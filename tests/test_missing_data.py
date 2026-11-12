"""Blending must keep going when members are missing or fail part-way."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pwpf_blender import export
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec
from pwpf_blender.pipeline import Pipeline
from pwpf_blender.sources import synthetic as syn
from pwpf_blender.sources.synthetic import synthetic_prism, synthetic_terrain

CFG = Path(__file__).resolve().parents[1] / "config" / "pwpf.yaml"
CYCLE = datetime(2026, 1, 15, 12)


def _pipe(over=None):
    cfg = load_config(CFG, "synthetic", dict({"grid": {"stride": 16}, "workers": 2}, **(over or {})))
    grid = GridSpec.ndfd_2p5km(16)
    lon, lat = grid.lonlat()
    return cfg, Pipeline(cfg, grid, synthetic_terrain(lon, lat), synthetic_prism(lon, lat))


def _break(monkeypatch, bad_members, from_h=0, kind="global", calls=None):
    """Make the given synthetic members raise for any bin starting at/after cycle+from_h."""
    orig = syn.SyntheticSource.load_steps

    def load_steps(self, init, valid_start, valid_end, member):
        if self.kind == kind and member in bad_members and valid_start >= CYCLE + timedelta(hours=from_h):
            if calls is not None:
                calls.append((member, valid_start))
            raise RuntimeError("simulated 404 for member %s" % member)
        return orig(self, init, valid_start, valid_end, member)
    monkeypatch.setattr(syn.SyntheticSource, "load_steps", load_steps)


def test_member_lost_midway_still_feeds_the_periods_it_covers(monkeypatch, tmp_path):
    cfg, pipe = _pipe()
    _break(monkeypatch, {5, 6, 7, 8, 9}, from_h=36)
    res = pipe.run(CYCLE)
    assert len(res.windows) == 15 and not res.skipped                  # nothing dropped, everything still produced
    assert res.windows["s24_f024"]["n_members"] == 34                  # bins 0-3 are complete for everybody
    assert res.windows["s24_f036"]["n_members"] == 34                  # ends at +36 h
    assert res.windows["s24_f042"]["n_members"] == 29                  # needs the +36..42 h bin
    assert res.windows["s72_f072"]["n_members"] == 29
    assert res.windows["s24_f042"]["reduced"] is False                 # 29 >= warn_members
    for w in res.windows.values():
        assert np.isfinite(w["mode"]).all() and (w["s_lo"] > 0).all() and (w["s_hi"] > 0).all()
    part = [m for m in res.members if not m["ok"]]
    assert len(part) == 5 and all(m["bins_ok"] == 6 and m["bins"] == "111111000000" for m in part)
    assert all("simulated 404" in m["error"] for m in part)
    export.write_site(res, cfg, tmp_path / "d")
    man = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert man["members_total"] == 29 and man["members_requested"] == 34 and len(man["members"]) == 34
    assert {w["id"]: w["n_members"] for w in man["windows"]}["s24_f066"] == 29


def test_failed_member_is_abandoned_after_the_failure_limit(monkeypatch):
    cfg, pipe = _pipe({"member_failure_limit": 2})
    calls = []
    _break(monkeypatch, {7}, from_h=0, calls=calls)   # 7: not a fallback target of a CAM member
    res = pipe.run(CYCLE)
    assert len(calls) == 2                                              # tried bins 0 and 1, then given up
    assert res.windows["s24_f024"]["n_members"] == 33 and len(res.windows) == 15


def test_periods_with_too_few_members_are_skipped_not_fatal(monkeypatch, tmp_path):
    cfg, pipe = _pipe({"min_members": 30})
    _break(monkeypatch, {5, 6, 7, 8, 9}, from_h=36)
    res = pipe.run(CYCLE)
    assert "s24_f024" in res.windows and "s24_f072" not in res.windows
    assert {x["id"] for x in res.skipped} >= {"s24_f042", "s72_f072"}
    assert all(x["n_members"] == 29 for x in res.skipped)
    export.write_site(res, cfg, tmp_path / "d")
    man = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert {w["id"] for w in man["windows"]} == set(res.windows) and man["windows_skipped"]
    assert not (tmp_path / "d" / "s72_f072.png").exists()


def test_everything_failing_is_an_error_with_reasons(monkeypatch):
    cfg, pipe = _pipe()
    _break(monkeypatch, set(range(1, 31)), kind="global")
    _break(monkeypatch, {1, 2, 3, 4}, kind="cam")
    with pytest.raises(RuntimeError, match="no member delivered any data"):
        pipe.run(CYCLE)


def test_one_bad_bin_in_the_middle_only_costs_the_windows_that_include_it(monkeypatch):
    cfg, pipe = _pipe({"member_failure_limit": 3})
    orig = syn.SyntheticSource.load_steps

    def flaky(self, init, valid_start, valid_end, member):
        if self.kind == "global" and member == 2 and valid_start == CYCLE + timedelta(hours=18):
            raise RuntimeError("transient")
        return orig(self, init, valid_start, valid_end, member)
    monkeypatch.setattr(syn.SyntheticSource, "load_steps", flaky)
    res = pipe.run(CYCLE)
    n = {k: v["n_members"] for k, v in res.windows.items()}
    # the lost bin is +18..+24 h (bin 3): every period containing it loses member 2, nothing else does
    assert n["s24_f024"] == n["s24_f030"] == n["s24_f036"] == n["s24_f042"] == 33
    assert n["s48_f048"] == n["s72_f072"] == 33
    assert n["s24_f048"] == n["s24_f054"] == n["s24_f072"] == 34       # these start at/after +24 h: unaffected


# ---------------------------------------------------------------- unpublished newest run -> previous run
class _Probe:
    """Stand-in source: only runs listed in `ready` (init, last published fxx) have data."""
    step_hours, native_ptype = 3, False

    def __init__(self, ready):
        self.ready, self.asked = ready, []

    def available(self, init, fxx, member):
        self.asked.append((init, fxx))
        return self.ready.get(init, -1) >= fxx


def test_incomplete_newest_run_steps_back_one_run(monkeypatch):
    from pwpf_blender import pipeline as pl
    cfg = {"members": [{"name": "gfs", "loader": "x", "members": [None], "run_hours": [0, 6, 12, 18], "age_h": 0,
                        "max_fxx_h": 384, "weight": 1}]}
    cycle = datetime(2026, 11, 12, 0)
    src = _Probe({datetime(2026, 11, 12, 0): 30, datetime(2026, 11, 11, 18): 120})   # 00Z only out to f030
    monkeypatch.setattr(pl, "make_source", lambda name: src)
    runs = pl.expand_members(cfg, cycle, need_h=72)
    assert [r.init for r in runs] == [datetime(2026, 11, 11, 18)]
    assert (datetime(2026, 11, 12, 0), 72) in src.asked and (datetime(2026, 11, 11, 18), 78) in src.asked
    assert "00Z run not fully published" in runs[0].note


def test_lagged_group_does_not_duplicate_runs_and_complete_run_is_kept(monkeypatch):
    from pwpf_blender import pipeline as pl
    cfg = {"members": [{"name": "hrrr", "loader": "x", "members": [None], "lags_h": [0, 6],
                        "run_hours": [0, 6, 12, 18], "age_h": 0, "max_fxx_h": 36, "weight": 1}]}
    cycle = datetime(2026, 11, 12, 0)
    # newest (00Z) not out yet; 18Z and 12Z complete -> 00Z request falls back to 18Z, the 6 h lag (18Z) moves to 12Z
    src = _Probe({datetime(2026, 11, 11, 18): 48, datetime(2026, 11, 11, 12): 48})
    monkeypatch.setattr(pl, "make_source", lambda name: src)
    runs = pl.expand_members(cfg, cycle, need_h=72)
    assert sorted(r.init for r in runs) == [datetime(2026, 11, 11, 12), datetime(2026, 11, 11, 18)]
    src2 = _Probe({datetime(2026, 11, 12, 0): 48, datetime(2026, 11, 11, 18): 48})
    monkeypatch.setattr(pl, "make_source", lambda name: src2)
    runs = pl.expand_members(cfg, cycle, need_h=72)
    assert sorted(r.init for r in runs) == [datetime(2026, 11, 11, 18), datetime(2026, 11, 12, 0)]
    assert all(not r.note for r in runs)
