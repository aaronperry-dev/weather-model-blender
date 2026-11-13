"""NOHRSC fetch/decode, scoring, archive + ledger, calibration fit.  Offline: GRIBs are generated locally, HTTP is a local server."""
import functools
import http.server
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pwpf_blender import binormal, cli, nohrsc, recalibrate, verify
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec

CFG = Path(__file__).resolve().parents[1] / "config" / "pwpf.yaml"


# ------------------------------------------------------------------------------------------ NOHRSC files
def _write_grib(path, ni=16, nj=31, scans_positively=1, valid=(2026, 11, 13, 12), hours=24):
    import eccodes as ec
    g = ec.codes_grib_new_from_samples("GRIB2")
    ec.codes_set(g, "gridType", "lambert")
    v = np.linspace(0, 0.2, ni * nj)                       # metres
    v[5] = 9999.0                                          # no analysis
    ec.codes_set(g, "bitmapPresent", 1)
    ec.codes_set(g, "missingValue", 9999.0)
    ec.codes_set_values(g, v)
    ec.codes_set(g, "jScansPositively", scans_positively)
    for k, val in zip(("year", "month", "day", "hour"), valid):
        try:
            ec.codes_set(g, k, val)
        except Exception:                                  # noqa: BLE001
            pass
    with open(path, "wb") as f:
        ec.codes_write(g, f)
    ec.codes_release(g)
    return v


def test_url_and_unit_conversion():
    assert nohrsc.url(datetime(2026, 11, 13, 12)) == \
        "https://www.nohrsc.noaa.gov/snowfall_v2/data/202611/sfav2_CONUS_24h_2026111312_grid184.grb2"
    assert nohrsc.url(datetime(2026, 12, 1, 0), 6).endswith("/202612/sfav2_CONUS_6h_2026120100_grid184.grb2")
    inches = nohrsc.to_inches(np.array([0.0254, 9999.0, -0.001, np.nan, 0.254]))
    assert inches[0] == pytest.approx(1.0) and inches[4] == pytest.approx(10.0)
    assert np.isnan(inches[1]) and np.isnan(inches[3]) and inches[2] == 0.0


@pytest.mark.parametrize("positive", [1, 0])
def test_read_grib_orientation_and_missing(tmp_path, monkeypatch, positive):
    monkeypatch.setattr(nohrsc, "NY", 31)
    monkeypatch.setattr(nohrsc, "NX", 16)
    f = tmp_path / "t.grb2"
    v = _write_grib(f, scans_positively=positive)
    g = nohrsc.read_grib(f)
    assert g["inches"].shape == (31, 16)
    raw = v.reshape(31, 16)
    want = raw if positive else raw[::-1]                   # always returned south-first
    ref = np.where(want >= 9998, np.nan, want / 0.0254)
    assert np.allclose(g["inches"], ref, equal_nan=True, atol=1e-3)
    assert np.isnan(g["inches"]).sum() == 1


def test_read_grib_rejects_other_grid(tmp_path):
    f = tmp_path / "t.grb2"
    _write_grib(f)
    with pytest.raises(nohrsc.nh.FetchError, match="expected the NDFD 2.5 km grid"):
        nohrsc.read_grib(f)


def _server(root: Path):
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            super().do_GET()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


def test_fetch_downloads_once_and_reports_unpublished(tmp_path):
    web = tmp_path / "web" / "202611"
    web.mkdir(parents=True)
    f = web / "sfav2_CONUS_24h_2026111312_grid184.grb2"
    f.write_bytes(b"GRIB" + b"\0" * 20000)
    srv, hits = _server(tmp_path / "web")
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        cache = tmp_path / "cache"
        p = nohrsc.fetch(datetime(2026, 11, 13, 12), 24, cache, base)
        assert p is not None and p.exists()
        n = len(hits)
        assert nohrsc.fetch(datetime(2026, 11, 13, 12), 24, cache, base) == p
        assert len(hits) == n                                      # cached: no second request
        assert nohrsc.fetch(datetime(2026, 11, 13, 18), 24, cache, base) is None     # 404 -> not published yet
        (web / "sfav2_CONUS_24h_2026111400_grid184.grb2").write_bytes(b"<html>" + b"x" * 20000)
        with pytest.raises(nohrsc.nh.FetchError, match="not a GRIB2"):
            nohrsc.fetch(datetime(2026, 11, 14, 0), 24, cache, base)
        assert not (cache / "sfav2_CONUS_24h_2026111400_grid184.grb2").exists()
    finally:
        srv.shutdown()


def test_nohrsc_host_has_its_own_conservative_policy():
    pol = nohrsc.nh.policy_for("https://www.nohrsc.noaa.gov/snowfall_v2/data/202611/x.grb2")
    assert pol.key == "nohrsc" and pol.max_conc == 1 and pol.rate <= 1.0


def test_prune_cache(tmp_path):
    for name in ("sfav2_CONUS_24h_2026110112_grid184.grb2", "sfav2_CONUS_24h_2026111312_grid184.grb2"):
        (tmp_path / name).write_bytes(b"x")
    assert nohrsc.prune_cache(tmp_path, 5, datetime(2026, 11, 14)) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["sfav2_CONUS_24h_2026111312_grid184.grb2"]


# ------------------------------------------------------------------------------------------ scoring
STRIDE = 12


def _field(seed=0, scale=1.0):
    g = GridSpec.ndfd_2p5km(STRIDE)
    lon, lat = g.lonlat()
    rng = np.random.default_rng(seed)
    base = np.clip(np.sin(np.radians(lon + 80) * 6) * np.cos(np.radians(lat - 36) * 9), 0, None) * 6 * scale
    base = base + rng.random(base.shape) * 0.05
    return lon, lat, base.astype(np.float32)


def test_score_case_perfect_forecast_and_additivity():
    _, _, m = _field()
    lo, hi = np.full_like(m, 0.3), np.full_like(m, 0.6)
    obs = m.copy()
    sc = verify.score_case(m, lo, hi, obs, STRIDE)
    d = verify.derived(sc["conus"])
    assert d["mode"]["bias"] == pytest.approx(0, abs=1e-6) and d["mode"]["rmse"] == pytest.approx(0, abs=1e-6)
    assert d["mode"]["corr_wet"] == pytest.approx(1.0, abs=1e-6)
    assert d["mode"]["cat"][1.0]["csi"] == pytest.approx(1.0)
    assert d["brier"][1.0]["bss"] > 0.5
    led = {"cases": [{"scores": sc}, {"scores": sc}]}
    d2 = verify.derived(verify.merge_cases(led["cases"], "conus"))
    assert d2["cells"] == 2 * d["cells"] and d2["mode"]["mae"] == pytest.approx(d["mode"]["mae"])


def test_score_case_detects_bias_and_nan_cells_are_ignored():
    _, _, m = _field()
    lo, hi = np.full_like(m, 0.3), np.full_like(m, 0.6)
    obs = m * 0.5
    obs[:20] = np.nan                                           # no analysis there
    sc = verify.score_case(m, lo, hi, obs, STRIDE)
    d = verify.derived(sc["conus"])
    assert d["cells"] == int(np.isfinite(obs).sum())
    assert d["mode"]["bias"] > 0.05                              # forecast too high
    assert d["mode"]["cat"][2.0]["freq_bias"] > 1.3
    assert d["coverage"][10] > 0.3 and d["coverage"][90] > 0.95   # forecast too high: the analysis sits low in the distribution
    txt = verify.summarize({"cases": [{"cycle": "2026-11-12T12:00:00", "window": "s24_f024", "end": "2026-11-13T12:00:00",
                                       "fingerprint": "abc", "label": "x", "scores": sc}]})[1]
    assert "settings abc" in txt and "Northeast box" in txt and "PIT histogram" in txt


# ------------------------------------------------------------------------------------------ archive + ledger
def _fake_results(cycle, scale=1.0, grid_stride=STRIDE):
    _, _, m = _field(scale=scale)
    wins = {}
    for end_h in (24, 30):
        wins[f"s24_f0{end_h}"] = {"mode": m, "s_lo": np.full_like(m, 0.4), "s_hi": np.full_like(m, 0.8), "length_h": 24,
                                  "start": cycle + timedelta(hours=end_h - 24), "end": cycle + timedelta(hours=end_h)}
    wins["s72_f072"] = {"mode": m, "s_lo": m * 0 + 1, "s_hi": m * 0 + 1, "length_h": 72, "start": cycle,
                        "end": cycle + timedelta(hours=72)}
    return SimpleNamespace(cycle=cycle, grid=GridSpec.ndfd_2p5km(grid_stride), windows=wins, meta={"n_members": 5})


def test_archive_keeps_only_24h_windows_and_prunes(tmp_path):
    cfg = load_config(CFG, "synthetic")
    old = datetime(2026, 10, 1, 0)
    verify.archive_results(_fake_results(old), cfg, tmp_path)
    new = datetime(2026, 11, 12, 12)
    p = verify.archive_results(_fake_results(new), cfg, tmp_path)
    assert not list(tmp_path.glob("fc_2026100100_*.npz"))         # older than archive_days
    z = np.load(p)
    meta = json.loads(str(z["meta"]))
    assert sorted(meta["windows"]) == ["s24_f024", "s24_f030"] and "s72_f072_mode" not in z.files
    assert meta["stride"] == STRIDE and len(meta["fingerprint"]) == 8


def test_verify_archive_scores_due_periods_once(tmp_path, monkeypatch):
    cfg = load_config(CFG, "synthetic")
    cycle = datetime(2026, 11, 12, 12)
    verify.archive_results(_fake_results(cycle), cfg, tmp_path / "arch")
    calls = []
    obs_full = np.kron(_field(scale=0.6)[2], np.ones((STRIDE, STRIDE), np.float32))[:1377, :2145]

    def fake_load(valid, hours, cache, base=None):
        calls.append(valid)
        return None if valid.hour == 18 else obs_full             # 18Z analysis is "not published"
    monkeypatch.setattr(verify.nohrsc, "load", fake_load)

    led = verify.verify_archive(cfg, tmp_path / "arch", tmp_path / "cache", now=cycle + timedelta(hours=28))   # only f024 is due (+3 h latency)
    assert [c["window"] for c in led["cases"]] == ["s24_f024"] and len(calls) == 1
    led = verify.verify_archive(cfg, tmp_path / "arch", tmp_path / "cache", now=cycle + timedelta(hours=40))   # f030 due, not published
    assert [c["window"] for c in led["cases"]] == ["s24_f024"]
    assert list(led["missing"].values()) == [1] and list(led["missing"])[0].startswith("2026-11-12T12:00:00|s24_f030|")
    led = verify.verify_archive(cfg, tmp_path / "arch", tmp_path / "cache", now=cycle + timedelta(hours=60))
    assert list(led["missing"].values()) == [2]
    assert len(calls) == 3                                          # f024 once, f030 twice - f024 never fetched again
    assert (tmp_path / "arch" / "pairs").glob("pairs_*.npz")
    c = led["cases"][0]
    assert c["scores"]["conus"]["mode"]["sum"] > c["scores"]["conus"]["sum_obs"]       # forecast was 1/0.6 too high
    # the same ledger file survives a reload
    assert verify.load_ledger(tmp_path / "arch" / "ledger.json")["cases"][0]["window"] == "s24_f024"


def test_missing_attempts_are_capped(tmp_path, monkeypatch):
    cfg = load_config(CFG, "synthetic", {"verification": {"max_tries": 2}})
    cycle = datetime(2026, 11, 12, 12)
    verify.archive_results(_fake_results(cycle), cfg, tmp_path / "arch")
    n = []
    monkeypatch.setattr(verify.nohrsc, "load", lambda *a, **k: n.append(1))     # always None
    for h in (30, 40, 50, 60):
        verify.verify_archive(cfg, tmp_path / "arch", tmp_path / "cache", now=cycle + timedelta(hours=h))
    assert len(n) == 4                                              # 2 windows x max_tries 2, then left alone


def test_fingerprint_follows_the_science_settings():
    a = load_config(CFG, "synthetic")
    b = load_config(CFG, "synthetic", {"accumulation": {"enabled": True}})
    c = load_config(CFG, "synthetic", {"verification": {"min_cases": 99}})
    assert verify.fingerprint(a)[0] != verify.fingerprint(b)[0]
    assert verify.fingerprint(a)[0] == verify.fingerprint(c)[0]       # verification settings do not change the products
    assert "accumulation off" in verify.label_of(verify.fingerprint(a)[1])


# ------------------------------------------------------------------------------------------ calibration fit
def _pairs(tmp_path, fp, a_true=1.3, cycles=10, n=4000):
    rng = np.random.default_rng(3)
    (tmp_path / "pairs").mkdir(exist_ok=True)
    for i in range(cycles):
        m = rng.uniform(0.3, 6, n).astype(np.float32)
        lo, hi = (m * 0.25 + 0.2).astype(np.float32), (m * 0.5 + 0.3).astype(np.float32)
        u = rng.uniform(0.0, 1.0, n)
        y = binormal.percentile(m * a_true, lo, hi, u).astype(np.float32)          # truth drawn from the *scaled* distribution
        np.savez_compressed(tmp_path / "pairs" / f"pairs_20261{i:03d}_s24_f024.npz", mode=m, s_lo=lo, s_hi=hi, obs=y,
                            meta=np.array(json.dumps({"fingerprint": fp, "cycle": f"c{i}", "window": "s24_f024", "end": "x"})))


def test_fit_recovers_a_known_bias(tmp_path):
    _pairs(tmp_path, "fp1")
    r = recalibrate.fit(tmp_path, "fp1", min_cycles=8)
    assert r["usable"] and r["cycles"] == 10
    assert r["a"] == pytest.approx(1.3, abs=0.08)
    assert r["loss_after"] < r["loss_before"]


def test_fit_not_usable_with_few_cycles_other_settings_or_no_gain(tmp_path):
    _pairs(tmp_path, "fp1", cycles=3)
    assert not recalibrate.fit(tmp_path, "fp1", min_cycles=8)["usable"]
    assert "no archived pairs" in recalibrate.fit(tmp_path, "other")["reason"]
    _pairs(tmp_path, "fp1", a_true=1.0, cycles=10)
    r = recalibrate.fit(tmp_path, "fp1", min_cycles=8)
    assert abs(r["a"] - 1.0) < 0.06 and abs(r["k_lo"] - 1) < 0.25 and abs(r["k_hi"] - 1) < 0.25


def test_apply_scales_windows_only_when_usable_and_matching(tmp_path):
    res = _fake_results(datetime(2026, 11, 12, 12))
    before = {k: v["mode"].copy() for k, v in res.windows.items()}
    fit_path = tmp_path / "fit.json"
    assert "no fit file" in recalibrate.apply(res, fit_path, "fp")
    recalibrate.write_fit({"fingerprint": "fp", "a": 1.2, "k_lo": 0.9, "k_hi": 1.5, "usable": True, "cycles": 9, "cells": 99999}, fit_path)
    assert "other science settings" in recalibrate.apply(res, fit_path, "different")
    assert np.array_equal(res.windows["s24_f024"]["mode"], before["s24_f024"])
    recalibrate.write_fit({"fingerprint": "fp", "a": 1.2, "k_lo": 0.9, "k_hi": 1.5, "usable": False, "reason": "x"}, fit_path)
    assert "not usable" in recalibrate.apply(res, fit_path, "fp")
    recalibrate.write_fit({"fingerprint": "fp", "a": 1.2, "k_lo": 0.9, "k_hi": 1.5, "usable": True, "cycles": 9, "cells": 99999}, fit_path)
    assert "calibration applied" in recalibrate.apply(res, fit_path, "fp")
    for k, w in res.windows.items():
        assert np.allclose(w["mode"], before[k] * 1.2, atol=1e-5)
    assert res.meta["calibration_applied"]["k_hi"] == 1.5


# ------------------------------------------------------------------------------------------ CLI
def test_run_cli_archives_raw_fields_and_accumulation_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "site" / "data"
    argv = ["run", "--config", str(CFG), "--synthetic", "--cycle", "2026-01-15T12", "--stride", "12", "--workers", "2",
            "--out", str(out), "--archive-dir", str(tmp_path / "arch"), "--accumulation", "on"]
    cli.main(argv)
    (f1,) = (tmp_path / "arch").glob("fc_2026011512_*.npz")
    meta = json.loads(str(np.load(f1)["meta"]))
    assert "accumulation on" in meta["label"] and meta["windows"]
    cli.main(argv[:-1] + ["off"])
    assert len(list((tmp_path / "arch").glob("fc_2026011512_*.npz"))) == 2          # A/B of one cycle: both kept
    (f2,) = [f for f in (tmp_path / "arch").glob("fc_2026011512_*.npz") if f != f1]
    meta2 = json.loads(str(np.load(f2)["meta"]))
    assert "accumulation off" in meta2["label"] and meta2["fingerprint"] != meta["fingerprint"]
    mf = json.loads((out / "manifest.json").read_text())
    assert mf["accumulation"] is None and mf["calibration_applied"] is None


def test_verify_archive_cli_writes_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "site" / "data"
    cli.main(["run", "--config", str(CFG), "--synthetic", "--cycle", "2026-01-15T12", "--stride", "12", "--workers", "2",
              "--out", str(out), "--archive-dir", str(tmp_path / "arch")])
    obs_full = np.zeros((1377, 2145), np.float32)
    monkeypatch.setattr(verify.nohrsc, "load", lambda *a, **k: obs_full)
    cli.main(["verify-archive", "--config", str(CFG), "--profile", "synthetic", "--archive", str(tmp_path / "arch"),
              "--cache", str(tmp_path / "c"), "--out", str(out), "--now", "2026-01-20T00:00"])
    txt = (out / "verification.txt").read_text()
    assert "verification against the NOHRSC" in txt and "calibration fit" in txt
    js = json.loads((out / "verification.json").read_text())
    assert js["groups"] and js["calibration_fit"]["usable"] is False
    assert json.loads((out / "manifest.json").read_text())["verification"] == "verification.txt"


def test_ab_runs_of_one_cycle_are_scored_separately(tmp_path, monkeypatch):
    cycle = datetime(2026, 11, 12, 12)
    cfg_off = load_config(CFG, "synthetic")
    cfg_on = load_config(CFG, "synthetic", {"accumulation": {"enabled": True}})
    verify.archive_results(_fake_results(cycle), cfg_off, tmp_path / "arch")
    verify.archive_results(_fake_results(cycle, scale=0.7), cfg_on, tmp_path / "arch")
    obs_full = np.kron(_field(scale=0.6)[2], np.ones((STRIDE, STRIDE), np.float32))[:1377, :2145]
    monkeypatch.setattr(verify.nohrsc, "load", lambda *a, **k: obs_full)
    led = verify.verify_archive(cfg_off, tmp_path / "arch", tmp_path / "c", now=cycle + timedelta(hours=60))
    assert len(led["cases"]) == 4 and len({c["fingerprint"] for c in led["cases"]}) == 2
    txt = verify.summarize(led)[1]
    assert "accumulation off" in txt and "accumulation on" in txt
    assert len(list((tmp_path / "arch" / "pairs").glob("pairs_*.npz"))) == 4
