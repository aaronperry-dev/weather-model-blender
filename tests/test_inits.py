"""Latest-run selection: newest run that has started by the cycle, else the previous (and the one before)."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from pwpf_blender import cli, noaa_http as nh
from pwpf_blender.pipeline import resolve_inits

CYCLE = datetime(2026, 11, 12, 12)
G = {"name": "gfs", "run_hours": [0, 6, 12, 18], "age_h": 0, "max_fxx_h": 384}


class Src:
    step_hours = 6

    def __init__(self, ready):
        self.ready, self.probes = set(ready), []

    def available(self, init, fxx, member):
        self.probes.append((init, fxx))
        return init in self.ready


def test_newest_run_is_used_when_published():
    src = Src({datetime(2026, 11, 12, 12)})
    (init, note), = resolve_inits(G, src, None, CYCLE, 72)
    assert init == datetime(2026, 11, 12, 12) and note == ""
    assert src.probes == [(datetime(2026, 11, 12, 12), 72)]                    # asks for the last hour the blend needs


def test_previous_run_when_the_newest_is_not_out_yet():
    src = Src({datetime(2026, 11, 12, 6), datetime(2026, 11, 12, 0)})
    (init, note), = resolve_inits(G, src, None, CYCLE, 72)
    assert init == datetime(2026, 11, 12, 6) and "12 12Z run not fully published yet; using 12 06Z" in note


def test_two_runs_back_and_then_the_nominal_run_is_kept_for_the_report():
    (init, _), = resolve_inits(G, Src({datetime(2026, 11, 12, 0)}), None, CYCLE, 72)
    assert init == datetime(2026, 11, 12, 0)                                    # at most two runs back
    (init, _), = resolve_inits(G, Src(set()), None, CYCLE, 72)                  # nothing complete: failures get reported
    assert init == datetime(2026, 11, 12, 12)


def test_twelve_hourly_models_step_back_twelve_hours():
    g = {"name": "ifs", "run_hours": [0, 12], "age_h": 0, "max_fxx_h": 144}
    (init, note), = resolve_inits(g, Src({datetime(2026, 11, 12, 0)}), None, CYCLE, 72)
    assert init == datetime(2026, 11, 12, 0) and "12 12Z" in note


def test_lagged_members_are_distinct_runs():
    g = dict(G, lags_h=[0, 6])
    out = resolve_inits(g, Src({datetime(2026, 11, 12, 6), datetime(2026, 11, 12, 0)}), None, CYCLE, 72)
    assert [i for i, _ in out] == [datetime(2026, 11, 12, 6), datetime(2026, 11, 12, 0)]


@pytest.mark.parametrize("now,expect", [((2026, 11, 12, 14, 15), (2026, 11, 12, 12)), ((2026, 11, 12, 11, 59), (2026, 11, 12, 0)),
                                        ((2026, 11, 12, 0, 5), (2026, 11, 12, 0)), ((2026, 11, 12, 12, 0), (2026, 11, 12, 12))])
def test_auto_cycle_is_the_newest_cycle_that_has_started(monkeypatch, now, expect):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(*now, tzinfo=tz)
    monkeypatch.setattr(cli, "datetime", FakeDT)
    assert cli._parse_cycle("auto", {"cycle_hours": [0, 12]}) == datetime(*expect)


# ---------------------------------------------------------------- ECMWF open data probe
class _FakeClient:
    def __init__(self, source=None, **kw):
        self.source = source

    def _get_urls(self, request=None, use_index=False, target=None, **kw):
        base = {"aws": "https://aws.example", "google": "https://g.example"}[self.source]
        return SimpleNamespace(urls=[f"{base}/20261112/12z/ifs/0p25/enfo/20261112120000-72h-enfo-ef.grib2"])


def _ifs(monkeypatch, answers):
    pytest.importorskip("ecmwf.opendata")
    import ecmwf.opendata as eo
    from pwpf_blender.sources.noaa_sources import IfsSource
    monkeypatch.setattr(eo, "Client", _FakeClient)
    seen = []

    def req(url, headers=None, **k):
        seen.append(url)
        a = answers[url.split("//")[1].split(".")[0]]
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(status_code=a, close=lambda: None)
    monkeypatch.setattr(nh, "_request", req)
    return IfsSource("ifs_ens", True, source="aws"), seen


def test_ifs_probe_true_when_the_index_exists(monkeypatch):
    src, seen = _ifs(monkeypatch, {"aws": 206})
    assert src.available(datetime(2026, 11, 12, 12), 72, 3)
    assert seen == ["https://aws.example/20261112/12z/ifs/0p25/enfo/20261112120000-72h-enfo-ef.index"]


def test_ifs_probe_false_only_on_a_definite_404(monkeypatch):
    src, _ = _ifs(monkeypatch, {"aws": 404, "g": 404})
    assert not src.available(datetime(2026, 11, 12, 12), 72, 3)
    src, _ = _ifs(monkeypatch, {"aws": 404, "g": 206})                      # another mirror already has it
    assert src.available(datetime(2026, 11, 12, 12), 72, 3)
    src, _ = _ifs(monkeypatch, {"aws": nh.FetchError("throttled"), "g": nh.FetchError("throttled")})
    assert src.available(datetime(2026, 11, 12, 12), 72, 3)                 # unknown is not "missing"


# ---------------------------------------------------------------- NOMADS answers 403 for a run that is not published yet
def test_noaa_probe_treats_403_as_not_published_without_penalising_the_host(monkeypatch):
    from pwpf_blender.sources.noaa_sources import NoaaSource
    nh._IDX_CACHE.clear()
    url_a = "https://nomads.example/cmce.20261112/12/f.idx"
    sent = []

    class R:
        def __init__(self, code):
            self.status_code, self.headers, self.text, self.content = code, {}, "", b""

        def close(self):
            pass

    class Sess:
        def get(self, url, **kw):
            sent.append(url)
            return R(403 if "/12/" in url else 200)
    monkeypatch.setattr(nh, "_session", lambda: Sess())
    monkeypatch.setattr(nh, "parse_idx", lambda text: ["rec"])
    monkeypatch.setattr(nh.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("probe must not sleep")))
    pol = nh.Policy("nomads.example", 1000.0, 1000, 4, (429, 503, 403))
    monkeypatch.setattr(nh, "policy_for", lambda u: pol)
    plan = lambda init, fxx, m: [SimpleNamespace(urls=[f"https://nomads.example/cmce.20261112/{init:%H}/f"])]   # noqa: E731
    src = NoaaSource("cmce", plan, 3, 25.0)
    assert not src.available(datetime(2026, 11, 12, 12), 72, 1)           # 403 = not published
    assert len(sent) == 1                                                  # one request: no retries
    assert pol.blocked_until == 0   # host not put on a cool-down
    assert pol.consecutive == 0
    assert src.available(datetime(2026, 11, 12, 0), 72, 1)                # the fall-back run is probed straight away


def test_noaa_probe_unknown_is_not_missing(monkeypatch):
    from pwpf_blender.sources.noaa_sources import NoaaSource
    nh._IDX_CACHE.clear()
    monkeypatch.setattr(nh, "_get_inventory_uncached", lambda url, probe=False: (_ for _ in ()).throw(nh.FetchError("timeout")))
    src = NoaaSource("x", lambda i, f, m: [SimpleNamespace(urls=["https://h/a", "https://h/b"])], 3, 25.0)
    assert src.available(datetime(2026, 11, 12, 12), 6, 1)
    monkeypatch.setattr(nh, "_get_inventory_uncached",
                        lambda url, probe=False: (_ for _ in ()).throw(nh.HttpStatusError("x", 404)))
    assert not src.available(datetime(2026, 11, 12, 12), 6, 1)
