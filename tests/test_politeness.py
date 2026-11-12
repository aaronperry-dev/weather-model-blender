"""Rate limiting, back-off, circuit breaker and request-volume behaviour of the download layer."""
import functools
import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from pwpf_blender import noaa_http as nh
from pwpf_blender import static_data as sd


# ------------------------------------------------------------------ token bucket (fake clock)
class FakeTime:
    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_limiter_enforces_rate_and_allows_small_burst():
    ft = FakeTime()
    lim = nh.Limiter(rate=2.0, burst=2, clock=ft.clock, sleep=ft.sleep)
    waits = [lim.acquire() for _ in range(6)]
    assert waits[:2] == [0.0, 0.0]                       # burst
    assert all(w == pytest.approx(0.5) for w in waits[2:])   # then exactly 2 requests/second
    elapsed = ft.t - 100.0
    assert elapsed == pytest.approx(2.0)                  # 6 requests, 2 free -> 4 x 0.5 s
    ft.t += 10                                            # idle: bucket refills, but only up to the burst
    assert [lim.acquire() for _ in range(3)] == [0.0, 0.0, pytest.approx(0.5)]


def test_limiter_is_thread_safe():
    lim = nh.Limiter(rate=1000.0, burst=1)
    t0 = time.monotonic()
    ths = [threading.Thread(target=lambda: [lim.acquire() for _ in range(50)]) for _ in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert time.monotonic() - t0 >= 0.15                  # 200 requests at 1000/s cannot finish faster


# ------------------------------------------------------------------ policies (documented limits)
def test_policies_match_published_limits(monkeypatch):
    monkeypatch.setattr(nh, "RATE_SCALE", 1.0)
    monkeypatch.setattr(nh, "_policies", {})
    ncep = [nh.policy_for(f"https://{h}/x") for h in
            ("nomads.ncep.noaa.gov", "ftp.ncep.noaa.gov", "ftpprd.ncep.noaa.gov")]
    assert ncep[0] is ncep[1] is ncep[2]                  # NCEP's 120 hits/min is summed over these hosts
    assert ncep[0].rate * 60 <= 60 and ncep[0].max_conc <= 2    # we stay at <= half of the 120/min limit
    assert 403 in ncep[0].throttle_codes and 429 in ncep[0].throttle_codes
    prism = nh.policy_for("https://data.prism.oregonstate.edu/normals/")
    assert prism is nh.policy_for("https://services.nacse.org/x") and prism.max_conc == 1 and prism.rate <= 0.5
    a = nh.policy_for("https://noaa-gefs-pds.s3.amazonaws.com/a")
    b = nh.policy_for("https://noaa-gfs-bdp-pds.s3.amazonaws.com/a")
    assert a is not b and a.rate <= 20 and a.max_conc <= 6        # per bucket (S3 allows thousands of GETs/s per prefix)
    assert 403 not in a.throttle_codes                            # S3 403 = forbidden, not "slow down"


# ------------------------------------------------------------------ local server with scripted behaviour
class Script:
    def __init__(self):
        self.hits = 0
        self.plan = []          # list of (status, headers dict, body) consumed in order; last one repeats


@pytest.fixture()
def server():
    sc = Script()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            sc.hits += 1
            status, hdr, body = sc.plan[min(sc.hits - 1, len(sc.plan) - 1)]
            self.send_response(status)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield sc, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _fresh_policies(monkeypatch):
    monkeypatch.setattr(nh, "_policies", {})
    monkeypatch.setattr(nh, "STATS", nh.Stats())
    monkeypatch.setattr(nh, "_IDX_CACHE", nh.OrderedDict())


def _sleeps(monkeypatch):
    rec = []
    monkeypatch.setattr(nh.time, "sleep", lambda s: rec.append(s))
    return rec


def test_retry_after_is_honoured_then_success(server, monkeypatch):
    sc, base = server
    sc.plan = [(503, {"Retry-After": "7"}, b"slow down"), (429, {"Retry-After": "11"}, b""), (200, {}, b"ok")]
    rec = _sleeps(monkeypatch)
    r = nh._get(base + "/f")
    assert r.status_code == 200 and sc.hits == 3
    assert rec == [7.0, 11.0]                                     # exactly what the server asked for


def test_exponential_backoff_with_jitter_when_no_retry_after(server, monkeypatch):
    sc, base = server
    sc.plan = [(503, {}, b""), (503, {}, b""), (200, {}, b"ok")]
    rec = _sleeps(monkeypatch)
    nh._get(base + "/f")
    assert 1.7 <= rec[0] <= 2.5 and 3.4 <= rec[1] <= 5.0 and rec[1] > rec[0]


def test_404_is_never_retried(server, monkeypatch):
    sc, base = server
    sc.plan = [(404, {}, b"nope")]
    rec = _sleeps(monkeypatch)
    assert nh._get(base + "/missing").status_code == 404
    assert sc.hits == 1 and rec == []                             # missing file: one request, no pause-and-retry


def test_persistent_throttling_gives_up_and_trips_the_breaker(server, monkeypatch):
    sc, base = server
    sc.plan = [(429, {"Retry-After": "1"}, b"")]
    _sleeps(monkeypatch)
    monkeypatch.setattr(nh, "RETRIES", 4)
    with pytest.raises(nh.FetchError, match="gave up|rate-limiting"):
        nh._get(base + "/f")
    first = sc.hits
    assert first <= nh.RETRIES + 1
    with pytest.raises(nh.FetchError, match="rate-limiting this client"):     # breaker open: fails fast...
        nh._get(base + "/g")
    assert sc.hits == first                                                    # ...without touching the server


def test_breaker_resets_after_cooldown_and_on_success(server, monkeypatch):
    sc, base = server
    sc.plan = [(429, {}, b"")] * 4 + [(200, {}, b"fine")]
    _sleeps(monkeypatch)
    monkeypatch.setattr(nh, "RETRIES", 3)
    with pytest.raises(nh.FetchError):
        nh._get(base + "/f")
    pol = nh.policy_for(base)
    assert pol.blocked_until > time.monotonic()
    pol.blocked_until = 0.0                                       # cool-down elapsed
    assert nh._get(base + "/f").status_code == 200 and pol.consecutive == 0


def test_requests_are_spaced_by_the_host_policy(server, monkeypatch):
    sc, base = server
    sc.plan = [(200, {}, b"x")]
    ft = FakeTime()
    pol = nh.policy_for(base)
    pol.limiter = nh.Limiter(rate=2.0, burst=1, clock=ft.clock, sleep=ft.sleep)
    for _ in range(5):
        nh._get(base + "/f")
    assert sc.hits == 5 and ft.t - 100.0 == pytest.approx(2.0)    # 5 requests at 2/s with a burst of 1 -> 4 x 0.5 s
    assert "5 requests" in nh.STATS.report()


def test_connection_cap_is_respected(server):
    sc, base = server
    pol = nh.policy_for(base)
    pol.sem = threading.BoundedSemaphore(1)
    live, peak, lock = [0], [0], threading.Lock()

    class H2(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            self.send_response(200)
            self.send_header("Content-Length", "1")
            self.end_headers()
            self.wfile.write(b"x")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H2)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    nh._policies.clear()
    nh.policy_for(url).sem = threading.BoundedSemaphore(1)
    ths = [threading.Thread(target=nh._get, args=(url,)) for _ in range(6)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    srv.shutdown()
    assert peak[0] == 1


def test_idx_is_fetched_once_per_file(server):
    sc, base = server
    sc.plan = [(200, {}, b"1:0:d=2026011512:TMP:2 m above ground:6 hour fcst:\n2:100:d=2026011512:PRES:surface:6 hour fcst:\n")]
    a = nh.get_inventory(base + "/file.grib2")
    b = nh.get_inventory(base + "/file.grib2")
    assert a is b and sc.hits == 1


def test_polite_context_counts_against_the_limit(monkeypatch):
    ft = FakeTime()
    pol = nh.policy_for("https://data.ecmwf.int/x")
    pol.limiter = nh.Limiter(rate=2.0, burst=2, clock=ft.clock, sleep=ft.sleep)
    with nh.polite("https://data.ecmwf.int/x", cost=5):
        pass
    assert ft.t - 100.0 == pytest.approx(1.5)                      # 5 requests: 2 from the burst, 3 more at 2/s
    assert "ecmwf: 5 requests" in nh.STATS.report()


# ------------------------------------------------------------------ PRISM download guard
def _listing_server(tmp_path):
    root = tmp_path / "srv" / "normals"
    root.mkdir(parents=True)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            return super().do_GET()

        def log_message(self, *a):
            pass
    handler = functools.partial(H, directory=str(root.parent))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return root, hits, srv, f"http://127.0.0.1:{srv.server_address[1]}/normals/"


def _zip_bytes():
    import io
    import zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.txt", "hello")
    return b.getvalue()


def test_prism_never_re_requests_files_it_has(tmp_path):
    root, hits, srv, url = _listing_server(tmp_path)
    for m in range(1, 13):
        (root / f"prism_ppt_us_30s_2020{m:02d}_avg_30y.zip").write_bytes(_zip_bytes())
    dest = tmp_path / "dl"
    sd.download_prism(dest, url)
    n = len(hits)
    assert n == 13                                                 # 1 listing + 12 files, once each
    sd.download_prism(dest, url)                                   # second call: everything on disk -> no requests
    assert len(hits) == n
    # one file lost but requested < 24 h ago: refuse to request it again
    next(dest.glob("05_*")).unlink()
    with pytest.raises(nh.FetchError, match="not requesting it again within 24 h"):
        sd.download_prism(dest, url)
    assert len(hits) == n + 1                                      # (only the directory listing was read)
    # ...but after the guard period it may be fetched again
    led = json.loads((dest / sd.LEDGER_NAME).read_text())
    led = {k: v - 25 * 3600 for k, v in led.items()}
    (dest / sd.LEDGER_NAME).write_text(json.dumps(led))
    files = sd.download_prism(dest, url)
    assert sorted(files) == list(range(1, 13)) and len(hits) == n + 3     # listing + the one missing file
    srv.shutdown()


def test_prism_request_that_was_not_served_is_not_counted(tmp_path):
    root, hits, srv, url = _listing_server(tmp_path)
    for m in range(1, 13):
        (root / f"prism_ppt_us_30s_2020{m:02d}_avg_30y.zip").write_bytes(_zip_bytes())
    (root / "prism_ppt_us_30s_202003_avg_30y.zip").unlink()         # listed in the page? no -> months missing
    with pytest.raises(nh.FetchError, match="could not identify"):
        sd.download_prism(tmp_path / "dl", url)
    # a listed-but-404 file (stale listing) must not poison the ledger
    (root / "prism_ppt_us_30s_202003_avg_30y.zip").write_bytes(_zip_bytes())
    orig = nh.download_file

    def flaky(u, dest, **k):
        if "202003" in u:
            raise nh.HttpStatusError(f"{u}: HTTP 404", 404)
        return orig(u, dest, **k)
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(nh, "download_file", flaky)
    with pytest.raises(nh.HttpStatusError):
        sd.download_prism(tmp_path / "dl2", url)
    mp.undo()
    led = json.loads((tmp_path / "dl2" / sd.LEDGER_NAME).read_text())
    assert not any("202003" in k for k in led) and len(led) == 2   # Jan and Feb were served and recorded
    srv.shutdown()


# ------------------------------------------------------------------ ECMWF client
class _FakeClient:
    """Stands in for ecmwf.opendata.Client; behaviour per mirror is scripted through `plan`."""
    plan: dict = {}
    seen: list = []

    def __init__(self, **kw):
        import requests
        type(self).seen.append(kw)
        self.kw, self.session = kw, requests.Session()

    def retrieve(self, **req):
        act = type(self).plan.get(self.kw["source"], "ok")
        if act != "ok":
            Path(req["target"]).write_bytes(b"GR")                 # a truncated file must not survive the failure
            raise RuntimeError(act)
        Path(req["target"]).write_bytes(b"GRIB")


def _ifs(monkeypatch, tmp_path, plan):
    pytest.importorskip("ecmwf.opendata")
    import ecmwf.opendata as eo
    from pwpf_blender.sources.noaa_sources import IfsSource
    _FakeClient.plan, _FakeClient.seen = plan, []
    monkeypatch.setattr(eo, "Client", _FakeClient)
    return IfsSource("ifs_ens", True, source="aws", save_dir=tmp_path)


def test_ifs_client_retries_are_capped(tmp_path, monkeypatch):
    from datetime import datetime
    src = _ifs(monkeypatch, tmp_path, {})
    src._retrieve(datetime(2026, 1, 15, 0), [6, 12], 3, "sfc")
    kw = _FakeClient.seen[0]
    assert kw["maximum_retries"] == 4 and kw["retry_after"] == [10, 40, 2]        # library default is 500 x 120 s
    assert any("ecmwf-forecasts" in k for k in nh._policies)


def test_ifs_falls_back_to_another_mirror(tmp_path, monkeypatch):
    from datetime import datetime
    src = _ifs(monkeypatch, tmp_path, {"aws": "503 Service Unavailable"})
    path = src._retrieve(datetime(2026, 1, 15, 0), [6, 12], 3, "sfc")
    assert [k["source"] for k in _FakeClient.seen] == ["aws", "google"] and path.read_bytes() == b"GRIB"


def test_ifs_gives_up_when_every_mirror_fails_and_stops_on_404(tmp_path, monkeypatch):
    from datetime import datetime
    src = _ifs(monkeypatch, tmp_path, {m: "503 Service Unavailable" for m in ("aws", "google", "azure", "ecmwf")})
    with pytest.raises(nh.FetchError, match="every mirror"):
        src._retrieve(datetime(2026, 1, 15, 0), [6], 3, "sfc")
    assert len(_FakeClient.seen) == 4 and not list(tmp_path.glob("*.grib2*"))
    src = _ifs(monkeypatch, tmp_path, {"aws": "404 Not Found"})
    with pytest.raises(nh.FetchError):
        src._retrieve(datetime(2026, 1, 15, 0), [6], 4, "sfc")
    assert [k["source"] for k in _FakeClient.seen] == ["aws"]                      # not published: no point asking others


def test_limiter_penalty_delays_every_thread():
    ft = FakeTime()
    lim = nh.Limiter(rate=2.0, burst=2, clock=ft.clock, sleep=ft.sleep)
    lim.penalise(10.0)                                    # 20 tokens of debt on a bucket holding 2
    assert lim.acquire() == pytest.approx(9.5)            # next request waits until the debt is paid (+1 token)
    lim.penalise(1000.0)
    assert lim.acquire() == pytest.approx(60.5, abs=0.6)  # debt is capped at 60 s


def test_session_adapter_paces_counts_and_penalises(server, monkeypatch):
    import requests
    sc, base = server
    sc.plan = [(200, {}, b"ok"), (503, {}, b""), (200, {}, b"ok")]
    ft = FakeTime()
    pol = nh.policy_for(base + "/x")
    pol.limiter = nh.Limiter(rate=2.0, burst=1, clock=ft.clock, sleep=ft.sleep)
    s = requests.Session()
    nh.mount_polite(s)
    assert s.get(base + "/a").status_code == 200
    assert s.get(base + "/b").status_code == 503                                   # 0.5 s pacing, then the 503 costs 15 s
    assert s.get(base + "/c").status_code == 200
    assert ft.t - 100.0 >= 0.5 + nh.THROTTLE_PENALTY_S - 1                        # third request waited out the penalty
    rep = nh.STATS.report()
    assert "3 requests" in rep and "1 throttled" in rep
