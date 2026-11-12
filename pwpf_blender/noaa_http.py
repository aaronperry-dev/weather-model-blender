"""Direct NOAA data access: plain HTTPS + GRIB2 ``.idx`` byte ranges, AWS Open Data first, NOMADS second.

No wrapper libraries.  The recipe is the one every NOAA GRIB2 feed supports:

1. GET ``<grib url>.idx``  (a text inventory: ``num:byte_offset:d=YYYYMMDDHH:VAR:LEVEL:FCST:...``),
2. pick the records you need by matching ``VAR:LEVEL:FCST`` against regular expressions,
3. GET only those byte ranges (``Range: bytes=a-b``) and concatenate them - a valid GRIB2 file,
4. decode with ECMWF's eccodes.

Because the file we assemble is exactly the records we selected (plus any small gap-fill records), message
*i* of the file is idx record *i*; fields are therefore identified by NOAA's own inventory text
(``("TMP", "2 m above ground", "3 hour fcst")``) instead of by cfgrib/eccodes short names, which differ
between versions.  Every failure raises :class:`FetchError` with the URL, HTTP status and, when the inventory
was readable but had nothing we wanted, a sample of the inventory - nothing is swallowed.
"""
from __future__ import annotations

import logging
import os
import random
import re
import tempfile
from urllib.parse import urlencode
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

USER_AGENT = "pwpf-blender/0.2 (+https://github.com/aaronperry-dev/weather-model-blender)"
TIMEOUT = (15, 180)                      # connect, read (s)
RETRIES = 4                              # per request (diagnose lowers this)
MERGE_GAP = 128 * 1024                   # merge byte ranges separated by less than this (fewer requests)
MISSING = 1e30

Key = tuple[str, str, str]               # (VAR, LEVEL, FCST) exactly as in the .idx


class FetchError(RuntimeError):
    """A download/decoding problem; the message always names the URL and the reason."""


class HttpStatusError(FetchError):
    """The server answered with a non-success HTTP status (the file was not served)."""

    def __init__(self, msg: str, status: int):
        super().__init__(msg)
        self.status = status


class NoMatch(FetchError):
    """The inventory was read but no record matched the requested patterns."""

    def __init__(self, msg: str, records: list["Record"] | None = None):
        super().__init__(msg)
        self.records = records or []


# ------------------------------------------------------------------------------------------ inventory
@dataclass(frozen=True)
class Record:
    num: int
    start: int
    end: int | None                      # inclusive last byte, None = to end of file
    date: str
    var: str
    level: str
    fcst: str
    extra: str = ""

    @property
    def key(self) -> Key:
        return (self.var, self.level, self.fcst)

    @property
    def text(self) -> str:
        return f"{self.var}:{self.level}:{self.fcst}"


def parse_idx(text: str) -> list[Record]:
    """Parse a wgrib2-style inventory.  Records sharing a byte offset (sub-messages) are rejected."""
    raw = []
    for line in text.splitlines():
        p = line.strip().split(":")
        if len(p) < 6 or not p[1].isdigit():
            continue
        if not p[0].isdigit():
            raise FetchError(f"idx has multi-field message '{p[0]}' - not supported ({line!r})")
        raw.append((int(p[0]), int(p[1]), p[2][2:] if p[2].startswith("d=") else p[2], p[3], p[4], p[5],
                    ":".join(p[6:]).strip(":")))
    if not raw:
        raise FetchError("idx file is empty or not a wgrib2 inventory")
    raw.sort(key=lambda r: r[1])
    out = []
    for i, (num, start, date, var, level, fcst, extra) in enumerate(raw):
        nxt = next((r[1] for r in raw[i + 1:] if r[1] > start), None)
        out.append(Record(num, start, None if nxt is None else nxt - 1, date, var, level, fcst, extra))
    return out


def select_records(records: list[Record], patterns) -> list[Record]:
    pats = [re.compile(p) for p in patterns]
    return [r for r in records if any(p.search(r.text) for p in pats)]


def plan_ranges(records: list[Record], selected: list[Record], merge_gap: int = MERGE_GAP):
    """Group selected records into contiguous byte ranges.

    Returns ``[(start, end_or_None, [records covered in file order])]``; gap records inside a merged range
    are downloaded too (and ignored at decode time)."""
    sel = sorted(selected, key=lambda r: r.start)
    groups: list[list[Record]] = []
    for r in sel:
        if groups and groups[-1][-1].end is not None and r.start - groups[-1][-1].end - 1 <= merge_gap:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = []
    for g in groups:
        a, b = g[0].start, g[-1].end
        covered = [r for r in records if r.start >= a and (b is None or r.start <= b)]
        out.append((a, b, covered))
    return out


# ------------------------------------------------------------------------------------------ politeness
# Every HTTP request in this package goes through `_request`, which enforces per-host limits so that no server's
# published (or unpublished) rate limit is approached:
#   * NCEP (nomads / ftp / ftpprd): policy is 120 hits/minute per IP summed over those hosts, HEAD counted like GET
#     (NWS SCN 21-32).  We stay at <= 60/minute in total and 2 connections.
#   * AWS S3 open-data buckets: no published limit, but S3 answers 503 "SlowDown" when a prefix gets too many
#     requests; we cap at 20 requests/s (NOAA buckets; 10 for others) and back off on 503.
#   * PRISM: one file at a time, one request every 2 s (its web service refuses a file fetched twice in 24 h).
#   * ECMWF open data: far below its 500-simultaneous-connection limit.
# On 429/503 (and 403 from NCEP, whose firewall blocks over-limit IPs) we honour Retry-After or back off
# exponentially with jitter; after several consecutive throttles a host is left alone for COOLDOWN_S and requests
# fail fast so the caller can fall back to another mirror.  404 (file not published yet) is never retried.  The
# NWS asks clients not to retry a missing file immediately and to limit retries to 1-minute intervals; one run
# makes a single pass over each file.
RATE_SCALE = float(os.environ.get("PWPF_HTTP_RATE_SCALE", "1"))     # tests raise this; never lower rates below 1 in CI
COOLDOWN_S = 300.0
BREAKER_TRIP = 4                         # consecutive throttle responses before a host is left alone
BACKOFF_CAP_S = 90.0
RETRY_AFTER_CAP_S = 180.0


class Limiter:
    """Thread-safe token bucket; `acquire` blocks until the request may be made (clock/sleep are injectable)."""

    def __init__(self, rate: float, burst: int, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.burst = float(rate), float(burst)
        self._tokens, self._t = float(burst), clock()
        self._clock, self._sleep, self._lock = clock, sleep, threading.Lock()

    def acquire(self, n: float = 1.0) -> float:
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
            self._t = now
            self._tokens -= n
            wait = 0.0 if self._tokens >= 0 else -self._tokens / self.rate
        if wait > 0:
            self._sleep(wait)
        return wait

    def penalise(self, seconds: float, max_debt_s: float = 60.0) -> None:
        """The server pushed back: owe it `seconds` of silence.  Every thread sharing this bucket waits (the debt is
        in tokens), without any of them having to know about the others.  Debt never exceeds `max_debt_s`."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
            self._t = now
            self._tokens = max(self._tokens - seconds * self.rate, -max_debt_s * self.rate)


@dataclass
class Policy:
    key: str
    rate: float                          # requests per second, whole group
    burst: int
    max_conc: int
    throttle_codes: tuple = (429, 503)

    def __post_init__(self):
        self.limiter = Limiter(self.rate * RATE_SCALE, self.burst)
        self.sem = threading.BoundedSemaphore(self.max_conc)
        self.lock = threading.Lock()
        self.consecutive = 0
        self.blocked_until = 0.0


_NCEP_HOSTS = ("nomads.ncep.noaa.gov", "ftp.ncep.noaa.gov", "www.ftp.ncep.noaa.gov", "ftpprd.ncep.noaa.gov")
_PRISM_HOSTS = ("prism.oregonstate.edu", "data.prism.oregonstate.edu", "ftp.prism.oregonstate.edu", "services.nacse.org")
_policies: dict[str, Policy] = {}
_policies_lock = threading.Lock()


def policy_for(url: str) -> Policy:
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    if host in _NCEP_HOSTS:
        key, mk = "ncep", lambda: Policy("ncep", 1.0, 3, 2, (429, 503, 403))
    elif host in _PRISM_HOSTS:
        key, mk = "prism", lambda: Policy("prism", 0.5, 1, 1)
    elif host == "dd.weather.gc.ca":                  # MSC Datamart (no published rate limit): stay modest
        key, mk = "msc", lambda: Policy("msc", 2.0, 4, 2)
    elif host.startswith("ecmwf-forecasts.s3"):       # ECMWF open-data mirror: answered 503 under 4 parallel pulls
        key, mk = host, lambda: Policy(host, 4.0, 8, 2)
    elif host.startswith("noaa-") and host.endswith(".amazonaws.com"):     # NOAA open-data buckets
        key, mk = host, lambda: Policy(host, 20.0, 40, 6)
    elif host.endswith(".amazonaws.com"):
        key, mk = host, lambda: Policy(host, 10.0, 20, 4)
    elif host.endswith("ecmwf.int"):
        key, mk = "ecmwf", lambda: Policy("ecmwf", 2.0, 4, 2)
    else:
        key, mk = host, lambda: Policy(host, 2.0, 4, 2)
    with _policies_lock:
        if key not in _policies:
            _policies[key] = mk()
        return _policies[key]


class Stats:
    """Per-policy request/byte/throttle counters for the run log."""

    def __init__(self):
        self._lock = threading.Lock()
        self.d = defaultdict(lambda: {"requests": 0, "bytes": 0, "throttled": 0, "waited_s": 0.0, "t0": time.monotonic()})

    def add(self, key, **kw):
        with self._lock:
            row = self.d[key]
            for k, v in kw.items():
                row[k] += v

    def report(self) -> str:
        with self._lock:
            rows = []
            for k, r in sorted(self.d.items()):
                mins = max((time.monotonic() - r["t0"]) / 60.0, 1e-9)
                rows.append(f"{k}: {r['requests']} requests ({r['requests'] / mins:.0f}/min avg), "
                            f"{r['bytes'] / 1e6:.0f} MB, {r['throttled']} throttled, {r['waited_s']:.0f} s spent waiting")
            return "\n".join(rows) or "no HTTP requests made"


STATS = Stats()


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), RETRY_AFTER_CAP_S)
        except ValueError:
            pass
    return min(BACKOFF_CAP_S, 2.0 * 2 ** attempt) * random.uniform(0.85, 1.25)


def polite(url: str, cost: float = 1.0):
    """Context manager for code that makes its own HTTP calls (e.g. the ecmwf-opendata client): reserves `cost`
    requests from the host's rate limit and holds one of its connection slots."""
    pol = policy_for(url)

    class _Ctx:
        def __enter__(self_inner):
            pol.sem.acquire()
            w = pol.limiter.acquire(cost)
            STATS.add(pol.key, requests=int(round(cost)), waited_s=w)
            return pol

        def __exit__(self_inner, *a):
            pol.sem.release()
    return _Ctx()


# ------------------------------------------------------------------------------------------ HTTP
_tls = threading.local()


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        import requests
        from requests.adapters import HTTPAdapter
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        for scheme in ("https://", "http://"):                       # retries are handled in _request (rate-aware)
            s.mount(scheme, HTTPAdapter(max_retries=0, pool_maxsize=8))
        _tls.s = s
    return s


def _request(url: str, headers=None, stream: bool = False, timeout=None):
    """One polite GET: rate limit + connection cap + Retry-After/backoff + circuit breaker.  Returns the
    Response (any final status, e.g. 404/206); raises FetchError if the host stays unreachable or throttled."""
    pol = policy_for(url)
    last = "no attempt"
    for attempt in range(RETRIES + 1):
        now = time.monotonic()
        if pol.blocked_until > now:
            raise FetchError(f"{url}: {pol.key} is rate-limiting this client; leaving it alone for another "
                             f"{pol.blocked_until - now:.0f} s (last response: {last})")
        r = None
        with pol.sem:
            w = pol.limiter.acquire()
            try:
                r = _session().get(url, headers=headers or {}, timeout=timeout or TIMEOUT, stream=stream)
            except Exception as e:                                       # noqa: BLE001 - network-level failure
                last = f"{type(e).__name__}: {e}"
            STATS.add(pol.key, requests=1, waited_s=w)
        if r is None:                                                    # connection problem: back off, retry
            if attempt == RETRIES:
                break
            time.sleep(_backoff(attempt, None))
            continue
        if r.status_code in pol.throttle_codes:
            last = f"HTTP {r.status_code}"
            STATS.add(pol.key, throttled=1)
            with pol.lock:
                pol.consecutive += 1
                if pol.consecutive >= BREAKER_TRIP:
                    pol.blocked_until = time.monotonic() + COOLDOWN_S
                    log.warning("%s throttled us %d times in a row: pausing all requests to it for %.0f s",
                                pol.key, pol.consecutive, COOLDOWN_S)
                    pol.consecutive = 0
            wait = _backoff(attempt, r.headers.get("Retry-After"))
            r.close()
            if attempt == RETRIES:
                break
            log.info("%s answered %s; waiting %.0f s before retrying %s", pol.key, last, wait, url)
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 504):                             # transient server trouble, not a throttle
            last = f"HTTP {r.status_code}"
            r.close()
            if attempt == RETRIES:
                break
            time.sleep(_backoff(attempt, None))
            continue
        with pol.lock:
            pol.consecutive = 0
        if not stream:
            STATS.add(pol.key, bytes=len(r.content))
        return r
    raise FetchError(f"{url}: gave up after {RETRIES + 1} attempts (last: {last})")


THROTTLE_PENALTY_S = 8.0                # shared pause after one throttle answer to a client that makes its own requests


def mount_polite(session) -> None:
    """Route every request of a `requests` session (e.g. the one inside the ecmwf-opendata client, whose downloader
    issues one ranged GET per GRIB message) through the host's rate limit.  Each real request waits for a token
    and is counted in STATS; a 429/503 answer makes *all* threads of that host pause (`Limiter.penalise`), on top
    of the library's own retry sleep."""
    from requests.adapters import HTTPAdapter

    class PoliteAdapter(HTTPAdapter):
        def send(self, request, **kw):
            pol = policy_for(request.url)
            w = pol.limiter.acquire()
            STATS.add(pol.key, requests=1, waited_s=w)
            resp = super().send(request, **kw)
            if resp.status_code in pol.throttle_codes:
                STATS.add(pol.key, throttled=1)
                pol.limiter.penalise(THROTTLE_PENALTY_S)
            return resp

    for scheme in ("https://", "http://"):
        session.mount(scheme, PoliteAdapter(max_retries=0, pool_maxsize=8))


def _get(url: str, headers=None):
    try:
        return _request(url, headers)
    except FetchError:
        raise
    except Exception as e:                                               # noqa: BLE001 - re-raised with context
        raise FetchError(f"{url}: {type(e).__name__}: {e}") from e


def download_file(url: str, dest: Path, timeout=(15, 300), chunk: int = 1 << 20) -> int:
    """Stream `url` to `dest` (atomic), politely; returns bytes written.  FetchError on any non-200."""
    r = _request(url, stream=True, timeout=timeout)
    try:
        if r.status_code != 200:
            raise HttpStatusError(f"{url}: HTTP {r.status_code}", r.status_code)
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        with open(tmp, "wb") as f:
            for blk in r.iter_content(chunk):
                f.write(blk)
                n += len(blk)
        STATS.add(policy_for(url).key, bytes=n)
        os.replace(tmp, dest)
        return n
    finally:
        r.close()


_IDX_CACHE: "OrderedDict[str, list]" = OrderedDict()
_IDX_LOCK = threading.Lock()
_IDX_MAX = 2000


def get_inventory(url: str) -> list[Record]:
    """Parsed .idx of `url`; successful reads are remembered for the run so each file's inventory is fetched once."""
    with _IDX_LOCK:
        hit = _IDX_CACHE.get(url)
        if hit is not None:
            _IDX_CACHE.move_to_end(url)
            return hit
    recs = _get_inventory_uncached(url)
    with _IDX_LOCK:
        _IDX_CACHE[url] = recs
        while len(_IDX_CACHE) > _IDX_MAX:
            _IDX_CACHE.popitem(last=False)
    return recs


def _get_inventory_uncached(url: str) -> list[Record]:
    r = _get(url + ".idx")
    if r.status_code != 200:
        raise FetchError(f"{url}.idx: HTTP {r.status_code}")
    try:
        return parse_idx(r.text)
    except FetchError as e:
        raise FetchError(f"{url}.idx: {e}") from e


def get_inventory_first(urls):
    """(records, url) for the first candidate whose .idx can be read; FetchError lists every failure."""
    errs = []
    for u in urls:
        try:
            return get_inventory(u), u
        except FetchError as e:
            errs.append(str(e))
    raise FetchError("all sources failed:\n  " + "\n  ".join(errs))


def download_ranges(url: str, ranges, dest: Path):
    """Write the byte ranges of `url` back-to-back into `dest` (atomic)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for a, b, _ in ranges:
            hdr = {"Range": f"bytes={a}-{'' if b is None else b}"}
            r = _get(url, hdr)
            if r.status_code != 206:
                raise FetchError(f"{url}: HTTP {r.status_code} for range {hdr['Range']} (expected 206)")
            if b is not None and len(r.content) != b - a + 1:
                raise FetchError(f"{url}: short read for {hdr['Range']}: got {len(r.content)} bytes")
            if r.content[:4] != b"GRIB":
                raise FetchError(f"{url}: range {hdr['Range']} does not start with a GRIB message")
            f.write(r.content)
    os.replace(tmp, dest)


# ------------------------------------------------------------------------------------------ decoding
@dataclass
class GridInfo:
    lat: np.ndarray                      # 1-D (regular lat/lon) or 2-D
    lon: np.ndarray
    shape: tuple[int, int]
    grid_type: str


def _ec():
    try:
        import eccodes
        return eccodes
    except ImportError as e:                                         # pragma: no cover
        raise FetchError("eccodes is required to decode GRIB2 (pip install -r requirements-live.txt)") from e


def _shape(ec, gid) -> tuple[int, int]:
    for ny_key, nx_key in (("Nj", "Ni"), ("Ny", "Nx")):
        try:
            ny, nx = int(ec.codes_get(gid, ny_key)), int(ec.codes_get(gid, nx_key))
            if ny > 0 and nx > 0 and ny * nx == int(ec.codes_get(gid, "numberOfPoints")):
                return ny, nx
        except Exception:                                            # noqa: BLE001 - key not present
            continue
    raise FetchError("could not determine grid shape of GRIB message")


def grid_of(ec, gid) -> GridInfo:
    ny, nx = _shape(ec, gid)
    gt = ec.codes_get(gid, "gridType")
    lat = ec.codes_get_array(gid, "latitudes").reshape(ny, nx)
    lon = ec.codes_get_array(gid, "longitudes").reshape(ny, nx)
    if gt in ("regular_ll", "regular_gg"):
        lat, lon = lat[:, 0].copy(), lon[0, :].copy()
    return GridInfo(lat, lon, (ny, nx), gt)


def values_of(ec, gid) -> np.ndarray:
    ny, nx = _shape(ec, gid)
    ec.codes_set(gid, "missingValue", MISSING)
    v = ec.codes_get_values(gid).astype(np.float32).reshape(ny, nx)
    v[v >= MISSING * 0.99] = np.nan
    return v


def decode_records(path: Path, covered: list[Record], wanted: set, want_grid: bool, post=None):
    """Decode `path` whose messages correspond one-to-one to `covered`; keep only `wanted` records."""
    ec = _ec()
    arrays: dict[Key, np.ndarray] = {}
    grid = None
    with open(path, "rb") as f:
        for rec in covered:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                raise FetchError(f"{path.name}: fewer GRIB messages than idx records (stopped at {rec.text})")
            try:
                if want_grid and grid is None:
                    grid = grid_of(ec, gid)
                if rec in wanted:
                    v = values_of(ec, gid)
                    arrays[rec.key] = post(v) if post else v
            finally:
                ec.codes_release(gid)
        extra = ec.codes_grib_new_from_file(f)
        if extra is not None:
            ec.codes_release(extra)
            raise FetchError(f"{path.name}: more GRIB messages than idx records")
    return arrays, grid


def fetch_fields(urls, patterns, want_grid: bool = False, post=None, merge_gap: int = MERGE_GAP):
    """Download + decode the records matching `patterns` from the first of `urls` that works.

    Returns ``(fields, grid_or_None, url_used)`` where fields maps ``(VAR, LEVEL, FCST) -> float32 array``.
    Raises FetchError (NoMatch if every reachable inventory lacked the fields) listing every attempt."""
    attempts, all_no_match = [], True
    for url in urls:
        try:
            records = get_inventory(url)
            selected = select_records(records, patterns)
            if not selected:
                raise NoMatch(f"{url}: inventory has {len(records)} records but none match {list(patterns)}; "
                              f"first records: {[r.text for r in records[:12]]}", records)
            ranges = plan_ranges(records, selected, merge_gap)
            with tempfile.TemporaryDirectory(prefix="pwpf_dl_") as td:
                dest = Path(td) / "sel.grib2"
                download_ranges(url, ranges, dest)
                covered = [r for _, _, cov in ranges for r in cov]
                fields, grid = decode_records(dest, covered, set(selected), want_grid, post)
            return fields, grid, url
        except NoMatch as e:
            attempts.append(str(e))
        except FetchError as e:
            all_no_match = False
            attempts.append(str(e))
    cls = NoMatch if all_no_match and attempts else FetchError
    msg = "all sources failed:\n  " + "\n  ".join(attempts)
    if cls is NoMatch:
        raise NoMatch(msg)
    raise FetchError(msg)


def _lev_slug(level: str) -> str:
    """idx level text -> grib-filter parameter suffix: '2 m above ground' -> '2_m_above_ground'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", level).strip("_")


def fetch_fields_filtered(idx_url: str, spec, patterns, want_grid: bool = False, post=None):
    """Fetch the records matching `patterns` through an NCEP NOMADS *grib filter* CGI (server-side subsetting: one
    request returns just the chosen variables x levels) instead of one byte-range request per record block.

    `spec` = (cgi_url, dir, file).  The file's .idx tells which messages the filter returns (every record whose
    variable AND level are selected, in file order), so the response decodes through the same path as range
    downloads.  Returns (fields, grid, url).  FetchError if the CGI returns anything but GRIB or the message count
    differs from the inventory's prediction."""
    cgi, dirpath, fname = spec
    records = get_inventory(idx_url)
    selected = select_records(records, patterns)
    if not selected:
        raise NoMatch(f"{idx_url}: inventory has {len(records)} records but none match {list(patterns)}", records)
    vars_ = sorted({r.var for r in selected})
    levs = sorted({r.level for r in selected})
    covered = sorted((r for r in records if r.var in vars_ and r.level in levs), key=lambda r: r.start)
    q = {"dir": dirpath, "file": fname}
    q.update({f"var_{v}": "on" for v in vars_})
    q.update({f"lev_{_lev_slug(lv)}": "on" for lv in levs})
    url = cgi + "?" + urlencode(q)
    with tempfile.TemporaryDirectory(prefix="pwpf_dl_") as td:
        dest = Path(td) / "sel.grib2"
        download_file(url, dest)
        with open(dest, "rb") as f:
            head = f.read(200)
        if head[:4] != b"GRIB":
            raise FetchError(f"{url}: grib filter did not return GRIB: {head[:160]!r}")
        fields, grid = decode_records(dest, covered, set(selected), want_grid, post)
    return fields, grid, url


# ------------------------------------------------------------------------------------------ helpers
# wgrib2 writes accumulations of whole days as "0-1 day acc fcst" (HRRR/RRFS at f24, f48, ...), not "0-24 hour acc"
ACC_RE = re.compile(r"(\d+)-(\d+) (hour|day) acc")


def find(fields: dict, var: str, level: str, fcst_re: str | None = None):
    """First field with this VAR and LEVEL (optionally FCST matching a regex), else None."""
    for (v, lv, fc), arr in fields.items():
        if v == var and lv == level and (fcst_re is None or re.search(fcst_re, fc)):
            return arr
    return None


def acc_intervals(fields: dict) -> dict:
    """{(start_h, end_h): array} for every APCP accumulation record."""
    out = {}
    for (v, lv, fc), arr in fields.items():
        m = ACC_RE.search(fc)
        if v == "APCP" and m:
            k = 24 if m.group(3) == "day" else 1
            out[(int(m.group(1)) * k, int(m.group(2)) * k)] = arr
    return out
