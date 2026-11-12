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
import re
import tempfile
import threading
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


# ------------------------------------------------------------------------------------------ HTTP
_tls = threading.local()


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        retry = Retry(total=RETRIES, connect=RETRIES, read=RETRIES, backoff_factor=1.5, status_forcelist=(500, 502, 503, 504),
                      allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
        for scheme in ("https://", "http://"):
            s.mount(scheme, HTTPAdapter(max_retries=retry, pool_maxsize=8))
        _tls.s = s
    return s


def _get(url: str, headers=None):
    try:
        r = _session().get(url, headers=headers or {}, timeout=TIMEOUT)
    except Exception as e:                                           # noqa: BLE001 - re-raised with context
        raise FetchError(f"{url}: {type(e).__name__}: {e}") from e
    return r


def get_inventory(url: str) -> list[Record]:
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


# ------------------------------------------------------------------------------------------ helpers
ACC_RE = re.compile(r"(\d+)-(\d+) hour acc")


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
            out[(int(m.group(1)), int(m.group(2)))] = arr
    return out
