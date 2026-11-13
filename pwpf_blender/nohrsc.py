"""NOHRSC National Snowfall Analysis (SNODAS-derived "sfav2") GRIB2 files: download and decode.

Files live at
    https://www.nohrsc.noaa.gov/snowfall_v2/data/<yyyymm>/sfav2_CONUS_<N>h_<yyyymmddhh>_grid184.grb2
where <N>h is the accumulation length (6, 24, 48, 72 ...) and <yyyymmddhh> the END (valid) time, UTC.  "grid184" is
the 2.5 km NDFD CONUS grid, i.e. exactly our full-resolution grid (2145 x 1377, south-west point first), values in
metres of snow, 9999 = no analysis (ocean, Canada, Mexico).

NOHRSC's analysis is the same dataset WPC calibrates the PWPF probabilities against, so these files serve both
verification (how good was an archived run?) and calibration (how should the spread/mode be scaled?).

Downloads go through noaa_http (one connection, 1 request/s, back-off on 429/503, `User-Agent` naming this project); a file
that is not published yet (404) is reported as `None`, never retried in a loop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import noaa_http as nh

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

BASE = "https://www.nohrsc.noaa.gov/snowfall_v2/data"
NY, NX = 1377, 2145
M_TO_IN = 1.0 / 0.0254
MISSING_FROM = 9998.0           # anything at or above this is "no data"
MIN_BYTES = 10_000              # a real file is ~1-3 MB; anything tiny is an error page


def url(valid: datetime, hours: int = 24, base: str = BASE) -> str:
    """URL of the analysis of `hours` ending at `valid` (UTC)."""
    return f"{base}/{valid:%Y%m}/sfav2_CONUS_{int(hours)}h_{valid:%Y%m%d%H}_grid184.grb2"


def local_name(valid: datetime, hours: int = 24) -> str:
    return f"sfav2_CONUS_{int(hours)}h_{valid:%Y%m%d%H}_grid184.grb2"


def fetch(valid: datetime, hours: int = 24, cache_dir: str | Path = "nohrsc_cache", base: str = BASE) -> Path | None:
    """Local path of the file, downloading it if needed.  None if NOHRSC has not published it (HTTP 404/403/410).
    A cached copy is never re-downloaded.  Network trouble raises nh.FetchError."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / local_name(valid, hours)
    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        return dest
    u = url(valid, hours, base)
    try:
        n = nh.download_file(u, dest)
    except nh.HttpStatusError as e:
        if e.status in (404, 403, 410):
            log.info("NOHRSC %dh analysis ending %s is not published (HTTP %d)", hours, valid, e.status)
            return None
        raise
    if n < MIN_BYTES or not _is_grib(dest):
        dest.unlink(missing_ok=True)
        raise nh.FetchError(f"{u}: not a GRIB2 file ({n} bytes)")
    log.info("NOHRSC %dh ending %s: %d bytes -> %s", hours, valid, n, dest)
    return dest


def _is_grib(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"GRIB"


def read_grib(path: str | Path) -> dict:
    """Decode one NOHRSC GRIB2 file -> {'inches': (1377, 2145) float32, NaN where there is no analysis (south-west first),
    'valid': datetime, 'hours': accumulation length}.  Checks the grid, so a different NOHRSC grid fails loudly."""
    try:
        import eccodes as ec
    except ImportError as e:                                         # pragma: no cover
        raise nh.FetchError("eccodes is required to decode GRIB2 (pip install -r requirements-live.txt)") from e
    with open(path, "rb") as fh:
        gid = ec.codes_grib_new_from_file(fh)
        if gid is None:
            raise nh.FetchError(f"{path}: no GRIB message")
        try:
            ni, nj = int(ec.codes_get(gid, "Ni")), int(ec.codes_get(gid, "Nj"))
            if (nj, ni) != (NY, NX):
                raise nh.FetchError(f"{path}: grid is {ni}x{nj}, expected the NDFD 2.5 km grid {NX}x{NY}")
            ec.codes_set(gid, "missingValue", 9999.0)
            v = ec.codes_get_values(gid).astype(np.float64).reshape(NY, NX)
            if int(ec.codes_get(gid, "jScansPositively")) == 0:        # stored north-first: flip to south-first
                v = v[::-1]
            if int(ec.codes_get(gid, "iScansNegatively")) == 1:
                v = v[:, ::-1]
            date, tm = int(ec.codes_get(gid, "validityDate")), int(ec.codes_get(gid, "validityTime"))
            try:
                hours = int(ec.codes_get(gid, "lengthOfTimeRange"))
            except Exception:                                        # noqa: BLE001
                hours = 0
        finally:
            ec.codes_release(gid)
    return {"inches": to_inches(v), "valid": datetime.strptime(f"{date:08d}{tm:04d}", "%Y%m%d%H%M"), "hours": hours}


def to_inches(metres: np.ndarray) -> np.ndarray:
    """metres of snow -> inches, NaN where the file says 'no data' (>= 9998) or is not finite; tiny negatives -> 0."""
    v = np.asarray(metres, np.float64)
    out = np.where(np.isfinite(v) & (v < MISSING_FROM), np.maximum(v, 0.0) * M_TO_IN, np.nan)
    return out.astype(np.float32)


def load(valid: datetime, hours: int = 24, cache_dir: str | Path = "nohrsc_cache", base: str = BASE) -> np.ndarray | None:
    """fetch + read_grib; the analysis in inches on the full 2.5 km grid, or None if not published yet."""
    p = fetch(valid, hours, cache_dir, base)
    if p is None:
        return None
    g = read_grib(p)
    if g["valid"] != valid.replace(minute=0, second=0, microsecond=0):
        log.warning("NOHRSC file %s is valid %s, expected %s", p, g["valid"], valid)
    return g["inches"]


def prune_cache(cache_dir: str | Path, keep_days: float, now: datetime | None = None) -> int:
    """Delete cached files whose valid time is older than keep_days; returns how many were removed."""
    now = now or _utcnow()
    n = 0
    for f in Path(cache_dir).glob("sfav2_CONUS_*h_??????????_grid184.grb2"):
        try:
            t = datetime.strptime(f.name.split("_")[3], "%Y%m%d%H")
        except (IndexError, ValueError):
            continue
        if now - t > timedelta(days=keep_days):
            f.unlink(missing_ok=True)
            n += 1
    return n
