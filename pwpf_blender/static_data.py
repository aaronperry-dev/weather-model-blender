"""One-off static inputs on the 2.5 km NDFD grid: terrain (RTMA) and PRISM monthly precipitation normals.

* Terrain: RTMA 2.5 km surface height (same grid as NDFD), fetched directly from AWS/NOMADS with .idx byte
  ranges.  If the RTMA inventory has no terrain record, falls back to RRFS 3 km model orography interpolated to
  the NDFD grid (smoother, logged as a warning).  Or supply your own (1377, 2145) .npy via --terrain-npy.
* PRISM: 1991-2020 monthly precipitation normals, 4 km, ESRI BIL in geographic coordinates, downloaded from
  the PRISM web service (URL is configurable because PRISM changed its services in Oct 2025) or read from
  a local folder of zipped BILs named like *_01_*.zip ... *_12_*.zip.
PRISM data are free to use; see PRISM's terms of use before redistributing the raw grids (this repo only
stores the derived ratio-ready array in the build cache, never in the published site).
"""
from __future__ import annotations

import logging
import re
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from . import noaa_http as nh
from .grid import GridSpec

log = logging.getLogger(__name__)

PRISM_URL = "http://services.nacse.org/prism/data/public/normals/4km/ppt/{mm:02d}"


def read_bil_zip(path: str | Path):
    """Return (array[rows, cols] float32 with NaN nodata, header dict) from a zipped ESRI BIL."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        bil = next(n for n in names if n.lower().endswith(".bil"))
        hdr = next(n for n in names if n.lower().endswith(".hdr"))
        h = {}
        for line in z.read(hdr).decode().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                h[parts[0].upper()] = parts[1]
        nrows, ncols = int(h["NROWS"]), int(h["NCOLS"])
        dtype = "<f4" if h.get("BYTEORDER", "I") == "I" else ">f4"
        arr = np.frombuffer(z.read(bil), dtype=dtype).reshape(nrows, ncols).astype(np.float32)
    nodata = float(h.get("NODATA", -9999))
    arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan
    return arr, {"ulx": float(h["ULXMAP"]), "uly": float(h["ULYMAP"]),
                 "dx": float(h["XDIM"]), "dy": float(h["YDIM"])}


def prism_to_grid(zip_paths: list[Path], grid: GridSpec) -> np.ndarray:
    """Bilinear-sample 12 PRISM monthly rasters onto `grid` -> (12, ny, nx) float32 (NaN outside CONUS)."""
    lon, lat = grid.lonlat()
    out = []
    for p in zip_paths:
        arr, h = read_bil_zip(p)
        col = (lon - h["ulx"]) / h["dx"] - 0.5
        row = (h["uly"] - lat) / h["dy"] - 0.5
        filled = np.nan_to_num(arr, nan=0.0)
        valid = (~np.isnan(arr)).astype(np.float32)
        v = map_coordinates(filled, [row, col], order=1, mode="nearest")
        w = map_coordinates(valid, [row, col], order=1, mode="nearest")
        out.append(np.where(w > 0.5, v / np.maximum(w, 1e-3), np.nan).astype(np.float32))
    return np.stack(out)


def download_prism(dest: Path, url_template: str = PRISM_URL) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for mm in range(1, 13):
        f = dest / f"prism_ppt_normal_4km_{mm:02d}.zip"
        if not f.exists():
            urllib.request.urlretrieve(url_template.format(mm=mm), f)
        files.append(f)
    return files


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


TERRAIN_PAT = (r"^(?:HGT|ORO)[A-Za-z]*:surface:",)
RTMA_NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod"


def rtma_urls(t: datetime) -> tuple[str, ...]:
    d, h = f"{t:%Y%m%d}", f"{t:%H}"
    name = f"rtma2p5.t{h}z.2dvaranl_ndfd.grb2_wexp"
    return (f"https://noaa-rtma-pds.s3.amazonaws.com/rtma2p5.{d}/{name}", f"{RTMA_NOMADS}/rtma2p5.{d}/{name}")


class GridMismatch(ValueError):
    """The decoded grid cannot be mapped onto the NDFD CONUS grid (deterministic: retrying other hours won't help)."""


def _wrap_lon(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def _fit_to_ndfd(arr: np.ndarray, info, grid_full: GridSpec, tol_deg: float = 0.02) -> np.ndarray:
    """Return `arr` on the NDFD CONUS window.

    RTMA ``*_ndfd.grb2_wexp`` files are on an *expanded* NDFD grid (same projection and spacing, extra rows and
    columns around the CONUS grid; 1597 x 2345 in 2026), so the CONUS window is located by latitude/longitude.  Also
    handles rows stored north-to-south.  Raises GridMismatch if the geometry does not line up within `tol_deg`."""
    if info is None or info.lat.ndim != 2:
        if arr.shape != grid_full.shape:
            raise GridMismatch(f"terrain shape {arr.shape} != NDFD grid {grid_full.shape} and no 2-D lat/lon to match")
        return arr
    lon_t, lat_t = grid_full.lonlat()
    lat, lon = info.lat, _wrap_lon(info.lon)
    if lat[0, 0] > lat[-1, 0]:                                    # north-first scan: flip to south-first
        arr, lat, lon = arr[::-1], lat[::-1], lon[::-1]
    ny, nx = grid_full.shape
    coslat = np.cos(np.radians(lat_t[0, 0]))
    d2 = (lat - lat_t[0, 0]) ** 2 + (_wrap_lon(lon - _wrap_lon(lon_t[0, 0])) * coslat) ** 2
    i0, j0 = (int(v) for v in np.unravel_index(np.argmin(d2), d2.shape))
    if i0 + ny > arr.shape[0] or j0 + nx > arr.shape[1]:
        raise GridMismatch(f"NDFD window ({ny}x{nx}) starting at source index ({i0},{j0}) does not fit the "
                           f"decoded grid {arr.shape}")
    err_lat = float(np.abs(lat[i0:i0 + ny, j0:j0 + nx] - lat_t).max())
    err_lon = float(np.abs(_wrap_lon(lon[i0:i0 + ny, j0:j0 + nx] - lon_t)).max())
    if max(err_lat, err_lon) > tol_deg:
        raise GridMismatch(f"decoded grid {arr.shape} does not line up with the NDFD grid at offset ({i0},{j0}): "
                           f"max lat error {err_lat:.4f} deg, lon error {err_lon:.4f} deg (tolerance {tol_deg})")
    if arr.shape != grid_full.shape:
        log.info("terrain: cropped %s source grid to NDFD window at offset (rows %d, cols %d)", arr.shape, i0, j0)
    return arr[i0:i0 + ny, j0:j0 + nx]


def _check_terrain(arr: np.ndarray, grid_full: GridSpec, min_finite: float = 0.95) -> np.ndarray:
    """Validate shape and plausibility; NaN (outside the source domain) becomes 0 m."""
    if arr.shape != grid_full.shape:
        raise ValueError(f"terrain shape {arr.shape} != NDFD grid {grid_full.shape}")
    ok = np.isfinite(arr)
    if ok.mean() < min_finite or np.nanmin(arr) < -200 or np.nanmax(arr) > 6000:
        raise ValueError(f"terrain values implausible: finite={ok.mean():.2f} (need >= {min_finite}) "
                         f"min={np.nanmin(arr):.0f} max={np.nanmax(arr):.0f}")
    return np.where(ok, arr, 0.0).astype(np.float32)


def terrain_from_rtma(grid_full: GridSpec, when: datetime | None = None, lookback_h: int = 12) -> np.ndarray:
    """RTMA 2.5 km surface height.  Raises with the real reason (HTTP status / inventory sample) on failure."""
    when = when or (_utcnow() - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    errors = []
    for back in range(lookback_h):
        t = when - timedelta(hours=back)
        try:
            f, grid, url = nh.fetch_fields(rtma_urls(t), TERRAIN_PAT, want_grid=True)
            arr = _check_terrain(_fit_to_ndfd(next(iter(f.values())), grid, grid_full), grid_full)
            log.info("terrain: RTMA %s (%s)", url, next(iter(f)))
            return arr
        except nh.NoMatch:
            raise                                       # inventory readable but no terrain record: retrying won't help
        except GridMismatch as e:
            raise nh.FetchError(f"RTMA terrain grid cannot be mapped to the NDFD grid: {e}") from e
        except (nh.FetchError, ValueError) as e:
            errors.append(f"{t:%Y-%m-%d %HZ}: {e}")
    raise nh.FetchError("RTMA terrain unavailable for the last %d hours:\n  %s" % (lookback_h, "\n  ".join(errors[:6])))


def terrain_from_rrfs(grid_full: GridSpec, when: datetime | None = None) -> np.ndarray:
    """Fallback: RRFS 3 km `HGT:surface` interpolated (inverse-distance, 4 neighbours) to the NDFD grid."""
    from .regrid import Regridder
    from .sources.noaa_sources import plan_rrfs
    when = when or _utcnow().replace(minute=0, second=0, microsecond=0)
    errors = []
    for back in range(0, 36, 6):
        init = (when - timedelta(hours=back)).replace(hour=((when - timedelta(hours=back)).hour // 6) * 6)
        plan = plan_rrfs(init, 0, None)[0]
        try:
            f, grid, url = nh.fetch_fields(plan.urls, (r"^HGT:surface:",), want_grid=True)
        except nh.FetchError as e:
            errors.append(f"{init:%Y-%m-%d %HZ}: {e}")
            continue
        z = next(iter(f.values()))
        lon, lat = grid_full.lonlat()
        out = Regridder.build(grid.lat, grid.lon, lat, lon)(z)
        log.warning("terrain: using RRFS 3 km model orography from %s (smoother than RTMA; RTMA terrain was unusable)", url)
        return _check_terrain(np.where(np.isfinite(out), out, np.nan), grid_full, min_finite=0.80)
    raise nh.FetchError("RRFS terrain unavailable:\n  " + "\n  ".join(errors[:6]))


def terrain_auto(grid_full: GridSpec) -> np.ndarray:
    """RTMA terrain; if RTMA cannot supply it (no terrain record, outage) fall back to RRFS orography.
    If both fail the error carries both reasons."""
    try:
        return terrain_from_rtma(grid_full)
    except nh.FetchError as e1:
        log.warning("RTMA terrain unusable, falling back to RRFS orography:\n%s", e1)
        try:
            return terrain_from_rrfs(grid_full)
        except nh.FetchError as e2:
            raise nh.FetchError(f"no terrain source worked.\nRTMA: {e1}\nRRFS: {e2}\n"
                                "Pass --terrain-npy with a (1377, 2145) float array in metres.") from e2


def prepare_static(out_dir: str | Path, terrain_npy: str | None = None, prism_dir: str | None = None,
                   prism_url: str = PRISM_URL, skip_existing: bool = True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = GridSpec.ndfd_2p5km(1)
    t_path, p_path = out / "terrain_2p5km.npy", out / "prism_ppt_monthly_2p5km.npy"
    if not (skip_existing and t_path.exists()):
        terr = np.load(terrain_npy) if terrain_npy else terrain_auto(full)
        np.save(t_path, terr.astype(np.float32))
    if not (skip_existing and p_path.exists()):
        if prism_dir:
            zips = sorted(Path(prism_dir).glob("*.zip"), key=lambda p: int(re.findall(r"_(\d{2})[_.]", p.name)[-1]))
        else:
            zips = download_prism(out / "prism_raw", prism_url)
        np.save(p_path, prism_to_grid(zips, full))
    return t_path, p_path
