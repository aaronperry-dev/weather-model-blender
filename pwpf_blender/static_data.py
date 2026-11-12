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
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from . import noaa_http as nh
from .grid import GridSpec

log = logging.getLogger(__name__)

# PRISM moved to Cloud Optimized GeoTIFF and a new web service on 2025-10-01.  The new web service does NOT serve
# the 30-year normals ("They can be downloaded via FTP, or on the PRISM website"), so the normals are fetched from
# PRISM's web-browsable copy of its FTP tree.  File names inside that directory are discovered from the listing
# (and printed if they don't look like 12 monthly grids) instead of being hard-coded.
PRISM_URL = "https://data.prism.oregonstate.edu/normals/us/4km/ppt/"
# only used if a GeoTIFF carries no georeferencing tags: PRISM 4 km CONUS grid, cell-centre of the upper-left cell
PRISM_4KM_FALLBACK = {"ulx": -125.0208333333, "uly": 49.9375, "dx": 1 / 24, "dy": 1 / 24}
_DATA_EXT = (".zip", ".tif", ".tiff", ".bil")
_SKIP = re.compile(r"annual|_all|stn|\.xml|\.txt|\.csv|readme|\.prj|\.stx|\.info", re.I)


def _is_tiff(head: bytes) -> bool:
    return head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def read_bil_zip(path: str | Path):
    """Return (array[rows, cols] float32 with NaN nodata, header dict) from a zipped ESRI BIL.
    Header: ulx/uly = *corner* of the upper-left cell, dx/dy = cell size (degrees)."""
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
    dx, dy = float(h["XDIM"]), float(h["YDIM"])
    # ULXMAP/ULYMAP are the CENTRE of the upper-left cell
    return arr, {"ulx": float(h["ULXMAP"]) - dx / 2, "uly": float(h["ULYMAP"]) + dy / 2, "dx": dx, "dy": dy}


def read_geotiff(src):
    """Return (array float32 NaN nodata, header) from a single-band float GeoTIFF/COG (path or file object),
    using Pillow only.  Header as in read_bil_zip."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        arr = np.array(im, dtype=np.float32)
        tags = getattr(im, "tag_v2", {})
        scale, tie = tags.get(33550), tags.get(33922)
        nodata = tags.get(42113)
        keys = tags.get(34735)
    if arr.ndim != 2:
        raise ValueError(f"expected a single-band raster, got shape {arr.shape}")
    if scale and tie and len(tie) >= 6:
        dx, dy = float(scale[0]), float(scale[1])
        ulx, uly = float(tie[3]), float(tie[4])
        # GTRasterTypeGeoKey (1025) = 2 -> PixelIsPoint: the tie point is a cell centre
        if keys:
            for i in range(4, len(keys) - 3, 4):
                if keys[i] == 1025 and keys[i + 3] == 2:
                    ulx, uly = ulx - dx / 2, uly + dy / 2
        hdr = {"ulx": ulx, "uly": uly, "dx": dx, "dy": dy}
    else:
        log.warning("GeoTIFF has no georeferencing tags: assuming the standard PRISM 4 km CONUS grid")
        f = PRISM_4KM_FALLBACK
        hdr = {"ulx": f["ulx"] - f["dx"] / 2, "uly": f["uly"] + f["dy"] / 2, "dx": f["dx"], "dy": f["dy"]}
    try:
        nd = float(str(nodata).strip("\x00 ")) if nodata is not None else -9999.0
    except ValueError:
        nd = -9999.0
    arr[arr == nd] = np.nan
    arr[arr < 0] = np.nan
    return arr, hdr


def read_prism_raster(path: str | Path):
    """Read one PRISM monthly grid from a .zip (GeoTIFF or BIL inside), .tif/.tiff or zipped/loose BIL."""
    path = Path(path)
    head = path.read_bytes()[:4]
    if _is_tiff(head):
        return read_geotiff(path)
    if not zipfile.is_zipfile(path):
        raise nh.FetchError(f"{path.name}: neither a zip nor a TIFF (starts with {head!r})")
    with zipfile.ZipFile(path) as z:
        tif = next((n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))), None)
        if tif is not None:
            import io
            return read_geotiff(io.BytesIO(z.read(tif)))
    return read_bil_zip(path)


def prism_month(name: str):
    """Month (1-12) a PRISM normals file name refers to, or None.  Handles `..._01_bil.zip`, `..._01.tif`,
    `..._202001.tif` (YYYYMM) and similar; annual / all-months / metadata files give None."""
    n = name.rsplit("/", 1)[-1]
    if _SKIP.search(n) or not n.lower().endswith(_DATA_EXT):
        return None
    stem = re.sub(r"(\.(zip|tif|tiff|bil|asc|nc))+$", "", n, flags=re.I)
    stem = re.sub(r"_(bil|tif|tiff|asc|nc)$", "", stem, flags=re.I)
    runs = re.findall(r"(?<!\d)(\d{2}|\d{6})(?!\d)", stem)
    if not runs:
        return None
    m = int(runs[-1][-2:])
    return m if 1 <= m <= 12 else None


def pick_monthly(names: list[str]) -> dict[int, str]:
    """Choose one file per month from a directory listing; GeoTIFF preferred over zipped BIL."""
    rank = {".tif": 0, ".tiff": 0, ".zip": 1, ".bil": 2}
    best: dict[int, tuple[int, str]] = {}
    for nme in names:
        m = prism_month(nme)
        if m is None:
            continue
        r = rank[Path(nme).suffix.lower()]
        if m not in best or r < best[m][0]:
            best[m] = (r, nme)
    return {m: v[1] for m, v in best.items()}


def list_directory(url: str) -> list[tuple[str, str]]:
    """(file name, absolute URL) for every link of an Apache-style index page."""
    import html
    from urllib.parse import unquote, urljoin
    r = nh._session().get(url, timeout=nh.TIMEOUT)
    if r.status_code != 200:
        raise nh.FetchError(f"{url}: HTTP {r.status_code}")
    out = []
    for href in re.findall(r'href="([^"?#]+)"', r.text):
        href = html.unescape(href)
        if href.endswith("/") or href.startswith(".."):        # sub-directories and "parent directory"
            continue
        out.append((unquote(href.rsplit("/", 1)[-1]), urljoin(url, href)))
    return out


def download_prism(dest: Path, url: str = PRISM_URL) -> dict[int, Path]:
    """Download the 12 monthly ppt normals; returns {month: local file}.  `url` is either the directory listing
    of the normals folder or a template containing `{mm}` (kept for custom mirrors)."""
    dest.mkdir(parents=True, exist_ok=True)
    if "{mm" in url:
        targets = {mm: (f"prism_ppt_normal_{mm:02d}", url.format(mm=mm)) for mm in range(1, 13)}
    else:
        listing = list_directory(url if url.endswith("/") else url + "/")
        chosen = pick_monthly([n for n, _ in listing])
        urls = dict(listing)
        missing = [m for m in range(1, 13) if m not in chosen]
        if missing:
            raise nh.FetchError(f"{url}: could not identify monthly files for months {missing}. "
                                f"Listing has {len(listing)} links, first 40: {[n for n, _ in listing][:40]}")
        targets = {m: (chosen[m], urls[chosen[m]]) for m in range(1, 13)}
    files = {}
    for mm, (name, u) in targets.items():
        f = dest / f"{mm:02d}_{Path(name).name}"
        if not f.exists():
            log.info("PRISM month %02d: %s", mm, u)
            r = nh._session().get(u, timeout=(10, 180))
            if r.status_code != 200:
                raise nh.FetchError(f"{u}: HTTP {r.status_code}")
            head = r.content[:4]
            if not (_is_tiff(head) or head[:2] == b"PK"):
                raise nh.FetchError(f"{u}: not a zip/TIFF (starts with {r.content[:120]!r})")
            f.write_bytes(r.content)
        files[mm] = f
    return files


def prism_to_grid(files, grid: GridSpec) -> np.ndarray:
    """Bilinear-sample 12 PRISM monthly rasters (list in Jan..Dec order, or {month: path}) onto `grid`
    -> (12, ny, nx) float32 (NaN outside CONUS)."""
    paths = [files[m] for m in range(1, 13)] if isinstance(files, dict) else list(files)
    lon, lat = grid.lonlat()
    out = []
    for p in paths:
        arr, h = read_prism_raster(p)
        col = (lon - h["ulx"]) / h["dx"] - 0.5               # ulx/uly are the upper-left cell corner
        row = (h["uly"] - lat) / h["dy"] - 0.5
        filled = np.nan_to_num(arr, nan=0.0)
        valid = (~np.isnan(arr)).astype(np.float32)
        v = map_coordinates(filled, [row, col], order=1, mode="nearest")
        w = map_coordinates(valid, [row, col], order=1, mode="nearest")
        inside = (row >= -0.5) & (row <= arr.shape[0] - 0.5) & (col >= -0.5) & (col <= arr.shape[1] - 0.5)
        out.append(np.where((w > 0.5) & inside, v / np.maximum(w, 1e-3), np.nan).astype(np.float32))
    return np.stack(out)


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
                   prism_url: str = PRISM_URL, skip_existing: bool = True, require_prism: bool = False):
    """Write terrain and PRISM arrays on the full 2.5 km grid.  Returns (terrain path, PRISM path or None).
    A PRISM failure is logged (and annotated for GitHub Actions) but not fatal unless `require_prism`: without
    PRISM the pipeline simply skips QPF downscaling, which the site's manifest/notice then states."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = GridSpec.ndfd_2p5km(1)
    t_path, p_path = out / "terrain_2p5km.npy", out / "prism_ppt_monthly_2p5km.npy"
    if not (skip_existing and t_path.exists()):
        terr = np.load(terrain_npy) if terrain_npy else terrain_auto(full)
        np.save(t_path, terr.astype(np.float32))
    if not (skip_existing and p_path.exists()):
        try:
            if prism_dir:
                files = {}
                for f in Path(prism_dir).iterdir():
                    m = prism_month(f.name)
                    if m is not None and (m not in files or f.suffix.lower() in (".tif", ".tiff")):
                        files[m] = f
                if len(files) != 12:
                    raise nh.FetchError(f"{prism_dir}: found monthly files for months {sorted(files)}, need all 12 "
                                        "(.zip/.tif/.tiff with the month as _01 ... _12 or YYYYMM in the name)")
            else:
                files = download_prism(out / "prism_raw", prism_url)
            np.save(p_path, prism_to_grid(files, full))
        except Exception as e:                                   # noqa: BLE001
            if require_prism:
                raise
            msg = (f"PRISM normals unavailable ({type(e).__name__}: {e}). Continuing WITHOUT PRISM QPF downscaling; "
                   "see README 'PRISM' for how to supply the 12 monthly grids with --prism-dir.")
            log.warning(msg)
            print(f"::warning title=PRISM normals unavailable::{msg}")
            return t_path, None
    return t_path, p_path
