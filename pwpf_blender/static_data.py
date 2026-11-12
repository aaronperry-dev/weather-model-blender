"""One-off static inputs on the 2.5 km NDFD grid: terrain (RTMA) and PRISM monthly precipitation normals.

* Terrain: RTMA 2.5 km orography (same grid as NDFD).  Fetched with Herbie, or supply a .npy.
* PRISM: 1991-2020 monthly precipitation normals, 4 km, ESRI BIL in geographic coordinates, downloaded from
  the PRISM web service (URL is configurable because PRISM changed its services in Oct 2025) or read from
  a local folder of zipped BILs named like *_01_*.zip ... *_12_*.zip.
PRISM data are free to use; see PRISM's terms of use before redistributing the raw grids (this repo only
stores the derived ratio-ready array in the build cache, never in the published site).
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .grid import GridSpec

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


def terrain_from_rtma(grid_full: GridSpec, when: datetime | None = None) -> np.ndarray:
    from herbie import Herbie
    when = when or (datetime.utcnow() - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    for back in range(0, 12):
        t = when - timedelta(hours=back)
        try:
            H = Herbie(t, model="rtma", product="anl", verbose=False)
            ds = H.xarray(r":(?:HGT|ORO)[A-Za-z]*:surface", remove_grib=True, verbose=False)
            ds = ds[0] if isinstance(ds, list) else ds
            name = next(v for v in ds.data_vars if v in ("orog", "gh", "h"))
            arr = ds[name].values.astype(np.float32)
            if arr.shape != grid_full.shape:
                raise ValueError(f"RTMA terrain shape {arr.shape} != NDFD grid {grid_full.shape}")
            return arr
        except Exception:                                    # noqa: BLE001
            continue
    raise RuntimeError("could not fetch RTMA terrain; pass --terrain-npy with a (1377, 2145) float array in metres")


def prepare_static(out_dir: str | Path, terrain_npy: str | None = None, prism_dir: str | None = None,
                   prism_url: str = PRISM_URL, skip_existing: bool = True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = GridSpec.ndfd_2p5km(1)
    t_path, p_path = out / "terrain_2p5km.npy", out / "prism_ppt_monthly_2p5km.npy"
    if not (skip_existing and t_path.exists()):
        terr = np.load(terrain_npy) if terrain_npy else terrain_from_rtma(full)
        np.save(t_path, terr.astype(np.float32))
    if not (skip_existing and p_path.exists()):
        if prism_dir:
            zips = sorted(Path(prism_dir).glob("*.zip"), key=lambda p: int(re.findall(r"_(\d{2})[_.]", p.name)[-1]))
        else:
            zips = download_prism(out / "prism_raw", prism_url)
        np.save(p_path, prism_to_grid(zips, full))
    return t_path, p_path
