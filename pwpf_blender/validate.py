"""Compare our output with WPC's published PWPF 2.5 km GRIB2 files, and calibrate the SLR adjustment.

WPC file names (from the READ_ME in ftp.wpc.ncep.noaa.gov/pwpf/conus_2.5km):
  percentiles : 2.5kmprcntil_24hsnow_<05|10|25|50|75|90|95>pt_<yyyymmddcc>f<hhh>.grb   (snow in METRES)
  probability : psnow24_ge<threshold>_<yyyymmddcc>f<hhh>.grb
  sigma       : 2.5kmsigma_24h_snow_<yyyymmddcc>f<hhh>.grb
where cc = 00|12 is the cycle and hhh is the lead time at the END of the 24-h period.
Download the files for one cycle into a folder (e.g. with wget) and point --wpc-dir at it.
Requires xarray + cfgrib.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import binormal

M_TO_IN = 1.0 / 0.0254


def read_grib(path) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    name = next(iter(ds.data_vars))
    arr = np.squeeze(ds[name].values).astype(np.float32)
    lat = ds["latitude"].values
    if lat.ndim == 2 and lat[0, 0] > lat[-1, 0]:
        arr = arr[::-1]
    return np.nan_to_num(arr, nan=0.0)


def metrics(ours: np.ndarray, ref: np.ndarray, wet: float) -> dict:
    m = (ours > wet) | (ref > wet)
    if not m.any():
        return {"n": 0}
    d = ours[m] - ref[m]
    c = np.corrcoef(ours[m], ref[m])[0, 1] if m.sum() > 2 else float("nan")
    return {"n": int(m.sum()), "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "corr": float(c)}


def compare(npz_path: str, wpc_dir: str, window: str, cycle: datetime, end_fhr: int) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    stride = json.loads(str(z["meta"]))["stride"]
    m, lo, hi = z[f"{window}_mode"], z[f"{window}_s_lo"], z[f"{window}_s_hi"]
    stamp = cycle.strftime("%Y%m%d%H")
    d = Path(wpc_dir)
    report = {"window": window, "percentiles": {}, "probabilities": {}}
    for q in (5, 10, 25, 50, 75, 90, 95):
        f = d / f"2.5kmprcntil_24hsnow_{q:02d}pt_{stamp}f{end_fhr:03d}.grb"
        if f.exists():
            ref = read_grib(f)[::stride, ::stride] * M_TO_IN
            ours = binormal.percentile(m, lo, hi, q / 100.0)
            report["percentiles"][q] = metrics(ours, ref, 0.1)
    for f in sorted(d.glob(f"psnow24_ge*_{stamp}f{end_fhr:03d}.grb")):
        mt = re.search(r"_ge([0-9p.]+)_", f.name)
        if not mt:
            continue
        thr = float(mt.group(1).replace("p", "."))
        ref = read_grib(f)[::stride, ::stride]
        if ref.max() > 1.5:
            ref = ref / 100.0
        ours = binormal.exceedance(m, lo, hi, thr)
        mk = (ours > 0.05) | (ref > 0.05)
        report["probabilities"][thr] = {
            "mae_prob": float(np.abs(ours - ref)[mk].mean()) if mk.any() else None,
            "area_ge50_ours": int((ours >= 0.5).sum()), "area_ge50_wpc": int((ref >= 0.5).sum())}
    return report


def calibrate_slr(npz_path: str, wpc_dir: str, window: str, cycle: datetime, end_fhr: int, current_adjust: float):
    """Suggest slr.adjust: median ratio of WPC to our median (50th pct) snowfall where both exceed 1 in."""
    z = np.load(npz_path, allow_pickle=False)
    stride = json.loads(str(z["meta"]))["stride"]
    ours = binormal.percentile(z[f"{window}_mode"], z[f"{window}_s_lo"], z[f"{window}_s_hi"], 0.5)
    f = Path(wpc_dir) / f"2.5kmprcntil_24hsnow_50pt_{cycle:%Y%m%d%H}f{end_fhr:03d}.grb"
    ref = read_grib(f)[::stride, ::stride] * M_TO_IN
    m = (ours > 1.0) & (ref > 1.0)
    if m.sum() < 50:
        return None
    return float(current_adjust * np.median(ref[m] / ours[m]))
