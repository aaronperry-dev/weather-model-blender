"""Snow-to-liquid ratio (SLR).

PWPF (current): "The SLR from the National Blend of Models (NBM), multiplied by an adjustment
factor, is used as the SLR for each ensemble member."  (The factor is not published.)
PWPF (legacy):  6-h SLR = average of [Roebber NN on NAM, Roebber NN on GFS, seasonal climatology, 11].

The NBM SLR and the Roebber network are not reproducible from public weights, so we provide:
  * "kuchera" - Kuchera et al. (2014): SLR from the maximum temperature below 500 hPa
  * "blend"   - legacy-style average:  0.5*Kuchera + 0.25*seasonal climatology + 0.25*11   (default)
  * "fixed"   - constant ratio
and a global `adjust` multiplier that can be calibrated against WPC's published percentiles
(`pwpf_blender validate --calibrate-slr`).
"""
from __future__ import annotations

import math

import numpy as np


def kuchera_slr(tmax_k):
    """Kuchera et al. (2014).  tmax_k: maximum temperature (K) in the column below 500 hPa."""
    tmax_k = np.asarray(tmax_k, dtype=np.float32)
    d = 271.16 - tmax_k
    return np.where(tmax_k > 271.16, 12.0 + 2.0 * d, 12.0 + d)


def climo_slr(day_of_year: int) -> float:
    """Smooth seasonal climatological SLR (assumption: ~13 mid-January, ~10 in summer)."""
    return 11.5 + 1.5 * math.cos(2 * math.pi * (day_of_year - 20) / 365.0)


def column_tmax_k(t2m_c, levels, psfc_hpa):
    """Max temperature (K) from the surface and all pressure levels above ground with p >= 500 hPa."""
    tmax = np.asarray(t2m_c, dtype=np.float32).copy()
    for p, lv in levels.items():
        if p < 500:
            continue
        t = np.where(p < psfc_hpa, lv["t"], np.nan)
        tmax = np.fmax(tmax, t)
    return tmax + 273.15


def slr_step(t2m_c, levels, psfc_hpa, day_of_year: int, cfg: dict) -> np.ndarray:
    method = cfg.get("method", "blend")
    lo, hi = float(cfg.get("min", 3.0)), float(cfg.get("max", 40.0))
    if method == "fixed":
        slr = np.full(np.shape(t2m_c), float(cfg.get("fixed", 11.0)), dtype=np.float32)
    else:
        k = kuchera_slr(column_tmax_k(t2m_c, levels, psfc_hpa))
        if method == "kuchera":
            slr = k
        elif method == "blend":
            slr = 0.5 * k + 0.25 * climo_slr(day_of_year) + 0.25 * float(cfg.get("fixed", 11.0))
        else:
            raise ValueError(f"unknown SLR method {method!r}")
    return (np.clip(slr, lo, hi) * float(cfg.get("adjust", 1.0))).astype(np.float32)
