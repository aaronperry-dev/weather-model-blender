"""Thermodynamics: wet-bulb temperature, surface height, and the WPC-style snow level.

WPC (PWPF documentation): "Each member's snow level is the height of the 0.5 degC wet-bulb
surface, found by a top-down search ... grid points [with terrain] above the snow level are
flagged as snow."
"""
from __future__ import annotations

import numpy as np

from .constants import SNOW_LEVEL_COLD, SNOW_LEVEL_WARM, WB_SNOW_THRESHOLD_C


def wetbulb_c(t_c, rh_pct):
    """Wet-bulb temperature (degC), Stull (2011) empirical formula (valid RH 5-99 %, T -20..50 C)."""
    t = np.asarray(t_c, dtype=np.float32)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float32), 1.0, 100.0)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def rh_from_t_td(t_c, td_c):
    """Relative humidity (%) from temperature and dew point (Magnus)."""
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))


def surface_height_from_levels(psfc_hpa, levels):
    """Estimate surface geopotential height (m) from surface pressure and pressure-level heights.

    levels: dict {p_hpa: {"z": ..., "t": ...}}.  Linear in ln(p) between bracketing levels;
    hypsometric extrapolation below the lowest level.
    """
    ps = np.asarray(psfc_hpa, dtype=np.float32)
    plev = sorted(levels, reverse=True)                      # 1000, 925, 850, ...
    z_out = np.full(ps.shape, np.nan, dtype=np.float32)
    # below the lowest level (e.g. surface pressure > 1000 hPa)
    p0 = plev[0]
    z0 = levels[p0]["z"]
    t0 = levels[p0]["t"] + 273.15
    below = ps >= p0
    z_out = np.where(below, z0 + 29.27 * t0 * np.log(p0 / np.maximum(ps, 1.0)), z_out)
    for pa, pb in zip(plev[:-1], plev[1:]):                  # pa > pb  (pa lower in atmosphere)
        m = (ps < pa) & (ps >= pb)
        if not m.any():
            continue
        f = (np.log(pa) - np.log(np.maximum(ps, 1.0))) / (np.log(pa) - np.log(pb))
        z = levels[pa]["z"] + f * (levels[pb]["z"] - levels[pa]["z"])
        z_out = np.where(m, z, z_out)
    top = plev[-1]
    z_out = np.where(ps < top, levels[top]["z"], z_out)
    return z_out.astype(np.float32)


def snow_level_height(levels, t2m_c, rh2m_pct, psfc_hpa, z_sfc_m, thr_c: float = WB_SNOW_THRESHOLD_C):
    """Height (m MSL) of the highest cold->warm crossing of the `thr_c` wet-bulb surface.

    Top-down search over [pressure levels above ground..., surface].  Returns SNOW_LEVEL_COLD where
    wet-bulb stays below the threshold to the surface, SNOW_LEVEL_WARM where the whole column is warm.
    """
    ps = np.asarray(psfc_hpa, dtype=np.float32)
    wb_sfc = wetbulb_c(t2m_c, rh2m_pct)
    z_sfc = np.asarray(z_sfc_m, dtype=np.float32) + 2.0

    stack_z, stack_wb = [], []
    for p in sorted(levels):                                  # ascending p = top first
        lv = levels[p]
        above_ground = p < ps
        wb = wetbulb_c(lv["t"], lv["rh"])
        # underground levels collapse onto the surface (zero-thickness layers, no false crossings)
        stack_z.append(np.where(above_ground, lv["z"], z_sfc))
        stack_wb.append(np.where(above_ground, wb, wb_sfc))
    stack_z.append(z_sfc)
    stack_wb.append(wb_sfc)

    result = np.full(ps.shape, np.nan, dtype=np.float32)
    found = np.zeros(ps.shape, dtype=bool)
    for k in range(len(stack_z) - 1):
        wa, wb_ = stack_wb[k], stack_wb[k + 1]
        za, zb = stack_z[k], stack_z[k + 1]
        cross = (wa < thr_c) & (wb_ >= thr_c) & ~found
        if cross.any():
            frac = (thr_c - wa) / np.where(wb_ - wa == 0, 1e-6, wb_ - wa)
            result = np.where(cross, za + frac * (zb - za), result)
            found |= cross
    result = np.where(~found & (wb_sfc < thr_c), SNOW_LEVEL_COLD, result)
    result = np.where(~found & (wb_sfc >= thr_c), SNOW_LEVEL_WARM, result)
    return np.clip(result, SNOW_LEVEL_COLD, SNOW_LEVEL_WARM).astype(np.float32)
