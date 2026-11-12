"""Warm-ground / marginal-air accumulation: how much of the snow that falls actually accumulates.

NOT part of WPC's published PWPF recipe (QPF x type x SLR) - it is our own, physically motivated extra, switchable with
`accumulation.enabled`, because the plain recipe piles up snow on warm November ground and in wet-bulb ~0 C air that
forecasters (who edit the WPC mode by hand) would not.  Two independent pieces:

1. Marginal-air efficiency (per 6-h bin).  The 0.5 C wet-bulb level (`sl`, m MSL) and the 2.5 km terrain give the
   wet-bulb temperature at the ground, Tw = 0.5 + lapse * (sl - terrain).  Snow falling at Tw <= wb_full_c accumulates
   fully; the efficiency falls linearly to min_eff at Tw >= wb_zero_c (wet snow melts on contact, settles, mixes with rain).

2. Warm-ground melt (cumulative over the run).  Soil that is still warm melts the first snow that lands on it.  The soil
   temperature at 10 cm is a climatology (latitude, day of year, elevation lapse); the first
   `ground_melt_in_per_c * T_soil` inches of snow of the run are lost, so a light event on warm lowland ground leaves
   nothing while a heavy event, or any event on cold high ground, is hardly touched.  Applied on the running total of each
   member, so 24/48/72 h windows stay consistent (their sums telescope).

All parameters are assumptions to calibrate (see the calibration report), not published values.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {"enabled": True, "wb_full_c": -1.0, "wb_zero_c": 1.5, "min_eff": 0.25, "wb_lapse_c_per_km": 6.0,
            "ground_melt_in_per_c": 0.15}


def settings(cfg: dict | None) -> dict:
    """Config block merged over DEFAULTS (used by the pipeline)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def active(cfg: dict | None) -> bool:
    return bool((cfg or {}).get("enabled", False))


def surface_wetbulb_c(sl_m, terrain_m, cfg: dict, wb_at_sl_c: float = 0.5):
    """Wet-bulb (C) at the ground: the snow level is where it equals wb_at_sl_c and it warms downward."""
    return wb_at_sl_c + float(cfg.get("wb_lapse_c_per_km", 6.0)) / 1000.0 * (np.asarray(sl_m, np.float32) - terrain_m)


def temp_efficiency(sl_m, terrain_m, cfg: dict, wb_at_sl_c: float = 0.5) -> np.ndarray:
    """0..1 multiplier from marginal air (1 = accumulates fully)."""
    tw = surface_wetbulb_c(sl_m, terrain_m, cfg, wb_at_sl_c)
    lo, hi = float(cfg.get("wb_full_c", -1.0)), float(cfg.get("wb_zero_c", 1.5))
    floor = float(cfg.get("min_eff", 0.25))
    frac = np.clip((tw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (1.0 - (1.0 - floor) * frac).astype(np.float32)


def soil_temp_c(doy: int, lat_deg, terrain_m=0.0) -> np.ndarray:
    """Climatological 10 cm soil temperature (C), clipped at 0: annual mean 41 - 0.75 lat, annual half-range 0.3 lat - 2,
    warmest ~3 August, minus 6.5 C/km of terrain height.  (Rough: ~7 C at 43 N on 12 Nov, ~1.5 C mid-December, 0 in January.)"""
    lat = np.asarray(lat_deg, np.float32)
    mean = 41.0 - 0.75 * lat
    amp = np.maximum(0.3 * lat - 2.0, 0.0)
    t = mean + amp * math.cos(2.0 * math.pi * (doy - 215) / 365.0) - 6.5e-3 * np.maximum(np.asarray(terrain_m, np.float32), 0.0)
    return np.maximum(t, 0.0).astype(np.float32)


def ground_melt_in(doy: int, lat_deg, terrain_m, cfg: dict) -> np.ndarray:
    """Inches of snow the warm ground removes from the first snow of the run (per grid cell)."""
    return (float(cfg.get("ground_melt_in_per_c", 0.15)) * soil_temp_c(doy, lat_deg, terrain_m)).astype(np.float32)


def apply_ground_melt(snow_bins: list, d0: np.ndarray, chunk_rows: int = 48) -> None:
    """In place: replace each bin's (n, ny, nx) snowfall by the increments of max(cumulative - d0, 0), per member."""
    if not snow_bins:
        return
    n, ny, nx = snow_bins[0].shape
    for r0 in range(0, ny, chunk_rows):
        r1 = min(ny, r0 + chunk_rows)
        d = d0[r0:r1][None]
        cum = np.zeros((n, r1 - r0, nx), np.float32)
        f_prev = np.zeros_like(cum)
        for sb in snow_bins:
            cum += sb[:, r0:r1].astype(np.float32)
            f = np.maximum(cum - d, 0.0)
            sb[:, r0:r1] = (f - f_prev).astype(sb.dtype)
            f_prev = f
