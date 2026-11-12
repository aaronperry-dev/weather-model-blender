"""Precipitation type.

PWPF documentation:
  * NCEP models:      dominant-type algorithm (Manikin 2005) applied to the model's categorical fields.
  * non-NCEP models:  "simple decision tree" using surface temperature and T at 925/850/700 hPa.
The decision tree below is our own simple implementation of that description (WPC's exact
thresholds are not published).
"""
from __future__ import annotations

import numpy as np

from .constants import TYPE_FZRA, TYPE_NONE, TYPE_RAIN, TYPE_SLEET, TYPE_SNOW


def decision_tree(t2m_c, aloft_max_c, snow_sfc_max_c: float = 1.5, frz_aloft_c: float = 2.0):
    """Return int8 type array (RAIN/SNOW/SLEET/FZRA) from 2 m temperature and the maximum
    temperature on the 925/850/700 hPa levels that are above ground (NaN if none)."""
    t2m = np.asarray(t2m_c)
    warm_aloft = np.nan_to_num(aloft_max_c, nan=-99.0) > 0.0
    out = np.full(t2m.shape, TYPE_RAIN, dtype=np.int8)
    out[~warm_aloft & (t2m <= snow_sfc_max_c)] = TYPE_SNOW
    frozen_sfc = warm_aloft & (t2m <= 0.0)
    out[frozen_sfc] = TYPE_SLEET
    out[frozen_sfc & (np.nan_to_num(aloft_max_c, nan=-99.0) > frz_aloft_c)] = TYPE_FZRA
    return out


def native_type(frac: dict, fallback: np.ndarray) -> np.ndarray:
    """Type from model categorical fractions {'rain','snow','sleet','fzra'} (0..1).
    Where the model flags nothing, use `fallback`."""
    keys = [("rain", TYPE_RAIN), ("snow", TYPE_SNOW), ("sleet", TYPE_SLEET), ("fzra", TYPE_FZRA)]
    best = np.zeros(fallback.shape, dtype=np.float32)
    out = np.full(fallback.shape, TYPE_NONE, dtype=np.int8)
    for name, code in keys:                       # later keys win ties -> fzra > sleet > snow > rain
        f = frac.get(name)
        if f is None:
            continue
        f = np.nan_to_num(np.asarray(f, dtype=np.float32), nan=0.0)
        m = (f >= best) & (f > 0)
        out[m] = code
        best = np.where(m, f, best)
    return np.where(out == TYPE_NONE, fallback, out).astype(np.int8)


def dominant_type(counts: np.ndarray) -> np.ndarray:
    """counts: (5, y, x) number of precipitating steps of each type.  Dominant = most frequent;
    ties resolved freezing rain > sleet > snow > rain (cautious ordering; an assumption)."""
    best = np.zeros(counts.shape[1:], dtype=np.int8)
    best_c = np.zeros(counts.shape[1:], dtype=counts.dtype)
    for code in (TYPE_RAIN, TYPE_SNOW, TYPE_SLEET, TYPE_FZRA):
        c = counts[code]
        m = (c >= best_c) & (c > 0)
        best[m] = code
        best_c = np.where(m, c, best_c)
    return best
