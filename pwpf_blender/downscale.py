"""PRISM-based QPF downscaling.

PWPF: "PRISM-based downscaling is applied to the QPF from each member."  PRISM supplies the
monthly precipitation climatology at ~800 m/4 km.  The downscaling factor is the ratio of that
fine climatology to the same climatology smoothed to the source model's resolution, so the factor
averages ~1 over a model grid cell and redistributes precipitation toward the climatologically
wetter (windward / high elevation) points.  The ratio limits are our assumption.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def interp_months(monthly: np.ndarray, doy: int) -> np.ndarray:
    """Linear-in-time interpolation of 12 monthly fields (Jan..Dec, mid-month centred)."""
    pos = ((doy - 15.5) / 30.4375) % 12.0
    m0 = int(np.floor(pos)) % 12
    m1 = (m0 + 1) % 12
    f = pos - np.floor(pos)
    return (1 - f) * monthly[m0] + f * monthly[m1]


class PrismDownscaler:
    def __init__(self, monthly_mm: np.ndarray | None, grid_dx_km: float,
                 ratio_min: float = 0.25, ratio_max: float = 4.0, smooth_factor: float = 1.0):
        self.monthly = monthly_mm
        self.grid_dx_km = grid_dx_km
        self.ratio_min, self.ratio_max = ratio_min, ratio_max
        self.smooth_factor = smooth_factor
        self._cache: dict = {}

    def ratio(self, doy: int, src_dx_km: float) -> np.ndarray | None:
        if self.monthly is None:
            return None
        key = (int(doy // 5), round(src_dx_km, 1))
        if key in self._cache:
            return self._cache[key]
        fine = interp_months(self.monthly, doy).astype(np.float32)
        valid = np.isfinite(fine) & (fine > 0.5)
        filled = np.where(valid, fine, 0.0)
        sigma_pix = max(0.5, self.smooth_factor * src_dx_km / self.grid_dx_km / 2.355)
        num = gaussian_filter(filled, sigma_pix, mode="nearest")
        den = gaussian_filter(valid.astype(np.float32), sigma_pix, mode="nearest")
        coarse = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
        r = np.where(valid & (coarse > 0.5), fine / np.maximum(coarse, 1e-3), 1.0)
        r = np.clip(r, self.ratio_min, self.ratio_max).astype(np.float32)
        self._cache[key] = r
        return r
