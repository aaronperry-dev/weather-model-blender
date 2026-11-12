"""Freezing-rain ice accumulation: the simplified Freezing Rain Accumulation Model (FRAM).

Sanders & Barjenbruch (2016, Wea. Forecasting 31, 1041-1060) regress the ice-to-liquid ratio (ILR) of
*flat* (elevated horizontal) ice on precipitation rate, wet-bulb temperature and wind speed.  The
precipitation-rate regression is the dominant term (Eq. 2):

    ILR_P = 0.1395 * P**-0.541           P = liquid precipitation rate, inches per hour
    flat ice   = ILR * liquid depth
    radial ice = 0.394 * flat ice        (Eq. 1, equivalent radial ice on a wire)

WPC's PWPF documentation describes a *simplified* FRAM driven by precipitation rate alone; that is what
is implemented here (no wind / wet-bulb terms, which would need extra fields from every data feed).
Caveats, all choices of this implementation rather than documented WPC behaviour:
  * P is floored at 0.02 in/h (the lower edge of the paper's training data) and ILR is capped at 1.0;
  * the rate is the model's *step-average* rate, so 3-6-hourly global-model steps give lower rates, hence
    higher ILR, than hourly CAM output would;
  * the paper flags P > 0.25 in/h as poorly sampled.
"""
from __future__ import annotations

import numpy as np

ILR_COEF = 0.1395
ILR_EXP = -0.541
P_MIN_IN_H = 0.02
ILR_MAX = 1.0
RADIAL_FACTOR = 0.394


def ilr_precip(rate_in_h) -> np.ndarray:
    """Flat-ice ice-to-liquid ratio from precipitation rate (in/h)."""
    p = np.maximum(np.asarray(rate_in_h, dtype=np.float32), P_MIN_IN_H)
    return np.minimum(ILR_COEF * p ** ILR_EXP, ILR_MAX).astype(np.float32)


def flat_ice_in(liquid_in, step_hours: float) -> np.ndarray:
    """Flat ice accretion (in) from the liquid depth (in) that fell as freezing rain during one model step."""
    liq = np.asarray(liquid_in, dtype=np.float32)
    return liq * ilr_precip(liq / max(step_hours, 1e-3))
