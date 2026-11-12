"""Binormal distribution fit used by WPC's PWPF (Toth & Szentimrey 1990).

PWPF documentation:
  * the WPC deterministic forecast is the MODE of the distribution
  * the WPC forecast's position among the ensemble order statistics sets the SKEWNESS
  * the VARIANCE is matched to the ensemble variance, with the WPC forecast included as an extra member
  * the fit is done separately at every grid point (method of moments)

A binormal has density  f(x) = sqrt(2/pi)/(s1+s2) * exp(-(x-m)^2 / (2 s_i^2)),  s_i = s1 for x<m, s2 for x>=m.
  P(X < m)  = s1/(s1+s2) = p          (set from the rank of the forecast in the ensemble)
  Var(X)    = (1-2/pi)(s2-s1)^2 + s1*s2        (Toth & Szentimrey 1990)
With s1 = p*s, s2 = (1-p)*s  =>  Var = s^2 [ (1-2/pi)(1-2p)^2 + p(1-p) ]  -> closed form for s.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

S_FLOOR = 1e-3   # inches; avoids a zero-width distribution


def fit(members: np.ndarray, mode: np.ndarray, tie_eps: float = 0.005, chunk_rows: int = 48):
    """members: (N, ny, nx) snowfall (in) - or a list of such arrays that are summed member-wise
    (e.g. two 24-h days -> 48 h) without materialising the sum; mode: (ny, nx) deterministic forecast (in).
    Returns dict(mode, s_lo, s_hi, std) with float32 arrays; std = member standard deviation."""
    parts = members if isinstance(members, (list, tuple)) else [members]
    n, ny, nx = parts[0].shape
    s_lo = np.zeros((ny, nx), np.float32)
    s_hi = np.zeros((ny, nx), np.float32)
    std = np.zeros((ny, nx), np.float32)
    lo_p, hi_p = 1.0 / (n + 1), n / (n + 1.0)
    for r0 in range(0, ny, chunk_rows):
        r1 = min(ny, r0 + chunk_rows)
        x = parts[0][:, r0:r1].astype(np.float32)
        for extra in parts[1:]:
            x = x + extra[:, r0:r1].astype(np.float32)
        md = mode[r0:r1]
        less = (x < md - tie_eps).sum(axis=0)
        eq = (np.abs(x - md) <= tie_eps).sum(axis=0)
        p = np.clip((less + 0.5 * eq) / n, lo_p, hi_p)
        mean_all = (x.sum(axis=0) + md) / (n + 1)
        var_all = (((x - mean_all) ** 2).sum(axis=0) + (md - mean_all) ** 2) / (n + 1)
        k2 = (1 - 2 / np.pi) * (1 - 2 * p) ** 2 + p * (1 - p)
        s = np.sqrt(var_all / k2)
        s_lo[r0:r1] = np.maximum(p * s, S_FLOOR)
        s_hi[r0:r1] = np.maximum((1 - p) * s, S_FLOOR)
        std[r0:r1] = x.std(axis=0)
    return {"mode": mode.astype(np.float32), "s_lo": s_lo, "s_hi": s_hi, "std": std}


def moments(m, s1, s2):
    """Mean and variance of the binormal."""
    mean = m + np.sqrt(2 / np.pi) * (s2 - s1)
    var = (1 - 2 / np.pi) * (s2 - s1) ** 2 + s1 * s2
    return mean, var


def exceedance(m, s1, s2, t):
    """P(X >= t)."""
    m, s1, s2 = (np.asarray(a, dtype=np.float64) for a in (m, s1, s2))
    s = s1 + s2
    right = 2 * s2 / s * (1 - ndtr((t - m) / s2))
    left = 1 - 2 * s1 / s * ndtr((t - m) / s1)
    return np.where(t >= m, right, left)


def percentile(m, s1, s2, q):
    """Value x with P(X <= x) = q  (q in 0..1), clipped at 0 (snowfall cannot be negative)."""
    m, s1, s2 = (np.asarray(a, dtype=np.float64) for a in (m, s1, s2))
    p = s1 / (s1 + s2)
    lo = m + s1 * ndtri(np.clip(q / (2 * p), 1e-12, 1 - 1e-12))
    hi = m + s2 * ndtri(np.clip(0.5 + (q - p) / (2 * (1 - p)), 1e-12, 1 - 1e-12))
    return np.maximum(np.where(q <= p, lo, hi), 0.0)
