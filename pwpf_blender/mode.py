"""The deterministic "mode" field.

In the real PWPF the mode is the human-edited 24-h snowfall forecast from the WPC Winter Weather
Desk (WWD), which is not machine-reproducible.  This module provides stand-ins, best first:

  external - read a deterministic grid you supply (e.g. NDFD snow amount, which carries the
             WWD/WFO collaborated forecast, or WPC's own deterministic grid if you have it)
  pmm      - probability-matched ensemble mean (Ebert 2001): the ensemble-mean *pattern* with the
             pooled member *amounts*, so peak amounts are not smoothed away; blended with the plain mean by
             `pmm_weight` (default 0.25) to temper PMM's overshoot where members disagree on placement
  mean     - (weighted) ensemble mean
"""
from __future__ import annotations

import numpy as np


def weighted_mean(members: np.ndarray, weights: np.ndarray | None = None, chunk_rows: int = 64) -> np.ndarray:
    n, ny, nx = members.shape
    w = np.ones(n, np.float32) if weights is None else np.asarray(weights, np.float32)
    w = w / w.sum()
    out = np.zeros((ny, nx), np.float32)
    for r0 in range(0, ny, chunk_rows):
        r1 = min(ny, r0 + chunk_rows)
        out[r0:r1] = np.tensordot(w, members[:, r0:r1].astype(np.float32), axes=(0, 0))
    return out


def probability_matched_mean(members: np.ndarray, weights: np.ndarray | None = None,
                             min_snow_in: float = 0.01) -> np.ndarray:
    n, ny, nx = members.shape
    mean = weighted_mean(members, weights)
    wet = mean > min_snow_in
    n_wet = int(wet.sum())
    out = np.zeros((ny, nx), np.float32)
    if n_wet == 0:
        return out
    pooled = members[:, wet].astype(np.float32).ravel()          # (n * n_wet)
    pooled = pooled[pooled > 0]
    pooled.sort()
    pooled = pooled[::-1]                                          # descending
    # every n-th pooled value, centred in each block of n
    pick = np.arange(n_wet) * n + n // 2
    vals = np.zeros(n_wet, np.float32)
    ok = pick < pooled.size
    vals[ok] = pooled[pick[ok]]
    order = np.argsort(-mean[wet], kind="stable")                  # wettest-first by mean
    tmp = np.zeros(n_wet, np.float32)
    tmp[order] = vals
    out[wet] = tmp
    return out


def compute_mode(members: np.ndarray, cfg: dict, weights=None, external: np.ndarray | None = None) -> np.ndarray:
    method = cfg.get("method", "pmm")
    if method == "external":
        if external is None:
            raise ValueError("mode.method=external but no external grid was provided")
        return np.asarray(external, np.float32)
    if method == "mean":
        return weighted_mean(members, weights)
    if method == "pmm":
        # Pure PMM keeps member peak amounts but can overshoot where members disagree on placement, so the
        # default blends it with the plain mean.  pmm_weight=1 -> pure PMM, 0 -> ensemble mean.
        w = float(cfg.get("pmm_weight", 0.25))
        mean = weighted_mean(members, weights)
        if w <= 0:
            return mean
        pm = probability_matched_mean(members, weights, cfg.get("min_snow_in", 0.01))
        return (w * pm + (1 - w) * mean).astype(np.float32)
    raise ValueError(f"unknown mode method {method!r}")
