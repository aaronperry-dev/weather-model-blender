"""Calibrating the blended distribution against the NOHRSC analysis.

WPC calibrates its PWPF probabilities statistically against NOHRSC snowfall analyses.  We do the same in the simplest
form that keeps the product a binormal (so the viewer, the percentiles and the probabilities stay consistent):

    mode'    = a    * mode
    sigma_lo' = k_lo * sigma_lo
    sigma_hi' = k_hi * sigma_hi

(a corrects the bias of the central value, k_lo / k_hi the lower / upper spread), fitted by minimising the mean quantile
(pinball) loss of the 5/10/25/50/75/90/95th percentiles over the archived forecast/analysis pairs (`verify.sample_pairs`:
cells where the forecast gave >= 2 % for 0.1 in, picked from the forecast alone, so the sample is not biased by what was
observed).  Only pairs made with the *current science settings* (same fingerprint) are used.

The fit is written to `calibration_fit.json`.  It is applied to the products only when `verification.apply_calibration: true`
AND the fit has enough independent cycles AND it improves the loss by at least `min_gain` AND its fingerprint matches the
run.  Parameters are clamped to a narrow range so a few odd storms cannot distort the guidance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import binormal

log = logging.getLogger(__name__)

QUANTILES = tuple(round(0.05 * i, 2) for i in range(1, 20))      # 5 %, 10 %, ... 95 %
A_RANGE = (0.6, 1.6)
K_RANGE = (0.6, 2.5)
MIN_GAIN = 0.01          # relative loss improvement needed
MIN_CELLS = 20000


def pinball(params, m, lo, hi, y) -> float:
    a, kl, kh = params
    m2, lo2, hi2 = m * a, np.maximum(lo * kl, binormal.S_FLOOR), np.maximum(hi * kh, binormal.S_FLOOR)
    tot = 0.0
    for q in QUANTILES:
        e = y - binormal.percentile(m2, lo2, hi2, q)
        tot += float(np.mean(np.maximum(q * e, (q - 1.0) * e)))
    return tot / len(QUANTILES)


def load_pairs(archive_dir: str | Path, fingerprint: str):
    """(mode, s_lo, s_hi, obs, cycles) pooled over the pair files of that fingerprint."""
    arrs, cycles = [[], [], [], []], set()
    for f in sorted((Path(archive_dir) / "pairs").glob("pairs_*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("fingerprint") != fingerprint:
            continue
        for i, k in enumerate(("mode", "s_lo", "s_hi", "obs")):
            arrs[i].append(z[k].astype(np.float64))
        cycles.add(meta["cycle"])
    if not cycles:
        return None
    return (*(np.concatenate(a) for a in arrs), sorted(cycles))


def fit(archive_dir: str | Path, fingerprint: str, min_cycles: int = 8, max_cells: int = 150000) -> dict:
    """Fit (a, k_lo, k_hi); always returns a dict (`usable` tells whether it should be applied)."""
    p = load_pairs(archive_dir, fingerprint)
    out = {"fingerprint": fingerprint, "a": 1.0, "k_lo": 1.0, "k_hi": 1.0, "usable": False, "cycles": 0, "cells": 0}
    if p is None:
        out["reason"] = "no archived pairs for these settings yet"
        return out
    m, lo, hi, y, cycles = p
    out.update(cycles=len(cycles), cells=int(m.size))
    if len(cycles) < min_cycles:
        out["reason"] = f"{len(cycles)} cycles of {min_cycles} needed"
        return out
    if m.size < MIN_CELLS:
        out["reason"] = f"{m.size} wet cells of {MIN_CELLS} needed"
        return out
    if m.size > max_cells:
        sel = np.random.default_rng(1).choice(m.size, max_cells, replace=False)
        m, lo, hi, y = m[sel], lo[sel], hi[sel], y[sel]
    base = pinball((1.0, 1.0, 1.0), m, lo, hi, y)

    def f(x):
        a, kl, kh = np.exp(x)
        return pinball((np.clip(a, *A_RANGE), np.clip(kl, *K_RANGE), np.clip(kh, *K_RANGE)), m, lo, hi, y)
    # a wide starting simplex matters: the loss has a long shallow valley (mode vs spread trade off) and the default tiny
    # simplex stops on its slope with parameters ~10 % off
    simplex = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], float)
    r = minimize(f, np.zeros(3), method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-9, "maxiter": 400, "initial_simplex": simplex})
    a, kl, kh = np.exp(r.x)
    a, kl, kh = float(np.clip(a, *A_RANGE)), float(np.clip(kl, *K_RANGE)), float(np.clip(kh, *K_RANGE))
    after = pinball((a, kl, kh), m, lo, hi, y)
    gain = (base - after) / base if base > 0 else 0.0
    out.update(a=round(a, 4), k_lo=round(kl, 4), k_hi=round(kh, 4), loss_before=round(base, 5), loss_after=round(after, 5),
               gain=round(gain, 4), usable=bool(gain >= MIN_GAIN),
               reason="" if gain >= MIN_GAIN else f"gain {100 * gain:.1f} % is below {100 * MIN_GAIN:.0f} %")
    return out


def write_fit(fit_result: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fit_result, indent=1))


def describe(r: dict) -> str:
    if not r.get("cells"):
        return f"calibration fit: {r.get('reason', 'nothing to fit')}"
    s = (f"calibration fit on {r['cycles']} cycles / {r['cells']} wet cells: mode x {r['a']:.3f}, sigma_low x {r['k_lo']:.3f}, "
         f"sigma_high x {r['k_hi']:.3f}")
    if "loss_before" in r:
        s += f"; quantile loss {r['loss_before']:.4f} -> {r['loss_after']:.4f} ({100 * r['gain']:.1f} % better)"
    return s + ("" if r["usable"] else f"  [not usable: {r.get('reason')}]")


def apply(res, fit_path: str | Path, fingerprint: str) -> str:
    """Scale every window of `res` in place with the stored fit when it is usable and made for these settings.
    Returns a one-line description of what happened (also useful for the log)."""
    p = Path(fit_path)
    if not p.exists():
        return "calibration: no fit file yet, products are uncalibrated"
    r = json.loads(p.read_text())
    if r.get("fingerprint") != fingerprint:
        return "calibration: the stored fit was made for other science settings, products are uncalibrated"
    if not r.get("usable"):
        return f"calibration: stored fit is not usable ({r.get('reason')}), products are uncalibrated"
    a, kl, kh = float(r["a"]), float(r["k_lo"]), float(r["k_hi"])
    for w in res.windows.values():
        w["mode"] = (w["mode"] * a).astype(np.float32)
        w["s_lo"] = np.maximum(w["s_lo"] * kl, binormal.S_FLOOR).astype(np.float32)
        w["s_hi"] = np.maximum(w["s_hi"] * kh, binormal.S_FLOOR).astype(np.float32)
    res.meta["calibration_applied"] = {"a": a, "k_lo": kl, "k_hi": kh, "cycles": r.get("cycles"), "cells": r.get("cells")}
    return f"calibration applied: mode x {a:.3f}, sigma_low x {kl:.3f}, sigma_high x {kh:.3f} (fit on {r.get('cycles')} cycles)"
