"""Calibration report: where does our snowfall sit relative to the official forecasts, and which models drive it?

Logged at the end of every run (CI log), for a few periods:
  * a Northeast box: area covered at several amounts, mean and 99th percentile, for the blended mode and for each model
    group on its own (so a wet or warm-biased group shows up immediately);
  * reference points: NWS WFO "expected snowfall" values read off the official 72-h maps (7 AM EST Thu 12 Nov -> 7 AM EST
    Sun 15 Nov 2026), next to our mode / 50th percentile and the per-group means at those points.

Edit REFERENCE_POINTS (or config `calibration.points`) for a different event; the numbers are the *labels printed on the
official maps* (T = trace, entered as 0.05 in), not verification data.  Browser-side terrain adjustment is not included.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from . import binormal

log = logging.getLogger(__name__)

BOX = {"lat": (40.0, 47.0), "lon": (-80.0, -67.0)}
AMOUNTS = (0.1, 1.0, 2.0, 4.0, 6.0, 12.0)

# name, lat, lon, NWS 72-h expected snowfall (in) - labels from the Albany, Binghamton, Gray and Boston/Norton maps
REFERENCE_POINTS = [
    ("Hunter NY", 42.20, -74.22, 2.0), ("Oneonta NY", 42.45, -75.06, 1.0), ("Delhi NY", 42.28, -74.92, 1.0),
    ("Norwich NY", 42.53, -75.52, 1.0), ("Old Forge NY", 43.71, -74.97, 1.0), ("Warrensburg NY", 43.50, -73.77, 1.0),
    ("Manchester VT", 43.16, -73.07, 1.0), ("Utica NY", 43.10, -75.23, 0.5), ("Syracuse NY", 43.05, -76.15, 0.5),
    ("Binghamton NY", 42.10, -75.91, 0.5), ("Ithaca NY", 42.44, -76.50, 0.5), ("Gloversville NY", 43.05, -74.34, 0.5),
    ("Hancock NY", 41.95, -75.28, 0.5), ("Pittsfield MA", 42.45, -73.25, 0.5), ("Torrington CT", 41.80, -73.12, 0.5),
    ("Lebanon NH", 43.64, -72.25, 0.5), ("Pittsburg NH", 45.05, -71.40, 0.5), ("Fryeburg ME", 44.02, -70.98, 0.5),
    ("Albany NY", 42.65, -73.75, 0.05), ("Glens Falls NY", 43.31, -73.64, 0.05), ("Rutland VT", 43.61, -72.97, 0.05),
    ("Keene NH", 42.93, -72.28, 0.05), ("Augusta ME", 44.31, -69.78, 0.05), ("Boston MA", 42.36, -71.06, 0.0),
]


def _window_bins(wid: str, bin_h: int):
    m = re.fullmatch(r"s(\d+)_f(\d+)", wid)
    if not m:
        return None
    length, end = int(m.group(1)), int(m.group(2))
    return (end - length) // bin_h, end // bin_h


def report(res, runs, avail, snow_bins, lon, lat, bin_h: int, cfg: dict | None = None) -> str:
    """Text report; never raises (a diagnostics problem must not fail a run)."""
    cfg = cfg or {}
    box = cfg.get("box") or BOX
    points = cfg.get("points") or REFERENCE_POINTS
    wanted = cfg.get("windows") or ["s72_f072", "s24_f030"]
    mask = (lat >= box["lat"][0]) & (lat <= box["lat"][1]) & (lon >= box["lon"][0]) & (lon <= box["lon"][1])
    if not mask.any():
        return "calibration: reference box is outside the grid"
    rr, cc = np.where(mask)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    bm = mask[r0:r1, c0:c1]
    idx_pts = []
    for name, la, lo, ref in points:
        d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.radians(la))) ** 2
        idx_pts.append((name, ref, np.unravel_index(int(np.argmin(d)), d.shape)))
    out = ["", "=== calibration report (blended mode vs models vs NWS) ==="]
    for wid in wanted:
        kk = _window_bins(wid, bin_h)
        fit = res.windows.get(wid)
        if kk is None or fit is None:
            out.append(f"[{wid}] not produced")
            continue
        k0, k1 = kk
        members = [i for i in range(len(runs)) if avail[i, k0:k1].all()]
        tot = np.zeros((len(members), r1 - r0, c1 - c0), np.float32)
        pt = np.zeros((len(members), len(idx_pts)), np.float32)
        for k in range(k0, k1):
            sb = snow_bins[k]
            for j, i in enumerate(members):
                tot[j] += sb[i][r0:r1, c0:c1]
                for q, (_, _, (pr, pc)) in enumerate(idx_pts):
                    pt[j, q] += float(sb[i][pr, pc])
        groups: dict = {}
        for j, i in enumerate(members):
            groups.setdefault(runs[i].group, []).append(j)

        def stats(a):
            v = a[bm]
            return [100.0 * float((v >= t).mean()) for t in AMOUNTS] + [float(v.mean()), float(np.percentile(v, 99)), float(v.max())]
        hdr = "  ".join(f"%>={t:g}" for t in AMOUNTS)
        out.append(f"[{wid}] {len(members)} members; Northeast box {box['lat']} N, {box['lon']} E;  columns: {hdr}   mean  p99  max (in)")
        rows = [("BLEND (mode)", stats(fit["mode"][r0:r1, c0:c1]))]
        p50 = binormal.percentile(fit["mode"], fit["s_lo"], fit["s_hi"], 0.5)
        rows.append(("BLEND (p50)", stats(p50[r0:r1, c0:c1])))
        rows.append(("members, mean", stats(tot.mean(axis=0))))
        for g, js in sorted(groups.items()):
            rows.append((f"{g} (n={len(js)})", stats(tot[js].mean(axis=0))))
        for name, s in rows:
            out.append(f"   {name:<20}" + "  ".join(f"{x:5.1f}" for x in s[:len(AMOUNTS)]) + "   " + "  ".join(f"{x:5.2f}" for x in s[len(AMOUNTS):]))
        if wid.startswith("s72"):
            gnames = sorted(groups)
            out.append("   points vs the NWS 72-h labels on the official maps (inches; ours has no browser terrain adjustment):")
            out.append(f"      {'point':<16}{'NWS':>5}{'mode':>7}{'p50':>7}{'ens.mean':>9}  " + "".join(f"{g[:9]:>10}" for g in gnames))
            for q, (name, ref, (pr, pc)) in enumerate(idx_pts):
                gv = "".join(f"{float(pt[groups[g], q].mean()):>10.2f}" for g in gnames)
                out.append(f"      {name:<16}{ref:>5.2f}{float(fit['mode'][pr, pc]):>7.2f}{float(p50[pr, pc]):>7.2f}"
                           f"{float(pt[:, q].mean()):>9.2f}  {gv}")
            ratios = [float(fit["mode"][pr, pc]) / ref for _, ref, (pr, pc) in idx_pts if ref >= 0.5]
            if ratios:
                out.append(f"   mode / NWS at the {len(ratios)} points with >= 0.5 in:  median {np.median(ratios):.1f}x  "
                           f"(min {min(ratios):.1f}x, max {max(ratios):.1f}x)")
    return "\n".join(out)


def log_report(res, runs, avail, snow_bins, lon, lat, bin_h: int, cfg: dict | None = None) -> None:
    if (cfg or {}).get("enabled", True) is False:
        return
    try:
        log.info("%s", report(res, runs, avail, snow_bins, lon, lat, bin_h, cfg))
    except Exception as e:                                                   # noqa: BLE001
        log.warning("calibration report failed: %s", e)
