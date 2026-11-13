"""Verification of archived forecasts against the NOHRSC snowfall analysis, and the case ledger it builds.

Flow (all driven by `python -m pwpf_blender verify-archive`, which the CI workflow runs after every build):

  1. `run --archive-dir archive` stores, per cycle, the RAW (uncalibrated) 24 h fields (mode, sigma_low, sigma_high) with a
     fingerprint of the science settings that produced them (`archive_results`).
  2. For every archived 24 h period whose end time is far enough in the past, the NOHRSC analysis for that end time is fetched
     (once; cached) and the forecast is scored over the Northeast box and over all CONUS cells NOHRSC covers (`score_case`).
  3. The scores are *additive counts* (so any set of cases can be merged exactly): amounts bias/MAE/RMSE, categorical hits
     per threshold for the mode and the median, percentile coverage, Brier sums + reliability bins, and a PIT histogram.
     They are appended to `ledger.json`; a text/JSON summary grouped by settings fingerprint is written for the site log
     (`summarize`), so an A/B of two settings (e.g. accumulation on/off) is two groups in the same table.
  4. A subsample of forecast/observation pairs is kept (`pairs/`) for the calibration fit in `recalibrate.py`.

Verification is on the forecast's own grid (the archive's stride); NOHRSC is subsampled to match.  Small amounts in
the analysis are uncertain (it is partly model-based, with limited station data), so read the 0.1 in scores with that in mind.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import binormal, calibration, nohrsc
from .grid import GridSpec

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

THRESHOLDS = (0.1, 1.0, 2.0, 4.0, 6.0)
BRIER_THRESHOLDS = (1.0, 4.0)
REL_EDGES = (0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0001)
PIT_BINS = 10
WET_P = 0.02                      # a cell is "forecast-wet" when P(>= 0.1 in) is at least this (forecast-only selection)
DEFAULTS = {"enabled": True, "obs_latency_h": 3, "archive_days": 14, "pair_keep_days": 180, "pairs_per_window": 6000,
            "max_fetch_per_run": 40, "max_tries": 6, "min_cases": 8, "apply_calibration": False}


def settings(cfg: dict | None) -> dict:
    return dict(DEFAULTS, **(cfg or {}))


# ---------------------------------------------------------------------------------------------- fingerprint
def fingerprint(cfg: dict) -> tuple[str, dict]:
    """Short hash + readable dict of the settings that change what the forecast fields mean (calibration is excluded)."""
    acc = cfg.get("accumulation") or {}
    d = {"accumulation": acc if acc.get("enabled", True) else "off",
         "ptype": cfg.get("ptype"), "slr": cfg.get("slr"), "mode": {k: v for k, v in (cfg.get("mode") or {}).items() if k != "external"},
         "downscale": cfg.get("downscale"), "snow_level": cfg.get("snow_level"),
         "stride": (cfg.get("grid") or {}).get("stride", 1)}
    h = hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return h, d


def label_of(d: dict) -> str:
    acc = d.get("accumulation")
    a = "accumulation off" if acc == "off" else "accumulation on (melt %.2f in/C, min_eff %.2f)" % (
        acc.get("ground_melt_in_per_c", 0.15), acc.get("min_eff", 0.25))
    m = d.get("mode") or {}
    return f"{a}; mode {m.get('method', 'pmm')}" + (f" w={m.get('pmm_weight')}" if m.get("method", "pmm") == "pmm" else "") + \
        f"; slr adjust {(d.get('slr') or {}).get('adjust', 1.0)}; stride {d.get('stride')}"


# ---------------------------------------------------------------------------------------------- archive
def archive_results(res, cfg: dict, archive_dir: str | Path) -> Path | None:
    """Store the raw 24 h fields of this run; prune cycles older than verification.archive_days."""
    vc = settings(cfg.get("verification"))
    d = Path(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrs, wins = {}, {}
    for wid, w in res.windows.items():
        if int(w.get("length_h", 0)) != 24:
            continue
        for k in ("mode", "s_lo", "s_hi"):
            arrs[f"{wid}_{k}"] = w[k].astype(np.float16)
        wins[wid] = [w["start"].isoformat(), w["end"].isoformat()]
    if not wins:
        return None
    fp, fd = fingerprint(cfg)
    arrs["meta"] = np.array(json.dumps({"cycle": res.cycle.isoformat(), "stride": res.grid.stride, "windows": wins,
                                         "fingerprint": fp, "settings": fd, "label": label_of(fd),
                                         "n_members": res.meta.get("n_members")}, default=str))
    path = d / f"fc_{res.cycle:%Y%m%d%H}.npz"
    np.savez_compressed(path, **arrs)
    for f in d.glob("fc_??????????.npz"):
        try:
            t = datetime.strptime(f.stem[3:], "%Y%m%d%H")
        except ValueError:
            continue
        if res.cycle - t > timedelta(days=vc["archive_days"]):
            f.unlink(missing_ok=True)
    log.info("archived %d 24 h periods of cycle %s (settings %s: %s) -> %s", len(wins), res.cycle, fp, label_of(fd), path)
    return path


# ---------------------------------------------------------------------------------------------- scoring
def _pct(m, lo, hi):
    return {q: binormal.percentile(m, lo, hi, q / 100.0) for q in (10, 50, 90)}


def _score(m, lo, hi, obs, mask) -> dict:
    """Additive counts for the cells in `mask` (see module docstring)."""
    o = obs[mask].astype(np.float64)
    md, l, h = (a[mask].astype(np.float64) for a in (m, lo, hi))
    pc = _pct(md, l, h)
    fcs = {"mode": md, "p50": pc[50]}
    out = {"n": int(o.size), "sum_obs": float(o.sum()), "obs_ge": {str(t): int((o >= t).sum()) for t in THRESHOLDS}}
    for k, x in fcs.items():
        d = x - o
        wet = (x >= 0.1) | (o >= 0.1)
        xw, ow = x[wet], o[wet]
        out[k] = {"sum": float(x.sum()), "abs": float(np.abs(d).sum()), "sq": float((d ** 2).sum()),
                  "wet": {"n": int(wet.sum()), "sx": float(xw.sum()), "sy": float(ow.sum()), "sxx": float((xw ** 2).sum()),
                          "syy": float((ow ** 2).sum()), "sxy": float((xw * ow).sum())},
                  "cat": {str(t): [int(((x >= t) & (o >= t)).sum()), int(((x >= t) & (o < t)).sum()),
                                   int(((x < t) & (o >= t)).sum())] for t in THRESHOLDS},
                  "fc_ge": {str(t): int((x >= t).sum()) for t in THRESHOLDS}}
    wet = (pc[50] >= 0.1) | (o >= 0.1)
    out["cover"] = {"n": int(wet.sum()), **{f"p{q}": int((o[wet] <= pc[q][wet]).sum()) for q in (10, 50, 90)}}
    out["brier"] = {}
    for t in BRIER_THRESHOLDS:
        p = binormal.exceedance(md, l, h, t)
        ev = (o >= t).astype(np.float64)
        bins = []
        for a, b in zip(REL_EDGES[:-1], REL_EDGES[1:]):
            s = (p >= a) & (p < b)
            bins.append([int(s.sum()), float(p[s].sum()), float(ev[s].sum())])
        out["brier"][str(t)] = {"n": int(o.size), "sum_bs": float(((p - ev) ** 2).sum()), "sum_ev": float(ev.sum()), "bins": bins}
    # PIT over forecast-wet cells (randomised below the mass at zero), for the spread/bias check
    pw = binormal.exceedance(md, l, h, 0.1) >= WET_P
    if pw.any():
        s1, s2 = l[pw], h[pw]
        u = 1.0 - binormal.exceedance(md[pw], s1, s2, o[pw])
        zero = o[pw] <= 0
        if zero.any():
            f0 = 1.0 - binormal.exceedance(md[pw], s1, s2, 0.0)
            u = np.where(zero, f0 * np.random.default_rng(0).random(u.shape), u)
        out["pit"] = np.histogram(np.clip(u, 0, 1 - 1e-9), bins=PIT_BINS, range=(0, 1))[0].astype(int).tolist()
    else:
        out["pit"] = [0] * PIT_BINS
    return out


def _domains(stride: int):
    g = GridSpec.ndfd_2p5km(stride)
    lon, lat = g.lonlat()
    b = calibration.BOX
    ne = (lat >= b["lat"][0]) & (lat <= b["lat"][1]) & (lon >= b["lon"][0]) & (lon <= b["lon"][1])
    return lon, lat, ne


def score_case(m, lo, hi, obs, stride: int) -> dict:
    """{'ne': counts, 'conus': counts} for one forecast period; obs already on the forecast grid (NaN = no analysis)."""
    _, _, ne = _domains(stride)
    valid = np.isfinite(obs)
    out = {}
    for name, mk in (("ne", ne & valid), ("conus", valid)):
        if mk.sum() > 0:
            out[name] = _score(m, lo, hi, obs, mk)
    return out


def sample_pairs(m, lo, hi, obs, n: int, seed: int = 0):
    """Random sample of forecast-wet, analysed cells -> (mode, s_lo, s_hi, obs) float32 arrays (forecast-only selection, so
    the sample is conditionally unbiased for calibration)."""
    ok = np.isfinite(obs) & (binormal.exceedance(m, lo, hi, 0.1) >= WET_P)
    idx = np.flatnonzero(ok.ravel())
    if idx.size > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return tuple(a.ravel()[idx].astype(np.float32) for a in (m, lo, hi, obs))


# ---------------------------------------------------------------------------------------------- ledger
def _merge(a, b):
    """Sum two count structures of identical shape."""
    if isinstance(a, dict):
        return {k: _merge(a[k], b[k]) if k in b else a[k] for k in a} | {k: b[k] for k in b if k not in a}
    if isinstance(a, list):
        return [_merge(x, y) for x, y in zip(a, b)]
    return a + b


def merge_cases(cases: list[dict], domain: str):
    acc = None
    for c in cases:
        s = c["scores"].get(domain)
        if s is not None:
            acc = s if acc is None else _merge(acc, s)
    return acc


def load_ledger(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("ledger %s is unreadable; starting a new one", p)
    return {"version": 1, "cases": [], "missing": {}}


def save_ledger(led: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(led, separators=(",", ":")))


def verify_archive(cfg: dict, archive_dir: str | Path, cache_dir: str | Path, now: datetime | None = None,
                   base: str = nohrsc.BASE) -> dict:
    """Score every archived 24 h period that NOHRSC has analysed and that is not in the ledger yet.  Returns the ledger."""
    vc = settings(cfg.get("verification"))
    now = now or _utcnow()
    arch = Path(archive_dir)
    led = load_ledger(arch / "ledger.json")
    done = {(c["cycle"], c["window"]) for c in led["cases"]}
    fetched = 0
    (arch / "pairs").mkdir(parents=True, exist_ok=True)
    for f in sorted(arch.glob("fc_??????????.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        stride = int(meta["stride"])
        for wid, (_, end) in sorted(meta["windows"].items(), key=lambda kv: kv[1][1]):
            key = (meta["cycle"], wid)
            end_t = datetime.fromisoformat(end)
            if key in done or now < end_t + timedelta(hours=vc["obs_latency_h"]):
                continue
            mk = f"{meta['cycle']}|{wid}"
            if led["missing"].get(mk, 0) >= vc["max_tries"] or fetched >= vc["max_fetch_per_run"]:
                continue
            try:
                full = nohrsc.load(end_t, 24, cache_dir, base)
                fetched += 1
            except Exception as e:                                           # noqa: BLE001 - one bad file must not stop the rest
                log.warning("NOHRSC %s: %s", end_t, e)
                led["missing"][mk] = led["missing"].get(mk, 0) + 1
                continue
            if full is None:
                led["missing"][mk] = led["missing"].get(mk, 0) + 1
                continue
            obs = full[::stride, ::stride]
            m, lo, hi = (z[f"{wid}_{k}"].astype(np.float32) for k in ("mode", "s_lo", "s_hi"))
            if obs.shape != m.shape:
                log.warning("%s %s: analysis grid %s != forecast grid %s", meta["cycle"], wid, obs.shape, m.shape)
                continue
            sc = score_case(m, lo, hi, obs, stride)
            led["cases"].append({"cycle": meta["cycle"], "window": wid, "start": meta["windows"][wid][0], "end": end,
                                 "fingerprint": meta["fingerprint"], "label": meta["label"], "stride": stride, "scores": sc})
            done.add(key)
            led["missing"].pop(mk, None)
            pr = sample_pairs(m, lo, hi, obs, int(vc["pairs_per_window"]))
            np.savez_compressed(arch / "pairs" / f"pairs_{meta['cycle'][:13].replace('-', '').replace('T', '')}_{wid}.npz",
                                mode=pr[0].astype(np.float16), s_lo=pr[1].astype(np.float16), s_hi=pr[2].astype(np.float16),
                                obs=pr[3].astype(np.float16),
                                meta=np.array(json.dumps({"fingerprint": meta["fingerprint"], "cycle": meta["cycle"],
                                                          "window": wid, "end": end})))
            ne = sc.get("ne") or sc.get("conus")
            log.info("verified %s %s: %d cells, obs mean %.3f in, mode mean %.3f in", meta["cycle"], wid, ne["n"],
                     ne["sum_obs"] / max(ne["n"], 1), ne["mode"]["sum"] / max(ne["n"], 1))
    # forget pair files and ledger detail that are older than pair_keep_days
    cutoff = now - timedelta(days=vc["pair_keep_days"])
    for pf in (arch / "pairs").glob("pairs_*.npz"):
        try:
            if datetime.strptime(pf.name.split("_")[1], "%Y%m%d%H") < cutoff:
                pf.unlink(missing_ok=True)
        except ValueError:
            pass
    led["cases"] = [c for c in led["cases"] if datetime.fromisoformat(c["end"]) >= cutoff]
    save_ledger(led, arch / "ledger.json")
    nohrsc.prune_cache(cache_dir, float(vc["archive_days"]) + 2, now)
    return led


# ---------------------------------------------------------------------------------------------- reporting
def derived(s: dict) -> dict:
    """Readable statistics from merged counts."""
    n = max(s["n"], 1)
    out = {"cells": s["n"], "obs_mean": s["sum_obs"] / n}
    for k in ("mode", "p50"):
        d = s[k]
        w = d["wet"]
        r = float("nan")
        if w["n"] > 2:
            cov = w["sxy"] / w["n"] - (w["sx"] / w["n"]) * (w["sy"] / w["n"])
            vx = w["sxx"] / w["n"] - (w["sx"] / w["n"]) ** 2
            vy = w["syy"] / w["n"] - (w["sy"] / w["n"]) ** 2
            r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        out[k] = {"mean": d["sum"] / n, "bias": (d["sum"] - s["sum_obs"]) / n, "mae": d["abs"] / n, "rmse": float(np.sqrt(d["sq"] / n)),
                  "corr_wet": float(r), "cat": {}}
        for t in THRESHOLDS:
            a, b, c = d["cat"][str(t)]
            if a + c == 0:
                continue
            out[k]["cat"][t] = {"pod": a / (a + c), "far": b / max(a + b, 1), "csi": a / max(a + b + c, 1),
                                "freq_bias": (a + b) / (a + c), "obs_cells": a + c}
    cv = s["cover"]
    out["coverage"] = {q: cv[f"p{q}"] / max(cv["n"], 1) for q in (10, 50, 90)} | {"n": cv["n"]}
    out["brier"] = {}
    for t, b in s["brier"].items():
        base = b["sum_ev"] / max(b["n"], 1)
        bsc = base * (1 - base)
        bs = b["sum_bs"] / max(b["n"], 1)
        out["brier"][float(t)] = {"bs": bs, "bss": (1 - bs / bsc) if bsc > 0 else float("nan"), "obs_freq": base,
                                  "bins": [{"lo": REL_EDGES[i], "hi": min(REL_EDGES[i + 1], 1.0), "n": x[0],
                                            "fc": x[1] / x[0] if x[0] else None, "obs": x[2] / x[0] if x[0] else None}
                                           for i, x in enumerate(b["bins"])]}
    tot = max(sum(s["pit"]), 1)
    out["pit"] = [x / tot for x in s["pit"]]
    return out


def summarize(led: dict, now: datetime | None = None) -> tuple[dict, str]:
    """(json summary, text) grouped by settings fingerprint, Northeast box and CONUS."""
    groups: dict = {}
    for c in led["cases"]:
        groups.setdefault(c["fingerprint"], []).append(c)
    js = {"generated": (now or _utcnow()).isoformat(timespec="minutes"), "groups": []}
    lines = ["=== verification against the NOHRSC snowfall analysis (24 h periods) ===",
             "Counts are pooled over cases; the Northeast box is " + f"{calibration.BOX['lat']} N, {calibration.BOX['lon']} E.",
             "NOHRSC small amounts (< ~0.5 in) are uncertain; judge the 0.1 in scores accordingly.", ""]
    if not groups:
        lines.append("no archived period has been analysed yet (NOHRSC files are fetched after the period ends).")
    for fp, cases in sorted(groups.items(), key=lambda kv: kv[1][0]["end"]):
        cycles = sorted({c["cycle"] for c in cases})
        g = {"fingerprint": fp, "label": cases[0]["label"], "cases": len(cases), "cycles": len(cycles),
             "first": cycles[0], "last": cycles[-1], "domains": {}}
        lines += [f"[settings {fp}] {cases[0]['label']}", f"   {len(cases)} periods from {len(cycles)} cycles ({cycles[0][:13]} .. {cycles[-1][:13]})"]
        for dom in ("ne", "conus"):
            m = merge_cases(cases, dom)
            if m is None:
                continue
            d = derived(m)
            g["domains"][dom] = d
            lines.append(f"   {'Northeast box' if dom == 'ne' else 'CONUS':<14} cells {d['cells']}  obs mean {d['obs_mean']:.3f} in")
            lines.append(f"      {'':6}{'mean':>7}{'bias':>7}{'MAE':>7}{'RMSE':>7}{'r(wet)':>8}   " +
                         "  ".join(f"CSI>={t:g}" for t in THRESHOLDS) + "   freq.bias>=" + "/".join(f"{t:g}" for t in (1.0, 4.0)))
            for k in ("mode", "p50"):
                x = d[k]
                csi = "  ".join(f"{x['cat'][t]['csi']:7.2f}" if t in x["cat"] else f"{'-':>7}" for t in THRESHOLDS)
                fb = "/".join(f"{x['cat'][t]['freq_bias']:.2f}" if t in x["cat"] else "-" for t in (1.0, 4.0))
                lines.append(f"      {k:<6}{x['mean']:7.3f}{x['bias']:7.3f}{x['mae']:7.3f}{x['rmse']:7.3f}{x['corr_wet']:8.2f}   {csi}   {fb}")
            cov = d["coverage"]
            lines.append(f"      obs below p10/p50/p90 (wet cells, ideal 10/50/90 %): {100 * cov[10]:.0f} / {100 * cov[50]:.0f} / {100 * cov[90]:.0f}  (n={cov['n']})")
            for t, b in d["brier"].items():
                lines.append(f"      P(>={t:g} in): Brier skill {b['bss']:.2f}; reliability (forecast % -> observed %): " +
                             ", ".join(f"{100 * x['fc']:.0f}->{100 * x['obs']:.0f}" for x in b["bins"] if x["n"] >= 50 and x["fc"] is not None))
            lines.append("      PIT histogram, forecast-wet cells (flat = calibrated; U-shape = too narrow; high first bin = forecast too high): " +
                         " ".join(f"{100 * x:.0f}" for x in d["pit"]))
        js["groups"].append(g)
        lines.append("")
    return js, "\n".join(lines)


def case_report(npz_path: str | Path, windows: list[str] | None, cache_dir: str | Path, towns: bool = True,
                base: str = nohrsc.BASE) -> str:
    """Score one `fields_*.npz` (from `run --save-npz`) period by period; used by the `verify` command."""
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    stride = int(meta["stride"])
    wins = windows or [w for w in sorted(meta["windows"]) if w.startswith("s24_")]
    lon, lat, _ = _domains(stride)
    out = []
    for wid in wins:
        end = datetime.fromisoformat(meta["windows"][wid][1])
        full = nohrsc.load(end, 24, cache_dir, base)
        if full is None:
            out.append(f"[{wid}] NOHRSC 24 h analysis ending {end:%Y-%m-%d %H}Z is not published yet")
            continue
        obs = full[::stride, ::stride]
        m, lo, hi = z[f"{wid}_mode"], z[f"{wid}_s_lo"], z[f"{wid}_s_hi"]
        sc = score_case(m, lo, hi, obs, stride)
        led = {"cases": [{"cycle": meta["cycle"], "window": wid, "end": end.isoformat(), "fingerprint": "-", "label": "this file",
                          "scores": sc}]}
        out.append(f"[{wid}] {meta['windows'][wid][0]} -> {meta['windows'][wid][1]}")
        out.append(summarize(led)[1])
        if towns:
            out.append(f"   {'point':<16}{'obs':>6}{'mode':>6}{'p10':>6}{'p50':>6}{'p90':>6}")
            for name, _, (r, c) in calibration.point_indices(lat, lon):
                sl = (slice(max(r - 1, 0), r + 2), slice(max(c - 1, 0), c + 2))
                ob = np.nanmean(obs[sl]) if np.isfinite(obs[sl]).any() else float("nan")
                q = [float(binormal.percentile(m[r, c], lo[r, c], hi[r, c], x)) for x in (0.1, 0.5, 0.9)]
                out.append(f"   {name:<16}{ob:6.2f}{m[r, c]:6.2f}{q[0]:6.2f}{q[1]:6.2f}{q[2]:6.2f}")
    return "\n".join(out)
