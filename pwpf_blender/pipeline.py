"""Orchestration: members -> per-member 24 h snowfall -> mode -> binormal fit -> windows."""
from __future__ import annotations

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from . import binormal, mode as modemod
from .downscale import PrismDownscaler
from .grid import GridSpec
from .member import BinContext, process_bin
from .regrid import Regridder
from .sources import make_source

log = logging.getLogger("pwpf")


# --------------------------------------------------------------------------- member bookkeeping
@dataclass
class MemberRun:
    group: str
    label: str
    source: object
    member: object
    init: datetime
    native_ptype: bool
    weight: float
    max_fxx_h: float
    fallback: "MemberRun | None" = None


def select_init(cycle: datetime, run_hours, age_h: float, lag_h: float = 0) -> datetime:
    t = cycle - timedelta(hours=age_h)
    for back in range(0, 72):
        cand = t - timedelta(hours=back)
        if cand.hour in run_hours:
            return cand - timedelta(hours=lag_h)
    raise ValueError("no run hour found")


def day_starts(cycle: datetime, ndays: int = 3) -> list[datetime]:
    t = cycle.replace(minute=0, second=0, microsecond=0)
    first = t.replace(hour=12) if t.hour <= 12 else (t + timedelta(days=1)).replace(hour=12)
    return [first + timedelta(days=d) for d in range(ndays)]


def expand_members(cfg: dict, cycle: datetime, sources: dict | None = None) -> list[MemberRun]:
    sources = {} if sources is None else sources
    groups: dict[str, list[MemberRun]] = {}
    for g in cfg["members"]:
        if g.get("enabled", True) is False:
            continue
        if g["loader"] not in sources:
            sources[g["loader"]] = make_source(g["loader"])
        src = sources[g["loader"]]
        if "members" in g:
            ids = list(g["members"])
        else:
            pool = list(range(1, g["pool"] + 1)) if isinstance(g["pool"], int) else list(g["pool"])
            n = min(g["count"], len(pool))
            if g.get("pick", "random") == "random":
                rng = random.Random(f"{cycle:%Y%m%d%H}-{g['name']}")
                ids = sorted(rng.sample(pool, n))
            else:
                ids = pool[:n]
        runs = []
        for lag in g.get("lags_h", [0]):
            init = select_init(cycle, g["run_hours"], g.get("age_h", 0), lag)
            for m in ids:
                runs.append(MemberRun(g["name"], f"{g['name']}:{m}@{init:%d%HZ}", src, m, init,
                                      src.native_ptype, float(g.get("weight", 1.0)), float(g["max_fxx_h"])))
        groups[g["name"]] = runs
    for g in cfg["members"]:
        fb = g.get("fallback")
        if fb and g["name"] in groups and fb in groups and groups[fb]:
            for i, r in enumerate(groups[g["name"]]):
                r.fallback = groups[fb][i % len(groups[fb])]
    return [r for runs in groups.values() for r in runs]


# --------------------------------------------------------------------------- results
@dataclass
class Results:
    cycle: datetime
    grid: GridSpec
    windows: dict = field(default_factory=dict)          # id -> dict(mode, s_lo, s_hi, std, start, end, label)
    members: list = field(default_factory=list)          # [{label, group, init, ok}]
    meta: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: dict, grid: GridSpec, terrain: np.ndarray, prism: np.ndarray | None,
                 external_mode: dict | None = None):
        self.cfg, self.grid, self.terrain = cfg, grid, terrain
        dsc = cfg.get("downscale", {})
        self.downscaler = PrismDownscaler(prism, grid.dx_km, dsc.get("ratio_min", 0.25),
                                          dsc.get("ratio_max", 4.0), dsc.get("smooth_factor", 1.0))
        self.external_mode = external_mode or {}
        self._regrid: dict = {}
        self._lock = threading.Lock()
        self._dst_lon, self._dst_lat = grid.lonlat()

    def regridder(self, source) -> Regridder:
        sg = source.grid()
        with self._lock:
            if sg.key not in self._regrid:
                log.info("building regridder for %s", sg.key)
                self._regrid[sg.key] = Regridder.build(sg.lat, sg.lon, self._dst_lat, self._dst_lon)
            return self._regrid[sg.key]

    # ------------------------------------------------------------------ one member, one bin
    def member_bin(self, run: MemberRun, b0: datetime, b1: datetime, ctx: BinContext) -> np.ndarray:
        r = run
        if (b1 - r.init).total_seconds() / 3600.0 > r.max_fxx_h and r.fallback is not None:
            r = r.fallback
        steps = r.source.load_steps(r.init, b0, b1, r.member)
        reg = self.regridder(r.source)
        return process_bin(steps, reg, ctx, native_ptype=r.native_ptype, src_dx_km=r.source.src_dx_km)

    # ------------------------------------------------------------------ whole run
    def run(self, cycle: datetime) -> Results:
        cfg = self.cfg
        runs = expand_members(cfg, cycle)
        n = len(runs)
        log.info("cycle %s: %d members", cycle, n)
        starts = day_starts(cycle, 3)
        bin_h = cfg.get("bin_hours", 6)
        ok = np.ones(n, bool)
        ny, nx = self.grid.shape
        day_members = []                                    # per day: (n, ny, nx) float16
        for d, d0 in enumerate(starts):
            acc = np.zeros((n, ny, nx), np.float32)
            for k in range(24 // bin_h):
                b0 = d0 + timedelta(hours=k * bin_h)
                b1 = b0 + timedelta(hours=bin_h)
                ctx = BinContext(self.terrain, self.downscaler, b0.timetuple().tm_yday,
                                 cfg.get("slr", {}), cfg.get("ptype", {}),
                                 cfg.get("snow_level", {}).get("wetbulb_c", 0.5))
                self._run_bin(runs, ok, acc, b0, b1, ctx)
                log.info("day %d bin %s done (%d/%d members ok)", d + 1, b0.strftime("%d %HZ"), ok.sum(), n)
            day_members.append(acc.astype(np.float16))
            del acc
        keep = np.where(ok)[0]
        if keep.size < cfg.get("min_members", 8):
            raise RuntimeError(f"only {keep.size} members succeeded (< min_members)")
        weights = np.array([runs[i].weight for i in keep], np.float32)
        if keep.size != n:
            day_members = [dm[keep] for dm in day_members]

        res = Results(cycle=cycle, grid=self.grid)
        res.members = [{"label": r.label, "group": r.group, "init": r.init.isoformat(), "ok": bool(ok[i])}
                       for i, r in enumerate(runs)]
        mode_days = []
        for d in range(3):
            mode_days.append(modemod.compute_mode(day_members[d], cfg.get("mode", {}), weights,
                                                  self.external_mode.get(f"d{d + 1}")))
        for w in cfg["windows"]:
            days = w["days"]
            mem = [day_members[d] for d in days]
            md = sum(mode_days[d] for d in days)
            fit = binormal.fit(mem, md)
            fit.update(start=starts[days[0]], end=starts[days[-1]] + timedelta(hours=24), label=w["label"])
            res.windows[w["id"]] = fit
            log.info("window %s fitted: max mode %.1f in, max p95-ish %.1f", w["id"], float(md.max()),
                     float(binormal.percentile(fit["mode"], fit["s_lo"], fit["s_hi"], 0.95).max()))
        res.meta = {"n_members": int(keep.size), "starts": [s.isoformat() for s in starts]}
        return res

    def _run_bin(self, runs, ok, acc, b0, b1, ctx):
        def work(i):
            if not ok[i]:
                return i, None
            try:
                return i, self.member_bin(runs[i], b0, b1, ctx)
            except Exception as e:                           # noqa: BLE001
                log.warning("member %s failed in bin %s: %s", runs[i].label, b0, e)
                return i, e
        workers = int(self.cfg.get("workers", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, out in ex.map(work, range(len(runs))):
                if isinstance(out, Exception):
                    ok[i] = False
                elif out is not None:
                    acc[i] += out
