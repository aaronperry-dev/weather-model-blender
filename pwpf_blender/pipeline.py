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
from .member import BinContext, bin_products
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


@dataclass(frozen=True)
class WindowSpec:
    """One accumulation period, relative to the cycle: (start, end] = cycle + (end_h - length_h, end_h]."""
    id: str
    length_h: int
    end_h: int
    start: datetime
    end: datetime
    label: str


def window_specs(cfg: dict, cycle: datetime) -> list[WindowSpec]:
    """PWPF periods: 24 h totals ending every `step_h` hours from +24 h to +72 h, 48 h totals (two adjacent
    24 h periods) and a single 72 h total.  Ids look like `s24_f030` (24 h ending 30 h after the cycle)."""
    wc = cfg.get("windows", {})
    step, max_h = int(wc.get("step_h", 6)), int(wc.get("max_h", 72))
    lengths = sorted(set(int(x) for x in wc.get("lengths_h", [24, 48, 72])) | {24})
    bin_h = int(cfg.get("bin_hours", 6))
    if step % bin_h or max_h % bin_h or any(L % 24 for L in lengths):
        raise ValueError("windows: step_h/max_h must be multiples of bin_hours and lengths_h multiples of 24")
    out = []
    for L in lengths:
        for end in range(L, max_h + 1, step):
            e = cycle + timedelta(hours=end)
            out.append(WindowSpec(f"s{L}_f{end:03d}", L, end, e - timedelta(hours=L), e, f"{e:%a %HZ}"))
    return out


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
    windows: dict = field(default_factory=dict)          # id -> snowfall fit: mode, s_lo, s_hi, std, start, end, label, ...
    ice: dict = field(default_factory=dict)              # id -> freezing-rain fit (only periods with an ice signal)
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
    def member_bin(self, run: MemberRun, b0: datetime, b1: datetime, ctx: BinContext):
        r = run
        if (b1 - r.init).total_seconds() / 3600.0 > r.max_fxx_h and r.fallback is not None:
            r = r.fallback
        steps = r.source.load_steps(r.init, b0, b1, r.member)
        reg = self.regridder(r.source)
        return bin_products(steps, reg, ctx, native_ptype=r.native_ptype, src_dx_km=r.source.src_dx_km)

    # ------------------------------------------------------------------ whole run
    def run(self, cycle: datetime) -> Results:
        cfg = self.cfg
        runs = expand_members(cfg, cycle)
        n = len(runs)
        log.info("cycle %s: %d members", cycle, n)
        specs = window_specs(cfg, cycle)
        bin_h = int(cfg.get("bin_hours", 6))
        nbins = max(s.end_h for s in specs) // bin_h
        ok = np.ones(n, bool)
        ny, nx = self.grid.shape
        ice_on = cfg.get("ice", {}).get("enabled", True)
        snow_bins: list[np.ndarray] = []                    # per 6-h bin: (n, ny, nx) float16 snowfall (in)
        ice_bins: list[dict] = []                           # per bin: {member index: (ny, nx) float16 ice (in)}
        for k in range(nbins):
            b0 = cycle + timedelta(hours=k * bin_h)
            b1 = b0 + timedelta(hours=bin_h)
            ctx = BinContext(self.terrain, self.downscaler, b0.timetuple().tm_yday,
                             cfg.get("slr", {}), cfg.get("ptype", {}),
                             cfg.get("snow_level", {}).get("wetbulb_c", 0.5),
                             ice_cfg=dict(cfg.get("ice", {}), enabled=ice_on), bin_hours=bin_h)
            sb = np.zeros((n, ny, nx), np.float16)
            ib: dict = {}
            self._run_bin(runs, ok, sb, ib, b0, b1, ctx)
            snow_bins.append(sb)
            ice_bins.append(ib)
            log.info("bin +%d..+%d h done (%d/%d members ok, %d with ice)", k * bin_h, (k + 1) * bin_h,
                     ok.sum(), n, len(ib))
        keep = np.where(ok)[0]
        if keep.size < cfg.get("min_members", 8):
            raise RuntimeError(f"only {keep.size} members succeeded (< min_members)")
        weights = np.array([runs[i].weight for i in keep], np.float32)
        if keep.size != n:
            remap = {int(old): new for new, old in enumerate(keep)}
            for k in range(nbins):
                snow_bins[k] = snow_bins[k][keep]
                ice_bins[k] = {remap[i]: a for i, a in ice_bins[k].items() if i in remap}
        n_ok = int(keep.size)

        res = Results(cycle=cycle, grid=self.grid)
        res.members = [{"label": r.label, "group": r.group, "init": r.init.isoformat(), "ok": bool(ok[i])}
                       for i, r in enumerate(runs)]
        self._fit_windows(res, specs, snow_bins, weights, n_ok, bin_h, ice=False, bins_ice=None)
        if ice_on:
            self._fit_windows(res, specs, snow_bins, weights, n_ok, bin_h, ice=True, bins_ice=ice_bins)
        res.meta = {"n_members": n_ok, "bin_hours": bin_h, "prism_downscale": self.downscaler.monthly is not None,
                    "windows": [s.id for s in specs], "ice_windows": sorted(res.ice)}
        return res

    # ------------------------------------------------------------------ window fits
    def _fit_windows(self, res, specs, snow_bins, weights, n, bin_h, *, ice: bool, bins_ice):
        """Fit the binormal for every window.  24 h windows get their own mode (ensemble-derived stand-in for
        the WWD forecast, or an external grid); 48/72 h modes are sums of the adjacent 24 h modes, as in PWPF."""
        cfg = self.cfg
        mode_cfg = dict(cfg.get("mode", {}))
        min_total = 0.0
        if ice:
            icfg = cfg.get("ice", {})
            min_total = float(icfg.get("min_total_in", 0.005))
            mode_cfg["min_snow_in"] = min(mode_cfg.get("min_snow_in", 0.01), min_total)
        ny, nx = self.grid.shape
        mode24: dict[int, np.ndarray | None] = {}
        for sp in sorted(specs, key=lambda x: (x.length_h, x.end_h)):
            k0, k1 = (sp.end_h - sp.length_h) // bin_h, sp.end_h // bin_h
            if sp.length_h == 24:
                x = self._sum_bins(snow_bins, bins_ice, k0, k1, n, ice)
                if x is None or float(x.max()) < min_total:
                    mode24[sp.end_h] = None
                    continue
                ext = self.external_mode.get(sp.id) if not ice else None
                md = modemod.compute_mode(x, mode_cfg, weights, ext)
                fit = binormal.fit(x, md)
                mode24[sp.end_h] = md
                del x
            else:
                parts = [mode24.get(sp.end_h - 24 * j) for j in range(sp.length_h // 24)]
                if not ice:
                    md = sum(parts)
                    fit = binormal.fit([snow_bins[k] for k in range(k0, k1)], md)
                else:
                    x = self._sum_bins(snow_bins, bins_ice, k0, k1, n, True)
                    if x is None or float(x.max()) < min_total:
                        continue
                    md = sum((p for p in parts if p is not None), np.zeros((ny, nx), np.float32))
                    fit = binormal.fit(x, md)
                    del x
            fit.update(start=sp.start, end=sp.end, label=sp.label, length_h=sp.length_h, end_h=sp.end_h)
            (res.ice if ice else res.windows)[sp.id] = fit
            log.info("%s %s fitted: max mode %.2f in, max p95 %.2f in", "ice" if ice else "snow", sp.id,
                     float(md.max()), float(binormal.percentile(fit["mode"], fit["s_lo"], fit["s_hi"], 0.95).max()))

    def _sum_bins(self, snow_bins, bins_ice, k0, k1, n, ice: bool):
        """(n, ny, nx) float32 member totals over bins [k0, k1); None when ice and nothing fell."""
        ny, nx = self.grid.shape
        if not ice:
            x = snow_bins[k0].astype(np.float32)
            for k in range(k0 + 1, k1):
                x += snow_bins[k]
            return x
        if not any(bins_ice[k] for k in range(k0, k1)):
            return None
        x = np.zeros((n, ny, nx), np.float32)
        for k in range(k0, k1):
            for i, a in bins_ice[k].items():
                x[i] += a
        return x

    def _run_bin(self, runs, ok, snow, ice, b0, b1, ctx):
        """Fill `snow[i]` (float16) and `ice[i]` (only when non-zero) for every member still ok."""
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
                    sn, ic = out
                    snow[i] = sn
                    if ic is not None:
                        ice[i] = ic.astype(np.float16)
