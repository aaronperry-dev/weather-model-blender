"""Orchestration: members -> per-member 24 h snowfall -> mode -> binormal fit -> windows."""
from __future__ import annotations

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .constants import SNOW_LEVEL_WARM

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
    note: str = ""                       # e.g. "00Z run not complete yet; used 18Z"


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


def _run_spacing(run_hours) -> int:
    hs = sorted(set(int(h) for h in run_hours))
    gaps = [(hs[(i + 1) % len(hs)] - h) % 24 or 24 for i, h in enumerate(hs)]
    return min(gaps)


def resolve_inits(g: dict, src, first_member, cycle: datetime, need_h: float | None) -> list[tuple[datetime, str]]:
    """Model initialisations for group `g`: the nominal run (see select_init) for each lag, but if that run has not
    finished publishing the lead times this blend needs (e.g. GFS 00Z at 03:40Z: f036 and later are not out yet),
    step back one run at a time (max 2) instead of losing the model.  Returns [(init, note)]."""
    probe = getattr(src, "available", None) if need_h else None
    spacing = _run_spacing(g["run_hours"])
    step = max(int(getattr(src, "step_hours", 1) or 1), 1)
    out, used = [], set()
    for lag in g.get("lags_h", [0]):
        nominal = select_init(cycle, g["run_hours"], g.get("age_h", 0), lag)
        chosen, note = None, ""
        for extra in range(0, 3):
            init = select_init(cycle, g["run_hours"], g.get("age_h", 0), lag + extra * spacing)
            if init in used:
                continue
            ok = True
            if probe is not None:
                last = min(float(g["max_fxx_h"]), (cycle + timedelta(hours=need_h) - init).total_seconds() / 3600.0)
                fxx = int(-(-last // step) * step)
                if fxx > 0:
                    ok = bool(probe(init, fxx, first_member))
            if ok:
                chosen = init
                if init != nominal:
                    note = f"{nominal:%d %HZ} run not fully published yet; using {init:%d %HZ}"
                break
        if chosen is None:                               # nothing complete: keep the nominal run, failures get reported
            chosen = nominal if nominal not in used else None
        if chosen is None:
            continue
        used.add(chosen)
        out.append((chosen, note))
    return out


def expand_members(cfg: dict, cycle: datetime, sources: dict | None = None, need_h: float | None = None) -> list[MemberRun]:
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
        for init, note in resolve_inits(g, src, ids[0], cycle, need_h):
            if note:
                log.warning("%s: %s", g["name"], note)
            for m in ids:
                runs.append(MemberRun(g["name"], f"{g['name']}:{m}@{init:%d%HZ}", src, m, init,
                                      src.native_ptype, float(g.get("weight", 1.0)), float(g["max_fxx_h"]), note=note))
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
    members: list = field(default_factory=list)          # per member: label, group, init, ok, bins_ok, error, ...
    skipped: list = field(default_factory=list)          # windows not produced: [{id, n_members, reason}]
    meta: dict = field(default_factory=dict)
    terrain: np.ndarray | None = None     # 2.5 km model terrain (m), exported for the browser's terrain adjustment


class AuxAccumulator:
    """Per 6-h bin sums that let the browser's terrain adjustment work: the weighted mean potential snowfall (inches if
    all liquid fell as snow) and the QPF-weighted snow level, over the members that delivered the bin.  Window values
    are the sum over bins (potential) and the ratio of sums (snow level)."""

    def __init__(self, nbins: int, shape):
        self.shape, self.nbins = shape, nbins
        self.pot = [None] * nbins
        self.sln = [None] * nbins
        self.sld = [None] * nbins
        self.w = np.zeros(nbins, np.float64)
        self._lock = threading.Lock()

    def add(self, k: int, weight: float, aux: dict | None):
        with self._lock:
            self.w[k] += weight
            if not aux:
                return
            if self.pot[k] is None:
                self.pot[k], self.sln[k], self.sld[k] = (np.zeros(self.shape, np.float32) for _ in range(3))
            self.pot[k] += weight * aux["pot"]
            self.sln[k] += weight * aux["q"] * aux["sl"]
            self.sld[k] += weight * aux["q"]

    def window(self, k0: int, k1: int):
        """(potential snowfall in, snow level m) for bins [k0, k1); snow level is the warm sentinel where nothing fell."""
        pot = np.zeros(self.shape, np.float32)
        num = np.zeros(self.shape, np.float32)
        den = np.zeros(self.shape, np.float32)
        for k in range(k0, k1):
            if self.pot[k] is None or self.w[k] <= 0:
                continue
            pot += self.pot[k] / np.float32(self.w[k])
            num += self.sln[k]
            den += self.sld[k]
        sl = np.where(den > 0, num / np.maximum(den, 1e-6), SNOW_LEVEL_WARM).astype(np.float32)
        return pot, sl


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
    def member_bin(self, run: MemberRun, b0: datetime, b1: datetime, ctx: BinContext, aux: dict | None = None):
        r = run
        if (b1 - r.init).total_seconds() / 3600.0 > r.max_fxx_h and r.fallback is not None:
            r = r.fallback
        steps = r.source.load_steps(r.init, b0, b1, r.member)
        reg = self.regridder(r.source)
        return bin_products(steps, reg, ctx, native_ptype=r.native_ptype, src_dx_km=r.source.src_dx_km, aux=aux)

    # ------------------------------------------------------------------ whole run
    def run(self, cycle: datetime) -> Results:
        cfg = self.cfg
        specs = window_specs(cfg, cycle)
        bin_h = int(cfg.get("bin_hours", 6))
        nbins = max(s.end_h for s in specs) // bin_h
        runs = expand_members(cfg, cycle, need_h=nbins * bin_h)
        n = len(runs)
        log.info("cycle %s: %d members", cycle, n)
        for src in {id(r.source): r.source for r in runs}.values():
            if hasattr(src, "set_span"):                  # let sources fetch each member once for the whole run
                src.set_span(cycle, cycle + timedelta(hours=nbins * bin_h))
        avail = np.zeros((n, nbins), bool)                  # member i delivered bin k
        fails = np.zeros(n, int)                            # consecutive failures
        dead = np.zeros(n, bool)                            # stopped being tried after too many failures
        errors: dict[int, str] = {}                         # first error per member
        limit = int(cfg.get("member_failure_limit", 2))
        ny, nx = self.grid.shape
        ice_on = cfg.get("ice", {}).get("enabled", True)
        self._aux = AuxAccumulator(nbins, (ny, nx)) if cfg.get("web", {}).get("terrain", {}).get("enabled", True) else None
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
            self._run_bin(runs, avail, fails, dead, errors, limit, k, sb, ib, b0, b1, ctx)
            snow_bins.append(sb)
            ice_bins.append(ib)
            log.info("bin +%d..+%d h done (%d/%d members delivered, %d with ice)", k * bin_h, (k + 1) * bin_h,
                     avail[:, k].sum(), n, len(ib))
        if not avail.any():
            raise RuntimeError("no member delivered any data; first errors: " +
                               "; ".join(f"{runs[i].label}: {e}" for i, e in list(errors.items())[:5]))
        weights = np.array([r.weight for r in runs], np.float32)

        res = Results(cycle=cycle, grid=self.grid)
        res.terrain = self.terrain if self._aux is not None else None
        res.members = [{"label": r.label, "group": r.group, "member": "" if r.member is None else str(r.member),
                        "init": r.init.isoformat(), "weight": r.weight, "max_fxx_h": r.max_fxx_h, "note": r.note,
                        "fallback": r.fallback.group if r.fallback else None,
                        "ok": bool(avail[i].all()), "bins_ok": int(avail[i].sum()), "bins_total": nbins,
                        "bins": "".join("1" if v else "0" for v in avail[i]),
                        "error": errors.get(i, "")[:240]} for i, r in enumerate(runs)]
        self._fit_windows(res, specs, snow_bins, weights, avail, bin_h, ice=False, bins_ice=None)
        if not res.windows:
            raise RuntimeError("no window had enough members (min_members=%s): %s" % (
                cfg.get("min_members", 4), "; ".join(f"{x['id']}:{x['n_members']}" for x in res.skipped[:8])))
        if ice_on:
            self._fit_windows(res, specs, snow_bins, weights, avail, bin_h, ice=True, bins_ice=ice_bins)
        from . import noaa_http as _nh
        log.info("HTTP request budget:\n%s", _nh.STATS.report())
        n_full = int(avail.all(axis=1).sum())
        if n_full != n:
            log.warning("blend used partial data: %d/%d members complete, %d windows skipped; failures: %s",
                        n_full, n, len(res.skipped),
                        "; ".join(f"{runs[i].label}: {e[:100]}" for i, e in errors.items()))
        res.meta = {"n_members": n_full, "n_requested": n, "bin_hours": bin_h,
                    "prism_downscale": self.downscaler.monthly is not None,
                    "windows": sorted(res.windows), "windows_planned": [s.id for s in specs],
                    "windows_skipped": res.skipped, "ice_windows": sorted(res.ice)}
        return res

    # ------------------------------------------------------------------ window fits
    def _fit_windows(self, res, specs, snow_bins, weights, avail, bin_h, *, ice: bool, bins_ice):
        """Fit the binormal for every window from the members that delivered *every* bin of that window (weights
        renormalised over them), so a member that is missing or fails part-way still contributes to the windows
        it fully covers.  24 h windows get their own mode (ensemble-derived stand-in for the WWD forecast, or an
        external grid); 48/72 h modes are sums of the adjacent 24 h modes, as in PWPF."""
        cfg = self.cfg
        mode_cfg = dict(cfg.get("mode", {}))
        min_members = int(cfg.get("min_members", 4))
        warn_members = int(cfg.get("warn_members", 8))
        min_total = 0.0
        if ice:
            icfg = cfg.get("ice", {})
            min_total = float(icfg.get("min_total_in", 0.005))
            mode_cfg["min_snow_in"] = min(mode_cfg.get("min_snow_in", 0.01), min_total)
        ny, nx = self.grid.shape
        n_all = avail.shape[0]
        mode24: dict[int, np.ndarray | None] = {}
        for sp in sorted(specs, key=lambda x: (x.length_h, x.end_h)):
            k0, k1 = (sp.end_h - sp.length_h) // bin_h, sp.end_h // bin_h
            idx = np.where(avail[:, k0:k1].all(axis=1))[0]
            n = int(idx.size)
            if n < min_members:
                if not ice:
                    res.skipped.append({"id": sp.id, "n_members": n,
                                        "reason": f"only {n} member(s) have data for the whole period (min {min_members})"})
                    log.warning("window %s skipped: only %d members with complete data", sp.id, n)
                mode24[sp.end_h] = None
                continue
            w = weights[idx]
            sub = (lambda arr: arr) if n == n_all else (lambda arr: arr[idx])      # noqa: E731
            if sp.length_h == 24:
                x = self._sum_bins(snow_bins, bins_ice, k0, k1, idx, ice)
                if x is None or float(x.max()) < min_total:
                    mode24[sp.end_h] = None
                    continue
                ext = self.external_mode.get(sp.id) if not ice else None
                md = modemod.compute_mode(x, mode_cfg, w, ext)
                fit = binormal.fit(x, md)
                mode24[sp.end_h] = md
                del x
            else:
                parts = [mode24.get(sp.end_h - 24 * j) for j in range(sp.length_h // 24)]
                if not ice:
                    if any(p is None for p in parts):
                        continue
                    md = sum(parts)
                    fit = binormal.fit([sub(snow_bins[k]) for k in range(k0, k1)], md)
                else:
                    x = self._sum_bins(snow_bins, bins_ice, k0, k1, idx, True)
                    if x is None or float(x.max()) < min_total:
                        continue
                    md = sum((p for p in parts if p is not None), np.zeros((ny, nx), np.float32))
                    fit = binormal.fit(x, md)
                    del x
            fit.update(start=sp.start, end=sp.end, label=sp.label, length_h=sp.length_h, end_h=sp.end_h,
                       n_members=n, reduced=bool(n < warn_members))
            if not ice and getattr(self, "_aux", None) is not None:
                fit["pot"], fit["sl"] = self._aux.window(k0, k1)
            (res.ice if ice else res.windows)[sp.id] = fit
            log.info("%s %s fitted from %d/%d members: max mode %.2f in, max p95 %.2f in",
                     "ice" if ice else "snow", sp.id, n, n_all, float(md.max()),
                     float(binormal.percentile(fit["mode"], fit["s_lo"], fit["s_hi"], 0.95).max()))

    def _sum_bins(self, snow_bins, bins_ice, k0, k1, idx, ice: bool):
        """(len(idx), ny, nx) float32 member totals over bins [k0, k1) for members `idx`; None when ice and
        nothing fell."""
        ny, nx = self.grid.shape
        if not ice:
            pick = (lambda arr: arr) if len(idx) == snow_bins[k0].shape[0] else (lambda arr: arr[idx])   # noqa: E731
            x = pick(snow_bins[k0]).astype(np.float32)
            for k in range(k0 + 1, k1):
                x += pick(snow_bins[k])
            return x
        if not any(bins_ice[k] for k in range(k0, k1)):
            return None
        pos = {int(o): j for j, o in enumerate(idx)}
        x = np.zeros((len(idx), ny, nx), np.float32)
        for k in range(k0, k1):
            for i, a in bins_ice[k].items():
                if i in pos:
                    x[pos[i]] += a
        return x

    def _run_bin(self, runs, avail, fails, dead, errors, limit, k, snow, ice, b0, b1, ctx):
        """Fill `snow[i]` (float16) and `ice[i]` (only when non-zero) for every member that delivers this bin.
        A member that fails is simply absent from this bin (the windows that need it will not use it); after
        `limit` consecutive failures it is no longer tried, so a missing run does not burn the request budget."""
        aux_acc = getattr(self, "_aux", None)

        def work(i):
            if dead[i]:
                return i, None
            try:
                aux = {} if aux_acc is not None else None
                return i, (self.member_bin(runs[i], b0, b1, ctx, aux), aux)
            except Exception as e:                           # noqa: BLE001
                log.warning("member %s failed in bin %s: %s", runs[i].label, b0, e)
                return i, e
        workers = int(self.cfg.get("workers", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, out in ex.map(work, range(len(runs))):
                if isinstance(out, Exception):
                    fails[i] += 1
                    errors.setdefault(i, f"bin +{k * int(self.cfg.get('bin_hours', 6))} h: {type(out).__name__}: {out}")
                    if fails[i] >= limit:
                        dead[i] = True
                elif out is not None:
                    (sn, ic), aux = out
                    if aux_acc is not None:
                        aux_acc.add(k, float(runs[i].weight), aux)
                    snow[i] = sn
                    avail[i, k] = True
                    fails[i] = 0
                    if ic is not None:
                        ice[i] = ic.astype(np.float16)
