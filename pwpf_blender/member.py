"""Per-member snowfall: QPF x precipitation type x SLR, following the PWPF recipe.

For one 6-h bin of one ensemble member:
  1. accumulate QPF over the member's native time steps (source grid)
  2. classify precipitation type each step (native categorical fields or decision tree) and take the
     dominant type over the bin
  3. wet-bulb snow level each step -> QPF-weighted bin mean
  4. SLR each step -> QPF-weighted bin mean
  5. regrid [QPF, dominant-snow flag, snow level, SLR] to the 2.5 km grid
  6. PRISM-downscale QPF; final snow mask = (native snow AND downscaled 2 m temperature <= ptype.snow_t2m_max_c)
     OR terrain above the snow level.  The 2 m temperature is the QPF-weighted bin mean, moved from the model's
     own terrain to the 2.5 km terrain with a lapse rate (WPC: ptype is downscaled from snow level and 2 m T).
  7. snowfall (in) = QPF/25.4 * SLR * mask
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .constants import (MM_PER_INCH, SNOW_LEVEL_COLD, SNOW_LEVEL_WARM, TYPE_FZRA, TYPE_SNOW)
from . import accumulation
from .downscale import PrismDownscaler
from .ice import RADIAL_FACTOR, flat_ice_in
from .ptype import decision_tree, dominant_type, native_type
from .regrid import Regridder
from .slr import slr_step
from .thermo import snow_level_height, surface_height_from_levels


@dataclass
class StepFields:
    """Standardised fields for one native model time step on the source grid."""
    valid: datetime
    qpf_mm: np.ndarray                    # accumulation over this step
    t2m_c: np.ndarray
    psfc_hpa: np.ndarray
    levels: dict                          # {p_hpa: {"t": degC, "rh": %, "z": m}}
    rh2m: np.ndarray | None = None
    z_sfc_m: np.ndarray | None = None
    native_type_frac: dict | None = None  # {"rain","snow","sleet","fzra"} -> fraction 0..1
    dt_h: float | None = None             # length of this step in hours (None: bin length / number of steps)


@dataclass
class BinContext:
    """Everything needed on the destination side."""
    terrain_m: np.ndarray                 # 2.5 km terrain height (RTMA)
    downscaler: PrismDownscaler | None
    doy: int
    slr_cfg: dict = field(default_factory=dict)
    ptype_cfg: dict = field(default_factory=dict)
    snow_level_wb_c: float = 0.5
    min_step_qpf_mm: float = 0.01
    ice_cfg: dict = field(default_factory=dict)   # {"enabled": bool, "report": "flat"|"radial"}
    bin_hours: float = 6.0
    accum_cfg: dict = field(default_factory=dict)   # warm-ground / marginal-air accumulation (see accumulation.py); {} = off


def _aloft_max(levels, psfc):
    out = None
    for p in (925, 850, 700):
        if p in levels:
            t = np.where(p < psfc, levels[p]["t"], np.nan)
            out = t if out is None else np.fmax(out, t)
    return out if out is not None else np.full(psfc.shape, np.nan, dtype=np.float32)


def process_bin(steps: list[StepFields], regrid: Regridder, ctx: BinContext,
                native_ptype: bool = False, src_dx_km: float = 25.0) -> np.ndarray:
    """Snowfall (inches) over the bin on the destination grid (float32)."""
    return bin_products(steps, regrid, ctx, native_ptype, src_dx_km)[0]


def bin_products(steps: list[StepFields], regrid: Regridder, ctx: BinContext,
                 native_ptype: bool = False, src_dx_km: float = 25.0, aux: dict | None = None):
    """(snowfall, ice) in inches over the bin on the destination grid (float32).
    `ice` is None when the member produced no freezing rain in the bin (or ice is disabled).
    When `aux` (a dict) is given and the member had precipitation, it is filled with the fields the browser's terrain
    adjustment needs: `pot` (inches if all the liquid fell as snow: QPF x SLR), `sl` (QPF-weighted snow level, m MSL,
    clipped to the sentinels' range) and `q` (downscaled QPF, mm; the weight for `sl`)."""
    dst_shape = regrid.dst_shape
    ice_on = ctx.ice_cfg.get("enabled", True)
    sshape = steps[0].qpf_mm.shape
    qpf = np.zeros(sshape, np.float32)
    wsum = np.zeros(sshape, np.float32)
    sl_acc = np.zeros(sshape, np.float32)
    slr_acc = np.zeros(sshape, np.float32)
    t2_wacc = np.zeros(sshape, np.float32)      # QPF-weighted 2 m temperature (wet cells)
    t2_all = np.zeros(sshape, np.float32)       # plain mean 2 m temperature (the fallback in dry cells)
    counts = np.zeros((5,) + sshape, np.int16)
    ice_src = np.zeros(sshape, np.float32) if ice_on else None

    pt = ctx.ptype_cfg
    for s in steps:
        q = np.clip(np.nan_to_num(s.qpf_mm, nan=0.0), 0.0, None).astype(np.float32)
        qpf += q
        t2 = np.nan_to_num(s.t2m_c, nan=0.0).astype(np.float32)
        t2_all += t2
        wet = q > ctx.min_step_qpf_mm
        if not wet.any():
            continue
        z_sfc = s.z_sfc_m if s.z_sfc_m is not None else surface_height_from_levels(s.psfc_hpa, s.levels)
        rh2m = s.rh2m if s.rh2m is not None else np.full(sshape, 90.0, np.float32)

        tree = decision_tree(s.t2m_c, _aloft_max(s.levels, s.psfc_hpa),
                             pt.get("snow_sfc_max_c", 1.5), pt.get("frz_aloft_c", 2.0))
        types = native_type(s.native_type_frac, tree) if (native_ptype and s.native_type_frac) else tree
        for code in (1, 2, 3, 4):
            counts[code] += ((types == code) & wet).astype(np.int16)
        if ice_on:
            fz = (types == TYPE_FZRA) & wet
            if fz.any():
                dt = s.dt_h if s.dt_h else ctx.bin_hours / max(len(steps), 1)
                ice_src += np.where(fz, flat_ice_in(q / MM_PER_INCH, dt), 0.0).astype(np.float32)

        sl = snow_level_height(s.levels, s.t2m_c, rh2m, s.psfc_hpa, z_sfc, ctx.snow_level_wb_c)
        slr = slr_step(s.t2m_c, s.levels, s.psfc_hpa, ctx.doy, ctx.slr_cfg)
        sl_acc += q * sl
        slr_acc += q * slr
        t2_wacc += q * t2
        wsum += q

    if float(qpf.max()) <= 0.0:
        return np.zeros(dst_shape, np.float32), None

    dom = dominant_type(counts)
    native_snow = (dom == TYPE_SNOW).astype(np.float32)
    sl_bin = np.where(wsum > 0, sl_acc / np.maximum(wsum, 1e-6), SNOW_LEVEL_WARM).astype(np.float32)
    slr_bin = np.where(wsum > 0, slr_acc / np.maximum(wsum, 1e-6), 10.0).astype(np.float32)

    gate = bool(pt.get("t2m_gate", False))
    t2_bin = np.where(wsum > 0, t2_wacc / np.maximum(wsum, 1e-6), t2_all / max(len(steps), 1)).astype(np.float32)
    if gate:
        last = steps[-1]
        zm = last.z_sfc_m if last.z_sfc_m is not None else surface_height_from_levels(last.psfc_hpa, last.levels)
        zm = np.nan_to_num(zm, nan=0.0).astype(np.float32)
    has_ice = ice_on and float(ice_src.max()) > 0.0
    layers = [qpf, native_snow, sl_bin, slr_bin] + ([ice_src] if has_ice else []) + ([t2_bin, zm] if gate else [])
    out = regrid(np.stack(layers))
    q2, ns2, sl2, slr2 = out[:4]
    q2 = np.nan_to_num(q2, nan=0.0)
    q_raw = q2.copy() if aux is not None else None         # before the PRISM ratio (diagnostics)
    ns2 = np.nan_to_num(ns2, nan=0.0)
    sl2 = np.nan_to_num(sl2, nan=SNOW_LEVEL_WARM)
    slr2 = np.nan_to_num(slr2, nan=10.0)

    ice2 = np.clip(np.nan_to_num(out[4], nan=0.0), 0.0, None) if has_ice else None
    native2 = ns2 >= 0.5
    if gate:
        t2_dst = out[-2] + float(pt.get("t2m_lapse_c_per_km", 6.5)) * (out[-1] - ctx.terrain_m) / 1000.0    # 2 m T at the 2.5 km terrain
        native2 = native2 & ~(t2_dst > float(pt.get("snow_t2m_max_c", 1.5)))      # NaN (outside the source grid) does not gate
    if ctx.downscaler is not None:
        r = ctx.downscaler.ratio(ctx.doy, src_dx_km)
        if r is not None:
            q2 = q2 * r
            if ice2 is not None:
                ice2 = ice2 * r

    snow_mask = native2 | (ctx.terrain_m > sl2)
    snow = (q2 / MM_PER_INCH * slr2 * snow_mask).astype(np.float32)
    snow_pre = snow
    if accumulation.active(ctx.accum_cfg):
        snow = (snow * accumulation.temp_efficiency(sl2, ctx.terrain_m, ctx.accum_cfg, ctx.snow_level_wb_c)).astype(np.float32)
    if aux is not None:
        aux["pot"] = (q2 / MM_PER_INCH * slr2).astype(np.float32)
        aux["sl"] = np.clip(sl2, SNOW_LEVEL_COLD, SNOW_LEVEL_WARM).astype(np.float32)
        aux["q"] = q2.astype(np.float32)
        aux["q0"] = q_raw.astype(np.float32)
        aux["snow_pre"] = snow_pre.astype(np.float32)
        aux["pot_nat"] = (q2 / MM_PER_INCH * slr2 * native2).astype(np.float32)      # snow if only the model's own snow type (and, if gated, the 2 m T) counted
    if ice2 is not None:
        if ctx.ice_cfg.get("report", "flat") == "radial":
            ice2 = ice2 * float(ctx.ice_cfg.get("radial_factor", RADIAL_FACTOR))
        ice2 = ice2.astype(np.float32)
        if float(ice2.max()) <= 0.0:
            ice2 = None
    return snow, ice2
