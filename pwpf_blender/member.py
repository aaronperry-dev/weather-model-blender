"""Per-member snowfall: QPF x precipitation type x SLR, following the PWPF recipe.

For one 6-h bin of one ensemble member:
  1. accumulate QPF over the member's native time steps (source grid)
  2. classify precipitation type each step (native categorical fields or decision tree) and take the
     dominant type over the bin
  3. wet-bulb snow level each step -> QPF-weighted bin mean
  4. SLR each step -> QPF-weighted bin mean
  5. regrid [QPF, dominant-snow flag, snow level, SLR] to the 2.5 km grid
  6. PRISM-downscale QPF; final snow mask = native snow  OR  terrain above the snow level
  7. snowfall (in) = QPF/25.4 * SLR * mask
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .constants import (MM_PER_INCH, SNOW_LEVEL_WARM, TYPE_SNOW)
from .downscale import PrismDownscaler
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
    dst_shape = regrid.dst_shape
    sshape = steps[0].qpf_mm.shape
    qpf = np.zeros(sshape, np.float32)
    wsum = np.zeros(sshape, np.float32)
    sl_acc = np.zeros(sshape, np.float32)
    slr_acc = np.zeros(sshape, np.float32)
    counts = np.zeros((5,) + sshape, np.int16)

    pt = ctx.ptype_cfg
    for s in steps:
        q = np.clip(np.nan_to_num(s.qpf_mm, nan=0.0), 0.0, None).astype(np.float32)
        qpf += q
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

        sl = snow_level_height(s.levels, s.t2m_c, rh2m, s.psfc_hpa, z_sfc, ctx.snow_level_wb_c)
        slr = slr_step(s.t2m_c, s.levels, s.psfc_hpa, ctx.doy, ctx.slr_cfg)
        sl_acc += q * sl
        slr_acc += q * slr
        wsum += q

    if float(qpf.max()) <= 0.0:
        return np.zeros(dst_shape, np.float32)

    dom = dominant_type(counts)
    native_snow = (dom == TYPE_SNOW).astype(np.float32)
    sl_bin = np.where(wsum > 0, sl_acc / np.maximum(wsum, 1e-6), SNOW_LEVEL_WARM).astype(np.float32)
    slr_bin = np.where(wsum > 0, slr_acc / np.maximum(wsum, 1e-6), 10.0).astype(np.float32)

    q2, ns2, sl2, slr2 = regrid(np.stack([qpf, native_snow, sl_bin, slr_bin]))
    q2 = np.nan_to_num(q2, nan=0.0)
    ns2 = np.nan_to_num(ns2, nan=0.0)
    sl2 = np.nan_to_num(sl2, nan=SNOW_LEVEL_WARM)
    slr2 = np.nan_to_num(slr2, nan=10.0)

    if ctx.downscaler is not None:
        r = ctx.downscaler.ratio(ctx.doy, src_dx_km)
        if r is not None:
            q2 = q2 * r

    snow_mask = (ns2 >= 0.5) | (ctx.terrain_m > sl2)
    return (q2 / MM_PER_INCH * slr2 * snow_mask).astype(np.float32)
