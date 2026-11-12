"""Synthetic ensemble used for tests, demos and CI smoke runs (no network).

Creates an idealised winter storm: a moving low with a precipitation shield, a north-south
temperature gradient, a standard-atmosphere vertical structure and member-to-member perturbations
in track, amplitude and temperature.  Two flavours exercise both regridder paths:
  synthetic_global : regular 0.25 degree lat/lon grid
  synthetic_cam    : Lambert-conformal 3 km-like grid (irregular -> KD-tree regridder), 36 h horizon
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..grid import GridSpec
from ..member import StepFields
from .base import Source, SourceGrid

P_LEVELS = (1000, 925, 850, 700, 500)
WARM_NOSE_C = 6.0        # degC added at 925/850 hPa along a band near 39.5N: sleet / freezing rain under it


def synthetic_terrain(lon, lat):
    """Smooth idealised CONUS terrain (m): Rockies, Sierra/Cascades, Appalachians."""
    def blob(lo0, la0, sx, sy, h):
        return h * np.exp(-(((lon - lo0) / sx) ** 2 + ((lat - la0) / sy) ** 2))
    z = (blob(-106, 40, 4.5, 7, 2600) + blob(-120.5, 38.5, 1.0, 5, 1800) + blob(-121.8, 46, 1.2, 3, 1400)
         + blob(-80.5, 37.5, 2.0, 5, 800) + blob(-111, 45, 3, 2.5, 1400) + 150 * np.exp(-((lon + 95) / 20) ** 2))
    return z.astype(np.float32)


def synthetic_prism(lon, lat):
    """Fake monthly PRISM-like precipitation (12, ny, nx) in mm with orographic enhancement."""
    terrain = synthetic_terrain(lon, lat)
    base = 60 + 0.08 * terrain + 25 * np.exp(-(((lon + 85) / 12) ** 2 + ((lat - 35) / 8) ** 2))
    seasonal = 1 + 0.25 * np.cos(2 * np.pi * (np.arange(12) - 0.5) / 12)[:, None, None]
    return (base[None] * seasonal).astype(np.float32)


class SyntheticSource(Source):
    def __init__(self, kind: str = "global", seed: int = 0, bias: float = 0.0):
        self.kind = kind
        self.name = f"synthetic_{kind}"
        self.seed = seed
        self.bias = bias
        self.native_ptype = False
        if kind == "global":
            self.step_hours, self.src_dx_km = 3, 27.0
            lat = np.arange(18.0, 58.01, 0.25)
            lon = np.arange(-128.0, -62.99, 0.25)
            self._grid = SourceGrid(lat.astype(np.float64), lon.astype(np.float64), "syn_global")
            self._lon2d, self._lat2d = np.meshgrid(lon, lat)
        else:
            self.step_hours, self.src_dx_km = 1, 4.0
            g = GridSpec.ndfd_2p5km(stride=2)           # ~5 km Lambert grid (rotated vs lat/lon -> KD path)
            lon2d, lat2d = g.lonlat()
            self._grid = SourceGrid(lat2d, lon2d, "syn_cam")
            self._lon2d, self._lat2d = lon2d, lat2d
        self._terrain = synthetic_terrain(self._lon2d, self._lat2d) * 0.8   # smoother model terrain

    def grid(self) -> SourceGrid:
        return self._grid

    # ------------------------------------------------------------------
    def load_steps(self, init: datetime, valid_start: datetime, valid_end: datetime, member):
        rng = np.random.default_rng(abs(hash((self.seed, str(member)))) % (2 ** 32))
        dxm, dym = rng.normal(0, 1.2), rng.normal(0, 1.0)         # track error (deg)
        amp = float(np.clip(rng.lognormal(0.0, 0.35), 0.4, 2.5))
        dT = rng.normal(0.0, 1.3) + self.bias
        speed = rng.normal(1.0, 0.1)
        steps = []
        t = valid_start + timedelta(hours=self.step_hours)
        lon, lat = self._lon2d, self._lat2d
        while t <= valid_end:
            h = (t - datetime(t.year, 1, 1)).total_seconds() / 3600.0
            hrs = (t - init).total_seconds() / 3600.0
            # low pressure track: Rockies -> Great Lakes -> northeast
            clon = -108 + 0.45 * hrs * speed + dxm
            clat = 38 + 0.20 * hrs * speed + dym
            r2 = ((lon - clon) / 5.5) ** 2 + ((lat - clat) / 3.2) ** 2
            shield = np.exp(-r2)
            q_rate = amp * (0.75 * shield + 0.10 * np.exp(-(((lon - clon) / 12) ** 2 + ((lat - clat - 4) / 2) ** 2)))
            oro = 1 + 0.6 * np.clip(self._terrain / 2000.0, 0, 1) * (shield > 0.2)
            qpf = (q_rate * oro * self.step_hours).astype(np.float32)
            qpf[qpf < 0.05] = 0.0
            diurnal = 1.2 * np.sin((h % 24 - 9) / 24 * 2 * np.pi)
            t2m = (9.0 - 0.9 * (lat - 35.0) - 0.0065 * self._terrain + dT + diurnal
                   - 2.5 * np.clip(shield, 0, 1)).astype(np.float32)
            zsfc = self._terrain
            nose = np.exp(-(((lat - (39.5 + 0.1 * dxm)) / 1.1) ** 2)) * (shield > 0.15) if WARM_NOSE_C else 0.0
            psfc = (1013.25 * np.exp(-zsfc / 8400.0)).astype(np.float32)
            levels = {}
            for p in P_LEVELS:
                zstd = 44330.0 * (1 - (p / 1013.25) ** 0.1903)
                tp = t2m - 0.0065 * np.maximum(zstd - zsfc, 0)
                if p in (925, 850):                              # warm nose -> a freezing-rain band
                    tp = tp + WARM_NOSE_C * nose
                levels[p] = {"t": tp.astype(np.float32),
                             "rh": np.where(shield > 0.1, 96.0, 65.0).astype(np.float32),
                             "z": np.full(lat.shape, zstd, np.float32)}
            steps.append(StepFields(valid=t, qpf_mm=qpf, t2m_c=t2m, psfc_hpa=psfc, levels=levels,
                                    rh2m=np.where(shield > 0.1, 95.0, 70.0).astype(np.float32), z_sfc_m=zsfc,
                                    dt_h=float(self.step_hours)))
            t += timedelta(hours=self.step_hours)
        return steps
