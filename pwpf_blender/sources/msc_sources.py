"""Canadian deterministic models from the MSC Datamart (dd.weather.gc.ca): RDPS (10 km, regional) and GDPS (15 km).

Layout (verified against the live directory listings, 2026-11):
  https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_rdps/10km/{HH}/{hhh}/{YYYYMMDD}T{HH}Z_MSC_RDPS_{Var}_{Level}_RLatLon0.09_PT{hhh}H.grib2
  https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_gdps/15km/{HH}/{hhh}/{YYYYMMDD}T{HH}Z_MSC_GDPS_{Var}_{Level}_LatLon0.15_PT{hhh}H.grib2
One variable x level x forecast hour per file, one GRIB message each, no .idx / byte ranges - so a step is ~19 small
files (about 13 MB for RDPS).  HRDPS (2.5 km, model_hrdps/continental) only covers southern Canada / the northern US,
so it is not used for a CONUS blend.  Units are normalised by value range (K or C, Pa or hPa, % or fraction,
gpm or dam) because the files' unit keys are not consistent between products."""
from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import noaa_http as nh
from ..member import StepFields
from .base import Source, SourceGrid
from .noaa_sources import LEVELS, _Cropper

log = logging.getLogger(__name__)

ROOT = "https://dd.weather.gc.ca"
Z_EXPECT = {1000: 100.0, 925: 760.0, 850: 1460.0, 700: 3010.0, 500: 5570.0}      # typical geopotential heights (m)


@dataclass(frozen=True)
class MscModel:
    key: str
    name: str                # as in the file names
    path: str                # directory under /{date}/WXO-DD/
    grid: str
    dx_km: float


MODELS = {"rdps": MscModel("rdps", "RDPS", "model_rdps/10km", "RLatLon0.09", 10.0),
          "gdps": MscModel("gdps", "GDPS", "model_gdps/15km", "LatLon0.15", 15.0)}


def to_celsius(a: np.ndarray) -> np.ndarray:
    return a - 273.15 if np.nanmedian(a) > 100 else a


def to_hpa(a: np.ndarray) -> np.ndarray:
    return a / 100.0 if np.nanmedian(a) > 2000 else a


def to_percent(a: np.ndarray) -> np.ndarray:
    return a * 100.0 if np.nanmax(a) <= 1.5 else a


def z_scale(zs: dict[int, np.ndarray]) -> float:
    """Geopotential height may be gpm or decametres depending on the product.  Decide once per step from the
    upper levels (850/700/500 hPa have well separated magnitudes); the low levels (1000/925 hPa, near zero) cannot
    tell the two apart on their own.  Returns the factor that converts to metres."""
    for p in (500, 700, 850):
        if p in zs and np.isfinite(zs[p]).any():
            return 10.0 if abs(float(np.nanmedian(zs[p]))) < 0.3 * Z_EXPECT[p] else 1.0
    return 1.0


class MscSource(Source):
    step_hours = 3
    native_ptype = False                 # decision-tree precipitation type (MSC's own type fields are not used)

    def __init__(self, key: str):
        self.m = MODELS[key]
        self.name = key
        self.src_dx_km = self.m.dx_km
        self._crop = _Cropper()
        self._grid: SourceGrid | None = None
        self._lock = threading.Lock()
        self._z_warned = False

    # ------------------------------------------------------------------ naming
    def url(self, init: datetime, fxx: int, var: str, level: str) -> str:
        d, h, f = f"{init:%Y%m%d}", f"{init:%H}", f"{fxx:03d}"
        name = f"{d}T{h}Z_MSC_{self.m.name}_{var}_{level}_{self.m.grid}_PT{f}H.grib2"
        return f"{ROOT}/{d}/WXO-DD/{self.m.path}/{h}/{f}/{name}"

    def available(self, init, fxx, member) -> bool:
        try:
            r = nh._request(self.url(init, fxx, "AirTemp", "AGL-2m"), headers={"Range": "bytes=0-0"})
            return r.status_code in (200, 206)
        except nh.FetchError:
            return False

    # ------------------------------------------------------------------ IO
    def grid(self) -> SourceGrid:
        if self._grid is None:
            raise RuntimeError("grid() called before the first load_steps()")
        return self._grid

    def _read(self, url: str) -> np.ndarray:
        """Download one single-message GRIB2 file and return its (cropped) float32 field."""
        ec = nh._ec()
        with tempfile.TemporaryDirectory(prefix="pwpf_msc_") as td:
            dest = Path(td) / "f.grib2"
            nh.download_file(url, dest)
            with open(dest, "rb") as f:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    raise nh.FetchError(f"{url}: no GRIB message in file")
                try:
                    grid = nh.grid_of(ec, gid)
                    v = nh.values_of(ec, gid)
                finally:
                    ec.codes_release(gid)
        with self._lock:
            if self._grid is None:
                self._crop.setup(grid)
                self._grid = SourceGrid(self._crop.lat, self._crop.lon, f"{self.name}_{self._crop.lat.shape}")
        return self._crop(v)

    def _qpf_step(self, init, fxx) -> np.ndarray:
        """3-h accumulation (kg m-2 = mm) ending at `fxx`."""
        return self._read(self.url(init, fxx, f"Precip-Accum{self.step_hours}h", "Sfc"))

    def _step(self, init: datetime, fxx: int) -> StepFields:
        t2 = to_celsius(self._read(self.url(init, fxx, "AirTemp", "AGL-2m")))
        ps = to_hpa(self._read(self.url(init, fxx, "Pressure", "Sfc")))
        try:
            rh2 = to_percent(self._read(self.url(init, fxx, "RelativeHumidity", "AGL-2m")))
        except nh.HttpStatusError:
            rh2 = None
        raw = {}
        for p in LEVELS:
            lv = f"IsbL-{p:04d}"
            try:
                raw[p] = (to_celsius(self._read(self.url(init, fxx, "AirTemp", lv))),
                          to_percent(self._read(self.url(init, fxx, "RelativeHumidity", lv))),
                          self._read(self.url(init, fxx, "GeopotentialHeight", lv)))
            except nh.HttpStatusError as e:
                if e.status != 404:
                    raise
        k = z_scale({p: v[2] for p, v in raw.items()})
        if k != 1.0 and not self._z_warned:
            self._z_warned = True
            log.info("%s: geopotential height appears to be in decametres; converting to metres", self.name)
        levels = {p: {"t": t, "rh": rh, "z": z * k} for p, (t, rh, z) in raw.items()}
        if not levels:
            raise nh.FetchError(f"{self.name} f{fxx:03d}: no pressure-level T/RH/HGT files found")
        qpf = np.nan_to_num(self._qpf_step(init, fxx), nan=0.0)
        return StepFields(valid=init + timedelta(hours=fxx), qpf_mm=np.clip(qpf, 0, None).astype(np.float32),
                          t2m_c=t2, rh2m=rh2, psfc_hpa=ps, levels=levels, dt_h=float(self.step_hours))

    def load_steps(self, init, valid_start, valid_end, member):
        out, t = [], valid_start + timedelta(hours=self.step_hours)
        while t <= valid_end:
            out.append(self._step(init, int(round((t - init).total_seconds() / 3600))))
            t += timedelta(hours=self.step_hours)
        return out
