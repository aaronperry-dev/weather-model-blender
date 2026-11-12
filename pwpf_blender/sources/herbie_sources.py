"""Live data loaders (NOAA via Herbie, ECMWF via ecmwf-opendata).

IMPORTANT - these loaders were written from the public feed documentation (GEFS pgrb2a record list,
ECMWF open-data parameter table, Herbie model templates).  They could NOT be exercised against the
live servers in the environment where this repository was authored, so expect to adjust variable
names / search strings on first contact.  The scientific core is independent of them: every loader
just has to return `StepFields` on its native grid.

Optional dependencies: herbie-data, xarray, cfgrib, eccodes, ecmwf-opendata.
"""
from __future__ import annotations

import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..member import StepFields
from .base import Source, SourceGrid

LEVELS = (1000, 925, 850, 700, 500)
LEVEL_RE = r":(?:TMP|RH|HGT):(?:" + "|".join(map(str, LEVELS)) + r") mb"
SFC_RE = (r":(?:APCP|PRES|HGT|CSNOW|CRAIN|CFRZR|CICEP):surface"
          r"|:(?:TMP|RH):2 m above ground")

# CONUS-ish crop for global grids (indices computed once from the coordinates)
CROP_LAT = (14.0, 62.0)
CROP_LON = (-135.0, -50.0)


def _need(mod: str):
    try:
        return __import__(mod)
    except ImportError as e:                                     # pragma: no cover
        raise RuntimeError(f"live loaders need the optional package '{mod}' "
                           f"(pip install -r requirements-live.txt)") from e


def _arrays(ds_or_list) -> dict:
    """Flatten the Dataset(s) returned by Herbie/cfgrib into {shortName: DataArray}."""
    out = {}
    for ds in (ds_or_list if isinstance(ds_or_list, list) else [ds_or_list]):
        for name, da in ds.data_vars.items():
            out[name] = da
    return out


def _step_range_start(da) -> int | None:
    sr = str(da.attrs.get("GRIB_stepRange", ""))
    m = re.match(r"^(\d+)-(\d+)$", sr)
    return int(m.group(1)) if m else None


class _Cropper:
    """Crop regular global lat/lon grids to the CONUS window; pass-through for projected grids."""

    def __init__(self):
        self.sl_lat = self.sl_lon = None
        self.lat = self.lon = None

    def setup(self, da):
        lat = da["latitude"].values
        lon = da["longitude"].values
        if lat.ndim == 1:
            lon180 = ((lon + 180) % 360) - 180
            ilat = np.where((lat >= CROP_LAT[0]) & (lat <= CROP_LAT[1]))[0]
            ilon = np.where((lon180 >= CROP_LON[0]) & (lon180 <= CROP_LON[1]))[0]
            self.sl_lat = slice(ilat.min(), ilat.max() + 1)
            self.sl_lon = slice(ilon.min(), ilon.max() + 1)
            self.lat, self.lon = lat[self.sl_lat], lon180[self.sl_lon]
        else:
            self.lat, self.lon = lat, ((lon + 180) % 360) - 180

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if self.sl_lat is None:
            return arr
        return arr[..., self.sl_lat, self.sl_lon]


class HerbieSource(Source):
    """Generic NOAA model through Herbie.

    file_specs: list of (herbie_kwargs, search_regex).  Single-file models (GEFS, GFS, NAM nest,
    HiResW) use one entry; HRRR uses two ('sfc' + 'prs')."""

    def __init__(self, name, model, file_specs, step_hours=3, src_dx_km=25.0, native_ptype=True,
                 save_dir=None, member_kw="member"):
        self.name, self.model = name, model
        self.file_specs = file_specs
        self.step_hours, self.src_dx_km, self.native_ptype = step_hours, src_dx_km, native_ptype
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="pwpf_")
        self.member_kw = member_kw
        self._crop = _Cropper()
        self._grid: SourceGrid | None = None
        self._cache: dict = {}

    # -------------------------------------------------------------- IO
    def _read(self, init: datetime, fxx: int, member) -> dict:
        key = (init, fxx, member)
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) > 24:                       # tiny cache: the previous step is needed for QPF
            self._cache.pop(next(iter(self._cache)))
        Herbie = _need("herbie").Herbie
        arrs = {}
        for kw, regex in self.file_specs:
            kw = dict(kw)
            if member is not None and self.member_kw:
                kw[self.member_kw] = member
            H = Herbie(init, model=self.model, fxx=fxx, save_dir=self.save_dir, verbose=False, **kw)
            arrs.update(_arrays(H.xarray(regex, remove_grib=True, verbose=False)))
        self._cache[key] = arrs
        return arrs

    def _qpf_step(self, init, f_end: int, member) -> np.ndarray:
        """Accumulation over (f_end - step_hours, f_end] in mm, handling running totals and buckets."""
        a_end = self._read(init, f_end, member)
        tp = a_end["tp"]
        field = np.asarray(tp.values, dtype=np.float32)
        prev = f_end - self.step_hours
        start = _step_range_start(tp)
        if start is None or start >= prev:
            return field                                  # already a per-step bucket (or instantaneous)
        a_prev = self._read(init, prev, member)
        if _step_range_start(a_prev["tp"]) == start:
            return field - np.asarray(a_prev["tp"].values, dtype=np.float32)
        return field

    def _step_fields(self, init, valid, member) -> StepFields:
        fxx = int(round((valid - init).total_seconds() / 3600))
        arrs = self._read(init, fxx, member)
        if self._grid is None:
            self._crop.setup(arrs["t2m"] if "t2m" in arrs else next(iter(arrs.values())))
            self._grid = SourceGrid(self._crop.lat, self._crop.lon, f"{self.name}_{self._crop.lat.shape}")
        c = self._crop
        t = arrs["t"]
        plevs = [int(p) for p in t["isobaricInhPa"].values]
        levels = {}
        for i, p in enumerate(plevs):
            if p in LEVELS and "r" in arrs and "gh" in arrs:
                levels[p] = {"t": c(t.values[i]) - 273.15, "rh": c(arrs["r"].values[i]),
                             "z": c(arrs["gh"].values[i])}
        frac = None
        if self.native_ptype and "csnow" in arrs:
            frac = {k: c(arrs[v].values) for k, v in
                    (("snow", "csnow"), ("rain", "crain"), ("fzra", "cfrzr"), ("sleet", "cicep")) if v in arrs}
        return StepFields(
            valid=valid,
            qpf_mm=c(self._qpf_step(init, fxx, member)),
            t2m_c=c(arrs["t2m"].values) - 273.15,
            rh2m=c(arrs["r2"].values) if "r2" in arrs else None,
            psfc_hpa=c(arrs["sp"].values) / 100.0,
            levels=levels,
            z_sfc_m=c(arrs["orog"].values) if "orog" in arrs else None,
            native_type_frac=frac,
        )

    # -------------------------------------------------------------- Source API
    def grid(self) -> SourceGrid:
        if self._grid is None:
            # prime the grid from a first analysis-time read
            raise RuntimeError("grid() called before first load_steps(); pipeline primes it")
        return self._grid

    def prime(self, init: datetime, member):
        if self._grid is None:
            self._step_fields(init, init + timedelta(hours=self.step_hours), member)

    def load_steps(self, init, valid_start, valid_end, member):
        self.prime(init, member)
        out, t = [], valid_start + timedelta(hours=self.step_hours)
        while t <= valid_end:
            out.append(self._step_fields(init, t, member))
            t += timedelta(hours=self.step_hours)
        return out


# ---------------------------------------------------------------------- factories
def gefs(member_label="gefs") -> HerbieSource:
    spec = [({"product": "atmos.5"}, f"{LEVEL_RE}|{SFC_RE}")]
    return HerbieSource(member_label, "gefs", spec, step_hours=3, src_dx_km=55.0, native_ptype=True)


def gfs() -> HerbieSource:
    spec = [({"product": "pgrb2.0p25"}, f"{LEVEL_RE}|{SFC_RE}")]
    return HerbieSource("gfs", "gfs", spec, step_hours=3, src_dx_km=27.0, native_ptype=True, member_kw=None)


def hrrr() -> HerbieSource:
    spec = [({"product": "sfc"}, SFC_RE), ({"product": "prs"}, LEVEL_RE)]
    return HerbieSource("hrrr", "hrrr", spec, step_hours=3, src_dx_km=3.0, native_ptype=True, member_kw=None)


def nam_nest() -> HerbieSource:
    spec = [({"product": "conusnest.hiresf"}, f"{LEVEL_RE}|{SFC_RE}")]
    return HerbieSource("nam_nest", "nam", spec, step_hours=3, src_dx_km=3.0, native_ptype=True, member_kw=None)


def hiresw_arw() -> HerbieSource:
    spec = [({"product": "arw_2p5km"}, f"{LEVEL_RE}|{SFC_RE}")]
    return HerbieSource("hiresw_arw", "hiresw", spec, step_hours=3, src_dx_km=3.2, native_ptype=True)


def hiresw_fv3() -> HerbieSource:
    spec = [({"product": "fv3_2p5km"}, f"{LEVEL_RE}|{SFC_RE}")]
    return HerbieSource("hiresw_fv3", "hiresw", spec, step_hours=3, src_dx_km=3.2, native_ptype=True, member_kw=None)


# ---------------------------------------------------------------------- ECMWF open data
class IfsSource(Source):
    """ECMWF open data (IFS ENS 'enfo'/'pf' or HRES 'oper'/'fc') at 0.25 deg, 6-hourly steps.

    Uses the ecmwf-opendata client (CC-BY-4.0 data).  No 2 m RH is published, so RH2m defaults to 90 %
    inside precipitation (see member.process_bin).  PWPF uses the decision tree for non-NCEP models, so
    ECMWF's own ptype field is not used."""

    step_hours = 6
    src_dx_km = 28.0
    native_ptype = False

    def __init__(self, name: str, ensemble: bool, source: str = "ecmwf", save_dir=None):
        self.name, self.ensemble, self.source = name, ensemble, source
        self.save_dir = Path(save_dir or tempfile.mkdtemp(prefix="pwpf_ifs_"))
        self._crop = _Cropper()
        self._grid = None

    def grid(self) -> SourceGrid:
        if self._grid is None:
            raise RuntimeError("grid() called before first load_steps()")
        return self._grid

    def _retrieve(self, init, steps, member, levtype):
        Client = _need("ecmwf.opendata").Client
        tag = f"{self.name}_{init:%Y%m%d%H}_{member}_{levtype}_{steps[0]}-{steps[-1]}.grib2"
        target = self.save_dir / tag
        if not target.exists():
            req = dict(date=init.strftime("%Y%m%d"), time=init.hour, step=steps, target=str(target))
            if self.ensemble:
                req.update(stream="enfo", type="pf", number=[int(member)])
            else:
                req.update(stream="oper", type="fc")
            if levtype == "sfc":
                req.update(param=["tp", "2t", "sp"])
            else:
                req.update(levtype="pl", levelist=list(LEVELS), param=["t", "gh", "r"])
            Client(source=self.source).retrieve(**req)
        return target

    def _open(self, path, filt):
        xr = _need("xarray")
        _need("cfgrib")
        return xr.open_dataset(path, engine="cfgrib",
                               backend_kwargs={"indexpath": "", "filter_by_keys": filt})

    def load_steps(self, init, valid_start, valid_end, member):
        f0 = int((valid_start - init).total_seconds() / 3600)
        f1 = int((valid_end - init).total_seconds() / 3600)
        steps = list(range(f0, f1 + 1, self.step_hours))
        sfc = self._retrieve(init, steps, member, "sfc")
        pl = self._retrieve(init, steps, member, "pl")
        d_tp = self._open(sfc, {"typeOfLevel": "surface", "shortName": "tp"})
        d_sp = self._open(sfc, {"typeOfLevel": "surface", "shortName": "sp"})
        d_2t = self._open(sfc, {"typeOfLevel": "heightAboveGround", "shortName": "2t"})
        d_pl = self._open(pl, {"typeOfLevel": "isobaricInhPa"})
        if self._grid is None:
            self._crop.setup(d_tp["tp"])
            self._grid = SourceGrid(self._crop.lat, self._crop.lon, f"{self.name}_{self._crop.lat.shape}")
        c = self._crop
        scale = 1000.0 if str(d_tp["tp"].attrs.get("GRIB_units", "m")) == "m" else 1.0
        out = []
        for k in range(1, len(steps)):
            tp = (d_tp["tp"].isel(step=k).values - d_tp["tp"].isel(step=k - 1).values) * scale
            levels = {}
            for i, p in enumerate(d_pl["isobaricInhPa"].values):
                p = int(p)
                if p in LEVELS:
                    levels[p] = {"t": c(d_pl["t"].isel(step=k, isobaricInhPa=i).values) - 273.15,
                                 "rh": c(d_pl["r"].isel(step=k, isobaricInhPa=i).values),
                                 "z": c(d_pl["gh"].isel(step=k, isobaricInhPa=i).values)}
            out.append(StepFields(
                valid=init + timedelta(hours=steps[k]),
                qpf_mm=c(np.clip(tp, 0, None).astype(np.float32)),
                t2m_c=c(d_2t["t2m"].isel(step=k).values) - 273.15,
                psfc_hpa=c(d_sp["sp"].isel(step=k).values) / 100.0,
                levels=levels, rh2m=None))
        return out
