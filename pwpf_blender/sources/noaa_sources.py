"""Live data loaders: NOAA models through direct HTTPS/.idx byte-range access, ECMWF open data via eccodes.

Models (all 2026 operational):

========== ============================== =========================================================
loader     data                           files used
========== ============================== =========================================================
gefs       GEFS v12, 0.5 deg              pgrb2ap5/{gep01..30,geavg}.tHHz.pgrb2a.0p50.fFFF
gfs        GFS, 0.25 deg                  atmos/gfs.tHHz.pgrb2.0p25.fFFF
hrrr       HRRR, 3 km                     conus/hrrr.tHHz.wrf{sfc,prs}fFF.grib2
rrfs       RRFS deterministic, 3 km       rrfs.tHHz.{2dfld,prslev}.3km.fFFF.conus.grib2
rrfs_ens   RRFS ensemble m001-m005, 3 km  rrfsens/mNNN/rrfs.tHHz.mNNN.{2dfldnomads,prslevnomads}...
ifs_ens/hres  ECMWF open data, 0.25 deg   ecmwf-opendata client (CC-BY-4.0)
========== ============================== =========================================================

RRFS/REFS replaced NAM, NAM-nest, HiResW, HREF and SREF on 2026-11-03 (NWS SCN 26-47/26-48); those loaders
were removed.  Each source tries AWS Open Data first and NOMADS second.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import noaa_http as nh
from ..member import StepFields
from .base import Source, SourceGrid

log = logging.getLogger(__name__)

LEVELS = (1000, 925, 850, 700, 500)
_LV = "|".join(map(str, LEVELS))
PAT_LEVELS = rf"^(?:TMP|RH|HGT):(?:{_LV}) mb:"
PAT_SFC = r"^(?:APCP|CSNOW|CRAIN|CFRZR|CICEP|PRES|HGT):surface:"
PAT_2M = r"^(?:TMP|RH):2 m above ground:"
PAT_APCP = r"^APCP:surface:"

NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com"
S3 = "https://{bucket}.s3.amazonaws.com"

# CONUS-ish crop for global grids
CROP_LAT = (14.0, 62.0)
CROP_LON = (-135.0, -50.0)


class _Cropper:
    """Crop regular global lat/lon grids to the CONUS window; pass-through for projected grids."""

    def __init__(self):
        self.sl_lat = self.sl_lon = None
        self.lat = self.lon = None

    def setup(self, grid: nh.GridInfo):
        lat, lon = grid.lat, grid.lon
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


# ------------------------------------------------------------------------------------------ file plans
@dataclass(frozen=True)
class FilePlan:
    urls: tuple[str, ...]                # candidates, tried in order (AWS, then NOMADS)
    patterns: tuple[str, ...]
    has_apcp: bool = True


def plan_gefs(init: datetime, fxx: int, member) -> list[FilePlan]:
    d, h = f"{init:%Y%m%d}", f"{init:%H}"
    who = "geavg" if member == "avg" else "gec00" if member in (0, "c00") else f"gep{int(member):02d}"
    name = f"{who}.t{h}z.pgrb2a.0p50.f{fxx:03d}"
    return [FilePlan((f"{S3.format(bucket='noaa-gefs-pds')}/gefs.{d}/{h}/atmos/pgrb2ap5/{name}",
                      f"{NOMADS}/gens/prod/gefs.{d}/{h}/atmos/pgrb2ap5/{name}"),
                     (PAT_LEVELS, PAT_SFC, PAT_2M))]


def plan_gfs(init: datetime, fxx: int, member) -> list[FilePlan]:
    d, h = f"{init:%Y%m%d}", f"{init:%H}"
    name = f"gfs.t{h}z.pgrb2.0p25.f{fxx:03d}"
    return [FilePlan((f"{S3.format(bucket='noaa-gfs-bdp-pds')}/gfs.{d}/{h}/atmos/{name}",
                      f"{NOMADS}/gfs/prod/gfs.{d}/{h}/atmos/{name}"),
                     (PAT_LEVELS, PAT_SFC, PAT_2M))]


def plan_hrrr(init: datetime, fxx: int, member) -> list[FilePlan]:
    d, h = f"{init:%Y%m%d}", f"{init:%H}"

    def urls(kind):
        n = f"hrrr.t{h}z.wrf{kind}f{fxx:02d}.grib2"
        return (f"{S3.format(bucket='noaa-hrrr-bdp-pds')}/hrrr.{d}/conus/{n}", f"{NOMADS}/hrrr/prod/hrrr.{d}/conus/{n}")
    return [FilePlan(urls("sfc"), (PAT_SFC, PAT_2M)), FilePlan(urls("prs"), (PAT_LEVELS,), has_apcp=False)]


def plan_rrfs(init: datetime, fxx: int, member) -> list[FilePlan]:
    d, h = f"{init:%Y%m%d}", f"{init:%H}"

    def urls(kind):
        n = f"rrfs.t{h}z.{kind}.3km.f{fxx:03d}.conus.grib2"
        return (f"{S3.format(bucket='noaa-rrfs-ops-pds')}/rrfs.{d}/{h}/{n}", f"{NOMADS}/rrfs/prod/rrfs.{d}/{h}/{n}")
    return [FilePlan(urls("2dfld"), (PAT_SFC, PAT_2M)), FilePlan(urls("prslev"), (PAT_LEVELS,), has_apcp=False)]


def plan_rrfs_ens(init: datetime, fxx: int, member) -> list[FilePlan]:
    d, h, m = f"{init:%Y%m%d}", f"{init:%H}", f"m{int(member):03d}"

    def urls(kind):
        n = f"rrfs.t{h}z.{m}.{kind}.3km.f{fxx:03d}.conus.grib2"
        return (f"{S3.format(bucket='noaa-rrfs-ops-pds')}/rrfsens.{d}/{h}/{m}/{n}",
                f"{NOMADS}/rrfs/prod/rrfsens.{d}/{h}/{m}/{n}")
    return [FilePlan(urls("2dfldnomads"), (PAT_SFC, PAT_2M)), FilePlan(urls("prslevnomads"), (PAT_LEVELS,), has_apcp=False)]


# ------------------------------------------------------------------------------------------ source
class NoaaSource(Source):
    """One NOAA model system.  `plan(init, fxx, member)` returns the FilePlans (files + record patterns)
    for one forecast hour."""

    def __init__(self, name, plan, step_hours=3, src_dx_km=25.0, native_ptype=True):
        self.name, self.plan = name, plan
        self.step_hours, self.src_dx_km, self.native_ptype = step_hours, src_dx_km, native_ptype
        self._crop = _Cropper()
        self._grid: SourceGrid | None = None
        self._lock = threading.Lock()
        self._acc: dict = {}             # (init, fxx, member) -> {(a, b): array}; small FIFO

    def available(self, init, fxx, member) -> bool:
        """Inventory (.idx) of every file for this hour exists on a mirror (the idx is written after the GRIB)."""
        try:
            for p in self.plan(init, fxx, member):
                nh.get_inventory_first(p.urls)
        except nh.FetchError:
            return False
        return True

    # ------------------------------------------------------------------ IO
    def _fetch(self, init, fxx, member, apcp_only=False) -> dict:
        fields: dict = {}
        for p in self.plan(init, fxx, member):
            if apcp_only and not p.has_apcp:
                continue
            f, grid, url = nh.fetch_fields(p.urls, (PAT_APCP,) if apcp_only else p.patterns, want_grid=True)
            log.debug("%s f%03d: %d fields from %s", self.name, fxx, len(f), url)
            self._set_grid(grid)
            fields.update({k: self._crop(v) for k, v in f.items()})
        return fields

    def _set_grid(self, grid: nh.GridInfo):
        with self._lock:
            if self._grid is None:
                self._crop.setup(grid)
                self._grid = SourceGrid(self._crop.lat, self._crop.lon, f"{self.name}_{self._crop.lat.shape}")

    def _remember_acc(self, key, fields):
        acc = nh.acc_intervals(fields)
        with self._lock:
            self._acc[key] = acc
            # keep recent accumulations (shared by all members/threads) within ~400 MB so that the first step of
            # the next bin finds its predecessor instead of downloading the same records again
            per = max((a.nbytes for d in acc.values() for a in [d]), default=1) * max(len(acc), 1)
            cap = max(24, int(400e6 // max(per, 1)))
            while len(self._acc) > cap:
                self._acc.pop(next(iter(self._acc)))
        return acc

    def _acc_at(self, init, fxx, member) -> dict:
        key = (init, fxx, member)
        with self._lock:
            hit = self._acc.get(key)
        if hit is not None:
            return hit
        return self._remember_acc(key, self._fetch(init, fxx, member, apcp_only=True))

    def _qpf_step(self, init, t1: int, member) -> np.ndarray:
        """Accumulation over (t1 - step, t1] in mm from whatever APCP records the model provides:
        a record for exactly that interval, or the difference of two run-total (0-h) records, or of two
        records of the same bucket start."""
        t0 = t1 - self.step_hours
        cur = self._acc_at(init, t1, member)
        if (t0, t1) in cur:
            return cur[(t0, t1)]
        if t0 > 0:
            prev = self._acc_at(init, t0, member)
            if (0, t1) in cur and (0, t0) in prev:
                return np.clip(cur[(0, t1)] - prev[(0, t0)], 0, None)
            for (a, b), arr in cur.items():
                if b == t1 and a < t0 and (a, t0) in prev:
                    return np.clip(arr - prev[(a, t0)], 0, None)
        # last resort: add up the hourly (h-1, h) records, e.g. a model that only publishes 1-h buckets at some hours
        hourly = []
        for h in range(t0 + 1, t1 + 1):
            acc_h = cur if h == t1 else self._acc_at(init, h, member)
            if (h - 1, h) not in acc_h:
                hourly = None
                break
            hourly.append(acc_h[(h - 1, h)])
        if hourly:
            return np.clip(sum(hourly[1:], hourly[0]), 0, None)
        raise nh.FetchError(f"{self.name} f{t1:03d}: no usable APCP accumulation for ({t0}, {t1}] h; "
                            f"have {sorted(cur)}")

    def _step_fields(self, init, valid, member) -> StepFields:
        fxx = int(round((valid - init).total_seconds() / 3600))
        F = self._fetch(init, fxx, member)
        self._remember_acc((init, fxx, member), F)
        g = lambda var, lev: nh.find(F, var, lev)               # noqa: E731
        t2, ps = g("TMP", "2 m above ground"), g("PRES", "surface")
        if t2 is None or ps is None:
            raise nh.FetchError(f"{self.name} f{fxx:03d}: missing 2 m TMP or surface PRES; have {sorted(F)[:20]}")
        levels = {}
        for p in LEVELS:
            t, rh, z = g("TMP", f"{p} mb"), g("RH", f"{p} mb"), g("HGT", f"{p} mb")
            if t is not None and rh is not None and z is not None:
                levels[p] = {"t": t - 273.15, "rh": rh, "z": z}
        if not levels:
            raise nh.FetchError(f"{self.name} f{fxx:03d}: no pressure-level T/RH/HGT found")
        frac = None
        if self.native_ptype:
            m = {"snow": "CSNOW", "rain": "CRAIN", "fzra": "CFRZR", "sleet": "CICEP"}
            frac = {k: g(v, "surface") for k, v in m.items() if g(v, "surface") is not None} or None
        return StepFields(valid=valid, qpf_mm=self._qpf_step(init, fxx, member), t2m_c=t2 - 273.15,
                          rh2m=g("RH", "2 m above ground"), psfc_hpa=ps / 100.0, levels=levels,
                          z_sfc_m=g("HGT", "surface"), native_type_frac=frac, dt_h=float(self.step_hours))

    # ------------------------------------------------------------------ Source API
    def grid(self) -> SourceGrid:
        if self._grid is None:
            raise RuntimeError("grid() called before the first load_steps()")
        return self._grid

    def load_steps(self, init, valid_start, valid_end, member):
        out, t = [], valid_start + timedelta(hours=self.step_hours)
        while t <= valid_end:
            out.append(self._step_fields(init, t, member))
            t += timedelta(hours=self.step_hours)
        return out


# ------------------------------------------------------------------------------------------ factories
def gefs() -> NoaaSource:
    return NoaaSource("gefs", plan_gefs, 3, 55.0)


def gfs() -> NoaaSource:
    return NoaaSource("gfs", plan_gfs, 3, 27.0)


def hrrr() -> NoaaSource:
    return NoaaSource("hrrr", plan_hrrr, 3, 3.0)


def rrfs() -> NoaaSource:
    return NoaaSource("rrfs", plan_rrfs, 3, 3.0)


def rrfs_ens() -> NoaaSource:
    return NoaaSource("rrfs_ens", plan_rrfs_ens, 3, 3.0)


# ------------------------------------------------------------------------------------------ ECMWF open data
class IfsSource(Source):
    """ECMWF open data (IFS ENS 'enfo'/'pf' or HRES 'oper'/'fc') at 0.25 deg, 6-hourly steps.

    Uses the `ecmwf-opendata` client (which itself fetches byte ranges) and decodes with eccodes.  The
    download source defaults to AWS (override with env PWPF_IFS_SOURCE=ecmwf|aws|azure|google).  No 2 m RH is
    published, so RH2m defaults to 90 % inside precipitation (see member.process_bin).  PWPF uses the
    decision tree for non-NCEP models, so ECMWF's own ptype is not used."""

    step_hours = 6
    src_dx_km = 28.0
    native_ptype = False

    def __init__(self, name: str, ensemble: bool, source: str | None = None, save_dir=None):
        self.name, self.ensemble = name, ensemble
        self.source = source or os.environ.get("PWPF_IFS_SOURCE", "aws")
        self.save_dir = Path(save_dir or tempfile.mkdtemp(prefix="pwpf_ifs_"))
        self._crop = _Cropper()
        self._grid = None
        self._lock = threading.Lock()
        self._span: tuple[datetime, datetime] | None = None
        self._dl_locks: dict[str, threading.Lock] = {}

    def set_span(self, start: datetime, end: datetime) -> None:
        """Valid-time span the whole run will need.  With it each member's file is fetched once for the span
        instead of once per 6-h bin (which re-downloaded the boundary steps and ~2.7x the bytes)."""
        self._span = (start, end)

    def grid(self) -> SourceGrid:
        if self._grid is None:
            raise RuntimeError("grid() called before the first load_steps()")
        return self._grid

    def _retrieve(self, init, steps, member, levtype) -> Path:
        target = self.save_dir / f"{self.name}_{init:%Y%m%d%H}_{member}_{levtype}_{steps[0]}-{steps[-1]}.grib2"
        with self._lock:
            dl_lock = self._dl_locks.setdefault(target.name, threading.Lock())
        with dl_lock:                      # two bins asking for the same member's file must not both download it
            self._download(target, init, steps, member, levtype)
        return target

    def _download(self, target: Path, init, steps, member, levtype) -> None:
        if not target.exists():
            os.environ.setdefault("TQDM_DISABLE", "1")       # multiurl's per-file bars flood the CI log
            try:
                from ecmwf.opendata import Client
            except ImportError as e:                                     # pragma: no cover
                raise nh.FetchError("ecmwf-opendata is required for IFS (pip install -r requirements-live.txt)") from e

            req = dict(date=init.strftime("%Y%m%d"), time=init.hour, step=steps, target=str(target))
            if self.ensemble:
                req.update(stream="enfo", type="pf", number=[int(member)])
            else:
                req.update(stream="oper", type="fc")
            if levtype == "sfc":
                req.update(param=["tp", "2t", "sp"])
            else:
                req.update(levtype="pl", levelist=list(LEVELS), param=["t", "gh", "r"])
            try:
                # ecmwf-opendata defaults to 500 retries x 120 s; keep it to a few, and count its requests
                # (1 index + about one ranged GET per step) against the host's rate limit
                host = {"aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/",
                        "azure": "https://ecmwf.blob.core.windows.net/",
                        "google": "https://storage.googleapis.com/"}.get(self.source, "https://data.ecmwf.int/")
                with nh.polite(host, cost=1 + len(steps)):
                    Client(source=self.source, maximum_retries=5, retry_after=20).retrieve(**req)
            except Exception as e:                                   # noqa: BLE001
                raise nh.FetchError(f"ECMWF open data ({self.source}) request failed: {type(e).__name__}: {e}; "
                                    f"request={ {k: v for k, v in req.items() if k != 'target'} }") from e

    def _decode(self, path: Path, steps=None) -> dict:
        """{(shortName, typeOfLevel, level, step_h): cropped float32 array}"""
        ec = nh._ec()
        out = {}
        with open(path, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    if steps is not None and int(ec.codes_get(gid, "endStep")) not in steps:
                        continue                                      # `finally` still releases gid
                    if self._grid is None:
                        with self._lock:
                            if self._grid is None:
                                self._crop.setup(nh.grid_of(ec, gid))
                                self._grid = SourceGrid(self._crop.lat, self._crop.lon,
                                                        f"{self.name}_{self._crop.lat.shape}")
                    key = (ec.codes_get(gid, "shortName"), ec.codes_get(gid, "typeOfLevel"),
                           int(ec.codes_get(gid, "level")), int(ec.codes_get(gid, "endStep")))
                    out[key] = self._crop(nh.values_of(ec, gid))
                finally:
                    ec.codes_release(gid)
        return out

    def load_steps(self, init, valid_start, valid_end, member):
        f0 = int((valid_start - init).total_seconds() / 3600)
        f1 = int((valid_end - init).total_seconds() / 3600)
        steps = list(range(f0, f1 + 1, self.step_hours))
        fetch = steps
        if self._span is not None:
            h = self.step_hours
            lo = max(0, int((self._span[0] - init).total_seconds() // 3600) // h * h)
            hi = -(-int((self._span[1] - init).total_seconds() // 3600) // h) * h
            if lo <= f0 and hi >= f1:
                fetch = list(range(lo, hi + 1, h))
        want = set(steps)
        sfc = self._decode(self._retrieve(init, fetch, member, "sfc"), want)
        pl = self._decode(self._retrieve(init, fetch, member, "pl"), want)
        get = lambda d, name, tol, lev, s: d.get((name, tol, lev, s))      # noqa: E731
        out = []
        for k in range(1, len(steps)):
            s0, s1 = steps[k - 1], steps[k]
            tp0, tp1 = get(sfc, "tp", "surface", 0, s0), get(sfc, "tp", "surface", 0, s1)
            t2, sp = get(sfc, "2t", "heightAboveGround", 2, s1), get(sfc, "sp", "surface", 0, s1)
            if tp1 is None or t2 is None or sp is None or (tp0 is None and s0 != 0):
                raise nh.FetchError(f"{self.name}: missing tp/2t/sp at step {s1} h (have "
                                    f"{sorted({(a, d) for a, b, c, d in sfc})[:12]})")
            tp = (tp1 - (tp0 if tp0 is not None else 0.0)) * 1000.0          # m -> mm
            levels = {}
            for p in LEVELS:
                t, r, z = (get(pl, n, "isobaricInhPa", p, s1) for n in ("t", "r", "gh"))
                if t is not None and r is not None and z is not None:
                    levels[p] = {"t": t - 273.15, "rh": r, "z": z}
            if not levels:
                raise nh.FetchError(f"{self.name}: no pressure-level fields at step {s1} h")
            out.append(StepFields(valid=init + timedelta(hours=s1), qpf_mm=np.clip(tp, 0, None).astype(np.float32),
                                  t2m_c=t2 - 273.15, psfc_hpa=sp / 100.0, levels=levels, rh2m=None,
                                  dt_h=float(s1 - s0)))
        return out
