"""Exporters: web data layers (`.bin.gz`, Web Mercator), manifest.json, and full-precision npz.

Web encoding (format "PWPB" v1)
-------------------------------
Each window is one gzip-compressed binary file of equally sized planes on the Web-Mercator grid (little endian):

    header   "PWPB" | u8 version=1 | u8 nplanes | u8 flags (bit0: 16-bit planes are byte-shuffled) | u8 0 | u16 W | u16 H
    per plane (12 B)   u8 id | u8 bits (8|16) | u8 kind (0 log, 1 linear) | u8 0 | f32 p0 | f32 p1
    plane data, in the order listed (byte-shuffled 16-bit: all high bytes, then all low bytes)

    ids: 1 mode, 2 sigma_low, 3 sigma_high, 4 potential snowfall (in), 5 snow level (m), 6 model terrain (m)
    kind 0 (log):    x = p0 * (exp(v / vmax * ln(1 + p1 / p0)) - 1)       p0 = A, p1 = xmax, vmax = 2^bits - 1
    kind 1 (linear): x = v * p0 + p1
Snowfall layers use A = 0.5, xmax = 200 in; freezing-rain layers (`<id>_ice.bin.gz`, planes 1-3 only) A = 0.02, xmax = 2 in.
16-bit planes give ~0.01 % precision (the 8-bit option, `web.bin_bits: 8`, ~2 %).  The page rebuilds exceedance
probabilities and percentiles in the browser with the same binormal CDF (web/binormal.js), so one file serves every
threshold/percentile product.  Planes 4-6 feed the browser's terrain adjustment (see web/terrain.js); the model terrain is
static and written once as `terrain_ref.bin.gz`.
"""
from __future__ import annotations

import gzip
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from . import __version__

ENC_A = 0.5
ENC_XMAX = 200.0
ICE_ENC_A = 0.02             # ice amounts are ~100x smaller than snowfall: finer log scale (~2 % from 0.002 in to 2 in)
ICE_ENC_XMAX = 2.0
R_MERC = 6378137.0


@dataclass
class MercatorGrid:
    xmin: float
    ymax: float
    px: float
    width: int
    height: int
    row: np.ndarray        # fractional source row per output pixel (H, W)
    col: np.ndarray
    inside: np.ndarray

    @classmethod
    def build(cls, grid, px_m: float = 3000.0) -> "MercatorGrid":
        lon, lat = grid.lonlat()
        xs = R_MERC * np.radians(lon)
        ys = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
        xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        w = int(math.ceil((xmax - xmin) / px_m))
        h = int(math.ceil((ymax - ymin) / px_m))
        xc = xmin + (np.arange(w) + 0.5) * px_m
        yc = ymax - (np.arange(h) + 0.5) * px_m
        X, Y = np.meshgrid(xc, yc)
        lonp = np.degrees(X / R_MERC)
        latp = np.degrees(2 * np.arctan(np.exp(Y / R_MERC)) - np.pi / 2)
        row, col = grid.lonlat_to_rowcol(lonp, latp)
        inside = (row >= 0) & (row <= grid.ny - 1) & (col >= 0) & (col <= grid.nx - 1)
        return cls(xmin, ymax, px_m, w, h, row.astype(np.float32), col.astype(np.float32), inside)

    def reproject(self, field: np.ndarray) -> np.ndarray:
        out = map_coordinates(field.astype(np.float32), [self.row, self.col], order=1, mode="nearest")
        return np.where(self.inside, out, 0.0)

    def bounds_latlon(self):
        def lat(y):
            return math.degrees(2 * math.atan(math.exp(y / R_MERC)) - math.pi / 2)
        ymin = self.ymax - self.height * self.px
        xmax = self.xmin + self.width * self.px
        return [[lat(ymin), math.degrees(self.xmin / R_MERC)], [lat(self.ymax), math.degrees(xmax / R_MERC)]]


MAGIC = b"PWPB"
P_MODE, P_LO, P_HI, P_POT, P_SL, P_TERRAIN = 1, 2, 3, 4, 5, 6
SL_SCALE_8 = 25.0
SL_SCALE, SL_OFFSET = 0.1, -500.0                   # snow level (m): u16 = (m + 500) / 0.1
TERRAIN_SCALE, TERRAIN_OFFSET = 0.25, -500.0        # terrain (m):    u16 = (m + 500) / 0.25


def encode_log(x: np.ndarray, A: float, xmax: float, bits: int = 16) -> np.ndarray:
    vmax = (1 << bits) - 1
    x = np.clip(np.nan_to_num(x, nan=0.0), 0, xmax)
    v = np.clip(np.round(vmax * np.log1p(x / A) / math.log1p(xmax / A)), 0, vmax)
    return v.astype(np.uint8 if bits == 8 else np.uint16)


def encode_linear(x: np.ndarray, scale: float, offset: float, bits: int = 16) -> np.ndarray:
    vmax = (1 << bits) - 1
    return np.clip(np.round((np.nan_to_num(x, nan=offset) - offset) / scale), 0, vmax).astype(np.uint8 if bits == 8 else np.uint16)


def pack_planes(width: int, height: int, planes: list[dict], shuffle: bool = True) -> bytes:
    """planes: [{"id", "bits", "kind": "log"|"linear", "p0", "p1", "data": uint array (H, W)}]"""
    head = MAGIC + struct.pack("<BBBBHH", 1, len(planes), 1 if shuffle else 0, 0, width, height)
    dirs, body = b"", b""
    for p in planes:
        d = np.ascontiguousarray(p["data"])
        assert d.shape == (height, width) and d.dtype == (np.uint8 if p["bits"] == 8 else np.uint16)
        dirs += struct.pack("<BBBBff", p["id"], p["bits"], 0 if p["kind"] == "log" else 1, 0, p["p0"], p["p1"])
        if p["bits"] == 16 and shuffle:
            le = d.astype("<u2").view(np.uint8).reshape(-1, 2)
            body += le[:, 1].tobytes() + le[:, 0].tobytes()
        else:
            body += d.astype("<u2" if p["bits"] == 16 else np.uint8).tobytes()
    return head + dirs + body


def write_bin(path: Path, width: int, height: int, planes: list[dict], shuffle: bool = True) -> None:
    raw = pack_planes(width, height, planes, shuffle)
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0) as g:
        g.write(raw)


def read_bin(src) -> dict:
    """Inverse of write_bin (tests, tooling): {"width", "height", "planes": {id: float64 (H, W)}}."""
    raw = Path(src).read_bytes() if not isinstance(src, (bytes, bytearray)) else bytes(src)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:4] == MAGIC, "not a PWPB file"
    ver, n, flags, _, w, h = struct.unpack_from("<BBBBHH", raw, 4)
    assert ver == 1
    pos, ents = 12, []
    for _ in range(n):
        ents.append(struct.unpack_from("<BBBBff", raw, pos))
        pos += 12
    out = {}
    for pid, bits, kind, _, p0, p1 in ents:
        cnt = w * h
        if bits == 16:
            if flags & 1:
                hi = np.frombuffer(raw, np.uint8, cnt, pos).astype(np.uint16)
                lo = np.frombuffer(raw, np.uint8, cnt, pos + cnt).astype(np.uint16)
                v = (hi << 8) | lo
            else:
                v = np.frombuffer(raw, "<u2", cnt, pos).astype(np.uint16)
            pos += 2 * cnt
        else:
            v = np.frombuffer(raw, np.uint8, cnt, pos)
            pos += cnt
        v = v.astype(np.float64).reshape(h, w)
        vmax = (1 << bits) - 1
        out[pid] = p0 * np.expm1(v / vmax * math.log1p(p1 / p0)) if kind == 0 else v * p0 + p1
    return {"width": w, "height": h, "planes": out}


def _write_layer(mg, fit, path: Path, A: float, xmax: float, bits: int = 16, aux: bool = False):
    inside = mg.inside
    planes = []
    for pid, key in ((P_MODE, "mode"), (P_LO, "s_lo"), (P_HI, "s_hi")):
        d = encode_log(mg.reproject(fit[key]), A, xmax, bits)
        d[~inside] = 0                               # outside the model domain: all zero (= no signal)
        planes.append({"id": pid, "bits": bits, "kind": "log", "p0": A, "p1": xmax, "data": d})
    if aux and fit.get("pot") is not None:
        d = encode_log(mg.reproject(fit["pot"]), A, xmax, bits)
        d[~inside] = 0
        planes.append({"id": P_POT, "bits": bits, "kind": "log", "p0": A, "p1": xmax, "data": d})
        scale = SL_SCALE if bits == 16 else SL_SCALE_8      # 0.1 m steps (16 bit) or 25 m steps (8 bit, range -500..5875 m)
        sl = encode_linear(mg.reproject(fit["sl"]), scale, SL_OFFSET, bits)
        sl[~inside] = encode_linear(np.array([5000.0]), scale, SL_OFFSET, bits)[0]
        planes.append({"id": P_SL, "bits": bits, "kind": "linear", "p0": scale, "p1": SL_OFFSET, "data": sl})
    write_bin(path, mg.width, mg.height, planes)


def _write_terrain(mg, terrain: np.ndarray, path: Path):
    t = encode_linear(mg.reproject(terrain), TERRAIN_SCALE, TERRAIN_OFFSET)
    write_bin(path, mg.width, mg.height, [{"id": P_TERRAIN, "bits": 16, "kind": "linear", "p0": TERRAIN_SCALE,
                                           "p1": TERRAIN_OFFSET, "data": t}])


def _iso(t) -> str:
    return t.replace(tzinfo=timezone.utc).isoformat()


DEFAULT_TERRAIN = {
    "enabled": True,
    "dem_url": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",   # AWS Terrain Tiles (Terrarium)
    "dem_max_zoom": 12,                 # finer map zooms reuse (upsample) this level
    "min_zoom": 5,                      # no adjustment below this map zoom (cells are far smaller than the data)
    "precip_gradient_per_km": 0.20,     # +20 % snowfall per km of elevation above the model terrain (our assumption)
    "precip_factor_range": [0.75, 1.35],
    "snow_level_width_m": 100.0,        # width of the rain/snow transition around the 0.5 degC wet-bulb level
    "max_phase_ratio": 3.0,             # cap on (snow fraction at the DEM height) / (at the model height)
    "hillshade": True,                  # offer a terrain-shading toggle in the viewer
}


def _clean_old(out: Path):
    for pat in ("s[0-9]*_f[0-9]*.bin.gz", "s[0-9]*_f[0-9]*.png", "s[0-9]*_f[0-9]*_ice.png", "terrain_ref.bin.gz"):
        for old in list(out.glob(pat)):        # drop layers of a previous cycle (window ids change) and legacy PNG layers
            old.unlink()


def write_site(results, cfg: dict, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    _clean_old(out)
    webcfg = cfg.get("web", {})
    bits = int(webcfg.get("bin_bits", 8))
    tcfg = dict(DEFAULT_TERRAIN, **(webcfg.get("terrain") or {}))
    use_terrain = bool(tcfg["enabled"]) and getattr(results, "terrain", None) is not None
    mg = MercatorGrid.build(results.grid, webcfg.get("px_m", 3000.0))
    wins = []
    for wid, w in results.windows.items():
        _write_layer(mg, w, out / f"{wid}.bin.gz", ENC_A, ENC_XMAX, bits, aux=use_terrain)
        entry = {"id": wid, "label": w["label"], "length_h": w["length_h"], "end_h": w["end_h"],
                 "file": f"{wid}.bin.gz", "start": _iso(w["start"]), "end": _iso(w["end"]), "ice_file": None,
                 "n_members": int(w.get("n_members", results.meta.get("n_members") or 0)),
                 "reduced": bool(w.get("reduced", False))}
        if wid in results.ice:
            _write_layer(mg, results.ice[wid], out / f"{wid}_ice.bin.gz", ICE_ENC_A, ICE_ENC_XMAX, bits)
            entry["ice_file"] = f"{wid}_ice.bin.gz"
        wins.append(entry)
    terrain_block = None
    if use_terrain:
        _write_terrain(mg, results.terrain, out / "terrain_ref.bin.gz")
        terrain_block = dict(tcfg, file="terrain_ref.bin.gz")
    groups: dict = {}
    for m in results.members:
        g = groups.setdefault(m["group"], {"group": m["group"], "members": 0, "requested": 0, "inits": set()})
        g["members"] += int(m["ok"])
        g["requested"] += 1
        g["inits"].add(m["init"])
    manifest = {
        "version": __version__,
        "cycle": results.cycle.replace(tzinfo=timezone.utc).isoformat(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "windows": wins,
        "products": {
            "snow": {"label": "Snowfall", "unit": "in", "decimals": 1, "file_key": "file",
                     "encoding": {"A": ENC_A, "xmax": ENC_XMAX},
                     "thresholds": cfg.get("thresholds_in", [1, 2, 4, 6, 8, 12, 18, 24])},
            "ice": {"label": "Freezing rain (ice)", "unit": "in", "decimals": 2, "file_key": "ice_file",
                    "encoding": {"A": ICE_ENC_A, "xmax": ICE_ENC_XMAX},
                    "thresholds": cfg.get("thresholds_ice_in", [0.01, 0.05, 0.1, 0.25, 0.5]),
                    "report": cfg.get("ice", {}).get("report", "flat")},
        },
        "encoding": {"A": ENC_A, "xmax": ENC_XMAX},
        "size": [mg.width, mg.height],
        "mercator": {"xmin": mg.xmin, "ymax": mg.ymax, "px": mg.px},
        "bounds": mg.bounds_latlon(),
        "thresholds_in": cfg.get("thresholds_in", [1, 2, 4, 6, 8, 12, 18, 24]),
        "percentiles": cfg.get("percentiles", [5, 10, 25, 50, 75, 90, 95]),
        "members_total": results.meta.get("n_members"),
        "members_requested": results.meta.get("n_requested", results.meta.get("n_members")),
        "groups": [{"group": g["group"], "members": g["members"], "requested": g["requested"],
                    "inits": sorted(g["inits"])} for g in groups.values()],
        "members": results.members,
        "windows_skipped": results.skipped,
        "format": "pwpb1",
        "bits": bits,
        "terrain": terrain_block,
        "mode_method": cfg.get("mode", {}).get("method", "pmm"),
        "slr": cfg.get("slr", {}),
        "accumulation": ({k: v for k, v in __import__("pwpf_blender.accumulation", fromlist=["x"]).settings(cfg.get("accumulation")).items()}
                         if (cfg.get("accumulation") or {}).get("enabled", True) else None),
        "calibration_applied": results.meta.get("calibration_applied"),
        "grid_stride": results.grid.stride,
        "prism_downscale": bool(results.meta.get("prism_downscale", False)),
        "notice": "Unofficial guidance built from public WPC PWPF documentation. Not a National Weather Service product.",
    }
    cal = out / "calibration.txt"
    txt = getattr(results, "calibration", "")
    if txt and (cfg.get("calibration") or {}).get("publish", True):
        cal.write_text(txt.strip() + "\n")
        manifest["calibration"] = "calibration.txt"
    elif cal.exists():
        cal.unlink()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out / "manifest.json"


def save_npz(results, path: str | Path):
    arrs = {}
    for wid, w in results.windows.items():
        for k in ("mode", "s_lo", "s_hi", "std"):
            arrs[f"{wid}_{k}"] = w[k].astype(np.float32)
    for wid, w in results.ice.items():
        for k in ("mode", "s_lo", "s_hi", "std"):
            arrs[f"{wid}_ice_{k}"] = w[k].astype(np.float32)
    arrs["meta"] = np.array(json.dumps({"cycle": results.cycle.isoformat(), "stride": results.grid.stride,
                                         "windows": {k: [v["start"].isoformat(), v["end"].isoformat()]
                                                     for k, v in results.windows.items()},
                                         "ice_windows": sorted(results.ice)}))
    np.savez_compressed(path, **arrs)
