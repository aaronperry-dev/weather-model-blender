"""Exporters: web data-PNGs (Web Mercator), manifest.json, and full-precision npz.

Web encoding
------------
For each window one RGB PNG holds the three binormal parameters per pixel:
    R = mode, G = sigma_low, B = sigma_high      (inches)
each log-encoded to 8 bits:   v = round(255 * ln(1 + x/A) / ln(1 + XMAX/A)),   A = 0.5, XMAX = 200
(~2 % relative precision from 0.01 in to 200 in).  Freezing-rain layers (`<id>_ice.png`) use the same scheme
with A = 0.02, XMAX = 2 in.  The page rebuilds exceedance probabilities and
percentiles in the browser with the same binormal CDF (web/binormal.js), so one small image serves
every threshold/percentile product.  Alpha is always 255 so browsers never premultiply the data.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from . import __version__

ENC_A = 0.5
ENC_XMAX = 200.0
ICE_ENC_A = 0.02             # ice amounts are ~100x smaller than snowfall: finer log scale (~2 % from 0.002 in to 2 in)
ICE_ENC_XMAX = 2.0
R_MERC = 6378137.0


def encode(x: np.ndarray, A: float = ENC_A, xmax: float = ENC_XMAX) -> np.ndarray:
    x = np.clip(np.nan_to_num(x, nan=0.0), 0, xmax)
    return np.clip(np.round(255 * np.log1p(x / A) / math.log1p(xmax / A)), 0, 255).astype(np.uint8)


def decode(v: np.ndarray, A: float = ENC_A, xmax: float = ENC_XMAX) -> np.ndarray:
    return A * np.expm1(np.asarray(v, dtype=np.float64) / 255 * math.log1p(xmax / A))


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


def _write_layer(mg, fit, path: Path, A: float, xmax: float):
    rgb = np.stack([encode(mg.reproject(fit["mode"]), A, xmax), encode(mg.reproject(fit["s_lo"]), A, xmax),
                    encode(mg.reproject(fit["s_hi"]), A, xmax)], axis=-1)
    rgb[~mg.inside] = 0                          # outside the model domain: all channels 0 (= no signal)
    Image.fromarray(rgb, "RGB").save(path, optimize=True)


def _iso(t) -> str:
    return t.replace(tzinfo=timezone.utc).isoformat()


def write_site(results, cfg: dict, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("s[0-9]*_f[0-9]*.png")):   # drop layers of a previous cycle (window ids change)
        old.unlink()
    mg = MercatorGrid.build(results.grid, cfg.get("web", {}).get("px_m", 3000.0))
    wins = []
    for wid, w in results.windows.items():
        _write_layer(mg, w, out / f"{wid}.png", ENC_A, ENC_XMAX)
        entry = {"id": wid, "label": w["label"], "length_h": w["length_h"], "end_h": w["end_h"],
                 "file": f"{wid}.png", "start": _iso(w["start"]), "end": _iso(w["end"]), "ice_file": None}
        if wid in results.ice:
            _write_layer(mg, results.ice[wid], out / f"{wid}_ice.png", ICE_ENC_A, ICE_ENC_XMAX)
            entry["ice_file"] = f"{wid}_ice.png"
        wins.append(entry)
    groups: dict = {}
    for m in results.members:
        g = groups.setdefault(m["group"], {"group": m["group"], "members": 0, "inits": set()})
        g["members"] += int(m["ok"])
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
        "groups": [{"group": g["group"], "members": g["members"], "inits": sorted(g["inits"])}
                   for g in groups.values()],
        "mode_method": cfg.get("mode", {}).get("method", "pmm"),
        "slr": cfg.get("slr", {}),
        "grid_stride": results.grid.stride,
        "prism_downscale": bool(results.meta.get("prism_downscale", False)),
        "notice": "Unofficial guidance built from public WPC PWPF documentation. Not a National Weather Service product.",
    }
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
