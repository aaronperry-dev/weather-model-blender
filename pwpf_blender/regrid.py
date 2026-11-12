"""Fast, reusable regridding from arbitrary source grids onto the 2.5 km target grid.

A Regridder stores, for every target point, k source indices and weights.  It is built once
per source grid and then applied to any number of fields.

* Regular lat/lon sources (GEFS, ECMWF, GFS ...)  -> exact bilinear interpolation (k = 4).
* Anything else (HRRR/NAM Lambert, rotated grids)  -> inverse-distance weighting over the 4 nearest
  source points found with a KD-tree on the unit sphere (close to bilinear on smooth grids).
Target points outside the source domain get NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _xyz(lat, lon):
    la = np.radians(lat)
    lo = np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


class Regridder:
    def __init__(self, idx: np.ndarray, w: np.ndarray, src_shape: tuple[int, int], dst_shape: tuple[int, int]):
        self.idx = idx.astype(np.int32, copy=False)
        self.w = w.astype(np.float32, copy=False)
        self.src_shape = tuple(src_shape)
        self.dst_shape = tuple(dst_shape)

    # ------------------------------------------------------------------ builders
    @classmethod
    def regular_latlon(cls, src_lat1d, src_lon1d, dst_lat, dst_lon) -> "Regridder":
        lat = np.asarray(src_lat1d, dtype=np.float64)
        lon = np.asarray(src_lon1d, dtype=np.float64)
        nlat, nlon = lat.size, lon.size
        if lat[0] > lat[-1]:
            fi = np.interp(dst_lat.ravel(), lat[::-1], np.arange(nlat)[::-1], left=-np.inf, right=np.inf)
        else:
            fi = np.interp(dst_lat.ravel(), lat, np.arange(nlat), left=-np.inf, right=np.inf)
        dlon = lon[1] - lon[0]
        is_global = abs(nlon * dlon) >= 360.0 - 1e-6
        fj = ((dst_lon.ravel() - lon[0]) % 360.0) / dlon
        outside = ~np.isfinite(fi)
        if not is_global:
            outside |= (fj > nlon - 1) | (fj < 0)
        i0 = np.clip(np.floor(np.where(outside, 0, fi)).astype(np.int64), 0, nlat - 2)
        fy = np.clip(np.where(outside, 0, fi) - i0, 0, 1)
        j0 = np.floor(np.where(outside, 0, fj)).astype(np.int64)
        fx = np.where(outside, 0, fj) - j0
        j0 = j0 % nlon if is_global else np.clip(j0, 0, nlon - 2)
        j1 = (j0 + 1) % nlon if is_global else np.clip(j0 + 1, 0, nlon - 1)
        i1 = i0 + 1
        idx = np.stack([i0 * nlon + j0, i0 * nlon + j1, i1 * nlon + j0, i1 * nlon + j1], axis=1)
        w = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=1)
        w[outside] = np.nan
        return cls(idx, w, (nlat, nlon), dst_lat.shape)

    @classmethod
    def from_points(cls, src_lat2d, src_lon2d, dst_lat, dst_lon, k: int = 4, max_gap: float = 3.0) -> "Regridder":
        pts = _xyz(src_lat2d.ravel(), src_lon2d.ravel())
        tree = cKDTree(pts)
        # typical nearest-neighbour spacing of the source grid (chord length on unit sphere)
        sample = pts[:: max(1, len(pts) // 5000)]
        spacing = np.median(tree.query(sample, k=2)[0][:, 1])
        d, idx = tree.query(_xyz(dst_lat.ravel(), dst_lon.ravel()), k=k)
        w = 1.0 / np.maximum(d, 1e-9) ** 2
        w /= w.sum(axis=1, keepdims=True)
        w[d[:, 0] > max_gap * spacing] = np.nan
        return cls(idx, w, src_lat2d.shape, dst_lat.shape)

    @classmethod
    def build(cls, src_lat, src_lon, dst_lat, dst_lon) -> "Regridder":
        src_lat = np.asarray(src_lat)
        src_lon = np.asarray(src_lon)
        if src_lat.ndim == 1 and src_lon.ndim == 1:
            return cls.regular_latlon(src_lat, src_lon, dst_lat, dst_lon)
        # 2-D lat/lon that is really a regular lat/lon grid?
        if src_lat.ndim == 2 and np.allclose(src_lat, src_lat[:, :1]) and np.allclose(src_lon, src_lon[:1, :]):
            return cls.regular_latlon(src_lat[:, 0], src_lon[0, :], dst_lat, dst_lon)
        return cls.from_points(src_lat, src_lon, dst_lat, dst_lon)

    # ------------------------------------------------------------------ apply
    def __call__(self, field: np.ndarray) -> np.ndarray:
        """field: (..., sy, sx) -> (..., dy, dx).  NaN where the target is outside the source."""
        field = np.asarray(field, dtype=np.float32)
        if field.shape[-2:] != self.src_shape:
            raise ValueError(f"field shape {field.shape[-2:]} != source grid {self.src_shape}")
        lead = field.shape[:-2]
        flat = field.reshape(*lead, -1)
        out = np.empty(lead + (self.idx.shape[0],), dtype=np.float32)
        it = np.ndindex(*lead) if lead else [()]
        for ix in it:
            vals = flat[ix][self.idx]                 # (P, k)
            out[ix] = np.einsum("pk,pk->p", vals, self.w, optimize=True)
        return out.reshape(*lead, *self.dst_shape)
