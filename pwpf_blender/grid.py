"""NDFD CONUS 2.5 km grid (spherical Lambert conformal) and helpers.

Grid definition from the NWS NDFD spatial reference documentation:
  2145 x 1377 points, dx = dy = 2539.703 m, tangent latitude 25N, orientation lon 95W,
  earth sphere R = 6371.2 km, first (south-west) point 20.192N, 121.554W.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EARTH_R = 6371200.0
NDFD_DX = 2539.703
NDFD_NX, NDFD_NY = 2145, 1377
NDFD_LA1, NDFD_LO1 = 20.192, -121.554


class LambertConformal:
    """Spherical Lambert conformal conic, one standard parallel (tangent cone)."""

    def __init__(self, lat_tangent: float = 25.0, lon0: float = -95.0, radius: float = EARTH_R):
        self.R = radius
        self.lon0 = math.radians(lon0)
        phi1 = math.radians(lat_tangent)
        self.n = math.sin(phi1)
        self.F = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** self.n / self.n
        self.rho0 = float(self._rho(phi1))

    def _rho(self, phi):
        return self.R * self.F / np.tan(np.pi / 4 + np.asarray(phi) / 2) ** self.n

    @staticmethod
    def _wrap(a):
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    def forward(self, lon, lat):
        lam = np.radians(lon)
        phi = np.radians(lat)
        rho = self._rho(phi)
        theta = self.n * self._wrap(lam - self.lon0)
        return rho * np.sin(theta), self.rho0 - rho * np.cos(theta)

    def inverse(self, x, y):
        dy = self.rho0 - np.asarray(y)
        x = np.asarray(x)
        rho = np.hypot(x, dy)
        theta = np.arctan2(x, dy)
        phi = 2 * np.arctan((self.R * self.F / rho) ** (1.0 / self.n)) - np.pi / 2
        lam = self.lon0 + theta / self.n
        return np.degrees(lam), np.degrees(phi)


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int
    dx: float
    x0: float          # projection x of grid point (0, 0) [m]
    y0: float
    stride: int = 1
    proj: LambertConformal = field(default_factory=LambertConformal, compare=False, repr=False)

    @classmethod
    def ndfd_2p5km(cls, stride: int = 1) -> "GridSpec":
        proj = LambertConformal()
        x0, y0 = proj.forward(NDFD_LO1, NDFD_LA1)
        return cls(
            nx=(NDFD_NX - 1) // stride + 1,
            ny=(NDFD_NY - 1) // stride + 1,
            dx=NDFD_DX * stride,
            x0=float(x0),
            y0=float(y0),
            stride=stride,
            proj=proj,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def dx_km(self) -> float:
        return self.dx / 1000.0

    def xy(self):
        xs = self.x0 + self.dx * np.arange(self.nx)
        ys = self.y0 + self.dx * np.arange(self.ny)
        return np.meshgrid(xs, ys)

    def lonlat(self):
        x, y = self.xy()
        lon, lat = self.proj.inverse(x, y)
        return lon.astype(np.float64), lat.astype(np.float64)

    def lonlat_to_rowcol(self, lon, lat):
        x, y = self.proj.forward(lon, lat)
        return (np.asarray(y) - self.y0) / self.dx, (np.asarray(x) - self.x0) / self.dx
