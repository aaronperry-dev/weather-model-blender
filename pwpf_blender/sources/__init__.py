"""Loader registry: config `loader:` name -> factory returning a Source."""
from __future__ import annotations

from .base import Source, SourceGrid          # noqa: F401


def make_source(loader: str, **kw):
    if loader == "synthetic_global":
        from .synthetic import SyntheticSource
        return SyntheticSource("global", **kw)
    if loader == "synthetic_cam":
        from .synthetic import SyntheticSource
        return SyntheticSource("cam", **kw)
    from . import noaa_sources as hs
    table = {
        "gefs": hs.gefs, "gfs": hs.gfs, "hrrr": hs.hrrr, "rrfs": hs.rrfs, "rrfs_ens": hs.rrfs_ens,
        "ifs_ens": lambda: hs.IfsSource("ifs_ens", ensemble=True),
        "ifs_hres": lambda: hs.IfsSource("ifs_hres", ensemble=False),
    }
    if loader not in table:
        raise KeyError(f"unknown loader {loader!r}; available: {sorted(table)} + synthetic_global/synthetic_cam")
    return table[loader]()
