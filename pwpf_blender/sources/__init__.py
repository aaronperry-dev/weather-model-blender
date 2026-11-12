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
    from . import herbie_sources as hs
    table = {
        "gefs": hs.gefs, "gfs": hs.gfs, "hrrr": hs.hrrr, "nam_nest": hs.nam_nest,
        "hiresw_arw": hs.hiresw_arw, "hiresw_fv3": hs.hiresw_fv3,
        "ifs_ens": lambda: hs.IfsSource("ifs_ens", ensemble=True),
        "ifs_hres": lambda: hs.IfsSource("ifs_hres", ensemble=False),
    }
    if loader not in table:
        raise KeyError(f"unknown loader {loader!r}; available: {sorted(table)} + synthetic_global/synthetic_cam")
    return table[loader]()
