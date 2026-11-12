"""Optional external deterministic forecast to use as the distribution mode (mode.method: external).

The PWPF mode is the WPC Winter Weather Desk forecast, which WPC collaborates with the local NWS offices.
The closest public gridded proxy is the NDFD snow-amount forecast on the very same 2.5 km grid.  This
loader sums its 6-h snow amounts into each 24 h window (NDFD periods are 6-hourly, so they line up).

EXPERIMENTAL: written from the NDFD GRIB2 documentation, not exercised against a live file.  Requires
xarray + cfgrib.  Units are converted from metres (GRIB2 'Total snowfall') to inches.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

M_TO_IN = 1.0 / 0.0254


def load_ndfd_windows(path: str, windows, stride: int = 1) -> dict:
    """Sum the 6-h NDFD snow amounts into each 24 h window.  `windows` = iterable of objects with `.id`,
    `.start`, `.end` (pipeline.WindowSpec).  Returns {window id: (ny, nx) inches or None}."""
    import cfgrib
    pieces = []
    for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
        for name, da in ds.data_vars.items():
            blob = f"{name} {da.attrs.get('GRIB_name', '')}".lower()
            if "snow" not in blob:
                continue
            vt = np.atleast_1d(da["valid_time"].values)
            arr = da.values if da.ndim == 3 else da.values[None]
            for k, t in enumerate(vt):
                end = datetime.utcfromtimestamp(int(t.astype("datetime64[s]").astype(int)))
                pieces.append((end, np.nan_to_num(arr[k].astype(np.float32), nan=0.0) * M_TO_IN))
    out = {}
    for w in windows:
        tot = None
        for end, a in pieces:
            if w.start < end <= w.end:
                tot = a if tot is None else tot + a
        out[w.id] = None if tot is None else tot[::stride, ::stride]
    return out
