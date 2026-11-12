"""Quick matplotlib preview of a fields_*.npz (mode, P(>=4in), 10th/90th percentile) - for sanity checks."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pwpf_blender import binormal
from pwpf_blender.grid import GridSpec

npz, out = sys.argv[1], sys.argv[2]
win = sys.argv[3] if len(sys.argv) > 3 else "s24_f024"
z = np.load(npz)
meta = json.loads(str(z["meta"]))
g = GridSpec.ndfd_2p5km(meta["stride"])
m, lo, hi = z[f"{win}_mode"], z[f"{win}_s_lo"], z[f"{win}_s_hi"]
lon, lat = g.lonlat()
fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
levels = [0.1, 1, 2, 4, 6, 8, 12, 18, 24, 36]
panels = [("mode (forecast), in", m, levels), ("P(>=4 in), %", 100 * binormal.exceedance(m, lo, hi, 4.0), [10, 20, 30, 40, 50, 60, 70, 80, 90]),
          ("P(>=12 in), %", 100 * binormal.exceedance(m, lo, hi, 12.0), [10, 20, 30, 40, 50, 60, 70, 80, 90]),
          ("10th pct, in", binormal.percentile(m, lo, hi, 0.10), levels), ("90th pct, in", binormal.percentile(m, lo, hi, 0.90), levels),
          ("ensemble std, in", z[f"{win}_std"], [0.1, 0.5, 1, 2, 3, 4, 6, 8])]
for a, (t, f, lv) in zip(ax.ravel(), panels):
    c = a.contourf(lon, lat, f, levels=lv, cmap="turbo", extend="max")
    a.set_title(f"{win}: {t}"); a.set_xlim(-125, -66); a.set_ylim(24, 52)
    fig.colorbar(c, ax=a, shrink=0.8)
fig.tight_layout(); fig.savefig(out, dpi=70)
print("saved", out)
