"""Command line interface:  python -m pwpf_blender <command>"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import binormal
from .config import load_config
from .export import save_npz, write_site
from .grid import GridSpec
from .pipeline import Pipeline, window_specs


def _parse_cycle(s: str, cfg: dict) -> datetime:
    if s != "auto":
        return datetime.fromisoformat(s).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=3)
    for back in range(0, 48):
        c = (now - timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        if c.hour in cfg.get("cycle_hours", [0, 12]):
            return c
    raise RuntimeError("could not pick a cycle")


def _load_static(cfg: dict, grid: GridSpec, synthetic: bool):
    if synthetic:
        from .sources.synthetic import synthetic_prism, synthetic_terrain
        lon, lat = grid.lonlat()
        return synthetic_terrain(lon, lat), synthetic_prism(lon, lat)
    s = grid.stride
    st = cfg["static"]
    if not Path(st["terrain"]).exists():
        sys.exit(f"missing {st['terrain']} - run:  python -m pwpf_blender prepare-static")
    terrain = np.load(st["terrain"])[::s, ::s]
    prism = None
    if Path(st["prism"]).exists():
        prism = np.load(st["prism"])[:, ::s, ::s]
    else:
        logging.warning("no PRISM file (%s): QPF downscaling disabled", st["prism"])
    return terrain, prism


def cmd_run(a):
    overrides = {}
    if a.stride:
        overrides["grid"] = {"stride": a.stride}
    if a.workers:
        overrides["workers"] = a.workers
    if a.slr_adjust is not None:
        overrides["slr"] = {"adjust": a.slr_adjust}
    cfg = load_config(a.config, "synthetic" if a.synthetic else a.profile, overrides)
    grid = GridSpec.ndfd_2p5km(cfg["grid"]["stride"])
    terrain, prism = _load_static(cfg, grid, a.synthetic)
    cycle = _parse_cycle(a.cycle, cfg)
    ext = None
    if a.external_snow:
        from .external import load_ndfd_windows
        cfg.setdefault("mode", {})["method"] = "external"
        ext = load_ndfd_windows(a.external_snow, [w for w in window_specs(cfg, cycle) if w.length_h == 24],
                                grid.stride)
    logging.info("grid %dx%d (stride %d), cycle %s", grid.nx, grid.ny, grid.stride, cycle)
    pipe = Pipeline(cfg, grid, terrain, prism, ext)
    res = pipe.run(cycle)
    out = Path(a.out)
    mf = write_site(res, cfg, out)
    if a.save_npz:
        save_npz(res, out.parent / f"fields_{cycle:%Y%m%d%H}.npz")
    print(f"wrote {mf}  ({res.meta['n_members']} members)")


def cmd_prepare(a):
    from .static_data import PRISM_URL, prepare_static
    t, p = prepare_static(a.out, a.terrain_npy, a.prism_dir, a.prism_url or PRISM_URL, skip_existing=not a.force,
                          require_prism=a.require_prism)
    print("static ready:", t, p if p else "(no PRISM: QPF downscaling will be off)")


def cmd_diagnose(a):
    from .diagnose import run
    sys.exit(run(download=a.download, only=a.only))


def cmd_validate(a):
    from . import validate
    cycle = datetime.fromisoformat(a.cycle).replace(tzinfo=None)
    z = np.load(a.ours, allow_pickle=False)
    wins = [w for w in json.loads(str(z["meta"]))["windows"] if w.startswith("s24_")]
    for wid in sorted(wins):
        end_fhr = int(wid.split("_f")[1])
        print(json.dumps(validate.compare(a.ours, a.wpc_dir, wid, cycle, end_fhr), indent=1))
    if a.calibrate_slr:
        wid = "s24_f024"
        v = validate.calibrate_slr(a.ours, a.wpc_dir, wid, cycle, 24, a.current_adjust)
        print("suggested slr.adjust:", v)


def cmd_point(a):
    z = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    grid = GridSpec.ndfd_2p5km(meta["stride"])
    r, c = grid.lonlat_to_rowcol(a.lon, a.lat)
    i, j = int(round(float(r))), int(round(float(c)))
    if not (0 <= i < grid.ny and 0 <= j < grid.nx):
        sys.exit("point is outside the NDFD CONUS grid")
    for wid in meta["windows"]:
        m, lo, hi = (float(z[f"{wid}_{k}"][i, j]) for k in ("mode", "s_lo", "s_hi"))
        pc = {q: round(float(binormal.percentile(m, lo, hi, q / 100)), 1) for q in (5, 10, 25, 50, 75, 90, 95)}
        pr = {t: round(float(binormal.exceedance(m, lo, hi, t)) * 100) for t in (1, 2, 4, 6, 8, 12, 18)}
        print(f"{wid}: mode {m:.1f} in | percentiles {pc} | P(>=in) % {pr}")
        if wid in meta.get("ice_windows", []):
            m, lo, hi = (float(z[f"{wid}_ice_{k}"][i, j]) for k in ("mode", "s_lo", "s_hi"))
            pc = {q: round(float(binormal.percentile(m, lo, hi, q / 100)), 2) for q in (5, 25, 50, 75, 95)}
            pr = {t: round(float(binormal.exceedance(m, lo, hi, t)) * 100) for t in (0.01, 0.1, 0.25)}
            print(f"{wid} ice: mode {m:.2f} in | percentiles {pc} | P(>=in) % {pr}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pwpf_blender", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline and write the static site data")
    r.add_argument("--config", default="config/pwpf.yaml")
    r.add_argument("--profile", default=None)
    r.add_argument("--cycle", default="auto", help="ISO time (UTC) of the 00/12Z cycle, or 'auto'")
    r.add_argument("--out", default="site/data")
    r.add_argument("--stride", type=int, default=None)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--synthetic", action="store_true", help="no-network demo with synthetic ensemble")
    r.add_argument("--save-npz", action="store_true", help="also write full-precision fields next to --out")
    r.add_argument("--external-snow", default=None, help="NDFD snow GRIB2 to use as the mode")
    r.add_argument("--slr-adjust", type=float, default=None)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("prepare-static", help="fetch terrain + PRISM normals onto the 2.5 km grid")
    s.add_argument("--out", default="static")
    s.add_argument("--terrain-npy", default=None)
    s.add_argument("--prism-dir", default=None)
    s.add_argument("--prism-url", default=None, help="normals directory listing URL (or a template with {mm})")
    s.add_argument("--require-prism", action="store_true", help="fail instead of continuing without PRISM")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_prepare)
    d = sub.add_parser("diagnose", help="check every live data feed (AWS/NOMADS inventories) and print the result")
    d.add_argument("--download", action="store_true", help="also read one real forecast step per model")
    d.add_argument("--only", nargs="*", default=None, help="models to check (gefs gfs hrrr rrfs rrfs_ens terrain)")
    d.set_defaults(fn=cmd_diagnose)
    v = sub.add_parser("validate", help="compare with WPC's published PWPF GRIB2 files")
    v.add_argument("--ours", required=True, help="fields_*.npz from `run --save-npz`")
    v.add_argument("--wpc-dir", required=True)
    v.add_argument("--cycle", required=True)
    v.add_argument("--calibrate-slr", action="store_true")
    v.add_argument("--current-adjust", type=float, default=1.0)
    v.set_defaults(fn=cmd_validate)
    q = sub.add_parser("point", help="print percentiles/probabilities at a lat/lon from an npz")
    q.add_argument("--npz", required=True)
    q.add_argument("--lat", type=float, required=True)
    q.add_argument("--lon", type=float, required=True)
    q.set_defaults(fn=cmd_point)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
