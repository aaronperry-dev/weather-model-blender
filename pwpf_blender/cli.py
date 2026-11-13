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
    now = datetime.now(timezone.utc).replace(tzinfo=None)      # newest cycle that has started; models fall back per member
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
    if a.accumulation:
        overrides["accumulation"] = {"enabled": a.accumulation == "on"}
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
    if a.archive_dir:                                  # raw fields first: the archive must never contain calibrated products
        from . import verify
        verify.archive_results(res, cfg, a.archive_dir)
        vc = verify.settings(cfg.get("verification"))
        if vc["apply_calibration"]:
            from . import recalibrate
            logging.info("%s", recalibrate.apply(res, Path(a.archive_dir) / "calibration_fit.json", verify.fingerprint(cfg)[0]))
    out = Path(a.out)
    mf = write_site(res, cfg, out)
    if a.save_npz:
        save_npz(res, out.parent / f"fields_{cycle:%Y%m%d%H}.npz")
    print(f"wrote {mf}  ({res.meta['n_members']} members)")


def cmd_nohrsc(a):
    from . import nohrsc
    valid = datetime.strptime(a.valid, "%Y%m%d%H")
    p = nohrsc.fetch(valid, a.hours, a.cache)
    if p is None:
        sys.exit(f"NOHRSC has not published the {a.hours} h analysis ending {valid:%Y-%m-%d %H}Z")
    g = nohrsc.read_grib(p)
    v = g["inches"][np.isfinite(g["inches"])]
    print(f"{p}: valid {g['valid']:%Y-%m-%d %H}Z, {g['hours']} h, {v.size} analysed cells, mean {v.mean():.3f} in, max {v.max():.1f} in")


def cmd_verify(a):
    from . import verify
    print(verify.case_report(a.npz, a.window, a.cache, towns=not a.no_towns))


def cmd_verify_archive(a):
    from . import recalibrate, verify
    cfg = load_config(a.config, a.profile)
    vc = verify.settings(cfg.get("verification"))
    if not vc["enabled"]:
        print("verification disabled in the config")
        return
    arch = Path(a.archive)
    now = datetime.fromisoformat(a.now).replace(tzinfo=None) if a.now else None
    led = verify.verify_archive(cfg, arch, a.cache, now=now)
    js, txt = verify.summarize(led, now)
    fp, fd = verify.fingerprint(cfg)
    fitr = recalibrate.fit(arch, fp, min_cycles=int(vc["min_cases"]))
    recalibrate.write_fit(fitr, arch / "calibration_fit.json")
    js["calibration_fit"] = fitr
    txt += "\n" + recalibrate.describe(fitr) + "\n" + (
        "(applied to the products)" if vc["apply_calibration"] else "(advisory: set verification.apply_calibration: true to apply it)") + "\n"
    print(txt)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "verification.txt").write_text(txt)
        (out / "verification.json").write_text(json.dumps(js, indent=1, default=float))
        mf = out / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            m["verification"] = "verification.txt"
            mf.write_text(json.dumps(m, indent=1))


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
    r.add_argument("--accumulation", choices=["on", "off"], default=None, help="override accumulation.enabled (A/B runs)")
    r.add_argument("--archive-dir", default=None, help="store the raw 24 h fields here for NOHRSC verification/calibration")
    r.set_defaults(fn=cmd_run)
    n = sub.add_parser("nohrsc", help="download + decode one NOHRSC snowfall analysis GRIB2 and print its summary")
    n.add_argument("--valid", required=True, help="end of the period, UTC, yyyymmddhh")
    n.add_argument("--hours", type=int, default=24)
    n.add_argument("--cache", default="nohrsc_cache")
    n.set_defaults(fn=cmd_nohrsc)
    vv = sub.add_parser("verify", help="score a fields_*.npz against NOHRSC (24 h periods that have ended)")
    vv.add_argument("--npz", required=True)
    vv.add_argument("--window", nargs="*", default=None, help="period ids (default: all 24 h periods)")
    vv.add_argument("--cache", default="nohrsc_cache")
    vv.add_argument("--no-towns", action="store_true")
    vv.set_defaults(fn=cmd_verify)
    va = sub.add_parser("verify-archive", help="score archived cycles against NOHRSC, update the ledger, fit the calibration")
    va.add_argument("--config", default="config/pwpf.yaml")
    va.add_argument("--profile", default=None)
    va.add_argument("--archive", default="archive")
    va.add_argument("--cache", default="nohrsc_cache")
    va.add_argument("--out", default=None, help="site data dir to receive verification.txt/.json")
    va.add_argument("--now", default=None, help="ISO UTC time to act as 'now' (back-fill / testing)")
    va.set_defaults(fn=cmd_verify_archive)
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
