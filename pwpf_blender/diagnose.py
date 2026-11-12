"""`python -m pwpf_blender diagnose` - check every live data feed and print exactly what the servers return.

For each model it finds the newest cycle whose forecast-hour-3 inventory is published, prints which URL answered
(AWS Open Data or NOMADS), how many inventory records match the fields the blender needs, and which required
fields are missing.  With --download it also reads one real forecast step through the same code path the
pipeline uses (byte-range download + eccodes decode) and prints the grid and value ranges.  Intended to be run
in CI before the pipeline so a failing feed is visible in the log with the real reason.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from . import noaa_http as nh
from .sources import noaa_sources as ns

# (name, factory, plan fn, member, run hours, forecast hour to probe)
MODELS = [
    ("gefs", ns.gefs, ns.plan_gefs, 1, (0, 6, 12, 18), 6),
    ("cmce", ns.cmce, ns.plan_cmce, 1, (0, 12), 6),
    ("gfs", ns.gfs, ns.plan_gfs, None, (0, 6, 12, 18), 6),
    ("hrrr", ns.hrrr, ns.plan_hrrr, None, tuple(range(24)), 6),
    ("rrfs", ns.rrfs, ns.plan_rrfs, None, (0, 6, 12, 18), 6),
    ("rrfs_ens", ns.rrfs_ens, ns.plan_rrfs_ens, 1, (0, 6, 12, 18), 6),
]
MSC_MODELS = [("rdps", (0, 6, 12, 18)), ("gdps", (0, 12))]           # MSC Datamart deterministic models (no .idx files)
REQUIRED = [r"^TMP:2 m above ground", r"^PRES:surface", r"^APCP:surface:\d+-\d+ hour acc", r"^TMP:850 mb",
            r"^RH:850 mb", r"^HGT:850 mb"]
OPTIONAL = [r"^CSNOW:surface", r"^RH:2 m above ground", r"^HGT:surface"]


def _latest_init(plan_fn, member, run_hours, fxx, now, max_back_h=None):
    max_back_h = max_back_h or (30 if len(run_hours) <= 4 else 8)
    for back in range(0, max_back_h + 1):
        init = (now - timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        if init.hour not in run_hours:
            continue
        try:
            plans = plan_fn(init, fxx, member)
            for p in plans:
                nh.get_inventory_first(p.urls)
            return init, plans
        except nh.FetchError:
            continue
    return None, None


def check_model(name, plan_fn, member, run_hours, fxx, now) -> bool:
    init, plans = _latest_init(plan_fn, member, run_hours, fxx, now)
    if init is None:
        print(f"[FAIL] {name}: no cycle with a published f{fxx:03d} inventory in the lookback window")
        # show why, for the most recent candidate
        init = next((now - timedelta(hours=b) for b in range(0, 24) if (now - timedelta(hours=b)).hour in run_hours), now)
        for p in plan_fn(init.replace(minute=0, second=0, microsecond=0), fxx, member):
            for u in p.urls:
                try:
                    nh.get_inventory(u)
                    print(f"        {u}.idx -> OK")
                except nh.FetchError as e:
                    print(f"        {e}")
        return False
    ok = True
    print(f"[ .. ] {name}: newest cycle {init:%Y-%m-%d %HZ}")
    all_text = []
    for p in plans:
        recs, url = nh.get_inventory_first(p.urls)
        sel = nh.select_records(recs, p.patterns)
        all_text += [r.text for r in sel]
        print(f"        {url}\n        {len(recs)} idx records, {len(sel)} selected")
    for pat in REQUIRED:
        hit = [t for t in all_text if re.search(pat, t)]
        if not hit:
            ok = False
        print(f"        {'ok     ' if hit else 'MISSING'} {pat}  {hit[:3] if hit else ''}")
    for pat in OPTIONAL:
        hit = [t for t in all_text if re.search(pat, t)]
        print(f"        {'ok     ' if hit else 'absent '} (optional) {pat}")
    print(f"[{'ok' if ok else 'FAIL':^5}] {name}")
    return ok


def download_probe(name, factory, init: datetime, member) -> bool:
    src = factory()
    try:
        st = src.load_steps(init, init + timedelta(hours=3), init + timedelta(hours=6), member)[0]
        g = src.grid()
        q = st.qpf_mm
        print(f"[ ok ] {name} read f006 {init:%d %HZ}: grid {g.shape}, qpf mm [{np.nanmin(q):.2f}, {np.nanmax(q):.2f}], "
              f"t2m C [{np.nanmin(st.t2m_c):.1f}, {np.nanmax(st.t2m_c):.1f}], levels {sorted(st.levels)}, "
              f"native ptype {sorted(st.native_type_frac) if st.native_type_frac else None}, z_sfc {st.z_sfc_m is not None}")
        return True
    except Exception as e:                                       # noqa: BLE001
        print(f"[FAIL] {name} download probe: {type(e).__name__}: {e}")
        return False


def check_msc(key: str, run_hours, now, download: bool) -> bool:
    """MSC Datamart has no inventories: probe a ranged GET of the 2 m temperature file, then (optionally) read a step."""
    from .sources import make_source
    src = make_source(key)
    for back in range(0, 31):
        init = (now - timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        if init.hour in run_hours and src.available(init, 6, None):
            break
    else:
        print(f"[FAIL] {key}: no cycle with a published f006 file in the lookback window "
              f"(e.g. {src.url(now.replace(minute=0, second=0, microsecond=0), 6, 'AirTemp', 'AGL-2m')})")
        return False
    print(f"[ .. ] {key}: newest cycle {init:%Y-%m-%d %HZ}\n        {src.url(init, 6, 'AirTemp', 'AGL-2m')}")
    if not download:
        print(f"[ ok ] {key}")
        return True
    try:
        st = src.load_steps(init, init + timedelta(hours=3), init + timedelta(hours=6), None)[0]
        g = src.grid()
        zr = {p: (round(float(np.nanmin(v['z']))), round(float(np.nanmax(v['z'])))) for p, v in sorted(st.levels.items())}
        print(f"[ ok ] {key} read f006 {init:%d %HZ}: grid {g.shape}, qpf mm [{np.nanmin(st.qpf_mm):.2f}, {np.nanmax(st.qpf_mm):.2f}], "
              f"t2m C [{np.nanmin(st.t2m_c):.1f}, {np.nanmax(st.t2m_c):.1f}], psfc hPa [{np.nanmin(st.psfc_hpa):.0f}, "
              f"{np.nanmax(st.psfc_hpa):.0f}], rh2m {st.rh2m is not None}\n        height range per level (m): {zr}")
        return True
    except Exception as e:                                       # noqa: BLE001
        print(f"[FAIL] {key} download probe: {type(e).__name__}: {e}")
        return False


def run(download: bool = False, only: list[str] | None = None) -> int:
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    nh.RETRIES, nh.TIMEOUT = 1, (8, 60)                          # a diagnostic should fail fast
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    print(f"diagnose @ {now:%Y-%m-%d %H:%MZ}")
    bad = 0
    for name, factory, plan_fn, member, run_hours, fxx in MODELS:
        if only and name not in only:
            continue
        good = check_model(name, plan_fn, member, run_hours, fxx, now)
        if good and download:
            init, _ = _latest_init(plan_fn, member, run_hours, fxx, now)
            good = download_probe(name, factory, init, member)
        bad += not good
    for key, run_hours in MSC_MODELS:
        if only and key not in only:
            continue
        bad += not check_msc(key, run_hours, now, download)
    if not only or "terrain" in only:
        from .grid import GridSpec
        from .static_data import terrain_auto
        try:
            t = terrain_auto(GridSpec.ndfd_2p5km(1))
            print(f"[ ok ] terrain {t.shape} min {t.min():.0f} max {t.max():.0f} m")
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print(f"[FAIL] terrain: {type(e).__name__}: {e}")
    print(f"{bad} feed(s) failing")
    print("request budget used:\n  " + nh.STATS.report().replace("\n", "\n  "))
    return 1 if bad else 0
