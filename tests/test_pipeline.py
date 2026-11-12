import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pwpf_blender import binormal, export
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec
from pwpf_blender.member import BinContext, StepFields, process_bin
from pwpf_blender.pipeline import Pipeline, day_starts, expand_members, select_init
from pwpf_blender.regrid import Regridder
from pwpf_blender.sources.synthetic import synthetic_prism, synthetic_terrain

CFG = Path(__file__).resolve().parents[1] / "config" / "pwpf.yaml"


# ---------------------------------------------------------------- member-level physics
def _identity_setup(shape=(6, 8)):
    lon, lat = np.meshgrid(np.linspace(-100, -90, shape[1]), np.linspace(35, 45, shape[0]))
    rg = Regridder.build(lat[:, 0], lon[0, :], lat, lon)
    return rg, lat, lon


def _step(shape, t2m, t850, qpf=10.0, psfc=1000.0):
    z = {1000: 100.0, 925: 760.0, 850: 1460.0, 700: 3010.0, 500: 5570.0}
    tt = {1000: t2m + 1, 925: (t2m + t850) / 2, 850: t850, 700: t850 - 6, 500: -28}
    levels = {p: {"t": np.full(shape, tt[p], np.float32), "rh": np.full(shape, 97.0, np.float32),
                  "z": np.full(shape, z[p], np.float32)} for p in z}
    return StepFields(valid=datetime(2026, 1, 15, 18), qpf_mm=np.full(shape, qpf, np.float32),
                      t2m_c=np.full(shape, t2m, np.float32), psfc_hpa=np.full(shape, psfc, np.float32),
                      levels=levels, rh2m=np.full(shape, 96.0, np.float32), z_sfc_m=np.full(shape, 100.0, np.float32))


def _ctx(shape, terrain=0.0, slr=None):
    return BinContext(terrain_m=np.full(shape, terrain, np.float32), downscaler=None, doy=15,
                      slr_cfg=slr or {"method": "fixed", "fixed": 10.0})


def test_cold_column_snowfall_equals_qpf_times_slr():
    rg, *_ = _identity_setup()
    out = process_bin([_step(rg.src_shape, -6.0, -10.0, qpf=10.0)], rg, _ctx(rg.dst_shape))
    assert np.allclose(out, 10.0 / 25.4 * 10.0, rtol=1e-3)


def test_warm_column_gives_no_snow():
    rg, *_ = _identity_setup()
    out = process_bin([_step(rg.src_shape, 8.0, 4.0, qpf=10.0)], rg, _ctx(rg.dst_shape))
    assert out.max() == 0.0


def test_high_terrain_above_snow_level_gets_snow_even_if_model_says_rain():
    """Valley rain (warm surface) but ridge above the 0.5C wet-bulb level -> snow only on the ridge."""
    rg, *_ = _identity_setup()
    step = _step(rg.src_shape, 4.0, -3.0, qpf=10.0)           # model: rain at surface
    terrain = np.zeros(rg.dst_shape, np.float32)
    terrain[:, 4:] = 2500.0                                   # ridge well above the snow level
    ctx = BinContext(terrain_m=terrain, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0})
    out = process_bin([step], rg, ctx)
    assert out[:, :3].max() == 0.0 and out[:, 5:].min() > 0.0


def test_dominant_type_over_steps_and_qpf_additivity():
    rg, *_ = _identity_setup()
    steps = [_step(rg.src_shape, -6, -10, qpf=4.0), _step(rg.src_shape, -6, -10, qpf=6.0)]
    out = process_bin(steps, rg, _ctx(rg.dst_shape))
    assert np.allclose(out, 10 / 25.4 * 10, rtol=1e-3)


def test_prism_downscaling_redistributes_but_preserves_mean():
    from pwpf_blender.downscale import PrismDownscaler
    rg, lat, lon = _identity_setup((40, 50))
    clim = np.ones((12,) + rg.dst_shape, np.float32) * 50
    clim[:, :, 20:30] = 150
    ds = PrismDownscaler(clim, grid_dx_km=2.54, ratio_min=0.25, ratio_max=4)
    ctx = BinContext(terrain_m=np.zeros(rg.dst_shape, np.float32), downscaler=ds, doy=15,
                     slr_cfg={"method": "fixed", "fixed": 10.0})
    out = process_bin([_step(rg.src_shape, -6, -10, qpf=10.0)], rg, ctx, src_dx_km=25.0)
    assert out[:, 25].mean() > out[:, 5].mean() * 1.1          # wet strip gets more (kernel is 25 km wide)
    assert abs(out.mean() / (10 / 25.4 * 10) - 1) < 0.15


# ---------------------------------------------------------------- scheduling helpers
def test_select_init_and_day_starts():
    c = datetime(2026, 1, 15, 12)
    assert select_init(c, [0, 6, 12, 18], 6) == datetime(2026, 1, 15, 6)       # GEFS 06Z for the 12Z cycle
    assert select_init(c, [0, 12], 12) == datetime(2026, 1, 15, 0)             # ECMWF 00Z
    assert select_init(c, [0, 6, 12, 18], 0, lag_h=6) == datetime(2026, 1, 15, 6)
    assert day_starts(c)[0] == datetime(2026, 1, 15, 12)
    assert day_starts(datetime(2026, 1, 15, 0))[0] == datetime(2026, 1, 15, 12)


def test_default_member_list_matches_pwpf_documentation_counts():
    cfg = load_config(CFG)
    # build the list without touching the network: swap in dummy sources
    import pwpf_blender.pipeline as pl

    class Dummy:
        native_ptype = True

    orig = pl.make_source
    pl.make_source = lambda loader, **k: Dummy()
    try:
        runs = expand_members(cfg, datetime(2026, 1, 15, 12))
    finally:
        pl.make_source = orig
    by = {}
    for r in runs:
        by[r.group] = by.get(r.group, 0) + 1
    assert by["gefs"] == 10 and by["ifs_ens"] == 25 and by["hrrr"] == 2 and by["nam_nest"] == 2
    assert by["hiresw_arw"] == 4 and by["gefs_mean"] == 1 and by["gfs"] == 1 and by["ifs_hres"] == 1
    gefs_inits = {r.init for r in runs if r.group == "gefs"}
    assert gefs_inits == {datetime(2026, 1, 15, 6)}
    assert {r.init for r in runs if r.group == "hrrr"} == {datetime(2026, 1, 15, 12), datetime(2026, 1, 15, 6)}
    assert all(r.fallback is not None for r in runs if r.group == "hrrr")


# ---------------------------------------------------------------- end to end (synthetic)
@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    cfg = load_config(CFG, "synthetic", {"grid": {"stride": 8}, "workers": 2})
    grid = GridSpec.ndfd_2p5km(8)
    lon, lat = grid.lonlat()
    pipe = Pipeline(cfg, grid, synthetic_terrain(lon, lat), synthetic_prism(lon, lat))
    res = pipe.run(datetime(2026, 1, 15, 12))
    out = tmp_path_factory.mktemp("site") / "data"
    export.write_site(res, cfg, out)
    return cfg, res, out


def test_synthetic_run_structure_and_physics(synthetic_run):
    cfg, res, out = synthetic_run
    assert set(res.windows) == {"d1", "d2", "d3", "d12", "d23"}
    assert res.meta["n_members"] == 34
    w = res.windows["d1"]
    assert np.isfinite(w["mode"]).all() and (w["mode"] >= 0).all()
    assert (w["s_lo"] > 0).all() and (w["s_hi"] > 0).all()
    q10, q50, q90 = (binormal.percentile(w["mode"], w["s_lo"], w["s_hi"], q) for q in (0.1, 0.5, 0.9))
    assert (q10 <= q50 + 1e-6).all() and (q50 <= q90 + 1e-6).all()
    lon, lat = res.grid.lonlat()
    assert w["mode"][lat < 30].max() == 0.0                         # warm south: no snow
    assert 3 < w["mode"].max() < 60                                  # plausible amounts
    # exceedance decreases with threshold
    p1 = binormal.exceedance(w["mode"], w["s_lo"], w["s_hi"], 1.0)
    p6 = binormal.exceedance(w["mode"], w["s_lo"], w["s_hi"], 6.0)
    assert (p6 <= p1 + 1e-9).all()
    # 48-h mode is the sum of the day modes (as PWPF sums adjacent 24-h forecasts)
    assert np.allclose(res.windows["d12"]["mode"], res.windows["d1"]["mode"] + res.windows["d2"]["mode"], atol=1e-4)


def test_png_roundtrip_matches_fields(synthetic_run):
    cfg, res, out = synthetic_run
    man = json.loads((out / "manifest.json").read_text())
    assert man["members_total"] == 34 and len(man["windows"]) == 5
    img = np.array(Image.open(out / "d1.png"))
    assert img.shape[:2] == (man["size"][1], man["size"][0]) and img.shape[2] == 3
    mg = export.MercatorGrid.build(res.grid, cfg["web"]["px_m"])
    ref = mg.reproject(res.windows["d1"]["mode"])
    dec = export.decode(img[..., 0])
    sel = ref > 0.5
    rel = np.abs(dec[sel] - ref[sel]) / ref[sel]
    assert rel.max() < 0.03                                          # ~2 % encoding error
    assert dec[~mg.inside].max() == 0.0


def test_encode_decode_precision():
    x = np.array([0.0, 0.01, 0.1, 1, 5, 20, 80, 150])
    d = export.decode(export.encode(x))
    assert d[0] == 0
    assert np.all(np.abs(d[2:] - x[2:]) <= np.maximum(0.006, 0.025 * x[2:]))   # <=0.006 in or 2.5 %


def test_lite_profile_reduces_members_and_stride():
    cfg = load_config(CFG, "lite")
    names = {m["name"]: m for m in cfg["members"]}
    assert names["ifs_ens"]["count"] == 10 and "nam_nest" not in names and "hiresw_arw" not in names
    assert cfg["grid"]["stride"] == 2 and cfg["mode"]["method"] == "pmm"
