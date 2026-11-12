import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pwpf_blender import binormal, export
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec
from pwpf_blender.member import BinContext, StepFields, bin_products, process_bin
from pwpf_blender.pipeline import Pipeline, expand_members, select_init, window_specs
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
def test_select_init():
    c = datetime(2026, 1, 15, 12)
    assert select_init(c, [0, 6, 12, 18], 6) == datetime(2026, 1, 15, 6)       # GEFS 06Z for the 12Z cycle
    assert select_init(c, [0, 12], 12) == datetime(2026, 1, 15, 0)             # ECMWF 00Z
    assert select_init(c, [0, 6, 12, 18], 0, lag_h=6) == datetime(2026, 1, 15, 6)


def test_window_specs_are_rolling_six_hourly_periods():
    cfg = load_config(CFG)
    cyc = datetime(2026, 1, 15, 12)
    specs = window_specs(cfg, cyc)
    ids = [s.id for s in specs]
    assert ids[:9] == [f"s24_f{h:03d}" for h in range(24, 73, 6)]            # 24 h ending +24 ... +72, every 6 h
    assert [s.id for s in specs if s.length_h == 48] == [f"s48_f{h:03d}" for h in range(48, 73, 6)]
    assert [s.id for s in specs if s.length_h == 72] == ["s72_f072"]
    assert len(specs) == 15
    s30 = next(s for s in specs if s.id == "s24_f030")
    assert (s30.start, s30.end) == (datetime(2026, 1, 15, 18), datetime(2026, 1, 16, 18)) and s30.label == "Fri 18Z"
    s72 = specs[-1]
    assert (s72.start, s72.end) == (cyc, datetime(2026, 1, 18, 12))
    with pytest.raises(ValueError):
        window_specs({"windows": {"lengths_h": [36]}}, cyc)


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
    assert by["gefs"] == 10 and by["ifs_ens"] == 25 and by["hrrr"] == 2 and by["rrfs"] == 2
    assert by["rrfs_ens"] == 10 and by["gefs_mean"] == 1 and by["gfs"] == 1 and by["ifs_hres"] == 1
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
    assert set(res.windows) == {s.id for s in window_specs(cfg, res.cycle)} and len(res.windows) == 15
    assert res.meta["n_members"] == 34
    w = res.windows["s24_f024"]
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
    # 48-h mode is the sum of the two adjacent 24-h modes, 72 h of three (as PWPF sums adjacent 24-h forecasts)
    W = res.windows
    assert np.allclose(W["s48_f054"]["mode"], W["s24_f030"]["mode"] + W["s24_f054"]["mode"], atol=1e-4)
    assert np.allclose(W["s72_f072"]["mode"], W["s24_f024"]["mode"] + W["s24_f048"]["mode"] + W["s24_f072"]["mode"],
                       atol=1e-4)
    # rolling windows overlap: the 24 h total ending +30 h shares bins with the one ending +24 h
    assert not np.allclose(W["s24_f024"]["mode"], W["s24_f030"]["mode"])
    assert W["s24_f030"]["start"] == res.cycle + timedelta(hours=6)


def test_manifest_reports_prism_downscaling(synthetic_run, tmp_path):
    cfg, res, out = synthetic_run
    assert json.loads((out / "manifest.json").read_text())["prism_downscale"] is True
    # no PRISM file (e.g. the download failed in CI): pipeline still runs, and says so
    small = load_config(CFG, "synthetic", {"grid": {"stride": 16}, "workers": 2, "windows": {"lengths_h": [24], "max_h": 24}})
    grid = GridSpec.ndfd_2p5km(16)
    lon, lat = grid.lonlat()
    res2 = Pipeline(small, grid, synthetic_terrain(lon, lat), None).run(datetime(2026, 1, 15, 12))
    assert res2.meta["prism_downscale"] is False and list(res2.windows) == ["s24_f024"]
    export.write_site(res2, small, tmp_path / "d")
    assert json.loads((tmp_path / "d" / "manifest.json").read_text())["prism_downscale"] is False


def test_png_roundtrip_matches_fields(synthetic_run):
    cfg, res, out = synthetic_run
    man = json.loads((out / "manifest.json").read_text())
    assert man["members_total"] == 34 and len(man["windows"]) == 15
    w0 = man["windows"][0]
    assert (w0["id"], w0["length_h"], w0["end_h"]) == ("s24_f024", 24, 24)
    img = np.array(Image.open(out / "s24_f024.png"))
    assert img.shape[:2] == (man["size"][1], man["size"][0]) and img.shape[2] == 3
    mg = export.MercatorGrid.build(res.grid, cfg["web"]["px_m"])
    ref = mg.reproject(res.windows["s24_f024"]["mode"])
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
    assert names["ifs_ens"]["count"] == 10 and "rrfs_ens" not in names and "rrfs" in names
    assert cfg["grid"]["stride"] == 2 and cfg["mode"]["method"] == "pmm"


# ---------------------------------------------------------------- freezing rain (simplified FRAM)
def test_fram_precip_rate_ilr_matches_paper():
    from pwpf_blender import ice
    # Eq. 2 of Sanders & Barjenbruch (2016): 0.1395 P^-0.541 (P in/h); worked values
    assert ice.ilr_precip(0.1) == pytest.approx(0.1395 * 0.1 ** -0.541, rel=1e-4) == pytest.approx(0.485, abs=0.002)
    assert ice.ilr_precip(0.25) == pytest.approx(0.1395 * 0.25 ** -0.541, rel=1e-4)
    # monotone: heavier rain -> lower ratio; floored at P = 0.02 in/h and capped at 1
    r = ice.ilr_precip(np.array([0.001, 0.02, 0.05, 0.1, 0.2, 0.5]))
    assert r[0] == r[1] == 1.0 and (np.diff(r[1:]) < 0).all()
    # liquid 0.1 in over one hour -> flat ice 0.0485 in; over 6 h (rate 0.0167 -> floored) -> ILR 1
    assert ice.flat_ice_in(0.1, 1.0) == pytest.approx(0.0485, abs=0.001)
    assert ice.flat_ice_in(0.1, 6.0) == pytest.approx(0.1, abs=1e-6)


def _fzra_step(shape, qpf=5.0, dt=1.0):
    st = _step(shape, -1.5, 4.0, qpf=qpf)                      # warm layer aloft, sub-freezing surface
    st.dt_h = dt
    return st


def test_freezing_rain_column_gives_ice_not_snow():
    rg, *_ = _identity_setup()
    ctx = _ctx(rg.dst_shape)
    ctx.ice_cfg = {"enabled": True}
    snow, ice = bin_products([_fzra_step(rg.src_shape, qpf=5.0, dt=1.0)], rg, ctx)
    assert snow.max() == 0.0
    assert ice is not None and np.allclose(ice, 5.0 / 25.4 * min(0.1395 * (5.0 / 25.4) ** -0.541, 1.0), rtol=1e-3)
    # radial reporting
    ctx.ice_cfg = {"enabled": True, "report": "radial"}
    _, ice_r = bin_products([_fzra_step(rg.src_shape, qpf=5.0, dt=1.0)], rg, ctx)
    assert np.allclose(ice_r, ice * 0.394, rtol=1e-4)
    # disabled -> no ice layer
    ctx.ice_cfg = {"enabled": False}
    assert bin_products([_fzra_step(rg.src_shape)], rg, ctx)[1] is None
    # snow and warm-rain columns produce no ice
    ctx.ice_cfg = {"enabled": True}
    assert bin_products([_step(rg.src_shape, -6, -10)], rg, ctx)[1] is None
    assert bin_products([_step(rg.src_shape, 8, 4)], rg, ctx)[1] is None


def test_ice_rate_uses_step_length():
    """The same liquid depth falling in a shorter step is a higher rate, hence a lower ice-to-liquid ratio."""
    rg, *_ = _identity_setup()
    ctx = _ctx(rg.dst_shape)
    ctx.ice_cfg = {"enabled": True}
    fast = bin_products([_fzra_step(rg.src_shape, qpf=5.0, dt=1.0)], rg, ctx)[1]
    slow = bin_products([_fzra_step(rg.src_shape, qpf=5.0, dt=6.0)], rg, ctx)[1]
    assert slow.mean() > fast.mean()


def test_synthetic_run_has_ice_layers(synthetic_run):
    cfg, res, out = synthetic_run
    assert res.ice, "synthetic warm nose should yield freezing rain in at least one window"
    w = res.ice["s24_f024"] if "s24_f024" in res.ice else next(iter(res.ice.values()))
    assert (w["mode"] >= 0).all() and (w["s_lo"] > 0).all() and w["mode"].max() < 2.0
    p01 = binormal.exceedance(w["mode"], w["s_lo"], w["s_hi"], 0.01)
    p25 = binormal.exceedance(w["mode"], w["s_lo"], w["s_hi"], 0.25)
    assert (p25 <= p01 + 1e-9).all() and p01.max() > 0.3
    man = json.loads((out / "manifest.json").read_text())
    with_ice = [x for x in man["windows"] if x["ice_file"]]
    assert {x["id"] for x in with_ice} == set(res.ice)
    assert man["products"]["ice"]["encoding"] == {"A": export.ICE_ENC_A, "xmax": export.ICE_ENC_XMAX}
    img = np.array(Image.open(out / with_ice[0]["ice_file"]))
    wid = with_ice[0]["id"]
    mg = export.MercatorGrid.build(res.grid, cfg["web"]["px_m"])
    ref = mg.reproject(res.ice[wid]["mode"])
    dec = export.decode(img[..., 0], export.ICE_ENC_A, export.ICE_ENC_XMAX)
    sel = ref > 0.01
    assert sel.any() and (np.abs(dec[sel] - ref[sel]) / ref[sel]).max() < 0.03


def test_ice_48h_mode_is_sum_of_24h_ice_modes(synthetic_run):
    cfg, res, _ = synthetic_run
    I = res.ice
    if "s48_f048" in I and "s24_f024" in I and "s24_f048" in I:
        assert np.allclose(I["s48_f048"]["mode"], I["s24_f024"]["mode"] + I["s24_f048"]["mode"], atol=1e-4)
