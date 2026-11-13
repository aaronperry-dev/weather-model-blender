"""Warm-ground / marginal-air accumulation (pwpf_blender/accumulation.py): our own extra, not WPC's recipe."""
from datetime import datetime

import numpy as np
import pytest

from pwpf_blender import accumulation as A
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec
from pwpf_blender.member import BinContext, process_bin
from pwpf_blender.pipeline import Pipeline
from pwpf_blender.sources.synthetic import synthetic_prism, synthetic_terrain
from test_pipeline import CFG, _identity_setup, _step

C = A.DEFAULTS


def test_soil_temperature_seasons_latitude_and_elevation():
    nov, dec, jan, jul = (float(A.soil_temp_c(d, 43.0, 0)) for d in (316, 349, 15, 200))
    assert 5.5 < nov < 8.5 and 0.5 < dec < 3 and jan == 0.0 and jul > 15            # warm in Nov, ~frozen by January
    assert float(A.soil_temp_c(316, 35.0, 0)) > nov > float(A.soil_temp_c(316, 48.0, 0))   # warmer to the south
    assert float(A.soil_temp_c(316, 43.0, 800)) == pytest.approx(nov - 5.2, abs=0.01)      # 6.5 C/km
    assert float(A.soil_temp_c(316, 43.0, 3000)) == 0.0                                    # never below 0 (clipped)


def test_temperature_efficiency_curve():
    sl = np.array([-500, -250, -100, 0, 100, 250, 4000], np.float32)                 # snow level minus terrain (terrain = 0)
    e = A.temp_efficiency(sl, np.zeros_like(sl), C)
    assert e[0] == 1.0 and e[1] == 1.0                                              # terrain well above the snow level: full
    assert np.all(np.diff(e) <= 1e-6) and e[-1] == pytest.approx(C["min_eff"])      # monotone, floor when the air is warm
    assert float(A.temp_efficiency(np.float32(0.0), np.float32(0.0), C)) == pytest.approx(0.55, abs=0.01)   # wet-bulb 0.5 C at the ground


def test_ground_melt_takes_the_first_inches_and_sums_telescope():
    ny, nx, n = 100, 6, 2                                                           # more than one 48-row chunk
    inc = [np.zeros((n, ny, nx), np.float16) for _ in range(4)]
    inc[0][0], inc[1][0], inc[2][0] = 0.3, 0.4, 0.8                                 # member 0: cumulative 0.3, 0.7, 1.5
    inc[0][1] = 0.5                                                                 # member 1: one burst of 0.5
    d0 = np.full((ny, nx), 0.6, np.float32)
    A.apply_ground_melt(inc, d0)
    m0 = [float(b[0, 50, 3]) for b in inc]
    assert m0[0] == 0.0 and m0[1] == pytest.approx(0.1, abs=2e-3) and m0[2] == pytest.approx(0.8, abs=2e-3)   # light snow vanishes, then it accumulates
    assert sum(m0) == pytest.approx(max(1.5 - 0.6, 0), abs=5e-3)                                            # total = cumulative - d0
    assert sum(float(b[1, 99, 0]) for b in inc) == 0.0                                                      # 0.5 in on 0.6 in of capacity: nothing left
    assert inc[0].dtype == np.float16 and all(float(b.min()) >= 0 for b in inc)


def test_marginal_air_reduces_snow_but_cold_high_ground_is_untouched():
    rg, *_ = _identity_setup()
    step = _step(rg.src_shape, 1.2, -3.0, qpf=10.0)                                 # decision-tree snow, but the 0.5 C wet-bulb level is above the ground
    terrain = np.zeros(rg.dst_shape, np.float32)
    off = process_bin([step], rg, BinContext(terrain_m=terrain, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0}))
    on = process_bin([step], rg, BinContext(terrain_m=terrain, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0},
                                            accum_cfg=dict(C)))
    assert on.max() < off.max() and on.min() > 0.2 * off.min()                       # reduced, not removed
    high = np.full(rg.dst_shape, 3000.0, np.float32)                                 # far above the snow level
    off2 = process_bin([step], rg, BinContext(terrain_m=high, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0}))
    on2 = process_bin([step], rg, BinContext(terrain_m=high, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0},
                                             accum_cfg=dict(C)))
    assert np.allclose(on2, off2)


def test_pipeline_switch_and_manifest(tmp_path):
    cfg = load_config(CFG, "synthetic", {"grid": {"stride": 8}, "workers": 2, "member_failure_limit": 2})
    grid = GridSpec.ndfd_2p5km(8)
    lon, lat = grid.lonlat()
    terr, prism = synthetic_terrain(lon, lat), synthetic_prism(lon, lat)
    on = Pipeline(cfg, grid, terr, prism).run(datetime(2026, 11, 12, 12))
    cfg_off = dict(cfg, accumulation=dict(cfg["accumulation"], enabled=False))
    off = Pipeline(cfg_off, grid, terr, prism).run(datetime(2026, 11, 12, 12))
    a, b = on.windows["s72_f072"]["mode"], off.windows["s72_f072"]["mode"]
    assert 0 < a.sum() < 0.98 * b.sum() and a.max() <= b.max() * 1.02                 # warm November ground + marginal air: less snow
    from pwpf_blender import export
    import json
    assert json.loads(export.write_site(on, cfg, tmp_path / "a").read_text())["accumulation"]["ground_melt_in_per_c"] == 0.15
    assert json.loads(export.write_site(off, cfg_off, tmp_path / "b").read_text())["accumulation"] is None


def test_t2m_gate_turns_off_native_snow_where_downscaled_2m_is_warm():
    """The model's cell says snow, but the 2.5 km point sits below the model terrain, so its 2 m temperature is warmer."""
    rg, *_ = _identity_setup()
    step = _step(rg.src_shape, 1.2, -3.0, qpf=10.0)                 # decision tree: snow; model surface at 100 m
    base = {"method": "fixed", "fixed": 10.0}

    def run(terrain, gate, cap=1.5):
        ctx = BinContext(terrain_m=np.full(rg.dst_shape, terrain, np.float32), downscaler=None, doy=15, slr_cfg=base,
                         ptype_cfg={"t2m_gate": gate, "snow_t2m_max_c": cap, "t2m_lapse_c_per_km": 6.5})
        return process_bin([step], rg, ctx)

    assert run(0.0, False).min() > 0                                # old behaviour: native snow everywhere
    assert run(0.0, True).max() == 0.0                              # 1.2 C + 6.5 C/km x 0.1 km = 1.85 C > 1.5 C: rain
    assert np.allclose(run(100.0, True), run(100.0, False))         # at the model's own terrain nothing changes (1.2 C)
    assert run(1000.0, True).min() > 0                              # higher than the model terrain: colder, still snow
    assert run(0.0, True, cap=2.5).min() > 0                        # the cap is configurable


def test_t2m_gate_never_removes_terrain_above_the_snow_level():
    rg, *_ = _identity_setup()
    step = _step(rg.src_shape, 4.0, -3.0, qpf=10.0)                 # model: rain at the surface
    terrain = np.full(rg.dst_shape, 2500.0, np.float32)             # ridge above the 0.5 C wet-bulb level
    ctx = lambda gate: BinContext(terrain_m=terrain, downscaler=None, doy=15, slr_cfg={"method": "fixed", "fixed": 10.0},   # noqa: E731
                                  ptype_cfg={"t2m_gate": gate})
    assert np.allclose(process_bin([step], rg, ctx(True)), process_bin([step], rg, ctx(False)))
    assert process_bin([step], rg, ctx(True)).min() > 0
