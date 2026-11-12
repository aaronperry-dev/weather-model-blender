"""The map layer colours every screen pixel from an interpolated field (web/core.js renderTile), with optional
terrain adjustment from elevation tiles."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def rep():
    p = subprocess.run(["node", str(ROOT / "tests" / "js_tile_check.js")], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def test_peak_pixel_has_the_peak_class_and_alpha(rep):
    assert rep["centre"] == rep["expect_centre"]


@pytest.mark.parametrize("key", ["walk", "walk_bilinear", "walk_bicubic"])
def test_classes_step_down_smoothly_across_one_cell(rep, key):
    walk = rep[key]
    assert rep["cell_px"] > 20                                   # at this zoom a data cell is many screen pixels wide
    seen = [w for w in walk if w >= 0]
    assert seen[0] == 7 and seen == sorted(seen, reverse=True)   # 80 % class first, then monotonically lower classes
    assert len(set(seen)) >= 5                                   # 80 -> 0 passes through several classes: a smooth ramp
    assert walk[-1] == -1                                        # and ends transparent (value below the first class)
    assert rep["far"][3] == 0


def test_bicubic_is_used_when_cells_are_coarse_and_bilinear_otherwise(rep):
    assert rep["bicubic_differs"] and rep["auto_is_bicubic"]     # cell ~39 px wide -> Catmull-Rom
    assert rep["low_zoom_auto_is_bilinear"]                      # cell < 8 px -> plain bilinear
    assert rep["walk_bicubic"].count(7) > rep["walk_bilinear"].count(7)     # bicubic holds the peak plateau longer


def test_tiles_without_data_are_transparent_and_scale_two_is_denser(rep):
    assert rep["empty_alpha_sum"] == 0
    assert rep["r2_len"] == 4 * 512 * 512 and rep["r2_centre"][3] > 0
    assert rep["low_zoom_pixels"] > 0


def test_terrain_adjustment_in_the_tile_renderer(rep):
    assert rep["t_off"] == 7                                     # point at the model terrain height: unchanged
    assert rep["t_uphill"] == 8 and rep["t_uphill_coarse_dem"] == 8     # +500 m inside an all-snow cell: x1.1 -> next class
    assert rep["t_valley"] == -1                                 # 500 m below a 0 m snow level: the snow disappears
    assert rep["t_none_without_flag"] == 7 and rep["t_none_without_pot"] == 7   # off switch / no aux planes: untouched
    assert rep["terrain_factor_readout"] == pytest.approx(1.1, abs=1e-6)


def test_field_and_dense_tile_speed(rep):
    from pwpf_blender import binormal
    import numpy as np
    m = np.array([0, 2, 4, 8, 0.01, 12.0])
    s1 = np.array([0.05, 0.5, 1, 2, 0.05, 3.0])
    s2 = np.array([0.05, 0.8, 2, 3, 0.05, 4.0])
    exp = np.where((m < 0.05) & (s2 < 0.1), 0, 100 * binormal.exceedance(m, s1, s2, 4.0))
    assert np.allclose(rep["field"], exp, atol=2e-3)
    assert rep["dense_pixels"] > 100_000 and rep["dense_terrain_pixels"] > 100_000
    assert rep["dense_bicubic_ms"] < 600 and rep["dense_terrain_ms"] < 900      # 512 x 512 retina tile


def test_cursor_sampling_agrees_with_the_painted_tile(rep):
    for interp, n, bad in rep["sample_vs_render"]:
        assert n > 100 and bad <= 1, (interp, n, bad)           # same interpolation -> same class (allow 1 float-rounding edge)
    assert rep["sample_outside"] is None
    m, expect, zr, pot, sl = rep["sample_centre"]
    assert m == pytest.approx(expect, rel=1e-5) and (zr, pot, round(sl)) == (321, 500, 1200)
    assert rep["value_of"][0] == 2.5 and rep["value_of"][1] > 99 and rep["value_of"][2] > 0
