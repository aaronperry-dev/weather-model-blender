"""web/terrain.js: Terrarium decoding, the snowfall terrain adjustment, hill-shading, tile cache."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def rep():
    p = subprocess.run(["node", str(ROOT / "tests" / "js_terrain_check.js")], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def test_terrarium_decoding_and_config_defaults(rep):
    assert rep["terrarium"] == pytest.approx([0.0, 131 * 256 + 232 + 0.5 - 32768, 127 * 256 + 255 + 255 / 256 - 32768], abs=1e-3)
    assert rep["cfg"]["precip_gradient_per_km"] == 0.2 and rep["cfg"]["precip_factor_range"] == [0.75, 1.35]
    assert rep["cfg_override"] == 50


def test_scale_cases(rep):
    assert rep["same"] == pytest.approx(1.0)                              # at model terrain height nothing changes
    assert rep["all_snow_up"] == pytest.approx(1.1, abs=1e-6)             # +500 m x 0.2/km, no phase change
    assert rep["cap_range_up"] == pytest.approx(1.3, abs=1e-6)            # dz limited to 1500 m
    assert rep["cap_range_down"] == pytest.approx(0.75, abs=1e-6)         # factor floor
    assert rep["valley"] < 0.15                                           # rain in the valley, snow on the model's hill
    assert rep["ridge"] > 2.0                                             # snow where the model cell is rain
    assert rep["pot_cap"] == pytest.approx(1.1 * 5.5 / 5, abs=0.2)        # limited by potential snowfall
    assert rep["pot_cap"] < rep["ridge"]
    assert rep["far_from_line"] == pytest.approx(1.02, abs=0.02)          # elevation factor only
    assert rep["snow_frac"][0] == pytest.approx(0.5) and rep["snow_frac"][1] > 0.99 and rep["snow_frac"][2] < 0.01


def test_hillshade_and_sample(rep):
    assert rep["shade_west_facing"][3] > 0 and rep["shade_west_facing"][0] == 255         # faces the NW sun: lit
    assert rep["shade_east_facing"][3] > 0 and rep["shade_east_facing"][0] == 0           # faces away: shadowed
    assert rep["shade_flat"][3] == 0                                                      # flat ground stays transparent
    assert rep["sample"] == pytest.approx([0, 5, 15, 20, 30])


def test_tile_cache(rep):
    assert rep["dedupe"] == [1, True]                      # concurrent requests share one load
    assert rep["size"] == 2                                # LRU limit
    assert rep["hit_loads"] == 3 and rep["reload_loads"] == 4
    assert rep["fail"] == [True, 2]                        # a failed load is not cached
