"""Viewer elevation filter (web/core.js renderTile minElev), checked in Node."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
ROOT = Path(__file__).resolve().parents[1]


def test_render_tile_hides_pixels_below_the_minimum_elevation():
    p = subprocess.run(["node", str(ROOT / "tests" / "js_elev_check.js")], capture_output=True, text=True, check=True)
    r = json.loads(p.stdout)
    w0, e0 = r["none"]
    assert w0 > 500 and e0 > 500                                   # the whole tile has data
    assert r["low"] == r["none"] and r["dem_only"] == r["none"]    # a filter below the ground, or only a DEM: unchanged
    assert r["mid"][0] == 0 and r["mid"][1] == e0                  # the low (west) half disappears, the high half stays intact
    assert r["high"] == [0, 0]
    assert r["nodem"] == [0, 0]                                   # no elevation data: nothing is drawn rather than unfiltered data


def test_viewer_text_is_imperial():
    for name in ("app.js", "index.html"):
        txt = (ROOT / "web" / name).read_text()
        for bad in ("' m'", "°C", "%/km", "per °C"):
            assert bad not in txt, (name, bad)
