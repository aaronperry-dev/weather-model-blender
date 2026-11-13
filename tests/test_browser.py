"""End-to-end check of the real viewer in headless Chrome (skipped unless playwright, Chrome and a Leaflet copy exist).

    pip install playwright    # the system Chrome is used, no browser download
    npm i leaflet             # or set PWPF_LEAFLET_DIR to a leaflet/dist directory

The page is served from a temp dir with a synthetic site; unpkg, basemap and terrain tiles are answered locally."""
import functools
import http.server
import io
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pw = pytest.importorskip("playwright.sync_api")
from PIL import Image  # noqa: E402

from pwpf_blender import export  # noqa: E402
from pwpf_blender.config import load_config  # noqa: E402
from pwpf_blender.grid import GridSpec  # noqa: E402
from pwpf_blender.pipeline import Pipeline  # noqa: E402
from pwpf_blender.sources.synthetic import synthetic_prism, synthetic_terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHROME = os.environ.get("PWPF_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
LEAFLET = next((Path(p) for p in (os.environ.get("PWPF_LEAFLET_DIR", ""), ROOT / "node_modules/leaflet/dist", "/tmp/bt/node_modules/leaflet/dist")
                if p and (Path(p) / "leaflet.js").exists()), None)
pytestmark = pytest.mark.skipif(not (CHROME and LEAFLET), reason="needs Chrome and a Leaflet copy")
# (note: never return the Leaflet map object from page.evaluate - Playwright would try to serialise the whole object graph)


def _png(rgba=(0, 0, 0, 0)):
    b = io.BytesIO()
    Image.new("RGBA", (256, 256), rgba).save(b, "PNG")
    return b.getvalue()


def _terrarium(x, y):
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    v = (600 + 500 * np.sin((x * 256 + xx) / 37.0) * np.cos((y * 256 + yy) / 53.0) + 32768) * 256
    b = io.BytesIO()
    Image.fromarray(np.stack([v // 65536, (v // 256) % 256, v % 256], -1).astype(np.uint8), "RGB").save(b, "PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    shutil.copytree(ROOT / "web", root, dirs_exist_ok=True)
    cfg = load_config(ROOT / "config" / "pwpf.yaml", "synthetic", {"grid": {"stride": 8}, "workers": 2})
    grid = GridSpec.ndfd_2p5km(8)
    lon, lat = grid.lonlat()
    res = Pipeline(cfg, grid, synthetic_terrain(lon, lat), synthetic_prism(lon, lat)).run(datetime(2026, 1, 15, 12))
    export.write_site(res, cfg, root / "data")

    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        br = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--disable-background-networking", "--disable-sync",
                                                              "--disable-component-update", "--no-first-run", "--disable-gpu"])
        yield br
        br.close()


@pytest.fixture()
def page(site, browser):
    if True:
        br = browser
        ctx = br.new_context(viewport={"width": 1500, "height": 900})
        dem = []

        def route(r):
            u = r.request.url
            if u.startswith("http://127.0.0.1"):
                return r.continue_()
            if "unpkg.com/leaflet" in u:
                return r.fulfill(path=str(LEAFLET / u.split("/dist/")[1]))
            if "elevation-tiles-prod" in u:
                z, x, y = (int(t) for t in u.split("terrarium/")[1][:-4].split("/"))
                dem.append(u)
                return r.fulfill(body=_terrarium(x, y), content_type="image/png", headers={"access-control-allow-origin": "*"})
            if "Reference" in u:
                return r.fulfill(body=_png(), content_type="image/png", headers={"access-control-allow-origin": "*"})
            if "arcgis" in u or u.endswith(".png"):
                return r.fulfill(body=_png((250, 245, 230, 255)), content_type="image/png", headers={"access-control-allow-origin": "*"})
            return r.abort()
        ctx.route("**/*", route)
        pg = ctx.new_page()
        pg.errors, pg.dem = [], dem
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.goto(site)
        pg.wait_for_selector("#map canvas.leaflet-tile.leaflet-tile-loaded", state="attached", timeout=20000)
        pg.wait_for_timeout(1500)
        yield pg
        ctx.close()


PAINTED = """() => { let n = 0, tiles = 0;
  document.querySelectorAll('.leaflet-overlay-pane canvas.leaflet-tile').forEach(c => {
    if (getComputedStyle(c).visibility !== 'visible') return;           // a tile Leaflet never marked as loaded stays hidden
    tiles++; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i]) n++; });
  return {tiles, painted: n}; }"""


def test_defaults_title_and_layout(page):
    assert page.title() == "Legacy WPC PWPF Redux"
    assert page.inner_text("header h1") == "Legacy WPC PWPF Redux"
    assert page.evaluate("parseFloat(getComputedStyle(document.querySelector('header h1')).fontSize)") >= 30
    assert page.evaluate("parseFloat(getComputedStyle(document.body).fontSize)") >= 16
    assert page.is_checked("#terrainAdj") and page.is_checked("#shade")
    assert page.get_attribute("#products button.on", "data-kind") == "mode"                   # Snowfall forecast (mode) first
    assert page.get_attribute("#hazards button.on", "data-prod") == "snow"
    m, t = page.locator("#map").bounding_box(), page.locator("#membersTable").bounding_box()
    assert m["height"] >= 520 and t["y"] > m["y"] + m["height"] - 5                          # table below the (tall) map
    assert page.inner_text("#membersSummary").startswith("34 of 34 members")
    assert page.is_visible("#calib summary") and "BLEND (mode)" in page.inner_text("#calibText") + page.evaluate("document.getElementById('calibText').textContent")
    assert not page.errors


def test_data_is_visible_with_and_without_terrain_adjustment(page):
    a = page.evaluate(PAINTED)
    assert a["tiles"] >= 4 and a["painted"] > 5000
    page.uncheck("#terrainAdj")                                                              # regression: used to show no data
    page.wait_for_timeout(1500)
    b = page.evaluate(PAINTED)
    assert b["tiles"] >= 4 and b["painted"] > 5000
    page.check("#terrainAdj")
    page.wait_for_timeout(1500)
    assert page.evaluate(PAINTED)["painted"] > 5000
    assert not page.errors


def test_terrain_shading_layer_and_elevation_tiles(page):
    page.evaluate("window.__pwpfMap.setView([44.5, -95], 6, {animate: false}); null")
    page.wait_for_timeout(2500)
    assert page.evaluate("document.querySelectorAll('.leaflet-shade-pane canvas.leaflet-tile').length") > 0
    assert len(page.dem) > 0
    page.uncheck("#shade")
    page.wait_for_timeout(500)
    assert page.evaluate("document.querySelectorAll('.leaflet-shade-pane canvas.leaflet-tile').length") == 0


def test_cursor_tooltip_follows_the_pointer_and_shows_exact_values(page):
    box = page.locator("#map").bounding_box()
    page.evaluate("window.__pwpfMap.setView([44.5, -95], 6, {animate: false}); null")
    page.wait_for_timeout(2000)
    page.mouse.move(box["x"] + 600, box["y"] + 400)
    page.wait_for_selector("#tip:not([hidden])")
    page.wait_for_timeout(800)
    txt = page.inner_text("#tip")
    assert "FORECAST (MODE)" in txt and " in" in txt and "10th percentile" in txt and "°N" in txt
    assert "Ground" in txt and "snowfall ×" in txt                                           # terrain-adjusted at zoom >= 5
    first = page.locator("#tip").bounding_box()
    page.mouse.move(box["x"] + 300, box["y"] + 200)
    page.wait_for_timeout(500)
    second = page.locator("#tip").bounding_box()
    assert abs(first["x"] - second["x"]) > 100                                               # it moved with the cursor
    page.click("#products button[data-kind=prob]")
    page.wait_for_timeout(800)
    page.mouse.move(box["x"] + 600, box["y"] + 400)
    page.wait_for_timeout(500)
    assert "CHANCE OF AT LEAST" in page.inner_text("#tip").upper()
    page.mouse.move(5, 5)                                                                    # off the map: hidden
    page.wait_for_timeout(300)
    assert page.is_hidden("#tip")


def test_boundary_hierarchy_layers_exist(page):
    page.evaluate("window.__pwpfMap.setView([44.5, -90], 7, {animate: false}); null")
    page.wait_for_timeout(2500)
    names = page.evaluate("[...document.querySelectorAll('.leaflet-pane')].map(p => p.className.replace('leaflet-pane leaflet-', '').replace('-pane', ''))")
    for n in ("bcase", "b-counties", "b-states", "b-coast", "b-countries"):
        assert n in names, n
    z = page.evaluate("['bcase','b-counties','b-states','b-coast','b-countries'].map(n => +getComputedStyle(document.querySelector('.leaflet-' + n + '-pane')).zIndex)")
    assert z == sorted(z) and z[-1] < 450                                                    # countries on top, below place labels


TILES = """() => { const r = {adj: 0, raw: 0, painted: 0};
  document.querySelectorAll('.leaflet-overlay-pane canvas.leaflet-tile').forEach(c => {
    if (getComputedStyle(c).visibility !== 'visible') return;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i]) n++;
    if (!n) return;
    r.painted += n; if (c.dataset.adjusted === '1') r.adj++; else r.raw++; });
  r.layers = document.querySelectorAll('.leaflet-overlay-pane .leaflet-layer').length; return r; }"""


def _zoom(page, z, lat=44.5, lon=-95):
    page.evaluate("window.__pwpfMap.setView([%s, %s], %s, {animate: false}); null" % (lat, lon, z))


def test_terrain_adjustment_off_then_on_returns_the_adjusted_data(page):
    """Regression: after off -> on the adjusted (high resolution) data did not come back."""
    _zoom(page, 7)
    page.wait_for_timeout(2500)
    a = page.evaluate(TILES)
    assert a["adj"] >= 4 and a["raw"] == 0 and a["painted"] > 5000
    for _ in range(2):
        page.uncheck("#terrainAdj")
        page.wait_for_timeout(1500)
        b = page.evaluate(TILES)
        assert b["raw"] >= 4 and b["adj"] == 0 and b["painted"] > 5000 and b["layers"] == 1
        page.check("#terrainAdj")
        page.wait_for_timeout(2500)
        c = page.evaluate(TILES)
        assert c["adj"] >= 4 and c["raw"] == 0 and c["painted"] > 5000 and c["layers"] == 1
    assert not page.errors


def test_rapid_toggling_with_slow_elevation_tiles_ends_in_the_chosen_state(page):
    import time

    def slow(r):
        z, x, y = (int(t) for t in r.request.url.split("terrarium/")[1][:-4].split("/"))
        time.sleep(0.25)
        r.fulfill(body=_terrarium(x, y), content_type="image/png", headers={"access-control-allow-origin": "*"})
    page.context.route("**/elevation-tiles-prod/**", slow)
    _zoom(page, 8, 40.5, -100)
    page.wait_for_timeout(500)
    for _ in range(4):
        page.uncheck("#terrainAdj")
        page.wait_for_timeout(200)
        page.check("#terrainAdj")
        page.wait_for_timeout(200)
    page.wait_for_timeout(9000)
    a = page.evaluate(TILES)
    assert a["adj"] >= 4 and a["raw"] == 0 and a["layers"] == 1                 # settled on adjusted, old layers gone
    page.uncheck("#terrainAdj")
    page.wait_for_timeout(1500)
    b = page.evaluate(TILES)
    assert b["raw"] >= 4 and b["adj"] == 0 and b["painted"] > 5000
    assert not page.errors


def test_data_stays_visible_when_elevation_tiles_fail_or_hang(page):
    _zoom(page, 7)
    page.wait_for_timeout(2500)                                                   # some elevation tiles load fine first
    assert page.evaluate(TILES)["adj"] >= 4
    page.context.route("**/elevation-tiles-prod/**", lambda r: r.abort())      # every elevation tile fails
    _zoom(page, 9, 38.0, -98)
    page.wait_for_timeout(6000)
    a = page.evaluate(TILES)
    assert a["raw"] >= 4 and a["painted"] > 5000                                  # un-adjusted, but not blank
    page.context.unroute("**/elevation-tiles-prod/**")
    page.context.route("**/elevation-tiles-prod/**", lambda r: None)             # requests that never answer
    page.evaluate("window.__pwpfDemTimeoutMs = 600; null")
    _zoom(page, 9, 36.0, -101)
    page.wait_for_timeout(7000)
    b = page.evaluate(TILES)
    assert b["raw"] >= 4 and b["painted"] > 5000
    assert not page.errors


def _set_step(page, v):
    page.evaluate("(v) => { const e = document.getElementById('step'); e.value = v; e.dispatchEvent(new Event('input')); e.dispatchEvent(new Event('change')); }", v)


def test_forecast_step_slider_shows_start_and_end_and_moves_the_map(page):
    n = page.evaluate("+document.getElementById('step').max") + 1
    assert n >= 3 and page.evaluate("+document.getElementById('step').value") == 0
    lab0 = page.inner_text("#stepLabel")
    assert "Step 1 of %d" % n in lab0 and "START" in lab0.upper() and "END" in lab0.upper() and "Z" in lab0
    assert page.is_disabled("#stepPrev") and page.is_enabled("#stepNext")
    h0 = page.evaluate("location.hash")
    page.click("#stepNext")
    page.wait_for_timeout(800)
    lab1 = page.inner_text("#stepLabel")
    assert "Step 2 of %d" % n in lab1 and lab1 != lab0 and page.evaluate("location.hash") != h0
    assert page.evaluate("+document.getElementById('step').value") == 1 and page.is_enabled("#stepPrev")
    _set_step(page, n - 1)                                                                   # the end of the slider
    page.wait_for_timeout(800)
    assert "Step %d of %d" % (n, n) in page.inner_text("#stepLabel") and page.is_disabled("#stepNext")
    assert page.evaluate(TILES)["painted"] >= 0 and not page.errors
    page.click("#lengths button:has-text('48 h')")                                           # another length: its own set of steps
    page.wait_for_timeout(800)
    m48 = page.evaluate("+document.getElementById('step').max") + 1
    assert "48 h ending" in page.inner_text("#stepLabel") and m48 != n


def test_elevation_filter_hides_low_ground_and_is_in_feet(page):
    _zoom(page, 7)
    page.wait_for_timeout(2500)
    full = page.evaluate(TILES)["painted"]
    assert full > 5000
    page.fill("#elevFt", "2000")                                                             # the test terrain spans about 330-3600 ft
    page.wait_for_timeout(3500)
    part = page.evaluate(TILES)
    assert 0 < part["painted"] < 0.9 * full and part["layers"] == 1
    assert "el=2000" in page.evaluate("location.hash") and "2,000 ft" in page.inner_text("#elevNote")
    page.fill("#elevFt", "20000")
    page.wait_for_timeout(3500)
    assert page.evaluate(TILES)["painted"] == 0
    box = page.locator("#map").bounding_box()
    page.mouse.move(box["x"] + 600, box["y"] + 400)
    page.wait_for_selector("#tip:not([hidden])")
    page.wait_for_timeout(800)
    assert "Hidden by the 20,000 ft elevation filter" in page.inner_text("#tip")
    page.click("#elevChips button[data-ft='1000']")                                          # quick value
    page.wait_for_timeout(3500)
    assert page.input_value("#elevFt") == "1000" and "on" in page.get_attribute("#elevChips button[data-ft='1000']", "class")
    mid = page.evaluate(TILES)["painted"]
    assert 0 < mid and mid >= part["painted"]
    page.click("#elevClear")
    page.wait_for_timeout(3500)
    back = page.evaluate(TILES)["painted"]
    assert back >= 0.95 * full and "el=" not in page.evaluate("location.hash")
    page.mouse.move(box["x"] + 600, box["y"] + 400)
    page.wait_for_timeout(1200)
    txt = page.inner_text("#tip")
    import re
    assert re.search(r"Ground [\d,]+ ft", txt) and not re.search(r"\d m\b", txt)            # feet, not metres
    assert not page.errors


def test_elevation_filter_hides_data_it_cannot_place_when_tiles_fail(page):
    _zoom(page, 7)
    page.wait_for_timeout(2000)
    page.fill("#elevFt", "1000")
    page.wait_for_timeout(3000)
    assert page.evaluate(TILES)["painted"] > 0
    page.context.route("**/elevation-tiles-prod/**", lambda r: r.abort())
    _zoom(page, 9, 38.0, -98)                                                                # new tiles whose elevation cannot be loaded
    page.wait_for_timeout(7000)
    assert page.evaluate(TILES)["painted"] == 0                                              # nothing is shown unfiltered
    assert page.is_visible("#elevNote") and ("failed" in page.inner_text("#elevNote") or "could not" in page.inner_text("#elevNote"))
