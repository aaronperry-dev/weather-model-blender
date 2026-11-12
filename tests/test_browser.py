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
