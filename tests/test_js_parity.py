"""The browser re-implements the binormal maths from the data PNG.  Verify it matches Python."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pwpf_blender import binormal, export

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_matches_python(tmp_path):
    rng = np.random.default_rng(3)
    n = 400
    m = rng.uniform(0, 30, n)
    s1 = rng.uniform(0.05, 6, n)
    s2 = rng.uniform(0.05, 12, n)
    t = rng.uniform(0, 40, n)
    q = rng.uniform(0.02, 0.98, n)
    enc = {"A": export.ENC_A, "xmax": export.ENC_XMAX}

    # a real data PNG from a tiny synthetic field -> RGBA bytes (alpha 255, like a browser canvas read)
    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    mode = 14 * np.exp(-(((xx - 40) / 14) ** 2 + ((yy - 30) / 10) ** 2))
    lo, hi = 0.15 * mode + 0.01, 0.4 * mode + 0.02
    rgb = np.stack([export.encode(mode), export.encode(lo), export.encode(hi)], -1)
    png = tmp_path / "t.png"
    Image.fromarray(rgb, "RGB").save(png)
    rgba = np.array(Image.open(png).convert("RGBA"))
    assert (rgba[..., 3] == 255).all()
    (tmp_path / "rgba.bin").write_bytes(rgba.tobytes())
    (tmp_path / "in.json").write_text(json.dumps({"triples": np.column_stack([m, s1, s2, t, q]).tolist(), "enc": enc}))

    subprocess.run(["node", str(ROOT / "tests" / "js_check.js"), str(tmp_path / "in.json"),
                    str(tmp_path / "rgba.bin"), str(tmp_path / "out.json")], check=True)
    out = json.loads((tmp_path / "out.json").read_text())

    assert np.allclose(out["exc"], binormal.exceedance(m, s1, s2, t), atol=2e-6)
    assert np.allclose(out["pct"], binormal.percentile(m, s1, s2, q), atol=2e-5, rtol=2e-5)
    assert np.allclose(out["lut"], export.decode(np.arange(256)), rtol=1e-5, atol=1e-6)

    dec = [export.decode(rgba[..., i].astype(int)) for i in range(3)]
    dm, dlo, dhi = dec[0].ravel(), np.maximum(dec[1].ravel(), 1e-3), np.maximum(dec[2].ravel(), 1e-3)
    assert np.allclose(out["mode"], dm, rtol=1e-5, atol=1e-6)
    active = ~((dm < 0.0119) & (dhi < 0.02))
    exp_prob = np.where(active, 100 * binormal.exceedance(dm, dlo, dhi, 4.0), 0)
    exp_pct = np.where(active, binormal.percentile(dm, dlo, dhi, 0.9), 0)
    assert np.allclose(np.array(out["prob_value"]), exp_prob, atol=2e-3)
    assert np.allclose(np.array(out["pct_value"]), exp_pct, atol=5e-4)
    assert out["prob_colored"] > 0


def test_js_ice_encoding_and_zero_handling(tmp_path):
    """Ice layers use a finer log encoding; the viewer must still treat encoded zeros as 'no signal'."""
    enc = {"A": export.ICE_ENC_A, "xmax": export.ICE_ENC_XMAX}
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w]
    mode = 0.3 * np.exp(-(((xx - 30) / 8) ** 2 + ((yy - 20) / 6) ** 2))
    mode[mode < 0.004] = 0.0                                   # most of the map: no ice at all
    lo, hi = 0.2 * mode, 0.5 * mode
    rgb = np.stack([export.encode(a, **{"A": enc["A"], "xmax": enc["xmax"]}) for a in (mode, lo, hi)], -1)
    png = tmp_path / "ice.png"
    Image.fromarray(rgb, "RGB").save(png)
    rgba = np.array(Image.open(png).convert("RGBA"))
    (tmp_path / "rgba.bin").write_bytes(rgba.tobytes())
    (tmp_path / "in.json").write_text(json.dumps({"triples": [[0.1, 0.02, 0.05, 0.1, 0.5]], "enc": enc,
                                                  "t": 0.1, "q": 90, "pal": "ice"}))
    subprocess.run(["node", str(ROOT / "tests" / "js_check.js"), str(tmp_path / "in.json"),
                    str(tmp_path / "rgba.bin"), str(tmp_path / "out.json")], check=True)
    out = json.loads((tmp_path / "out.json").read_text())
    dec = [export.decode(rgba[..., i].astype(int), enc["A"], enc["xmax"]) for i in range(3)]
    assert np.allclose(out["mode"], dec[0].ravel(), rtol=1e-5, atol=1e-7)
    prob = np.array(out["prob_value"]).reshape(h, w)
    zero = (rgba[..., 0] == 0) & (rgba[..., 2] == 0)
    assert zero.sum() > h * w * 0.5 and (prob[zero] == 0).all()              # encoded zeros are skipped
    dm, dlo, dhi = dec[0], np.maximum(dec[1], 1e-3), np.maximum(dec[2], 1e-3)
    exp = 100 * binormal.exceedance(dm, dlo, dhi, 0.1)
    assert np.allclose(prob[~zero], exp[~zero], atol=2e-3)
    assert prob.max() > 5                                                    # a real ice core is rendered
    assert out["prob_colored"] > 0
