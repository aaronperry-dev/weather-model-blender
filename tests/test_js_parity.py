"""The browser re-implements the binormal maths and decodes the PWPB data layers itself.  Verify it matches Python."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pwpf_blender import binormal, export

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _layer(path, w, h, mode, lo, hi, A, xmax, bits, extra=()):
    planes = [{"id": i, "bits": bits, "kind": "log", "p0": A, "p1": xmax, "data": export.encode_log(a, A, xmax, bits)}
              for i, a in ((1, mode), (2, lo), (3, hi))]
    planes += list(extra)
    export.write_bin(path, w, h, planes)


def _run_node(tmp_path, triples, layer, **extra):
    (tmp_path / "in.json").write_text(json.dumps({"triples": triples, **extra}))
    subprocess.run(["node", str(ROOT / "tests" / "js_check.js"), str(tmp_path / "in.json"), str(layer), str(tmp_path / "out.json")],
                   check=True)
    return json.loads((tmp_path / "out.json").read_text())


@pytest.mark.parametrize("bits", [16, 8])
def test_js_matches_python(tmp_path, bits):
    rng = np.random.default_rng(3)
    n = 400
    m, s1, s2 = rng.uniform(0, 30, n), rng.uniform(0.05, 6, n), rng.uniform(0.05, 12, n)
    t, q = rng.uniform(0, 40, n), rng.uniform(0.02, 0.98, n)

    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    mode = 14 * np.exp(-(((xx - 40) / 14) ** 2 + ((yy - 30) / 10) ** 2))
    lo, hi = 0.15 * mode + 0.01, 0.4 * mode + 0.02
    pot = encoded = export.encode_linear(mode * 1.3, 0.01, 0.0)
    sl = export.encode_linear(900 + 3 * xx, export.SL_SCALE, export.SL_OFFSET)
    extra = [{"id": 4, "bits": 16, "kind": "linear", "p0": 0.01, "p1": 0.0, "data": pot},
             {"id": 5, "bits": 16, "kind": "linear", "p0": export.SL_SCALE, "p1": export.SL_OFFSET, "data": sl}]
    layer = tmp_path / "t.bin.gz"
    _layer(layer, w, h, mode, lo, hi, 0.5, 200.0, bits, extra)
    out = _run_node(tmp_path, np.column_stack([m, s1, s2, t, q]).tolist(), layer)

    assert np.allclose(out["exc"], binormal.exceedance(m, s1, s2, t), atol=2e-6)
    assert np.allclose(out["pct"], binormal.percentile(m, s1, s2, q), atol=2e-5, rtol=2e-5)

    ref = export.read_bin(layer)["planes"]              # Python reader == JS reader, bit for bit (up to float32)
    assert (out["W"], out["H"]) == (w, h) and set(out["planes"]) == {1, 2, 3, 4, 5}
    dm = ref[1].ravel()
    assert np.allclose(out["mode"], dm, rtol=1e-5, atol=1e-6)
    assert np.allclose(out["pot"], ref[4].ravel(), atol=1e-4) and np.allclose(out["sl"], ref[5].ravel(), atol=1e-3)
    dlo, dhi = np.maximum(ref[2].ravel(), 1e-3), np.maximum(ref[3].ravel(), 1e-3)
    assert np.allclose(out["lo"], dlo, rtol=1e-5, atol=1e-6) and np.allclose(out["hi"], dhi, rtol=1e-5, atol=1e-6)
    active = ~((dm < out["zero_mode"]) & (dhi < out["zero_hi"]))
    assert active.sum() > 1000
    assert np.allclose(np.array(out["prob_value"]), np.where(active, 100 * binormal.exceedance(dm, dlo, dhi, 4.0), 0), atol=2e-3)
    assert np.allclose(np.array(out["pct_value"]), np.where(active, binormal.percentile(dm, dlo, dhi, 0.9), 0), atol=5e-4)
    assert np.allclose(np.array(out["mode_value"]), np.where(active, dm, 0), rtol=1e-5, atol=1e-6)
    # and the decoded values are close to the originals (quantisation only)
    sel = mode > 0.5
    assert (np.abs(ref[1][sel] - mode[sel]) / mode[sel]).max() < (3e-2 if bits == 8 else 2e-3)


def test_js_ice_encoding_and_zero_handling(tmp_path):
    """Ice layers use a finer log scale; the viewer must still treat encoded zeros as 'no signal'."""
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w]
    mode = 0.3 * np.exp(-(((xx - 30) / 8) ** 2 + ((yy - 20) / 6) ** 2))
    mode[mode < 0.004] = 0.0                                   # most of the map: no ice at all
    lo, hi = 0.2 * mode, 0.5 * mode
    layer = tmp_path / "ice.bin.gz"
    _layer(layer, w, h, mode, lo, hi, export.ICE_ENC_A, export.ICE_ENC_XMAX, 16)
    out = _run_node(tmp_path, [[0.1, 0.02, 0.05, 0.1, 0.5]], layer, t=0.1, q=90)
    ref = export.read_bin(layer)["planes"]
    assert np.allclose(out["mode"], ref[1].ravel(), rtol=1e-5, atol=1e-7)
    prob = np.array(out["prob_value"]).reshape(h, w)
    zero = (ref[1] == 0) & (ref[3] == 0)
    assert zero.sum() > h * w * 0.5 and (prob[zero] == 0).all()              # encoded zeros are skipped
    dm, dlo, dhi = ref[1], np.maximum(ref[2], 1e-3), np.maximum(ref[3], 1e-3)
    exp = 100 * binormal.exceedance(dm, dlo, dhi, 0.1)
    assert np.allclose(prob[~zero], exp[~zero], atol=2e-3)
    assert prob.max() > 5                                                    # a real ice core is rendered
