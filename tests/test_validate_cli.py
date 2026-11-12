"""validate / point CLI paths, with WPC GRIB reading replaced by arrays derived from our own fields."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from pwpf_blender import binormal, cli, export, validate
from pwpf_blender.config import load_config
from pwpf_blender.grid import GridSpec
from pwpf_blender.pipeline import Pipeline
from pwpf_blender.sources.synthetic import synthetic_prism, synthetic_terrain

CFG = Path(__file__).resolve().parents[1] / "config" / "pwpf.yaml"
STRIDE = 12


def _small_run(tmp_path):
    cfg = load_config(CFG, "synthetic", {"grid": {"stride": STRIDE}, "workers": 2})
    grid = GridSpec.ndfd_2p5km(STRIDE)
    lon, lat = grid.lonlat()
    res = Pipeline(cfg, grid, synthetic_terrain(lon, lat), synthetic_prism(lon, lat)).run(datetime(2026, 1, 15, 12))
    npz = tmp_path / "fields.npz"
    export.save_npz(res, npz)
    return res, npz


def _full(a):
    return np.kron(a, np.ones((STRIDE, STRIDE), np.float32))[:1377, :2145]


def test_validate_compare_and_calibrate(tmp_path, monkeypatch):
    res, npz = _small_run(tmp_path)
    w = res.windows["s24_f024"]
    m, lo, hi = w["mode"], w["s_lo"], w["s_hi"]
    wpc = tmp_path / "wpc"
    wpc.mkdir()
    stamp, fhr = "2026011512", 24
    for q in (5, 10, 25, 50, 75, 90, 95):
        (wpc / f"2.5kmprcntil_24hsnow_{q:02d}pt_{stamp}f{fhr:03d}.grb").touch()
    (wpc / f"psnow24_ge4_{stamp}f{fhr:03d}.grb").touch()

    def fake_read(path):
        name = Path(path).name
        if "prcntil" in name:
            q = int(name.split("_")[2][:2])
            return _full(binormal.percentile(m, lo, hi, q / 100).astype(np.float32)) * 0.0254   # metres, like WPC
        return _full((100 * binormal.exceedance(m, lo, hi, 4.0)).astype(np.float32))            # percent

    monkeypatch.setattr(validate, "read_grib", fake_read)
    rep = validate.compare(str(npz), str(wpc), "s24_f024", datetime(2026, 1, 15, 12), 24)
    assert rep["percentiles"][50]["rmse"] < 1e-3 and rep["percentiles"][50]["n"] > 0
    assert rep["probabilities"][4.0]["mae_prob"] < 1e-4
    # WPC (fake) 50 % heavier than ours -> calibration suggests ~1.5
    monkeypatch.setattr(validate, "read_grib", lambda p: fake_read(p) * 1.5)
    adj = validate.calibrate_slr(str(npz), str(wpc), "s24_f024", datetime(2026, 1, 15, 12), 24, 1.0)
    assert 1.4 < adj < 1.6


def test_point_cli_prints_every_window(tmp_path, capsys):
    res, npz = _small_run(tmp_path)
    cli.main(["point", "--npz", str(npz), "--lat", "40.0", "--lon", "-105.0"])
    out = capsys.readouterr().out
    assert out.count(" mode ") >= 15 and "percentiles" in out
    assert "s72_f072:" in out
