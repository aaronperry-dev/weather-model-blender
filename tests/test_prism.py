"""PRISM normals: file discovery, BIL + GeoTIFF readers, directory download, non-fatal failure."""
import functools
import http.server
import io
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from pwpf_blender import noaa_http as nh
from pwpf_blender import static_data as sd
from pwpf_blender.grid import GridSpec

# a small PRISM-like grid: 1/24 degree cells covering the CONUS box used by the NDFD grid
NCOLS, NROWS, D = 1405, 621, 1 / 24
ULX_C, ULY_C = -125.0208333333, 49.9375            # centre of the upper-left cell (PRISM .hdr convention)


def _field(month):
    """Smooth synthetic precipitation climatology: depends on lon, lat and month."""
    lon = ULX_C + np.arange(NCOLS) * D
    lat = ULY_C - np.arange(NROWS) * D
    X, Y = np.meshgrid(lon, lat)
    return (50 + 20 * np.sin(np.radians(X * 3)) + 10 * (Y - 35) / 10 + month * 4).astype(np.float32)


def _expected(month, grid):
    lon, lat = grid.lonlat()
    return (50 + 20 * np.sin(np.radians(lon * 3)) + 10 * (lat - 35) / 10 + month * 4).astype(np.float32)


def _write_tif(path, arr, corner=True, nodata="-9999", georef=True, compression="tiff_adobe_deflate"):
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    if georef:
        ifd[33550] = (D, D, 0.0)
        ifd.tagtype[33550] = 12
        ulx, uly = (ULX_C - D / 2, ULY_C + D / 2) if corner else (ULX_C, ULY_C)
        ifd[33922] = (0.0, 0.0, 0.0, ulx, uly, 0.0)
        ifd.tagtype[33922] = 12
        if not corner:                                  # GTRasterTypeGeoKey = PixelIsPoint
            ifd[34735] = (1, 1, 0, 1, 1025, 0, 1, 2)
            ifd.tagtype[34735] = 3
    ifd[42113] = nodata
    im.save(path, tiffinfo=ifd, compression=compression)


def _write_bil_zip(path, arr):
    hdr = (f"BYTEORDER I\nLAYOUT BIL\nNROWS {arr.shape[0]}\nNCOLS {arr.shape[1]}\nNBANDS 1\nNBITS 32\n"
           f"PIXELTYPE FLOAT\nULXMAP {ULX_C}\nULYMAP {ULY_C}\nXDIM {D}\nYDIM {D}\nNODATA -9999\n")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("PRISM_ppt_test.hdr", hdr)
        z.writestr("PRISM_ppt_test.bil", arr.astype("<f4").tobytes())


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("name,month", [
    ("PRISM_ppt_30yr_normal_4kmM4_01_bil.zip", 1),
    ("PRISM_ppt_30yr_normal_4kmM4_12_bil.zip", 12),
    ("prism_ppt_us_25m_202003.tif", 3),
    ("prism_ppt_us_25m_2020_07.zip", 7),
    ("prism_ppt_us_25m_09.tif", 9),
    ("PRISM_ppt_30yr_normal_4kmM4_annual_bil.zip", None),
    ("PRISM_ppt_30yr_normal_4kmM4_all_bil.zip", None),
    ("prism_ppt_us_25m_2020.tif", None),                   # 4-digit run = annual / year, not a month
    ("PRISM_ppt_30yr_normal_4kmM4_01_bil.stn.csv", None),
    ("readme_4km_normals_M4.txt", None),
    ("PRISM_ppt_30yr_normal_4kmM4_13_bil.zip", None),
])
def test_prism_month(name, month):
    assert sd.prism_month(name) == month


def test_pick_monthly_prefers_geotiff_and_ignores_extras():
    names = [f"PRISM_ppt_30yr_normal_4kmM4_{m:02d}_bil.zip" for m in range(1, 13)]
    names += ["PRISM_ppt_30yr_normal_4kmM4_annual_bil.zip", "PRISM_ppt_30yr_normal_4kmM4_all_bil.zip", "x.xml"]
    names += ["prism_ppt_us_25m_202005.tif"]
    got = sd.pick_monthly(names)
    assert sorted(got) == list(range(1, 13)) and got[5].endswith(".tif") and got[4].endswith("_04_bil.zip")


# ------------------------------------------------------------------ readers
def test_read_geotiff_corner_and_nodata(tmp_path):
    a = _field(3)
    a[:5, :5] = -9999
    _write_tif(tmp_path / "m.tif", a)
    arr, h = sd.read_geotiff(tmp_path / "m.tif")
    assert arr.shape == (NROWS, NCOLS) and np.isnan(arr[:5, :5]).all() and np.allclose(arr[10:20, 10:20], a[10:20, 10:20])
    assert h["ulx"] == pytest.approx(ULX_C - D / 2) and h["uly"] == pytest.approx(ULY_C + D / 2)
    assert h["dx"] == pytest.approx(D)


def test_read_geotiff_pixel_is_point_and_missing_tags(tmp_path):
    _write_tif(tmp_path / "p.tif", _field(1), corner=False)
    _, h = sd.read_geotiff(tmp_path / "p.tif")
    assert h["ulx"] == pytest.approx(ULX_C - D / 2) and h["uly"] == pytest.approx(ULY_C + D / 2)
    _write_tif(tmp_path / "n.tif", _field(1), georef=False)
    _, h2 = sd.read_geotiff(tmp_path / "n.tif")                 # falls back to the documented PRISM grid
    assert h2["ulx"] == pytest.approx(ULX_C - D / 2)


def test_bil_zip_header_is_converted_to_corner(tmp_path):
    _write_bil_zip(tmp_path / "b.zip", _field(2))
    arr, h = sd.read_prism_raster(tmp_path / "b.zip")
    assert h["ulx"] == pytest.approx(ULX_C - D / 2) and h["uly"] == pytest.approx(ULY_C + D / 2)
    assert arr.shape == (NROWS, NCOLS)


def test_zip_containing_geotiff_and_garbage(tmp_path):
    _write_tif(tmp_path / "in.tif", _field(4))
    with zipfile.ZipFile(tmp_path / "z.zip", "w") as z:
        z.write(tmp_path / "in.tif", "prism_ppt_04.tif")
    arr, _ = sd.read_prism_raster(tmp_path / "z.zip")
    assert arr.shape == (NROWS, NCOLS)
    (tmp_path / "bad.zip").write_bytes(b"<html>Service moved</html>")
    with pytest.raises(nh.FetchError, match="neither a zip nor a TIFF"):
        sd.read_prism_raster(tmp_path / "bad.zip")


def test_prism_to_grid_samples_correctly_for_both_formats(tmp_path):
    grid = GridSpec.ndfd_2p5km(40)
    exp = [_expected(m, grid) for m in range(1, 13)]
    tifs, bils = {}, {}
    for m in range(1, 13):
        tifs[m] = tmp_path / f"t{m:02d}.tif"
        _write_tif(tifs[m], _field(m))
        bils[m] = tmp_path / f"b{m:02d}.zip"
        _write_bil_zip(bils[m], _field(m))
    for files in (tifs, bils):
        out = sd.prism_to_grid(files, grid)
        assert out.shape == (12,) + grid.shape
        ok = np.isfinite(out)
        assert 0.5 < ok.mean() < 1.0                          # the NDFD grid reaches beyond PRISM's CONUS extent
        lon, lat = grid.lonlat()
        outside = (lat > 49.96) | (lat < 24.04) | (lon < -125.05) | (lon > -66.45)
        assert not ok[0][outside].any()                       # nothing invented outside the raster
        for m in (0, 5, 11):
            assert np.nanmax(np.abs(out[m][ok[m]] - exp[m][ok[m]])) < 0.5     # smooth field, 1/24 deg cells


# ------------------------------------------------------------------ directory download
@pytest.fixture()
def prism_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("prism_srv")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}/normals/"
    srv.shutdown()


def _populate(root: Path, names_to_months, as_tif=True):
    d = root / "normals"
    d.mkdir(exist_ok=True)
    for name, m in names_to_months.items():
        if name.endswith(".tif"):
            _write_tif(d / name, _field(m))
        else:
            (d / name).write_bytes(b"x")
    return d


def test_download_discovers_files_from_listing(prism_server, tmp_path):
    root, url = prism_server
    _populate(root, {f"prism_ppt_us_25m_2020{m:02d}.tif": m for m in range(1, 13)}
              | {"prism_ppt_us_25m_2020.tif": 6, "readme.txt": 0})
    files = sd.download_prism(tmp_path / "dl", url)
    assert sorted(files) == list(range(1, 13))
    arr, _ = sd.read_prism_raster(files[7])
    assert np.allclose(arr[100, 100], _field(7)[100, 100])


def test_download_reports_unrecognised_listing(prism_server, tmp_path):
    root, url = prism_server
    _populate(root, {"prism_ppt_us_25m_202001.tif": 1, "mystery_file.tif": 2})
    with pytest.raises(nh.FetchError, match=r"months \[2, 3.*first 40.*mystery_file"):
        sd.download_prism(tmp_path / "dl2", url)


def test_download_rejects_html_error_pages(prism_server, tmp_path):
    root, url = prism_server
    d = _populate(root, {f"p_{m:02d}.tif": m for m in range(1, 13)})
    (d / "p_05.tif").write_text("<html><body>Service unavailable</body></html>")
    with pytest.raises(nh.FetchError, match="not a zip/TIFF.*Service unavailable"):
        sd.download_prism(tmp_path / "dl3", url)


# ------------------------------------------------------------------ non-fatal PRISM
def test_prepare_static_survives_prism_failure(tmp_path, monkeypatch, capsys):
    terr = np.zeros((1377, 2145), np.float32)
    np.save(tmp_path / "terr.npy", terr)

    def boom(*a, **k):
        raise nh.FetchError("HTTP 404")
    monkeypatch.setattr(sd, "download_prism", boom)
    t, p = sd.prepare_static(tmp_path / "out", str(tmp_path / "terr.npy"))
    assert t.exists() and p is None
    assert "::warning" in capsys.readouterr().out
    with pytest.raises(nh.FetchError):
        sd.prepare_static(tmp_path / "out2", str(tmp_path / "terr.npy"), require_prism=True)


def test_prepare_static_from_local_folder(tmp_path, monkeypatch):
    folder = tmp_path / "prism"
    folder.mkdir()
    for m in range(1, 13):
        _write_tif(folder / f"prism_ppt_us_25m_2020{m:02d}.tif", _field(m))
    np.save(tmp_path / "terr.npy", np.zeros((1377, 2145), np.float32))
    t, p = sd.prepare_static(tmp_path / "out", str(tmp_path / "terr.npy"), str(folder))
    arr = np.load(p)
    assert arr.shape == (12, 1377, 2145) and 0.5 < np.isfinite(arr).mean() < 1.0
    # an incomplete folder is reported clearly (non-fatal) rather than silently accepted
    (folder / "prism_ppt_us_25m_202012.tif").unlink()
    t2, p2 = sd.prepare_static(tmp_path / "out3", str(tmp_path / "terr.npy"), str(folder))
    assert p2 is None
