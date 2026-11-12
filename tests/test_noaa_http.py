"""Tests for the direct NOAA/.idx access layer, the RRFS/GEFS/HRRR file plans and the QPF step logic.

Real GRIB2 messages are written with eccodes and served from a local HTTP server that supports byte ranges,
so the download + decode path is exercised end to end without network access."""
from __future__ import annotations

import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

ec = pytest.importorskip("eccodes")

from pwpf_blender import noaa_http as nh                      # noqa: E402
from pwpf_blender.sources import noaa_sources as ns            # noqa: E402


# ---------------------------------------------------------------------------- GRIB + server fixtures
def make_msg(vals: np.ndarray, lat0=50.0, lon0=-100.0, d=1.0) -> bytes:
    gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ny, nx = vals.shape
    ec.codes_set(gid, "Ni", nx)
    ec.codes_set(gid, "Nj", ny)
    ec.codes_set(gid, "latitudeOfFirstGridPointInDegrees", lat0)
    ec.codes_set(gid, "longitudeOfFirstGridPointInDegrees", lon0 % 360)
    ec.codes_set(gid, "latitudeOfLastGridPointInDegrees", lat0 - (ny - 1) * d)
    ec.codes_set(gid, "longitudeOfLastGridPointInDegrees", (lon0 + (nx - 1) * d) % 360)
    ec.codes_set(gid, "iDirectionIncrementInDegrees", d)
    ec.codes_set(gid, "jDirectionIncrementInDegrees", d)
    ec.codes_set(gid, "bitsPerValue", 24)
    ec.codes_set_values(gid, vals.astype(np.float64).ravel())
    buf = bytes(ec.codes_get_message(gid))
    ec.codes_release(gid)
    return buf


INVENTORY = [("TMP", "2 m above ground", "3 hour fcst"), ("PRES", "surface", "3 hour fcst"),
             ("APCP", "surface", "2-3 hour acc fcst"), ("APCP", "surface", "0-3 hour acc fcst"),
             ("CSNOW", "surface", "3 hour fcst"), ("TMP", "850 mb", "3 hour fcst"),
             ("RH", "850 mb", "3 hour fcst"), ("HGT", "850 mb", "3 hour fcst"), ("UGRD", "850 mb", "3 hour fcst")]


def build_file(shape=(5, 7)):
    """-> (grib bytes, idx text, {key: array})"""
    arrays, blob, lines = {}, b"", []
    for i, key in enumerate(INVENTORY):
        a = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100 * i)
        arrays[key] = a
        lines.append(f"{i + 1}:{len(blob)}:d=2026111112:{key[0]}:{key[1]}:{key[2]}:")
        blob += make_msg(a)
    return blob, "\n".join(lines) + "\n", arrays


class Handler(BaseHTTPRequestHandler):
    files: dict = {}
    log: list = []

    def log_message(self, *a):                                 # silence
        pass

    def do_GET(self):
        body = self.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        self.log.append((self.path, rng))
        if rng:
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else len(body) - 1
            part = body[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{a + len(part) - 1}/{len(body)}")
        else:
            part = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


@pytest.fixture()
def server():
    Handler.files, Handler.log = {}, []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", Handler
    srv.shutdown()


# ---------------------------------------------------------------------------- idx handling
def test_parse_select_and_ranges():
    _, idx, _ = build_file()
    recs = nh.parse_idx(idx)
    assert len(recs) == len(INVENTORY) and recs[0].end == recs[1].start - 1 and recs[-1].end is None
    sel = nh.select_records(recs, [r"^TMP:2 m above ground:", r"^(?:TMP|RH|HGT):850 mb:"])
    assert [r.var for r in sel] == ["TMP", "TMP", "RH", "HGT"]
    # tiny merge gap: separate ranges; huge gap: one range that also covers the records in between
    assert len(nh.plan_ranges(recs, sel, merge_gap=0)) == 2
    one = nh.plan_ranges(recs, sel, merge_gap=10**9)
    assert len(one) == 1 and len(one[0][2]) == 8


def test_parse_idx_extra_fields_and_errors():
    recs = nh.parse_idx("1:0:d=2026111100:HGT:500 mb:3 hour fcst:ENS=+1\n2:900:d=2026111100:TMP:500 mb:3 hour fcst:ENS=+1\n")
    assert recs[0].extra == "ENS=+1" and recs[0].end == 899
    with pytest.raises(nh.FetchError):
        nh.parse_idx("not an inventory")
    with pytest.raises(nh.FetchError):
        nh.parse_idx("1.2:0:d=2026111100:HGT:500 mb:3 hour fcst:\n")


# ---------------------------------------------------------------------------- download + decode
@pytest.mark.parametrize("merge_gap", [0, 10**9])
def test_fetch_fields_roundtrip(server, merge_gap):
    base, h = server
    grib, idx, arrays = build_file()
    h.files["/f.grib2"], h.files["/f.grib2.idx"] = grib, idx.encode()
    pats = (r"^TMP:2 m above ground:", r"^(?:TMP|RH|HGT):850 mb:", r"^APCP:surface:")
    fields, grid, url = nh.fetch_fields([base + "/f.grib2"], pats, want_grid=True, merge_gap=merge_gap)
    assert set(fields) == {k for k in arrays if re.search("|".join(pats), ":".join(k))}
    for k, v in fields.items():
        np.testing.assert_allclose(v, arrays[k], atol=1e-3)           # 24-bit packing
    assert grid.shape == (5, 7) and grid.lat.ndim == 1 and grid.lat[0] > grid.lat[-1]
    assert grid.lon[0] == pytest.approx(260.0)                         # 0-360 convention kept; Cropper converts
    ranged = [r for p, r in h.log if p == "/f.grib2"]
    assert all(r is not None for r in ranged)                          # never downloads the whole file
    # selected: rec 0 | recs 2-3 (APCP x2) | recs 5-7 (850 mb) -> 3 separate ranges, or 1 when gaps are merged
    assert len(ranged) == (3 if merge_gap == 0 else 1)


def test_fallback_to_second_url_and_error_text(server):
    base, h = server
    grib, idx, arrays = build_file()
    h.files["/ok.grib2"], h.files["/ok.grib2.idx"] = grib, idx.encode()
    fields, _, url = nh.fetch_fields([base + "/missing.grib2", base + "/ok.grib2"], (r"^PRES:surface:",))
    assert url.endswith("/ok.grib2") and len(fields) == 1
    with pytest.raises(nh.FetchError) as e:
        nh.fetch_fields([base + "/missing.grib2"], (r"^PRES:surface:",))
    assert "HTTP 404" in str(e.value) and "missing.grib2.idx" in str(e.value)


def test_no_match_reports_inventory(server):
    base, h = server
    grib, idx, _ = build_file()
    h.files["/f.grib2"], h.files["/f.grib2.idx"] = grib, idx.encode()
    with pytest.raises(nh.NoMatch) as e:
        nh.fetch_fields([base + "/f.grib2"], (r"^HGT:surface:",))
    msg = str(e.value)
    assert "9 records" in msg and "TMP:2 m above ground:3 hour fcst" in msg      # the real inventory is shown


def test_server_without_range_support_is_an_error(server):
    base, h = server
    grib, idx, _ = build_file()
    h.files["/f.grib2"], h.files["/f.grib2.idx"] = grib, idx.encode()
    orig = Handler.do_GET

    def ignore_range(self):                      # a server that answers 200 + the whole file
        self.headers.__delitem__("Range")
        orig(self)
    Handler.do_GET = ignore_range
    try:
        with pytest.raises(nh.FetchError, match="expected 206"):
            nh.fetch_fields([base + "/f.grib2"], (r"^PRES:surface:",))
    finally:
        Handler.do_GET = orig


# ---------------------------------------------------------------------------- file plans
def test_url_templates():
    init = datetime(2026, 11, 11, 12)
    g = ns.plan_gefs(init, 6, 3)[0]
    assert g.urls[0] == ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.20261111/12/atmos/pgrb2ap5/"
                         "gep03.t12z.pgrb2a.0p50.f006")
    assert ns.plan_gefs(init, 6, "avg")[0].urls[0].endswith("geavg.t12z.pgrb2a.0p50.f006")
    assert ns.plan_gfs(init, 6, None)[0].urls[0] == ("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261111/12/atmos/"
                                                     "gfs.t12z.pgrb2.0p25.f006")
    sfc, prs = ns.plan_hrrr(init, 6, None)
    assert sfc.urls[0] == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261111/conus/hrrr.t12z.wrfsfcf06.grib2"
    assert prs.urls[0].endswith("hrrr.t12z.wrfprsf06.grib2") and not prs.has_apcp
    d2, dp = ns.plan_rrfs(init, 3, None)
    assert d2.urls[0] == ("https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.20261111/12/"
                          "rrfs.t12z.2dfld.3km.f003.conus.grib2")
    assert d2.urls[1].startswith("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.20261111/12/")
    assert dp.urls[0].endswith("rrfs.t12z.prslev.3km.f003.conus.grib2")
    e2, ep = ns.plan_rrfs_ens(init, 3, 2)
    assert e2.urls[0] == ("https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfsens.20261111/12/m002/"
                          "rrfs.t12z.m002.2dfldnomads.3km.f003.conus.grib2")
    assert ep.urls[0].endswith("m002/rrfs.t12z.m002.prslevnomads.3km.f003.conus.grib2")


def test_patterns_select_what_blender_needs():
    txt = ["TMP:2 m above ground:3 hour fcst", "RH:2 m above ground:3 hour fcst", "PRES:surface:3 hour fcst",
           "HGT:surface:3 hour fcst", "APCP:surface:0-3 hour acc fcst", "CSNOW:surface:3 hour fcst",
           "TMP:850 mb:3 hour fcst", "HGT:1000 mb:3 hour fcst", "UGRD:850 mb:3 hour fcst", "TMP:2 mb:3 hour fcst",
           "TMP:250 mb:3 hour fcst", "SNMR:850 mb:3 hour fcst"]
    pats = [re.compile(p) for p in (ns.PAT_LEVELS, ns.PAT_SFC, ns.PAT_2M)]
    got = {t for t in txt if any(p.search(t) for p in pats)}
    assert got == {t for t in txt if not t.startswith(("UGRD", "SNMR")) and "2 mb" not in t and "250 mb" not in t}


# ---------------------------------------------------------------------------- QPF step logic
class FakeSource(ns.NoaaSource):
    """NoaaSource whose _fetch returns canned APCP intervals."""

    def __init__(self, table, step=3):
        super().__init__("fake", lambda *a: [], step_hours=step)
        self.table, self.calls = table, 0

    def _fetch(self, init, fxx, member, apcp_only=False):
        self.calls += 1
        return {("APCP", "surface", f"{a}-{b} hour acc fcst"): np.full((2, 2), v, np.float32)
                for (a, b), v in self.table.get(fxx, {}).items()}


INIT = datetime(2026, 11, 11, 12)


def test_qpf_exact_interval_record():
    s = FakeSource({6: {(3, 6): 2.0, (0, 6): 5.0}})
    assert float(s._qpf_step(INIT, 6, None)[0, 0]) == 2.0


def test_qpf_difference_of_run_totals_rrfs_hrrr_style():
    # 1-h bucket + run total at each hour (RRFS/HRRR): 3-h accumulation = total(f06) - total(f03)
    s = FakeSource({3: {(2, 3): 0.4, (0, 3): 1.5}, 6: {(5, 6): 0.1, (0, 6): 4.0}})
    assert float(s._qpf_step(INIT, 6, None)[0, 0]) == pytest.approx(2.5)
    assert float(s._qpf_step(INIT, 3, None)[0, 0]) == pytest.approx(1.5)      # first step: total since init


def test_qpf_bucket_resets_gfs_gefs_style():
    # f09 carries an exact (6, 9) record -> used directly; f12 has only run totals (0, 12) and f09 has (0, 9) ->
    # the 3-h amount is their difference
    s = FakeSource({6: {(0, 6): 3.0}, 9: {(6, 9): 1.25, (0, 9): 4.25}, 12: {(0, 12): 6.0, (6, 12): 3.0}})
    assert float(s._qpf_step(INIT, 9, None)[0, 0]) == pytest.approx(1.25)
    assert float(s._qpf_step(INIT, 12, None)[0, 0]) == pytest.approx(1.75)   # 6.0 - 4.25


def test_qpf_missing_raises_with_inventory():
    s = FakeSource({6: {(0, 6): 3.0}, 3: {}})
    with pytest.raises(nh.FetchError) as e:
        s._qpf_step(INIT, 6, None)
    assert "no usable APCP" in str(e.value)


# ---------------------------------------------------------------------------- full NoaaSource step (mocked I/O)
def test_step_fields_from_inventory_fields(monkeypatch):
    shape = (4, 6)
    lat = np.linspace(60, 20, shape[0])
    lon = np.linspace(230, 300, shape[1])
    grid = nh.GridInfo(lat, lon, shape, "regular_ll")
    base = lambda v: np.full(shape, v, np.float32)                      # noqa: E731
    fields = {("TMP", "2 m above ground", "6 hour fcst"): base(272.15), ("RH", "2 m above ground", "6 hour fcst"): base(90),
              ("PRES", "surface", "6 hour fcst"): base(95000), ("HGT", "surface", "6 hour fcst"): base(300),
              ("APCP", "surface", "3-6 hour acc fcst"): base(2.0), ("CSNOW", "surface", "6 hour fcst"): base(1),
              ("CRAIN", "surface", "6 hour fcst"): base(0)}
    for p in ns.LEVELS:
        fields[("TMP", f"{p} mb", "6 hour fcst")] = base(270)
        fields[("RH", f"{p} mb", "6 hour fcst")] = base(80)
        fields[("HGT", f"{p} mb", "6 hour fcst")] = base(1000 + p)

    calls = []

    def fake_fetch(urls, patterns, want_grid=False, **kw):
        calls.append(urls[0])
        return fields, grid, urls[0]
    monkeypatch.setattr(nh, "fetch_fields", fake_fetch)
    src = ns.gfs()
    out = src.load_steps(INIT, datetime(2026, 11, 11, 15), datetime(2026, 11, 11, 18), None)
    assert len(out) == 1
    st = out[0]
    assert st.t2m_c[0, 0] == pytest.approx(-1.0, abs=1e-4) and st.psfc_hpa[0, 0] == pytest.approx(950.0)
    assert float(st.qpf_mm[0, 0]) == 2.0 and sorted(st.levels) == sorted(ns.LEVELS)
    assert st.levels[850]["t"][0, 0] == pytest.approx(-3.15, abs=1e-4) and st.z_sfc_m[0, 0] == 300
    assert set(st.native_type_frac) == {"snow", "rain"}
    g = src.grid()
    assert g.lon.max() < 0 and g.lat.ndim == 1                           # cropped + converted to -180..180


def test_step_fields_missing_levels_is_explicit(monkeypatch):
    grid = nh.GridInfo(np.linspace(60, 20, 3), np.linspace(230, 300, 3), (3, 3), "regular_ll")
    f = {("TMP", "2 m above ground", "6 hour fcst"): np.zeros((3, 3), np.float32),
         ("PRES", "surface", "6 hour fcst"): np.full((3, 3), 1e5, np.float32),
         ("APCP", "surface", "0-6 hour acc fcst"): np.zeros((3, 3), np.float32)}
    monkeypatch.setattr(nh, "fetch_fields", lambda urls, patterns, want_grid=False, **k: (f, grid, urls[0]))
    with pytest.raises(nh.FetchError) as e:
        ns.gfs()._step_fields(INIT, datetime(2026, 11, 11, 18), None)
    assert "pressure-level" in str(e.value)


# ---------------------------------------------------------------------------- ECMWF open-data decode path
def _ifs_msg(short, tol, level, step, vals):
    gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ny, nx = vals.shape
    for k, v in (("Ni", nx), ("Nj", ny), ("latitudeOfFirstGridPointInDegrees", 60.0),
                 ("longitudeOfFirstGridPointInDegrees", 230.0), ("latitudeOfLastGridPointInDegrees", 60.0 - (ny - 1) * 10),
                 ("longitudeOfLastGridPointInDegrees", 230.0 + (nx - 1) * 10), ("iDirectionIncrementInDegrees", 10.0),
                 ("jDirectionIncrementInDegrees", 10.0), ("bitsPerValue", 24)):
        ec.codes_set(gid, k, v)
    ec.codes_set(gid, "shortName", short)
    if tol == "isobaricInhPa":
        ec.codes_set(gid, "typeOfLevel", tol)
        ec.codes_set(gid, "level", level)
    ec.codes_set(gid, "step", step)
    ec.codes_set_values(gid, vals.astype(np.float64).ravel())
    buf = bytes(ec.codes_get_message(gid))
    ec.codes_release(gid)
    return buf


def test_ifs_decode_and_tp_differencing(tmp_path, monkeypatch):
    shape = (4, 6)
    c = lambda v: np.full(shape, v, np.float32)                       # noqa: E731
    sfc, pl = b"", b""
    for step, tp in ((0, 0.0), (6, 0.002), (12, 0.005)):              # metres, accumulated from 0
        sfc += _ifs_msg("2t", "heightAboveGround", 2, step, c(270.0))
        sfc += _ifs_msg("sp", "surface", 0, step, c(95000.0))
        if step:
            sfc += _ifs_msg("tp", "surface", 0, step, c(tp))
        for p in ns.LEVELS:
            pl += _ifs_msg("t", "isobaricInhPa", p, step, c(268.0))
            pl += _ifs_msg("r", "isobaricInhPa", p, step, c(75.0))
            pl += _ifs_msg("gh", "isobaricInhPa", p, step, c(1000.0 + p))
    (tmp_path / "sfc.grib2").write_bytes(sfc)
    (tmp_path / "pl.grib2").write_bytes(pl)
    src = ns.IfsSource("ifs_ens", ensemble=True, save_dir=tmp_path)
    monkeypatch.setattr(src, "_retrieve", lambda init, steps, member, levtype:
                        tmp_path / ("sfc.grib2" if levtype == "sfc" else "pl.grib2"))
    out = src.load_steps(INIT, datetime(2026, 11, 11, 12), datetime(2026, 11, 12, 0), 3)
    assert [o.valid.hour for o in out] == [18, 0]
    assert float(out[0].qpf_mm[0, 0]) == pytest.approx(2.0, abs=1e-3)   # (0.002 - 0) m
    assert float(out[1].qpf_mm[0, 0]) == pytest.approx(3.0, abs=1e-3)   # (0.005 - 0.002) m
    assert out[0].t2m_c[0, 0] == pytest.approx(-3.15, abs=1e-3) and sorted(out[0].levels) == sorted(ns.LEVELS)
    assert out[0].levels[850]["z"][0, 0] == pytest.approx(1850, abs=0.1)
    assert src.grid().lon.max() < 0


def test_lambert_grid_decodes_to_2d_latlon_south_first():
    gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ec.codes_set(gid, "gridType", "lambert")                       # GRIB2 template 3.30 like RRFS/HRRR/RTMA
    for k, v in (("Nx", 8), ("Ny", 6), ("latitudeOfFirstGridPointInDegrees", 20.19),
                 ("longitudeOfFirstGridPointInDegrees", 238.446), ("LoVInDegrees", 265.0), ("Latin1InDegrees", 25.0),
                 ("Latin2InDegrees", 25.0), ("DxInMetres", 2539.703), ("DyInMetres", 2539.703), ("scanningMode", 64),
                 ("bitsPerValue", 24)):
        ec.codes_set(gid, k, v)
    ec.codes_set_values(gid, np.arange(48, dtype=float))
    g = nh.grid_of(ec, gid)
    v = nh.values_of(ec, gid)
    ec.codes_release(gid)
    assert g.grid_type == "lambert" and g.lat.shape == (6, 8) and v.shape == (6, 8)
    assert g.lat[0, 0] == pytest.approx(20.19, abs=1e-3) and g.lat[-1, 0] > g.lat[0, 0]
    assert v[0, 1] == 1 and v[1, 0] == 8                           # row-major, row 0 = first scanned row
