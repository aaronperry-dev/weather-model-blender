"""MSC Datamart loaders (RDPS / GDPS)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pwpf_blender import noaa_http as nh
from pwpf_blender.config import load_config
from pwpf_blender.pipeline import expand_members
from pwpf_blender.sources import make_source
from pwpf_blender.sources import msc_sources as ms

CFG = Path(__file__).resolve().parents[1] / "config" / "pwpf.yaml"
INIT = datetime(2026, 11, 12, 0)
SHAPE = (4, 5)


def test_urls_match_the_datamart_layout():
    r, g = make_source("rdps"), make_source("gdps")
    assert r.url(INIT, 24, "AirTemp", "AGL-2m") == (
        "https://dd.weather.gc.ca/20261112/WXO-DD/model_rdps/10km/00/024/"
        "20261112T00Z_MSC_RDPS_AirTemp_AGL-2m_RLatLon0.09_PT024H.grib2")
    assert g.url(datetime(2026, 11, 12, 12), 6, "Precip-Accum3h", "Sfc") == (
        "https://dd.weather.gc.ca/20261112/WXO-DD/model_gdps/15km/12/006/"
        "20261112T12Z_MSC_GDPS_Precip-Accum3h_Sfc_LatLon0.15_PT006H.grib2")
    assert (r.src_dx_km, g.src_dx_km, r.step_hours) == (10.0, 15.0, 3)


def test_unit_helpers():
    assert to_c(np.array([263.15])) == pytest.approx(-10)
    assert ms.to_celsius(np.array([-5.0, 3.0]))[0] == -5.0
    assert ms.to_hpa(np.array([101325.0]))[0] == pytest.approx(1013.25)
    assert ms.to_hpa(np.array([1013.0]))[0] == 1013.0
    assert ms.to_percent(np.array([0.2, 0.9]))[1] == pytest.approx(90)
    assert ms.to_percent(np.array([20.0, 99.0]))[1] == 99.0
    assert ms.z_scale({500: np.full(3, 5500.0)}) == 1.0                  # gpm
    assert ms.z_scale({500: np.full(3, 550.0), 850: np.full(3, 146.0)}) == 10.0      # decametres
    assert ms.z_scale({1000: np.full(3, 5.0)}) == 1.0                    # low levels alone cannot decide


def to_c(a):
    return ms.to_celsius(a)


class _Fake:
    """Stands in for _read: returns a field depending on the file asked for and records the requests."""

    def __init__(self, missing=(), dam=False):
        self.urls, self.missing, self.dam = [], missing, dam

    def __call__(self, url):
        self.urls.append(url)
        if any(m in url for m in self.missing):
            raise nh.HttpStatusError(f"{url}: HTTP 404", 404)
        v = lambda x: np.full(SHAPE, x, np.float32)                       # noqa: E731
        if "AirTemp_AGL-2m" in url:
            return v(271.0)
        if "AirTemp_IsbL" in url:
            return v(260.0)
        if "RelativeHumidity" in url:
            return v(0.8)
        if "Pressure_Sfc" in url:
            return v(95000.0)
        if "GeopotentialHeight" in url:
            p = int(url.split("IsbL-")[1][:4])
            z = {1000: 100, 925: 760, 850: 1460, 700: 3010, 500: 5570}[p]
            return v(z / 10 if self.dam else z)
        if "Precip-Accum3h" in url:
            return v(1.5)
        raise AssertionError(url)


def _src(monkeypatch, fake):
    src = make_source("rdps")
    monkeypatch.setattr(src, "_read", fake)
    return src


def test_step_assembles_standard_fields(monkeypatch):
    fake = _Fake()
    st = _src(monkeypatch, fake).load_steps(INIT, INIT, datetime(2026, 11, 12, 6), None)
    assert [s.valid for s in st] == [datetime(2026, 11, 12, 3), datetime(2026, 11, 12, 6)]
    s = st[1]
    assert s.t2m_c[0, 0] == pytest.approx(-2.15, abs=1e-3) and s.psfc_hpa[0, 0] == pytest.approx(950)
    assert s.rh2m[0, 0] == pytest.approx(80) and s.qpf_mm[0, 0] == 1.5 and s.dt_h == 3.0
    assert sorted(s.levels) == [500, 700, 850, 925, 1000] and s.levels[850]["z"][0, 0] == 1460
    assert s.levels[850]["t"][0, 0] == pytest.approx(-13.15, abs=1e-3) and s.levels[850]["rh"][0, 0] == pytest.approx(80)
    assert len(fake.urls) == 2 * 19                                       # 19 small single-message files per step
    assert not any("PT006H" in u and "PT003H" in u for u in fake.urls)


def test_decametre_heights_are_converted(monkeypatch):
    st = _src(monkeypatch, _Fake(dam=True)).load_steps(INIT, INIT, datetime(2026, 11, 12, 3), None)[0]
    assert st.levels[500]["z"][0, 0] == pytest.approx(5570) and st.levels[1000]["z"][0, 0] == pytest.approx(100)


def test_missing_level_files_are_skipped_but_not_all(monkeypatch):
    st = _src(monkeypatch, _Fake(missing=("IsbL-0925",))).load_steps(INIT, INIT, datetime(2026, 11, 12, 3), None)[0]
    assert sorted(st.levels) == [500, 700, 850, 1000]
    with pytest.raises(nh.FetchError, match="no pressure-level"):
        _src(monkeypatch, _Fake(missing=("IsbL",))).load_steps(INIT, INIT, datetime(2026, 11, 12, 3), None)


def test_missing_precip_is_an_error(monkeypatch):
    with pytest.raises(nh.HttpStatusError):
        _src(monkeypatch, _Fake(missing=("Precip-Accum3h",))).load_steps(INIT, INIT, datetime(2026, 11, 12, 3), None)


def test_available_probe(monkeypatch):
    src = make_source("rdps")

    class R:
        def __init__(self, c):
            self.status_code = c

    seen = []
    monkeypatch.setattr(nh, "_request", lambda url, headers=None, **k: (seen.append((url, headers)) or R(206)))
    assert src.available(INIT, 72, None) and seen[0][1] == {"Range": "bytes=0-0"} and "PT072H" in seen[0][0]
    monkeypatch.setattr(nh, "_request", lambda url, headers=None, **k: R(404))
    assert not src.available(INIT, 72, None)
    monkeypatch.setattr(nh, "_request", lambda *a, **k: (_ for _ in ()).throw(nh.FetchError("down")))
    assert not src.available(INIT, 72, None)


def test_default_config_includes_rdps_and_not_gdps():
    cfg = load_config(CFG, "lite", {})
    names = {m.group for m in expand_members(cfg, datetime(2026, 11, 12, 0))}
    assert "rdps" in names and "gdps" not in names
    assert make_source("rdps").name == "rdps" and make_source("gdps").name == "gdps"


def test_datamart_host_is_rate_limited_politely():
    pol = nh.policy_for("https://dd.weather.gc.ca/20261112/WXO-DD/model_rdps/10km/00/003/x.grib2")
    assert pol.rate <= 2.0 and pol.max_conc <= 2
