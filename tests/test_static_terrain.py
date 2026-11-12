"""Terrain acquisition logic (network mocked): RTMA first, RRFS fallback, explicit errors, orientation check."""
import numpy as np
import pytest

from pwpf_blender import noaa_http as nh
from pwpf_blender import static_data as sd
from pwpf_blender.grid import GridSpec

FULL = GridSpec.ndfd_2p5km(1)


def _terrain():
    lon, lat = FULL.lonlat()
    return (1500 + 1000 * np.sin(np.radians(lat * 6)) * np.cos(np.radians(lon * 4))).astype(np.float32), lon, lat


def test_rtma_terrain_flipped_rows_are_corrected(monkeypatch):
    z, lon, lat = _terrain()
    g = nh.GridInfo(lat[::-1], lon[::-1] % 360, z.shape, "lambert")        # a file stored north-to-south, 0-360 lon
    monkeypatch.setattr(nh, "fetch_fields", lambda urls, pats, want_grid=False, **k:
                        ({("HGT", "surface", "anl"): z[::-1].copy()}, g, urls[0]))
    out = sd.terrain_from_rtma(FULL)
    np.testing.assert_allclose(out, z)


def test_rtma_expanded_grid_is_cropped_to_ndfd_window(monkeypatch):
    """Regression for the 2026-11 CI failure: RTMA *_wexp is 1597 x 2345 (expanded NDFD grid)."""
    from dataclasses import replace
    z, lon, lat = _terrain()
    r0, c0 = 110, 100                                                      # asymmetric margins like the real file
    big = replace(FULL, nx=FULL.nx + 2 * c0, ny=FULL.ny + 2 * r0, x0=FULL.x0 - c0 * FULL.dx, y0=FULL.y0 - r0 * FULL.dx)
    blon, blat = big.lonlat()
    bz = (1500 + 1000 * np.sin(np.radians(blat * 6)) * np.cos(np.radians(blon * 4))).astype(np.float32)
    assert bz.shape == (1597, 2345)
    g = nh.GridInfo(blat, blon % 360, bz.shape, "lambert")
    monkeypatch.setattr(nh, "fetch_fields", lambda urls, pats, want_grid=False, **k:
                        ({("HGT", "surface", "anl"): bz}, g, urls[0]))
    np.testing.assert_allclose(sd.terrain_from_rtma(FULL), z, atol=1e-3)


def test_misaligned_grid_is_reported_not_retried(monkeypatch):
    z, lon, lat = _terrain()
    calls = []
    g = nh.GridInfo(lat + 0.5, lon, z.shape, "lambert")                    # half a degree off
    monkeypatch.setattr(nh, "fetch_fields", lambda urls, pats, want_grid=False, **k:
                        (calls.append(1) or {("HGT", "surface", "anl"): z}, g, urls[0]))
    with pytest.raises(nh.FetchError, match="cannot be mapped"):
        sd.terrain_from_rtma(FULL)
    assert len(calls) == 1


def test_rtma_without_terrain_falls_back_to_rrfs(monkeypatch):
    z, lon, lat = _terrain()
    rl = np.arange(15.0, 60.0, 0.05)
    ro = np.arange(-135.0, -55.0, 0.05)
    la, lo = np.meshgrid(rl, ro, indexing="ij")
    src = (1500 + 1000 * np.sin(np.radians(la * 6)) * np.cos(np.radians(lo * 4))).astype(np.float32)

    def fake(urls, pats, want_grid=False, **k):
        if "rtma" in urls[0]:
            raise nh.NoMatch(f"{urls[0]}: inventory has 20 records but none match", [])
        return {("HGT", "surface", "anl"): src}, nh.GridInfo(rl, ro % 360, src.shape, "regular_ll"), urls[0]
    monkeypatch.setattr(nh, "fetch_fields", fake)
    out = sd.terrain_auto(FULL)
    assert out.shape == FULL.shape and np.isfinite(out).all()
    inside = np.isfinite(z)
    assert np.abs(out - z)[inside].mean() < 25                              # interpolated smooth analytic field


def test_both_sources_failing_reports_both(monkeypatch):
    def fake(urls, pats, want_grid=False, **k):
        raise nh.FetchError(f"{urls[0]}.idx: HTTP 404")
    monkeypatch.setattr(nh, "fetch_fields", fake)
    with pytest.raises(nh.FetchError) as e:
        sd.terrain_auto(FULL)
    msg = str(e.value)
    assert "RTMA" in msg and "RRFS" in msg and "HTTP 404" in msg and "--terrain-npy" in msg


def test_implausible_terrain_rejected():
    bad = np.full(FULL.shape, 99999.0, np.float32)
    with pytest.raises(ValueError):
        sd._check_terrain(bad, FULL)
    with pytest.raises(ValueError):
        sd._check_terrain(np.zeros((3, 3), np.float32), FULL)


def test_fallback_accepts_partial_domain_coverage_like_rrfs():
    """RRFS covers ~91 % of the NDFD grid (CI log 2026-11-12); the rest is filled with 0 m."""
    z = np.full(FULL.shape, 500.0, np.float32)
    z[:, int(FULL.nx * 0.91):] = np.nan
    with pytest.raises(ValueError):
        sd._check_terrain(z, FULL)                                          # default 95 % threshold rejects it
    out = sd._check_terrain(z, FULL, min_finite=0.80)
    assert np.isfinite(out).all() and out[0, -1] == 0.0 and out[0, 0] == 500.0
