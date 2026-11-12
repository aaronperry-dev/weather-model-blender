import numpy as np
import pytest

from pwpf_blender import binormal, mode, slr, thermo
from pwpf_blender.constants import (SNOW_LEVEL_COLD, SNOW_LEVEL_WARM, TYPE_FZRA, TYPE_RAIN, TYPE_SLEET, TYPE_SNOW)
from pwpf_blender.downscale import PrismDownscaler, interp_months
from pwpf_blender.grid import GridSpec, LambertConformal
from pwpf_blender.ptype import decision_tree, dominant_type, native_type
from pwpf_blender.regrid import Regridder


# ---------------------------------------------------------------- grid
def test_ndfd_grid_matches_published_corners():
    """GMOS/NDFD docs: 2145x1597 grid, LL 20.192N/-121.554, UR 54.3728N/-59.0421 (dx 2539.703 m)."""
    g = GridSpec.ndfd_2p5km()
    x = g.x0 + g.dx * 2144
    y = g.y0 + g.dx * 1596
    lon, lat = g.proj.inverse(x, y)
    assert abs(lat - 54.3728) < 0.02
    assert abs(lon - (-59.0421)) < 0.02


def test_grid_roundtrip_and_shape():
    g = GridSpec.ndfd_2p5km(stride=8)
    lon, lat = g.lonlat()
    assert lon.shape == g.shape == (173, 269)
    r, c = g.lonlat_to_rowcol(lon, lat)
    assert np.allclose(r, np.arange(g.ny)[:, None], atol=1e-3)
    assert np.allclose(c, np.arange(g.nx)[None, :], atol=1e-3)
    p = LambertConformal()
    lo, la = p.inverse(*p.forward(-100.0, 40.0))
    assert abs(lo + 100) < 1e-6 and abs(la - 40) < 1e-6


# ---------------------------------------------------------------- regrid
def test_regular_regrid_recovers_linear_field():
    lat = np.arange(20, 56, 0.5)
    lon = np.arange(-130, -60, 0.5)
    LON, LAT = np.meshgrid(lon, lat)
    field = 2 * LAT + 0.5 * LON
    g = GridSpec.ndfd_2p5km(stride=16)
    dlon, dlat = g.lonlat()
    rg = Regridder.build(lat, lon, dlat, dlon)
    out = rg(field)
    assert np.nanmax(np.abs(out - (2 * dlat + 0.5 * dlon))) < 1e-3


def test_kdtree_regrid_close_to_truth_on_smooth_field():
    g_src = GridSpec.ndfd_2p5km(stride=12)           # irregular (Lambert) source
    slon, slat = g_src.lonlat()
    g_dst = GridSpec.ndfd_2p5km(stride=20)
    dlon, dlat = g_dst.lonlat()
    rg = Regridder.build(slat, slon, dlat, dlon)
    out = rg(np.sin(np.radians(slat)) * 10)
    truth = np.sin(np.radians(dlat)) * 10
    ok = np.isfinite(out)
    assert ok.mean() > 0.95
    assert np.abs(out[ok] - truth[ok]).max() < 0.1


# ---------------------------------------------------------------- thermo
def test_wetbulb_known_values():
    assert abs(float(thermo.wetbulb_c(20.0, 50.0)) - 13.7) < 0.2      # Stull (2011) example
    assert abs(float(thermo.wetbulb_c(0.0, 100.0))) < 0.5            # saturated: Tw ~ T
    assert thermo.wetbulb_c(2.0, 60.0) < 2.0


def _levels(shape, t850, t925=None, t700=None, rh=95.0):
    zs = {1000: 100.0, 925: 760.0, 850: 1460.0, 700: 3010.0, 500: 5570.0}
    ts = {1000: 0, 925: t925 if t925 is not None else t850 + 3, 850: t850, 700: t700 if t700 is not None else t850 - 6,
          500: -25}
    return {p: {"t": np.full(shape, ts[p], np.float32), "rh": np.full(shape, rh, np.float32),
                "z": np.full(shape, zs[p], np.float32)} for p in zs}


def test_snow_level_crossing_between_levels():
    sh = (2, 2)
    lv = _levels(sh, t850=-4, t925=1.0)
    lv[1000]["t"][:] = 4.0
    sl = thermo.snow_level_height(lv, np.full(sh, 4.0), np.full(sh, 95.0), np.full(sh, 1010.0),
                                  np.full(sh, 50.0))
    assert np.all((sl > 760.0 - 400) & (sl < 1460.0))      # between 850 and 925 / just below 925


def test_snow_level_cold_and_warm_columns():
    sh = (1, 1)
    cold = thermo.snow_level_height(_levels(sh, -10), np.full(sh, -8.0), np.full(sh, 90.0),
                                    np.full(sh, 1000.0), np.full(sh, 100.0))
    assert cold[0, 0] == SNOW_LEVEL_COLD
    lv = _levels(sh, 8, t925=10, t700=3)
    lv[500]["t"][:] = 5.0
    warm = thermo.snow_level_height(lv, np.full(sh, 12.0), np.full(sh, 80.0), np.full(sh, 1000.0), np.full(sh, 100.0))
    assert warm[0, 0] == SNOW_LEVEL_WARM


def test_snow_level_underground_levels_ignored():
    sh = (1, 1)
    lv = _levels(sh, t850=-6, t925=-4)
    lv[1000]["t"][:] = 15.0            # nonsense underground value must not matter
    sl = thermo.snow_level_height(lv, np.full(sh, -2.0), np.full(sh, 95.0), np.full(sh, 850.0), np.full(sh, 1450.0))
    assert sl[0, 0] == SNOW_LEVEL_COLD


def test_surface_height_from_levels():
    sh = (1, 3)
    lv = _levels(sh, -5)
    z = thermo.surface_height_from_levels(np.array([[1000.0, 925.0, 850.0]]), lv)
    assert np.allclose(z[0], [100.0, 760.0, 1460.0], atol=1)


# ---------------------------------------------------------------- ptype
def test_decision_tree_cases():
    t2m = np.array([-3.0, 3.0, -2.0, -2.0, 1.0])
    aloft = np.array([-6.0, -5.0, 1.0, 4.0, np.nan])
    out = decision_tree(t2m, aloft)
    assert list(out) == [TYPE_SNOW, TYPE_RAIN, TYPE_SLEET, TYPE_FZRA, TYPE_SNOW]


def test_dominant_type_and_ties():
    counts = np.zeros((5, 1, 3), np.int16)
    counts[TYPE_RAIN, 0, 0] = 3
    counts[TYPE_SNOW, 0, 0] = 1            # rain dominates
    counts[TYPE_RAIN, 0, 1] = 2
    counts[TYPE_SNOW, 0, 1] = 2            # tie -> snow beats rain
    counts[TYPE_SNOW, 0, 2] = 1
    counts[TYPE_FZRA, 0, 2] = 1            # tie -> freezing rain beats snow
    assert list(dominant_type(counts)[0]) == [TYPE_RAIN, TYPE_SNOW, TYPE_FZRA]


def test_native_type_falls_back_to_tree():
    fb = np.array([[TYPE_RAIN, TYPE_SNOW]], np.int8)
    frac = {"snow": np.array([[0.9, 0.0]]), "rain": np.array([[0.1, 0.0]])}
    assert list(native_type(frac, fb)[0]) == [TYPE_SNOW, TYPE_SNOW]


# ---------------------------------------------------------------- SLR
def test_kuchera_values():
    assert np.isclose(slr.kuchera_slr(271.16), 12.0)
    assert np.isclose(slr.kuchera_slr(265.16), 18.0)
    assert np.isclose(slr.kuchera_slr(275.16), 4.0)


def test_slr_step_clipping_and_adjust():
    sh = (1, 2)
    lv = _levels(sh, -10)
    out = slr.slr_step(np.full(sh, -12.0, np.float32), lv, np.full(sh, 1000.0), 20, {"method": "kuchera", "adjust": 0.5})
    assert np.all(out <= 40 * 0.5 + 1e-6)


# ---------------------------------------------------------------- binormal
def test_binormal_variance_formula_matches_numerical_integration():
    m, s1, s2 = 4.0, 1.5, 4.0
    x = np.linspace(-30, 60, 400001)
    s = np.where(x < m, s1, s2)
    f = np.sqrt(2 / np.pi) / (s1 + s2) * np.exp(-(x - m) ** 2 / (2 * s ** 2))
    dx = x[1] - x[0]
    assert abs(f.sum() * dx - 1) < 1e-4
    mean_n = (x * f).sum() * dx
    var_n = ((x - mean_n) ** 2 * f).sum() * dx
    mean_f, var_f = binormal.moments(m, s1, s2)
    assert abs(mean_n - mean_f) < 1e-3 and abs(var_n - var_f) < 1e-2


def test_binormal_fit_matches_variance_and_rank():
    rng = np.random.default_rng(1)
    members = rng.gamma(2.0, 2.0, size=(60, 4, 5)).astype(np.float32)
    md = np.full((4, 5), 3.0, np.float32)
    f = binormal.fit(members, md)
    # variance of the fitted binormal equals ensemble(+mode) variance
    allv = np.concatenate([members, md[None]], axis=0)
    _, var = binormal.moments(f["mode"], f["s_lo"], f["s_hi"])
    assert np.allclose(var, allv.var(axis=0), rtol=1e-3)
    # P(X<mode) equals the rank fraction
    p = f["s_lo"] / (f["s_lo"] + f["s_hi"])
    rank = ((members < md).sum(0) + 0.5 * (np.abs(members - md) <= 0.005).sum(0)) / 60
    assert np.allclose(p, np.clip(rank, 1 / 61, 60 / 61), atol=1e-4)


def test_exceedance_percentile_consistency():
    m, s1, s2 = 6.0, 1.0, 3.0
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        x = binormal.percentile(m, s1, s2, q)
        assert abs(1 - binormal.exceedance(m, s1, s2, x) - q) < 1e-6
    # probability at the mode equals s2/(s1+s2)
    assert abs(binormal.exceedance(m, s1, s2, m) - s2 / (s1 + s2)) < 1e-9
    assert binormal.percentile(0.0, 0.01, 0.05, 0.05) >= 0


def test_percentile_monotone_and_exceedance_decreasing():
    t = np.linspace(0, 30, 50)
    pr = binormal.exceedance(5.0, 2.0, 4.0, t)
    assert np.all(np.diff(pr) <= 1e-12)
    qs = binormal.percentile(5.0, 2.0, 4.0, np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
    assert np.all(np.diff(qs) >= 0)


# ---------------------------------------------------------------- mode / downscale
def test_pmm_keeps_member_amounts_and_mean_pattern():
    rng = np.random.default_rng(0)
    base = np.zeros((20, 30))
    base[8:12, 10:20] = 6.0
    members = np.stack([np.roll(base, rng.integers(-3, 4), axis=1) * rng.uniform(0.6, 1.6) for _ in range(30)]).astype(np.float32)
    pm = mode.probability_matched_mean(members)
    mean = members.mean(0)
    assert pm.max() > mean.max() * 1.05                                 # keeps peak amounts
    assert abs(pm.max() - members.max()) < members.max() * 0.4
    assert np.corrcoef(pm.ravel(), mean.ravel())[0, 1] > 0.95           # same pattern


def test_prism_ratio_mean_one_and_enhances_wet_pixels():
    ny, nx = 60, 80
    clim = np.ones((12, ny, nx), np.float32) * 50
    clim[:, 20:40, 30:34] = 200      # narrow wet ridge
    d = PrismDownscaler(clim, grid_dx_km=2.54)
    r = d.ratio(15, src_dx_km=25.0)
    assert r[30, 31] > 1.5 and r[30, 40] < 1.0
    assert abs(r.mean() - 1.0) < 0.1
    assert interp_months(clim, 1).shape == (ny, nx)
    assert PrismDownscaler(None, 2.5).ratio(10, 25) is None
