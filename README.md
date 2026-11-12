# pwpf-blender — an open, GitHub-Pages-ready re-implementation of WPC's 2.5 km PWPF snowfall guidance

> **Unofficial.** Built from WPC's *public* documentation of its Probabilistic Winter Precipitation
> Forecast (PWPF). It is not an NWS/WPC product and must not be used for warnings or life-safety decisions.
> Always use official forecasts from [weather.gov](https://www.weather.gov) and [wpc.ncep.noaa.gov](https://www.wpc.ncep.noaa.gov).

GitHub Pages only serves static files, so the project splits into two parts:

```
GitHub Actions (cron, 2x/day)                      GitHub Pages (static)
  Python pipeline  ──►  data PNGs + manifest.json ──►  web/ viewer (Leaflet + binormal maths in JS)
  (this package)                                      one small image per period serves every
                                                      threshold / percentile product
```

## What the real PWPF does (and where this repo stands)

From WPC's "About PWPF Products" page and the 2.5 km file README:

| PWPF step (public documentation) | Here | Status |
|---|---|---|
| ~60 member multi-model ensemble: 10 GEFS (06Z), 25 ECMWF ENS (00Z), 10 CMC ENS, SREF NMMB, 2 HRRR, 2 CMC RDPS, 2 NAM-nest, 4 HiResW ARW, FV3-LAM, GFS, ECMWF HRES, GEFS mean | `config/pwpf.yaml` groups with the same init-time logic (`age_h`, `lags_h`), seeded random member picks, and PWPF-style fall-back to time-lagged GFS/GEFS beyond a CAM's range | **Implemented** except CMC GEPS/RDPS and SREF (no loader yet) |
| PRISM-based QPF downscaling of every member | `downscale.py`: ratio of PRISM monthly normals to the same field smoothed to the member's resolution | **Implemented** (ratio clip limits are our assumption) |
| Snow level = height of the 0.5 °C wet-bulb surface, top-down search, intersected with RTMA terrain; terrain above it ⇒ snow | `thermo.py` (Stull wet-bulb, top-down crossing) + RTMA 2.5 km terrain | **Implemented** |
| Precip type: NCEP models use dominant-type (Manikin 2005); non-NCEP use a simple decision tree on T2m and T925/850/700 | `ptype.py`: dominant type from native categorical fields; decision tree otherwise | **Implemented** (tree thresholds / tie order are our assumptions) |
| Snowfall = QPF × ptype × SLR; SLR = NBM SLR × an adjustment factor | `slr.py`: Kuchera-based blend (legacy-style average with climatology and 11:1) × `slr.adjust` | **Approximated** — NBM SLR and WPC's factor aren't reproducible; calibrate `slr.adjust` (below) |
| WWD (human-edited) 24-h deterministic forecast = **mode** of the distribution | `mode.py`: probability-matched mean blended with the mean (default), or `external` (supply NDFD/WWD-like grid) | **Stand-in** — the human forecast isn't machine-reproducible |
| Binormal PDF per grid point (Toth & Szentimrey 1990), method of moments: mode = WWD forecast, skewness from the forecast's rank in the ensemble, variance matched to the ensemble **plus the forecast as an extra member** | `binormal.py` — closed form: `s = sqrt(Var / [(1-2/π)(1-2p)² + p(1-p)])`, `σ_lo = p·s`, `σ_hi = (1-p)·s` | **Implemented**; verified numerically (tests) |
| Products: exceedance probabilities and percentile accumulations for 24 h; 48 h = sum of adjacent 24 h; 12Z–12Z periods | `windows` in config: Day 1/2/3, Days 1-2, Days 2-3 | **Implemented** |
| 2.5 km NDFD CONUS grid (2145×1377, spherical Lambert, 2539.703 m) | `grid.py`, checked against published corner coordinates | **Implemented** |

So the *distributional machinery* follows the documentation closely; the pieces that WPC does not publish
(human-edited mode, NBM SLR, SLR adjustment factor, exact thresholds) are explicit, tunable stand-ins.
`validate` exists to measure and close the remaining gap against WPC's own 2.5 km files.

## Quick start (no network: synthetic ensemble)

```bash
pip install -r requirements.txt pytest
pytest -q                                              # 34 tests incl. Python↔JavaScript parity
python -m pwpf_blender run --synthetic --out site/data --save-npz
cp web/* site/ && python -m http.server -d site 8000   # open http://localhost:8000
PYTHONPATH=. python scripts/preview.py site/fields_*.npz preview.png d1   # needs matplotlib
```

## Deploy on GitHub Pages

1. Push this repo to GitHub. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. **Actions → "Build PWPF-style snowfall guidance" → Run workflow** with profile `synthetic` to confirm the site deploys.
3. Run again with profile `lite` (5 km, ~35 members) then `full` (2.5 km, ~60 members).
   Set the repository variable `PWPF_PROFILE` to choose what the twice-daily schedule runs (default `lite`).

The workflow restores/creates `static/` (RTMA terrain + PRISM normals, cached), runs the pipeline, uploads the
site to Pages and keeps the full-precision `fields_*.npz` as a 3-day artifact for validation.

### Live-data caveat (read this)

The loaders in `pwpf_blender/sources/herbie_sources.py` (GEFS, GFS, HRRR, NAM nest, HiResW via
[Herbie](https://github.com/blaylockbk/Herbie); ECMWF open data via `ecmwf-opendata`) and
`static_data.py` (RTMA terrain, PRISM download) were written from the feed documentation but **could not be run against the
live servers in the environment where this repo was authored** (outbound access was blocked). Everything downstream
of them — wet-bulb snow level, ptype, SLR, downscaling, binormal fit, export, viewer — is covered by tests on a
synthetic ensemble. Expect to adjust a variable name or regex on first contact; start with `--profile lite`
and look at the Actions log (a member that fails is dropped with a warning, the run aborts below `min_members`).

PRISM changed its web services in Oct 2025; if the default URL fails, download the 12 monthly 4 km `ppt` normals
from <https://prism.oregonstate.edu/normals/> and run `python -m pwpf_blender prepare-static --prism-dir <folder>`.

## Getting "very close" to the real PWPF — the calibration loop

1. Run with `--save-npz` for a 00Z or 12Z cycle.
2. Download WPC's files for the same cycle from `ftp.wpc.ncep.noaa.gov/pwpf/conus_2.5km/` (percentiles
   `2.5kmprcntil_24hsnow_<pp>pt_<yyyymmddcc>f<hhh>.grb`, probabilities `psnow24_ge*`).
3. `python -m pwpf_blender validate --ours site/fields_2026011512.npz --wpc-dir wpc/ --cycle 2026-01-15T12 --calibrate-slr`
   prints bias/MAE/RMSE/correlation per percentile and per-threshold probability error, plus a suggested `slr.adjust`.
4. Put the suggestion in `config/pwpf.yaml` (`slr.adjust`), and tune `mode.pmm_weight`
   (0 = ensemble mean … 1 = pure PMM) until the 50th-percentile bias is near zero.
5. Biggest remaining lever: `--external-snow <NDFD snow GRIB2>` makes the mode the NDFD snow forecast (which carries
   the WWD/WFO collaboration) on the identical 2.5 km grid — expected to be the closest public proxy for WPC's mode
   (experimental loader, see `external.py`).

## Layout

```
pwpf_blender/            Python package
  grid.py regrid.py      NDFD 2.5 km Lambert grid; reusable bilinear / KD-tree regridders
  thermo.py ptype.py slr.py downscale.py member.py    per-member snowfall (QPF × ptype × SLR, snow level, PRISM)
  mode.py binormal.py    deterministic mode + binormal fit / CDF / percentiles
  pipeline.py            members → bins → days → windows
  export.py              Web-Mercator data PNGs, manifest.json, npz
  sources/               synthetic (tests/demo) and live loaders
  validate.py external.py static_data.py cli.py
web/                     static viewer: Leaflet UI (app.js), DOM-free logic (core.js, binormal.js)
config/pwpf.yaml         members, science settings, profiles (lite, synthetic)
.github/workflows/       scheduled build + Pages deploy
tests/                   physics, grid, binormal, end-to-end, Python↔JS parity
```

### How the viewer gets every product from one PNG per period
Each period's PNG stores three numbers per pixel — mode, σ_low, σ_high (log-encoded to 8 bits, ≈2 % precision,
alpha always 255). The browser rebuilds P(snow ≥ t) and any percentile with the same binormal CDF as Python
(`tests/test_js_parity.py` verifies agreement), so one ~0.3 MB image serves ten thresholds and seven percentiles, plus
hover read-outs. Full precision stays in the npz artifact.

## Performance
Full 2.5 km grid (3.0 M points), synthetic 34-member run: ~75 s and ~3.6 GB peak RAM. Live runs are dominated by
downloads (ECMWF ENS is the heaviest: use `lite` or fewer members on small runners).

## Licences / attribution
Code: no licence file is included yet — add the one you want before publishing. Model data: NOAA/NCEP (public domain); ECMWF open data is CC-BY-4.0 (attribution is in
the page footer); PRISM normals are free to use but check their terms before redistributing raw grids (this repo never
publishes them). Basemap © OpenStreetMap contributors © CARTO.

## Roadmap
CMC GEPS/RDPS and SREF loaders · NBM SLR / snow-level percentiles as optional inputs · ice (freezing-rain) products
using the simplified FRAM ILR · per-pixel NaN handling at CAM domain edges · automatic WPC-file download in `validate`.
