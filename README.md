# pwpf-blender — an open, GitHub-Pages-ready re-implementation of WPC's 2.5 km PWPF snowfall guidance

> **Unofficial.** Built from WPC's *public* documentation of its Probabilistic Winter Precipitation
> Forecast (PWPF). It is not an NWS/WPC product and must not be used for warnings or life-safety decisions.
> Always use official forecasts from [weather.gov](https://www.weather.gov) and [wpc.ncep.noaa.gov](https://www.wpc.ncep.noaa.gov).

GitHub Pages only serves static files, so the project splits into two parts:

```
GitHub Actions (cron, 2x/day)                      GitHub Pages (static)
  Python pipeline  ──►  .bin.gz layers + manifest ──►  web/ viewer (Leaflet; binormal maths, interpolation,
  (this package)                                      terrain adjustment and colouring all in the browser;
                                                      one small file per period serves every product)
```

## Status of the real product (read first)

NWS Service Change Notice 26-96 says the WPC Super Ensemble PWPF/PQPF data - the binormal product this repo emulates - is
replaced by Probabilistic Precipitation Portal (PPP) data **on or about 2026-11-30**; the PPP is derived from the
National Blend of Models and adjusted by forecasters. So this repo emulates the 2013-2026 WSE/binormal PWPF, which is
what the public documentation describes in detail. See [`docs/wpc_method_notes.md`](docs/wpc_method_notes.md) for the
documented-vs-implemented comparison, the 2024 nonparametric method, and open assumptions.

## What the real PWPF does (and where this repo stands)

From WPC's "About PWPF Products" page and the 2.5 km file README:

| PWPF step (public documentation) | Here | Status |
|---|---|---|
| ~60 member multi-model ensemble: 10 GEFS, 25 ECMWF ENS, 10 CMC ENS, 2 HRRR, 2 CMC RDPS, GFS, ECMWF HRES, GEFS mean, plus (until Nov 2026) SREF, 2 NAM-nest, 4 HiResW ARW, FV3-LAM | `config/pwpf.yaml` groups with the same init-time logic (`age_h`, `lags_h`), seeded random member picks, and PWPF-style fall-back to time-lagged GFS/GEFS beyond a CAM's range. **NAM, NAM-nest, HiResW, HREF and SREF were retired on 2026-11-03** (NWS SCN 26-47/48); their CAM role is filled by **RRFS** (deterministic + 5-member ensemble, 3 km) and HRRR, the members NCEP itself combines into REFS | **Implemented**; CMC RDPS comes from the MSC Datamart (`dd.weather.gc.ca`), CMC GEPS comes from NCEP's NAEFS copy on NOMADS via the grib filter. WPC hasn't published its post-transition constituent list, so the CAM part is the closest public analogue |
| PRISM-based QPF downscaling of every member | `downscale.py`: ratio of PRISM monthly normals to the same field smoothed to the member's resolution | **Implemented** (ratio clip limits are our assumption) |
| Snow level = height of the 0.5 °C wet-bulb surface, top-down search, intersected with RTMA terrain; terrain above it ⇒ snow | `thermo.py` (Stull wet-bulb, top-down crossing) + RTMA 2.5 km terrain | **Implemented** |
| Precip type: NCEP models use dominant-type (Manikin 2005); non-NCEP use a simple decision tree on T2m and T925/850/700 | `ptype.py`: dominant type from native categorical fields; decision tree otherwise | **Implemented** (tree thresholds / tie order are our assumptions) |
| Snowfall = QPF × ptype × SLR; SLR = NBM SLR × an adjustment factor | `slr.py`: Kuchera-based blend (legacy-style average with climatology and 11:1) × `slr.adjust` | **Approximated** — NBM SLR and WPC's factor aren't reproducible; calibrate `slr.adjust` (below) |
| WWD (human-edited) 24-h deterministic forecast = **mode** of the distribution | `mode.py`: probability-matched mean blended with the mean (default), or `external` (supply NDFD/WWD-like grid) | **Stand-in** — the human forecast isn't machine-reproducible |
| Binormal PDF per grid point (Toth & Szentimrey 1990), method of moments: mode = WWD forecast, skewness from the forecast's rank in the ensemble, variance matched to the ensemble **plus the forecast as an extra member** | `binormal.py` — closed form: `s = sqrt(Var / [(1-2/π)(1-2p)² + p(1-p)])`, `σ_lo = p·s`, `σ_hi = (1-p)·s` | **Implemented**; verified numerically (tests) |
| Products: exceedance probabilities and percentile accumulations for **24 h periods at 6-hour increments through 72 h**; 48 h = sum of two adjacent 24 h; one 72 h set | `windows` in config → 15 periods: 24 h ending +24, +30 … +72 h, 48 h ending +48 … +72 h, 72 h ending +72 h (ids `s24_f030` etc., measured from the cycle). 48/72 h modes are sums of the 24 h modes | **Implemented** (the 12Z–12Z day boundaries of the older docs are replaced by cycle-relative periods) |
| Freezing rain: ice-to-liquid ratio from the simplified FRAM (Sanders & Barjenbruch 2016), precipitation rate only | `ice.py`: ILR = 0.1395·P^−0.541 (P in/h), flat ice = ILR × liquid, optional radial (×0.394); applied per model step in freezing-rain grid cells, then the same binormal/PRISM/web path as snow; thresholds 0.01–0.5 in | **Implemented, experimental** — which ptype WPC counts as freezing rain, its rate averaging and any caps are unpublished, so amounts are indicative |
| 2.5 km NDFD CONUS grid (2145×1377, spherical Lambert, 2539.703 m) | `grid.py`, checked against published corner coordinates | **Implemented** |

So the *distributional machinery* follows the documentation closely; the pieces that WPC does not publish
(human-edited mode, NBM SLR, SLR adjustment factor, exact thresholds) are explicit, tunable stand-ins.
`validate` exists to measure and close the remaining gap against WPC's own 2.5 km files.

## Quick start (no network: synthetic ensemble)

```bash
pip install -r requirements.txt pytest
pytest -q                                              # tests incl. Python↔JavaScript parity and the download layer
python -m pwpf_blender run --synthetic --out site/data --save-npz
cp web/* site/ && python -m http.server -d site 8000   # open http://localhost:8000
PYTHONPATH=. python scripts/preview.py site/fields_*.npz preview.png s24_f024   # needs matplotlib
```

## Deploy on GitHub Pages

1. Push this repo to GitHub. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. **Actions → "Build PWPF-style snowfall guidance" → Run workflow** with profile `synthetic` to confirm the site deploys.
3. Run again with profile `lite` (5 km, ~30 members) then `full` (2.5 km, ~50 members).
   Memory: the pipeline keeps all twelve 6-h member bins (float16) in RAM - roughly 0.5 GB for `lite`, 3-4 GB for `full`
   (fine on current 16 GB hosted runners).
   Set the repository variable `PWPF_PROFILE` to choose what the twice-daily schedule runs (default `lite`).

The workflow restores/creates `static/` (RTMA terrain + PRISM normals, cached separately), runs the pipeline, uploads the
site to Pages and keeps the full-precision `fields_*.npz` as a 3-day artifact for validation.

### Where the data comes from

All NOAA feeds are read **directly** over HTTPS - no wrapper library. For each GRIB2 file the loader fetches
`<url>.idx` (a text inventory), selects the records it needs by matching `VAR:LEVEL:FCST`, downloads only those byte
ranges and decodes them with eccodes (`pwpf_blender/noaa_http.py`). Records are identified by NOAA's own inventory
text, so there is no cfgrib/eccodes short-name guessing. AWS Open Data is tried first, NOMADS second:

| Loader | Source | Files |
|---|---|---|
| `gefs` | `noaa-gefs-pds` | `gefs.YYYYMMDD/HH/atmos/pgrb2ap5/{gep01..30,geavg}.tHHz.pgrb2a.0p50.fFFF` |
| `gfs` | `noaa-gfs-bdp-pds` | `gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF` |
| `hrrr` | `noaa-hrrr-bdp-pds` | `hrrr.YYYYMMDD/conus/hrrr.tHHz.wrf{sfc,prs}fFF.grib2` |
| `rrfs` | `noaa-rrfs-ops-pds` / NOMADS `rrfs/prod` | `rrfs.YYYYMMDD/HH/rrfs.tHHz.{2dfld,prslev}.3km.fFFF.conus.grib2` |
| `rrfs_ens` | same | `rrfsens.YYYYMMDD/HH/mNNN/rrfs.tHHz.mNNN.{2dfldnomads,prslevnomads}.3km.fFFF.conus.grib2` |
| `rdps` / `gdps` | MSC Datamart `dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km` (RDPS, 10 km) and `model_gdps/15km` (GDPS, 0.15°; registered, off by default). One small GRIB2 per variable x level x hour (19 files/step), no `.idx` | decoded with eccodes; 2 req/s, 2 parallel (no published limit); height/RH/T/pressure units auto-detected. HRDPS 2.5 km is not used (covers only southern Canada / northern US) |
| `cmce` | NOMADS `naefs/prod/cmce.*` (CMC GEPS, 0.5°), via the NOMADS grib filter (server-side subsetting, ~2 requests per step) | decoded with eccodes; NCEP rate rules apply |
| `ifs_ens`, `ifs_hres` | ECMWF open data (`ecmwf-opendata`, default AWS mirror) | decoded with eccodes |
| terrain | RTMA 2.5 km `rtma2p5.tHHz.2dvaranl_ndfd.grb2_wexp` (fallback: RRFS `HGT:surface`) | `prepare-static` |

RRFS `prslev` files are ~550 MB each, which is why byte-range subsetting matters: a step downloads a few tens of MB.

### Missing data

Members that are late, missing or fail part-way never abort the run.  Every period (e.g. `s24_f042`) is
blended from the members that delivered *all* of its 6-h bins, with weights renormalised, so a member that only
reaches +36 h still counts for the earlier periods.  A member is retried per bin and given up on after
`member_failure_limit` consecutive failures (saves request budget).  Periods with fewer than `min_members` complete
members are skipped (listed under `windows_skipped` in `manifest.json` and shown in the viewer); periods with fewer
than `warn_members` are flagged "reduced ensemble".  The run only fails if no member delivers anything or no period
can be produced.  The viewer's table under the map lists every member, its model run, which 6-h bins arrived and
the first error, so failures are visible without reading the CI log.

## Respecting server limits

All HTTP goes through one polite client (`noaa_http._request`), so no server's rate or download limit is approached:

| Server | Published limit | What this client does |
|---|---|---|
| NOMADS / NCEP ftp (`nomads.ncep.noaa.gov`, `ftp(prd).ncep.noaa.gov`) | 120 hits/min per IP, summed over those hosts, HEAD counts like GET ([SCN 21-32](https://www.weather.gov/media/notification/pdf2/scn21-32nomad_changes.pdf)) | ≤ 60 requests/min in total, 2 connections; only used if AWS fails |
| AWS open-data buckets (NOAA, ECMWF mirror) | none published; S3 answers `503 SlowDown` when a prefix is hammered | ≤ 20 requests/s and 6 connections per NOAA bucket (10 and 4 elsewhere, 3/s and 2 for the ECMWF mirror, which also halves its allowed rate on every 503 (down to 1/8) and creeps back after good answers, and where each `.index` file is fetched once per run instead of once per member; where the IFS byte ranges are fetched by our own resumable downloader: each range is paced, counted and retried with back-off, finished ranges are never fetched twice, even after a switch of mirror); back-off on 503, shared by all threads; if a mirror keeps failing the IFS download moves on to google, azure, then data.ecmwf.int and the failing mirror is tried last from then on; IFS downloads stop after 45 min (`PWPF_IFS_BUDGET_MIN`) so the run still finishes with the members it has; `PWPF_IFS_SOURCE=aws|google|azure|ecmwf` picks the first mirror |
| ECMWF open data | 500 simultaneous connections | ≤ 2 requests/s; `ecmwf-opendata` retries capped at 3 (library default is 500 × 120 s) |
| PRISM | refuses a file downloaded twice within 24 h | one request every 2 s, one connection, files streamed to disk once, a ledger refuses to re-request a file inside 24 h, raw files cached between workflow runs |

On `429`/`503` (and `403` from NCEP, whose firewall blocks over-limit IPs; but see the availability probes below) it honours `Retry-After` or backs off
exponentially with jitter; four consecutive throttles put a host on a 5-minute cool-down during which requests fail fast
(callers fall back to another mirror or drop the member). A missing file (`404`) is never retried, matching the NWS
guidance not to re-request a file that isn't published yet. Each file's `.idx` is fetched once per run, and the previous
step's accumulation is kept so the next bin doesn't download the same records again. The run log ends with the
**request budget** (requests, MB, throttles and seconds spent waiting, per server); `PWPF_HTTP_RATE_SCALE` exists only
for the unit tests and must stay at 1 in CI. Schedules fire twice a day, after the data are published, and runs never
overlap (`concurrency` group).

### Diagnosing a feed

```bash
python -m pwpf_blender diagnose --download      # per model: newest cycle, which URL answered, which records match,
                                                # and one real f006 read (grid, value ranges)
python -m pwpf_blender diagnose --only rrfs terrain
```

The workflow runs this before the pipeline (informational), so the Actions log always shows the real state of every
feed. Errors name the URL and HTTP status; when an inventory is readable but lacks a field, the first inventory
records are printed.

### Live-data caveat

The live loaders are unit-tested against real GRIB2 messages served by a local byte-range HTTP server, and their URL
layouts were checked against the AWS/NOMADS directory listings and NCO inventories, but the authoring environment
could not reach NOAA, so a full live run has only ever happened on GitHub's runners. Everything downstream (wet-bulb snow level, ptype, SLR,
downscaling, binormal fit, export, viewer) is covered by tests on a synthetic ensemble. Start with `--profile lite` and
read the `diagnose` output and Actions log; a member that fails is dropped with a warning that carries the reason, and
a period with fewer than `min_members` complete members is skipped (see "Missing data").

### PRISM normals

PRISM retired its BIL files and old web service on 2025-10-01 (everything is now Cloud Optimized GeoTIFF), and its new
web service does **not** serve the 30-year normals - those come from PRISM's FTP tree, browsable over HTTPS. `prepare-static`
therefore reads the directory listing at `https://data.prism.oregonstate.edu/normals/us/800m/ppt/monthly/`
(files `prism_ppt_us_30s_2020MM_avg_30y.zip`, ~56 MB each, ~0.7 GB once; the 12 months are then area-averaged onto the
2.5 km grid and cached). Pass `--prism-url https://data.prism.oregonstate.edu/normals/us/4km/ppt/monthly/` for the
lighter 4 km set. Files are picked by name (GeoTIFF or zipped BIL), each download is checked to really be a zip/TIFF,
and if the names don't look like twelve months the listing it saw is printed.

PRISM is **not fatal**: if it can't be fetched the build continues without QPF downscaling, a GitHub Actions warning is
raised, and the site's notice says "PRISM QPF downscaling was not applied". The workflow caches terrain and PRISM
separately so a failed PRISM download is retried on the next run. To supply the data yourself, download the twelve
monthly 4 km `ppt` normals from <https://prism.oregonstate.edu/normals/> into a folder and run
`python -m pwpf_blender prepare-static --prism-dir <folder>` (add `--require-prism` to fail instead of continue).

## Getting "very close" to the real PWPF — the calibration loop

1. Run with `--save-npz` for a 00Z or 12Z cycle.
2. Download WPC's files for the same cycle from `ftp.wpc.ncep.noaa.gov/pwpf/conus_2.5km/` (percentiles
   `2.5kmprcntil_24hsnow_<pp>pt_<yyyymmddcc>f<hhh>.grb`, probabilities `psnow24_ge*`).  `validate` walks every
   24 h window (`s24_f024` … `s24_f072`) and skips lead times for which you didn't download files.
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
  ice.py                 simplified FRAM freezing-rain accumulation
  pipeline.py            members → 6-h bins → rolling 24/48/72 h windows (snow + ice)
  export.py              Web-Mercator .bin.gz data layers (PWPB format), manifest.json, npz
  sources/               synthetic (tests/demo) and live loaders
  validate.py external.py static_data.py cli.py
web/                     static viewer: Leaflet UI (app.js), DOM-free logic (core.js, binormal.js, bin.js, terrain.js)
config/pwpf.yaml         members, science settings, profiles (lite, synthetic)
.github/workflows/       scheduled build + Pages deploy
tests/                   physics, grid, binormal, end-to-end, Python↔JS parity
```

### How the viewer gets every product from one `.bin.gz` per period
There are no image layers: every period is one gzip-compressed binary file, `<id>.bin.gz` (snowfall) and `<id>_ice.bin.gz`
(freezing rain, only for periods where some member produces ice), plus `terrain_ref.bin.gz` (the model's own terrain height
on the same grid). The format, "PWPB" v1 (`pwpf_blender/export.py`, reader in `web/bin.js`):

```
header   "PWPB" | u8 version=1 | u8 planes | u8 flags (bit0: 16-bit planes are byte-shuffled) | u8 0 | u16 W | u16 H
per plane (12 B)  u8 id | u8 bits (8|16) | u8 kind (0 log, 1 linear) | u8 0 | f32 p0 | f32 p1
data     the planes, row-major; 16-bit planes store all high bytes, then all low bytes (compresses far better)
log      x = p0 * (exp(v / vmax * ln(1 + p1/p0)) - 1)        linear   x = v * p0 + p1
planes   1 mode, 2 sigma_low, 3 sigma_high (in; log: snow A=0.5, max 200 in; ice A=0.02, max 2 in)
         4 potential snowfall (in), 5 snow level (m), 6 model terrain (m)   <- planes 4-6 only for snowfall (terrain adjustment)
```
`web.bin_bits: 8` (default) gives ~2 % steps in the log-encoded planes and 25 m steps in snow level, ~1-3 MB per period for a CONUS-size grid; `16` keeps ~0.01 % / 0.1 m but is about four times larger (the low bytes are noise to gzip). 2 % steps are far below the width of a colour class and are smoothed away by the per-pixel interpolation. Everything after download is
done by the browser: gunzip (`DecompressionStream`; a server that already decoded the gzip is handled too), the binormal CDF
and percentiles at every pixel (`tests/test_js_parity.py` checks them against Python, also for the 8-bit variant), palette
classification, terrain adjustment, hill-shading and the point read-outs. Full precision stays in the npz artifact.

**Drawing and interpolation.** The map layer is a Leaflet `GridLayer` (`DataLayer` in `web/app.js`). Every *screen* pixel of
a tile interpolates mode / σ_low / σ_high from the data grid, evaluates the product and is classified, so class boundaries
are smooth curves at every zoom level instead of blocky data cells. Interpolation is bilinear, switching to Catmull-Rom
bicubic once a data cell spans 8 or more screen pixels (i.e. when the grid would otherwise look coarse); `PWPFCore.renderTile`
takes `interp: 'auto' | 'bilinear' | 'bicubic'`. A dense retina tile with bicubic + terrain adjustment renders in about
0.1 s in Node (`tests/test_tiles.py`). Interpolation only smooths: the real detail is the model grid (2.5 km data,
resampled to `web.px_m`), which no file format can add to.

### Terrain adjustment (browser side, optional)
When "Terrain adjustment" is ticked (default; `web.terrain.enabled` in the config switches the planes on/off for everybody),
the viewer fetches [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) (Terrarium PNG elevation, keyless,
CORS-enabled; the only PNGs the project uses are these third-party elevation tiles) for the visible map at zoom
`min(z, dem_max_zoom)` and, per screen pixel, multiplies mode and both sigmas by a factor `k` that depends on the difference
between the true ground height `zp` and the model's terrain height `zr`:

* **Elevation gradient** `fq = 1 + 0.20 /km * (zp - zr)`, limited to [0.75, 1.35].
* **Rain/snow line** `rho` = ratio of the snow fraction at `zp` to that at `zr` (a logistic of width 100 m centred on the
  per-pixel 0.5 °C wet-bulb snow level the pipeline exports), regularised and capped at 3. Valleys below the line lose snow,
  ridges above it gain it.
* `k = fq * rho`, never more than `fq * potential_snowfall / mode`: you can't get more snow than all the precipitation
  falling as snow. At the model terrain height `k` is exactly 1.

**These numbers are our assumptions, not part of WPC's published method** (WPC's PWPF relies on PRISM downscaling plus the
snow-level mask, which the pipeline already applies). The adjustment is therefore a display-time option: untick it to see
the pure blend. It only applies from zoom `min_zoom` (5) and to snowfall, not freezing rain. If the elevation tiles can't be
fetched (offline, blocked) the viewer falls back to the unadjusted field and says so. The point read-out shows ground height
vs model height and the factor applied. An optional hill-shade layer (same tiles) helps to see why the pattern follows
ridges. Terrain data: Mapzen / AWS Terrain Tiles (USGS 3DEP, SRTM, GMTED2010, ETOPO1 and others); the attribution is added to
the map when the layer is in use.

### Calibration report
Every run ends with a `=== calibration report ===` block in the log (`calibration:` in `config/pwpf.yaml`): for the 72-h period and the 24-h period ending 30 h after the cycle it prints, for a Northeast box, the percentage of area at or above 0.1/1/2/4/6/12 in plus mean, 99th percentile and maximum - for the blended mode, the blended 50th percentile, the plain member mean, and **each model group on its own** - and, for the 72-h period, our value at ~24 reference towns next to the NWS WFO "expected snowfall" labels read off the official maps (`pwpf_blender/calibration.py`; edit the list for another event). It shows at a glance whether the blend is high or low against the official forecasts and which model group drives it, before any SLR / mode / QPF knob is touched. The browser-side terrain adjustment is not included.

### Viewer behaviour
* **Defaults:** Snowfall, "Forecast (mode)", terrain adjustment **on**, terrain shading **on** (URL hash `#...&k=prob&ta=0&sh=0` overrides). The map fills the window height; the ensemble-member table sits directly below it (link in the header).
* **Cursor tooltip:** hovering shows the exact values under the pointer - the selected product, forecast mode, chance of exceeding the chosen threshold, 10/50/90th percentile, and (from zoom 5) ground elevation vs model terrain with the terrain factor - computed from the same interpolated field the map paints (`PWPFCore.sampleAt`). Click pins a point and fills the side-panel point forecast; on touch screens a tap shows the tooltip.
* **Member freshness:** each model uses the **newest run that has started by the cycle** (`age_h: 0`; cycle `auto` = the newest 00/12Z cycle that has started). Before a run is used its last needed forecast hour must be published (NOAA/CMC/ECMWF inventories are probed - a probe is one request, never retried: NOMADS answers **403** for a run directory that does not exist yet, which means "not published", not "throttled"); if not, the **previous run** is used, and then the one before it, and the member table notes the substitution. `PWPF_IFS_SOURCE`, the retry/back-off and the 45-min IFS budget are described under politeness above.
* **Terrain toggle:** switching the adjustment (or the data) builds a *new* map layer beneath the old one and removes the old one when the new one has painted (or after 10 s), so the map never goes blank; elevation tiles are fetched with a timeout and two retries, and a tile whose elevation cannot be loaded is drawn un-adjusted instead of staying hidden. Painted tiles carry `data-adjusted="1"|"0"` for debugging.
* **Browser tests (optional):** `tests/test_browser.py` drives the real page in headless Chrome (`pip install playwright`, `npm i leaflet`, system Chrome); it is skipped when they are missing. It checks the defaults, that data is visible with and without terrain adjustment, the adjustment off->on round trip, slow/failed/hung elevation tiles, the shading layer, the tooltip and the boundary layers.

## Performance
Full 2.5 km grid (3.0 M points), synthetic 34-member run: ~75 s and ~3.6 GB peak RAM. Live runs are dominated by
downloads (ECMWF ENS is the heaviest: use `lite` or fewer members on small runners).

## Licences / attribution
Code: no licence file is included yet — add the one you want before publishing. Model data: NOAA/NCEP (public domain); ECMWF open data is CC-BY-4.0 (attribution is in
the page footer); PRISM normals are free to use but check their terms before redistributing raw grids (this repo never
publishes them). Basemap © OpenStreetMap contributors © CARTO. Terrain tiles (optional adjustment / hill-shade): AWS Terrain Tiles (Mapzen; USGS 3DEP, SRTM, GMTED2010, ETOPO1).

## Roadmap
SREF loader · NBM SLR / snow-level percentiles as optional inputs · ice (freezing-rain) products
using the simplified FRAM ILR · per-pixel NaN handling at CAM domain edges · automatic WPC-file download in `validate`.


*Colours:* snowfall amounts use the NWS "Expected Snowfall" ramp; probabilities use ColorBrewer YlGnBu; basemaps are keyless (Esri light gray by default, OSM / USGS Topo selectable, with automatic fallback).

*Borders:* `web/borders/*.json` (country, state and county lines, coastline, small islands/lakes) are generated once by `scripts/make_borders.py` from the public-domain us-atlas / world-atlas TopoJSON plus Natural Earth 10 m lakes (command in the script's docstring) and committed. The viewer draws them above the data as a cartographic hierarchy: country borders heaviest (near-black, white casing), then coastlines and lake shores (thin dark blue-grey, light casing), state lines (medium dark grey, light casing) and county lines (hairline, from zoom 6, no casing); widths grow with zoom so the hierarchy holds from the continental view to street level. Small islands and lakes appear from zoom 6, so the continental view stays calm. Each class has a toggle in the layer button.
