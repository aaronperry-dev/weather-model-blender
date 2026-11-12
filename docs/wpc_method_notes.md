# WPC PWPF - what the literature says, and what this repo assumes

Sources reviewed: *About WPC's PWPF Products* (WPC web page, last modified May 2022), B. Veenhuis, *Exploring new
techniques for WPC's PWPF* (13 Feb 2024), NWS Service Change Notices 26-75, 26-96 (PPP), 26-47/26-48 (RRFS/REFS).

## 1. Product status (important)

* The WSE-based PWPF/PQPF that this repo emulates is scheduled to be replaced on or about **2026-11-30** by data from the
  Probabilistic Precipitation Portal (PPP), derived from the National Blend of Models and adjusted by forecasters
  (SCN 26-96). Legacy PWPF web pages end the same day; WSE-derived data stops on the SBN.
* PPP products: snow exceedance 0.1, 1, 2, 4, 6, 8, 12, 18 in; radial ice 0.01-1.5 in; periods of 6, 12, 24, 36, 48,
  72 h; 10/90 % (and 25/50/75, 5/95 in GRIB2) percentiles; hourly updates; GRIB2 on `ftp-wpc.ncep.noaa.gov/prob_precip_portal/`
  (SCN 26-75, SCN 26-96).  SCN 26-96 states that standard deviation is no longer produced.
* Consequence: this repo is best described as an open **emulation of the WSE/binormal PWPF** (2013-2026), not of the
  forthcoming PPP. A PPP-like product would be built from NBM probabilistic snowfall instead (see README).

## 2. Documented method vs. implementation

| Documented (source) | Implemented | Notes |
|---|---|---|
| 24-h forecasts for 3 days; 48-h = sum of two 24-h; a single set for 72 h (About) | `s24_f024..f072`, `s48_f048..f072`, `s72_f072` | 48/72-h modes are sums of the 24-h modes, member totals are sums of 6-h bins |
| 24-h products generated at **6-hour increments** through 72 h (About) | 24 h periods ending every 6 h (+24 ... +72 h from the cycle) | implemented with a 12-bin buffer; WPC's handling of off-cycle deterministic periods is unpublished, so every 24-h mode comes from the ensemble stand-in (or `--external-snow`) |
| Members: 10 GEFS (06Z), 25 ECMWF ENS, 10 CMC ENS, SREF, 2 HRRR, 2 RDPS, 2 NAM-nest, 4 ARW, FV3, GFS, ECMWF, GEFS mean, WPC fcst (About) | same groups, NAM/HiResW/SREF replaced by RRFS (+ens); no CMC | page predates the Nov-2026 retirements; WPC's new list is unpublished |
| PRISM QPF downscaling per member (About) | `downscale.py` | ratio clip limits are assumptions |
| Ptype: Manikin dominant type (NCEP) / decision tree on T2m, T925/850/700 (non-NCEP) (About) | `ptype.py` | tree thresholds are assumptions |
| Snow level = 0.5 C wet-bulb, top-down, vs **5-km RTMA topography**; terrain above => snow (About) | `thermo.py` + RTMA terrain (2.5 km) | WPC does the member processing at 5 km and a later step produces the 2.5 km product (Veenhuis) - the `lite` profile (5 km) is structurally closer |
| Ptype grids downscaled with **snow level + downscaled 2-m temperature** (Veenhuis 2024) | snow level only | gap: no terrain-adjusted 2-m T; combination rule unpublished |
| SLR = NBM SLR x adjustment factor (About) | Kuchera-based blend x `slr.adjust` | NBM SLR / factor not reproducible |
| Freezing rain: ILR from simplified FRAM (Sanders & Barjenbruch 2016), precipitation rate only (About) | `ice.py`: ILR_P = 0.1395 P^-0.541 (P in/h, Eq. 2 of the paper), flat ice = ILR x liquid, radial = 0.394 x flat; evaluated per model step where the ptype is freezing rain | experimental; see section 4 |
| Binormal PDF, method of moments, mode = WPC forecast, rank => skew, variance incl. forecast (About) | `binormal.py` | exact |
| Mode (WWD forecast) | ensemble PMM stand-in | human forecast unavailable |

## 3. The 2024 nonparametric method (Veenhuis)

Why: the binormal with mode = deterministic forecast gives too-high 10th percentiles (overconfidence) and a tail that
drops off too fast; a right-skewed PDF with mode somewhat below the deterministic forecast verifies better.

Algorithm (slides): (1) downscale members to 5 km; (2) evaluate regressions that turn ensemble relative frequencies of
exceeding thresholds, plus PRISM daily QPF (alone and as interaction terms), into calibrated probabilities - NOHRSC
snowfall analysis is the binary predictand; (3) build a calibrated ensemble CDF; (4) evaluate quantile regressions
(single CONUS-wide set) on the WPC deterministic forecast to give a second CDF; (5) merge on percentiles
[1, 5, 10, 15, 25, 30, 40, 50, 60, 70, 75, 85, 90, 95, 99]: for p in {99, 95, 90} take max of the two CDFs; for the middle
percentiles use the calibrated ensemble; for p in {15, 10, 5, 1} use the quantile-regression CDF where the WPC
forecast is below the calibrated-ensemble 15th percentile, else the calibrated ensemble; (6) read exceedance
probabilities off the final CDF.

Not reproducible from public documents: the regression/quantile-regression coefficients (they need a multi-season
archive of the WSE members paired with NOHRSC analyses) and the human deterministic forecast. Slide inconsistencies to
be aware of: the percentile list omits 20 although step 5c uses it, and step 5d writes `WPC_CAL_CDF` where the
accompanying text says "calibrated WSE".  Reported 2023-24 results: Brier skill higher than operational WPC and NBM at
all thresholds for CONUS F024 and F072 (significant at 4, 8, 12 in F024); still some overconfidence at high
probabilities; ensemble underdispersion remains an issue.  The deck says binormal was still operational in Feb 2024.

## 4. Freezing rain / ice (this repo)

* Equation source: Sanders & Barjenbruch (2016), Wea. Forecasting 31, 1041-1060. Eq. 2 `ILR_P = 0.1395 P^-0.541`
  (P in in/h); Eq. 1 radial = 0.394 x flat. The full FRAM adds wet-bulb and wind regressions with regime weights
  (Eq. 5-7); WPC's description is "simplified, precipitation rate only", so only Eq. 2 is used.
* Choices that are ours, not documented by WPC: P floored at 0.02 in/h (lower edge of the paper's data), ILR capped
  at 1, the rate is the model step average (3-6-hourly global-model steps therefore give a higher ILR than hourly CAM
  steps), freezing rain is whatever the member's ptype says (native categorical field, or our decision tree: warm
  layer aloft above 2 C and T2m <= 0 C), ice is not reduced where terrain is above the snow level, and the ice
  distribution uses the same binormal machinery as snow with a PMM/mean mode.
* Output is flat ice by default (`ice.report: radial` switches to radial). PPP publishes *radial* ice 0.01-1.5 in, so
  set `radial` if you compare with PPP.
* Per-period ice layers are written only when some member reaches `ice.min_total_in`; the viewer greys out periods
  without one.
