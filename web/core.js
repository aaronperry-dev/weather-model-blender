/* DOM-free viewer logic: palettes, field computation, point read-out.  Testable in Node. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./binormal.js'), require('./terrain.js'));
  else root.PWPFCore = factory(root.Binormal, root.PWPFTerrain);
})(typeof self !== 'undefined' ? self : this, function (Binormal, Terrain) {
  'use strict';

  /* Probabilities: ColorBrewer sequential BuPu (blue -> purple), one class per 10 % step from 10 %.  The two
     near-white BuPu classes are merged into #cfe0ee so that 10-20 % stays visible on a light basemap, and one step
     (#670763) is interpolated between the two darkest classes to get nine. */
  var PROB_STOPS = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  var PROB_COLORS = ['#cfe0ee', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#670763', '#4d004b'];
  /* Snowfall amounts: the NWS "Expected Snowfall" ramp (colours sampled from the official legend); each colour
     starts at its stop, e.g. 6-8 in = pale yellow, 72+ in = purple. */
  var SNOW_STOPS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36, 48, 60, 72];
  var SNOW_COLORS = ['#bdd7e7', '#6baed6', '#3182bd', '#005199', '#002694', '#ffff96', '#ffc400', '#ff8700',
                     '#db1400', '#9e0000', '#690000', '#360000', '#ccccff', '#9f8cd8', '#7c52a5'];

  var ICE_STOPS = [0.01, 0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0];
  var ICE_COLORS = ['#ffe3ef', '#ffc2da', '#ff9ec3', '#f46fa8', '#d6408f', '#b0237a', '#8a1a6e', '#5f1360', '#3a0d4a'];

  function hexToRGB(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }

  function makePalette(stops, colors) {
    return { stops: stops, colors: colors, rgb: colors.map(hexToRGB) };
  }
  var palettes = { prob: makePalette(PROB_STOPS, PROB_COLORS), snow: makePalette(SNOW_STOPS, SNOW_COLORS),
                   ice: makePalette(ICE_STOPS, ICE_COLORS) };

  function binIndex(stops, v) {            // highest i with v >= stops[i], or -1
    var lo = 0, hi = stops.length - 1, ans = -1;
    while (lo <= hi) { var mid = (lo + hi) >> 1; if (v >= stops[mid]) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
    return ans;
  }

  /* Parameters from a parsed .bin layer (PWPFBin.parse): mode / sigma_low / sigma_high in inches, plus the optional
     potential snowfall (`pot`) and snow level (`sl`) planes the terrain adjustment uses.  An encoded zero is "no signal". */
  function paramsFromBin(bin) {
    var P = bin.planes, n = bin.W * bin.H, i, step1 = bin.meta[1].step1;
    var lo = new Float32Array(n), hi = new Float32Array(n), a = P[2], b = P[3], fl = Binormal.S_FLOOR;
    for (i = 0; i < n; i++) { lo[i] = a[i] > fl ? a[i] : fl; hi[i] = b[i] > fl ? b[i] : fl; }
    /* "no signal" = encoded zero: mode below the first non-zero level and sigma_high below ~1.5 levels (scales with the
       encoding, so it works for snowfall A=0.5/200 in and ice A=0.02/2 in, 8 or 16 bit) */
    return { mode: P[1], lo: lo, hi: hi, n: n, W: bin.W, H: bin.H, pot: P[4] || null, sl: P[5] || null,
             zeroMode: step1 * 0.99, zeroHi: Math.max(step1 * 1.5, fl * 1.01) };
  }

  /* Per-cell value of a product (probability %, percentile or mode, in inches), 0 where there is no signal.  Same numbers
     as render().value but without building colours: the map layer colours it per screen pixel instead. */
  function field(params, product) {
    var n = params.n, val = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var m = params.mode[i], s1 = params.lo[i], s2 = params.hi[i];
      if (m < params.zeroMode && s2 < params.zeroHi) continue;
      val[i] = product.kind === 'prob' ? 100 * Binormal.exceedance(m, s1, s2, product.t)
             : product.kind === 'pct' ? Binormal.percentile(m, s1, s2, product.q / 100) : m;
    }
    return val;
  }

  var HALF = 20037508.342789244;           // half the Web-Mercator world width (m)

  function catmull(t, w) {                 // Catmull-Rom weights for the 4 taps around a point at fraction t
    var t2 = t * t, t3 = t2 * t;
    w[0] = -0.5 * t3 + t2 - 0.5 * t; w[1] = 1.5 * t3 - 2.5 * t2 + 1; w[2] = -1.5 * t3 + 2 * t2 + 0.5 * t; w[3] = 0.5 * t3 - 0.5 * t2;
  }

  /* Colour one map tile at screen resolution, entirely in the browser.
       o.params   decoded layer (paramsFromBin); o.W/H/merc: its Web-Mercator grid (cells of merc.px metres)
       o.product  {kind:'prob', t} | {kind:'pct', q} | {kind:'mode'};  o.pal palette;  o.alpha 0..1
       o.z/tx/ty  map tile; o.size/scale  tile size in CSS px and device scale (canvas is size*scale wide)
       o.zref     model terrain per cell (m) and o.dem {data, tz, tx, ty, size} the elevation tile covering this map
                  tile (same or a coarser zoom), o.adjust terrain-adjustment config: when all are given the
                  distribution at every pixel is rescaled for the true elevation (see terrain.js scale())
       o.interp   'auto' | 'bilinear' | 'bicubic'
     Every output pixel interpolates the parameters (bilinear; Catmull-Rom bicubic once a data cell spans >= 8 output
     pixels, i.e. when the grid would otherwise look coarse), applies the terrain factor, evaluates the product and is
     classified, so class boundaries are smooth curves at any zoom.  Output: RGBA, alpha = o.alpha where coloured. */
  function renderTile(o) {
    var P = o.params, W = P.W, H = P.H, merc = o.merc, pal = o.pal, size = o.size, scale = o.scale, z = o.z, tx = o.tx, ty = o.ty;
    var n = size * scale, out = new Uint8ClampedArray(n * n * 4), world = size * Math.pow(2, z), a = Math.round(o.alpha * 255);
    var prod = o.product, mode = P.mode, lo = P.lo, hi = P.hi, zm = P.zeroMode, zh = P.zeroHi, FL = Binormal.S_FLOOR;
    var dem = o.dem, adj = o.adjust, zref = o.zref, pot = P.pot, sl = P.sl;
    var useAdj = !!(adj && dem && zref && pot && sl);
    var cellPx = merc.px * world / (2 * HALF) * scale;
    var cubic = o.interp === 'bicubic' || (o.interp !== 'bilinear' && cellPx >= 8);
    var ci = new Int32Array(n), cf = new Float32Array(n), cw = cubic ? new Float32Array(4 * n) : null, du = useAdj ? new Float32Array(n) : null;
    var wy = new Float32Array(4), wxs = new Float32Array(4), p, q, k;
    var dd = dem ? z - dem.tz : 0, dsc = Math.pow(2, dd);
    for (p = 0; p < n; p++) {
      var g = (((tx * size + (p + 0.5) / scale) / world) * 2 * HALF - HALF - merc.xmin) / merc.px - 0.5, i0 = Math.floor(g);
      ci[p] = i0; cf[p] = g - i0;
      if (cubic) { catmull(g - i0, wxs); for (k = 0; k < 4; k++) cw[4 * p + k] = wxs[k]; }
      if (useAdj) du[p] = (((tx + (p + 0.5) / n) / dsc) - dem.tx) * dem.size - 0.5;
    }
    var first = pal.stops[0], tcfg = adj;
    for (q = 0; q < n; q++) {
      var gy = (merc.ymax - (HALF - ((ty * size + (q + 0.5) / scale) / world) * 2 * HALF)) / merc.px - 0.5;
      var j0 = Math.floor(gy), fy = gy - j0;
      if (j0 < -1 || j0 >= H) continue;
      var dv = useAdj ? (((ty + (q + 0.5) / n) / dsc) - dem.ty) * dem.size - 0.5 : 0;
      if (cubic) catmull(fy, wy);
      var j1 = j0 + 1, r0 = j0 >= 0 ? j0 * W : -1, r1 = j1 < H ? j1 * W : -1;
      for (p = 0; p < n; p++) {
        var i = ci[p];
        if (i < -1 || i >= W) continue;
        var i1 = i + 1, c00 = r0 >= 0 && i >= 0 ? r0 + i : -1, c10 = r0 >= 0 && i1 < W ? r0 + i1 : -1;
        var c01 = r1 >= 0 && i >= 0 ? r1 + i : -1, c11 = r1 >= 0 && i1 < W ? r1 + i1 : -1;
        if (!((c00 >= 0 && (mode[c00] >= zm || hi[c00] >= zh)) || (c10 >= 0 && (mode[c10] >= zm || hi[c10] >= zh)) ||
              (c01 >= 0 && (mode[c01] >= zm || hi[c01] >= zh)) || (c11 >= 0 && (mode[c11] >= zm || hi[c11] >= zh)))) continue;
        var fx = cf[p], m, s1, s2;
        if (cubic && i >= 1 && i + 2 < W && j0 >= 1 && j0 + 2 < H) {
          m = 0; s1 = 0; s2 = 0;
          for (var dj = 0; dj < 4; dj++) {
            var rb = (j0 - 1 + dj) * W + i - 1, wr = wy[dj];
            for (var di = 0; di < 4; di++) {
              var wgt = wr * cw[4 * p + di], cc = rb + di;
              m += wgt * mode[cc]; s1 += wgt * lo[cc]; s2 += wgt * hi[cc];
            }
          }
          if (m < 0) m = 0; if (s1 < FL) s1 = FL; if (s2 < FL) s2 = FL;
        } else {
          var w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
          m = (c00 >= 0 ? w00 * mode[c00] : 0) + (c10 >= 0 ? w10 * mode[c10] : 0) + (c01 >= 0 ? w01 * mode[c01] : 0) + (c11 >= 0 ? w11 * mode[c11] : 0);
          s1 = (c00 >= 0 ? w00 * lo[c00] : w00 * FL) + (c10 >= 0 ? w10 * lo[c10] : w10 * FL) + (c01 >= 0 ? w01 * lo[c01] : w01 * FL) + (c11 >= 0 ? w11 * lo[c11] : w11 * FL);
          s2 = (c00 >= 0 ? w00 * hi[c00] : w00 * FL) + (c10 >= 0 ? w10 * hi[c10] : w10 * FL) + (c01 >= 0 ? w01 * hi[c01] : w01 * FL) + (c11 >= 0 ? w11 * hi[c11] : w11 * FL);
        }
        if (m < zm && s2 < zh) continue;                       // no signal here
        if (useAdj) {
          if (c00 >= 0 && c10 >= 0 && c01 >= 0 && c11 >= 0) {      // cells outside the grid carry no terrain / snow level
            var w0 = (1 - fx) * (1 - fy), w1 = fx * (1 - fy), w2 = (1 - fx) * fy, w3 = fx * fy;
            var zr = w0 * zref[c00] + w1 * zref[c10] + w2 * zref[c01] + w3 * zref[c11];
            var slv = w0 * sl[c00] + w1 * sl[c10] + w2 * sl[c01] + w3 * sl[c11];
            var pt = w0 * pot[c00] + w1 * pot[c10] + w2 * pot[c01] + w3 * pot[c11];
            var zp = Terrain.sample(dem.data, dem.size, du[p], dv), f = Terrain.scale(zp, zr, slv, pt, m, tcfg);
            m *= f; s1 *= f; s2 *= f;
            if (s1 < FL) s1 = FL; if (s2 < FL) s2 = FL;
          }
        }
        var v = prod.kind === 'prob' ? 100 * Binormal.exceedance(m, s1, s2, prod.t)
              : prod.kind === 'pct' ? Binormal.percentile(m, s1, s2, prod.q / 100) : m;
        if (v < first) continue;
        var col = pal.rgb[binIndex(pal.stops, v)], o2 = (q * n + p) * 4;
        out[o2] = col[0]; out[o2 + 1] = col[1]; out[o2 + 2] = col[2]; out[o2 + 3] = a;
      }
    }
    return out;
  }

  /* Parameters at an exact Web-Mercator position (metres), interpolated exactly like renderTile does for a pixel centre
     (bilinear, or Catmull-Rom when `cubic`): {m, s1, s2, zr, pot, sl}, with zr/pot/sl null where unavailable.
     Returns null outside the grid or where there is no signal.  Used by the cursor tooltip. */
  function sampleAt(P, merc, x, y, cubic, zref) {
    var W = P.W, H = P.H, FL = Binormal.S_FLOOR, g = (x - merc.xmin) / merc.px - 0.5, h = (merc.ymax - y) / merc.px - 0.5;
    var i = Math.floor(g), j = Math.floor(h), fx = g - i, fy = h - j;
    if (i < -1 || i >= W || j < -1 || j >= H) return null;
    var c = [i >= 0 && j >= 0 ? j * W + i : -1, i + 1 < W && j >= 0 ? j * W + i + 1 : -1,
             i >= 0 && j + 1 < H ? (j + 1) * W + i : -1, i + 1 < W && j + 1 < H ? (j + 1) * W + i + 1 : -1], k;
    var live = false;
    for (k = 0; k < 4; k++) if (c[k] >= 0 && (P.mode[c[k]] >= P.zeroMode || P.hi[c[k]] >= P.zeroHi)) live = true;
    if (!live) return null;
    var w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], m, s1, s2;
    if (cubic && i >= 1 && i + 2 < W && j >= 1 && j + 2 < H) {
      var wx = [0, 0, 0, 0], wy = [0, 0, 0, 0], a, b;
      catmull(fx, wx); catmull(fy, wy);
      m = s1 = s2 = 0;
      for (b = 0; b < 4; b++) for (a = 0; a < 4; a++) {
        var cc = (j - 1 + b) * W + i - 1 + a, wt = wy[b] * wx[a];
        m += wt * P.mode[cc]; s1 += wt * P.lo[cc]; s2 += wt * P.hi[cc];
      }
      if (m < 0) m = 0; if (s1 < FL) s1 = FL; if (s2 < FL) s2 = FL;
    } else {
      m = s1 = s2 = 0;
      for (k = 0; k < 4; k++) { m += c[k] >= 0 ? w[k] * P.mode[c[k]] : 0; s1 += w[k] * (c[k] >= 0 ? P.lo[c[k]] : FL); s2 += w[k] * (c[k] >= 0 ? P.hi[c[k]] : FL); }
    }
    if (m < P.zeroMode && s2 < P.zeroHi) return null;
    var out = { m: m, s1: s1, s2: s2, zr: null, pot: null, sl: null };
    if (c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[3] >= 0) {
      var mix = function (arr) { return arr ? w[0] * arr[c[0]] + w[1] * arr[c[1]] + w[2] * arr[c[2]] + w[3] * arr[c[3]] : null; };
      out.zr = mix(zref); out.pot = mix(P.pot); out.sl = mix(P.sl);
    }
    return out;
  }

  /* Value of a product ('prob' %, 'pct' in, 'mode' in) for distribution parameters. */
  function valueOf(product, m, s1, s2) {
    return product.kind === 'prob' ? 100 * Binormal.exceedance(m, s1, s2, product.t)
         : product.kind === 'pct' ? Binormal.percentile(m, s1, s2, product.q / 100) : m;
  }

  /* Factor the terrain adjustment applies at one data cell for a point at elevation zp (read-outs). */
  function terrainFactor(params, i, zref, zp, adj) {
    if (!adj || !zref || !params.pot || !params.sl) return 1;
    return Terrain.scale(zp, zref[i], params.sl[i], params.pot[i], params.mode[i], adj);
  }

  function pointReadout(params, i, thresholds, percentiles) {
    var m = params.mode[i], s1 = params.lo[i], s2 = params.hi[i];
    return {
      mode: m,
      probs: thresholds.map(function (t) { return [t, 100 * Binormal.exceedance(m, s1, s2, t)]; }),
      pcts: percentiles.map(function (q) { return [q, Binormal.percentile(m, s1, s2, q / 100)]; })
    };
  }

  function lonLatToPixel(lon, lat, manifest, mercator) {
    var R = 6378137, x = R * lon * Math.PI / 180, y = R * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
    var col = Math.floor((x - mercator.xmin) / mercator.px), row = Math.floor((mercator.ymax - y) / mercator.px);
    var w = manifest.size[0], h = manifest.size[1];
    if (col < 0 || row < 0 || col >= w || row >= h) return -1;
    return row * w + col;
  }

  return { palettes: palettes, binIndex: binIndex, paramsFromBin: paramsFromBin, field: field, renderTile: renderTile, sampleAt: sampleAt, valueOf: valueOf,
           terrainFactor: terrainFactor, pointReadout: pointReadout, lonLatToPixel: lonLatToPixel };
});
