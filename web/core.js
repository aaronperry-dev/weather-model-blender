/* DOM-free viewer logic: palettes, field computation, point read-out.  Testable in Node. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./binormal.js'));
  else root.PWPFCore = factory(root.Binormal);
})(typeof self !== 'undefined' ? self : this, function (Binormal) {
  'use strict';

  var PROB_STOPS = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  var PROB_COLORS = ['#cfe8f3', '#93cfe6', '#5fb5e0', '#3b8fd6', '#4f6fc9', '#7a4fba', '#a63fa8', '#cf3c86', '#e8505b'];
  var SNOW_STOPS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36, 48];
  var SNOW_COLORS = ['#d6ecf9', '#a7d3f0', '#74b4e6', '#4a92d9', '#2f6fc6', '#3a4cb5', '#6b3fae', '#8e2c9f', '#b02a86', '#cf3d63', '#e8622f', '#f59b25', '#fbd24a'];

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

  /* RGBA(+decoded params) from the data PNG's RGBA pixel buffer. */
  function decodeParams(rgba, enc) {
    var lut = Binormal.decodeLUT(enc), n = rgba.length / 4;
    var mode = new Float32Array(n), lo = new Float32Array(n), hi = new Float32Array(n);
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      mode[i] = lut[rgba[j]]; lo[i] = Math.max(lut[rgba[j + 1]], Binormal.S_FLOOR); hi[i] = Math.max(lut[rgba[j + 2]], Binormal.S_FLOOR);
    }
    /* "no signal" = encoded zero: mode below the first non-zero level and sigma_high below ~1.5 levels
       (scales with the encoding, so it works for snowfall A=0.5/200 in and ice A=0.02/2 in) */
    return { mode: mode, lo: lo, hi: hi, n: n, zeroMode: lut[1] * 0.99, zeroHi: Math.max(lut[1] * 1.5, Binormal.S_FLOOR * 1.01) };
  }

  /* product: {kind:'prob', t:inches} | {kind:'pct', q:0..100} | {kind:'mode'}, plus pal:'snow'|'ice' for amount maps
   * returns {rgba: Uint8ClampedArray, value: Float32Array} (value used for read-outs/tests) */
  function render(params, product, alpha) {
    var n = params.n, out = new Uint8ClampedArray(n * 4), val = new Float32Array(n);
    var pal = product.kind === 'prob' ? palettes.prob : (palettes[product.pal] || palettes.snow);
    var a = Math.round((alpha === undefined ? 0.85 : alpha) * 255);
    for (var i = 0; i < n; i++) {
      var m = params.mode[i], s1 = params.lo[i], s2 = params.hi[i];
      if (m < params.zeroMode && s2 < params.zeroHi) continue;   // no signal (encoded zeros)
      var v, idx;
      if (product.kind === 'prob') { v = 100 * Binormal.exceedance(m, s1, s2, product.t); }
      else if (product.kind === 'pct') { v = Binormal.percentile(m, s1, s2, product.q / 100); }
      else { v = m; }
      val[i] = v;
      idx = binIndex(pal.stops, v);
      if (idx < 0) continue;
      var c = pal.rgb[idx], o = i * 4;
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = a;
    }
    return { rgba: out, value: val };
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

  return { palettes: palettes, binIndex: binIndex, decodeParams: decodeParams, render: render,
           pointReadout: pointReadout, lonLatToPixel: lonLatToPixel };
});
