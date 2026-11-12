/* Terrain handling for the viewer, DOM-free (testable in Node): Terrarium elevation decoding, the terrain adjustment of
 * snowfall, hill-shading and a small tile cache.  Elevation source: AWS Terrain Tiles (Terrarium encoding). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PWPFTerrain = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* Terrarium: elevation (m) = R * 256 + G + B / 256 - 32768.  rgba: Uint8 RGBA pixels of a 256 x 256 tile. */
  function decodeTerrarium(rgba, n) {
    var out = new Float32Array(n);
    for (var i = 0, j = 0; i < n; i++, j += 4) out[i] = rgba[j] * 256 + rgba[j + 1] + rgba[j + 2] / 256 - 32768;
    return out;
  }

  var DEFAULTS = { precip_gradient_per_km: 0.2, precip_factor_range: [0.75, 1.35], snow_level_width_m: 100, max_phase_ratio: 3.0 };
  function config(c) {
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    for (k in (c || {})) if (c[k] !== undefined && c[k] !== null) o[k] = c[k];
    return o;
  }

  function snowFrac(z, sl, w) {                    // 0..1: how much of the precipitation falls as snow at height z
    return 1 / (1 + Math.exp(-(z - sl) / w));
  }

  /* Multiplier for the snowfall distribution (mode and both sigmas) at a point whose true elevation is `zp`, inside a
     model cell whose terrain height is `zr`:
       - elevation gradient: snowfall x (1 + gradient * (zp - zr)), limited to precip_factor_range;
       - rain/snow line: ratio of the snow fraction at zp to that at zr (logistic of width snow_level_width_m around the
         snow level `sl`), regularised (+0.02) so a cell far from the line is left alone and capped at max_phase_ratio;
       - the result never exceeds (potential snowfall `pot` x elevation factor) / mode: you cannot get more snow than
         all the liquid falling as snow.
     A point at the model terrain height gets exactly 1. */
  function scale(zp, zr, sl, pot, mode, cfg) {
    var dz = zp - zr;
    if (dz > 1500) dz = 1500; else if (dz < -1500) dz = -1500;
    var r = cfg.precip_factor_range, fq = 1 + cfg.precip_gradient_per_km * dz / 1000;
    if (fq < r[0]) fq = r[0]; else if (fq > r[1]) fq = r[1];
    var w = cfg.snow_level_width_m, eps = 0.02;
    var rho = (snowFrac(zp, sl, w) + eps) / (snowFrac(zr, sl, w) + eps);
    if (rho > cfg.max_phase_ratio) rho = cfg.max_phase_ratio;
    var k = fq * rho;
    if (pot > 0 && mode > 0) { var cap = fq * pot / mode; if (k > cap) k = cap; }
    return k;
  }

  /* Hill-shading of a (size x size) elevation grid; returns RGBA: black where shaded, white where lit, transparent
     on flat ground.  cellM = ground size of one cell (m), exag = vertical exaggeration; light from azimuth deg
     (clockwise from north) at altitude deg. */
  function hillshade(dem, size, cellM, exag, azimuth, altitude) {
    var az = (azimuth === undefined ? 315 : azimuth) * Math.PI / 180, alt = (altitude === undefined ? 45 : altitude) * Math.PI / 180;
    var lx = Math.sin(az) * Math.cos(alt), ly = Math.cos(az) * Math.cos(alt), lz = Math.sin(alt);
    var flat = lz, out = new Uint8ClampedArray(size * size * 4), r, c;
    for (r = 0; r < size; r++) {
      var r0 = r > 0 ? r - 1 : r, r1 = r < size - 1 ? r + 1 : r;
      for (c = 0; c < size; c++) {
        var c0 = c > 0 ? c - 1 : c, c1 = c < size - 1 ? c + 1 : c;
        var zE = exag * (dem[r * size + c1] - dem[r * size + c0]) / ((c1 - c0 || 1) * cellM);
        var zN = exag * (dem[r0 * size + c] - dem[r1 * size + c]) / ((r1 - r0 || 1) * cellM);
        var s = (-zE * lx - zN * ly + lz) / Math.sqrt(1 + zE * zE + zN * zN), d = s - flat, o = (r * size + c) * 4;
        if (d < 0) { out[o + 3] = Math.min(255, -d * 1.1 * 255); }
        else { out[o] = out[o + 1] = out[o + 2] = 255; out[o + 3] = Math.min(255, d * 0.6 * 255); }
      }
    }
    return out;
  }

  /* LRU cache of terrain tiles keyed z/x/y; `loader(z, x, y)` returns a Promise of a Float32Array (256 x 256).
     Concurrent requests for one tile share a single load; failures are not cached. */
  function TileCache(loader, max) {
    var map = new Map(), pending = new Map();
    this.get = function (z, x, y) {
      var key = z + '/' + x + '/' + y;
      if (map.has(key)) { var v = map.get(key); map.delete(key); map.set(key, v); return Promise.resolve(v); }
      if (pending.has(key)) return pending.get(key);
      var p = loader(z, x, y).then(function (d) {
        pending.delete(key); map.set(key, d);
        while (map.size > max) map.delete(map.keys().next().value);
        return d;
      }, function (e) { pending.delete(key); throw e; });
      pending.set(key, p);
      return p;
    };
    this.size = function () { return map.size; };
  }

  /* Bilinear elevation at fractional tile pixel (u, v) of a 256-grid, clamped to the tile. */
  function sample(dem, size, u, v) {
    if (u < 0) u = 0; else if (u > size - 1) u = size - 1;
    if (v < 0) v = 0; else if (v > size - 1) v = size - 1;
    var i = Math.floor(u), j = Math.floor(v), i1 = i + 1 < size ? i + 1 : i, j1 = j + 1 < size ? j + 1 : j, fx = u - i, fy = v - j;
    return (dem[j * size + i] * (1 - fx) + dem[j * size + i1] * fx) * (1 - fy) + (dem[j1 * size + i] * (1 - fx) + dem[j1 * size + i1] * fx) * fy;
  }

  return { decodeTerrarium: decodeTerrarium, config: config, snowFrac: snowFrac, scale: scale, hillshade: hillshade,
           TileCache: TileCache, sample: sample, DEFAULTS: DEFAULTS };
});
