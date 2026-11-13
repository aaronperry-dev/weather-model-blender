/* Leaflet front-end.  All numbers shown come from the .bin.gz data layers (bin.js) + binormal maths in binormal.js/core.js. */
(function () {
  'use strict';
  var DATA = 'data/';
  var state = { win: null, prod: 'snow', kind: 'mode', thr: null, pct: 50, pt: null, adj: true, shade: true, minFt: 0 };   // minFt: hide data where the ground is lower than this many feet (0 = off)
  var manifest, map, overlay, shown, shadeLayer, cache = {}, cacheOrder = [], current = null, mercator, zref = null, zrefP = null;

  function $(id) { return document.getElementById(id); }
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  /* Everything the viewer shows is imperial: snowfall in inches, heights in feet, temperatures in degrees F, distances in miles. */
  var M2FT = 3.28084, FT2M = 0.3048, MAX_FT = 20000;
  function ftNum(m) { return Math.round(m * M2FT).toLocaleString('en-US'); }
  function ftTxt(m) { return ftNum(m) + ' ft'; }
  function minElevM() { return state.minFt > 0 ? state.minFt * FT2M : null; }
  function utc(iso) { return new Date(/(Z|[+-]\d\d:?\d\d)$/.test(iso) ? iso : iso + 'Z'); }   // naive ISO strings are UTC
  function fmtTime(iso) {
    var d = utc(iso);
    return d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' + d.getUTCFullYear() + ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + 'Z';
  }
  function fmtShort(iso) {
    var d = utc(iso);
    return DOW[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' + pad(d.getUTCHours()) + 'Z';
  }
  function parseHash() {
    (location.hash.slice(1) || '').split('&').forEach(function (kv) {
      var p = kv.split('=');
      if (p[0] === 'w') state.win = p[1];
      if (p[0] === 'x') state.prod = p[1];
      if (p[0] === 'k') state.kind = p[1];
      if (p[0] === 't') state.thr = +p[1];
      if (p[0] === 'q') state.pct = +p[1];
      if (p[0] === 'ta') state.adj = p[1] !== '0';
      if (p[0] === 'sh') state.shade = p[1] !== '0';
      if (p[0] === 'el') state.minFt = Math.max(0, Math.min(MAX_FT, Math.round(+p[1]) || 0));
      if (p[0] === 'p') { var ll = p[1].split(','); if (ll.length === 2) state.pt = { lat: +ll[0], lng: +ll[1] }; }
    });
  }
  function writeHash() {
    history.replaceState(null, '', '#w=' + state.win + '&x=' + state.prod + '&k=' + state.kind + '&t=' + state.thr + '&q=' + state.pct + '&ta=' + (state.adj ? 1 : 0) + '&sh=' + (state.shade ? 1 : 0) + (state.minFt > 0 ? '&el=' + state.minFt : '') +
      (state.pt ? '&p=' + state.pt.lat.toFixed(3) + ',' + state.pt.lng.toFixed(3) : ''));
  }

  function curWin() { return manifest.windows.filter(function (x) { return x.id === state.win; })[0]; }
  function prodInfo() { return manifest.products[state.prod]; }
  function lengths() {
    var s = {};
    manifest.windows.forEach(function (w) { s[w.length_h] = 1; });
    return Object.keys(s).map(Number).sort(function (a, b) { return a - b; });
  }
  var MODEL = { gefs: 'GEFS', gefs_mean: 'GEFS mean', cmce: 'CMC GEPS', rdps: 'CMC RDPS', gdps: 'CMC GDPS', ifs_ens: 'ECMWF IFS ENS', ifs_hres: 'ECMWF IFS HRES', gfs: 'GFS',
                hrrr: 'HRRR', rrfs: 'RRFS', rrfs_ens: 'RRFS ensemble', syn_global: 'Synthetic global', syn_cam: 'Synthetic CAM' };
  function modelName(g) { return MODEL[g] || g; }
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function buildMembers() {
    var rank = function (x) { return x.ok ? 2 : (x.bins_ok > 0 ? 1 : 0); };        // problems first, then by model
    var ms = (manifest.members || []).map(function (x, i) { return [x, i]; }).sort(function (a, b) {
      return rank(a[0]) - rank(b[0]) || a[1] - b[1]; }).map(function (p) { return p[0]; });
    var sum = $('membersSummary'), tb = $('membersTable');
    if (!ms.length) { $('members').hidden = true; return; }
    var full = ms.filter(function (x) { return x.ok; }).length, part = ms.filter(function (x) { return !x.ok && x.bins_ok > 0; }).length;
    var none = ms.length - full - part, bits = [];
    sum.textContent = full + ' of ' + ms.length + ' members complete' + (part ? ' · ' + part + ' partial' : '') + (none ? ' · ' + none + ' missing' : '') +
      '. Each period is blended from the members that delivered all of its data (weights renormalised); ensemble size per period is shown under the date chips.';
    var html = '<tr><th>Model</th><th>Member</th><th>Run (UTC)</th><th title="6-h bins delivered, 0 to 72 h">Data</th><th>Status</th><th>Notes</th></tr>';
    ms.forEach(function (x) {
      var cov = '';
      for (var i = 0; i < x.bins.length; i++) cov += '<i class="' + (x.bins[i] === '1' ? 'y' : '') + '" title="+' + i * 6 + '–' + (i + 1) * 6 + ' h"></i>';
      var cls = x.ok ? 'ok' : (x.bins_ok > 0 ? 'part' : 'bad');
      var txt = x.ok ? 'Used' : (x.bins_ok > 0 ? 'Partial (' + x.bins_ok + '/' + x.bins_total + ')' : 'Missing');
      var note = [x.note ? esc(x.note) : '', x.error ? esc(x.error) : '',
                  (!x.note && !x.error && x.fallback) ? 'Uses ' + esc(modelName(x.fallback)) + ' beyond +' + x.max_fxx_h + ' h from run start' : ''].filter(Boolean).join(' · ');
      html += '<tr><td>' + esc(modelName(x.group)) + '</td><td>' + esc(x.member || 'det.') + '</td><td>' + fmtShort(x.init) +
        '</td><td><span class="cov">' + cov + '</span></td><td class="st ' + cls + '">' + txt + '</td><td class="note">' + note + '</td></tr>';
    });
    tb.innerHTML = html;
  }

  function skipNote() {
    var sk = manifest.windows_skipped || [], box = $('skipnote');
    var t0 = utc(manifest.cycle).getTime();
    var lab = sk.filter(function (x) { return /^s(\d+)_f(\d+)$/.test(x.id); }).map(function (x) {
      var m = /^s(\d+)_f(\d+)$/.exec(x.id);
      return m[1] + ' h ending ' + fmtShort(new Date(t0 + (+m[2]) * 3600000).toISOString());
    });
    box.hidden = !lab.length;
    if (lab.length) box.textContent = 'Not available this run (too few members with complete data): ' + lab.join('; ') + '.';
  }

  /* Optional run diagnostics (calibration.txt, written by the pipeline): shown in a collapsed panel under the member table. */
  function loadCalibration() {
    if (!manifest.calibration) return;
    fetch(DATA + manifest.calibration + '?v=' + encodeURIComponent(manifest.generated))
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
      .then(function (t) { $('calibText').textContent = t; $('calib').hidden = false; })
      .catch(function () { /* diagnostics are optional */ });
  }

  function hasIce() { return manifest.windows.some(function (w) { return !!w.ice_file; }); }

  /* Load (and cache, newest 3) the decoded parameters of window w for the active hazard: a gzip .bin layer (see
     pwpf_blender/export.py); resolves null if the window has no layer for it (e.g. no freezing rain in that period). */
  function loadLayer(w, prod) {
    var info = manifest.products[prod], file = w[info.file_key], key = w.id + '|' + prod;
    if (!file) return Promise.resolve(null);
    if (cache[key]) return Promise.resolve(cache[key]);
    return PWPFBin.load(DATA + file + '?v=' + encodeURIComponent(manifest.generated)).then(function (bin) {
      cache[key] = PWPFCore.paramsFromBin(bin);
      cacheOrder.push(key);
      while (cacheOrder.length > 3) delete cache[cacheOrder.shift()];
      return cache[key];
    });
  }

  /* Model terrain per data cell (static for the run), needed by the terrain adjustment. */
  function loadTerrainRef() {
    if (!manifest.terrain || !manifest.terrain.enabled) return Promise.resolve(null);
    if (!zrefP) {
      zrefP = PWPFBin.load(DATA + manifest.terrain.file + '?v=' + encodeURIComponent(manifest.generated))
        .then(function (b) { zref = b.planes[6] || null; return zref; })
        .catch(function (e) { console.warn('terrain reference unavailable', e); return null; });
    }
    return zrefP;
  }

  /* ---- AWS Terrain Tiles (Terrarium PNGs: elevation = R*256 + G + B/256 - 32768 m) ---- */
  var demFails = 0, demOk = 0, demCache = null;
  function pixelsOf(url) {
    function fromBitmap(bmp) {
      var c = document.createElement('canvas'); c.width = 256; c.height = 256;
      var cx = c.getContext('2d', { willReadFrequently: true });
      cx.drawImage(bmp, 0, 0);
      return cx.getImageData(0, 0, 256, 256).data;
    }
    var ms = window.__pwpfDemTimeoutMs || 20000;       // a request that never answers must fail, not leave its map tile hidden
    if (window.createImageBitmap) {                     // no colour conversion / premultiplication: the PNG holds data
      var ctl = window.AbortController ? new AbortController() : null, timer = ctl ? setTimeout(function () { ctl.abort(); }, ms) : null;
      return fetch(url, { mode: 'cors', signal: ctl ? ctl.signal : undefined })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); })
        .then(function (b) { return createImageBitmap(b, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }); })
        .then(function (bmp) { clearTimeout(timer); return fromBitmap(bmp); }, function (e) { clearTimeout(timer); throw e; });
    }
    return new Promise(function (resolve, reject) {
      var img = new Image(), timer = setTimeout(function () { img.src = ''; reject(new Error('terrain tile timed out')); }, ms);
      img.crossOrigin = 'anonymous';
      img.onload = function () { clearTimeout(timer); resolve(fromBitmap(img)); };
      img.onerror = function () { clearTimeout(timer); reject(new Error('terrain tile failed')); };
      img.src = url;
    });
  }
  function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function pixelsWithRetry(url) {                      // transient errors (503 SlowDown, dropped connections) get two more tries
    var tries = [0, 500, 1500];
    function go(i) {
      return wait(tries[i]).then(function () { return pixelsOf(url); }).catch(function (e) { if (i + 1 < tries.length) return go(i + 1); throw e; });
    }
    return go(0);
  }
  function demTile(z, x, y) {
    if (!demCache) demCache = new PWPFTerrain.TileCache(function (zz, xx, yy) {
      var url = manifest.terrain.dem_url.replace('{z}', zz).replace('{x}', xx).replace('{y}', yy);
      return pixelsWithRetry(url).then(function (px) { demOk++; return PWPFTerrain.decodeTerrarium(px, 65536); },
                                function (e) { demFails++; throw e; });
    }, 220);
    return demCache.get(z, x, y);
  }
  function demDown() { return demOk === 0 && demFails >= 6; }
  /* The elevation tile covering map tile (z, x, y): same zoom, or the finest available (dem_max_zoom) parent. */
  function demFor(c) {
    var tz = Math.min(c.z, manifest.terrain.dem_max_zoom), f = Math.pow(2, c.z - tz), tx = Math.floor(c.x / f), ty = Math.floor(c.y / f);
    return demTile(tz, tx, ty).then(function (data) { return { data: data, tz: tz, tx: tx, ty: ty, size: 256 }; });
  }
  function elevZoom() { return Math.min(Math.max(Math.round(map.getZoom()), manifest.terrain.min_zoom), manifest.terrain.dem_max_zoom); }
  function elevationAt(latlng) {                         // metres, from the tile the map shows at this zoom (usually already cached)
    var z = elevZoom(), n = Math.pow(2, z);
    var xw = (latlng.lng + 180) / 360 * n, lat = latlng.lat * Math.PI / 180;
    var yw = (1 - Math.log(Math.tan(lat) + 1 / Math.cos(lat)) / Math.PI) / 2 * n;
    var tx = Math.floor(xw), ty = Math.floor(yw);
    return demTile(z, tx, ty).then(function (d) { return PWPFTerrain.sample(d, 256, (xw - tx) * 256 - 0.5, (yw - ty) * 256 - 0.5); });
  }
  function adjustOn() { return !!(state.adj && manifest.terrain && manifest.terrain.enabled && zref && current && current.pot && !demDown()); }
  function adjustHere() { return adjustOn() && Math.round(map.getZoom()) >= manifest.terrain.min_zoom; }     // the layer only adjusts from min_zoom
  function elevationAvail() { return !!(manifest.terrain && manifest.terrain.enabled && !demDown()); }
  function elevationOn() { return !!(manifest.terrain && manifest.terrain.enabled && !demDown() && Math.round(map.getZoom()) >= manifest.terrain.min_zoom); }

  function product() {
    if (state.kind === 'prob') return { kind: 'prob', t: state.thr, pal: state.prod };
    if (state.kind === 'pct') return { kind: 'pct', q: state.pct, pal: state.prod };
    return { kind: 'mode', pal: state.prod };
  }

  function drawLegend() {
    var pal = state.kind === 'prob' ? PWPFCore.palettes.prob : PWPFCore.palettes[state.prod];
    var unit = state.kind === 'prob' ? '%' : ' in';
    var html = '';
    for (var i = pal.stops.length - 1; i >= 0; i--) {
      var next = pal.stops[i + 1];
      var label = (next === undefined ? '≥ ' + pal.stops[i] : pal.stops[i] + ' – ' + next) + unit;
      html += '<div class="row"><span class="sw" style="background:' + pal.colors[i] + '"></span>' + label + '</div>';
    }
    if (state.minFt > 0) html += '<div class="muted small" style="margin-top:6px">Hidden where the ground is below ' + state.minFt.toLocaleString('en-US') + ' ft.</div>';
    $('legend').innerHTML = html;
  }

  function clearOverlay() { [overlay, shown].forEach(function (l) { if (l) map.removeLayer(l); }); overlay = shown = null; }

  /* The data layer: map tiles coloured per screen pixel in the browser (PWPFCore.renderTile) from the interpolated
     parameters, with the terrain adjustment where the elevation tiles are available. */
  var DataLayer = L.GridLayer.extend({
    options: { pane: 'overlayPane', tileSize: 256, updateWhenZooming: false, keepBuffer: 1, noWrap: true },
    /* `adjust` is fixed for the life of the layer (a toggle builds a new layer, see setField), so a tile never has to
       guess which variant it belongs to. */
    setData: function (params, product, pal, adjust, minElev) {
      this._params = params; this._product = product; this._pal = pal; this._adjust = !!adjust;
      this._minElev = typeof minElev === 'number' ? minElev : null;            // elevation filter (m), fixed for the life of the layer too
    },
    createTile: function (coords, done) {
      var self = this, size = this.getTileSize().x, sc = this.options.scale, n = size * sc, finished = false;
      var c = L.DomUtil.create('canvas', 'leaflet-tile');
      c.width = c.height = n;
      /* Leaflet ignores a done() called before createTile returns (the tile would stay hidden), and a tile that never calls
         it stays hidden forever: so done() is always called, once, asynchronously - whatever happens while painting. */
      function finish() { if (finished) return; finished = true; setTimeout(function () { done(null, c); }, 0); }
      var needAdj = self._adjust && coords.z >= manifest.terrain.min_zoom, needFilter = self._minElev !== null;
      function draw(dem) {
        var adj = !!(dem && needAdj);
        var rgba = PWPFCore.renderTile({ params: self._params, merc: manifest.mercator, pal: self._pal, product: self._product,
          alpha: 1, z: coords.z, tx: coords.x, ty: coords.y, size: size, scale: sc,
          zref: adj ? zref : null, dem: dem, adjust: adj ? PWPFTerrain.config(manifest.terrain) : null, minElev: self._minElev });
        c.getContext('2d').putImageData(new ImageData(rgba, n, n), 0, 0);
        c.dataset.adjusted = adj ? '1' : '0';
        if (needFilter && !dem) { c.dataset.elev = 'missing'; elevMissing++; setTimeout(elevNote, 0); }      // nothing is drawn: we cannot place it above the line
        else if (needFilter) c.dataset.elev = 'ok';
      }
      function paint(dem) {
        try {
          if (self._params) draw(dem);
        } catch (e) {
          console.error('data tile failed' + (dem ? ' (with terrain adjustment); drawing it without' : ''), e);
          if (dem && !needFilter) { try { draw(null); } catch (e2) { console.error(e2); } }
        }
        finish();
      }
      if ((needAdj || needFilter) && !demDown()) {
        demFor(coords).then(paint, function () { paint(null); });
      } else paint(null);
      return c;
    }
  });

  /* Optional hill-shading from the same elevation tiles (pane above the data, below the borders). */
  var ShadeLayer = L.GridLayer.extend({
    options: { pane: 'shade', tileSize: 256, noWrap: true, updateWhenZooming: false, keepBuffer: 1, opacity: 0.6 },
    createTile: function (coords, done) {
      var n = 256 * this.options.scale, c = L.DomUtil.create('canvas', 'leaflet-tile');
      c.width = c.height = n;
      demFor(coords).then(function (dem) {
        var lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (dem.ty + 0.5) / Math.pow(2, dem.tz))));
        var cellM = 40075016.686 / (256 * Math.pow(2, dem.tz)) * Math.cos(lat);
        var key = dem.tz + '/' + dem.tx + '/' + dem.ty, tmp = shadeCache[key];
        if (!tmp) {
          tmp = document.createElement('canvas'); tmp.width = tmp.height = 256;
          tmp.getContext('2d').putImageData(new ImageData(PWPFTerrain.hillshade(dem.data, 256, cellM, 2.0), 256, 256), 0, 0);
          shadeCache[key] = tmp; shadeKeys.push(key);
          while (shadeKeys.length > 60) delete shadeCache[shadeKeys.shift()];
        }
        var f = Math.pow(2, coords.z - dem.tz), sub = 256 / f;
        c.getContext('2d').drawImage(tmp, (coords.x / f - dem.tx) * 256, (coords.y / f - dem.ty) * 256, sub, sub, 0, 0, n, n);
        done(null, c);
      }, function () { done(null, c); });
      return c;
    }
  });
  var shadeCache = {}, shadeKeys = [], elevMissing = 0;

  var TERRAIN_ATTR = 'Terrain: AWS Terrain Tiles (Mapzen; USGS 3DEP, SRTM, GMTED2010, ETOPO1)';
  function dprScale() { return Math.min(2, Math.max(1, Math.round(window.devicePixelRatio || 1))); }

  /* A change of data or of the terrain adjustment builds a NEW layer under the old one; the old layer is removed only when
     the new one has painted all its tiles (or after 10 s).  The map never goes blank while elevation tiles load, and a
     layer that cannot finish can never take the visible one away. */
  function setField(params, prod, pal) {
    var layer = new DataLayer({ bounds: L.latLngBounds(manifest.bounds), scale: dprScale(), opacity: +$('opacity').value,
                                minZoom: 0, maxZoom: 18,
                                attribution: manifest.terrain && manifest.terrain.enabled ? TERRAIN_ATTR : '' });
    elevMissing = 0;
    layer.setData(params, prod, pal, adjustOn(), minElevM());
    if (overlay && overlay !== shown) map.removeLayer(overlay);          // superseded before it ever showed
    overlay = layer;
    var swapped = false;
    function swap() {
      if (swapped) return;
      swapped = true;
      if (overlay !== layer) return;                                     // an even newer layer took over
      if (shown && shown !== layer) map.removeLayer(shown);
      shown = layer;
    }
    layer.once('load', swap);
    setTimeout(swap, 10000);
    layer.addTo(map);
  }

  function syncShade() {
    var want = state.shade && manifest.terrain && manifest.terrain.enabled && manifest.terrain.hillshade && !demDown();
    if (want && !shadeLayer) {
      if (!map.getPane('shade')) { map.createPane('shade'); map.getPane('shade').style.zIndex = 410; map.getPane('shade').style.pointerEvents = 'none'; }
      shadeLayer = new ShadeLayer({ scale: dprScale(), minZoom: 3, maxZoom: 18, attribution: TERRAIN_ATTR }).addTo(map);
    } else if (!want && shadeLayer) { map.removeLayer(shadeLayer); shadeLayer = null; }
  }

  function terrainNote() {
    var box = $('terrainNote');
    if (!manifest.terrain || !manifest.terrain.enabled) { box.hidden = true; return; }
    var msg = '';
    if (demDown() && (state.adj || state.shade)) msg = 'Terrain tiles could not be loaded; showing the un-adjusted data.';
    else if (state.adj && current && !current.pot) msg = 'No terrain adjustment for this layer.';
    else if (state.adj && map && Math.round(map.getZoom()) < manifest.terrain.min_zoom) msg = 'Terrain adjustment applies from zoom level ' + manifest.terrain.min_zoom + ' (zoom in).';
    box.hidden = !msg; box.textContent = msg;
  }

  /* Status line under the elevation filter. */
  function elevNote() {
    var box = $('elevNote');
    if (!box) return;
    var msg = '';
    if (state.minFt > 0) {
      if (demDown()) msg = 'Elevation tiles could not be loaded, so the filter cannot be applied and the data is hidden.';
      else if (elevMissing > 0) msg = 'Some elevation tiles failed to load; those parts of the map are hidden while the filter is on.';
      else msg = 'Showing data only where the ground is at or above ' + state.minFt.toLocaleString('en-US') + ' ft.';
    }
    box.hidden = !msg; box.textContent = msg;
    box.classList.toggle('warn', msg.indexOf('could not') >= 0 || msg.indexOf('failed') >= 0);
  }

  function redraw() {
    var w = curWin(), info = prodInfo();
    $('validity').textContent = w.n_members ? w.n_members + ' ensemble members' + (w.reduced ? ' (reduced ensemble, lower confidence)' : '') : '';
    elevNote();
    $('thrLabel').textContent = state.prod === 'ice' ? 'Ice accretion at least' : 'Snowfall at least';
    drawLegend();
    Promise.all([loadLayer(w, state.prod), loadTerrainRef()]).then(function (res) {
      var params = res[0];
      current = params;
      $('nodata').hidden = !!params;
      if (!params) { clearOverlay(); $('readout').innerHTML = '<h2>Point forecast</h2><div class="muted">No ' +
        info.label.toLowerCase() + ' signal in any ensemble member for this period.</div>'; writeHash(); return; }
      var prod = product(), pal = prod.kind === 'prob' ? PWPFCore.palettes.prob : (PWPFCore.palettes[prod.pal] || PWPFCore.palettes.snow);
      terrainNote();
      setField(params, prod, pal);
      if (state.pt) readout(state.pt);
      writeHash();
    }).catch(function (e) { $('runinfo').textContent = String(e); });
  }

  function fillThresholds() {
    var thr = $('thr'), info = prodInfo();
    thr.innerHTML = '';
    info.thresholds.forEach(function (t) { thr.add(new Option(t, t)); });
    if (info.thresholds.indexOf(state.thr) < 0) state.thr = info.thresholds[Math.min(3, info.thresholds.length - 1)];
    if (state.prod === 'ice' && info.thresholds.indexOf(0.1) >= 0 && !state._thrSet) state.thr = 0.1;
    thr.value = state.thr;
  }

  /* ---- forecast-step slider: one stop per period of the chosen length, with the exact start / end valid times ---- */
  function stepPool() { var len = curWin().length_h; return manifest.windows.filter(function (w) { return w.length_h === len; }); }
  function fmtLocal(iso) {
    try {
      return new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }).format(utc(iso));
    } catch (e) { return ''; }
  }
  function stepHtml(w, idx, n) {
    var loc = fmtLocal(w.start);
    return '<b>Step ' + (idx + 1) + ' of ' + n + '</b> · ' + w.length_h + ' h ending ' + fmtShort(w.end) +
      '<div class="tm"><span>Start</span> ' + fmtTime(w.start) + '</div><div class="tm"><span>End</span> ' + fmtTime(w.end) + '</div>' +
      (loc ? '<div class="muted small">Your time: ' + fmtLocal(w.start) + ' → ' + fmtLocal(w.end) + '</div>' : '') +
      (state.prod === 'ice' && !w.ice_file ? '<div class="muted small">No freezing-rain signal in this period.</div>' : '');
  }
  function syncStepper() {
    var p = stepPool(), cur = curWin(), idx = Math.max(0, p.indexOf(cur)), sl = $('step');
    sl.max = String(p.length - 1); sl.value = String(idx); sl.disabled = p.length < 2;
    $('stepPrev').disabled = idx <= 0; $('stepNext').disabled = idx >= p.length - 1;
    $('stepLabel').innerHTML = stepHtml(cur, idx, p.length);
    $('tickA').textContent = fmtShort(p[0].end); $('tickB').textContent = p.length > 1 ? fmtShort(p[p.length - 1].end) : '';
  }
  var stepTimer = null;
  function goStep(idx, now) {
    var p = stepPool(); idx = Math.max(0, Math.min(p.length - 1, idx));
    $('stepLabel').innerHTML = stepHtml(p[idx], idx, p.length);                  // the label follows the thumb immediately
    clearTimeout(stepTimer);
    var go = function () { if (state.win !== p[idx].id) { state.win = p[idx].id; syncChips(); redraw(); } };
    if (now) go(); else stepTimer = setTimeout(go, 120);                          // while dragging, load only where the thumb rests
  }
  function wireStepper() {
    $('step').oninput = function () { goStep(+$('step').value, false); };
    $('step').onchange = function () { goStep(+$('step').value, true); };
    $('stepPrev').onclick = function () { goStep(stepPool().indexOf(curWin()) - 1, true); };
    $('stepNext').onclick = function () { goStep(stepPool().indexOf(curWin()) + 1, true); };
  }

  /* ---- elevation filter: hide the data where the ground (AWS Terrain Tiles) is lower than the entered height in feet ---- */
  function setMinFt(v, immediate) {
    v = Math.max(0, Math.min(MAX_FT, Math.round(+v) || 0));
    if (v === state.minFt) return;
    state.minFt = v; syncElevChips(); writeHash();
    redraw();
  }
  function syncElevChips() {
    Array.prototype.forEach.call($('elevChips').children, function (b) { b.classList.toggle('on', +b.dataset.ft === state.minFt); });
  }
  function wireElevation() {
    var ei = $('elevFt'), timer = null;
    ei.value = state.minFt > 0 ? state.minFt : '';
    ei.oninput = function () { clearTimeout(timer); timer = setTimeout(function () { setMinFt(ei.value); }, 350); };
    ei.onchange = function () { clearTimeout(timer); setMinFt(ei.value); ei.value = state.minFt > 0 ? state.minFt : ''; };
    $('elevClear').onclick = function () { clearTimeout(timer); ei.value = ''; setMinFt(0); };
    Array.prototype.forEach.call($('elevChips').children, function (b) {
      b.onclick = function () { clearTimeout(timer); ei.value = b.dataset.ft; setMinFt(+b.dataset.ft); };
    });
    syncElevChips();
  }

  function buildControls() {
    var lbox = $('lengths');
    lengths().forEach(function (L) {
      var b = document.createElement('button');
      b.textContent = L + ' h'; b.dataset.len = L;
      b.onclick = function () {                       // same end time if the other length has it, else the nearest later one
        var cur = curWin();
        var pool = manifest.windows.filter(function (w) { return w.length_h === L; });
        var pick = pool.filter(function (w) { return w.end_h >= cur.end_h; })[0] || pool[pool.length - 1];
        state.win = pick.id; syncChips(); redraw();
      };
      lbox.appendChild(b);
    });
    var hz = $('hazards');
    Array.prototype.forEach.call(hz.children, function (b) {
      if (b.dataset.prod === 'ice' && !hasIce()) { b.hidden = true; return; }
      b.onclick = function () { state.prod = b.dataset.prod; state._thrSet = false; fillThresholds(); syncChips(); redraw(); };
    });
    if (state.prod === 'ice' && !hasIce()) state.prod = 'snow';
    var pct = $('pct');
    manifest.percentiles.forEach(function (q) { pct.add(new Option(q + 'th', q)); });
    pct.value = state.pct;
    fillThresholds();
    $('thr').onchange = function () { state.thr = +$('thr').value; state._thrSet = true; redraw(); };
    pct.onchange = function () { state.pct = +pct.value; redraw(); };
    $('opacity').oninput = function () { [overlay, shown].forEach(function (l) { if (l) l.setOpacity(+$('opacity').value); }); };
    var tOn = !!(manifest.terrain && manifest.terrain.enabled);
    $('terrainBox').hidden = !tOn;
    $('elevBox').hidden = !tOn;
    if (tOn) wireElevation(); else state.minFt = 0;
    wireStepper();
    $('shadeRow').hidden = !(tOn && manifest.terrain.hillshade);
    $('terrainAdj').checked = state.adj; $('shade').checked = state.shade;
    $('terrainAdj').onchange = function () { state.adj = $('terrainAdj').checked; redraw(); };
    $('shade').onchange = function () { state.shade = $('shade').checked; syncShade(); writeHash(); };
    Array.prototype.forEach.call($('products').children, function (b) {
      b.onclick = function () { state.kind = b.dataset.kind; syncChips(); redraw(); };
    });
    syncChips();
  }
  function syncChips() {
    var cur = curWin();
    syncStepper();
    Array.prototype.forEach.call($('lengths').children, function (b) { b.classList.toggle('on', +b.dataset.len === cur.length_h); });
    Array.prototype.forEach.call($('hazards').children, function (b) { b.classList.toggle('on', b.dataset.prod === state.prod); });
    Array.prototype.forEach.call($('products').children, function (b) { b.classList.toggle('on', b.dataset.kind === state.kind); });
    $('thrRow').hidden = state.kind !== 'prob';
    $('pctRow').hidden = state.kind !== 'pct';
  }

  var readTok = 0;
  function readout(latlng) {
    if (!current) return;
    var i = PWPFCore.lonLatToPixel(latlng.lng, latlng.lat, manifest, mercator);
    if (i < 0) return;
    var tok = ++readTok, filt = state.minFt > 0 && elevationAvail();
    if (!filt) showReadout(latlng, i, null);                // with the filter on, wait for the elevation: the point may be hidden
    if (filt || elevationOn()) {
      elevationAt(latlng).then(function (zp) { if (tok === readTok) showReadout(latlng, i, zp); },
                               function () { if (tok === readTok) showReadout(latlng, i, null, filt); });     // keep the plain read-out
    }
  }

  function showReadout(latlng, i, zp, elevFailed) {
    var info = prodInfo(), dec = info.decimals, box = $('readout');
    var FL = Binormal.S_FLOOR, adj = PWPFTerrain.config(manifest.terrain), k = 1;
    var head = '<h2>Point forecast · ' + info.label + '</h2><div class="muted">' + latlng.lat.toFixed(2) + ', ' + latlng.lng.toFixed(2) + '</div>';
    if (current.mode[i] < current.zeroMode && current.hi[i] < current.zeroHi) {
      box.innerHTML = head + '<div class="big">No ' + (state.prod === 'ice' ? 'ice' : 'snow') + ' signal</div>';
      return;
    }
    if (state.minFt > 0 && elevFailed) {
      box.innerHTML = head + '<div class="big">Elevation unavailable</div><div class="muted small">The ' + state.minFt.toLocaleString('en-US') + ' ft elevation filter cannot be checked here, so the map hides this point.</div>';
      return;
    }
    if (state.minFt > 0 && zp !== null && zp !== undefined && zp < state.minFt * FT2M) {
      box.innerHTML = head + '<div class="big">Hidden</div><div class="muted">Ground is ' + ftTxt(zp) + ', below your ' + state.minFt.toLocaleString('en-US') + ' ft elevation filter.</div>';
      return;
    }
    var adjusting = zp !== null && zp !== undefined && adjustHere();
    if (adjusting) k = PWPFCore.terrainFactor(current, i, zref, zp, adj);
    var view = { mode: [current.mode[i] * k], lo: [Math.max(current.lo[i] * k, FL)], hi: [Math.max(current.hi[i] * k, FL)] };
    var thr = state.prod === 'ice' ? info.thresholds : info.thresholds.filter(function (t) { return t >= 1 && t <= 18; });
    var r = PWPFCore.pointReadout(view, 0, thr, [5, 10, 25, 50, 75, 90, 95]);
    var rows = '<tr><th>' + (state.prod === 'ice' ? 'Ice ≥' : 'Snow ≥') + '</th>' + r.probs.map(function (p) { return '<td>' + p[0] + '"</td>'; }).join('') + '</tr>' +
      '<tr><th>Prob.</th>' + r.probs.map(function (p) { return '<td>' + Math.round(p[1]) + '%</td>'; }).join('') + '</tr>';
    var prow = '<tr><th>Pctile</th>' + r.pcts.map(function (p) { return '<td>' + p[0] + '</td>'; }).join('') + '</tr>' +
      '<tr><th>Inches</th>' + r.pcts.map(function (p) { return '<td>' + p[1].toFixed(dec) + '</td>'; }).join('') + '</tr>';
    var terr = (zp === null || zp === undefined) ? ''
      : adjusting ? '<div class="muted small">Terrain adjustment: ground ' + ftTxt(zp) + ' vs model ' + ftTxt(zref[i]) + ' → snowfall × ' + k.toFixed(2) + '</div>'
      : '<div class="muted small">Ground ' + ftTxt(zp) + (zref ? ' (model ' + ftTxt(zref[i]) + ')' : '') + '</div>';
    box.innerHTML = head + '<div class="big">' + r.mode.toFixed(dec) + ' in <span class="muted small">forecast (mode)</span></div>' + terr +
      '<table>' + rows + '</table><table style="margin-top:8px">' + prow + '</table>' +
      '<div class="muted small" style="margin-top:6px">Chance of at least the amount shown; percentiles are amounts not exceeded with that probability.' +
      (state.prod === 'ice' ? ' Ice is flat (elevated horizontal) accretion from a precipitation-rate FRAM; treat as experimental.' : '') + '</div>';
  }

  /* ---- cursor tooltip: exact values under the pointer, computed from the same interpolated field the map shows ---- */
  var tipEl, tipRaf = 0, tipEvt = null, tipTok = 0, pinMarker = null, HALF = 20037508.342789244;
  function fmtLat(v) { return Math.abs(v).toFixed(3) + '°' + (v < 0 ? 'S' : 'N'); }
  function fmtLng(v) { return Math.abs(v).toFixed(3) + '°' + (v < 0 ? 'W' : 'E'); }
  function tipOn(e) {
    tipEvt = e;
    if (!tipRaf) tipRaf = requestAnimationFrame(function () { tipRaf = 0; if (tipEvt) updateTip(tipEvt); });
  }
  function tipOff() { tipTok++; tipEvt = null; if (tipEl) tipEl.hidden = true; }
  function placeTip(pt) {
    var box = map.getContainer(), w = tipEl.offsetWidth, h = tipEl.offsetHeight, x = pt.x + 18, y = pt.y + 18;
    if (x + w > box.clientWidth - 8) x = pt.x - w - 14;
    if (y + h > box.clientHeight - 8) y = pt.y - h - 14;
    tipEl.style.left = Math.max(6, x) + 'px'; tipEl.style.top = Math.max(6, y) + 'px';
  }
  function updateTip(e) {
    if (!current || !tipEl) return;
    var ll = e.latlng, R = 6378137, tok = ++tipTok;
    var x = R * ll.lng * Math.PI / 180, y = R * Math.log(Math.tan(Math.PI / 4 + ll.lat * Math.PI / 360));
    var z = Math.round(map.getZoom()), cellPx = mercator.px * 256 * Math.pow(2, z) / (2 * HALF) * dprScale();
    var s = PWPFCore.sampleAt(current, mercator, x, y, cellPx >= 8, adjustOn() ? zref : null), filt = state.minFt > 0 && elevationAvail();
    var inside = PWPFCore.lonLatToPixel(ll.lng, ll.lat, manifest, mercator) >= 0;
    var head = '<div class="ll">' + fmtLat(ll.lat) + ', ' + fmtLng(ll.lng) + '</div>';
    if (!s) {
      tipEl.innerHTML = head + '<div class="val"><small>' + (inside ? 'No ' + (state.prod === 'ice' ? 'ice' : 'snow') + ' signal here' : 'Outside the forecast domain') + '</small></div>';
      tipEl.hidden = false; placeTip(e.containerPoint);
      if (inside && elevationOn()) elevationAt(ll).then(function (zp) { if (tok === tipTok) { tipEl.innerHTML = head + '<div class="val"><small>No ' + (state.prod === 'ice' ? 'ice' : 'snow') + ' signal here</small></div><div class="terr">Ground ' + ftTxt(zp) + '</div>'; placeTip(e.containerPoint); } }, function () {});
      return;
    }
    function draw(zp, failed) {
      if (filt && (failed || (zp !== null && zp < state.minFt * FT2M))) {             // hidden on the map, so no values here either
        tipEl.innerHTML = head + '<div class="val"><small>' + (failed ? 'Elevation unavailable: hidden by the ' : 'Hidden by the ') + state.minFt.toLocaleString('en-US') + ' ft elevation filter</small></div>' +
          (failed ? '' : '<div class="terr">Ground ' + ftTxt(zp) + '</div>');
        tipEl.hidden = false; placeTip(e.containerPoint);
        return;
      }
      var info = prodInfo(), prod = product(), FL = Binormal.S_FLOOR, k = 1, adjusting = false;
      if (zp !== null && s.zr !== null && s.pot !== null && s.sl !== null && adjustHere()) {
        k = PWPFTerrain.scale(zp, s.zr, s.sl, s.pot, s.m, PWPFTerrain.config(manifest.terrain)); adjusting = true;
      }
      var m = s.m * k, s1 = Math.max(s.s1 * k, FL), s2 = Math.max(s.s2 * k, FL);
      var dec = state.prod === 'ice' ? 3 : 2, unit = state.prod === 'ice' ? 'in of ice' : 'in';
      var pv = PWPFCore.valueOf(prod, m, s1, s2), pal = prod.kind === 'prob' ? PWPFCore.palettes.prob : PWPFCore.palettes[state.prod];
      var bi = PWPFCore.binIndex(pal.stops, pv), sw = bi >= 0 ? '<span class="sw" style="background:' + pal.colors[bi] + '"></span>' : '<span class="sw" style="background:#fff"></span>';
      var title = prod.kind === 'prob' ? 'Chance of at least ' + state.thr + ' in' : prod.kind === 'pct' ? 'Amount not exceeded at ' + state.pct + '% (' + state.pct + 'th percentile)' : 'Forecast (mode)';
      var bigTxt = prod.kind === 'prob' ? pv.toFixed(1) + ' <small>%</small>' : pv.toFixed(dec) + ' <small>' + unit + '</small>';
      var rows = '';
      function row(a, b) { rows += '<tr><td>' + a + '</td><td>' + b + '</td></tr>'; }
      if (prod.kind !== 'mode') row('Forecast (mode)', m.toFixed(dec) + ' in');
      if (prod.kind !== 'prob') row('Chance ≥ ' + state.thr + ' in', (100 * Binormal.exceedance(m, s1, s2, state.thr)).toFixed(1) + ' %');
      [10, 50, 90].forEach(function (q) { row(q + 'th percentile', Binormal.percentile(m, s1, s2, q / 100).toFixed(dec) + ' in'); });
      var terr = zp === null ? '' : adjusting
        ? '<div class="terr">Ground ' + ftTxt(zp) + ' vs model ' + ftTxt(s.zr) + ' → snowfall × ' + k.toFixed(2) + '</div>'
        : '<div class="terr">Ground ' + ftTxt(zp) + (s.zr !== null ? ' (model ' + ftTxt(s.zr) + ')' : '') + '</div>';
      tipEl.innerHTML = head + '<div class="head">' + title + '</div><div class="val">' + sw + bigTxt + '</div><table>' + rows + '</table>' + terr;
      tipEl.hidden = false; placeTip(e.containerPoint);
    }
    if (filt) {                                              // the elevation decides whether this point is shown at all
      elevationAt(ll).then(function (zp) { if (tok === tipTok) draw(zp); }, function () { if (tok === tipTok) draw(null, true); });
      return;
    }
    draw(null);
    if (elevationOn()) elevationAt(ll).then(function (zp) { if (tok === tipTok) draw(zp); }, function () {});
  }
  function setPin(ll) {
    if (!map.getPane('pin')) { map.createPane('pin'); map.getPane('pin').style.zIndex = 460; map.getPane('pin').style.pointerEvents = 'none'; }
    if (!pinMarker) pinMarker = L.circleMarker(ll, { radius: 7, weight: 3, color: '#fff', fillColor: '#d6336c', fillOpacity: 1, interactive: false, pane: 'pin' }).addTo(map);
    else pinMarker.setLatLng(ll);
  }

  function init(m) {
    manifest = m; mercator = m.mercator;
    parseHash();
    if (!state.win || !curWin()) state.win = m.windows[0].id;
    $('runinfo').innerHTML = 'Cycle ' + fmtTime(m.cycle) + ' · <a href="#members" id="toMembers">' + m.members_total + ' ensemble members ↓</a>' +
      (m.members_requested && m.members_requested !== m.members_total ? ' (of ' + m.members_requested + ' requested)' : '') + ' · built ' + fmtTime(m.generated);
    $('toMembers').onclick = function (e) { e.preventDefault(); $('members').scrollIntoView({ behavior: 'smooth', block: 'start' }); };   // (the URL hash is the viewer state)
    $('about').innerHTML = '<b>Notice.</b> ' + m.notice + ' Methodology: multi-model ensemble → QPF (PRISM-downscaled) × precipitation type × SLR → binormal fit (mode = ' +
      m.mode_method + ' forecast, variance from the ensemble). Freezing rain: simplified FRAM ice-to-liquid ratio.' +
      (m.terrain && m.terrain.enabled ? ' <b>Terrain adjustment</b> (browser-side, optional): snowfall is rescaled at each map pixel for the ground elevation from AWS Terrain Tiles relative to the model terrain (1.6-mile grid) — an elevation gradient (+' + Math.round(100 * m.terrain.precip_gradient_per_km * FT2M) + "% per 1,000 ft, limited) and the rain/snow line at the model snow level; never more than all the liquid falling as snow. These are our assumptions, not part of WPC\'s method." : '') +
      (m.accumulation ? ' <b>Accumulation</b> (our assumption, not part of WPC\'s method): snow in marginal air (ground wet-bulb above about 30 °F) is reduced, and warm early-season ground melts the first ~' + (m.accumulation.ground_melt_in_per_c * 5 / 9).toFixed(2) + ' in per °F of soil temperature of snow.' : '') +
      (m.prism_downscale === false ? ' <b>PRISM QPF downscaling was not applied in this build.</b>' : '') + ' Model data: NOAA/NCEP, ECMWF open data (CC-BY-4.0).';
    map = L.map('map', { zoomSnap: 0.25, minZoom: 3, maxZoom: 12 }).fitBounds(m.bounds);
    L.control.scale({ metric: false, imperial: true, position: 'bottomleft' }).addTo(map);
    setupBasemap();
    buildMembers(); skipNote();
    loadCalibration();
    buildControls();
    syncShade();
    window.__pwpfMap = map;                                   // handle for debugging and browser tests
    tipEl = $('tip');
    map.on('mousemove', tipOn);
    map.on('mouseout', tipOff);
    map.on('movestart zoomstart', tipOff);
    map.on('zoomend', terrainNote);
    map.on('click', function (e) { state.pt = e.latlng; setPin(e.latlng); readout(e.latlng); tipOn(e); writeHash(); });
    if (state.pt) setPin(state.pt);
    redraw();
  }

  /* Keyless basemaps (Carto's free tiles now need an API key).  If the active one produces only errors the
     next one is switched on automatically; the layer control lets the user pick. */
  var BASES = [
    { name: 'Light gray (Esri)', url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      labels: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      attr: 'Basemap © Esri, HERE, Garmin, OpenStreetMap contributors', native: 16 },
    { name: 'OpenStreetMap', url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', attr: '© OpenStreetMap contributors', native: 19 },
    { name: 'USGS Topo', url: 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
      attr: 'Basemap courtesy of the U.S. Geological Survey', native: 16 }
  ];
  /* Boundaries drawn above the data (below the basemap's place labels, pane z 450), styled as a cartographic hierarchy:
       country borders  heaviest, near-black, white casing        coastlines + lake shores  thin dark blue-grey, light casing
       state lines       medium, dark grey, light casing          county lines (zoom 6+)    hairline mid-grey, no casing
     Widths grow with zoom (linear between the two zoom stops) so the hierarchy holds from the continental view to street
     level; the casing keeps the lines legible over every colour of the data ramp.  Data: web/borders/*.json. */
  function lerpZ(z, z0, w0, z1, w1) { var t = Math.max(0, Math.min(1, (z - z0) / (z1 - z0))); return w0 + (w1 - w0) * t; }
  var BORDER_DEFS = [          // drawn in this order: later = on top
    { key: 'counties',  label: 'County lines (zoom 6+)',  minZoom: 6, color: '#5b6571', opacity: 0.6,  w: function (z) { return lerpZ(z, 6, 0.5, 12, 1.1); } },
    { key: 'states',    label: 'State lines',             color: '#2e3744', opacity: 0.88, w: function (z) { return lerpZ(z, 3, 0.9, 11, 2.1); },
      casing: { color: '#ffffff', opacity: 0.5, extra: 1.8 } },
    { key: 'coast_minor', label: 'Small islands & lakes (zoom 6+)', minZoom: 6, color: '#2a4a68', opacity: 0.75, w: function (z) { return lerpZ(z, 6, 0.5, 12, 1.1); } },
    { key: 'coast',     label: 'Coastlines & lake shores', color: '#17324d', opacity: 0.92, w: function (z) { return lerpZ(z, 3, 0.7, 11, 1.6); },
      casing: { color: '#ffffff', opacity: 0.4, extra: 1.5 } },
    { key: 'countries', label: 'Country borders',         color: '#0a0e14', opacity: 0.96, w: function (z) { return lerpZ(z, 3, 1.7, 11, 3.4); },
      casing: { color: '#ffffff', opacity: 0.6, extra: 2.6 } }
  ];
  function setupBorders(ctl) {
    function pane(name, z) { map.createPane(name); map.getPane(name).style.zIndex = z; map.getPane(name).style.pointerEvents = 'none'; }
    pane('bcase', 430);
    var caseRend = L.canvas({ pane: 'bcase' }), internal = false, defs = BORDER_DEFS;
    defs.forEach(function (d, i) {
      pane('b-' + d.key, 436 + i);
      d.line = L.geoJSON(null, { pane: 'b-' + d.key, renderer: L.canvas({ pane: 'b-' + d.key }), interactive: false,
        style: { color: d.color, opacity: d.opacity, weight: d.w(map.getZoom()), lineCap: 'round', lineJoin: 'round' } });
      var parts = [d.line];
      if (d.casing) {
        d.case = L.geoJSON(null, { pane: 'bcase', renderer: caseRend, interactive: false,
          style: { color: d.casing.color, opacity: d.casing.opacity, weight: d.w(map.getZoom()) + d.casing.extra, lineCap: 'round', lineJoin: 'round' } });
        parts.unshift(d.case);
      }
      d.layer = L.layerGroup(parts);
      d.on = true;
      ctl.addOverlay(d.layer, d.label);
      fetch('borders/' + d.key + '.json').then(function (r) { return r.json(); })
        .then(function (g) {
          var f = { type: 'Feature', properties: {}, geometry: g };
          if (d.case) d.case.addData(f);
          d.line.addData(f);
        }).catch(function () { /* borders are cosmetic; ignore a missing file */ });
    });
    window.__pwpfDefs = defs;
    function of(layer) { return defs.filter(function (d) { return d.layer === layer; })[0]; }
    function sync() {
      internal = true;
      var z = map.getZoom();
      defs.forEach(function (d) {
        d.line.setStyle({ weight: d.w(z) });
        if (d.case) d.case.setStyle({ weight: d.w(z) + d.casing.extra });
        var show = d.on && (!d.minZoom || z >= d.minZoom), has = map.hasLayer(d.layer);
        if (show && !has) map.addLayer(d.layer); else if (!show && has) map.removeLayer(d.layer);
      });
      internal = false;
    }
    map.on('overlayadd', function (e) { var d = of(e.layer); if (d && !internal) { d.on = true; sync(); } });
    map.on('overlayremove', function (e) { var d = of(e.layer); if (d && !internal) d.on = false; });
    map.on('zoomend', sync);
    sync();
  }

  function setupBasemap() {
    map.createPane('labels'); map.getPane('labels').style.zIndex = 450; map.getPane('labels').style.pointerEvents = 'none';
    var layers = {}, active = 0, refLayers = [], notice = null;
    BASES.forEach(function (b, i) {
      var t = L.tileLayer(b.url, { attribution: b.attr, maxNativeZoom: b.native, maxZoom: 12 });
      t._st = { ok: 0, err: 0 };
      t.on('tileload', function () { t._st.ok++; });
      t.on('tileerror', function () {
        t._st.err++;
        if (t._st.ok === 0 && t._st.err >= 4 && i === active && i + 1 < BASES.length) switchTo(i + 1, true);
      });
      layers[b.name] = t;
      refLayers[i] = b.labels ? L.tileLayer(b.labels, { pane: 'labels', maxNativeZoom: b.native, maxZoom: 12 }) : null;
    });
    function show(i) {
      BASES.forEach(function (b, j) {
        var t = layers[b.name];
        if (j !== i && map.hasLayer(t)) map.removeLayer(t);
        if (j !== i && refLayers[j] && map.hasLayer(refLayers[j])) map.removeLayer(refLayers[j]);
      });
      layers[BASES[i].name].addTo(map);
      if (refLayers[i]) refLayers[i].addTo(map);
      active = i;
    }
    function switchTo(i, auto) {
      show(i);
      if (auto && !notice) {
        notice = document.createElement('div'); notice.className = 'basemsg';
        notice.textContent = 'Basemap unavailable; switched to ' + BASES[i].name + '.';
        map.getContainer().appendChild(notice);
        setTimeout(function () { if (notice && notice.parentNode) notice.parentNode.removeChild(notice); notice = null; }, 6000);
      }
    }
    show(0);
    var ctl = L.control.layers(layers, null, { position: 'topright', collapsed: true }).addTo(map);
    setupBorders(ctl);
    map.on('baselayerchange', function (e) {
      BASES.forEach(function (b, j) { if (layers[b.name] === e.layer) { active = j; } });
      BASES.forEach(function (b, j) { if (refLayers[j]) { if (j === active) refLayers[j].addTo(map); else map.removeLayer(refLayers[j]); } });
    });
  }

  fetch(DATA + 'manifest.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(init)
    .catch(function (e) { $('runinfo').textContent = 'No data yet: ' + e; });
})();
