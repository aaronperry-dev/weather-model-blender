/* Leaflet front-end.  All numbers shown come from the data PNGs + binormal maths in binormal.js/core.js. */
(function () {
  'use strict';
  var DATA = 'data/';
  var state = { win: null, prod: 'snow', kind: 'prob', thr: null, pct: 50, pt: null };
  var manifest, map, overlay, cache = {}, current = null, mercator;

  function $(id) { return document.getElementById(id); }
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function fmtTime(iso) {
    var d = new Date(iso);
    return d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' + d.getUTCFullYear() + ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + 'Z';
  }
  function fmtShort(iso) {
    var d = new Date(iso);
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
      if (p[0] === 'p') { var ll = p[1].split(','); if (ll.length === 2) state.pt = { lat: +ll[0], lng: +ll[1] }; }
    });
  }
  function writeHash() {
    history.replaceState(null, '', '#w=' + state.win + '&x=' + state.prod + '&k=' + state.kind + '&t=' + state.thr + '&q=' + state.pct +
      (state.pt ? '&p=' + state.pt.lat.toFixed(3) + ',' + state.pt.lng.toFixed(3) : ''));
  }

  function curWin() { return manifest.windows.filter(function (x) { return x.id === state.win; })[0]; }
  function prodInfo() { return manifest.products[state.prod]; }
  function lengths() {
    var s = {};
    manifest.windows.forEach(function (w) { s[w.length_h] = 1; });
    return Object.keys(s).map(Number).sort(function (a, b) { return a - b; });
  }
  function hasIce() { return manifest.windows.some(function (w) { return !!w.ice_file; }); }

  /* Load (and cache) the decoded parameter image of window w for the active hazard; resolves null if the
     window has no layer for it (e.g. no freezing rain forecast in that period). */
  function loadLayer(w, prod) {
    var info = manifest.products[prod], file = w[info.file_key], key = w.id + '|' + prod;
    if (!file) return Promise.resolve(null);
    if (cache[key]) return Promise.resolve(cache[key]);
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        var c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        var ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        var data = ctx.getImageData(0, 0, c.width, c.height).data;
        cache[key] = PWPFCore.decodeParams(data, info.encoding);
        resolve(cache[key]);
      };
      img.onerror = function () { reject(new Error('could not load ' + file)); };
      img.src = DATA + file + '?v=' + encodeURIComponent(manifest.generated);
    });
  }

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
    $('legend').innerHTML = html;
  }

  function clearOverlay() { if (overlay) { map.removeLayer(overlay); overlay = null; } }

  function redraw() {
    var w = curWin(), info = prodInfo();
    $('validity').textContent = w.length_h + ' h total · valid ' + fmtTime(w.start) + ' → ' + fmtTime(w.end);
    $('thrLabel').textContent = state.prod === 'ice' ? 'Ice accretion at least' : 'Snowfall at least';
    drawLegend();
    loadLayer(w, state.prod).then(function (params) {
      current = params;
      $('nodata').hidden = !!params;
      if (!params) { clearOverlay(); $('readout').innerHTML = '<h2>Point forecast</h2><div class="muted">No ' +
        info.label.toLowerCase() + ' signal in any ensemble member for this period.</div>'; writeHash(); return; }
      var r = PWPFCore.render(params, product(), +$('opacity').value);
      var c = document.createElement('canvas');
      c.width = manifest.size[0]; c.height = manifest.size[1];
      var ctx = c.getContext('2d');
      ctx.putImageData(new ImageData(r.rgba, c.width, c.height), 0, 0);
      var url = c.toDataURL('image/png');
      clearOverlay();
      overlay = L.imageOverlay(url, manifest.bounds, { opacity: 1, interactive: false }).addTo(map);
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

  function endButtons() {
    var box = $('ends'), len = curWin().length_h;
    box.innerHTML = '';
    manifest.windows.filter(function (w) { return w.length_h === len; }).forEach(function (w) {
      var b = document.createElement('button');
      b.textContent = fmtShort(w.end); b.dataset.id = w.id;
      if (state.prod === 'ice' && !w.ice_file) b.classList.add('dim');
      b.onclick = function () { state.win = w.id; syncChips(); redraw(); };
      box.appendChild(b);
    });
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
    $('opacity').oninput = function () { redraw(); };
    Array.prototype.forEach.call($('products').children, function (b) {
      b.onclick = function () { state.kind = b.dataset.kind; syncChips(); redraw(); };
    });
    syncChips();
  }
  function syncChips() {
    var cur = curWin();
    endButtons();
    Array.prototype.forEach.call($('ends').children, function (b) { b.classList.toggle('on', b.dataset.id === state.win); });
    Array.prototype.forEach.call($('lengths').children, function (b) { b.classList.toggle('on', +b.dataset.len === cur.length_h); });
    Array.prototype.forEach.call($('hazards').children, function (b) { b.classList.toggle('on', b.dataset.prod === state.prod); });
    Array.prototype.forEach.call($('products').children, function (b) { b.classList.toggle('on', b.dataset.kind === state.kind); });
    $('thrRow').hidden = state.kind !== 'prob';
    $('pctRow').hidden = state.kind !== 'pct';
  }

  function readout(latlng) {
    if (!current) return;
    var info = prodInfo(), dec = info.decimals;
    var i = PWPFCore.lonLatToPixel(latlng.lng, latlng.lat, manifest, mercator);
    var box = $('readout');
    if (i < 0) return;
    var thr = state.prod === 'ice' ? info.thresholds : info.thresholds.filter(function (t) { return t >= 1 && t <= 18; });
    var r = PWPFCore.pointReadout(current, i, thr, [5, 10, 25, 50, 75, 90, 95]);
    var head = '<h2>Point forecast · ' + info.label + '</h2><div class="muted">' + latlng.lat.toFixed(2) + ', ' + latlng.lng.toFixed(2) + '</div>';
    if (r.mode < current.zeroMode && current.hi[i] < current.zeroHi) {
      box.innerHTML = head + '<div class="big">No ' + (state.prod === 'ice' ? 'ice' : 'snow') + ' signal</div>';
      return;
    }
    var rows = '<tr><th>' + (state.prod === 'ice' ? 'Ice ≥' : 'Snow ≥') + '</th>' + r.probs.map(function (p) { return '<td>' + p[0] + '"</td>'; }).join('') + '</tr>' +
      '<tr><th>Prob.</th>' + r.probs.map(function (p) { return '<td>' + Math.round(p[1]) + '%</td>'; }).join('') + '</tr>';
    var prow = '<tr><th>Pctile</th>' + r.pcts.map(function (p) { return '<td>' + p[0] + '</td>'; }).join('') + '</tr>' +
      '<tr><th>Inches</th>' + r.pcts.map(function (p) { return '<td>' + p[1].toFixed(dec) + '</td>'; }).join('') + '</tr>';
    box.innerHTML = head + '<div class="big">' + r.mode.toFixed(dec) + ' in <span class="muted small">forecast (mode)</span></div>' +
      '<table>' + rows + '</table><table style="margin-top:8px">' + prow + '</table>' +
      '<div class="muted small" style="margin-top:6px">Chance of at least the amount shown; percentiles are amounts not exceeded with that probability.' +
      (state.prod === 'ice' ? ' Ice is flat (elevated horizontal) accretion from a precipitation-rate FRAM; treat as experimental.' : '') + '</div>';
  }

  function init(m) {
    manifest = m; mercator = m.mercator;
    parseHash();
    if (!state.win || !curWin()) state.win = m.windows[0].id;
    $('runinfo').textContent = 'Cycle ' + fmtTime(m.cycle) + ' · ' + m.members_total + ' ensemble members · built ' + fmtTime(m.generated);
    $('about').innerHTML = '<b>Notice.</b> ' + m.notice + ' Methodology: multi-model ensemble → QPF (PRISM-downscaled) × precipitation type × SLR → binormal fit (mode = ' +
      m.mode_method + ' forecast, variance from the ensemble). Freezing rain: simplified FRAM ice-to-liquid ratio. Basemap © OpenStreetMap contributors © CARTO. Model data: NOAA/NCEP, ECMWF open data (CC-BY-4.0).';
    map = L.map('map', { zoomSnap: 0.25, minZoom: 3 }).fitBounds(m.bounds);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      { attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 10 }).addTo(map);
    buildControls();
    map.on('mousemove', function (e) { readout(e.latlng); });
    map.on('click', function (e) { state.pt = e.latlng; readout(e.latlng); writeHash(); });
    redraw();
  }

  fetch(DATA + 'manifest.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(init)
    .catch(function (e) { $('runinfo').textContent = 'No data yet: ' + e; });
})();
