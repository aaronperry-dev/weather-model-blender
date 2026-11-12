/* Leaflet front-end.  All numbers shown come from the data PNGs + binormal maths in binormal.js/core.js. */
(function () {
  'use strict';
  var DATA = 'data/';
  var state = { win: null, kind: 'prob', thr: 4, pct: 50, pt: null };
  var manifest, map, overlay, cache = {}, current = null, mercator;

  function $(id) { return document.getElementById(id); }
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function fmtTime(iso) {
    var d = new Date(iso);
    return d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' + d.getUTCFullYear() + ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + 'Z';
  }
  function parseHash() {
    (location.hash.slice(1) || '').split('&').forEach(function (kv) {
      var p = kv.split('=');
      if (p[0] === 'w') state.win = p[1];
      if (p[0] === 'k') state.kind = p[1];
      if (p[0] === 't') state.thr = +p[1];
      if (p[0] === 'q') state.pct = +p[1];
      if (p[0] === 'p') { var ll = p[1].split(','); if (ll.length === 2) state.pt = { lat: +ll[0], lng: +ll[1] }; }
    });
  }
  function writeHash() {
    history.replaceState(null, '', '#w=' + state.win + '&k=' + state.kind + '&t=' + state.thr + '&q=' + state.pct +
      (state.pt ? '&p=' + state.pt.lat.toFixed(3) + ',' + state.pt.lng.toFixed(3) : ''));
  }

  function loadWindow(w) {
    if (cache[w.id]) return Promise.resolve(cache[w.id]);
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        var c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        var ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        var data = ctx.getImageData(0, 0, c.width, c.height).data;
        cache[w.id] = PWPFCore.decodeParams(data, manifest.encoding);
        resolve(cache[w.id]);
      };
      img.onerror = function () { reject(new Error('could not load ' + w.file)); };
      img.src = DATA + w.file + '?v=' + encodeURIComponent(manifest.generated);
    });
  }

  function product() {
    if (state.kind === 'prob') return { kind: 'prob', t: state.thr };
    if (state.kind === 'pct') return { kind: 'pct', q: state.pct };
    return { kind: 'mode' };
  }

  function drawLegend() {
    var pal = state.kind === 'prob' ? PWPFCore.palettes.prob : PWPFCore.palettes.snow;
    var unit = state.kind === 'prob' ? '%' : ' in';
    var html = '';
    for (var i = pal.stops.length - 1; i >= 0; i--) {
      var next = pal.stops[i + 1];
      var label = (next === undefined ? '≥ ' + pal.stops[i] : pal.stops[i] + ' – ' + next) + unit;
      html += '<div class="row"><span class="sw" style="background:' + pal.colors[i] + '"></span>' + label + '</div>';
    }
    $('legend').innerHTML = html;
  }

  function redraw() {
    var w = manifest.windows.filter(function (x) { return x.id === state.win; })[0];
    $('validity').textContent = 'Valid ' + fmtTime(w.start) + ' → ' + fmtTime(w.end);
    drawLegend();
    loadWindow(w).then(function (params) {
      current = params;
      var r = PWPFCore.render(params, product(), +$('opacity').value);
      var c = document.createElement('canvas');
      c.width = manifest.size[0]; c.height = manifest.size[1];
      var ctx = c.getContext('2d');
      ctx.putImageData(new ImageData(r.rgba, c.width, c.height), 0, 0);
      var url = c.toDataURL('image/png');
      if (overlay) map.removeLayer(overlay);
      overlay = L.imageOverlay(url, manifest.bounds, { opacity: 1, interactive: false }).addTo(map);
      if (state.pt) readout(state.pt);
      writeHash();
    }).catch(function (e) { $('runinfo').textContent = String(e); });
  }

  function buildControls() {
    var wbox = $('windows');
    manifest.windows.forEach(function (w) {
      var b = document.createElement('button');
      b.textContent = w.label; b.dataset.id = w.id;
      b.onclick = function () { state.win = w.id; syncChips(); redraw(); };
      wbox.appendChild(b);
    });
    var thr = $('thr'), pct = $('pct');
    manifest.thresholds_in.forEach(function (t) { thr.add(new Option(t, t)); });
    manifest.percentiles.forEach(function (q) { pct.add(new Option(q + 'th', q)); });
    thr.value = state.thr; pct.value = state.pct;
    thr.onchange = function () { state.thr = +thr.value; redraw(); };
    pct.onchange = function () { state.pct = +pct.value; redraw(); };
    $('opacity').oninput = function () { redraw(); };
    Array.prototype.forEach.call($('products').children, function (b) {
      b.onclick = function () { state.kind = b.dataset.kind; syncChips(); redraw(); };
    });
    syncChips();
  }
  function syncChips() {
    Array.prototype.forEach.call($('windows').children, function (b) { b.classList.toggle('on', b.dataset.id === state.win); });
    Array.prototype.forEach.call($('products').children, function (b) { b.classList.toggle('on', b.dataset.kind === state.kind); });
    $('thrRow').hidden = state.kind !== 'prob';
    $('pctRow').hidden = state.kind !== 'pct';
  }

  function readout(latlng) {
    if (!current) return;
    var i = PWPFCore.lonLatToPixel(latlng.lng, latlng.lat, manifest, mercator);
    var box = $('readout');
    if (i < 0) return;
    var r = PWPFCore.pointReadout(current, i, manifest.thresholds_in.filter(function (t) { return t >= 1 && t <= 18; }), [5, 10, 25, 50, 75, 90, 95]);
    if (r.mode < 0.0119 && current.hi[i] < 0.02) {
      box.innerHTML = '<h2>Point forecast</h2><div class="muted">' + latlng.lat.toFixed(2) + ', ' + latlng.lng.toFixed(2) +
        '</div><div class="big">No snow signal</div>';
      return;
    }
    var rows = '<tr><th>Snow ≥</th>' + r.probs.map(function (p) { return '<td>' + p[0] + '"</td>'; }).join('') + '</tr>' +
      '<tr><th>Prob.</th>' + r.probs.map(function (p) { return '<td>' + Math.round(p[1]) + '%</td>'; }).join('') + '</tr>';
    var prow = '<tr><th>Pctile</th>' + r.pcts.map(function (p) { return '<td>' + p[0] + '</td>'; }).join('') + '</tr>' +
      '<tr><th>Inches</th>' + r.pcts.map(function (p) { return '<td>' + p[1].toFixed(1) + '</td>'; }).join('') + '</tr>';
    box.innerHTML = '<h2>Point forecast</h2><div class="muted">' + latlng.lat.toFixed(2) + ', ' + latlng.lng.toFixed(2) +
      '</div><div class="big">' + r.mode.toFixed(1) + ' in <span class="muted small">forecast (mode)</span></div>' +
      '<table>' + rows + '</table><table style="margin-top:8px">' + prow + '</table>' +
      '<div class="muted small" style="margin-top:6px">Chance of at least the amount shown; percentiles are amounts not exceeded with that probability.</div>';
  }

  function init(m) {
    manifest = m; mercator = m.mercator;
    parseHash();
    if (!state.win) state.win = m.windows[0].id;
    $('runinfo').textContent = 'Cycle ' + fmtTime(m.cycle) + ' · ' + m.members_total + ' ensemble members · built ' + fmtTime(m.generated);
    $('about').innerHTML = '<b>Notice.</b> ' + m.notice + ' Methodology: multi-model ensemble → QPF (PRISM-downscaled) × precipitation type × SLR → binormal fit (mode = ' +
      m.mode_method + ' forecast, variance from the ensemble). Basemap © OpenStreetMap contributors © CARTO. Model data: NOAA/NCEP, ECMWF open data (CC-BY-4.0).';
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
