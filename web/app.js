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
  var MODEL = { gefs: 'GEFS', gefs_mean: 'GEFS mean', ifs_ens: 'ECMWF IFS ENS', ifs_hres: 'ECMWF IFS HRES', gfs: 'GFS',
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
    $('validity').textContent = w.length_h + ' h total · valid ' + fmtTime(w.start) + ' → ' + fmtTime(w.end) +
      (w.n_members ? ' · ' + w.n_members + ' members' + (w.reduced ? ' (reduced ensemble, lower confidence)' : '') : '');
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
    $('runinfo').textContent = 'Cycle ' + fmtTime(m.cycle) + ' · ' + m.members_total + ' ensemble members' +
      (m.members_requested && m.members_requested !== m.members_total ? ' (of ' + m.members_requested + ' requested)' : '') + ' · built ' + fmtTime(m.generated);
    $('about').innerHTML = '<b>Notice.</b> ' + m.notice + ' Methodology: multi-model ensemble → QPF (PRISM-downscaled) × precipitation type × SLR → binormal fit (mode = ' +
      m.mode_method + ' forecast, variance from the ensemble). Freezing rain: simplified FRAM ice-to-liquid ratio.' +
      (m.prism_downscale === false ? ' <b>PRISM QPF downscaling was not applied in this build.</b>' : '') + ' Model data: NOAA/NCEP, ECMWF open data (CC-BY-4.0).';
    map = L.map('map', { zoomSnap: 0.25, minZoom: 3, maxZoom: 12 }).fitBounds(m.bounds);
    setupBasemap();
    buildMembers(); skipNote();
    buildControls();
    map.on('mousemove', function (e) { readout(e.latlng); });
    map.on('click', function (e) { state.pt = e.latlng; readout(e.latlng); writeHash(); });
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
  /* Administrative boundaries drawn above the data (pane z 440, below the basemap's place labels at 450):
     international borders, state lines, and county lines (only from zoom 6).  Data: web/borders/*.json. */
  function setupBorders(ctl) {
    map.createPane('borders'); map.getPane('borders').style.zIndex = 440; map.getPane('borders').style.pointerEvents = 'none';
    var rend = L.canvas({ pane: 'borders' }), internal = false;
    var defs = [
      { key: 'countries', label: 'Country borders', style: { color: '#161616', weight: 1.9, opacity: 0.9 } },
      { key: 'states', label: 'State lines', style: { color: '#2b2b2b', weight: 1.1, opacity: 0.8 } },
      { key: 'counties', label: 'County lines (zoom 6+)', style: { color: '#4a4a4a', weight: 0.6, opacity: 0.55 }, minZoom: 6 }
    ];
    defs.forEach(function (d) {
      d.layer = L.geoJSON(null, { pane: 'borders', renderer: rend, style: d.style, interactive: false });
      d.on = true;
      ctl.addOverlay(d.layer, d.label);
      fetch('borders/' + d.key + '.json').then(function (r) { return r.json(); })
        .then(function (g) { d.layer.addData({ type: 'Feature', properties: {}, geometry: g }); })
        .catch(function () { /* borders are cosmetic; ignore a missing file */ });
    });
    function of(layer) { return defs.filter(function (d) { return d.layer === layer; })[0]; }
    function sync() {
      internal = true;
      defs.forEach(function (d) {
        var show = d.on && (!d.minZoom || map.getZoom() >= d.minZoom), has = map.hasLayer(d.layer);
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
