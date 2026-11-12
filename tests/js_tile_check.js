// Node helper for test_tiles.py: prints a JSON report about PWPFCore.renderTile / field / terrain adjustment.
const Core = require('../web/core.js');
const Terrain = require('../web/terrain.js');
const HALF = 20037508.342789244;
const W = 24, H = 20, merc = { xmin: -8.8e6, ymax: 5.0e6, px: 3000 };
const pal = Core.palettes.prob;
const FL = 1e-3;
const out = {};

function mkParams(w, h, mode, extra) {
  const n = w * h, lo = new Float32Array(n).fill(FL), hi = new Float32Array(n).fill(FL);
  return Object.assign({ W: w, H: h, n, mode, lo, hi, zeroMode: 0.05, zeroHi: 0.1, pot: null, sl: null }, extra || {});
}
// one hot cell (value 85, the 'mode' product reads it directly) in an otherwise empty field
const val = new Float32Array(W * H);
const ci = 12, cj = 10;
val[cj * W + ci] = 85;
const params = mkParams(W, H, val);
const z = 11, size = 256, scale = 1;
const xm = merc.xmin + (ci + 0.5) * merc.px, ym = merc.ymax - (cj + 0.5) * merc.px;
const wx = (xm + HALF) / (2 * HALF) * size * Math.pow(2, z), wy = (HALF - ym) / (2 * HALF) * size * Math.pow(2, z);
const tx = Math.floor(wx / size), ty = Math.floor(wy / size);
const px = Math.round(wx - tx * size - 0.5), py = Math.round(wy - ty * size - 0.5);   // pixel holding the cell centre
const base = { params, merc, pal, product: { kind: 'mode' }, alpha: 0.9, z, tx, ty, size, scale };
const R = (o) => Core.renderTile(Object.assign({}, base, o));
const rgba = R({});
const at = (r, x, y, s) => { s = s || size; const o = (y * s + x) * 4; return Array.from(r.slice(o, o + 4)); };
const cls = (c) => c[3] ? pal.rgb.findIndex(r => r[0] === c[0] && r[1] === c[1] && r[2] === c[2]) : -1;
out.centre = at(rgba, px, py);
out.expect_centre = pal.rgb[Core.binIndex(pal.stops, 85)].concat([Math.round(0.9 * 255)]);
const cellpx = merc.px / (2 * HALF / (size * Math.pow(2, z)));
out.cell_px = cellpx;
const walkOf = (r) => { const w = []; for (let d = 0; d <= Math.ceil(cellpx); d++) w.push(cls(at(r, Math.min(size - 1, px + d), py))); return w; };
out.walk = walkOf(rgba);                                        // auto -> bicubic (cell is ~39 px wide)
out.walk_bilinear = walkOf(R({ interp: 'bilinear' }));
out.walk_bicubic = walkOf(R({ interp: 'bicubic' }));
const rBil = R({ interp: 'bilinear' }), rCub = R({ interp: 'bicubic' });
out.bicubic_differs = Array.from(rBil).some((v, i) => v !== rCub[i]);
out.auto_is_bicubic = Array.from(rgba).every((v, i) => v === rCub[i]);
out.far = at(rgba, Math.min(size - 1, px + Math.round(2.2 * cellpx)), py);
// tiles that do not touch the data are fully transparent
const empty = R({ tx: tx + 40 });
out.empty_alpha_sum = empty.reduce((s, v, i) => s + (i % 4 === 3 ? v : 0), 0);
// retina scale: same picture sampled twice as densely
const r2 = R({ scale: 2 });
out.r2_len = r2.length;
out.r2_centre = at(r2, px * 2 + 1, py * 2 + 1, size * 2);
// low zoom (cell < 8 px): auto picks bilinear, the whole grid fits one tile, the hot cell still shows up
const lz = 6, lwx = (xm + HALF) / (2 * HALF) * size * Math.pow(2, lz);
const lbase = { z: lz, tx: Math.floor(lwx / size), ty: Math.floor((HALF - ym) / (2 * HALF) * Math.pow(2, lz)) };
const lt = R(lbase), ltb = R(Object.assign({ interp: 'bilinear' }, lbase));
out.low_zoom_pixels = lt.reduce((s, v, i) => s + (i % 4 === 3 && v ? 1 : 0), 0);
out.low_zoom_auto_is_bilinear = Array.from(lt).every((v, i) => v === ltb[i]);

// --- terrain adjustment: DEM tile (same zoom) of constant elevation vs model terrain of 0 m
const zref = new Float32Array(W * H);
const demOf = (elev, dtz) => { const d = new Float32Array(256 * 256).fill(elev); return { data: d, tz: z - (dtz || 0), tx: tx >> (dtz || 0), ty: ty >> (dtz || 0), size: 256 }; };
const cfg = Terrain.config({});
const pot = new Float32Array(W * H).fill(1000), sl = new Float32Array(W * H).fill(3000);
const tparams = mkParams(W, H, val, { pot, sl });
const adjBase = { params: tparams, zref, adjust: cfg };
out.t_off = cls(at(R(Object.assign({}, adjBase, { dem: demOf(0) })), px, py));           // dz = 0: unchanged (85 -> class of 80)
out.t_uphill = cls(at(R(Object.assign({}, adjBase, { dem: demOf(500) })), px, py));      // +500 m, all snow: x1.1 -> 93.5 -> 90 class
out.t_uphill_coarse_dem = cls(at(R(Object.assign({}, adjBase, { dem: demOf(500, 1) })), px, py));   // DEM from a coarser zoom
out.t_valley = cls(at(R(Object.assign({}, adjBase, { dem: demOf(-500), params: mkParams(W, H, val, { pot, sl: new Float32Array(W * H).fill(0) }) })), px, py));
out.t_none_without_flag = cls(at(R({ params: tparams, zref, dem: demOf(500) }), px, py));  // no `adjust` -> untouched
out.t_none_without_pot = cls(at(R({ zref, adjust: cfg, dem: demOf(500) }), px, py));       // layer lacks pot/sl planes -> untouched
out.terrain_factor_readout = Core.terrainFactor(tparams, cj * W + ci, zref, 500, cfg);

// --- speed: dense retina tile, bicubic and terrain adjustment on
const n = 1500 * 1000, big = new Float32Array(n);
for (let i = 0; i < n; i++) big[i] = 2 + (i % 1500) / 1500 * 40;
const bp = mkParams(1500, 1000, big, { pot: new Float32Array(n).fill(200), sl: new Float32Array(n).fill(1500) });
bp.lo.fill(0.3); bp.hi.fill(0.8);
const bm = { xmin: -8.8e6, ymax: 5.9e6, px: 3000 }, bz = 9, bcx = bm.xmin + 750 * 3000, bcy = bm.ymax - 500 * 3000;
const btx = Math.floor((bcx + HALF) / (2 * HALF) * Math.pow(2, bz)), bty = Math.floor((HALF - bcy) / (2 * HALF) * Math.pow(2, bz));
const dem = { data: new Float32Array(256 * 256).map((_, i) => 800 + (i % 256) * 3), tz: bz, tx: btx, ty: bty, size: 256 };
const bzref = new Float32Array(n).fill(700);
const timed = (o) => { const t0 = Date.now(); const r = Core.renderTile(Object.assign({ params: bp, merc: bm, pal: Core.palettes.prob, alpha: 1, z: bz, tx: btx, ty: bty, size: 256, scale: 2, product: { kind: 'prob', t: 6 } }, o)); return [Date.now() - t0, r.reduce((s, v, i) => s + (i % 4 === 3 && v ? 1 : 0), 0)]; };
const t1 = timed({ interp: 'bicubic' }), t2 = timed({ interp: 'bicubic', dem, zref: bzref, adjust: cfg });
out.dense_bicubic_ms = t1[0]; out.dense_terrain_ms = t2[0]; out.dense_pixels = t1[1]; out.dense_terrain_pixels = t2[1];


// --- sampleAt (cursor tooltip) agrees with what renderTile paints at the same spot
{
  const m2 = Float32Array.from({ length: W * H }, (_, i) => 3 + 40 * Math.exp(-(((i % W) - 12) ** 2 + (Math.floor(i / W) - 10) ** 2) / 30));
  const p2 = mkParams(W, H, m2, { pot: new Float32Array(W * H).fill(500), sl: new Float32Array(W * H).fill(1200) });
  p2.lo.fill(0.4); p2.hi.fill(0.9);
  const zz = 11, tt = { tx, ty };
  const palS = Core.palettes.snow, test = [];
  for (const interp of ['bilinear', 'bicubic']) {
    const rr = Core.renderTile({ params: p2, merc, pal: palS, product: { kind: 'pct', q: 50 }, alpha: 1, z: zz, tx, ty, size, scale: 1, interp });
    let bad = 0, n = 0;
    for (let py2 = 3; py2 < size; py2 += 17) for (let px2 = 3; px2 < size; px2 += 19) {
      const x = ((tx * size + px2 + 0.5) / (size * Math.pow(2, zz))) * 2 * HALF - HALF, y = HALF - ((ty * size + py2 + 0.5) / (size * Math.pow(2, zz))) * 2 * HALF;
      const sm = Core.sampleAt(p2, merc, x, y, interp === 'bicubic', null);
      const o = (py2 * size + px2) * 4, c = [rr[o], rr[o + 1], rr[o + 2], rr[o + 3]];
      const expectCls = sm ? Core.binIndex(palS.stops, Core.valueOf({ kind: 'pct', q: 50 }, sm.m, sm.s1, sm.s2)) : -1;
      const gotCls = c[3] ? palS.rgb.findIndex(r => r[0] === c[0] && r[1] === c[1] && r[2] === c[2]) : -1;
      n++; if (expectCls !== gotCls) bad++;
    }
    test.push([interp, n, bad]);
  }
  out.sample_vs_render = test;
  out.sample_outside = Core.sampleAt(p2, merc, merc.xmin - 5 * merc.px, merc.ymax - 3 * merc.px, false, null);
  const cx = merc.xmin + (3 + 0.5) * merc.px, cy = merc.ymax - (4 + 0.5) * merc.px;       // exactly at a cell centre
  const sc2 = Core.sampleAt(p2, merc, cx, cy, false, new Float32Array(W * H).fill(321));
  out.sample_centre = [sc2.m, p2.mode[4 * W + 3], sc2.zr, sc2.pot, sc2.sl];
  out.value_of = [Core.valueOf({ kind: 'mode' }, 2.5, 1, 2), Core.valueOf({ kind: 'prob', t: 0 }, 2.5, 1, 2), Core.valueOf({ kind: 'pct', q: 50 }, 2.5, 1, 2)];
}

// field() on decoded params
const p6 = { n: 6, mode: Float32Array.from([0, 2, 4, 8, 0.01, 12]), lo: Float32Array.from([0.05, 0.5, 1, 2, 0.05, 3]),
             hi: Float32Array.from([0.05, 0.8, 2, 3, 0.05, 4]), zeroMode: 0.05, zeroHi: 0.1 };
out.field = Array.from(Core.field(p6, { kind: 'prob', t: 4 }));
process.stdout.write(JSON.stringify(out));
