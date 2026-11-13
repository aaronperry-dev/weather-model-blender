// renderTile elevation filter: pixels whose ground is below o.minElev stay transparent
const Core = require('../web/core.js');
const HALF = 20037508.342789244, W = 64, n = W;
const params = { mode: new Float32Array(W * W).fill(5), lo: new Float32Array(W * W).fill(1), hi: new Float32Array(W * W).fill(1),
                 n: W * W, W: W, H: W, pot: null, sl: null, zeroMode: 0.01, zeroHi: 0.02 };
const merc = { xmin: -HALF, ymax: HALF, px: 2 * HALF / W };
const dem = new Float32Array(256 * 256);
for (let r = 0; r < 256; r++) for (let c = 0; c < 256; c++) dem[r * 256 + c] = c < 128 ? 100 : 1000;      // west half low, east half high
const base = { params: params, merc: merc, pal: Core.palettes.snow, product: { kind: 'mode' }, alpha: 1, z: 2, tx: 1, ty: 1, size: n, scale: 1 };
function cols(rgba) {                                        // [painted pixels in the west half, in the east half]
  let w = 0, e = 0;
  for (let q = 0; q < n; q++) for (let p = 0; p < n; p++) if (rgba[(q * n + p) * 4 + 3]) { if (p < n / 2) w++; else e++; }
  return [w, e];
}
const d = { data: dem, tz: 2, tx: 1, ty: 1, size: 256 };
const out = {
  none: cols(Core.renderTile(Object.assign({}, base))),
  low: cols(Core.renderTile(Object.assign({}, base, { dem: d, minElev: 50 }))),                 // below both halves: nothing hidden
  mid: cols(Core.renderTile(Object.assign({}, base, { dem: d, minElev: 500 }))),                // hides the west half
  high: cols(Core.renderTile(Object.assign({}, base, { dem: d, minElev: 5000 }))),              // hides everything
  nodem: cols(Core.renderTile(Object.assign({}, base, { minElev: 500 }))),                      // filter without elevation data: draw nothing
  dem_only: cols(Core.renderTile(Object.assign({}, base, { dem: d }))),                         // dem without a filter changes nothing
};
process.stdout.write(JSON.stringify(out));
