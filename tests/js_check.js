// Node helper for test_js_parity.py: node js_check.js <input.json> <layer.bin.gz> <output.json>
const fs = require('fs');
const Binormal = require('../web/binormal.js');
const Bin = require('../web/bin.js');
const Core = require('../web/core.js');

const inp = JSON.parse(fs.readFileSync(process.argv[2]));
const out = {};
out.exc = inp.triples.map(([m, s1, s2, t]) => Binormal.exceedance(m, s1, s2, t));
out.pct = inp.triples.map(([m, s1, s2, , q]) => Binormal.percentile(m, s1, s2, q));

Bin.gunzip(fs.readFileSync(process.argv[3])).then((bytes) => {
  const bin = Bin.parse(bytes);
  const params = Core.paramsFromBin(bin);
  const T = inp.t || 4, Q = inp.q || 90;
  out.W = bin.W; out.H = bin.H;
  out.planes = Object.keys(bin.planes).map(Number);
  out.mode = Array.from(params.mode);
  out.lo = Array.from(params.lo);
  out.hi = Array.from(params.hi);
  out.prob_value = Array.from(Core.field(params, { kind: 'prob', t: T }));
  out.pct_value = Array.from(Core.field(params, { kind: 'pct', q: Q }));
  out.mode_value = Array.from(Core.field(params, { kind: 'mode' }));
  out.zero_mode = params.zeroMode; out.zero_hi = params.zeroHi;
  if (bin.planes[4]) out.pot = Array.from(bin.planes[4]);
  if (bin.planes[5]) out.sl = Array.from(bin.planes[5]);
  fs.writeFileSync(process.argv[4], JSON.stringify(out));
}).catch((e) => { console.error(e); process.exit(1); });
