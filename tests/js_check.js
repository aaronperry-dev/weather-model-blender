// Node helper for test_js_parity.py: node js_check.js <input.json> <rgba.bin> <output.json>
const fs = require('fs');
const Binormal = require('../web/binormal.js');
const Core = require('../web/core.js');

const inp = JSON.parse(fs.readFileSync(process.argv[2]));
const out = {};
out.exc = inp.triples.map(([m, s1, s2, t]) => Binormal.exceedance(m, s1, s2, t));
out.pct = inp.triples.map(([m, s1, s2, , q]) => Binormal.percentile(m, s1, s2, q));
out.lut = Array.from(Binormal.decodeLUT(inp.enc));

const buf = fs.readFileSync(process.argv[3]);
const rgba = new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.length);
const params = Core.decodeParams(rgba, inp.enc);
const rp = Core.render(params, { kind: 'prob', t: 4 }, 0.85);
const rq = Core.render(params, { kind: 'pct', q: 90 }, 0.85);
out.prob_value = Array.from(rp.value);
out.pct_value = Array.from(rq.value);
out.prob_colored = Array.from(rp.rgba).filter((v, i) => i % 4 === 3 && v > 0).length;
out.mode = Array.from(params.mode);
fs.writeFileSync(process.argv[4], JSON.stringify(out));
