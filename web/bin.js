/* Reader for the gzip-compressed "PWPB" data layers written by pwpf_blender/export.py (format described there).
 * UMD: browser global (window.PWPFBin) and CommonJS module for Node tests. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('zlib'));
  else root.PWPFBin = factory(null);
})(typeof self !== 'undefined' ? self : this, function (nodeZlib) {
  'use strict';

  var lutCache = {};
  function logLUT(bits, A, xmax) {                 // code -> value for a log-encoded plane
    var key = bits + '|' + A + '|' + xmax;
    if (lutCache[key]) return lutCache[key];
    var n = 1 << bits, vmax = n - 1, k = Math.log(1 + xmax / A), lut = new Float32Array(n);
    for (var v = 0; v < n; v++) lut[v] = A * (Math.exp(v / vmax * k) - 1);
    return (lutCache[key] = lut);
  }

  /* bytes: Uint8Array of the *decompressed* file.  Returns {W, H, planes: {id: Float32Array}, meta: {id: {...}}}.
     `meta[id].step1` is the decoded value of code 1, the smallest non-zero level of that plane. */
  function parse(bytes) {
    if (bytes[0] !== 0x50 || bytes[1] !== 0x57 || bytes[2] !== 0x50 || bytes[3] !== 0x42) throw new Error('not a PWPB file');
    var dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var ver = dv.getUint8(4), np = dv.getUint8(5), shuffled = (dv.getUint8(6) & 1) === 1;
    if (ver !== 1) throw new Error('unsupported PWPB version ' + ver);
    var W = dv.getUint16(8, true), H = dv.getUint16(10, true), n = W * H, pos = 12 + 12 * np, i, v;
    var planes = {}, meta = {};
    for (var e = 0; e < np; e++) {
      var o = 12 + 12 * e, id = dv.getUint8(o), bits = dv.getUint8(o + 1), kind = dv.getUint8(o + 2);
      var p0 = dv.getFloat32(o + 4, true), p1 = dv.getFloat32(o + 8, true);
      var out = new Float32Array(n), lut = kind === 0 ? logLUT(bits, p0, p1) : null;
      if (bits === 16) {
        if (shuffled) {
          var hiB = bytes.subarray(pos, pos + n), loB = bytes.subarray(pos + n, pos + 2 * n);
          for (i = 0; i < n; i++) { v = (hiB[i] << 8) | loB[i]; out[i] = lut ? lut[v] : v * p0 + p1; }
        } else {
          for (i = 0; i < n; i++) { v = dv.getUint16(pos + 2 * i, true); out[i] = lut ? lut[v] : v * p0 + p1; }
        }
        pos += 2 * n;
      } else {
        for (i = 0; i < n; i++) { v = bytes[pos + i]; out[i] = lut ? lut[v] : v * p0 + p1; }
        pos += n;
      }
      planes[id] = out;
      meta[id] = { bits: bits, kind: kind, p0: p0, p1: p1, step1: lut ? lut[1] : p0 };
    }
    return { W: W, H: H, planes: planes, meta: meta };
  }

  /* gzip -> bytes.  A server that already applied Content-Encoding hands us the plain file: pass it through. */
  function gunzip(buf) {
    var u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    if (!(u8[0] === 0x1f && u8[1] === 0x8b)) return Promise.resolve(u8);
    if (nodeZlib) return Promise.resolve(new Uint8Array(nodeZlib.gunzipSync(u8)));
    if (typeof DecompressionStream === 'undefined') return Promise.reject(new Error('this browser cannot decompress gzip (DecompressionStream missing)'));
    var stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer().then(function (b) { return new Uint8Array(b); });
  }

  function load(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
      return r.arrayBuffer();
    }).then(gunzip).then(parse);
  }

  return { parse: parse, gunzip: gunzip, load: load };
});
