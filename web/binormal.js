/* Binormal distribution maths (mirror of pwpf_blender/binormal.py) + data-PNG decoding.
 * UMD: works as a browser global (window.Binormal) and as a CommonJS module for tests. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Binormal = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // complementary error function, fractional error < 1.2e-7 (Numerical Recipes erfcc)
  function erfc(x) {
    var z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
      t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  function ndtr(x) { return 0.5 * erfc(-x / Math.SQRT2); }

  // inverse normal CDF: Acklam's rational approximation + one Halley refinement step
  var A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  var B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  var C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  var D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  function ndtri(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    var q, r, x;
    if (p < 0.02425) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
    } else if (p <= 1 - 0.02425) {
      q = p - 0.5; r = q * q;
      x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
    }
    var e = ndtr(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  }

  // P(X >= t) for a binormal with mode m, lower sigma s1, upper sigma s2
  function exceedance(m, s1, s2, t) {
    var s = s1 + s2;
    if (t >= m) return 2 * s2 / s * (1 - ndtr((t - m) / s2));
    return 1 - 2 * s1 / s * ndtr((t - m) / s1);
  }
  // value x with P(X <= x) = q, clipped at 0
  function percentile(m, s1, s2, q) {
    var p = s1 / (s1 + s2), x;
    var cl = function (v) { return Math.min(Math.max(v, 1e-12), 1 - 1e-12); };
    if (q <= p) x = m + s1 * ndtri(cl(q / (2 * p)));
    else x = m + s2 * ndtri(cl(0.5 + (q - p) / (2 * (1 - p))));
    return Math.max(x, 0);
  }

  // 8-bit log decoding used by pwpf_blender/export.py
  function decodeLUT(enc) {
    var lut = new Float32Array(256), k = Math.log(1 + enc.xmax / enc.A);
    for (var v = 0; v < 256; v++) lut[v] = enc.A * (Math.exp(v / 255 * k) - 1);
    return lut;
  }
  var S_FLOOR = 1e-3;

  return { erfc: erfc, ndtr: ndtr, ndtri: ndtri, exceedance: exceedance, percentile: percentile,
           decodeLUT: decodeLUT, S_FLOOR: S_FLOOR };
});
