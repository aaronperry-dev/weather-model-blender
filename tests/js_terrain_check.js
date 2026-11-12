// Node helper for test_terrain.py: Terrarium decoding, scale() cases, hillshade, TileCache, sample().
const T = require('../web/terrain.js');
const out = {};
const cfg = T.config({});
out.cfg = cfg;
out.cfg_override = T.config({ snow_level_width_m: 50, precip_factor_range: null }).snow_level_width_m;
// Terrarium: R=128,G=0,B=0 -> 0 m;  R=131,G=232,B=128 -> 3*256+232+0.5-... = (131*256+232+0.5)-32768
const px = new Uint8Array([128, 0, 0, 255, 131, 232, 128, 255, 127, 255, 255, 255]);
out.terrarium = Array.from(T.decodeTerrarium(px, 3));
const sc = (zp, zr, sl, pot, mode) => T.scale(zp, zr, sl, pot, mode, cfg);
out.same = sc(1200, 1200, 1000, 50, 10);
out.all_snow_up = sc(1500, 1000, 5000, 1000, 5);          // fq = 1.1, no phase change
out.cap_range_up = sc(4000, 0, 9000, 1e6, 5);             // dz capped at 1500 m -> fq = 1.3 (< 1.35)
out.cap_range_down = sc(-3000, 0, 9000, 1e6, 5);
out.valley = sc(0, 500, 400, 100, 5);                     // valley below the snow level at 400 m, model cell above
out.ridge = sc(900, 400, 600, 100, 5);                    // ridge above the snow level, cell below: more snow, capped at 3 x fq
out.pot_cap = sc(900, 400, 600, 5.5, 5);                  // potential snowfall 5.5 in: cannot exceed fq*5.5/5
out.far_from_line = sc(2100, 2000, 800, 100, 5);          // everything far above the snow level: just the elevation factor
out.snow_frac = [T.snowFrac(0, 0, 100), T.snowFrac(500, 0, 100), T.snowFrac(-500, 0, 100)];
// hillshade (sun in the NW): a slope that rises to the east faces west = lit; one that falls to the east faces east = shadowed
const n = 8, dem = new Float32Array(n * n);
for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) dem[r * n + c] = 100 * c;        // rises eastward => faces west (lit)
const dem2 = new Float32Array(n * n);
for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) dem2[r * n + c] = -100 * c;      // faces east (shadowed)
const hs = (d) => Array.from(T.hillshade(d, n, 100, 1, 315, 45).slice((3 * n + 4) * 4, (3 * n + 4) * 4 + 4));
out.shade_west_facing = hs(dem);
out.shade_east_facing = hs(dem2);
out.shade_flat = Array.from(T.hillshade(new Float32Array(n * n), n, 100, 1).slice(0, 4));
// sample()
const g = Float32Array.from([0, 10, 20, 30]);
out.sample = [T.sample(g, 2, 0, 0), T.sample(g, 2, 0.5, 0), T.sample(g, 2, 0.5, 0.5), T.sample(g, 2, -3, 5), T.sample(g, 2, 9, 9)];
// TileCache: LRU, dedupe, failures not cached
let loads = 0, failNext = false;
const cache = new T.TileCache((z, x, y) => { loads++; if (failNext) { failNext = false; return Promise.reject(new Error('x')); } return Promise.resolve(new Float32Array([z, x, y])); }, 2);
(async () => {
  const [a, b] = await Promise.all([cache.get(5, 1, 1), cache.get(5, 1, 1)]);
  out.dedupe = [loads, a === b];
  await cache.get(5, 2, 2); await cache.get(5, 3, 3);      // evicts 1/1 (LRU)
  out.size = cache.size();
  await cache.get(5, 3, 3);                                  // hit
  out.hit_loads = loads;
  await cache.get(5, 1, 1);                                  // reload
  out.reload_loads = loads;
  failNext = true;
  let err = false; try { await cache.get(6, 9, 9); } catch (e) { err = true; }
  await cache.get(6, 9, 9);
  out.fail = [err, loads - out.reload_loads];                // failed once, loaded again afterwards (not cached)
  process.stdout.write(JSON.stringify(out));
})();
