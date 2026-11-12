"""Build the viewer's border overlays (web/borders/*.json) from TopoJSON files.

    npm i us-atlas world-atlas            # Census cartographic boundaries / Natural Earth, public domain
    python scripts/make_borders.py node_modules/us-atlas node_modules/world-atlas web/borders ne_10m_lakes.geojson

Only *interior* boundaries are kept for countries / states / counties (arcs shared by two features), clipped to the model
domain and rounded to 0.001 deg, as one MultiLineString per layer.  `coast.json` / `coast_minor.json` are the coastline (Natural Earth 10 m land
outline from world-atlas) plus the shores of the larger lakes (Natural Earth 10 m lakes, scalerank <= 4: the Great Lakes,
Champlain, Finger Lakes, Great Salt Lake ...), simplified to ~0.003 deg (about 300 m).  Lakes come from
https://github.com/nvkelso/natural-earth-vector (geojson/ne_10m_lakes.geojson, public domain); pass it as the 4th argument.
The output is committed, so CI never needs these packages."""
import json
import sys
from pathlib import Path

BBOX = (-135.0, 14.0, -50.0, 62.0)           # lon0, lat0, lon1, lat1


def arcs_of(topo):
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    out = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        out.append(pts)
    return out


def walk(node, fn):
    if isinstance(node, int):
        fn(node)
    else:
        for n in node:
            walk(n, fn)


def interior_lines(topo, obj):
    arcs = arcs_of(topo)
    count = [0] * len(arcs)

    def hit(i):
        count[i if i >= 0 else ~i] += 1
    for g in topo["objects"][obj]["geometries"]:
        if g.get("arcs") is not None:
            walk(g["arcs"], hit)
    lines = []
    for i, c in enumerate(count):
        if c < 2:
            continue
        pts = [(round(x, 3), round(y, 3)) for x, y in arcs[i]]
        if not any(BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3] for x, y in pts):
            continue
        dedup = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
        if len(dedup) > 1:
            lines.append([list(p) for p in dedup])
    return lines


def rdp(pts, eps):
    """Douglas-Peucker line simplification (iterative, numpy)."""
    import numpy as np
    p = np.asarray(pts, float)
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        seg = p[a + 1:b] - p[a]
        n = float(np.hypot(*d))
        dist = np.hypot(seg[:, 0], seg[:, 1]) if n == 0 else np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / n
        i = int(np.argmax(dist))
        if dist[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return p[keep]


def clip_runs(pts, bbox=BBOX):
    """Split a polyline into the runs of consecutive points that lie inside bbox."""
    runs, cur = [], []
    for x, y in pts:
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            cur.append((x, y))
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return runs


def coast_lines(land_topo, lakes_geojson=None, eps=0.003, min_extent=0.03, max_scalerank=4, major_extent=0.6):
    """Ocean coast (land outline) + shores of the larger lakes, simplified, clipped to the domain.  Returns
    (major, minor): outlines at least `major_extent` degrees across (continents, big islands, big lakes) are drawn at
    every zoom, the rest (small islands, small lakes) only when zoomed in, so the continental view stays calm."""
    rings = [a for a in arcs_of(land_topo)]
    if lakes_geojson:
        for f in lakes_geojson["features"]:
            if (f["properties"].get("scalerank") if f["properties"].get("scalerank") is not None else 99) > max_scalerank:
                continue
            g = f["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                rings += [[tuple(c[:2]) for c in ring] for ring in poly]
    major, minor = [], []
    for ring in rings:
        for run in clip_runs(ring):
            q = rdp(run, eps)
            ext = max(q[:, 0].max() - q[:, 0].min(), q[:, 1].max() - q[:, 1].min())
            if len(q) >= 2 and ext >= min_extent:
                (major if ext >= major_extent else minor).append([[round(float(x), 3), round(float(y), 3)] for x, y in q])
    return major, minor


def main(us_dir, world_dir, out_dir, lakes_path=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    us_c = json.load(open(Path(us_dir) / "counties-10m.json"))
    us_s = json.load(open(Path(us_dir) / "states-10m.json"))
    world = json.load(open(Path(world_dir) / "countries-50m.json"))
    land = json.load(open(Path(world_dir) / "land-10m.json"))
    lakes = json.load(open(lakes_path)) if lakes_path else None
    for name, lines in zip(("coast", "coast_minor"), coast_lines(land, lakes)):
        (out / f"{name}.json").write_text(json.dumps({"type": "MultiLineString", "coordinates": lines}, separators=(",", ":")))
        print(name, len(lines), "lines,", sum(map(len, lines)), "points")
    if lakes_path == "--coast-only":
        return
    for name, topo, obj in (("counties", us_c, "counties"), ("states", us_s, "states"), ("countries", world, "countries")):
        lines = interior_lines(topo, obj)
        (out / f"{name}.json").write_text(json.dumps({"type": "MultiLineString", "coordinates": lines},
                                                     separators=(",", ":")))
        print(name, len(lines), "lines,", sum(map(len, lines)), "points")


if __name__ == "__main__":
    main(*sys.argv[1:5])
