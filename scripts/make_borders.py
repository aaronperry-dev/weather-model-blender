"""Build the viewer's border overlays (web/borders/*.json) from TopoJSON files.

    npm i us-atlas world-atlas            # Census cartographic boundaries / Natural Earth, public domain
    python scripts/make_borders.py node_modules/us-atlas node_modules/world-atlas web/borders

Only *interior* boundaries are kept (arcs shared by two features: country-country, state-state, county-county),
clipped to the model domain and rounded to 0.001 deg, as one MultiLineString per layer.  Coastlines are left to the
basemap.  The output is committed, so CI never needs these packages."""
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


def main(us_dir, world_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    us_c = json.load(open(Path(us_dir) / "counties-10m.json"))
    us_s = json.load(open(Path(us_dir) / "states-10m.json"))
    world = json.load(open(Path(world_dir) / "countries-50m.json"))
    for name, topo, obj in (("counties", us_c, "counties"), ("states", us_s, "states"), ("countries", world, "countries")):
        lines = interior_lines(topo, obj)
        (out / f"{name}.json").write_text(json.dumps({"type": "MultiLineString", "coordinates": lines},
                                                     separators=(",", ":")))
        print(name, len(lines), "lines,", sum(map(len, lines)), "points")


if __name__ == "__main__":
    main(*sys.argv[1:4])
