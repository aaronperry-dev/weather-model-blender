import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / "web" / "borders"


def _lines(name):
    g = json.loads((B / f"{name}.json").read_text())
    assert g["type"] == "MultiLineString"
    return g["coordinates"]


def test_border_layers_exist_and_cover_conus():
    c, s, k = _lines("countries"), _lines("states"), _lines("counties")
    assert len(c) >= 5 and len(s) >= 50 and len(k) > 3000
    pts = [p for ln in c for p in ln]
    assert any(abs(lat - 49.0) < 0.05 and -120 < lon < -100 for lon, lat in pts)      # US-Canada 49th parallel
    assert any(30 < lat < 33 and -115 < lon < -100 for lon, lat in pts)                # US-Mexico border
    assert all(-136 < lon < -49 or lat < 20 for ln in s for lon, lat in ln)


def test_workflow_copies_web_subfolders():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pwpf.yml").read_text()
    assert "cp -r web/. site/" in wf
