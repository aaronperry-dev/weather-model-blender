"""Config loading with profiles and simple overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, profile: str | None = None, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    profiles = cfg.pop("profiles", {}) or {}
    if profile:
        if profile not in profiles:
            raise KeyError(f"profile {profile!r} not in {sorted(profiles)}")
        prof = copy.deepcopy(profiles[profile])
        counts = prof.pop("member_counts", {})
        disable = set(prof.pop("disable", []))
        replace = prof.pop("members_replace", None)
        cfg = _deep_merge(cfg, prof)
        if replace is not None:
            cfg["members"] = replace
        else:
            cfg["members"] = [dict(m, count=counts.get(m["name"], m.get("count")))
                              if "count" in m else m for m in cfg["members"] if m["name"] not in disable]
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
