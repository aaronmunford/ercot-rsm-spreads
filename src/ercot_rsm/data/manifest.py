"""Provenance for every raw file: where it came from, when, and what it holds.

One JSON file per window, ``data/raw/<window>/manifest.json``, keyed by a
source name (``gridstatus/ercot_spp_day_ahead_hourly``, ``weather/iem``, ...).
Entries are upserted with a recursive merge, so re-running a fetch that only
validates cached chunks keeps the original retrieval times of those chunks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import Study


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path(study: Study) -> Path:
    return study.raw_dir / "manifest.json"


def _merge(old: dict, new: dict) -> dict:
    out = dict(old)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(study: Study) -> dict:
    p = path(study)
    return json.loads(p.read_text()) if p.exists() else {}


def record(study: Study, name: str, **meta) -> dict:
    """Upsert ``meta`` under ``name``; returns the merged entry."""
    p = path(study)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load(study)
    data[name] = _merge(data.get(name, {}), {**meta, "recorded_utc": utc_now()})
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(p)  # atomic: an interrupted run never leaves half a manifest
    return data[name]
