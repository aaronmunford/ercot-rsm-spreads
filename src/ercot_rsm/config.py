"""Load ``config/study.toml`` into plain, validated objects."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "study.toml"


@dataclass(frozen=True)
class Zone:
    name: str
    price_point: str
    load_column: str
    stations: dict[str, float]


@dataclass(frozen=True)
class Study:
    raw: dict
    start: pd.Timestamp  # local midnight, tz-aware
    end: pd.Timestamp
    market_tz: str
    regime_break: pd.Timestamp
    zones: dict[str, Zone]
    stations: dict[str, tuple[float, float]]

    @property
    def start_utc(self) -> pd.Timestamp:
        return self.start.tz_convert("UTC")

    @property
    def end_utc(self) -> pd.Timestamp:
        return self.end.tz_convert("UTC")

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})

    @property
    def tag(self) -> str:
        """The window as a path component, e.g. ``2025-01-01_2026-01-01``.

        Every cache and output is keyed by it, so a one-week sample pulled to
        check the columns can never be mistaken for the full year.
        """
        return f"{self.start:%Y-%m-%d}_{self.end:%Y-%m-%d}"

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw" / self.tag

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.tag

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" / self.tag


def load(path: str | Path = DEFAULT_CONFIG, start: str | None = None, end: str | None = None) -> Study:
    """Read the config; ``start``/``end`` override the study window (for samples)."""
    raw = tomllib.loads(Path(path).read_text())
    s = raw["study"]
    if start:
        s["start"] = start
    if end:
        s["end"] = end
    tz = s["market_tz"]
    zones = {
        name: Zone(name, z["price_point"], z["load_column"], dict(z["stations"]))
        for name, z in raw["zones"].items()
    }
    stations = {k: (float(v[0]), float(v[1])) for k, v in raw.get("stations", {}).items()}

    for z in zones.values():
        missing = set(z.stations) - set(stations)
        if missing:
            raise ValueError(f"zone {z.name}: stations without coordinates {sorted(missing)}")
        total = sum(z.stations.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"zone {z.name}: station weights sum to {total}, not 1")

    start = pd.Timestamp(s["start"]).tz_localize(tz)
    end = pd.Timestamp(s["end"]).tz_localize(tz)
    if end <= start:
        raise ValueError("study end must be after start")

    return Study(
        raw=raw,
        start=start,
        end=end,
        market_tz=tz,
        regime_break=pd.Timestamp(s["regime_break"]).tz_localize(tz),
        zones=zones,
        stations=stations,
    )
