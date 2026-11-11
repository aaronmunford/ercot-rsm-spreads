"""Print what each raw source and the joined panel actually look like.

Run on a one-week sample before pulling a year: columns, dtypes, time range,
sampling step, locations, and a few rows of each, so that a wrong dataset id,
a column name or a timestamp convention is caught on a few thousand rows.
"""

from __future__ import annotations

import pandas as pd

TIME_COLUMNS = ("interval_start_utc", "valid_utc", "trade_date")


def describe_frame(name: str, df: pd.DataFrame, rows: int = 3) -> str:
    lines = [f"=== {name}: {len(df):,} rows x {df.shape[1]} columns"]
    lines.append("  columns: " + ", ".join(f"{c} [{t}]" for c, t in df.dtypes.astype(str).items()))
    tcol = next((c for c in TIME_COLUMNS if c in df.columns), None)
    if tcol is not None and len(df):
        t = pd.to_datetime(df[tcol])
        lines.append(f"  {tcol}: {t.min()} -> {t.max()}")
        key = next((c for c in ("location", "station") if c in df.columns), None)
        groups = [g for _, g in t.groupby(df[key])] if key else [t]
        steps = sorted({str(g.sort_values().diff().median()) for g in groups})
        lines.append(f"  median step: {', '.join(steps)}")
    for c in ("location", "station", "zone"):
        if c in df.columns:
            vals = df[c].value_counts().sort_index()
            lines.append(f"  {c}: " + ", ".join(f"{k} ({v:,})" for k, v in vals.items()))
    lines.append(df.head(rows).to_string(max_cols=12))
    return "\n".join(lines)


def report(raw, panel: pd.DataFrame, quality: pd.DataFrame) -> str:
    parts = [
        describe_frame("DA prices (hourly)", raw.da),
        describe_frame("RT prices (expect 15-min)", raw.rt),
        describe_frame("Load by forecast zone (wide)", raw.load_wide),
        describe_frame("Station temperatures", raw.station_obs),
        describe_frame("Henry Hub gas (daily, by trade date)", raw.gas),
    ]
    if raw.fuel_mix is not None:
        parts.append(describe_frame("Fuel mix (hourly)", raw.fuel_mix))
    core = ["interval_start_local", "zone", "da_price", "rt_price", "spread", "load_mw",
            "temp_f", "gas_price", "imhr_da"]
    parts.append(describe_frame("PANEL (one row per zone-hour)", panel[[c for c in core if c in panel]], rows=8))
    parts.append("=== Coverage (expected hours are DST-correct)\n" + quality.to_string())
    return "\n\n".join(parts)
