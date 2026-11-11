"""ERCOT's own hourly native load archive, used to cross-check the gridstatus load.

ERCOT publishes one workbook per year on https://www.ercot.com/gridinfo/load/load_hist
("Native_Load_<year>.zip"), hourly by **weather zone** (COAST, EAST, FWEST,
NORTH, NCENT, SOUTH, SCENT, WEST) plus the ERCOT total. Weather zones are not
the load forecast zones the panel uses, so only the system total is compared;
the weather-zone columns are kept for the zone-mapping discussion.

Timestamps are "Hour Ending" in Central Prevailing Time, ``MM/DD/YYYY HH:00``
with HE01..HE24. The spring-forward day has no HE03, and on the fall-back day
the repeated hour is the row suffixed ``DST`` (the second, CST, occurrence).
"""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from ..config import Study
from . import manifest

LOAD_HIST_URL = "https://www.ercot.com/gridinfo/load/load_hist"
UTC = "interval_start_utc"
HEADERS = {"User-Agent": "ercot-rsm research pipeline (python-requests)"}


def find_archive_url(html: str, year: int) -> str:
    """The Native_Load_<year> zip link on the load_hist page (its path carries an upload date)."""
    links = re.findall(rf'href="([^"]*Native_Load_{year}[^"]*\.zip)"', html, flags=re.I)
    if not links:
        raise ValueError(f"no Native_Load_{year} archive linked from {LOAD_HIST_URL}")
    url = links[0]
    return url if url.startswith("http") else "https://www.ercot.com" + url


def parse_native_load(df: pd.DataFrame, tz: str = "America/Chicago") -> pd.DataFrame:
    """Workbook frame -> (interval_start_utc, <weather zones>, ERCOT), hour-beginning UTC."""
    he = df.iloc[:, 0].astype(str).str.strip()
    m = he.str.extract(r"^(\d{2}/\d{2}/\d{4})\s+(\d{1,2}):00\s*(DST)?$")
    if m[0].isna().any():
        bad = he[m[0].isna()].head(3).tolist()
        raise ValueError(f"unexpected Hour Ending values, e.g. {bad}")
    day = pd.to_datetime(m[0], format="%m/%d/%Y")
    local_start = day + pd.to_timedelta(m[1].astype(int) - 1, unit="h")
    repeated = m[2].notna().to_numpy()
    # ambiguous=True means "the DST occurrence"; the row marked DST is the later, CST one.
    start = pd.DatetimeIndex(local_start).tz_localize(tz, ambiguous=~repeated, nonexistent="raise")
    out = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    out.insert(0, UTC, start.tz_convert("UTC"))
    if out[UTC].duplicated().any():
        raise ValueError("duplicate hours after DST handling")
    return out.sort_values(UTC).reset_index(drop=True)


def fetch_native_load(study: Study, force: bool = False) -> pd.DataFrame:
    year = study.start.year
    raw = study.raw_dir / "ercot"
    raw.mkdir(parents=True, exist_ok=True)
    out_file = raw / f"native_load_{year}.parquet"
    if out_file.exists() and not force:
        return pd.read_parquet(out_file)
    page = requests.get(LOAD_HIST_URL, headers=HEADERS, timeout=60)
    page.raise_for_status()
    url = find_archive_url(page.text, year)
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    (raw / url.rsplit("/", 1)[-1]).write_bytes(r.content)  # keep the original archive
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls")))
        df = parse_native_load(pd.read_excel(z.open(name)), study.market_tz)
    df.to_parquet(out_file, index=False)
    manifest.record(
        study, "ercot/native_load",
        source=LOAD_HIST_URL, url=url, file=name, last_modified=r.headers.get("Last-Modified"),
        retrieved_utc=manifest.utc_now(), rows=int(len(df)),
        span_utc=[str(df[UTC].min()), str(df[UTC].max())],
        units="MW, hourly native load", frequency="1h",
        time="Hour Ending, Central Prevailing Time, converted to interval_start_utc (hour beginning)",
        geography="ERCOT weather zones COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT, WEST; ERCOT total",
        purpose="cross-check of the gridstatus forecast-zone load total; not used in the panel",
    )
    print(f"ERCOT native load {year}: {len(df):,} hours from {url}")
    return df


def compare_totals(gridstatus_load: pd.DataFrame, native: pd.DataFrame,
                   gs_total: str = "total", ercot_total: str = "ERCOT") -> dict:
    """Agreement of the gridstatus forecast-zone total with ERCOT's native-load total."""
    m = gridstatus_load[[UTC, gs_total]].merge(native[[UTC, ercot_total]], on=UTC, how="inner").dropna()
    diff = m[gs_total] - m[ercot_total]
    return {
        "hours_compared": int(len(m)),
        "hours_gridstatus": int(gridstatus_load[UTC].nunique()),
        "hours_ercot": int(native[UTC].nunique()),
        "corr": float(np.corrcoef(m[gs_total], m[ercot_total])[0, 1]) if len(m) > 2 else float("nan"),
        "mape_pct": float((diff.abs() / m[ercot_total]).mean() * 100),
        "mean_diff_mw": float(diff.mean()),
        "max_abs_diff_mw": float(diff.abs().max()),
        "max_abs_diff_at": str(m.loc[diff.abs().idxmax(), UTC]) if len(m) else None,
    }
