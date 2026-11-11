"""Henry Hub daily spot natural gas price, $/MMBtu, keyed by trade date.

Three sources, one series (RNGWHHD) and one output schema ``(trade_date, price)``:

* ``eia_xls`` — EIA's own history workbook for the series, no key. Checked
  against FRED's mirror (DHHNGSP) on 2026-10-01: identical on every one of the
  454 trade dates from 2024-12-01 on.
* ``eia_api`` — EIA's API v2 (free key at https://www.eia.gov/opendata/, in
  ``EIA_API_KEY``).
* ``csv`` — a file you downloaded by hand: any two columns that parse as a
  date and a price.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests

from ..config import ROOT, Study
from . import manifest

EIA_URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
EIA_XLS_URL = "https://www.eia.gov/dnav/ng/hist_xls/{series}d.xls"  # trailing d = daily
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# A month of lead-in so the first operating day has a prior trade date.
LEAD_IN = pd.Timedelta(days=31)


def eia_params(series: str, start: pd.Timestamp, end: pd.Timestamp, api_key: str) -> list[tuple[str, str]]:
    a = (start - LEAD_IN).strftime("%Y-%m-%d")
    return [
        ("api_key", api_key), ("frequency", "daily"), ("data[0]", "value"),
        ("facets[series][]", series), ("start", a), ("end", end.strftime("%Y-%m-%d")),
        ("sort[0][column]", "period"), ("sort[0][direction]", "asc"), ("length", "5000"),
    ]


def parse_eia(payload: dict) -> pd.DataFrame:
    rows = payload["response"]["data"]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("EIA returned no rows — check the series id and dates")
    return pd.DataFrame({
        "trade_date": pd.to_datetime(df["period"]).dt.date,
        "price": pd.to_numeric(df["value"], errors="coerce"),
    }).dropna().sort_values("trade_date").reset_index(drop=True)


def parse_eia_sheet(sheet: pd.DataFrame, series: str) -> pd.DataFrame:
    """The "Data 1" sheet of an EIA history workbook, read with ``header=None``.

    Layout: a ``Sourcekey`` row naming the series, then a ``Date`` header row,
    then one (date, price) row per trade date.
    """
    first = sheet.iloc[:, 0].astype(str).str.strip()
    at = first.index[first.eq("Sourcekey")]
    if len(at) != 1 or str(sheet.iloc[at[0], 1]).strip() != series:
        raise ValueError(f"not an EIA workbook for {series}: {sheet.head(4).values.tolist()!r}")
    if first.iloc[at[0] + 1] != "Date":
        raise ValueError(f"expected a Date header after Sourcekey, got {first.iloc[at[0] + 1]!r}")
    data = sheet.iloc[at[0] + 2:, :2]
    return pd.DataFrame({
        "trade_date": pd.to_datetime(data.iloc[:, 0]).dt.date,
        "price": pd.to_numeric(data.iloc[:, 1], errors="coerce"),
    }).dropna().sort_values("trade_date").reset_index(drop=True)


def parse_eia_xls(content: bytes, series: str) -> pd.DataFrame:
    return parse_eia_sheet(pd.read_excel(io.BytesIO(content), sheet_name="Data 1", header=None), series)


def in_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Trade dates from the lead-in through the window's last day."""
    a, b = (start - LEAD_IN).date(), end.date()
    return df[(df["trade_date"] >= a) & (df["trade_date"] <= b)].reset_index(drop=True)


def parse_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9)
    price_col = next(c for c in df.columns
                     if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9)
    return pd.DataFrame({
        "trade_date": pd.to_datetime(df[date_col]).dt.date,
        "price": pd.to_numeric(df[price_col], errors="coerce"),
    }).dropna().sort_values("trade_date").reset_index(drop=True)


def fetch(study: Study, force: bool = False) -> pd.DataFrame:
    cfg = study.section("gas")
    raw = study.raw_dir / "gas"
    raw.mkdir(parents=True, exist_ok=True)
    out_file = raw / "henry_hub_daily.parquet"
    if out_file.exists() and not force:
        return pd.read_parquet(out_file)
    src = cfg.get("source", "eia_xls")
    if src == "eia_xls":
        r = requests.get(EIA_XLS_URL.format(series=cfg["series"]), timeout=120)
        r.raise_for_status()
        df = in_window(parse_eia_xls(r.content, cfg["series"]), study.start, study.end)
    elif src == "csv":
        df = in_window(parse_csv(ROOT / cfg["csv_path"]), study.start, study.end)
    elif src == "eia_api":
        key = os.environ.get("EIA_API_KEY")
        if not key:
            raise SystemExit('set EIA_API_KEY, or use [gas] source = "eia_xls", which needs no key')
        r = requests.get(EIA_URL, params=eia_params(cfg["series"], study.start, study.end, key), timeout=120)
        r.raise_for_status()
        df = parse_eia(r.json())
    else:
        raise ValueError(f"unknown gas source {src!r}")
    if df.empty:
        raise ValueError(f"no {cfg['series']} prices for {study.tag}")
    print(f"gas: {len(df)} trade dates {df.trade_date.min()}..{df.trade_date.max()}")
    df.to_parquet(out_file, index=False)
    manifest.record(
        study, "gas/henry_hub",
        source={"eia_xls": EIA_XLS_URL.format(series=cfg["series"]), "eia_api": EIA_URL,
                "csv": cfg.get("csv_path")}[src],
        series=f"EIA {cfg['series']}: Henry Hub Natural Gas Spot Price (EIA credits Refinitiv/LSEG)",
        retrieved_utc=manifest.utc_now(), output="gas/henry_hub_daily.parquet", rows=int(len(df)),
        trade_dates=[str(df.trade_date.min()), str(df.trade_date.max())],
        units="$/MMBtu, nominal", frequency="daily, trade dates only (no weekends or exchange holidays)",
        lead_in_days=LEAD_IN.days, geography="Henry Hub, Erath, Louisiana (national benchmark)",
        alignment=cfg.get("alignment", "previous_trade_date"),
    )
    return df


def parse_fred_csv(text: str) -> pd.DataFrame:
    """FRED ``fredgraph.csv`` -> (trade_date, price); blank or '.' values (holidays) dropped."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2:
        raise ValueError(f"unexpected FRED columns {list(df.columns)}")
    return pd.DataFrame({
        "trade_date": pd.to_datetime(df.iloc[:, 0]).dt.date,
        "price": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
    }).dropna().reset_index(drop=True)


def compare_series(primary: pd.DataFrame, other: pd.DataFrame, tol: float = 0.005) -> dict:
    """Date-by-date agreement of two (trade_date, price) series over the primary's dates."""
    m = primary.merge(other, on="trade_date", how="outer", suffixes=("", "_other"), indicator=True)
    m = m[m["trade_date"] >= primary["trade_date"].min()]
    m = m[m["trade_date"] <= primary["trade_date"].max()]
    both = m[m["_merge"] == "both"]
    diff = (both["price"] - both["price_other"]).abs()
    return {
        "dates_primary": int((m["_merge"] != "right_only").sum()),
        "dates_other": int((m["_merge"] != "left_only").sum()),
        "dates_both": int(len(both)),
        "only_primary": [str(d) for d in m.loc[m["_merge"] == "left_only", "trade_date"]],
        "only_other": [str(d) for d in m.loc[m["_merge"] == "right_only", "trade_date"]],
        "mismatches": [{"trade_date": str(r.trade_date), "price": r.price, "other": r.price_other}
                       for r in both[diff > tol].itertuples()],
        "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
    }


def crosscheck_fred(study: Study, gas: pd.DataFrame) -> dict:
    """Compare the fetched Henry Hub series with FRED's mirror (DHHNGSP, no key needed)."""
    a, b = gas["trade_date"].min(), gas["trade_date"].max()
    r = requests.get(FRED_URL, params={"id": "DHHNGSP", "cosd": str(a), "coed": str(b)}, timeout=60)
    r.raise_for_status()
    res = compare_series(gas, parse_fred_csv(r.text))
    manifest.record(study, "gas/fred_crosscheck", source=FRED_URL + "?id=DHHNGSP",
                    retrieved_utc=manifest.utc_now(),
                    purpose="cross-check of the EIA Henry Hub series; not used in the panel", result=res)
    return res
