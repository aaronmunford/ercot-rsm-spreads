"""ERCOT's official annual settlement-point price archives: every load zone and hub.

* DAM: "Historical DAM Load Zone and Hub Prices" (NP4-180-ER, report type 13060),
  ``DAMLZHBSPP_<year>.zip``: hourly, Hour Ending ``HH:00`` (01..24).
* RTM: "Historical RTM Load Zone and Hub Prices" (NP6-785-ER, report type 13061),
  ``RTMLZHBSPP_<year>.zip``: 15-minute, Delivery Hour 1..24 x Delivery Interval 1..4.

Both are public (no login, no API quota) and carry a ``Repeated Hour Flag``:
``Y`` marks the second, standard-time occurrence of the fall-back hour. One
workbook per year, one sheet per month. Load zones appear in the RTM file
under two point types, ``LZ`` and ``LZEW``; which one matches the series used
elsewhere is settled by the cross-check in ``compare_with``.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from ..config import Study
from . import manifest

LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rid}"
DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc}"
REPORTS = {"da": (13060, "DAMLZHBSPP"), "rt": (13061, "RTMLZHBSPP")}
HEADERS = {"User-Agent": "ercot-rsm research pipeline (python-requests)"}
UTC = "interval_start_utc"


def list_documents(kind: str) -> pd.DataFrame:
    rid, _ = REPORTS[kind]
    r = requests.get(LIST_URL.format(rid=rid), headers=HEADERS, timeout=60)
    r.raise_for_status()
    docs = [d["Document"] for d in r.json()["ListDocsByRptTypeRes"]["DocumentList"]]
    return pd.DataFrame({"name": [d["FriendlyName"] for d in docs], "doc_id": [d["DocID"] for d in docs],
                         "published": [d["PublishDate"] for d in docs], "bytes": [int(d["ContentSize"]) for d in docs]})


def _localize(local: pd.Series, repeated: pd.Series, tz: str) -> pd.Series:
    # ambiguous=True means the DST (first) occurrence; the flagged row is the second, CST one
    idx = pd.DatetimeIndex(local).tz_localize(tz, ambiguous=~repeated.to_numpy(bool), nonexistent="raise")
    return pd.Series(idx.tz_convert("UTC"), index=local.index)


def parse_da(sheets: dict[str, pd.DataFrame], tz: str = "America/Chicago") -> pd.DataFrame:
    df = pd.concat(sheets.values(), ignore_index=True)
    day = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    he = df["Hour Ending"].astype(str).str.slice(0, 2).astype(int)
    rep = df["Repeated Hour Flag"].astype(str).str.strip().eq("Y")
    out = pd.DataFrame({UTC: _localize(day + pd.to_timedelta(he - 1, unit="h"), rep, tz),
                        "location": df["Settlement Point"].astype(str), "point_type": "DA",
                        "price": pd.to_numeric(df["Settlement Point Price"], errors="coerce")})
    return _check(out, "DA")


def parse_rt(sheets: dict[str, pd.DataFrame], tz: str = "America/Chicago") -> pd.DataFrame:
    df = pd.concat(sheets.values(), ignore_index=True)
    day = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    off = pd.to_timedelta(df["Delivery Hour"].astype(int) - 1, unit="h") + \
        pd.to_timedelta((df["Delivery Interval"].astype(int) - 1) * 15, unit="min")
    rep = df["Repeated Hour Flag"].astype(str).str.strip().eq("Y")
    out = pd.DataFrame({UTC: _localize(day + off, rep, tz), "location": df["Settlement Point Name"].astype(str),
                        "point_type": df["Settlement Point Type"].astype(str),
                        "price": pd.to_numeric(df["Settlement Point Price"], errors="coerce")})
    return _check(out, "RT")


def _check(out: pd.DataFrame, kind: str) -> pd.DataFrame:
    dup = out.duplicated([UTC, "location", "point_type"]).sum()
    if dup:
        raise ValueError(f"{kind} archive: {dup} duplicate (interval, point, type) rows after DST handling")
    if out["price"].isna().any():
        raise ValueError(f"{kind} archive: {int(out['price'].isna().sum())} unparseable prices")
    return out.sort_values([UTC, "location", "point_type"]).reset_index(drop=True)


def read_workbook(content: bytes) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls")))
        data = z.read(name)
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=None, engine="calamine")
    except ImportError:  # pragma: no cover
        return pd.read_excel(io.BytesIO(data), sheet_name=None)


def fetch(study: Study, kind: str, force: bool = False) -> pd.DataFrame:
    """Every settlement point in the archive for the study window, cached as parquet."""
    raw = study.raw_dir / "ercot"
    raw.mkdir(parents=True, exist_ok=True)
    out_file = raw / f"{kind}_prices_archive.parquet"
    if out_file.exists() and not force:
        return pd.read_parquet(out_file)
    docs = list_documents(kind)
    _, prefix = REPORTS[kind]
    years = range(study.start.year, (study.end - pd.Timedelta(seconds=1)).year + 1)
    parts, files = [], []
    for y in years:
        hit = docs[docs["name"] == f"{prefix}_{y}"]
        if hit.empty:
            raise ValueError(f"no {prefix}_{y} in ERCOT report list")
        doc = hit.iloc[0]
        zpath = raw / f"{prefix}_{y}.zip"
        if not zpath.exists() or force:
            r = requests.get(DOWNLOAD_URL.format(doc=doc["doc_id"]), headers=HEADERS, timeout=600)
            r.raise_for_status()
            zpath.write_bytes(r.content)
        sheets = read_workbook(zpath.read_bytes())
        parts.append(parse_da(sheets, study.market_tz) if kind == "da" else parse_rt(sheets, study.market_tz))
        files.append({"name": doc["name"], "doc_id": str(doc["doc_id"]), "published": doc["published"],
                      "bytes": int(doc["bytes"])})
    df = pd.concat(parts, ignore_index=True)
    df = df[(df[UTC] >= study.start_utc) & (df[UTC] < study.end_utc)].reset_index(drop=True)
    df.to_parquet(out_file, index=False)
    manifest.record(
        study, f"ercot/{kind}_price_archive",
        source=LIST_URL.format(rid=REPORTS[kind][0]), download=DOWNLOAD_URL.format(doc="<doc_id>"), files=files,
        retrieved_utc=manifest.utc_now(), rows=int(len(df)), points=sorted(df["location"].unique()),
        point_types=sorted(df["point_type"].unique()),
        units="$/MWh settlement point price", frequency="1h" if kind == "da" else "15min",
        time="Delivery Date + Hour Ending (+ Interval), Central Prevailing Time with Repeated Hour Flag, "
             "converted to interval_start_utc",
    )
    print(f"ERCOT {kind.upper()} archive: {len(df):,} rows, {df['location'].nunique()} points")
    return df


def to_prices(archive: pd.DataFrame, points: list[str], point_type: str | None) -> pd.DataFrame:
    """Archive rows -> the pipeline's (interval_start_utc, location, price) for ``points``."""
    d = archive[archive["location"].isin(points)]
    if point_type is not None:
        d = d[d["point_type"].isin([point_type, "HU", "AH", "SH"]) | ~d["location"].str.startswith("LZ_")]
        d = d[~(d["location"].str.startswith("LZ_") & (d["point_type"] != point_type))]
    out = d[[UTC, "location", "price"]]
    if out.duplicated([UTC, "location"]).any():
        raise ValueError("more than one price per (interval, point): choose a point_type")
    return out.reset_index(drop=True)


def compare_with(archive: pd.DataFrame, other: pd.DataFrame, point_type: str | None) -> dict:
    """Agreement of archive prices with another source (interval_start_utc, location, price)."""
    a = to_prices(archive, sorted(other["location"].unique()), point_type)
    m = a.merge(other, on=[UTC, "location"], how="outer", suffixes=("_ercot", "_other"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["price_ercot"] - both["price_other"]).abs()
    return {"point_type": point_type, "rows_both": int(len(both)),
            "only_ercot": int((m["_merge"] == "left_only").sum()), "only_other": int((m["_merge"] == "right_only").sum()),
            "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
            "rows_differ": int((diff > 0.005).sum())}
