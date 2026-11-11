"""Hourly temperature for the configured stations.

Two sources, one output schema ``(station, valid_utc, tmpf)``:

* ``iem`` — ASOS/METAR observations from the Iowa Environmental Mesonet.
  Measured at the airports, no key. These are the same NWS/FAA ASOS reports
  that NOAA archives in its Integrated Surface Database; ``ncei_crosscheck``
  compares the two for a station-month.
* ``open_meteo`` — ERA5-family reanalysis at each station's lat/lon from the
  Open-Meteo archive API. Spatially complete, no gaps, but modelled.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

from ..config import Study
from . import manifest

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
# NOAA ISD (USAF+WBAN) ids for the cross-check stations.
NCEI_IDS = {"IAH": "72243012960", "DFW": "72259003927"}
RETRIABLE = {429, 500, 502, 503, 504}
PAUSE_S = 1.0  # between requests to the same public service


def get_with_retry(http: requests.Session, url: str, params, tries: int = 4, timeout: int = 300,
                   base_delay: float = 5.0) -> requests.Response:
    """GET with exponential backoff on network errors and 429/5xx; other 4xx fail at once."""
    for i in range(tries):
        try:
            r = http.get(url, params=params, timeout=timeout)
            if r.status_code not in RETRIABLE:
                r.raise_for_status()
                return r
            err: Exception = requests.HTTPError(f"HTTP {r.status_code}", response=r)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e
        if i == tries - 1:
            raise err
        delay = base_delay * 2 ** i
        print(f"  {type(err).__name__}: {err}; retrying in {delay:.0f}s")
        time.sleep(delay)
    raise AssertionError("unreachable")


def iem_params(station: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    a, b = start.tz_convert("UTC"), end.tz_convert("UTC") + pd.Timedelta(days=1)
    return [
        ("station", station), ("data", "tmpf"),
        ("year1", str(a.year)), ("month1", str(a.month)), ("day1", str(a.day)),
        ("year2", str(b.year)), ("month2", str(b.month)), ("day2", str(b.day)),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),  # routine + specials
    ]


def parse_iem(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), na_values=["M"], comment="#")
    if not {"station", "valid", "tmpf"} <= set(df.columns):
        raise ValueError(f"unexpected IEM response columns {list(df.columns)}: {text[:200]!r}")
    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    return df[["station", "valid_utc", "tmpf"]]


def open_meteo_params(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return {
        "latitude": lat, "longitude": lon,
        "start_date": start.tz_convert("UTC").strftime("%Y-%m-%d"),
        "end_date": end.tz_convert("UTC").strftime("%Y-%m-%d"),
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": "GMT",
    }


def parse_open_meteo(payload: dict, station: str) -> pd.DataFrame:
    h = payload["hourly"]
    return pd.DataFrame({
        "station": station,
        "valid_utc": pd.to_datetime(h["time"]).tz_localize("UTC"),
        "tmpf": pd.to_numeric(pd.Series(h["temperature_2m"]), errors="coerce"),
    })


def station_coverage(obs: pd.DataFrame, stations: list[str], start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    """Per station: observations, share of window hours with a reading, longest gap (hours)."""
    hours = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    valid = obs.dropna(subset=["tmpf"])
    rows = []
    for st in stations:
        o = valid[valid["station"] == st]
        present = hours.isin(pd.DatetimeIndex(o["valid_utc"].dt.round("h").unique()))
        miss = ~present
        runs = pd.Series(miss).groupby((~pd.Series(miss)).cumsum()).sum()
        rows.append({
            "station": st,
            "observations": int(len(o)),
            "hours_expected": int(len(hours)),
            "hours_with_obs": int(present.sum()),
            "pct_hours_with_obs": round(100 * present.mean(), 2),
            "longest_gap_hours": int(runs.max()) if len(runs) else int(len(hours)),
            "tmpf_min": float(o["tmpf"].min()) if len(o) else np.nan,
            "tmpf_max": float(o["tmpf"].max()) if len(o) else np.nan,
        })
    return pd.DataFrame(rows).set_index("station")


def fetch_all(study: Study, force: bool = False) -> pd.DataFrame:
    src = study.section("weather").get("source", "iem")
    raw = study.raw_dir / "weather"
    raw.mkdir(parents=True, exist_ok=True)
    out_file = raw / f"stations_{src}.parquet"
    if out_file.exists() and not force:
        return pd.read_parquet(out_file)
    stations = sorted({s for z in study.zones.values() for s in z.stations})
    parts = []
    with requests.Session() as http:
        for i, st in enumerate(stations):
            if i:
                time.sleep(PAUSE_S)
            if src == "iem":
                r = get_with_retry(http, IEM_URL, iem_params(st, study.start, study.end))
                df = parse_iem(r.text)
            elif src == "open_meteo":
                lat, lon = study.stations[st]
                r = get_with_retry(http, OPEN_METEO_URL, open_meteo_params(lat, lon, study.start, study.end))
                df = parse_open_meteo(r.json(), st)
            else:
                raise ValueError(f"unknown weather source {src!r}")
            print(f"{st}: {len(df):,} observations, {df['tmpf'].isna().mean():.1%} missing")
            parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(out_file, index=False)
    cov = station_coverage(out, stations, study.start, study.end)
    cov.to_csv(raw / f"station_coverage_{src}.csv")
    manifest.record(
        study, f"weather/{src}",
        source=IEM_URL if src == "iem" else OPEN_METEO_URL,
        description=("Iowa Environmental Mesonet ASOS archive: NWS/FAA ASOS METAR reports, routine and "
                     "special (report_type 3, 4)" if src == "iem" else "Open-Meteo ERA5-family reanalysis"),
        retrieved_utc=manifest.utc_now(), output=f"weather/{out_file.name}",
        query={"start_utc": str(study.start_utc), "end_utc_plus_1d": str(study.end_utc + pd.Timedelta(days=1)),
               "variable": "tmpf"},
        units="degrees Fahrenheit", time="valid_utc: observation time, UTC (METAR routine reports near :53)",
        geography={st: list(study.stations[st]) for st in stations},
        rows=int(len(out)), coverage=cov.to_dict(orient="index"),
    )
    return out


def parse_ncei_tmp(text: str) -> pd.DataFrame:
    """NCEI global-hourly CSV -> (valid_utc, report_type, tmpf); drops 9999 and failed QC codes."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if not {"DATE", "TMP", "REPORT_TYPE"} <= set(df.columns):
        raise ValueError(f"unexpected NCEI columns {list(df.columns)}: {text[:200]!r}")
    val, qc = df["TMP"].str.split(",", n=1, expand=True)[0], df["TMP"].str.split(",", n=1, expand=True)[1]
    c = pd.to_numeric(val, errors="coerce") / 10.0
    ok = c.notna() & (val.str.lstrip("+-") != "9999") & ~qc.isin(["2", "3", "6", "7"])
    return pd.DataFrame({
        "valid_utc": pd.to_datetime(df["DATE"], utc=True),
        "report_type": df["REPORT_TYPE"].str.strip(),
        "tmpf": c * 9 / 5 + 32,
    })[ok].reset_index(drop=True)


def ncei_crosscheck(study: Study, station: str, month: str, iem_obs: pd.DataFrame) -> dict:
    """Compare IEM with NOAA NCEI global-hourly METARs (FM-15/16) for one station-month."""
    a = pd.Timestamp(month + "-01", tz="UTC")
    b = a + pd.offsets.MonthBegin(1)
    with requests.Session() as http:
        r = get_with_retry(http, NCEI_URL, {
            "dataset": "global-hourly", "stations": NCEI_IDS[station], "dataTypes": "TMP", "format": "csv",
            "startDate": a.strftime("%Y-%m-%dT%H:%M:%S"), "endDate": b.strftime("%Y-%m-%dT%H:%M:%S")})
    n = parse_ncei_tmp(r.text)
    n = n[n["report_type"].isin(["FM-15", "FM-16"])]
    i = iem_obs[(iem_obs["station"] == station) & iem_obs["valid_utc"].between(a, b, inclusive="left")]
    m = n.merge(i.dropna(subset=["tmpf"]), on="valid_utc", suffixes=("_ncei", "_iem"))
    d = m["tmpf_iem"] - m["tmpf_ncei"]
    res = {
        "station": station, "ncei_id": NCEI_IDS[station], "month": month,
        "ncei_metars": int(len(n)), "iem_obs": int(i["tmpf"].notna().sum()), "matched_timestamps": int(len(m)),
        "mean_abs_diff_f": float(d.abs().mean()), "max_abs_diff_f": float(d.abs().max()),
        "share_within_1f": float((d.abs() <= 1.0).mean()),
    }
    manifest.record(study, f"weather/ncei_crosscheck/{station}_{month}",
                    source=NCEI_URL, dataset="global-hourly (NOAA ISD)", retrieved_utc=manifest.utc_now(),
                    purpose="cross-check of IEM ASOS temperatures; not used in the panel", result=res)
    return res
