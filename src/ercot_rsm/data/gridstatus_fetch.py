"""ERCOT prices and load from gridstatus.io, cached month by month.

Needs ``GRIDSTATUS_API_KEY`` in the environment (free account at
https://gridstatus.io). Every price pull is filtered server-side to the
configured settlement points: an unfiltered ERCOT price pull covers ~1,000
locations and would burn a free plan's row allowance in one call.

A successful response is not proof of a complete one, so every monthly chunk
is checked against the intervals [start, end) must contain (``validate_chunk``)
before it is cached, and again whenever it is read back from the cache. Calls
are paced to the free plan's 1 request/s, and each dataset is priced against
the remaining monthly quota before any request is made (``check_budget``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ..config import Study
from . import manifest

UTC = "interval_start_utc"
PRICE_COLUMNS = ("spp", "lmp", "price")
HOUR = pd.Timedelta(hours=1)
QUARTER = pd.Timedelta(minutes=15)

MIN_SECONDS_BETWEEN_CALLS = 1.1  # free plan: 1 request/s
PAGE_ROW_LIMIT = 50_000          # free plan: rows per response
ROW_RESERVE = 50_000             # a pull may never leave less than this of the monthly row quota
REQUEST_RESERVE = 20             # ... or fewer requests than this


class BudgetError(RuntimeError):
    """A pull would exceed, or is being charged beyond, the free-plan quota."""


class ChunkError(ValueError):
    """A response contains rows it should not (wrong location, window, grid, conflicts)."""


@dataclass(frozen=True)
class Spec:
    """One gridstatus dataset as this study uses it."""

    name: str          # normalised output: data/raw/<window>/gridstatus/<name>.parquet
    config_key: str    # key in [gridstatus] holding the dataset id
    freq: pd.Timedelta
    by_location: bool  # filtered to the configured settlement points
    units: str
    extra: dict = field(default_factory=dict)  # extra query arguments
    chunk: str = "month"  # "month" | "quarter" (UTC boundaries)
    variant: str = ""     # cache sub-directory, so two vintages of one dataset never share chunks
    cutoff: str | None = None  # HH:MM:SS market time on D-1; every row must be published by then


SPECS = {
    "da": Spec("da_prices", "da_dataset", HOUR, True,
               "spp in $/MWh: Day-Ahead Market settlement point price, hourly"),
    "rt": Spec("rt_prices", "rt_dataset", QUARTER, True,
               "spp in $/MWh: real-time settlement point price, 15-minute settlement interval"),
    "load": Spec("load_wide", "load_dataset", HOUR, False,
                 "MW: hourly actual load by ERCOT load forecast zone (north, south, west, houston) and total"),
    "fuel_mix": Spec("fuel_mix_hourly", "fuel_mix_dataset", HOUR, False,
                     "MW: ERCOT system generation by fuel, server-side hourly mean of 5-minute data",
                     {"resample": "1 hour", "resample_function": "mean"}),
}

# DA-time forecast vintages (D-28). ``latest_before:-1 day@HH:MM:SS`` returns, for each
# operating hour, the newest forecast published at or before that market-time clock
# time on the previous day. 09:45 is the primary cutoff: DAM bids close at 10:00 and
# can only be extended, never brought forward (Protocols 4.5.1, NPRR478).
FORECASTS = {
    "load_fc": ("load_forecast_dataset", "MW: ERCOT seven-day load forecast by forecast zone (NP3-560-CD)"),
    "wind_fc": ("wind_dataset", "MW: ERCOT STWPF / WGRPP wind forecasts by region (NP4-742-CD)"),
    "solar_fc": ("solar_dataset", "MW: ERCOT STPPF / PVGRPP solar forecasts by region (NP4-745-CD)"),
}
ACTUALS = {
    "wind_act": ("wind_dataset", "MW: ERCOT hourly wind generation and HSL by region, latest report (actuals)"),
    "solar_act": ("solar_dataset", "MW: ERCOT hourly solar generation and HSL by region, latest report (actuals)"),
}


def _hhmm(cutoff: str) -> str:
    return cutoff.replace(":", "")[:4]


def all_specs(study: Study) -> dict[str, Spec]:
    """Base datasets plus the forecast vintages and actuals configured in [forecasts]."""
    cfg = study.section("forecasts")
    out = dict(SPECS)
    cutoffs = {"": cfg.get("cutoff", "09:45:00")}
    if cfg.get("sensitivity_cutoff"):
        cutoffs["_sens"] = cfg["sensitivity_cutoff"]
    for key, (cfg_key, units) in FORECASTS.items():
        for suffix, cut in cutoffs.items():
            v = f"latest_before_{_hhmm(cut)}"
            out[key + suffix] = Spec(f"{key}_{_hhmm(cut)}", cfg_key, HOUR, False, units,
                                     {"publish_time": f"latest_before:-1 day@{cut}", "timezone": "market"},
                                     chunk="quarter", variant=v, cutoff=cut)
    for key, (cfg_key, units) in ACTUALS.items():
        out[key] = Spec(key, cfg_key, HOUR, False, units, {"publish_time": "latest"},
                        chunk="quarter", variant="latest")
    return out


# --------------------------------------------------------------------------
# Client, pacing, quota
# --------------------------------------------------------------------------


def client():
    from gridstatusio import GridStatusClient

    return GridStatusClient()  # reads GRIDSTATUS_API_KEY


_last_call = 0.0


def _paced(fn, *args, **kw):
    """Call ``fn`` no sooner than MIN_SECONDS_BETWEEN_CALLS after the previous API call ended."""
    global _last_call
    wait = _last_call + MIN_SECONDS_BETWEEN_CALLS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        return fn(*args, **kw)
    finally:
        _last_call = time.monotonic()


def usage(c) -> dict:
    u = _paced(c.get_api_usage)
    lim, cur = u["limits"], u["current_period_usage"]
    return {
        "requests": int(cur["total_requests"]),
        "rows": int(cur["total_api_rows_returned"]),
        "requests_limit": int(lim["api_requests_limit"]),
        "rows_limit": int(lim["api_rows_returned_limit"]),
        "period_end": u.get("current_usage_period_end"),
    }


def check_budget(u: dict, rows: int, requests: int,
                 row_reserve: int = ROW_RESERVE, request_reserve: int = REQUEST_RESERVE) -> None:
    rows_left = u["rows_limit"] - u["rows"]
    req_left = u["requests_limit"] - u["requests"]
    if rows > rows_left - row_reserve or requests > req_left - request_reserve:
        raise BudgetError(
            f"pull needs ~{rows:,} rows / {requests} requests, but only {rows_left:,} rows / "
            f"{req_left} requests are left this period (keeping {row_reserve:,} rows / "
            f"{request_reserve} requests in reserve)")


def check_charge(charged: int, returned: int) -> None:
    """Stop if the API bills far more rows than it returned (e.g. counts before a resample)."""
    if charged > 2 * returned + 1_000:
        raise BudgetError(f"charged {charged:,} rows for {returned:,} returned; stopping before "
                          f"the rest of the year is billed at that rate")


# --------------------------------------------------------------------------
# Windows and validation
# --------------------------------------------------------------------------


def month_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) split at UTC month boundaries."""
    a, b = start.tz_convert("UTC"), end.tz_convert("UTC")
    months = pd.date_range(a.normalize().replace(day=1), b, freq="MS")
    edges = sorted({a, *[m for m in months if a < m < b], b})
    return list(zip(edges[:-1], edges[1:]))


def quarter_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) split at UTC calendar-quarter boundaries."""
    a, b = start.tz_convert("UTC"), end.tz_convert("UTC")
    qs = pd.date_range(a.normalize().replace(day=1), b, freq="QS")
    edges = sorted({a, *[q for q in qs if a < q < b], b})
    return list(zip(edges[:-1], edges[1:]))


def windows(start, end, chunk: str = "month"):
    return quarter_windows(start, end) if chunk == "quarter" else month_windows(start, end)


def cutoff_times(interval_start_utc: pd.Series, cutoff: str, tz: str) -> pd.Series:
    """For each interval, the cutoff instant: ``cutoff`` market time on (operating day - 1)."""
    op_day = pd.to_datetime(interval_start_utc, utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    local = op_day - pd.Timedelta(days=1) + pd.to_timedelta(cutoff)
    return local.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")


def vintage_check(df: pd.DataFrame, cutoff: str, tz: str) -> dict:
    """Every selected forecast row must have been published by its cutoff. Also records
    how far before the cutoff it was published (lag) and the forecast horizon."""
    t = pd.to_datetime(df[UTC], utc=True)
    pub = pd.to_datetime(df["publish_time_utc"], utc=True)
    cut = cutoff_times(t, cutoff, tz)
    lag_h = (cut - pub) / pd.Timedelta(hours=1)
    horizon_h = (t - pub) / pd.Timedelta(hours=1)
    return {
        "late_rows": int((pub > cut).sum()),
        "lag_h_min": float(lag_h.min()), "lag_h_median": float(lag_h.median()), "lag_h_max": float(lag_h.max()),
        "stale_rows_lag_gt_2h": int((lag_h > 2).sum()), "stale_rows_lag_gt_26h": int((lag_h > 26).sum()),
        "horizon_h_min": float(horizon_h.min()), "horizon_h_max": float(horizon_h.max()),
        "publish_minutes": sorted({f"{m:02d}" for m in pub.dt.minute.unique()}),
    }


def validate_chunk(df: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp, freq: pd.Timedelta,
                   locations: list[str] | None = None, location_column: str = "location",
                   cutoff: str | None = None, tz: str = "America/Chicago") -> dict:
    """Check one response against what [a, b) at ``freq`` must contain.

    ``hard`` issues mean the response itself is wrong: no rows, a row count
    that is a whole number of pages (the signature of a truncated pull),
    locations that were not requested, timestamps outside the window or off
    the interval grid, or one interval published with two different values.
    ``missing`` lists absent intervals per series. Those may be genuine gaps
    at the source, so they are reported (and audited downstream), not fatal.
    """
    expected = int((b - a) / freq)
    s: dict = {"rows": int(len(df)), "expected_per_series": expected, "hard": [], "missing": {},
               "exact_duplicates": 0}
    if df.empty:
        s["hard"].append("no rows")
        return s
    if len(df) % PAGE_ROW_LIMIT == 0:
        s["hard"].append(f"{len(df):,} rows is a whole number of {PAGE_ROW_LIMIT:,}-row pages: "
                         "possible truncation")
    t = pd.to_datetime(df[UTC], utc=True)
    if ((t < a) | (t >= b)).any():
        s["hard"].append(f"{int(((t < a) | (t >= b)).sum())} rows outside [{a}, {b})")
    off = ((t - a) % freq) != pd.Timedelta(0)
    if off.any():
        s["hard"].append(f"{int(off.sum())} timestamps off the {freq} grid, e.g. {t[off].iloc[0]}")

    d = df.assign(**{UTC: t})
    d = d.loc[:, [c for c in d.columns if d[c].map(type).ne(list).all()]].drop_duplicates()
    s["exact_duplicates"] = int(len(df) - len(d))
    keys = [UTC, location_column] if locations else [UTC]
    conflicts = int(d.duplicated(keys).sum())
    if conflicts:
        s["hard"].append(f"{conflicts} intervals published more than once with different values")
    if cutoff is not None:
        if "publish_time_utc" not in df.columns:
            s["hard"].append("forecast vintage without publish_time_utc")
        else:
            v = vintage_check(df, cutoff, tz)
            s["vintage"] = v
            if v["late_rows"]:
                s["hard"].append(f"{v['late_rows']} rows published after the {cutoff} D-1 cutoff")

    if locations:
        extra = sorted(set(d[location_column]) - set(locations))
        if extra:
            s["hard"].append(f"unrequested locations {extra[:10]}")
        groups = {loc: d.loc[d[location_column] == loc, UTC] for loc in locations}
    else:
        groups = {"all": d[UTC]}
    grid = pd.date_range(a, b, freq=freq, inclusive="left")
    for name, ts in groups.items():
        miss = grid.difference(pd.DatetimeIndex(ts.unique()))
        if len(miss):
            s["missing"][name] = {"n": int(len(miss)), "first": str(miss[0]), "last": str(miss[-1]),
                                  "reaches_window_end": bool(miss[-1] == grid[-1])}
    return s


def check_datasets(c, ids: list[str]) -> dict[str, bool]:
    """Which configured dataset ids exist; prints ERCOT candidates for any that do not."""
    found = {d["id"] for d in _paced(c.list_datasets, filter_term="ercot", return_list=True)}
    status = {i: i in found for i in ids}
    if not all(status.values()):
        print("Configured ids not found:", [i for i, ok in status.items() if not ok])
        print("ERCOT datasets available:")
        for i in sorted(found):
            print("  ", i)
    return status


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _fetch(c, dataset: str, start, end, **kw) -> pd.DataFrame:
    # sleep_time paces the client's own page loop; a chunk here is far below one page,
    # but a multi-page response must not burst past 1 request/s either.
    return c.get_dataset(dataset=dataset, start=start.isoformat(), end=end.isoformat(),
                         verbose=False, return_format="pandas", sleep_time=1, **kw)


def raw_dir(study: Study) -> Path:
    return study.raw_dir / "gridstatus"


def price_points(study: Study) -> list[str]:
    return sorted({z.price_point for z in study.zones.values()})


def _chunk_file(out_dir: Path, a: pd.Timestamp, b: pd.Timestamp) -> Path:
    return out_dir / f"{a:%Y-%m-%d}_{b:%Y-%m-%d}.parquet"


def _cached_ok(f: Path, a, b, freq, locations, location_column, cutoff=None, tz="America/Chicago") -> dict | None:
    """Validation summary of a cached chunk, or None if it is absent or must be re-fetched."""
    if not f.exists():
        return None
    s = validate_chunk(pd.read_parquet(f), a, b, freq, locations, location_column, cutoff, tz)
    if s["hard"]:
        print(f"{f.parent.name}/{f.name}: cached chunk fails validation {s['hard']}; re-fetching")
        return None
    return s


def fetch_cached(c, dataset: str, study: Study, freq: pd.Timedelta, locations: list[str] | None = None,
                 force: bool = False, location_column: str = "location", on_first_fetch=None,
                 chunk: str = "month", variant: str = "", cutoff: str | None = None,
                 **kw) -> tuple[pd.DataFrame, dict]:
    """Every chunk of ``dataset`` for the study window, validated; -> (frame, chunk summaries)."""
    out_dir = raw_dir(study) / dataset / variant if variant else raw_dir(study) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tz = study.market_tz
    parts, chunks = [], {}
    for a, b in windows(study.start_utc, study.end_utc, chunk):
        f = _chunk_file(out_dir, a, b)
        cached = None if force else _cached_ok(f, a, b, freq, locations, location_column, cutoff, tz)
        if cached is not None:
            parts.append(pd.read_parquet(f))
            chunks[f.name] = {**cached, "status": "cached"}
            continue
        df = _paced(_fetch, c, dataset, a, b, **kw)
        s = validate_chunk(df, a, b, freq, locations, location_column, cutoff, tz)
        s.update(status="fetched", retrieved_utc=manifest.utc_now(),
                 query_start_utc=str(a), query_end_utc=str(b))
        if s["hard"]:
            bad = f.with_suffix(".rejected.parquet")
            df.to_parquet(bad, index=False)
            raise ChunkError(f"{dataset} {a}..{b}: {s['hard']} (response kept at {bad})")
        df.to_parquet(f, index=False)
        n_miss = sum(m["n"] for m in s["missing"].values())
        print(f"{dataset}: {a:%Y-%m-%d %H:%M}..{b:%Y-%m-%d %H:%M} UTC  {len(df):,} rows"
              + (f"  ({n_miss} intervals MISSING: {s['missing']})" if n_miss else "  complete"))
        if on_first_fetch is not None:
            on_first_fetch(len(df))
            on_first_fetch = None
        parts.append(df)
        chunks[f.name] = s
    return pd.concat(parts, ignore_index=True), chunks


def normalize_prices(df: pd.DataFrame, location_column: str = "location") -> pd.DataFrame:
    """-> (interval_start_utc, location, price), deduplicated."""
    price = next((c for c in PRICE_COLUMNS if c in df.columns), None)
    if price is None:
        raise KeyError(f"no price column among {PRICE_COLUMNS} in {list(df.columns)}")
    out = df[[UTC, location_column, price]].rename(columns={location_column: "location", price: "price"})
    out[UTC] = pd.to_datetime(out[UTC], utc=True)
    dup = out.duplicated([UTC, "location"], keep=False)
    if dup.any():
        # ERCOT occasionally republishes an interval; identical repeats are
        # harmless, conflicting ones are not.
        conflicts = out[dup].groupby([UTC, "location"])["price"].nunique().gt(1).sum()
        if conflicts:
            raise ValueError(f"{conflicts} intervals published with conflicting prices")
        out = out.drop_duplicates([UTC, "location"])
    return out.sort_values([UTC, "location"]).reset_index(drop=True)


def normalize_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[UTC] = pd.to_datetime(out[UTC], utc=True)
    return out.drop_duplicates(UTC).sort_values(UTC).reset_index(drop=True)


def _estimate(study: Study, spec: Spec, dataset: str, locations, location_column, force) -> tuple[int, int]:
    """Rows and requests still to be fetched for one dataset (cached valid chunks cost nothing)."""
    n_series = len(locations) if locations else 1
    rows = requests = 0
    d = raw_dir(study) / dataset / spec.variant if spec.variant else raw_dir(study) / dataset
    for a, b in windows(study.start_utc, study.end_utc, spec.chunk):
        f = _chunk_file(d, a, b)
        if not force and _cached_ok(f, a, b, spec.freq, locations, location_column, spec.cutoff,
                                    study.market_tz) is not None:
            continue
        n = int((b - a) / spec.freq) * n_series
        rows += n
        requests += max(1, math.ceil(n / PAGE_ROW_LIMIT))
    return rows, requests


def fetch_dataset(c, study: Study, key: str, force: bool = False) -> Path:
    """Fetch, validate, normalise and document one configured dataset for the study window."""
    spec = all_specs(study)[key]
    cfg = study.section("gridstatus")
    dataset, loc_col = cfg[spec.config_key], cfg.get("location_column", "location")
    points = price_points(study) if spec.by_location else None
    kw = dict(spec.extra)
    if points:
        kw.update(filter_column=loc_col, filter_value=points, filter_operator="in")

    est_rows, est_req = _estimate(study, spec, dataset, points, loc_col, force)
    u0 = usage(c)
    print(f"{dataset}: ~{est_rows:,} rows / {est_req} requests to fetch; "
          f"usage {u0['rows']:,}/{u0['rows_limit']:,} rows, {u0['requests']}/{u0['requests_limit']} requests")
    check_budget(u0, est_rows, est_req)

    def first_chunk_charge(returned: int) -> None:
        charged = usage(c)["rows"] - u0["rows"]
        print(f"  first chunk: {returned:,} rows returned, {charged:,} rows charged"
              + ("  (usage counter may lag)" if charged == 0 else ""))
        check_charge(charged, returned)

    df, chunks = fetch_cached(c, dataset, study, spec.freq, points, force, loc_col,
                              on_first_fetch=first_chunk_charge, chunk=spec.chunk, variant=spec.variant,
                              cutoff=spec.cutoff, **kw)
    u1 = usage(c)
    norm = normalize_prices(df, loc_col) if spec.by_location else normalize_wide(df)
    out = raw_dir(study) / f"{spec.name}.parquet"
    norm.to_parquet(out, index=False)
    manifest.record(
        study, f"gridstatus/{dataset}" + (f"/{spec.variant}" if spec.variant else ""),
        source="gridstatus.io API, GridStatusClient.get_dataset", dataset_id=dataset,
        client=f"gridstatusio {version('gridstatusio')}",
        query={"start_utc": str(study.start_utc), "end_utc": str(study.end_utc),
               "chunking": f"UTC calendar {spec.chunk}s, end exclusive", **kw},
        vintage_cutoff=(f"{spec.cutoff} market time on D-1" if spec.cutoff else None),
        units=spec.units, frequency=str(spec.freq),
        time="interval_start_utc (UTC, interval beginning); local time America/Chicago is derived downstream",
        geography=points or ("ERCOT load forecast zones (one column each)" if key == "load" else "ERCOT system"),
        raw_chunks_dir=f"gridstatus/{dataset}/", output=f"gridstatus/{out.name}",
        rows_raw=int(len(df)), rows_normalised=int(len(norm)),
        missing_intervals=int(sum(m["n"] for s in chunks.values() for m in s["missing"].values())),
        chunks=chunks, usage_before=u0, usage_after=u1,
        rows_charged_this_run=u1["rows"] - u0["rows"], requests_this_run=u1["requests"] - u0["requests"],
    )
    return out


def fetch_all(study: Study, force: bool = False, include_fuel_mix: bool = True,
              keys: list[str] | None = None) -> dict[str, Path]:
    cfg = study.section("gridstatus")
    keys = keys or ["da", "rt", "load"] + (["fuel_mix"] if include_fuel_mix else [])
    specs = all_specs(study)
    c = client()
    status = check_datasets(c, sorted({cfg[specs[k].config_key] for k in keys}))
    if not all(status.values()):
        raise SystemExit("fix the dataset ids in config/study.toml, then re-run")
    return {k: fetch_dataset(c, study, k, force) for k in keys}


# --------------------------------------------------------------------------
# DART cross-check: gridstatus's own DA-RT spread, to confirm sign and alignment
# --------------------------------------------------------------------------


def dart_file(study: Study, start: str, end: str) -> Path:
    ds = study.section("gridstatus").get("dart_dataset", "ercot_spp_dart_15_min")
    return raw_dir(study) / "crosscheck" / f"{ds}_{start}_{end}.parquet"


def fetch_dart_crosscheck(study: Study, start: str = "2025-07-07", end: str = "2025-07-14",
                          force: bool = False) -> Path:
    """One week of ``ercot_spp_dart_15_min`` for the configured points (one request)."""
    cfg = study.section("gridstatus")
    dataset, loc_col = cfg.get("dart_dataset", "ercot_spp_dart_15_min"), cfg.get("location_column", "location")
    out = dart_file(study, start, end)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    a = pd.Timestamp(start, tz=study.market_tz).tz_convert("UTC")
    b = pd.Timestamp(end, tz=study.market_tz).tz_convert("UTC")
    points = price_points(study)
    c = client()
    u0 = usage(c)
    check_budget(u0, int((b - a) / QUARTER) * len(points), 1)
    df = _paced(_fetch, c, dataset, a, b, filter_column=loc_col, filter_value=points, filter_operator="in")
    s = validate_chunk(df, a, b, QUARTER, points, loc_col)
    if s["hard"]:
        raise ChunkError(f"{dataset} {a}..{b}: {s['hard']}")
    df.to_parquet(out, index=False)
    u1 = usage(c)
    manifest.record(
        study, f"gridstatus/{dataset}",
        source="gridstatus.io API, GridStatusClient.get_dataset", dataset_id=dataset,
        purpose="cross-check of the DA-RT spread sign and alignment only; not used in the panel",
        client=f"gridstatusio {version('gridstatusio')}",
        query={"start_utc": str(a), "end_utc": str(b), "filter_column": loc_col, "filter_value": points},
        units="spp in $/MWh (gridstatus DART spread)", frequency=str(QUARTER), geography=points,
        output=str(out.relative_to(study.raw_dir)), rows_raw=int(len(df)), chunk=s,
        usage_before=u0, usage_after=u1, retrieved_utc=manifest.utc_now(),
    )
    print(f"{dataset}: {start}..{end} {len(df):,} rows -> {out}")
    return out
