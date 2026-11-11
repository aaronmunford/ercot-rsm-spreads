"""Data audit: is every series complete, aligned and plausible, and do outside sources agree?

``run(study, raw, panel)`` writes ``data/processed/<window>/audit/``: one CSV per
check plus ``audit.md``, a PASS/WARN/FAIL summary to read before any modelling.
Checks that need a cross-check file that was never fetched are reported as
SKIPPED, never silently passed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Study
from .data import ercot_official, gridstatus_fetch as gs, manifest, weather
from .pipeline import RawInputs

UTC = "interval_start_utc"
HOUR, QUARTER = pd.Timedelta(hours=1), pd.Timedelta(minutes=15)
RT_SPIKE = 1_000.0   # $/MWh
OFFER_CAP = 5_000.0  # ERCOT high system-wide offer cap in 2025, $/MWh
TEMP_RANGE_F = (-20.0, 125.0)


def _grid(study: Study, freq: pd.Timedelta) -> pd.DatetimeIndex:
    return pd.date_range(study.start_utc, study.end_utc, freq=freq, inclusive="left")


def _coverage_row(series: str, ident: str, ts: pd.Series, grid: pd.DatetimeIndex) -> dict:
    ts = pd.to_datetime(ts, utc=True)
    present = pd.DatetimeIndex(ts.unique())
    miss = grid.difference(present)
    return {
        "series": series, "id": ident, "expected": len(grid), "present": int(grid.isin(present).sum()),
        "missing": len(miss), "duplicates": int(ts.duplicated().sum()),
        "outside_window": int((~present.isin(grid)).sum()),
        "first_missing": str(miss[0]) if len(miss) else "", "last_missing": str(miss[-1]) if len(miss) else "",
    }


def series_coverage(study: Study, raw: RawInputs, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hours, quarters = _grid(study, HOUR), _grid(study, QUARTER)
    for name, df, grid in (("da_price", raw.da, hours), ("rt_price_15min", raw.rt, quarters)):
        for loc in gs.price_points(study):
            rows.append(_coverage_row(name, loc, df.loc[df["location"] == loc, UTC], grid))
    for z in study.zones.values():
        col = next(c for c in raw.load_wide.columns if c.lower() == z.load_column.lower())
        rows.append(_coverage_row("load_mw", z.load_column, raw.load_wide.loc[raw.load_wide[col].notna(), UTC], hours))
    if raw.fuel_mix is not None:
        rows.append(_coverage_row("fuel_mix", "system", raw.fuel_mix[UTC], hours))
    for zone, d in panel.groupby("zone"):
        rows.append(_coverage_row("temp_f", zone, d.loc[d["temp_f"].notna(), UTC], hours))
    return pd.DataFrame(rows)


def dst_days(panel: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Rows per zone on each operating day whose length is not 24 h, against that length."""
    days = pd.Series(sorted(set(panel["operating_date"])))
    naive = pd.DatetimeIndex(pd.to_datetime(days))
    length = np.asarray(((naive + pd.Timedelta(days=1)).tz_localize(tz) - naive.tz_localize(tz)) / HOUR)
    odd = length != 24
    rows = []
    for d, L in zip(days[odd], length[odd]):
        sub = panel[panel["operating_date"] == d]
        for zone, z in sub.groupby("zone"):
            rows.append({"operating_date": str(d), "zone": zone, "day_hours": int(L), "panel_rows": len(z),
                         "hours_with_spread": int(z["spread"].notna().sum()),
                         "hours_with_load": int(z["load_mw"].notna().sum()),
                         "hours_with_temp": int(z["temp_f"].notna().sum())})
    return pd.DataFrame(rows)


def price_stats(panel: pd.DataFrame, rt15: pd.DataFrame, point_to_zone: dict) -> pd.DataFrame:
    rows = []
    r = rt15.assign(zone=rt15["location"].map(point_to_zone))
    series = [(z, "rt_15min", d["price"]) for z, d in r.groupby("zone")]
    for z, d in panel.groupby("zone"):
        series += [(z, "da_hourly", d["da_price"]), (z, "rt_hourly", d["rt_price"]),
                   (z, "spread_da_minus_rt", d["spread"])]
    for z, name, s in series:
        s = s.dropna()
        q = s.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
        rows.append({"zone": z, "series": name, "n": len(s), "mean": s.mean(), "std": s.std(),
                     "min": s.min(), "p0.1": q[0.001], "p1": q[0.01], "median": q[0.5], "p99": q[0.99],
                     "p99.9": q[0.999], "max": s.max(), "n_negative": int((s < 0).sum()),
                     "n_ge_1000": int((s >= RT_SPIKE).sum()), "n_ge_offer_cap": int((s >= OFFER_CAP).sum())})
    return pd.DataFrame(rows).round(3)


def spikes(panel: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["interval_start_local", "zone", "da_price", "rt_price", "rt_intrahour_std", "spread",
            "load_mw", "temp_f", "gas_price"]
    top_rt = panel.nlargest(n, "rt_price")[cols].assign(rank_by="rt_price")
    top_sp = panel.loc[panel["spread"].abs().nlargest(n).index, cols].assign(rank_by="abs_spread")
    return pd.concat([top_rt, top_sp], ignore_index=True)


def exclusions(panel: pd.DataFrame) -> pd.DataFrame:
    """Why zone-hours fall out of the estimation sample (reasons overlap)."""
    rows = []
    for zone, d in panel.groupby("zone"):
        rows.append({
            "zone": zone, "panel_rows": len(d), "model_ok": int(d["model_ok"].sum()),
            "no_da": int(d["da_price"].isna().sum()), "rt_incomplete": int((~d["rt_complete"]).sum()),
            "no_load": int(d["load_mw"].isna().sum()), "no_temp": int(d["temp_f"].isna().sum()),
            "no_gas": int(d["gas_price"].isna().sum()), "imhr_undefined": int(d["imhr_da"].isna().sum()),
        })
    return pd.DataFrame(rows)


def gas_gaps(gas: pd.DataFrame, study: Study) -> pd.DataFrame:
    """Gaps between consecutive trade dates longer than a weekend plus one holiday."""
    t = pd.to_datetime(gas["trade_date"]).sort_values().reset_index(drop=True)
    gap = t.diff().dt.days
    out = pd.DataFrame({"from": t.shift(1), "to": t, "calendar_days": gap})
    return out[out["calendar_days"] > 4].assign(**{"from": lambda d: d["from"].dt.date, "to": lambda d: d["to"].dt.date})


def dart_check(da: pd.DataFrame, rt15: pd.DataFrame, dart: pd.DataFrame) -> pd.DataFrame:
    """Our DA (hourly, repeated over its four intervals) minus RT (15-min) vs gridstatus's DART ``spp``."""
    d = dart[[UTC, "location", "spp"]].assign(**{UTC: pd.to_datetime(dart[UTC], utc=True)})
    r = rt15.rename(columns={"price": "rt"})
    a = da.rename(columns={"price": "da"}).assign(hour=lambda x: x[UTC])
    m = (d.merge(r, on=[UTC, "location"], how="inner")
          .assign(hour=lambda x: x[UTC].dt.floor("h"))
          .merge(a[["hour", "location", "da"]], on=["hour", "location"], how="inner"))
    rows = []
    for loc, g in m.groupby("location"):
        da_minus_rt, rt_minus_da = g["da"] - g["rt"], g["rt"] - g["da"]
        rows.append({"location": loc, "intervals": len(g),
                     "max_abs_diff_vs_da_minus_rt": float((g["spp"] - da_minus_rt).abs().max()),
                     "max_abs_diff_vs_rt_minus_da": float((g["spp"] - rt_minus_da).abs().max()),
                     "corr_with_da_minus_rt": float(np.corrcoef(g["spp"], da_minus_rt)[0, 1])})
    return pd.DataFrame(rows)


def run(study: Study, raw: RawInputs, panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    point_to_zone = {z.price_point: n for n, z in study.zones.items()}
    checks: list[dict] = []

    def check(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    cov = series_coverage(study, raw, panel)
    cov.to_csv(out_dir / "series_coverage.csv", index=False)
    for series, g in cov.groupby("series", sort=False):
        miss, dup, out = int(g["missing"].sum()), int(g["duplicates"].sum()), int(g["outside_window"].sum())
        status = "PASS" if miss == dup == out == 0 else ("FAIL" if dup or out else "WARN")
        check(f"coverage: {series}", status,
              f"{int(g['present'].sum()):,}/{int(g['expected'].sum()):,} present across {len(g)} series; "
              f"{miss} missing, {dup} duplicates, {out} outside window")

    dst = dst_days(panel, study.market_tz)
    dst.to_csv(out_dir / "dst_days.csv", index=False)
    if dst.empty:
        check("DST days", "WARN", "no 23- or 25-hour operating day in the window")
    else:
        ok = (dst["panel_rows"] == dst["day_hours"]).all() and (dst["hours_with_spread"] == dst["day_hours"]).all()
        days = ", ".join(f"{d} ({h} h)" for d, h in dst[["operating_date", "day_hours"]].drop_duplicates().values)
        check("DST days", "PASS" if ok else "FAIL", f"{days}: rows and spreads per zone match the day length")

    ps = price_stats(panel, raw.rt, point_to_zone)
    ps.to_csv(out_dir / "price_stats.csv", index=False)
    rt15 = ps[ps["series"] == "rt_15min"]
    check("prices: negatives and spikes", "INFO",
          "; ".join(f"{r.zone}: {r.n_negative} negative 15-min RT, {r.n_ge_1000} >= $1,000, "
                    f"{r.n_ge_offer_cap} >= cap, max ${r.max:,.0f}" for r in rt15.itertuples()))
    spikes(panel).to_csv(out_dir / "spikes_top20.csv", index=False)

    ex = exclusions(panel)
    ex.to_csv(out_dir / "model_sample_exclusions.csv", index=False)
    check("estimation sample (model_ok)", "INFO",
          "; ".join(f"{r.zone}: {r.model_ok:,}/{r.panel_rows:,}" for r in ex.itertuples()))

    stations = sorted({s for z in study.zones.values() for s in z.stations})
    sc = weather.station_coverage(raw.station_obs, stations, study.start, study.end)
    sc.to_csv(out_dir / "station_coverage.csv")
    weak = sc[sc["pct_hours_with_obs"] < 90]
    lo, hi = TEMP_RANGE_F
    implausible = int(((raw.station_obs["tmpf"] < lo) | (raw.station_obs["tmpf"] > hi)).sum())
    check("weather stations", "PASS" if weak.empty and not implausible else "WARN",
          f"{len(stations)} stations; below 90% hourly coverage: "
          + (", ".join(f"{s} {p}% (longest gap {g} h)" for s, p, g in
                       weak[["pct_hours_with_obs", "longest_gap_hours"]].itertuples()) or "none")
          + f"; {implausible} readings outside {lo:.0f}..{hi:.0f} F (dropped as sensor faults before averaging)")

    gg = gas_gaps(raw.gas, study)
    gg.to_csv(out_dir / "gas_gaps.csv", index=False)
    g_in = raw.gas[pd.to_datetime(raw.gas["trade_date"]) >= study.start.tz_localize(None)]
    check("gas trade dates", "PASS" if gg.empty else "WARN",
          f"{len(g_in)} trade dates in window, ${g_in['price'].min():.2f}..${g_in['price'].max():.2f}/MMBtu; "
          f"{len(gg)} gaps > 4 calendar days" + (f": {gg[['from', 'to']].values.tolist()}" if len(gg) else ""))

    m = manifest.load(study)
    fred = m.get("gas/fred_crosscheck", {}).get("result")
    if fred is None:
        check("gas vs FRED DHHNGSP", "SKIPPED", "run `fetch --only crosscheck`")
    else:
        ok = not fred["mismatches"] and not fred["only_primary"] and not fred["only_other"]
        check("gas vs FRED DHHNGSP", "PASS" if ok else "WARN",
              f"{fred['dates_both']} common trade dates, {len(fred['mismatches'])} price mismatches, "
              f"{len(fred['only_primary'])} only in EIA, {len(fred['only_other'])} only in FRED")

    ncei = {k: v["result"] for k, v in m.items() if k.startswith("weather/ncei_crosscheck/")}
    if not ncei:
        check("IEM vs NOAA NCEI", "SKIPPED", "run `fetch --only crosscheck`")
    for k, r in ncei.items():
        check(f"IEM vs NOAA NCEI ({r['station']} {r['month']})",
              "PASS" if r["share_within_1f"] > 0.98 else "WARN",
              f"{r['matched_timestamps']} matched METARs, mean |diff| {r['mean_abs_diff_f']:.2f} F, "
              f"max {r['max_abs_diff_f']:.1f} F, {100 * r['share_within_1f']:.1f}% within 1 F")

    dart_files = sorted((gs.raw_dir(study) / "crosscheck").glob("*dart*.parquet"))
    if not dart_files:
        check("DART sign (gridstatus spread)", "SKIPPED", "run `fetch --only crosscheck`")
    for f in dart_files:
        dc = dart_check(raw.da, raw.rt, pd.read_parquet(f))
        dc.to_csv(out_dir / f"dart_check_{f.stem}.csv", index=False)
        ok = len(dc) and (dc["max_abs_diff_vs_da_minus_rt"] < 0.01).all()
        flipped = len(dc) and (dc["max_abs_diff_vs_rt_minus_da"] < 0.01).all()
        check("DART sign (gridstatus spread)", "PASS" if ok else ("WARN" if flipped else "FAIL"),
              f"{f.stem}: {int(dc['intervals'].sum()) if len(dc) else 0} intervals; max |spp - (DA - RT)| = "
              f"{dc['max_abs_diff_vs_da_minus_rt'].max():.4f}, max |spp - (RT - DA)| = "
              f"{dc['max_abs_diff_vs_rt_minus_da'].max():.4f}")

    arch = study.raw_dir / "ercot"
    gsd = gs.raw_dir(study)
    if (arch / "rt_prices_archive.parquet").exists() and (gsd / "rt_prices.parquet").exists():
        from .data import ercot_archive
        rows = []
        for kind, t in (("da", None), ("rt", study.section("prices").get("rt_point_type", "LZ"))):
            a_ = pd.read_parquet(arch / f"{kind}_prices_archive.parquet")
            g_ = pd.read_parquet(gsd / f"{kind}_prices.parquet")
            cmp = ercot_archive.compare_with(a_, g_, t)
            pa = ercot_archive.to_prices(a_, sorted(g_["location"].unique()), t)
            m = pa.merge(g_, on=[UTC, "location"], suffixes=("_ercot", "_gs"))
            bad = m[(m["price_ercot"] - m["price_gs"]).abs() > 0.005]
            days = sorted({str(x) for x in bad[UTC].dt.tz_convert(study.market_tz).dt.date})
            rows.append({"market": kind.upper(), **cmp, "days_with_differences": ";".join(days)})
        pc = pd.DataFrame(rows)
        pc.to_csv(out_dir / "price_source_crosscheck.csv", index=False)
        ok = (pc["only_ercot"] == 0).all() and (pc["only_other"] == 0).all()
        check("prices: ERCOT settled archive vs gridstatus", "PASS" if ok else "FAIL",
              "; ".join(f"{r.market}: {r.rows_both:,} matched intervals, {r.rows_differ} "
                        f"differ (max ${r.max_abs_diff:,.2f}) on {len(r.days_with_differences.split(';')) if r.days_with_differences else 0} "
                        f"corrected days" for r in pc.itertuples()) + "; panel uses the settled archive (D-31)")

    nl = study.raw_dir / "ercot" / f"native_load_{study.start.year}.parquet"
    if not nl.exists():
        check("load total vs ERCOT native load", "SKIPPED", "run `fetch --only ercot`")
    else:
        lc = ercot_official.compare_totals(raw.load_wide, pd.read_parquet(nl))
        pd.Series(lc).to_csv(out_dir / "load_total_check.csv", header=["value"])
        check("load total vs ERCOT native load", "PASS" if lc["corr"] > 0.99 and lc["mape_pct"] < 2 else "WARN",
              f"{lc['hours_compared']:,} hours; corr {lc['corr']:.5f}, MAPE {lc['mape_pct']:.3f}%, mean diff "
              f"{lc['mean_diff_mw']:+,.0f} MW, max |diff| {lc['max_abs_diff_mw']:,.0f} MW at {lc['max_abs_diff_at']}")

    summary = pd.DataFrame(checks)
    summary.to_csv(out_dir / "summary.csv", index=False)
    lines = [f"# Data audit: {study.tag}", "",
             f"Window {study.start:%Y-%m-%d} to {study.end:%Y-%m-%d} (operating days, {study.market_tz}, end "
             f"exclusive). Generated by `python -m ercot_rsm.cli audit`; the CSVs next to this file hold "
             f"the detail.", "", "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['status']} | {c['detail']} |" for c in checks]
    (out_dir / "audit.md").write_text("\n".join(lines) + "\n")
    return summary
