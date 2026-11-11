"""Acquisition safeguards: chunk validation, quota guard, manifest, cross-check parsers, audit.

All offline. The gridstatus client is replaced by a fake that serves
synthetic frames and keeps a usage counter, so the caching, re-validation
and budget logic run exactly as they do against the live API.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import config, pipeline, synthetic, validate
from ercot_rsm.data import ercot_official, gas, gridstatus_fetch as gs, manifest, weather

UTC = "interval_start_utc"
H, Q = pd.Timedelta(hours=1), pd.Timedelta(minutes=15)
LOCS = ["LZ_HOUSTON", "LZ_NORTH"]


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def frame(a, b, freq, locs=LOCS):
    t = pd.date_range(a, b, freq=freq, inclusive="left")
    return pd.concat([pd.DataFrame({UTC: t, "location": loc, "spp": np.arange(len(t), dtype=float)})
                      for loc in locs], ignore_index=True)


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    monkeypatch.setattr(gs, "MIN_SECONDS_BETWEEN_CALLS", 0.0)


# --------------------------------------------------------------------------
# validate_chunk
# --------------------------------------------------------------------------


def test_complete_chunk_passes():
    a, b = ts("2025-03-01"), ts("2025-04-01")
    s = gs.validate_chunk(frame(a, b, Q), a, b, Q, LOCS)
    assert s["hard"] == [] and s["missing"] == {}
    assert s["expected_per_series"] == 31 * 96 and s["rows"] == 2 * 31 * 96


def test_truncated_chunk_reports_missing_tail():
    a, b = ts("2025-03-01"), ts("2025-04-01")
    df = frame(a, b, Q)
    df = df[~((df["location"] == "LZ_NORTH") & (df[UTC] >= ts("2025-03-31 20:00")))]
    s = gs.validate_chunk(df, a, b, Q, LOCS)
    assert s["hard"] == []
    assert s["missing"]["LZ_NORTH"]["n"] == 16 and s["missing"]["LZ_NORTH"]["reaches_window_end"]
    assert "LZ_HOUSTON" not in s["missing"]


def test_page_sized_response_is_suspect():
    a = ts("2025-01-01")
    df = frame(a, a + 25_000 * H, H)  # 2 locations x 25,000 = exactly one 50k page
    assert any("truncation" in h for h in gs.validate_chunk(df, a, a + 25_000 * H, H, LOCS)["hard"])


@pytest.mark.parametrize("mutate, message", [
    (lambda d: pd.concat([d, d.iloc[:1].assign(location="HB_HUBAVG")]), "unrequested"),
    (lambda d: pd.concat([d, d.iloc[:1].assign(spp=-1.0)]), "different values"),
    (lambda d: pd.concat([d, d.iloc[:1].assign(**{UTC: ts("2025-04-01")})]), "outside"),
    (lambda d: pd.concat([d, d.iloc[:1].assign(**{UTC: ts("2025-03-01 00:07")})]), "off the"),
    (lambda d: d.iloc[:0], "no rows"),
])
def test_wrong_rows_are_hard_failures(mutate, message):
    a, b = ts("2025-03-01"), ts("2025-04-01")
    s = gs.validate_chunk(mutate(frame(a, b, H)), a, b, H, LOCS)
    assert any(message in h for h in s["hard"]), s["hard"]


def test_identical_republication_is_not_a_failure():
    a, b = ts("2025-03-01"), ts("2025-04-01")
    d = frame(a, b, H)
    s = gs.validate_chunk(pd.concat([d, d.iloc[:3]]), a, b, H, LOCS)
    assert s["hard"] == [] and s["exact_duplicates"] == 3 and s["missing"] == {}


def test_fall_back_day_has_25_hourly_intervals():
    tz = "America/Chicago"
    a = pd.Timestamp("2025-11-02", tz=tz).tz_convert("UTC")
    b = pd.Timestamp("2025-11-03", tz=tz).tz_convert("UTC")
    s = gs.validate_chunk(frame(a, b, H), a, b, H, LOCS)
    assert s["expected_per_series"] == 25 and s["hard"] == [] and s["missing"] == {}


def test_wide_frames_validate_without_locations():
    a, b = ts("2025-03-01"), ts("2025-03-02")
    df = pd.DataFrame({UTC: pd.date_range(a, b, freq="h", inclusive="left"), "north": 1.0})
    assert gs.validate_chunk(df.iloc[1:], a, b, H)["missing"]["all"]["n"] == 1


# --------------------------------------------------------------------------
# Quota
# --------------------------------------------------------------------------

USAGE = {"requests": 14, "rows": 10_119, "requests_limit": 250, "rows_limit": 500_000}


def test_budget_guard():
    gs.check_budget(USAGE, rows=200_000, requests=60)
    with pytest.raises(gs.BudgetError, match="reserve"):
        gs.check_budget(USAGE, rows=450_000, requests=60)
    with pytest.raises(gs.BudgetError):
        gs.check_budget(USAGE, rows=1_000, requests=230)


def test_charge_check_stops_on_overbilling():
    gs.check_charge(charged=744, returned=744)
    gs.check_charge(charged=0, returned=744)  # counter lag is not an error
    with pytest.raises(gs.BudgetError, match="charged"):
        gs.check_charge(charged=8_928, returned=744)  # billed at 5-minute resolution


# --------------------------------------------------------------------------
# Fetching against a fake client
# --------------------------------------------------------------------------


class FakeClient:
    def __init__(self, drop=None, charge_factor=1):
        self.calls, self.rows, self.drop, self.charge_factor = [], 10_119, drop, charge_factor

    def get_api_usage(self):
        return {"limits": {"api_requests_limit": 250, "api_rows_returned_limit": 500_000},
                "current_period_usage": {"total_requests": 14 + len(self.calls),
                                         "total_api_rows_returned": self.rows}}

    def list_datasets(self, filter_term, return_list):
        return [{"id": i} for i in ("ercot_spp_day_ahead_hourly", "ercot_spp_real_time_15_min",
                                    "ercot_load_by_forecast_zone", "ercot_fuel_mix")]

    def get_dataset(self, dataset, start, end, filter_value=None, **kw):
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((dataset, a, kw.get("sleep_time")))
        if dataset == "ercot_load_by_forecast_zone":
            t = pd.date_range(a, b, freq="h", inclusive="left")
            df = pd.DataFrame({UTC: t, "north": 1.0, "south": 1.0, "west": 1.0, "houston": 1.0, "total": 4.0})
        else:
            freq = Q if "15_min" in dataset else H
            df = frame(a, b, freq, filter_value)
            if self.drop:
                df = self.drop(df)
        self.rows += len(df) * self.charge_factor
        return df


@pytest.fixture
def month_study(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return config.load(start="2025-02-01", end="2025-04-01")  # 3 UTC chunks


def test_fetch_dataset_caches_validates_and_documents(month_study):
    c = FakeClient()
    out = gs.fetch_dataset(c, month_study, "da")
    da = pd.read_parquet(out)
    assert len(c.calls) == 3 and all(s == 1 for *_, s in c.calls)  # client pages paced too
    assert len(da) == 4 * int((month_study.end_utc - month_study.start_utc) / H)
    entry = manifest.load(month_study)["gridstatus/ercot_spp_day_ahead_hourly"]
    assert entry["missing_intervals"] == 0 and entry["rows_charged_this_run"] == entry["rows_raw"]
    assert entry["query"]["filter_value"] == ["LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST"]
    assert all(ch["status"] == "fetched" for ch in entry["chunks"].values())
    first = sorted(entry["chunks"].values(), key=lambda ch: ch["query_start_utc"])[0]["retrieved_utc"]

    gs.fetch_dataset(c, month_study, "da")  # everything cached: no new requests
    assert len(c.calls) == 3
    entry = manifest.load(month_study)["gridstatus/ercot_spp_day_ahead_hourly"]
    assert all(ch["status"] == "cached" for ch in entry["chunks"].values())
    kept = sorted(entry["chunks"].values(), key=lambda ch: ch["query_start_utc"])[0]["retrieved_utc"]
    assert kept == first  # the merge keeps the original retrieval time


def test_corrupt_cached_chunk_is_refetched_alone(month_study):
    c = FakeClient()
    gs.fetch_dataset(c, month_study, "da")
    f = sorted((gs.raw_dir(month_study) / "ercot_spp_day_ahead_hourly").glob("*.parquet"))[1]
    df = pd.read_parquet(f)
    pd.concat([df, df.iloc[:1].assign(location="HB_HUBAVG")]).to_parquet(f)
    gs.fetch_dataset(c, month_study, "da")
    assert len(c.calls) == 4 and c.calls[-1][1] == pd.Timestamp(f.stem.split("_")[0], tz="UTC")


def test_bad_response_is_rejected_not_cached(month_study):
    c = FakeClient(drop=lambda d: pd.concat([d, d.iloc[:1].assign(location="HB_BUSAVG")]))
    with pytest.raises(gs.ChunkError, match="unrequested"):
        gs.fetch_dataset(c, month_study, "da")
    d = gs.raw_dir(month_study) / "ercot_spp_day_ahead_hourly"
    assert list(d.glob("*.rejected.parquet")) and not [p for p in d.glob("*.parquet") if "rejected" not in p.name]


def test_overbilling_stops_after_the_first_chunk(month_study):
    c = FakeClient(charge_factor=12)
    with pytest.raises(gs.BudgetError, match="charged"):
        gs.fetch_dataset(c, month_study, "da")
    assert len(c.calls) == 1


def test_source_gaps_are_kept_and_counted(month_study):
    c = FakeClient(drop=lambda d: d.iloc[5:])
    gs.fetch_dataset(c, month_study, "rt")
    entry = manifest.load(month_study)["gridstatus/ercot_spp_real_time_15_min"]
    assert entry["missing_intervals"] == 3 * 5


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def test_manifest_merges_nested_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = config.load(start="2025-02-01", end="2025-02-08")
    manifest.record(s, "x", chunks={"a": {"status": "fetched", "retrieved_utc": "t0"}}, rows=1)
    manifest.record(s, "x", chunks={"a": {"status": "cached"}, "b": {"status": "fetched"}}, rows=2)
    e = manifest.load(s)["x"]
    assert e["rows"] == 2 and e["chunks"]["a"] == {"status": "cached", "retrieved_utc": "t0"}
    assert set(e["chunks"]) == {"a", "b"} and "recorded_utc" in e


# --------------------------------------------------------------------------
# Cross-check parsers
# --------------------------------------------------------------------------


def test_ercot_native_load_dst_rows():
    he = (["03/09/2025 01:00", "03/09/2025 02:00", "03/09/2025 04:00"]
          + ["11/02/2025 01:00", "11/02/2025 02:00", "11/02/2025 02:00 DST", "11/02/2025 03:00",
             "12/31/2025 24:00"])
    df = pd.DataFrame({"Hour Ending": he, "COAST": 1.0, "ERCOT": np.arange(len(he), dtype=float)})
    out = ercot_official.parse_native_load(df)
    assert out[UTC].tolist() == [ts(x) for x in (
        "2025-03-09 06:00", "2025-03-09 07:00", "2025-03-09 08:00",  # 00:00 CST, 01:00 CST, 03:00 CDT
        "2025-11-02 05:00", "2025-11-02 06:00", "2025-11-02 07:00", "2025-11-02 08:00",  # CDT, CDT, CST, CST
        "2026-01-01 05:00")]
    with pytest.raises(ValueError, match="Hour Ending"):
        ercot_official.parse_native_load(pd.DataFrame({"Hour Ending": ["2025-01-01T01"], "ERCOT": [1.0]}))


def test_find_ercot_archive_link():
    html = '<a href="https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip">2025</a>' \
           '<a href="/files/docs/2024/02/06/Native_Load_2024.zip">2024</a>'
    assert ercot_official.find_archive_url(html, 2025).endswith("2025/02/11/Native_Load_2025.zip")
    assert ercot_official.find_archive_url(html, 2024).startswith("https://www.ercot.com/files/")
    with pytest.raises(ValueError):
        ercot_official.find_archive_url(html, 2023)


def test_compare_load_totals():
    t = pd.date_range(ts("2025-01-01 06:00"), periods=48, freq="h")
    g = pd.DataFrame({UTC: t, "total": 40_000 + 100 * np.arange(48.0)})
    n = pd.DataFrame({UTC: t[:-1], "ERCOT": g["total"].iloc[:-1] * 1.01})
    r = ercot_official.compare_totals(g, n)
    assert r["hours_compared"] == 47 and r["corr"] > 0.999 and r["mape_pct"] == pytest.approx(100 / 101, rel=1e-6)


def test_fred_parse_and_compare():
    fred = gas.parse_fred_csv("observation_date,DHHNGSP\n2025-12-24,3.31\n2025-12-25,\n2025-12-26,.\n"
                              "2025-12-29,4.35\n")
    assert fred["trade_date"].tolist() == [date(2025, 12, 24), date(2025, 12, 29)]
    eia = pd.DataFrame({"trade_date": [date(2025, 12, 24), date(2025, 12, 29)], "price": [3.31, 4.30]})
    r = gas.compare_series(eia, fred)
    assert r["dates_both"] == 2 and len(r["mismatches"]) == 1 and r["max_abs_diff"] == pytest.approx(0.05)


def test_ncei_parser_and_station_coverage():
    text = ('"STATION","DATE","SOURCE","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","TMP"\n'
            '"72243012960","2025-07-01T00:53:00","7","FM-15","KIAH ","V020","+0283,5"\n'
            '"72243012960","2025-07-01T01:53:00","7","FM-15","KIAH ","V020","+9999,9"\n'
            '"72243012960","2025-07-01T02:53:00","7","FM-15","KIAH ","V020","+0267,3"\n')
    n = weather.parse_ncei_tmp(text)
    assert len(n) == 1 and n["tmpf"].iloc[0] == pytest.approx(28.3 * 9 / 5 + 32)
    tz = "America/Chicago"
    a, b = pd.Timestamp("2025-07-01", tz=tz), pd.Timestamp("2025-07-02", tz=tz)
    hrs = pd.date_range(a.tz_convert("UTC"), b.tz_convert("UTC"), freq="h", inclusive="left")
    obs = pd.DataFrame({"station": "IAH", "valid_utc": hrs[[0, 1, 2, 10, 11]] - pd.Timedelta(minutes=7),
                        "tmpf": 80.0})
    cov = weather.station_coverage(obs, ["IAH", "HOU"], a, b)
    assert cov.loc["IAH", "hours_with_obs"] == 5 and cov.loc["IAH", "longest_gap_hours"] == 12
    assert cov.loc["HOU", "hours_with_obs"] == 0 and cov.loc["HOU", "longest_gap_hours"] == 24


def test_dart_check_identifies_the_sign():
    t = pd.date_range(ts("2025-07-07 05:00"), periods=8, freq="15min")
    rt = pd.DataFrame({UTC: t, "location": "LZ_HOUSTON", "price": np.linspace(20, 60, 8)})
    da = pd.DataFrame({UTC: t[::4], "location": "LZ_HOUSTON", "price": [30.0, 50.0]})
    da_rep = np.repeat([30.0, 50.0], 4)
    dart = pd.DataFrame({UTC: t, "location": "LZ_HOUSTON", "spp": da_rep - rt["price"].to_numpy()})
    r = validate.dart_check(da, rt, dart).iloc[0]
    assert r["intervals"] == 8 and r["max_abs_diff_vs_da_minus_rt"] < 1e-12
    assert r["max_abs_diff_vs_rt_minus_da"] > 1


# --------------------------------------------------------------------------
# Panel and audit
# --------------------------------------------------------------------------


@pytest.fixture
def week(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = config.load(start="2025-11-01", end="2025-11-08")  # contains the fall-back day
    return s, synthetic.make(s)


def test_missing_hours_stay_in_the_panel(week):
    s, raw = week
    gone = ts("2025-11-03 12:00")
    raw.da = raw.da[~((raw.da[UTC] == gone) & (raw.da["location"] == "LZ_WEST"))]
    raw.rt = raw.rt[~((raw.rt[UTC].dt.floor("h") == gone) & (raw.rt["location"] == "LZ_WEST"))]
    raw.rt = raw.rt[~((raw.rt[UTC] == ts("2025-11-04 12:15")) & (raw.rt["location"] == "LZ_NORTH"))]
    p = pipeline.build_panel(raw, s)
    assert (p.groupby("zone").size() == 7 * 24 + 1).all()
    row = p[(p["zone"] == "west") & (p[UTC] == gone)].iloc[0]
    assert np.isnan(row["spread"]) and not row["model_ok"] and row["rt_n"] == 0
    part = p[(p["zone"] == "north") & (p[UTC] == ts("2025-11-04 12:00"))].iloc[0]
    assert part["rt_n"] == 3 and not part["rt_complete"] and not part["model_ok"]
    assert p["model_ok"].sum() >= len(p) - 10
    assert np.allclose(p["imhr_spread"], p["spread"] / p["gas_price"], equal_nan=True)


def test_audit_writes_summary(week):
    s, raw = week
    panel = pipeline.build_panel(raw, s)
    out = s.processed_dir / "audit"
    summary = validate.run(s, raw, panel, out).set_index("check")
    assert summary.loc["DST days", "status"] == "PASS" and "2025-11-02 (25 h)" in summary.loc["DST days", "detail"]
    assert summary.loc["coverage: da_price", "status"] == "PASS"
    assert summary.loc["gas vs FRED DHHNGSP", "status"] == "SKIPPED"
    assert (out / "audit.md").exists() and (out / "series_coverage.csv").exists()


# --------------------------------------------------------------------------
# ERCOT settled price archive
# --------------------------------------------------------------------------


def test_archive_da_and_rt_dst_and_point_types():
    from ercot_rsm.data import ercot_archive as ea

    da = {"Nov": pd.DataFrame({
        "Delivery Date": ["11/02/2025"] * 4 + ["03/09/2025"] * 2,
        "Hour Ending": ["01:00", "02:00", "02:00", "03:00", "02:00", "03:00"],
        "Repeated Hour Flag": ["N", "N", "Y", "N", "N", "N"],
        "Settlement Point": "LZ_WEST", "Settlement Point Price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})}
    with pytest.raises(Exception):  # 03/09 HE03 would be 02:00 local, which does not exist
        ea.parse_da(da)
    da["Nov"] = da["Nov"].iloc[:4]
    d = ea.parse_da(da)
    assert d[UTC].tolist() == [ts("2025-11-02 05:00"), ts("2025-11-02 06:00"), ts("2025-11-02 07:00"),
                               ts("2025-11-02 08:00")]
    rt = {"Nov": pd.DataFrame({
        "Delivery Date": "11/02/2025", "Delivery Hour": [2, 2, 2], "Delivery Interval": [4, 4, 1],
        "Repeated Hour Flag": ["N", "Y", "Y"], "Settlement Point Name": "LZ_WEST",
        "Settlement Point Type": ["LZ", "LZ", "LZEW"], "Settlement Point Price": [1.0, 2.0, 3.0]})}
    r = ea.parse_rt(rt)
    assert set(r[UTC]) == {ts("2025-11-02 06:45"), ts("2025-11-02 07:45"), ts("2025-11-02 07:00")}
    p = ea.to_prices(r, ["LZ_WEST"], "LZ")
    assert len(p) == 2 and set(p["price"]) == {1.0, 2.0}


def test_archive_comparison_counts_differences():
    from ercot_rsm.data import ercot_archive as ea

    t = pd.date_range(ts("2025-06-30 05:00"), periods=4, freq="15min")
    arch = pd.DataFrame({UTC: t, "location": "LZ_SOUTH", "point_type": "LZ", "price": [28.17, 30.0, 31.0, 32.0]})
    other = pd.DataFrame({UTC: t, "location": "LZ_SOUTH", "price": [1623.90, 30.0, 31.0, 32.0]})
    c = ea.compare_with(arch, other, "LZ")
    assert c["rows_both"] == 4 and c["rows_differ"] == 1 and c["max_abs_diff"] == pytest.approx(1595.73)
