"""EDA tables on a synthetic panel: shapes, definitions, and the event-exclusion check."""

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import config, eda, pipeline, synthetic


@pytest.fixture(scope="module")
def panel():
    s = config.load(start="2025-01-01", end="2025-04-01")
    return pipeline.build_panel(synthetic.make(s), s)


def test_effective_n_of_white_noise_and_a_random_walk():
    rng = np.random.default_rng(0)
    e = rng.normal(size=5000)
    assert 4000 < eda.effective_n(e) < 6500
    assert eda.effective_n(np.cumsum(e)) < 200


def test_tail_flags_mark_one_percent_per_zone(panel):
    f = eda.tail_flags(panel)
    share = f.groupby(panel["zone"])["extreme"].mean()
    assert share.between(0.009, 0.011).all()  # above the 99th percentile, up to rounding of n
    assert (f["extreme"] == (f["extreme_pos"] | f["extreme_neg"])).all()


def test_spread_summary_columns(panel):
    s = eda.spread_summary(panel)
    assert list(s.index) == ["houston", "north", "south", "west"]
    assert (s["n"] == panel.groupby("zone").size()).all()
    assert s["top1pct_share_of_abs_sum"].between(0.01, 1).all()
    assert (s["p1"] <= s["median"]).all() and (s["median"] <= s["p99"]).all()


def test_temp_load_quadratic_beats_linear(panel):
    t = eda.temp_load(panel)
    assert (t["r2_T_T2"] >= t["r2_T"]).all() and (t["r2_T_T2_hour_weekday"] >= t["r2_T_T2"]).all()


def test_volatility_by_counts_and_exclusion(panel):
    full = eda.volatility_by(panel, "temperature")
    assert full["hours"].sum() == len(panel)
    day = sorted(set(panel["operating_date"]))[10]
    ex = eda.volatility_by(panel, "temperature", exclude_days=[day])
    assert full["hours"].sum() - ex["hours"].sum() == (panel["operating_date"] == day).sum()
    for by in ("load_decile", "net_load_decile", "hour"):
        assert eda.volatility_by(panel, by)["hours"].sum() == len(panel)


def test_extremes_tables_add_up(panel):
    ex = eda.extremes(panel)
    n = int(ex["summary"]["extreme_zone_hours"].iloc[0])
    assert ex["by_hour"].to_numpy().sum() == n == ex["by_day"]["zone_hours"].sum()
    assert len(ex["by_hour"]) == 24


def test_run_returns_every_table(panel):
    tables = eda.run(panel)
    assert {"q1_spread_summary", "q2_temp_load", "q3_by_temperature_excl_event",
            "q5_extremes_by_hour", "dependence_effective_n"} <= set(tables)
    assert all(isinstance(t, pd.DataFrame) and len(t) for t in tables.values())


def test_event_days_is_top_day_plus_busier_neighbour(panel):
    p = panel.copy()
    days = sorted(set(p["operating_date"]))
    d0, d1 = days[20], days[21]
    p.loc[(p["operating_date"] == d0) & p["hour_beginning"].between(5, 12), "spread"] = -5000.0
    p.loc[(p["operating_date"] == d1) & p["hour_beginning"].between(5, 7), "spread"] = 4000.0
    assert eda.event_days(p) == (d0, d1)
    assert eda.event_label((d0, d1)) == f"{d0:%Y-%m-%d}/{d1:%d}"


def test_zonal_summary_matches_its_sources(panel):
    z = eda.zonal_summary(panel)
    s = eda.spread_summary(panel)
    assert list(z.index) == ["houston", "north", "south", "west"]
    assert np.allclose(z["spread_median"], s["median"]) and np.allclose(z["spread_p99"], s["p99"])
    assert np.allclose(z["share_DA_above_RT"], s["share_positive"])
    assert np.allclose(z["load_r2_T_T2"], eda.temp_load(panel)["r2_T_T2"])
    assert z["load_factor"].between(0, 1).all()
    assert (z["spread_p1"] <= z["spread_median"]).all() and (z["abs_spread_mean"] >= 0).all()
    # hour-weighted on/off-peak means recombine to the overall mean
    on = panel.groupby("zone")["on_peak"].mean()
    assert np.allclose(on * z["spread_onpeak_mean"] + (1 - on) * z["spread_offpeak_mean"], z["spread_mean"])


def test_tail_vs_normal_counts(panel):
    t = eda.tail_vs_normal(panel, threshold=50.0)
    assert t.loc["all zones", "observed_beyond"] == (panel["spread"].abs() > 50).sum()
    assert t.drop("all zones")["observed_beyond"].sum() == t.loc["all zones", "observed_beyond"]
    assert (t["normal_expected_beyond"] >= 0).all()


def test_curvature_contribution_nests_and_reports_every_cell():
    from ercot_rsm import analysis

    s = config.load(start="2025-01-01", end="2025-06-01")  # 5 months: two forward folds
    p = pipeline.build_panel(synthetic.make(s), s)
    c = analysis.curvature_contribution(p, s)
    cfg = s.section("rsm")
    assert len(c) == len(cfg["responses"]) * p["zone"].nunique()
    # nested OLS fits never lose in-sample R²
    assert (c["r2_M2"] >= c["r2_M1"] - 1e-12).all() and (c["r2_M3"] >= c["r2_M1H"] - 1e-12).all()
    assert (c["r2_M1H"] >= c["r2_M1"] - 1e-12).all() and (c["r2_M3"] >= c["r2_M2"] - 1e-12).all()
    for col in ("p_quadratic", "p_interaction", "p_second_order", "p_second_order_with_hour"):
        assert c[col].between(0, 1).all()
    assert c[[f"fwd_skill_{m}" for m in ("M1", "M2", "M1H", "M3")]].notna().all().all()
    assert c["share_fitted_negative_M2"].between(0, 1).all()


# --------------------------------------------------------------------------
# Q6: extremes vs system net load, the profile, and the event window
# --------------------------------------------------------------------------


def _with_forecast_columns(panel, seed=1):
    """The synthetic panel has no DA-time forecasts; add plausible columns for the tests."""
    rng = np.random.default_rng(seed)
    p = panel.copy()
    n = len(p)
    for c in ("wind_fc", "solar_fc", "sys_load_fc", "load_fc_mw"):
        p[c] = rng.uniform(1000, 20000, n)
    for c in ("wind_err_hsl", "solar_err_hsl", "load_err", "net_load_ramp"):
        p[c] = rng.normal(0, 1000, n)
    p["net_load_fc"] = p["sys_load_fc"] - p["wind_fc"] - p["solar_fc"]
    p["net_load_err_avail"] = rng.normal(0, 1500, n)
    return p


def test_net_load_deciles_balance_hours_within_hour(panel):
    d = eda.net_load_deciles(panel)
    assert d["nl_decile"].between(1, 10).all() and d["nl_decile_within_hour"].between(1, 10).all()
    counts = pd.crosstab(panel["hour_beginning"], d["nl_decile_within_hour"])
    assert (counts.max(axis=1) - counts.min(axis=1)).max() <= 4  # 4 zones share each system hour


def test_tail_by_net_load_adds_up_and_bootstrap_brackets(panel):
    rows, summ = eda.tail_by_net_load(panel, n_boot=300)
    ext = eda.tail_flags(panel)["extreme"]
    r = rows[(rows.decile_kind == "nl_decile") & (rows["sample"] == "all days")]
    per_zone = r[r.zone != "all zones"].groupby("zone")["extreme_hours"].sum()
    assert (per_zone == ext.groupby(panel["zone"]).sum()).all()
    assert r[r.zone == "all zones"]["extreme_hours"].sum() == ext.sum()
    s = summ.dropna(subset=["ratio_top_vs_rest", "ratio_ci05"])
    assert (s.ratio_ci05 <= s.ratio_top_vs_rest + 1e-9).all() and (s.ratio_top_vs_rest <= s.ratio_ci95 + 1e-9).all()


def test_fixed_threshold_counts_and_undefined_ratio(panel):
    f = eda.tail_by_net_load_fixed(panel, thresholds=(5, 1e9), n_boot=50)
    for zone, d in panel.groupby("zone"):
        row = f[(f.zone == zone) & (f.definition == "|spread| > 5")].iloc[0]
        assert row.events == (d["spread"].abs() > 5).sum()
        row = f[(f.zone == zone) & (f.definition == "RT above DA by > 5")].iloc[0]
        assert row.events == (d["spread"] < -5).sum()
    never = f[f.threshold == 1e9]
    assert (never.events == 0).all() and never.ratio_top_vs_rest.isna().all()
    assert np.isfinite(f.ratio_top_vs_rest.dropna()).all()


def test_extreme_profile_groups_partition_the_panel(panel):
    pr = eda.extreme_profile(_with_forecast_columns(panel))
    n = pr[pr.variable == "zone_hours"].set_index("group")["median"]
    assert n.sum() == len(panel)
    assert n["RT_above_DA_extreme"] + n["DA_above_RT_extreme"] == eda.tail_flags(panel)["extreme"].sum()
    assert set(pr.variable) >= set(eda.PROFILE_VARS)


def test_event_window_rt15_matches_the_hourly_panel(panel):
    p = _with_forecast_columns(panel)
    day = sorted(p["operating_date"].unique())[20]
    d = p[p["operating_date"] == day]
    offs = np.array([-3.0, 1.0, -1.0, 3.0])  # four quarter-hours that average to the hourly RT
    starts = pd.DatetimeIndex(np.repeat(d["interval_start_utc"].dt.tz_localize(None).to_numpy(), 4))
    starts = (starts + pd.to_timedelta(np.tile([0, 15, 30, 45], len(d)), unit="min")).tz_localize("UTC")
    rt15 = pd.DataFrame({
        "interval_start_utc": starts,
        "location": np.repeat(d["location"].to_numpy(), 4),
        "price": np.repeat(d["rt_price"].to_numpy(), 4) + np.tile(offs, len(d))})
    hourly, rt = eda.event_window(p, rt15, day)
    hours = d.groupby("zone").size()
    assert (hourly.groupby("zone").size() == hours).all()
    assert (rt.groupby("zone").size() == 4 * hours).all()
    m = rt.assign(h=rt["interval_start_utc"].dt.floor("h")).groupby(["zone", "h"])["price"].mean()
    ref = hourly.set_index(["zone", "interval_start_utc"])["rt_price"]
    assert np.allclose(m.to_numpy(), ref.loc[m.index].to_numpy())
    assert np.allclose(hourly["net_load_avail"] - hourly["net_load_fc"], hourly["net_load_err_avail"])
    with pytest.raises(ValueError, match="not in the panel"):
        eda.event_window(p, rt15, pd.Timestamp("1999-01-01").date())


def test_temp_load_forward_prefers_the_quadratic_on_a_v_shape():
    s = config.load(start="2025-01-01", end="2025-07-01")  # 6 months: three forward folds
    p = pipeline.build_panel(synthetic.make(s), s)
    t = eda.temp_load_forward(p)
    assert (t["test_hours"] > 0).all()
    assert (t["sse_quadratic"] <= t["sse_linear"] * 1.05).all()  # never materially worse on synthetic V-shaped load
    short = eda.temp_load_forward(p[p["month"].astype(str) <= "2025-03"])
    assert short["mse_reduction_quadratic_vs_linear"].isna().all()  # no forward fold -> undefined, not a crash


def test_event_forward_predictions_train_only_on_earlier_months():
    from ercot_rsm import surprise

    s = config.load(start="2025-01-01", end="2025-06-01")
    p = _with_forecast_columns(pipeline.build_panel(synthetic.make(s), s))
    day = sorted(d for d in p["operating_date"].unique() if str(d) >= "2025-05-01")[3]
    e = surprise.event_forward_predictions(p, s.section("rsm"), day, zone="houston")
    assert len(e) == (p[(p.zone == "houston") & (p.operating_date == day)]).shape[0]
    assert e["train_months"].iloc[0] == "2025-01..2025-04"
    assert e[["pred_S_forward", "pred_M2_forward"]].notna().all().all()
