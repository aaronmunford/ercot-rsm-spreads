"""Forecast vintages, surprise features, the pre-specified ladder and the ex-ante guard."""

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import config, rsm, surprise
from ercot_rsm import features as F
from ercot_rsm.data import gridstatus_fetch as gs

UTC = "interval_start_utc"
TZ = "America/Chicago"


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def test_cutoff_is_market_time_on_the_previous_day_across_dst():
    # operating day 2025-03-10 (CDT); cutoff 09:45 CDT on 03-09 (the spring-forward day itself) = 14:45 UTC
    t = pd.Series([pd.Timestamp("2025-03-10 00:00", tz=TZ).tz_convert("UTC"),
                   pd.Timestamp("2025-11-03 00:00", tz=TZ).tz_convert("UTC")])
    cut = gs.cutoff_times(t, "09:45:00", TZ)
    assert cut.tolist() == [utc("2025-03-09 14:45"), utc("2025-11-02 15:45")]  # 09:45 CDT, 09:45 CST


def test_vintage_check_rejects_a_row_published_after_the_cutoff():
    a = pd.Timestamp("2025-07-15", tz=TZ).tz_convert("UTC")
    b = a + pd.Timedelta(days=1)
    t = pd.date_range(a, b, freq="h", inclusive="left")
    ok = pd.DataFrame({UTC: t, "publish_time_utc": utc("2025-07-14 14:30"), "system_total": 1.0})
    s = gs.validate_chunk(ok, a, b, gs.HOUR, cutoff="09:45:00", tz=TZ)
    assert s["hard"] == [] and s["vintage"]["late_rows"] == 0 and s["vintage"]["lag_h_max"] == pytest.approx(0.25)
    late = ok.assign(publish_time_utc=[utc("2025-07-14 14:55")] + [utc("2025-07-14 14:30")] * 23)  # 09:55 CDT
    assert any("after the 09:45:00" in h for h in gs.validate_chunk(late, a, b, gs.HOUR, cutoff="09:45:00", tz=TZ)["hard"])


def _frames(n=48):
    t = pd.date_range(utc("2025-07-15 05:00"), periods=n, freq="h")
    lw = pd.DataFrame({UTC: t, "total": 50_000.0 + 100 * np.arange(n)})
    lf = pd.DataFrame({UTC: t, "system_total": 49_000.0 + 100 * np.arange(n), "north": 1.0, "south": 1.0,
                       "west": 1.0, "houston": 1.0})
    wf = pd.DataFrame({UTC: t, "stwpf_system_wide": 10_000.0, "stwpf_west": 3_000.0, "stwpf_panhandle": 2_000.0})
    sf = pd.DataFrame({UTC: t, "stppf_system_wide": 5_000.0})
    wa = pd.DataFrame({UTC: t, "gen_system_wide": 9_000.0, "hsl_system_wide": 10_500.0, "gen_west": 2_500.0,
                       "gen_panhandle": 2_000.0})
    sa = pd.DataFrame({UTC: t, "gen_system_wide": 4_000.0, "hsl_system_wide": 5_000.0})
    return lw, lf, wf, sf, wa, sa


def test_error_signs_and_ramps():
    x = F.system_expectations(*_frames())
    r = x.iloc[5]
    assert r["sys_load_err"] == 1_000            # 1 GW more load than forecast: tighter
    assert r["wind_err"] == -1_000 and r["wind_err_hsl"] == 500   # curtailed below forecast, availability above
    assert r["net_load_fc"] == r["sys_load_fc"] - 15_000
    assert r["net_load_err"] == 1_000 + 1_000 + 1_000             # load up, wind and solar down
    assert r["net_load_err_avail"] == 1_000 - 500 - 0
    assert r["wind_err_west"] == -500
    assert np.isnan(x["net_load_ramp"].iloc[0]) and (x["net_load_ramp"].iloc[1:] == 100).all()
    gappy = [f.drop(index=10) for f in _frames()]
    g = F.system_expectations(*gappy)
    assert np.isnan(g.loc[g[UTC] == utc("2025-07-15 16:00"), "net_load_ramp"]).all()  # no ramp across a gap


def test_one_vintage_per_hour_is_enforced():
    lw, lf, *rest = _frames()
    with pytest.raises(ValueError, match="one vintage per hour"):
        F.system_expectations(lw, pd.concat([lf, lf.iloc[:1]]), *rest)


def test_exante_gas_is_two_trade_days_old():
    gas = pd.DataFrame({"trade_date": pd.to_datetime(["2025-07-10", "2025-07-11", "2025-07-14", "2025-07-15"]).date,
                        "price": [1.0, 2.0, 3.0, 4.0]})
    days = pd.Series(pd.to_datetime(["2025-07-16", "2025-07-15", "2025-07-14"]).date)
    # D=07-16: at 09:45 on 07-15 the 07-15 index is not final -> last trade before 07-15 is 07-14
    assert F.exante_gas(gas, days).tolist() == [3.0, 2.0, 2.0]
    assert F.gas_for_operating_dates(gas, days).tolist() == [4.0, 3.0, 2.0]


def test_ladder_is_nested_and_hierarchical():
    specs = surprise.ladder_specs(config.load().section("rsm"))
    for prev, cur in zip("ABC", "BCD"):
        assert set(specs[prev]["terms"]) < set(specs[cur]["terms"])
    for k in "ABCD":
        exps = rsm.term_exponents(tuple(specs[k]["factors"]), 2)
        rsm.check_hierarchical(specs[k]["terms"], exps)
    assert len(specs["D"]["terms"]) == 9 + 3 + 3 + 4
    assert specs["D"]["added"] == ["wind_err_hsl", "solar_err_hsl"]


def test_exante_model_rejects_realised_inputs():
    specs = surprise.ladder_specs(config.load().section("rsm"))
    surprise.assert_exante(specs["E"])
    for leak in ({"factors": ["net_load_fc", "temp_f"], "blocks": []},
                 {"factors": ["net_load_fc"], "blocks": ["regime", "month"]},
                 {"factors": ["load_fc_mw", "gas_price"], "blocks": []}):
        with pytest.raises(ValueError, match="not available"):
            surprise.assert_exante(leak)


def test_out_of_year_refuses_overlapping_years():
    from ercot_rsm import crossyear, pipeline, synthetic

    s = config.load(start="2025-01-01", end="2025-03-01")
    p = pipeline.build_panel(synthetic.make(s), s)
    with pytest.raises(AssertionError, match="must end before"):
        crossyear.out_of_year(p, p, s.section("rsm"))


def test_basis_identity_and_covariance_shares():
    from ercot_rsm import basis

    rng = np.random.default_rng(4)
    t = pd.date_range(utc("2025-07-15 05:00"), periods=200, freq="h")
    hubs = pd.DataFrame({UTC: t, "da_HUBAVG": 30 + rng.normal(0, 5, 200), "rt_HUBAVG": 30 + rng.normal(0, 20, 200)})
    da = hubs["da_HUBAVG"] + rng.normal(0, 2, 200)
    rt = hubs["rt_HUBAVG"] + rng.normal(0, 8, 200)
    panel = pd.DataFrame({UTC: t, "zone": "west", "operating_date": t.date, "hour_beginning": t.hour,
                          "da_price": da, "rt_price": rt, "spread": da - rt, "load_mw": 9000.0,
                          "extreme": False})
    d = basis.decompose(panel, hubs, "HUBAVG")
    assert np.allclose(d["spread"], d["spread_hub"] + d["basis_surprise"])
    v = basis.variance_shares(d).iloc[0]
    assert v["cov_share_hub"] + v["cov_share_basis"] == pytest.approx(1.0)
    assert v["share_var_hub"] + v["share_var_basis"] + v["share_2cov"] == pytest.approx(1.0)
    bad = hubs.assign(da_HUBAVG=hubs["da_HUBAVG"] + 1)
    d2 = basis.decompose(panel, bad, "HUBAVG")  # the identity holds for any reference by construction
    assert np.allclose(d2["spread"], d2["spread_hub"] + d2["basis_surprise"])
