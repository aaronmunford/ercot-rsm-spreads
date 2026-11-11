from datetime import date

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import features as F

TZ = "America/Chicago"
UTC = F.UTC


def utc(*args):
    return pd.Timestamp(*args, tz="UTC")


def test_fall_back_day_has_25_distinct_hours():
    # 2025-11-02 local: 01:00 CDT and 01:00 CST are different UTC hours
    hours = pd.date_range(pd.Timestamp("2025-11-02", tz=TZ), pd.Timestamp("2025-11-03", tz=TZ),
                          freq="h", inclusive="left").tz_convert("UTC")
    df = F.add_local_time(pd.DataFrame({UTC: hours}), TZ)
    assert len(df) == 25
    assert (df["hour_beginning"] == 1).sum() == 2
    assert df[UTC].is_unique


def test_spring_forward_day_has_23_hours():
    hours = pd.date_range(pd.Timestamp("2026-03-08", tz=TZ), pd.Timestamp("2026-03-09", tz=TZ),
                          freq="h", inclusive="left")
    assert len(hours) == 23
    df = F.add_local_time(pd.DataFrame({UTC: hours.tz_convert("UTC")}), TZ)
    assert 2 not in set(df["hour_beginning"])


def test_nerc_holidays_sunday_moves_saturday_does_not():
    h = nerc_2026 = F.nerc_holidays([2026])
    assert date(2026, 7, 4) in h          # Saturday: not moved
    assert date(2026, 7, 3) not in h
    assert date(2026, 5, 25) in h         # Memorial Day
    assert date(2026, 9, 7) in h          # Labor Day
    assert date(2026, 11, 26) in h        # Thanksgiving
    assert date(2026, 12, 25) in h and date(2026, 1, 1) in h
    assert len(nerc_2026) == 6
    # 2022-12-25 was a Sunday -> Monday 26th
    assert date(2022, 12, 26) in F.nerc_holidays([2022])


def test_on_peak_definition():
    local = pd.DatetimeIndex([
        "2026-07-06 05:00", "2026-07-06 06:00", "2026-07-06 21:00", "2026-07-06 22:00",  # Monday
        "2026-07-04 12:00",  # Saturday holiday
        "2026-09-07 12:00",  # Labor Day (Monday)
        "2026-09-08 12:00",  # ordinary Tuesday
    ]).tz_localize(TZ)
    df = F.add_local_time(pd.DataFrame({UTC: local.tz_convert("UTC")}), TZ)
    assert F.on_peak(df).tolist() == [False, True, True, False, False, False, True]


def test_rt_to_hourly_mean_std_and_completeness():
    t0 = utc("2026-01-01 06:00")
    rt = pd.DataFrame({
        UTC: [t0 + pd.Timedelta(minutes=15 * i) for i in range(4)] + [t0 + pd.Timedelta(hours=1)],
        "location": "LZ_X",
        "price": [10.0, 20.0, 30.0, 40.0, 99.0],
    })
    h = F.rt_to_hourly(rt)
    first = h.iloc[0]
    assert first["rt_price"] == 25.0 and first["rt_n"] == 4 and first["rt_complete"]
    assert first["rt_intrahour_std"] == pytest.approx(np.std([10, 20, 30, 40], ddof=1))
    assert not h.iloc[1]["rt_complete"]


def test_spread_sign_and_join():
    t = [utc("2026-01-01 06:00"), utc("2026-01-01 07:00")]
    da = pd.DataFrame({UTC: t, "location": "LZ_X", "price": [50.0, 30.0]})
    rt = pd.DataFrame({UTC: t, "location": "LZ_X", "rt_price": [40.0, 45.0],
                       "rt_intrahour_std": 0.0, "rt_n": 4, "rt_complete": True})
    s = F.da_rt_spread(da, rt)
    assert s["spread"].tolist() == [10.0, -15.0]  # positive = DA premium
    assert s["abs_spread"].tolist() == [10.0, 15.0]


def test_spread_join_refuses_duplicate_hours():
    t = [utc("2026-01-01 06:00")] * 2
    da = pd.DataFrame({UTC: t, "location": "LZ_X", "price": [50.0, 51.0]})
    rt = pd.DataFrame({UTC: t[:1], "location": "LZ_X", "rt_price": [40.0],
                       "rt_intrahour_std": 0.0, "rt_n": 4, "rt_complete": True})
    with pytest.raises(pd.errors.MergeError):
        F.da_rt_spread(da, rt)


def test_gas_previous_trade_date_covers_weekends():
    gas = pd.DataFrame({"trade_date": [date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 12)],
                        "price": [3.0, 3.5, 4.0]})  # Thu, Fri, Mon
    days = pd.Series([date(2026, 1, 9), date(2026, 1, 10), date(2026, 1, 11),
                      date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 8)])
    got = F.gas_for_operating_dates(gas, days, "previous_trade_date")
    # Fri<-Thu, Sat/Sun/Mon<-Fri, Tue<-Mon, and nothing before the first trade date
    assert got.iloc[:5].tolist() == [3.0, 3.5, 3.5, 3.5, 4.0]
    assert np.isnan(got.iloc[5])
    same = F.gas_for_operating_dates(gas, days, "same_day")
    assert same.iloc[:5].tolist() == [3.5, 3.5, 3.5, 4.0, 4.0]


def test_implied_heat_rate_guard():
    power = pd.Series([40.0, -10.0, 40.0, 40.0])
    gas = pd.Series([4.0, 4.0, 0.2, -1.0])
    hr = F.implied_heat_rate(power, gas, min_gas_price=0.5)
    assert hr.iloc[0] == 10.0
    assert hr.iloc[1] == -2.5      # negative power kept
    assert hr.iloc[2:].isna().all()  # near-zero and negative gas -> undefined


def test_load_long_is_case_insensitive_and_hourly():
    t0 = utc("2026-01-01 06:00")
    wide = pd.DataFrame({UTC: [t0, t0 + pd.Timedelta(minutes=30)], "HOUSTON": [100.0, 200.0]})
    out = F.load_long(wide, {"houston": "houston"})
    assert len(out) == 1 and out["load_mw"].iloc[0] == 150.0
    with pytest.raises(KeyError):
        F.load_long(wide, {"north": "north"})


def test_load_measures():
    hours = pd.date_range(utc("2026-01-05"), periods=24 * 14, freq="h")
    df = F.add_local_time(pd.DataFrame({UTC: hours, "zone": "z",
                                        "load_mw": np.arange(len(hours), dtype=float) + 1}), TZ)
    out = F.add_load_measures(df)
    assert out["load_norm"].max() == 1.0
    assert out["load_z"].mean() == pytest.approx(0, abs=1e-12)
    # two weeks: each hour-of-week has two observations a week apart, median in between
    assert np.allclose(out["load_anom"].abs(), 84)


def test_obs_to_hourly_rounds_and_fills_only_short_gaps():
    base = utc("2026-01-01 00:53")
    times = [base + pd.Timedelta(hours=h) for h in range(12) if h not in (3, 6, 7, 8, 9)]
    obs = pd.DataFrame({"station": "AAA", "valid_utc": times, "tmpf": np.arange(len(times), dtype=float)})
    h = F.obs_to_hourly(obs, max_gap_hours=2).set_index(UTC)["temp_f"]
    assert h.index[0] == utc("2026-01-01 01:00")
    assert not np.isnan(h[utc("2026-01-01 04:00")])         # 1-hour gap filled
    assert h[[utc(f"2026-01-01 {x:02d}:00") for x in (7, 8, 9, 10)]].isna().all()  # 4-hour gap left


def test_zone_temperature_renormalises_and_requires_coverage():
    t = utc("2026-01-01 06:00")
    st = pd.DataFrame({UTC: [t, t, t + pd.Timedelta(hours=1)], "station": ["A", "B", "B"],
                       "temp_f": [60.0, 80.0, 80.0]})
    z = F.zone_temperature(st, {"z": {"A": 0.75, "B": 0.25}}, min_coverage=0.5).set_index(UTC)
    assert z.loc[t, "temp_f"] == pytest.approx(65.0)
    assert np.isnan(z.loc[t + pd.Timedelta(hours=1), "temp_f"])  # only 25% of weight reporting


def test_degree_hours():
    d = F.degree_hours(pd.Series([50.0, 65.0, 90.0]))
    assert d["cdh"].tolist() == [0, 0, 25] and d["hdh"].tolist() == [15, 0, 0]


def test_implausible_temperature_readings_are_dropped():
    t = pd.date_range(utc("2024-04-03 06:53"), periods=4, freq="h")
    obs = pd.DataFrame({"station": "BRO", "valid_utc": t, "tmpf": [59.0, 53.0, -63.0, 44.0]})
    h = F.obs_to_hourly(obs, max_gap_hours=1)
    assert h["temp_f"].min() > 0          # the -63 F fault never reaches the hourly series
    assert h["temp_f"].iloc[2] == pytest.approx((53.0 + 44.0) / 2)  # its hour is a 1-h gap, interpolated
