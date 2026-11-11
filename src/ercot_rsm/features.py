"""Derived measures: DA-RT spreads, implied heat rates, load and weather. Pure functions.

Conventions used throughout:

* Every join is on ``interval_start_utc`` (tz-aware UTC). Local time is
  derived, never joined on, so the DST fall-back hour (two 01:00s) and the
  spring-forward gap cannot silently collide or vanish.
* An "hour" is identified by its start. ERCOT labels hours by their end
  (HE01..HE24); ``hour_beginning = HE - 1``.
* Prices are $/MWh, gas $/MMBtu, so heat rates are MMBtu/MWh.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

UTC = "interval_start_utc"

# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------


def add_local_time(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = df.copy()
    local = out[UTC].dt.tz_convert(tz)
    out["interval_start_local"] = local
    out["operating_date"] = local.dt.date
    out["hour_beginning"] = local.dt.hour
    out["weekday"] = local.dt.weekday
    out["month"] = local.dt.strftime("%Y-%m")
    return out


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def nerc_holidays(years) -> set[date]:
    """The six NERC off-peak holidays. A Sunday holiday moves to Monday; Saturday ones do not."""
    out = set()
    for y in years:
        days = [
            date(y, 1, 1),
            _last_weekday(y, 5, 0),       # Memorial Day
            date(y, 7, 4),
            _nth_weekday(y, 9, 0, 1),     # Labor Day
            _nth_weekday(y, 11, 3, 4),    # Thanksgiving
            date(y, 12, 25),
        ]
        out |= {d + timedelta(days=1) if d.weekday() == 6 else d for d in days}
    return out


def on_peak(df: pd.DataFrame, hours: tuple[int, int] = (6, 21)) -> pd.Series:
    """Weekday, non-holiday, hour-beginning in ``hours`` inclusive (HE07–HE22 by default)."""
    years = {d.year for d in df["operating_date"]}
    hol = nerc_holidays(years)
    return (
        (df["weekday"] < 5)
        & ~df["operating_date"].isin(hol)
        & df["hour_beginning"].between(*hours)
    )


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------


def rt_to_hourly(rt: pd.DataFrame, expected_per_hour: int = 4) -> pd.DataFrame:
    """Average 15-minute RT settlement point prices into hours.

    Also returns the intra-hour standard deviation (a within-hour volatility
    measure the hourly DA market cannot see) and the interval count, so
    incomplete hours can be found rather than silently averaged.
    """
    g = rt.assign(**{UTC: rt[UTC].dt.floor("h")}).groupby([UTC, "location"])["price"]
    out = g.agg(rt_price="mean", rt_intrahour_std="std", rt_n="count").reset_index()
    out["rt_complete"] = out["rt_n"] == expected_per_hour
    return out


def da_rt_spread(da: pd.DataFrame, rt_hourly: pd.DataFrame) -> pd.DataFrame:
    """DA minus RT at each settlement point and hour. Positive = DA premium.

    Outer join: an hour present in only one market keeps its row (with the
    spread missing) rather than vanishing from the panel unnoticed.
    """
    da = da.rename(columns={"price": "da_price"})[[UTC, "location", "da_price"]]
    m = da.merge(rt_hourly, on=[UTC, "location"], how="outer", validate="one_to_one")
    m["rt_n"] = m["rt_n"].fillna(0).astype(int)
    m["rt_complete"] = m["rt_complete"].astype("boolean").fillna(False).astype(bool)
    m["spread"] = m["da_price"] - m["rt_price"]
    m["abs_spread"] = m["spread"].abs()
    return m


# --------------------------------------------------------------------------
# Gas and implied heat rate
# --------------------------------------------------------------------------


def gas_for_operating_dates(
    gas: pd.DataFrame, operating_dates: pd.Series, alignment: str = "previous_trade_date"
) -> pd.Series:
    """Gas price applicable to each operating date.

    ``gas`` has ``trade_date`` (date) and ``price``. With
    ``previous_trade_date`` the price for day D is the one traded on the last
    trade date strictly before D, matching the day-ahead timing: DA power for
    D clears on D-1, and gas for flow on D (and the weekend package) trades on
    the prior business day. ``same_day`` uses the last trade date on or before D.
    """
    g = gas.dropna(subset=["price"]).sort_values("trade_date")
    t = pd.to_datetime(g["trade_date"]).to_numpy()
    p = g["price"].to_numpy(float)
    d = pd.to_datetime(pd.Series(operating_dates)).to_numpy()
    if alignment == "previous_trade_date":
        idx = np.searchsorted(t, d, side="left") - 1
    elif alignment == "same_day":
        idx = np.searchsorted(t, d, side="right") - 1
    else:
        raise ValueError(f"unknown gas alignment {alignment!r}")
    out = np.where(idx >= 0, p[np.clip(idx, 0, None)], np.nan)
    return pd.Series(out, index=pd.Series(operating_dates).index)


def implied_heat_rate(power: pd.Series, gas: pd.Series, min_gas_price: float = 0.5) -> pd.Series:
    """Implied market heat rate, MMBtu/MWh = power $/MWh / gas $/MMBtu.

    Undefined (NaN) where gas is below ``min_gas_price``: the ratio explodes
    as gas approaches zero and changes sign below it, which says nothing
    about generation economics. Negative power prices are kept, since they
    are real market outcomes, and produce negative heat rates.
    """
    gas = gas.astype(float)
    return (power.astype(float) / gas).where(gas >= min_gas_price)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def load_long(load_wide: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Wide forecast-zone load -> (interval_start_utc, zone, load_mw), hourly mean."""
    lower = {c.lower(): c for c in load_wide.columns}
    missing = [c for c in columns.values() if c.lower() not in lower]
    if missing:
        raise KeyError(f"load columns {missing} not in {list(load_wide.columns)}")
    parts = []
    for zone, col in columns.items():
        s = load_wide[[UTC, lower[col.lower()]]].rename(columns={lower[col.lower()]: "load_mw"})
        parts.append(s.assign(zone=zone))
    out = pd.concat(parts, ignore_index=True)
    out[UTC] = out[UTC].dt.floor("h")
    return out.groupby([UTC, "zone"], as_index=False)["load_mw"].mean()


def add_load_measures(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-zone load normalisations that make zones comparable.

    * ``load_norm``  load / the zone's peak load in the sample (0–1).
    * ``load_anom``  load minus the zone's median for that hour-of-week, MW.
    * ``load_z``     load standardised within zone.
    """
    out = panel.copy()
    g = out.groupby("zone")["load_mw"]
    out["load_norm"] = out["load_mw"] / g.transform("max")
    out["load_z"] = (out["load_mw"] - g.transform("mean")) / g.transform("std")
    how = out["weekday"] * 24 + out["hour_beginning"]
    out["load_anom"] = out["load_mw"] - out.groupby(["zone", how])["load_mw"].transform("median")
    return out


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


def obs_to_hourly(obs: pd.DataFrame, max_gap_hours: int = 3,
                  plausible_f: tuple[float, float] = (-20.0, 125.0)) -> pd.DataFrame:
    """Station observations -> one temperature per station-hour.

    Readings outside ``plausible_f`` are sensor faults (e.g. -63 F at Brownsville
    on 2024-04-03, just before the sensor went missing) and are dropped first.

    Routine METARs are issued a few minutes before the hour (e.g. :53), so
    each observation is assigned to the nearest hour, then averaged.
    Gaps up to ``max_gap_hours`` are linearly interpolated per station;
    longer gaps stay missing.
    """
    o = obs.dropna(subset=["tmpf"])
    o = o[o["tmpf"].between(*plausible_f)].copy()
    o[UTC] = o["valid_utc"].dt.round("h")
    h = o.groupby(["station", UTC], as_index=False)["tmpf"].mean()
    parts = []
    for st, g in h.groupby("station"):
        idx = pd.date_range(g[UTC].min(), g[UTC].max(), freq="h")
        s = g.set_index(UTC)["tmpf"].reindex(idx)
        filled = s.interpolate(limit=max_gap_hours, limit_area="inside")
        # interpolate(limit=) fills the first `limit` hours of a longer gap;
        # blank those again so only short gaps are filled.
        run = s.isna().groupby(s.notna().cumsum()).transform("sum")
        filled[s.isna() & (run > max_gap_hours)] = np.nan
        parts.append(pd.DataFrame({UTC: idx, "station": st, "temp_f": filled.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def zone_temperature(
    station_hourly: pd.DataFrame, weights: dict[str, dict[str, float]], min_coverage: float = 0.5
) -> pd.DataFrame:
    """Weighted zonal temperature, renormalising over the stations reporting.

    Missing if the reporting stations carry less than ``min_coverage`` of the
    zone's weight, so one outlying station cannot stand in for a whole zone.
    """
    rows = []
    for zone, w in weights.items():
        s = station_hourly[station_hourly["station"].isin(w)].dropna(subset=["temp_f"])
        s = s.assign(w=s["station"].map(w))
        g = s.groupby(UTC)
        cov = g["w"].sum()
        t = g.apply(lambda d: np.average(d["temp_f"], weights=d["w"]), include_groups=False)
        rows.append(pd.DataFrame({UTC: cov.index, "zone": zone,
                                  "temp_f": t.where(cov >= min_coverage).to_numpy(),
                                  "temp_coverage": cov.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def degree_hours(temp_f: pd.Series, base: float = 65.0) -> pd.DataFrame:
    return pd.DataFrame({"cdh": (temp_f - base).clip(lower=0), "hdh": (base - temp_f).clip(lower=0)})


# --------------------------------------------------------------------------
# Representative spreads
# --------------------------------------------------------------------------


def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def daily_spread_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Representative daily DA–RT spreads per zone.

    Mean and median over all hours, mean over on-peak and off-peak hours,
    mean absolute spread, and the within-day standard deviation of RT prices
    (a daily realised-volatility measure).
    """
    def agg(d: pd.DataFrame) -> pd.Series:
        on = d["on_peak"]
        return pd.Series({
            "hours": len(d),
            "spread_mean": d["spread"].mean(),
            "spread_median": d["spread"].median(),
            "spread_onpeak": d.loc[on, "spread"].mean(),
            "spread_offpeak": d.loc[~on, "spread"].mean(),
            "abs_spread_mean": d["abs_spread"].mean(),
            "rt_price_std": d["rt_price"].std(),
            "da_price_mean": d["da_price"].mean(),
            "rt_price_mean": d["rt_price"].mean(),
            "temp_f_max": d["temp_f"].max(),
            "load_mw_peak": d["load_mw"].max(),
        })

    return (panel.groupby(["zone", "operating_date"]).apply(agg, include_groups=False)
            .reset_index())


# --------------------------------------------------------------------------
# Expectations: DA-time forecasts and their errors (D-28)
# --------------------------------------------------------------------------

# West load zone wind: ERCOT's West and Panhandle wind regions.
WEST_WIND_REGIONS = ("west", "panhandle")


def _hourly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[[UTC, *cols]].copy()
    out[UTC] = pd.to_datetime(out[UTC], utc=True)
    if out[UTC].duplicated().any():
        raise ValueError("forecast frame has more than one row per hour; expected one vintage per hour")
    return out.sort_values(UTC).reset_index(drop=True)


def system_expectations(load_wide: pd.DataFrame, load_fc: pd.DataFrame, wind_fc: pd.DataFrame,
                        solar_fc: pd.DataFrame, wind_act: pd.DataFrame, solar_act: pd.DataFrame,
                        suffix: str = "") -> pd.DataFrame:
    """System-level forecasts (one DA-time vintage), actuals from the same ERCOT report
    families, and errors defined as actual - forecast.

    Positive ``*_load_err``/``net_load_err``: tighter than expected. Positive
    ``wind_err``/``solar_err``: more renewable output than expected. Every
    forecast-derived quantity, including the forecast ramp, uses the same
    vintage; realised series appear only on the actual side of an error.
    """
    lw = _hourly(load_wide, ["total"]).rename(columns={"total": "sys_load_act"})
    lf = _hourly(load_fc, ["system_total"]).rename(columns={"system_total": "sys_load_fc"})
    wf = _hourly(wind_fc, ["stwpf_system_wide", *[f"stwpf_{r}" for r in WEST_WIND_REGIONS]])
    sf = _hourly(solar_fc, ["stppf_system_wide"])
    wa = _hourly(wind_act, ["gen_system_wide", "hsl_system_wide", *[f"gen_{r}" for r in WEST_WIND_REGIONS]])
    sa = _hourly(solar_act, ["gen_system_wide", "hsl_system_wide"])
    d = (lw.merge(lf, on=UTC, how="left").merge(wf, on=UTC, how="left").merge(sf, on=UTC, how="left")
           .merge(wa, on=UTC, how="left").merge(sa, on=UTC, how="left", suffixes=("_wind", "_solar")))
    out = pd.DataFrame({UTC: d[UTC]})
    out["sys_load_fc"] = d["sys_load_fc"]
    out["sys_load_err"] = d["sys_load_act"] - d["sys_load_fc"]
    out["wind_fc"] = d["stwpf_system_wide"]
    out["wind_act"] = d["gen_system_wide_wind"]
    out["wind_err"] = out["wind_act"] - out["wind_fc"]
    out["wind_err_hsl"] = d["hsl_system_wide_wind"] - out["wind_fc"]  # availability surprise, curtailment-free
    out["solar_fc"] = d["stppf_system_wide"]
    out["solar_act"] = d["gen_system_wide_solar"]
    out["solar_err"] = out["solar_act"] - out["solar_fc"]
    out["solar_err_hsl"] = d["hsl_system_wide_solar"] - out["solar_fc"]
    out["wind_fc_west"] = sum(d[f"stwpf_{r}"] for r in WEST_WIND_REGIONS)
    out["wind_err_west"] = sum(d[f"gen_{r}"] for r in WEST_WIND_REGIONS) - out["wind_fc_west"]
    out["net_load_fc"] = out["sys_load_fc"] - out["wind_fc"] - out["solar_fc"]
    out["net_load_act"] = d["sys_load_act"] - out["wind_act"] - out["solar_act"]
    out["net_load_err"] = out["net_load_act"] - out["net_load_fc"]
    # availability-based: renewable surprise measured against HSL, so economic curtailment
    # (output cut because RT prices went negative) is not counted as a supply surprise
    out["net_load_err_avail"] = out["sys_load_err"] - out["wind_err_hsl"] - out["solar_err_hsl"]
    # consecutive hours of the system series (UTC order, no gaps assumed: checked below)
    gap = out[UTC].diff() != pd.Timedelta(hours=1)
    out["net_load_ramp"] = out["net_load_act"].diff().where(~gap)
    out["net_load_fc_ramp"] = out["net_load_fc"].diff().where(~gap)
    if suffix:
        keep = ["sys_load_fc", "sys_load_err", "net_load_fc", "net_load_err", "net_load_err_avail", "net_load_fc_ramp"]
        out = out[[UTC, *keep]].rename(columns={c: c + suffix for c in keep})
    return out


def zonal_load_forecast(load_fc: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Wide forecast-zone load forecast -> (interval_start_utc, zone, load_fc_mw)."""
    lf = load_fc.copy()
    lf[UTC] = pd.to_datetime(lf[UTC], utc=True)
    parts = [lf[[UTC, col]].rename(columns={col: "load_fc_mw"}).assign(zone=z) for z, col in columns.items()]
    return pd.concat(parts, ignore_index=True)


def exante_gas(gas: pd.DataFrame, operating_dates: pd.Series) -> pd.Series:
    """Gas known at the 09:45 D-1 cutoff: the last trade date strictly before D-1,
    because the D-1 next-day index is still trading at 09:45 (D-30)."""
    d = pd.to_datetime(pd.Series(operating_dates)) - pd.Timedelta(days=1)
    return gas_for_operating_dates(gas, d.dt.date, "previous_trade_date").set_axis(pd.Series(operating_dates).index)
