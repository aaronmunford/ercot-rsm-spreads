"""Assemble the hourly zone panel from normalised raw inputs, and audit it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from .config import Study

UTC = F.UTC


@dataclass
class RawInputs:
    da: pd.DataFrame          # interval_start_utc, location, price   (hourly)
    rt: pd.DataFrame          # interval_start_utc, location, price   (15-min)
    load_wide: pd.DataFrame   # interval_start_utc, <zone columns>
    station_obs: pd.DataFrame  # station, valid_utc, tmpf
    gas: pd.DataFrame         # trade_date, price
    fuel_mix: pd.DataFrame | None = None  # interval_start_utc, wind, solar, ... (hourly)
    forecasts: dict | None = None  # load_fc, wind_fc, solar_fc (+ *_sens), wind_act, solar_act


def _in_window(df: pd.DataFrame, study: Study) -> pd.DataFrame:
    return df[(df[UTC] >= study.start_utc) & (df[UTC] < study.end_utc)]


def build_panel(raw: RawInputs, study: Study) -> pd.DataFrame:
    """One row per (zone, hour): prices, spread, gas, IMHR, load, temperature, calendar."""
    zones = study.zones
    point_to_zone = {z.price_point: name for name, z in zones.items()}

    # prices -> spread
    rt_h = F.rt_to_hourly(raw.rt)
    da = raw.da.assign(**{UTC: raw.da[UTC].dt.floor("h")})
    px = F.da_rt_spread(da, rt_h)
    px["zone"] = px["location"].map(point_to_zone)
    px = _in_window(px.dropna(subset=["zone"]), study)
    # One row for every zone-hour in the window: an hour missing from both
    # markets shows up as a row of NaNs (and model_ok = False), not as nothing.
    grid = pd.MultiIndex.from_product(
        [pd.date_range(study.start_utc, study.end_utc, freq="h", inclusive="left"), list(zones)],
        names=[UTC, "zone"])
    px = px.set_index([UTC, "zone"]).reindex(grid).reset_index()
    px["location"] = px["zone"].map({n: z.price_point for n, z in zones.items()})
    px["rt_n"] = px["rt_n"].fillna(0).astype(int)
    px["rt_complete"] = px["rt_complete"].astype("boolean").fillna(False).astype(bool)

    # load
    load = F.load_long(raw.load_wide, {n: z.load_column for n, z in zones.items()})

    # temperature
    w = study.section("weather")
    st = F.obs_to_hourly(raw.station_obs, w.get("max_gap_hours", 3), tuple(w.get("plausible_range_f", [-20, 125])))
    temp = F.zone_temperature(st, {n: z.stations for n, z in zones.items()},
                              w.get("min_weight_coverage", 0.5))

    panel = (px.merge(load, on=[UTC, "zone"], how="left", validate="one_to_one")
               .merge(temp, on=[UTC, "zone"], how="left", validate="one_to_one"))
    panel = F.add_local_time(panel, study.market_tz)

    # gas and implied heat rates
    g = study.section("gas")
    panel["gas_price"] = F.gas_for_operating_dates(
        raw.gas, panel["operating_date"], g.get("alignment", "previous_trade_date"))
    mg = g.get("min_price", 0.5)
    panel["imhr_da"] = F.implied_heat_rate(panel["da_price"], panel["gas_price"], mg)
    panel["imhr_rt"] = F.implied_heat_rate(panel["rt_price"], panel["gas_price"], mg)
    # = spread / gas: the DA-RT spread in heat-rate units (MMBtu/MWh)
    panel["imhr_spread"] = panel["imhr_da"] - panel["imhr_rt"]

    # load measures, degree hours, calendar flags
    panel = F.add_load_measures(panel)
    panel = pd.concat([panel, F.degree_hours(panel["temp_f"], w.get("degree_base_f", 65.0))], axis=1)
    hours = tuple(study.section("features").get("on_peak_hour_beginning", [6, 21]))
    panel["on_peak"] = F.on_peak(panel, hours)
    panel["regime"] = np.where(panel["interval_start_local"] >= study.regime_break, "post", "pre")

    if raw.fuel_mix is not None:
        fm = raw.fuel_mix.assign(**{UTC: raw.fuel_mix[UTC].dt.floor("h")})
        cols = {c.lower(): c for c in fm.columns}
        ren = [cols[c] for c in ("wind", "solar") if c in cols]
        if ren:
            fm = fm.groupby(UTC, as_index=False)[ren].mean()
            fm["renewables_mw"] = fm[ren].sum(axis=1)
            panel = panel.merge(fm[[UTC, "renewables_mw"]], on=UTC, how="left")
            sys_load = panel.groupby(UTC)["load_mw"].transform("sum")
            panel["system_load_mw"] = sys_load
            panel["system_net_load_mw"] = sys_load - panel["renewables_mw"]

    if raw.forecasts:
        fc = raw.forecasts
        cols = {n: z.load_column for n, z in zones.items()}
        panel = panel.merge(F.zonal_load_forecast(fc["load_fc"], cols), on=[UTC, "zone"], how="left",
                            validate="one_to_one")
        panel["load_err"] = panel["load_mw"] - panel["load_fc_mw"]
        sysx = F.system_expectations(raw.load_wide, fc["load_fc"], fc["wind_fc"], fc["solar_fc"],
                                     fc["wind_act"], fc["solar_act"])
        panel = panel.merge(sysx, on=UTC, how="left", validate="many_to_one")
        if "load_fc_sens" in fc:
            sens = F.system_expectations(raw.load_wide, fc["load_fc_sens"], fc["wind_fc_sens"], fc["solar_fc_sens"],
                                         fc["wind_act"], fc["solar_act"], suffix="_sens")
            panel = panel.merge(sens, on=UTC, how="left", validate="many_to_one")
            zs = F.zonal_load_forecast(fc["load_fc_sens"], cols).rename(columns={"load_fc_mw": "load_fc_mw_sens"})
            panel = panel.merge(zs, on=[UTC, "zone"], how="left", validate="one_to_one")
            panel["load_err_sens"] = panel["load_mw"] - panel["load_fc_mw_sens"]
        panel["gas_price_exante"] = F.exante_gas(raw.gas, panel["operating_date"])

    # The estimation sample: every input observed, RT hour built from all four
    # intervals, IMHR defined. Nothing is imputed; excluded hours stay in the
    # panel so the exclusion can be audited.
    panel["model_ok"] = (
        panel["rt_complete"]
        & panel[["da_price", "rt_price", "load_mw", "temp_f", "gas_price", "imhr_da"]].notna().all(axis=1)
    )
    return panel.sort_values(["zone", UTC]).reset_index(drop=True)


def quality_report(panel: pd.DataFrame, study: Study) -> pd.DataFrame:
    """Per-zone coverage against the hours the study window actually contains.

    The expected count is computed in UTC, so the 23- and 25-hour DST days
    are counted correctly rather than assumed to be 24.
    """
    expected = int((study.end_utc - study.start_utc) / pd.Timedelta(hours=1))
    rows = []
    for zone, d in panel.groupby("zone"):
        rows.append({
            "zone": zone,
            "expected_hours": expected,
            "panel_rows": int(len(d)),
            "hours_with_spread": int(d["spread"].notna().sum()),
            "model_ok_hours": int(d["model_ok"].sum()),
            "rt_incomplete_hours": int((~d["rt_complete"]).sum()),
            "missing_load": int(d["load_mw"].isna().sum()),
            "missing_temp": int(d["temp_f"].isna().sum()),
            "missing_gas": int(d["gas_price"].isna().sum()),
            "imhr_undefined": int(d["imhr_da"].isna().sum()),
            "negative_rt_hours": int((d["rt_price"] < 0).sum()),
            "rt_price_max": float(d["rt_price"].max()),
            "spread_p01": float(d["spread"].quantile(0.01)),
            "spread_p99": float(d["spread"].quantile(0.99)),
        })
    return pd.DataFrame(rows).set_index("zone")
