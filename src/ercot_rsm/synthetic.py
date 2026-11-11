"""Synthetic raw inputs in exactly the normalised schemas the fetchers produce.

For exercising the pipeline end to end before real data exists. NOTHING
produced from these frames may appear in the report: every output written
from them is labelled SYNTHETIC and lands under ``data/synthetic`` and
``results/synthetic``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Study
from .pipeline import RawInputs

ZONE_SHAPE = {  # mean temp offset (F), base load (MW), heat-rate convexity
    "houston": (3.0, 12000, 1.0),
    "north": (-1.0, 16000, 1.2),
    "south": (2.0, 11000, 1.1),
    "west": (-2.0, 7000, 0.6),
}


def make(study: Study, seed: int = 7) -> RawInputs:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(study.start_utc, study.end_utc, freq="h", inclusive="left")
    n = len(hours)
    local = hours.tz_convert(study.market_tz)
    doy = local.dayofyear.to_numpy()
    hod = local.hour.to_numpy()

    # gas: business-day random walk around $3
    bdays = pd.bdate_range(study.start.tz_localize(None) - pd.Timedelta(days=40), study.end.tz_localize(None))
    gas_px = np.clip(3.0 + np.cumsum(rng.normal(0, 0.06, len(bdays))), 1.5, 6.0)
    gas = pd.DataFrame({"trade_date": bdays.date, "price": gas_px})
    gas_by_day = pd.Series(gas_px, index=bdays.normalize()).reindex(
        pd.DatetimeIndex(local.tz_localize(None).normalize()), method="ffill").to_numpy()

    da_rows, rt_rows, load_cols, obs_rows = [], [], {}, []
    for name, zone in study.zones.items():
        dt, base, conv = ZONE_SHAPE.get(name, (0.0, 10000, 1.0))
        temp = (68 + dt + 17 * np.sin(2 * np.pi * (doy - 110) / 365.25)
                + 8 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 3, n))
        load = base * (1 + 0.00035 * (temp - 62) ** 2 + 0.08 * np.sin(2 * np.pi * (hod - 11) / 24))
        load *= 1 + rng.normal(0, 0.02, n)
        load_cols[zone.load_column] = load
        ln = load / load.max()
        heat_rate = 7 + 6 * conv * ln ** 3 + 0.01 * np.clip(temp - 95, 0, None) ** 2
        da = gas_by_day * heat_rate + rng.normal(0, 2, n)
        spike = rng.random(n) < 0.004 * ln ** 4 * 10
        rt_h = da + rng.standard_t(3, n) * 4 + spike * rng.exponential(400, n) - 1.5
        da_rows.append(pd.DataFrame({"interval_start_utc": hours, "location": zone.price_point, "price": da}))
        for q in range(4):
            rt_rows.append(pd.DataFrame({
                "interval_start_utc": hours + pd.Timedelta(minutes=15 * q),
                "location": zone.price_point,
                "price": rt_h + rng.normal(0, 3, n),
            }))
        for st in zone.stations:
            keep = rng.random(n) > 0.02
            obs_rows.append(pd.DataFrame({
                "station": st,
                "valid_utc": (hours - pd.Timedelta(minutes=7))[keep],
                "tmpf": (temp + rng.normal(0, 1.5, n))[keep],
            }))

    load_wide = pd.DataFrame({"interval_start_utc": hours, **load_cols})
    fuel_mix = pd.DataFrame({
        "interval_start_utc": hours,
        "wind": np.clip(rng.normal(12000, 6000, n), 0, None),
        "solar": np.clip(18000 * np.sin(np.pi * (hod - 6) / 13), 0, None),
    })
    return RawInputs(
        da=pd.concat(da_rows, ignore_index=True),
        rt=pd.concat(rt_rows, ignore_index=True),
        load_wide=load_wide,
        station_obs=pd.concat(obs_rows, ignore_index=True),
        gas=gas,
        fuel_mix=fuel_mix,
    )
