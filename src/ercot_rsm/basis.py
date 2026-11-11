"""Hub-relative basis decomposition of the zonal DA-RT spread (DECISIONS.md D-32).

For each zone and hour, with basis = load-zone SPP - reference hub SPP:

    spread_LZ = spread_HUB + (basis_DA - basis_RT)

The identity is exact. ``spread_HUB`` is the system-wide part of the DA-to-RT
surprise as seen at the reference hub; ``basis_DA - basis_RT`` (the "basis
surprise") is how much the zone's price moved *relative to that hub* between
DA and RT: a relative congestion surprise. It is not total congestion and not
an attribution to constraints. The primary reference is ``HB_HUBAVG``, which
contains ``HB_WEST`` with one-quarter weight and so dampens West; the mean of
the Houston, North and South hubs is the ex-West sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Study
from .data import ercot_archive

UTC = "interval_start_utc"
HUBS = ["HB_HUBAVG", "HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_PAN"]
EX_WEST = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH"]


def hub_prices(study: Study) -> pd.DataFrame:
    """Hourly DA and RT hub prices from the settled archive; RT hour = mean of 4 intervals."""
    arch = study.raw_dir / "ercot"
    da = pd.read_parquet(arch / "da_prices_archive.parquet")
    rt = pd.read_parquet(arch / "rt_prices_archive.parquet")
    da = da[da["location"].isin(HUBS)][[UTC, "location", "price"]]
    rt = rt[rt["location"].isin(HUBS)]
    g = rt.assign(**{UTC: rt[UTC].dt.floor("h")}).groupby([UTC, "location"])["price"]
    rth = g.agg(rt="mean", n="count").reset_index()
    if (rth["n"] != 4).any():
        raise ValueError(f"{int((rth['n'] != 4).sum())} incomplete RT hub hours")
    out = da.rename(columns={"price": "da"}).merge(rth[[UTC, "location", "rt"]], on=[UTC, "location"],
                                                   how="inner", validate="one_to_one")
    wide = out.pivot(index=UTC, columns="location", values=["da", "rt"])
    wide.columns = [f"{m}_{loc.removeprefix('HB_')}" for m, loc in wide.columns]
    for m in ("da", "rt"):
        wide[f"{m}_EXWEST"] = wide[[f"{m}_{h.removeprefix('HB_')}" for h in EX_WEST]].mean(axis=1)
    return wide.reset_index()


def decompose(panel: pd.DataFrame, hubs: pd.DataFrame, ref: str = "HUBAVG") -> pd.DataFrame:
    p = panel[[UTC, "zone", "operating_date", "hour_beginning", "da_price", "rt_price", "spread", "load_mw",
               *[c for c in ("load_norm", "wind_err_west", "net_load_err_avail", "extreme") if c in panel]]].merge(
        hubs[[UTC, f"da_{ref}", f"rt_{ref}"]], on=UTC, how="left", validate="many_to_one")
    p["spread_hub"] = p[f"da_{ref}"] - p[f"rt_{ref}"]
    p["basis_da"] = p["da_price"] - p[f"da_{ref}"]
    p["basis_rt"] = p["rt_price"] - p[f"rt_{ref}"]
    p["basis_surprise"] = p["basis_da"] - p["basis_rt"]
    gap = (p["spread"] - p["spread_hub"] - p["basis_surprise"]).abs().max()
    if not gap < 1e-6:
        raise AssertionError(f"decomposition identity fails by {gap}")
    return p.assign(reference=ref)


def variance_shares(dec: pd.DataFrame) -> pd.DataFrame:
    """Var(spread) = Var(hub) + Var(basis surprise) + 2 Cov: each as a share, per zone; and
    for the zone's top-1% |spread| hours, how often the basis surprise is the larger term."""
    rows = []
    for zone, d in dec.groupby("zone"):
        v = d["spread"].var()
        row = {"zone": zone, "reference": d["reference"].iloc[0],
               "share_var_hub": d["spread_hub"].var() / v, "share_var_basis": d["basis_surprise"].var() / v,
               "share_2cov": 2 * d["spread_hub"].cov(d["basis_surprise"]) / v,
               # covariance ("beta") shares: Cov(term, spread)/Var(spread); they sum to exactly 1
               "cov_share_hub": d["spread_hub"].cov(d["spread"]) / v,
               "cov_share_basis": d["basis_surprise"].cov(d["spread"]) / v,
               "median_abs_basis_surprise": d["basis_surprise"].abs().median(),
               "p95_abs_basis_surprise": d["basis_surprise"].abs().quantile(0.95),
               "share_hours_abs_basis_gt_20": (d["basis_surprise"].abs() > 20).mean()}
        if "extreme" in d:
            e = d[d["extreme"]]
            row["extreme_hours"] = len(e)
            row["extreme_share_basis_dominant"] = (e["basis_surprise"].abs() > e["spread_hub"].abs()).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def west_wind_availability(study: Study) -> pd.DataFrame:
    """West + Panhandle wind: DA-time forecast (09:45 vintage), realised generation, and the
    real-time COP high sustained limit (availability as resource owners reported it),
    so curtailment caused by congestion is not mistaken for a wind surprise."""
    from .data.gridstatus_fetch import all_specs, raw_dir

    specs = all_specs(study)
    fc = pd.read_parquet(raw_dir(study) / f"{specs['wind_fc'].name}.parquet")
    act = pd.read_parquet(raw_dir(study) / f"{specs['wind_act'].name}.parquet")
    regs = ["west", "panhandle"]
    f = pd.DataFrame({UTC: pd.to_datetime(fc[UTC], utc=True), "wind_fc_wp": sum(fc[f"stwpf_{r}"] for r in regs)})
    a = pd.DataFrame({UTC: pd.to_datetime(act[UTC], utc=True), "wind_gen_wp": sum(act[f"gen_{r}"] for r in regs),
                      "wind_hsl_wp": sum(act[f"cop_hsl_{r}"] for r in regs)})
    out = f.merge(a, on=UTC, how="inner")
    out["wind_err_wp_gen"] = out["wind_gen_wp"] - out["wind_fc_wp"]
    out["wind_err_wp_avail"] = out["wind_hsl_wp"] - out["wind_fc_wp"]
    out["curtailment_proxy_wp"] = (out["wind_hsl_wp"] - out["wind_gen_wp"]).clip(lower=0)
    return out


def west_profile(dec: pd.DataFrame, wind: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """West's basis surprise by West/Panhandle wind availability surprise decile, by DA-time
    West wind forecast decile, and by zonal load decile."""
    w = dec[dec["zone"] == "west"].merge(wind, on=UTC, how="left")
    out = {}
    for name, col in (("by_wind_avail_surprise_decile", "wind_err_wp_avail"),
                      ("by_wind_forecast_decile", "wind_fc_wp"), ("by_load_decile", "load_mw")):
        dec_ = pd.qcut(w[col], 10, labels=False) + 1
        g = w.groupby(dec_)
        out[name] = pd.DataFrame({
            "hours": g.size(), f"{col}_mean": g[col].mean(),
            "mean_spread": g["spread"].mean(), "mean_spread_hub": g["spread_hub"].mean(),
            "mean_basis_surprise": g["basis_surprise"].mean(),
            "median_abs_basis_surprise": g["basis_surprise"].apply(lambda s: s.abs().median()),
            "p95_abs_basis_surprise": g["basis_surprise"].apply(lambda s: s.abs().quantile(0.95)),
            "mean_curtailment_proxy_mw": g["curtailment_proxy_wp"].mean(),
        }).rename_axis(f"{col}_decile")
    return out


def run(panel: pd.DataFrame, study: Study, out_dir, fig_dir) -> dict[str, pd.DataFrame]:
    from . import eda, plots

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    hubs = hub_prices(study)
    panel = panel.assign(extreme=eda.tail_flags(panel)["extreme"])
    tables = {}
    decs = {}
    for ref in ("HUBAVG", "EXWEST"):
        d = decompose(panel, hubs, ref)
        decs[ref] = d
        tables[f"variance_shares_{ref.lower()}"] = variance_shares(d)
    wind = west_wind_availability(study)
    for k, t in west_profile(decs["HUBAVG"], wind).items():
        tables[f"west_{k}"] = t
    for k, t in tables.items():
        t.to_csv(out_dir / f"{k}.csv")
    plots.basis_decomposition(tables["variance_shares_hubavg"], tables["west_by_wind_avail_surprise_decile"],
                              tables["west_by_load_decile"], fig_dir / "basis_decomposition.png")
    return tables
