"""Exploratory analysis behind the model specification. Pure functions, one per question.

1. What does the DA-RT spread distribution look like?      ``spread_summary``
2. How does load vary with temperature?                     ``temp_load``
3. Is the spread more dispersed in hot (or cold) hours?     ``volatility_by``
4. Do zones differ?                                         ``zone_comparison``
5. Are extreme spreads concentrated in certain hours?       ``extremes``

plus ``effective_n``: hourly observations are serially dependent, so 8,760
hours per zone carry far fewer independent observations than 8,760.

Vocabulary (DECISIONS.md D-24), all in $/MWh levels because prices go negative:

* spread: DA SPP minus the hourly mean of the four 15-min RT SPPs;
* spread magnitude: |spread|;
* conditional spread dispersion: std and quantile ranges of the spread within a group;
* intra-hour RT volatility: std of the four 15-min RT SPPs within an hour;
* tail frequency: share of hours with |spread| above the zone's 99th
  percentile (relative) or above $100/MWh (absolute).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

UTC = "interval_start_utc"
TAIL_Q = 0.99
RAMP_HOURS = ((5, 7), (17, 21))  # hour-beginning spans, inclusive: morning and evening ramps


def in_ramp(hour_beginning: pd.Series) -> pd.Series:
    """True in the morning (HB 5-7) and evening (HB 17-21) ramp hours."""
    return pd.concat([hour_beginning.between(a, b) for a, b in RAMP_HOURS], axis=1).any(axis=1)
ABS_THRESHOLD = 100.0  # $/MWh
EVENT_DAYS = (date(2025, 2, 19), date(2025, 2, 20))  # 2025's episode; use event_days() for any window
TEMP_BINS = [-np.inf, 32, 45, 55, 65, 75, 85, 95, np.inf]
TEMP_LABELS = ["<=32", "32-45", "45-55", "55-65", "65-75", "75-85", "85-95", ">95"]


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["zone", UTC])


def acf(x, lag: int) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    return float(x[:-lag] @ x[lag:] / (x @ x))


def effective_n(x, max_lag: int = 168) -> float:
    """n / (1 + 2 sum_k w_k rho_k) with Bartlett weights: the number of independent
    observations carrying the same information about the mean as this series."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = x @ x
    k = np.arange(1, max_lag + 1)
    rho = np.array([x[:-j] @ x[j:] / v for j in k])
    return float(len(x) / max(1.0, 1 + 2 * np.sum((1 - k / (max_lag + 1)) * rho)))


def effective_n_table(panel: pd.DataFrame, cols=("spread", "abs_spread", "rt_intrahour_std",
                                                  "temp_f", "load_mw", "gas_price")) -> pd.DataFrame:
    rows = []
    for zone, d in _ordered(panel).groupby("zone"):
        row = {"zone": zone, "n": len(d)}
        row.update({f"neff_{c}": round(effective_n(d[c])) for c in cols})
        row["distinct_gas_values"] = int(d["gas_price"].nunique())
        row["operating_days"] = int(d["operating_date"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).set_index("zone")


def event_days(panel: pd.DataFrame, q: float = TAIL_Q) -> tuple:
    """The window's largest extreme-spread episode: the operating day with the most top-1%
    zone-hours, plus whichever adjacent day has more (if any). 2025: Feb 19-20."""
    e = panel.loc[tail_flags(panel, q)["extreme"]]
    counts = e.groupby("operating_date").size()
    if counts.empty:
        return ()
    top = counts.idxmax()
    nb = {d: counts.get(d, 0) for d in (top - pd.Timedelta(days=1), top + pd.Timedelta(days=1))}
    best = max(nb, key=nb.get)
    return tuple(sorted([top, best])) if nb[best] > 0 else (top,)


def event_label(days) -> str:
    days = sorted(days)
    if not days:
        return "no event"
    return f"{days[0]:%Y-%m-%d}" + (f"/{days[-1]:%d}" if len(days) > 1 else "")


def tail_flags(panel: pd.DataFrame, q: float = TAIL_Q) -> pd.DataFrame:
    """``extreme``: |spread| above the zone's q-quantile; ``pos``/``neg`` split it by sign."""
    thr = panel.groupby("zone")["spread"].transform(lambda s: s.abs().quantile(q))
    ext = panel["spread"].abs() > thr
    return pd.DataFrame({"extreme": ext, "extreme_pos": ext & (panel["spread"] > 0),
                         "extreme_neg": ext & (panel["spread"] < 0), "threshold": thr}, index=panel.index)


# --------------------------------------------------------------------------
# Q1 distribution
# --------------------------------------------------------------------------


def spread_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone, d in _ordered(panel).groupby("zone"):
        s = d["spread"]
        a = s.abs().sort_values(ascending=False)
        top = max(1, int(round(len(a) * (1 - TAIL_Q))))
        pre, post = d[d["regime"] == "pre"]["spread"], d[d["regime"] == "post"]["spread"]
        q = s.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        rows.append({
            "zone": zone, "n": len(s), "mean": s.mean(), "median": q[0.5], "std": s.std(),
            "skew": s.skew(), "excess_kurtosis": s.kurt(), "min": s.min(),
            "p0.1": q[0.001], "p1": q[0.01], "p5": q[0.05], "p25": q[0.25], "p75": q[0.75],
            "p95": q[0.95], "p99": q[0.99], "p99.9": q[0.999], "max": s.max(),
            "share_positive": (s > 0).mean(), "share_abs_gt_100": (a > ABS_THRESHOLD).mean(),
            "top1pct_share_of_abs_sum": a.iloc[:top].sum() / a.sum(),
            "mean_excl_top1pct_abs": s[s.abs() <= s.abs().quantile(TAIL_Q)].mean(),
            "acf_lag1": acf(s, 1), "acf_lag24": acf(s, 24), "neff": effective_n(s),
            "std_pre_rtcb": pre.std(), "std_post_rtcb": post.std(),
            "days_post_rtcb": int(d.loc[d["regime"] == "post", "operating_date"].nunique()),
        })
    return pd.DataFrame(rows).set_index("zone")


# --------------------------------------------------------------------------
# Q2 temperature and load
# --------------------------------------------------------------------------


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return float(1 - e.var() / y.var())


def temp_load(panel: pd.DataFrame, branch_f: float = 65.0) -> pd.DataFrame:
    rows = []
    for zone, d in panel.groupby("zone"):
        t, y = d["temp_f"].to_numpy(float), d["load_mw"].to_numpy(float)
        hot, cold = d[d["temp_f"] >= branch_f], d[d["temp_f"] < branch_f]
        cal = np.column_stack([pd.get_dummies(d["hour_beginning"], drop_first=True, dtype=float),
                               pd.get_dummies(d["weekday"], drop_first=True, dtype=float)])
        c2, c1, _ = np.polyfit(t, y, 2)
        rows.append({
            "zone": zone,
            "pearson": d["temp_f"].corr(d["load_mw"]),
            "spearman": d["temp_f"].corr(d["load_mw"], method="spearman"),
            f"pearson_T_ge_{branch_f:.0f}F": hot["temp_f"].corr(hot["load_mw"]),
            f"pearson_T_lt_{branch_f:.0f}F": cold["temp_f"].corr(cold["load_mw"]),
            "r2_T": _r2(y, t), "r2_T_T2": _r2(y, np.column_stack([t, t ** 2])),
            "r2_T_T2_hour_weekday": _r2(y, np.column_stack([t, t ** 2, cal])),
            "quadratic_min_load_at_F": -c1 / (2 * c2) if c2 > 0 else np.nan,
            "corr_T_hour": d["temp_f"].corr(d["hour_beginning"]),
            "load_mw_min": d["load_mw"].min(), "load_mw_max": d["load_mw"].max(),
        })
    return pd.DataFrame(rows).set_index("zone")


def load_by_temperature(panel: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(panel["temp_f"], TEMP_BINS, labels=TEMP_LABELS)
    g = panel.groupby(["zone", b], observed=True)
    return pd.DataFrame({"hours": g.size(), "load_mw_mean": g["load_mw"].mean(),
                         "load_norm_mean": g["load_norm"].mean()})


# --------------------------------------------------------------------------
# Q3 dispersion and tails by condition
# --------------------------------------------------------------------------


def _grouping(panel: pd.DataFrame, by: str) -> pd.Series:
    if by == "temperature":
        return pd.cut(panel["temp_f"], TEMP_BINS, labels=TEMP_LABELS).rename("temp_bin_f")
    if by == "load_decile":  # within zone: each zone's own load distribution
        return panel.groupby("zone")["load_mw"].transform(
            lambda s: pd.qcut(s, 10, labels=False) + 1).rename("zone_load_decile")
    if by == "net_load_decile":  # system-wide, the same for every zone
        return (pd.qcut(panel["system_net_load_mw"], 10, labels=False) + 1).rename("system_net_load_decile")
    if by == "hour":
        return panel["hour_beginning"]
    raise ValueError(f"unknown grouping {by!r}")


def volatility_by(panel: pd.DataFrame, by: str, exclude_days=()) -> pd.DataFrame:
    """Every D-24 measure by ``zone`` x condition. Tail thresholds come from the full sample,
    so excluding days changes the counts, not the definition of 'extreme'."""
    flags = tail_flags(panel)
    grp = _grouping(panel, by)
    p = panel.assign(_g=grp, _ext=flags["extreme"])
    if exclude_days:
        p = p[~p["operating_date"].isin(set(exclude_days))]
    g = p.groupby(["zone", "_g"], observed=True)
    s = g["spread"]
    out = pd.DataFrame({
        "hours": g.size(), "days": g["operating_date"].nunique(),
        "mean_spread": s.mean(), "median_spread": s.median(), "std_spread": s.std(),
        "p1": s.quantile(0.01), "p5": s.quantile(0.05), "p95": s.quantile(0.95), "p99": s.quantile(0.99),
        "median_abs_spread": g["abs_spread"].median(), "p95_abs_spread": g["abs_spread"].quantile(0.95),
        "mean_intrahour_rt_std": g["rt_intrahour_std"].mean(),
        "median_intrahour_rt_std": g["rt_intrahour_std"].median(),
        "tail_freq_rel_pct": 100 * g["_ext"].mean(),
        "tail_freq_abs_pct": 100 * g["abs_spread"].apply(lambda x: (x > ABS_THRESHOLD).mean()),
    })
    out.index = out.index.set_names(["zone", grp.name or by])
    return out


# --------------------------------------------------------------------------
# Q4 zones
# --------------------------------------------------------------------------


def zone_comparison(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    w = panel.pivot(index=UTC, columns="zone", values="spread")
    flags = tail_flags(panel)
    ext = panel.assign(_e=flags["extreme"]).pivot(index=UTC, columns="zone", values="_e")
    k = ext.sum(axis=1)
    k = k[k > 0]
    levels = panel.groupby(["zone", "on_peak"])[["da_price", "rt_price", "spread", "imhr_da", "imhr_rt"]].mean()
    return {
        "spread_corr_pearson": w.corr(),
        "spread_corr_spearman": w.corr(method="spearman"),
        "imhr_da_corr": panel.pivot(index=UTC, columns="zone", values="imhr_da").corr(),
        "levels_by_peak": levels.unstack("on_peak"),
        "extreme_cooccurrence": k.value_counts().sort_index().rename_axis("zones_extreme_same_hour")
                                  .rename("hours").to_frame(),
    }


# --------------------------------------------------------------------------
# Q5 timing of extremes
# --------------------------------------------------------------------------


def extremes(panel: pd.DataFrame, q: float = TAIL_Q) -> dict[str, pd.DataFrame]:
    f = tail_flags(panel, q)
    e = panel.assign(**f)[lambda d: d["extreme"]]
    sign = np.where(e["spread"] > 0, "DA_above_RT", "RT_above_DA")
    by_hour = (e.assign(sign=sign).groupby(["hour_beginning", "sign"]).size().unstack(fill_value=0)
               .reindex(range(24), fill_value=0))
    by_hour_zone = (e.groupby(["hour_beginning", "zone"]).size().unstack(fill_value=0)
                    .reindex(range(24), fill_value=0))
    by_month = e.assign(sign=sign).groupby(["month", "sign"]).size().unstack(fill_value=0)
    by_day = (e.groupby("operating_date")
               .agg(zone_hours=("zone", "size"), zones=("zone", lambda z: ",".join(sorted(set(z)))),
                    min_spread=("spread", "min"), max_spread=("spread", "max"))
               .sort_values("zone_hours", ascending=False))
    ramps = in_ramp(e["hour_beginning"])
    ramp_hours_share = in_ramp(panel["hour_beginning"]).mean()
    summary = pd.DataFrame({
        "extreme_zone_hours": [len(e)],
        "thresholds": [";".join(f"{z}:{t:.1f}" for z, t in
                                panel.assign(t=f["threshold"]).groupby("zone")["t"].first().items())],
        "share_RT_above_DA": [(e["spread"] < 0).mean()],
        "distinct_days": [e["operating_date"].nunique()],
        "share_on_top5_days": [by_day["zone_hours"].head(5).sum() / len(e)],
        "share_in_ramp_hours": [ramps.mean()],
        "ramp_hours_share_of_all_hours": [ramp_hours_share],
        "share_DA_above_RT_in_hb17_21": [(e.loc[e["spread"] > 0, "hour_beginning"].between(17, 21)).mean()],
    })
    return {"summary": summary, "by_hour": by_hour, "by_hour_zone": by_hour_zone,
            "by_month": by_month, "by_day": by_day}


def spread_by_hour_quantiles(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby(["zone", "hour_beginning"])["spread"]
    return pd.DataFrame({f"p{int(q * 100)}": g.quantile(q) for q in (0.01, 0.05, 0.5, 0.95, 0.99)})


# --------------------------------------------------------------------------
# Q6. What makes an extreme hour different? Tightness, surprise, one event.
# All descriptive associations: high net load co-occurs with season, weather,
# renewable output and outages, none of which is held fixed here (D-35).
# --------------------------------------------------------------------------

TOP_DECILE = 10


def net_load_deciles(panel: pd.DataFrame) -> pd.DataFrame:
    """System net-load (load - wind - solar) deciles per row.

    ``nl_decile`` ranks over all hours, the convention of the q3 table.
    ``nl_decile_within_hour`` ranks within each hour of day, so every decile holds
    the same mix of hours: a time-of-day robustness check, not a control for
    season, weather or renewable conditions.
    """
    nl = panel["system_net_load_mw"]
    if nl.isna().any():
        raise ValueError("system_net_load_mw has missing values; deciles would be undefined")
    within = panel.groupby("hour_beginning")["system_net_load_mw"].transform(
        lambda s: pd.qcut(s, 10, labels=False) + 1)
    return pd.DataFrame({"nl_decile": pd.qcut(nl, 10, labels=False) + 1,
                         "nl_decile_within_hour": within}, index=panel.index)


def _top_vs_rest(event: pd.Series, top: pd.Series, day: pd.Series, n_boot: int, rng) -> dict:
    """Event rate in ``top`` hours vs the rest, their ratio, and a 90% day-block bootstrap
    interval for the ratio (operating days resampled with replacement, since extremes
    cluster on a few days). The ratio is NaN, never infinite, when the rest-rate is zero."""
    g = pd.DataFrame({"n_top": top, "x_top": event & top, "n_rest": ~top, "x_rest": event & ~top,
                      "day": day.to_numpy()}).groupby("day").sum()
    c = g[["n_top", "x_top", "n_rest", "x_rest"]].to_numpy(float)

    def ratio(t):
        num = np.divide(t[..., 1], t[..., 0], out=np.full(t.shape[:-1], np.nan), where=t[..., 0] > 0)
        den = np.divide(t[..., 3], t[..., 2], out=np.full(t.shape[:-1], np.nan), where=t[..., 2] > 0)
        return np.divide(num, den, out=np.full(t.shape[:-1], np.nan), where=den > 0)

    tot = c.sum(axis=0)
    boot = ratio(c[rng.integers(0, len(c), (n_boot, len(c)))].sum(axis=1)) if n_boot else np.array([np.nan])
    ok = boot[np.isfinite(boot)]
    events = int(event.sum())
    return {"hours_top": int(tot[0]), "hours_rest": int(tot[2]), "events": events,
            "event_days": int(day[event].nunique()),
            "rate_top_pct": 100 * tot[1] / tot[0] if tot[0] else np.nan,
            "rate_rest_pct": 100 * tot[3] / tot[2] if tot[2] else np.nan,
            "ratio_top_vs_rest": float(ratio(tot)),
            "ratio_ci05": float(np.percentile(ok, 5)) if len(ok) else np.nan,
            "ratio_ci95": float(np.percentile(ok, 95)) if len(ok) else np.nan,
            "boot_draws_defined": int(len(ok)),
            "share_of_events_in_top": tot[1] / events if events else np.nan}


def _by_zone_and_pooled(panel: pd.DataFrame):
    yield "all zones", panel.index
    for zone, d in panel.groupby("zone"):
        yield zone, d.index


def tail_by_net_load(panel: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zone-relative extremes (top 1% |spread| per zone) by system net-load decile.

    Returns (per-decile rows, top-decile-vs-rest summary), for both decile kinds,
    with and without the window's storm days, per zone and pooled. "Extreme" is
    relative: every zone's base rate is 1% by construction, so ratios are relative
    risks, not dollar exposures (see ``tail_by_net_load_fixed``).
    """
    rng = np.random.default_rng(seed)
    ext = tail_flags(panel)["extreme"]
    dec = net_load_deciles(panel)
    storm = panel["operating_date"].isin(set(event_days(panel)))
    rows, summ = [], []
    for kind in ("nl_decile", "nl_decile_within_hour"):
        everything = pd.Series(True, index=panel.index)
        for sample, keep in (("all days", everything), (f"excl. {event_label(event_days(panel))}", ~storm)):
            d = panel[keep]
            for zone, idx in _by_zone_and_pooled(d):
                e, k = ext.loc[idx], dec.loc[idx, kind]
                base = e.mean()
                by = pd.DataFrame({"e": e, "k": k}).groupby("k")["e"].agg(["size", "sum"])
                for decile, r in by.iterrows():
                    rows.append({"decile_kind": kind, "sample": sample, "zone": zone, "decile": int(decile),
                                 "hours": int(r["size"]), "extreme_hours": int(r["sum"]),
                                 "rate_pct": 100 * r["sum"] / r["size"], "base_rate_pct": 100 * base,
                                 "lift_vs_base": (r["sum"] / r["size"]) / base if base else np.nan,
                                 "share_of_extremes": r["sum"] / e.sum() if e.sum() else np.nan})
                summ.append({"decile_kind": kind, "sample": sample, "zone": zone,
                             **_top_vs_rest(e, k == TOP_DECILE, d.loc[idx, "operating_date"], n_boot, rng)})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def tail_by_net_load_fixed(panel: pd.DataFrame, thresholds=(50, 100, 250), n_boot: int = 2000,
                           seed: int = 0) -> pd.DataFrame:
    """Fixed-dollar sensitivity for the net-load concentration: for each threshold X,
    hours with |spread| > X and hours with RT above DA by more than X (spread < -X, the
    per-MWh cost of an unhedged real-time shortfall), top net-load decile vs the rest."""
    rng = np.random.default_rng(seed)
    top = net_load_deciles(panel)["nl_decile"] == TOP_DECILE
    rows = []
    for x in thresholds:
        for definition, event in ((f"|spread| > {x}", panel["spread"].abs() > x),
                                  (f"RT above DA by > {x}", panel["spread"] < -x)):
            for zone, idx in _by_zone_and_pooled(panel):
                rows.append({"threshold": x, "definition": definition, "zone": zone,
                             "base_rate_pct": 100 * event.loc[idx].mean(),
                             **_top_vs_rest(event.loc[idx], top.loc[idx], panel.loc[idx, "operating_date"],
                                            n_boot, rng)})
    return pd.DataFrame(rows)


PROFILE_VARS = {  # column -> (label, divisor)
    "system_net_load_mw": ("System net load (GW)", 1000),
    "net_load_fc": ("Net load expected at 09:45 D−1 (GW)", 1000),
    "net_load_err_avail": ("Net-load surprise, + = tighter (GW)", 1000),
    "wind_err_hsl": ("Wind availability surprise (GW)", 1000),
    "solar_err_hsl": ("Solar availability surprise (GW)", 1000),
    "load_err": ("Zone load surprise (GW)", 1000),
    "net_load_ramp": ("Net-load change from previous hour (GW)", 1000),
    "temp_f": ("Zone temperature (°F)", 1),
    "rt_intrahour_std": ("Std of the four 15-min RT prices ($/MWh)", 1),
}


def extreme_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Ordinary hours vs zone-relative extremes split by sign: medians and quartiles of the
    conditions in PROFILE_VARS, plus the share in the top net-load decile and in ramp hours."""
    f = tail_flags(panel)
    group = pd.Series("ordinary", index=panel.index)
    group[f["extreme"] & (panel["spread"] < 0)] = "RT_above_DA_extreme"
    group[f["extreme"] & (panel["spread"] > 0)] = "DA_above_RT_extreme"
    top = net_load_deciles(panel)["nl_decile"] == TOP_DECILE
    rows = []
    for g in ("ordinary", "RT_above_DA_extreme", "DA_above_RT_extreme"):
        m = group == g
        d = panel[m]
        for col, (label, div) in PROFILE_VARS.items():
            v = d[col] / div
            rows.append({"group": g, "variable": col, "label": label, "median": v.median(),
                         "p25": v.quantile(0.25), "p75": v.quantile(0.75)})
        for col, label, v in (("share_top_decile", "Share in top net-load decile", top[m].mean()),
                              ("share_ramp_hours", "Share in ramp hours (HB 5–7, 17–21)", in_ramp(d["hour_beginning"]).mean()),
                              ("zone_hours", "Zone-hours", float(m.sum())),
                              ("days", "Operating days", float(d["operating_date"].nunique()))):
            rows.append({"group": g, "variable": col, "label": label, "median": v, "p25": np.nan, "p75": np.nan})
    return pd.DataFrame(rows)


def representative_event_day(panel: pd.DataFrame):
    """Reproducible event-selection rule (D-35): the operating day with the most RT-above-DA
    zone-relative extremes in the top system net-load decile, outside the window's storm days.
    The rule was settled after candidate events had been examined, so the day it returns is a
    retrospectively selected illustration, not an out-of-sample test."""
    f = tail_flags(panel)
    top = net_load_deciles(panel)["nl_decile"] == TOP_DECILE
    hit = f["extreme"] & (panel["spread"] < 0) & top & ~panel["operating_date"].isin(set(event_days(panel)))
    counts = panel.loc[hit, "operating_date"].value_counts()
    if counts.empty:
        raise ValueError("no RT-above-DA extreme in the top net-load decile outside the storm days")
    return counts.index[0]


def event_window(panel: pd.DataFrame, rt15: pd.DataFrame, day) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One operating day: (hourly frame, 15-minute RT frame).

    ``rt15`` is the pipeline's (interval_start_utc, location, price) RT series. The
    hourly frame carries, per zone, DA, the panel's hourly RT, load and temperature;
    and two system net-load measures, kept distinct (D-35):

    * tightness: ``system_net_load_mw`` = load - wind and solar *generation*, the measure
      ranked by ``net_load_deciles``, with that ranking's top-decile threshold;
    * surprise: ``net_load_avail`` = load - wind and solar *available capacity* (forecast +
      availability surprise), compared with ``net_load_fc`` expected at 09:45 D-1, so
      actual - expected is the curtailment-free surprise used throughout.

    Wind and solar available capacity vs forecast are included as well.
    """
    d = panel[panel["operating_date"] == day].copy()
    if d.empty:
        raise ValueError(f"{day} is not in the panel")
    d["extreme"] = tail_flags(panel).loc[d.index, "extreme"]
    d["wind_avail"] = d["wind_fc"] + d["wind_err_hsl"]
    d["solar_avail"] = d["solar_fc"] + d["solar_err_hsl"]
    d["net_load_avail"] = d["net_load_fc"] + d["net_load_err_avail"]
    # the Fig.-4 tightness measure is generation-based; carry its top-decile threshold
    dec = net_load_deciles(panel)["nl_decile"]
    d["nl_top_decile_threshold"] = panel.loc[dec == TOP_DECILE, "system_net_load_mw"].min()
    cols = ["interval_start_utc", "interval_start_local", "hour_beginning", "zone", "location", "da_price",
            "rt_price", "spread", "extreme", "load_mw", "load_fc_mw", "temp_f", "system_net_load_mw",
            "net_load_fc", "net_load_avail", "net_load_err_avail", "wind_fc", "wind_avail", "solar_fc",
            "solar_avail", "sys_load_fc", "nl_top_decile_threshold"]
    hourly = d[cols].sort_values(["zone", UTC]).reset_index(drop=True)
    lo = d[UTC].min()
    hi = d[UTC].max() + pd.Timedelta(hours=1)
    loc_zone = dict(zip(d["location"], d["zone"]))
    r = rt15[(rt15[UTC] >= lo) & (rt15[UTC] < hi) & rt15["location"].isin(loc_zone)].copy()
    r["zone"] = r["location"].map(loc_zone)
    tz = d["interval_start_local"].dt.tz
    r["interval_start_local"] = r[UTC].dt.tz_convert(tz)
    rt = r.sort_values(["zone", UTC]).reset_index(drop=True)[[UTC, "interval_start_local", "zone", "location", "price"]]
    return hourly, rt


def temp_load_forward(panel: pd.DataFrame, min_train: int = 3) -> pd.DataFrame:
    """Does the T² term in the temperature-load fit carry forward? For each zone, on the
    report's expanding monthly folds, fit load ~ T and load ~ T + T² on training months only
    and compare pooled squared error in the test months (realised test-hour temperature, so
    this is explanatory validation, not a load forecast)."""
    from .rsm import time_folds

    rows = []
    for zone, d in panel.groupby("zone"):
        d = d.sort_values(UTC)
        t, y = d["temp_f"].to_numpy(float), d["load_mw"].to_numpy(float)
        sse = {1: 0.0, 2: 0.0}
        n = 0
        for train, test, _ in time_folds(d["month"], "forward", min_train):
            for deg in (1, 2):
                c = np.polyfit(t[train], y[train], deg)
                sse[deg] += float(((y[test] - np.polyval(c, t[test])) ** 2).sum())
            n += int(test.sum())
        rows.append({"zone": zone, "test_hours": n, "sse_linear": sse[1], "sse_quadratic": sse[2],
                     "mse_reduction_quadratic_vs_linear": 1 - sse[2] / sse[1] if n and sse[1] > 0 else np.nan})
    return pd.DataFrame(rows).set_index("zone")


def zonal_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """The report's zonal summary: load measures, representative DA-RT spreads and DA IMHR.

    One row per zone. Load factor is mean / peak hourly load; the load-temperature
    R² is the T + T² fit of ``temp_load``, so the two tables cannot disagree. On-
    and off-peak follow the panel's ``on_peak`` flag (weekday HE07-HE22, excl. NERC
    holidays). Spreads are untrimmed; the 1st/99th percentiles show the tails.
    """
    tl = temp_load(panel)
    rows = {}
    for zone, d in panel.groupby("zone"):
        on = d["on_peak"].astype(bool)
        s = d["spread"]
        rows[zone] = {
            "hours": len(d),
            "load_mean_mw": d["load_mw"].mean(),
            "load_peak_mw": d["load_mw"].max(),
            "load_factor": d["load_mw"].mean() / d["load_mw"].max(),
            "load_r2_T_T2": tl.loc[zone, "r2_T_T2"],
            "spread_mean": s.mean(),
            "spread_median": s.median(),
            "spread_onpeak_mean": s[on].mean(),
            "spread_offpeak_mean": s[~on].mean(),
            "abs_spread_mean": d["abs_spread"].mean(),
            "spread_p1": s.quantile(0.01),
            "spread_p99": s.quantile(0.99),
            "share_DA_above_RT": (s > 0).mean(),
            "imhr_da_mean": d["imhr_da"].mean(),
            "imhr_rt_mean": d["imhr_rt"].mean(),
            "imhr_da_onpeak_mean": d.loc[on, "imhr_da"].mean(),
            "imhr_da_offpeak_mean": d.loc[~on, "imhr_da"].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("zone")


def tail_vs_normal(panel: pd.DataFrame, threshold: float = 170.0) -> pd.DataFrame:
    """Observed hours with |spread| > threshold against the count a normal distribution
    with the same mean and SD would expect, pooled over zones and per zone (Fig. 1)."""
    from scipy.stats import norm

    groups = [("all zones", panel)] + list(panel.groupby("zone"))
    rows = []
    for name, d in groups:
        s = d["spread"]
        mu, sd = s.mean(), s.std()
        rows.append({"group": name, "hours": len(s), "mean": mu, "std": sd, "threshold": threshold,
                     "observed_beyond": int((s.abs() > threshold).sum()),
                     "normal_expected_beyond": len(s) * (norm.sf(threshold, mu, sd) + norm.cdf(-threshold, mu, sd))})
    return pd.DataFrame(rows).set_index("group")


def run(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every table, keyed by the file name it is written under."""
    zc = zone_comparison(panel)
    ex = extremes(panel)
    tables = {
        "q1_spread_summary": spread_summary(panel),
        "q2_temp_load": temp_load(panel),
        "q2_load_by_temperature": load_by_temperature(panel),
        "q3_by_temperature": volatility_by(panel, "temperature"),
        "q3_by_temperature_excl_event": volatility_by(panel, "temperature", event_days(panel)),
        "q3_by_zone_load_decile": volatility_by(panel, "load_decile"),
        "q3_by_system_net_load_decile": volatility_by(panel, "net_load_decile"),
        "q5_spread_quantiles_by_hour": spread_by_hour_quantiles(panel),
        "dependence_effective_n": effective_n_table(panel),
    }
    tables.update({f"q4_{k}": v for k, v in zc.items()})
    tables.update({f"q5_extremes_{k}": v for k, v in ex.items()})
    return tables
