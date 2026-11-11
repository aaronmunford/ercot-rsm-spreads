"""Fit the response-surface model ladder for every zone and write tables and figures.

Models (DECISIONS.md D-11, D-25):

* M0  hour-of-day mean of the response, the climatology benchmark
* M1  linear in temperature, actual load and gas (+ regime nuisance block)
* M2  full second-order RSM in the same factors (the primary model)
* M3  M2 + an hour-of-day block, so ramp structure is not credited to weather and load
* ALT M2 with the load anomaly in place of actual load (collinearity comparison)

The primary spread response is not winsorised (D-14): its tails are the
object of interest. Winsorised and asinh fits appear only as sensitivities.
All factors are realised, contemporaneous values, so every result here is
explanatory. The forward-CV numbers are honest in time but are not a trading
backtest (D-25).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import eda
from . import features as F
from . import plots, rsm
from .config import Study

SPEC_LABELS = {
    "M0": "M0 hour-of-day mean",
    "M1": "M1 linear (T, L, G)",
    "M2": "M2 quadratic RSM (T, L, G)",
    "M3": "M3 quadratic RSM + hour block",
    "ALT": "ALT quadratic RSM (T, load anomaly, G)",
}


# --------------------------------------------------------------------------
# Single fits (kept for the notebooks and tests)
# --------------------------------------------------------------------------


def prepare(panel: pd.DataFrame, zone: str, response: str, cfg: dict) -> pd.DataFrame:
    """One zone, time-ordered, estimation sample only; winsorised only if configured."""
    d = panel[panel["zone"] == zone].sort_values(F.UTC)
    if "model_ok" in d:
        d = d[d["model_ok"]]
    d = d.copy()
    lo, hi = cfg.get("winsorize", [0.0, 1.0])
    if (lo, hi) != (0.0, 1.0):
        d[response] = F.winsorize(d[response], lo, hi)
    return d


def _blocks(d: pd.DataFrame, cols) -> list[str] | None:
    keep = [b for b in (cols or []) if d[b].nunique() > 1]
    return keep or None


def fit_zone(panel: pd.DataFrame, zone: str, response: str, cfg: dict, **override) -> rsm.RSMFit:
    d = prepare(panel, zone, response, cfg)
    kw = dict(
        order=cfg.get("order", 2),
        coding_method=cfg.get("coding", "range"),
        coding_quantiles=tuple(cfg.get("coding_quantiles", (0.01, 0.99))),
        block_cols=_blocks(d, cfg.get("block_cols", [])),
        cov_type=cfg.get("cov_type", "HAC"),
        hac_maxlags=cfg.get("hac_maxlags", 24),
    )
    kw.update(override)
    factors = kw.pop("factors", cfg["factors"])
    return rsm.fit_rsm(d, response, factors, **kw)


def canonical_row(zone: str, fit: rsm.RSMFit) -> dict:
    can = rsm.canonical_analysis(fit)
    row = {"zone": zone, "response": fit.response, "n": len(fit.y),
           "r2": fit.ols.rsquared, "kind": can.kind, "inside": can.inside_coded_cube,
           "y_stationary": can.predicted_at_stationary}
    for name in fit.names:
        row[f"{name}_stationary"] = can.stationary_natural[name]
    for i, lam in enumerate(can.eigenvalues, start=1):
        row[f"lambda_{i}"] = lam
    return row


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def specs(cfg: dict) -> dict[str, dict]:
    base = list(cfg.get("block_cols", []))
    hour = cfg.get("hour_block", "hour_beginning")
    return {
        "M1": {"factors": cfg["factors"], "order": 1, "blocks": base},
        "M2": {"factors": cfg["factors"], "order": 2, "blocks": base},
        "M3": {"factors": cfg["factors"], "order": 2, "blocks": base + [hour]},
        "ALT": {"factors": cfg.get("alt_factors", cfg["factors"]), "order": 2, "blocks": base},
    }


def fit_spec(d: pd.DataFrame, response: str, spec: dict, cfg: dict, **kw) -> rsm.RSMFit:
    args = dict(order=spec["order"], coding_method=cfg.get("coding", "quantile"),
                coding_quantiles=tuple(cfg.get("coding_quantiles", (0.01, 0.99))),
                block_cols=_blocks(d, spec["blocks"]), cov_type=cfg.get("cov_type", "HAC"),
                hac_maxlags=cfg.get("hac_maxlags", 48), terms=spec.get("terms"))
    args.update(kw)
    return rsm.fit_rsm(d, response, list(spec["factors"]), **args)


def curvature_test(fit: rsm.RSMFit) -> dict:
    """Joint Wald test (fit's covariance) that every second-order term is zero."""
    cols = list(fit.X.columns)
    members = [c for c in cols if fit.terms[c] in ("quadratic", "lin x lin")]
    if not members:
        return {"curv_df": 0, "curv_chi2": np.nan, "curv_p": np.nan}
    R = np.zeros((len(members), len(cols)))
    for r, c in enumerate(members):
        R[r, cols.index(c)] = 1.0
    w = fit.ols.wald_test(R, scalar=True, use_f=False)
    return {"curv_df": len(members), "curv_chi2": float(w.statistic), "curv_p": float(w.pvalue)}


def evaluate(d: pd.DataFrame, response: str, spec_map: dict, cfg: dict, scheme: str,
             extreme: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample errors for M0 and every spec under ``scheme`` (forward | lomo).

    Returns (metrics per model, row-level predictions). Each fold refits on
    its own training months only, coding included. ``extreme`` flags the
    zone's top-1% |spread| hours, a fixed evaluation subset.
    """
    preds = []
    folds = rsm.time_folds(d["month"], scheme, cfg.get("cv_min_train_months", 3))
    if not folds:
        print(f"  {scheme} CV skipped: {d['month'].nunique()} months is too short a history")
        return pd.DataFrame(), pd.DataFrame()
    for train, test, month in folds:
        tr, te = d[train], d[test]
        out = pd.DataFrame({F.UTC: te[F.UTC].to_numpy(), "month": month, "y": te[response].to_numpy(),
                            "extreme": extreme.loc[te.index].to_numpy(),
                            "train_mean": tr[response].mean()})
        out["M0"] = te["hour_beginning"].map(tr.groupby("hour_beginning")[response].mean()).to_numpy()
        for name, spec in spec_map.items():
            out[name] = fit_spec(tr, response, spec, cfg, cov_type="nonrobust").predict_frame(te)
        preds.append(out)
    p = pd.concat(preds, ignore_index=True)
    rows = []
    sse0 = ((p["y"] - p["M0"]) ** 2).sum()
    ssem = ((p["y"] - p["train_mean"]) ** 2).sum()
    for name in ["M0", *spec_map]:
        e = p["y"] - p[name]
        ext = p["extreme"]
        rows.append({"model": name, "scheme": scheme, "n": len(p), "months": p["month"].nunique(),
                     "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()), "median_ae": e.abs().median(),
                     "r2_vs_train_mean": 1 - (e ** 2).sum() / ssem, "skill_vs_M0": 1 - (e ** 2).sum() / sse0,
                     "mae_extreme_hours": e[ext].abs().mean(), "n_extreme_hours": int(ext.sum()),
                     "mae_other_hours": e[~ext].abs().mean()})
    return pd.DataFrame(rows), p


def reference_conditions(d: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Representative observed conditions at which every variant's prediction is compared:
    mean factor values over the coldest 1%, the mild 60-70 F band, and the hottest 1% of hours."""
    t = d["temp_f"]
    groups = {"cold_1pct": t <= t.quantile(0.01), "mild_60_70F": t.between(60, 70),
              "hot_1pct": t >= t.quantile(0.99)}
    cols = list(dict.fromkeys(["temp_f", "load_mw", "load_anom", "gas_price", *factors]))
    return pd.DataFrame({k: d.loc[m, cols].mean() for k, m in groups.items()}).T.assign(
        hours=[int(m.sum()) for m in groups.values()])


def sensitivity(d: pd.DataFrame, response: str, spec_map: dict, cfg: dict, refs: pd.DataFrame,
                event=eda.EVENT_DAYS) -> pd.DataFrame:
    """Predicted response at the reference conditions under each variant (block-averaged)."""
    rows = []

    def add(label: str, fit: rsm.RSMFit, back=lambda v: v, n=None):
        pred = fit.predict(refs[list(fit.names)].to_numpy(float))
        row = {"variant": label, "n": n or len(fit.y), "r2": fit.ols.rsquared, **curvature_test(fit)}
        row.update({f"pred_{k}": back(v) for k, v in zip(refs.index, pred)})
        rows.append(row)

    m2 = spec_map["M2"]
    add("M2 raw (primary)", fit_spec(d, response, m2, cfg))
    if response == "spread":
        lo, hi = cfg.get("sensitivity_winsorize", [0.005, 0.995])
        add(f"M2 winsorised {lo:.1%}/{hi:.1%}", fit_spec(d.assign(**{response: F.winsorize(d[response], lo, hi)}),
                                                         response, m2, cfg))
        s = cfg.get("asinh_scale", 10.0)
        add(f"M2 asinh(spread/{s:g}), back-transformed",
            fit_spec(d.assign(**{response: np.arcsinh(d[response] / s)}), response, m2, cfg),
            back=lambda v: s * np.sinh(v))
    add(f"M2 excl. {eda.event_label(event)}", fit_spec(d[~d["operating_date"].isin(set(event))], response, m2, cfg))
    add("M2 excl. post-RTC+B (27 days)", fit_spec(d[d["regime"] == "pre"], response, m2, cfg))
    add("M3 (+ hour block)", fit_spec(d, response, spec_map["M3"], cfg))
    add("ALT (load anomaly)", fit_spec(d, response, spec_map["ALT"], cfg))
    return pd.DataFrame(rows)


def se_comparison(d: pd.DataFrame, response: str, spec: dict, cfg: dict) -> pd.DataFrame:
    """M2 coefficient SEs: classical vs HAC vs clustered by operating day."""
    out = {}
    for label, kw in (("se_ols", {"cov_type": "nonrobust"}), ("se_hac", {}),
                      ("se_cluster_day", {"cov_type": "cluster", "cluster_col": "operating_date"})):
        f = fit_spec(d, response, spec, cfg, **kw)
        out["coef"] = f.params
        out[label] = f.ols.bse
    t = pd.DataFrame(out)
    t["hac_over_ols"] = t["se_hac"] / t["se_ols"]
    t["cluster_over_ols"] = t["se_cluster_day"] / t["se_ols"]
    return t


def quantile_surfaces(d: pd.DataFrame, response: str, spec: dict, cfg: dict, refs: pd.DataFrame,
                      taus=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Conditional quantiles of the response on the M2 design, at the reference conditions.

    The tails of the spread are the extreme-risk object; a mean surface
    averages them away. Point estimates only: valid inference for quantile
    regression on dependent hourly data would need a block bootstrap.
    """
    import statsmodels.api as sm

    base = fit_spec(d, response, spec, cfg, cov_type="nonrobust")
    Zr = base.coding.encode(refs[list(base.names)].to_numpy(float))
    blocks = None
    if base.block_columns:
        blocks = pd.DataFrame({c: np.full(len(Zr), base.block_means[c]) for c in base.block_columns})
    Xr, _ = rsm.design_matrix(Zr, base.names, base.order, blocks)
    Xr = Xr[base.X.columns]
    rows = []
    for tau in taus:
        res = sm.QuantReg(base.y, base.X).fit(q=tau, max_iter=5000)
        rows.append({"tau": tau, **{f"pred_{k}": v for k, v in zip(refs.index, Xr.to_numpy() @ res.params.to_numpy())}})
    return pd.DataFrame(rows)


def influence(fit: rsm.RSMFit, d: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    cd = fit.ols.get_influence().cooks_distance[0]
    idx = np.argsort(cd)[::-1][:top]
    rows = d.reset_index(drop=True).iloc[idx]
    return pd.DataFrame({"interval_start_local": rows["interval_start_local"].to_numpy(),
                         "y": fit.y[idx], "fitted": fit.ols.fittedvalues[idx], "cooks_d": cd[idx],
                         "temp_f": rows["temp_f"].to_numpy(), "load_mw": rows["load_mw"].to_numpy(),
                         "gas_price": rows["gas_price"].to_numpy()})


def run(panel: pd.DataFrame, study: Study, out_dir: Path, fig_dir: Path, label: str = "") -> pd.DataFrame:
    """The full ladder for every (zone, response). Returns the M2 canonical table."""
    cfg = study.section("rsm")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    zones = [z for z in study.zones if z in set(panel["zone"])]
    spec_map = specs(cfg)
    flags = eda.tail_flags(panel)
    event = eda.event_days(panel)
    canon, comp, sens, quant, diag, lof_rows, sel_rows, wald_rows = [], [], [], [], [], [], [], []
    fwd_preds = []

    for response in cfg["responses"]:
        fits = {}
        for zone in zones:
            d = prepare(panel, zone, response, cfg)
            extreme = flags.loc[d.index, "extreme"]
            refs = reference_conditions(d, list(spec_map["ALT"]["factors"]) + list(cfg["factors"]))
            refs.to_csv(out_dir / f"reference_conditions_{zone}.csv")

            # in-sample: every spec, HAC inference
            in_rows = {}
            for name, spec in spec_map.items():
                f = fit_spec(d, response, spec, cfg)
                in_rows[name] = {"r2": f.ols.rsquared, "adj_r2": f.ols.rsquared_adj, "params": len(f.params),
                                 **curvature_test(f)}
                for g, r in rsm.robust_term_tests(f).iterrows():
                    wald_rows.append({"zone": zone, "response": response, "model": name, "group": g, **r})
                if name == "M2":
                    fits[zone] = f
                    rsm.coefficient_table(f).to_csv(out_dir / f"coef_{zone}_{response}_M2.csv")
                    dd = rsm.design_diagnostics(f)
                    diag.append({"zone": zone, "response": response,
                                 "condition_number": dd.attrs["condition_number"],
                                 **{f"vif_{t}": v for t, v in dd["vif"].items()}})
                    _, sel = rsm.hierarchical_selection(f, cfg.get("selection_cutoff", 0.01))
                    sel_rows += [{"zone": zone, "response": response, **r} for r in sel.to_dict("records")]
                    lof = rsm.lack_of_fit(f, cfg.get("lof_bins", 8))
                    lof_rows.append({"zone": zone, "response": response, **lof})
                    (out_dir / f"summary_{zone}_{response}_M2.txt").write_text(
                        rsm.summary_text(f, lof_bins=cfg.get("lof_bins", 8), selection_log=sel))
                    canon.append(canonical_row(zone, f))
                    influence(f, d).to_csv(out_dir / f"influence_{zone}_{response}_M2.csv", index=False)
                    se_comparison(d, response, spec, cfg).to_csv(out_dir / f"se_comparison_{zone}_{response}_M2.csv")
                if name == "M3":
                    rsm.coefficient_table(f).to_csv(out_dir / f"coef_{zone}_{response}_M3.csv")

            # out of sample: forward (honest in time) and leave-one-month-out (interpolation)
            for scheme in ("forward", "lomo"):
                m, p = evaluate(d, response, spec_map, cfg, scheme, extreme)
                if m.empty:
                    continue
                m.insert(0, "zone", zone)
                m.insert(1, "response", response)
                m["in_sample_r2"] = m["model"].map({k: v["r2"] for k, v in in_rows.items()})
                m["curv_p_hac"] = m["model"].map({k: v["curv_p"] for k, v in in_rows.items()})
                comp.append(m)
                if scheme == "forward":
                    fwd_preds.append(p.assign(zone=zone, response=response))

            s = sensitivity(d, response, spec_map, cfg, refs, event)
            sens.append(s.assign(zone=zone, response=response))
            if response == "spread" and cfg.get("quantile_surfaces", True):
                quant.append(quantile_surfaces(d, response, spec_map["M2"], cfg, refs)
                             .assign(zone=zone, sample="all hours"))
                ex = d[~d["operating_date"].isin(set(event))]
                quant.append(quantile_surfaces(ex, response, spec_map["M2"], cfg, refs)
                             .assign(zone=zone, sample=f"excl. {eda.event_label(event)}"))

        title = f"{plots.LABELS.get(response, response)}, M2 at median gas{label}"
        plots.surfaces_by_zone(fits, response, fig_dir / f"surface_{response}.png", title=title,
                               data={z: prepare(panel, z, response, cfg) for z in fits})
        plots.residual_diagnostics(fits, response, fig_dir / f"residuals_{response}.png")

    comp_df = pd.concat(comp, ignore_index=True)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    pd.concat(sens, ignore_index=True).to_csv(out_dir / "sensitivity.csv", index=False)
    if quant:
        pd.concat(quant, ignore_index=True).to_csv(out_dir / "quantile_surfaces_spread.csv", index=False)
    pd.DataFrame(diag).to_csv(out_dir / "design_diagnostics.csv", index=False)
    pd.DataFrame(wald_rows).to_csv(out_dir / "wald_term_groups.csv", index=False)
    pd.DataFrame(lof_rows).to_csv(out_dir / "lack_of_fit.csv", index=False)
    pd.DataFrame(sel_rows, columns=["zone", "response", "removed", "r2_decrease"]).to_csv(
        out_dir / "hierarchical_selection.csv", index=False)
    if fwd_preds:
        pd.concat(fwd_preds, ignore_index=True).to_parquet(out_dir / "forward_cv_predictions.parquet", index=False)
    plots.model_comparison(comp_df, fig_dir / "model_comparison.png")
    canon_df = pd.DataFrame(canon)
    canon_df.to_csv(out_dir / "canonical.csv", index=False)
    return canon_df


def describe(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-zone descriptive statistics for the report's data table."""
    cols = ["da_price", "rt_price", "spread", "imhr_da", "temp_f", "load_mw", "gas_price"]
    d = panel.groupby("zone")[cols].agg(["mean", "std", "min", "median", "max"])
    return d.round(2)


def correlation_check(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Factor correlations and VIFs per zone: how far from orthogonal the 'design' is."""
    rows = []
    for zone, d in panel.groupby("zone"):
        x = d[factors].dropna()
        r = np.corrcoef(x.to_numpy().T)
        vif = np.diag(np.linalg.inv(r))
        row = {"zone": zone}
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                row[f"corr_{factors[i]}_{factors[j]}"] = r[i, j]
        row.update({f"vif_{f}": v for f, v in zip(factors, vif)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("zone")


def curvature_contribution(panel: pd.DataFrame, study: Study) -> pd.DataFrame:
    """What the second-order terms add, per zone and response (Table 3, D-34).

    Compares nested models that differ only by the six second-order terms:
    M1 vs M2 (no hour control) and M1H vs M3 (both with the hour-of-day block,
    so the hour profile is not credited to curvature). Reports in-sample R²,
    HAC Wald p-values for the squared terms, the interactions and both jointly,
    forward-CV skill against M0, and M1 vs M2 predictions at the cold, mild and
    hot reference conditions (the shape a linear model cannot produce).
    """
    cfg = study.section("rsm")
    sm = specs(cfg)
    hour = cfg.get("hour_block", "hour_beginning")
    ladder = {"M1": sm["M1"], "M2": sm["M2"],
              "M1H": {**sm["M1"], "blocks": list(sm["M1"]["blocks"]) + [hour]}, "M3": sm["M3"]}
    flags = eda.tail_flags(panel)
    rows = []
    for response in cfg["responses"]:
        for zone in [z for z in study.zones if z in set(panel["zone"])]:
            d = prepare(panel, zone, response, cfg)
            fits = {k: fit_spec(d, response, s, cfg) for k, s in ladder.items()}
            groups = rsm.robust_term_tests(fits["M2"])
            row = {"zone": zone, "response": response,
                   **{f"r2_{k}": f.ols.rsquared for k, f in fits.items()},
                   "p_quadratic": groups.loc["quadratic", "p_value"],
                   "p_interaction": groups.loc["lin x lin", "p_value"],
                   "p_second_order": curvature_test(fits["M2"])["curv_p"],
                   "p_second_order_with_hour": curvature_test(fits["M3"])["curv_p"],
                   "share_fitted_negative_M2": float((fits["M2"].ols.fittedvalues < 0).mean())}
            m, _ = evaluate(d, response, ladder, cfg, "forward", flags.loc[d.index, "extreme"])
            row.update({f"fwd_skill_{r.model}": r.skill_vs_M0 for r in m.itertuples() if r.model != "M0"})
            refs = reference_conditions(d, list(cfg["factors"]))
            for k in ("M1", "M2"):
                pred = fits[k].predict(refs[list(fits[k].names)].to_numpy(float))
                row.update({f"pred_{k}_{ref}": v for ref, v in zip(refs.index, pred)})
            rows.append(row)
    return pd.DataFrame(rows)
