"""Expectations and surprises: the pre-specified ladder A-D, the surprise surface S and
the ex-ante model E (DECISIONS.md D-28 to D-30).

The spread is DA (priced on expectations at bid close) minus RT (the realisation),
so the natural explanatory variables are forecast errors measured against the
forecast vintage available before the 10:00 D-1 bid close (here 09:45).

Evidence rule (D-29): an added term group counts only if its HAC Wald test
survives Holm adjustment across the pre-specified tests AND forward CV improves
on the previous rung. E is evaluated by forward CV only, with DA-time inputs only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import analysis, eda, plots, rsm
from .config import Study

CORE = ["temp_f", "load_mw", "gas_price"]
# Inputs known by 09:45 on D-1 (D-30). Anything else in E is a leak.
EXANTE_ALLOWED = {"net_load_fc", "load_fc_mw", "gas_price_exante", "sys_load_fc", "wind_fc", "solar_fc",
                  "net_load_fc_ramp"}
EXANTE_BLOCKS_ALLOWED = {"regime", "hour_beginning"}
LADDER = ("A", "B", "C", "D")


def _terms(factors: list[str], core: list[str], extras: list[tuple[str, bool, bool]]) -> list[str]:
    """Hierarchical term subset: the full quadratic in ``core`` plus, for each extra
    factor x, x and (optionally) x^2 and x*load_mw."""
    names = tuple(factors)
    exps = rsm.term_exponents(names, 2)
    idx = {f: i for i, f in enumerate(names)}
    core_i = {idx[c] for c in core}
    keep = []
    for term, e in exps.items():
        support = {i for i, v in enumerate(e) if v}
        if support <= core_i:
            keep.append(term)
            continue
        for x, square, with_load in extras:
            xi = idx[x]
            if support == {xi} and (e[xi] == 1 or (e[xi] == 2 and square)):
                keep.append(term)
            elif with_load and support == {xi, idx["load_mw"]} and e[xi] == 1 and e[idx["load_mw"]] == 1:
                keep.append(term)
    return keep


def ladder_specs(cfg: dict) -> dict[str, dict]:
    base = list(cfg.get("block_cols", ["regime"]))
    hour = cfg.get("hour_block", "hour_beginning")
    steps = {
        "A": [],
        "B": [("net_load_ramp", True, True)],
        "C": [("net_load_ramp", True, True), ("load_err", True, True)],
        "D": [("net_load_ramp", True, True), ("load_err", True, True),
              ("wind_err_hsl", True, False), ("solar_err_hsl", True, False)],
    }
    out = {}
    for k, extras in steps.items():
        factors = CORE + [x for x, _, _ in extras]
        out[k] = {"factors": factors, "order": 2, "blocks": base, "terms": _terms(factors, CORE, extras),
                  "added": [x for x, _, _ in extras[len(steps[LADDER[LADDER.index(k) - 1]]):]] if k != "A" else []}
    out["S"] = {"factors": ["net_load_err_avail", "net_load_fc"], "order": 2, "blocks": base}
    out["E"] = {"factors": ["net_load_fc", "load_fc_mw", "gas_price_exante"], "order": 2, "blocks": base + [hour]}
    assert_exante(out["E"])
    return out


def assert_exante(spec: dict) -> None:
    bad = set(spec["factors"]) - EXANTE_ALLOWED
    badb = set(spec["blocks"]) - EXANTE_BLOCKS_ALLOWED
    if bad or badb:
        raise ValueError(f"ex-ante model uses information not available at 09:45 D-1: {sorted(bad | badb)}")


def group_test(fit: rsm.RSMFit, factors: list[str]) -> dict:
    """HAC Wald test that every term involving ``factors`` is zero."""
    cols = list(fit.X.columns)
    members = [c for c in cols if c in fit.exponents and any(
        fit.exponents[c][fit.names.index(f)] for f in factors)]
    R = np.zeros((len(members), len(cols)))
    for r, c in enumerate(members):
        R[r, cols.index(c)] = 1.0
    w = fit.ols.wald_test(R, scalar=True, use_f=False)
    return {"group_df": len(members), "group_chi2": float(w.statistic), "group_p": float(w.pvalue)}


def s_reference(d: pd.DataFrame) -> pd.DataFrame:
    """Surprise x expected-tightness grid for S: net-load error at its 5/50/95th
    percentiles crossed with the net-load forecast at its 10/50/90th."""
    e = d["net_load_err_avail"].quantile([0.05, 0.5, 0.95]).to_numpy()
    f = d["net_load_fc"].quantile([0.1, 0.5, 0.9]).to_numpy()
    rows = [{"net_load_err_avail": a, "net_load_fc": b, "label": f"err_p{q1}_fc_p{q2}"}
            for a, q1 in zip(e, (5, 50, 95)) for b, q2 in zip(f, (10, 50, 90))]
    return pd.DataFrame(rows).set_index("label")


def e_reference(d: pd.DataFrame) -> pd.DataFrame:
    nf = d["net_load_fc"]
    groups = {"fc_low_10pct": nf <= nf.quantile(0.1), "fc_mid": nf.between(nf.quantile(0.45), nf.quantile(0.55)),
              "fc_high_10pct": nf >= nf.quantile(0.9)}
    cols = ["net_load_fc", "load_fc_mw", "gas_price_exante"]
    return pd.DataFrame({k: d.loc[m, cols].mean() for k, m in groups.items()}).T


def run(panel: pd.DataFrame, study: Study, out_dir: Path, fig_dir: Path) -> dict[str, pd.DataFrame]:
    cfg = study.section("rsm")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = ladder_specs(cfg)
    flags = eda.tail_flags(panel)
    event = set(eda.event_days(panel))
    ev_label = f"excl. {eda.event_label(event)}"
    zones = [z for z in study.zones if z in set(panel["zone"])]
    ins, fwd, groups, sens, quant, canon, diag = [], [], [], [], [], [], []
    for response in ("spread", "abs_spread"):
        s_fits, s_data = {}, {}
        for zone in zones:
            d = analysis.prepare(panel, zone, response, cfg)
            d = d.dropna(subset=sorted({f for s in specs.values() for f in s["factors"]})).copy()
            extreme = flags.loc[d.index, "extreme"]
            for name, spec in specs.items():
                f = analysis.fit_spec(d, response, spec, cfg)
                dd = rsm.design_diagnostics(f)
                row = {"zone": zone, "response": response, "model": name, "n": len(f.y), "params": len(f.params),
                       "r2": f.ols.rsquared, "adj_r2": f.ols.rsquared_adj,
                       "vif_max": float(dd["vif"].drop([c for c in dd.index if dd.loc[c, "group"] == "block"]).max()),
                       "condition_number": dd.attrs["condition_number"]}
                if spec.get("added"):
                    row.update(group_test(f, spec["added"]))
                    gx = analysis.fit_spec(d[~d["operating_date"].isin(event)], response, spec, cfg)
                    row["group_p_excl_event"] = group_test(gx, spec["added"])["group_p"]
                    groups.append({"zone": zone, "response": response, "model": name,
                                   "added": "+".join(spec["added"]), **{k: row[k] for k in ("group_df", "group_chi2", "group_p")},
                                   "group_p_excl_event": row["group_p_excl_event"]})
                if name == "S":
                    s_fits[zone], s_data[zone] = f, d
                    can = rsm.canonical_analysis(f)
                    canon.append({"zone": zone, "response": response, "kind": can.kind, "inside": can.inside_coded_cube,
                                  **{f"lambda_{i + 1}": v for i, v in enumerate(can.eigenvalues)},
                                  **analysis.curvature_test(f),
                                  "interaction_coef": float(f.params.get("net_load_err_avail*net_load_fc", np.nan)),
                                  "interaction_p": float(f.ols.pvalues.get("net_load_err_avail*net_load_fc", np.nan))})
                    refs = s_reference(d)
                    pred = f.predict(refs[list(f.names)].to_numpy(float))
                    q = analysis.quantile_surfaces(d, response, spec, cfg, refs[list(f.names)]) if response == "spread" else None
                    for (lab, r), pm in zip(refs.iterrows(), pred):
                        quant.append({"zone": zone, "response": response, "model": "S", "point": lab,
                                      "net_load_err_avail": r["net_load_err_avail"], "net_load_fc": r["net_load_fc"],
                                      "mean": pm, **({f"q{t:.2f}": q.loc[q.tau == t, f"pred_{lab}"].iloc[0]
                                                      for t in q.tau} if q is not None else {})})
                    # sensitivities of S (D-28)
                    for label, col in (("gen-based net-load error", "net_load_err"),
                                       ("09:00 vintage (availability)", "net_load_err_avail_sens")):
                        if col not in d.columns:
                            continue  # the 09:00 vintage is pulled for 2025 only
                        alt = {**spec, "factors": [col, "net_load_fc" if "sens" not in col else "net_load_fc_sens"]}
                        dd2 = d.dropna(subset=alt["factors"])
                        fa = analysis.fit_spec(dd2, response, alt, cfg)
                        ix = f"{alt['factors'][0]}*{alt['factors'][1]}"
                        sens.append({"zone": zone, "response": response, "variant": label, "r2": fa.ols.rsquared,
                                     **analysis.curvature_test(fa),
                                     "interaction_p": float(fa.ols.pvalues.get(ix, np.nan))})
                    fe = analysis.fit_spec(d[~d["operating_date"].isin(event)], response, spec, cfg)
                    sens.append({"zone": zone, "response": response, "variant": ev_label,
                                 "r2": fe.ols.rsquared, **analysis.curvature_test(fe),
                                 "interaction_p": float(fe.ols.pvalues.get("net_load_err_avail*net_load_fc", np.nan))})
                if name == "E" and response == "spread":
                    refs = e_reference(d)
                    q = analysis.quantile_surfaces(d, response, spec, cfg, refs)
                    for lab in refs.index:
                        quant.append({"zone": zone, "response": response, "model": "E", "point": lab,
                                      "net_load_fc": refs.loc[lab, "net_load_fc"],
                                      **{f"q{t:.2f}": q.loc[q.tau == t, f"pred_{lab}"].iloc[0] for t in q.tau}})
                ins.append(row)
            # forward CV (honest in time); E only under forward CV (D-30)
            m, p = analysis.evaluate(d, response, specs, cfg, "forward", extreme)
            if not m.empty:
                order = ["M0", "A", "B", "C", "D"]
                for k in range(1, len(order)):
                    prev, cur = order[k - 1], order[k]
                    m.loc[m["model"] == cur, "skill_vs_previous"] = 1 - ((p["y"] - p[cur]) ** 2).sum() / ((p["y"] - p[prev]) ** 2).sum()
                for k in ("S", "E"):
                    m.loc[m["model"] == k, "skill_vs_A"] = 1 - ((p["y"] - p[k]) ** 2).sum() / ((p["y"] - p["A"]) ** 2).sum()
                fwd.append(m.assign(zone=zone, response=response))
        plots.surfaces_by_zone(s_fits, response, fig_dir / f"surface_S_{response}.png",
                               title=f"{plots.LABELS.get(response, response)}: surprise surface S (2025)", data=s_data)
    gdf = pd.DataFrame(groups)
    for resp, g in gdf.groupby("response"):
        gdf.loc[g.index, "group_p_holm"] = multipletests(g["group_p"], method="holm")[1]
    tables = {"ladder_insample": pd.DataFrame(ins), "ladder_forward": pd.concat(fwd, ignore_index=True),
              "ladder_group_tests": gdf, "S_canonical": pd.DataFrame(canon), "S_E_quantiles": pd.DataFrame(quant),
              "S_sensitivity": pd.DataFrame(sens)}
    for k, t in tables.items():
        t.to_csv(out_dir / f"{k}.csv", index=False)
    plots.ladder_skill(tables["ladder_forward"], fig_dir / "ladder_skill.png")
    return tables


def event_forward_predictions(panel: pd.DataFrame, cfg: dict, day, zone: str = "houston") -> pd.DataFrame:
    """Spread predictions for one operating day from models trained only on the months
    before the event month (the report's forward fold for that month): the surprise surface S
    (net-load surprise x expected net load) and the core RSM M2 (realised T, L, G).
    S is given the realised surprise, so it is retrospective; the point is whether even that
    information accounts for the size of the day's spread."""
    d = analysis.prepare(panel, zone, "spread", cfg)
    s_spec = ladder_specs(cfg)["S"]
    m2_spec = analysis.specs(cfg)["M2"]
    d = d.dropna(subset=list(s_spec["factors"]) + list(m2_spec["factors"]))
    month = str(d.loc[d["operating_date"] == day, "month"].iloc[0])
    train = d[d["month"].astype(str) < month]
    test = d[d["operating_date"] == day]
    if train.empty or test.empty:
        raise ValueError(f"no training months before {month} or no rows on {day}")
    out = test[["interval_start_utc", "hour_beginning", "spread", "net_load_fc", "net_load_err_avail"]].copy()
    out["pred_S_forward"] = analysis.fit_spec(train, "spread", s_spec, cfg, cov_type="nonrobust").predict_frame(test)
    out["pred_M2_forward"] = analysis.fit_spec(train, "spread", m2_spec, cfg, cov_type="nonrobust").predict_frame(test)
    out["train_months"] = f"{train['month'].astype(str).min()}..{train['month'].astype(str).max()}"
    out["zone"] = zone
    return out.reset_index(drop=True)
