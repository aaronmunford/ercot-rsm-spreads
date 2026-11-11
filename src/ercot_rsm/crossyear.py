"""Does a finding survive a different calendar year? (Step 2 of the 2026-10-10 plan)

* ``replication_table``: the same pre-specified quantities per zone and year, read
  from each window's own result CSVs, plus whether signed findings keep their sign.
* ``out_of_year``: fit on one year, predict the next one: the cleanest honest
  out-of-sample test (every training hour precedes every test hour). Reports the
  share of test hours inside the training data's convex hull, because a
  polynomial surface is not trustworthy outside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, eda, rsm, surprise
from .config import Study


def _read(results: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(results / name, **kw)


def year_quantities(study: Study) -> pd.DataFrame:
    """One row per zone: the quantities the report's claims rest on, from this window's CSVs."""
    R = study.results_dir
    q1 = _read(R, "eda/tables/q1_spread_summary.csv", index_col=0)
    tl = _read(R, "eda/tables/q2_temp_load.csv", index_col=0)
    t = _read(R, "eda/tables/q3_by_temperature.csv")
    sens = _read(R, "sensitivity.csv")
    comp = _read(R, "model_comparison.csv")
    quant = _read(R, "quantile_surfaces_spread.csv")
    lf = _read(R, "surprise/ladder_forward.csv")
    sc = _read(R, "surprise/S_canonical.csv")
    gt = _read(R, "surprise/ladder_group_tests.csv")
    ex = _read(R, "eda/tables/q5_extremes_summary.csv")
    rows = []
    for z in q1.index:
        def tf(b):
            r = t[(t.zone == z) & (t.temp_bin_f == b)]
            return float(r["tail_freq_rel_pct"].iloc[0]) if len(r) else np.nan

        def pred(resp, col):
            r = sens[(sens.zone == z) & (sens.response == resp) & (sens.variant == "M2 raw (primary)")]
            return float(r[col].iloc[0])

        def fwd(resp, model, scheme="forward"):
            c = comp[(comp.zone == z) & (comp.response == resp) & (comp.model == model) & (comp.scheme == scheme)]
            return float(c["skill_vs_M0"].iloc[0]) if len(c) else np.nan

        def lad(resp, model, col="skill_vs_M0"):
            c = lf[(lf.zone == z) & (lf.response == resp) & (lf.model == model)]
            return float(c[col].iloc[0]) if len(c) else np.nan

        qa = quant[(quant.zone == z) & (quant["sample"] == "all hours")].set_index("tau")
        s_sp = sc[(sc.zone == z) & (sc.response == "spread")].iloc[0]
        g = gt[(gt.zone == z) & (gt.response == "spread")].set_index("model")
        curv = comp[(comp.zone == z) & (comp.response == "abs_spread") & (comp.model == "M2") & (comp.scheme == "forward")]
        rows.append({
            "zone": z, "year": study.start.year,
            "spread_median": q1.loc[z, "median"], "spread_skew": q1.loc[z, "skew"],
            "top1pct_share_of_abs": q1.loc[z, "top1pct_share_of_abs_sum"], "share_DA_above_RT": q1.loc[z, "share_positive"],
            "load_r2_T_T2": tl.loc[z, "r2_T_T2"], "load_min_at_F": tl.loc[z, "quadratic_min_load_at_F"],
            "tailfreq_55_65F": tf("55-65"), "tailfreq_le32F": tf("<=32"), "tailfreq_gt95F": tf(">95"),
            "extremes_share_ramp_hours": float(ex["share_in_ramp_hours"].iloc[0]),
            "hot1pct_spread_M2": pred("spread", "pred_hot_1pct"),
            "abs_spread_mild": pred("abs_spread", "pred_mild_60_70F"), "abs_spread_hot": pred("abs_spread", "pred_hot_1pct"),
            "abs_spread_curv_p": float(curv["curv_p_hac"].iloc[0]),
            "q_band_mild": float(qa.loc[0.95, "pred_mild_60_70F"] - qa.loc[0.05, "pred_mild_60_70F"]),
            "q_band_hot": float(qa.loc[0.95, "pred_hot_1pct"] - qa.loc[0.05, "pred_hot_1pct"]),
            "fwd_skill_spread_M2": fwd("spread", "M2"), "fwd_skill_imhr_M3": fwd("imhr_da", "M3"),
            "fwd_skill_spread_S": lad("spread", "S"), "fwd_skill_spread_D": lad("spread", "D"),
            "fwd_skill_abs_E": lad("abs_spread", "E"),
            "S_interaction_coef": s_sp["interaction_coef"], "S_interaction_p": s_sp["interaction_p"],
            "D_group_p_holm": float(g.loc["D", "group_p_holm"]), "C_group_p_holm": float(g.loc["C", "group_p_holm"]),
        })
    return pd.DataFrame(rows)


def replication_table(studies: list[Study]) -> pd.DataFrame:
    """Long table zone x quantity x year, with a same-sign flag for signed quantities."""
    q = pd.concat([year_quantities(s) for s in studies], ignore_index=True)
    long = q.melt(id_vars=["zone", "year"], var_name="quantity").pivot_table(
        index=["zone", "quantity"], columns="year", values="value").reset_index()
    years = sorted(q["year"].unique())
    if len(years) == 2:
        a, b = years
        long["same_sign"] = np.sign(long[a]) == np.sign(long[b])
    return long


def _hull_share(train: pd.DataFrame, test: pd.DataFrame, factors: list[str]) -> float:
    c = rsm.Coding.from_data(train[factors], "quantile", (0.01, 0.99))
    return float(rsm.in_hull(c.encode(train[factors].to_numpy(float)), c.encode(test[factors].to_numpy(float))).mean())


def out_of_year(train_panel: pd.DataFrame, test_panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fit complete common training rows and predict complete common test rows.

    A year label does not imply complete calendar coverage: unavailable forecast
    inputs can truncate the training sample. Report its size and dates explicitly.
    """
    specs = {"M2 (A)": analysis.specs(cfg)["M2"], "M3": analysis.specs(cfg)["M3"]}
    lad = surprise.ladder_specs(cfg)
    specs.update({"D": lad["D"], "S": lad["S"], "E": lad["E"]})
    flags = eda.tail_flags(test_panel)
    rows = []
    for response in ("spread", "abs_spread", "imhr_da"):
        for zone in sorted(set(test_panel["zone"])):
            tr = analysis.prepare(train_panel, zone, response, cfg)
            te = analysis.prepare(test_panel, zone, response, cfg)
            if te["operating_date"].min() <= tr["operating_date"].max():
                raise AssertionError("out-of-year test: the training year must end before the test year starts")
            need = sorted({f for s in specs.values() for f in s["factors"]} | {"hour_beginning"})
            tr, te = tr.dropna(subset=need), te.dropna(subset=need)
            y = te[response].to_numpy()
            m0 = te["hour_beginning"].map(tr.groupby("hour_beginning")[response].mean()).to_numpy()
            sse0 = float(((y - m0) ** 2).sum())
            ext = flags.loc[te.index, "extreme"].to_numpy()
            for name, spec in specs.items():
                if response == "imhr_da" and name not in ("M2 (A)", "M3"):
                    continue
                f = analysis.fit_spec(tr, response, spec, cfg, cov_type="nonrobust")
                e = y - f.predict_frame(te)
                k = len(spec["factors"])
                rows.append({"response": response, "zone": zone, "model": name,
                             "train_year": int(pd.Timestamp(tr["operating_date"].min()).year),
                             "n_train": len(tr),
                             "first_train_utc": str(tr["interval_start_utc"].min()),
                             "last_train_utc": str(tr["interval_start_utc"].max()),
                             "test_year": int(pd.Timestamp(te["operating_date"].min()).year), "n_test": len(te),
                             "skill_vs_M0": 1 - float((e ** 2).sum()) / sse0, "mae": float(np.abs(e).mean()),
                             "mae_M0": float(np.abs(y - m0).mean()),
                             "mae_extreme_hours": float(np.abs(e[ext]).mean()),
                             "mae_extreme_hours_M0": float(np.abs((y - m0)[ext]).mean()),
                             "test_share_in_train_hull": _hull_share(tr, te, spec["factors"]) if k <= 3 else np.nan})
    return pd.DataFrame(rows)
