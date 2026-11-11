"""Isolated experiment: does the quadratic RSM improve with (1) calendar/seasonal controls
or (2) net-load ramp variables?

Run:  .venv/bin/python experiments/rsm_controls.py
Writes only to results/2025-01-01_2026-01-01/experiments/rsm_controls/. It reads cached
panels; it downloads nothing and touches neither src/ nor report/ nor reports/.

Models (each zone x response; all share the report's M2 core: full quadratic in coded
realised temperature, zonal load and Henry Hub gas, plus the RTC+B regime intercept):

  M2        the report's model (baseline, reproduced exactly; see verify_baseline)
  M2+C      + calendar: hour-of-day dummies, non-business-day flag (weekend or NERC holiday)
  M2+CS     + calendar + season: annual Fourier terms, 2 harmonics of day of year
  M2+Rrt    + realised net-load ramp (this hour minus previous hour) and its square
  M2+Rfc    + DA-time forecast net-load ramp (09:45 D-1 vintage) and its square
  M2+CS+Rfc + calendar + season + forecast ramp
  M2+C+Rrt, M2+C+Rfc  calendar + ramp: is the ramp informative beyond the hour of day?
                      (half of the ramps' variance is explained by calendar and season)
  EX        ex-ante only: calendar + season + forecast ramp + forecast net load (+ square);
            no realised factor, so the only model here usable at bid time
  M0        hour-of-day mean of the training sample (the report's benchmark)

Information sets: every M2 variant uses realised T, L, G for the hour it describes, so it
is RETROSPECTIVE (explanatory) however its extra terms are built; M2+Rrt is doubly so.
Calendar, season and forecast ramp/net load are known at the 10:00 D-1 bid close, so EX
is EX-ANTE. Forward validation of a retrospective model tests stability, not forecasting.

Preprocessing is fitted on training rows only, inside every fold: the RSM coding
(1st/99th percentiles -> +/-1), the mean/SD scaling of continuous extra terms, and which
regime dummy exists. Calendar and Fourier terms are deterministic functions of the date.
Month dummies are NOT used: in forward folds the test month is never in training.

Validation, identical observations and folds for every model:
  A. forward expanding window within 2025, train on months < m, test month m (Apr-Dec),
     exactly the report's folds (rsm.time_folds, min 3 training months);
  B. fit on all of 2024, predict all of 2025 (season fully covered by training).
Rows missing any feature of any model (<= 3 per zone: first hour, DST) are dropped from
every model, so all comparisons use the same observations.

Pre-specified criteria (fixed before the first run, 2026-10-10):
  material       pooled forward skill gain over M2 >= 0.02 (2% of M0's SSE) in the median
                 zone, or median MAE reduction >= 5%;
  stable         in >= 3 of 4 zones: beats M2 in >= 7 of 9 forward folds, Diebold-Mariano
                 p < 0.05 on daily squared-error differentials (Newey-West, 7 lags), and a
                 positive gain over M2 in the year-ahead test (B);
  interpretable  added-term signs match economics in >= 80% of refits (ramp: up-ramp ->
                 RT above DA, i.e. negative for the spread, positive for |spread| and DA IMHR)
                 and the term is not redundant (VIF < 10 against the rest of the design).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ercot_rsm import analysis, config, features, rsm  # noqa: E402

W24, W25 = "2024-01-01_2025-01-01", "2025-01-01_2026-01-01"
OUT = ROOT / "results" / W25 / "experiments" / "rsm_controls"
PROTECTED = [ROOT / "reports" / "report_v2.pdf"]
ZONES = ["houston", "north", "south", "west"]
RESPONSES = ["spread", "abs_spread", "imhr_da"]
FACTORS = ["temp_f", "load_mw", "gas_price"]
CRIT = {"material_skill": 0.02, "material_mae": 0.05, "stable_folds": 7, "dm_alpha": 0.05, "zones": 3,
        "sign_share": 0.8, "vif": 10.0}
EXPECTED_RAMP_SIGN = {"spread": -1, "abs_spread": +1, "imhr_da": +1}

SPECS = {  # name -> (uses RSM core, calendar, season, ramp variable or None, forecast net load)
    "M2": (True, False, False, None, False),
    "M2+C": (True, True, False, None, False),
    "M2+CS": (True, True, True, None, False),
    "M2+Rrt": (True, False, False, "net_load_ramp", False),
    "M2+Rfc": (True, False, False, "net_load_fc_ramp", False),
    "M2+CS+Rfc": (True, True, True, "net_load_fc_ramp", False),
    "M2+C+Rrt": (True, True, False, "net_load_ramp", False),
    "M2+C+Rfc": (True, True, False, "net_load_fc_ramp", False),
    "EX": (False, True, True, "net_load_fc_ramp", True),
}
INFO = {m: ("ex-ante" if m == "EX" else "retrospective") for m in SPECS}
NEEDED = ["net_load_ramp", "net_load_fc_ramp", "net_load_fc", *FACTORS]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Features and design (training-only preprocessing)
# --------------------------------------------------------------------------


def deterministic_terms(d: pd.DataFrame) -> pd.DataFrame:
    """Calendar and Fourier columns: functions of the timestamp only, no fitted parameters."""
    od = pd.to_datetime(d["operating_date"])
    hol = features.nerc_holidays(set(od.dt.year))
    out = pd.DataFrame(index=d.index)
    for h in range(1, 24):
        out[f"hour_{h}"] = (d["hour_beginning"] == h).astype(float)
    out["nonbusiness"] = ((d["weekday"] >= 5) | od.dt.date.isin(hol)).astype(float)
    doy = od.dt.dayofyear.to_numpy(float)
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return out


class Design:
    """Everything learned from the training rows that is needed to build a design matrix."""

    def __init__(self, train: pd.DataFrame, model: str):
        core, cal, season, ramp, fcnl = SPECS[model]
        self.model, self.core, self.cal, self.season, self.ramp, self.fcnl = model, core, cal, season, ramp, fcnl
        self.coding = rsm.Coding.from_data(train[FACTORS], "quantile", (0.01, 0.99)) if core else None
        self.regime = core and train["regime"].nunique() > 1
        self.scale = {}
        for c in [x for x in (ramp, "net_load_fc" if fcnl else None) if x]:
            sd = float(train[c].std())
            if not sd > 0:
                raise ValueError(f"{c} has no spread in training")
            self.scale[c] = (float(train[c].mean()), sd)
        self.columns = None

    def matrix(self, d: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.core:
            blocks = None
            if self.regime:
                blocks = pd.DataFrame({"regime_post": (d["regime"] == "post").to_numpy(float)})
            X, _ = rsm.design_matrix(self.coding.encode(d[FACTORS]), tuple(FACTORS), 2, blocks)
            parts.append(X)
        else:
            parts.append(pd.DataFrame({"Intercept": np.ones(len(d))}))
        det = deterministic_terms(d).reset_index(drop=True)
        if self.cal:
            parts.append(det[[c for c in det.columns if c.startswith("hour_")] + ["nonbusiness"]])
        if self.season:
            parts.append(det[["sin1", "cos1", "sin2", "cos2"]])
        for c, (mu, sd) in self.scale.items():
            z = (d[c].to_numpy(float) - mu) / sd
            parts.append(pd.DataFrame({f"{c}_z": z, f"{c}_z2": z ** 2}))
        X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        if self.columns is None:
            keep = [c for c in X.columns if c == "Intercept" or X[c].std() > 0]  # drop constant-in-training
            self.columns = keep
        return X.reindex(columns=self.columns, fill_value=0.0)


def fit(train: pd.DataFrame, model: str, response: str, hac: bool = False):
    des = Design(train, model)
    X = des.matrix(train)
    y = train[response].to_numpy(float)
    res = (sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 48}) if hac else sm.OLS(y, X).fit())
    return des, res


def predict(des: Design, res, test: pd.DataFrame) -> np.ndarray:
    return des.matrix(test).to_numpy(float) @ res.params.to_numpy(float)


# --------------------------------------------------------------------------
# Checks: baseline reproduces the report, preprocessing never sees test rows
# --------------------------------------------------------------------------


def verify_baseline(panel: pd.DataFrame, cfg: dict) -> float:
    """Max |our M2 forward prediction - the report's| over every zone, response and fold."""
    ref = pd.read_parquet(ROOT / "results" / W25 / "forward_cv_predictions.parquet")
    worst = 0.0
    for response in RESPONSES:
        for zone in ZONES:
            d = analysis.prepare(panel, zone, response, cfg)
            for train, test, month in rsm.time_folds(d["month"], "forward", cfg.get("cv_min_train_months", 3)):
                des, res = fit(d[train], "M2", response)
                ours = predict(des, res, d[test])
                r = ref[(ref.zone == zone) & (ref.response == response) & (ref.month == month)]["M2"].to_numpy()
                worst = max(worst, float(np.max(np.abs(ours - r))))
    return worst


def verify_no_leakage(d: pd.DataFrame, response: str) -> bool:
    """Fit on a training fold; corrupt every test row; the fitted model must not change."""
    train, test, _ = rsm.time_folds(d["month"], "forward", 3)[0]
    a_des, a = fit(d[train], "M2+CS+Rfc", response)
    bad = d.copy()
    for c in [*FACTORS, "net_load_fc_ramp", response]:
        bad.loc[test, c] = bad.loc[test, c] * 1000 + 1e6
    b_des, b = fit(bad[train], "M2+CS+Rfc", response)
    same_coding = np.allclose(a_des.coding.center, b_des.coding.center) and np.allclose(a_des.coding.half_range,
                                                                                         b_des.coding.half_range)
    return bool(same_coding and a_des.scale == b_des.scale and np.allclose(a.params, b.params))


# --------------------------------------------------------------------------
# Metrics and tests
# --------------------------------------------------------------------------


def dm_pvalue(loss_base: np.ndarray, loss_model: np.ndarray, day: np.ndarray, lags: int = 7) -> float:
    """Diebold-Mariano: mean daily loss differential (base - model) = 0, Newey-West SE.
    Positive differential = the model has lower loss. Two-sided p-value."""
    dd = pd.Series(loss_base - loss_model).groupby(day).sum().to_numpy()
    res = sm.OLS(dd, np.ones((len(dd), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.pvalues[0])


def metrics(y, p, base_mean):
    e = y - p
    sse = float((e ** 2).sum())
    return {"sse": sse, "oos_r2": 1 - sse / float(((y - base_mean) ** 2).sum()),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}


def forward_run(d: pd.DataFrame, response: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Predictions for every model on the report's forward folds, plus refit coefficients."""
    rows, coefs = [], []
    for train, test, month in rsm.time_folds(d["month"], "forward", 3):
        tr, te = d[train], d[test]
        out = pd.DataFrame({"month": month, "day": te["operating_date"].astype(str).to_numpy(),
                            "y": te[response].to_numpy(float), "train_mean": tr[response].mean(),
                            "M0": te["hour_beginning"].map(tr.groupby("hour_beginning")[response].mean()).to_numpy()})
        for m in SPECS:
            des, res = fit(tr, m, response)
            out[m] = predict(des, res, te)
            if des.ramp:
                coefs.append({"fold": month, "model": m, "coef_ramp_z": float(res.params[f"{des.ramp}_z"])})
        rows.append(out)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coefs), []


def summarise(pred: pd.DataFrame, scheme: str) -> pd.DataFrame:
    y = pred["y"].to_numpy()
    sse0 = float(((y - pred["M0"]) ** 2).sum())
    base = pred["M2"].to_numpy()
    out = []
    for m in ["M0", *SPECS]:
        p = pred[m].to_numpy()
        r = {"model": m, "information": "benchmark" if m == "M0" else INFO[m], "scheme": scheme, "n": len(y),
             **metrics(y, p, pred["train_mean"].to_numpy())}
        r["skill_vs_M0"] = 1 - r["sse"] / sse0
        if m not in ("M0", "M2"):
            r["dm_p_sq_vs_M2"] = dm_pvalue((y - base) ** 2, (y - p) ** 2, pred["day"].to_numpy())
            r["dm_p_abs_vs_M2"] = dm_pvalue(np.abs(y - base), np.abs(y - p), pred["day"].to_numpy())
            if "month" in pred:
                g = pred.assign(eb=(y - base) ** 2, em=(y - p) ** 2).groupby("month")[["eb", "em"]].sum()
                r["folds_beating_M2"] = int((g["em"] < g["eb"]).sum())
                r["folds"] = len(g)
        out.append(r)
    c = pred["M2+C"].to_numpy()
    for r in out:
        if r["model"] in ("M2+C+Rrt", "M2+C+Rfc"):
            p = pred[r["model"]].to_numpy()
            r["dm_p_sq_vs_M2C"] = dm_pvalue((y - c) ** 2, (y - p) ** 2, pred["day"].to_numpy())
            if "month" in pred:
                g = pred.assign(eb=(y - c) ** 2, em=(y - p) ** 2).groupby("month")[["eb", "em"]].sum()
                r["folds_beating_M2C"] = int((g["em"] < g["eb"]).sum())
    t = pd.DataFrame(out)
    t["skill_gain_vs_M2C"] = t["skill_vs_M0"] - t.set_index("model").loc["M2+C", "skill_vs_M0"]
    b = t.set_index("model").loc["M2"]
    t["skill_gain_vs_M2"] = t["skill_vs_M0"] - b["skill_vs_M0"]
    t["mae_change_vs_M2"] = t["mae"] / b["mae"] - 1
    t["rmse_change_vs_M2"] = t["rmse"] / b["rmse"] - 1
    return t


# --------------------------------------------------------------------------
# Retrospective in-sample fit, redundancy, coefficient stability
# --------------------------------------------------------------------------


def vif_of(X: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    others = X.drop(columns=["Intercept"])
    for c in cols:
        if c not in others:
            continue
        rest = others.drop(columns=[c])
        r2 = sm.OLS(others[c].to_numpy(), sm.add_constant(rest.to_numpy())).fit().rsquared
        out[c] = 1 / max(1 - r2, 1e-12)
    return out


def insample_and_redundancy(d: pd.DataFrame, response: str) -> list[dict]:
    rows = []
    for m in SPECS:
        des, res = fit(d, m, response, hac=True)
        X = des.matrix(d)
        added = [c for c in X.columns if c.startswith(("hour_", "nonbusiness", "sin", "cos", "net_load"))]
        r = {"model": m, "information": INFO[m], "r2": res.rsquared, "adj_r2": res.rsquared_adj, "params": X.shape[1],
             "cond_number": float(np.linalg.cond(X.to_numpy() / np.linalg.norm(X.to_numpy(), axis=0)))}
        if SPECS[m][0] and added:
            R = np.zeros((len(added), X.shape[1]))
            for i, c in enumerate(added):
                R[i, list(X.columns).index(c)] = 1
            w = res.wald_test(R, scalar=True, use_f=False)
            r["added_terms"] = len(added)
            r["hac_wald_p_added"] = float(w.pvalue)
            v = vif_of(X, [c for c in added if c.startswith("net_load") or c in ("sin1", "cos1", "nonbusiness")])
            r["max_vif_added"] = max(v.values()) if v else np.nan
            r["vif_detail"] = "; ".join(f"{k} {x:.1f}" for k, x in v.items())
        if des.ramp:
            r["coef_ramp_z"] = float(res.params[f"{des.ramp}_z"])
            r["p_ramp_z"] = float(res.pvalues[f"{des.ramp}_z"])
        rows.append(r)
    # how much of the realised ramp is just time of day and season?
    det = deterministic_terms(d)
    for ramp in ("net_load_ramp", "net_load_fc_ramp"):
        Xd = sm.add_constant(det.to_numpy())
        rows.append({"model": f"R² of {ramp} on hour + calendar + season", "information": "diagnostic",
                     "r2": sm.OLS(d[ramp].to_numpy(), Xd).fit().rsquared})
    rows.append({"model": "corr(realised ramp, forecast ramp)", "information": "diagnostic",
                 "r2": float(d["net_load_ramp"].corr(d["net_load_fc_ramp"]))})
    return rows


# --------------------------------------------------------------------------


def main() -> int:
    before = {p.name: sha(p) for p in PROTECTED}
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(start="2025-01-01", end="2026-01-01").section("rsm")
    p25 = pd.read_parquet(ROOT / "data" / "processed" / W25 / "panel.parquet")
    p24 = pd.read_parquet(ROOT / "data" / "processed" / W24 / "panel.parquet")

    worst = verify_baseline(p25, cfg)
    print(f"baseline M2 vs report forward predictions: max |diff| = {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("baseline does not reproduce the report's M2; stopping")

    fwd, yr, ins, coefs, checks = [], [], [], [], []
    for response in RESPONSES:
        for zone in ZONES:
            d = analysis.prepare(p25, zone, response, cfg).dropna(subset=NEEDED).reset_index(drop=True)
            t = analysis.prepare(p24, zone, response, cfg).dropna(subset=NEEDED).reset_index(drop=True)
            checks.append({"zone": zone, "response": response, "rows_2025": len(d),
                           "rows_dropped_2025": int((p25.zone == zone).sum() - len(d)),
                           "no_leakage": verify_no_leakage(d, response)})
            pred, cf, _ = forward_run(d, response)
            fwd.append(summarise(pred, "forward 2025").assign(zone=zone, response=response))
            coefs.append(cf.assign(zone=zone, response=response, sample="forward fold"))
            # B: fit 2024, predict 2025, same 2025 observations for every model
            ypred = pd.DataFrame({"day": d["operating_date"].astype(str).to_numpy(), "y": d[response].to_numpy(float),
                                  "train_mean": t[response].mean(),
                                  "M0": d["hour_beginning"].map(t.groupby("hour_beginning")[response].mean()).to_numpy()})
            for m in SPECS:
                des, res = fit(t, m, response)
                ypred[m] = predict(des, res, d)
                if des.ramp:
                    coefs.append(pd.DataFrame([{"fold": "fit 2024", "model": m,
                                                "coef_ramp_z": float(res.params[f"{des.ramp}_z"]),
                                                "zone": zone, "response": response, "sample": "2024 full"}]))
            yr.append(summarise(ypred, "fit 2024 -> 2025").assign(zone=zone, response=response))
            ins += [dict(r, zone=zone, response=response) for r in insample_and_redundancy(d, response)]

    fwd, yr = pd.concat(fwd, ignore_index=True), pd.concat(yr, ignore_index=True)
    ins, coefs = pd.DataFrame(ins), pd.concat(coefs, ignore_index=True)
    checks = pd.DataFrame(checks)
    for name, df in (("forward_metrics", fwd), ("year_ahead_metrics", yr), ("insample_redundancy", ins),
                     ("ramp_coefficients", coefs), ("checks", checks)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    verdict = []
    for response in RESPONSES:
        for m in SPECS:
            if m == "M2":
                continue
            f = fwd[(fwd.response == response) & (fwd.model == m)].set_index("zone")
            y = yr[(yr.response == response) & (yr.model == m)].set_index("zone")
            material = (f.skill_gain_vs_M2.median() >= CRIT["material_skill"]
                        or -f.mae_change_vs_M2.median() >= CRIT["material_mae"])
            zone_stable = ((f.folds_beating_M2 >= CRIT["stable_folds"]) & (f.dm_p_sq_vs_M2 < CRIT["dm_alpha"])
                           & (f.skill_gain_vs_M2 > 0) & (y.skill_gain_vs_M2 > 0))
            row = {"response": response, "model": m, "information": INFO[m],
                   "fwd_skill_gain_median": f.skill_gain_vs_M2.median(),
                   "fwd_skill_gain_range": f"{f.skill_gain_vs_M2.min():+.3f} to {f.skill_gain_vs_M2.max():+.3f}",
                   "fwd_mae_change_median": f.mae_change_vs_M2.median(),
                   "fwd_rmse_change_median": f.rmse_change_vs_M2.median(),
                   "folds_beating_M2": ",".join(str(int(v)) for v in f.folds_beating_M2),
                   "yr_skill_gain_range": f"{y.skill_gain_vs_M2.min():+.3f} to {y.skill_gain_vs_M2.max():+.3f}",
                   "material": bool(material), "zones_stable": int(zone_stable.sum()),
                   "stable": bool(zone_stable.sum() >= CRIT["zones"])}
            if SPECS[m][3]:
                c = coefs[(coefs.response == response) & (coefs.model == m)]
                row["ramp_sign_share_expected"] = float((np.sign(c.coef_ramp_z) == EXPECTED_RAMP_SIGN[response]).mean())
                vifs = ins[(ins.response == response) & (ins.model == m)]["max_vif_added"]
                row["max_vif_added"] = float(vifs.max())
                row["interpretable"] = bool(row["ramp_sign_share_expected"] >= CRIT["sign_share"]
                                            and row["max_vif_added"] < CRIT["vif"])
            verdict.append(row)
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(OUT / "verdict.csv", index=False)
    inc = []
    for response in RESPONSES:
        for m in ("M2+C+Rrt", "M2+C+Rfc"):
            f = fwd[(fwd.response == response) & (fwd.model == m)].set_index("zone")
            y = yr[(yr.response == response) & (yr.model == m)].set_index("zone")
            ok = ((f.folds_beating_M2C >= CRIT["stable_folds"]) & (f.dm_p_sq_vs_M2C < CRIT["dm_alpha"])
                  & (f.skill_gain_vs_M2C > 0) & (y.skill_gain_vs_M2C > 0))
            inc.append({"response": response, "model": m, "reference": "M2+C",
                        "fwd_gain_vs_M2C_range": f"{f.skill_gain_vs_M2C.min():+.3f} to {f.skill_gain_vs_M2C.max():+.3f}",
                        "fwd_gain_vs_M2C_median": f.skill_gain_vs_M2C.median(),
                        "folds_beating_M2C": ",".join(str(int(v)) for v in f.folds_beating_M2C),
                        "yr_gain_vs_M2C_range": f"{y.skill_gain_vs_M2C.min():+.3f} to {y.skill_gain_vs_M2C.max():+.3f}",
                        "material": bool(f.skill_gain_vs_M2C.median() >= CRIT["material_skill"]),
                        "zones_stable": int(ok.sum()), "stable": bool(ok.sum() >= CRIT["zones"])})
    inc = pd.DataFrame(inc)
    inc.to_csv(OUT / "verdict_ramp_beyond_calendar.csv", index=False)

    after = {p.name: sha(p) for p in PROTECTED}
    assert after == before, "a protected report file changed"
    (OUT / "protected_hashes.txt").write_text("".join(f"{k} {v}\n" for k, v in after.items()))
    pd.set_option("display.width", 250)
    print(checks.to_string(index=False))
    print(verdict.round(3).to_string(index=False))
    print(inc.round(3).to_string(index=False))
    print("protected files unchanged:", after == before)
    return 0


if __name__ == "__main__":
    sys.exit(main())
