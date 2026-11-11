"""NCSS Chapter 330's worked example, reproduced to the printed precision.

The Odor dataset (15 runs of a three-factor Box-Behnken design, pages 330-4
to 330-14 of the NCSS documentation) and every number NCSS prints for it.
This is the regression fixture for the RSM engine: if a change moves any
of these, the change is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import rsm

ODOR = pd.DataFrame(
    [(66, -1, -1, 0), (58, -1, 0, -1), (65, 0, -1, -1), (-31, 0, 0, 0), (39, 1, -1, 0),
     (17, 1, 0, -1), (7, 0, 1, -1), (-35, 0, 0, 0), (43, -1, 1, 0), (-5, -1, 0, 1),
     (43, 0, -1, 1), (-26, 0, 0, 0), (49, 1, 1, 0), (-40, 1, 0, 1), (-22, 0, 1, 1)],
    columns=["Odor", "Temp", "Ratio", "Height"],
)
FACTORS = ["Temp", "Ratio", "Height"]


@pytest.fixture(scope="module")
def fit():
    return rsm.fit_rsm(ODOR, "Odor", FACTORS, cov_type="nonrobust")


@pytest.fixture(scope="module")
def lof(fit):
    return rsm.lack_of_fit(fit, bins=None)  # exact duplicates, as NCSS does


def test_data_is_already_coded(fit):
    # NCSS does not code the factors; the example's data were coded -1/0/+1 by hand
    assert np.allclose(fit.coding.center, 0) and np.allclose(fit.coding.half_range, 1)
    assert ODOR["Odor"].mean() == pytest.approx(15.2)


def test_r_squared(fit):
    assert fit.ols.rsquared == pytest.approx(0.881989, abs=5e-7)


# Estimation section, page 330-10: coefficient, SE, t, p, last R-squared,
# and SE under "Estimation Using Pure Error".
ESTIMATION = {
    "Temp":         (-12.125,   7.947353, -1.53, 0.187613, 0.054938, 1.594261),
    "Ratio":        (-17.0,     7.947353, -2.14, 0.085417, 0.107995, 1.594261),
    "Height":       (-21.375,   7.947353, -2.69, 0.043321, 0.170733, 1.594261),
    "Temp^2":       (32.08333, 11.69819,   2.74, 0.040667, 0.177530, 2.346688),
    "Ratio^2":      (47.83333, 11.69819,   4.09, 0.009457, 0.394616, 2.346688),
    "Height^2":     (6.083333, 11.69819,   0.52, 0.625242, 0.006383, 2.346688),
    "Temp*Ratio":   (8.25,     11.23925,   0.73, 0.495884, 0.012717, 2.254625),
    "Temp*Height":  (1.5,      11.23925,   0.13, 0.899034, 0.000420, 2.254625),
    "Ratio*Height": (-1.75,    11.23925,  -0.16, 0.882357, 0.000572, 2.254625),
}
PURE_ERROR_P = {"Temp": 0.016853, "Ratio": 0.008680, "Height": 0.005517, "Temp^2": 0.005307,
                "Ratio^2": 0.002398, "Height^2": 0.122137, "Temp*Ratio": 0.067241,
                "Temp*Height": 0.574315, "Ratio*Height": 0.518860}


def test_estimation(fit, lof):
    t = rsm.coefficient_table(fit, lof)
    assert t.loc["Intercept", "coef"] == pytest.approx(-30.66667, abs=5e-6)
    for term, (coef, se, tval, p, last, se_pe) in ESTIMATION.items():
        row = t.loc[term]
        assert row["coef"] == pytest.approx(coef, abs=5e-6), term
        assert row["se"] == pytest.approx(se, abs=5e-6), term
        assert row["t"] == pytest.approx(tval, abs=5e-3), term
        assert row["p_value"] == pytest.approx(p, abs=5e-7), term
        assert row["last_r2"] == pytest.approx(last, abs=5e-7), term
        assert row["se_pe"] == pytest.approx(se_pe, abs=5e-7), term
        assert row["p_pe"] == pytest.approx(PURE_ERROR_P[term], abs=5e-7), term


def test_sequential_anova(fit, lof):
    a = rsm.sequential_anova(fit, lof)
    expected = {  # page 330-6/7: df, SS, MS, F, p, incremental R^2, F_pe, p_pe
        "regression":  (9, 18881.98, 2097.998, 4.15, 0.065691, 0.881989, 103.18, 0.009635),
        "linear":      (3, 7143.25, 2381.083, 4.71, 0.064071, 0.333666, 117.10, 0.008479),
        "quadratic":   (3, 11445.23, 3815.078, 7.55, 0.026426, 0.534614, 187.63, 0.005306),
        "lin x lin":   (3, 293.5, 97.83334, 0.19, 0.896470, 0.013710, 4.81, 0.176872),
    }
    for src, (df, ss, ms, F, p, r2, Fpe, ppe) in expected.items():
        row = a.loc[src]
        assert row["df"] == df, src
        assert row["ss"] == pytest.approx(ss, abs=5e-3), src
        assert row["ms"] == pytest.approx(ms, abs=5e-3), src
        assert row["F"] == pytest.approx(F, abs=5e-3), src
        assert row["p_value"] == pytest.approx(p, abs=5e-7), src
        assert row["incr_r2"] == pytest.approx(r2, abs=5e-7), src
        assert row["F_pe"] == pytest.approx(Fpe, abs=5e-3), src
        assert row["p_pe"] == pytest.approx(ppe, abs=5e-7), src
    err = {"total error": (5, 2526.417, 505.2833, 0.118011),
           "lack of fit": (3, 2485.75, 828.5833, 0.116111),
           "pure error": (2, 40.66667, 20.33333, 0.001900)}
    for src, (df, ss, ms, r2) in err.items():
        row = a.loc[src]
        assert row["df"] == df and row["ss"] == pytest.approx(ss, abs=5e-4), src
        assert row["ms"] == pytest.approx(ms, abs=5e-4) and row["incr_r2"] == pytest.approx(r2, abs=5e-7), src
    assert a.loc["lack of fit", "F"] == pytest.approx(40.75, abs=5e-3)
    assert a.loc["lack of fit", "p_value"] == pytest.approx(0.024047, abs=5e-7)


def test_factor_anova(fit, lof):
    a = rsm.factor_anova(fit, lof)
    expected = {  # page 330-9: df, last SS, MS, F, p, term R^2, F_pe, p_pe
        "Temp":   (4, 5258.016, 1314.504, 2.60, 0.161334, 0.245605, 64.65, 0.015291),
        "Ratio":  (4, 11044.6, 2761.151, 5.46, 0.045377, 0.515900, 135.79, 0.007324),
        "Height": (4, 3813.016, 953.254, 1.89, 0.251025, 0.178108, 46.88, 0.020994),
    }
    for f, (df, ss, ms, F, p, r2, Fpe, ppe) in expected.items():
        row = a.loc[f]
        assert row["df"] == df, f
        assert row["last_ss"] == pytest.approx(ss, abs=0.05), f  # NCSS prints 11044.6
        assert row["ms"] == pytest.approx(ms, abs=5e-3), f
        assert row["F"] == pytest.approx(F, abs=5e-3), f
        assert row["p_value"] == pytest.approx(p, abs=5e-7), f
        assert row["term_r2"] == pytest.approx(r2, abs=5e-7), f
        assert row["F_pe"] == pytest.approx(Fpe, abs=5e-3), f
        assert row["p_pe"] == pytest.approx(ppe, abs=5e-7), f


def test_optimum_equals_stationary_point(fit):
    # NCSS finds this by numerical search (378 evaluations), page 330-12; for a
    # quadratic with a true minimum it is the stationary point, here outside
    # the design region, which NCSS notes it does not constrain.
    can = rsm.canonical_analysis(fit)
    assert can.kind == "minimum" and not can.inside_coded_cube
    assert can.stationary_natural.to_numpy() == pytest.approx([0.1219125, 0.1995746, 1.770525], abs=5e-7)
    assert can.predicted_at_stationary == pytest.approx(-52.02463, abs=5e-6)


def test_predicted_values(fit):
    # Residuals section, page 330-12: row 1 and the out-of-sample row 16
    assert fit.predict(np.array([[-1, -1, 0]]))[0] == pytest.approx(86.625, abs=1e-9)
    assert fit.predict(np.array([[1, 1, 0]]))[0] == pytest.approx(28.375, abs=1e-9)


def test_no_terms_removed_at_ncss_cutoff_would_differ():
    # NCSS's example runs the full model (0 terms removed) while printing a 0.01
    # cutoff; selection was evidently not enabled. With it enabled, our reading
    # of the five-step procedure removes the two cross terms that each cost
    # well under 0.01 of R^2 and then stops.
    fit = rsm.fit_rsm(ODOR, "Odor", FACTORS, cov_type="nonrobust")
    final, log = rsm.hierarchical_selection(fit, cutoff=0.01)
    assert list(log["removed"])[:2] == ["Temp*Height", "Ratio*Height"]
    assert (log["r2_decrease"] < 0.01).all()


def test_summary_runs_with_exact_duplicates(fit):
    s = rsm.summary_text(fit, lof_bins=None)
    assert "MINIMUM" in s and "pure error" in s
