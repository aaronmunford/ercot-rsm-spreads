"""The RSM engine against surfaces whose answers are known in closed form."""

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import rsm

# y = 10 + 2 z1 - 3 z2 - 4 z1^2 - 2 z2^2 + 1.5 z1 z2   (coded units)
TRUE = {"Intercept": 10.0, "x1": 2.0, "x2": -3.0, "x1^2": -4.0, "x2^2": -2.0, "x1*x2": 1.5}
B_TRUE = np.array([[-4.0, 0.75], [0.75, -2.0]])
b_TRUE = np.array([2.0, -3.0])


def quad_frame(n=4000, noise=0.05, seed=1, coefs=TRUE, lo=(10, 0.2), hi=(30, 0.8)):
    """Natural-unit factors on known ranges, response from coded-unit coefficients."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n, 2))
    # pin the range so coding is exact: first two rows hit the corners
    x[0], x[1] = lo, hi
    lo, hi = np.array(lo), np.array(hi)
    z = (x - (lo + hi) / 2) / ((hi - lo) / 2)
    y = (coefs["Intercept"] + coefs["x1"] * z[:, 0] + coefs["x2"] * z[:, 1]
         + coefs["x1^2"] * z[:, 0] ** 2 + coefs["x2^2"] * z[:, 1] ** 2
         + coefs["x1*x2"] * z[:, 0] * z[:, 1] + rng.normal(0, noise, n))
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})


def test_coding_round_trip_and_range():
    df = quad_frame()
    c = rsm.Coding.from_data(df[["x1", "x2"]])
    z = c.encode(df[["x1", "x2"]])
    assert np.allclose(z.min(axis=0), -1) and np.allclose(z.max(axis=0), 1)
    assert np.allclose(c.decode(z), df[["x1", "x2"]].to_numpy())


def test_quantile_coding_lets_tails_fall_outside():
    df = quad_frame()
    c = rsm.Coding.from_data(df[["x1", "x2"]], method="quantile", quantiles=(0.05, 0.95))
    z = c.encode(df[["x1", "x2"]])
    assert z.min() < -1 and z.max() > 1


def test_flat_factor_is_refused():
    df = quad_frame()
    df["x2"] = 3.0
    with pytest.raises(ValueError, match="no spread"):
        rsm.Coding.from_data(df[["x1", "x2"]])


def test_design_matrix_columns_and_groups():
    X, g = rsm.design_matrix(np.zeros((3, 2)), ("a", "b"), order=3)
    assert list(X.columns) == ["Intercept", "a", "b", "a^2", "a*b", "b^2",
                               "a^3", "a^2*b", "a*b^2", "b^3"]
    assert g["a*b"] == "lin x lin" and g["b^2"] == "quadratic"
    assert g["a^2*b"] == "lin x quad" and g["a^3"] == "cubic"


def test_recovers_coefficients_and_B():
    fit = rsm.fit_rsm(quad_frame(), "y", ["x1", "x2"], cov_type="nonrobust")
    for name, v in TRUE.items():
        assert fit.params[name] == pytest.approx(v, abs=0.02)
    assert np.allclose(fit.B, B_TRUE, atol=0.02)
    assert np.allclose(fit.b, b_TRUE, atol=0.02)


def test_stationary_point_matches_closed_form():
    fit = rsm.fit_rsm(quad_frame(), "y", ["x1", "x2"], cov_type="nonrobust")
    can = rsm.canonical_analysis(fit)
    zs = -0.5 * np.linalg.solve(B_TRUE, b_TRUE)
    assert np.allclose(can.stationary_coded, zs, atol=0.01)
    assert can.kind == "maximum" and can.inside_coded_cube
    assert np.allclose(np.sort(can.eigenvalues), np.sort(np.linalg.eigvalsh(B_TRUE)), atol=0.02)
    # natural units = decode(coded), and the prediction there equals y_s
    assert np.allclose(can.stationary_natural.to_numpy(), fit.coding.decode(zs), atol=0.05)
    assert can.predicted_at_stationary == pytest.approx(
        fit.predict(can.stationary_natural.to_numpy()[None, :])[0], abs=1e-9)


@pytest.mark.parametrize("coefs,kind", [
    ({**TRUE, "x1^2": 4.0, "x2^2": 2.0, "x1*x2": -1.5}, "minimum"),
    ({**TRUE, "x1^2": 4.0, "x2^2": -2.0, "x1*x2": 0.0}, "saddle"),
    ({**TRUE, "x1^2": -4.0, "x2^2": -0.01, "x1*x2": 0.0}, "ridge"),
])
def test_classification(coefs, kind):
    fit = rsm.fit_rsm(quad_frame(coefs=coefs, noise=0.001), "y", ["x1", "x2"], cov_type="nonrobust")
    assert rsm.canonical_analysis(fit).kind == kind


def test_coding_does_not_change_predictions():
    """A quadratic in coded units is the same function as one in natural units."""
    df = quad_frame()
    a = rsm.fit_rsm(df, "y", ["x1", "x2"], coding_method="range", cov_type="nonrobust")
    b = rsm.fit_rsm(df, "y", ["x1", "x2"], coding_method="quantile", cov_type="nonrobust")
    probe = df[["x1", "x2"]].sample(50, random_state=0).to_numpy()
    assert np.allclose(a.predict(probe), b.predict(probe), atol=1e-8)
    za = rsm.canonical_analysis(a).stationary_natural
    zb = rsm.canonical_analysis(b).stationary_natural
    assert np.allclose(za, zb, atol=1e-6)


def test_anova_partitions_total_ss():
    fit = rsm.fit_rsm(quad_frame(), "y", ["x1", "x2"], cov_type="nonrobust")
    a = rsm.sequential_anova(fit)
    sst = ((fit.y - fit.y.mean()) ** 2).sum()
    assert a.loc["regression", "ss"] + a.loc["total error", "ss"] == pytest.approx(sst, rel=1e-9)
    assert a.loc[["linear", "quadratic", "lin x lin"], "ss"].sum() == pytest.approx(a.loc["regression", "ss"])
    assert a.loc[["regression", "total error"], "incr_r2"].sum() == pytest.approx(1.0)


def test_hac_changes_standard_errors_not_estimates():
    df = quad_frame()
    a = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    b = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="HAC", hac_maxlags=24)
    assert np.allclose(a.params, b.params)
    assert not np.allclose(a.ols.bse, b.ols.bse)


def test_lack_of_fit_detects_a_cubic_and_passes_a_quadratic():
    rng = np.random.default_rng(3)
    # replicated grid: 7x7 design points, 30 replicates each
    g = np.linspace(-1, 1, 7)
    z = np.array([(a, b) for a in g for b in g for _ in range(30)])
    quad = 1 + z[:, 0] - z[:, 1] ** 2 + rng.normal(0, 0.1, len(z))
    cubic = quad + 3 * z[:, 0] ** 3
    for y, should_reject in ((quad, False), (cubic, True)):
        df = pd.DataFrame({"x1": z[:, 0], "x2": z[:, 1], "y": y})
        fit = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
        lof = rsm.lack_of_fit(fit, bins=7)
        assert lof["cells"] == 49
        assert (lof["p_value"] < 0.001) is should_reject


def test_lack_of_fit_matches_classical_formula_on_exact_replicates():
    rng = np.random.default_rng(5)
    g = np.linspace(-1, 1, 5)
    z = np.array([(a, b) for a in g for b in g for _ in range(4)])
    y = 2 + z[:, 0] + 0.5 * z[:, 0] ** 3 + rng.normal(0, 0.2, len(z))
    df = pd.DataFrame({"x1": z[:, 0], "x2": z[:, 1], "y": y})
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    lof = rsm.lack_of_fit(fit, bins=5)
    cell_mean = df.groupby(["x1", "x2"])["y"].transform("mean")
    ss_pe = float(((df["y"] - cell_mean) ** 2).sum())
    assert lof["ss_pe"] == pytest.approx(ss_pe, rel=1e-9)
    assert lof["ss_lof"] == pytest.approx(fit.ols.ssr - ss_pe, rel=1e-9)
    assert (lof["df_lof"], lof["df_pe"]) == (25 - 6, len(z) - 25)


def test_lack_of_fit_never_negative_on_binned_data():
    # Coarse cells over a steep surface: the classical split would go negative here.
    df = quad_frame(n=5000, noise=0.05)
    df["y"] += 20 * (df["x1"] - 20) / 10
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    lof = rsm.lack_of_fit(fit, bins=3)
    assert lof["ss_lof"] >= 0 and lof["F"] >= 0 and lof["p_value"] > 0.001


def test_ridge_path_is_on_the_sphere_and_is_the_constrained_max():
    fit = rsm.fit_rsm(quad_frame(coefs={**TRUE, "x1^2": 4.0, "x2^2": -2.0, "x1*x2": 0.5}),
                      "y", ["x1", "x2"], cov_type="nonrobust")  # saddle: max is on the boundary
    path = rsm.ridge_analysis(fit, [0, 0.25, 0.5, 1.0], direction="max")
    theta = np.linspace(0, 2 * np.pi, 20000)
    for _, row in path.iterrows():
        z = row[["x1_coded", "x2_coded"]].to_numpy(float)
        assert np.linalg.norm(z) == pytest.approx(row["radius"], abs=1e-6)
        if row["radius"] > 0:
            circle = row["radius"] * np.column_stack([np.cos(theta), np.sin(theta)])
            assert row["predicted"] >= fit.predict_coded(circle).max() - 1e-6
    # predicted response rises along the ascent path
    assert path["predicted"].is_monotonic_increasing


def test_ridge_min_direction_descends():
    fit = rsm.fit_rsm(quad_frame(), "y", ["x1", "x2"], cov_type="nonrobust")
    path = rsm.ridge_analysis(fit, [0, 0.5, 1.0], direction="min")
    assert path["predicted"].is_monotonic_decreasing


def test_blocks_shift_intercept_only():
    df = quad_frame()
    df["regime"] = np.where(np.arange(len(df)) % 2, "post", "pre")
    df.loc[df.regime == "post", "y"] += 5.0
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], block_cols=["regime"], cov_type="nonrobust")
    assert fit.params["regime_pre"] == pytest.approx(-5.0, abs=0.02)
    assert fit.params["Intercept"] == pytest.approx(15.0, abs=0.02)  # baseline = 'post'
    assert np.allclose(fit.B, B_TRUE, atol=0.02)
    lof = rsm.lack_of_fit(fit, bins=6)
    assert lof["p_value"] > 0.001  # blocks are respected when forming replicates
    a = rsm.sequential_anova(fit)
    assert a.loc[["blocks", "regression", "total error"], "incr_r2"].sum() == pytest.approx(1.0)


def test_hull_masks_unsupported_region():
    # strongly correlated factors: the off-diagonal corners hold no data
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, 2000)
    df = pd.DataFrame({"x1": t, "x2": t + rng.normal(0, 0.05, 2000)})
    df["y"] = df.x1 + df.x2
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    _, _, yhat = rsm.surface_grid(fit, n=40)
    assert np.isnan(yhat[0, -1]) and np.isnan(yhat[-1, 0])  # corners off the diagonal
    assert not np.isnan(yhat[20, 20])


def test_blocked_cv_prefers_the_true_order():
    df = quad_frame(n=6000, noise=0.5)
    df["fold"] = np.arange(len(df)) % 6
    cv = rsm.blocked_cv(df, "y", ["x1", "x2"], fold_col="fold", orders=(1, 2, 3))
    assert cv.loc[2, "rmse"] < cv.loc[1, "rmse"]
    assert cv.loc[2, "rmse"] <= cv.loc[3, "rmse"] * 1.01


def test_summary_text_runs():
    fit = rsm.fit_rsm(quad_frame(), "y", ["x1", "x2"])
    s = rsm.summary_text(fit)
    assert "MAXIMUM" in s and "lack of fit" in s and "ANOVA by factor" in s


def test_terms_subset_must_be_hierarchical():
    df = quad_frame()
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], terms=["x1", "x2", "x1^2"], cov_type="nonrobust")
    assert fit.polynomial_terms == ["x1", "x2", "x1^2"]
    assert fit.B[1, 1] == 0 and fit.B[0, 1] == 0  # dropped terms read as zero
    with pytest.raises(ValueError, match="not hierarchical"):
        rsm.fit_rsm(df, "y", ["x1", "x2"], terms=["x1", "x1*x2"], cov_type="nonrobust")


def test_power_needs_enough_distinct_values():
    df = quad_frame()
    df["x2"] = np.where(df["x2"] > 0.5, 0.8, 0.2)  # two values: linear only
    with pytest.raises(ValueError, match="distinct values"):
        rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    rsm.fit_rsm(df, "y", ["x1", "x2"], terms=["x1", "x2", "x1^2"], cov_type="nonrobust")


def test_hierarchical_selection_drops_only_what_costs_little():
    # truth has no x2^2 and no interaction
    coefs = {**TRUE, "x2^2": 0.0, "x1*x2": 0.0}
    fit = rsm.fit_rsm(quad_frame(coefs=coefs, noise=0.05), "y", ["x1", "x2"], cov_type="nonrobust")
    final, log = rsm.hierarchical_selection(fit, cutoff=0.001)
    assert set(log["removed"]) == {"x2^2", "x1*x2"}
    assert final.polynomial_terms == ["x1", "x2", "x1^2"]
    assert (log["r2_decrease"] < 0.001).all()
    # a cutoff of zero removes nothing
    assert rsm.hierarchical_selection(fit, cutoff=0.0)[1].empty


def test_hierarchical_selection_never_orphans_a_child():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.uniform(-1, 1, (3000, 2)), columns=["x1", "x2"])
    df["y"] = df.x1 ** 2 + rng.normal(0, 0.01, 3000)  # x1 itself contributes nothing
    fit = rsm.fit_rsm(df, "y", ["x1", "x2"], cov_type="nonrobust")
    final, log = rsm.hierarchical_selection(fit, cutoff=0.01)
    assert "x1" in final.polynomial_terms and "x1^2" in final.polynomial_terms
    rsm.check_hierarchical(final.polynomial_terms, rsm.term_exponents(("x1", "x2"), 2))


# --------------------------------------------------------------------------
# Observational extensions: clustered SEs, block-aware prediction, slices, time folds
# --------------------------------------------------------------------------


def _three_factor_frame(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(-1, 1, (3, n))
    hour = np.arange(n) % 24
    y = 2 + x1 - 0.5 * x2 + 0.3 * x3 + 1.5 * x1 ** 2 + 0.8 * x1 * x2 + 3.0 * (hour >= 18) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "hour": hour, "day": np.arange(n) // 24,
                         "month": [f"2025-{1 + i * 12 // n:02d}" for i in range(n)]})


def test_cluster_covariance_runs_and_differs():
    df = _three_factor_frame()
    a = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="nonrobust")
    b = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="cluster", cluster_col="day")
    assert np.allclose(a.params, b.params)
    assert not np.allclose(a.ols.bse, b.ols.bse)
    with pytest.raises(ValueError, match="cluster_col"):
        rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="cluster")


def test_predict_frame_uses_each_rows_block_and_defaults_to_the_block_mix():
    df = _three_factor_frame()
    df["evening"] = np.where(df["hour"] >= 18, "evening", "other")
    fit = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], block_cols=["evening"], cov_type="nonrobust")
    p = fit.predict_frame(df)
    assert np.corrcoef(p, df["y"])[0, 1] > 0.95
    probe = pd.DataFrame({"x1": [0.0, 0.0], "x2": [0.0, 0.0], "x3": [0.0, 0.0], "evening": ["evening", "other"]})
    gap = np.diff(fit.predict_frame(probe))[0]
    assert gap == pytest.approx(-3.0, abs=0.1)
    # default prediction = surface averaged over the block mix (25% evening)
    share = (df["evening"] == "evening").mean()
    avg = fit.predict(np.zeros((1, 3)))[0]
    assert avg == pytest.approx(fit.predict_frame(probe) @ np.array([share, 1 - share]), abs=1e-9)
    can = rsm.canonical_analysis(fit)
    assert can.predicted_at_stationary == pytest.approx(
        fit.predict(can.stationary_natural.to_numpy()[None, :])[0], abs=1e-9)


def test_slice_grid_matches_the_known_surface():
    df = _three_factor_frame()
    fit = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="nonrobust", coding_method="range")
    X, Y, Z = rsm.slice_grid(fit, "x1", "x2", {"x3": 0.0}, n=25)
    truth = 2 + X - 0.5 * Y + 1.5 * X ** 2 + 0.8 * X * Y  # the evening shift is not in this fit
    resid_sd = np.sqrt(fit.ols.scale)  # includes the unmodelled evening shift (~1.3)
    assert np.nanmax(np.abs(Z - (truth + 3.0 * (df["hour"] >= 18).mean()))) < 0.15 * resid_sd / 0.3
    _, _, Zm = rsm.slice_grid(fit, "x1", "x2", {"x3": 0.0}, data=df, n=25)
    assert np.isnan(Zm).any() and np.isfinite(Zm).any()
    with pytest.raises(ValueError, match="fix every factor"):
        rsm.slice_grid(fit, "x1", "x2", {})


def test_density_support_excludes_the_empty_inside_of_a_curved_cloud():
    # a V-shaped cloud: its convex hull contains the empty region above the V
    rng = np.random.default_rng(4)
    t = rng.uniform(-1, 1, 3000)
    df = pd.DataFrame({"x1": t, "x2": t ** 2 + rng.normal(0, 0.03, 3000), "x3": rng.normal(0, 0.1, 3000)})
    df["y"] = df.x1 + df.x2 + df.x3 + rng.normal(0, 0.1, 3000)
    fit = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="nonrobust", coding_method="range")
    X, Y, hull = rsm.slice_grid(fit, "x1", "x2", {"x3": 0.0}, data=df, n=41)
    _, _, dens = rsm.slice_grid(fit, "x1", "x2", {"x3": 0.0}, data=df, n=41, support="density")
    inside_v = (np.abs(X) < 0.3) & (Y > 0.6)  # inside the hull, far from any observation
    assert np.isfinite(hull[inside_v]).any() and np.isnan(dens[inside_v]).all()
    on_curve = np.abs(Y - X ** 2) < 0.02
    assert np.isfinite(dens[on_curve]).mean() > 0.8
    # density support is never wider than the hull
    assert not (np.isfinite(dens) & np.isnan(hull)).any()
    sup = rsm.slice_support(fit, "x1", "x2", {"x3": 0.0}, df)
    assert sup.dtype == bool and 0 < sup.sum() < len(df)
    with pytest.raises(ValueError, match="support"):
        rsm.slice_grid(fit, "x1", "x2", {"x3": 0.0}, data=df, support="alpha")


def test_forward_folds_never_train_on_the_test_month_or_later():
    months = pd.Series([f"2025-{m:02d}" for m in range(1, 13) for _ in range(5)])
    folds = rsm.time_folds(months, "forward", min_train=3)
    assert [f[2] for f in folds] == [f"2025-{m:02d}" for m in range(4, 13)]
    for train, test, month in folds:
        assert months[train].max() < month and set(months[test]) == {month}
    lomo = rsm.time_folds(months, "lomo")
    assert len(lomo) == 12 and all(months[tr].max() > m for tr, _, m in lomo if m < "2025-12")


def test_design_diagnostics_flags_collinear_factors():
    df = _three_factor_frame()
    df["x2"] = df["x1"] + np.random.default_rng(1).normal(0, 0.05, len(df))
    fit = rsm.fit_rsm(df, "y", ["x1", "x2", "x3"], cov_type="nonrobust")
    dd = rsm.design_diagnostics(fit)
    assert dd.loc["x1", "vif"] > 50 and dd.loc["x3", "vif"] < 5
    assert dd.attrs["condition_number"] > 10
