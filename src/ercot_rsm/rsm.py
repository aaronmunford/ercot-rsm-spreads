"""Second-order response surface regression (RSM). Pure functions, no I/O.

Follows the classical treatment (Box & Wilson 1951; Myers, Montgomery &
Anderson-Cook 2016) and reproduces the reports of NCSS Chapter 330,
"Response Surface Regression": hierarchical polynomial models, sequential
ANOVA by term group, ANOVA by factor, estimation with last R-squared,
lack of fit against pure error, the optimum (here via canonical analysis),
and hierarchical backward elimination with an R-squared cutoff. The NCSS
worked example (Odor data) is reproduced to the printed precision in
``tests/test_ncss_odor.py``. Ridge analysis is an addition NCSS lacks.

The data here are observational hourly time series, not a designed
experiment, and two departures follow from that. Both are deliberate and
both belong in the report's methodology section:

* Inference can use HAC (Newey-West) covariance. Hourly residuals are
  strongly autocorrelated, so OLS standard errors are far too small.
* There are no true replicates, so the lack-of-fit test uses
  near-replicates: observations sharing a cell of a grid laid over the
  coded factor space, posed as a nested model comparison (see
  ``lack_of_fit`` for why the classical split fails on binned data).

Notation. With k coded factors z, the second-order model is

    y = b0 + z'b + z'Bz + e

where b holds the linear coefficients and B is symmetric with the pure
quadratic coefficients b_ii on its diagonal and half of each interaction
coefficient b_ij / 2 off it. The stationary point is z_s = -B^{-1} b / 2,
and the eigenvalues of B classify it.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.spatial import Delaunay, cKDTree

# --------------------------------------------------------------------------
# Factor coding
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Coding:
    """Linear map between natural units and coded units.

    ``z = (x - center) / half_range``. With ``method="range"`` the observed
    min and max map to -1 and +1, which is the convention for designed
    experiments. On observational data a few extreme hours then set the
    scale, so ``method="quantile"`` maps chosen quantiles to -1 and +1
    instead and lets the tails fall outside.
    """

    names: tuple[str, ...]
    center: np.ndarray
    half_range: np.ndarray

    @classmethod
    def from_data(
        cls,
        X: pd.DataFrame,
        method: str = "range",
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> Coding:
        if method == "range":
            lo, hi = X.min(), X.max()
        elif method == "quantile":
            lo, hi = X.quantile(quantiles[0]), X.quantile(quantiles[1])
        else:
            raise ValueError(f"unknown coding method {method!r}")
        if (hi <= lo).any():
            flat = list(X.columns[(hi <= lo).to_numpy()])
            raise ValueError(f"factor(s) with no spread: {flat}")
        return cls(
            names=tuple(X.columns),
            center=((hi + lo) / 2).to_numpy(float),
            half_range=((hi - lo) / 2).to_numpy(float),
        )

    def encode(self, X) -> np.ndarray:
        return (np.asarray(X, float) - self.center) / self.half_range

    def decode(self, Z) -> np.ndarray:
        return np.asarray(Z, float) * self.half_range + self.center


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------


def _term_name(names: tuple[str, ...], idx: tuple[int, ...]) -> str:
    counts: dict[int, int] = {}
    for i in idx:
        counts[i] = counts.get(i, 0) + 1
    parts = [names[i] if c == 1 else f"{names[i]}^{c}" for i, c in counts.items()]
    return "*".join(parts)


_POWER = {1: "linear", 2: "quadratic", 3: "cubic"}
_PART = {1: "lin", 2: "quad", 3: "cubic"}


def _term_group(idx: tuple[int, ...]) -> str:
    """NCSS group name: Linear, Quadratic, Cubic, Lin x Lin, Lin x Quad, ..."""
    powers = sorted(Counter(idx).values())
    if len(powers) == 1:
        return _POWER[powers[0]]
    if len(powers) == 2:
        return f"{_PART[powers[0]]} x {_PART[powers[1]]}"
    return "three-way"


def _exponents(k: int, idx: tuple[int, ...]) -> tuple[int, ...]:
    c = Counter(idx)
    return tuple(c.get(i, 0) for i in range(k))


def design_matrix(
    Z: np.ndarray,
    names: tuple[str, ...],
    order: int = 2,
    blocks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Polynomial design matrix in coded factors, plus a term -> group map.

Groups follow NCSS: ``linear``, ``quadratic``, ``cubic`` (pure powers),
    then ``lin x lin``, ``lin x quad`` ... (two-factor cross products), plus
    ``intercept``, ``block`` and ``three-way`` (A*B*C, order 3 only; NCSS
    itself only builds two-factor cross products). Block columns are
    additive shifts in the sense of RSM blocking: they move the intercept
    and are excluded from the canonical analysis.
    """
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")
    Z = np.atleast_2d(np.asarray(Z, float))
    n, k = Z.shape
    if k != len(names):
        raise ValueError("Z columns do not match names")

    cols: dict[str, np.ndarray] = {"Intercept": np.ones(n)}
    groups: dict[str, str] = {"Intercept": "intercept"}

    if blocks is not None and blocks.shape[1]:
        for c in blocks.columns:
            cols[c] = np.asarray(blocks[c], float)
            groups[c] = "block"

    for degree in range(1, order + 1):
        for idx in itertools.combinations_with_replacement(range(k), degree):
            name = _term_name(names, idx)
            cols[name] = np.prod(Z[:, list(idx)], axis=1)
            groups[name] = _term_group(idx)

    return pd.DataFrame(cols), pd.Series(groups)


def term_exponents(names: tuple[str, ...], order: int) -> dict[str, tuple[int, ...]]:
    """Polynomial term name -> exponent vector, e.g. 'A^2*B' -> (2, 1)."""
    k = len(names)
    return {
        _term_name(names, idx): _exponents(k, idx)
        for degree in range(1, order + 1)
        for idx in itertools.combinations_with_replacement(range(k), degree)
    }


def _is_child(e: tuple[int, ...], parent: tuple[int, ...]) -> bool:
    return e != parent and all(a <= b for a, b in zip(e, parent))


def check_hierarchical(terms: list[str], exponents: dict[str, tuple[int, ...]]) -> None:
    """Raise unless every lower-order term implied by a kept term is also kept."""
    kept = {exponents[t] for t in terms}
    by_exp = {v: k for k, v in exponents.items()}
    for t in terms:
        for e in exponents.values():
            if any(e) and _is_child(e, exponents[t]) and e not in kept:
                raise ValueError(f"not hierarchical: {t} needs {by_exp[e]}")


def block_dummies(df: pd.DataFrame, block_cols: list[str] | None,
                  drop_first: bool = True) -> pd.DataFrame | None:
    """Treatment-coded dummies for categorical block columns (first level dropped)."""
    if not block_cols:
        return None
    parts = [
        pd.get_dummies(df[c].astype("category"), prefix=c, drop_first=drop_first, dtype=float)
        for c in block_cols
    ]
    out = pd.concat(parts, axis=1)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass(eq=False)
class RSMFit:
    response: str
    coding: Coding
    order: int
    terms: pd.Series  # design column -> group
    ols: object  # statsmodels RegressionResultsWrapper
    y: np.ndarray
    Z: np.ndarray  # coded factors actually used in the fit
    X: pd.DataFrame  # design matrix actually used in the fit
    block_columns: tuple[str, ...]
    cov_type: str
    exponents: dict = field(default_factory=dict)  # polynomial term -> exponent vector
    cov_kwds: dict = field(default_factory=dict)
    block_sources: tuple[str, ...] = ()  # the categorical columns the block dummies came from
    block_means: dict = field(default_factory=dict)  # dummy column -> share of fitted rows

    @property
    def polynomial_terms(self) -> list[str]:
        return [c for c in self.X.columns if c in self.exponents]

    @property
    def sst(self) -> float:
        """Corrected total sum of squares, the denominator of every R-squared here."""
        return float(((self.y - self.y.mean()) ** 2).sum())

    @property
    def names(self) -> tuple[str, ...]:
        return self.coding.names

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def params(self) -> pd.Series:
        return self.ols.params

    @property
    def b0(self) -> float:
        return float(self.params["Intercept"])

    @property
    def b(self) -> np.ndarray:
        return np.array([self.params.get(n, 0.0) for n in self.names])

    @property
    def B(self) -> np.ndarray:
        if self.order < 2:
            raise ValueError("B is defined only for a second-order model")
        k = self.k
        B = np.zeros((k, k))
        for i in range(k):
            B[i, i] = self.params.get(_term_name(self.names, (i, i)), 0.0)
            for j in range(i + 1, k):
                half = self.params.get(_term_name(self.names, (i, j)), 0.0) / 2
                B[i, j] = B[j, i] = half
        return B

    @property
    def block_offset(self) -> float:
        """Block contribution at the fitted sample's block mix (0 without blocks)."""
        return float(sum(self.params.get(c, 0.0) * m for c, m in self.block_means.items()))

    def predict_coded(self, Z, blocks: pd.DataFrame | None = None) -> np.ndarray:
        """Predict at coded factor values. Without ``blocks`` the block dummies are
        set to their sample shares, i.e. the surface averaged over the block mix
        (every hour of day, both regimes), not one arbitrary baseline block."""
        n = np.atleast_2d(Z).shape[0]
        if blocks is None and self.block_columns:
            blocks = pd.DataFrame({c: np.full(n, self.block_means.get(c, 0.0)) for c in self.block_columns})
        X, _ = design_matrix(Z, self.names, self.order, blocks)
        X = X[self.X.columns]
        return X.to_numpy() @ self.params.to_numpy()

    def predict(self, X_natural, blocks: pd.DataFrame | None = None) -> np.ndarray:
        """Predict at natural-unit factor values (block-averaged unless ``blocks`` given)."""
        return self.predict_coded(self.coding.encode(X_natural), blocks)

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Predict rows of ``df`` using their own factor values and block categories.
        A block level never seen in fitting gets no dummy, i.e. the baseline level."""
        blocks = None
        if self.block_columns:
            d = block_dummies(df, list(self.block_sources), drop_first=False)
            blocks = d.reindex(columns=list(self.block_columns), fill_value=0.0)
        return self.predict(df[list(self.names)].to_numpy(float), blocks)


def fit_rsm(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    *,
    order: int = 2,
    coding: Coding | None = None,
    coding_method: str = "range",
    coding_quantiles: tuple[float, float] = (0.01, 0.99),
    block_cols: list[str] | None = None,
    cov_type: str = "HAC",
    hac_maxlags: int = 24,
    terms: list[str] | None = None,
    cluster_col: str | None = None,
) -> RSMFit:
    """Fit a polynomial response surface by OLS on coded factors.

    ``df`` must be in time order if ``cov_type="HAC"``: the Newey-West
    estimator weights residual autocovariances by lag in row order.
    ``cov_type`` may be ``"HAC"``, ``"HC3"``, ``"cluster"`` (by ``cluster_col``,
    e.g. operating day) or ``"nonrobust"`` (classical OLS, which is what NCSS reports). ``terms`` restricts the polynomial to
    a hierarchical subset of the full ``order`` model.
    """
    if cov_type == "cluster" and cluster_col is None:
        raise ValueError('cov_type="cluster" needs cluster_col')
    cols = list(dict.fromkeys([response, *factors, *(block_cols or []), *([cluster_col] if cluster_col else [])]))
    data = df[cols].dropna().reset_index(drop=True)
    if len(data) == 0:
        raise ValueError("no complete rows to fit")

    if coding is None:
        coding = Coding.from_data(data[factors], coding_method, coding_quantiles)
    Z = coding.encode(data[factors])
    blocks = block_dummies(data, block_cols)
    X, groups = design_matrix(Z, tuple(factors), order, blocks)
    exps = term_exponents(tuple(factors), order)
    if terms is not None:
        unknown = set(terms) - set(exps)
        if unknown:
            raise ValueError(f"unknown terms {sorted(unknown)}")
        check_hierarchical(list(terms), exps)
        X = X[[c for c in X.columns if c not in exps or c in terms]]
        exps = {t: e for t, e in exps.items() if t in terms}
    _check_unique_values(data[factors], exps)
    if len(data) <= X.shape[1]:
        raise ValueError(f"{len(data)} rows cannot fit {X.shape[1]} terms")
    y = data[response].to_numpy(float)
    clusters = pd.factorize(data[cluster_col])[0] if cov_type == "cluster" else None
    res, cov_kwds = _ols(y, X, cov_type, hac_maxlags, clusters)

    return RSMFit(
        response=response,
        coding=coding,
        order=order,
        terms=groups,
        ols=res,
        y=y,
        Z=Z,
        X=X,
        block_columns=tuple(blocks.columns) if blocks is not None else (),
        cov_type=cov_type,
        exponents=exps,
        cov_kwds=cov_kwds,
        block_sources=tuple(block_cols or []) if blocks is not None else (),
        block_means=blocks.mean().to_dict() if blocks is not None else {},
    )


def _ols(y, X, cov_type: str, hac_maxlags: int = 24, groups=None):
    model = sm.OLS(y, X)
    if cov_type == "HAC":
        kw = {"maxlags": hac_maxlags}
        return model.fit(cov_type="HAC", cov_kwds=kw), kw
    if cov_type == "cluster":
        if groups is None:
            raise ValueError("cluster covariance needs groups")
        kw = {"groups": np.asarray(groups)}
        return model.fit(cov_type="cluster", cov_kwds=kw), kw
    if cov_type in ("HC3", "nonrobust"):
        return model.fit(cov_type=cov_type), {}
    raise ValueError(f"unknown cov_type {cov_type!r}")


def _check_unique_values(F: pd.DataFrame, exps: dict[str, tuple[int, ...]]) -> None:
    """NCSS: a factor raised to power k needs at least k + 1 distinct values."""
    for j, name in enumerate(F.columns):
        k = max((e[j] for e in exps.values()), default=0)
        if F[name].nunique() < k + 1:
            raise ValueError(f"{name} has {F[name].nunique()} distinct values; power {k} needs {k + 1}")


def _refit(fit: RSMFit, columns: list[str]) -> RSMFit:
    """Same data and coding, a subset of the design columns."""
    X = fit.X[columns]
    hac = fit.cov_kwds.get("maxlags", 24)
    res, kw = _ols(fit.y, X, fit.cov_type, hac, fit.cov_kwds.get("groups"))
    return RSMFit(
        response=fit.response, coding=fit.coding, order=fit.order,
        terms=fit.terms[columns], ols=res, y=fit.y, Z=fit.Z, X=X,
        block_columns=fit.block_columns, cov_type=fit.cov_type,
        exponents={t: e for t, e in fit.exponents.items() if t in columns},
        cov_kwds=kw, block_sources=fit.block_sources,
        block_means={c: m for c, m in fit.block_means.items() if c in columns},
    )


def _sse(y: np.ndarray, X: pd.DataFrame) -> float:
    coef, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    r = y - X.to_numpy() @ coef
    return float(r @ r)


# --------------------------------------------------------------------------
# ANOVA and lack of fit
# --------------------------------------------------------------------------

_SEQUENCE = ("linear", "quadratic", "cubic", "lin x lin", "lin x quad",
             "quad x quad", "lin x cubic", "three-way")


def _with_pure_error(df: pd.DataFrame, lof: dict | None) -> pd.DataFrame:
    """Append NCSS's second report: the same F tests over the pure-error mean square."""
    if lof is None or not lof.get("df_pe"):
        return df
    ms_pe = lof["ss_pe"] / lof["df_pe"]
    df["F_pe"] = df["ms"] / ms_pe
    df["p_pe"] = stats.f.sf(df["F_pe"], df["df"], lof["df_pe"])
    return df


def sequential_anova(fit: RSMFit, lof: dict | None = None) -> pd.DataFrame:
    """NCSS "Sequential ANOVA": Type I sums of squares by term group.

    Groups enter in the order linear -> quadratic -> cubic -> lin x lin ->
    lin x quad ..., after the intercept and any blocks. Each F uses the
    total-error mean square, so each tests its group given the groups above
    it ("sequential ... should be considered from the bottom up").
    ``incr_r2`` is the R-squared each group adds. Passing ``lack_of_fit``'s
    result appends the lack-of-fit / pure-error rows and the pure-error F
    columns, as NCSS's "Sequential ANOVA Using Pure Error".

    These F tests assume independent errors. On hourly data read them as
    descriptive and use ``robust_term_tests`` for inference.
    """
    y, X, n = fit.y, fit.X, len(fit.y)
    sst = fit.sst
    base = [c for c in X.columns if fit.terms[c] in ("intercept", "block")]
    sse_prev = _sse(y, X[base])
    used = list(base)
    df_err = n - X.shape[1]
    sse_full = float(fit.ols.ssr)
    ms_err = sse_full / df_err

    rows = []
    for group in _SEQUENCE:
        add = [c for c in X.columns if fit.terms[c] == group]
        if not add:
            continue
        used += add
        sse = _sse(y, X[used])
        rows.append({"source": group, "df": len(add), "ss": sse_prev - sse})
        sse_prev = sse

    reg_ss = sum(r["ss"] for r in rows)
    reg_df = sum(r["df"] for r in rows)
    rows.insert(0, {"source": "regression", "df": reg_df, "ss": reg_ss})
    n_blocks = len(base) - 1
    if n_blocks:
        # Not in NCSS (no blocking there); shown so the table still partitions SST.
        rows.insert(0, {"source": "blocks", "df": n_blocks, "ss": sst - _sse(y, X[base])})
    out = pd.DataFrame(rows).set_index("source")
    out["ms"] = out["ss"] / out["df"]
    out["F"] = out["ms"] / ms_err
    out["p_value"] = stats.f.sf(out["F"], out["df"], df_err)
    out["incr_r2"] = out["ss"] / sst
    out = _with_pure_error(out, lof)

    err = [{"source": "total error", "df": df_err, "ss": sse_full, "ms": ms_err,
            "incr_r2": sse_full / sst}]
    if lof is not None and lof.get("df_pe"):
        err.append({"source": "lack of fit", "df": lof["df_lof"], "ss": lof["ss_lof"],
                    "ms": lof["ss_lof"] / lof["df_lof"], "F": lof["F"], "p_value": lof["p_value"],
                    "incr_r2": lof["ss_lof"] / sst})
        err.append({"source": "pure error", "df": lof["df_pe"], "ss": lof["ss_pe"],
                    "ms": lof["ss_pe"] / lof["df_pe"], "incr_r2": lof["ss_pe"] / sst})
    return pd.concat([out, pd.DataFrame(err).set_index("source")])


def factor_anova(fit: RSMFit, lof: dict | None = None) -> pd.DataFrame:
    """NCSS "ANOVA": for each factor, drop every term that contains it.

    ``last_ss`` is the regression sum of squares lost; ``term_r2`` is the
    R-squared lost. Not sequential: each factor is tested given all the
    others. Dropping all of a factor's terms keeps the model hierarchical.
    """
    df_err = len(fit.y) - fit.X.shape[1]
    sse_full = float(fit.ols.ssr)
    ms_err = sse_full / df_err
    rows = []
    for j, name in enumerate(fit.names):
        drop = [t for t, e in fit.exponents.items() if e[j] > 0]
        keep = [c for c in fit.X.columns if c not in drop]
        ss = _sse(fit.y, fit.X[keep]) - sse_full
        rows.append({"factor": name, "df": len(drop), "last_ss": ss})
    out = pd.DataFrame(rows).set_index("factor")
    out["ms"] = out["last_ss"] / out["df"]
    out["F"] = out["ms"] / ms_err
    out["p_value"] = stats.f.sf(out["F"], out["df"], df_err)
    out["term_r2"] = out["last_ss"] / fit.sst
    return _with_pure_error(out, lof)


def robust_term_tests(fit: RSMFit) -> pd.DataFrame:
    """Wald test that each term group is jointly zero, using the fit's covariance.

    Under ``cov_type="HAC"`` this is the autocorrelation-robust counterpart
    of the sequential ANOVA's F tests (but each group is tested given all
    others, not sequentially).
    """
    rows = []
    cols = list(fit.X.columns)
    for group in _SEQUENCE:
        members = [c for c in cols if fit.terms[c] == group]
        if not members:
            continue
        R = np.zeros((len(members), len(cols)))
        for r, c in enumerate(members):
            R[r, cols.index(c)] = 1.0
        w = fit.ols.wald_test(R, scalar=True, use_f=False)
        rows.append({"group": group, "df": len(members),
                     "chi2": float(w.statistic), "p_value": float(w.pvalue)})
    return pd.DataFrame(rows).set_index("group")


def near_replicate_cells(Z: np.ndarray, bins: int) -> np.ndarray:
    """Integer cell id for each row on a ``bins``^k grid over [-1, 1]^k.

    Points outside [-1, 1] (possible under quantile coding) fall in the edge cells.
    """
    edges = np.linspace(-1, 1, bins + 1)[1:-1]
    idx = np.stack([np.digitize(Z[:, j], edges) for j in range(Z.shape[1])], axis=1)
    return np.ravel_multi_index(idx.T, (bins,) * Z.shape[1])


def lack_of_fit(fit: RSMFit, bins: int | None = 8) -> dict:
    """Lack-of-fit F test against pure error, as a nested model comparison.

    ``bins=None`` uses exact duplicate factor settings as replicates, which
    is what NCSS does ("automatically scans the data for duplicates").
    Observational data has almost none, so ``bins=k`` instead treats rows
    sharing a cell of a k^d grid over the coded factors as near-replicates.

    Binned near-replicates break the classical identity SSE = SS_pe +
    SS_lof: the fitted surface still varies inside a cell, so within-cell
    variation of y can exceed the model's own SSE and the classical lack-of-fit
    sum of squares goes negative. So the test is posed as a nested
    comparison: augment the model with one indicator per occupied cell
    (block-specific when blocks are present) and F-test the augmentation:

        SS_lof = SSE(model) - SSE(model + cells),  df_lof = rank gain
        SS_pe  = SSE(model + cells),                df_pe  = n - rank

    This is never negative, and when cells are exact design points (the
    polynomial is constant within each cell) it is the classical test exactly.
    """
    if bins is None:
        cells = pd.DataFrame(np.round(fit.Z, 12)).astype(str).agg("|".join, axis=1).to_numpy()
    else:
        cells = near_replicate_cells(fit.Z, bins)
    if fit.block_columns:
        key = pd.DataFrame(fit.X[list(fit.block_columns)].to_numpy()).astype(str).agg("|".join, axis=1)
        cells = pd.Series(cells).astype(str).str.cat(key, sep="#").to_numpy()
    codes, uniques = pd.factorize(cells)
    n_cells = len(uniques)
    D = np.zeros((len(codes), n_cells))
    D[np.arange(len(codes)), codes] = 1.0

    X = fit.X.to_numpy()
    aug = np.hstack([X, D])
    rank_x = np.linalg.matrix_rank(X)
    rank_aug = np.linalg.matrix_rank(aug)
    coef, *_ = np.linalg.lstsq(aug, fit.y, rcond=None)
    resid = fit.y - aug @ coef
    ss_pe = float(resid @ resid)
    sse = float(fit.ols.ssr)
    ss_lof = max(sse - ss_pe, 0.0)  # clip float noise only; nesting guarantees >= 0
    n = len(fit.y)
    df_lof = int(rank_aug - rank_x)
    df_pe = int(n - rank_aug)
    out = {"bins": bins, "cells": n_cells, "df_lof": df_lof, "df_pe": df_pe,
           "ss_lof": ss_lof, "ss_pe": ss_pe, "F": np.nan, "p_value": np.nan}
    if df_lof > 0 and df_pe > 0:
        F = (ss_lof / df_lof) / (ss_pe / df_pe)
        out.update(F=F, p_value=float(stats.f.sf(F, df_lof, df_pe)))
    return out


def hierarchical_selection(fit: RSMFit, cutoff: float = 0.01) -> tuple[RSMFit, pd.DataFrame]:
    """NCSS's hierarchical backward elimination with an R-squared cutoff.

    1. Start from the full model.
    2. Mark the terms that no remaining term needs for hierarchy.
    3. For each, compute how much R-squared falls if it is removed.
    4. Remove the one that costs least, never to return; go to 2.
    5. Stop when every removable term would cost at least ``cutoff``.

    The criterion is an R-squared amount, not a p-value, so it does not
    degenerate as n grows. With ~8,760 hourly rows every term is
    "significant", and this is the parsimony tool that still works.
    """
    current = fit
    log = []
    while True:
        terms = current.polynomial_terms
        exps = current.exponents
        removable = [t for t in terms
                     if not any(_is_child(exps[t], exps[u]) for u in terms if u != t)]
        if not removable:
            break
        sse_full = float(current.ols.ssr)
        cost = {t: (_sse(current.y, current.X.drop(columns=[t])) - sse_full) / current.sst
                for t in removable}
        t_min = min(cost, key=cost.get)
        if cost[t_min] >= cutoff:
            break
        log.append({"removed": t_min, "r2_decrease": cost[t_min]})
        current = _refit(current, [c for c in current.X.columns if c != t_min])
    return current, pd.DataFrame(log, columns=["removed", "r2_decrease"])


# --------------------------------------------------------------------------
# Canonical analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Canonical:
    stationary_coded: np.ndarray
    stationary_natural: pd.Series
    predicted_at_stationary: float
    eigenvalues: np.ndarray  # descending
    eigenvectors: np.ndarray  # columns, matching eigenvalues
    kind: str  # maximum | minimum | saddle | ridge
    inside_coded_cube: bool


def canonical_analysis(fit: RSMFit, ridge_tol: float = 0.05) -> Canonical:
    """Stationary point and eigen-classification of the fitted quadratic.

    ``kind`` is ``ridge`` when the smallest |eigenvalue| is below
    ``ridge_tol`` times the largest. The surface is then nearly flat along
    that axis, so the stationary point is poorly determined (and may be far
    outside the data) even though the fitted surface is well behaved.
    """
    if fit.order != 2:
        raise ValueError("canonical analysis applies to a second-order fit")
    B, b = fit.B, fit.b
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    scale = np.max(np.abs(evals)) if np.any(evals) else 0.0
    if scale == 0 or np.min(np.abs(evals)) < ridge_tol * scale:
        kind = "ridge"
    elif np.all(evals < 0):
        kind = "maximum"
    elif np.all(evals > 0):
        kind = "minimum"
    else:
        kind = "saddle"

    zs = -0.5 * np.linalg.lstsq(B, b, rcond=None)[0]
    ys = fit.b0 + fit.block_offset + 0.5 * float(zs @ b)
    return Canonical(
        stationary_coded=zs,
        stationary_natural=pd.Series(fit.coding.decode(zs), index=list(fit.names)),
        predicted_at_stationary=ys,
        eigenvalues=evals,
        eigenvectors=evecs,
        kind=kind,
        inside_coded_cube=bool(np.all(np.abs(zs) <= 1)),
    )


# --------------------------------------------------------------------------
# Ridge analysis (Hoerl 1959; Draper 1963)
# --------------------------------------------------------------------------


def _ridge_point(B: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """Maximiser of z'b + z'Bz on the sphere |z| = radius.

    Lagrange: (B - mu I) z = -b/2 with mu > lambda_max. |z(mu)| falls
    monotonically from infinity to zero on that interval, so a root in mu
    gives the constrained maximum.
    """
    k = len(b)
    lam_max = float(np.linalg.eigvalsh(B).max())

    def z_of(mu: float) -> np.ndarray:
        return -0.5 * np.linalg.solve(B - mu * np.eye(k), b)

    def f(mu: float) -> float:
        return float(np.linalg.norm(z_of(mu)) - radius)

    span = max(1.0, abs(lam_max), float(np.abs(b).max()))
    lo, hi = lam_max + 1e-9 * span, lam_max + span
    while f(hi) > 0:
        hi = lam_max + 2 * (hi - lam_max)
    if f(lo) > 0:
        mu = optimize.brentq(f, lo, hi, xtol=1e-12)
        return z_of(mu), mu

    # Degenerate: b has no component on the top eigenvector, so |z(mu)| stays
    # bounded as mu -> lambda_max. Fall back to direct constrained search.
    def neg(z):
        return -(z @ b + z @ B @ z)

    best = None
    rng = np.random.default_rng(0)
    for start in rng.normal(size=(16, k)):
        start = radius * start / np.linalg.norm(start)
        r = optimize.minimize(
            neg, start, method="SLSQP",
            constraints=[{"type": "eq", "fun": lambda z: z @ z - radius**2}],
        )
        if best is None or r.fun < best.fun:
            best = r
    return best.x, np.nan


def ridge_analysis(fit: RSMFit, radii, direction: str = "max") -> pd.DataFrame:
    """Path of constrained optima of the fitted surface at increasing radius.

    Radii are in coded units from the design centre. ``direction="min"``
    traces the path of steepest descent instead.
    """
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")
    sign = 1.0 if direction == "max" else -1.0
    B, b = sign * fit.B, sign * fit.b
    rows = []
    for R in radii:
        if R == 0:
            z, mu = np.zeros(fit.k), np.nan
        else:
            z, mu = _ridge_point(B, b, float(R))
        row = {"radius": float(R), "mu": mu}
        x = fit.coding.decode(z)
        for j, name in enumerate(fit.names):
            row[f"{name}_coded"] = z[j]
            row[name] = x[j]
        row["predicted"] = float(fit.predict_coded(z[None, :])[0])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Surfaces, support, and model-order comparison
# --------------------------------------------------------------------------


def in_hull(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """True where ``query`` lies inside the convex hull of ``points``.

    Observational factors are correlated (hot hours are high-load hours), so
    much of the rectangle spanned by the factor ranges holds no data. A
    fitted surface there is extrapolation and should not be drawn as if it
    were estimated.
    """
    return Delaunay(points).find_simplex(query) >= 0


def near_data(points: np.ndarray, query: np.ndarray, radius: float, min_points: int) -> np.ndarray:
    """True where at least ``min_points`` of ``points`` lie within ``radius`` of ``query``.

    A stricter notion of support than the convex hull. Observed (temperature,
    load) pairs form a curved V, and the hull of a V includes its empty inside;
    a local-density mask does not. Use coded units so the radius means the same
    fraction of each factor's range.
    """
    counts = cKDTree(points).query_ball_point(query, r=radius, return_length=True)
    return np.asarray(counts) >= min_points


def surface_grid(fit: RSMFit, n: int = 80, mask_hull: bool = True):
    """Fitted surface on an n x n grid over the observed range of two factors."""
    if fit.k != 2:
        raise ValueError("surface_grid needs exactly two factors")
    lo = fit.Z.min(axis=0)
    hi = fit.Z.max(axis=0)
    g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    Zg = np.column_stack([g1.ravel(), g2.ravel()])
    yhat = fit.predict_coded(Zg)
    if mask_hull:
        yhat = np.where(in_hull(fit.Z, Zg), yhat, np.nan)
    Xg = fit.coding.decode(Zg)
    return Xg[:, 0].reshape(n, n), Xg[:, 1].reshape(n, n), yhat.reshape(n, n)


def blocked_cv(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    fold_col: str,
    orders=(1, 2, 3),
    **fit_kwargs,
) -> pd.DataFrame:
    """Leave-one-block-out RMSE for each polynomial order.

    Folds are whole blocks (e.g. calendar months) so that autocorrelated
    neighbours of a test hour are never in its training set. Coding is
    frozen from the full sample so every fold predicts on the same scale.
    """
    if fit_kwargs.get("block_cols"):
        # Block dummies are rebuilt per fold and predict() uses the baseline
        # block, so a held-out fold in another block would be mis-predicted.
        raise ValueError("blocked_cv does not support block_cols")
    data = df[[response, *factors, fold_col]].dropna()
    coding = Coding.from_data(
        data[factors],
        fit_kwargs.pop("coding_method", "range"),
        fit_kwargs.pop("coding_quantiles", (0.01, 0.99)),
    )
    fit_kwargs.setdefault("cov_type", "nonrobust")
    rows = []
    for fold in sorted(data[fold_col].unique()):
        train = data[data[fold_col] != fold]
        test = data[data[fold_col] == fold]
        for order in orders:
            fit = fit_rsm(train, response, factors, order=order, coding=coding, **fit_kwargs)
            err = test[response].to_numpy() - fit.predict(test[factors])
            rows.append({"fold": fold, "order": order, "n": len(test),
                         "sse": float(err @ err)})
    out = pd.DataFrame(rows)
    agg = out.groupby("order").agg(n=("n", "sum"), sse=("sse", "sum"))
    agg["rmse"] = np.sqrt(agg["sse"] / agg["n"])
    return agg[["n", "rmse"]]


def coefficient_table(fit: RSMFit, lof: dict | None = None) -> pd.DataFrame:
    """NCSS "Estimation": coefficients, SEs, t, p and last R-squared.

    SEs follow the fit's covariance (HAC, HC3, or classical). ``last_r2`` is
    the R-squared lost by deleting that one term, as NCSS defines it.
    Passing ``lack_of_fit``'s result adds NCSS's "Estimation Using Pure
    Error" columns, which rescale the classical SEs by sqrt(MS_pe / MS_error).
    """
    res = fit.ols
    sse_full = float(res.ssr)
    last = {c: (_sse(fit.y, fit.X.drop(columns=[c])) - sse_full) / fit.sst
            for c in fit.X.columns if c != "Intercept"}
    out = pd.DataFrame({
        "group": fit.terms,
        "coef": res.params,
        "se": res.bse,
        "t": res.tvalues,
        "p_value": res.pvalues,
        "last_r2": pd.Series(last),
    }).reindex(fit.X.columns)  # dict-of-Series construction sorts the index
    if lof is not None and lof.get("df_pe"):
        df_err = len(fit.y) - fit.X.shape[1]
        unscaled = pd.Series(np.diag(res.normalized_cov_params), index=fit.X.columns)
        se_pe = np.sqrt(unscaled * lof["ss_pe"] / lof["df_pe"])
        out["se_pe"] = se_pe
        out["p_pe"] = 2 * stats.t.sf(np.abs(res.params / se_pe), lof["df_pe"])
    return out


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:,.6g}", na_rep="")


def summary_text(fit: RSMFit, *, lof_bins: int | None = 8, ridge_tol: float = 0.05,
                 selection_log: pd.DataFrame | None = None) -> str:
    """Plain-text report in NCSS Chapter 330's section order, plus the HAC extras."""
    lof = lack_of_fit(fit, lof_bins)
    lines = [f"RESPONSE SURFACE REGRESSION — {fit.response}", ""]

    desc = pd.DataFrame(fit.coding.decode(fit.Z), columns=list(fit.names))
    desc[fit.response] = fit.y
    lines += ["Descriptive Statistics", _fmt(desc.agg(["count", "mean", "min", "max"]).T), ""]

    lines.append("Model Summary")
    lines.append(f"  n = {len(fit.y):,}   order = {fit.order}   terms = {fit.X.shape[1]}"
                 f"   R^2 = {fit.ols.rsquared:.6f}   adj R^2 = {fit.ols.rsquared_adj:.6f}")
    if selection_log is not None:
        removed = ", ".join(selection_log["removed"]) or "none"
        lines.append(f"  hierarchical selection removed: {removed}")
    lines.append("  coding  z = (x - center) / half_range")
    for j, n in enumerate(fit.names):
        lines.append(f"    {n:<14} center {fit.coding.center[j]:>10.4g}   half_range {fit.coding.half_range[j]:>10.4g}")
    lines.append("")

    which = "exact duplicates" if lof_bins is None else f"{lof_bins}^k near-replicate cells"
    lines += [f"Sequential ANOVA (classical F; pure error from {which}, {lof['cells']} occupied)",
              _fmt(sequential_anova(fit, lof)), ""]
    lines += ["ANOVA by factor (all terms containing the factor)", _fmt(factor_anova(fit, lof)), ""]
    lines += [f"Estimation (coded units; SEs: {fit.cov_type})", _fmt(coefficient_table(fit, lof)), ""]
    lines += [f"Term-group Wald tests ({fit.cov_type})", _fmt(robust_term_tests(fit)), ""]

    if fit.order == 2:
        can = canonical_analysis(fit, ridge_tol)
        lines.append("Optimum / canonical analysis")
        lines.append(f"  stationary point is a {can.kind.upper()}"
                     f"{'' if can.inside_coded_cube else ' (OUTSIDE the coded region)'}")
        for name in fit.names:
            lines.append(f"  {name:<14} {can.stationary_natural[name]:>12.6g}")
        lines.append(f"  function at stationary point: {can.predicted_at_stationary:.6g}")
        lines.append(f"  eigenvalues: {np.array2string(can.eigenvalues, precision=4)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Observational-design diagnostics, sliced surfaces, time-ordered validation
# --------------------------------------------------------------------------


def design_diagnostics(fit: RSMFit) -> pd.DataFrame:
    """VIF of every polynomial column, plus the condition number of the design.

    Observational factors are correlated (and T, L are nonlinearly linked),
    so the coded design is far from orthogonal. Large VIFs mean individual
    coefficients are poorly determined even when the fitted surface, a
    linear combination of them, is well determined inside the data.
    """
    X = fit.X.drop(columns=["Intercept"])
    Xc = (X - X.mean()).to_numpy(float)
    rows = []
    for j, c in enumerate(X.columns):
        others = np.delete(Xc, j, axis=1)
        beta = np.linalg.lstsq(others, Xc[:, j], rcond=None)[0]
        resid = Xc[:, j] - others @ beta
        r2 = 1 - (resid @ resid) / (Xc[:, j] @ Xc[:, j])
        rows.append({"term": c, "group": fit.terms[c], "vif": 1 / max(1e-12, 1 - r2)})
    out = pd.DataFrame(rows).set_index("term")
    Xs = fit.X.to_numpy(float)
    out.attrs["condition_number"] = float(np.linalg.cond(Xs / np.linalg.norm(Xs, axis=0)))
    return out


def _check_slice(fit: RSMFit, x: str, y: str, fixed: dict) -> tuple[list[str], int, int]:
    names = list(fit.names)
    if set(fixed) | {x, y} != set(names):
        raise ValueError(f"fix every factor except {x}, {y}: got {sorted(fixed)}")
    return names, names.index(x), names.index(y)


def slice_support(fit: RSMFit, x: str, y: str, fixed: dict, data: pd.DataFrame,
                  band: float = 0.25) -> np.ndarray:
    """Boolean mask over ``data`` rows whose fixed factors lie within ``band`` coded
    units of the slice: the observed hours that support a slice over (x, y)."""
    names, _, _ = _check_slice(fit, x, y, fixed)
    zfix = fit.coding.encode(np.array([[fixed.get(nm, 0.0) for nm in names]]))[0]
    Zd = fit.coding.encode(data[names].to_numpy(float))
    near = np.ones(len(Zd), bool)
    for k, nm in enumerate(names):
        if nm in fixed:
            near &= np.abs(Zd[:, k] - zfix[k]) <= band
    return near


def slice_grid(fit: RSMFit, x: str, y: str, fixed: dict, data: pd.DataFrame | None = None,
               n: int = 80, band: float = 0.25, support: str = "hull", radius: float = 0.1,
               min_points: int = 5):
    """The fitted surface over (x, y) with every other factor held at ``fixed``.

    With ``data``, the grid is masked to the convex hull of the (x, y) points
    whose other factors lie near the slice (within ``band`` coded units, see
    ``slice_support``), so a slice is drawn only where comparable hours were
    actually observed. ``support="density"`` additionally requires
    ``min_points`` such hours within ``radius`` coded units of each grid point
    (``near_data``). Returns natural-unit grids X, Y and the block-averaged
    prediction.
    """
    if support not in ("hull", "density"):
        raise ValueError(f"support must be 'hull' or 'density', got {support!r}")
    names, i, j = _check_slice(fit, x, y, fixed)
    lo, hi = fit.Z.min(axis=0), fit.Z.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[i], hi[i], n), np.linspace(lo[j], hi[j], n))
    Zg = np.zeros((n * n, len(names)))
    Zg[:, i], Zg[:, j] = gx.ravel(), gy.ravel()
    zfix = fit.coding.encode(np.array([[fixed.get(nm, 0.0) for nm in names]]))[0]
    for k, nm in enumerate(names):
        if nm in fixed:
            Zg[:, k] = zfix[k]
    yhat = fit.predict_coded(Zg)
    if data is not None:
        near = slice_support(fit, x, y, fixed, data, band)
        pts = fit.coding.encode(data[names].to_numpy(float))[near][:, [i, j]]
        if len(pts) < 10:
            raise ValueError(f"only {len(pts)} observations near the slice {fixed}")
        keep = in_hull(pts, Zg[:, [i, j]])
        if support == "density":
            keep &= near_data(pts, Zg[:, [i, j]], radius, min_points)
        yhat = np.where(keep, yhat, np.nan)
    Xg = fit.coding.decode(Zg)
    return Xg[:, i].reshape(n, n), Xg[:, j].reshape(n, n), yhat.reshape(n, n)


def time_folds(months: pd.Series, scheme: str, min_train: int = 3) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(train mask, test mask, test month) for each fold.

    ``forward``: expanding window, train on every month before the test month
    and nothing after it (the honest-in-time scheme). ``lomo``: leave one
    month out, training on months on both sides (interpolation, not forecasting).
    """
    m = pd.Series(months).astype(str).to_numpy()
    uniq = sorted(set(m))
    folds = []
    for k, test in enumerate(uniq):
        if scheme == "forward":
            if k < min_train:
                continue
            train = m < test
            assert m[train].max() < test, "forward fold trains on the test month or later"
        elif scheme == "lomo":
            train = m != test
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        folds.append((train, m == test, test))
    return folds
