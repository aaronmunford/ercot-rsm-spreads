"""Report figures. Static matplotlib, sized for a two-column-free 10-page PDF.

Colour follows the job it does: a magnitude surface (heat rate) uses one hue
light->dark; a signed surface (DA-RT spread) uses a blue<->red diverging
ramp with a grey midpoint pinned at zero, so "no premium" always reads as
neutral. Zones keep fixed colours in every figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import rsm  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ZONE_COLORS = {"houston": "#2a78d6", "north": "#eb6834", "south": "#1baf7a", "west": "#eda100"}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a5a4", "#e34948", "#a62b2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2,
})

LABELS = {
    "temp_f": "Temperature (°F)",
    "load_norm": "Load / zonal peak",
    "imhr_da": "DA implied heat rate (MMBtu/MWh)",
    "imhr_rt": "RT implied heat rate (MMBtu/MWh)",
    "spread": "DA − RT spread ($/MWh)",
    "abs_spread": "|DA − RT| spread magnitude ($/MWh)",
    "load_mw": "Zone load (MW)",
    "load_anom": "Load anomaly (MW)",
    "gas_price": "Henry Hub gas ($/MMBtu)",
    "net_load_err_avail": "Net-load surprise (GW)",
    "net_load_fc": "DA-time net-load forecast (GW)",
}
GW_FACTORS = {"load_mw", "net_load_err_avail", "net_load_fc", "load_fc_mw", "net_load_err"}


def _surface(fit: rsm.RSMFit, data=None, n: int = 90):
    """(x grid, y grid, fitted surface, slice label). Two factors: the full masked surface.
    Three or more: a slice over the first two factors with the rest at their medians."""
    if fit.k == 2:
        x1, x2, yhat = rsm.surface_grid(fit, n=n, mask_hull=True)
        return x1, x2, yhat, ""
    if data is None:
        raise ValueError("a sliced surface needs the fitted data for its medians and hull mask")
    x, y, rest = fit.names[0], fit.names[1], fit.names[2:]
    fixed = {r: float(data[r].median()) for r in rest}
    x1, x2, yhat = rsm.slice_grid(fit, x, y, fixed, data=data, n=n)
    return x1, x2, yhat, ", ".join(f"{LABELS.get(r, r).split(' (')[0]} = {v:.2f}" for r, v in fixed.items())


def surfaces_by_zone(fits: dict, response: str, path, title: str | None = None, data: dict | None = None,
                     levels: int = 14):
    """Small multiples, one panel per zone, on ONE colour scale shared by every panel."""
    zones = list(fits)
    signed = response == "spread"
    grids = {z: _surface(fits[z], (data or {}).get(z)) for z in zones}
    finite = np.concatenate([g[2][np.isfinite(g[2])] for g in grids.values()])
    from matplotlib.ticker import MaxNLocator

    if signed:
        lim = float(np.nanmax(np.abs(finite))) or 1.0
        lv = MaxNLocator(nbins=levels, symmetric=True).tick_values(-lim, lim)
        norm, cmap = TwoSlopeNorm(0.0, lv[0], lv[-1]), DIVERGING
    else:
        lv = MaxNLocator(nbins=levels).tick_values(float(np.nanmin(finite)), float(np.nanmax(finite)))
        norm, cmap = plt.Normalize(lv[0], lv[-1]), SEQUENTIAL
    fig, axes = plt.subplots(1, len(zones), figsize=(1.9 * len(zones) + 0.8, 2.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    cs = None
    for ax, z in zip(axes, zones):
        x1, x2, yhat, sl = grids[z]
        fit = fits[z]
        gx = 1000.0 if fit.names[0] in GW_FACTORS else 1.0
        gy = 1000.0 if fit.names[1] in GW_FACTORS else 1.0
        d = (data or {}).get(z)
        if d is not None:
            step = max(1, len(d) // 3000)
            ax.scatter(d[fit.names[0]].to_numpy()[::step] / gx, d[fit.names[1]].to_numpy()[::step] / gy,
                       s=1.5, color=INK_2, alpha=0.15, linewidths=0, zorder=1, rasterized=True)
        cs = ax.contourf(x1 / gx, x2 / gy, yhat, levels=lv, cmap=cmap, norm=norm, alpha=0.92, zorder=2)
        ax.contour(x1 / gx, x2 / gy, yhat, levels=lv, colors=SURFACE, linewidths=0.4, zorder=3)
        if gx != 1.0:
            ax.axvline(0, color=INK, lw=0.6, ls=":", zorder=4) if fit.names[0].endswith(("err", "err_avail")) else None
        if fit.k == 2 and fit.order == 2:
            can = rsm.canonical_analysis(fit)
            sx, sy = can.stationary_natural.iloc[0], can.stationary_natural.iloc[1]
            if x1.min() <= sx <= x1.max() and x2.min() <= sy <= x2.max():
                ax.plot(sx / gx, sy / gy, marker="o", ms=7, mfc=SURFACE, mec=INK, mew=2, zorder=5)
        ax.set_title(z.capitalize(), color=ZONE_COLORS.get(z, INK), loc="left", fontweight="bold")
        ax.set_xlabel(LABELS.get(fit.names[0], fit.names[0]), fontsize=8)
        ylab = ("Zone load (GW)" if fit.names[1] == "load_mw" else LABELS.get(fit.names[1], fit.names[1]))
        ax.set_ylabel(ylab if ax is axes[0] else "", fontsize=8)
        if sl:
            ax.text(0.02, 0.98, sl, transform=ax.transAxes, fontsize=6.5, va="top", color=INK_2)
    fig.colorbar(cs, ax=axes, shrink=0.85, label=LABELS.get(response, response))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


SPEC_SHORT = {"M1": "M1 linear", "M2": "M2 quadratic RSM", "M3": "M3 RSM + hour block", "ALT": "ALT load anomaly"}


def ladder_skill(fwd, path):
    """Forward-CV skill of the surprise ladder and the ex-ante model vs the hour-of-day mean."""
    models = ["A", "B", "C", "D", "S", "E"]
    shades = {"A": "#c9d9ef", "B": "#9ec5f4", "C": "#3987e5", "D": "#0d366b", "S": "#e34948", "E": "#eda100"}
    labels = {"A": "A realised T, L, G", "B": "B + ramp", "C": "C + load surprise", "D": "D + wind/solar surprise",
              "S": "S surprise surface", "E": "E ex-ante (09:45 D−1)"}
    responses = [r for r in ("spread", "abs_spread") if r in set(fwd["response"])]
    fig, axes = plt.subplots(1, len(responses), figsize=(7.4, 2.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    w = 0.13
    for ax, resp in zip(axes, responses):
        c = fwd[fwd["response"] == resp]
        zones = [z for z in ZONES if z in set(c["zone"])]
        for i, m in enumerate(models):
            r = c[c["model"] == m].set_index("zone").reindex(zones)
            ax.bar(np.arange(len(zones)) + (i - 2.5) * w, r["skill_vs_M0"], width=w, color=shades[m], label=labels[m])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(zones)), [z.capitalize() for z in zones])
        ax.grid(axis="x", visible=False)
        ax.set_title({"spread": "DA − RT spread", "abs_spread": "|spread| magnitude"}[resp], loc="left", fontsize=9)
    axes[0].set_ylabel("Skill vs hour-of-day mean")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="outside lower center")
    fig.suptitle("Forward (expanding-window) CV, test months Apr–Dec: 1 − SSE/SSE(M0)", x=0.01, ha="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def basis_decomposition(shares, west_wind, west_load, path):
    """A: share of each zone's spread variance from the hub spread, the basis surprise and their
    covariance. B, C: West's basis surprise by wind-availability surprise and by load decile."""
    zones = [z for z in ZONES if z in set(shares["zone"])]
    s = shares.set_index("zone").reindex(zones)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.1, 1, 1]})
    x = np.arange(len(zones))
    a1.bar(x, s["cov_share_hub"], color="#9ec5f4", label="hub DA − RT spread")
    a1.bar(x, s["cov_share_basis"], bottom=s["cov_share_hub"], color="#eda100", label="basis surprise")
    a1.axhline(1, color=INK, lw=0.6, ls=":")
    a1.set_xticks(x, [z[:1].upper() for z in zones])
    a1.set_ylabel("Cov(term, spread) / Var(spread)")
    a1.set_ylim(0, 1.32)
    a1.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1)
    a1.set_title("A. Spread variance by source", loc="left", fontsize=9)
    for ax, t, xlab, title in ((a2, west_wind, "West+Panhandle wind availability\nsurprise (GW)", "B. West, by wind surprise"),
                              (a3, west_load, "West zonal load (GW)", "C. West, by load decile")):
        xv = t.iloc[:, 1].to_numpy() / 1000
        ax.plot(xv, t["median_abs_basis_surprise"], color=ZONE_COLORS["west"], marker="o", ms=3, label="median |basis surprise|")
        ax.plot(xv, t["p95_abs_basis_surprise"], color=ZONE_COLORS["west"], ls="--", marker="o", ms=3, label="95th pct")
        ax.plot(xv, t["mean_basis_surprise"], color=INK, lw=1, label="mean (signed)")
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
    a2.set_ylabel("$/MWh")
    a3.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def model_comparison(comp, path, scheme: str = "forward"):
    """Out-of-sample skill of every model relative to the hour-of-day mean (M0), per response.
    Skill = 1 - SSE/SSE(M0) over every held-out hour; below zero is worse than M0."""
    models = ["M1", "M2", "M3", "ALT"]
    shades = {"M1": "#9ec5f4", "M2": "#256abf", "M3": "#0d366b", "ALT": "#eda100"}
    responses = [r for r in ("spread", "abs_spread", "imhr_da") if r in set(comp["response"])]
    fig, axes = plt.subplots(1, len(responses), figsize=(7.4, 2.6), constrained_layout=True, sharey=False)
    axes = np.atleast_1d(axes)
    w = 0.2
    for ax, resp in zip(axes, responses):
        c = comp[(comp["response"] == resp) & (comp["scheme"] == scheme)]
        zones = [z for z in ZONES if z in set(c["zone"])]
        for i, m in enumerate(models):
            r = c[c["model"] == m].set_index("zone").reindex(zones)
            ax.bar(np.arange(len(zones)) + (i - 1.5) * w, r["skill_vs_M0"], width=w, color=shades[m], label=m)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(zones)), [z[:1].upper() for z in zones])
        ax.grid(axis="x", visible=False)
        ax.set_title({"spread": "DA − RT spread", "abs_spread": "|spread| magnitude",
                      "imhr_da": "DA implied heat rate"}[resp], loc="left", fontsize=9)
    axes[0].set_ylabel("Skill vs hour-of-day mean")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [SPEC_SHORT.get(x, x) for x in l], frameon=False, fontsize=7, ncol=4, loc="outside lower center")
    fig.suptitle(f"{scheme.capitalize()} (expanding-window) CV, test months Apr–Dec 2025: 1 − SSE/SSE(M0)",
                 x=0.01, ha="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def spread_distribution(panel, path):
    """Hourly DA-RT spread quantiles by zone on a symmetric-log axis.

    Quantile bars rather than a density: a kernel estimate drawn on a
    symlog axis is computed in linear space and distorts badly in the tails,
    which is exactly where ERCOT spreads live.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
    zones = [z for z in ZONE_COLORS if z in set(panel["zone"])]
    for i, z in enumerate(zones):
        q = panel.loc[panel.zone == z, "spread"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        c = ZONE_COLORS[z]
        ax.plot([q[0.05], q[0.95]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot([q[0.25], q[0.75]], [i, i], color=c, lw=9, solid_capstyle="butt")
        ax.plot([q[0.01], q[0.99]], [i, i], ls="none", marker="o", ms=4, color=c)
        ax.plot(q[0.5], i, marker="|", ms=14, mew=2.5, color=INK)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yticks(range(len(zones)), [z.capitalize() for z in zones])
    ax.set_ylim(-0.6, len(zones) - 0.4)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(LABELS["spread"] + " (symmetric log, linear within ±10)")
    ax.set_title("Bar: 25–75th pct · line: 5–95th · dots: 1st, 99th · tick: median",
                 loc="left", fontsize=8, color=INK_2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def residual_diagnostics(fits: dict, response: str, path, max_lag: int = 72):
    """Per zone: normal probability plot (NCSS's diagnostic) and residual autocorrelation.

    The ACF is the evidence for HAC inference. Hourly residuals that stay
    correlated for a day or more mean classical standard errors, which
    assume independence, are far too small.
    """
    from scipy import stats

    zones = list(fits)
    fig, axes = plt.subplots(2, len(zones), figsize=(3.0 * len(zones), 5.4), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(2, len(zones))
    for j, z in enumerate(zones):
        r = np.asarray(fits[z].ols.resid)
        color = ZONE_COLORS.get(z, INK)
        (osm, osr), (slope, icpt, _) = stats.probplot(r, dist="norm")
        step = max(1, len(osm) // 3000)
        ax = axes[0, j]
        ax.scatter(osm[::step], osr[::step], s=4, color=color, linewidths=0)
        ax.plot(osm[[0, -1]], icpt + slope * osm[[0, -1]], color=INK, lw=1)
        ax.set_title(z.capitalize(), color=color, loc="left", fontweight="bold")
        ax.set_xlabel("Normal quantile")
        ax.set_ylabel("Residual" if j == 0 else "")

        rc = r - r.mean()
        acf = np.array([1.0] + [np.dot(rc[:-k], rc[k:]) / np.dot(rc, rc) for k in range(1, max_lag + 1)])
        ax = axes[1, j]
        ax.vlines(np.arange(max_lag + 1), 0, acf, color=color, lw=1.5)
        band = 1.96 / np.sqrt(len(r))
        ax.axhspan(-band, band, color=GRID, zorder=0)
        for lag in (24, 48):
            ax.axvline(lag, color=INK_2, lw=0.6, ls=":")
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Residual ACF" if j == 0 else "")
        ax.set_ylim(min(-0.2, acf.min() - 0.05), 1.02)
    fig.suptitle(f"Residual diagnostics — {LABELS.get(response, response)}", x=0.01, ha="left", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# EDA figures
# --------------------------------------------------------------------------

ZONES = ("houston", "north", "south", "west")
RAMPS = ((5, 8), (17, 22))  # hour-beginning spans shaded as morning / evening ramps


def _zones(panel):
    return [z for z in ZONES if z in set(panel["zone"])]


def fig_spread_distribution(panel, summary, path):
    """Left: hourly spread histogram per zone on a symlog axis, log counts, so the bulk
    and the tails are both visible. Right: tail exceedance P(|spread| > x), log-log,
    with RT-above-DA (negative spread) and DA-above-RT (positive) tails drawn separately."""
    zones = _zones(panel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    s_all = panel["spread"]
    lim = float(np.ceil(s_all.abs().max() / 100) * 100)
    edges = 10 * np.sinh(np.linspace(-np.arcsinh(lim / 10), np.arcsinh(lim / 10), 121))
    for z in zones:
        counts, _ = np.histogram(panel.loc[panel.zone == z, "spread"], bins=edges)
        a1.stairs(np.where(counts > 0, counts, np.nan), edges, color=ZONE_COLORS[z], lw=1.4,
                  label=z.capitalize())
    a1.set_xscale("symlog", linthresh=10)
    a1.set_yscale("log")
    a1.axvline(0, color=INK_2, lw=0.8)
    a1.set_xlabel("DA − RT spread ($/MWh, symmetric log)")
    a1.set_ylabel("Hours per bin (log)")
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    a1.set_title("A. Hourly spread, 2025 (8,760 h per zone)", loc="left", fontsize=9)

    for z in zones:
        s = panel.loc[panel.zone == z, "spread"].to_numpy()
        n = len(s)
        for tail, ls in ((-s[s < 0], "-"), (s[s > 0], "--")):
            x = np.sort(tail)[::-1]
            a2.plot(x, np.arange(1, len(x) + 1) / n, ls=ls, color=ZONE_COLORS[z], lw=1.3)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlim(1, None)
    a2.set_xlabel("|spread| threshold x ($/MWh)")
    a2.set_ylabel("Share of hours beyond x")
    a2.plot([], [], "-", color=INK_2, label="RT above DA (spread < 0)")
    a2.plot([], [], "--", color=INK_2, label="DA above RT (spread > 0)")
    a2.legend(frameon=False, fontsize=7.5, loc="lower left")
    txt = "          skew  top-1% share\n" + "\n".join(
        f"{z.capitalize():8s}{summary.loc[z, 'skew']:6.1f}  {100 * summary.loc[z, 'top1pct_share_of_abs_sum']:5.0f}%"
        for z in zones)
    a2.text(0.98, 0.98, txt, transform=a2.transAxes, ha="right", va="top", fontsize=6.8,
            family="monospace", color=INK)
    a2.text(0.98, 0.70, "top-1% share: share of Σ|spread|\nfrom the 1% largest hours", transform=a2.transAxes,
            ha="right", va="top", fontsize=6, color=INK_2)
    a2.set_title("B. Tail exceedance by sign", loc="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_temp_load(panel, tl, path):
    """Temperature vs actual zonal load, coloured by hour of day, with the 2 °F binned mean
    and a least-squares quadratic in temperature. R² of that quadratic is annotated."""
    zones = _zones(panel)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("twilight_shifted")
    sc = None
    for ax, z in zip(axes.ravel(), zones):
        d = panel[panel.zone == z]
        t, y = d["temp_f"].to_numpy(), d["load_mw"].to_numpy() / 1000
        sc = ax.scatter(t, y, c=d["hour_beginning"], cmap=cmap, vmin=0, vmax=24, s=1.5, alpha=0.45,
                        linewidths=0, rasterized=True)
        b = np.floor(t / 2) * 2 + 1
        m = d.assign(b=b, y=y).groupby("b")["y"].agg(["mean", "size"])
        m = m[m["size"] >= 10]
        ax.plot(m.index, m["mean"], color=INK, lw=1.8, label="2 °F bin mean")
        c = np.polyfit(t, y, 2)
        tt = np.linspace(t.min(), t.max(), 100)
        ax.plot(tt, np.polyval(c, tt), color=INK, lw=1.2, ls="--", label="Quadratic fit")
        ax.set_title(f"{z.capitalize()}   R²(T, T²) = {tl.loc[z, 'r2_T_T2']:.2f}, "
                     f"minimum ≈ {tl.loc[z, 'quadratic_min_load_at_F']:.0f} °F",
                     loc="left", color=ZONE_COLORS[z], fontweight="bold", fontsize=8.5)
        ax.set_xlabel("Zone temperature (°F)")
        ax.set_ylabel("Zone load (GW)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper center")
    # an opaque mappable for the colour bar: the points themselves are translucent
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 24))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, ticks=[0, 6, 12, 18, 24])
    cb.set_label("Hour of day (local, beginning)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_extremes_compact(panel, by_hour, path):
    """Report version: the day x hour heatmap (A) and top-1% counts by hour and sign (B)."""
    from matplotlib.colors import SymLogNorm

    fig, (a1, a3) = plt.subplots(2, 1, figsize=(7.4, 4.4), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1.35, 0.8]})
    avg = panel.groupby(["operating_date", "hour_beginning"])["spread"].mean().unstack("operating_date")
    days = pd.to_datetime(avg.columns)
    vmax = float(np.nanpercentile(np.abs(avg.to_numpy()), 99.8))
    im = a1.pcolormesh(np.arange(len(days) + 1), np.arange(25), avg.to_numpy(), cmap=DIVERGING,
                       norm=SymLogNorm(linthresh=10, vmin=-vmax, vmax=vmax), rasterized=True)
    months = [i for i, d in enumerate(days) if d.day == 1]
    a1.set_xticks(months, [days[i].strftime("%b") for i in months])
    a1.set_yticks([0, 6, 12, 18, 23])
    a1.set_ylabel("Hour beginning")
    a1.grid(False)
    a1.set_title("A. Zone-average DA − RT spread by day and hour (blue: RT above DA; red: DA above RT)",
                 loc="left", fontsize=9)
    fig.colorbar(im, ax=a1, pad=0.01, label="$/MWh (symlog)")
    hrs = np.arange(24)
    for lo, hi in RAMPS:
        a3.axvspan(lo - 0.5, hi - 0.5, color=GRID, alpha=0.7, zorder=0)
    neg = by_hour.get("RT_above_DA", pd.Series(0, index=hrs))
    pos = by_hour.get("DA_above_RT", pd.Series(0, index=hrs))
    a3.bar(hrs, neg, color=DIVERGING(0.12), label="RT above DA")
    a3.bar(hrs, pos, bottom=neg, color=DIVERGING(0.88), label="DA above RT")
    a3.set_xlim(-0.5, 23.5)
    a3.set_xticks(range(0, 24, 3))
    a3.set_xlabel("Hour beginning (local); shaded: morning and evening ramps")
    a3.set_ylabel("Zone-hours")
    a3.legend(frameon=False, fontsize=7, loc="upper left")
    a3.set_title("B. Top-1% |spread| zone-hours (all zones) by hour and sign", loc="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_extremes_by_hour(panel, quantiles, by_hour, path):
    """A: hour-of-day x operating-day heatmap of the zone-average spread (diverging,
    symmetric log). B: 1st/99th and 5th/95th percentiles of the spread by hour, per zone.
    C: counts of top-1% |spread| zone-hours by hour and sign. Ramps shaded."""
    from matplotlib.colors import SymLogNorm

    fig = plt.figure(figsize=(7.4, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1, 0.75])
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))

    avg = panel.groupby(["operating_date", "hour_beginning"])["spread"].mean().unstack("operating_date")
    days = pd.to_datetime(avg.columns)
    vmax = float(np.nanpercentile(np.abs(avg.to_numpy()), 99.8))
    im = a1.pcolormesh(np.arange(len(days) + 1), np.arange(25), avg.to_numpy(), cmap=DIVERGING,
                       norm=SymLogNorm(linthresh=10, vmin=-vmax, vmax=vmax), rasterized=True)
    months = [i for i, d in enumerate(days) if d.day == 1]
    a1.set_xticks(months, [days[i].strftime("%b") for i in months])
    a1.set_yticks([0, 6, 12, 18, 23])
    a1.set_ylabel("Hour beginning")
    a1.grid(False)
    a1.set_title("A. Zone-average DA − RT spread by day and hour (blue: RT above DA; red: DA above RT)",
                 loc="left", fontsize=9)
    fig.colorbar(im, ax=a1, pad=0.01, label="$/MWh (symlog)")

    hrs = np.arange(24)
    for lo, hi in RAMPS:
        for ax in (a2, a3):
            ax.axvspan(lo - 0.5, hi - 0.5, color=GRID, alpha=0.7, zorder=0)
    for z in _zones(panel):
        q = quantiles.loc[z]
        c = ZONE_COLORS[z]
        a2.plot(hrs, q["p1"], color=c, lw=1.4)
        a2.plot(hrs, q["p99"], color=c, lw=1.4, label=z.capitalize())
        a2.plot(hrs, q["p5"], color=c, lw=0.8, ls=":")
        a2.plot(hrs, q["p95"], color=c, lw=0.8, ls=":")
    a2.axhline(0, color=INK_2, lw=0.8)
    a2.set_yscale("symlog", linthresh=10)
    a2.set_xlim(-0.5, 23.5)
    a2.set_xticks(range(0, 24, 3))
    a2.set_ylabel("Spread percentile ($/MWh)")
    a2.legend(frameon=False, fontsize=7, ncol=4, loc="center left")
    a2.set_title("B. 1st/99th (solid) and 5th/95th (dotted) percentiles by hour; shaded: ramp hours",
                 loc="left", fontsize=9)

    neg = by_hour.get("RT_above_DA", pd.Series(0, index=hrs))
    pos = by_hour.get("DA_above_RT", pd.Series(0, index=hrs))
    a3.bar(hrs, neg, color=DIVERGING(0.12), label="RT above DA")
    a3.bar(hrs, pos, bottom=neg, color=DIVERGING(0.88), label="DA above RT")
    a3.set_xlim(-0.5, 23.5)
    a3.set_xticks(range(0, 24, 3))
    a3.set_xlabel("Hour beginning (local)")
    a3.set_ylabel("Zone-hours")
    a3.legend(frameon=False, fontsize=7, loc="upper left")
    a3.set_title("C. Top-1% |spread| zone-hours (all zones) by hour and sign", loc="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report figures: one message each, drawn at print size
# (width in inches = width on the page), so 9 pt here is 9 pt in the PDF.
# --------------------------------------------------------------------------

V2_FONT = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
           "ytick.labelsize": 8.5, "legend.fontsize": 8.5}


def _normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fig_spread_simple(panel, summary, path):
    """Hourly DA-RT spread, all zones pooled, against a normal curve with the same mean
    and SD. A: the core (-$60..+$60), linear axes: the bulk is far more peaked than the
    normal. B: the full range on log counts: the normal is gone by about ±$150 while
    observed hours reach -$2,100. Annotations come from the per-zone summary table."""
    s = panel["spread"].to_numpy(float)
    mu, sd, n = s.mean(), s.std(), len(s)
    with plt.rc_context(V2_FONT):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1, 1.25]})
        w = 2.0
        edges = np.arange(-60, 60 + w, w)
        a1.hist(s, bins=edges, color="#9ec5f4", edgecolor=SURFACE, linewidth=0.3, label="Observed hours")
        xx = np.linspace(-60, 60, 400)
        a1.plot(xx, n * w * _normal_pdf(xx, mu, sd), color=INK, lw=1.6, label="Normal, same mean and SD")
        a1.axvline(0, color=INK_2, lw=0.7)
        a1.set_xlim(-60, 60)
        a1.set_xlabel("DA − RT spread ($/MWh)")
        a1.set_ylabel(f"Zone-hours per \\${w:.0f} bin")
        a1.set_title("A. Typical hours (−\\$60 to +\\$60)", loc="left")
        a1.legend(frameon=False, loc="upper left", fontsize=8)
        med = summary["median"]
        pos = summary["share_positive"]
        a1.text(0.98, 0.70, f"Median +\\${med.min():.1f}\nto +\\${med.max():.1f}\n\n"
                f"DA > RT in\n{100 * pos.min():.0f}–{100 * pos.max():.0f}%\nof hours",
                transform=a1.transAxes, ha="right", va="top", color=INK)

        w2 = 25.0
        lo = np.floor(s.min() / w2) * w2
        hi = np.ceil(s.max() / w2) * w2
        edges2 = np.arange(lo, hi + w2, w2)
        counts, _ = np.histogram(s, bins=edges2)
        a2.stairs(np.where(counts > 0, counts, np.nan), edges2, color="#256abf", lw=1.3, fill=False,
                  label="Observed hours")
        xx = np.linspace(lo, hi, 2000)
        expct = n * w2 * _normal_pdf(xx, mu, sd)
        a2.plot(xx[expct >= 0.1], expct[expct >= 0.1], color=INK, lw=1.6, label="Normal, same mean and SD")
        a2.set_yscale("log")
        a2.set_ylim(0.6, None)
        a2.set_xlabel("DA − RT spread ($/MWh)")
        a2.set_ylabel(f"Zone-hours per \\${w2:.0f} bin (log)")
        a2.set_title("B. All hours: the tails (log scale)", loc="left")
        a2.legend(frameon=False, loc="upper left", fontsize=8)
        top = summary["top1pct_share_of_abs_sum"]
        a2.text(0.03, 0.66, f"Top 1% of hours carry\n{100 * top.min():.0f}–{100 * top.max():.0f}% of Σ|spread|\n\n"
                f"Skew {summary['skew'].min():.0f} to {summary['skew'].max():.0f}".replace("-", "−"),
                transform=a2.transAxes, ha="left", va="top", color=INK)
        fig.savefig(path, dpi=220)
        plt.close(fig)


def fig_temp_load_simple(panel, tl, path):
    """Temperature vs zonal load, 2x2: neutral points, a dashed least-squares line (load on T)
    and a bold quadratic (load on T and T²), with the in-sample R² of each in the title, so the
    gain from the squared term is visible at a glance."""
    zones = _zones(panel)
    with plt.rc_context(V2_FONT):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), constrained_layout=True, sharex=True)
        for ax, z in zip(axes.ravel(), zones):
            d = panel[panel.zone == z]
            t, y = d["temp_f"].to_numpy(float), d["load_mw"].to_numpy(float) / 1000
            ax.scatter(t, y, s=1.2, color=INK_2, alpha=0.18, linewidths=0, rasterized=True)
            tt = np.linspace(t.min(), t.max(), 200)
            ax.plot(tt, np.polyval(np.polyfit(t, y, 1), tt), color=INK_2, lw=1.4, ls="--",
                    label="Linear in T")
            ax.plot(tt, np.polyval(np.polyfit(t, y, 2), tt), color=ZONE_COLORS[z], lw=2.4,
                    label="Quadratic in T")
            ax.set_title(f"{z.capitalize()}:  R² {tl.loc[z, 'r2_T']:.2f} → {tl.loc[z, 'r2_T_T2']:.2f}",
                         loc="left", fontweight="bold")
            ax.set_ylabel("Load (GW)")
        axes[0, 0].legend(frameon=False, loc="upper center", fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("Zone temperature (°F)")
        fig.savefig(path, dpi=220)
        plt.close(fig)


def fig_surface_support(fits: dict, data: dict, path, response: str = "abs_spread", band: float = 0.25,
                        radius: float = 0.1, min_points: int = 5):
    """Fitted M2 surface over temperature x zonal load at median gas, one panel per zone,
    with three regions that cannot be confused:

    * observed support (colour, outlined): inside the convex hull of hours with
      near-median gas AND within ``radius`` coded units of at least ``min_points`` of
      them (``rsm.slice_grid(support="density")``);
    * extrapolation (grey, thin contours only): the same polynomial with no comparable hours;
    * nonphysical predictions (hatched): a negative fitted magnitude, wherever it occurs.

    The colour scale starts at zero and is shared across panels.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    zones = list(fits)
    grids = {}
    for z in zones:
        fit, d = fits[z], data[z]
        x, y, rest = fit.names[0], fit.names[1], fit.names[2:]
        fixed = {r: float(d[r].median()) for r in rest}
        full = rsm.slice_grid(fit, x, y, fixed, data=None, n=160)
        masked = rsm.slice_grid(fit, x, y, fixed, data=d, n=160, band=band, support="density",
                                radius=radius, min_points=min_points)
        sup = d.loc[rsm.slice_support(fit, x, y, fixed, d, band), [x, y]].to_numpy(float)
        grids[z] = (full, masked, sup, fixed)
    inside = np.concatenate([g[1][2][np.isfinite(g[1][2])] for g in grids.values()])
    lv = MaxNLocator(nbins=10).tick_values(0.0, float(inside.max()))
    norm = plt.Normalize(lv[0], lv[-1])
    extrap, hatch_ink = "#e4e3df", INK
    with plt.rc_context({**V2_FONT, "hatch.linewidth": 0.6}):
        fig, axes = plt.subplots(1, len(zones), figsize=(7.0, 3.05), constrained_layout=True)
        axes = np.atleast_1d(axes)
        cs = None
        for ax, z in zip(axes, zones):
            (X, Y, F_full), (_, _, F_in), sup, fixed = grids[z]
            Yg = Y / 1000
            supported = np.isfinite(F_in)
            ax.set_facecolor(extrap)
            ax.contour(X, Yg, F_full, levels=lv, colors="#b9b8b3", linewidths=0.5, zorder=1)
            ax.contourf(X, Yg, supported.astype(float), levels=[0.5, 1.5], colors=[SURFACE], zorder=2)
            cs = ax.contourf(X, Yg, F_in, levels=lv, cmap=SEQUENTIAL, norm=norm, extend="max", zorder=3)
            cl = ax.contour(X, Yg, F_in, levels=lv, colors=SURFACE, linewidths=0.5, zorder=4)
            ax.clabel(cl, levels=lv[1::2], fmt=lambda v: f"\\${v:.0f}", fontsize=7.5, colors=INK, inline=True)
            if (F_full < 0).any():
                with plt.rc_context({"hatch.color": "#8a8984"}):
                    ax.contourf(X, Yg, np.where(supported, np.nan, F_full), levels=[F_full.min() - 1, 0],
                                colors="none", hatches=["////"], zorder=5)
            if (F_in < 0).any():
                with plt.rc_context({"hatch.color": hatch_ink}):
                    ax.contourf(X, Yg, F_in, levels=[np.nanmin(F_in) - 1, 0], colors="none",
                                hatches=["////"], zorder=5)
                ax.contour(X, Yg, F_in, levels=[0], colors=hatch_ink, linewidths=0.8, linestyles="--", zorder=6)
            ax.scatter(sup[:, 0], sup[:, 1] / 1000, s=1.0, color=INK, alpha=0.22, linewidths=0, zorder=7,
                       rasterized=True)
            ax.contour(X, Yg, supported.astype(float), levels=[0.5], colors=INK, linewidths=1.1, zorder=8)
            ax.set_title(f"{z.capitalize()}", loc="left", fontweight="bold", color=ZONE_COLORS.get(z, INK))
            ax.set_xlabel("Zone temperature (°F)")
            ax.set_ylabel("Zone load (GW)" if ax is axes[0] else "")
            ax.grid(False)
        cb = fig.colorbar(cs, ax=axes, shrink=0.85, pad=0.02)
        cb.set_label("Fitted |DA − RT| (\\$/MWh)")
        handles = [Patch(facecolor=SEQUENTIAL(0.55), edgecolor=INK, label="Observed support"),
                   Patch(facecolor=extrap, edgecolor="#b9b8b3", label="Extrapolation"),
                   Patch(facecolor=SURFACE, edgecolor=INK, hatch="////", label="Fitted |spread| < 0 (not physical)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.4)
        gas = grids[zones[0]][3].get("gas_price")
        if gas is not None:
            fig.suptitle(f"Gas held at its 2025 median (\\${gas:.2f}/MMBtu)", x=0.01, ha="left",
                         fontsize=8.5, color=INK_2)
        fig.savefig(path, dpi=220)
        plt.close(fig)


def fig_tail_by_net_load(rows: pd.DataFrame, summary: pd.DataFrame, path, sample: str = "all days"):
    """Zone-relative extreme rate (top 1% |spread| per zone) by system net-load decile.
    A: deciles over all hours. B: deciles within each hour of day, a time-of-day
    robustness check (season, weather and renewables are not held fixed). Bars: zones
    pooled; dots: each zone; dashed: the 1% base rate. The top-vs-rest ratio and its
    day-block bootstrap 90% interval are printed in each panel."""
    with plt.rc_context(V2_FONT):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True, sharey=True)
        titles = {"nl_decile": "A. Deciles of system net load",
                  "nl_decile_within_hour": "B. Deciles within each hour of day"}
        for ax, kind in zip(axes, titles):
            r = rows[(rows.decile_kind == kind) & (rows["sample"] == sample)]
            pooled = r[r.zone == "all zones"].sort_values("decile")
            ax.bar(pooled.decile, pooled.rate_pct, color="#9ec5f4", edgecolor=SURFACE, width=0.8,
                   label="All zones", zorder=2)
            for z in ZONES:
                zz = r[r.zone == z].sort_values("decile")
                ax.scatter(zz.decile, zz.rate_pct, s=12, color=ZONE_COLORS[z], zorder=3, label=z.capitalize(),
                           linewidths=0)
            ax.axhline(1.0, color=INK_2, lw=0.9, ls="--", zorder=1)
            ax.text(1.0, 1.15, "1% base rate", fontsize=8, color=INK_2, va="bottom")
            s = summary[(summary.decile_kind == kind) & (summary["sample"] == sample) &
                        (summary.zone == "all zones")].iloc[0]
            ax.text(0.03, 0.97, f"Top decile vs the rest:\n{s.ratio_top_vs_rest:.1f}× "
                    f"(90% CI {s.ratio_ci05:.1f}–{s.ratio_ci95:.1f})",
                    transform=ax.transAxes, va="top", ha="left", color=INK)
            ax.set_xticks(range(1, 11))
            ax.set_xlabel("Net-load decile (1 = loosest, 10 = tightest)")
            ax.set_title(titles[kind], loc="left")
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Extreme hours (% of hours)")
        axes[1].legend(frameon=False, loc="center left", fontsize=8, handletextpad=0.3)
        fig.savefig(path, dpi=220)
        plt.close(fig)


def fig_event_day(hourly: pd.DataFrame, rt15: pd.DataFrame, path, focus: str = "houston"):
    """One operating day, local time, three stacked panels sharing the x-axis; the hours in
    which any zone had a zone-relative extreme are shaded. Each net-load measure appears once
    and is labelled with its definition (D-35):

    1. settled RT 15-minute prices for every zone and the focus zone's hourly DA (symlog);
    2. tightness: realised net load (load - wind and solar generation), the measure ranked
       in the decile figure, against that ranking's top-decile threshold;
    3. surprise: net load expected at 09:45 D-1 vs actual with renewables at available
       capacity, the gap filled where the system was tighter than expected.
    """
    from matplotlib.ticker import FixedLocator

    def x_of(local):
        return local.dt.hour + local.dt.minute / 60

    sysh = hourly[hourly.zone == focus].sort_values("interval_start_utc")
    xh = x_of(sysh["interval_start_local"]).to_numpy()
    edges = np.append(xh, xh[-1] + 1)
    ext_hours = sorted(set(x_of(hourly.loc[hourly.extreme, "interval_start_local"])))
    with plt.rc_context(V2_FONT):
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 3.7), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1.5, 1.0, 1.0]})
        for ax in axes:
            for h0 in ext_hours:
                ax.axvspan(h0, h0 + 1, color="#f2a5a4", alpha=0.25, lw=0, zorder=0)
        a1, a2, a3 = axes
        for z in ZONES:
            r = rt15[rt15.zone == z]
            if r.empty:
                continue
            xr = x_of(r["interval_start_local"]).to_numpy()
            a1.step(np.append(xr, xr[-1] + 0.25), np.append(r.price.to_numpy(), r.price.iloc[-1]), where="post",
                    color=ZONE_COLORS[z], lw=1.8 if z == focus else 0.9, alpha=1 if z == focus else 0.8,
                    label=f"RT 15-min, {z.capitalize()}")
        a1.stairs(sysh.da_price.to_numpy(), edges, baseline=None, color=INK, lw=1.8,
                  label=f"DA hourly, {focus.capitalize()}")
        a1.set_yscale("symlog", linthresh=100, linscale=1.2)
        a1.yaxis.set_major_locator(FixedLocator([0, 50, 100, 300, 1000, 3000]))
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"\\${v:,.0f}"))
        a1.set_ylabel("Price ($/MWh)")
        a1.legend(frameon=False, ncol=3, fontsize=7.8, loc="upper left")
        a1.set_title("Prices: settled RT far above DA in the evening ramp", loc="left")

        gen = sysh.system_net_load_mw.to_numpy() / 1000
        thr = float(sysh.nl_top_decile_threshold.iloc[0]) / 1000
        a2.stairs(gen, edges, baseline=None, color=INK, lw=1.8, label="Realised net load (renewables as generated)")
        a2.axhline(thr, color="#e34948", lw=1.1, ls=":", label=f"Tightest-10% threshold ({thr:.1f} GW)")
        a2.set_ylim(gen.min() - 2, gen.max() + 0.6 * (gen.max() - gen.min()))
        a2.set_ylabel("GW")
        a2.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2, handlelength=1.6)
        a2.set_title("Tightness: system load − wind and solar generation (the Fig. 3 measure)", loc="left")

        fc, av = sysh.net_load_fc.to_numpy() / 1000, sysh.net_load_avail.to_numpy() / 1000
        a3.stairs(fc, edges, baseline=None, color=INK_2, lw=1.5, ls="--", label="Expected at 09:45 D−1")
        a3.stairs(av, edges, baseline=None, color=INK, lw=1.8, label="Actual, renewables at available capacity")
        a3.stairs(np.maximum(av, fc), edges, baseline=fc, fill=True, color="#e34948", alpha=0.35,
                  label="Tighter than expected")
        lo3, hi3 = min(fc.min(), av.min()), max(fc.max(), av.max())
        a3.set_ylim(lo3 - 2, hi3 + 0.6 * (hi3 - lo3))
        a3.set_ylabel("GW")
        a3.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=3, handlelength=1.6, columnspacing=1.0)
        a3.set_title("Surprise: net load vs what the DA market expected", loc="left")
        a3.set_xlim(0, 24)
        a3.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
        a3.set_xlabel(f"{pd.Timestamp(sysh['interval_start_local'].iloc[0]):%A %d %B %Y}, Central time "
                      "(shaded: hours with an extreme spread in any zone)")
        fig.savefig(path, dpi=220)
        plt.close(fig)
