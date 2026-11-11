"""Command line: ``python -m ercot_rsm.cli [--start D --end D] <command>``.

    datasets   list gridstatus.io ERCOT dataset ids and check the configured ones
    sample     fetch + build + print every source and the joined panel (use on ~1 week)
    fetch      download prices, load, weather and gas into data/raw/<window>/
               (--only grid|da|rt|load|fuel_mix|weather|gas|ercot|crosscheck)
    build      assemble data/processed/<window>/panel.parquet and a coverage report
    audit      data audit -> data/processed/<window>/audit/ (read audit.md before modelling)
    eda        exploratory tables and figures -> results/<window>/eda/
    surprise   forecast-surprise ladder A-D, surprise surface S, ex-ante model E -> results/<window>/surprise/
    crossyear  replication table and fit-on-one-year/predict-the-next test -> results/cross_year/ (--years 2024 2025)
    peek       print the raw sources and panel for a window already fetched
    fit        fit every zone x response -> results/<window>/ (tables, summaries, figures)
    report-assets  the tables and five figures behind the report -> results/<window>/report/
    synthetic  run build + fit end to end on synthetic inputs (results/synthetic/)

``--start``/``--end`` override the window in config/study.toml. Every cache
and output is keyed by the window, so samples never mix with the full year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import analysis, basis, config, crossyear, eda, peek, pipeline, surprise, synthetic, validate
from .config import ROOT


def _load_raw(study) -> pipeline.RawInputs:
    from .data import gas, gridstatus_fetch as gs, weather

    from .data import ercot_archive

    raw = gs.raw_dir(study)
    pcfg = study.section("prices")
    arch = study.raw_dir / "ercot"
    use_archive = pcfg.get("source") == "ercot_archive"
    price_files = (["load_wide"] if use_archive else ["da_prices", "rt_prices", "load_wide"])
    missing = [f for f in price_files if not (raw / f"{f}.parquet").exists()]
    if use_archive:
        missing += [f for f in ("da_prices_archive", "rt_prices_archive") if not (arch / f"{f}.parquet").exists()]
    if missing:
        raise SystemExit(f"no {', '.join(missing)} for {study.tag}; run fetch for this window first")
    if use_archive:
        points = gs.price_points(study)
        da = ercot_archive.to_prices(pd.read_parquet(arch / "da_prices_archive.parquet"), points, None)
        rt = ercot_archive.to_prices(pd.read_parquet(arch / "rt_prices_archive.parquet"), points,
                                     pcfg.get("rt_point_type", "LZ"))
    else:
        da, rt = pd.read_parquet(raw / "da_prices.parquet"), pd.read_parquet(raw / "rt_prices.parquet")
    fm = raw / "fuel_mix_hourly.parquet"
    from .data.gridstatus_fetch import all_specs
    specs = all_specs(study)
    fc_files = {k: raw / f"{specs[k].name}.parquet" for k in FORECAST_KEYS}
    forecasts = {k: pd.read_parquet(f) for k, f in fc_files.items() if f.exists()}
    if forecasts and not {"load_fc", "wind_fc", "solar_fc", "wind_act", "solar_act"} <= set(forecasts):
        raise SystemExit(f"incomplete forecast set for {study.tag}: have {sorted(forecasts)}")
    return pipeline.RawInputs(
        da=da,
        rt=rt,
        load_wide=pd.read_parquet(raw / "load_wide.parquet"),
        station_obs=weather.fetch_all(study),  # cached per window after the first call
        gas=gas.fetch(study),
        fuel_mix=pd.read_parquet(fm) if fm.exists() else None,
        forecasts=forecasts or None,
    )


def cmd_datasets(study, args):
    from .data import gridstatus_fetch as gs

    cfg = study.section("gridstatus")
    ok = gs.check_datasets(gs.client(), [cfg["da_dataset"], cfg["rt_dataset"],
                                         cfg["load_dataset"], cfg["fuel_mix_dataset"]])
    for k, v in ok.items():
        print(f"  {'ok     ' if v else 'MISSING'} {k}")


GRID_KEYS = ("da", "rt", "load", "fuel_mix")
FORECAST_KEYS = ("load_fc", "wind_fc", "solar_fc", "load_fc_sens", "wind_fc_sens", "solar_fc_sens",
                 "wind_act", "solar_act")
CROSSCHECK_STATIONS = (("IAH", "2025-07"), ("DFW", "2025-01"))  # a summer and a winter month


def cmd_fetch(study, args):
    from .data import ercot_official, gas, gridstatus_fetch as gs, weather

    print(f"window {study.tag} -> {study.raw_dir}")
    only = set(args.only or ["grid", "weather", "gas"])
    keys = [k for k in GRID_KEYS if k in only or "grid" in only]
    keys += [k for k in FORECAST_KEYS if k in only or "forecasts" in only]
    if args.no_fuel_mix:
        keys = [k for k in keys if k != "fuel_mix"]
    if keys:
        gs.fetch_all(study, force=args.force, keys=keys)
    if "weather" in only:
        weather.fetch_all(study, force=args.force)
    if "gas" in only:
        gas.fetch(study, force=args.force)
    if "ercot" in only:
        ercot_official.fetch_native_load(study, force=args.force)
    if "prices" in only or "ercot" in only:
        from .data import ercot_archive
        ercot_archive.fetch(study, "da", force=args.force)
        ercot_archive.fetch(study, "rt", force=args.force)
    if "crosscheck" in only:
        gs.fetch_dart_crosscheck(study, force=args.force)
        print("FRED:", gas.crosscheck_fred(study, gas.fetch(study)))
        obs = weather.fetch_all(study)
        for st, month in CROSSCHECK_STATIONS:
            print("NCEI:", weather.ncei_crosscheck(study, st, month, obs))


def _build(study, raw, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pipeline.build_panel(raw, study)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    q = pipeline.quality_report(panel, study)
    q.to_csv(out_dir / "quality_report.csv")
    return panel, q


def cmd_build(study, args):
    _, q = _build(study, _load_raw(study), study.processed_dir)
    print(q.to_string())


def cmd_audit(study, args):
    raw = _load_raw(study)
    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.processed_dir / "audit"
    print(validate.run(study, raw, panel, out).to_string(index=False))
    print(f"\n(see {out / 'audit.md'})")


def cmd_eda(study, args):
    from . import plots

    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.results_dir / "eda"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tables = eda.run(panel)
    for name, t in tables.items():
        t.to_csv(out / "tables" / f"{name}.csv")
    ex = eda.extremes(panel)
    plots.fig_spread_distribution(panel, tables["q1_spread_summary"], out / "figures" / "fig1_spread_distribution.png")
    plots.fig_temp_load(panel, tables["q2_temp_load"], out / "figures" / "fig2_temp_load.png")
    plots.fig_extremes_by_hour(panel, tables["q5_spread_quantiles_by_hour"], ex["by_hour"],
                               out / "figures" / "fig3_extremes_by_hour.png")
    plots.fig_extremes_compact(panel, ex["by_hour"], out / "figures" / "fig3_extremes_compact.png")
    print(f"{len(tables)} tables and 3 figures -> {out}")


def cmd_surprise(study, args):
    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.results_dir / "surprise"
    tables = surprise.run(panel, study, out, out / "figures")
    print(f"{len(tables)} tables -> {out}")


def cmd_crossyear(study, args):
    years = sorted(int(y) for y in args.years)
    studies = [config.load(args.config, f"{y}-01-01", f"{y + 1}-01-01") for y in years]
    out = ROOT / "results" / "cross_year"
    out.mkdir(parents=True, exist_ok=True)
    rep = crossyear.replication_table(studies)
    rep.to_csv(out / "replication.csv", index=False)
    panels = [pd.read_parquet(s.processed_dir / "panel.parquet") for s in studies]
    oy = pd.concat([crossyear.out_of_year(panels[i], panels[i + 1], studies[i + 1].section("rsm"))
                    for i in range(len(studies) - 1)], ignore_index=True)
    oy.to_csv(out / "out_of_year.csv", index=False)
    print(rep.to_string(index=False))
    print(oy.round(3).to_string(index=False))


def cmd_basis(study, args):
    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.results_dir / "basis"
    tables = basis.run(panel, study, out, out / "figures")
    for k in ("variance_shares_hubavg", "variance_shares_exwest"):
        print(tables[k].round(3).to_string(index=False))


def cmd_peek(study, args):
    raw = _load_raw(study)
    panel, q = _build(study, raw, study.processed_dir)
    text = peek.report(raw, panel, q)
    (study.processed_dir / "peek.txt").write_text(text)
    print(text)
    print(f"\n(saved to {study.processed_dir / 'peek.txt'})")


def cmd_sample(study, args):
    days = (study.end - study.start).days
    if days > 31 and not args.allow_long:
        raise SystemExit(f"sample window is {days} days; pass --start/--end for ~1 week, or --allow-long")
    args.only, args.no_fuel_mix, args.force = None, False, False
    cmd_fetch(study, args)
    cmd_peek(study, args)


def cmd_fit(study, args):
    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.results_dir
    out.mkdir(parents=True, exist_ok=True)
    print(analysis.correlation_check(panel, study.section("rsm")["factors"]).to_string())
    analysis.describe(panel).to_csv(out / "describe.csv")
    print(analysis.run(panel, study, out, out / "figures").to_string())


def cmd_report_assets(study, args):
    """Report inputs: the tables behind the report and its five figures, numbered as in the report."""
    from . import plots

    panel = pd.read_parquet(study.processed_dir / "panel.parquet")
    out = study.results_dir / "report"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    zs = eda.zonal_summary(panel)
    zs.to_csv(out / "zonal_summary.csv")
    eda.tail_vs_normal(panel).to_csv(out / "tail_vs_normal.csv")
    analysis.curvature_contribution(panel, study).to_csv(out / "curvature_contribution.csv", index=False)
    # Q6: what makes an extreme hour different (D-35)
    rows, summ = eda.tail_by_net_load(panel)
    rows.to_csv(out / "tail_by_net_load.csv", index=False)
    summ.to_csv(out / "tail_lift_summary.csv", index=False)
    eda.tail_by_net_load_fixed(panel).to_csv(out / "tail_fixed_threshold.csv", index=False)
    eda.extreme_profile(panel).to_csv(out / "extreme_profile.csv", index=False)
    plots.fig_tail_by_net_load(rows, summ, out / "figures" / "fig3_tail_net_load.png")
    day = (pd.Timestamp(args.event_day).date() if args.event_day else eda.representative_event_day(panel))
    from .data import ercot_archive, gridstatus_fetch as gs

    arch = study.raw_dir / "ercot" / "rt_prices_archive.parquet"
    if not arch.exists():
        raise SystemExit(f"no {arch}; the event figure needs ERCOT's 15-minute settled RT archive")
    rt15 = ercot_archive.to_prices(pd.read_parquet(arch), gs.price_points(study),
                                   study.section("prices").get("rt_point_type", "LZ"))
    hourly, rt = eda.event_window(panel, rt15, day)
    hourly.to_csv(out / f"event_{day}_hourly.csv", index=False)
    rt.to_csv(out / f"event_{day}_rt15.csv", index=False)
    (out / "event_day.txt").write_text(f"{day}\n")
    from . import surprise as sp

    sp.event_forward_predictions(panel, study.section("rsm"), day).to_csv(out / f"event_{day}_model_predictions.csv",
                                                                          index=False)
    eda.temp_load_forward(panel).to_csv(out / "temp_load_forward.csv")
    plots.fig_event_day(hourly, rt, out / "figures" / "fig4_event_day.png")
    plots.fig_spread_simple(panel, eda.spread_summary(panel), out / "figures" / "fig1_spread.png")
    plots.fig_temp_load_simple(panel, eda.temp_load(panel), out / "figures" / "fig2_temp_load.png")
    cfg = study.section("rsm")
    m2 = analysis.specs(cfg)["M2"]
    zones = [z for z in args.surface_zones if z in set(panel["zone"])]
    data = {z: analysis.prepare(panel, z, "abs_spread", cfg) for z in zones}
    fits = {z: analysis.fit_spec(data[z], "abs_spread", m2, cfg) for z in zones}
    plots.fig_surface_support(fits, data, out / "figures" / "fig5_surface.png")
    print(zs.round(2).T.to_string())
    print(f"report tables and 5 figures -> {out} (event day {day})")


def cmd_synthetic(study, args):
    raw = synthetic.make(study)
    panel, q = _build(study, raw, ROOT / "data" / "synthetic")
    print(q.to_string())
    out = ROOT / "results" / "synthetic"
    canon = analysis.run(panel, study, out, out / "figures", label="  [SYNTHETIC DATA]")
    print(canon.to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="ercot_rsm")
    p.add_argument("--config", default=str(config.DEFAULT_CONFIG))
    p.add_argument("--start", help="override study start (YYYY-MM-DD, local, inclusive)")
    p.add_argument("--end", help="override study end (YYYY-MM-DD, local, exclusive)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets")
    s = sub.add_parser("sample")
    s.add_argument("--allow-long", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("--only", nargs="*",
                   choices=["grid", *GRID_KEYS, "forecasts", *FORECAST_KEYS, "weather", "gas", "ercot", "prices",
                            "crosscheck"])
    f.add_argument("--force", action="store_true")
    f.add_argument("--no-fuel-mix", action="store_true")
    sub.add_parser("build")
    sub.add_parser("audit")
    sub.add_parser("eda")
    sub.add_parser("surprise")
    sub.add_parser("basis")
    cy = sub.add_parser("crossyear")
    cy.add_argument("--years", nargs="+", default=["2024", "2025"])
    sub.add_parser("peek")
    sub.add_parser("fit")
    ra = sub.add_parser("report-assets")
    ra.add_argument("--surface-zones", nargs="+", default=["houston", "west"])
    ra.add_argument("--event-day", help="operating day for the event figure (default: eda.representative_event_day)")
    sub.add_parser("synthetic")
    args = p.parse_args(argv)
    study = config.load(args.config, args.start, args.end)
    {"datasets": cmd_datasets, "sample": cmd_sample, "fetch": cmd_fetch, "build": cmd_build, "audit": cmd_audit, "eda": cmd_eda, "surprise": cmd_surprise, "crossyear": cmd_crossyear, "basis": cmd_basis,
     "peek": cmd_peek, "fit": cmd_fit, "report-assets": cmd_report_assets, "synthetic": cmd_synthetic}[args.cmd](study, args)


if __name__ == "__main__":
    main()
