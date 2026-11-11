"""Parsers, request builders, config, and the end-to-end pipeline on synthetic inputs.

None of this touches the network: the sources were unreachable from the
environment the code was written in, so the fetchers' first live run is
the real test of the request shapes. These tests pin everything after the
HTTP call.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ercot_rsm import analysis, config, pipeline, synthetic
from ercot_rsm.data import gas, gridstatus_fetch as gs, weather


@pytest.fixture(scope="module")
def study():
    return config.load()


def test_config_loads_and_validates(study, tmp_path):
    assert set(study.zones) == {"houston", "north", "south", "west"}
    assert study.start.tz is not None
    assert all(abs(sum(z.stations.values()) - 1) < 1e-9 for z in study.zones.values())
    bad = config.DEFAULT_CONFIG.read_text().replace("IAH = 0.5, HOU = 0.5", "IAH = 0.5, HOU = 0.6")
    p = tmp_path / "bad.toml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="sum to"):
        config.load(p)


def test_month_windows_tile_the_study(study):
    w = gs.month_windows(study.start, study.end)
    assert w[0][0] == study.start_utc and w[-1][1] == study.end_utc
    assert all(a < b for a, b in w)
    assert all(w[i][1] == w[i + 1][0] for i in range(len(w) - 1))


def test_normalize_prices_detects_column_and_conflicts():
    t = pd.Timestamp("2026-01-01 06:00", tz="UTC")
    df = pd.DataFrame({"interval_start_utc": [t, t], "location": ["LZ_X", "LZ_X"], "spp": [1.0, 1.0],
                       "market": "DAY_AHEAD_HOURLY"})
    assert len(gs.normalize_prices(df)) == 1  # identical republish collapses
    df.loc[1, "spp"] = 2.0
    with pytest.raises(ValueError, match="conflicting"):
        gs.normalize_prices(df)
    with pytest.raises(KeyError):
        gs.normalize_prices(df.rename(columns={"spp": "value"}))


def test_iem_parser():
    text = "station,valid,tmpf\nIAH,2026-01-01 00:53,61.0\nIAH,2026-01-01 01:53,M\n"
    df = weather.parse_iem(text)
    assert df["valid_utc"].dt.tz is not None
    assert df["tmpf"].iloc[0] == 61.0 and np.isnan(df["tmpf"].iloc[1])
    with pytest.raises(ValueError):
        weather.parse_iem("error\nbad station\n")


def test_iem_params_cover_the_window(study):
    p = dict(weather.iem_params("IAH", study.start, study.end))
    assert p["station"] == "IAH" and p["tz"] == "Etc/UTC"
    assert (int(p["year1"]), int(p["month1"])) == (study.start_utc.year, study.start_utc.month)


def test_open_meteo_parser():
    payload = {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "temperature_2m": [50.1, None]}}
    df = weather.parse_open_meteo(payload, "IAH")
    assert str(df["valid_utc"].dt.tz) == "UTC" and np.isnan(df["tmpf"].iloc[1])


def test_eia_parser_and_csv(tmp_path):
    payload = {"response": {"data": [{"period": "2026-01-02", "value": "3.10"},
                                     {"period": "2026-01-05", "value": "3.25"}]}}
    df = gas.parse_eia(payload)
    assert df["trade_date"].tolist() == [date(2026, 1, 2), date(2026, 1, 5)]
    p = tmp_path / "hh.csv"
    p.write_text("Day,Henry Hub Natural Gas Spot Price Dollars per Million Btu\n01/02/2026,3.10\n01/05/2026,3.25\n")
    assert gas.parse_csv(p)["price"].tolist() == [3.10, 3.25]


def test_eia_workbook_sheet_and_window():
    # The layout of EIA's RNGWHHDd.xls "Data 1" sheet as fetched on 2026-10-01.
    sheet = pd.DataFrame([
        ["Back to Contents", "Data 1: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)"],
        ["Sourcekey", "RNGWHHD"],
        ["Date", "Henry Hub Natural Gas Spot Price (Dollars per Million Btu)"],
        [pd.Timestamp("2024-11-29"), 3.05],
        [pd.Timestamp("2025-01-02"), 3.10],
        [pd.Timestamp("2025-01-03"), None],
        [pd.Timestamp("2025-01-06"), 3.25],
    ])
    df = gas.parse_eia_sheet(sheet, "RNGWHHD")
    assert df["trade_date"].tolist() == [date(2024, 11, 29), date(2025, 1, 2), date(2025, 1, 6)]
    start, end = (pd.Timestamp(d, tz="America/Chicago") for d in ("2025-01-01", "2025-01-06"))
    # 31 days of lead-in reaches back to 2024-12-01, not to November
    assert gas.in_window(df, start, end)["trade_date"].tolist() == [date(2025, 1, 2), date(2025, 1, 6)]
    with pytest.raises(ValueError, match="not an EIA workbook"):
        gas.parse_eia_sheet(sheet, "RNGC1")


@pytest.fixture(scope="module")
def short_panel(study):
    # Two months spanning the fall-back DST change keep this fast but non-trivial.
    s = config.Study(**{**study.__dict__,
                        "start": pd.Timestamp("2025-10-15", tz=study.market_tz),
                        "end": pd.Timestamp("2025-12-15", tz=study.market_tz)})
    return s, pipeline.build_panel(synthetic.make(s), s)


def test_panel_shape_and_coverage(short_panel):
    s, panel = short_panel
    expected = int((s.end_utc - s.start_utc) / pd.Timedelta(hours=1))
    assert expected == 61 * 24 + 1  # one 25-hour day
    q = pipeline.quality_report(panel, s)
    assert (q["hours_with_spread"] == expected).all()
    assert not panel.duplicated(["zone", "interval_start_utc"]).any()
    for col in ("spread", "imhr_da", "temp_f", "load_norm", "gas_price", "on_peak", "regime"):
        assert panel[col].notna().mean() > 0.95, col
    assert set(panel["regime"]) == {"pre", "post"}


def test_fit_runs_on_panel(short_panel, tmp_path):
    s, panel = short_panel
    canon = analysis.run(panel, s, tmp_path / "out", tmp_path / "fig")
    assert len(canon) == len(s.section("rsm")["responses"]) * 4
    comp = pd.read_csv(tmp_path / "out" / "model_comparison.csv")
    assert set(comp["model"]) == {"M0", "M1", "M2", "M3", "ALT"}
    assert set(comp["scheme"]) == {"lomo"}  # 3 months: too short for forward CV, which is skipped
    assert (tmp_path / "out" / "sensitivity.csv").exists() and (tmp_path / "fig" / "model_comparison.png").exists()
    assert (tmp_path / "fig" / "surface_spread.png").exists()
    assert (tmp_path / "out" / "lack_of_fit.csv").exists()


def test_window_override_keys_every_cache(study):
    sample = config.load(start="2025-07-07", end="2025-07-14")
    assert sample.tag == "2025-07-07_2025-07-14" and study.tag == "2025-01-01_2026-01-01"
    for attr in ("raw_dir", "processed_dir", "results_dir"):
        assert getattr(sample, attr) != getattr(study, attr)
    assert gs.raw_dir(sample) != gs.raw_dir(study)


def _write_raw_like_fetchers(study):
    """Synthetic inputs laid out on disk exactly as the fetchers leave them."""
    raw = synthetic.make(study)
    g = gs.raw_dir(study)
    g.mkdir(parents=True)
    raw.da.to_parquet(g / "da_prices.parquet")
    raw.rt.to_parquet(g / "rt_prices.parquet")
    raw.load_wide.to_parquet(g / "load_wide.parquet")
    raw.fuel_mix.to_parquet(g / "fuel_mix_hourly.parquet")
    (study.raw_dir / "weather").mkdir(parents=True)
    raw.station_obs.to_parquet(study.raw_dir / "weather" / "stations_iem.parquet")
    (study.raw_dir / "gas").mkdir(parents=True)
    raw.gas.to_parquet(study.raw_dir / "gas" / "henry_hub_daily.parquet")
    # settled-price archive layout (the configured price source, D-31)
    (study.raw_dir / "ercot").mkdir(parents=True)
    raw.da.assign(point_type="DA").to_parquet(study.raw_dir / "ercot" / "da_prices_archive.parquet")
    raw.rt.assign(point_type="LZ").to_parquet(study.raw_dir / "ercot" / "rt_prices_archive.parquet")


def test_peek_on_a_sample_week_offline(tmp_path, monkeypatch, capsys):
    from ercot_rsm import cli

    monkeypatch.setattr(config, "ROOT", tmp_path)
    week = config.load(start="2025-11-01", end="2025-11-08")  # contains the fall-back day
    _write_raw_like_fetchers(week)
    cli.main(["--start", "2025-11-01", "--end", "2025-11-08", "peek"])
    out = capsys.readouterr().out
    for section in ("DA prices", "RT prices", "Load by forecast zone", "Station temperatures",
                    "Henry Hub gas", "PANEL", "Coverage"):
        assert section in out, section
    assert "median step: 0 days 00:15:00" in out  # RT is 15-minute
    q = pd.read_csv(week.processed_dir / "quality_report.csv", index_col=0)
    assert (q["expected_hours"] == 7 * 24 + 1).all()  # 25-hour Nov 2
    assert (q["hours_with_spread"] == q["expected_hours"]).all()
    assert (week.processed_dir / "peek.txt").exists()
    # the full-year paths were never touched
    assert not config.load().processed_dir.exists()


def test_build_refuses_a_window_that_was_never_fetched(tmp_path, monkeypatch):
    from ercot_rsm import cli

    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="run fetch for this window"):
        cli.main(["--start", "2025-03-01", "--end", "2025-03-08", "build"])


def test_sample_refuses_a_long_window():
    from ercot_rsm import cli

    with pytest.raises(SystemExit, match="allow-long"):
        cli.main(["sample"])  # the configured full year


def test_default_fetch_downloads_everything_build_needs(tmp_path, monkeypatch):
    from ercot_rsm import cli
    from ercot_rsm.data import ercot_archive

    monkeypatch.setattr(config, "ROOT", tmp_path)
    calls = {}
    monkeypatch.setattr(gs, "fetch_all", lambda study, force=False, keys=(): calls.setdefault("grid", list(keys)))
    monkeypatch.setattr(weather, "fetch_all", lambda study, force=False: calls.setdefault("weather", True))
    monkeypatch.setattr(gas, "fetch", lambda study, force=False: calls.setdefault("gas", True))
    monkeypatch.setattr(ercot_archive, "fetch",
                        lambda study, kind, force=False: calls.setdefault("archive", []).append(kind))
    cli.main(["--start", "2025-03-01", "--end", "2025-03-08", "fetch"])
    assert set(cli.GRID_KEYS) <= set(calls["grid"])
    assert set(cli.FORECAST_KEYS) <= set(calls["grid"])          # DA-time forecasts (D-28)
    assert calls["archive"] == ["da", "rt"]                        # settled prices (D-31)
    assert calls["weather"] and calls["gas"]


def test_build_refuses_a_panel_without_forecasts(tmp_path, monkeypatch):
    from ercot_rsm import cli

    monkeypatch.setattr(config, "ROOT", tmp_path)
    _write_raw_like_fetchers(config.load(start="2025-11-01", end="2025-11-08"))  # prices, load, weather, gas only
    with pytest.raises(SystemExit, match="fetch --only forecasts"):
        cli.main(["--start", "2025-11-01", "--end", "2025-11-08", "build"])
