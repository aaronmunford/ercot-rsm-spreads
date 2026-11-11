# Data: sources, alignment, validation and dictionary

Study window: **operating days 2025-01-01 to 2025-12-31** in Central Prevailing Time
(`America/Chicago`), i.e. `[2025-01-01 06:00 UTC, 2026-01-01 06:00 UTC)`. That is 8,760
hours, including the 23-hour 2025-03-09 and the 25-hour 2025-11-02.

The data was pulled on 2026-10-09. Rebuild it with:

```bash
set -a; source .env; set +a
make fetch crosscheck build audit
```

Every cache is reused, so a re-run costs no API quota. Provenance for every raw file is in
`data/raw/2025-01-01_2026-01-01/manifest.json`. Each entry records the source, dataset id,
query, filters, retrieval time, units, timezone, geography, per-chunk row counts and
validation, and the API usage before and after the pull.

**The audit:** `data/processed/2025-01-01_2026-01-01/audit/audit.md` holds the results, and
the CSVs alongside it hold the detail.

## Sources

| Data | Source | Id / URL | Native resolution | Units | Rows (2025) |
|---|---|---|---|---|---:|
| Day-ahead prices | gridstatus.io API | `ercot_spp_day_ahead_hourly`, `location in (LZ_HOUSTON, LZ_NORTH, LZ_SOUTH, LZ_WEST)` | hourly | $/MWh (settlement point price) | 35,040 |
| Real-time prices | gridstatus.io API | `ercot_spp_real_time_15_min`, same filter | 15 min | $/MWh (SPP) | 140,160 |
| Real-time load | gridstatus.io API | `ercot_load_by_forecast_zone` (ERCOT NP6-346, "Actual System Load by Forecast Zone") | hourly | MW | 8,760 |
| Generation mix | gridstatus.io API | `ercot_fuel_mix`, server-side `resample=1 hour, mean` | 5 min → hourly | MW | 8,760 |
| Temperature | Iowa Environmental Mesonet ASOS archive | `mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (NWS/FAA ASOS METARs, routine and special) | ~hourly (:51–:56) plus specials | °F | 153,212 obs, 15 stations |
| Natural gas | U.S. EIA (spot data credited to Refinitiv/LSEG) | series `RNGWHHD`, Henry Hub spot, `eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls` | daily, trade dates | $/MMBtu | 248 trade dates in 2025, plus a 31-day lead-in |

Each gridstatus dataset was pulled as 13 monthly requests, split at UTC month boundaries
with the end exclusive. Every request was filtered server-side to the four load-zone
settlement points; nothing was downloaded unfiltered.

### Cross-check sources (validation only, not in the panel)

| Check | Source | Result |
|---|---|---|
| Spread sign and alignment | gridstatus `ercot_spp_dart_15_min`, 2025-07-07..14, 4 zones (2,688 rows) | gridstatus `spp` = our DA(hourly) − RT(15-min) **exactly** (max abs diff 0.0000). Against RT − DA the max diff is $3,429. The gridstatus DART convention is DA − RT, the same as ours. |
| Gas | FRED `DHHNGSP` (no key) | 269 of 269 trade dates identical, 0 mismatches |
| Temperature | NOAA NCEI Global Hourly (ISD) FM-15/16 METARs; IAH Jul 2025, DFW Jan 2025 | Mean abs diff 0.04–0.05 °F, max 0.08 °F (°C-to-°F rounding); 100% within 1 °F |
| Load total | ERCOT `Native_Load_2025.xlsx` (load_hist page, weather zones + ERCOT total) | Correlation 0.997, MAPE 1.33%, mean +704 MW (gridstatus higher); see the note below |

## Validation summary (2025, all four zones)

- **Prices:** DA 8,760/8,760 hours and RT 35,040/35,040 intervals per zone, with zero
  duplicates and zero timestamps outside the window. **Every RT hour has all four 15-minute
  intervals.**
- **Load and fuel mix:** 8,760/8,760 hours. The four forecast zones sum to the `total`
  column to within 0.02 MW.
- **Zonal temperature:** 8,760/8,760 hours per zone after the station rules below.
- **Station coverage:** Waco (ACT, weight 0.1 in North) has readings in only 86% of hours,
  with a longest gap of 73 h; every other station is ≥90%. No reading falls outside
  −20..125 °F.
- **Gas:** two gaps longer than 4 calendar days, both at holidays (Thanksgiving: 11-26 →
  12-01; Christmas: 12-24 → 12-29). These are bridged by the previous-trade-date rule.
- **DST:** 2025-03-09 has 23 rows and 2025-11-02 has 25 rows per zone, with prices, load
  and temperature on every row.
- **Estimation sample:** `model_ok` is true for **35,040 of 35,040** zone-hours. Nothing is
  imputed except station-level temperature gaps of ≤3 h.
- **15-minute RT extremes:**
  - negative intervals: Houston 584, North 891, South 878, West 1,974;
  - intervals ≥ $1,000/MWh: 8, 13, 20 and 13 respectively;
  - none at the $5,000 offer cap;
  - maximum $3,860 (Houston).

  Hourly-averaged RT peaks range from $1,450 (South) to $2,189 (North).

### Load cross-check note

The gridstatus forecast-zone total exceeds ERCOT native load by +704 MW on average. The
timestamps align: a ±1 h shift makes the fit worse (correlation 0.98 instead of 0.997).

- **When the gap occurs:** it is largest overnight and at 08–10 local, and grows from about
  530 MW in January to about 950 MW in October.
- **The December drop:** it falls to about 176 MW in December, after the RTC+B go-live on
  2025-12-05.
- **Hypothesis:** this pattern fits telemetered forecast-zone load counting battery (ESR)
  charging while native load does not. The gap is negatively correlated with net storage
  output (r = −0.30).

This is a plausible explanation, not a verified one. It does not affect alignment, but it
means the zonal load measure includes some storage charging.

## Zone mapping

The assignment asks for *zonal* data. ERCOT has three different zone systems, and only two
of them line up.

| Price: load zone (settlement point) | Load: forecast zone | Temperature stations (weight) |
|---|---|---|
| `LZ_HOUSTON` | Houston | IAH 0.5, HOU 0.5 |
| `LZ_NORTH` | North | DFW 0.5, DAL 0.4, ACT 0.1 |
| `LZ_SOUTH` | South | SAT 0.38, AUS 0.33, MFE 0.13, CRP 0.06, BRO 0.06, LRD 0.04 |
| `LZ_WEST` | West | MAF 0.35, ODO 0.35, ABI 0.18, SJT 0.12 |

- **Forecast zones** (North, South, West, Houston) have the boundaries of ERCOT's 2003
  congestion management zones (ERCOT NPRR1048). They therefore share names with the four
  competitive load zones.
- **The NOIE load zones** (`LZ_AEN`, `LZ_CPS`, `LZ_LCRA`, `LZ_RAYBN`) are settled
  separately, but they lie inside these regions. The market monitor maps CPS, AEN and LCRA
  into South and Rayburn into North.
- **Consequence:** a forecast-zone load covers a geographically larger population than the
  `LZ_*` price does, especially in South. This is a limitation (D-03).
- **Evidence:** the forecast zones sum to the system total, so all ERCOT load, NOIE
  included, sits in one of the four.
- **ERCOT's eight weather zones** (COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT, WEST)
  are a third system. They appear only in the native-load cross-check, and the station
  choice does not rely on them.

## Temporal alignment

- **Join key:** every join is on `interval_start_utc`, interval-beginning and in UTC. Local
  time is derived from it and never joined on, so the repeated 01:00 on 2025-11-02 cannot
  collide.
- **RT to hourly:** each hour's RT price is the mean of its four 15-minute SPPs. The
  intervals have equal length, so this is exactly interval-length weighting. `rt_n` counts
  the intervals present and `rt_complete` requires all four. A short hour is never averaged
  silently; there were none in 2025.
- **DA:** the hourly DA SPP sits on the same hour-beginning grid.
- **Load:** the hourly actual load is on the same grid.
- **Temperature:**
  - METAR observations, usually at :51–:56, are rounded to the nearest hour, and the
    readings within an hour are averaged per station.
  - Station gaps of ≤3 h are linearly interpolated; longer gaps stay missing.
  - The zone value is the weighted mean, renormalised over the stations that are reporting.
    It is set to missing if those stations carry less than 50% of the zone's weight.
- **Gas:** operating day *D* uses the Henry Hub price from the **last trade date strictly
  before D**. DA power for D clears on D−1, and gas for D (and the weekend strip) trades on
  the prior business day.
  - Each daily price is therefore repeated across the 23–25 hours of an operating day. It is
    one observation per day, not 24.
  - Any inference involving gas must account for this. Cluster by day, or use HAC with at
    least 24 lags (D-15).

## Data dictionary (`data/processed/2025-01-01_2026-01-01/panel.parquet`)

The panel has one row per zone and hour: 35,040 rows.

| Column | Type | Units | Definition |
|---|---|---|---|
| `interval_start_utc` | datetime, UTC | — | Hour beginning; the join key |
| `interval_start_local` | datetime, America/Chicago | — | Same instant in Central Prevailing Time |
| `zone` | str | — | houston / north / south / west |
| `location` | str | — | ERCOT settlement point (`LZ_*`) |
| `operating_date` | date | — | Local calendar date of the hour |
| `hour_beginning` | int | 0–23 | Local hour beginning (ERCOT HE = hour_beginning + 1) |
| `weekday` | int | 0 = Mon | Local day of week |
| `month` | str | YYYY-MM | Local month |
| `on_peak` | bool | — | Weekday, not a NERC holiday, HE07–HE22 |
| `regime` | str | — | `pre` / `post` the RTC+B go-live, operating day 2025-12-05 |
| `da_price` | float | $/MWh | Day-ahead SPP |
| `rt_price` | float | $/MWh | Mean of the four 15-min real-time SPPs in the hour |
| `rt_intrahour_std` | float | $/MWh | Sample std of the four 15-min RT SPPs (intra-hour volatility) |
| `rt_n` | int | count | 15-min RT intervals present (4 expected) |
| `rt_complete` | bool | — | `rt_n == 4` |
| `spread` | float | $/MWh | **DA − RT**: positive means DA settled above RT (a DA premium) |
| `abs_spread` | float | $/MWh | `abs(spread)` |
| `gas_price` | float | $/MMBtu | Henry Hub spot, last trade date before the operating day |
| `imhr_da` | float | MMBtu/MWh | `da_price / gas_price`; NaN if gas < $0.50 (never in 2025) |
| `imhr_rt` | float | MMBtu/MWh | `rt_price / gas_price`. **Mechanically contains RT price**, so never use it as a regressor for the spread |
| `imhr_spread` | float | MMBtu/MWh | `imhr_da − imhr_rt` = spread / gas |
| `load_mw` | float | MW | Hourly actual load, ERCOT forecast zone |
| `load_norm` | float | 0–1 | `load_mw` / the zone's 2025 maximum |
| `load_z` | float | sd | `load_mw` standardised within zone |
| `load_anom` | float | MW | `load_mw` − the zone's median for that hour-of-week |
| `system_load_mw` | float | MW | Sum of the four zones' load (= ERCOT total) |
| `renewables_mw` | float | MW | ERCOT system wind + solar generation, hourly mean |
| `system_net_load_mw` | float | MW | `system_load_mw − renewables_mw` (system-level, the same for all zones) |
| `temp_f` | float | °F | Zone temperature: weighted mean of the station temperatures |
| `temp_coverage` | float | 0–1 | Share of the zone's station weight reporting that hour |
| `cdh`, `hdh` | float | °F·h | Cooling / heating degree-hours, base 65 °F |
| `model_ok` | bool | — | RT complete, and DA, RT, load, temp, gas and IMHR all present: the estimation sample |

## Licences and attribution

- **gridstatus.io:** data is used under the free-plan terms. Raw pulls are gitignored and not
  redistributed.
- **EIA, NOAA/NCEI and FRED:** public data (FRED republishes the EIA series).
- **IEM:** ASOS archive, data courtesy of Iowa State University.
- **ERCOT:** load archive, a public market information file.

## Additions (2026-10-10)

### Prices: ERCOT settled archive (D-31)

The panel's DA and RT prices now come from ERCOT's annual archives:
- `DAMLZHBSPP_<year>.zip` (NP4-180-ER, report type 13060);
- `RTMLZHBSPP_<year>.zip` (NP6-785-ER, report type 13061).

**Contents.** The archives cover all 15 hubs and load zones. The RT load-zone series uses point type `LZ`.

**Where it is cached.** `data/raw/<window>/ercot/{da,rt}_prices_archive.parquet`, with the original zips beside it.

**Comparison with gridstatus.** gridstatus is kept as a cross-check. It holds prices as first published, and ERCOT corrected 2025 prices on 8 days. The details are in `audit/price_source_crosscheck.csv`.

### DA-time forecasts and actuals (D-28)

**Datasets.** All come from gridstatus, with one row per hour:

| Dataset | Pulled with |
|---|---|
| `ercot_load_forecast_by_forecast_zone` (NP3-560-CD) | `publish_time="latest_before:-1 day@09:45:00"`, `timezone="market"` |
| `ercot_wind_actual_and_forecast_by_geo_region_hourly` (NP4-742-CD) | the same |
| `ercot_solar_actual_and_forecast_by_geo_region_hourly` (NP4-745-CD) | the same |
| The same three at `@09:00:00` | sensitivity |
| Wind and solar actuals | `publish_time="latest"` |

**Vintages selected.** These are the 09:30 load report and the 08:55 wind and solar reports.
- **Audit:** every row was published before its cutoff, and horizons are 14.5–38.5 h.
- **Where it is recorded:** the per-chunk vintage audit (lag, horizon, publish minutes, stale rows) is in `manifest.json`.

**New panel columns** (MW unless noted; error = actual − forecast):

| Column | Definition |
|---|---|
| `load_fc_mw`, `load_err` | Zonal DA-time load forecast, and its error |
| `sys_load_fc`, `sys_load_err` | System load forecast, and its error |
| `wind_fc`, `wind_act`, `wind_err` | Wind STWPF forecast, actual generation, and generation error |
| `wind_err_hsl` | Wind availability (HSL) error |
| `solar_fc`, `solar_act`, `solar_err`, `solar_err_hsl` | The same for solar (STPPF) |
| `wind_fc_west`, `wind_err_west` | West + Panhandle regions |
| `net_load_fc` | `sys_load_fc − wind_fc − solar_fc` |
| `net_load_act` | Actual net load |
| `net_load_err` | Generation-based net-load error |
| `net_load_err_avail` | Load error − wind HSL error − solar HSL error. **The primary surprise.** |
| `net_load_ramp` | Realised Δ net load, per hour |
| `net_load_fc_ramp` | Forecast Δ net load (D−2-cutoff value for hour 23 of D−1, published before D's cutoff) |
| `*_sens` | The same, at the 09:00 vintage |
| `gas_price_exante` | Henry Hub from the last trade date before D−1, i.e. known at 09:45 on D−1 ($/MMBtu) |
