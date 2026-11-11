# Decision log

Every modelling and data choice in the study, with what the code does, the alternatives
considered, and why the current option was chosen. Entries are numbered in the order the
choices were made; where a choice was later revised, the original reasoning is kept below
the decision.

The tags (D-01 …) match the comments in `config/study.toml`. Page references like
"330-6" are to NCSS Chapter 330, *Response Surface Regression*, and the engine reproduces that chapter's worked example (the Odor data) to every
printed digit; see `tests/test_ncss_odor.py`.

---

## Data scope

### D-01 Study window: calendar 2025 (operating days, Central time)
"One recent year." There were two clean options:

| Option | For | Against |
|---|---|---|
| **Calendar 2025** (chosen) | A complete, settled calendar year, easy to cite. **Almost all of it falls in one RT market regime**: only the last ~4 weeks come after RTC+B (D-02) | About 10 months older than the alternative |
| Trailing 12 months, Oct 2025 – Sep 2026 | Most recent, and also a complete year | It straddles RTC+B about 2 months in |

(Calendar 2026 would be incomplete, but the trailing window would not have been: a
"recent year" doesn't have to be a calendar year.)

The window is a complete calendar year, chosen so that the sample sits almost entirely
within a single real-time market design. The December post-RTC+B weeks
stay in the sample, absorbed by the regime block. Dropping them is a one-line robustness
check.

### D-02 Regime indicator at RTC+B go-live, 2025-12-05
ERCOT's Real-Time Co-optimization plus Batteries went live on **operating day 2025-12-05**.
Source: ERCOT's news release of that day, which announces "the successful implementation
of the first operating day" and refers to "the December 5 go-live"
(https://www.ercot.com/news/release/12052025-ercot-goes-live). In the RSM it enters as a **block**: an additive intercept shift that is kept out
of the canonical analysis. This is the textbook RSM device for a nuisance factor.

### D-03 Four zones: Houston, North, South, West
These are the four competitive load zones (`LZ_HOUSTON`, `LZ_NORTH`, `LZ_SOUTH`,
`LZ_WEST`), and their names match ERCOT's four load *forecast* zones. That match is what
makes a zonal join of price to load possible at all. Left out:
* the NOIE zones (`LZ_AEN`, `LZ_CPS`, `LZ_LCRA`, `LZ_RAYBN`), which are municipal and
  co-op load with their own settlement points;
* the hubs (`HB_*`), which are trading points. Hubs are the natural choice if the
  question were about virtual-bidding profitability.

**Limitation.** The forecast-zone load series include NOIE load in their geography (for
example, San Antonio's CPS load in "South"), so the load measure covers a larger population
than the `LZ_SOUTH` price does. D-22 confirms this empirically, and the report states it.

### D-04 Prices: settlement point prices (SPP), not LMPs
Load pays SPPs. For load zones, an SPP is the load-weighted average of the zone's
nodal LMPs plus real-time adders. Using SPPs keeps DA and RT on the same settlement
basis.

### D-05 RT hourly = mean of the four 15-min SPPs
This puts RT on the DA market's hourly grid, so the spread is defined. The **intra-hour
standard deviation** of the four intervals is kept as a separate volatility measure.
The hourly DA market cannot see this variation at all. Hours with fewer than four RT
intervals are flagged (`rt_complete`), not dropped silently.

---

## Weather

### D-06 Zonal temperature = weighted average of airport stations
The weights are rough metro-population shares, and West is tilted toward the Permian
(Midland/Odessa), where oil-and-gas load sits. They are judgment calls, written into the
config so they can be challenged. Better weights would use ERCOT's own weather-zone
station weights, or each county's share of zone load.

Missing data: a station's gaps of up to 3 h are linearly interpolated, and longer gaps
stay missing. The zone value is renormalised over the stations that are reporting. It is
set to missing if those stations carry less than 50% of the zone's weight.

### D-07 Source: IEM ASOS observations, not Open-Meteo reanalysis
* **IEM ASOS** gives measured temperatures at the airports, with no API key. The METAR
  report at :53 is assigned to the next hour.
* **Open-Meteo (ERA5)** is modelled and spatially complete, with no gaps.

ASOS is used because it is measured rather than modelled; Open-Meteo stays in the code as an
alternative source (`[weather] source`).

The project title says *mesoscale*. A gridded field averaged over each zone is the more
literally mesoscale choice, and the report names it as a next step.

---

## Gas and implied heat rate

### D-08 Gas price: Henry Hub daily spot (EIA RNGWHHD)
The intra-Texas hubs would match ERCOT generators better: Houston Ship Channel for
Houston, North and South, and Waha for West. Daily history for those hubs is not freely
available.

**Waha also traded at negative prices on many days** (pipeline constraints in the
Permian), and that makes an implied heat rate meaningless on those days. Henry Hub is
liquid, free, and close to Ship Channel most days. Its weakness: West IMHRs computed
against Henry Hub overstate the gas cost West generators actually faced; the report notes
this.

**Where it comes from (2026-10-01):** EIA's own history workbook for the series
(`hist_xls/RNGWHHDd.xls`), which needs no key. EIA credits the spot prices to Refinitiv
(LSEG). On 2026-10-01 it matched FRED's mirror (`DHHNGSP`) exactly on all 454 trade dates
from 2024-12-01 on. The API v2 route (`source = "eia_api"`) is still in the code for
anyone with a key. Both routes return the same series.

### D-09 Gas–power timing: the gas price for operating day D is the last trade before D
DA power for day D clears on D−1. Gas for flow on D trades on D−1, and the weekend
package trades on Friday. Matching on "the last trade date strictly before D" follows
how both markets are actually traded. The `same_day` option is in the code as a
sensitivity.

EIA does not say whether RNGWHHD is dated by trade date or by flow date. The data says
trade date: the series has no weekend dates and skips exchange holidays. In the July sample,
2025-07-04 is missing, and Monday 07-07 is priced off Thursday 07-03. A flow-dated series
would carry prices for every calendar day.

### D-10 IMHR is undefined when gas < $0.50/MMBtu
IMHR = power price ($/MWh) ÷ gas price ($/MMBtu), in MMBtu/MWh. Near zero the ratio
explodes, and below zero it flips sign. Negative *power* prices are kept, so they produce
negative IMHRs. They are real outcomes, and frequent in West.

---

## Response surface model

### D-11 Factors: temperature, ACTUAL zonal load and Henry Hub gas
**Decision.** The factors are T (°F), L (MW, actual zonal load) and G ($/MMBtu), in a full quadratic.
The assignment is about temperature, demand and price, so actual load stays in the model even
though it is collinear with temperature. That collinearity is *handled and reported*, not
avoided:
- coding to ±1 at the 1st/99th percentiles;
- VIFs and the condition number (`design_diagnostics.csv`);
- surfaces read as slices inside the data's convex hull;
- no interpretation of individual coefficients.

The load-anomaly specification (`ALT`) is fitted as a comparison only. It does not change the
conclusions (`sensitivity.csv`).

*Original discussion, kept for the record:*
In ERCOT, temperature and load are **strongly correlated**: hot hours are high-load hours.
The factor "design" is therefore far from the orthogonal one RSM assumes.
`analysis.correlation_check` reports the correlation and VIFs per zone, and the contour
plots mask everything outside the data's convex hull, so there is no extrapolation into
corners with no data. Alternatives:

* **temperature + load anomaly** (load minus the median for that hour-of-week). This is
  much less collinear, and asks "given the temperature, does unusually high load move
  the price?"
* **temperature + system net load** (load − wind − solar). Net load is the best single
  explanatory variable for ERCOT prices, but it is system-level, not zonal.
* **three factors** (temperature, load, gas). This needs a 3-D canonical analysis and
  loses the contour plot.

### D-12 Responses: DA IMHR and the DA−RT spread
IMHR takes the gas price level out of power prices, so what is left is scarcity and
heat-rate structure. The spread is the "price volatility" leg: it is the RT surprise
relative to DA expectations. Candidates for robustness checks: |spread|, the RT IMHR, and
the intra-hour RT standard deviation.

### D-13 Coding: 1st/99th percentiles → ±1 (not min/max)
**NCSS does not code the factors itself.** Its example data "have been recoded so that
they are -1, 0, and 1" by hand (330-4). What NCSS does say is that hierarchical models
allow "the ability to change the scale (coding) of a variable" (330-1). That is why coding
is harmless here: it changes the coordinates, not the fitted function. A test confirms
that predictions and the stationary point are identical under min/max and quantile coding.

Coding still matters in two places: where "inside the design region" falls, and how
readable the coded coefficients are. Min/max coding would let a handful of extreme hours
set the scale, which is why the 1st/99th percentiles are mapped to ±1 instead.

### D-14 The primary response is NOT winsorised
**Decision.** RiskEcon is about extreme risk, so the rare, enormous spreads are the observations
of most interest. The primary spread response is raw.
- **Sensitivities:** winsorised (0.5/99.5%) and asinh(spread/10) fits are kept, to show how much
  each conclusion depends on the tails.
- **Extreme-risk view:** quantile surfaces (τ = 0.05, 0.95) and the |spread| response look at
  the tails directly.
- **Influence:** the 20 most influential hours are listed (`influence_*.csv`).

*Original default, superseded:*
ERCOT RT prices hit the offer cap. A handful of scarcity hours would otherwise dominate an
OLS quadratic. Alternatives:
* an asinh transform;
* a robust (Huber) fit;
* modelling the median with quantile regression.

Report the winsorised fit and say how much the canonical analysis moves without
winsorising.

### D-15 Inference: Newey–West HAC standard errors (48 lags, D-26)
NCSS takes its assumptions from multiple regression (330-3), which include independent
errors, and it prints every test twice. One version divides by the total-error mean
square, the other by the pure-error mean square (330-6 to 330-10). Both are reproduced
here. Neither is valid for hourly data: residuals stay correlated for a day or more (see
`results/<window>/figures/residuals_*.png`), so classical standard errors badly overstate precision.

* **Coefficient SEs and the term-group Wald tests use HAC.**
* **The NCSS tables are still printed**, labelled descriptive, so the report can show the
  standard RSM output alongside the corrected inference.
* **Model comparisons** (first against second order, with and without hour of day) rest on
  out-of-sample validation (D-25), not on in-sample F tests.

### D-16 Lack of fit uses near-replicate cells, posed as a nested test
NCSS "automatically scans the data for duplicates" (330-1) and prints lack-of-fit and
pure-error lines "only ... if you have repeated observations" (330-8). Hourly temperature
and load essentially never repeat exactly, so NCSS would print no lack-of-fit test on this
data at all. The code supports both:
* `bins=None` is NCSS's exact-duplicate mode, and it reproduces the Odor example's
  F = 40.75 (p = 0.024047).
* `bins=8` treats observations that share a cell of an 8×8 grid over the coded factors
  as replicates.

The textbook split, SSE = pure error + lack of fit, **breaks** on binned data. The fitted
surface still varies inside each cell, so the "pure error" can exceed the model's SSE.
On a test surface the classical lack-of-fit sum of squares came out at −95,000. The code
therefore F-tests whether adding one indicator per cell improves the quadratic. That
comparison is never negative, and it reduces to the classical test when the cells are
exact design points (both properties are tested).

With n ≈ 8,760 the test has enormous power, so expect it to reject on real data. Read it
as a description, and lean on the CV comparison (D-15).

### D-18 NCSS's hierarchical elimination is reported, not used to choose the model
NCSS fits only hierarchical models (330-1). Its model selection is a backward
elimination: repeatedly remove the removable term that lowers R² least, and stop once
every candidate would cost at least the cutoff, set here to 0.01 (330-2). The cutoff is
an *amount of R²*, not a p-value, so unlike significance tests it still works at
n = 8,760.

It runs for every zone and response, and the result goes to
`results/hierarchical_selection.csv` and the summaries. The canonical analysis is still
done on the full second-order model, for two reasons:
1. The cutoff is absolute. When the total R² is modest, as it will be for spreads, 0.01
   can remove curvature that is real, and that curvature is exactly what RSM is there to
   study.
2. Dropping a term is the same as forcing its coefficient to zero, which moves the
   stationary point.

The comparison belongs in the robustness section: which terms the procedure would drop,
and whether that changes the conclusions.

One caveat: the Odor example prints "Number of Terms Removed: 0" alongside a 0.01 cutoff,
so selection was evidently off in that run. That means the implementation follows the
five steps as written but has no worked NCSS output to check it against.

---

## Repository and data

### D-17 Data are not committed; they are rebuilt from the public sources
`data/raw/` and `data/processed/` are not committed. `make fetch build` rebuilds them from the
sources in `DATA.md`, and every cache is keyed by window, so a re-run downloads nothing it
already has. The results under `results/` (CSVs and figures) are committed, so every number in
the report can be traced without rebuilding the data.

---

## Data acquisition (2026-10-09)

The full 2025 pull, its validation and the zone mapping are documented in
[`DATA.md`](DATA.md). The audit is in `data/processed/2025-01-01_2026-01-01/audit/audit.md`.

### D-19 Spread sign: DA − RT
This convention matches the assignment's wording ("DA–RT spread"). A positive spread means
DA settled above RT, i.e. a DA premium, which is what a buyer who hedges in DA pays relative
to RT.

gridstatus's own `ercot_spp_dart_15_min` uses the same sign. On 2025-07-07..14, across all
four zones and 2,688 intervals, its `spp` equals our DA(hourly) − RT(15-min) exactly. The
largest disagreement with RT − DA is $3,429.

### D-20 All four zones in one filtered request per month, not Houston first
The original plan was to pull Houston first. With a server-side `location in (...)` filter, each
monthly request carries all four zones at the same request cost. A Houston-first sequence
would have spent about 26 more requests to learn nothing new: the filter had already been
verified on two sample weeks.

Risk was controlled in other ways instead:
- a budget guard before each dataset;
- a check after each dataset's first chunk that the rows charged equal the rows returned (they did, for all four);
- per-chunk completeness validation against the expected interval count.

### D-21 Missing data is never imputed in prices, load or gas; the panel keeps every zone-hour
- **The grid:** the panel is reindexed onto the full zone × hour grid, so a missing hour
  appears as a row with missing values instead of disappearing.
- **The estimation sample:** `model_ok` marks which rows enter estimation.
- **2025 in practice:** nothing needed excluding. `model_ok` is true for 35,040 of 35,040
  zone-hours.
- **The one imputation:** station temperature gaps of ≤3 h are interpolated (D-06).

### D-22 Load = forecast-zone actual load (gridstatus), validated against ERCOT native load
The forecast-zone total tracks ERCOT's official native load closely: r = 0.997, MAPE 1.3%.
It runs about 700 MW higher.

That gap is concentrated overnight and at 08–10 local. It grows through autumn and nearly
vanishes after the RTC+B go-live. This is consistent with battery charging being counted
in telemetered load, but that explanation is unverified.

The NOIE question in D-03 is settled empirically: the four forecast zones sum to the system
total, so NOIE load is inside them, while the `LZ_*` prices exclude NOIE settlement points.

### D-23 External cross-checks for every source
| Source checked | Checked against | Result |
|---|---|---|
| EIA gas | FRED DHHNGSP | identical on all 269 dates |
| IEM temperatures | NOAA NCEI ISD (IAH Jul, DFW Jan) | max 0.08 °F apart |
| Spread sign | gridstatus DART | exact |
| Load total | ERCOT native load | see D-22 |

These results are reproducible with `make crosscheck audit`. They are the evidence that the
joins are right, and Section 2 of the report summarises them.

---

## Specification and inference (2026-10-10)

The evidence for each decision is in `EDA_FINDINGS.md`; the results are in `MODEL_RESULTS.md`.

### D-24 "Volatility" always means a named measure
All measures are levels in $/MWh, not log returns, because prices go negative.

| Measure | Definition |
|---|---|
| Spread magnitude | \|DA − RT\| in one hour |
| Conditional spread dispersion | Std **and** quantiles of the spread within a group. The std is flagged as tail-dominated. |
| Intra-hour RT volatility | Std of the four 15-min RT SPPs in an hour |
| Tail frequency | Share of hours beyond the zone's 99th-percentile \|spread\| (relative), or beyond $100 (absolute) |

### D-25 Validation: forward CV is primary; leave-one-month-out is secondary; neither is a backtest
- **Forward CV** is an expanding window: train on months < m, test on month m, for April–December. `rsm.time_folds` asserts that no fold trains on the test month or later.
- **Leave-one-month-out** trains on both sides of the test month. It is interpolation, and it is reported second.
- Both use **realised** T, L and G, so both are *explanatory*. A trading backtest would need DA-time forecasts, a bidding rule and costs; that is listed as an extension.
- The benchmark for skill is the hour-of-day mean (M0), not the overall mean.

### D-26 Dependence: 8,760 rows ≠ 8,760 experiments
- **Effective sample size** (Bartlett, 168 lags) per zone: spread ≈ 1,700–3,400; temperature ≈ 70–79; load ≈ 102–124; gas 73, with 127 distinct values.
- **Standard errors:** HAC with 48 lags. Clustering by operating day is a check: its SEs are similar to or larger than HAC, and both are 2–9× the classical SEs.
- **Evidence:** joint Wald tests on term groups and out-of-sample skill, never single-coefficient p-values. The NCSS F-tables are printed as descriptive only.

### D-27 RTC+B is a nuisance intercept, not a finding
There are 27 post-go-live days (648 hours per zone), all in late December, which confounds the market change with season. Dropping them is a sensitivity check, and no conclusion about RTC+B is drawn.

---

## Expectations: forecast surprises (pre-specified 2026-10-10, before any surprise model was fitted)

### D-28 The DA information set: forecasts as of 09:45 on D−1; errors are actual − forecast
**The cutoff.**
- DAM bids and offers close at 10:00 on D−1 (Protocols §4.5.1).
- System problems can only *extend* the window (NPRR478), so a 09:45 cutoff precedes bid close on every day.
- Grid Status `publish_time="latest_before:-1 day@09:45:00"` (`timezone="market"`) selects, per hour, the newest report published by then:
  - the **09:30** load forecast (NP3-560-CD, posted at :30);
  - the **08:55** wind and solar forecasts (NP4-742/745-CD, posted at about :55).
- A 10:00 cutoff would admit the 09:55 wind and solar report, five minutes before close. That is not conservative, so it is not used.

**Sensitivity: a 09:00 cutoff.** This changes only the load vintage (to 08:30), in case the 09:30 stamp is a nominal rather than a posting time.

**Audit (2025).**
- All 26,280 selected rows were published before their cutoff.
- Horizons run 14.5–38.5 h, and no row is stale (none was published more than 2 h before its cutoff).
- The per-chunk detail is in `manifest.json`.

**Load pairing, by definition.** The forecast-zone forecast is paired with forecast-zone actuals (NP6-346), the same ERCOT zone family. The realised system error has a mean of +9.5 MW, i.e. it is unbiased. A constant bias would not change a hierarchical polynomial fit in any case.

**Renewables.** Errors are measured against **HSL (available capability)**, not generation. Generation-based errors are biased (wind −793 MW, solar −547 MW) because economic curtailment at negative prices pushes output below forecast. That bias is caused by price, so it would contaminate the regressor. The HSL-based errors have mean +197 and −44 MW. Generation-based errors are kept as a sensitivity.

**Primary surprise:** `net_load_err_avail` = system load error − wind HSL error − solar HSL error.

**Endogeneity that remains.** Large flexible loads curtail when RT prices are high, so the load error is partly a *response* to RT prices. Coefficients are therefore associations with surprises, not causal effects of exogenous shocks.

### D-29 Pre-specified surprise ladder and evidence rule
| Rung | Specification |
|---|---|
| A | M2: T, L, G full quadratic |
| B | A + realised system net-load ramp R: {R, R², R·L} |
| C | B + zonal load error E_L: {E_L, E_L², E_L·L} |
| D | C + wind and solar availability errors: {E_W, E_S, E_W², E_S²} |
| S | 2-factor full quadratic in (net_load_err_avail, net_load_fc), with canonical analysis |

All rungs share the regime nuisance block and use HAC with 48 lags.

**Evidence rule.** An added group counts only if two things hold:
1. its HAC Wald test is significant after Holm adjustment across the 12 pre-specified group tests per response (3 groups × 4 zones);
2. **forward CV improves on the previous rung.**

In-sample ΔR² alone never counts. No terms are added after results are seen. Event-day exclusion is rerun for C, D and S.

### D-30 The ex-ante model E uses only information known by 09:45 on D−1
**Factors:**
- the net-load forecast;
- the zonal load forecast;
- gas from the last trade date strictly **before D−1**, because the D−1 next-day index is still trading at 09:45;
- an hour-of-day block and the regime block (calendar).

**Excluded:**
- realised temperature, load and renewables;
- full-sample statistics: `load_norm`, `load_anom`, and full-sample coding or thresholds as inputs.

**Fitting and evaluation.**
- Coding and the M0 benchmark are fitted on training months only.
- Headline numbers come from forward CV only.
- This is an ex-ante evaluation of the information set, not a trading backtest: there is no bidding rule, transaction costs or collateral.

### D-31 Prices come from ERCOT's settled annual archives; gridstatus is the cross-check
**Sources.**
- **ERCOT archives:** NP4-180-ER (DAM) and NP6-785-ER (RTM) give *final* settlement-point prices for every load zone and hub. They are public, need no quota, and include 2024.
- **gridstatus:** holds prices *as first published*.

**Comparison for 2025.**
- **Keys:** they match on every one of the 35,040 DA and 140,160 RT keys.
- **Values:** they differ on 89 DA hours and 135 RT intervals, all on days ERCOT corrected prices afterwards. These include:
  - the Board-approved DAM price correction for 2025-06-27;
  - an RT interval on 2025-06-30 at 15:30 that gridstatus shows at $1,624–$1,636 in every zone, where ERCOT's final price is $28–$40;
  - smaller differences on the remaining dates. Per the audit (`price_source_crosscheck.csv`), the DA differences fall on two dates (2025-06-27, 2025-12-06) and the RT differences on eight dates (03-28, 03-29, 06-30, 10-07, 10-08, 10-12, 11-20, 12-04): ten distinct dates in all.
- **RT point type:** the archive's `LZ` type matches gridstatus. The `LZEW` (energy-weighted) type does not.

**Decision.** The spread is a settlement quantity, and a $1,596 phantom spike would have entered the extreme-spread analysis. The panel therefore uses the settled archive (`[prices] source = "ercot_archive"`).

**Audit.** The `prices: ERCOT settled archive vs gridstatus` audit check lists every corrected day (`price_source_crosscheck.csv`).

**Impact on 2025.** Re-running everything on settled prices moved no conclusion. Individual figures changed slightly, and `EDA_FINDINGS.md` and `MODEL_RESULTS.md` were updated to match the CSVs.

### D-32 What the West basis decomposition measures
The decomposition is spread_LZ = spread_HUB + (basis_DA − basis_RT), with basis = LZ SPP − `HB_HUBAVG` SPP.

**What the second term is.** It is the DA-to-RT change in the zone's price *relative to the hub average*: a relative congestion surprise.

**What it is not.**
- It is not total congestion.
- It is not constraint-level attribution.
- `HB_HUBAVG` includes `HB_WEST` (one quarter of its weight), which dampens West. An ex-West reference (the mean of Houston, North and South hubs) is a sensitivity.

### D-33 Response surfaces are drawn on a density-based support, not the convex hull alone
**Problem.** Observed (temperature, load) pairs form a curved V. The convex hull of a V includes the empty inside of the V, so a hull-masked slice draws extrapolated values as if they were supported. In Houston at median gas, the hull covers 50–80 °F at 16–20 GW, where no hours lie.

**Choice (report Figure 5).** A grid point is drawn solid only if it is inside the hull **and** at least 5 hours with near-median gas (within 0.25 coded units) lie within 0.1 coded units of it (`rsm.slice_grid(support="density")`, `rsm.near_data`). Everything else is drawn faded, as extrapolation.

**Alternatives.**
- Hull only: the earlier default, and still the default for `slice_grid`.
- Alpha shapes.
- A kernel-density contour.
- Fitting on a normalised (load / predicted load) factor instead.

**Encoding (revised 2026-10-10).**
- Colour appears only on the supported region.
- Extrapolation is drawn in grey.
- Fitted |spread| < 0 is hatched wherever it occurs, because it is not physical.

In the supported region, West's surface is negative over about 9% of the slice, and 6.3% of West's fitted hours are below zero; South's are 2.8%. An earlier version of the figure clipped these values to the lightest colour.

**What it affects.** Presentation only: no fit, test or table changes. The radius (0.1 ≈ 5% of each factor's 1st–99th-percentile range) and the minimum count (5) are judgement calls. Checked 2026-10-10: radius 0.2, or a minimum count of 10, changes the outline but not the message. Houston's magnitude rises along the hot, high-load arm; West's is highest at low load.

### D-34 Linear + hour-of-day comparator (M1H) for the report's "what did curvature add?" question
**Why.** M3 (the quadratic RSM plus an hour block) was compared only with M0 (the hour-of-day mean) and with M2 (no hour block). Its DA-IMHR skill could therefore come from the hour profile and linear load rather than from the six second-order terms. M1H is M1 plus the same hour block. M1H vs M3 then differs *only* by the second-order terms, as M1 vs M2 does.

**What it is.** A comparator, not a new candidate model. No factor, response or sample changed. `analysis.curvature_contribution` writes `results/<window>/report/curvature_contribution.csv` with:
- in-sample R² for M1, M2, M1H and M3;
- HAC Wald p-values for the squares, the interactions and all six, with and without hour;
- forward-CV skill;
- M1 and M2 predictions at the reference conditions.

**What it changed in the report.** In Houston, North and South, M1H forecasts the DA IMHR better than M3: +0.12 to +0.17, against +0.09 to +0.13. The earlier wording, which credited the 9–13% gain to the response surface, was wrong for those zones. Only in West does curvature add out-of-sample skill (−0.04 → +0.09).

**Alternatives.**
- Report M3 vs M0 only, as an earlier draft did. This overstates what curvature adds.
- Add M1H to the main ladder in `analysis.run`. That is a larger change and was not made.

### D-35 Extreme hours vs system tightness: definitions, robustness and the event day
**Tightness** is the decile of ERCOT system net load (load − wind − solar), ranked over all hours. This is the q3-table convention.

**Extreme** is the zone-relative top 1% of |spread| (`tail_flags`). Every zone's base rate is 1% by construction, so the ratios are relative risks. A **fixed-dollar sensitivity** ($50, $100 and $250; for |spread| and for RT above DA by more than X) reports the same comparison in dollars.

**Two net-load measures, never mixed (verified 2026-10-10).**
- **Tightness** (the deciles) = system load − wind and solar *generation* (`system_net_load_mw`, from the fuel mix).
- **Surprise** = system load − wind and solar *available capacity* (HSL), compared with the 09:45 D−1 forecast (`net_load_fc`, `net_load_err_avail`). This keeps price-driven curtailment out of the error (D-28).

The two series correlate at 0.985. The availability-based series averages 1.5 GW lower, and the 95th percentile of the absolute gap is 5.9 GW. The event figure plots each measure in its own panel, under its own label.

**Uncertainty.** 90% intervals come from a day-block bootstrap (2,000 resamples of operating days, seed 0), because extremes cluster on few days. When the rest-rate is zero, the ratio is undefined (NaN), never infinite.

**Time of day.** Deciles ranked *within* each hour of day are a time-of-day robustness check only. Season, weather, renewable output and outages are not held fixed, so the report does not call it a control or a causal effect.

**Event day.** Selected retrospectively, after candidate events had been examined, using a reproducible rule: the operating day with the most RT-above-DA extreme zone-hours in the top decile, excluding the window's storm days (`event_days`). For 2025 that is 2025-05-16; for 2024, 2024-05-08. `--event-day` overrides it.

**Alternatives.**
- Rank tightness by a reserve-margin measure. ERCOT's PRC/reserve data are not cached.
- Rank within month × hour. The cells are too thin: about 30 days per cell.
- A regression of the extreme indicator on net load with season and hour fixed effects. That is a model, not the clear descriptive result asked for.
- Show the 19–20 February storm as the event. It is the year's outlier, not a representative day.

### D-36 Matched comparisons for curvature and forecast surprises, and the event-day predictions (2026-10-10)
A retrospective diagnostic, not a newly pre-registered study. It asks where the second-order
terms help, whether forecast surprises explain the spread, and what the models said on the
event day.

**Comparisons fixed before the diagnostic was run.**
- Temperature-only linear versus quadratic load models, with the same expanding-month
  validation (`eda.temp_load_forward`). Realised temperature makes this explanatory.
- Linear versus quadratic price models on identical inputs, hours and controls
  (`analysis.curvature_contribution`, D-34).
- Pooled squared-error skill is the headline; absolute error, monthly results and refitting
  without the event day are diagnostics. No new headline model is selected on its best
  metric, nothing is clipped, and no further predictors are searched for.

**Event.** The D-35 day (16 May 2025), for Houston, to stay consistent with the report.
Actual and forecast net load are both on the availability basis; the generation-based series
is never subtracted from the forecast. The predictions shown for May come from models trained
through April (`surprise.event_forward_predictions`). The event is illustrative, not a
representative sample or a causal attribution of the price spike.

**Coverage correction.** The cached availability-surprise inputs for 2024 are complete only
from 24 September 2024. The fit-2024/predict-2025 test therefore trains on 2,364 hours, not a
full calendar year, and `out_of_year.csv` records each model's training count and dates.
Missing hours are not imputed, and no further downloads were made.

**Alternatives deferred.** New weather fields, outages, regional gas hubs, higher-order
polynomials and a wider model search would expand the scope before this basic comparison is
understood.
