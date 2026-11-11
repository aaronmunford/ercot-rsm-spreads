# Response-surface results: 2025, four ERCOT load zones

**How to regenerate:** `make fit` writes everything under `results/2025-01-01_2026-01-01/`. File names are given in brackets.

**Specification:** see `EDA_FINDINGS.md` (the last table) and DECISIONS D-11, D-14, D-24 to D-27.

- **Factors:** temperature (°F), actual zonal load (MW) and Henry Hub gas ($/MMBtu), coded to ±1 at their 1st/99th percentiles.
- **Sample:** 8,760 hours per zone.
- **Inference:** HAC with 48 lags unless stated otherwise.

## The model ladder

| Model | Specification |
|---|---|
| M0 | hour-of-day mean |
| M1 | linear in T, L, G |
| **M2** | **full quadratic RSM in (T, L, G)**, 10 terms |
| M3 | M2 + 23 hour-of-day dummies |
| ALT | M2 with load anomaly instead of load |

M1–M3 and ALT also carry a nuisance intercept for the 27 post-RTC+B days. Surfaces and reference predictions are averaged over the block mix.

## 1. The mean DA − RT spread is barely explained by temperature, load and gas

[model_comparison.csv, wald_term_groups.csv, sensitivity.csv]

| | Houston | North | South | West |
|---|---:|---:|---:|---:|
| In-sample R², M2 | 0.005 | 0.009 | 0.031 | 0.049 |
| Curvature (all 6 second-order terms), HAC Wald p | 0.36 | 0.077 | 0.23 | **0.0006** |
| Forward-CV skill vs M0, M1 | 0.002 | 0.007 | 0.010 | −0.011 |
| Forward-CV skill vs M0, M2 | −0.001 | 0.004 | −0.042 | −0.048 |

- **No predictive gain.** No model predicts held-out months better than the hour-of-day mean. The quadratic terms make out-of-sample error *worse* (M2 vs M1), so the curvature does not earn its place for the mean spread.
- **Interpretation:** the day-ahead price already embeds expected weather and load, so the average RT surprise is not systematically related to realised conditions. This is a negative result, reported as such.
- **One robust pattern: a DA premium in the hottest hours.** At the mean conditions of each zone's hottest 1% of hours, M2 predicts +$5.6 to +$9.9 in Houston, North and South. Across all seven variants (winsorised, asinh, without Feb 19–20, without post-RTC+B, M3, ALT) the prediction stays within +$3.3 to +$11.7. West is ≈ 0 (−$2.6 to +$2.4).
- **The cold end is not robust.** North's coldest-1% prediction moves from −$17 to +$8 when 2025-02-19/20 is dropped, and South's from −$49 to −$3. Feb 19–20 also accounts for 60–70% of the 20 most influential hours in North and South [influence_*_spread_M2.csv].

## 2. The magnitude and the tails of the spread *are* shaped by temperature and load

[model_comparison.csv, sensitivity.csv, quantile_surfaces_spread.csv]

Spread magnitude |spread| under M2:

| | Houston | North | South | West |
|---|---:|---:|---:|---:|
| In-sample R² | 0.035 | 0.062 | 0.088 | 0.114 |
| Curvature, HAC Wald p | 0.020 | 0.033 | 0.004 | <0.001 |
| Predicted \|spread\| at the mild 60–70 °F band ($/MWh) | 7.9 | 8.6 | 7.5 | 10.8 |
| … at the hottest 1% of hours | 20.6 | 26.0 | 22.7 | 34.6 |
| … at the coldest 1% of hours | 42.8 | 83.1 | 93.6 | 70.4 |
| … coldest 1%, excluding Feb 19–20 | 21.7 | 32.4 | 28.5 | 52.4 |

- **The surface is convex (U-shaped).** The curvature is significant in every zone even under HAC inference.
- **Hot end:** the predicted magnitude is about 2.6–3.2× its mild level in the primary fit, and $19.4–$34.6 across every variant.
- **Cold end:** larger still, but about half of it rests on a single storm.

The quantile surfaces (τ = 0.05 and 0.95, on the same M2 design; point estimates) show the same thing in the tails:

| Spread quantile ($/MWh) | Mild | Hottest 1% | Coldest 1% | Coldest 1%, excl. Feb 19–20 |
|---|---|---|---|---|
| 5th percentile | −15 to −29 | −35 to −49 | −66 to −195 | −25 to −94 |
| 95th percentile | +16 to +24 | +49 to +74 | +56 to +104 | +42 to +93 |

The median barely moves (+$1.5 to +$16). **Temperature and load shift the *scale* of DA−RT risk, not its centre.** This is the result most relevant to a study of extreme price risk.

**This structure does not translate into forecasting skill.** Out of sample, no model beats the hour-of-day mean for |spread|: forward-CV skill runs from −0.09 to +0.02. Extremes are few and episodic (57 days hold every top-1% hour), so an expanding window has seen too few comparable events. The magnitude surface is an explanatory description, not a forecast.

## 3. DA implied heat rate: load and time of day explain it; curvature adds out-of-sample skill only in West

[model_comparison.csv, surface_imhr_da.png]

| | Houston | North | South | West |
|---|---:|---:|---:|---:|
| In-sample R², M2 | 0.184 | 0.150 | 0.208 | 0.205 |
| In-sample R², M3 (+ hour) | 0.381 | 0.360 | 0.374 | 0.398 |
| Curvature, HAC Wald p (M2) | 0.29 | 0.13 | **0.0006** | **<0.001** |
| Forward-CV skill vs M0, M3 | **0.133** | **0.089** | **0.085** | **0.087** |
| Forward-CV skill vs M0, M2 | −0.091 | −0.187 | −0.149 | −0.226 |

- **Load drives the heat rate.** In Houston, North and South the DA heat rate rises mainly with load at fixed temperature (merit order). Without hour of day the weather/load surface is worse than the hourly profile; with it (M3) the model beats the hourly profile by 9–13% in held-out months.
- **Most of that gain is not curvature (D-34).** M1H has the same hour block as M3 but only linear terms [report/curvature_contribution.csv]. In Houston, North and South it forecasts better than M3 (skill +0.12 to +0.17, against +0.09 to +0.13), so adding the six second-order terms raises later-month error by 3.5–7.7%. Only in West does curvature add skill (M1H −0.04, M3 +0.09; error −12.0%).
- **West differs:** its heat rate is highest at *low* load.
- **Gas caveat.** Gas enters a ratio whose denominator *is* gas, so a gas effect here measures how far power prices scale with gas, not a causal effect.

## 4. Inference and design: what the 35,040 rows can and cannot support

[se_comparison_*_M2.csv, design_diagnostics.csv, EDA dependence_effective_n.csv]

- **Dependence.** HAC standard errors are a median of **1.9–6.5×** the classical OLS standard errors (up to 8.3×). Clustering by operating day gives 1.9–7.3× (up to 9.4×). Classical NCSS-style F tests are therefore printed as descriptive only.
- **Effective sample size.** The effective number of independent observations is about 1,700–3,400 per zone for the spread, but only about **70–120 for temperature and load**. Gas has 127 distinct daily values (73 effective). So the gas terms (G², T·G, L·G) are weakly identified and largely encode season.
- **Collinearity.** VIFs reach 51 (linear temperature, North), and the condition number of the coded design is 35–43. Individual coefficients are therefore not interpreted. Surfaces are read as slices inside the convex hull of observed (T, L) at median gas.
  - The coldest hours lie outside that slice, because cold snaps coincided with high gas prices.
  - The quadratic |spread| surface dips slightly below zero at one hull edge, a polynomial artefact for a non-negative response; the quantile surfaces do not share it.
- **Lack of fit.** The near-replicate lack-of-fit test (8 bins per factor) rejects in every case [lack_of_fit.csv]. With n = 8,760 that is expected, and it says the quadratic is an approximation, as RSM intends.

## Validation framing (D-25)

- **Forward CV** trains on months before the test month only, and tests April–December. It is honest in time.
- **Leave-one-month-out CV** trains on both sides of the test month. It measures interpolation and is reported as secondary [model_comparison.csv, scheme = lomo]. It is uniformly *harsher* on the quadratic models; for example, North spread M2 skill is −0.26.
- **Neither is a trading backtest.** Both use realised temperature, load and gas, which are unknown when DA bids are due. A backtest would need the DA-time weather and load forecasts, a bidding rule, and transaction and collateral costs. That is an extension.

## RTC+B

The 27 post-go-live days enter as a nuisance intercept only. Dropping them changes no conclusion above (sensitivity.csv, "excl. post-RTC+B" rows), and nothing is concluded about the market change itself.

---

# Expectations: forecast surprises (2025)

**How to regenerate:** `make surprise` writes `results/2025-01-01_2026-01-01/surprise/`.

**Specification.** The ladder was fixed in DECISIONS D-28 to D-30 before any of these models was fitted.

**Information set.**
- **Cutoff:** ERCOT forecasts as of **09:45 market time on D−1**. DAM bids close at 10:00 and can only be extended.
- **Vintages:** this selects the 09:30 load report and the 08:55 wind and solar reports.
- **Audit:** all 26,280 selected rows were published before their cutoff, and no row is stale. Forecast horizons are 14.5–38.5 h [manifest.json].

**Surprise definition.** A surprise is actual − forecast. Renewables are measured against **available capability (HSL)**, so curtailment caused by negative prices is not counted as a supply surprise.

## The raw association

Spearman correlations with the spread [`panel.parquet`]:

| Variable | Houston | North | South | West |
|---|---:|---:|---:|---:|
| Net-load surprise (availability) | **−0.47** | **−0.47** | **−0.44** | **−0.35** |
| Net-load surprise (generation-based) | −0.38 | −0.38 | −0.32 | −0.32 |
| Wind availability surprise | +0.33 | +0.36 | +0.36 | +0.32 |
| Realised temperature | +0.05 | +0.03 | +0.05 | −0.01 |
| Realised load | +0.07 | +0.03 | +0.00 | +0.09 |

- **Sign.** A tighter-than-expected system goes with RT above DA (a negative spread), as economics predicts.
- **Curtailment.** Removing curtailment from the renewable surprise *strengthens* the relationship, consistent with generation-based errors being partly a response to price.

## The ladder: what counts under the evidence rule (D-29)

The rule: the group's HAC Wald test is significant after Holm adjustment, **and** forward CV improves on the previous rung [ladder_group_tests.csv, ladder_forward.csv].

| Added group (spread response) | Holm p (H / N / S / W) | Forward-CV gain over previous rung (H / N / S / W) | Counts? |
|---|---|---|---|
| B: net-load ramp | 0.018 / 0.007 / 0.0004 / 0.079 | −0.001 / −0.003 / −0.023 / +0.005 | **No** (fails out of sample) |
| C: zonal load surprise | <0.001 / 0.025 / 0.067 / <0.001 | −0.065 / −0.029 / −0.079 / +0.050 | **West only** |
| D: wind and solar availability surprises | <0.001 in all four zones | +0.025 / +0.034 / +0.007 / +0.026 | **Yes, in all four zones** |

- **In-sample R² of the spread** rises from 0.005–0.049 (A) to 0.055–0.102 (D).
- **Why C fails out of sample in H/N/S.** The load-surprise group adds a quadratic and a load interaction, and those do not carry forward.
- **Collinearity.** The ladder's max VIF stays at about 34–52, driven by the realised T/L core.

## The surprise surface S

[S_canonical.csv, S_E_quantiles.csv, S_sensitivity.csv, figures/surface_S_spread.png]

**The model.** S is a 2-factor full quadratic in the net-load surprise (availability-based) and the DA-time net-load forecast, i.e. how tight the system was *expected* to be.
- Max VIF is 1.2, so it is essentially orthogonal.
- In-sample R² is 0.048–0.068. With 6 terms, that exceeds model A's 10 realised-condition terms in Houston, North and South.

**Out of sample.** S is the **only model with positive forward-CV skill against the hour-of-day benchmark**: Houston +0.054, North +0.038, West +0.045; South −0.039. It **beats A in all four zones** (+0.003 to +0.088). Its top-1% MAE is the lowest in Houston, and within $2 of the lowest (model D) in North and South. In West, D is lower.

**The shape.**
- **Canonical analysis:** a saddle in every zone.
- **The key term** is the interaction (surprise × expected tightness). It is negative in every zone (HAC p ≤ 0.037), and it stays significant under all three sensitivities: generation-based surprise, the 09:00 vintage, and excluding 2025-02-19/20 (p ≤ 0.04 everywhere, except South with the generation-based surprise at 0.069).
- **What the interaction means:** the same surprise moves the spread far more when the system was expected to be tight.

At a +4.6 GW (95th-percentile) tighter-than-forecast surprise:

| Expected net load | Mean spread (H/N/S/W) | 5th-percentile spread |
|---|---|---|
| 18 GW (10th pct) | −1.3 / −3.5 / −3.3 / −4.1 | −21 to −30 |
| 35 GW (median) | −13 / −17 / −20 / −16 | −42 to −54 |
| 50 GW (90th pct) | **−18 / −25 / −32 / −21** | **−73 to −89** |

A −4.9 GW (5th-percentile, looser-than-forecast) surprise does the reverse: in tight hours the mean DA premium is +$16 to +$21, against +$0 to +$5 in loose hours.

**For |spread|,** S is a convex bowl (a minimum, curvature p ≤ 0.003): magnitude grows with the size of the surprise in either direction. The interaction is not significant for |spread|.

## The ex-ante model E (DA-time information only)

[ladder_forward.csv, S_E_quantiles.csv]

**Inputs:** the net-load forecast, the zonal load forecast, gas lagged two trade days, and an hour block.

**Signed spread:** **no point-forecast skill.** Forward-CV skill against M0 is −0.12 to −0.02. As expected, the *direction* of the RT surprise is not knowable at bid close.

**|spread|:**
- Forward-CV skill against M0 is mixed: Houston −0.018, North −0.094, South −0.321, West +0.014.
- The South result is unstable, so this is not a forecasting result.
- E does have the **lowest MAE on extreme hours** in Houston, North and South ($150 / $156 / $142, against $160–$178 for M0).

**Conditional quantiles at bid close** are in-sample point estimates, not validated out of sample. The spread's 5th–95th-percentile band widens with forecast tightness:
- **lowest 10% of forecast net load:** about −$13 to −$23 / +$20 to +$28;
- **highest 10%:** about −$28 to −$55 / +$34 to +$55.

**Interpretation.** What DA-time information carries is about the *scale* of spread risk, not its sign. That matches the structure of the realised-condition results.

## What this changes

The realised T, L and G surface could not explain the signed spread. The reason is that the spread prices the *deviation from expectations*, and the realised conditions were largely expected.
- **Supply surprises** count on both conditions in all zones, and so does the **interaction of the net-load surprise with expected tightness**.
- **Load surprises** count only in West.
- **The convexity of the supply stack** appears where RSM is designed to find it: in the curvature and cross-product of the surface.

**Caveats.**
- Flexible loads curtail at high RT prices, so the load surprise is partly endogenous (D-28).
- All results are associations within one year. The 2024 replication follows.

---

# Replication: 2024 vs 2025

**How to regenerate:** `make crossyear` reads `results/2024-01-01_2025-01-01/` and `results/2025-01-01_2026-01-01/`, and writes `results/cross_year/{replication.csv, out_of_year.csv}`.

**The 2024 run.** 2024 was run through the identical pipeline:
- 8,784 hours, with the 23-h day on 2024-03-10 and the 25-h day on 2024-11-03;
- settled ERCOT prices;
- the same 09:45 forecast vintages, with 0 late rows;
- every audit check passing, FRED gas identical on all 271 dates, and ERCOT native load correlation 0.9996.

**Two data findings along the way.**
- **Sensor fault.** A −63 °F reading at Brownsville on 2024-04-03 was a sensor fault. A physical-plausibility filter (−20 to 125 °F) now drops such readings before averaging. 2025 had none.
- **Load gap.** The gridstatus-vs-native-load gap is +343 MW in 2024, against +704 MW in 2025. That fits the battery-charging explanation (D-22), but does not prove it.

## What replicates (same direction in every zone, both years) [replication.csv]

The 2024 rows for the surprise models (group D, S and its interaction) use only 24 September to 31 December 2024 (2,364 hours), because ERCOT's available-capacity forecasts are incomplete before then. They are a partial check, not a full-year replication; the report therefore treats the surprise results as unreplicated.

| Finding | 2024 | 2025 |
|---|---|---|
| DA settles above RT in most hours | 55–61% of hours | 59–63% |
| Negative skew (RT spikes) | −4.6 to −10.4 | −17.7 to −25.1 |
| Top 1% of hours' share of Σ\|spread\| | 24–34% | 17–21% |
| Load V-shape: R² of T + T² (H/N/S; West) | 0.84–0.89; 0.37 | 0.85–0.89; 0.32 |
| Load minimum at | 52–60 °F | 52–61 °F |
| Tail frequency at 55–65 °F vs ≤32 °F | 0–0.4% vs 8–28% | 0% vs 7–22% |
| Share of extremes in ramp hours | 78% | 75% |
| \|spread\| curvature (HAC p) | ≤ 0.006 | ≤ 0.033 |
| Spread 5–95% band, hottest 1% vs mild hours ($/MWh) | 88–215 vs 25–68 | 85–112 vs 31–51 |
| Renewable-availability surprise group D (Holm p) | ≤ 0.032 | < 0.001 |
| Surprise surface S, forward-CV skill vs M0 | **+0.085 to +0.163 (all zones)** | +0.038 to +0.054 (3 of 4) |
| S interaction (surprise × expected tightness), sign | negative in all zones | negative in all zones |

## What does not replicate

- **The S interaction's strength.** It is negative in every zone in both years, but only −3 to −9 in 2024 (HAC p 0.23–0.62) against −23 to −31 in 2025 (p ≤ 0.037). The *amplification* of surprises by expected tightness is a 2025-strength effect, not an established regularity.
- **The hot-1% DA premium.** It is positive in Houston and North in both years, but flips in South (2024: −$8.1) and West (2025: −$0.6). It is not robust.
- **M3's DA-IMHR forward skill:** +0.085 to +0.133 in 2025, but −0.19 to +0.01 in 2024.
- **The load-surprise group C:** it is not significant in Houston and North in 2024 (Holm p = 1.0).
- **Forward-CV gains of the full ladder D:** negative in all zones in 2024.

## Fit on 2024, predict 2025 [out_of_year.csv]

**Coverage.** This is not a full-calendar-2024 fit. The surprise models need ERCOT's forecasts of
available wind and solar capacity, which are complete only from 24 September 2024, and every
model is fitted on the same complete-case rows. Training therefore covers **2,364 hours** (24
September to 31 December 2024), and the benchmark is the hour-of-day mean of those hours. Every
training hour precedes the test year, but the training sample covers only autumn and early
winter. `out_of_year.csv` records each model's training count and dates.

| Model | Spread skill (H / N / S / W) | \|spread\| skill | 2025 hours inside the 2024 factor hull |
|---|---|---|---|
| M2 (realised T, L, G) | −0.72 / −1.10 / −0.86 / −9.83 | −1.07 / −1.16 / −0.62 / −4.95 | 24–33% |
| M3 (+ hour block) | −0.50 / −0.52 / −0.71 / −5.93 | −0.50 / −0.30 / −0.25 / −2.05 | 24–33% |
| D (full surprise ladder) | −0.79 / −1.37 / −0.70 / −4.12 | −1.22 / −1.52 / −0.59 / −1.21 | n/a (7 factors) |
| E (ex-ante) | −0.33 / −0.32 / −0.32 / −0.74 | −0.65 / −1.10 / −0.70 / −0.91 | 24–34% |
| **S (surprise surface)** | **+0.076 / +0.060 / +0.047 / +0.058** | **+0.040 / +0.040 / +0.012 / +0.006** | **94%** |

**Why the realised-condition surfaces fail.**
- 2025's combinations of temperature, load and gas mostly lie *outside* what 2024 observed, so a quadratic fitted on one year extrapolates badly into the next.
- This is the practical cost of the collinear, seasonal factor design flagged in D-26. It is also exactly why the report reads surfaces only inside the data hull.

**Why S succeeds.** S is the only specification that transfers across years, although its training sample covers only autumn and early winter 2024. Its two factors (the net-load surprise and the expected net load) span almost the same region in both years, and its relationship has the same sign.

**What still holds.** None of this is a trading result: every factor except the forecast is realised ex post (D-25).

---

# West: the hub-relative basis decomposition (D-32)

**How to regenerate:** `make basis` writes `results/<window>/basis/`.

**The decomposition.** Per zone and hour:

spread_LZ = spread_HUB + (basis_DA − basis_RT)

- **The hub term** is the DA − RT spread at `HB_HUBAVG`, the system-wide part of the RT surprise.
- **The basis surprise** is the DA-to-RT change in the zone's price *relative to the hub average*.

**What the basis surprise is not.**
- It is not total congestion, and not a constraint-level attribution.
- `HB_HUBAVG` includes `HB_WEST` at one-quarter weight. The ex-West reference (the mean of the Houston, North and South hubs) is reported as a sensitivity and makes West's basis slightly larger.

**Data.** Hub prices come from the settled ERCOT archive, with RT hours built only from complete sets of four intervals. The identity holds to machine precision and is asserted in code.

## Results [variance_shares_hubavg.csv, west_by_*.csv]

| HUBAVG reference | Houston | North | South | West |
|---|---|---|---|---|
| Basis share of spread variance, Cov(basis, spread)/Var(spread), 2025 / 2024 | 0.06 / 0.00 | 0.27 / 0.03 | **0.46 / 0.22** | 0.30 / 0.11 |
| Median \|basis surprise\| ($/MWh), 2025 / 2024 | 1.2 / 1.0 | 1.5 / 1.2 | 2.1 / 1.7 | **6.0 / 4.9** |
| Hours with \|basis surprise\| > $20, 2025 / 2024 | 1.5% / 1.1% | 1.4% / 1.6% | 4.7% / 4.9% | **10.9% / 12.8%** |
| Top-1% spread hours where the basis term is the larger term, 2025 / 2024 | 13% / 1% | 9% / 2% | 26% / 27% | 39% / 19% |

**What explains West.** West's *typical* hour is different:
- its basis surprise is 3–5× the other zones' at the median;
- it exceeds $20 in about 11–13% of hours.

**Variance tells a different story.** The largest basis share of variance belongs to South, which is driven by a few large episodes. West's extremes remain mostly system-wide (the hub term), in line with the EDA finding that West's ordinary hours are idiosyncratic but its extremes are shared.

**Where West's basis surprises concentrate** (both years):
- **The lowest West-load decile:**

  | | 2025 | 2024 |
  |---|---|---|
  | Median \|basis surprise\| | $11.9 | $7.8 |
  | 95th percentile | $55 | $73 |
  | Mean basis surprise | −$8.0 | −$7.2 |
  | Mean spread | −$18.7 | −$15.4 |

  In the highest-load decile, the median is $1.7–1.8 and the 95th percentile $10–14. **About half of West's low-load spread anomaly is the local basis term.** This accounts for the EDA finding that West's tail frequency peaks at low load.
- **High DA-time West/Panhandle wind forecasts:** the median |basis surprise| rises from $2 to $7–9 across forecast deciles.
- **Curtailment** behaves the same way. The proxy (COP HSL − generation, West and Panhandle) runs from about 170 MW in the lowest wind-forecast decile to about 1,900 MW in the highest, and from about 310 MW in the highest load decile to about 900 MW in the lowest.

**The wind-availability surprise itself** moves the West *spread* monotonically (2025: −$7.5 to +$9.3 across deciles; 2024: −$13.2 to +$13.1), mostly through the hub term. It barely moves the *magnitude* of the basis surprise.

**Interpretation.** These are descriptive associations. They point to export-constrained, high-wind, low-local-load conditions as the source of West's distinct spread behaviour. Testing that would require constraint-level data (shadow prices), listed as an extension.
