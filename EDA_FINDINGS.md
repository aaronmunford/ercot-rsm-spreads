# EDA findings: 2025, four ERCOT load zones

**How to regenerate:** `make eda` writes 19 tables to `results/2025-01-01_2026-01-01/eda/tables/` and 3 figures to `.../eda/figures/`. Every number below comes from the table named in brackets.

**Sample:** 8,760 hours per zone, with no hour excluded (see `DATA.md`).

## Vocabulary (D-24)

All measures are price **levels** in $/MWh, not log returns, because ERCOT prices go negative.

- **Spread:** the DA settlement point price minus the hourly mean of the four 15-minute RT settlement point prices. Positive means DA settled above RT.
- **Spread magnitude:** |spread|.
- **Conditional spread dispersion:** the std *and* the quantiles of the spread within a group (a temperature bin, a load decile, an hour). The std is reported, but it is driven by a handful of hours, so quantiles are read first.
- **Intra-hour RT volatility:** the std of the four 15-minute RT prices within an hour.
- **Tail frequency:** the share of hours with |spread| above the zone's 99th percentile (*relative*: $81, $91, $87 and $120 for H/N/S/W), or above $100 (*absolute*).

"Volatility" never appears without one of these qualifiers.

## Q1. What does the spread distribution look like? (EDA figure 1) [q1_spread_summary]

- **Small positive centre.** The median is +$1.9 to +$2.8, and DA > RT in 59–63% of hours. The means are near zero (−$0.2 to +$1.4), because rare large negative spreads offset the typical small DA premium.
- **Extremely heavy, asymmetric tails:**
  - skew −17.7 to −25.1 and excess kurtosis 659–1,135;
  - min −$1,416 to −$2,129, against max +$385 to +$727.
  - **The 1% largest hours carry 17–21% of all |spread|.** The left tail (RT above DA) is longer than the right in every zone (EDA figure 1B).
- **Implication for modelling.** These tails are the object of interest, so the primary response is the raw spread, with no winsorising (D-14 revised). Winsorised and asinh fits are sensitivity checks that show how much of each result the tails drive.
- **Persistence:** lag-1 autocorrelation is 0.46–0.75, while lag-24 is ≈ 0.

## Q2. How does load vary with temperature? (EDA figure 2) [q2_temp_load]

- **V-shaped in Houston, North and South.** Load falls with temperature below about 57–61 °F and rises above it. Within each branch the correlation is strong: r = 0.92–0.94 above 65 °F and −0.70 to −0.82 below. The overall Pearson r (0.35–0.68) badly understates the link.
- **A quadratic in temperature captures most of it.** T + T² explains **85–89%** of hourly load variance in H/N/S, against 12–46% for linear T. Adding hour-of-day and weekday raises this to 91–94%.
- **West is different.** The R² is only **0.32**, and load ranges over just 7.1–11.9 GW. Its load is largely industrial (Permian oil and gas) and only weakly weather-driven.
- **Implications:**
  - The motivation for second-order terms is direct.
  - Temperature and load are strongly dependent through a *nonlinear* relationship. A (T, L) quadratic design therefore has multicollinearity among T, T², L, L² and T·L. Actual load stays in the model, as the assignment is about temperature, demand and price. The collinearity is handled by coding, VIFs, the convex-hull mask and slice-based interpretation. The load-anomaly specification is reported only as a comparison.

## Q3. Is the spread more dispersed in hot periods? [q3_by_temperature, q3_by_temperature_excl_event, q3_by_system_net_load_decile, q3_by_zone_load_decile]

- **Yes, and in cold periods too. The pattern is U-shaped in temperature.**
  - In the 55–65 °F band the relative tail frequency is **0.0% in every zone**, and the median |spread| is $5–9.
  - At 85–95 °F the tail frequency is 1.1–2.6% and the median |spread| $7.9–8.4.
  - Above 95 °F it is 1.3–7.1% and $9.8–13.2.
  - Median intra-hour RT volatility shows the same U: $1.2–1.6 in the 55–75 °F bands (West $2.7), against $2.8–6.4 at ≤32 °F and $3.6–4.8 above 95 °F.
- **Hot and cold tails differ in sign:**
  - Above 95 °F the **mean spread is positive in every zone** (+$6.5 to +$19.2), i.e. DA priced scarcity that RT did not deliver.
  - Cold hours (≤45 °F) lean negative, i.e. RT above DA.
- **The cold-end magnitudes rest on a few events.** There are only 60–305 hours at ≤32 °F, spread over 7–35 days.
  - Excluding 2025-02-19/20 changes South's ≤32 °F mean spread from **−$92.6 to −$3.7**, and its std from $338 to $44.
  - The U-shape in median |spread| and in tail frequency survives that exclusion. **The size of the cold tail does not**, and the report says so.
- **System net load (load − wind − solar) separates calm from extreme hours most cleanly.**
  - The median |spread| is $5–6 in deciles 1–6, then $9.1 in decile 9 and $14.4 in decile 10.
  - The tail frequency rises from 0.0% to **6.0%** in the top decile.
- **West is again the exception.** Its tail frequency is highest at its *lowest* zonal load (decile 1: 4.2%, median |spread| $16). This fits renewable- and congestion-driven price swings rather than demand scarcity.

## Q4. Do relationships differ across zones? [q4_* tables]

- **Spreads co-move:**
  - Pearson r = 0.59–0.89; North–West is the highest pair (0.89).
  - Spearman r is 0.70–0.84 among H/N/S but only **0.52 with West**. West's ordinary hours are its own; its extremes are shared.
- **Extreme hours are mostly system-wide.** Of the 156 hours with an extreme in any zone, **70 have 3 or 4 zones extreme simultaneously**. The four zone fits are therefore not four independent pieces of evidence.
- **Levels:**
  - West DA averages about $42.7/MWh, against $29–40 elsewhere, and West has no on-peak premium.
  - DA IMHR is 8.5–8.8 MMBtu/MWh off-peak and 11.0–11.7 on-peak in H/N/S; West sits at about 12.4 in both.
- **Implications:**
  - Fit zone by zone, so each zone has its own surface. Report West separately.
  - Pooled inference must cluster by hour or day, not by zone.

## Q5. Are extremes concentrated in certain hours? (EDA figure 3) [q5_extremes_*]

- **Yes: at the ramps.** 75% of the 352 top-1% zone-hours fall in hours-beginning 5–7 and 17–21, which make up only 33% of all hours.
  - The evening peak (hours-beginning 19–21) alone holds 170.
  - **80% of the DA-above-RT extremes are in the evening (17–21).** RT-above-DA extremes (66% of all extremes) are spread over both ramps and the night.
- **They are concentrated in time.**
  - All 352 fall on **56 of 365 days**, and the top 5 days hold 30%.
  - 2025-02-19 (a cold snap: RT spikes, spreads down to −$1,416) contributes 40 zone-hours by itself.
  - The next day, 02-20, shows the reverse: DA priced continued scarcity and RT fell (+$121 to +$727).
- **Months:** extremes cluster in February, May, July–August and December. August's 35 extremes are all DA above RT.
- **Implication:** hour of day must be controlled for (model M3), so that the weather and load surface is not credited with ramp structure.

## Q6. What makes an extreme hour different? (report Figures 3–4) [results/<window>/report/: tail_by_net_load, tail_lift_summary, tail_fixed_threshold, extreme_profile, event_2025-05-16_*]

**How to regenerate:** `python -m ercot_rsm.cli report-assets` (and with `--start 2024-01-01 --end 2025-01-01`). Everything here is a descriptive association (D-35).

**Definitions.**
- **Tight hours:** the top decile of ERCOT system net load (load − wind − solar).
- **Extreme:** a zone's own top 1% of |spread|. The base rate is therefore 1% in every zone by construction, and ratios are relative risks.

**Concentration.**
- The tightest 10% of hours hold **53–68%** of each zone's extremes (47–65% without 2025-02-19/20).
- Pooled, extremes occur in 6.0% of tight hours vs 0.45% of the others: **13× (90% day-block bootstrap 8–21×)**.
- 2024: 12× (7–19×).

**Time-of-day robustness check.**
- With deciles ranked within each hour of day, the ratio roughly halves, to **7.4× (4.2–12.5×)**; 5.4× (2.9–9.2×) without the storm; 2024: 5.2× (2.5–9.5×).
- So the concentration is not explained by time of day alone.
- Season, weather, renewables and outages are *not* held fixed, so this is not a control and not a causal effect of net load.

**Fixed-dollar sensitivity (all zones pooled, tight vs other hours).**

| Event | Tight hours | Other hours | Ratio (90% CI) |
|---|---|---|---|
| \|spread\| > $50 | 13.4% | 1.78% | 7.5× (5.8–9.7) |
| \|spread\| > $100 | 5.3% | 0.39% | 13× (8–21) |
| \|spread\| > $250 | 1.6% | 0.06% | 25× (9–59); only 15 days |
| RT above DA by > $100 (unhedged real-time shortfall) | 2.5% | 0.35% | 7.1× (3.3–12.5) |

The ratio grows with the threshold. West has the highest base rates at fixed thresholds, because its typical spreads are larger.

**The two signs are different hours (medians).**
- **RT > DA extremes** (231 zone-hours on 39 days):
  - net-load surprise +1.1 GW (tighter than expected at 09:45 D−1);
  - wind 2.0 GW below forecast;
  - 43 °F;
  - intra-hour RT standard deviation $51 (vs $2 in ordinary hours);
  - only 48% in the tightest decile.
- **DA > RT extremes** (121 zone-hours on 21 days):
  - expected net load 64 GW;
  - surprise −2.7 GW (looser than expected);
  - net load up 2.5 GW from the previous hour;
  - 85 °F;
  - 83% in the tightest decile and 93% in ramp hours.

**Event: 16 May 2025.** The day was chosen by the D-35 rule; `representative_event_day` returns it.
- **Morning:** the system ran up to 6.7 GW tighter than the 09:45 forecast, with a slow solar ramp and a wind shortfall.
- **Afternoon:** wind sat 2.9–3.7 GW below forecast during HB 16–20; temperatures reached 97 °F.
- **Evening:** net load peaked at 59 GW. Houston's hourly RT reached $1,507 at 19:00 against a $184 DA price, and a West 15-minute interval cleared at $2,621.
- **Extremes:** 14 zone-hours across all four zones, HB 16–20, with spreads down to −$1,533.
- **Check:** the hourly means of the 15-minute RT prices equal the panel's RT exactly.

## Dependence: 8,760 hours ≠ 8,760 experiments [dependence_effective_n]

The effective sample size uses a Bartlett estimate with 168 lags. Per zone, out of 8,760 hours:

| Series | Effective sample size |
|---|---:|
| Spread | 1,700–3,364 |
| \|spread\| | 1,121–2,016 |
| Intra-hour RT volatility | 3,863–4,693 |
| Temperature | **70–79** |
| Load | **102–124** |
| Gas price | **73** (only 127 distinct daily values) |

Hourly persistence and seasonality mean the response surface is identified from far fewer independent observations than the row count suggests. This has three consequences:

- **Standard errors:** use HAC with 48 lags (cluster-by-day as a check).
- **Evidence:** prefer joint Wald tests on groups of terms, plus out-of-sample comparison, over individual p-values.
- **Gas:** the gas terms (G², T·G, L·G) are weakly identified. They largely encode season.

## RTC+B (December 5)

The post-go-live sample is 27 days (648 hours per zone), all in late December. The spread std there is lower in three zones (for example Houston $10 against $33), but that comparison confounds the market change with season and with a short window. **No structural conclusion is drawn.** The regime enters only as a nuisance intercept, and dropping those 27 days is a sensitivity check.

## Specification carried forward

| Choice | Decision |
|---|---|
| Primary response | Raw DA − RT spread (no winsorising). Also DA IMHR. |
| Factors (D-11) | Temperature, **actual** zonal load, Henry Hub gas, coded to ±1 at the 1st/99th percentiles. The load-anomaly variant is a comparison. |
| Models | M0 hour-of-day mean; M1 linear; **M2 full quadratic RSM**; M3 = M2 plus an hour-of-day block |
| Inference | HAC with 48 lags; grouped Wald tests; n_eff reported |
| Validation | Forward expanding-window CV (honest in time) and leave-one-month-out CV (interpolation). Both are explanatory: realised T, L and G are not known at DA time, so neither is a trading backtest. |
| Sensitivity | Winsorised and asinh responses; excluding Feb 19–20; excluding the post-RTC+B days; the load-anomaly variant |
| Extreme-risk (secondary) | Quantile surfaces at τ = 0.05 and 0.95, and \|spread\|, if time permits |
