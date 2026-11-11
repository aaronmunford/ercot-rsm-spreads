# Experiment: calendar/seasonal controls and net-load ramps in the quadratic RSM (2026-10-10)

**Status:** isolated. The report (`reports/report_v2.pdf`) is unchanged; its hash is checked before and after the run (`protected_hashes.txt`). No data were downloaded.

**Run:** `.venv/bin/python experiments/rsm_controls.py` (about 45 s). **Outputs:** `results/2025-01-01_2026-01-01/experiments/rsm_controls/*.csv`. Every number below comes from those CSVs.

## Design

**Models.** Every model shares the report's M2 core: a full quadratic in coded realised temperature, zonal load and Henry Hub gas, plus the RTC+B intercept. Each is fitted per zone for three responses: the spread, |spread| and the DA heat rate.

| Model | Adds | Information set |
|---|---|---|
| M2 | (baseline; reproduces the report's forward predictions **exactly**, max \|diff\| = 0) | retrospective |
| M2+C | hour-of-day dummies, non-business-day flag | retrospective (realised T, L, G) |
| M2+CS | + annual Fourier terms (2 harmonics) | retrospective |
| M2+Rrt / M2+Rfc | realised / 09:45 D−1 forecast net-load ramp, linear + square | retrospective |
| M2+C+Rrt / M2+C+Rfc | calendar + ramp, to test whether ramp adds anything beyond hour of day | retrospective |
| M2+CS+Rfc | calendar + season + forecast ramp | retrospective |
| EX | calendar + season + forecast ramp + forecast net load (no realised factors) | **ex-ante** |

**Retrospective vs ex-ante.** Every M2 variant uses the realised T, L and G of the hour it describes, so it is explanatory, whatever ex-ante terms are added. Only EX uses bid-time information alone. A forward test of a retrospective model measures stability over time, not forecasting skill.

**Same observations, same folds.**
- 8,757 of 8,760 hours per zone, the same rows for every model. Three hours lack a ramp: the first hour of the year and the DST changes.
- **A. Forward folds within 2025:** the report's own folds (train on months < m; test April–December, 6,601 hours).
- **B. Fit on 2024, predict 2025:** here training covers every season.

**Training-only preprocessing.** Inside every fold, the training rows alone set:
- the RSM coding (1st/99th percentiles → ±1);
- the ramp and net-load scaling;
- whether a regime dummy exists.

Calendar and Fourier terms are deterministic functions of the date. A leakage check confirms this: corrupting every test row leaves the fitted coding, scaling and coefficients unchanged in all 12 zone × response cases. Month dummies are impossible here, because a forward test month never appears in training.

**Criteria, fixed before the first run.**
- **Material:** median-zone skill gain ≥ 0.02 (2% of M0's squared error) or MAE −5%.
- **Stable:** in ≥ 3 of 4 zones, all of the following:
  - beats the reference in ≥ 7 of 9 folds;
  - Diebold–Mariano p < 0.05 (daily loss differentials, Newey–West);
  - gains in the year-ahead test as well.
- **Interpretable:** expected sign in ≥ 80% of refits, and VIF < 10.

## Results

**Bottom line: no addition passes all three criteria for any response. Nothing here justifies changing the report.**

| Response | Addition | Forward 2025, skill gain vs M2 (median; range) | Folds beating M2 | Fit-2024 → 2025, gain vs M2 | Verdict |
|---|---|---|---|---|---|
| Spread | calendar | −0.011 (−0.015 to −0.006) | 1–3 of 9 | +0.001 to +0.047 (all models still below M0: skill −0.13 to −0.67) | no gain |
| Spread | ramps (realised or forecast) | ≈ 0 (−0.009 to +0.001) | 2–6 | −0.007 to +0.002 | no gain |
| Spread | season (Fourier) | −0.36 (to −5.8) | 0–2 | −0.05 to +0.005 | harmful |
| \|spread\| | calendar | +0.019 (−0.003 to +0.029) | 4–7 | +0.008 to +0.040 | small; MAE worse in S, W; not stable |
| \|spread\| | ramps | +0.003 to +0.005 | 5–7 | −0.024 to −0.004 | not material, reverses across years |
| DA IMHR | calendar | **+0.265 (+0.23 to +0.32)** | **8–9** | −0.10 to +0.29 (worse than M2 in H, N, S) | material within 2025 (≈ the report's M3), **not stable across years** |
| DA IMHR | ramp beyond calendar | +0.021 to +0.025 median (+0.009 to +0.028) | 5–8 | **−0.06 to −0.29** | material and significant within 2025, **reverses out of year** |
| DA IMHR | EX (ex-ante) | +0.297 median (−0.19 to +0.43) | 8 in 3 zones | skill vs M0 −0.27 to −0.57 | unstable |

**Retrospective explanation (in-sample, all of 2025, HAC tests).**
- **Calendar.** It raises the IMHR R² from 0.15–0.21 to 0.37–0.41 (HAC Wald p < 0.001), and |spread| R² by 0.015–0.036.
- **Ramps.** They are significant for the IMHR (p < 0.001) and for |spread| (p ≤ 0.003), but not for the signed spread (p 0.09–0.14).
- **Season.** It adds about 0.02 of R² in sample. Out of sample it extrapolates badly whenever training has seen only part of the year.

**Redundancy.**
- **Ramps vs the RSM core:** nearly unrelated (VIF ≈ 1.2).
- **Ramps vs time:** **hour of day, calendar and season explain 51–54% of each ramp's variance.**
- **Realised vs forecast ramp:** correlation 0.92, so the two ramps carry nearly the same information.
- **Combined model:** with calendar included, the ramp VIF rises to about 2.4–2.5; the condition number of the full design is about 51.

**Stability and economic interpretation of the ramp coefficient (standardised, every refit).**
- **DA IMHR and |spread|:** positive in 100% (DA IMHR) and 98–100% (|spread|) of fold refits and in the 2024 fit. Up-ramps go with higher heat rates and larger mismatches, as the merit order and ramping scarcity would suggest.
- **Signed spread:** the expected negative sign (up-ramp → RT above DA) appears in 85% (M2+Rrt) to 100% (M2+C+Rrt) of refits for the realised ramp but only 68% for the forecast ramp, and the term is never significant.

## Interpretation

1. **Calendar controls are the only material improvement**, and only for the DA heat rate within 2025. That is essentially the report's M3 hour block; the non-business-day flag adds 0.005–0.012 of skill. Fitted on 2024, the same model predicts 2025 worse than M2 in three zones.
   - **One possible reason (not tested):** the hourly heat-rate profile moved between years, as solar growth shifted the evening peak.
2. **Ramps are economically sensible but not robust.** The signs are right and the terms are significant in sample. But half their variance is time of day; beyond calendar they add about 0.02 of skill within 2025 for the heat rate only; and that gain reverses when fitting on 2024 and predicting 2025. They add nothing for the signed spread.
3. **Seasonal Fourier terms are harmful with one year of expanding-window training.** In-year forward folds have not seen the season being predicted. A multi-year sample would be needed to estimate seasonality honestly.
4. **Ex-ante.** The only bid-time model (EX) does not beat the hour-of-day benchmark for the signed spread in either test. That is consistent with the report: direction is not predictable from bid-time calendar, ramp and net-load information.

**Recommendation.** Keep the report's specification. If anything is mentioned, it is a single limitation: hour-of-day and ramp effects on the heat rate are strong within a year but do not transfer across years. A multi-year sample is the precondition for seasonal and ramp controls.

## Caveats
- The criteria are strict on purpose: stable means within-year *and* across years.
- Two years of data is a small basis for the year-ahead test, and 2025 differs from 2024: RTC+B, solar growth, and only 24–33% of 2025's (T, L, G) hours fall inside 2024's range.
- The ramp is the hour-over-hour change in net load. Sub-hourly ramps are not in the cached hourly panel.
