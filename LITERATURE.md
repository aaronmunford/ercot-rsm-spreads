# Related work

Prior research motivates the questions in this study; it does not settle the answers. Each entry
below was checked against a publisher or RePEc record (most recently on 2026-10-10).

The report cites only the method references (NCSS Chapter 330; Myers et al. 2016; Newey & West
1987) and the two results it examines directly: Engle et al. (1986), on curvature in demand
(Figure 2), and Bessembinder & Lemmon (2002), on studying the size of the spread as well as its
sign (Table 3). The remaining papers are context: they suggest explanations that this study
does not test.

The "Result here" column says whether the 2025 ERCOT findings **confirm**, **qualify** or
**contrast with** each paper. The numbers come from `EDA_FINDINGS.md` and `MODEL_RESULTS.md`.

## Cited in the report

| Paper | Finding | Result here |
|---|---|---|
| Engle, Granger, Rice & Weiss (1986), *JASA* 81(394), 310–320 | Electricity demand is nonlinear in temperature | **Confirms.** Load is V-shaped in temperature, with its minimum at 57–61 °F, and T + T² explains 85–89% of load in Houston, North and South. **Qualifies** for West, where R² is 0.32 because load is largely industrial. |
| Bessembinder & Lemmon (2002), *J. Finance* 57(3), 1347–1382 | The forward premium depends on the variance and skewness of spot prices | **Partly confirms.** The median DA premium is small and positive (+$1.9 to +$2.8), and the tails carry the risk (skewness −18 to −25). The mean premium is about zero, and the realised-condition surface does not predict its sign. |
| Myers, Montgomery & Anderson-Cook (2016); NCSS Chapter 330; Newey & West (1987) | Methods | — |

## Context, not cited

| Paper | Finding | Result here |
|---|---|---|
| Longstaff & Wang (2004), *J. Finance* 59(4), 1877–1900 | Significant DA premia in PJM | **Qualifies.** The typical-hour premium is positive (DA > RT in 59–63% of hours), but rare RT spikes offset it on average. |
| Haugom & Ullrich (2012), *Energy Economics* 34(6), 1931–1941 | PJM short-term forwards converged toward unbiased predictors of spot | **Consistent:** mean spreads are near zero. |
| Jha & Wolak (2023), *AEJ: Economic Policy* 15(2), 292–330 | Convergence (virtual) trading made California DA prices track RT | **Consistent:** neither the realised-condition nor the ex-ante models have out-of-sample skill for the sign. |
| Ito & Reguant (2016), *AER* 106(7), 1921–1957 | Market power and limited arbitrage sustain a systematic DA premium (Iberia) | **Contrast, not tested.** ERCOT's small median premium could reflect limits to arbitrage, but market power is not examined here. |
| Birge, Hortaçsu, Mercadal & Pavlin (2018), *Energy Economics* 75, 518–533 | The MISO DA–RT premium persisted because of capital and regulatory limits on financial traders | Same contrast as Ito & Reguant. |
| Hogan (2013), *Economics of Energy & Environmental Policy* 2(2), 65–86 | ORDC scarcity pricing | Consistent with RT spikes dominating the left tail: the top 1% of hours carry 17–21% of the summed \|spread\|. |
| Woo, Horowitz, Moore & Pacheco (2011), *Energy Policy* 39(7), 3939–3944 | Wind lowers ERCOT price *levels* but raises their *variance* | **Consistent in spirit:** the structure found here is in magnitude and dispersion, not level. |
| Zarnikau, Woo, Moore & Horowitz (2011), *Energy Policy* 39(7), 3928–3938 | West wind and low West load drive zonal price divergence | **Confirms.** West's basis surprise concentrates at low West load and high DA-time wind forecasts. |
| Zarnikau, Woo & Zhu (2016), *Journal of Energy Markets* 9(4), 17–47, DOI 10.21314/JEM.2016.153 | Wind moves RT more than DA, and load and wind forecast errors make DA and RT diverge | **Confirms.** The net-load surprise has a Spearman ρ of −0.35 to −0.47 with the spread. |

## Recent and working papers

- **Hubert, Lolas & Sircar, "Trading Electrons: Predicting DART Spread Spikes in ISO Electricity
  Markets"**, arXiv 2601.05085 (2026; not peer-reviewed). It predicts positive and negative DA–RT
  spikes and sizes zonal virtual bids, with an application to NYISO. It is the closest recent
  work on spread *spikes*, but it is a forecasting and trading study rather than a weather-response
  study.
- **Graf, Quaglia & Wolak (2020)**, NBER Working Paper 27262. DA offers price in expected RT
  re-dispatch.
- **Weron (2014)**, *International Journal of Forecasting* 30(4), 1030–1081. A survey of
  electricity price models.

## Searched for, not found

No peer-reviewed study was found that fits a response surface, or any joint nonlinear model, of
the ERCOT DA–RT spread in temperature, load and gas. The nearest work falls into two groups:
forecasting studies of DA–RT spikes, and merit-order and forecast-error studies. The search was
not exhaustive.
