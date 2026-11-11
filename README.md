# Response Surface Analysis of ERCOT Day-Ahead–Real-Time Price Spreads

*The role of temperature, electricity demand and natural gas prices* — Aaron Munford, New York University, October 2026.

**📄 Report: [`reports/report_v2.pdf`](reports/report_v2.pdf)** (10 pages).
**📓 Notebooks:** [`notebooks/01_research_log.ipynb`](notebooks/01_research_log.ipynb) (the dated research log) and [`notebooks/02_analysis.ipynb`](notebooks/02_analysis.ipynb) (the analysis, recomputing every table and figure in the report).

The study uses every hour of 2025 for ERCOT's Houston, North, South and West load zones. For each zone it builds:
- the day-ahead (DA) minus real-time (RT) price spread;
- implied market heat rates;
- load measures.

It then fits second-order response surface regressions (NCSS Chapter 330) in temperature, load and Henry Hub gas. 2024 is processed through the same pipeline as a partial replication.

## Key findings

1. **Load is strongly curved in temperature.**
   - In Houston, North and South, adding temperature squared raises R² from 0.12–0.46 to 0.85–0.89.
   - It also cuts later-month load error by 84–86% (25% in West, where load is much less weather-sensitive).
2. **Spreads are small in typical hours but have extreme tails.**
   - The median spread is a DA premium of +$1.9 to +$2.8/MWh, and DA settles above RT in 59–63% of hours.
   - Tails: a normal distribution would expect 0.4 zone-hours beyond ±$170; 144 occur.
3. **Adding quadratic and interaction terms does not improve predictions for later months.** Each model is compared with the same model without the six second-order terms:

   | Response | In-sample R², linear → quadratic | Later-month error, quadratic vs linear |
   |---|---|---|
   | Signed spread | 0.2–1.8% → 0.5–4.9% | worse in every zone (+0.3% to +5.2%) |
   | Spread magnitude \|S\| | 2.7–4.8% → 3.5–11.4%; terms jointly significant in every zone (HAC p ≤ 0.033) | worse in every zone (+0.5% to +6.3%) |
   | DA heat rate, with hour-of-day controls | 0.33–0.37 → 0.36–0.40 | worse in Houston, North and South (+3.5% to +7.7%); better only in West (−12.0%) |

4. **Extreme spreads concentrate in tight hours.**
   - In the tightest 10% of hours by system net load (load − wind − solar), a zone's top-1% |spread| occurs about **13×** as often as in other hours (90% day-block bootstrap interval 8–21×; 11× without the 19–20 February storm).
   - Ranking within each hour of day, the ratio is 7.4× (4.2–12.5×).
   - In dollars: |spread| exceeds $100/MWh in 5.3% of tight hours, against 0.39% of other hours.
   - 2024 gives 12×.
   - This is a descriptive association: season, weather and renewable output are not held fixed.
5. **16 May 2025 illustrates the limits.**
   - Houston's RT price reached $1,507/MWh against $184 day-ahead in the hour beginning 19:00, a spread of −$1,324.
   - A surprise model fitted on January–April and given the realised forecast error predicts −$1.90.
   - Knowing the system was tighter than expected matched the sign of the spike, not its size.

**Scope.** Every model uses conditions known only after the fact. The results explain past hours; they are not real-time trading forecasts.

## Repository guide

| File | Contents |
|---|---|
| [`reports/report_v2.pdf`](reports/report_v2.pdf) | The report |
| [`notebooks/01_research_log.ipynb`](notebooks/01_research_log.ipynb) | Dated research log: what was done, decided and learned |
| [`notebooks/02_analysis.ipynb`](notebooks/02_analysis.ipynb) | The analysis in the report's order, with every table and figure recomputed |
| [`DATA.md`](DATA.md) | Sources, alignment, validation and data dictionary |
| [`DECISIONS.md`](DECISIONS.md) | Every modelling choice with its alternatives (D-01 to D-35) |
| [`EDA_FINDINGS.md`](EDA_FINDINGS.md) | Exploratory results, with the CSV behind each number |
| [`MODEL_RESULTS.md`](MODEL_RESULTS.md) | RSM, surprise models, 2024 replication, West basis decomposition |
| [`LITERATURE.md`](LITERATURE.md) | Verified references and what each motivates |
| [`experiments/RSM_CONTROLS.md`](experiments/RSM_CONTROLS.md) | Isolated test of calendar/seasonal controls and net-load ramps (none improved the model robustly) |

```
config/study.toml       every tunable choice, tagged to DECISIONS.md
src/ercot_rsm/
  rsm.py                response surface regression (reproduces NCSS's worked example to every printed digit)
  features.py           spreads, heat rates, gas alignment, load, weather, calendar, forecast surprises
  pipeline.py           raw sources -> hourly zone panel
  eda.py, analysis.py   exploratory tables; the model ladder, validation and diagnostics
  surprise.py, basis.py, crossyear.py   extensions: forecast surprises, West basis, 2024 replication
  plots.py              report figures
  data/                 fetchers (Grid Status, ERCOT archives, Iowa Mesonet, EIA)
notebooks/              research log and analysis notebook
results/<window>/       every table and figure, as CSV and PNG (report inputs under report/)
experiments/            isolated experiments that do not touch the report
tests/                  pytest (125 tests, no network)
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env            # add a free gridstatus.io API key
set -a; source .env; set +a

make test                       # 125 tests, no network
make fetch build audit          # pull, assemble and audit the 2025 panel (cached; re-runs cost nothing)
make eda fit surprise basis     # exploratory tables, model ladder and extensions -> results/<window>/
make report-assets              # the report's tables and five figures -> results/<window>/report/
```

For the 2024 replication, run the same steps with `python -m ercot_rsm.cli --start 2024-01-01 --end 2025-01-01 <command>`, then `make crossyear`. Raw and processed data are not committed; `make fetch` rebuilds them.
