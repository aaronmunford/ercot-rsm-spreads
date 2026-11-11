PY ?= python
CLI = $(PY) -m ercot_rsm.cli
# a summer week by default; try START=2025-11-01 END=2025-11-08 for the DST fall-back day
START ?= 2025-07-07
END ?= 2025-07-14

.PHONY: test synthetic datasets sample fetch crosscheck build audit eda fit surprise basis crossyear report-assets

test:
	$(PY) -m pytest

synthetic:
	$(CLI) synthetic

datasets:
	$(CLI) datasets

sample:
	$(CLI) --start $(START) --end $(END) sample

fetch:
	$(CLI) fetch

crosscheck:
	$(CLI) fetch --only ercot crosscheck

build:
	$(CLI) build

audit:
	$(CLI) audit

eda:
	$(CLI) eda

surprise:
	$(CLI) surprise

basis:
	$(CLI) basis

crossyear:
	$(CLI) crossyear --years 2024 2025

fit:
	$(CLI) fit

report-assets:  # the tables and five figures behind the report -> results/<window>/report/
	$(CLI) report-assets
