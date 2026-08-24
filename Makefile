# warpcore-benchmarks — figure regeneration
#
#   make figs     regenerate every figure from committed raw artifacts
#   make data     just re-derive the intermediate CSV/JSON
#   make clean    remove generated figures
#
# Figures are plots-as-code: nothing is hand-edited, and `make figs` on a clean
# checkout must reproduce byte-identical SVGs.

PYTHON ?= python3
VIZ    := viz

DATA_FILES := $(VIZ)/data/throughput_all.csv $(VIZ)/data/bench_matrix.json
FIGS       := fig1_pareto fig2_swebench fig3_discrimination

.PHONY: all figs data clean check
all: figs

data: $(DATA_FILES)

$(VIZ)/data/throughput_all.csv: $(VIZ)/parse_sweeps.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) parse_sweeps.py

$(VIZ)/data/bench_matrix.json: $(VIZ)/collect_matrix.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) collect_matrix.py

figs: data
	@for f in $(FIGS); do cd $(VIZ) && $(PYTHON) $$f.py && cd ..; done

# Verify the checked-in figures match what the code produces right now.
check: figs
	@git diff --stat --exit-code -- $(VIZ)/out $(VIZ)/data \
		&& echo "OK: committed figures match regenerated output" \
		|| (echo "STALE: run 'make figs' and commit the result"; exit 1)

clean:
	rm -f $(VIZ)/out/*.png $(VIZ)/out/*.svg
