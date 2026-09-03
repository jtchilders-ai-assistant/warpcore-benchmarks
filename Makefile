# warpcore-benchmarks — figure regeneration
#
#   make figs             regenerate every figure from committed raw artifacts
#   make data             just re-derive the intermediate CSV/JSON
#   make preflight        verify the SWE-bench container images are cached before a run
#   make manifest         scaffold a manifest.json  (MODEL=<m> BENCH=<b>)
#   make check-artifacts  fail if a reported number is missing its artifact
#   make samples          fail on silent scoring failures (empty responses)
#   make ci               everything CI runs
#   make clean            remove generated figures
#
# Figures are plots-as-code: nothing is hand-edited, and `make figs` on a clean
# checkout must reproduce byte-identical SVGs.

PYTHON ?= python3
VIZ    := viz

DATA_FILES := $(VIZ)/data/throughput_all.csv $(VIZ)/data/bench_matrix.json
FIGS       := fig1_pareto fig2_swebench fig3_discrimination

# Instance set for the SWE-bench pre-flight check (seed-42 n=100, shared by all models).
SWEBENCH_INSTANCES ?= results/qwen3.6-35b-a3b/raw/swebench/preds_shuffle100.json

.PHONY: all figs data clean check preflight manifest check-artifacts audit samples ci

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

# Verify every SWE-bench container image is cached before launching a run.
# A warm cache makes this a fast no-op; a cold one is why the 2026-08-04 Qwen3.6
# run lost 22/100 instances to a 120s docker pull timeout. Add PULL=1 to fetch.
preflight:
	@$(PYTHON) $(VIZ)/swebench_preflight.py --instances $(SWEBENCH_INSTANCES) $(if $(PULL),--pull,)

clean:
	rm -f $(VIZ)/out/*.png $(VIZ)/out/*.svg

# --- provenance enforcement (PROVENANCE.md §4) ---------------------------------
#
# These are the targets PROVENANCE.md promised. Until now they were prose, and
# audit_provenance.py ended in an unconditional `return 0` -- it printed 17 gaps
# and still reported success, so nothing could ever fail on them.

# Scaffold a manifest. Auto-fills what it can probe; writes "unrecorded" -- never
# a guess -- for the rest.  make manifest MODEL=ornith-35b BENCH=swebench
manifest:
	@test -n "$(MODEL)" || (echo "usage: make manifest MODEL=<m> BENCH=<b> [ENDPOINT=url]"; exit 2)
	@test -n "$(BENCH)" || (echo "usage: make manifest MODEL=<m> BENCH=<b> [ENDPOINT=url]"; exit 2)
	@$(PYTHON) $(VIZ)/manifest_scaffold.py --model $(MODEL) --bench $(BENCH) \
		$(if $(ENDPOINT),--endpoint $(ENDPOINT),) $(if $(FORCE),--force,)

# Fail if a reported number is missing its artifact. Ratcheted: the 17 known gaps
# are accepted via viz/data/provenance_baseline.json, but a NEW gap exits 1.
# STRICT=1 fails on any gap at all (the end goal, once the backlog is cleared).
check-artifacts:
	@$(PYTHON) $(VIZ)/audit_provenance.py $(if $(STRICT),--strict,)

audit: check-artifacts

# Detect silent scoring failures (ISSUES #15): items where the model returned no
# content, were scored 0, and quietly dragged a published average down.
samples:
	@$(PYTHON) $(VIZ)/validate_samples.py $(if $(MAX_EMPTY),--max-empty-rate $(MAX_EMPTY),)

# What CI runs. Kept as one target so `make ci` locally == the GitHub job.
ci: check check-artifacts
	@$(PYTHON) $(VIZ)/validate_samples.py --warn-only
	@echo "OK: figures reproducible, no new provenance gaps."
