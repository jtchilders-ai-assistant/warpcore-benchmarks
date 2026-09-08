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

DATA_FILES := $(VIZ)/data/throughput_all.csv $(VIZ)/data/bench_matrix.json \
              $(VIZ)/data/swebench_fair.json \
              $(VIZ)/data/swebench_paired.json \
              $(VIZ)/data/swebench_reweighted.json
FIGS       := fig1_pareto fig2_swebench fig3_discrimination

# Instance set for the SWE-bench pre-flight check (seed-42 n=100, shared by all models).
SWEBENCH_INSTANCES ?= results/qwen3.6-35b-a3b/raw/swebench/preds_shuffle100.json

.PHONY: all figs data clean check preflight manifest check-artifacts audit samples ci preflight-serving preflight-selftest quality-preflight quality-preflight-selftest

all: figs

data: $(DATA_FILES)

$(VIZ)/data/throughput_all.csv: $(VIZ)/parse_sweeps.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) parse_sweeps.py

$(VIZ)/data/bench_matrix.json: $(VIZ)/collect_matrix.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) collect_matrix.py

# Infrastructure-fair SWE-bench denominators (the `fair n/N` in the README table).
$(VIZ)/data/swebench_fair.json: $(VIZ)/swebench_fair.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) swebench_fair.py

$(VIZ)/data/swebench_paired.json: $(VIZ)/swebench_paired.py $(VIZ)/swebench_fair.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) swebench_paired.py

$(VIZ)/data/swebench_reweighted.json: $(VIZ)/swebench_reweighted.py $(VIZ)/common.py
	cd $(VIZ) && $(PYTHON) swebench_reweighted.py

figs: data
	@for f in $(FIGS); do cd $(VIZ) && $(PYTHON) $$f.py && cd ..; done

# Verify regeneration is idempotent relative to the current worktree. Comparing
# with HEAD incorrectly rejects legitimate, intentionally uncommitted generated
# updates during pre-commit verification.
check:
	@$(PYTHON) $(VIZ)/check_generated.py \
		$(foreach f,$(DATA_FILES),--file $(f)) \
		$(foreach f,$(FIGS),--file $(VIZ)/out/$(f).png --file $(VIZ)/out/$(f).svg) \
		-- $(MAKE) --no-print-directory figs

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

# Refuse to launch against an endpoint that silently drops answers (ISSUES #15).
# Run this BEFORE any quality run: 30 s here vs a whole corrupted eval.
#   make preflight-serving                       # probe the default endpoint
#   make preflight-serving ENDPOINT=http://h:8000/v1 MODEL=name
# Exit 0 = usable, 1 = defect (do not launch), 2 = could not probe (also do not launch).
preflight-serving:
	@$(PYTHON) $(VIZ)/preflight_serving.py \
		$(if $(ENDPOINT),--endpoint $(ENDPOINT),) $(if $(MODEL),--model $(MODEL),) \
		$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS),)

# Same classifier, fixture-driven: no GPU, no network. This is what CI can run.
preflight-selftest:
	@$(PYTHON) $(VIZ)/preflight_serving.py --self-test

# What CI runs. Kept as one target so `make ci` locally == the GitHub job.
ci: check check-artifacts
	@$(PYTHON) $(VIZ)/preflight_serving.py --self-test
	@$(PYTHON) $(VIZ)/validate_samples.py --warn-only
	@$(PYTHON) $(VIZ)/quality_preflight.py --self-test
	@echo "OK: figures reproducible, no new provenance gaps."

# Mandatory quality-run gate: serving preflight + output budget + timeout arithmetic.
# Run this BEFORE any quality run. All three checks must pass.
#
# Full gate (requires a live endpoint):
#   make quality-preflight MODE=live ENDPOINT=http://h:8000/v1 MODEL=name \
#       MAX_GEN_TOKS=32768 AGGREGATE_TOK_S=64 CONCURRENCY=16 CLIENT_TIMEOUT=14400
#
# Arithmetic-only (offline, no GPU):
#   make quality-preflight MODE=arithmetic MAX_GEN_TOKS=32768 AGGREGATE_TOK_S=64 CONCURRENCY=16 CLIENT_TIMEOUT=14400
#
# MODE is explicit so a missing endpoint/model can never silently downgrade a requested live gate.
# Exit 0 = safe to launch, 1 = defect (do not launch), 2 = inconclusive (do not launch).
quality-preflight:
	@if [ "$(MODE)" = "live" ]; then \
		test -n "$(ENDPOINT)" && test -n "$(MODEL)" || { echo "ERROR: ENDPOINT and MODEL must be set together for MODE=live" >&2; exit 2; }; \
		mode_args="--endpoint $(ENDPOINT) --model $(MODEL)"; \
	elif [ "$(MODE)" = "arithmetic" ]; then \
		test -z "$(ENDPOINT)$(MODEL)" || { echo "ERROR: ENDPOINT and MODEL are invalid for MODE=arithmetic" >&2; exit 2; }; \
		mode_args="--check-timeout-only"; \
	else \
		echo "ERROR: set MODE=live or MODE=arithmetic; ENDPOINT and MODEL must be set together for live checks" >&2; exit 2; \
	fi; \
	$(PYTHON) $(VIZ)/quality_preflight.py $$mode_args \
		$(if $(MAX_TOKENS_PROBE),--max-tokens-probe $(MAX_TOKENS_PROBE),) \
		$(if $(MAX_GEN_TOKS),--max-gen-toks $(MAX_GEN_TOKS),) \
		$(if $(AGGREGATE_TOK_S),--aggregate-tok-s $(AGGREGATE_TOK_S),) \
		$(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) \
		$(if $(CLIENT_TIMEOUT),--client-timeout $(CLIENT_TIMEOUT),) \
		$(if $(SAFETY_FACTOR),--safety-factor $(SAFETY_FACTOR),)

# Fixture-driven self-test (no GPU, no network, runs in CI).
quality-preflight-selftest:
	@$(PYTHON) $(VIZ)/quality_preflight.py --self-test
