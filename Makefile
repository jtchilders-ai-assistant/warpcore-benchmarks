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

.PHONY: all figs data clean check preflight manifest check-artifacts audit samples ci preflight-serving preflight-selftest quality-preflight quality-preflight-selftest contract run-quality run-swebench validate-campaign publish-campaign

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
ci: check check-artifacts contract
	@$(PYTHON) $(VIZ)/preflight_serving.py --self-test
	@$(PYTHON) $(VIZ)/validate_samples.py --warn-only
	@$(PYTHON) $(VIZ)/quality_preflight.py --self-test
	@$(PYTHON) -m pytest tests/test_run_quality.py tests/test_task5_acceptance.py tests/test_run_swebench.py tests/test_task6_hardening.py tests/test_validate_campaign.py tests/test_publish_campaign.py tests/test_task7_adversarial.py tests/test_task7_contracts.py tests/test_task7_authoritative_validator.py tests/test_task7_evidence_paths.py tests/test_task7_runner_integration.py tests/test_task7_submitted_and_scoring_provenance.py tests/test_task7_swe_digest.py tests/test_task7_swebench_contracts.py tests/test_lmeval_sidecar.py tests/test_result_registry.py tests/test_task9_readiness.py -q
	@echo "OK: figures reproducible, no new provenance gaps, suite contract valid."

# Suite and adapter contract validation (warpcore-v1 design §12 step 2).
# Validates that suite canonical file hashes are fresh and the suite schema is
# satisfied.  Exit 1 = diagnosed defect, 2 = unreadable input.
contract:
	@$(PYTHON) $(VIZ)/validate_suite.py suite/warpcore-v1.yaml \
		--adapter adapters/qwen3.6-35b-a3b.yaml \
		--prompt-tokens gsm8k=256,ifeval=373,gpqa_diamond=2808

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

# Contract-aware quality runner (Task 5).
# Required variables: SUITE, ADAPTER, BENCH, ENDPOINT, THROUGHPUT, CONCURRENCY, TIMEOUT, PROMPT_TOKENS
# Optional: RUN_ID, RUN_DIR, DRY_RUN=1, ALLOW_NO_SCREEN=1, RESUME=1
#
# PROMPT_TOKENS — measured tokenized prompt maxima for each benchmark (required).
#   Format: bench=N,bench2=N2  e.g. PROMPT_TOKENS=gsm8k=500,ifeval=2000
#
# Dry-run (inspect generated command, no network/GPU):
#   make run-quality SUITE=suite/warpcore-v1.yaml ADAPTER=adapters/qwen3.6-35b-a3b.yaml \
#       BENCH=gsm8k ENDPOINT=http://h:8000/v1 THROUGHPUT=64 CONCURRENCY=8 TIMEOUT=14400 DRY_RUN=1
#
# Live run (must be inside /usr/bin/screen):
#   screen -S quality-gsm8k
#   make run-quality SUITE=suite/warpcore-v1.yaml ADAPTER=adapters/qwen3.6-35b-a3b.yaml \
#       BENCH=gsm8k ENDPOINT=http://h:8000/v1 THROUGHPUT=64 CONCURRENCY=8 TIMEOUT=14400
#
# Exit 0 = success + DONE written, 1 = preflight/harness failure, 2 = screen guard, 3 = config error.
run-quality:
	@test -n "$(SUITE)"         || { echo "ERROR: SUITE is required (e.g. SUITE=suite/warpcore-v1.yaml)" >&2; exit 3; }
	@test -n "$(ADAPTER)"       || { echo "ERROR: ADAPTER is required (e.g. ADAPTER=adapters/qwen3.6-35b-a3b.yaml)" >&2; exit 3; }
	@test -n "$(BENCH)"         || { echo "ERROR: BENCH is required (e.g. BENCH=gsm8k)" >&2; exit 3; }
	@test -n "$(ENDPOINT)"      || { echo "ERROR: ENDPOINT is required (e.g. ENDPOINT=http://host:8000/v1)" >&2; exit 3; }
	@test -n "$(THROUGHPUT)"    || { echo "ERROR: THROUGHPUT is required (measured aggregate tok/s)" >&2; exit 3; }
	@test -n "$(CONCURRENCY)"   || { echo "ERROR: CONCURRENCY is required (number of parallel workers)" >&2; exit 3; }
	@test -n "$(TIMEOUT)"       || { echo "ERROR: TIMEOUT is required (lm-eval --timeout in seconds)" >&2; exit 3; }
	@test -n "$(PROMPT_TOKENS)" || { echo "ERROR: PROMPT_TOKENS is required (measured prompt maxima, e.g. gsm8k=500,ifeval=2000)" >&2; exit 3; }
	@$(PYTHON) $(VIZ)/run_quality.py \
		--suite $(SUITE) \
		--adapter $(ADAPTER) \
		--benchmark $(BENCH) \
		--endpoint $(ENDPOINT) \
		--throughput $(THROUGHPUT) \
		--concurrency $(CONCURRENCY) \
		--timeout $(TIMEOUT) \
		--prompt-tokens $(PROMPT_TOKENS) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		$(if $(RUN_DIR),--run-dir $(RUN_DIR),) \
		$(if $(DRY_RUN),--dry-run,) \
		$(if $(ALLOW_NO_SCREEN),--allow-no-screen,) \
		$(if $(RESUME),--resume,)

# Contract-aware SWE-bench runner (Task 6).
# Required variables: SUITE, ADAPTER, ENDPOINT, PROMPT_TOKENS
# Optional: RUN_ID, RUN_DIR, API_KEY, WORKERS, DRY_RUN=1, ALLOW_NO_SCREEN=1, RESUME=1
#
# PROMPT_TOKENS — measured tokenized prompt maxima for quality benchmarks (required for
#   adapter campaign-readiness validation even for SWE-bench runs).
#   Format: bench=N,bench2=N2  e.g. PROMPT_TOKENS=gsm8k=500,ifeval=2000,gpqa_diamond=1000
#
# Dry-run (inspect scaffold config, no network/GPU):
#   make run-swebench SUITE=suite/warpcore-v1.yaml ADAPTER=adapters/qwen3.6-35b-a3b.yaml \
#       ENDPOINT=http://h:8000/v1 PROMPT_TOKENS=gsm8k=500,ifeval=2000,gpqa_diamond=1000 DRY_RUN=1
#
# Live run (must be inside /usr/bin/screen on Mac mini):
#   screen -S swebench-run
#   make run-swebench SUITE=suite/warpcore-v1.yaml ADAPTER=adapters/qwen3.6-35b-a3b.yaml \
#       ENDPOINT=http://h:8000/v1 PROMPT_TOKENS=gsm8k=500,ifeval=2000,gpqa_diamond=1000
#
# Exit 0 = success + DONE written, 1 = preflight/generation/grading failure,
#          2 = screen guard or inconclusive, 3 = config error.
run-swebench:
	@test -n "$(SUITE)"         || { echo "ERROR: SUITE is required (e.g. SUITE=suite/warpcore-v1.yaml)" >&2; exit 3; }
	@test -n "$(ADAPTER)"       || { echo "ERROR: ADAPTER is required (e.g. ADAPTER=adapters/qwen3.6-35b-a3b.yaml)" >&2; exit 3; }
	@test -n "$(ENDPOINT)"      || { echo "ERROR: ENDPOINT is required (e.g. ENDPOINT=http://host:8000/v1)" >&2; exit 3; }
	@test -n "$(PROMPT_TOKENS)" || { echo "ERROR: PROMPT_TOKENS is required (measured prompt maxima, e.g. gsm8k=500,ifeval=2000,gpqa_diamond=1000)" >&2; exit 3; }
	@$(PYTHON) $(VIZ)/run_swebench.py \
		--suite $(SUITE) \
		--adapter $(ADAPTER) \
		--endpoint $(ENDPOINT) \
		--prompt-tokens $(PROMPT_TOKENS) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		$(if $(RUN_DIR),--run-dir $(RUN_DIR),) \
		$(if $(API_KEY),--api-key $(API_KEY),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(DRY_RUN),--dry-run,) \
		$(if $(ALLOW_NO_SCREEN),--allow-no-screen,) \
		$(if $(RESUME),--resume,)

# Campaign validator (Task 7).
# Validate a completed run directory before transitioning to 'validated'.
# Required variables: RUN_DIR, SUITE, ADAPTER
# Optional: FOR_PUBLICATION=1
#
# Exit 0 = all gates pass, 1 = at least one gate failed.
validate-campaign:
	@test -n "$(RUN_DIR)" || { echo "ERROR: RUN_DIR is required (path to the run directory)" >&2; exit 2; }
	@test -n "$(SUITE)"   || { echo "ERROR: SUITE is required (e.g. SUITE=suite/warpcore-v1.yaml)" >&2; exit 2; }
	@test -n "$(ADAPTER)" || { echo "ERROR: ADAPTER is required (e.g. ADAPTER=adapters/model.yaml)" >&2; exit 2; }
	@$(PYTHON) $(VIZ)/validate_campaign.py $(RUN_DIR) \
		--suite $(SUITE) \
		--adapter $(ADAPTER) \
		$(if $(FOR_PUBLICATION),--for-publication,)

# Canonical publication gate (Task 7).
# Read validated+current manifests and write canonical_matrix.json transactionally.
# Optional: OUTPUT=path/to/canonical_matrix.json, SUITE_ID=warpcore-v1
#
# Exit 0 = published successfully.
publish-campaign:
	@$(PYTHON) $(VIZ)/publish_campaign.py \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(SUITE_ID),--suite-id $(SUITE_ID),)
