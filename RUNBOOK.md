# RUNBOOK — Adding a new model to warpcore-benchmarks

**One document. Every new model follows this procedure exactly.**
The goal: every model in this repo was measured under identical conditions,
so comparisons are valid and results can be reproduced from the committed artifacts.

This runbook was derived from the actual commands, configs, and failure post-mortems
for the 7 models already in the repo. It replaces ad-hoc adaptation of prior scripts.

---

## Before you start

### Hardware
- **Inference host**: warpcore (DGX Spark, GB10, aarch64, ~121 GiB unified memory)
  SSH alias: `warpcore` → `140.221.17.30`, user `jchilders`
  warpcore runs: vLLM serving + on-box vllm throughput sweeps only.
- **Benchmark clients (lm-eval, SWE-bench)**: run on the Mac mini / Tribble (x86_64,
  `csi0359637.cels.anl.gov`), calling the warpcore remote endpoint at
  `http://csi370295.alcf.anl.gov:8000/v1`. Use `/usr/bin/screen` and stable paths under `$HOME`
  (not `/tmp`) for long-running client sessions — `/tmp` is reaped by macOS and has caused
  multiple data losses (see PROVENANCE.md §5).
  Exception: `vllm bench serve` throughput sweeps run ON warpcore (inside the serving container),
  as they need local access to the engine's metrics and completions port.

### Fixed versions — do not change without updating this file
| Component | Version | Where pinned |
|---|---|---|
| lm-eval | 0.4.12 | Mac mini venv (install under `~/workspaces/`) |
| Task configs | v3.0 (GPQA), v1.0 (GSM8K) | `results/<prior-model>/raw/*.yaml` |
| SWE-bench scaffold | mini-swe-agent, `qwen3_xml` parser | model config yaml |
| SWE-bench instance set | n=100, `--shuffle --seed 42 --slice 0:100` | all run scripts |
| viz toolchain | matplotlib 3.9.4, numpy 2.0.2 | `requirements-viz.txt` |
| Python (Mac viz) | `/usr/bin/python3` (3.9.6) | `Makefile` PYTHON var |

---

## Step 0 — Decide the container

**Every new model needs a container decision before anything else.**
The wrong container crashes silently or gives wrong numbers.

### Decision tree

```
Is the model a compressed-tensors FP8 checkpoint?
  (quant_method: compressed-tensors, format: float-quantized in config.json)
  → YES: use vllm/vllm-openai:cu129-nightly-aarch64
          add env: VLLM_TEST_FORCE_FP8_MARLIN=1
          add flag: --moe-backend marlin  (if MoE)
          example: Ornith-1.0-35B-FP8

Is it an MXFP4/NVFP4 MoE?
  (quantization: gpt_oss_mxfp4 or mxfp4 in config.json)
  → use eugr/spark-vllm:latest (prebuilt)
     launch via: ./launch-cluster.sh --solo -t eugr/spark-vllm:latest
     example: gpt-oss-120b, Nemotron-3-Super, Qwen3.6

Is it a standard BF16 or GPTQ-Marlin checkpoint?
  → use vllm/vllm-openai:cu129-nightly-aarch64
     no special flags needed

Is it a vendor-recommended specific version?
  (model HF README says "use vLLM vX.Y.Z")
  → use that exact image, e.g. vllm/vllm-openai:v0.27.1 (Lightning)
```

**NEVER use the `openai-gpt-oss-120b` recipe for tool-calling/agentic workloads.**
It uses CUTLASS MXFP4 + FlashInfer which crashes under concurrent structured-output requests.
For SWE-bench or pi-30, use the `eugr/spark-vllm:latest` container with `--moe-backend marlin`.

---

## Step 1 — Serve the model

### 1a. Write the launch script
Copy the nearest prior model's launch script. Required flags for ALL models:

```bash
# Template: results/<model>/raw/launch_<shortname>.sh
docker run -d --rm \
  --name vllm_prebuilt \
  --network host \
  --gpus all \
  --shm-size=8g \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \      # only for compressed-tensors FP8
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \       # only for GPTQ-Marlin
  -v /home/jchilders/.cache/huggingface:/root/.cache/huggingface \
  <IMAGE> \
  vllm serve <MODEL_ID> \
    --max-model-len <CTX> \
    --gpu-memory-utilization 0.90 \        # use 0.55 if running agent harness ON warpcore
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --reasoning-parser <PARSER> \          # see parser table below
    --tool-call-parser <PARSER> \          # for SWE-bench / agentic
    --enable-auto-tool-choice \
    --moe-backend marlin                    # for MoE models
```

**Reasoning parser table** (look up from model HF README or vLLM docs):
| Model family | `--reasoning-parser` | `--tool-call-parser` |
|---|---|---|
| gpt-oss-120b | `openai_gptoss` | `openai` |
| Nemotron-3-Super | `super_v3` via the checkpoint's `super_v3_reasoning_parser.py` plugin | `qwen3_coder` |
| Nemotron-3.5-Lightning | `nemotron_v3` | `openai` |
| Ornith-1.0-35B | `qwen3` | `qwen3_xml` |
| Laguna-S-2.1 | none | none (instruct model) |
| Qwen3-family | `qwen3` | `qwen3_xml` |

**gpu-memory-utilization rule:**
- Remote-client workloads (lm-eval from Mac mini, SWE-bench client from Mac mini): `0.90`
  These clients call warpcore over the network; agent processes run elsewhere and do not compete
  with vLLM for the GB10's unified memory pool.
- On-host agent harness (pi-30, any harness running ON warpcore): `0.55`
  → GB10 has unified memory; agent processes compete with vLLM for the same pool

### 1b. Wait for startup and verify

```bash
# On warpcore — startup takes 2–10 min depending on model size and repacking
watch -n 10 'curl -s -m6 http://localhost:8000/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print([m["id"] for m in d["data"]])"'

# Check for the startup-warning that predicts silent failures (ISSUES #15):
docker logs vllm_prebuilt 2>&1 | grep -i "reasoning\|Auto-initialization\|parser"
```

If you see `Auto-initialization of reasoning token IDs failed` → **stop here**.
The reasoning parser is broken. Fix before proceeding (see ISSUES.md #15).

### 1c. Preflight gate — mandatory, no exceptions

```bash
# From Mac mini / Tribble, from the repo root. Values must come from this model's
# retained sweep and intended client configuration.
make quality-preflight MODE=live \
  ENDPOINT=http://csi370295.alcf.anl.gov:8000/v1 \
  MODEL=<exact-model-id> MAX_TOKENS_PROBE=<usable-canary-budget> \
  MAX_GEN_TOKS=<generation-ceiling> AGGREGATE_TOK_S=<measured-at-C> \
  CONCURRENCY=<C> CLIENT_TIMEOUT=<seconds>
```

- Exit 0 → proceed
- Exit 1 → defect detected, fix before proceeding
- Exit 2 → endpoint unreachable, also stop

**This takes ~30 seconds. It has prevented multiple multi-hour runs from producing
zero-scored results.**

---

## Step 2 — Throughput sweep (derive concurrency)

The concurrency setting for quality runs must come from measurement, not copying
from another model. A 120B model at its saturation point ≠ a 35B model at its
saturation point.

```bash
# On warpcore, inside the serving container
MODEL="<MODEL_ID>"
docker exec vllm_prebuilt vllm bench serve \
  --base-url http://localhost:8000 \
  --model "$MODEL" \
  --backend openai \                    # NOT openai-chat (models stop early)
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --ignore-eos \
  --num-prompts <N> \                   # start at 3×concurrency
  --max-concurrency <C>                  # sweep: 1, 2, 4, 8, 16, 32, 64, 128
```

**Reading the results:**
- Output tok/s climbs then plateaus → use the concurrency at plateau for quality runs
- If still climbing at your highest concurrency → this is a **floor**, not a ceiling.
  The `--max-num-seqs` cap reading was **falsified on Laguna** (§3a): a large rise at c=128 looked
  like a cap, but c=192 added only +10.9% and c=256 only +2.9% — the heuristic overstated headroom.
  `SchedulerConfig.max_num_seqs` also admitted 150–172 concurrent while its documented default was 128.
  **Always extend the sweep past the knee** (to c=256 or c=384) before reporting a peak.
  Report the measured top point as a measured floor if you cannot extend, not as a cap-limited ceiling.
- For quality runs: use **½ × saturation concurrency** to leave headroom for long reasoning traces

**Save the sweep log** to `results/<model>/raw/throughput_sweep/vllm_sweep.sh` and
`sweep.log`. This is a required artifact (PROVENANCE.md §2).

**Tokenizer override** (for models with novel tokenizer_class):
```bash
docker exec vllm_prebuilt vllm bench serve ... --tokenizer <BASE_REPO>
```
Example: Intel AutoRound repos declare `TokenizersBackend` → pass `--tokenizer Qwen/Qwen3.5-122B-A10B`.

---

## Step 3 — Quality suite

Run all four benchmarks. Order matters: GPQA is longest, run it first.

### Before every run: verify endpoint still alive
```bash
curl -s -m6 http://csi370295.alcf.anl.gov:8000/v1/models
```

### 3a. GPQA-Diamond (canonical task config)

**Task config**: copy `results/laguna-s-2.1-118b/raw/quality/gpqa/gpqa_diamond_cot_zeroshot_clean.yaml`
(v3.0). Do not modify it. This is the canonical config for all models.

**Time budget arithmetic** (do this before launching, not after a timeout):
```
output_budget_tokens = 65536          # canonical for all models
model_tok_per_sec = <from sweep>      # single-stream warm throughput
max_item_time_s = output_budget_tokens / model_tok_per_sec
required_timeout_s = 2 * max_item_time_s  # default quality-preflight safety factor
# e.g. at 4 tok/s: 65536/4 = 16384s (~4.5h) per item in the worst case
# required timeout at the default 2x safety factor = 32768s (~9.1h)
# With concurrency=4: effective throughput = 4 * tok/s
# Total time ≈ (198 items / concurrency) * max_item_time_s * 0.5 (most finish early)
```

Gate on p90, not mean. If the arithmetic says a single item could take >2h at your concurrency, reduce concurrency and increase timeout accordingly.

```bash
# Run script template (adapt concurrency and timeout from arithmetic above)
# Runs ON Mac mini / Tribble — calls warpcore remote endpoint
# LMEVAL: path to lm_eval binary on the Mac mini (e.g. ~/workspaces/lmeval-venv/bin/lm_eval)
MODEL="<MODEL_ID>"
BASE="http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
TASK_DIR="$HOME/workspaces/warpcore-benchmarks/results/<model>/raw/quality/gpqa"
LMEVAL_BIN="$HOME/workspaces/lmeval-venv/bin/lm_eval"
CLIENT_TIMEOUT=<TIMEOUT_FROM_QUALITY_PREFLIGHT>

"$LMEVAL_BIN" \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=${CLIENT_TIMEOUT}" \
  --tasks gpqa_diamond_cot_zeroshot_clean \
  --include_path "${TASK_DIR}" \
  --gen_kwargs "max_gen_toks=65536,temperature=0" \
  --output_path "${TASK_DIR}" \
  --log_samples \
  --seed 42 \
  2>&1 | tee "${TASK_DIR}/gpqa.log"
```

**Always run in /usr/bin/screen on the Mac mini under $HOME paths.** A session disconnect kills the
client and wastes the GPU time. Use stable `$HOME`-relative output paths — `/tmp` is reaped by macOS.

```bash
/usr/bin/screen -dmS gpqa_<shortname> bash run_gpqa.sh
```

### 3b–c. GSM8K and IFEval

**GSM8K:** copy `results/nemotron-3.5-lightning-30b/raw/gsm8k_cot_zeroshot_clean.yaml` (v1.0).
Use its short-form **8192-token** ceiling.

**IFEval:** use lm-eval's `ifeval` task with the same **65536-token ceiling** as GPQA for reasoning models. Although many IFEval prompts are short, a reasoning model can spend its completion budget in hidden thinking before emitting final content; 8k silently depressed Ornith by 2.96 points even after replay.

```bash
# GSM8K (short-form) — run from Mac mini / Tribble
<LMEVAL_BIN> \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=http://csi370295.alcf.anl.gov:8000/v1/chat/completions,num_concurrent=<N_FROM_SWEEP>,max_retries=8,tokenized_requests=False,timeout=3600" \
  --tasks gsm8k_cot_zeroshot_clean \
  --include_path "$HOME/workspaces/warpcore-benchmarks/results/<model>/raw" \
  --gen_kwargs "max_gen_toks=8192,temperature=0" \
  --output_path "$HOME/workspaces/warpcore-benchmarks/results/<model>/raw/quality/gsm8k" --log_samples --seed 42

# IFEval (offline reasoning ceiling) — run from Mac mini / Tribble
<LMEVAL_BIN> \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=http://csi370295.alcf.anl.gov:8000/v1/chat/completions,num_concurrent=<N_FROM_SWEEP>,max_retries=8,tokenized_requests=False,timeout=7200" \
  --tasks ifeval \
  --gen_kwargs "max_gen_toks=65536,temperature=0" \
  --output_path "$HOME/workspaces/warpcore-benchmarks/results/<model>/raw/quality/ifeval" --log_samples --seed 42
```

### 3d. SWE-bench (runs from the Mac mini, not warpcore)

SWE-bench test containers are x86 Docker images and cannot run on warpcore's aarch64/GB10.
Run the client from the Mac mini; it calls warpcore over the network.

**Instance set — fixed, same for every model:**
```bash
# In the mini-swe-agent config yaml:
dataset_name: princeton-nlp/SWE-bench_Verified
dataset_split: test
dataset_slice: 0:100
shuffle: true
shuffle_seed: 42
```

**Agent config**: copy `results/ornith-35b/raw/swebench/swebench_ornith_config.yaml` and change:
- `model_name`: `hosted_vllm/<MODEL_ID>`
- `api_base`: `http://csi370295.alcf.anl.gov:8000/v1`
- `tool_call_parser`: match the model's parser

**Warm the Docker cache before the run:**
```bash
# Pull the SWE-bench test images on the Mac mini first (avoids 120s pull timeouts)
# Qwen3.6 lost 22/100 instances to cold-cache Docker pull timeouts
bash results/<prior>/raw/swebench/run_smoke.sh   # 3-item smoke test
```

**Run in screen on the Mac mini** (Mac mini has no tmux):
```bash
/usr/bin/screen -dmS swe_<shortname> bash results/<model>/raw/swebench/run_gen_n100.sh
```

---

## Step 4 — Post-run validation

After each benchmark completes, before committing:

### 4a. Check empty-content rate
```bash
# Check samples JSONL for empty responses
python3 scripts/check_empty_rate.py results/<model>/raw/quality/<bench>/samples_*.jsonl
# Must be < 2%. If > 2%, re-serve before committing (see ISSUES.md #15)
```

If you see empty items:
1. Classify by signature (AGENTS.md table: BUDGET / PARSER / EMPTY)
2. BUDGET → raise `max_gen_toks`, re-run the affected items
3. PARSER → re-serve using `replay_empties.py` against a live endpoint

### 4b. Write the manifest
```bash
make manifest MODEL=<model-dir>
# Writes results/<model>/raw/<bench>/manifest.json
# Fill in anything marked "unrecorded" — especially the HF model revision sha
```

Get the HF revision sha:
```bash
curl -s "https://huggingface.co/api/models/<ORG>/<MODEL>" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('sha','unknown'))"
```

### 4c. Run CI gates
```bash
# From the repo root on the Mac mini
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
```

All four targets must pass:
- `make check` — figures reproduce from committed artifacts
- `make check-artifacts` — provenance ratchet (no new gaps)
- `make preflight-selftest` — classifier fixtures
- `make samples` — empty-rate check (warn-only currently; will become hard gate)

**CI green locally == CI green on GitHub.**

---

## Step 5 — Commit

```bash
cd ~/workspaces/warpcore-benchmarks

# Stage everything for this model
git add results/<model>/

# Commit with a consistent message format
git commit -m "Add <MODEL_ID> results

Benchmarks: GPQA-Diamond, GSM8K, IFEval, SWE-bench
Hardware: warpcore (DGX Spark GB10, aarch64)
vLLM: <VERSION>, container: <IMAGE>
lm-eval: 0.4.12
Empty-content rate: GPQA <N>%, GSM8K <N>%, IFEval <N>%
SWE-bench: <N>/100 resolved (n=100, seed-42, McNemar vs prior)

manifest.json: results/<model>/raw/*/manifest.json"

git push origin main
```

Update the README table with the new model's scores.

---

## Comparison rules (do not skip)

**GPQA, GSM8K, IFEval** — direct comparison is valid only if:
- Same task config version (check `metadata.version` in the YAML)
- Same output budget (`max_gen_toks`)
- Empty-content rate < 2% for both models

**SWE-bench** — use McNemar on discordant pairs, not comparison of raw rates:
```bash
make data    # runs viz/swebench_paired.py from committed artifacts
```
The paired test accounts for the fact that both models ran the identical seed-42 instance set.
Never compare two models whose submitted sets differ.

**"Not distinguishable" ≠ "tied."** Report the p-value and say which conclusion is supported.

---

## Quick reference: what goes wrong and how to catch it

| Failure | Symptom | Catch |
|---|---|---|
| Reasoning parser broken (ISSUES #15) | content=null, finish=stop, reasoning populated | `make quality-preflight MODE=live ...` |
| Output budget too small | content=null, finish=length | Budget arithmetic before launch |
| Wrong concurrency | Run abandoned at timeout | Throughput sweep first |
| Docker cold cache (SWE-bench) | 22/100 instances lost | Smoke test + pre-pull |
| CUTLASS crash (tool-calling) | Engine dies, docker ps -a empty | Use marlin container |
| Unified-memory OOM | Engine OOM-killed, no logs | Use --gpu-memory-utilization 0.55 for on-host harness |
| Stale task config | KeyError: choices, or wrong dataset_path | Copy canonical v3.0 YAML |
| Lost screen session | Run killed mid-way | Always /usr/bin/screen on Mac mini, $HOME paths |

---

*Last updated: 2026-09-06. Derived from LESSONS.md, PROVENANCE.md, AGENTS.md,
and the actual run artifacts for 7 models (gpt-oss-120b, Nemotron-3-Super,
Nemotron-3.5-Lightning, Qwen3.5-122B, Qwen3.6-35B, Ornith-1.0-35B, Laguna-S-2.1).*
