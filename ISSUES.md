# Serving & Benchmarking Issues on Warpcore — and Fixes

Every problem hit while benchmarking gpt-oss-120b on the Warpcore DGX Spark (GB10), with root cause
and the fix that worked. Recorded so the next model/harness run doesn't rediscover them.

Hardware context: single GB10 unified-memory device, vLLM `0.23.1rc1.dev…` (nightly arm64) in the
`vllm_prebuilt` container (image `eugr/spark-vllm:latest`, MARLIN MoE + TRITON attention). See
[HARDWARE.md](HARDWARE.md).

---

## 1. gpt-oss needs the CHAT endpoint, not raw completions
**Symptom:** pointing lm-eval at `.../v1/completions` (`--model local-completions`) produced garbage —
the model blurted a short answer then rambled — and lm-eval's chat-template rendering on that path
failed with a masked error: `RuntimeError: Session is closed` → `UnboundLocalError: local variable
'outputs' referenced before assignment` (an lm-eval `api_models.py` error-handler bug hides the real
cause).
**Root cause:** gpt-oss is a reasoning model; it needs its harmony/reasoning chat format applied
server-side, which only the chat endpoint does.
**Fix:** use `--model local-chat-completions` with `base_url=<host>/v1/chat/completions`.

## 2. MCQ answer-extraction was unreliable (GSM8K strict 0%, GPQA false positives)
**Symptom:** GSM8K flexible-extract 90% but strict-match **0%**; GPQA stock flexible-extract gave a
**false positive** (matched a stray `(B)` when the model actually concluded "option c").
**Root cause:** reasoning-model output buries the answer in prose/LaTeX. The stock `strict-match` regex
wants a literal "The answer is X" gpt-oss never writes → 0%. The stock `flexible-extract` grabs the
*last* `(X)` anywhere in the text (`group_select: -1`) → lucky/incorrect matches.
**Fix (MCQ):** a custom task (`gpqa_diamond_cot_zeroshot_clean`, in `results/gpt-oss-120b/raw/`) that
(a) appends a strict format instruction — *"give your final answer on a new line in EXACTLY this
format: The answer is (X)"* — and (b) uses a regex anchored to that exact line:
`[Tt]he answer is \(?([A-D])\)?`. Verified deterministic: parsed letter == the model's stated final
answer on every sampled item; misses were then real model errors, not parse errors.
**Fix (GSM8K):** report the **flexible-extract** metric (last number), not strict-match — verified it
pulls the model's actual answer.

## 3. IFEval scorer crashed on empty (None) reasoning-model output
**Symptom:** the whole run completed inference (~47 min) then crashed at the scoring step with
`AttributeError: 'NoneType' object has no attribute 'strip'` in `ifeval/utils.py`. Because lm-eval
writes results/samples **only after all scoring finishes**, this **lost the entire run's inference.**
**Root cause:** gpt-oss can return `content: null` when it spends its whole token budget in the
reasoning channel and emits no final answer. IFEval's scorer didn't guard against `None`.
**Fix:** patch `ifeval/utils.py` `process_results` to coerce `None → ""` right after
`response = results[0]` (an empty response then correctly scores 0 instead of crashing). Also: **run
each task as its own lm_eval process** so a scorer crash in one task can't discard the others' work.

## 4. Server disconnects on very long generations (GPQA tail)
**Symptom:** GPQA runs died at ~94% with `aiohttp.client_exceptions.ServerDisconnectedError: Server
disconnected` (again masked by the `outputs` UnboundLocalError). The last, hardest GPQA items generate
enormous reasoning traces (16K-token budget, 100–145 s each); the server dropped the connection on
those. Cost two full GPQA runs.
**Root cause:** many simultaneous multi-minute generations at high concurrency strain the endpoint /
its timeout, and a dropped connection escapes lm-eval's retry.
**Fix:** for tasks with long generations, **drop `num_concurrent` to 4**, raise `max_retries` to 5 and
`timeout` to 1200 s. This completed GPQA-198 cleanly. If the tail still fails, use lm-eval `--use_cache`
so completed requests survive a crash and a re-run only redoes failures.

## 5. Miscellaneous setup gotchas
- **numpy 2.x ABI:** lm-eval + torch throw `_ARRAY_API not found` mid-run → pin `numpy<2`.
- **IFEval deps:** hard-requires `langdetect`, `immutabledict`, `nltk` (else `ModuleNotFoundError`).
- **Gated dataset:** GPQA (`Idavidrein/gpqa`) is gated on HuggingFace — needs an `HF_TOKEN` with access
  granted (accept terms on the Hub page first).
- **Dummy API key required:** vLLM ignores the key but lm-eval/AutomationBench require one be set
  (`OPENAI_API_KEY=dummy`, `--api-key dummy`).
- **SSH target:** use the `warpcore` alias, not the FQDN (`csi370295.alcf.anl.gov`) — the FQDN isn't in
  `known_hosts` and fails SSH host-key verification (it does work for the HTTP endpoint on :8000).

## 6. Orphaned benchmark process polluted the throughput sweep
**Symptom:** during the concurrency sweep, `curl .../metrics | grep num_requests_running` showed MORE
in-flight requests than the current `--max-concurrency` (e.g. 2 when running c=1). The c=1 result came
out low (29 tok/s vs the expected ~34).
**Root cause:** a `vllm bench serve` started via a foreground `ssh warpcore 'docker exec ...'` that
*timed out client-side* did NOT die — the `docker exec` kept running inside the container and its
requests kept hitting the engine, competing with the real sweep. Killing the host-side `docker exec`
wrapper does not stop the in-container process.
**Fix:** kill it *inside* the container:
`docker exec vllm_prebuilt bash -c 'ps -eo pid,args | grep "bench serve" | grep -v grep'` → `docker exec
vllm_prebuilt kill <PID>`. Then re-run the polluted level (c=1 clean → 34.1 tok/s, matching baseline).
**Prevention:** run sweeps via a persistent background session (not a foreground call that can time out),
and assert `num_requests_running == 0` before each level.

## 7. `vllm bench serve --result-dir` writes INSIDE the container
Results saved to a path under `--result-dir` land in the **container** filesystem, not the host. Pull
them with `docker cp <container>:/path/file ./`. (The per-level metrics also print to stdout, so the
run log captures them regardless.)

---

## 8. NVFP4 MoE `cutlass` backend crashes on GB10 (Nemotron-3-Super-120B) — use MARLIN
**Symptom (two failure modes):** (a) the stock `spark-vllm-docker` recipe forces `--moe-backend
cutlass` → engine init fails with `ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support the
deployment configuration since kernel does not support no act_and_mul MLP layer`. (b) Dropping the
flag lets vLLM's oracle auto-select `FLASHINFER_CUTLASS`, which loads fine but **crashes on the first
decode** with `CUDA error: an illegal instruction was encountered` (`cudaErrorIllegalInstruction`).
**Root cause:** the CUTLASS-family NVFP4 MoE kernels are incompatible with this model's MLP config /
unstable on GB10 CUDA-graph execution. Same lesson as the gpt-oss MXFP4 CUTLASS crash.
**Fix:** force `--moe-backend marlin`. MARLIN NVFP4 MoE is the GB10-stable path — verified correct
output and survives sustained concurrent load (server up 2+ days, zero crashes across the full
benchmark). Also bump `--max-num-seqs` (recipe default 10 → 24) to relieve the eval-throughput
bottleneck.

## 9. lm-eval aborts the WHOLE run when one request disconnects — patch the client to survive
**Symptom:** on a very verbose reasoning model (Nemotron-3-Super generates 5–12 min traces on hard
GPQA items), a request occasionally drops (`RuntimeError: Session is closed`) and lm-eval dies with
`UnboundLocalError: local variable 'outputs' referenced before assignment` at `api_models.py:504` —
**discarding the entire run's completed inference** (this killed three runs, ~2.5h + 8h wasted). A
related crash: `RegexFilter`/scorers throw `TypeError: expected string or bytes-like object` on
`content: null` (reasoning model spent its whole budget reasoning without a final answer).
**Root cause:** (i) the error handler logs an unbound `outputs` var, turning a *retryable* disconnect
into a fatal crash; (ii) `asyncio.gather(*tasks)` propagates the first exhausted-retry exception and
cancels all siblings; (iii) filters/scorers don't guard `None` responses.
**Fix (three source patches to lm-eval, applied per-venv):** (1) drop the unbound `{outputs}` from the
`amodel_call` except-handler log; (2) make `get_batched_requests` gather with `return_exceptions=True`
and substitute an empty-answer fallback for failed items (they score 0, run COMPLETES); (3) coerce
`None → ""` in `RegexFilter` and the IFEval scorer. Also run reasoning-heavy models at LOW concurrency
(c=4) with a LONG client timeout (≥3600 s). Full copy-paste patch: see the `lm-eval-vllm-endpoint`
Hermes skill (`references/api_models_survive_patch.md`).

## 10. Verbose reasoning + null-content: a real capability/cost characteristic, not just a harness bug
Nemotron-3-Super-120B routinely exhausts even a 16 384-token budget in its reasoning channel on hard
GPQA items and returns `content: null` (scored 0). Combined with 3–6 min/item generation, GPQA-Diamond
took **9h15m** to evaluate (vs a fraction of that for gpt-oss). This depresses its GPQA score and makes
it expensive to serve — worth recording as a model property, not dismissing as noise.

## 11. gpt-oss pi-30 = 6/30 on the `--exp-mxfp4` build (server crash), 30/30 on the crash-fixed image
Running the pi-30 agentic-coding sweep against gpt-oss-120b served by the recipe's default
`vllm-node-mxfp4` build produced a bogus **6/30**: P1–P6 passed, then the engine hit
`cudaErrorIllegalAddress` (the CUTLASS MXFP4 MoE crash, Signature 1) under the sustained
tool-calling / guided-decoding load, and every later problem failed. Tell-tale: the throughput
problems mid-run read absurd values (P15 ~0.08 Mtok/s vs the expected ~21; P23 ~0.11 Mn/s vs ~105) —
a 250–1000× collapse = the server degraded/died, not the model. By the **ZERO-SCORE RULE**, a score
that contradicts a known-good baseline (gpt-oss is a 30/30-class model) is an infra failure, not a
result. **Fix:** re-serve gpt-oss on the crash-fixed `eugr/spark-vllm:latest` image (MARLIN MXFP4 MoE +
TRITON), container `vllm_prebuilt`, then **verify with the concurrent guided-decoding stress test
before re-running** (`stress_guided_decoding.py 60 16` → 60/60 OK, engine alive). On the fixed image
gpt-oss scored a clean **30/30**. **The serving container is a first-class benchmark variable** — the
same weights score 6/30 or 30/30 depending on it.

---

## 12. Qwen3.5-122B int4 fails to load: `Tokenizer class TokenizersBackend does not exist`
The `Intel/Qwen3.5-122B-A10B-int4-AutoRound` repo ships a `tokenizer_config.json` that declares
`tokenizer_class: TokenizersBackend` — a newer/nonstandard class the box's (older) vLLM/transformers
does not know, so both the **server** and the **`vllm bench serve` client** crash at tokenizer init:
`ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported`. The
underlying vocab is a standard `Qwen2Tokenizer`. **Fix:** override the tokenizer to the base repo,
`--tokenizer Qwen/Qwen3.5-122B-A10B`, on **both** the serve command and the bench command (the bench
tool loads the tokenizer to build the random dataset, so it needs the flag independently). Same vocab
→ no correctness impact.

Two more Qwen3.5-122B bring-up gotchas on this single-Spark box:
- **All `spark-vllm-docker` recipes default to a 2-Spark cluster** (`tensor_parallel: 2` + Ray).
  Warpcore is ONE Spark, so a solo recipe with `tensor_parallel: 1` and no Ray backend is required.
  The `-tp` CLI shorthand collides with the launcher's `-t` (image-name) flag and mangles the command
  (`Container: p`, stray `1`), so set `tensor_parallel: 1` in the recipe YAML rather than via CLI.
- **`<think>` reasoning split is imperfect.** The `qwen3` reasoning parser + `unsloth.jinja` chat
  template leave `<think>…</think>` tags inline in `content` and `reasoning_content` empty. Output is
  correct; only the separation is off — fix the template/parser before trusting MCQ answer-extraction
  in lm-eval.
- **FP8 does not fit one Spark.** `Qwen/Qwen3.5-122B-A10B-FP8` is ~125 GiB of weights (> 128 GB
  unified memory). INT4 (62.65 GiB weights + 26.26 GiB KV, 256K context) is the only single-Spark fit.

---

## 13. GB10 has NO working CUTLASS W8A8 **FP8 dense/linear** kernel — `VLLM_TEST_FORCE_FP8_MARLIN=1`
`ornith-ai/Ornith-1.0-35B-FP8` is a **compressed-tensors W8A8 FP8** checkpoint. On the GB10 the engine
core dies during init with:
```
cutlass_gemm_caller ... Error Internal
RuntimeError: ... (EngineCore init failed)
```
This is **distinct from issue 8**. Issue 8 is the *MoE experts* CUTLASS path (fixed by
`--moe-backend marlin`); this one is the **dense / linear** FP8 GEMM
(`CompressedTensorsW8A8Fp8` → `CutlassFP8ScaledMMLinearKernel`), which `--moe-backend` does not touch.
An FP8 MoE model needs **both** fixes or it will not start:

```bash
-e VLLM_TEST_FORCE_FP8_MARLIN=1   # dense/linear FP8 -> MarlinFP8ScaledMMLinearKernel
--moe-backend marlin              # MoE experts -> Marlin
```

Despite the `VLLM_TEST_` prefix this is the production-correct setting on GB10 — Marlin is the only
FP8 GEMM path that works on this SM. Cost: a slower load (Marlin repack, ~19 s/shard, ~300 s total for
34.85 GiB), no correctness impact (GSM8K 97.19%, GPQA-Diamond 69.70%). See the
[Ornith-1.0-35B card](results/ornith-35b/README.md).

Related bring-up notes for the same model:
- **`--reasoning-parser qwen3` returns an EMPTY `reasoning_content`** while `content` is correctly
  CoT-stripped. Output is right; only the field split is off (same shape as the Qwen3.5 note above).
- **It is a reasoning model with a large thinking budget even on trivial prompts** (218 reasoning
  tokens to answer a one-word question). Always send a generous `max_tokens` — ≥4k for chat, 32k+ for
  GPQA/agentic — or answers get truncated to empty and silently score wrong.

---

## 14. GB10 is **sm_121** and FlashInfer ships no sm_121 cubin — `no kernel image is available`

Hit bringing up `poolside/Laguna-S-2.1-NVFP4` (vLLM `0.27.2rc1.dev193`,
`vllm/vllm-openai:cu129-nightly-aarch64`). Two separate problems stack here.

### 14a. The mixed-quantization `--moe-backend` catch-22

Laguna quantizes **layers 0–39 experts to NVFP4** but leaves **layers 40–47 experts BF16**
(`quantization_config.ignore` carries `re:^model\.layers\.4[0-7]\.mlp\.experts.*`). vLLM therefore
builds *two* MoE method objects with **disjoint** legal backend sets, while `--moe-backend` is a
single **global** flag:

```
--moe-backend marlin  -> ValueError: moe_backend='marlin' is not supported for unquantized MoE.
--moe-backend triton  -> ValueError: moe_backend='triton' is not supported for NvFP4 MoE.
```

| Layer group | vLLM method | Accepts | Rejects |
|---|---|---|---|
| 0–39 experts (NVFP4) | `CompressedTensorsW4A4Fp4MoE` | cutlass, flashinfer_\*, **marlin**, humming, emulation | triton |
| 40–47 experts (BF16) | `UnquantizedFusedMoEMethod` | **triton**, batched_triton, flashinfer_\*, aiter | marlin |

### 14b. Omitting the flag "works" at init, then dies at first decode

Dropping `--moe-backend` lets each oracle auto-select. Both pick FlashInfer, the engine initialises,
all 49 shards load — and then the **first decode** dies:

```
MemoryError: CUDA error: no kernel image is available for execution on the device
  ... flashinfer/fused_moe/core.py -> cutlass_fused_moe
```

**That `MemoryError` is NOT an OOM.** Don't tune `--gpu-memory-utilization`. The real cause:

```
torch.cuda.get_device_capability(0) -> (12, 1)          # GB10 = sm_121
torch.cuda.get_arch_list() -> [sm_80, sm_90, sm_100, sm_120]   # no sm_121
```

FlashInfer's CUTLASS/TRTLLM MoE kernels are arch-exact SASS with no PTX fallback, so sm_120 code will
not run on sm_121. vLLM's oracle hardcodes FlashInfer first for CUDA and demotes it only for SM90 /
`dp_size>1` / SWIGLUOAI — **it never checks whether a cubin exists for the running SM**
(`vllm.envs.VLLM_HAS_FLASHINFER_CUBIN` is already `False` here and is not consulted). Upstream applies
this reasoning in exactly one spot: NvFP4 auto-selection excludes `FLASHINFER_B12X` pending "the
upstream CUTLASS SM121 MMA op guard".

### The fix

Pass `--moe-backend marlin` (GB10-stable NVFP4 path, cf. issue 8) **and** alias marlin→TRITON for the
unquantized group via a `sitecustomize.py` on `PYTHONPATH`. It must be `sitecustomize` — a monkeypatch
in the parent process is lost in the spawned `EngineCore` child, which is where model construction runs.
The shim is gated on `get_device_capability(0) == (12, 1)`, so it is a no-op on any other GPU.

```bash
-v $HOME/vllm_patch:/patch -e PYTHONPATH=/patch   # + --moe-backend marlin
```

Confirm both lines appear at startup, and that neither says FlashInfer:
```
[nvfp4.py:244]       Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]
[unquantized.py:282] Using TRITON Unquantized MoE backend out of potential backends: [...]
```

Verified: smoke test OK, tool call `finish_reason: tool_calls`, and **60/60 concurrent guided-decoding
requests passed in 20.2 s** with the engine alive afterwards. Shim + launch script are archived in
[`results/laguna-s-2.1-118b/raw/`](results/laguna-s-2.1-118b/raw/).

**Generalises:** on GB10 prefer **MARLIN (quantized) / TRITON (unquantized)** and treat *any* FlashInfer
MoE selection as a latent first-decode crash. Always unwrap the real exception — every one of these
hides behind `RuntimeError: Engine core initialization failed ... Failed core proc(s): {}`:
```bash
docker logs <ctr> 2>&1 | sed -e 's/\r$//' | grep 'core.py:1346' | tail -20
```
(the `\r` strip is mandatory; a C++ `frame #N` stack means a kernel-level fault, not a config error).

---

## 15. **lm-eval silently scores 0 when vLLM puts the answer in `message.reasoning`** (repo-wide)
**Symptom:** a fraction of items in any lm-eval run return HTTP 200, `finish_reason: "stop"`, completion
tokens billed — and an **empty response** that scores zero. No errors, no retries, no exceptions, no
truncation. Rates observed in this repo: Laguna GSM8K **14.1%** (186/1319), Laguna IFEval **5.4%**,
Ornith GPQA-Diamond **21.2%** (42/198), Ornith IFEval **5.2%**, Ornith GSM8K 0.1%.

**Root cause — two halves that only bite together:**

1. **Server.** With `--reasoning-parser <x>`, vLLM may fail to initialize its reasoning token IDs:
   ```
   WARNING [vllm.py:1689] Auto-initialization of reasoning token IDs failed. Please check whether
   your reasoning parser has implemented the `reasoning_start_str` and `reasoning_end_str`.
   ```
   Without delimiters the parser cannot find where reasoning ends, so on some generations it classifies
   the **entire output** as reasoning and emits `content: null`, with the full answer in `reasoning`.
   Note the field is **`reasoning`**, not the conventional `reasoning_content` — probes that check only
   `reasoning_content` see nothing and wrongly conclude the output vanished.

2. **Client.** lm-eval's `ChatCompletions` parser is one line, with no notion of `reasoning`:
   ```python
   tmp[choices["index"]] = choices["message"]["content"]   # null -> "" -> scores 0
   ```

**Proof:** for a prompt that returns `content: null` on `/v1/chat/completions`, the raw
`/v1/completions` endpoint returns 1579 chars of normal text. Running the **full lm-eval path** on 200
GSM8K questions reproduced 11.5% empty; hand-rolled direct API calls reproduced **0%** — localizing the
bug to chat-completions response handling, not the engine. Falsified along the way: truncation,
request errors, prompt-specific failure, concurrency/load (0/64 empty at c=32), and lm-eval's `until`
stop strings (60% still empty with stop strings removed).

**Fix / detection:** re-serve the affected items and read `reasoning` when `content` is null. Detect
with `raw/quality/audit_empty_responses.py`, recover with
`raw/quality/recover_empties_via_reasoning_field.py` (both under `results/laguna-s-2.1-118b/`).

**Impact on published numbers — DO NOT ignore:**

| Model / task | Published | Served-item rate | Empty | Status |
| --- | ---: | ---: | ---: | --- |
| Laguna GSM8K | 83.40% | 97.09% | 14.1% | **corrected to 96.13%** (all 186 re-served & graded) |
| Ornith GPQA-Diamond | **69.70%** | **88.46%** | 21.2% | **suspect — needs re-serve** |
| Ornith IFEval prompt-strict | 85.58% | 90.25% | 5.2% | floor |
| Ornith IFEval inst-strict | 88.39% | 93.21% | 5.2% | floor |
| Laguna IFEval | 75.79% / 81.41% | — | 5.4% | floor |
| Ornith GSM8K | 97.19% | 97.27% | 0.1% | effectively unaffected |

Ornith's GPQA is the serious one: **21.2% of items scored zero without being answered**, so 69.70% is
an underestimate of unknown size. The served-item rate (88.46%) is **not** a substitute — on Laguna the
recovered items scored 90.3% vs 97.09% for served ones, proving the defect does **not** drop items
uniformly at random, so exclusion-based estimates are optimistically biased. Only a re-serve gives a
defensible number.

**Cross-model caveat:** any score in this repo taken through lm-eval against a vLLM endpoint with a
reasoning parser is suspect until audited. Runs whose sample files were not retained cannot be checked.

---

## Non-issues (ruled out — don't chase)
- The GB10 `nvidia-smi` memory `N/A` is normal (unified memory), not a broken GPU.
- The `fatal: not a git repository` line lm-eval prints at the end is harmless (it tries to record a
  git hash of the cwd).
