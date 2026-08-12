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

## Non-issues (ruled out — don't chase)
- The GB10 `nvidia-smi` memory `N/A` is normal (unified memory), not a broken GPU.
- The `fatal: not a git repository` line lm-eval prints at the end is harmless (it tries to record a
  git hash of the cwd).
