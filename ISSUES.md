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

## Non-issues (ruled out — don't chase)
- The GB10 `nvidia-smi` memory `N/A` is normal (unified memory), not a broken GPU.
- The `fatal: not a git repository` line lm-eval prints at the end is harmless (it tries to record a
  git hash of the cwd).
