# NVIDIA Qwen3.8-Flash-Next-NVFP4 — Warpcore Benchmark Card

- **Campaign dates:** 2026-10-08 → 2026-10-10
- **Served model ID:** `qwen3.8-flash-next`
- **Checkpoint:** `nvidia/Qwen3.8-Flash-Next-NVFP4`
- **Pinned checkpoint revision:** `fc694b54fb0174e0913e6adf86691ef85a4ead47`
- **Host:** Warpcore, NVIDIA DGX Spark / GB10
- **Serving image ID:** `sha256:ce9fb15b8b64aa4facc84d6fdd8691d57a5bf8a71b9d4b51aeac2148ca1b9fb2`
- **vLLM:** `0.31.0`
- **Suite:** `warpcore-v2`

## Summary

All four canonical benchmark cells passed the repository's publication validator over retained full-denominator evidence.

- **GSM8K:** **97.73%** anchored answer-line exact match (1289/1319). Four responses ended at the 8,192-token ceiling; retained parser recovery restored three responses whose answer text had been captured outside the initially parsed field. All residual failures remain in the denominator.
- **IFEval:** **89.09%** prompt-level strict (482/541), with 18 responses ending at the frozen 65,536-token ceiling and remaining wrong.
- **GPQA-Diamond:** **87.88%** anchored answer-line exact match (174/198). Eight responses ended at the 65,536-token ceiling; seven had empty visible content. All eight remain wrong in the denominator.
- **SWE-bench Verified:** **97/100 resolved (97.0%)** on the frozen seed-42 n=100 inventory. The model submitted a nonempty patch for every assigned instance; three patches were unresolved and official grading had zero errors or incomplete instances.
- **Throughput:** **129.93 output tok/s at concurrency 64**. The measured plateau spans c=32–256 (125.17–129.93 tok/s); the first point within 95% of peak is c=32.

## Canonical quality

- **GSM8K:** 1289/1319 = **97.73%**; 95% Wilson CI **96.77%–98.40%**.
- **IFEval prompt-strict:** 482/541 = **89.09%**; 95% Wilson CI **86.19%–91.45%**.
- **GPQA-Diamond:** 174/198 = **87.88%**; 95% Wilson CI **82.60%–91.72%**.

All quality runs used lm-evaluation-harness 0.4.12, zero-shot prompts, greedy decoding (`temperature=0`, `do_sample=false`), no automatic retries, and request-bound response metadata. GSM8K used an 8,192-token output ceiling; IFEval and GPQA-Diamond used 65,536 tokens. Budget-limited responses are counted as failures, never excluded or replaced by served-only rates.

Canonical evidence:

- [GSM8K run](runs/warpcore-v2/gsm8k/qwen38-gsm8k-20261009/)
- [IFEval run](runs/warpcore-v2/ifeval/qwen38-ifeval-20261009/)
- [GPQA-Diamond run](runs/warpcore-v2/gpqa_diamond/qwen38-gpqa-diamond-20261009/)

## SWE-bench Verified — frozen seed-42 n=100

mini-swe-agent 2.4.6 ran the frozen 250-step native-tool-calling scaffold with four workers on the x86_64 Mac mini. Warpcore served inference only. The official grader used SWE-bench 4.1.0 and cached x86 evaluation images.

- **Resolved:** 97/100 (**97.0%**; full assigned denominator; 95% Wilson CI **91.55%–98.97%**)
- **Nonempty submissions:** 100/100
- **Submitted but unresolved:** 3
- **Empty patches / non-submissions:** 0
- **Official grading errors:** 0
- **Incomplete instances:** 0
- **Unresolved:** `astropy__astropy-8872`, `pydata__xarray-4075`, `sphinx-doc__sphinx-8595`

Because every instance produced a nonempty patch, the conditional submitted-patch rate is also 97/100; the canonical headline nevertheless remains resolved divided by all 100 assigned instances.

The [SWE-bench run directory](runs/warpcore-v2/swebench/qwen38-swebench-n100-20261010/) retains all 100 predictions, source and normalized trajectories, exit statuses, official grading, per-instance grader logs, manifest, lifecycle history, invocation, and completion sentinel.

## Throughput and latency

The retained sweep used `vllm bench serve` on-box against `/v1/completions`, fixed 512-input/256-output-token random requests, `--ignore-eos`, and `temperature=0`. All **1,904/1,904** requests in the validated summary succeeded.

- c=1: 24.95 output tok/s
- c=2: 38.53
- c=4: 56.25
- c=8: 80.28
- c=16: 107.80
- c=32: 126.06
- c=64: **129.93**
- c=96: 128.51
- c=128: 127.38
- c=192: 125.17
- c=256: 127.09

This is a measured throughput plateau, not an interactive recommendation: median TTFT was 81.98 s at c=64 and queueing rose further at higher concurrency. Concurrency 4 was therefore used for the quality and SWE-bench campaigns. The [validated summary, sweep log, and runner](raw/throughput_sweep/) retain the exact measurement and serving identities. The c=1 point is retained from the validated pre-restart run; c=2–256 are the clean continuation after recovery from a separately preserved failed c=2 attempt.

## Reproducibility

The canonical [adapter](../../adapters/qwen3.8-flash-next.yaml) binds the checkpoint revision, serving image ID, vLLM version, mixed NVFP4 quantization, Qwen3 reasoning and tool parsers, tokenizer path, 262,144-token context, `gpu-memory-utilization=0.80`, and `max-num-seqs=256`. Each campaign run retains its exact command, manifest, immutable suite-input hashes, per-item evidence, response metadata, status history, and `DONE` sentinel.

To revalidate a retained run:

```bash
python3 viz/validate_campaign.py \
  --suite suite/warpcore-v2.yaml \
  --adapter adapters/qwen3.8-flash-next.yaml \
  --for-publication \
  results/qwen3.8-flash-next/runs/warpcore-v2/<benchmark>/<run-id>
```
