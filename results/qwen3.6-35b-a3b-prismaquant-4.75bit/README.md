# rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm — Warpcore Benchmark Card

- **Campaign dates:** 2026-10-01 → 2026-10-03
- **Model:** `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`
- **Pinned model revision:** `e347d86b2a6cba4b54ea6f87ca247f60439eed07`
- **Host:** Warpcore, NVIDIA DGX Spark / GB10
- **Serving image:** `eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`
- **vLLM:** `0.29.1rc1.dev427`
- **Suite:** `warpcore-v2`

## Summary

All four canonical cells passed the repository's publication validator over their retained full-denominator evidence.

- **GSM8K:** **96.51%** anchored answer-line exact match (1273/1319). The flexible-fallback diagnostic was **96.44%** (1272/1319). All 1,319 responses were nonempty and ended normally.
- **IFEval:** **86.69%** prompt-level strict (469/541), **89.09%** prompt-level loose, **89.69%** instruction-level strict, and **91.25%** instruction-level loose. Twenty-six prompts exhausted the frozen 65,536-token output ceiling without visible final content and remain failures in the denominator.
- **GPQA-Diamond:** **70.71%** anchored answer-line exact match (140/198). Thirty-seven items exhausted the frozen 65,536-token output ceiling without visible final content and remain failures in the denominator.
- **SWE-bench Verified:** **61/100 resolved (61.0%)** on the frozen seed-42 n=100 inventory. The model submitted 79 nonempty patches; 18 were unresolved and 21 instances produced no patch. Official grading had zero errors and zero incomplete instances.
- **Throughput:** **802.07 output tok/s at concurrency 128**, the highest tested point. Throughput was still rising, so this is a measured floor rather than a proven plateau.

## Canonical quality

| Benchmark | n | Canonical metric | Result | Dispositions |
| --- | ---: | --- | ---: | --- |
| GSM8K clean, zero-shot CoT | 1319 | anchored answer-line exact match | **96.51%** (1273/1319; 95% Wilson CI 95.38%–97.38%) | 1273 correct, 46 wrong; 0 empty or budget-exhausted |
| IFEval | 541 | prompt-level strict accuracy | **86.69%** (469/541; 95% Wilson CI 83.57%–89.30%) | 469 strict-correct, 46 nonempty strict-wrong, 26 budget-exhausted |
| GPQA-Diamond clean, zero-shot CoT | 198 | anchored answer-line exact match | **70.71%** (140/198; 95% Wilson CI 64.02%–76.60%) | 140 correct, 21 nonempty wrong, 37 budget-exhausted |

All quality runs used lm-evaluation-harness 0.4.12, zero-shot prompts, greedy decoding (`temperature=0`, `do_sample=false`), no automatic retries, and request-bound sidecar capture of finish reasons and full response fields. GSM8K used an 8,192-token output ceiling; IFEval and GPQA-Diamond used 65,536 tokens. Empty budget-limited responses are counted as failures, never excluded or replaced by served-only rates.

Canonical evidence:

- [GSM8K run](runs/warpcore-v2/gsm8k/qwen36-prisma-gsm8k-20261002/)
- [IFEval run](runs/warpcore-v2/ifeval/qwen36-prisma-ifeval-20261002/)
- [GPQA-Diamond run](runs/warpcore-v2/gpqa_diamond/qwen36-prisma-gpqa-20261001/)

## SWE-bench Verified — `warpcore-v2` frozen n=100

The campaign directly ran the fixed seed-42 100-instance inventory after mandatory production endpoint, model-identity, container-cache, and native-tool-call preflight. mini-swe-agent 2.4.6 ran with four workers on the x86_64 Mac mini; Warpcore served only the model. The official grader used SWE-bench 4.1.0.

- **Resolved:** 61/100 (**61.0%**, full assigned denominator; 95% Wilson CI **51.2%–70.0%**)
- **Nonempty submissions:** 79/100
- **Submitted but unresolved:** 18
- **Empty patches / non-submissions:** 21
- **Generation dispositions:** 79 `Submitted`, 13 `LimitsExceeded`, 7 `ContextWindowExceededError`, 1 `RepeatedFormatError`
- **Official grading errors:** 0
- **Incomplete instances:** 0

The conditional submitted-patch resolution rate is **61/79 = 77.2%**. It is diagnostic only; it is not the headline because all 100 assigned instances remain in the canonical denominator.

Canonical evidence is under [the SWE-bench run directory](runs/warpcore-v2/swebench/qwen36-prisma-swebench-n100-20261002/), including predictions, exit statuses, all 100 normalized trajectories, a compressed archive of the byte-identical source trajectories, official grading, per-instance grader logs, manifest, lifecycle history, invocation, and completion sentinel.

## Throughput and latency

`vllm bench serve`, on-box against `localhost:8000`, raw-completions path, fixed 512-input/256-output-token requests, `--ignore-eos`, and `temperature=0`:

| Concurrency | Requests | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Mean ITL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 72.14 | 153 ms | 163 ms | 13.19 ms | 41.87 ms |
| 2 | 32 | 129.77 | 207 ms | 223 ms | 13.29 ms | 47.44 ms |
| 4 | 32 | 199.33 | 239 ms | 350 ms | 16.28 ms | 59.48 ms |
| 8 | 32 | 290.18 | 301 ms | 717 ms | 23.43 ms | 80.07 ms |
| 16 | 48 | 412.28 | 431 ms | 1.35 s | 30.63 ms | 109.66 ms |
| 32 | 96 | 567.43 | 640 ms | 2.94 s | 46.14 ms | 160.46 ms |
| 64 | 192 | 694.66 | 1.03 s | 6.66 s | 74.28 ms | 262.63 ms |
| 96 | 288 | 762.64 | 1.35 s | 11.43 s | 100.18 ms | 349.79 ms |
| 128 | 384 | **802.07** | 1.64 s | 16.56 s | 126.63 ms | 437.86 ms |

All **1,136/1,136** requests succeeded. Concurrency 96→128 gained **5.17%** throughput. Because the live server was configured with `--max-num-seqs 128`, this sweep does **not** establish a plateau above 128; 802.07 tok/s is the maximum measured rate and a floor on the deployment's possible saturation throughput. The raw [sweep log](raw/throughput_sweep/sweep.log) and [exact sweep script](raw/throughput_sweep/vllm_sweep.sh) are retained.

## Reproducibility

The canonical serving adapter is [`../../adapters/qwen3.6-35b-a3b-prismaquant-4.75bit.yaml`](../../adapters/qwen3.6-35b-a3b-prismaquant-4.75bit.yaml). It binds every run to the exact checkpoint revision, serving image digest, vLLM version, mixed NVFP4/MXFP8/BF16 quantization, Qwen3 reasoning and tool parsers, 262,144-token context, `gpu-memory-utilization=0.80`, and `max-num-seqs=128`.

The retained [launch script](raw/launch_qwen36_prisma.sh) additionally records prefix caching, three-token MTP speculative decoding, cache mounts, and the exact `docker run` invocation. Its SHA-256 is `f835d7326229492e3244859b67e763c690d4bad53b1285a4beb2f2f31d8f4757`, matching the serving-host copy captured for publication.

The adapter hash embedded in all four manifests is `945a99886a706ccc1c51e08039022c1bf23ff867a458454874bc91966ce23199`; all runs also bind the frozen suite/task or SWE-bench inventory hashes. The canonical matrix is generated by `viz/publish_campaign.py` only from `validated + current` runs after the authoritative publication validator passes.

### Exact commands

The per-run `command.txt` files retain the complete quality and SWE-bench invocations. The throughput command is retained in [`raw/throughput_sweep/vllm_sweep.sh`](raw/throughput_sweep/vllm_sweep.sh). To reproduce the serving profile and validate a retained run:

```bash
bash results/qwen3.6-35b-a3b-prismaquant-4.75bit/raw/launch_qwen36_prisma.sh

python3 viz/validate_campaign.py \
  --suite suite/warpcore-v2.yaml \
  --adapter adapters/qwen3.6-35b-a3b-prismaquant-4.75bit.yaml \
  --for-publication \
  results/qwen3.6-35b-a3b-prismaquant-4.75bit/runs/warpcore-v2/<benchmark>/<run-id>

python3 viz/publish_campaign.py --suite-id warpcore-v2 .
```
