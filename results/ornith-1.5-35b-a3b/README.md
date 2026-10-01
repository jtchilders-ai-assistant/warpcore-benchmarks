# ornith-ai/Ornith-1.5-35B-A3B-FP8 — Warpcore Benchmark Card

- **Campaign dates:** 2026-09-22 → 2026-10-01
- **Model:** `ornith-ai/Ornith-1.5-35B-A3B-FP8`
- **Pinned model revision:** `fab11c26e2325a42f4b32da0249c819a0bade1b1`
- **Host:** Warpcore, NVIDIA DGX Spark / GB10
- **Serving image:** `eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`
- **vLLM:** `0.29.1rc1.dev427+g0748d3bd5.d20260920`
- **Harnesses:** `warpcore-v1` quality/throughput and `warpcore-v2` SWE-bench

## Summary

- **Serving qualification passed:** ordinary chat, native tool use, and a 60-request strict-schema stress test completed successfully.
- **GSM8K validated:** **88.02%** canonical anchored answer-line exact match (1161/1319), with flexible-fallback diagnostic **97.04%**. Three responses (0.23%) exhausted the 8,192-token generation ceiling and are counted as failures.
- **SWE-bench Verified:** **59/100 resolved (59.0%)** on the frozen seed-42 n=100 inventory. Submission reliability was **69/100 nonempty patches**; 29 instances hit the step limit and 2 ended in `RepeatedFormatError`. Official grading had zero errors and zero incomplete instances.
- **GPQA-Diamond is invalid/nonpublishable:** 35/198 responses exhausted the 65,536-token ceiling without a final answer. The diagnostic aggregate was 77.27% anchored answer-line, but it is not promoted as a canonical score.
- **IFEval was not measured** in this campaign.

## Canonical quality

| Benchmark | n | Canonical metric | Result | Dispositions |
| --- | ---: | --- | ---: | --- |
| GSM8K clean, zero-shot CoT | 1319 | anchored answer-line exact match | **88.02%** (±0.89) | 1161 correct, 155 wrong, 3 budget-truncated |
| IFEval | — | prompt-level strict | **not measured** | Not launched |
| GPQA-Diamond | 198 | anchored answer-line exact match | **invalid / not published** | 153 correct, 10 wrong, 35 budget-truncated |

GSM8K used greedy decoding (`temperature=0`), concurrency 8, and an 8,192-token output ceiling. The exact artifacts are under [`runs/warpcore-v1/gsm8k/run-2026-09-24T18-26-28/`](runs/warpcore-v1/gsm8k/run-2026-09-24T18-26-28/). The flexible fallback is retained as a diagnostic because it accepts an unanchored integer anywhere in the response; it is not the suite's headline metric.

The GPQA evidence is retained under [`runs/warpcore-v1/gpqa_diamond/run-2026-09-23T01-48-07/`](runs/warpcore-v1/gpqa_diamond/run-2026-09-23T01-48-07/). Its failed/invalid lifecycle must not be interpreted as a score of record.

## SWE-bench Verified — `warpcore-v2` frozen n=100

The canonical v2 campaign used the fixed seed-42 100-instance inventory and four mini-swe-agent 2.4.6 workers. Correctness and submission reliability are deliberately separate:

- **Resolved:** 59/100 (**59.0%**, full assigned denominator; 95% Wilson interval **49.2%–68.1%**)
- **Nonempty submissions:** 69/100
- **Unresolved nonempty patches:** 10
- **Empty patches / non-submissions:** 31
- **Generation dispositions among non-submissions:** 29 `LimitsExceeded`, 2 `RepeatedFormatError`
- **Official grading errors:** 0
- **Incomplete instances:** 0

The first grading invocation failed after all 100 predictions and trajectories had been retained because the runner used `sys.executable`, whose Python environment lacked `swebench`. The official grader was rerun against the **unchanged** `raw/preds.json` with the pinned campaign interpreter and SWE-bench 4.1.0. No inference was replayed. `status.json` preserves the original failure and records an explicit recovery transition; [`RECOVERY.md`](runs/warpcore-v2/swebench/ornith15-swebench-n100-20260929/RECOVERY.md) gives the full procedure and evidence map.

Canonical evidence is under [`runs/warpcore-v2/swebench/ornith15-swebench-n100-20260929/`](runs/warpcore-v2/swebench/ornith15-swebench-n100-20260929/). The earlier v1 n=20 qualification remains retained separately as diagnostic history; v2's launch policy directly executes the frozen n=100 campaign after mandatory production preflight, so the obsolete v1 gate does not govern this score.

## Throughput and latency

`vllm bench serve`, on-box, fixed 512-input/256-output-token requests with `--ignore-eos`:

| Concurrency | Output tok/s | Median TTFT | Mean TPOT |
| ---: | ---: | ---: | ---: |
| 1 | 37.34 | 159 ms | 26.26 ms |
| 8 | 153.72 | 626 ms | 49.67 ms |
| 64 | 379.63 | 1.14 s | 160.31 ms |
| 128 | 491.12 | 1.39 s | 244.56 ms |
| 192 | 478.22 | 11.72 s | 243.27 ms |
| 256 | 491.15 | 67.12 s | 251.81 ms |
| 384 | **492.89** | 133.86 s | 252.96 ms |

Throughput plateaus around concurrency 128. Higher concurrency adds essentially no throughput while sharply increasing time to first token. Concurrency 8 was therefore used for quality evaluation. Raw sweep and serving evidence are retained under [`raw/`](raw/).

## Reproducibility and interpretation

The serving adapter is [`../../adapters/ornith-1.5-35b-a3b.yaml`](../../adapters/ornith-1.5-35b-a3b.yaml). It records the exact checkpoint revision, serving image digest, parser settings, context limit, memory utilization, and sequence capacity.

This card distinguishes four states deliberately:

1. **Validated measurement:** GSM8K.
2. **Validated `warpcore-v2` measurement:** SWE-bench Verified, including the preserved grading-recovery history.
3. **Not measured:** IFEval.
4. **Completed but scientifically invalid diagnostic:** GPQA-Diamond.

Neither an old failed qualification nor a completed harness process is silently promoted into a capability score. The SWE-bench value is published only from the complete retained n=100 evidence and official grader partitions.
