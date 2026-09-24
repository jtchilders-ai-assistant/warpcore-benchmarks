# ornith-ai/Ornith-1.5-35B-A3B-FP8 — Warpcore Benchmark Card

- **Campaign date:** 2026-09-22 → 2026-09-24
- **Model:** `ornith-ai/Ornith-1.5-35B-A3B-FP8`
- **Pinned model revision:** `fab11c26e2325a42f4b32da0249c819a0bade1b1`
- **Host:** Warpcore, NVIDIA DGX Spark / GB10
- **Serving image:** `eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`
- **vLLM:** `0.29.1rc1.dev427+g0748d3bd5.d20260920`
- **Harness:** `warpcore-v1`, campaign branch commit `b4874b38b567dfa73409f512b417cf4ee699a91b`

## Summary

- **Serving qualification passed:** ordinary chat, native tool use, and a 60-request strict-schema stress test completed successfully.
- **GSM8K validated:** **88.02%** canonical anchored answer-line exact match (1161/1319), with flexible-fallback diagnostic **97.04%**. Three responses (0.23%) exhausted the 8,192-token generation ceiling and are counted as failures.
- **SWE-bench is not measured:** the required n=20 launch qualification failed, so the canonical n=100 run was not started. At termination, 11 instances had submitted nonempty patches, five had exhausted the 250-step limit, one had a 1,800-second transport timeout, and three had not completed. The qualification was not graded and is not a SWE-bench capability score.
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

## SWE-bench qualification — failed; n=100 not run

The frozen suite requires a fresh 20-instance production qualification with 20/20 clean submissions before authorizing the canonical n=100 run. This model did not pass that gate.

Observed terminal evidence before fail-closed termination:

- **11 `Submitted`**, each with a nonempty patch
- **5 `LimitsExceeded`**, each consuming the full configured 250 API calls without submitting a patch
- **1 `Timeout`**: `matplotlib__matplotlib-25332`, a LiteLLM/HTTP read timeout after 1,800 seconds
- **3 incomplete** when 20/20 success had become impossible

The five step-limit failures were:

- `astropy__astropy-14096`
- `psf__requests-6028`
- `pylint-dev__pylint-4551`
- `pylint-dev__pylint-6903`
- `pydata__xarray-6938`

This result supports a narrow conclusion: **under the frozen mini-swe-agent scaffold and 250-step budget, Ornith-1.5 was not reliable enough to proceed to n=100.** It does not establish a SWE-bench resolution rate because the qualification was stopped early and no official grading run was performed. Eleven successful nonempty submissions and functioning native tool calls argue against blanket harness incompatibility; the dominant observed failure was model/agent non-convergence, plus one infrastructure timeout.

Retained qualification evidence is under [`qualification/warpcore-v1/swebench/run/`](qualification/warpcore-v1/swebench/run/). No `qualification.json` was sealed.

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

This card distinguishes three states deliberately:

1. **Validated measurement:** GSM8K.
2. **Failed qualification / not measured:** SWE-bench.
3. **Completed but scientifically invalid diagnostic:** GPQA-Diamond.

Neither a failed qualification nor a completed harness process is silently promoted into a capability score.
