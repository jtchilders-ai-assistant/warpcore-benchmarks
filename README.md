# Warpcore Benchmarks

Benchmark results for LLMs served on **Warpcore** — an NVIDIA DGX Spark (GB10) host at ALCF —
running vLLM. This repo aggregates quality and (planned) throughput benchmarks on a **per-model**
basis, and documents the serving issues encountered on the hardware and how they were fixed.

## Models

| Model | Serving | GSM8K | IFEval (prompt-strict) | GPQA-Diamond | Peak tok/s | Full card |
| ----- | ------- | ----- | ---------------------- | ------------ | ---------- | --------- |
| [openai/gpt-oss-120b](results/gpt-oss-120b/README.md) | vLLM MXFP4 | 83.7% | 83.7% | 72.7% | ~709 (c≈256) | [card](results/gpt-oss-120b/README.md) |

## What's measured

- **Quality** — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  (GSM8K, IFEval, GPQA-Diamond) run against the live vLLM OpenAI chat endpoint, plus
  [Zapier AutomationBench](https://github.com/zapier/AutomationBench) for agentic business workflows.
- **Throughput** — `vllm bench serve` concurrency sweep (output tok/s, TTFT, ITL/TPOT), climbed until
  throughput plateaus. See each card's Throughput section. (Note: the quality harnesses above record
  no tok/s — throughput is a separate on-box measurement.)

## Hardware

See [HARDWARE.md](HARDWARE.md) for the Warpcore (DGX Spark / GB10) specs.

## Serving issues & fixes

See [ISSUES.md](ISSUES.md) for every problem hit while benchmarking on Warpcore and how it was resolved
(chat-vs-completions endpoint, MCQ answer-parse artifacts, scorer crashes on empty reasoning-model
output, server disconnects on very long generations, etc.).

## Layout

```
results/<model>/
  README.md          # the model's benchmark card
  raw/               # raw harness output (results JSON, custom task configs)
HARDWARE.md          # host specs
ISSUES.md            # serving issues encountered + fixes
```

## Reproducing

Each model card lists exact harness versions, endpoints, and flags. The lm-eval pipeline (chat endpoint,
custom clean-extraction tasks, all pitfall fixes) is captured for the maintainer in a Hermes skill
`lm-eval-vllm-endpoint`; the commands are reproduced in each card.
