# Warpcore Benchmarks

Benchmark results for LLMs served on **Warpcore** — an NVIDIA DGX Spark (GB10) host at ALCF —
running vLLM. This repo aggregates quality and (planned) throughput benchmarks on a **per-model**
basis, and documents the serving issues encountered on the hardware and how they were fixed.

## Models

| Model | Serving | GSM8K | IFEval (prompt-strict) | GPQA-Diamond | pi-30 | SWE-bench Verified | Peak tok/s | Full card |
| ----- | ------- | ----- | ---------------------- | ------------ | ----- | ------------------ | ---------- | --------- |
| [openai/gpt-oss-120b](results/gpt-oss-120b/README.md) | vLLM MXFP4 | 83.7% | 83.7% | 72.7% | 30/30 | blocked² | ~709 (c≈256) | [card](results/gpt-oss-120b/README.md) |
| [nvidia/Nemotron-3-Super-120B-A12B-NVFP4](results/nemotron-3-super-120b/README.md) | vLLM NVFP4 (MARLIN) | 76.65% | 85.40% | 63.64% | 30/30 | — | ~190 (c≈128) | [card](results/nemotron-3-super-120b/README.md) |
| [Qwen/Qwen3.6-35B-A3B-FP8](results/qwen3.6-35b-a3b/README.md) | vLLM FP8 (TRITON) | 97.04% | 84.84% | 82.32% | 29/30 | 44% (n=100)¹ | ~487 (c=128) | [card](results/qwen3.6-35b-a3b/README.md) |

¹ SWE-bench Verified, **representative random 100-instance sample** (11 repos), not the full 500 — indicative score ±~5%. See the [Qwen3.6 card](results/qwen3.6-35b-a3b/README.md#agentic-coding-swe-bench-verified) for the per-repo breakdown and caveats.

² gpt-oss-120b's SWE-bench run is **blocked by a vLLM serving bug**, not a model limitation — the model was actively and correctly solving instances (median 12 successful shell commands per instance) when vLLM's tool-call parser corrupted the JSON arguments and aborted the run (79/100 instances). No trustworthy score can be produced on this serving stack. See the [gpt-oss-120b card](results/gpt-oss-120b/README.md#agentic-coding-swe-bench-verified-blocked) for the full diagnosis.

## What's measured

- **Quality** — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  (GSM8K, IFEval, GPQA-Diamond) run against the live vLLM OpenAI chat endpoint, plus
  [Zapier AutomationBench](https://github.com/zapier/AutomationBench) for agentic business workflows.
- **Throughput** — `vllm bench serve` concurrency sweep (output tok/s, TTFT, ITL/TPOT), climbed until
  throughput plateaus. See each card's Throughput section. (Note: the quality harnesses above record
  no tok/s — throughput is a separate on-box measurement.)
- **Agentic coding** — [pi-30](https://github.com/rick-stevens-ai/pi-30) (30 agentic-coding problems, verifier-graded)
  and [SWE-bench Verified](https://www.swebench.com/) via
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (bash-only agent loop; the agent + x86 test
  containers run on a client Mac, the model is served on Warpcore). SWE-bench is reported on a representative
  random 100-instance sample (not the full 500). Qwen3.6 has a trustworthy number; gpt-oss-120b's SWE-bench
  run is blocked by a vLLM tool-call serving bug (see its card).

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
