# GPT-OSS tool-call corruption boundary replay

Date: 2026-09-21
Instance: `django__django-13820`

## Result

The malformed JSON is present in the model's generated token sequence before
Chat Completions serialization, SSE assembly, LiteLLM conversion, or
mini-swe-agent parsing. The server's unconstrained `tool_choice=auto` path
allowed the model to emit an array-style closing sequence (`"]}`) for the
object-only `bash` schema.

The smallest verified serving repair is to enable vLLM's existing parameter
schema constraint for every tool call:

```text
--tool-strict-level parameter
```

This changes only serving-side constrained decoding. It does not modify the
frozen SWE-bench prompt, tools, retries, limits, agent parser, or grader.

## Reproduction

Source campaign evidence:

- archive: `results/gpt-oss-120b/runs/warpcore-v1/swebench/gptoss-swebench-n100-20260921/raw/trajectories.tar.gz`
- member: `django__django-13820/django__django-13820.traj.json`
- first malformed retained response: message index 6
- extracted request SHA-256: `cc3e6c0c03a14c5b27c530db65b10aca6f6ca7c6c0afd6ce13119b1707006d17`
- retained malformed arguments: 2,051 characters, JSON error at character 2,049

The exact retained conversation prefix and production request parameters were
replayed at temperature 0 against:

- image: `eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`
- local image ID: `sha256:fb47fdf2a242ea97c5b0ae0cebc18d44fc53ab0e7bf2055cb121c00b559ae3b4`
- vLLM: `0.29.1rc1.dev427+g0748d3bd5.d20260920`
- checkpoint: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
- parsers: `openai_gptoss` reasoning and `openai` tool calling

## Boundary observations

- Direct non-streaming Chat Completions: malformed, 1,866-character arguments.
- Direct streaming Chat Completions plus deterministic SSE reassembly:
  malformed, 2,476-character arguments.
- Chat Completions with `parallel_tool_calls=false`: still malformed.
- Responses API: supported by this server and malformed on the replay.
- LiteLLM conversion: malformed; it preserved the server response rather than
  introducing the defect.
- Raw completion using the exact rendered prompt token IDs: generated the same
  807 output token IDs as Chat Completions with token IDs enabled.
- Decoding those token IDs with `openai_harmony` produced the malformed tool
  payload ending in `PATCH\"]}`. Therefore the invalid representation exists
  before the Harmony parser constructs OpenAI `tool_calls`.
- Non-streaming and streaming token-ID captures were byte-for-byte identical
  for the token-capture replay (807 IDs, SHA-256
  `455c058d622496a268d5552025f3cbbb16daa4d1c658e14e444d0ad680293aee`).
  Their decoded Harmony text was also identical (SHA-256
  `8d74df70111cca781f99ac157e357f9aab56d4a9ec7f31c10dcbaa91faa44cac`).
  Streaming assembly is therefore not the source.

The replay is deterministic for a given request form, but separate requests can
produce different wording and argument lengths. Every unconstrained replay
above ended with the same invalid array-style closure.

## Why the repair works

In this vLLM build, `HarmonyParser.parse()` consumes engine output token IDs and
passes the completed Harmony message text through unchanged when it is invalid
JSON. With `tool_choice=auto` and a tool lacking `strict: true`, vLLM's
`ToolStrictLevel.AUTO` intentionally applies no structural tag. The malformed
model emission therefore reaches clients unchanged.

Marking the tool `strict: true` made the same replay valid. Setting the server
floor to `--tool-strict-level parameter` applies the same parameter-schema
constraint without changing mini-swe-agent's tool declaration.

## Post-restart checks

The live launch script was changed on Warpcore to include
`--tool-strict-level parameter` and the same immutable image/checkpoint was
restarted. Verified after readiness:

- ordinary content probe: `STRICT_OK`, `finish_reason=stop`;
- simple bash tool call: valid JSON, `finish_reason=tool_calls`;
- retained multi-turn failure replay: valid 2,432-character bash arguments;
- strict-schema concurrency stress: 60/60 valid tool calls in 11.1 seconds,
  followed by a successful engine-liveness probe.

This establishes the repair for the deterministic trigger. It does not replace
the required production-path smoke, stress test, official grading, or n=20
qualification.
