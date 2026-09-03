"""Refuse to launch a benchmark against an endpoint that silently drops answers.

ISSUES #15: with `--reasoning-parser`, vLLM can return `content: null` while the
real answer sits in `message.reasoning`. lm-eval reads only `content`, so the
item scores 0 with no error, no warning and no retry. Laguna's GPQA published
53.03% when the served-only rate was 90.52%; 82 of 198 items were empty.

The warning that predicts this was ALREADY in the run log:

    WARNING [vllm.py:1689] Auto-initialization of reasoning token IDs failed.

Nothing was watching for it. This script watches, in ~30 s, before the GPU-hours
are spent.

TWO FAILURES THAT LOOK IDENTICAL AND ARE NOT
--------------------------------------------
Empty content alone proves nothing. Measured on a live, HEALTHY endpoint
(RedHatAI/Muse-Glimmer-30B-NVFP4, 2026-09-03):

    max_tokens=64    finish=length  content=None  -> budget starvation
    max_tokens=512   finish=stop    content='4'   -> same endpoint, fine

That model needs ~109 completion tokens before it emits any answer, so a naive
"empty content = broken parser" check CONDEMNS A WORKING ENDPOINT. The two are
separated by finish_reason:

    finish_reason=length + empty content            -> BUDGET (raise max_gen_toks)
    finish_reason=stop   + empty + reasoning filled -> PARSER (ISSUES #15)
    finish_reason=stop   + empty + nothing anywhere -> EMPTY  (genuinely no output)

Exit codes are distinct so a runner can branch on them:
    0  every probe returned usable content
    1  DEFECT: answers are being dropped (do not launch)
    2  INCONCLUSIVE: endpoint unreachable / HTTP error (do not launch either)

Exit 2 is not success. An unprobeable endpoint must never read as "passed".

    preflight_serving.py --endpoint http://host:8000 --model NAME
    preflight_serving.py --self-test        # fixtures, no GPU, used by CI
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Fields that have held the answer text in the wild. `reasoning` is what vLLM
# actually emits for poolside_v1 -- NOT the OpenAI-conventional
# `reasoning_content`. A probe checking only the latter sees nothing and wrongly
# concludes the output vanished (LESSONS.md, Laguna card line 200).
REASONING_FIELDS = ("reasoning", "reasoning_content")

# Deliberately answerable in few tokens, but each demands a visible final answer
# so a parser that swallows the tail is caught.
PROBES = (
    ("arith", "What is 2+2? Reply with just the number."),
    ("format", "What is 17*23? End your reply with exactly 'The answer is X'."),
    ("short", "Name the capital of France. One word."),
)

# Must exceed the reasoning preamble or every probe trips the BUDGET branch.
# The live 30B model needed ~109 completion tokens for a one-word answer.
DEFAULT_MAX_TOKENS = 1024


def post_chat(base: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def classify(resp: dict) -> tuple[str, str]:
    """Return (verdict, detail). Verdicts: ok | parser | budget | empty | malformed."""
    choices = resp.get("choices") or []
    if not choices:
        return "malformed", "response carried no choices[]"

    choice = choices[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    content = (msg.get("content") or "").strip()

    if content:
        return "ok", f"finish={finish}, {len(content)} chars"

    # Empty content. Find out WHY -- the whole point of the tool.
    hidden = [(f, msg[f]) for f in REASONING_FIELDS
              if isinstance(msg.get(f), str) and msg[f].strip()]

    if finish == "length":
        # Budget, not corruption. Raising max_tokens fixes it; the parser is fine.
        got = (resp.get("usage") or {}).get("completion_tokens", "?")
        return "budget", (f"finish=length, content empty after {got} tokens -- "
                          f"the answer never fit. Raise max_gen_toks; not a parser fault")

    if hidden:
        field, text = hidden[0]
        return "parser", (f"finish={finish}, content EMPTY but message.{field} holds "
                          f"{len(text)} chars: {text.strip()[:90]!r}")

    return "empty", f"finish={finish}, no content and no reasoning field -- nothing generated"


def probe_models(base: str, timeout: int) -> list:
    req = urllib.request.Request(base.rstrip("/") + "/models")
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return [m.get("id") for m in (json.load(fh).get("data") or [])]


def run_live(base: str, requested: str | None, max_tokens: int, timeout: int) -> int:
    try:
        served = probe_models(base, timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"INCONCLUSIVE: cannot reach {base}/models -- {exc}")
        print("Not a pass. Bring the endpoint up, or fix the URL, then re-run.")
        return 2

    if not served:
        print(f"INCONCLUSIVE: {base}/models returned an empty catalogue.")
        return 2

    if requested is None:
        model = served[0]
    elif requested not in served:
        print(f"INCONCLUSIVE: model {requested!r} is not served. Available: {served}")
        return 2
    else:
        model = requested

    if not isinstance(model, str):
        print(f"INCONCLUSIVE: served model id is not a string: {model!r}")
        return 2

    print(f"endpoint : {base}")
    print(f"model    : {model}")
    print(f"budget   : max_tokens={max_tokens}\n")

    verdicts = []
    for name, prompt in PROBES:
        try:
            resp = post_chat(base, model, prompt, max_tokens, timeout)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"  [{name}] INCONCLUSIVE -- request failed: {exc}")
            verdicts.append("inconclusive")
            continue
        verdict, detail = classify(resp)
        print(f"  [{name}] {verdict.upper()}: {detail}")
        verdicts.append(verdict)

    print()
    if "parser" in verdicts:
        print("DEFECT: the endpoint is returning answers that lm-eval will score as 0.")
        print("This is ISSUES #15. Every number from a run started now would be")
        print("silently depressed. Fix the reasoning parser before launching:")
        print("  - check the vLLM log for 'Auto-initialization of reasoning token IDs failed'")
        print("  - or set LM_EVAL_REASONING_FALLBACK=1 so the client reads the reasoning field")
        return 1
    if "empty" in verdicts:
        print("DEFECT: the endpoint returned no text at all. Do not launch.")
        return 1
    if "budget" in verdicts:
        print("DEFECT: answers did not fit the probe budget.")
        print(f"A benchmark using max_gen_toks near {max_tokens} would score those items 0.")
        print("Raise the generation budget, then re-run this preflight.")
        return 1
    if "inconclusive" in verdicts:
        print("INCONCLUSIVE: at least one probe could not be completed.")
        return 2
    print(f"OK: {len(verdicts)}/{len(verdicts)} probes returned usable content.")
    return 0


# --- fixtures: let CI exercise every branch with no GPU and no network --------
FIXTURES = {
    "healthy": ({"choices": [{"finish_reason": "stop",
                              "message": {"content": "4", "reasoning": "2+2 is 4"}}]}, "ok"),
    "issues15": ({"choices": [{"finish_reason": "stop",
                               "message": {"content": None,
                                           "reasoning": "2 cm/hour x 4 = 8\nThe answer is 8"}}]},
                 "parser"),
    "issues15_alt_field": ({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "",
                                                     "reasoning_content": "The answer is 8"}}]},
                           "parser"),
    "budget_starved": ({"choices": [{"finish_reason": "length",
                                     "message": {"content": None, "reasoning": "thinking..."}}],
                        "usage": {"completion_tokens": 64}}, "budget"),
    "nothing": ({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}, "empty"),
    "malformed": ({"choices": []}, "malformed"),
}


def run_self_test() -> int:
    failures = 0
    for name, (payload, expected) in FIXTURES.items():
        got, detail = classify(payload)
        ok = got == expected
        failures += not ok
        print(f"  [{'pass' if ok else 'FAIL'}] {name:20s} expected={expected:9s} got={got}")
        if not ok:
            print(f"         detail: {detail}")
    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures} classification(s) wrong.")
        return 1
    print(f"OK: {len(FIXTURES)}/{len(FIXTURES)} classifications correct "
          f"(incl. budget-vs-parser separation).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Refuse to launch a benchmark against an endpoint that drops answers.")
    ap.add_argument("--endpoint", default="http://csi370295.alcf.anl.gov:8000/v1",
                    help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", default=None, help="defaults to the first served model")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--self-test", action="store_true",
                    help="classify fixtures instead of probing (no GPU, used by CI)")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    return run_live(args.endpoint, args.model, args.max_tokens, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
