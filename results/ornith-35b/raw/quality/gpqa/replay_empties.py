"""Re-serve the 42 empty-content GPQA-Diamond items for Ornith-1.0-35B-FP8 (ISSUES #15).

WHY THIS EXISTS
---------------
lm-eval reads only `message.content`. When vLLM's reasoning parser strands the
answer in `message.reasoning`, the item is recorded as an empty string and
scored 0 -- HTTP 200, finish_reason "stop", no error, no retry. 42 of 198
GPQA-Diamond items hit this on the 2026-08-19 run, so the published 69.70% is
an underestimate of unknown size.

The retained samples_*.jsonl does NOT contain the reasoning field (lm-eval
discards what it does not read), so recovery REQUIRES re-serving against a live
endpoint. There is no offline path.

METHOD
------
Prompts are replayed BYTE-IDENTICALLY from the original run's stored
`arguments.gen_args_0.arg_0[0]`, with the same generation parameters
(max_gen_toks=32768, temperature=0, until=["</s>"]). Nothing is re-templated:
a reconstructed prompt would silently change the measurement.

Per item we record content, reasoning, finish_reason and completion_tokens so
the three failure signatures stay distinguishable (AGENTS.md):

    finish=length + empty                        -> BUDGET, raise max_gen_toks
    finish=stop   + empty + reasoning populated  -> PARSER, ISSUES #15
    finish=stop   + empty + nothing anywhere     -> genuinely EMPTY

Grading applies the SAME filter as the clean task: the last
"The answer is (X)" line, X in A-D, compared against the recorded target.

Usage:
    python3 replay_empties.py --ids ornith_empty_ids.json --out replay_results.json
"""
import argparse
import concurrent.futures
import glob
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.0-35B-FP8"

# VERBATIM from gpqa_diamond_cot_zeroshot_clean.yaml, filter "answer-line":
#   regex_pattern: "[Tt]he answer is \\(?([A-D])\\)?"   group_select: -1
# Do NOT "improve" this regex. A looser pattern (e.g. \s* for the literal space)
# grades items correct that lm-eval files as [invalid], which silently inflates
# the recovered score above anything the harness can reproduce.
ANS_RE = re.compile(r"[Tt]he answer is \(?([A-D])\)?")
TARGET_RE = re.compile(r"\(?([ABCD])\)?")


def load_originals(pattern):
    """Return {doc_id: {"prompt": str, "target": str, "gen": dict}} from the run's samples."""
    out = {}
    for path in glob.glob(pattern):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = rec.get("doc_id")
                if did in out:
                    continue
                args = (rec.get("arguments") or {}).get("gen_args_0") or {}
                arg0 = args.get("arg_0")
                if not arg0:
                    continue
                raw = arg0[0] if isinstance(arg0, list) else arg0
                try:
                    messages = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                out[did] = {
                    "prompt": messages,
                    "target": rec.get("target"),
                    "gen": args.get("arg_1") or {},
                }
    return out


def serve_one(doc_id, item, timeout, retries=3, max_tokens_override=None):
    gen = item["gen"]
    payload = {
        "model": MODEL,
        "messages": item["prompt"],
        "temperature": gen.get("temperature", 0),
        "max_tokens": max_tokens_override or gen.get("max_gen_toks", 32768),
    }
    until = gen.get("until")
    if until:
        payload["stop"] = until

    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                BASE, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                doc = json.load(resp)
            choice = doc["choices"][0]
            msg = choice.get("message") or {}
            usage = doc.get("usage") or {}
            # vLLM emits `reasoning`; some builds use `reasoning_content`.
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            return {
                "doc_id": doc_id,
                "content": msg.get("content") or "",
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "error": None,
            }
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            last_err = repr(exc)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return {
        "doc_id": doc_id,
        "content": "",
        "reasoning": "",
        "finish_reason": None,
        "completion_tokens": None,
        "error": last_err,
    }


def classify(rec):
    """AGENTS.md signature table."""
    has_content = bool((rec["content"] or "").strip())
    has_reasoning = bool((rec["reasoning"] or "").strip())
    if rec["error"]:
        return "ERROR"
    if has_content:
        return "RECOVERED"
    if rec["finish_reason"] == "length":
        return "BUDGET"
    if has_reasoning:
        return "PARSER"
    return "EMPTY"


def grade(rec, target):
    """Apply the clean task's answer-line filter to `content` ONLY.

    Grading `reasoning` would be cheating: lm-eval reads only `content`, so an
    answer recovered from the reasoning channel is not a score the harness can
    reproduce. Recovery must come from the model actually emitting content.
    """
    tm = TARGET_RE.search(str(target or ""))
    want = tm.group(1) if tm else None
    hits = ANS_RE.findall(rec.get("content") or "")
    if hits:
        return hits[-1] == want, hits[-1], "content"  # group_select: -1
    return False, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/home/jchilders/lmeval_samples_backup/"
                                         "lmeval_results__ornith35b__gpqa__*.jsonl.gz")
    ap.add_argument("--ids", required=True, help="JSON list of empty doc_ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the original run's max_gen_toks (budget sweep)")
    args = ap.parse_args()

    empty_ids = json.load(open(args.ids))
    originals = load_originals(args.samples)
    missing = [d for d in empty_ids if d not in originals]
    if missing:
        sys.exit(f"FATAL: no stored prompt for doc_ids {missing}; cannot replay faithfully.")

    print(f"replaying {len(empty_ids)} items at c={args.concurrency}, "
          f"timeout={args.timeout}s", flush=True)

    results = {}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(serve_one, d, originals[d], args.timeout,
                        max_tokens_override=args.max_tokens): d for d in empty_ids
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            rec = fut.result()
            did = rec["doc_id"]
            rec["target"] = originals[did]["target"]
            rec["verdict"] = classify(rec)
            ok, picked, src = grade(rec, rec["target"])
            rec["correct"] = ok
            rec["picked"] = picked
            rec["answer_from"] = src
            rec["content_chars"] = len(rec["content"])
            rec["reasoning_chars"] = len(rec["reasoning"])
            results[did] = rec
            print(f"[{i}/{len(empty_ids)}] doc {did}: {rec['verdict']} "
                  f"finish={rec['finish_reason']} tok={rec['completion_tokens']} "
                  f"correct={ok} ({time.time()-started:.0f}s)", flush=True)
            # Persist incrementally: a crash must not lose completed work.
            with open(args.out, "w") as fh:
                json.dump([results[k] for k in sorted(results)], fh, indent=1)

    ordered = [results[k] for k in sorted(results)]
    with open(args.out, "w") as fh:
        json.dump(ordered, fh, indent=1)

    counts = {}
    for r in ordered:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    recovered = [r for r in ordered if r["verdict"] == "RECOVERED"]
    n_correct = sum(1 for r in ordered if r["correct"])
    print("\n=== verdicts ===")
    for k in sorted(counts):
        print(f"  {k:10s} {counts[k]}")
    print(f"\nrecovered content: {len(recovered)}/{len(ordered)}")
    print(f"newly correct    : {n_correct}")
    print(f"elapsed          : {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
