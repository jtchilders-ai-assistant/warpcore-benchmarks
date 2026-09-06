"""Re-serve the 28 empty-content IFEval items for Ornith-1.0-35B-FP8.

WHY THIS EXISTS
---------------
IFEval measures instruction-following at the level of individual instructions
within each prompt (prompt_level and instruction_level accuracy). 28 of 541
prompts got empty content on the 2026-08-19 run due to ISSUES #15. These 28
items score 0 on ALL their instructions, depressing the published 85.58% below
the true rate.

METHOD
------
Prompts are replayed BYTE-IDENTICALLY from the original run's stored
arguments.gen_args_0.arg_0[0]. Recovery uses the committed JSONL, not the
gzipped backup, because the JSONL is the source-of-record.

Output: a JSON list of {doc_id, content, reasoning, finish_reason,
completion_tokens, error} for the 28 items. IFEval re-scoring is done
separately by scoring the recovered content against the original prompt
instructions using the IFEval evaluator.
"""
import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.0-35B-FP8"


def load_originals(jsonl_path):
    """Return {doc_id: {prompt, gen}} from committed lm-eval samples JSONL."""
    out = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = rec.get("doc_id")
            if did is None or did in out:
                continue
            args = (rec.get("arguments") or {}).get("gen_args_0") or {}
            arg0 = args.get("arg_0")
            if not arg0:
                continue
            raw = arg0[0] if isinstance(arg0, list) else arg0
            try:
                messages = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # IFEval sometimes stores the prompt as a plain string
                messages = [{"role": "user", "content": raw}]
            out[did] = {
                "prompt": messages,
                "gen": args.get("arg_1") or {},
            }
    return out


def serve_one(doc_id, item, timeout, retries=3):
    gen = item["gen"]
    payload = {
        "model": MODEL,
        "messages": item["prompt"],
        "temperature": gen.get("temperature", 0),
        "max_tokens": gen.get("max_gen_toks", 8192),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="Committed lm-eval samples JSONL (not .gz)")
    ap.add_argument("--ids", required=True, help="JSON list of empty doc_ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    empty_ids = json.load(open(args.ids))
    originals = load_originals(args.jsonl)
    missing = [d for d in empty_ids if d not in originals]
    if missing:
        sys.exit(f"FATAL: no stored prompt for doc_ids {missing}")

    print(f"replaying {len(empty_ids)} IFEval items at c={args.concurrency}, "
          f"timeout={args.timeout}s", flush=True)

    results = {}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(serve_one, d, originals[d], args.timeout): d
            for d in empty_ids
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            rec = fut.result()
            did = rec["doc_id"]
            has_content = bool((rec["content"] or "").strip())
            has_reasoning = bool((rec["reasoning"] or "").strip())
            if rec["error"]:
                verdict = "ERROR"
            elif has_content:
                verdict = "RECOVERED"
            elif rec["finish_reason"] == "length":
                verdict = "BUDGET"
            elif has_reasoning:
                verdict = "PARSER"
            else:
                verdict = "EMPTY"
            rec["verdict"] = verdict
            rec["content_chars"] = len(rec["content"] or "")
            rec["reasoning_chars"] = len(rec["reasoning"] or "")
            results[did] = rec
            print(f"[{i}/{len(empty_ids)}] doc {did}: {verdict} "
                  f"finish={rec['finish_reason']} tok={rec['completion_tokens']} "
                  f"({time.time()-started:.0f}s)", flush=True)
            with open(args.out, "w") as fh:
                json.dump([results[k] for k in sorted(results)], fh, indent=1)

    ordered = [results[k] for k in sorted(results)]
    with open(args.out, "w") as fh:
        json.dump(ordered, fh, indent=1)

    counts = {}
    for r in ordered:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n=== verdicts ===")
    for k in sorted(counts):
        print(f"  {k:10s} {counts[k]}")
    print(f"\nrecovered content: {counts.get('RECOVERED',0)}/{len(ordered)}")
    print(f"elapsed          : {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
