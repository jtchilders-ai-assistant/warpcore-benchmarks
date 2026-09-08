"""Replay selected Ornith IFEval prompts at 64k and score with lm-eval's evaluator.

This preserves the stored chat messages byte-for-byte from the original lm-eval
samples, records generation metadata, and applies the exact lm-eval 0.4.12
IFEval strict/loose evaluators to recovered content.
"""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

from lm_eval.tasks.ifeval import utils as ifeval_utils

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.0-35B-FP8"


def load_originals(path):
    originals = {}
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            doc_id = record["doc_id"]
            args = record["arguments"]["gen_args_0"]
            originals[doc_id] = {
                "doc": record["doc"],
                "messages": json.loads(args["arg_0"][0]),
                "temperature": args["arg_1"].get("temperature", 0),
                "until": args["arg_1"].get("until", []),
            }
    return originals


def serve_one(doc_id, item, timeout, retries=3):
    payload = {
        "model": MODEL,
        "messages": item["messages"],
        "temperature": item["temperature"],
        "max_tokens": 65536,
    }
    if item["until"]:
        payload["stop"] = item["until"]
    body = json.dumps(payload).encode()
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                BASE, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            message = choice.get("message") or {}
            return {
                "doc_id": doc_id,
                "content": message.get("content") or "",
                "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
                "error": None,
            }
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            error = repr(exc)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return {"doc_id": doc_id, "content": "", "reasoning": "", "finish_reason": None,
            "completion_tokens": None, "error": error}


def score(doc, response):
    metrics = ifeval_utils.process_results(doc, [response])
    return {
        "prompt_level_strict_acc": bool(metrics["prompt_level_strict_acc"]),
        "inst_level_strict_acc": [bool(x) for x in metrics["inst_level_strict_acc"]],
        "prompt_level_loose_acc": bool(metrics["prompt_level_loose_acc"]),
        "inst_level_loose_acc": [bool(x) for x in metrics["inst_level_loose_acc"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    originals = load_originals(args.jsonl)
    ids = json.load(open(args.ids))
    missing = [d for d in ids if d not in originals]
    if missing:
        raise SystemExit(f"Missing stored prompts: {missing}")

    started = time.time()
    results = {}
    print(f"Replaying {len(ids)} IFEval budget-limited prompts at max_tokens=65536, c={args.concurrency}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(serve_one, d, originals[d], args.timeout): d for d in ids}
        for ordinal, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            has_content = bool(record["content"].strip())
            has_reasoning = bool(record["reasoning"].strip())
            if record["error"]:
                record["verdict"] = "ERROR"
            elif has_content:
                record["verdict"] = "RECOVERED"
                record["metrics"] = score(originals[record["doc_id"]]["doc"], record["content"])
            elif record["finish_reason"] == "length":
                record["verdict"] = "BUDGET"
            elif has_reasoning:
                record["verdict"] = "PARSER"
            else:
                record["verdict"] = "EMPTY"
            record["content_chars"] = len(record["content"])
            record["reasoning_chars"] = len(record["reasoning"])
            results[record["doc_id"]] = record
            print(f"[{ordinal}/{len(ids)}] doc {record['doc_id']}: {record['verdict']} "
                  f"finish={record['finish_reason']} tok={record['completion_tokens']} "
                  f"elapsed={time.time()-started:.0f}s", flush=True)
            with open(args.out, "w") as fh:
                json.dump([results[k] for k in sorted(results)], fh, indent=2)

    ordered = [results[k] for k in sorted(results)]
    with open(args.out, "w") as fh:
        json.dump(ordered, fh, indent=2)
    counts = {}
    for r in ordered:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("verdicts:", counts)


if __name__ == "__main__":
    main()
