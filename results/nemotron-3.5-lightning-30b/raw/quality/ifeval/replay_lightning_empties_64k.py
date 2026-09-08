#!/usr/bin/env python3
"""Replay Lightning's 47 empty IFEval responses at the 64k standard ceiling.

Uses the exact stored lm-eval messages and the lm-eval 0.4.12 IFEval scorer.
Writes checkpointed JSON after every completed request and a DONE sentinel only
when all target IDs have terminal records.
"""
import argparse
import concurrent.futures
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from lm_eval.tasks.ifeval import utils as ifeval_utils

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
BASE = "http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
BUDGET = 65536


def load_originals(path: Path):
    originals = {}
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            args = record["arguments"]["gen_args_0"]
            raw = args["arg_0"][0]
            originals[int(record["doc_id"])] = {
                "doc": record["doc"],
                "messages": json.loads(raw),
                "temperature": args["arg_1"].get("temperature", 0),
                "until": args["arg_1"].get("until", []),
            }
    return originals


def load_empty_ids(path: Path):
    with path.open(newline="") as fh:
        return sorted(int(row["doc_id"]) for row in csv.DictReader(fh)
                      if int(row["empty_content"]) == 1)


def serve_one(doc_id, item, timeout, retries=3):
    payload = {
        "model": MODEL,
        "messages": item["messages"],
        "temperature": item["temperature"],
        "max_tokens": BUDGET,
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


def classify(record, doc):
    has_content = bool(record["content"].strip())
    has_reasoning = bool(record["reasoning"].strip())
    if record["error"]:
        record["verdict"] = "ERROR"
    elif has_content:
        record["verdict"] = "RECOVERED"
        record["metrics"] = score(doc, record["content"])
    elif record["finish_reason"] == "length":
        record["verdict"] = "BUDGET"
    elif has_reasoning:
        record["verdict"] = "PARSER"
    else:
        record["verdict"] = "EMPTY"
    record["content_chars"] = len(record["content"])
    record["reasoning_chars"] = len(record["reasoning"])
    return record


def save(path, results):
    path.write_text(json.dumps([results[k] for k in sorted(results)], indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    originals = load_originals(args.jsonl)
    ids = load_empty_ids(args.csv)
    missing = [doc_id for doc_id in ids if doc_id not in originals]
    if len(ids) != 47 or missing:
        raise SystemExit(f"precondition failed: ids={len(ids)} missing={missing}")
    print(f"VALIDATED exact stored prompts for {len(ids)} target IDs", flush=True)
    if args.validate_only:
        return

    output = args.run_dir / "results_64k.json"
    done = args.run_dir / "DONE"
    failed = args.run_dir / "FAILED"
    done.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    started = time.time()
    results = {}
    print(f"Replaying {len(ids)} Lightning IFEval prompts at max_tokens={BUDGET}, "
          f"c={args.concurrency}", flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(serve_one, d, originals[d], args.timeout): d for d in ids}
            for ordinal, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                doc_id = record["doc_id"]
                classify(record, originals[doc_id]["doc"])
                results[doc_id] = record
                save(output, results)
                print(f"[{ordinal}/{len(ids)}] doc {doc_id}: {record['verdict']} "
                      f"finish={record['finish_reason']} tok={record['completion_tokens']} "
                      f"elapsed={time.time()-started:.0f}s", flush=True)
        if len(results) != len(ids):
            raise RuntimeError(f"completed {len(results)}/{len(ids)}")
        counts = {}
        for record in results.values():
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
        print("verdicts:", counts, flush=True)
        done.touch()
    except BaseException as exc:
        failed.write_text(repr(exc) + "\n")
        raise


if __name__ == "__main__":
    main()
