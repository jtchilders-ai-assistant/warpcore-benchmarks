#!/usr/bin/env python3
"""Replay Nemotron-3-Super GPQA rows with empty original content at 64K.

Preserves each stored prompt and original temperature/stop settings, changing only
max_tokens. Retains the complete response message, finish reason, usage, errors,
and a conservative answer-letter extraction for later composite scoring.
"""
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "samples_gpqa_diamond_cot_zeroshot_clean_2026-07-30T18-26-57.972871.jsonl"
OUT = ROOT / "replay_64k_2026-09-09"
ENDPOINT = "http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
BUDGET = 65536
WORKERS = 8
TIMEOUT = 30000


def response_text(row):
    resps = row.get("resps") or []
    if not resps:
        return ""
    first = resps[0]
    if isinstance(first, list):
        return (first[0] if first else "") or ""
    return first if isinstance(first, str) else ""


def answer_letter(text):
    matches = re.findall(r"(?i)the answer is\s*\(?([A-D])\)?", text or "")
    return matches[-1].upper() if matches else None


def target_letter(target):
    matches = re.findall(r"[A-D]", str(target).upper())
    return matches[-1] if matches else None

rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
per_doc = {}
for row in rows:
    entry = per_doc.setdefault(row["doc_id"], row)
    if not response_text(entry).strip() and response_text(row).strip():
        per_doc[row["doc_id"]] = row
empties = [per_doc[k] for k in sorted(per_doc) if not response_text(per_doc[k]).strip()]
if len(empties) != 56:
    raise SystemExit(f"refusing replay: expected 56 unique empty docs, found {len(empties)}")
OUT.mkdir(parents=True, exist_ok=True)


def replay(row):
    gen = row["arguments"]["gen_args_0"]
    arg0 = gen["arg_0"]
    if isinstance(arg0, list):
        arg0 = arg0[0]
    messages = json.loads(arg0)
    original = gen.get("arg_1") or {}
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": BUDGET,
        "temperature": original.get("temperature", 0),
    }
    if original.get("until"):
        body["stop"] = original["until"]
    started = time.time()
    result = {
        "doc_id": row["doc_id"],
        "target": row.get("target"),
        "gold": target_letter(row.get("target")),
        "request": body,
        "original_generation_args": original,
    }
    try:
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            data = json.load(response)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        result.update(
            status="ok",
            finish_reason=choice.get("finish_reason"),
            message=message,
            usage=data.get("usage"),
            prediction=answer_letter(content),
            correct=answer_letter(content) == result["gold"],
        )
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", correct=False)
    result["elapsed_s"] = round(time.time() - started, 3)
    return result

started = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(replay, row): row["doc_id"] for row in empties}
    completed = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        completed.append(result)
        print(json.dumps({k: result.get(k) for k in ("doc_id", "status", "finish_reason", "prediction", "gold", "correct", "elapsed_s")}), flush=True)
completed.sort(key=lambda row: row["doc_id"])
(OUT / "results.json").write_text(json.dumps(completed, indent=2) + "\n")
summary = {
    "model": MODEL,
    "endpoint": ENDPOINT,
    "source": str(SOURCE.relative_to(ROOT.parents[3])),
    "source_unique_docs": len(per_doc),
    "replayed_docs": len(completed),
    "max_tokens": BUDGET,
    "workers": WORKERS,
    "timeout_seconds": TIMEOUT,
    "elapsed_seconds": round(time.time() - started, 3),
    "http_errors": sum(r["status"] != "ok" for r in completed),
    "finish_stop": sum(r.get("finish_reason") == "stop" for r in completed),
    "finish_length": sum(r.get("finish_reason") == "length" for r in completed),
    "content_nonempty": sum(bool((r.get("message") or {}).get("content")) for r in completed),
    "correct": sum(bool(r.get("correct")) for r in completed),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "DONE").touch()
print(json.dumps(summary, indent=2), flush=True)
