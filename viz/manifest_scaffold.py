"""Scaffold a PROVENANCE.md-compliant manifest.json for a run.

PROVENANCE.md §4 promises `make manifest MODEL=<m> BENCH=<b>`: auto-fill what can
be probed, prompt for nothing, and write the literal string "unrecorded" for
anything genuinely unknown.

That last rule is the point. A missing key is ambiguous -- did nobody record the
image digest, or was there no digest? "unrecorded" is a fact you can grep for and
later fill in. This tool NEVER guesses a value: fields it cannot probe are
emitted as "unrecorded", not inferred from a sibling model's manifest.

Auto-filled when available: repo commit, model id (from the card), engine version
and image (from a live endpoint via --endpoint, or a launch script via --launch-script),
harness versions, and file timestamps. Everything else is "unrecorded".

  manifest_scaffold.py --model ornith-35b --bench swebench
  manifest_scaffold.py --model ornith-35b --bench swebench --endpoint http://warpcore:8000
  manifest_scaffold.py --model ornith-35b --bench swebench --force   # overwrite
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
UNREC = "unrecorded"


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return UNREC


def probe_endpoint(url: str) -> dict:
    """Read model id + engine version off a live vLLM endpoint."""
    out = {"model_id": UNREC, "engine_version": UNREC}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as r:
            data = json.load(r)
        served = (data.get("data") or [{}])[0]
        out["model_id"] = served.get("id", UNREC)
        out["engine_version"] = served.get("vllm_version", UNREC)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"  ! endpoint probe failed ({exc}); leaving '{UNREC}'", file=sys.stderr)
    return out


def scan_launch_script(path: pathlib.Path) -> dict:
    """Extract serving args from a committed launch script.

    Recorded as launch_script_matches_run=UNREC deliberately: the script is what
    is COMMITTED, not necessarily what RAN. Ornith's script pins
    --gpu-memory-utilization 0.90 while the SWE-bench run used 0.55. Only a human
    who watched the run can assert they match.
    """
    out = {"args": [], "image": UNREC, "env": {}}
    if not path.exists():
        return out
    text = path.read_text()
    out["args"] = sorted(set(re.findall(r"(--[a-z0-9][a-z0-9-]*(?:\s+[^\\\n\"']+)?)", text)))
    m = re.search(r"(vllm/vllm-openai:[^\s\"'\\]+)", text)
    if m:
        out["image"] = m.group(1)
    for k, v in re.findall(r"(?:^|\s)-e\s+([A-Z_][A-Z0-9_]*)=([^\s\\]+)", text):
        out["env"][k] = v
    return out


def card_model_id(model_dir: pathlib.Path) -> str:
    card = model_dir / "README.md"
    if not card.exists():
        return UNREC
    m = re.search(r"huggingface\.co/([\w.\-]+/[\w.\-]+)", card.read_text())
    return m.group(1) if m else UNREC


def build(model: str, bench: str, endpoint: str | None,
          launch_script: pathlib.Path | None) -> dict:
    model_dir = RESULTS / model
    probed = probe_endpoint(endpoint) if endpoint else {}
    launch = scan_launch_script(launch_script) if launch_script else {}

    return {
        "schema_version": 1,
        "model": {
            "id": probed.get("model_id") or card_model_id(model_dir),
            "revision": UNREC,
            "quantization": UNREC,
        },
        "benchmark": {
            "name": bench,
            "sample": UNREC,
            "harness": UNREC,
            "harness_version": UNREC,
            "grading_harness": UNREC,
            "run_id": UNREC,
        },
        "serving": {
            "engine": "vLLM",
            "version": probed.get("engine_version", UNREC),
            "image": launch.get("image", UNREC),
            "image_digest": UNREC,
            "host": UNREC,
            "launch_script": str(launch_script) if launch_script else UNREC,
            "args": launch.get("args") or UNREC,
            "env": launch.get("env") or UNREC,
            # Never auto-assert this: the committed script may not be what ran.
            "launch_script_matches_run": UNREC,
            "launch_script_note": UNREC,
            "generation_config": UNREC,
            "checkpoint_max_new_tokens": UNREC,
        },
        "client": {
            "host": UNREC,
            "concurrency": UNREC,
            "agent_config": UNREC,
            "limits": UNREC,
        },
        "timing": {"started": UNREC, "finished": UNREC},
        "segments": [],
        "repo_commit": repo_commit(),
        "notes": "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="directory name under results/")
    ap.add_argument("--bench", required=True,
                    help="benchmark subdir under raw/ (e.g. swebench, quality/gpqa)")
    ap.add_argument("--endpoint", help="live vLLM base URL to probe (optional)")
    ap.add_argument("--launch-script", type=pathlib.Path,
                    help="committed launch script to scan for serving args")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args(argv)

    model_dir = RESULTS / args.model
    if not model_dir.is_dir():
        print(f"error: no such model dir: {model_dir}", file=sys.stderr)
        print(f"known: {', '.join(sorted(p.name for p in RESULTS.iterdir() if p.is_dir()))}",
              file=sys.stderr)
        return 2

    launch = args.launch_script
    if launch is None:
        cands = sorted((model_dir / "raw").glob("launch*.sh"))
        launch = cands[0] if cands else None
        if launch:
            print(f"  using launch script: {launch.relative_to(REPO)}")

    manifest = build(args.model, args.bench, args.endpoint, launch)

    if args.stdout:
        print(json.dumps(manifest, indent=2))
        return 0

    out = model_dir / "raw" / args.bench / "manifest.json"
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out.relative_to(REPO)} (use --force)",
              file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    unrec = json.dumps(manifest).count(f'"{UNREC}"')
    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {unrec} field(s) marked '{UNREC}' -- fill these in from the run log.")
    print("  Do NOT guess: an unrecorded fact is better than a plausible fiction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
