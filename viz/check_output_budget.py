#!/usr/bin/env python3
"""Report the EFFECTIVE server-side output-token ceiling for a served model.

WHY THIS EXISTS
vLLM's `--generation-config` defaults to `auto`, which loads the model's own
`generation_config.json` at startup. Per `vllm serve --help=generation-config`:

    "If max_new_tokens is specified in generation config, then it sets a
     server-wide limit on the number of output tokens for all requests."

So a checkpoint can impose a hard output cap that appears NOWHERE in the launch
command, is not echoed in the startup banner, and produces no error when a
client asks for more -- the request is silently truncated. A benchmark run with
`max_gen_toks=65536` against such a model measures the CAP, not the model, and
the depressed score is indistinguishable from a capability result.

This is not hypothetical: poolside/Laguna-XS-2.1-NVFP4 ships
`"max_new_tokens": 32768` while the S variant ships none.

USAGE
    python3 viz/check_output_budget.py --model poolside/Laguna-S-2.1-NVFP4 \
        [--base-url http://csi370295.alcf.anl.gov:8000] \
        [--require-budget 32768] [--json]

Exit codes: 0 = budget satisfied (or only informational), 1 = REQUIRED budget
not satisfiable, 2 = could not determine (never silently pass).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://csi370295.alcf.anl.gov:8000"


def _get_json(url: str, timeout: int = 20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def probe_max_model_len(base_url: str, model: str):
    """Read max_model_len from /v1/models (authoritative, from the live engine)."""
    try:
        data = _get_json(f"{base_url}/v1/models")
    except Exception as e:
        return None, f"/v1/models unreachable: {type(e).__name__}"
    for m in data.get("data", []):
        if m.get("id") == model:
            return m.get("max_model_len"), None
    ids = [m.get("id") for m in data.get("data", [])]
    return None, f"model {model!r} not served; endpoint has {ids}"


def probe_effective_cap(base_url: str, model: str, ask: int):
    """Ask for `ask` output tokens on a trivial prompt.

    A server-wide max_new_tokens cap does NOT reject the request -- it silently
    clamps. So we cannot detect it from an accepted short answer alone. What we
    CAN do cheaply and reliably: detect an explicit 400 (bound exceeded), and
    report the error text, which names the binding limit.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": ask,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        return {"accepted": True,
                "finish_reason": d["choices"][0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error", {}).get("message", "")
        except Exception:
            msg = e.read().decode(errors="replace")[:300]
        return {"accepted": False, "status": e.code, "message": msg}
    except Exception as e:
        return {"accepted": None, "error": f"{type(e).__name__}: {e}"}


def read_generation_config(path: str | None):
    """Read max_new_tokens from a local generation_config.json, if provided."""
    if not path:
        return None, "not checked (pass --generation-config PATH)"
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception as e:
        return None, f"unreadable: {type(e).__name__}"
    return cfg.get("max_new_tokens", "absent"), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--require-budget", type=int, default=None,
                    help="Fail (exit 1) if this many output tokens is not permitted.")
    ap.add_argument("--generation-config", default=None,
                    help="Path to the model's generation_config.json (checks max_new_tokens).")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = {"model": a.model, "base_url": a.base_url}

    mml, err = probe_max_model_len(a.base_url, a.model)
    out["max_model_len"] = mml
    if err:
        out["max_model_len_error"] = err

    gen_cap, gen_err = read_generation_config(a.generation_config)
    out["generation_config_max_new_tokens"] = gen_cap
    if gen_err:
        out["generation_config_note"] = gen_err

    if a.require_budget:
        out["required_budget"] = a.require_budget
        out["probe"] = probe_effective_cap(a.base_url, a.model, a.require_budget)

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model:                {a.model}")
        print(f"max_model_len:        {mml}{'  (' + err + ')' if err else ''}")
        print(f"gen-config max_new_tokens: {gen_cap}"
              f"{'  (' + gen_err + ')' if gen_err else ''}")
        if a.require_budget:
            print(f"required budget:      {a.require_budget}")
            print(f"probe:                {out['probe']}")

    # --- verdict ---
    # Order matters: an unreachable/unserved model must FAIL, never fall through
    # to a PASS. A check that cannot see the endpoint has proven nothing.
    if mml is None:
        print(f"VERDICT: UNDETERMINED - could not read max_model_len"
              f"{': ' + err if err else ''}", file=sys.stderr)
        return 2

    if isinstance(gen_cap, int) and a.require_budget and gen_cap < a.require_budget:
        print(f"VERDICT: FAIL - checkpoint generation_config caps output at {gen_cap} "
              f"tokens (server-wide), below the required {a.require_budget}. "
              f"Serve with --generation-config vllm to disable, or lower the budget.",
              file=sys.stderr)
        return 1

    if a.require_budget:
        p = out["probe"]
        if p.get("accepted") is False:
            print(f"VERDICT: FAIL - endpoint rejected max_tokens={a.require_budget}: "
                  f"{p.get('message')}", file=sys.stderr)
            return 1
        if p.get("accepted") is None:
            print(f"VERDICT: UNDETERMINED - probe error: {p.get('error')}", file=sys.stderr)
            return 2
        if gen_cap == "absent" or gen_cap is None:
            print(f"VERDICT: PASS - max_tokens={a.require_budget} accepted; no checkpoint "
                  f"max_new_tokens cap detected.")
        else:
            print(f"VERDICT: PASS - max_tokens={a.require_budget} accepted "
                  f"(checkpoint cap {gen_cap}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
