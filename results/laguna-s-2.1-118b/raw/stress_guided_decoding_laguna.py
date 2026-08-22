#!/usr/bin/env python3
"""Stress-test a gpt-oss vLLM endpoint with the workload that crashes the CUTLASS MXFP4 build.

Reproduces warpcore crash Signature 1: concurrent structured-output / tool-calling requests
(strict JSON schema => guided decoding). Use to verify a build survives before declaring a fix.

Usage (run ON warpcore, or point at the host):
    python3 stress_guided_decoding.py [N_REQUESTS] [CONCURRENCY] [BASE_URL]
    python3 stress_guided_decoding.py 60 16
    python3 stress_guided_decoding.py 60 16 http://localhost:8000

Exit code 0 = all OK; 1 = at least one failure (engine likely died — check `docker logs`).
A healthy result: all requests return tool_calls, and the post-run "engine alive" probe replies.
The OLD --exp-mxfp4 (CUTLASS) build dies here with cudaErrorIllegalAddress; the newer
eugr/spark-vllm:latest (MARLIN backend) passed 84/84.
"""
import json, urllib.request, concurrent.futures, time, sys

BASE = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
URL = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = __import__("os").environ.get("STRESS_MODEL", "openai/gpt-oss-120b")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
        "strict": True,  # forces guided decoding — the crash trigger
    },
}]
PROMPTS = [
    "What's the weather in Chicago in celsius? Use the tool.",
    "Get the fahrenheit weather for Tokyo.",
    "Weather in Paris, celsius please.",
    "Check London weather in fahrenheit.",
    "What is the temperature in Berlin (celsius)?",
    "Weather for Sydney in fahrenheit.",
]


def one_request(i):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": int(__import__("os").environ.get("STRESS_MAXTOK", "300")),
        "temperature": 0.7,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.load(r)
        msg = resp["choices"][0]["message"]
        kind = "tool_call" if msg.get("tool_calls") else "text"
        return (i, "OK", round(time.time() - t0, 1), kind)
    except Exception as e:
        return (i, "FAIL", round(time.time() - t0, 1), str(e)[:120])


def engine_alive():
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Say ALIVE"}],
            "max_tokens": int(__import__("os").environ.get("STRESS_MAXTOK","300"))}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["choices"][0]["message"].get("content")
    except Exception as e:
        return f"DEAD: {str(e)[:120]}"


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    print(f"Firing {N} reqs, {CONC} concurrent, strict-schema tools (guided decoding) at {URL}")
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        for f in concurrent.futures.as_completed([ex.submit(one_request, i) for i in range(N)]):
            r = f.result()
            results.append(r)
            print(f"  req {r[0]:2d}: {r[1]:4s} {r[2]:5.1f}s {r[3]}")
    ok = sum(1 for r in results if r[1] == "OK")
    fail = N - ok
    print(f"\n=== {ok}/{N} OK, {fail} FAIL in {time.time()-t0:.1f}s ===")
    print(f"engine alive probe: {engine_alive()!r}")
    sys.exit(1 if fail else 0)
