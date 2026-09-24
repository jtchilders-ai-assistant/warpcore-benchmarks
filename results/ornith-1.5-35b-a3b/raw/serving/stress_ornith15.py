#!/usr/bin/env python3
import concurrent.futures
import json
import sys
import time
import urllib.request

BASE = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
URL = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-FP8"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}, "strict": True}}]
PROMPTS = [
    "What's the weather in Chicago in celsius? Use the tool.",
    "Get the fahrenheit weather for Tokyo.",
    "Weather in Paris, celsius please.",
    "Check London weather in fahrenheit.",
    "What is the temperature in Berlin (celsius)?",
    "Weather for Sydney in fahrenheit.",
]

def one_request(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "tools": TOOLS, "tool_choice": "auto", "max_tokens": 300, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.load(r)
        choice = resp["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        if choice.get("finish_reason") != "tool_calls" or not calls:
            raise ValueError(f"not a parsed tool call: {choice}")
        for call in calls:
            json.loads(call["function"]["arguments"])
        return i, "OK", round(time.time() - t0, 1), len(calls)
    except Exception as exc:
        return i, "FAIL", round(time.time() - t0, 1), str(exc)[:160]

def engine_alive():
    req = urllib.request.Request(URL, data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Reply ALIVE"}], "max_tokens": 256, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"].get("content")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one_request, range(n)))
    for result in results:
        print(result)
    failures = [result for result in results if result[1] != "OK"]
    print(json.dumps({"ok": n-len(failures), "total": n, "failures": len(failures), "elapsed_s": round(time.time()-started, 1), "engine_alive": engine_alive()}))
    raise SystemExit(1 if failures else 0)
