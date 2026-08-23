"""Is the 1800s timeout simply too small for Laguna's speed?

Measured sweep data: at c=4 Laguna delivers ~42 tok/s aggregate = ~10.5 tok/s PER USER.
The SWE-bench config allows max_tokens=32768 per step. If a step runs to the cap:
    32768 tok / 10.5 tok/s ~= 3100 s  >>  1800 s litellm timeout
=> the client kills a HEALTHY, still-generating request. litellm then retries, burning
   another 1800s. That matches the observed 2 retries / ~60 min stuck instances.

This measures the actual sustained per-request rate under the smoke's concurrency and
reports the wall-clock a full 32768-token step would need.
"""
import json, urllib.request, time

BASE = "http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"
MAX_TOK = 32768
BUDGET = 120  # seconds of streaming to sample the rate

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content":
                  "Write an extremely detailed technical essay about the Python "
                  "descriptor protocol. Be exhaustive and do not stop early."}],
    "max_tokens": 4096,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}

req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})

t0 = time.time()
first = None
n = 0
with urllib.request.urlopen(req, timeout=BUDGET + 60) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        try:
            d = json.loads(body)
        except Exception:
            continue
        ch = d.get("choices") or []
        if ch:
            delta = ch[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                if first is None:
                    first = time.time() - t0
                n += 1
        if time.time() - t0 > BUDGET:
            break

el = time.time() - t0
rate = n / el if el else 0
print(f"concurrent load during test: (see metrics)")
print(f"TTFT                : {first:.2f}s" if first else "no tokens")
print(f"sampled tokens      : {n} in {el:.1f}s")
print(f"per-request rate    : {rate:.2f} tok/s")
print()
if rate > 0:
    need = MAX_TOK / rate
    print(f"time for a FULL {MAX_TOK}-token step: {need:.0f}s ({need/60:.1f} min)")
    print(f"configured litellm timeout          : 1800s (30.0 min)")
    print()
    if need > 1800:
        print(f"VERDICT: timeout is TOO SMALL by {need-1800:.0f}s. A healthy long step gets")
        print(f"         killed mid-generation. Need timeout >= ~{int(need*1.3//600+1)*600}s.")
    else:
        print("VERDICT: 1800s is sufficient at this rate -- look elsewhere.")
