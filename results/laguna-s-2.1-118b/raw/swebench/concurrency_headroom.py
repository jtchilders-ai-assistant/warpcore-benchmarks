"""Does raising concurrency actually buy aggregate throughput on this box right now?

The run uses w=4 and the endpoint delivers ~34 tok/s aggregate. The sweep measured a
259 tok/s peak, implying ~7.6x unused capacity. But that sweep ran with NO other load and
short prompts; SWE-bench sends long agent contexts. So MEASURE it against the live server
while the run is in flight, rather than trusting the old sweep.

Fires N extra concurrent requests alongside the running job and reports the marginal
aggregate gain. If aggregate rises roughly linearly, more workers = proportionally faster
wall clock. If flat, we are already saturated and raising w only slows each request.
"""
import json, threading, time, urllib.request

EP = "http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"
DUR = 75
PROMPT = ("Explain in exhaustive technical detail how a copy-on-write B-tree works, "
          "including split/merge, concurrency, and crash recovery. Be very thorough.")


def worker(counts, idx):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 3000, "temperature": 0, "stream": True}
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=DUR + 40) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                b = line[6:]
                if b == "[DONE]":
                    break
                try:
                    d = json.loads(b)
                except Exception:
                    continue
                ch = d.get("choices") or []
                if ch:
                    dl = ch[0].get("delta") or {}
                    if dl.get("content") or dl.get("reasoning"):
                        n += 1
                if time.time() - t0 > DUR:
                    break
    except Exception as e:
        counts[idx] = (0, str(e)[:60])
        return
    counts[idx] = (n / max(time.time() - t0, 1e-9), "")


def server_rate(sec=20):
    """Aggregate gen tok/s straight from vLLM metrics (includes the real run)."""
    import subprocess
    def g():
        out = subprocess.run(
            ["ssh", "warpcore",
             "curl -s -m10 http://localhost:8000/metrics | awk '/^vllm:generation_tokens_total/{print $2}'"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out.splitlines()[0])
    a = g(); time.sleep(sec); b = g()
    return (b - a) / sec


print("baseline (run only, w=4):")
base = server_rate(20)
print(f"  aggregate {base:.1f} tok/s\n")

for extra in (4, 8):
    counts = {}
    ths = [threading.Thread(target=worker, args=(counts, i)) for i in range(extra)]
    for t in ths: t.start()
    time.sleep(25)                      # let them ramp
    agg = server_rate(20)
    for t in ths: t.join()
    per = [v[0] for v in counts.values() if v[0]]
    errs = [v[1] for v in counts.values() if v[1]]
    print(f"+{extra} extra concurrent (total ~{4+extra}):")
    print(f"  aggregate {agg:.1f} tok/s   (baseline {base:.1f}, gain {agg-base:+.1f})")
    if per:
        print(f"  per-probe-request mean {sum(per)/len(per):.2f} tok/s")
    if errs:
        print(f"  errors: {errs[:2]}")
    print()

print("READ: if aggregate scales up materially, raise -w on the run.")
print("      if it is flat, the box is saturated and w=4 is already right.")
