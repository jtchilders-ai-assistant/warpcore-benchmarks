"""Measure finish_reason + generation-length distribution FOR THE SWE-BENCH WINDOW ONLY.

Why: I previously quoted 'finished_reason=length is 46%' from CUMULATIVE counters. Those
counters span the whole server lifetime -- including the GSM8K / IFEval / GPQA lm-eval runs,
which used much smaller max_tokens. Using them to characterise SWE-bench behaviour is wrong.
This samples DELTAS over a live window so the numbers describe the current run only.
"""
import subprocess, time, re

def metrics():
    out = subprocess.run(
        ["ssh", "warpcore", "curl -s -m10 http://localhost:8000/metrics"],
        capture_output=True, text=True, timeout=90).stdout
    d = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^(vllm:[a-z_]+)\{([^}]*)\}\s+([0-9.e+]+)$', line)
        if not m:
            continue
        name, labels, val = m.groups()
        key = None
        if name == "vllm:request_success_total":
            fr = re.search(r'finished_reason="([^"]+)"', labels)
            if fr: key = f"finish:{fr.group(1)}"
        elif name == "vllm:request_generation_tokens_bucket":
            le = re.search(r'le="([^"]+)"', labels)
            if le: key = f"bucket:{le.group(1)}"
        if key:
            d[key] = float(val)
    return d

WINDOW = 100
a = metrics()
time.sleep(WINDOW)
b = metrics()

print(f"=== DELTAS over {WINDOW}s of the live SWE-bench run ===\n")

fin = {k[7:]: b.get(k, 0) - a.get(k, 0) for k in b if k.startswith("finish:")}
tot_f = sum(fin.values())
print("finish reasons:")
for k, v in sorted(fin.items(), key=lambda x: -x[1]):
    pct = f"{v/tot_f*100:5.1f}%" if tot_f else "  n/a"
    print(f"  {k:<12} {v:6.0f}  {pct}")
print(f"  TOTAL        {tot_f:6.0f} completions in {WINDOW}s")

buckets = sorted(((float(k[7:]) if k[7:] != "+Inf" else float("inf")),
                  b.get(k, 0) - a.get(k, 0)) for k in b if k.startswith("bucket:"))
print("\ngeneration-length distribution (non-cumulative):")
prev_le, prev_c = 0, 0
for le, c in buckets:
    band = c - prev_c
    if band > 0:
        lo = int(prev_le)
        hi = "inf" if le == float("inf") else int(le)
        print(f"  {lo:>6} - {str(hi):>6} tok : {band:5.0f}")
    prev_le, prev_c = le, c

if tot_f:
    lp = fin.get("length", 0) / tot_f * 100
    print(f"\n=> {lp:.1f}% of SWE-bench completions hit the max_tokens cap")
    print("   Each capped step produces NO tool call and is 100% wasted work.")
    print(f"   At ~8.8 tok/s a 32768-token wasted step burns ~{32768/8.8/60:.0f} min.")
