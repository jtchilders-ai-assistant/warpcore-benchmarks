#!/usr/bin/env python3
# Replay the 41 GPQA items that truncated at 32k, now at a 64k budget.
# Measure: completion_tokens needed, finish_reason, and correctness.
# Answers "how big a budget does Lightning actually need for GPQA, and is it realistic?"
import json, re, sys, time, urllib.request

PROMPTS = json.load(open("/tmp/trunc_prompts.json"))
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
BUDGET = 65536

def norm(x): 
    m = re.findall(r"[A-Da-d]", str(x))
    return m[-1].upper() if m else None

def ask(messages):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0, "max_tokens": BUDGET}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)

rows=[]
for i,item in enumerate(PROMPTS):
    t0=time.time()
    try:
        d=ask(item["messages"])
        c=d["choices"][0]; content=c["message"].get("content") or ""
        fr=c["finish_reason"]; ct=d["usage"]["completion_tokens"]
        # extract answer letter
        m=re.findall(r"[Tt]he answer is \(?([A-D])\)?", content)
        pred=m[-1] if m else norm(content[-20:])
        gold=norm(item["target"])
        correct = (pred==gold)
        rows.append(dict(doc=item["doc_id"], fr=fr, ct=ct, pred=pred, gold=gold, correct=correct))
        print(f"[{i+1}/41] ct={ct} fr={fr} pred={pred} gold={gold} {'OK' if correct else 'x'} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        rows.append(dict(doc=item["doc_id"], fr="ERROR", ct=None, pred=None, gold=norm(item["target"]), correct=False))
        print(f"[{i+1}/41] ERROR {e}", flush=True)

json.dump(rows, open("/tmp/trunc_replay_64k.json","w"), indent=2)
fin=[r for r in rows if r["fr"]=="stop"]
still=[r for r in rows if r["fr"]=="length"]
cts=sorted(r["ct"] for r in fin if r["ct"])
print("\n===== SUMMARY (41 items that truncated at 32k, replayed at 64k) =====")
print(f"now finished (stop): {len(fin)}   still truncated (length): {len(still)}")
if cts:
    print(f"completion_tokens of newly-finished: min={cts[0]} p50={cts[len(cts)//2]} p90={cts[int(len(cts)*0.9)]} max={cts[-1]}")
print(f"correct among newly-finished: {sum(r['correct'] for r in fin)}/{len(fin)}")
open("/tmp/trunc_replay_64k.DONE","w").close()
