import json, urllib.request, gzip, glob, concurrent.futures, collections, sys

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"
STOP = ["</s>", "\n\nQ:"]

def load_empty_prompts(pattern, limit=40):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except Exception: pass
    per = {}
    for r in rows:
        did = r.get("doc_id")
        resp = r.get("resps") or []
        t = ""
        if resp:
            f0 = resp[0]
            t = f0[0] if isinstance(f0, list) and f0 else (f0 if isinstance(f0, str) else "")
        ent = per.setdefault(did, {"text": "", "args": r.get("arguments")})
        if t.strip():
            ent["text"] = t
    out = []
    for did, e in sorted(per.items()):
        if e["text"].strip() or not e["args"]:
            continue
        try:
            a0 = e["args"]["gen_args_0"]["arg_0"]
            if isinstance(a0, list) and a0:
                a0 = a0[0]
            msgs = json.loads(a0) if isinstance(a0, str) else a0
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
                out.append((did, msgs))
        except Exception as ex:
            pass
        if len(out) >= limit:
            break
    return out

def call(msgs, stop):
    body = {"model": MODEL, "messages": msgs, "max_tokens": 8192, "temperature": 0}
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        ch = d["choices"][0]; m = ch["message"]
        return (ch.get("finish_reason"), len(m.get("content") or ""),
                len(m.get("reasoning_content") or ""), d["usage"]["completion_tokens"])
    except Exception as e:
        return ("EXC:" + type(e).__name__, -1, -1, -1)

pattern = "/tmp/lmeval_bisect/**/samples_*.jsonl*"
items = load_empty_prompts(pattern, int(sys.argv[1]) if len(sys.argv) > 1 else 20)
print("replaying %d previously-EMPTY questions (from bisect run)\n" % len(items))
if not items:
    sys.exit("no prompts extracted")

for label, stop in [("WITH lm-eval stop strings", STOP), ("WITHOUT stop strings", None)]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(lambda it: call(it[1], stop), items))
    empty = [r for r in res if r[1] == 0]
    empty_with_reasoning = [r for r in empty if r[2] > 0]
    print("--- %s ---" % label)
    print("   finish_reasons : %s" % dict(collections.Counter(r[0] for r in res)))
    print("   EMPTY content  : %d/%d (%.1f%%)" % (len(empty), len(res), 100*len(empty)/max(1,len(res))))
    print("   ...of those, reasoning_content populated: %d" % len(empty_with_reasoning))
    for e in empty[:5]:
        print("      (finish=%s content=%d reasoning=%d tokens=%d)" % e)
    print()
