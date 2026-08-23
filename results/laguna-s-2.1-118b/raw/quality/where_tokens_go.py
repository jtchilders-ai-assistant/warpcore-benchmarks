"""Where do the generated tokens go? Compare the CHAT endpoint (parsed) against the
COMPLETIONS endpoint (raw, no chat template, no reasoning parser) for the same prompt.

If the raw completions endpoint returns real text while chat returns empty content,
the tokens are being swallowed by the chat-layer parsing (reasoning/tool-call parser),
not by the model."""
import json, urllib.request, gzip, glob, sys

MODEL = "poolside/Laguna-S-2.1-NVFP4"

def post(path, body):
    req = urllib.request.Request("http://localhost:8000" + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

# grab one prompt that reproduced empty
rows = []
for f in glob.glob("/tmp/lmeval_bisect/**/samples_*.jsonl*", recursive=True):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except Exception: pass
per = {}
for r in rows:
    did = r.get("doc_id"); resp = r.get("resps") or []
    t = ""
    if resp:
        f0 = resp[0]; t = f0[0] if isinstance(f0, list) and f0 else (f0 if isinstance(f0, str) else "")
    e = per.setdefault(did, {"text": "", "args": r.get("arguments")})
    if t.strip(): e["text"] = t

msgs = None
for did, e in sorted(per.items()):
    if not e["text"].strip() and e["args"]:
        a0 = e["args"]["gen_args_0"]["arg_0"]
        if isinstance(a0, list) and a0: a0 = a0[0]
        msgs = json.loads(a0)
        print("using doc_id", did)
        break
if not msgs:
    sys.exit("no empty prompt found")

user_text = msgs[0]["content"]

print("\n=== 1. CHAT endpoint (what lm-eval uses) ===")
d = post("/v1/chat/completions", {"model": MODEL, "messages": msgs,
                                  "max_tokens": 8192, "temperature": 0})
ch = d["choices"][0]; m = ch["message"]
print("  finish_reason :", ch.get("finish_reason"))
print("  completion_tok:", d["usage"]["completion_tokens"])
print("  content len   :", len(m.get("content") or ""))
print("  reasoning len :", len(m.get("reasoning_content") or ""))
print("  message keys  :", sorted(m.keys()))
print("  tool_calls    :", json.dumps(m.get("tool_calls"))[:400])
print("  RAW message   :", json.dumps(m)[:800])

print("\n=== 2. COMPLETIONS endpoint (raw, no chat parsing) ===")
try:
    d2 = post("/v1/completions", {"model": MODEL, "prompt": user_text,
                                  "max_tokens": 512, "temperature": 0})
    c2 = d2["choices"][0]
    print("  finish_reason :", c2.get("finish_reason"))
    print("  completion_tok:", d2["usage"]["completion_tokens"])
    print("  text len      :", len(c2.get("text") or ""))
    print("  text head     : %r" % (c2.get("text") or "")[:400])
except Exception as e:
    print("  completions endpoint error:", type(e).__name__, e)
