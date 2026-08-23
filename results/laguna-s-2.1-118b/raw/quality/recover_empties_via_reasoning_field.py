"""Recover the GSM8K score by re-serving the 186 empty-content questions and reading
the `reasoning` field (where vLLM actually put the answer) instead of `content`.

Applies the SAME grading filters as the clean task:
  answer-line       : last  "[Tt]he answer is  <num>"
  flexible-fallback : last  number anywhere
"""
import json, urllib.request, gzip, glob, re, concurrent.futures, collections, sys

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"
STOP = ["</s>", "\n\nQ:"]

ANS_RE = re.compile(r"[Tt]he answer is\s*\$?\s*(-?[0-9][0-9,]*)")
NUM_RE = re.compile(r"(-?[0-9][0-9,]*)")

def norm(s):
    return s.replace(",", "").replace("$", "").replace(".", "").strip() if s else None

def grade(text, target):
    a = ANS_RE.findall(text or "")
    al = norm(a[-1]) if a else None
    n = NUM_RE.findall(text or "")
    fl = norm(n[-1]) if n else None
    t = norm(str(target))
    return (al == t), (fl == t)

def load(pattern):
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
        did = r.get("doc_id"); resp = r.get("resps") or []
        t = ""
        if resp:
            f0 = resp[0]; t = f0[0] if isinstance(f0, list) and f0 else (f0 if isinstance(f0, str) else "")
        e = per.setdefault(did, {"text": "", "args": r.get("arguments"), "target": r.get("target")})
        if t.strip(): e["text"] = t
    return per

def call(msgs):
    body = {"model": MODEL, "messages": msgs, "max_tokens": 8192,
            "temperature": 0, "stop": STOP}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=1800))
            m = d["choices"][0]["message"]
            return (m.get("content") or "") or (m.get("reasoning") or "")
        except Exception:
            continue
    return ""

pattern = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
per = load(pattern)
empties = [(d, e) for d, e in sorted(per.items()) if not e["text"].strip()][:limit]
print("re-serving %d empty-content questions, reading the `reasoning` field\n" % len(empties))

def work(item):
    did, e = item
    a0 = e["args"]["gen_args_0"]["arg_0"]
    if isinstance(a0, list) and a0: a0 = a0[0]
    msgs = json.loads(a0)
    txt = call(msgs)
    al, fl = grade(txt, e["target"])
    return (did, len(txt), al, fl)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(work, empties))

got   = sum(1 for r in res if r[1] > 0)
al_ok = sum(1 for r in res if r[2])
fl_ok = sum(1 for r in res if r[3])
n = len(res)
print("recovered non-empty text : %d/%d (%.1f%%)" % (got, n, 100*got/max(1,n)))
print("of the %d recovered, correct:" % n)
print("   answer-line       : %d (%.1f%%)" % (al_ok, 100*al_ok/max(1,n)))
print("   flexible-fallback : %d (%.1f%%)" % (fl_ok, 100*fl_ok/max(1,n)))
json.dump([{"doc_id": r[0], "chars": r[1], "answer_line": r[2], "flex": r[3]} for r in res],
          open("/tmp/recovery_results.json", "w"), indent=1)
print("\nwrote /tmp/recovery_results.json")
