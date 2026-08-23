import json, urllib.request, concurrent.futures, collections, sys

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"

QS = [
 "Q: Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?\nA:",
 "Q: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
 "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA:",
 "Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA:",
]

def one(i):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": QS[i % len(QS)]}],
            "max_tokens": 8192, "temperature": 0}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        ch = d["choices"][0]
        c = ch["message"].get("content") or ""
        rc = ch["message"].get("reasoning_content") or ""
        return (ch.get("finish_reason"), len(c), len(rc),
                d["usage"]["completion_tokens"])
    except Exception as e:
        return ("EXC:" + type(e).__name__, -1, -1, -1)

conc = int(sys.argv[1]); n = int(sys.argv[2])
with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
    res = list(ex.map(one, range(n)))

empty = [r for r in res if r[1] == 0]
fr = collections.Counter(r[0] for r in res)
print("concurrency=%d  n=%d" % (conc, n))
print("  finish_reasons:", dict(fr))
print("  EMPTY content: %d/%d (%.1f%%)" % (len(empty), n, 100*len(empty)/n))
if empty:
    print("  empty samples (finish_reason, content_len, reasoning_len, completion_tokens):")
    for e in empty[:6]:
        print("   ", e)
nonempty = [r for r in res if r[1] > 0]
if nonempty:
    print("  non-empty completion_tokens: min=%d max=%d" % (
        min(r[3] for r in nonempty), max(r[3] for r in nonempty)))
