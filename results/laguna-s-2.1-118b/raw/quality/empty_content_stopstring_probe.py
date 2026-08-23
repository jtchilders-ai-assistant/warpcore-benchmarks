import json, urllib.request

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"

# Exact lm-eval prompt for a doc that came back EMPTY (doc_id 19).
PROMPT = ("Q: Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, "
          "then another hour to walk the next two miles. If she wants her average speed to be "
          "4 miles per hour, what speed (in miles per hour) does she need to walk the remaining "
          "distance?\n\nSolve this step by step. Then, on the final line, give your answer in "
          "EXACTLY this format:\nThe answer is <number>\nwhere <number> is a plain integer with "
          "no commas, units, currency symbols, or formatting.\nA:")

def run(label, stop):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 8192, "temperature": 0}
    if stop is not None:
        body["stop"] = stop
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = d["choices"][0]
    c = ch["message"].get("content") or ""
    print("--- %s ---" % label)
    print("   stop param      :", stop)
    print("   finish_reason   :", ch.get("finish_reason"))
    print("   completion_tok  :", d["usage"]["completion_tokens"])
    print("   content chars   :", len(c))
    print("   first 160 chars : %r" % c[:160])
    print()

# A: exactly what lm-eval sends (until -> stop)
run("WITH lm-eval stop strings", ["</s>", "\n\nQ:"])
# B: identical request, no stop strings
run("WITHOUT stop strings", None)
