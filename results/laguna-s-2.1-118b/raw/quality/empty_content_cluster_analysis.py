import json, glob, gzip, collections

rows = []
for f in glob.glob("/tmp/lmeval_results/laguna/gsm8k/**/samples_*.jsonl*", recursive=True):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except Exception: pass

# collapse to one entry per doc (answer-line filter view)
per_doc = {}
for r in rows:
    did = r.get("doc_id")
    resp = r.get("resps") or []
    txt = ""
    if resp:
        f0 = resp[0]
        txt = f0[0] if isinstance(f0, list) and f0 else (f0 if isinstance(f0, str) else "")
    e = per_doc.setdefault(did, {"text": ""})
    if txt.strip():
        e["text"] = txt

ids = sorted(per_doc)
empty_flags = [(d, not per_doc[d]["text"].strip()) for d in ids]

n = len(ids); nempty = sum(1 for _, e in empty_flags if e)
print("docs=%d empty=%d (%.1f%%)" % (n, nempty, 100*nempty/n))

# Are empties clustered in time (doc_id order ~ submission order)? Bucket by decile.
print("\nempty rate by doc_id decile (submission order):")
B = 10
for b in range(B):
    lo, hi = b*n//B, (b+1)*n//B
    seg = empty_flags[lo:hi]
    k = sum(1 for _, e in seg if e)
    bar = "#" * int(40*k/max(1,len(seg)))
    print("  docs %5d-%5d: %5.1f%%  %s" % (lo, hi-1, 100*k/max(1,len(seg)), bar))

# longest consecutive run of empties -> burst signature
best = cur = 0
for _, e in empty_flags:
    cur = cur+1 if e else 0
    best = max(best, cur)
print("\nlongest consecutive empty run: %d" % best)
