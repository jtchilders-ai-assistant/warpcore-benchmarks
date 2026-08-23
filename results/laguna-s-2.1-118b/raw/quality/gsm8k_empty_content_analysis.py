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

# Keep one row per (doc_id, filter) -> collapse to per-doc using the answer-line filter
per_doc = {}
for r in rows:
    did = r.get("doc_id")
    filt = r.get("filter")
    resp = r.get("resps") or []
    txt = ""
    if resp:
        f0 = resp[0]
        txt = f0[0] if isinstance(f0, list) and f0 else (f0 if isinstance(f0, str) else "")
    ent = per_doc.setdefault(did, {"text": txt, "scores": {}})
    ent["scores"][filt] = r.get("exact_match")
    if txt.strip():
        ent["text"] = txt

n = len(per_doc)
empty = [d for d, e in per_doc.items() if not e["text"].strip()]
nonempty = [d for d, e in per_doc.items() if e["text"].strip()]

def acc(ids, filt):
    vals = [per_doc[d]["scores"].get(filt) for d in ids]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals) * 100) if vals else float("nan"), len(vals)

print("unique docs:", n)
print("empty-content docs:", len(empty), "(%.1f%%)" % (100*len(empty)/n))
print()
for filt in ("answer-line", "flexible-fallback"):
    a_all, n_all = acc(list(per_doc), filt)
    a_ne, n_ne = acc(nonempty, filt)
    print("%-18s overall %.2f%% (n=%d)   |  excluding empty-content: %.2f%% (n=%d)"
          % (filt, a_all, n_all, a_ne, n_ne))
print()
print("=> If the empties were served correctly, the score would land near the")
print("   'excluding empty-content' column. The gap is the serving-loss, not capability.")
