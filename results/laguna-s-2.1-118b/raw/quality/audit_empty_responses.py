"""Audit every retained lm-eval sample file for the empty-content defect.
Reports the empty rate per task per model so we can tell which PUBLISHED scores
in this repo are silently depressed."""
import json, glob, gzip, collections

files = sorted(glob.glob("/tmp/lmeval_results/**/samples_*.jsonl*", recursive=True))
print("%-58s %-8s %-8s %s" % ("file", "docs", "empty", "empty%"))
print("-" * 90)
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    rows = []
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
        if did not in per or not per[did]:
            per[did] = t
        elif t.strip():
            per[did] = t
    n = len(per)
    em = sum(1 for v in per.values() if not (v or "").strip())
    short = "/".join(f.split("/")[-3:])[:56]
    print("%-58s %-8d %-8d %.1f%%" % (short, n, em, 100*em/max(1, n)))
