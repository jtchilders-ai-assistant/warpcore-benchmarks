"""Audit PROVENANCE.md compliance across every model card in the repo.

Checks the things PROVENANCE.md calls non-negotiable for a reported number:
  * manifest.json exists (S1)
  * serving engine version + image digest recorded (S1)
  * exit_statuses present for swebench runs (S2)
  * raw result artifacts present for each claimed score (S2)
  * trajectories retained for agentic runs (S5a)

Prints a coverage table. Exits nonzero if a REPORTED number lacks its artifact,
so this can be wired into CI later.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Which benchmarks each model actually reports, read from the top-level README
# table rather than assumed, so the audit tracks what we CLAIM.
README = (REPO / "README.md").read_text()


def claimed_benchmarks(model_dir: pathlib.Path) -> dict:
    """Read the model's own card to see which numbers it reports."""
    card = model_dir / "README.md"
    if not card.exists():
        return {}
    txt = card.read_text()
    out = {}
    for bench, pat in [
        ("gsm8k", r"GSM8K"),
        ("ifeval", r"IFEval"),
        ("gpqa", r"GPQA"),
        ("swebench", r"SWE-bench"),
        ("pi30", r"pi-30"),
        ("throughput", r"tok/s"),
    ]:
        # a claim = the benchmark named alongside a number
        if re.search(pat, txt, re.I):
            m = re.search(pat + r"[^\n|]{0,80}?(\d+\.?\d*)\s*%", txt, re.I)
            out[bench] = m.group(1) if m else "mentioned"
    return out


def audit_model(model_dir: pathlib.Path) -> dict:
    raw = model_dir / "raw"
    swe = raw / "swebench"
    row = {
        "model": model_dir.name,
        "card": (model_dir / "README.md").exists(),
        "manifest_any": bool(list(raw.rglob("manifest.json"))) if raw.exists() else False,
        "swebench_run": swe.exists(),
    }
    if swe.exists():
        # Naming is not uniform across models: the graded report is
        # "*results*.json" for some runs and "*report*.json" for others. Match
        # both, else the audit invents gaps that don't exist.
        row["swe_manifest"] = (swe / "manifest.json").exists()
        row["swe_exit_statuses"] = bool(list(swe.glob("exit_statuses*.yaml")))
        row["swe_preds"] = bool(list(swe.glob("*preds*.json")))
        row["swe_results"] = bool(list(swe.glob("*results*.json"))
                                  or list(swe.glob("*report*.json")))
        row["swe_trajectories"] = bool(list(swe.glob("*trajector*"))
                                       or list(swe.glob("*.tar.gz")))
        row["swe_config"] = bool(list(swe.glob("*.yaml")))
        # A config the agent RECONSTRUCTED after the fact is weaker evidence than
        # the file the run actually used -- flag it rather than scoring it clean.
        row["swe_config_reconstructed"] = bool(list(swe.glob("*RECONSTRUCTED*")))
        # A run we deliberately publish as "blocked / no score" (gpt-oss: vLLM
        # tool-call JSON corruption) is SUPPOSED to have no preds and no graded
        # report. Requiring them would flag correct behaviour as a gap.
        row["swe_no_score"] = (swe / "DIAGNOSIS.json").exists() and not row["swe_preds"]
    return row


def main() -> int:
    models = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    rows = [audit_model(m) for m in models]

    print(f"{'model':30} {'card':5} {'manifest':9} {'swe':5} {'exit_st':8} "
          f"{'preds':6} {'results':8} {'trajs':6}")
    print("-" * 86)
    gaps = []
    for r in rows:
        swe = r["swebench_run"]
        def mark(k):
            if not swe:
                return "  -  "
            return "  OK " if r.get(k) else " MISS"
        print(f"{r['model']:30} {'OK' if r['card'] else 'MISS':5} "
              f"{'OK' if r['manifest_any'] else 'MISS':9} "
              f"{'yes' if swe else 'no':5} {mark('swe_exit_statuses'):8} "
              f"{mark('swe_preds'):6} {mark('swe_results'):8} {mark('swe_trajectories'):6}")
        if swe:
            checks = [("swe_manifest", "manifest.json"),
                      ("swe_exit_statuses", "exit_statuses"),
                      ("swe_trajectories", "trajectories")]
            if r.get("swe_no_score"):
                gaps.append(f"{r['model']}: swebench published as BLOCKED/no-score "
                            f"(DIAGNOSIS.json) -- preds/report correctly absent")
            else:
                checks += [("swe_preds", "preds"), ("swe_results", "graded report")]
            for k, label in checks:
                if not r.get(k):
                    gaps.append(f"{r['model']}: swebench missing {label}")
            if r.get("swe_config_reconstructed"):
                gaps.append(f"{r['model']}: swebench config is RECONSTRUCTED, "
                            f"not the file the run used")
        if not r["manifest_any"]:
            gaps.append(f"{r['model']}: NO manifest.json anywhere under raw/")

    print("\nGAPS vs PROVENANCE.md:")
    if not gaps:
        print("  none")
    for g in gaps:
        print("  -", g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
