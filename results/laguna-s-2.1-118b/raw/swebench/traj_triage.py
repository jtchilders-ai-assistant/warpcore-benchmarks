"""Trajectory triage for a native tool-calling SWE-bench run.

Per the swebench-vllm-endpoint skill:
  - extract commands from message.tool_calls[].function.arguments (JSON .command),
    NOT from message.content -- tool-call-native models leave content empty and a
    content-based repeat counter reports a huge FAKE loop.
  - a valid submission starts with 'diff --git' / '--- a/'; anything else means the
    submit-capture broke (scaffold bug), not a model failure.
"""
import json, glob, collections, os, sys

pat = sys.argv[1] if len(sys.argv) > 1 else "/tmp/laguna_swe_smoke/*/*.traj.json"

for tj in sorted(glob.glob(pat)):
    inst = os.path.basename(os.path.dirname(tj))
    d = json.load(open(tj))
    msgs = d.get("messages", [])
    info = d.get("info", {}) or {}

    cmds = []
    for m in msgs:
        for t in (m.get("tool_calls") or []):
            try:
                cmds.append(json.loads(t["function"]["arguments"]).get("command", "")[:80])
            except Exception:
                cmds.append("<UNPARSEABLE ARGS>")

    parse_err = sum(1 for m in msgs
                    if "Error parsing tool call arguments" in (m.get("content") or ""))
    execs = sum(1 for m in msgs if "returncode" in (m.get("content") or ""))
    sub = (info.get("submission") or "")
    valid = sub.lstrip().startswith(("diff --git", "--- a/"))

    print(f"=== {inst}")
    print(f"  exit_status     : {info.get('exit_status')}")
    print(f"  tool calls      : {len(cmds)} total / {len(set(cmds))} distinct")
    if cmds:
        top = collections.Counter(cmds).most_common(2)
        print(f"  most repeated   : {top}")
    print(f"  real executions : {execs}")
    print(f"  tool-arg JSON errors: {parse_err}")
    print(f"  submission valid: {valid}  (len={len(sub)})")
    print(f"  submission head : {sub[:80]!r}")
    print()
