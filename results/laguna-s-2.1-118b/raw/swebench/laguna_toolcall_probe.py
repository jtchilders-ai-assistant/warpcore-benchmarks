"""Gate before any SWE-bench run: does Laguna emit a NATIVE tool call?

Per the swebench-vllm-endpoint skill, poolside models are tool-call-native and score
0/4 on the bash-in-content (backticks) scaffold. Laguna is served WITH
--enable-auto-tool-choice --tool-call-parser poolside_v1, so mini-swe-agent's DEFAULT
tool-calling config is the right scaffold -- but prove it before committing hours.

Want: finish_reason == 'tool_calls' and a well-formed bash command in the arguments.
"""
import json, urllib.request

BASE = "http://csi370295.alcf.anl.gov:8000/v1/chat/completions"
MODEL = "poolside/Laguna-S-2.1-NVFP4"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a software engineer fixing a bug in a git repository. Use the bash tool to inspect the repo."},
        {"role": "user", "content": "There is a failing test in the django repository at /testbed. Start by listing the files in the current directory."},
    ],
    "tools": [BASH_TOOL],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 2048,
}

req = urllib.request.Request(
    BASE, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)

ch = d["choices"][0]
msg = ch["message"]
print("finish_reason :", ch.get("finish_reason"))
print("content       :", repr((msg.get("content") or ""))[:200])
print("reasoning     :", repr((msg.get("reasoning") or ""))[:200])
print("reasoning_cont:", repr((msg.get("reasoning_content") or ""))[:120])

tcs = msg.get("tool_calls") or []
print("tool_calls    :", len(tcs))
for t in tcs:
    fn = t.get("function", {})
    print("  name:", fn.get("name"))
    raw = fn.get("arguments")
    print("  raw args:", repr(raw)[:300])
    try:
        args = json.loads(raw)
        print("  PARSED OK -> command:", repr(args.get("command"))[:200])
    except Exception as e:
        print("  !! ARGUMENTS DO NOT PARSE AS JSON:", e)

print()
if ch.get("finish_reason") == "tool_calls" and tcs:
    print("VERDICT: native tool-calling WORKS -> use mini-swe-agent DEFAULT swebench.yaml")
else:
    print("VERDICT: NO tool call emitted -> investigate before running SWE-bench")
