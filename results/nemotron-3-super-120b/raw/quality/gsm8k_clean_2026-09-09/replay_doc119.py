#!/usr/bin/env python3
import json, glob, urllib.request
from pathlib import Path
base=Path.home()/"warpcore-benchmark-runs/nemotron-super-gsm8k-clean-20260909"
p=glob.glob(str(base/"results/**/samples_*.jsonl"),recursive=True)[0]
rows=[json.loads(x) for x in open(p) if x.strip()]
r=next(x for x in rows if x.get("doc_id")==119)
arg=r["arguments"]["gen_args_0"]
prompt=json.loads(arg["arg_0"][0])
params=dict(arg["arg_1"])
params.pop("do_sample",None)
params.update(messages=prompt,model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",stream=False)
params["max_tokens"]=params.pop("max_gen_toks")
req=urllib.request.Request("http://csi370295.alcf.anl.gov:8000/v1/chat/completions",data=json.dumps(params).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=1800) as f: d=json.load(f)
json.dump({"doc_id":119,"request":params,"response":d},open(base/"doc119_replay.json","w"),indent=2)
c=d["choices"][0]
summary={"finish_reason":c.get("finish_reason"),"content":c["message"].get("content"),"reasoning_len":len(c["message"].get("reasoning") or ""),"usage":d.get("usage")}
json.dump(summary,open(base/"doc119_replay_summary.json","w"),indent=2)
(base/"REPLAY_DONE").write_text("ok\n")
print(json.dumps(summary))
