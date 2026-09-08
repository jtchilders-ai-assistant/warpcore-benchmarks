# Working in this repo

Benchmark results for models served on **warpcore** (NVIDIA DGX Spark, GB10,
aarch64) at `http://csi370295.alcf.anl.gov:8000/v1`.

This repo is an **evidence archive**, not a notebook. Every published number must
trace to a committed artifact. The rules below exist because each one was learned
by losing GPU-hours or publishing a wrong number.

- `TODO.md` — the backlog (what still needs doing)
- `LESSONS.md` — the seven failure classes, with measurements
- `PROVENANCE.md` — what artifacts a run must leave behind
- `ISSUES.md` — serving bugs, notably #15
- `HARDWARE.md` — GB10 sizing rules

---

## Before launching any quality run

```bash
make quality-preflight MODE=live ENDPOINT=http://csi370295.alcf.anl.gov:8000/v1 \
  MODEL=<exact-model-id> MAX_GEN_TOKS=<budget> AGGREGATE_TOK_S=<measured> \
  CONCURRENCY=<workers> CLIENT_TIMEOUT=<seconds>
```

Exit `0` usable · `1` defect, do not launch · `2` could not probe, **also** do
not launch. Exit 2 is deliberately not success: an unprobeable endpoint must
never read as a pass.

**Why this exists.** vLLM's reasoning parser can fail to initialize and return
`content: null` with the real answer stranded in `message.reasoning`. lm-eval
reads only `content`, so the item scores **0** — no error, no retry,
`finish_reason: "stop"`. Lightning GPQA published **53.03%** when the served-only
rate was **90.52%**; 82 of 198 items were empty. The warning that predicts this
was already in the run log and nothing was watching for it.

**Empty content alone does NOT mean a broken parser.** Measured on live warpcore
2026-09-03 (`RedHatAI/Muse-Glimmer-30B-NVFP4`): the same healthy model returns
`content=None, finish=length` at `max_tokens=64` and `content='4', finish=stop`
at 512 — it needs ~109 completion tokens before emitting any answer. Assuming
emptiness means corruption would block working endpoints. Classify on
`finish_reason`:

| signature | verdict |
| --- | --- |
| `finish=length` + empty | BUDGET — raise `max_gen_toks`, parser is fine |
| `finish=stop` + empty + `reasoning` populated | PARSER — ISSUES #15 |
| `finish=stop` + empty + nothing anywhere | EMPTY — nothing generated |

vLLM emits `reasoning`, **not** the conventional `reasoning_content`. A probe
checking only the latter sees nothing and wrongly concludes output vanished.

## Size the time budget with arithmetic, not intuition

A GPQA run was abandoned at **110/198 after ~13 hours**. The engine was healthy;
the client timeout was simply smaller than the work: at c=16 and ~4 tok/s per
request, a ~32k-token answer needs ~2 h, against `--timeout 3600` (1 h). Every
long item timed out, **retried from scratch**, and hit the same wall — 352
timeout/retry events. It could never have converged.

Probe a few items, extrapolate, and refuse to launch if the tail doesn't fit.
The automated gate uses measured aggregate throughput divided by concurrency and a default 2×
safety factor. That is a conservative screening estimate, not a substitute for tail data: when a
p90 per-request generation rate is available, pass an equally conservative aggregate equivalent or
raise `SAFETY_FACTOR`. Gate on **p90, not the mean** — that was a tail failure a mean would hide.
Existing timeouts are ad-hoc (`3600 / 14400 / 30000`); don't copy one blindly.

---

## Reproducing figures

The toolchain is **pinned exactly** (`requirements-viz.txt`: matplotlib 3.9.4,
numpy 2.0.2). Byte-identical SVG output is the repo standard and matplotlib does
not guarantee it across minors — CI once installed 3.11.1 against figures
rendered with 3.9.4 and `make check` failed with ~5,400 changed SVG lines while
every derived CSV was byte-identical. The numbers were fine; only rendering
differed.

**Trap:** `PYTHON ?= python3` picks up whatever is first on `PATH`. An unrelated
virtualenv active in the shell will shadow the system interpreter and `make ci`
dies with `ModuleNotFoundError: No module named 'matplotlib'` — which looks like
a repo regression and is not. On this Mac the working interpreter is
`/usr/bin/python3` (3.9.6, matplotlib 3.9.4). Check `which python3` first, or:

```bash
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
```

If you bump the pins, regenerate and commit the figures in the **same commit**,
or `make check` goes red for everyone.

## Finish with `make ci`

Same target CI runs, so local green == CI green:

```
make check            figures reproduce from committed artifacts
make check-artifacts  every number has its artifact (ratcheted)
make preflight-selftest  classifier fixtures, no GPU
make samples          silent-zero detector (warn-only, on purpose)
```

`check-artifacts` is a **ratchet**: 17 known gaps are accepted in
`viz/data/provenance_baseline.json`; a *new* gap exits 1. `STRICT=1` fails on any
gap — the end goal once the backlog clears.

`make samples` is **warn-only in CI deliberately**: three committed Lightning
tasks already breach the 2% threshold (GPQA 41.4%, GPQA-32k 20.7%, IFEval 8.7%).
Failing today would wedge CI red on documented debt. Flipping it to a hard gate
is the definition of done for the ISSUES #15 re-serve backlog (TODO 6i).

Historical runs often retained only the slim `*.per_item.csv`; the validator reads that fallback as
well as newly committed `samples_*.jsonl.gz`. Without the CSV fallback, CI audited 2 of 6 historical
tasks and reported a false all-clear. New runs must retain compressed sample JSONL, and any new
sample-reading check still needs the CSV path for legacy evidence or it will silently pass.

---

## Reporting rules

**Publish the denominator.** Report attempts, not the nominal instance count.
Qwen3.6 lost **22/100** SWE-bench instances to a 120 s Docker pull timeout on a
cold cache — the model was never invoked, so those measure image-pull throughput,
not capability.

**A fair rate explains *why* a model missed. It cannot rank two models.** Two
models' fair rates sit on different populations, so comparing them head to head
answers no single question. This file used to sanction exactly that, and the
error propagated into the README: one model's exclusions were dropped from its
denominator while the other's stayed at the full set, and the resulting "tie" was
an artifact of the mismatch. Excluded instances are not a random subset — the
other model solved half of them. Use a fair rate to characterise a single model's
failures, never to rank.

**Compare paired.** Every model runs the identical seed-42 instance set, so any
two are a paired design: use McNemar on the discordant pairs, not a comparison of
two rates. `viz/swebench_paired.py` derives this from the committed artifacts
(`make data`); read the numbers there, not from prose. It refuses to compare two
models whose submitted sets differ.

**"Not distinguishable" is not "tied."** A non-significant result is a statement
about insufficient evidence, not about equality. Say which, and never promote the
second reading.

**Never argue significance from overlapping confidence intervals.** Two intervals
can overlap while a paired test is decisively significant; "they overlap, so
there is no difference" is a known fallacy, and it is what propped up the tie
above — on intervals that had been computed at different denominators besides.
For paired runs use McNemar, prefer the exact binomial at these counts, and
**name the convention**: an unqualified χ² is ambiguous between the Yates-corrected
and uncorrected values, and the two differ enough here to read as an error.

**Publish an interval with anything reweighted or subsetted.** Per-repo cells hold
4–10 instances, where a single problem moves a rate 10–25 pp and the bootstrap
intervals come out tens of points wide. A bare reweighted point estimate invites
the reader to treat a coin flip as a finding.

**One word, one statistic.** "Fair" means the infrastructure-adjusted denominator
from `viz/swebench_fair.py` and nothing else. The per-submission column in
LESSONS.md §3 is **"graded"** — it was once also labelled "fair", so `fair 55/75`
read as "55 of 75 submissions". Per-submission accuracy also drops model-side
failures that the rule below keeps in the denominator, so it flatters a model
that fails by giving up.

Exclude an instance **only** when it never received a test verdict *and* the
cause was infrastructure (timeout, 5xx, parser fault). Model-side outcomes —
step limit, context window, a wrong patch — stay in the denominator; they are
the model's own. Ground truth for "did it get a verdict?" is the results JSON
(`resolved_ids | unresolved_ids`), **never** an exit-status file: those can
cover a partial re-run segment. Reading one as though it covered all 100
instances once produced a fabricated 56.0% for Lightning — the file described 28
instances and 8 of its 9 `InternalServerError` entries had in fact been graded.
`viz/swebench_fair.py` derives this from committed artifacts (`make data`) and
marks a model with no exit-status artifact `attributed: false` rather than
silently adjusting it — Ornith is currently the one such case.

**A served-only rate is an upper bound, not a corrected score.** Scoring an empty
response 0 understates the model; excluding it overstates. On Laguna the
recovered items scored 90.3% vs 97.09% for served items — dropped items are
harder. Publish neither as a capability number; re-serve instead.

**Recompute statistics at the real n.** The README's ±9 pp Wilson interval
assumes n=100; at Laguna's fair n=75 it is wider.

**When an artifact is genuinely absent, emit `"unrecorded"`** and drop the entry
from the ranking. Do not backfill from a review, a summary, or memory — a number
in a plan is not an artifact.

**One known exception, kept visible.** Ornith's SWE-bench run has no exit-status
artifact (TODO 6a), so its ungraded instances are attributed from the card's run
log rather than from a committed file. `swebench_fair.py` marks it
`attributed: false` and it still appears in the fair column. That is tolerable
only because the flag rides with the data and no ranking claim depends on it —
the paired test needs no attribution at all. Either commit the artifact or drop
the entry; do not let a second such case appear without the same disclosure.

**Mind the vocabulary.** Repo prose calls empty-patch instances "non-submissions",
but the harness did receive a submission — an empty one; they land in
`empty_patch_ids` and `submitted_ids` still counts them. Say **ungraded** or
**empty patch**. Nor is `completed_ids` the verdict set: it can undercount when
an instance is re-graded after a harness timeout. `resolved_ids | unresolved_ids`
is the only ground truth for "did it get a verdict?", and `submitted_ids` is the
right pool for a paired comparison.

**Config files must match the run they document.** `launch_ornith.sh` declares
`--gpu-memory-utilization 0.90` while the SWE-bench run used **0.55**; both are
in the repo and you cannot tell from inside which is right. The runners are
hand-edited clones, so drift is structural — when changing one, diff it against
the historical command and explain every difference in the commit message.

## Repo hygiene

- Commit only on explicit approval from the maintainer.
- Negative controls: break it, prove exit 1, restore, prove exit 0. A guard that
  has never failed is not known to work. Mutation-test the classifier boundaries.
- Never leave artifacts moved or deleted after a test.
- `.hermes/` is gitignored agent scratch — nothing durable belongs there.
