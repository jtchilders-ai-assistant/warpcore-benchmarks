# Recovery provenance — `ornith15-swebench-n100-20260929`

## Scope

This record documents recovery of the **grading phase only**. Generation completed for all 100 frozen SWE-bench Verified instances before the original runner entered a terminal failed state. No inference was replayed and no prediction was changed.

## Immutable campaign identity

- Suite: `warpcore-v2`
- Frozen inventory: `suite/swebench/instances-seed42-n100.json`
- Inventory SHA-256: `d46f257dc87482e432307a6e96d5ee2d14b372df0c43b413eba48a573c6d0a24`
- Model: `ornith-ai/Ornith-1.5-35B-A3B-FP8`
- Model revision: `fab11c26e2325a42f4b32da0249c819a0bade1b1`
- Serving image: `eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`
- vLLM: `0.29.1rc1.dev427+g0748d3bd5.d20260920`
- Generation harness: mini-swe-agent `2.4.6`, four workers
- Grading harness: SWE-bench `4.1.0`

## Initial failure

The production runner completed 100/100 trajectories and wrote `raw/preds.json`, then invoked grading with `sys.executable`. That interpreter did not contain the `swebench` package, so the grading subprocess failed. The failure is retained in `status.json` as the `failed` history entry at `2026-09-30T23:40:35Z`.

The runner defect was corrected so grading uses the campaign's configured `--python-executable`, matching generation and preflight. Regression coverage is in `tests/test_task6_hardening.py`.

## Recovery procedure

Official grading was rerun against the unchanged `raw/preds.json` (SHA-256 `a6c4c9c0baa42e7364aa6592beab9e9dfe937cd01c580abcdf96d6128730b1e0`) with:

- interpreter: `/Users/jchilders/swebench-run/venv/bin/python`
- package: SWE-bench `4.1.0`
- dataset: `princeton-nlp/SWE-bench_Verified`
- run ID: `ornith15-swebench-n100-20260929-recovery`

The original grader report is retained as `raw/hosted_vllm__ornith-ai__Ornith-1.5-35B-A3B-FP8.ornith15-swebench-n100-20260929-recovery.json`; the complete grader log is `raw/grading-recovery.log`. Its partitions were normalized without changing membership into `raw/grading_results.json` for the repository validator and publisher.

## Reconciliation

All 100 frozen IDs have exactly one official grading disposition:

- resolved: 59
- unresolved: 10
- empty patch: 31
- grading error: 0
- incomplete: 0

Generation evidence independently records:

- `Submitted`: 69
- `LimitsExceeded`: 29
- `RepeatedFormatError`: 2
- `RuntimeError`: 0

Thus the headline capability metric is **59/100 resolved (59.0%)** on the full frozen denominator. Submission reliability is reported separately as **69/100 nonempty patches**. The 31 non-submissions are not excluded from the capability denominator.

## Lifecycle treatment

`status.json` is append-only. It retains the initial `failed` transition and adds an explicit marked recovery to `completed`; ordinary terminal-state transitions remain forbidden. Publication validation must pass before the separate `validated` transition is appended.
