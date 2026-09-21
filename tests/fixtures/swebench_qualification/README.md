# SWE-bench qualification test fixture

**This is synthetic test data, not measurement evidence.** No model produced it, no
SWE-bench harness graded it, and nothing here may be cited, published, or copied into
`results/`. It exists only so `tests/test_swebench_qualification.py` has a
production-shaped positive fixture to assert against and to mutate into negatives.

`run/` mirrors the on-disk layout a real 20-instance qualification run leaves behind:

```
run/raw/preds.json                       20 predictions, each a nonempty unified diff
run/raw/exit_statuses.json               20 terminal dispositions, all "Submitted"
run/raw/trajectories/<id>.traj           per-instance trajectory with well-formed tool calls
run/raw/gpt-oss-120b.qual-….json         official SWE-bench schema-v2 grader report
```

The instance IDs are the real suite-owned qualification set
(`suite/swebench/qualification-ids-v1.json`), so a test that mutates the fixture is
exercising the same ID authority the gate enforces in production. The `gpt-oss-120b`
prefix on the report filename only reproduces the harness's
`<model>.<run_id>.json` naming convention; it makes no claim about that model.

Regenerating: the fixture is plain JSON and is edited in place. If the suite-owned
qualification set changes (a suite-version change), the per-instance files here must be
regenerated to match, or `TestProductionPositiveFixture` will fail — which is the
intended signal, not a nuisance.
