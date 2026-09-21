"""tests/test_swebench_circuit_breaker.py — fail-closed SWE-bench campaign breaker.

TDD: written BEFORE viz/swebench_circuit_breaker.py exists. RED until implemented.

Why this exists
---------------
The 2026-09-17 Qwen SWE-bench campaign burned a full run while every instance
recorded RuntimeError; the 2026-08-04 run lost 22/100 to a docker pull timeout.
Both were decidable from the first few completed instances. This breaker watches
the live mini-swe-agent progress YAML and aborts generation once partial evidence
already proves a systemic parser/serving/infrastructure failure.

What it must NOT do
-------------------
Model/config operational outcomes — step limit, cost limit, context window,
a wrong patch, a plain format mistake — stay in the full denominator and must
never trip the breaker (design §10.2). A partially written progress YAML is not
evidence. Only the exact owned generation process group is ever signalled.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_swebench  # noqa: E402
import swebench_circuit_breaker as cb  # noqa: E402  (expected ImportError during RED)

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_REAL_INSTANCES = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"
_REAL_POLICY = _REPO / "suite" / "swebench" / "circuit_breaker_policy.yaml"

_FROZEN_IDS = json.loads(_REAL_INSTANCES.read_text())

# Canonical adapter fixture (same shape as tests/test_run_swebench.py).
_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-canonical-model",
        "id": "testorg/TestCanonicalModel",
        "revision": "a" * 40,
    },
    "serving": {
        "image": "testregistry.example.com/test@sha256:" + "b" * 64,
        "engine": "vllm",
        "engine_version": "0.6.6",
        "quantization": "fp8",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "tokenizer": None,
        "moe_backend": None,
        "max_model_len": 300000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 256,
        "environment": {},
    },
}
_ADAPTER_SLUG = _CANONICAL_ADAPTER["model"]["slug"]
_MODEL_ID = _CANONICAL_ADAPTER["model"]["id"]
_PROMPT_TOKEN_MAXIMA = {"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_canonical_adapter(path: pathlib.Path, extra: dict | None = None) -> None:
    import yaml
    doc = json.loads(json.dumps(_CANONICAL_ADAPTER))
    if extra:
        doc.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, default_flow_style=False))


def _build_run_dir(repo: pathlib.Path, run_id: str = "run-test") -> pathlib.Path:
    run_dir = repo / "results" / _ADAPTER_SLUG / "runs" / "warpcore-v1" / "swebench" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
    }))
    (run_dir / "manifest.json").write_text(json.dumps({
        "suite_id": "warpcore-v1",
        "run_id": run_id,
        "benchmark": "swebench",
        "model": {"slug": _ADAPTER_SLUG, "id": _MODEL_ID, "revision": "a" * 40},
        "item_inventory": {"expected": 100},
    }))
    return run_dir


def _progress_yaml_text(by_status: dict) -> str:
    lines = ["instances_by_exit_status:"]
    for status, ids in by_status.items():
        lines.append(f"  {status}:")
        for iid in ids:
            lines.append(f"    - {iid}")
    return "\n".join(lines) + "\n"


def _write_progress(raw_dir: pathlib.Path, by_status: dict, stamp: str = "1000000.0") -> pathlib.Path:
    """Write a mini-swe-agent progress YAML the way the real harness does."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"exit_statuses_{stamp}.yaml"
    tmp = raw_dir / f".exit_statuses_{stamp}.part"
    tmp.write_text(_progress_yaml_text(by_status))
    os.replace(tmp, path)
    return path


def _write_traj(raw_dir: pathlib.Path, iid: str, payload: dict) -> None:
    """Write the live (pre-normalization) mini-swe-agent trajectory for *iid*."""
    d = raw_dir / iid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{iid}.traj.json").write_text(json.dumps(payload))


def _traj_malformed_tool_call(iid: str) -> dict:
    """Persisted tool call whose arguments JSON is truncated — parser corruption."""
    return {
        "info": {"exit_status": "RepeatedFormatError", "instance_id": iid},
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls -la'},
                    }
                ],
            },
        ],
    }


def _traj_parse_error_marker(iid: str) -> dict:
    """Explicit parser diagnostic persisted in the trajectory."""
    return {
        "info": {"exit_status": "RepeatedFormatError", "instance_id": iid},
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {"role": "user", "content": "Error parsing tool call arguments: unterminated string"},
        ],
    }


def _traj_plain_format_mistake(iid: str) -> dict:
    """The model simply answered in prose. No parser or tool-argument corruption."""
    return {
        "info": {"exit_status": "RepeatedFormatError", "instance_id": iid},
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "I think we should edit the file manually."},
            {"role": "user", "content": "Please always provide exactly one action in triple backticks."},
        ],
    }


def _traj_ok(iid: str, exit_status: str) -> dict:
    return {
        "info": {"exit_status": exit_status, "instance_id": iid},
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
                    }
                ],
            },
        ],
    }


class _BreakerCase(unittest.TestCase):
    """Base: a raw/ dir plus the committed suite-owned policy."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir(parents=True)
        self.policy = cb.load_policy(_REPO)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def monitor(self):
        return cb.BreakerMonitor(raw_dir=self.raw, policy=self.policy)


# ---------------------------------------------------------------------------
# 1. The policy is suite-owned, versioned, and not overridable
# ---------------------------------------------------------------------------


class TestPolicyIsSuiteOwned(unittest.TestCase):

    def test_policy_file_lives_under_suite(self):
        self.assertTrue(
            _REAL_POLICY.is_file(),
            f"The circuit-breaker policy must be a suite-owned file at {_REAL_POLICY}.",
        )
        self.assertEqual(_REAL_POLICY.parent.parent.name, "suite")

    def test_policy_is_versioned_and_content_addressed(self):
        policy = cb.load_policy(_REPO)
        self.assertEqual(policy.policy_version, 1)
        self.assertTrue(policy.policy_id)
        self.assertEqual(policy.suite_id, "warpcore-v1")
        self.assertEqual(policy.benchmark, "swebench")
        expected = hashlib.sha256(_REAL_POLICY.read_bytes()).hexdigest()
        self.assertEqual(policy.policy_sha256, expected)

    def test_policy_thresholds_match_the_approved_control(self):
        policy = cb.load_policy(_REPO)
        self.assertTrue(policy.enabled)
        self.assertTrue(policy.first_completion_systemic)
        self.assertEqual(policy.min_completed, 3)
        self.assertEqual(policy.ratio_threshold, 0.5)

    def test_operational_outcomes_are_never_in_the_systemic_allowlist(self):
        policy = cb.load_policy(_REPO)
        for status in (
            "Submitted",
            "LimitsExceeded",
            "ContextWindowExceeded",
            "ContextWindowExceededError",
            "CostLimitExceeded",
            "StepLimitExceeded",
        ):
            self.assertNotIn(status, policy.systemic_exit_statuses)
            self.assertNotIn(status, policy.conditional_systemic_exit_statuses)

    def test_missing_policy_file_fails_closed(self):
        empty = pathlib.Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(cb.PolicyError):
                cb.load_policy(empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_malformed_policy_file_fails_closed(self):
        fake_repo = pathlib.Path(tempfile.mkdtemp())
        try:
            dest = fake_repo / "suite" / "swebench" / "circuit_breaker_policy.yaml"
            dest.parent.mkdir(parents=True)
            dest.write_text("policy_version: 1\nratio_threshold: 'not-a-number'\n")
            with self.assertRaises(cb.PolicyError):
                cb.load_policy(fake_repo)
        finally:
            shutil.rmtree(fake_repo, ignore_errors=True)


class TestPolicyCannotBeOverridden(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self, **kw):
        return run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            **kw,
        )

    def test_adapter_declaring_a_circuit_breaker_is_rejected(self):
        _write_canonical_adapter(
            self.adapter_path,
            extra={"circuit_breaker": {"enabled": False, "min_completed": 999}},
        )
        with self.assertRaises(ValueError):
            self._runner()

    def test_adapter_declaring_nested_serving_override_is_rejected(self):
        import yaml
        doc = json.loads(json.dumps(_CANONICAL_ADAPTER))
        doc["serving"]["circuit_breaker_ratio_threshold"] = 0.99
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(doc, default_flow_style=False))
        with self.assertRaises(ValueError):
            self._runner()

    def test_runner_uses_the_committed_suite_policy_verbatim(self):
        _write_canonical_adapter(self.adapter_path)
        runner = self._runner()
        self.assertEqual(
            runner.circuit_breaker_policy.policy_sha256,
            cb.load_policy(_REPO).policy_sha256,
        )

    def test_cli_exposes_no_circuit_breaker_option(self):
        parser = run_swebench._build_arg_parser()
        options = [opt for action in parser._actions for opt in action.option_strings]
        for opt in options:
            lowered = opt.lower()
            for banned in ("circuit", "breaker", "systemic", "threshold", "abort"):
                self.assertNotIn(
                    banned, lowered,
                    f"CLI option {opt!r} would let an operator tune a suite-owned control.",
                )

    def test_cli_rejects_an_arbitrary_circuit_breaker_override(self):
        _write_canonical_adapter(self.adapter_path)
        with self.assertRaises(SystemExit):
            run_swebench.main([
                "--suite", str(_REAL_SUITE),
                "--adapter", str(self.adapter_path),
                "--endpoint", "http://localhost:8000/v1",
                "--dry-run",
                "--circuit-breaker-min-completed", "999",
            ])


# ---------------------------------------------------------------------------
# 2. Trip rules
# ---------------------------------------------------------------------------


class TestTripRules(_BreakerCase):

    def test_first_completed_parser_corrupt_instance_trips(self):
        iid = _FROZEN_IDS[0]
        _write_traj(self.raw, iid, _traj_malformed_tool_call(iid))
        _write_progress(self.raw, {"RepeatedFormatError": [iid]})

        decision = self.monitor().observe()

        self.assertIsNotNone(decision, "A first systemic completion must trip the breaker.")
        self.assertEqual(decision.rule, "first_completion_systemic")
        self.assertEqual(decision.completed_count, 1)
        self.assertEqual(decision.systemic_count, 1)
        self.assertIn(iid, [c.instance_id for c in decision.classifications])

    def test_first_completed_explicit_parse_error_marker_trips(self):
        iid = _FROZEN_IDS[0]
        _write_traj(self.raw, iid, _traj_parse_error_marker(iid))
        _write_progress(self.raw, {"RepeatedFormatError": [iid]})

        decision = self.monitor().observe()

        self.assertIsNotNone(decision)
        self.assertEqual(decision.rule, "first_completion_systemic")

    def test_first_completed_internal_server_error_trips(self):
        iid = _FROZEN_IDS[0]
        _write_progress(self.raw, {"InternalServerError": [iid]})

        decision = self.monitor().observe()

        self.assertIsNotNone(decision)
        self.assertEqual(decision.rule, "first_completion_systemic")

    def test_first_completed_runtime_error_trips(self):
        iid = _FROZEN_IDS[0]
        _write_progress(self.raw, {"RuntimeError": [iid]})

        decision = self.monitor().observe()

        self.assertIsNotNone(decision)

    def test_two_of_three_parser_corrupt_trips(self):
        bad = _FROZEN_IDS[:2]
        good = _FROZEN_IDS[2]
        for iid in bad:
            _write_traj(self.raw, iid, _traj_malformed_tool_call(iid))
        _write_traj(self.raw, good, _traj_ok(good, "Submitted"))
        # Arrive one at a time so the first observed batch is unambiguous and
        # deliberately NOT systemic — only the majority rule may fire here.
        mon = self.monitor()
        _write_progress(self.raw, {"Submitted": [good]}, stamp="1000000.0")
        self.assertIsNone(mon.observe())
        _write_progress(self.raw, {"Submitted": [good], "RepeatedFormatError": bad[:1]},
                        stamp="1000001.0")
        self.assertIsNone(mon.observe())
        _write_progress(self.raw, {"Submitted": [good], "RepeatedFormatError": bad},
                        stamp="1000002.0")

        decision = mon.observe()

        self.assertIsNotNone(decision, "2 of 3 systemic completions must trip the breaker.")
        self.assertEqual(decision.rule, "systemic_majority")
        self.assertEqual(decision.completed_count, 3)
        self.assertEqual(decision.systemic_count, 2)

    def test_one_of_three_parser_corrupt_does_not_trip(self):
        bad = _FROZEN_IDS[0]
        good = _FROZEN_IDS[1:3]
        _write_traj(self.raw, bad, _traj_malformed_tool_call(bad))
        for iid in good:
            _write_traj(self.raw, iid, _traj_ok(iid, "Submitted"))
        _write_progress(self.raw, {"Submitted": good, "RepeatedFormatError": [bad]})

        self.assertIsNone(
            self.monitor().observe(),
            "1 of 3 is not a systemic majority and the first batch is ambiguous.",
        )

    def test_exactly_half_systemic_does_not_trip(self):
        bad = _FROZEN_IDS[:2]
        good = _FROZEN_IDS[2:4]
        for iid in bad:
            _write_traj(self.raw, iid, _traj_malformed_tool_call(iid))
        for iid in good:
            _write_traj(self.raw, iid, _traj_ok(iid, "Submitted"))
        _write_progress(self.raw, {"Submitted": good, "RepeatedFormatError": bad})

        self.assertIsNone(
            self.monitor().observe(),
            "The threshold is strictly greater than 50%; exactly half must not trip.",
        )

    def test_two_systemic_of_two_does_not_reach_the_majority_minimum(self):
        bad = _FROZEN_IDS[:2]
        for iid in bad:
            _write_traj(self.raw, iid, _traj_malformed_tool_call(iid))
        mon = self.monitor()
        # First batch is both instances at once and both are systemic, so the
        # first-completion rule legitimately fires. Force ambiguity instead by
        # seeding a non-systemic first batch, then adding the two systemic ones.
        good = _FROZEN_IDS[2]
        _write_traj(self.raw, good, _traj_ok(good, "Submitted"))
        _write_progress(self.raw, {"Submitted": [good]}, stamp="1000000.0")
        self.assertIsNone(mon.observe())
        _write_progress(self.raw, {"Submitted": [good], "RepeatedFormatError": bad[:1]},
                        stamp="1000001.0")
        self.assertIsNone(mon.observe(), "2 completions is below the 3-completion minimum.")


# ---------------------------------------------------------------------------
# 3. Narrow RepeatedFormatError semantics and operational outcomes
# ---------------------------------------------------------------------------


class TestNonSystemicOutcomes(_BreakerCase):

    def test_plain_repeated_format_error_without_parser_proof_does_not_trip(self):
        iid = _FROZEN_IDS[0]
        _write_traj(self.raw, iid, _traj_plain_format_mistake(iid))
        _write_progress(self.raw, {"RepeatedFormatError": [iid]})

        decision = self.monitor().observe()

        self.assertIsNone(
            decision,
            "A model format mistake is not proof of parser/tool-argument corruption.",
        )

    def test_repeated_format_error_majority_without_proof_does_not_trip(self):
        ids = _FROZEN_IDS[:4]
        for iid in ids:
            _write_traj(self.raw, iid, _traj_plain_format_mistake(iid))
        _write_progress(self.raw, {"RepeatedFormatError": ids})

        self.assertIsNone(self.monitor().observe())

    def test_repeated_format_error_with_missing_trajectory_does_not_trip(self):
        iid = _FROZEN_IDS[0]
        _write_progress(self.raw, {"RepeatedFormatError": [iid]})

        self.assertIsNone(
            self.monitor().observe(),
            "Absent trajectory evidence is not proof of parser corruption.",
        )

    def test_limits_exceeded_does_not_trip(self):
        ids = _FROZEN_IDS[:4]
        for iid in ids:
            _write_traj(self.raw, iid, _traj_ok(iid, "LimitsExceeded"))
        _write_progress(self.raw, {"LimitsExceeded": ids})

        self.assertIsNone(
            self.monitor().observe(),
            "Step/cost limits are model/config outcomes and stay in the denominator.",
        )

    def test_context_window_exceeded_does_not_trip(self):
        ids = _FROZEN_IDS[:4]
        for iid in ids:
            _write_traj(self.raw, iid, _traj_ok(iid, "ContextWindowExceededError"))
        _write_progress(self.raw, {"ContextWindowExceededError": ids})

        self.assertIsNone(self.monitor().observe())

    def test_cost_and_step_limits_do_not_trip(self):
        ids = _FROZEN_IDS[:4]
        for iid in ids:
            _write_traj(self.raw, iid, _traj_ok(iid, "CostLimitExceeded"))
        _write_progress(self.raw, {
            "CostLimitExceeded": ids[:2],
            "StepLimitExceeded": ids[2:],
        })

        self.assertIsNone(self.monitor().observe())

    def test_submitted_and_wrong_patches_do_not_trip(self):
        ids = _FROZEN_IDS[:5]
        for iid in ids:
            _write_traj(self.raw, iid, _traj_ok(iid, "Submitted"))
        _write_progress(self.raw, {"Submitted": ids})

        self.assertIsNone(self.monitor().observe())

    def test_unrecognised_exit_status_does_not_trip(self):
        ids = _FROZEN_IDS[:4]
        _write_progress(self.raw, {"SomeBrandNewStatus": ids})

        decision = self.monitor().observe()

        self.assertIsNone(
            decision,
            "An unknown status is reported, not silently treated as systemic.",
        )

    def test_unrecognised_exit_status_is_classified_explicitly(self):
        iid = _FROZEN_IDS[0]
        _write_progress(self.raw, {"SomeBrandNewStatus": [iid]})
        mon = self.monitor()
        mon.observe()

        classified = {c.instance_id: c for c in mon.classifications()}

        self.assertEqual(classified[iid].classification, "unclassified")
        self.assertFalse(classified[iid].systemic)


# ---------------------------------------------------------------------------
# 4. Snapshot safety — a partial write is never evidence
# ---------------------------------------------------------------------------


class TestSnapshotSafety(_BreakerCase):

    def test_no_progress_file_yields_no_snapshot(self):
        self.assertIsNone(cb.read_stable_progress(self.raw, policy=self.policy))

    def test_malformed_yaml_yields_no_snapshot(self):
        (self.raw / "exit_statuses_1.yaml").write_text("instances_by_exit_status: [oops\n")

        self.assertIsNone(cb.read_stable_progress(self.raw, policy=self.policy))

    def test_malformed_yaml_does_not_trip(self):
        (self.raw / "exit_statuses_1.yaml").write_text(
            "instances_by_exit_status:\n  RuntimeError:\n    - unterminated"
            "\n    - {broken\n"
        )

        self.assertIsNone(self.monitor().observe())

    def test_wrong_shaped_yaml_does_not_trip(self):
        (self.raw / "exit_statuses_1.yaml").write_text(
            "instances_by_exit_status:\n  RuntimeError: not-a-list\n"
        )

        self.assertIsNone(self.monitor().observe())

    def test_truncated_half_written_yaml_does_not_trip(self):
        # mini-swe-agent rewrote the file and we caught it mid-write: the status
        # key is present but its list has not been flushed yet.
        (self.raw / "exit_statuses_1.yaml").write_text("instances_by_exit_status:\n")

        self.assertIsNone(self.monitor().observe())

    def test_a_file_still_changing_is_not_read_as_evidence(self):
        path = self.raw / "exit_statuses_1.yaml"
        path.write_text(_progress_yaml_text({"RuntimeError": [_FROZEN_IDS[0]]}))

        calls = {"n": 0}

        def mutating_sleep(_seconds):
            calls["n"] += 1
            path.write_text(
                _progress_yaml_text({"RuntimeError": _FROZEN_IDS[: calls["n"] + 1]})
            )

        snapshot = cb.read_stable_progress(self.raw, policy=self.policy, sleep=mutating_sleep)

        self.assertIsNone(
            snapshot,
            "A progress file that changes between reads is a partial write, not evidence.",
        )
        self.assertGreaterEqual(calls["n"], 2, "The reader must retry before giving up.")

    def test_a_settled_file_is_read_as_evidence(self):
        _write_progress(self.raw, {"Submitted": _FROZEN_IDS[:2]})

        snapshot = cb.read_stable_progress(self.raw, policy=self.policy, sleep=lambda _s: None)

        self.assertIsNotNone(snapshot)
        self.assertEqual(set(snapshot.id_to_status), set(_FROZEN_IDS[:2]))

    def test_latest_progress_file_wins(self):
        _write_progress(self.raw, {"Submitted": [_FROZEN_IDS[0]]}, stamp="1000000.0")
        time.sleep(0.01)
        _write_progress(self.raw, {"Submitted": _FROZEN_IDS[:2]}, stamp="1000009.0")

        snapshot = cb.read_stable_progress(self.raw, policy=self.policy, sleep=lambda _s: None)

        self.assertEqual(len(snapshot.id_to_status), 2)

    def test_reconciliation_never_drops_an_already_observed_instance(self):
        mon = self.monitor()
        _write_progress(self.raw, {"Submitted": _FROZEN_IDS[:3]}, stamp="1000000.0")
        mon.observe()
        # A rewrite that momentarily shows fewer instances must not erase evidence.
        _write_progress(self.raw, {"Submitted": [_FROZEN_IDS[0]]}, stamp="1000001.0")
        mon.observe()

        self.assertEqual(len(mon.classifications()), 3)

    def test_partial_rewrite_cannot_manufacture_a_systemic_majority(self):
        good = _FROZEN_IDS[:3]
        bad = _FROZEN_IDS[3]
        for iid in good:
            _write_traj(self.raw, iid, _traj_ok(iid, "Submitted"))
        _write_traj(self.raw, bad, _traj_malformed_tool_call(bad))
        mon = self.monitor()
        _write_progress(self.raw, {"Submitted": good}, stamp="1000000.0")
        self.assertIsNone(mon.observe())
        # A truncated rewrite that only lists the failing instance must not look
        # like "100% systemic" — the three healthy completions are retained.
        _write_progress(self.raw, {"RepeatedFormatError": [bad]}, stamp="1000001.0")

        self.assertIsNone(mon.observe())
        self.assertEqual(
            len(mon.classifications()), 4,
            "All four completions must still be in the denominator the breaker reasons over.",
        )
        self.assertEqual(sum(1 for c in mon.classifications() if c.systemic), 1)

    def test_partial_trajectory_json_is_not_parser_proof(self):
        iid = _FROZEN_IDS[0]
        d = self.raw / iid
        d.mkdir(parents=True)
        (d / f"{iid}.traj.json").write_text('{"messages": [{"role": "assistant", "tool_c')
        _write_progress(self.raw, {"RepeatedFormatError": [iid]})

        self.assertIsNone(
            self.monitor().observe(),
            "A half-written trajectory is not proof of parser corruption.",
        )


# ---------------------------------------------------------------------------
# 5. Production path — live subprocess monitoring, termination, artifacts
# ---------------------------------------------------------------------------


_FAKE_GENERATION = '''
import json, os, pathlib, signal, subprocess, sys, time

raw = pathlib.Path(sys.argv[1]); raw.mkdir(parents=True, exist_ok=True)
marker = pathlib.Path(sys.argv[2])
mode = sys.argv[3]
ids = json.loads(sys.argv[4])
status = sys.argv[5]

child = None
if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])

if status == "RepeatedFormatError":
    for iid in ids:
        d = raw / iid
        d.mkdir(parents=True, exist_ok=True)
        (d / (iid + ".traj.json")).write_text(json.dumps({
            "info": {"exit_status": status},
            "messages": [{"role": "assistant", "tool_calls": [
                {"function": {"name": "bash", "arguments": '{"command": "ls'}}]}],
        }))

lines = ["instances_by_exit_status:", "  " + status + ":"]
lines += ["    - " + iid for iid in ids]
tmp = raw / ".progress.part"
tmp.write_text("\\n".join(lines) + "\\n")
os.replace(str(tmp), str(raw / "exit_statuses_1000000.0.yaml"))

marker.write_text(json.dumps({
    "pid": os.getpid(), "pgid": os.getpgrp(),
    "child": child.pid if child is not None else None,
}))
sys.stdout.write("fake mini-swe-agent running\\n")
sys.stdout.flush()

if mode == "clean":
    sys.exit(0)
if mode == "sigterm":
    time.sleep(0.3)
    os.kill(os.getpid(), signal.SIGTERM)
time.sleep(300)
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


class _ProductionRunner(run_swebench.SwebenchRunner):
    """Production runner with only the generation argv replaced.

    The subprocess launch, live progress monitoring, trip evaluation, process-group
    termination, artifact write and lifecycle transition all run unmodified.
    """

    fake_script: pathlib.Path
    marker: pathlib.Path
    mode: str = "hang"
    fake_ids: list = []
    fake_status: str = "RuntimeError"
    interrupt_after: int = 0

    def _build_generation_argv(self, config_path, raw_dir):
        return [
            sys.executable, str(self.fake_script), str(raw_dir), str(self.marker),
            self.mode, json.dumps(self.fake_ids), self.fake_status,
        ]

    def _monitor_poll_interval_s(self) -> float:
        return 0.05

    def _monitor_sleep(self, seconds: float) -> None:
        # Interrupt only once the subprocess has published its pids, so the
        # "did we orphan it?" assertion has something to check.
        if self.interrupt_after and self.marker.exists():
            self.interrupt_after -= 1
            if self.interrupt_after == 0:
                raise KeyboardInterrupt()
        time.sleep(seconds)


class TestProductionGenerationPath(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)
        self.script = self.tmp / "fake_generation.py"
        self.script.write_text(_FAKE_GENERATION)
        self.marker = self.tmp / "marker.json"
        self.grading_calls = []
        self.bystander = None

    def tearDown(self):
        if self.bystander is not None and self.bystander.poll() is None:
            self.bystander.kill()
            self.bystander.wait()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self, **overrides) -> _ProductionRunner:
        runner = _ProductionRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda _model_id: 0,
            grading_runner=self._record_grading,
        )
        runner.fake_script = self.script
        runner.marker = self.marker
        runner.fake_ids = [_FROZEN_IDS[0]]
        for key, value in overrides.items():
            setattr(runner, key, value)
        return runner

    def _record_grading(self, preds_path, run_dir):
        self.grading_calls.append(preds_path)
        return 0

    # -- trip -------------------------------------------------------------

    def test_systemic_first_instance_aborts_the_live_generation(self):
        runner = self._runner(fake_status="RuntimeError")

        rc = runner.run()

        self.assertEqual(rc, run_swebench.EXIT_DEFECT)
        self.assertTrue(self.marker.exists(), "The fake generation must have started.")

    def test_termination_kills_exactly_the_owned_process_group(self):
        self.bystander = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        runner = self._runner(fake_status="RuntimeError")

        runner.run()

        observed = json.loads(self.marker.read_text())
        self.assertTrue(_wait_gone(observed["pid"]), "The owned subprocess must be terminated.")
        self.assertTrue(
            _wait_gone(observed["child"]),
            "The whole owned process group must be terminated, not just the direct child.",
        )
        self.assertNotEqual(
            observed["pgid"], os.getpgrp(),
            "Generation must run in its own process group so the kill cannot reach the runner.",
        )
        self.assertIsNone(
            self.bystander.poll(),
            "A process outside the owned group must survive the circuit breaker.",
        )

    def test_trip_does_not_grade_and_does_not_write_done(self):
        runner = self._runner(fake_status="RuntimeError")

        runner.run()

        self.assertEqual(self.grading_calls, [], "A tripped campaign must never be graded.")
        self.assertFalse((self.run_dir / "DONE").exists())
        self.assertFalse((self.run_dir / "raw" / "grading_results.json").exists())

    def test_trip_transitions_the_campaign_to_failed_and_invalid(self):
        runner = self._runner(fake_status="RuntimeError")

        runner.run()

        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "failed")
        self.assertEqual(status["lifecycle"], "invalid")
        self.assertIn("circuit breaker", status["history"][-1].get("note", "").lower())

    def test_trip_writes_a_structured_diagnostic_artifact(self):
        runner = self._runner(fake_status="RuntimeError")

        runner.run()

        artifact_path = self.run_dir / "circuit_breaker.json"
        self.assertTrue(artifact_path.is_file(), "circuit_breaker.json must be written.")
        artifact = json.loads(artifact_path.read_text())

        self.assertTrue(artifact["tripped"])
        self.assertEqual(artifact["benchmark"], "swebench")
        self.assertEqual(artifact["suite_id"], "warpcore-v1")
        self.assertEqual(artifact["run_id"], self.run_dir.name)
        self.assertTrue(artifact["reason"])
        self.assertIn(artifact["rule"], ("first_completion_systemic", "systemic_majority"))

        policy = cb.load_policy(_REPO)
        self.assertEqual(artifact["policy"]["policy_version"], policy.policy_version)
        self.assertEqual(artifact["policy"]["policy_sha256"], policy.policy_sha256)
        self.assertEqual(artifact["policy"]["policy_id"], policy.policy_id)

        observed = artifact["observed"]
        self.assertEqual(observed["completed_count"], 1)
        self.assertEqual(observed["systemic_count"], 1)
        entry = observed["instances"][0]
        self.assertEqual(entry["instance_id"], _FROZEN_IDS[0])
        self.assertEqual(entry["exit_status"], "RuntimeError")
        self.assertTrue(entry["systemic"])
        self.assertTrue(entry["classification"])

        # UTC, second precision, Zulu — same convention as status.json.
        self.assertRegex(artifact["tripped_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

        termination = artifact["termination"]
        self.assertEqual(termination["terminated_by"], "warpcore_circuit_breaker")
        self.assertEqual(termination["scope"], "owned_process_group")
        self.assertIsInstance(termination["pid"], int)

    def test_trip_preserves_the_partial_generation_evidence(self):
        runner = self._runner(fake_status="RepeatedFormatError")

        runner.run()

        raw = self.run_dir / "raw"
        self.assertTrue(
            list(raw.glob("exit_statuses_*.yaml")),
            "The partial progress YAML must be preserved as evidence.",
        )
        traj = raw / _FROZEN_IDS[0] / f"{_FROZEN_IDS[0]}.traj.json"
        self.assertTrue(traj.is_file(), "Partial trajectories must be preserved.")
        self.assertTrue((raw / "run.log").exists())

    def test_parser_corrupt_first_instance_aborts_the_live_generation(self):
        runner = self._runner(fake_status="RepeatedFormatError")

        rc = runner.run()

        self.assertEqual(rc, run_swebench.EXIT_DEFECT)
        artifact = json.loads((self.run_dir / "circuit_breaker.json").read_text())
        self.assertEqual(artifact["observed"]["instances"][0]["exit_status"],
                         "RepeatedFormatError")
        self.assertIn("parser", artifact["observed"]["instances"][0]["classification"])

    # -- negative controls ------------------------------------------------

    def test_a_healthy_partial_generation_is_not_tripped(self):
        runner = self._runner(mode="clean", fake_status="Submitted",
                              fake_ids=list(_FROZEN_IDS[:3]))

        runner.run()

        self.assertFalse(
            (self.run_dir / "circuit_breaker.json").exists(),
            "Submitted outcomes must never trip the breaker.",
        )
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertNotIn("circuit breaker", status["history"][-1].get("note", "").lower())

    def test_unrelated_sigterm_is_not_recorded_as_a_circuit_breaker_trip(self):
        runner = self._runner(mode="sigterm", fake_status="Submitted",
                              fake_ids=list(_FROZEN_IDS[:3]))

        rc = runner.run()

        self.assertEqual(rc, run_swebench.EXIT_DEFECT)
        self.assertFalse(
            (self.run_dir / "circuit_breaker.json").exists(),
            "An externally signalled subprocess is not a circuit-breaker abort.",
        )
        status = json.loads((self.run_dir / "status.json").read_text())
        note = status["history"][-1].get("note", "").lower()
        self.assertNotIn("circuit breaker", note)
        self.assertIn("generation failed", note)

    def test_operator_interruption_is_distinguishable_from_a_trip(self):
        runner = self._runner(mode="hang", fake_status="Submitted",
                              fake_ids=list(_FROZEN_IDS[:3]), interrupt_after=1)

        rc = runner.run()

        self.assertEqual(
            rc, run_swebench.EXIT_INCONCLUSIVE,
            "An operator interrupt is inconclusive, not a diagnosed systemic defect.",
        )
        self.assertFalse((self.run_dir / "circuit_breaker.json").exists())
        status = json.loads((self.run_dir / "status.json").read_text())
        note = status["history"][-1].get("note", "").lower()
        self.assertIn("operator", note)
        self.assertNotIn("circuit breaker", note)

    def test_operator_interruption_still_terminates_the_owned_subprocess(self):
        runner = self._runner(mode="hang", fake_status="Submitted",
                              fake_ids=list(_FROZEN_IDS[:3]), interrupt_after=1)

        runner.run()

        observed = json.loads(self.marker.read_text())
        self.assertTrue(
            _wait_gone(observed["pid"]),
            "An interrupted run must not leave the generation subprocess orphaned.",
        )


if __name__ == "__main__":
    unittest.main()
