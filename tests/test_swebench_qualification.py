"""tests/test_swebench_qualification.py — executable SWE-bench qualification gate.

TDD: these tests are written BEFORE viz/swebench_qualification.py exists and must
fail (RED) until it does.

WHY THIS EXISTS
---------------
External cron/shell logic promoted a gpt-oss smoke whose terminal status was
``RepeatedFormatError``.  Qualification was prose, not repository policy, so
nothing in the repo could refuse the promotion.  This file makes the refusal
executable and adversarially tested.

The gate is authoritative and fail-closed.  Every test below either proves the
production path qualifies, or proves that a specific mismatch, staleness, forgery,
or convenience shortcut *cannot* authorize a canonical launch.

Design contract:
  docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md §4.4, §8.2
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import swebench_qualification as sq  # noqa: E402  (expected ImportError during RED)
import contract  # noqa: E402
import run_swebench  # noqa: E402

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_REAL_FROZEN_IDS = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"
_REAL_QUAL_IDS = _REPO / "suite" / "swebench" / "qualification-ids-v1.json"
_REAL_SCAFFOLD = _REPO / "suite" / "swebench" / "scaffold.yaml"
_REAL_ADAPTER = _REPO / "adapters" / "gpt-oss-120b.yaml"
_FIXTURE_RUN = _TESTS_DIR / "fixtures" / "swebench_qualification" / "run"

_ENDPOINT = "http://csi370295.alcf.anl.gov:8000/v1"


def _frozen_ids() -> list:
    return json.loads(_REAL_FROZEN_IDS.read_text())


def _qual_ids() -> list:
    return json.loads(_REAL_QUAL_IDS.read_text())


# ---------------------------------------------------------------------------
# Environment helper: a self-contained qualification directory
# ---------------------------------------------------------------------------


class _Env:
    """A writable copy of the committed production-path qualification fixture.

    Layout (self-contained, exactly what the gate expects on disk):

        <root>/qualification.json
        <root>/run/raw/preds.json
        <root>/run/raw/exit_statuses.json
        <root>/run/raw/trajectories/<id>.traj
        <root>/run/raw/<model>.<run-id>.json     official grader report
    """

    def __init__(self, tmp: pathlib.Path, adapter_path: pathlib.Path = _REAL_ADAPTER,
                 endpoint: str = _ENDPOINT):
        self.tmp = tmp
        self.endpoint = endpoint
        self.root = tmp / "qualification"
        self.root.mkdir(parents=True, exist_ok=True)
        self.run = self.root / "run"
        shutil.copytree(_FIXTURE_RUN, self.run)
        self.artifact_path = self.root / "qualification.json"
        self.adapter_path = adapter_path
        self.raw = self.run / "raw"
        self.report = next(
            p for p in sorted(self.raw.glob("*.json"))
            if p.name not in ("preds.json", "exit_statuses.json", "grading_results.json")
        )

    # -- evidence mutators (all rewrite the artifact afterwards via emit) ----

    def preds(self) -> dict:
        return json.loads((self.raw / "preds.json").read_text())

    def write_preds(self, data: dict) -> None:
        (self.raw / "preds.json").write_text(json.dumps(data, indent=2) + "\n")

    def statuses(self) -> dict:
        return json.loads((self.raw / "exit_statuses.json").read_text())

    def write_statuses(self, data: dict) -> None:
        (self.raw / "exit_statuses.json").write_text(json.dumps(data, indent=2) + "\n")

    def grading(self) -> dict:
        return json.loads(self.report.read_text())

    def write_grading(self, data: dict) -> None:
        self.report.write_text(json.dumps(data, indent=2) + "\n")

    def traj(self, iid: str) -> dict:
        return json.loads((self.raw / "trajectories" / f"{iid}.traj").read_text())

    def write_traj(self, iid: str, data) -> None:
        (self.raw / "trajectories" / f"{iid}.traj").write_text(json.dumps(data, indent=2) + "\n")

    # -- artifact emission / verification ------------------------------------

    def emit(self, now=None, **overrides) -> dict:
        """Rebuild the artifact from the *current* evidence and write it."""
        artifact = sq.build_qualification_artifact(
            repo=_REPO,
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint=self.endpoint,
            served_model_id=_served_model_id(self.adapter_path),
            evidence_run_dir=self.run,
            artifact_path=self.artifact_path,
            now=now,
        )
        for key, value in overrides.items():
            artifact[key] = value
        self.write_artifact(artifact)
        return artifact

    def write_artifact(self, artifact) -> None:
        self.artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

    def verify(self, now=None, endpoint: str = None, **kw):
        return sq.verify_qualification_for_launch(
            repo=_REPO,
            suite_path=_REAL_SUITE,
            adapter_path=kw.pop("adapter_path", self.adapter_path),
            endpoint=endpoint if endpoint is not None else self.endpoint,
            artifact_path=self.artifact_path,
            now=now,
            **kw,
        )


def _served_model_id(adapter_path: pathlib.Path) -> str:
    import yaml
    return (yaml.safe_load(adapter_path.read_text()).get("model") or {}).get("id", "")


class _EnvCase(unittest.TestCase):
    """Base class providing a fresh writable qualification environment."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.env = _Env(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertBlocked(self, result, *needles):
        self.assertFalse(
            result.ok,
            f"qualification must be blocked but passed; summary={result.summary}",
        )
        blob = " ".join(result.errors).lower()
        for needle in needles:
            self.assertIn(needle.lower(), blob, f"missing {needle!r} in {result.errors}")


# ---------------------------------------------------------------------------
# 1. Suite-owned qualification instance set
# ---------------------------------------------------------------------------


class TestSuiteOwnedQualificationIds(unittest.TestCase):
    """The 20 qualification IDs are suite-owned, frozen, and derivable."""

    def test_qualification_ids_file_exists(self):
        self.assertTrue(_REAL_QUAL_IDS.is_file(), f"missing {_REAL_QUAL_IDS}")

    def test_exactly_twenty_unique_ids(self):
        ids = _qual_ids()
        self.assertEqual(len(ids), sq.REQUIRED_QUALIFICATION_COUNT)
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20, "qualification IDs must be unique")

    def test_all_ids_are_members_of_the_frozen_n100_set(self):
        self.assertTrue(set(_qual_ids()) <= set(_frozen_ids()))

    def test_ids_equal_the_documented_deterministic_derivation(self):
        """Not convenience cases: a pure function of the frozen seed-42 set."""
        derived = sq.derive_qualification_instance_ids(_frozen_ids())
        self.assertEqual(_qual_ids(), derived)

    def test_derivation_covers_every_repository_in_the_frozen_set(self):
        derived = sq.derive_qualification_instance_ids(_frozen_ids())
        repos_all = {i.rsplit("-", 1)[0] for i in _frozen_ids()}
        repos_sel = {i.rsplit("-", 1)[0] for i in derived}
        self.assertEqual(repos_sel, repos_all, "every repository must be represented")

    def test_derivation_is_deterministic_and_order_independent(self):
        frozen = _frozen_ids()
        self.assertEqual(
            sq.derive_qualification_instance_ids(frozen),
            sq.derive_qualification_instance_ids(list(frozen)),
        )

    def test_suite_declares_qualification_block_with_matching_hash(self):
        import yaml
        suite = yaml.safe_load(_REAL_SUITE.read_text())
        qual = suite["benchmarks"]["swebench"]["qualification"]
        self.assertEqual(qual["expected_count"], 20)
        self.assertEqual(qual["ids_file"], "suite/swebench/qualification-ids-v1.json")
        self.assertEqual(qual["ids_sha256"], contract.sha256_file(_REAL_QUAL_IDS))
        self.assertEqual(qual["required_terminal_status"], "Submitted")

    def test_real_suite_passes_contract_validation(self):
        self.assertEqual(contract.validate_suite(_REPO, _REAL_SUITE), [])

    def test_contract_rejects_stale_qualification_ids_hash(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(_REPO / "suite", tmp / "suite")
        suite_path = tmp / "suite" / "warpcore-v1.yaml"
        text = suite_path.read_text().replace(
            contract.sha256_file(_REAL_QUAL_IDS), "0" * 64
        )
        suite_path.write_text(text)
        errors = contract.validate_suite(tmp, suite_path)
        self.assertTrue(any("qualification" in e.lower() for e in errors), errors)

    def test_contract_rejects_qualification_set_of_wrong_size(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(_REPO / "suite", tmp / "suite")
        ids_path = tmp / "suite" / "swebench" / "qualification-ids-v1.json"
        shrunk = _qual_ids()[:5]
        ids_path.write_text(json.dumps(shrunk, indent=2) + "\n")
        suite_path = tmp / "suite" / "warpcore-v1.yaml"
        suite_path.write_text(
            suite_path.read_text().replace(
                contract.sha256_file(_REAL_QUAL_IDS), contract.sha256_file(ids_path)
            )
        )
        errors = contract.validate_suite(tmp, suite_path)
        self.assertTrue(any("20" in e for e in errors), errors)

    def test_contract_rejects_foreign_qualification_id(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(_REPO / "suite", tmp / "suite")
        ids_path = tmp / "suite" / "swebench" / "qualification-ids-v1.json"
        bad = _qual_ids()[:19] + ["not-in-the-frozen-set-1"]
        ids_path.write_text(json.dumps(bad, indent=2) + "\n")
        suite_path = tmp / "suite" / "warpcore-v1.yaml"
        suite_path.write_text(
            suite_path.read_text().replace(
                contract.sha256_file(_REAL_QUAL_IDS), contract.sha256_file(ids_path)
            )
        )
        errors = contract.validate_suite(tmp, suite_path)
        self.assertTrue(errors, "a foreign qualification ID must fail suite validation")


# ---------------------------------------------------------------------------
# 2. Versioned artifact and schema
# ---------------------------------------------------------------------------


class TestArtifactSchema(_EnvCase):
    """The qualification artifact is versioned and schema-validated."""

    def test_schema_file_exists(self):
        self.assertTrue(sq.QUALIFICATION_SCHEMA_PATH.is_file())

    def test_schema_version_constant(self):
        self.assertEqual(sq.QUALIFICATION_SCHEMA_VERSION, 1)

    def test_emitted_artifact_is_schema_valid(self):
        artifact = self.env.emit()
        self.assertEqual(
            contract.validate_json(artifact, sq.QUALIFICATION_SCHEMA_PATH), []
        )

    def test_artifact_records_every_required_binding(self):
        artifact = self.env.emit()
        for key in (
            "qualification_schema_version",
            "qualification_id",
            "generated_utc",
            "repo_sha",
            "suite_id",
            "suite_schema_version",
            "suite_input_hashes",
            "adapter_hash",
            "serving_profile_digest",
            "model",
            "endpoint",
            "scaffold_sha256",
            "production_scaffold_hash",
            "launch_path",
            "qualification_instance_ids",
            "qualification_ids_sha256",
            "evidence",
            "dispositions",
            "grading",
        ):
            self.assertIn(key, artifact, f"artifact must bind {key}")

    def test_unknown_schema_version_is_rejected(self):
        self.env.emit(qualification_schema_version=2)
        self.assertBlocked(self.env.verify(), "schema_version")

    def test_malformed_artifact_json_is_rejected(self):
        self.env.emit()
        self.env.artifact_path.write_text("{not json")
        self.assertBlocked(self.env.verify(), "json")

    def test_missing_artifact_is_rejected(self):
        self.env.emit()
        self.env.artifact_path.unlink()
        self.assertBlocked(self.env.verify(), "not found")

    def test_artifact_is_separate_from_campaign_execution_state(self):
        """The artifact must not be a status.json and must not carry run state."""
        artifact = self.env.emit()
        for forbidden in ("execution_state", "history", "lifecycle"):
            self.assertNotIn(forbidden, artifact)


# ---------------------------------------------------------------------------
# 3. Production-path positive fixture
# ---------------------------------------------------------------------------


class TestProductionPositiveFixture(_EnvCase):
    """The committed production-path fixture qualifies."""

    def test_committed_fixture_qualifies(self):
        self.env.emit()
        result = self.env.verify()
        self.assertTrue(result.ok, result.errors)

    def test_summary_names_the_twenty_ids_and_grader(self):
        self.env.emit()
        result = self.env.verify()
        self.assertIn("20", result.summary)

    def test_all_twenty_ids_terminal_submitted(self):
        self.assertEqual(
            sorted(self.env.statuses()), sorted(_qual_ids())
        )
        self.assertEqual(set(self.env.statuses().values()), {"Submitted"})

    def test_production_scaffold_hash_matches_the_runner_builder(self):
        """The artifact's scaffold hash is exactly what the production runner builds."""
        import yaml
        artifact = self.env.emit()
        scaffold = yaml.safe_load(_REAL_SCAFFOLD.read_text())
        built = sq.build_production_scaffold_config(
            scaffold=scaffold,
            model_id=_served_model_id(_REAL_ADAPTER),
            endpoint=_ENDPOINT,
            api_key="warpcore",
        )
        self.assertEqual(artifact["production_scaffold_hash"], sq.production_scaffold_hash(built))

    def test_production_scaffold_hash_ignores_api_key(self):
        import yaml
        scaffold = yaml.safe_load(_REAL_SCAFFOLD.read_text())
        a = sq.build_production_scaffold_config(
            scaffold=scaffold, model_id="m", endpoint=_ENDPOINT, api_key="warpcore")
        b = sq.build_production_scaffold_config(
            scaffold=scaffold, model_id="m", endpoint=_ENDPOINT, api_key="something-else")
        self.assertEqual(sq.production_scaffold_hash(a), sq.production_scaffold_hash(b))

    def test_production_scaffold_hash_changes_with_endpoint(self):
        import yaml
        scaffold = yaml.safe_load(_REAL_SCAFFOLD.read_text())
        a = sq.build_production_scaffold_config(
            scaffold=scaffold, model_id="m", endpoint=_ENDPOINT, api_key="k")
        b = sq.build_production_scaffold_config(
            scaffold=scaffold, model_id="m", endpoint="http://other:8000/v1", api_key="k")
        self.assertNotEqual(sq.production_scaffold_hash(a), sq.production_scaffold_hash(b))


# ---------------------------------------------------------------------------
# 4. Forbidden dispositions — the RepeatedFormatError class
# ---------------------------------------------------------------------------


_FORBIDDEN_STATUSES = [
    "RepeatedFormatError",
    "RuntimeError",
    "FormatError",
    "ParseError",
    "ParserError",
    "APIConnectionError",
    "ConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "TimeoutExpired",
    "infrastructure_error",
    "infra_failure",
    "DockerPullTimeout",
]


class TestForbiddenDispositions(_EnvCase):
    """No forbidden terminal disposition may ever authorize a launch."""

    def test_repeated_format_error_blocks_launch(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "RepeatedFormatError")

    def test_every_forbidden_disposition_blocks_launch(self):
        for status in _FORBIDDEN_STATUSES:
            with self.subTest(status=status):
                env = _Env(pathlib.Path(tempfile.mkdtemp()))
                self.addCleanup(shutil.rmtree, env.tmp, True)
                statuses = env.statuses()
                statuses[_qual_ids()[3]] = status
                env.write_statuses(statuses)
                env.emit()
                self.assertBlocked(env.verify(), status)

    def test_forbidden_disposition_is_case_insensitive(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "repeatedformaterror"
        self.env.write_statuses(statuses)
        self.env.emit()
        self.assertBlocked(self.env.verify())

    def test_any_non_submitted_terminal_status_blocks_launch(self):
        """Even a benign-looking model-side status is not a qualification pass."""
        for status in ("LimitsExceeded", "ContextWindowExceededError", "Submitted "):
            with self.subTest(status=status):
                env = _Env(pathlib.Path(tempfile.mkdtemp()))
                self.addCleanup(shutil.rmtree, env.tmp, True)
                statuses = env.statuses()
                statuses[_qual_ids()[1]] = status
                env.write_statuses(statuses)
                env.emit()
                self.assertBlocked(env.verify(), "Submitted")

    def test_classifier_names_the_forbidden_class(self):
        self.assertIsNotNone(sq.classify_forbidden_disposition("RepeatedFormatError"))
        self.assertIsNotNone(sq.classify_forbidden_disposition("RuntimeError"))
        self.assertIsNone(sq.classify_forbidden_disposition("Submitted"))

    def test_artifact_cannot_launder_a_forbidden_disposition(self):
        """Hand-editing dispositions in the artifact does not beat the evidence."""
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        artifact = self.env.emit()
        artifact["dispositions"] = {iid: "Submitted" for iid in _qual_ids()}
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify(), "RepeatedFormatError")

    def test_missing_disposition_blocks_launch(self):
        statuses = self.env.statuses()
        del statuses[_qual_ids()[0]]
        self.env.write_statuses(statuses)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "exit_statuses")

    def test_foreign_disposition_id_blocks_launch(self):
        statuses = self.env.statuses()
        statuses["some__other-123"] = "Submitted"
        self.env.write_statuses(statuses)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "foreign")


# ---------------------------------------------------------------------------
# 5. preds.json model_patch checks
# ---------------------------------------------------------------------------


class TestPredictionPatches(_EnvCase):
    """Every qualification prediction must be a nonempty, patch-like diff."""

    def test_empty_model_patch_blocks_launch(self):
        preds = self.env.preds()
        preds[_qual_ids()[2]]["model_patch"] = ""
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "model_patch")

    def test_whitespace_only_model_patch_blocks_launch(self):
        preds = self.env.preds()
        preds[_qual_ids()[2]]["model_patch"] = "   \n\t\n"
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "model_patch")

    def test_null_model_patch_blocks_launch(self):
        preds = self.env.preds()
        preds[_qual_ids()[2]]["model_patch"] = None
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "model_patch")

    def test_non_patch_like_model_patch_blocks_launch(self):
        preds = self.env.preds()
        preds[_qual_ids()[4]]["model_patch"] = "I could not solve this task, sorry."
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "patch-like")

    def test_missing_prediction_blocks_launch(self):
        preds = self.env.preds()
        del preds[_qual_ids()[5]]
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "preds.json")

    def test_foreign_prediction_blocks_launch(self):
        preds = self.env.preds()
        preds["foreign__repo-1"] = {"model_patch": "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"}
        self.env.write_preds(preds)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "foreign")

    def test_patch_like_helper_boundaries(self):
        self.assertTrue(sq.is_patch_like("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"))
        self.assertTrue(sq.is_patch_like("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"))
        self.assertFalse(sq.is_patch_like(""))
        self.assertFalse(sq.is_patch_like("   "))
        self.assertFalse(sq.is_patch_like(None))
        self.assertFalse(sq.is_patch_like("diff --git a/x b/x\n"))  # no hunk
        self.assertFalse(sq.is_patch_like("just prose"))


# ---------------------------------------------------------------------------
# 6. Trajectory evidence and malformed tool calls
# ---------------------------------------------------------------------------


class TestTrajectoryEvidence(_EnvCase):
    """Trajectories must exist, be nonempty, and hold well-formed tool calls."""

    def test_missing_trajectory_blocks_launch(self):
        (self.env.raw / "trajectories" / f"{_qual_ids()[0]}.traj").unlink()
        self.env.emit()
        self.assertBlocked(self.env.verify(), "trajectory")

    def test_empty_trajectory_blocks_launch(self):
        (self.env.raw / "trajectories" / f"{_qual_ids()[0]}.traj").write_text("")
        self.env.emit()
        self.assertBlocked(self.env.verify(), "trajectory")

    def test_malformed_tool_call_arguments_block_launch(self):
        iid = _qual_ids()[1]
        traj = self.env.traj(iid)
        traj["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"command": '
        self.env.write_traj(iid, traj)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "tool-call")

    def test_empty_tool_call_arguments_block_launch(self):
        iid = _qual_ids()[1]
        traj = self.env.traj(iid)
        traj["messages"][2]["tool_calls"][0]["function"]["arguments"] = ""
        self.env.write_traj(iid, traj)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "tool-call")

    def test_non_object_tool_call_arguments_block_launch(self):
        iid = _qual_ids()[2]
        traj = self.env.traj(iid)
        traj["messages"][2]["tool_calls"][0]["function"]["arguments"] = "[1, 2, 3]"
        self.env.write_traj(iid, traj)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "tool-call")

    def test_trajectory_with_no_tool_calls_at_all_blocks_launch(self):
        iid = _qual_ids()[3]
        traj = self.env.traj(iid)
        for msg in traj["messages"]:
            msg.pop("tool_calls", None)
        self.env.write_traj(iid, traj)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "tool call")

    def test_scanner_finds_malformed_arguments_at_any_depth(self):
        good = {"a": {"b": [{"tool_calls": [
            {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}]}}
        bad = {"a": {"b": [{"tool_calls": [
            {"function": {"name": "bash", "arguments": '{"command":'}}]}]}}
        self.assertEqual(sq.scan_tool_calls(good)[1], [])
        self.assertTrue(sq.scan_tool_calls(bad)[1])

    def test_scanner_counts_well_formed_calls(self):
        obj = {"tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
        count, problems = sq.scan_tool_calls(obj)
        self.assertEqual(count, 1)
        self.assertEqual(problems, [])


# ---------------------------------------------------------------------------
# 7. Official grader evidence
# ---------------------------------------------------------------------------


class TestOfficialGraderEvidence(_EnvCase):
    """Every ID must appear exactly once in official terminal grading."""

    def test_resolved_or_unresolved_both_acceptable(self):
        grading = self.env.grading()
        grading["resolved_ids"] = []
        grading["unresolved_ids"] = list(_qual_ids())
        self.env.write_grading(grading)
        self.env.emit()
        self.assertTrue(self.env.verify().ok, self.env.verify().errors)

        grading["resolved_ids"] = list(_qual_ids())
        grading["unresolved_ids"] = []
        self.env.write_grading(grading)
        self.env.emit()
        self.assertTrue(self.env.verify().ok, self.env.verify().errors)

    def test_grading_error_id_blocks_launch(self):
        grading = self.env.grading()
        moved = grading["unresolved_ids"].pop()
        grading["error_ids"] = [moved]
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "error_ids")

    def test_grading_incomplete_id_blocks_launch(self):
        grading = self.env.grading()
        moved = grading["unresolved_ids"].pop()
        grading["incomplete_ids"] = [moved]
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "incomplete_ids")

    def test_grading_empty_patch_id_blocks_launch(self):
        grading = self.env.grading()
        moved = grading["unresolved_ids"].pop()
        grading["empty_patch_ids"] = [moved]
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "empty_patch_ids")

    def test_duplicate_grading_disposition_blocks_launch(self):
        grading = self.env.grading()
        grading["resolved_ids"] = grading["resolved_ids"] + [grading["unresolved_ids"][0]]
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "duplicate")

    def test_missing_grading_disposition_blocks_launch(self):
        grading = self.env.grading()
        grading["unresolved_ids"].pop()
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "grading")

    def test_foreign_grading_id_blocks_launch(self):
        grading = self.env.grading()
        grading["unresolved_ids"].append("foreign__repo-9")
        self.env.write_grading(grading)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "foreign")

    def test_missing_grader_report_blocks_launch(self):
        self.env.emit()
        self.env.report.unlink()
        self.assertBlocked(self.env.verify(), "grader")

    def test_grader_report_digest_mismatch_blocks_launch(self):
        self.env.emit()
        grading = self.env.grading()
        grading["resolved_instances"] = 999
        self.env.write_grading(grading)  # artifact NOT re-emitted: digest now stale
        self.assertBlocked(self.env.verify(), "sha256")

    def test_preds_digest_mismatch_blocks_launch(self):
        self.env.emit()
        preds = self.env.preds()
        preds[_qual_ids()[0]]["model_patch"] += "\n"
        self.env.write_preds(preds)
        self.assertBlocked(self.env.verify(), "sha256")

    def test_exit_statuses_digest_mismatch_blocks_launch(self):
        self.env.emit()
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "Submitted"
        (self.env.raw / "exit_statuses.json").write_text(json.dumps(statuses))  # reformatted
        self.assertBlocked(self.env.verify(), "sha256")

    def test_missing_evidence_directory_blocks_launch(self):
        self.env.emit()
        shutil.rmtree(self.env.run)
        self.assertBlocked(self.env.verify())


# ---------------------------------------------------------------------------
# 8. Identity binding at launch
# ---------------------------------------------------------------------------


class TestLaunchBinding(_EnvCase):
    """Every recorded digest must match the launch context exactly."""

    def test_repo_sha_mismatch_blocks_launch(self):
        self.env.emit(repo_sha="f" * 40)
        self.assertBlocked(self.env.verify(), "repo_sha")

    def test_unresolvable_repo_sha_blocks_launch(self):
        self.env.emit()
        with patch.object(sq, "resolve_repo_sha", return_value=None):
            self.assertBlocked(self.env.verify(), "repo")

    def test_suite_id_mismatch_blocks_launch(self):
        self.env.emit(suite_id="warpcore-v2")
        self.assertBlocked(self.env.verify(), "suite_id")

    def test_suite_input_hash_mismatch_blocks_launch(self):
        artifact = self.env.emit()
        hashes = dict(artifact["suite_input_hashes"])
        hashes["suite/warpcore-v1.yaml"] = "0" * 64
        self.env.emit(suite_input_hashes=hashes)
        self.assertBlocked(self.env.verify(), "suite_input_hashes")

    def test_missing_suite_input_hash_blocks_launch(self):
        artifact = self.env.emit()
        hashes = dict(artifact["suite_input_hashes"])
        hashes.pop("suite/swebench/scaffold.yaml", None)
        self.env.emit(suite_input_hashes=hashes)
        self.assertBlocked(self.env.verify(), "suite_input_hashes")

    def test_adapter_hash_mismatch_blocks_launch(self):
        self.env.emit(adapter_hash="1" * 64)
        self.assertBlocked(self.env.verify(), "adapter_hash")

    def test_serving_profile_digest_mismatch_blocks_launch(self):
        self.env.emit(serving_profile_digest="sha256:" + "2" * 64)
        self.assertBlocked(self.env.verify(), "serving_profile_digest")

    def test_model_id_mismatch_blocks_launch(self):
        artifact = self.env.emit()
        model = dict(artifact["model"])
        model["id"] = "someoneelse/other-model"
        self.env.emit(model=model)
        self.assertBlocked(self.env.verify(), "model")

    def test_model_revision_mismatch_blocks_launch(self):
        artifact = self.env.emit()
        model = dict(artifact["model"])
        model["revision"] = "d" * 40
        self.env.emit(model=model)
        self.assertBlocked(self.env.verify(), "revision")

    def test_scaffold_hash_mismatch_blocks_launch(self):
        self.env.emit(scaffold_sha256="3" * 64)
        self.assertBlocked(self.env.verify(), "scaffold")

    def test_production_scaffold_hash_mismatch_blocks_launch(self):
        self.env.emit(production_scaffold_hash="sha256:" + "4" * 64)
        self.assertBlocked(self.env.verify(), "production_scaffold_hash")

    def test_non_production_launch_path_blocks_launch(self):
        self.env.emit(launch_path="scripts/quick_smoke.sh")
        self.assertBlocked(self.env.verify(), "launch_path")

    def test_endpoint_url_mismatch_blocks_launch(self):
        self.env.emit()
        self.assertBlocked(
            self.env.verify(endpoint="http://someotherhost:8000/v1"), "endpoint"
        )

    def test_endpoint_model_identity_mismatch_blocks_launch(self):
        artifact = self.env.emit()
        endpoint = dict(artifact["endpoint"])
        endpoint["served_model_id"] = "openai/gpt-oss-20b"
        self.env.emit(endpoint=endpoint)
        self.assertBlocked(self.env.verify(), "served_model_id")

    def test_qualification_ids_mismatch_blocks_launch(self):
        """Adapter/CLI-chosen convenience cases cannot stand in for the suite set."""
        convenient = _frozen_ids()[:20]
        self.assertNotEqual(convenient, _qual_ids())
        self.env.emit(qualification_instance_ids=convenient)
        self.assertBlocked(self.env.verify(), "qualification_instance_ids")

    def test_qualification_ids_hash_mismatch_blocks_launch(self):
        self.env.emit(qualification_ids_sha256="5" * 64)
        self.assertBlocked(self.env.verify(), "qualification_ids_sha256")

    def test_wrong_number_of_qualification_ids_blocks_launch(self):
        self.env.emit(qualification_instance_ids=_qual_ids()[:19])
        self.assertBlocked(self.env.verify(), "20")

    def test_duplicate_qualification_id_blocks_launch(self):
        ids = _qual_ids()[:19] + [_qual_ids()[0]]
        self.env.emit(qualification_instance_ids=ids)
        self.assertBlocked(self.env.verify())

    def test_another_adapter_cannot_reuse_the_qualification(self):
        other = self.tmp / "adapters" / "qwen3.6-35b-a3b.yaml"
        other.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / "adapters" / "qwen3.6-35b-a3b.yaml", other)
        self.env.emit()
        self.assertBlocked(self.env.verify(adapter_path=other))


# ---------------------------------------------------------------------------
# 9. Freshness
# ---------------------------------------------------------------------------


class TestFreshness(_EnvCase):
    """A qualification expires; a stale one cannot authorize a launch."""

    def test_fresh_qualification_passes(self):
        now = datetime.now(tz=timezone.utc)
        self.env.emit(now=now - timedelta(hours=1))
        self.assertTrue(self.env.verify(now=now).ok)

    def test_stale_qualification_blocks_launch(self):
        now = datetime.now(tz=timezone.utc)
        self.env.emit(now=now - timedelta(days=30))
        self.assertBlocked(self.env.verify(now=now), "stale")

    def test_boundary_just_inside_window_passes(self):
        now = datetime.now(tz=timezone.utc)
        hours = sq.qualification_max_age_hours(_REAL_SUITE)
        self.env.emit(now=now - timedelta(hours=hours) + timedelta(minutes=1))
        self.assertTrue(self.env.verify(now=now).ok)

    def test_boundary_just_outside_window_blocks(self):
        now = datetime.now(tz=timezone.utc)
        hours = sq.qualification_max_age_hours(_REAL_SUITE)
        self.env.emit(now=now - timedelta(hours=hours) - timedelta(minutes=1))
        self.assertBlocked(self.env.verify(now=now), "stale")

    def test_future_dated_qualification_blocks_launch(self):
        now = datetime.now(tz=timezone.utc)
        self.env.emit(now=now + timedelta(hours=5))
        self.assertBlocked(self.env.verify(now=now), "future")

    def test_non_utc_timestamp_is_rejected(self):
        self.env.emit(generated_utc="2026-09-21 12:00:00")
        self.assertBlocked(self.env.verify(), "generated_utc")


# ---------------------------------------------------------------------------
# 10. Runner integration — launch fails closed
# ---------------------------------------------------------------------------


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
_PROMPT_TOKEN_MAXIMA = {"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000}


class TestRunnerIntegration(unittest.TestCase):
    """viz/run_swebench.py must consult the authoritative gate before launching."""

    def setUp(self):
        import yaml
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))
        self.slug = _CANONICAL_ADAPTER["model"]["slug"]
        self.run_dir = self.tmp / "results" / self.slug / "runs" / "warpcore-v1" / "swebench" / "run-test"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "status.json").write_text(json.dumps({
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
        }))
        (self.run_dir / "manifest.json").write_text(json.dumps({
            "suite_id": "warpcore-v1", "run_id": "run-test", "benchmark": "swebench",
            "model": _CANONICAL_ADAPTER["model"], "item_inventory": {"expected": 100},
        }))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def _qual_dir(self) -> pathlib.Path:
        return sq.default_artifact_path(self.tmp, "warpcore-v1", self.slug).parent

    def _install_qualification(self, mutate_statuses=None) -> _Env:
        qual_dir = self._qual_dir()
        qual_dir.parent.mkdir(parents=True, exist_ok=True)
        env = _Env.__new__(_Env)
        env.tmp = self.tmp
        env.endpoint = "http://localhost:8000/v1"
        env.root = qual_dir
        qual_dir.mkdir(parents=True, exist_ok=True)
        env.run = qual_dir / "run"
        shutil.copytree(_FIXTURE_RUN, env.run)
        env.raw = env.run / "raw"
        env.artifact_path = qual_dir / "qualification.json"
        env.adapter_path = self.adapter_path
        env.report = next(
            p for p in sorted(env.raw.glob("*.json"))
            if p.name not in ("preds.json", "exit_statuses.json", "grading_results.json")
        )
        if mutate_statuses is not None:
            statuses = env.statuses()
            mutate_statuses(statuses)
            env.write_statuses(statuses)
        env.emit()
        return env

    def _runner(self, **kw):
        return run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            **kw,
        )

    def _make_generation_artifacts(self, run_dir):
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        instances = _frozen_ids()
        (raw_dir / "preds.json").write_text(json.dumps(
            {i: {"model_patch": "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n", "instance_id": i}
             for i in instances}))
        (raw_dir / "exit_statuses.json").write_text(json.dumps({i: 0 for i in instances}))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for i in instances:
            (traj_dir / f"{i}.traj").write_text("{}")
        (raw_dir / "run.log").write_text("generation completed\n")

    def _make_grading_artifacts(self, run_dir):
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        instances = _frozen_ids()
        (raw_dir / "grading_results.json").write_text(json.dumps({
            "resolved_ids": instances[:5], "unresolved_ids": instances[5:],
            "empty_patch_ids": [], "error_ids": [],
        }))

    # -- tests -----------------------------------------------------------

    def test_live_launch_without_qualification_fails_closed(self):
        rc = self._runner(allow_no_screen=True).run()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_live_launch_without_qualification_writes_nothing(self):
        self._runner(allow_no_screen=True).run()
        self.assertFalse((self.run_dir / "command.txt").exists())
        self.assertFalse((self.run_dir / "DONE").exists())
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "planned")

    def test_live_launch_with_repeated_format_error_qualification_fails_closed(self):
        def mutate(statuses):
            statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self._install_qualification(mutate_statuses=mutate)
        rc = self._runner(allow_no_screen=True).run()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)
        self.assertFalse((self.run_dir / "command.txt").exists())

    def test_live_launch_with_valid_qualification_proceeds(self):
        self._install_qualification()
        rc = self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=lambda cfg, rd, **kw: (self._make_generation_artifacts(rd), 0)[1],
            grading_runner=lambda p, rd, **kw: (self._make_grading_artifacts(rd), 0)[1],
        ).run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

    def test_stale_qualification_fails_closed(self):
        env = self._install_qualification()
        env.emit(now=datetime.now(tz=timezone.utc) - timedelta(days=30))
        rc = self._runner(allow_no_screen=True).run()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_qualification_for_another_model_fails_closed(self):
        env = self._install_qualification()
        artifact = json.loads(env.artifact_path.read_text())
        artifact["model"] = dict(artifact["model"], id="someoneelse/other")
        env.write_artifact(artifact)
        rc = self._runner(allow_no_screen=True).run()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_dry_run_reports_qualification_ok_and_is_side_effect_free(self):
        self._install_qualification()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._runner(dry_run=True).run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertIn("QUALIFICATION", buf.getvalue().upper())
        self.assertIn("OK", buf.getvalue().upper())
        self.assertFalse((self.run_dir / "command.txt").exists())

    def test_dry_run_reports_qualification_blocked(self):
        def mutate(statuses):
            statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self._install_qualification(mutate_statuses=mutate)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._runner(dry_run=True).run()
        out = buf.getvalue()
        # A dry run inspects and reports; it stays exit 0 and side-effect-free
        # even when the verdict is BLOCKED. Only a live launch refuses.
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertIn("BLOCKED", out.upper())
        self.assertIn("RepeatedFormatError", out)
        self.assertFalse((self.run_dir / "command.txt").exists())
        self.assertFalse((self.run_dir / "raw").exists())

    def test_dry_run_reports_missing_qualification(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._runner(dry_run=True).run()
        self.assertIn("BLOCKED", buf.getvalue().upper())

    def test_runner_build_scaffold_config_uses_the_production_builder(self):
        self._install_qualification()
        runner = self._runner(dry_run=True)
        built = runner.build_scaffold_config(endpoint="http://localhost:8000/v1", api_key="k")
        import yaml
        expected = sq.build_production_scaffold_config(
            scaffold=yaml.safe_load(_REAL_SCAFFOLD.read_text()),
            model_id=_CANONICAL_ADAPTER["model"]["id"],
            endpoint="http://localhost:8000/v1",
            api_key="k",
        )
        self.assertEqual(built, expected)

    def test_main_fails_closed_before_creating_a_campaign(self):
        """No qualification ⇒ main() must not even create the campaign directory."""
        rc = run_swebench.main([
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.adapter_path),
            "--endpoint", "http://localhost:8000/v1",
            "--repo", str(self.tmp),
            "--run-id", "run-never-created",
            "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
            "--allow-no-screen",
        ])
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse(
            (self.tmp / "results" / self.slug / "runs" / "warpcore-v1" / "swebench"
             / "run-never-created").exists(),
            "campaign directory must not be created when qualification is missing",
        )


# ---------------------------------------------------------------------------
# 10b. The qualification run itself — produced by the same production path
# ---------------------------------------------------------------------------


class TestQualificationRunMode(unittest.TestCase):
    """The 20-instance qualification run is the production runner in a narrow mode.

    Without this the gate would be unreachable: a qualification run cannot itself
    require a qualification, but it must still be the production path, or the
    production_scaffold_hash it records would authorize a config nothing ran.
    """

    def setUp(self):
        import yaml
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))
        self.slug = _CANONICAL_ADAPTER["model"]["slug"]
        self.evidence = (
            self.tmp / "results" / self.slug / "qualification"
            / "warpcore-v1" / "swebench" / "run"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self, **kw):
        return run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.evidence,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            qualification_run=True,
            **kw,
        )

    def test_qualification_run_uses_exactly_the_suite_owned_twenty(self):
        runner = self._runner(dry_run=True)
        self.assertEqual(runner.get_instance_ids(), _qual_ids())

    def test_campaign_run_still_uses_the_frozen_hundred(self):
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.evidence,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
        )
        self.assertEqual(len(runner.get_instance_ids()), 100)

    def test_qualification_run_does_not_require_a_qualification(self):
        """No chicken-and-egg: the run that produces the record is not gated by it."""
        calls = []

        def gen(cfg, run_dir, **kw):
            calls.append(cfg)
            shutil.copytree(_FIXTURE_RUN / "raw", run_dir / "raw", dirs_exist_ok=True)
            (run_dir / "raw" / "run.log").write_text("qualification generation\n")
            return 0

        rc = self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=gen,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS, "qualification run must not be gated")
        self.assertEqual(len(calls), 1)

    def test_qualification_run_uses_the_production_config_builder(self):
        import yaml
        seen = {}

        def gen(cfg, run_dir, **kw):
            seen["cfg"] = cfg
            shutil.copytree(_FIXTURE_RUN / "raw", run_dir / "raw", dirs_exist_ok=True)
            (run_dir / "raw" / "run.log").write_text("qualification generation\n")
            return 0

        self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=gen,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        expected = sq.build_production_scaffold_config(
            scaffold=yaml.safe_load(_REAL_SCAFFOLD.read_text()),
            model_id=_CANONICAL_ADAPTER["model"]["id"],
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        self.assertEqual(seen["cfg"], expected)

    def test_qualification_run_output_seals_into_a_passing_record(self):
        """End to end: production qualification run -> emit -> gate says OK."""
        def gen(cfg, run_dir, **kw):
            shutil.copytree(_FIXTURE_RUN / "raw", run_dir / "raw", dirs_exist_ok=True)
            (run_dir / "raw" / "run.log").write_text("qualification generation\n")
            return 0

        rc = self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=gen,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

        artifact_path = self.evidence.parent / "qualification.json"
        artifact = sq.build_qualification_artifact(
            repo=_REPO,
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            served_model_id=_CANONICAL_ADAPTER["model"]["id"],
            evidence_run_dir=self.evidence,
            artifact_path=artifact_path,
        )
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
        result = sq.verify_qualification_for_launch(
            repo=_REPO,
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            artifact_path=artifact_path,
        )
        self.assertTrue(result.ok, result.errors)

    def test_qualification_run_refuses_a_forbidden_disposition(self):
        """The mode does not launder a bad run: it fails before any record exists."""
        def gen(cfg, run_dir, **kw):
            shutil.copytree(_FIXTURE_RUN / "raw", run_dir / "raw", dirs_exist_ok=True)
            (run_dir / "raw" / "run.log").write_text("qualification generation\n")
            statuses_path = run_dir / "raw" / "exit_statuses.json"
            statuses = json.loads(statuses_path.read_text())
            statuses[_qual_ids()[0]] = "RepeatedFormatError"
            statuses_path.write_text(json.dumps(statuses, indent=2) + "\n")
            return 0

        rc = self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=gen,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_qualification_run_writes_no_campaign_state(self):
        """A qualification is not a campaign: no status.json, manifest, or DONE."""
        def gen(cfg, run_dir, **kw):
            shutil.copytree(_FIXTURE_RUN / "raw", run_dir / "raw", dirs_exist_ok=True)
            (run_dir / "raw" / "run.log").write_text("qualification generation\n")
            return 0

        self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 0,
            generation_runner=gen,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        for name in ("status.json", "manifest.json", "DONE"):
            self.assertFalse((self.evidence / name).exists(), name)

    def test_qualification_run_blocks_on_failed_preflight(self):
        rc = self._runner(
            allow_no_screen=True,
            preflight_runner=lambda model_id: 1,
            generation_runner=lambda cfg, rd, **kw: 0,
            grading_runner=lambda p, rd, **kw: 0,
        ).run_qualification()
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_qualification_run_refuses_outside_screen(self):
        runner = self._runner(preflight_runner=lambda model_id: 0)
        env_without_sty = {k: v for k, v in os.environ.items() if k != "STY"}
        with patch.dict(os.environ, env_without_sty, clear=True):
            self.assertEqual(runner.run_qualification(), run_swebench.EXIT_INCONCLUSIVE)


# ---------------------------------------------------------------------------
# 11. RepeatedFormatError can never authorize a launch — exhaustive sweep
# ---------------------------------------------------------------------------


class TestRepeatedFormatErrorCanNeverAuthorize(_EnvCase):
    """Sweep every place a RepeatedFormatError could be hidden."""

    def test_hidden_in_exit_statuses(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[7]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "RepeatedFormatError")

    def test_hidden_in_trajectory_info_exit_status(self):
        iid = _qual_ids()[7]
        traj = self.env.traj(iid)
        traj["info"]["exit_status"] = "RepeatedFormatError"
        self.env.write_traj(iid, traj)
        self.env.emit()
        self.assertBlocked(self.env.verify(), "RepeatedFormatError")

    def test_hidden_behind_a_hand_written_artifact(self):
        """A fully hand-written artifact with no evidence cannot pass."""
        artifact = self.env.emit()
        shutil.rmtree(self.env.run)
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify())

    def test_hidden_by_shrinking_the_qualification_set(self):
        """Dropping the failing instance from the set does not qualify the rest."""
        statuses = self.env.statuses()
        bad = _qual_ids()[9]
        statuses[bad] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        artifact = self.env.emit()
        artifact["qualification_instance_ids"] = [i for i in _qual_ids() if i != bad]
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify())

    def test_hidden_by_claiming_a_different_grader_report(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[9]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        artifact = self.env.emit()
        evidence = dict(artifact["evidence"])
        evidence["exit_statuses_file"] = "raw/does-not-exist.json"
        artifact["evidence"] = evidence
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify())

    def test_evidence_path_escape_is_rejected(self):
        artifact = self.env.emit()
        evidence = dict(artifact["evidence"])
        evidence["run_dir"] = "../../elsewhere"
        artifact["evidence"] = evidence
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify())

    def test_absolute_evidence_path_is_rejected(self):
        artifact = self.env.emit()
        evidence = dict(artifact["evidence"])
        evidence["run_dir"] = "/tmp/elsewhere"
        artifact["evidence"] = evidence
        self.env.write_artifact(artifact)
        self.assertBlocked(self.env.verify())


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------


class TestCli(_EnvCase):
    """The module is runnable as a make/CI target."""

    def test_verify_cli_exit_zero_on_valid(self):
        self.env.emit()
        rc = sq.main([
            "verify",
            "--artifact", str(self.env.artifact_path),
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.env.adapter_path),
            "--endpoint", _ENDPOINT,
            "--repo", str(_REPO),
        ])
        self.assertEqual(rc, 0)

    def test_verify_cli_exit_one_on_forbidden_disposition(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        self.env.emit()
        rc = sq.main([
            "verify",
            "--artifact", str(self.env.artifact_path),
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.env.adapter_path),
            "--endpoint", _ENDPOINT,
            "--repo", str(_REPO),
        ])
        self.assertEqual(rc, 1)

    def test_emit_cli_refuses_to_write_a_failing_artifact(self):
        statuses = self.env.statuses()
        statuses[_qual_ids()[0]] = "RepeatedFormatError"
        self.env.write_statuses(statuses)
        target = self.env.root / "emitted.json"
        rc = sq.main([
            "emit",
            "--artifact", str(target),
            "--evidence", str(self.env.run),
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.env.adapter_path),
            "--endpoint", _ENDPOINT,
            "--served-model-id", _served_model_id(self.env.adapter_path),
            "--repo", str(_REPO),
        ])
        self.assertEqual(rc, 1)
        self.assertFalse(target.exists(), "a failing qualification must not be written")

    def test_self_test_runs_offline(self):
        self.assertEqual(sq.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
