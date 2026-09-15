"""
Task 2 acceptance tests: contract validator for warpcore-v1 suite and adapters.

RED before viz/contract.py and viz/validate_suite.py exist.
GREEN after both are implemented.

Run: /usr/bin/python3 -m pytest tests/test_validate_suite.py -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
SUITE_DIR = REPO / "suite"
SUITE_FILE = SUITE_DIR / "warpcore-v1.yaml"
SCHEMAS_DIR = SUITE_DIR / "schemas"
TASKS_DIR = SUITE_DIR / "tasks"
SWEBENCH_DIR = SUITE_DIR / "swebench"

ADAPTER_SCHEMA = SCHEMAS_DIR / "adapter.schema.json"
SUITE_SCHEMA = SCHEMAS_DIR / "suite.schema.json"

# ---------------------------------------------------------------------------
# Import under test (will fail RED before files exist)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO / "viz"))

from contract import (  # noqa: E402
    load_yaml,
    sha256_file,
    validate_json,
    validate_suite,
    validate_adapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_minimal_suite(tmpdir: Path) -> tuple[Path, Path]:
    """Copy real suite files into a tmp directory and return (repo_root, suite_path)."""
    import shutil

    # Mirror the structure we need
    suite_dir = tmpdir / "suite"
    suite_dir.mkdir(parents=True)
    schemas_dst = suite_dir / "schemas"
    schemas_dst.mkdir()
    tasks_dst = suite_dir / "tasks"
    tasks_dst.mkdir()
    swe_dst = suite_dir / "swebench"
    swe_dst.mkdir()
    viz_dst = tmpdir / "viz"
    viz_dst.mkdir()

    # Copy schemas
    for p in SCHEMAS_DIR.iterdir():
        shutil.copy(p, schemas_dst / p.name)

    # Copy task files
    for p in TASKS_DIR.iterdir():
        shutil.copy(p, tasks_dst / p.name)

    # Copy swebench files
    for p in SWEBENCH_DIR.iterdir():
        shutil.copy(p, swe_dst / p.name)

    # Copy suite YAML verbatim
    suite_path = suite_dir / "warpcore-v1.yaml"
    shutil.copy(SUITE_FILE, suite_path)

    return tmpdir, suite_path


def _make_minimal_adapter(tmpdir: Path) -> Path:
    """Write a valid minimal adapter YAML inside *tmpdir* and return its path.

    The adapter is placed inside tmpdir so that validate_adapter(tmpdir, path)
    does not trigger a path-escape error.  Callers that need a self-contained
    mini-repo should use _make_adapter_repo() instead.
    """
    adapter = {
        "adapter_schema_version": 1,
        "model": {
            "slug": "test-model",
            "id": "TestOrg/TestModel",
            "revision": "abc123def456abc123def456abc123def456abc1",
        },
        "serving": {
            "image": "registry/img@sha256:" + "de" * 32,
            "engine": "vllm",
            "engine_version": "0.8.5",
            "quantization": None,
            "reasoning_parser": None,
            "tool_call_parser": None,
            "tokenizer": None,
            "moe_backend": None,
            "max_model_len": 131072,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 32,
            "environment": {},
        },
    }
    path = tmpdir / "adapters" / "test-model.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(adapter))
    return path


def _make_adapter_repo(tmpdir: Path) -> tuple[Path, Path]:
    """Create a minimal repo layout inside *tmpdir* with a valid adapter.

    Returns (repo_root, adapter_path).  The repo root contains a copy of the
    real adapter schema so validate_adapter can resolve it.
    """
    import shutil

    schemas_dst = tmpdir / "suite" / "schemas"
    schemas_dst.mkdir(parents=True)
    for p in SCHEMAS_DIR.iterdir():
        shutil.copy(p, schemas_dst / p.name)

    adapter_path = _make_minimal_adapter(tmpdir)
    return tmpdir, adapter_path


# ===========================================================================
# 1.  load_yaml
# ===========================================================================

class TestLoadYaml:
    def test_returns_dict(self):
        result = load_yaml(SUITE_FILE)
        assert isinstance(result, dict)

    def test_parses_suite_id(self):
        result = load_yaml(SUITE_FILE)
        assert result["suite_id"] == "warpcore-v1"

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_yaml(Path("/nonexistent/file.yaml"))


# ===========================================================================
# 2.  sha256_file
# ===========================================================================

class TestSha256File:
    def test_known_file(self):
        # Compute reference hash for the suite file
        expected = hashlib.sha256(SUITE_FILE.read_bytes()).hexdigest()
        assert sha256_file(SUITE_FILE) == expected

    def test_returns_64_hex(self):
        h = sha256_file(SUITE_FILE)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            sha256_file(Path("/nonexistent/file.txt"))


# ===========================================================================
# 3.  validate_json
# ===========================================================================

class TestValidateJson:
    """validate_json must enforce JSON Schema formats via FormatChecker."""

    def _valid_adapter_dict(self) -> dict:
        return {
            "adapter_schema_version": 1,
            "model": {
                "slug": "qwen3.6-35b-a3b",
                "id": "Qwen/Qwen3.6-35B-A3B-FP8",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
            "serving": {
                "image": "eugr/spark-vllm@sha256:" + "de" * 32,
                "engine": "vllm",
                "engine_version": "0.8.5",
                "quantization": "fp8",
                "reasoning_parser": None,
                "tool_call_parser": None,
                "tokenizer": None,
                "moe_backend": None,
                "max_model_len": 131072,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 32,
                "environment": {},
            },
        }

    def test_valid_adapter_returns_empty_list(self):
        errors = validate_json(self._valid_adapter_dict(), ADAPTER_SCHEMA)
        assert errors == []

    def test_returns_list(self):
        result = validate_json(self._valid_adapter_dict(), ADAPTER_SCHEMA)
        assert isinstance(result, list)

    def test_invalid_adapter_returns_errors(self):
        bad = {**self._valid_adapter_dict(), "generation_ceiling": 99999}
        errors = validate_json(bad, ADAPTER_SCHEMA)
        assert len(errors) > 0
        assert isinstance(errors[0], str)

    def test_unknown_key_returns_error(self):
        bad = {**self._valid_adapter_dict(), "unknown_experiment_key": "bad"}
        errors = validate_json(bad, ADAPTER_SCHEMA)
        assert len(errors) > 0

    def test_missing_required_field_returns_error(self):
        bad = copy.deepcopy(self._valid_adapter_dict())
        del bad["model"]["id"]
        errors = validate_json(bad, ADAPTER_SCHEMA)
        assert len(errors) > 0

    def test_valid_suite_returns_empty_list(self):
        suite = load_yaml(SUITE_FILE)
        errors = validate_json(suite, SUITE_SCHEMA)
        assert errors == []


# ===========================================================================
# 4.  validate_suite — valid suite => []
# ===========================================================================

class TestValidateSuiteValid:
    def test_real_suite_is_valid(self):
        """The committed warpcore-v1.yaml must validate clean."""
        errors = validate_suite(REPO, SUITE_FILE)
        assert errors == [], f"Unexpected errors: {errors}"


# ===========================================================================
# 5.  validate_suite — stale task hash
# ===========================================================================

class TestValidateSuiteStaleTaskHash:
    def test_stale_gsm8k_hash_detected(self):
        """If task_sha256 declared in suite doesn't match actual file => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            # Corrupt the declared hash in the suite
            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["gsm8k"]["task_sha256"] = "a" * 64
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert any("gsm8k" in e.lower() or "task_sha256" in e.lower() or "hash" in e.lower() for e in errors), \
                f"Expected hash mismatch error, got: {errors}"

    def test_stale_gpqa_utils_hash_detected(self):
        """Stale utils_sha256 must be flagged."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["gpqa_diamond"]["utils_sha256"] = "b" * 64
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert any("gpqa" in e.lower() or "utils" in e.lower() or "hash" in e.lower() for e in errors), \
                f"Expected hash mismatch error, got: {errors}"


# ===========================================================================
# 6.  validate_suite — changed canonical file without version/hash update
# ===========================================================================

class TestValidateSuiteChangedCanonicalFile:
    def test_mutated_task_file_detected(self):
        """Mutating a canonical task file without updating the declared hash => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            # Append a byte to the gsm8k task file
            task_file = tmpdir / "suite" / "tasks" / "gsm8k_clean_v1.yaml"
            task_file.write_bytes(task_file.read_bytes() + b"\n# mutated")

            # Suite still declares old hash — should fail
            errors = validate_suite(repo_root, suite_path)
            assert any("hash" in e.lower() or "gsm8k" in e.lower() for e in errors), \
                f"Expected hash mismatch error, got: {errors}"


# ===========================================================================
# 7.  validate_suite — duplicate or missing SWE-bench IDs
# ===========================================================================

class TestValidateSuiteSwebenchIds:
    def test_duplicate_swebench_ids_detected(self):
        """Duplicate instance IDs in instances-seed42-n100.json => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            instances_file = tmpdir / "suite" / "swebench" / "instances-seed42-n100.json"
            instances = json.loads(instances_file.read_text())
            # Introduce a duplicate
            instances[1] = instances[0]
            instances_file.write_text(json.dumps(instances))

            # Update hash in suite YAML to match new content
            new_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()
            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["swebench"]["instances_sha256"] = new_hash
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert any("duplicate" in e.lower() or "unique" in e.lower() for e in errors), \
                f"Expected duplicate ID error, got: {errors}"

    def test_expected_count_mismatch_detected(self):
        """Instance count != expected_item_count in suite => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            instances_file = tmpdir / "suite" / "swebench" / "instances-seed42-n100.json"
            instances = json.loads(instances_file.read_text())
            # Remove one instance
            instances = instances[:-1]
            instances_file.write_text(json.dumps(instances))

            # Update hash in suite YAML but leave expected_item_count=100
            new_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()
            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["swebench"]["instances_sha256"] = new_hash
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert any("count" in e.lower() or "expected" in e.lower() for e in errors), \
                f"Expected count mismatch error, got: {errors}"

    def test_missing_swebench_ids_detected(self):
        """Empty instance list (all IDs missing) => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            instances_file = tmpdir / "suite" / "swebench" / "instances-seed42-n100.json"
            instances_file.write_text(json.dumps([]))

            new_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()
            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["swebench"]["instances_sha256"] = new_hash
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert len(errors) > 0, "Expected errors for empty instance set"


# ===========================================================================
# 8.  validate_suite — unknown suite keys
# ===========================================================================

class TestValidateSuiteUnknownKeys:
    def test_unknown_top_level_key_detected(self):
        """Adding an unknown top-level key to suite YAML => schema error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            suite["unknown_experiment_key"] = "this should not be here"
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert len(errors) > 0, f"Expected schema error for unknown key, got: {errors}"


# ===========================================================================
# 9.  validate_suite — missing required schema fields
# ===========================================================================

class TestValidateSuiteMissingRequired:
    def test_missing_suite_id_detected(self):
        """Removing suite_id from suite YAML => schema error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            del suite["suite_id"]
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert len(errors) > 0, "Expected error for missing suite_id"

    def test_missing_statistical_policy_detected(self):
        """Removing statistical_policy from suite YAML => schema error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            del suite["statistical_policy"]
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert len(errors) > 0, "Expected error for missing statistical_policy"


# ===========================================================================
# 10.  validate_adapter — valid adapter => []
# ===========================================================================

class TestValidateAdapterValid:
    def test_valid_adapter_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)
            errors = validate_adapter(repo_root, adapter_path)
            assert errors == [], f"Unexpected errors: {errors}"


# ===========================================================================
# 11.  validate_adapter — generation_ceiling prohibited
# ===========================================================================

class TestValidateAdapterGenerationCeilingProhibited:
    def test_generation_ceiling_in_adapter_detected(self):
        """Adapter containing generation_ceiling => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)

            adapter = yaml.safe_load(adapter_path.read_text())
            adapter["generation_ceiling"] = 99999
            adapter_path.write_text(yaml.dump(adapter))

            errors = validate_adapter(repo_root, adapter_path)
            assert len(errors) > 0, "Expected error for generation_ceiling in adapter"
            assert any("generation_ceiling" in e.lower() or "experiment" in e.lower() or "prohibited" in e.lower() or "additional" in e.lower() for e in errors), \
                f"Error message should name the prohibited key, got: {errors}"

    def test_sampling_in_adapter_detected(self):
        """Adapter containing sampling (experiment-level key) => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)

            adapter = yaml.safe_load(adapter_path.read_text())
            adapter["sampling"] = {"temperature": 0.5}
            adapter_path.write_text(yaml.dump(adapter))

            errors = validate_adapter(repo_root, adapter_path)
            assert len(errors) > 0, "Expected error for sampling in adapter"

    def test_task_key_in_adapter_detected(self):
        """Adapter containing task key => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)

            adapter = yaml.safe_load(adapter_path.read_text())
            adapter["task"] = "gpqa_diamond_cot_zeroshot_clean"
            adapter_path.write_text(yaml.dump(adapter))

            errors = validate_adapter(repo_root, adapter_path)
            assert len(errors) > 0, "Expected error for task key in adapter"


# ===========================================================================
# 12.  validate_adapter — repository-relative path escape
# ===========================================================================

class TestValidateAdapterPathEscape:
    def test_path_escape_outside_repo_detected(self):
        """Adapter stored outside repo root => path escape error."""
        with tempfile.TemporaryDirectory() as td_adapter:
            with tempfile.TemporaryDirectory() as td_repo:
                repo_root = Path(td_repo)
                adapter_path = Path(td_adapter) / "escaped-adapter.yaml"

                adapter = yaml.safe_load(_make_minimal_adapter(Path(td_adapter)).read_text())
                adapter_path.write_text(yaml.dump(adapter))

                errors = validate_adapter(repo_root, adapter_path)
                assert len(errors) > 0, "Expected error for path outside repo"
                assert any("path" in e.lower() or "outside" in e.lower() or "escape" in e.lower() for e in errors), \
                    f"Expected path escape error, got: {errors}"


# ===========================================================================
# 13.  validate_suite — path escape for canonical files
# ===========================================================================

class TestValidateSuitePathEscape:
    def test_task_file_outside_repo_detected(self):
        """Suite declaring a task_file path outside the repo root => error."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            # Inject a path escape into gsm8k task_file
            suite["benchmarks"]["gsm8k"]["task_file"] = "/etc/passwd"
            suite_path.write_text(yaml.dump(suite))

            errors = validate_suite(repo_root, suite_path)
            assert len(errors) > 0, "Expected error for path escape in task_file"


# ===========================================================================
# 14.  CLI: viz/validate_suite.py exit codes and output
# ===========================================================================

class TestValidateSuiteCLI:
    """validate_suite.py CLI must print every error and exit with correct code."""

    def _run_cli(self, *args: str) -> tuple[int, str]:
        cmd = ["/usr/bin/python3", str(REPO / "viz" / "validate_suite.py")] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def test_valid_suite_exits_0(self):
        rc, output = self._run_cli(str(SUITE_FILE))
        assert rc == 0, f"Expected exit 0 for valid suite, got {rc}:\n{output}"

    def test_invalid_suite_exits_1(self):
        """A diagnosed defect (stale hash) => exit 1."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["gsm8k"]["task_sha256"] = "a" * 64
            suite_path.write_text(yaml.dump(suite))

            cmd = [
                "/usr/bin/python3", str(REPO / "viz" / "validate_suite.py"),
                "--repo", str(repo_root), str(suite_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 1, \
                f"Expected exit 1 for stale hash, got {result.returncode}:\n{result.stdout}{result.stderr}"

    def test_unreadable_input_exits_2(self):
        """An unreadable/nonexistent file => exit 2."""
        rc, output = self._run_cli("/nonexistent/suite.yaml")
        assert rc == 2, f"Expected exit 2 for nonexistent file, got {rc}:\n{output}"

    def test_errors_printed_to_output(self):
        """Every error must appear in stdout or stderr."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["gsm8k"]["task_sha256"] = "a" * 64
            suite_path.write_text(yaml.dump(suite))

            cmd = [
                "/usr/bin/python3", str(REPO / "viz" / "validate_suite.py"),
                "--repo", str(repo_root), str(suite_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            combined = result.stdout + result.stderr
            assert len(combined.strip()) > 0, "Expected error output but got nothing"


# ===========================================================================
# 15.  CLI: viz/validate_suite.py negative controls (mutation testing)
# ===========================================================================

class TestValidateSuiteCLINegativeControls:
    """AGENTS.md §Negative controls: break it, prove exit 1, restore, prove exit 0."""

    def _cli(self, *args: str) -> tuple[int, str]:
        cmd = ["/usr/bin/python3", str(REPO / "viz" / "validate_suite.py")] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def test_mutate_canonical_file_exits_1_restore_exits_0(self):
        """Mutate one byte in a canonical task file → exit 1. Restore → exit 0."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, suite_path = _make_minimal_suite(tmpdir)

            task_file = tmpdir / "suite" / "tasks" / "gsm8k_clean_v1.yaml"
            original_bytes = task_file.read_bytes()

            # Mutate
            task_file.write_bytes(original_bytes + b"\n# injected")

            cmd_args = ["--repo", str(repo_root), str(suite_path)]
            rc_bad, out_bad = self._cli(*cmd_args)
            assert rc_bad == 1, f"Expected exit 1 after mutation, got {rc_bad}:\n{out_bad}"

            # Restore
            task_file.write_bytes(original_bytes)
            rc_good, out_good = self._cli(*cmd_args)
            assert rc_good == 0, f"Expected exit 0 after restore, got {rc_good}:\n{out_good}"

    def test_prohibited_adapter_key_exits_1(self):
        """Adapter with generation_ceiling → CLI exits 1."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)

            adapter = yaml.safe_load(adapter_path.read_text())
            adapter["generation_ceiling"] = 99999
            adapter_path.write_text(yaml.dump(adapter))

            cmd = [
                "/usr/bin/python3", str(REPO / "viz" / "validate_suite.py"),
                "--adapter", str(adapter_path),
                "--repo", str(repo_root)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 1, \
                f"Expected exit 1 for prohibited adapter key, got {result.returncode}:\n{result.stdout}{result.stderr}"

    def test_valid_adapter_exits_0(self):
        """Valid adapter in same repo → CLI exits 0."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            repo_root, adapter_path = _make_adapter_repo(tmpdir)

            cmd = [
                "/usr/bin/python3", str(REPO / "viz" / "validate_suite.py"),
                "--adapter", str(adapter_path),
                "--repo", str(repo_root)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, \
                f"Expected exit 0 for valid adapter, got {result.returncode}:\n{result.stdout}{result.stderr}"
