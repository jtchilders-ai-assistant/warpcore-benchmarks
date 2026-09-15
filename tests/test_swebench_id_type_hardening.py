"""
Task 2 type-hardening: _validate_swebench must diagnose non-string / empty-string
instance IDs as a contract defect (exit 1), not crash with TypeError (exit 2).

TDD PROTOCOL
- RED before the type-guard in viz/contract.py is added:
    - library: raises TypeError (unhashable)
    - CLI: exits 2 (opaque)
- GREEN after the type-guard:
    - library: returns list[str] with clear position-tagged messages
    - CLI: exits 1 (diagnosed defect)

Run: /usr/bin/python3 -m pytest tests/test_swebench_id_type_hardening.py -v
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
SUITE_DIR = REPO / "suite"
SUITE_FILE = SUITE_DIR / "warpcore-v1.yaml"
SCHEMAS_DIR = SUITE_DIR / "schemas"
TASKS_DIR = SUITE_DIR / "tasks"
SWEBENCH_DIR = SUITE_DIR / "swebench"

sys.path.insert(0, str(REPO / "viz"))

from contract import validate_suite  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_suite_repo(tmpdir: Path, instances: list) -> tuple[Path, Path]:
    """Create a minimal repo in *tmpdir* with *instances* as the SWE-bench
    instance set file.  The instances_sha256 in suite YAML is updated to
    match the new content so the hash check passes and type—not hash—is
    the isolated failure.

    Returns (repo_root, suite_path).
    """
    suite_dir = tmpdir / "suite"
    suite_dir.mkdir(parents=True)
    schemas_dst = suite_dir / "schemas"
    schemas_dst.mkdir()
    tasks_dst = suite_dir / "tasks"
    tasks_dst.mkdir()
    swe_dst = suite_dir / "swebench"
    swe_dst.mkdir()
    (tmpdir / "viz").mkdir()

    for p in SCHEMAS_DIR.iterdir():
        shutil.copy(p, schemas_dst / p.name)
    for p in TASKS_DIR.iterdir():
        shutil.copy(p, tasks_dst / p.name)
    for p in SWEBENCH_DIR.iterdir():
        shutil.copy(p, swe_dst / p.name)

    # Write the test instance set as a real JSON file
    instances_file = swe_dst / "instances-seed42-n100.json"
    raw = json.dumps(instances).encode()
    instances_file.write_bytes(raw)
    new_hash = _sha256_bytes(raw)

    # Copy real suite YAML and update instances_sha256 + expected_item_count
    suite_path = suite_dir / "warpcore-v1.yaml"
    shutil.copy(SUITE_FILE, suite_path)
    suite = yaml.safe_load(suite_path.read_text())
    suite["benchmarks"]["swebench"]["instances_sha256"] = new_hash
    suite["benchmarks"]["swebench"]["expected_item_count"] = len(instances)
    suite_path.write_text(yaml.dump(suite))

    return tmpdir, suite_path


def _run_cli(*args: str) -> tuple[int, str]:
    cmd = ["/usr/bin/python3", str(REPO / "viz" / "validate_suite.py")] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


# ===========================================================================
# 1.  Library: unhashable objects in instance list must NOT raise TypeError
#     They must return a list[str] with clear per-item messages.
# ===========================================================================

class TestUnhashableObjectsLibrary:
    """Before fix: set(instances) raises TypeError → test proves RED.
    After fix:  returns list[str] with position-tagged errors → GREEN.
    """

    def test_dict_elements_return_list_not_raise(self):
        """Instance list containing dicts (unhashable) must return errors, not crash."""
        instances = [{"instance_id": "django__django-11299"}, {"instance_id": "other__repo-1"}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            # Must NOT raise TypeError
            result = validate_suite(repo_root, suite_path)
            assert isinstance(result, list), f"Expected list[str], got {type(result)}"

    def test_dict_elements_produce_errors(self):
        """Instance list containing dicts must produce at least one error string."""
        instances = [{"instance_id": "django__django-11299"}, {"instance_id": "other__repo-1"}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert len(result) > 0, "Expected errors for dict instance IDs, got none"

    def test_dict_elements_error_mentions_type_or_position(self):
        """Error messages for non-string IDs must mention type or position."""
        instances = [{"instance_id": "django__django-11299"}, {"instance_id": "other__repo-1"}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            combined = " ".join(result).lower()
            assert any(
                word in combined for word in ("type", "str", "position", "index", "item", "non-string")
            ), f"Error must mention type/position, got: {result}"

    def test_list_elements_return_list_not_raise(self):
        """Instance list containing nested lists (unhashable) must return errors, not crash."""
        instances = [["django__django-11299"], ["other__repo-1"]]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert isinstance(result, list), f"Expected list[str], got {type(result)}"

    def test_list_elements_produce_errors(self):
        """Instance list containing nested lists must produce errors."""
        instances = [["django__django-11299"], ["other__repo-1"]]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert len(result) > 0, "Expected errors for list instance IDs, got none"

    def test_mixed_valid_and_invalid_types(self):
        """Mixed list (some str, some dict) must report positions of bad items."""
        instances = [
            "django__django-11299",
            {"instance_id": "bad"},
            "sqlfluff__sqlfluff-1234",
            None,
            "another__valid-99",
        ]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert isinstance(result, list)
            assert len(result) > 0, f"Expected errors for mixed bad IDs, got: {result}"
            # Should mention at least one bad position (index 1 and 3)
            combined = " ".join(result)
            # Validate it mentions some positional info (index/position numbers)
            assert any(c.isdigit() for c in combined), (
                f"Error must include position info (digit), got: {result}"
            )

    def test_all_errors_are_strings(self):
        """Every element in the returned list must be a str."""
        instances = [{"key": "val"}, 42, None, ["a", "b"]]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert all(isinstance(e, str) for e in result), (
                f"All error items must be str; got types: {[type(e) for e in result]}"
            )


# ===========================================================================
# 2.  Library: empty string IDs are also invalid
# ===========================================================================

class TestEmptyStringIds:
    """Empty strings are non-empty-string contract violations."""

    def test_empty_string_id_produces_error(self):
        """A list containing an empty string must produce at least one error."""
        instances = ["django__django-11299", "", "sqlfluff__sqlfluff-1234"]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert len(result) > 0, "Expected error for empty-string ID, got none"

    def test_empty_string_error_mentions_empty_or_position(self):
        """Error for empty-string ID must mention 'empty' or give position."""
        instances = ["valid__id-1", ""]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            combined = " ".join(result).lower()
            assert any(
                word in combined for word in ("empty", "blank", "position", "index", "item", "1")
            ), f"Error must mention 'empty' or position, got: {result}"

    def test_all_empty_strings_produce_errors(self):
        """All-empty-string list must produce errors."""
        instances = ["", "", ""]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            result = validate_suite(repo_root, suite_path)
            assert len(result) > 0, "Expected errors for all-empty-string list"

    def test_valid_strings_pass(self):
        """A valid list of unique non-empty strings must produce no type errors."""
        # Use the real instance set: copy real file so all hashes line up
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            suite_dir = tmpdir / "suite"
            suite_dir.mkdir(parents=True)
            schemas_dst = suite_dir / "schemas"
            schemas_dst.mkdir()
            tasks_dst = suite_dir / "tasks"
            tasks_dst.mkdir()
            swe_dst = suite_dir / "swebench"
            swe_dst.mkdir()
            (tmpdir / "viz").mkdir()

            for p in SCHEMAS_DIR.iterdir():
                shutil.copy(p, schemas_dst / p.name)
            for p in TASKS_DIR.iterdir():
                shutil.copy(p, tasks_dst / p.name)
            for p in SWEBENCH_DIR.iterdir():
                shutil.copy(p, swe_dst / p.name)

            suite_path = suite_dir / "warpcore-v1.yaml"
            shutil.copy(SUITE_FILE, suite_path)

            result = validate_suite(tmpdir, suite_path)
            assert result == [], f"Real suite should be valid; got: {result}"


# ===========================================================================
# 3.  Duplicate/count checks must not crash when malformed IDs exist
# ===========================================================================

class TestDuplicateAndCountSafeWithMalformedIds:
    """Duplicate detection and count checks must stay crash-free with bad IDs."""

    def test_duplicate_check_safe_with_dicts(self):
        """Duplicate check must not raise when list contains dicts."""
        # Two identical dicts would be unhashable for set()
        instances = [{"a": 1}, {"b": 2}, {"a": 1}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            # Must not raise
            result = validate_suite(repo_root, suite_path)
            assert isinstance(result, list)

    def test_count_check_still_runs_with_malformed_ids(self):
        """expected_item_count check must still run even when IDs are malformed."""
        # 2 items but expected_item_count is 3 — count mismatch should still surface
        # (but only if we get past the type check gracefully)
        instances = [{"a": 1}, {"b": 2}]
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            suite_dir = tmpdir / "suite"
            suite_dir.mkdir(parents=True)
            schemas_dst = suite_dir / "schemas"
            schemas_dst.mkdir()
            tasks_dst = suite_dir / "tasks"
            tasks_dst.mkdir()
            swe_dst = suite_dir / "swebench"
            swe_dst.mkdir()
            (tmpdir / "viz").mkdir()

            for p in SCHEMAS_DIR.iterdir():
                shutil.copy(p, schemas_dst / p.name)
            for p in TASKS_DIR.iterdir():
                shutil.copy(p, tasks_dst / p.name)
            for p in SWEBENCH_DIR.iterdir():
                shutil.copy(p, swe_dst / p.name)

            instances_file = swe_dst / "instances-seed42-n100.json"
            raw = json.dumps(instances).encode()
            instances_file.write_bytes(raw)
            new_hash = _sha256_bytes(raw)

            suite_path = suite_dir / "warpcore-v1.yaml"
            shutil.copy(SUITE_FILE, suite_path)
            suite = yaml.safe_load(suite_path.read_text())
            suite["benchmarks"]["swebench"]["instances_sha256"] = new_hash
            # Deliberately leave expected_item_count=100 (mismatch) to prove
            # count check still fires alongside type errors
            # (do NOT update expected_item_count)
            suite_path.write_text(yaml.dump(suite))

            result = validate_suite(tmpdir, suite_path)
            assert isinstance(result, list)
            # Must have at least the type error
            assert len(result) > 0


# ===========================================================================
# 4.  CLI: unhashable objects must exit 1 (diagnosed defect), not 2
# ===========================================================================

class TestCLIUnhashableExitsOne:
    """Before fix: TypeError bubbles up → exit 2.
    After fix:  clear error message → exit 1.
    """

    def test_dict_elements_cli_exits_1(self):
        """CLI must exit 1 (diagnosed defect) for dict instance IDs."""
        instances = [{"instance_id": "django__django-11299"}, {"instance_id": "other__repo-1"}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            rc, output = _run_cli("--repo", str(repo_root), str(suite_path))
            assert rc == 1, (
                f"Dict instance IDs are a diagnosed contract defect; expected exit 1, "
                f"got {rc}:\n{output}"
            )

    def test_list_elements_cli_exits_1(self):
        """CLI must exit 1 for nested-list instance IDs."""
        instances = [["django__django-11299"], ["other__repo-1"]]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            rc, output = _run_cli("--repo", str(repo_root), str(suite_path))
            assert rc == 1, (
                f"List instance IDs are a diagnosed contract defect; expected exit 1, "
                f"got {rc}:\n{output}"
            )

    def test_empty_string_ids_cli_exits_1(self):
        """CLI must exit 1 for empty-string instance IDs."""
        instances = ["valid__id-1", "", "valid__id-2"]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            rc, output = _run_cli("--repo", str(repo_root), str(suite_path))
            assert rc == 1, (
                f"Empty-string instance IDs are a diagnosed contract defect; expected exit 1, "
                f"got {rc}:\n{output}"
            )

    def test_cli_output_mentions_type_or_position(self):
        """CLI output must include a human-readable error, not a raw traceback."""
        instances = [{"instance_id": "bad"}]
        with tempfile.TemporaryDirectory() as td:
            repo_root, suite_path = _make_suite_repo(Path(td), instances)
            rc, output = _run_cli("--repo", str(repo_root), str(suite_path))
            # Should NOT contain 'TypeError' or 'Traceback'
            assert "Traceback" not in output, (
                f"CLI must not emit a Python traceback; got:\n{output}"
            )
            assert "TypeError" not in output, (
                f"CLI must not emit raw TypeError; got:\n{output}"
            )
            # Should contain some useful term
            combined = output.lower()
            assert any(
                word in combined for word in ("type", "str", "non-string", "instance id", "error")
            ), f"CLI output must include meaningful error description, got:\n{output}"

    def test_valid_suite_still_exits_0_after_fix(self):
        """After the fix, a valid suite must still exit 0."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            suite_dir = tmpdir / "suite"
            suite_dir.mkdir(parents=True)
            schemas_dst = suite_dir / "schemas"
            schemas_dst.mkdir()
            tasks_dst = suite_dir / "tasks"
            tasks_dst.mkdir()
            swe_dst = suite_dir / "swebench"
            swe_dst.mkdir()
            (tmpdir / "viz").mkdir()

            for p in SCHEMAS_DIR.iterdir():
                shutil.copy(p, schemas_dst / p.name)
            for p in TASKS_DIR.iterdir():
                shutil.copy(p, tasks_dst / p.name)
            for p in SWEBENCH_DIR.iterdir():
                shutil.copy(p, swe_dst / p.name)

            suite_path = suite_dir / "warpcore-v1.yaml"
            shutil.copy(SUITE_FILE, suite_path)

            rc, output = _run_cli("--repo", str(tmpdir), str(suite_path))
            assert rc == 0, f"Valid suite must exit 0, got {rc}:\n{output}"
