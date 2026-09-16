"""
tests/test_task4_hardening.py — Adversarial-review regression tests for Task 4.

Requirements covered (all RED before implementation, GREEN after):

  H1.  create_campaign resume=True path: reuse dir only when identity+inventory match;
       resume=False always collides (raises OutputDirectoryCollisionError).

  H2.  Full suite validation via contract.validate_suite before creation:
       unknown benchmark, missing/escaped input, hash mismatch all fail; suite YAML
       hash is included in suite_input_hashes.

  H3.  Derive expected inventory from suite expected_item_count; do not default
       silently to 1. Benchmark types without expected_item_count require explicit
       positive item_count. Conflicting caller item_count rejected.

  H4.  Build and validate manifest + status against schemas via contract.validate_json
       before filesystem commit; invalid documents raise before any file is written.

  H5.  Transactional directory publication: stage sibling dir, write files, atomically
       rename. First-write and second-write failures each clean staging and leave no
       final run dir.

  H6.  serving_profile_digest and manifest effective_args include all effective adapter
       serving settings (parsers/tokenizer/moe/backend/resource controls); digest
       changes when each field changes.

  H7.  campaign_state rejects nonmonotonic timestamps and history whose tail does not
       match execution_state. Invalid lifecycle override on non-failure transition rejected.

  H8.  No dead imports in production modules (smoke import test).

Run:
    /usr/bin/python3 -m pytest tests/test_task4_hardening.py -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile

import pytest
import yaml

# Allow imports from viz/ and tests/
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_state
import create_campaign
import contract

from schema_helpers import (
    SCHEMA_MANIFEST,
    SCHEMA_STATUS,
    VALID_MANIFEST,
    VALID_STATUS,
    validate_doc,
)

# Real suite and schema paths
_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_SCHEMAS_DIR = _REPO / "suite" / "schemas"


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_canonical_adapter(
    *,
    slug: str = "test-model-canonical",
    model_id: str = "TestOrg/TestModel-FP8",
    revision: str = "a" * 40,
    image: str = "test/image@sha256:" + "dead" * 16,
    reasoning_parser: object = None,
    tool_call_parser: object = None,
    tokenizer: object = None,
    moe_backend: object = None,
    quantization: object = "fp8",
    max_model_len: int = 131072,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 32,
) -> dict:
    return {
        "adapter_schema_version": 1,
        "campaign_status": "canonical",
        "model": {
            "slug": slug,
            "id": model_id,
            "revision": revision,
        },
        "serving": {
            "image": image,
            "engine": "vllm",
            "engine_version": "0.8.5",
            "quantization": quantization,
            "reasoning_parser": reasoning_parser,
            "tool_call_parser": tool_call_parser,
            "tokenizer": tokenizer,
            "moe_backend": moe_backend,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "environment": {},
        },
    }


def _setup_real_suite_repo(
    tmp_path: pathlib.Path,
    adapter_data: dict | None = None,
    slug: str = "test-model-canonical",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Build a tmp repo using the real suite YAML and all required task/schema files.
    This ensures validate_suite passes.
    Returns (repo, suite_path, adapter_path).
    """
    # Copy schemas
    schemas_dst = tmp_path / "suite" / "schemas"
    schemas_dst.mkdir(parents=True)
    for sf in _SCHEMAS_DIR.glob("*.json"):
        shutil.copy2(sf, schemas_dst / sf.name)

    # Copy the entire suite/ directory tree (tasks, swebench, etc.)
    suite_src = _REPO / "suite"
    suite_dst = tmp_path / "suite"
    for item in suite_src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(suite_src)
            dst = suite_dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    suite_path = suite_dst / "warpcore-v1.yaml"

    # Adapter
    if adapter_data is None:
        adapter_data = _make_canonical_adapter(slug=slug)
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    adapter_file = adapters_dir / f"{slug}.yaml"
    adapter_file.write_text(yaml.dump(adapter_data))

    # Results dir for the model slug
    (tmp_path / "results" / slug).mkdir(parents=True, exist_ok=True)

    return tmp_path, suite_path, adapter_file


def _create(
    tmp_path: pathlib.Path,
    suite_yaml: pathlib.Path,
    adapter_file: pathlib.Path,
    *,
    benchmark: str = "gsm8k",
    run_id: str = "run-2026-09-15T00-00-00",
    resume: bool = False,
    item_count: int | None = None,
) -> pathlib.Path:
    kwargs: dict = dict(
        repo=tmp_path,
        suite_path=suite_yaml,
        adapter_path=adapter_file,
        benchmark=benchmark,
        run_id=run_id,
        resume=resume,
        prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
    )
    if item_count is not None:
        kwargs["item_count"] = item_count
    return create_campaign.create_campaign(**kwargs)


# ===========================================================================
# H1. resume=True path
# ===========================================================================

class TestResumePath:
    """H1: resume=True reuses existing dir only when identity+inventory match;
    resume=False always collides (raises OutputDirectoryCollisionError)."""

    def test_resume_false_always_collides(self, tmp_path):
        """resume=False (default) raises on second call."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        _create(tmp_path, suite, adapter)
        with pytest.raises(create_campaign.OutputDirectoryCollisionError):
            _create(tmp_path, suite, adapter, resume=False)

    def test_resume_true_succeeds_when_identity_matches(self, tmp_path):
        """resume=True returns existing dir when all identity fields match."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir1 = _create(tmp_path, suite, adapter)
        run_dir2 = _create(tmp_path, suite, adapter, resume=True)
        assert run_dir1 == run_dir2

    def test_resume_true_fails_when_adapter_hash_differs(self, tmp_path):
        """resume=True must fail when the adapter file has changed (hash mismatch)."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        _create(tmp_path, suite, adapter)
        # Mutate adapter (change a harmless comment)
        adapter.write_text(adapter.read_text() + "# changed\n")
        with pytest.raises((create_campaign.OutputDirectoryCollisionError,
                             create_campaign.ResumeIdentityMismatchError, ValueError)):
            _create(tmp_path, suite, adapter, resume=True)

    def test_resume_true_fails_when_run_id_dir_contains_no_manifest(self, tmp_path):
        """resume=True on an existing dir with no manifest.json must fail closed."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        # Manually create the run dir without a manifest
        slug = "test-model-canonical"
        run_dir = (
            tmp_path / "results" / slug / "runs"
            / "warpcore-v1" / "gsm8k" / "run-2026-09-15T00-00-00"
        )
        run_dir.mkdir(parents=True)
        with pytest.raises((create_campaign.OutputDirectoryCollisionError,
                             create_campaign.ResumeIdentityMismatchError,
                             ValueError, FileNotFoundError)):
            _create(tmp_path, suite, adapter, resume=True)

    def test_resume_true_fails_when_expected_item_count_differs(self, tmp_path):
        """resume=True must fail if the item count in the existing manifest is wrong.

        We create a run, then manually tamper with manifest's item_inventory.expected,
        and verify that a subsequent resume=True call detects the mismatch.
        """
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        # Tamper the manifest
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["item_inventory"]["expected"] = 9999
        manifest_path.write_text(json.dumps(manifest, indent=2))
        # Now resume should detect the mismatch
        with pytest.raises((create_campaign.OutputDirectoryCollisionError,
                             create_campaign.ResumeIdentityMismatchError, ValueError)):
            _create(tmp_path, suite, adapter, resume=True)


# ===========================================================================
# H2. Full suite validation before creation
# ===========================================================================

class TestSuiteValidationBeforeCreation:
    """H2: contract.validate_suite is called before creating the run dir."""

    def test_unknown_benchmark_rejected(self, tmp_path):
        """Requesting a benchmark not in the suite must raise."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        with pytest.raises((create_campaign.SuiteValidationError, ValueError, KeyError, Exception)):
            _create(tmp_path, suite, adapter, benchmark="no_such_benchmark_xyz")

    def test_suite_yaml_hash_included_in_suite_input_hashes(self, tmp_path):
        """manifest.suite_input_hashes must include the suite YAML or at least one task file."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        hashes = manifest["suite_input_hashes"]
        # Must be non-empty and contain at least one suite/ key
        assert len(hashes) >= 1, "suite_input_hashes must be non-empty"
        assert all(k.startswith("suite/") for k in hashes), (
            f"All suite_input_hashes keys must start with 'suite/'; got: {list(hashes)}"
        )

    def test_suite_yaml_key_in_suite_input_hashes(self, tmp_path):
        """manifest.suite_input_hashes must include the suite YAML itself."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        hashes = manifest["suite_input_hashes"]
        # The suite YAML must be included (either as 'suite/warpcore-v1.yaml' or similar)
        suite_keys = [k for k in hashes if k.endswith(".yaml") and "tasks" not in k]
        assert len(suite_keys) >= 1, (
            f"suite_input_hashes must include the suite YAML file; got keys: {list(hashes)}"
        )

    def test_suite_yaml_hash_is_correct(self, tmp_path):
        """The suite YAML hash in suite_input_hashes must equal the actual file hash."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        hashes = manifest["suite_input_hashes"]
        # Find the suite yaml key
        suite_key = next(
            (k for k in hashes if k.endswith(".yaml") and "tasks" not in k), None
        )
        assert suite_key is not None
        actual_hash = _sha256_hex(suite.read_bytes())
        assert hashes[suite_key] == actual_hash, (
            f"Suite YAML hash mismatch: manifest says {hashes[suite_key]!r}, "
            f"actual is {actual_hash!r}"
        )

    def test_hash_mismatch_rejects_before_creating_dir(self, tmp_path):
        """Suite with corrupt task hash is rejected; no run dir is created."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        # Corrupt the task hash in the suite file
        suite_data = yaml.safe_load(suite.read_text())
        suite_data["benchmarks"]["gsm8k"]["task_sha256"] = "0" * 64
        suite.write_text(yaml.dump(suite_data))
        slug = "test-model-canonical"
        run_dir_path = (
            tmp_path / "results" / slug / "runs"
            / "warpcore-v1" / "gsm8k" / "run-2026-09-15T00-00-00"
        )
        with pytest.raises(Exception):
            _create(tmp_path, suite, adapter)
        assert not run_dir_path.exists(), "Run dir must not be created when suite hash fails"


# ===========================================================================
# H3. Inventory from suite expected_item_count
# ===========================================================================

class TestInventoryFromSuite:
    """H3: expected inventory derived from suite, not silently defaulted to 1."""

    def test_inventory_from_suite_expected_item_count(self, tmp_path):
        """manifest.item_inventory.expected equals suite expected_item_count."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        # Real suite gsm8k expected_item_count = 1319
        assert manifest["item_inventory"]["expected"] == 1319, (
            f"Expected inventory=1319 from suite; got {manifest['item_inventory']['expected']}"
        )

    def test_no_silent_default_of_1_when_suite_has_count(self, tmp_path):
        """item_inventory.expected must NOT be 1 when suite has expected_item_count=1319."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["item_inventory"]["expected"] != 1, (
            "item_inventory.expected must not silently default to 1"
        )

    def test_conflicting_caller_item_count_rejected(self, tmp_path):
        """Caller item_count that conflicts with suite expected_item_count must be rejected."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        # suite says 1319; caller says 999 — must be rejected
        with pytest.raises((create_campaign.InventoryConflictError, ValueError, AssertionError)):
            _create(tmp_path, suite, adapter, item_count=999)

    def test_caller_item_count_matching_suite_accepted(self, tmp_path):
        """Caller item_count matching suite expected_item_count is accepted."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        # suite says 1319; caller explicitly says 1319 — accepted
        run_dir = _create(tmp_path, suite, adapter, item_count=1319)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["item_inventory"]["expected"] == 1319

    def test_missing_expected_item_count_requires_explicit_caller_count(self, tmp_path):
        """When benchmark has no expected_item_count, caller must provide positive item_count.

        We use the 'throughput' benchmark which lacks expected_item_count in the real suite.
        """
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        suite_data = yaml.safe_load(suite.read_text())
        throughput_bench = suite_data.get("benchmarks", {}).get("throughput", {})
        if "expected_item_count" in throughput_bench:
            pytest.skip("throughput benchmark has expected_item_count in this suite version")
        # Without item_count, must raise
        with pytest.raises((ValueError, TypeError)):
            _create(tmp_path, suite, adapter, benchmark="throughput")

    def test_explicit_item_count_accepted_when_no_suite_count(self, tmp_path):
        """Explicit positive item_count is accepted when suite benchmark lacks expected_item_count."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        suite_data = yaml.safe_load(suite.read_text())
        throughput_bench = suite_data.get("benchmarks", {}).get("throughput", {})
        if "expected_item_count" in throughput_bench:
            pytest.skip("throughput benchmark has expected_item_count in this suite version")
        run_dir = _create(tmp_path, suite, adapter, benchmark="throughput", item_count=500)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["item_inventory"]["expected"] == 500


# ===========================================================================
# H4. Schema validation before filesystem commit
# ===========================================================================

class TestSchemaValidationBeforeCommit:
    """H4: manifest and status are validated via contract.validate_json before write."""

    def test_create_campaign_manifest_passes_schema(self, tmp_path):
        """The manifest created by create_campaign is schema-valid."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        errors = contract.validate_json(manifest, SCHEMA_MANIFEST)
        assert errors == [], f"Manifest schema errors: {errors}"

    def test_create_campaign_status_passes_schema(self, tmp_path):
        """The status created by create_campaign is schema-valid."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        status = json.loads((run_dir / "status.json").read_text())
        errors = contract.validate_json(status, SCHEMA_STATUS)
        assert errors == [], f"Status schema errors: {errors}"

    def test_manifest_item_inventory_minimum_is_one(self, tmp_path):
        """item_inventory.expected >= 1 per schema; must never be 0 or negative."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["item_inventory"]["expected"] >= 1


# ===========================================================================
# H5. Transactional directory publication
# ===========================================================================

class TestTransactionalPublication:
    """H5: creation is staged in a sibling dir, then atomically renamed.
    Failures during first or second write clean staging and leave no final dir."""

    def test_first_write_failure_leaves_no_run_dir(self, tmp_path, monkeypatch):
        """If the first write (manifest) fails, the final run dir must not exist."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        slug = "test-model-canonical"

        write_count = [0]
        original_write = campaign_state.write_status

        def failing_first_write(dest, status, run_dir=None):
            write_count[0] += 1
            if write_count[0] == 1:
                raise OSError("injected first-write failure")
            return original_write(dest, status, run_dir=run_dir)

        monkeypatch.setattr(campaign_state, "write_status", failing_first_write)

        import importlib
        import create_campaign as _cc
        importlib.reload(_cc)

        run_dir_path = (
            tmp_path / "results" / slug / "runs"
            / "warpcore-v1" / "gsm8k" / "run-2026-09-15T00-00-00"
        )
        try:
            with pytest.raises(OSError):
                _cc.create_campaign(
                    repo=tmp_path,
                    suite_path=suite,
                    adapter_path=adapter,
                    benchmark="gsm8k",
                    run_id="run-2026-09-15T00-00-00",
                    resume=False,
                    prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
                )
        finally:
            importlib.reload(_cc)

        assert not run_dir_path.exists(), (
            "Final run directory must not exist after first-write failure"
        )

    def test_second_write_failure_leaves_no_run_dir(self, tmp_path, monkeypatch):
        """If the second write (status) fails, the final run dir must not exist."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        slug = "test-model-canonical"

        write_count = [0]
        original_write = campaign_state.write_status

        def failing_second_write(dest, status, run_dir=None):
            write_count[0] += 1
            if write_count[0] == 2:
                raise OSError("injected second-write failure")
            return original_write(dest, status, run_dir=run_dir)

        monkeypatch.setattr(campaign_state, "write_status", failing_second_write)

        import importlib
        import create_campaign as _cc
        importlib.reload(_cc)

        run_dir_path = (
            tmp_path / "results" / slug / "runs"
            / "warpcore-v1" / "gsm8k" / "run-2026-09-15T00-00-00"
        )
        try:
            with pytest.raises(OSError):
                _cc.create_campaign(
                    repo=tmp_path,
                    suite_path=suite,
                    adapter_path=adapter,
                    benchmark="gsm8k",
                    run_id="run-2026-09-15T00-00-00",
                    resume=False,
                    prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
                )
        finally:
            importlib.reload(_cc)

        assert not run_dir_path.exists(), (
            "Final run directory must not exist after second-write failure"
        )

    def test_staging_sibling_cleaned_on_failure(self, tmp_path, monkeypatch):
        """After a failed creation, no .staging* directory remains under results/."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)

        original_write = campaign_state.write_status

        def always_fail(dest, status, run_dir=None):
            raise OSError("injected failure")

        monkeypatch.setattr(campaign_state, "write_status", always_fail)

        import importlib
        import create_campaign as _cc
        importlib.reload(_cc)

        try:
            with pytest.raises(OSError):
                _cc.create_campaign(
                    repo=tmp_path,
                    suite_path=suite,
                    adapter_path=adapter,
                    benchmark="gsm8k",
                    run_id="run-2026-09-15T00-00-00",
                    resume=False,
                    prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
                )
        finally:
            importlib.reload(_cc)

        results_dir = tmp_path / "results"
        if results_dir.exists():
            leftovers = [
                p for p in results_dir.rglob("*")
                if p.is_dir() and (p.name.startswith(".staging") or p.name.startswith(".tmp"))
            ]
            assert leftovers == [], f"Staging directories left behind: {leftovers}"

    def test_successful_creation_leaves_clean_final_dir(self, tmp_path):
        """Successful creation produces only the final run dir, no staging artifacts."""
        repo, suite, adapter = _setup_real_suite_repo(tmp_path)
        run_dir = _create(tmp_path, suite, adapter)
        assert run_dir.exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "status.json").exists()
        results_dir = tmp_path / "results"
        leftovers = [
            p for p in results_dir.rglob("*")
            if p.is_dir() and (p.name.startswith(".staging") or p.name.startswith(".tmp"))
        ]
        assert leftovers == [], f"Unexpected staging artifacts: {leftovers}"


# ===========================================================================
# H6. serving_profile_digest coverage and effective_args
# ===========================================================================

class TestServingProfileDigest:
    """H6: serving_profile_digest changes when any effective adapter serving field changes;
    effective_args in manifest includes all non-null serving settings."""

    def _digest_for(
        self, adapter_data: dict, tmp_path: pathlib.Path, idx: int
    ) -> str:
        """Build a fresh repo and return the serving_profile_digest from its manifest."""
        subtmp = tmp_path / f"repo_{idx}"
        subtmp.mkdir()
        adapter_slug = adapter_data["model"]["slug"]

        # Copy suite
        suite_src = _REPO / "suite"
        suite_dst = subtmp / "suite"
        for item in suite_src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(suite_src)
                dst = suite_dst / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        suite_yaml = suite_dst / "warpcore-v1.yaml"

        (subtmp / "results" / adapter_slug).mkdir(parents=True, exist_ok=True)
        adapter_file = subtmp / "adapters" / f"{adapter_slug}.yaml"
        adapter_file.parent.mkdir(exist_ok=True)
        adapter_file.write_text(yaml.dump(adapter_data))

        run_dir = create_campaign.create_campaign(
            repo=subtmp,
            suite_path=suite_yaml,
            adapter_path=adapter_file,
            benchmark="gsm8k",
            run_id="run-001",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
            resume=False,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        return manifest["serving_profile_digest"]

    def test_digest_changes_when_reasoning_parser_changes(self, tmp_path):
        a1 = _make_canonical_adapter(reasoning_parser=None)
        a2 = _make_canonical_adapter(reasoning_parser="deepseek_r1")
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when reasoning_parser changes"

    def test_digest_changes_when_tool_call_parser_changes(self, tmp_path):
        a1 = _make_canonical_adapter(tool_call_parser=None)
        a2 = _make_canonical_adapter(tool_call_parser="hermes")
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when tool_call_parser changes"

    def test_digest_changes_when_tokenizer_changes(self, tmp_path):
        a1 = _make_canonical_adapter(tokenizer=None)
        a2 = _make_canonical_adapter(tokenizer="custom-tokenizer-v2")
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when tokenizer changes"

    def test_digest_changes_when_moe_backend_changes(self, tmp_path):
        a1 = _make_canonical_adapter(moe_backend=None)
        a2 = _make_canonical_adapter(moe_backend="marlin")
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when moe_backend changes"

    def test_digest_changes_when_quantization_changes(self, tmp_path):
        a1 = _make_canonical_adapter(quantization="fp8")
        a2 = _make_canonical_adapter(quantization=None)
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when quantization changes"

    def test_digest_changes_when_max_num_seqs_changes(self, tmp_path):
        a1 = _make_canonical_adapter(max_num_seqs=32)
        a2 = _make_canonical_adapter(max_num_seqs=64)
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when max_num_seqs changes"

    def test_digest_changes_when_gpu_memory_utilization_changes(self, tmp_path):
        a1 = _make_canonical_adapter(gpu_memory_utilization=0.90)
        a2 = _make_canonical_adapter(gpu_memory_utilization=0.95)
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a2, tmp_path, 2)
        assert d1 != d2, "serving_profile_digest must change when gpu_memory_utilization changes"

    def test_digest_is_deterministic(self, tmp_path):
        """Same adapter must always produce the same digest."""
        a1 = _make_canonical_adapter()
        d1 = self._digest_for(a1, tmp_path, 1)
        d2 = self._digest_for(a1, tmp_path, 2)
        assert d1 == d2, "serving_profile_digest must be deterministic"

    def test_effective_args_includes_non_null_parser_fields(self, tmp_path):
        """manifest.serving.effective_args must reflect non-null adapter serving fields."""
        adapter_data = _make_canonical_adapter(reasoning_parser="deepseek_r1")
        repo, suite, adapter_file = _setup_real_suite_repo(
            tmp_path, adapter_data=adapter_data
        )
        run_dir = _create(tmp_path, suite, adapter_file)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        effective_args = manifest["serving"]["effective_args"]
        assert isinstance(effective_args, list)
        combined = " ".join(str(a) for a in effective_args)
        assert "deepseek_r1" in combined or "reasoning" in combined, (
            f"effective_args should include reasoning_parser=deepseek_r1; got: {effective_args}"
        )

    def test_effective_args_not_empty_when_settings_present(self, tmp_path):
        """effective_args must not be an empty list when adapter has non-null settings."""
        adapter_data = _make_canonical_adapter(
            quantization="fp8", reasoning_parser="deepseek_r1"
        )
        repo, suite, adapter_file = _setup_real_suite_repo(
            tmp_path, adapter_data=adapter_data
        )
        run_dir = _create(tmp_path, suite, adapter_file)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        effective_args = manifest["serving"]["effective_args"]
        assert len(effective_args) > 0, (
            "effective_args must not be empty when adapter has non-null serving settings"
        )


# ===========================================================================
# H7. campaign_state: nonmonotonic timestamps and history consistency
# ===========================================================================

class TestCampaignStateHardening:
    """H7: campaign_state rejects nonmonotonic timestamps and inconsistent history."""

    def test_nonmonotonic_timestamp_rejected(self):
        """A transition with a timestamp earlier than the last history entry must be rejected."""
        status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T10:00:00Z"},
            ],
        }
        earlier_ts = "2026-09-15T09:00:00Z"
        with pytest.raises((campaign_state.InvalidTimestampError,
                             campaign_state.InvalidTransitionError, ValueError)):
            campaign_state.apply_transition(status, "preflight_passed", earlier_ts)

    def test_monotonic_timestamp_accepted(self):
        """A transition with a timestamp equal to or later than last entry must succeed."""
        status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T10:00:00Z"},
            ],
        }
        # Equal timestamp — must not raise (or raise, but consistently; test just later ts)
        later_ts = "2026-09-15T11:00:00Z"
        result = campaign_state.apply_transition(status, "preflight_passed", later_ts)
        assert result["execution_state"] == "preflight_passed"

    def test_history_tail_must_match_execution_state(self):
        """A status where history[-1].state != execution_state must be rejected on transition."""
        status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "running",   # claims running
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T00:00:00Z"},
                # history tail says "planned" but execution_state says "running"
            ],
        }
        with pytest.raises((campaign_state.InvalidTransitionError, ValueError, AssertionError)):
            campaign_state.apply_transition(status, "completed", "2026-09-15T01:00:00Z")

    def test_invalid_lifecycle_override_on_non_failure_rejected(self):
        """Supplying an invalid lifecycle value for a non-failure transition must be rejected."""
        status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T00:00:00Z"},
            ],
        }
        with pytest.raises((campaign_state.InvalidTransitionError, ValueError)):
            campaign_state.apply_transition(
                status, "preflight_passed", "2026-09-15T01:00:00Z",
                lifecycle="not_a_valid_lifecycle"
            )

    def test_valid_lifecycle_override_accepted(self):
        """A valid lifecycle override value is accepted on a non-failure transition."""
        status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T00:00:00Z"},
            ],
        }
        result = campaign_state.apply_transition(
            status, "preflight_passed", "2026-09-15T01:00:00Z",
            lifecycle="diagnostic"
        )
        assert result["lifecycle"] == "diagnostic"


# ===========================================================================
# H8. No dead imports in production modules
# ===========================================================================

class TestNoDeadImports:
    """H8: production modules import cleanly without unused dead imports."""

    def test_create_campaign_importable(self):
        import importlib
        import create_campaign as _cc
        importlib.reload(_cc)
        assert hasattr(_cc, "create_campaign")

    def test_campaign_state_importable(self):
        import importlib
        import campaign_state as _cs
        importlib.reload(_cs)
        assert hasattr(_cs, "apply_transition")

    def test_contract_importable(self):
        import importlib
        import contract as _c
        importlib.reload(_c)
        assert hasattr(_c, "validate_suite")
        assert hasattr(_c, "validate_json")

    def test_create_campaign_no_dead_copy_import(self):
        """create_campaign.py must not import 'copy' without using it."""
        import ast
        src = (_REPO / "viz" / "create_campaign.py").read_text()
        tree = ast.parse(src)
        has_copy_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "copy":
                        has_copy_import = True
        if has_copy_import:
            assert "copy." in src or "copy(" in src, (
                "Module 'copy' is imported but not referenced in create_campaign.py"
            )
