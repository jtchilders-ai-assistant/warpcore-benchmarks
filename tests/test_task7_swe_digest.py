"""tests/test_task7_swe_digest.py — TDD: SWE-bench frozen instance-set digest.

Strict TDD: tests written BEFORE implementation. Must run RED until
create_campaign.py gains the SWE digest feature.

Feature requirements:
  1. For benchmark=swebench, derive the canonical frozen ID set from
     suite benchmarks.swebench.instance_set_file.
  2. Validate: JSON list of unique non-empty strings, contained under repo
     after symlink resolution, exact count equals expected_item_count.
  3. Compute instance_ids_hash using the EXACT canonical algorithm:
       sha256(json.dumps(sorted(ids), sort_keys=True).encode()).hexdigest()
  4. Record it in manifest.item_inventory.instance_ids_hash at campaign
     creation.
  5. instance_set_file hash must be covered by suite_input_hashes (already
     hashed by _collect_suite_input_hashes via instance_set_file field).
  6. Resume identity comparison must include instance_ids_hash — a manifest
     without a hash OR with a drifted hash must be rejected.
  7. Fail closed BEFORE creating a directory on malformed/missing/escaping
     instance set.
  8. Non-SWE benchmarks must NOT have instance_ids_hash in item_inventory.

Run:
    /usr/bin/python3 -m pytest tests/test_task7_swe_digest.py -v
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import pytest

# Allow imports from viz/ and tests/
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import create_campaign  # viz/create_campaign.py

# ---------------------------------------------------------------------------
# Constants / shared data
# ---------------------------------------------------------------------------

# The real frozen instance file (exists in repo)
_REAL_INSTANCES_PATH = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"

# Expected canonical hash of sorted frozen IDs from the real file
_REAL_IDS = json.loads(_REAL_INSTANCES_PATH.read_text(encoding="utf-8"))
_CANONICAL_HASH = hashlib.sha256(
    json.dumps(sorted(_REAL_IDS), sort_keys=True).encode()
).hexdigest()

# Number of real instances
_REAL_N = len(_REAL_IDS)  # 100


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_adapter_data() -> dict:
    return {
        "adapter_schema_version": 1,
        "campaign_status": "canonical",
        "model": {
            "slug": "test-model",
            "id": "TestOrg/TestModel-7B",
            "revision": "a" * 40,
        },
        "serving": {
            "image": "registry/spark-vllm@sha256:" + "dead" * 16,
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


def _build_repo(
    tmp_path: pathlib.Path,
    *,
    instance_ids: list | None = None,
    instance_ids_raw: object = None,  # override entire JSON content (any type)
    inject_instances_sha256: str | None = None,  # override sha256 in suite YAML
    omit_instance_set_file: bool = False,
    instance_file_escape: bool = False,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Build a minimal repo with real suite + adjusted swebench instance set.

    Returns (tmp_path, suite_yaml, adapter_path).
    """
    import yaml

    # --- Copy real suite/ directory ---
    suite_src = _REPO / "suite"
    suite_dst = tmp_path / "suite"
    for item in suite_src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(suite_src)
            dst = suite_dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    suite_yaml_path = suite_dst / "warpcore-v1.yaml"

    # --- Optionally modify the swebench instance set ---
    if instance_ids is not None or instance_ids_raw is not None:
        if instance_ids_raw is not None:
            content = json.dumps(instance_ids_raw)
        else:
            content = json.dumps(instance_ids)
        actual_path = suite_dst / "swebench" / "instances-seed42-n100.json"
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        actual_path.write_text(content, encoding="utf-8")

        # Re-compute sha256 for the new content and patch the suite YAML
        new_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        suite_data = yaml.safe_load(suite_yaml_path.read_text(encoding="utf-8"))
        swe = suite_data["benchmarks"]["swebench"]
        swe["instances_sha256"] = inject_instances_sha256 or new_sha
        # Also update expected_item_count if we changed the IDs
        if instance_ids is not None:
            swe["expected_item_count"] = len(instance_ids)
        suite_yaml_path.write_text(
            yaml.dump(suite_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    elif inject_instances_sha256 is not None:
        # Only patch sha256, keep existing file
        suite_data = yaml.safe_load(suite_yaml_path.read_text(encoding="utf-8"))
        suite_data["benchmarks"]["swebench"]["instances_sha256"] = inject_instances_sha256
        suite_yaml_path.write_text(
            yaml.dump(suite_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    if omit_instance_set_file:
        # Patch suite YAML to remove instance_set_file key
        import yaml as _yaml
        suite_data = _yaml.safe_load(suite_yaml_path.read_text(encoding="utf-8"))
        suite_data["benchmarks"]["swebench"].pop("instance_set_file", None)
        suite_yaml_path.write_text(
            _yaml.dump(suite_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    if instance_file_escape:
        # Point instance_set_file to a path that escapes the repo
        import yaml as _yaml
        suite_data = _yaml.safe_load(suite_yaml_path.read_text(encoding="utf-8"))
        suite_data["benchmarks"]["swebench"]["instance_set_file"] = "../../../etc/passwd"
        suite_yaml_path.write_text(
            _yaml.dump(suite_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # --- Adapter ---
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapters_dir / "test-model.yaml"
    adapter_file.write_text(
        yaml.dump(_make_adapter_data(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # --- Results dir ---
    (tmp_path / "results" / "test-model").mkdir(parents=True, exist_ok=True)

    return tmp_path, suite_yaml_path, adapter_file


# ---------------------------------------------------------------------------
# FEATURE 1 & 3 & 4: Positive digest — swebench creates instance_ids_hash
# ---------------------------------------------------------------------------

class TestSweDigestCreation:
    """create_campaign must compute and record instance_ids_hash for swebench."""

    @pytest.fixture
    def swe_repo(self, tmp_path):
        return _build_repo(tmp_path)

    # Prompt token maxima required for quality-bench context feasibility check
    _PTM = {"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1}

    def test_manifest_has_instance_ids_hash_for_swebench(self, swe_repo):
        """swebench campaign must write instance_ids_hash in item_inventory."""
        repo, suite_yaml, adapter = swe_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-digest-01",
            prompt_token_maxima=self._PTM,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        inv = manifest.get("item_inventory", {})
        assert "instance_ids_hash" in inv, (
            "swebench manifest.item_inventory must contain instance_ids_hash"
        )

    def test_instance_ids_hash_matches_canonical_algorithm(self, swe_repo):
        """instance_ids_hash must equal sha256(json.dumps(sorted(ids),...))."""
        repo, suite_yaml, adapter = swe_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-digest-02",
            prompt_token_maxima=self._PTM,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        recorded = manifest["item_inventory"]["instance_ids_hash"]
        assert recorded == _CANONICAL_HASH, (
            f"instance_ids_hash mismatch.\n"
            f"  Recorded:  {recorded}\n"
            f"  Expected:  {_CANONICAL_HASH}"
        )

    def test_instance_ids_hash_is_64_char_hex(self, swe_repo):
        """instance_ids_hash must be a 64-char lowercase hex string."""
        repo, suite_yaml, adapter = swe_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-digest-03",
            prompt_token_maxima=self._PTM,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        h = manifest["item_inventory"]["instance_ids_hash"]
        import re
        assert re.fullmatch(r"[0-9a-f]{64}", h), (
            f"instance_ids_hash must be 64-char hex; got: {h!r}"
        )

    def test_instance_ids_hash_covered_by_suite_input_hashes(self, swe_repo):
        """instance_set_file must appear in suite_input_hashes (file integrity)."""
        repo, suite_yaml, adapter = swe_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-digest-04",
            prompt_token_maxima=self._PTM,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        hashes = manifest.get("suite_input_hashes", {})
        # instance_set_file is 'suite/swebench/instances-seed42-n100.json'
        assert "suite/swebench/instances-seed42-n100.json" in hashes, (
            "suite_input_hashes must contain the instance_set_file key"
        )

    def test_instance_ids_hash_schema_valid(self, swe_repo):
        """The manifest with instance_ids_hash must pass JSON schema."""
        from contract import validate_json
        schema_path = _REPO / "suite" / "schemas" / "manifest.schema.json"

        repo, suite_yaml, adapter = swe_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-schema-01",
            prompt_token_maxima=self._PTM,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        errors = validate_json(manifest, schema_path)
        assert not errors, (
            f"swebench manifest must pass schema; errors:\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# FEATURE 8: Non-SWE benchmarks must NOT have instance_ids_hash
# ---------------------------------------------------------------------------

class TestNonSweNoDigest:
    """Non-SWE quality benchmarks must not get instance_ids_hash."""

    @pytest.fixture
    def quality_repo(self, tmp_path):
        return _build_repo(tmp_path)

    @pytest.mark.parametrize("bench", ["gsm8k", "ifeval", "gpqa_diamond"])
    def test_non_swe_manifest_has_no_instance_ids_hash(self, quality_repo, bench):
        """Quality benchmark manifests must NOT contain instance_ids_hash."""
        repo, suite_yaml, adapter = quality_repo
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark=bench,
            run_id="run-no-hash-01",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        inv = manifest.get("item_inventory", {})
        assert "instance_ids_hash" not in inv, (
            f"Quality benchmark {bench!r} must NOT have instance_ids_hash in item_inventory; "
            f"got: {inv}"
        )


# ---------------------------------------------------------------------------
# FEATURE 7: Fail-closed BEFORE directory creation on malformed instance set
# ---------------------------------------------------------------------------

class TestSweDigestFailClosed:
    """Malformed/missing/escaping instance sets must fail before directory creation."""

    def test_duplicate_ids_fail_before_dir_creation(self, tmp_path):
        """Duplicate IDs in instance file must fail before any dir is created."""
        # Use first real ID duplicated to create a duplicate list of length 100
        ids = list(_REAL_IDS)  # 100 unique
        ids[1] = ids[0]  # introduce a duplicate

        repo, suite_yaml, adapter = _build_repo(tmp_path, instance_ids=ids)
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-dup-01"

        with pytest.raises(Exception) as exc_info:
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-dup-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance set is malformed (duplicates)"
        )
        err = str(exc_info.value).lower()
        assert "duplicate" in err or "instance" in err or "unique" in err, (
            f"Error must mention duplicates/instances/unique; got: {exc_info.value}"
        )

    def test_non_string_id_fails_before_dir_creation(self, tmp_path):
        """Non-string entries in instance file must fail before any dir is created."""
        ids = list(_REAL_IDS[:99]) + [42]  # one integer entry

        repo, suite_yaml, adapter = _build_repo(tmp_path, instance_ids_raw=ids)
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-nonstr-01"

        with pytest.raises(Exception):
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-nonstr-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance set contains non-string IDs"
        )

    def test_empty_string_id_fails_before_dir_creation(self, tmp_path):
        """Empty string entries in instance file must fail before dir creation."""
        ids = list(_REAL_IDS[:99]) + [""]  # one empty string

        repo, suite_yaml, adapter = _build_repo(tmp_path, instance_ids_raw=ids)
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-emptyid-01"

        with pytest.raises(Exception):
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-emptyid-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance set contains empty string IDs"
        )

    def test_missing_instance_set_file_fails_before_dir_creation(self, tmp_path):
        """Missing instance_set_file must fail before directory creation."""
        repo, suite_yaml, adapter = _build_repo(tmp_path)
        # Remove the instance set file after building repo
        inst_file = repo / "suite" / "swebench" / "instances-seed42-n100.json"
        inst_file.unlink()
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-missing-01"

        with pytest.raises(Exception):
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-missing-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance_set_file is missing"
        )

    def test_count_mismatch_fails_before_dir_creation(self, tmp_path):
        """Count mismatch (ids != expected_item_count in suite) fails before dir creation."""
        # Use only 50 of the 100 IDs — suite says 100 so count will mismatch
        import yaml
        ids = list(_REAL_IDS[:50])
        new_content = json.dumps(ids)

        repo, suite_yaml, adapter = _build_repo(tmp_path)
        # Write 50 IDs but keep suite expected_item_count=100
        # Compute the sha256 for these 50 IDs so the file hash check passes
        inst_file = repo / "suite" / "swebench" / "instances-seed42-n100.json"
        inst_file.write_text(new_content, encoding="utf-8")
        new_sha = hashlib.sha256(inst_file.read_bytes()).hexdigest()

        # Patch suite YAML to update instances_sha256 BUT keep expected_item_count=100
        suite_data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
        suite_data["benchmarks"]["swebench"]["instances_sha256"] = new_sha
        # Do NOT change expected_item_count — keep at 100, so count mismatch occurs
        suite_yaml.write_text(
            yaml.dump(suite_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-count-mismatch-01"

        with pytest.raises(Exception) as exc_info:
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-count-mismatch-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance count mismatches expected_item_count"
        )
        err = str(exc_info.value).lower()
        assert "count" in err or "expected" in err or "instance" in err, (
            f"Error must mention count/expected/instance; got: {exc_info.value}"
        )

    def test_escape_path_fails_before_dir_creation(self, tmp_path):
        """Path-escaping instance_set_file must fail before directory creation."""
        repo, suite_yaml, adapter = _build_repo(tmp_path, instance_file_escape=True)
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-escape-01"

        with pytest.raises(Exception):
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-escape-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance_set_file escapes the repo"
        )

    def test_json_not_a_list_fails_before_dir_creation(self, tmp_path):
        """instance_set_file that is a JSON object (not list) must fail before dir creation."""
        bad_content = {"id1": True, "id2": True}  # dict, not list
        repo, suite_yaml, adapter = _build_repo(tmp_path, instance_ids_raw=bad_content)
        run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "swebench" / "run-notlist-01"

        with pytest.raises(Exception):
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-notlist-01",
            )
        assert not run_dir.exists(), (
            "run directory must NOT be created when instance_set_file is not a JSON list"
        )


# ---------------------------------------------------------------------------
# FEATURE 6: Resume identity — instance_ids_hash must match exactly
# ---------------------------------------------------------------------------

class TestSweResumeIdentity:
    """Resume must compare instance_ids_hash exactly and reject drift."""

    _PTM = {"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1}

    @pytest.fixture
    def swe_repo_with_run(self, tmp_path):
        """Build a repo and create a valid swebench campaign."""
        repo, suite_yaml, adapter = _build_repo(tmp_path)
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-resume-01",
            prompt_token_maxima=self._PTM,
        )
        return repo, suite_yaml, adapter, run_dir

    def test_resume_same_identity_succeeds(self, swe_repo_with_run):
        """resume=True with identical instance set must succeed."""
        repo, suite_yaml, adapter, run_dir = swe_repo_with_run
        resumed = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="swebench",
            run_id="run-swe-resume-01",
            resume=True,
            prompt_token_maxima=self._PTM,
        )
        assert resumed == run_dir, "resume must return the same run directory"

    def test_resume_missing_instance_ids_hash_rejected(self, swe_repo_with_run):
        """Existing manifest without instance_ids_hash must be rejected on resume."""
        repo, suite_yaml, adapter, run_dir = swe_repo_with_run

        # Remove instance_ids_hash from existing manifest
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["item_inventory"].pop("instance_ids_hash", None)
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(create_campaign.ResumeIdentityMismatchError) as exc_info:
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-swe-resume-01",
                resume=True,
                prompt_token_maxima=self._PTM,
            )
        err = str(exc_info.value).lower()
        assert "instance_ids_hash" in err or "instance" in err, (
            f"ResumeIdentityMismatchError must mention instance_ids_hash; got: {exc_info.value}"
        )

    def test_resume_wrong_instance_ids_hash_rejected(self, swe_repo_with_run):
        """Existing manifest with wrong instance_ids_hash must be rejected on resume."""
        repo, suite_yaml, adapter, run_dir = swe_repo_with_run

        # Write a wrong hash into the existing manifest
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["item_inventory"]["instance_ids_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(create_campaign.ResumeIdentityMismatchError) as exc_info:
            create_campaign.create_campaign(
                repo=repo,
                suite_path=suite_yaml,
                adapter_path=adapter,
                benchmark="swebench",
                run_id="run-swe-resume-01",
                resume=True,
                prompt_token_maxima=self._PTM,
            )
        err = str(exc_info.value).lower()
        assert "instance_ids_hash" in err or "instance" in err, (
            f"ResumeIdentityMismatchError must mention instance_ids_hash; got: {exc_info.value}"
        )

    def test_resume_non_swe_benchmark_ignores_instance_ids_hash(self, tmp_path):
        """Non-SWE resume must not fail due to absent instance_ids_hash."""
        import yaml

        # Build repo, create gsm8k campaign, then try to resume
        repo, suite_yaml, adapter = _build_repo(tmp_path)
        run_dir = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="gsm8k",
            run_id="run-gsm-resume-01",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )

        # Verify no instance_ids_hash was written
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "instance_ids_hash" not in manifest["item_inventory"]

        # Resume must succeed (not raise about missing instance_ids_hash)
        resumed = create_campaign.create_campaign(
            repo=repo,
            suite_path=suite_yaml,
            adapter_path=adapter,
            benchmark="gsm8k",
            run_id="run-gsm-resume-01",
            resume=True,
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        assert resumed == run_dir
