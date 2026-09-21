"""
tests/test_adapters.py — Task 3: serving adapter validation tests.

Tests cover:
  T1. Schema validity of both adapter files
  T2. Slugs match result directory names
  T3. Noncanonical gating: adapters with unresolved provenance fail
      validate_adapter_campaign_ready(), even if schema-valid
  T4. Rejection of aliases, mutable image-only tags, missing revisions,
      forbidden experiment keys
  T5. Rejection of duplicate model slugs across adapters
  T6. Rejection of unsupported adapter_schema_version values
  T7. Context-length cross-validation is honest about what it can/cannot prove
      (does NOT accept max_model_len >= output_ceiling as proof of context fit)
  T8. validate_adapters_dir() finds all adapters, reports noncanonical status

Run:
    /usr/bin/python3 -m pytest tests/test_adapters.py -v

All tests in this file are RED before implementation, GREEN after.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

# Allow import from viz/ when run as a script
_TESTS_DIR = Path(__file__).parent
_REPO = _TESTS_DIR.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
_VIZ_DIR = _REPO / "viz"
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

from schema_helpers import SCHEMA_ADAPTER, validate_doc
from contract import validate_adapter, validate_adapters_dir, validate_adapter_campaign_ready

ADAPTERS_DIR = _REPO / "adapters"
RESULTS_DIR = _REPO / "results"
SUITE_FILE = _REPO / "suite" / "warpcore-v1.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_suite() -> dict:
    return yaml.safe_load(SUITE_FILE.read_text())


def _make_canonical_adapter(**overrides) -> dict:
    """Return a minimal schema-valid canonical adapter for mutation tests."""
    base = {
        "adapter_schema_version": 1,
        "campaign_status": "canonical",
        "model": {
            "slug": "test-model-7b",
            "id": "org/test-model-7b",
            "revision": "a" * 40,
        },
        "serving": {
            "image": "eugr/spark-vllm@sha256:" + "d" * 64,
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
    base.update(overrides)
    return base


def _make_noncanonical_adapter(**overrides) -> dict:
    """Return a schema-valid noncanonical adapter for mutation tests."""
    base = {
        "adapter_schema_version": 1,
        "campaign_status": "noncanonical",
        "noncanonical_reason": "model_revision_unrecorded",
        "model": {
            "slug": "test-model-7b",
            "id": "org/test-model-7b",
            "revision": "unresolved",
        },
        "serving": {
            "image": "unresolved",
            "engine": "vllm",
            "engine_version": "0.23.1rc1.dev961+gbc6fbf472",
            "quantization": "mxfp4",
            "reasoning_parser": None,
            "tool_call_parser": None,
            "tokenizer": None,
            "moe_backend": "marlin",
            "max_model_len": 131072,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 32,
            "environment": {},
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# T1: Schema validity of both adapter files
# ---------------------------------------------------------------------------

class TestT1_AdapterFileSchemaValidity:
    """Both adapter files must be schema-valid (whether canonical or noncanonical)."""

    def test_gpt_oss_adapter_file_exists(self):
        assert (ADAPTERS_DIR / "gpt-oss-120b.yaml").exists(), (
            "adapters/gpt-oss-120b.yaml does not exist; create it."
        )

    def test_qwen36_adapter_file_exists(self):
        assert (ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml").exists(), (
            "adapters/qwen3.6-35b-a3b.yaml does not exist; create it."
        )

    def test_gpt_oss_adapter_is_schema_valid(self):
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        # Should not raise
        validate_doc(adapter, SCHEMA_ADAPTER)

    def test_qwen36_adapter_is_schema_valid(self):
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        validate_doc(adapter, SCHEMA_ADAPTER)

    def test_validate_adapter_function_accepts_gpt_oss(self):
        """validate_adapter() must return [] for the gpt-oss adapter."""
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        errors = validate_adapter(_REPO, adapter_path)
        assert errors == [], f"Schema validation errors for gpt-oss-120b.yaml: {errors}"

    def test_validate_adapter_function_accepts_qwen36(self):
        """validate_adapter() must return [] for the qwen3.6 adapter."""
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        errors = validate_adapter(_REPO, adapter_path)
        assert errors == [], f"Schema validation errors for qwen3.6-35b-a3b.yaml: {errors}"


# ---------------------------------------------------------------------------
# T2: Slugs match result directory names
# ---------------------------------------------------------------------------

class TestT2_SlugMatchesResultDirectory:
    """Each adapter's model.slug must exactly match an existing results/<slug>/ directory."""

    def test_gpt_oss_slug_matches_results_dir(self):
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        slug = adapter["model"]["slug"]
        assert slug == "gpt-oss-120b", (
            f"gpt-oss adapter slug must be 'gpt-oss-120b' (matching results/gpt-oss-120b/), got {slug!r}"
        )
        assert (RESULTS_DIR / slug).is_dir(), (
            f"results/{slug}/ directory does not exist (slug must match result directory name)"
        )

    def test_qwen36_slug_matches_results_dir(self):
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        slug = adapter["model"]["slug"]
        assert slug == "qwen3.6-35b-a3b", (
            f"qwen3.6 adapter slug must be 'qwen3.6-35b-a3b', got {slug!r}"
        )
        assert (RESULTS_DIR / slug).is_dir(), (
            f"results/{slug}/ directory does not exist (slug must match result directory name)"
        )


# ---------------------------------------------------------------------------
# T3: Noncanonical gating
# ---------------------------------------------------------------------------

class TestT3_NoncanonicalGating:
    """
    Adapters with unresolved provenance must carry campaign_status=noncanonical
    and fail validate_adapter_campaign_ready().
    A noncanonical adapter remains schema-valid; it just cannot launch a canonical campaign.
    """

    def test_gpt_oss_is_canonical_for_new_closure_campaigns(self):
        """The verified live gpt-oss profile is canonical for prospective campaigns."""
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        assert adapter.get("campaign_status") == "canonical"
        assert adapter["model"]["revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
        assert adapter["serving"]["image"] == (
            "eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8"
        )
        assert adapter["serving"]["engine_version"] == "0.29.1rc1.dev427+g0748d3bd5.d20260920"
        assert adapter["serving"]["tool_call_parser"] == "openai"
        assert adapter["serving"]["moe_backend"] == "marlin"

    def test_gpt_oss_passes_campaign_ready_with_measured_prompt_evidence(self):
        """The verified live gpt-oss adapter passes readiness with frozen-suite maxima."""
        adapter = _load_yaml(ADAPTERS_DIR / "gpt-oss-120b.yaml")
        errors = validate_adapter_campaign_ready(
            adapter,
            "gpt-oss-120b",
            suite=_load_suite(),
            prompt_token_maxima={"gsm8k": 256, "ifeval": 373, "gpqa_diamond": 2808},
        )
        assert errors == []

    def test_qwen36_is_canonical_for_new_closure_campaigns(self):
        """Qwen's live profile is canonical without rewriting historical provenance."""
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        assert adapter.get("campaign_status") == "canonical"
        assert adapter["model"]["revision"] != "unresolved"
        assert adapter["serving"]["image"] != "unresolved"

    def test_gpt_oss_passes_campaign_ready(self):
        """validate_adapter_campaign_ready() accepts the verified live gpt-oss profile."""
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        errors = validate_adapter_campaign_ready(adapter, "gpt-oss-120b")
        assert errors == []

    def test_qwen36_passes_campaign_ready_with_measured_prompt_evidence(self):
        """The live Qwen adapter passes readiness with measured frozen-suite maxima."""
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        errors = validate_adapter_campaign_ready(
            adapter,
            "qwen3.6-35b-a3b",
            suite=_load_suite(),
            prompt_token_maxima={"gsm8k": 256, "ifeval": 373, "gpqa_diamond": 2808},
        )
        assert errors == []

    def test_noncanonical_reason_present_gpt_oss(self):
        """noncanonical adapters must carry a noncanonical_reason field."""
        adapter_path = ADAPTERS_DIR / "gpt-oss-120b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        if adapter.get("campaign_status") != "noncanonical":
            pytest.skip("adapter is not noncanonical")
        assert adapter.get("noncanonical_reason"), (
            "noncanonical adapters must carry a noncanonical_reason field explaining what is unresolved"
        )

    def test_noncanonical_reason_present_qwen36(self):
        """noncanonical adapters must carry a noncanonical_reason field."""
        adapter_path = ADAPTERS_DIR / "qwen3.6-35b-a3b.yaml"
        if not adapter_path.exists():
            pytest.skip("adapter file not yet created")
        adapter = _load_yaml(adapter_path)
        if adapter.get("campaign_status") != "noncanonical":
            pytest.skip("adapter is not noncanonical")
        assert adapter.get("noncanonical_reason"), (
            "noncanonical adapters must carry a noncanonical_reason field explaining what is unresolved"
        )

    def test_canonical_adapter_passes_campaign_ready(self):
        """A fully-resolved canonical adapter must pass validate_adapter_campaign_ready()."""
        adapter = _make_canonical_adapter()
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b")
        assert errors == [], f"Canonical adapter failed campaign_ready: {errors}"

    def test_noncanonical_adapter_fails_campaign_ready(self):
        """A noncanonical adapter must fail validate_adapter_campaign_ready()."""
        adapter = _make_noncanonical_adapter()
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b")
        assert errors, "Noncanonical adapter should fail campaign_ready but returned []"

    def test_campaign_ready_error_names_missing_field(self):
        """validate_adapter_campaign_ready() errors must name which field is unresolved."""
        adapter = _make_noncanonical_adapter()
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b")
        # At least one error must mention a field name (revision or image)
        combined = " ".join(errors).lower()
        assert "revision" in combined or "image" in combined or "noncanonical" in combined, (
            f"validate_adapter_campaign_ready() errors must name the unresolved field; got: {errors}"
        )


# ---------------------------------------------------------------------------
# T4: Rejection of invalid adapter content
# ---------------------------------------------------------------------------

class TestT4_AdapterSchemaRejections:
    """Schema must reject aliases, mutable tags, missing revisions, forbidden keys."""

    # -- Experiment-level forbidden keys (additionalProperties: false) --------

    def test_forbidden_generation_ceiling_rejected(self):
        bad = {
            "adapter_schema_version": 1,
            "campaign_status": "canonical",
            "model": {"slug": "m", "id": "org/m", "revision": "a" * 40},
            "serving": {
                "image": "repo@sha256:" + "d" * 64,
                "engine": "vllm", "engine_version": "0.8.5",
                "quantization": None, "reasoning_parser": None, "tool_call_parser": None,
                "tokenizer": None, "moe_backend": None,
                "max_model_len": 65536, "gpu_memory_utilization": 0.90, "max_num_seqs": 32,
                "environment": {},
            },
            "generation_ceiling": 65536,  # forbidden experiment key
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_forbidden_task_key_rejected(self):
        bad = {
            "adapter_schema_version": 1,
            "campaign_status": "canonical",
            "model": {"slug": "m", "id": "org/m", "revision": "a" * 40},
            "serving": {
                "image": "repo@sha256:" + "d" * 64,
                "engine": "vllm", "engine_version": "0.8.5",
                "quantization": None, "reasoning_parser": None, "tool_call_parser": None,
                "tokenizer": None, "moe_backend": None,
                "max_model_len": 65536, "gpu_memory_utilization": 0.90, "max_num_seqs": 32,
                "environment": {},
            },
            "task": "gpqa_diamond",  # forbidden experiment key
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_forbidden_sampling_key_rejected(self):
        bad = {
            "adapter_schema_version": 1,
            "campaign_status": "canonical",
            "model": {"slug": "m", "id": "org/m", "revision": "a" * 40},
            "serving": {
                "image": "repo@sha256:" + "d" * 64,
                "engine": "vllm", "engine_version": "0.8.5",
                "quantization": None, "reasoning_parser": None, "tool_call_parser": None,
                "tokenizer": None, "moe_backend": None,
                "max_model_len": 65536, "gpu_memory_utilization": 0.90, "max_num_seqs": 32,
                "environment": {},
            },
            "sampling": {"temperature": 0},  # forbidden experiment key
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    # -- Mutable tags / alias image references for canonical adapters ----------

    def test_canonical_mutable_tag_only_image_rejected(self):
        """Canonical adapter with mutable tag image must be rejected by schema."""
        bad = _make_canonical_adapter()
        bad["serving"] = {**bad["serving"], "image": "repo:latest"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_canonical_image_with_tag_no_digest_rejected(self):
        """Canonical adapter with tag but no digest must be rejected."""
        bad = _make_canonical_adapter()
        bad["serving"] = {**bad["serving"], "image": "eugr/spark-vllm:v0.23.1"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    # -- Missing / malformed revision for canonical adapters ------------------

    def test_canonical_revision_unrecorded_rejected(self):
        """Canonical adapter with revision='unrecorded' must be rejected by schema."""
        bad = _make_canonical_adapter()
        bad["model"] = {**bad["model"], "revision": "unrecorded"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_canonical_revision_empty_rejected(self):
        bad = _make_canonical_adapter()
        bad["model"] = {**bad["model"], "revision": ""}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_canonical_revision_short_hex_rejected(self):
        bad = _make_canonical_adapter()
        bad["model"] = {**bad["model"], "revision": "abc123"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)


# ---------------------------------------------------------------------------
# T5: Duplicate model slugs across adapter files
# ---------------------------------------------------------------------------

class TestT5_DuplicateSlugsAcrossAdapters:
    """validate_adapters_dir() must reject a directory containing duplicate model slugs."""

    def test_duplicate_slug_detection(self, tmp_path):
        """Two adapters with the same slug must be flagged."""
        # Create two adapters with the same slug
        slug = "duplicate-model-7b"
        adapter1 = _make_canonical_adapter()
        adapter1["model"]["slug"] = slug
        adapter2 = _make_canonical_adapter()
        adapter2["model"]["slug"] = slug
        adapter2["model"]["id"] = "org2/different-model"
        # Use different revisions to be clear they're meant to be separate models
        adapter2["model"]["revision"] = "b" * 40

        (tmp_path / "adapter1.yaml").write_text(yaml.dump(adapter1))
        (tmp_path / "adapter2.yaml").write_text(yaml.dump(adapter2))

        errors = validate_adapters_dir(_REPO, tmp_path)
        slugs_mentioned = [e for e in errors if slug in e]
        assert slugs_mentioned, (
            f"validate_adapters_dir() must report duplicate slug {slug!r}; errors: {errors}"
        )

    def test_unique_slugs_accepted(self, tmp_path):
        """Two adapters with distinct slugs must pass duplicate check."""
        adapter1 = _make_canonical_adapter()
        adapter1["model"]["slug"] = "model-a-7b"
        adapter2 = _make_canonical_adapter()
        adapter2["model"]["slug"] = "model-b-13b"
        adapter2["model"]["revision"] = "b" * 40

        (tmp_path / "adapter1.yaml").write_text(yaml.dump(adapter1))
        (tmp_path / "adapter2.yaml").write_text(yaml.dump(adapter2))

        errors = validate_adapters_dir(_REPO, tmp_path)
        dup_errors = [e for e in errors if "duplicate" in e.lower() or "slug" in e.lower()]
        assert not dup_errors, (
            f"Adapters with distinct slugs should not produce slug errors; got: {errors}"
        )


# ---------------------------------------------------------------------------
# T6: Unsupported adapter_schema_version
# ---------------------------------------------------------------------------

class TestT6_AdapterSchemaVersion:
    """Unsupported adapter_schema_version values must be rejected by the schema."""

    def test_schema_version_0_rejected(self):
        bad = _make_canonical_adapter()
        bad["adapter_schema_version"] = 0
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_schema_version_2_rejected(self):
        bad = _make_canonical_adapter()
        bad["adapter_schema_version"] = 2
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_schema_version_string_rejected(self):
        bad = _make_canonical_adapter()
        bad["adapter_schema_version"] = "1"
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_ADAPTER)

    def test_schema_version_1_accepted(self):
        good = _make_canonical_adapter()
        good["adapter_schema_version"] = 1
        validate_doc(good, SCHEMA_ADAPTER)  # must not raise


# ---------------------------------------------------------------------------
# T7: Context-length cross-validation honesty
# ---------------------------------------------------------------------------

class TestT7_ContextLengthValidation:
    """
    max_model_len cross-validation must be honest about what it can prove.

    The suite stores only output ceilings (generation_ceiling), NOT prompt token counts.
    validate_adapter_campaign_ready() must NOT claim context feasibility from
    max_model_len >= generation_ceiling alone — that ignores prompt tokens entirely.
    It should report that prompt-plus-output validation is blocked pending tokenization.
    """

    def test_max_model_len_less_than_output_ceiling_flagged(self):
        """
        When max_model_len < the largest output ceiling in the suite, that's a
        definite problem (output alone can't fit) and must be flagged.
        """
        suite = _load_suite()
        max_ceiling = max(
            b.get("generation_ceiling", 0)
            for b in suite["benchmarks"].values()
            if isinstance(b, dict)
        )
        # Set max_model_len to less than the largest ceiling
        adapter = _make_canonical_adapter()
        adapter["serving"]["max_model_len"] = max_ceiling - 1
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b", suite=suite)
        assert errors, (
            f"max_model_len={max_ceiling-1} is less than the largest output ceiling "
            f"{max_ceiling}; validate_adapter_campaign_ready() must flag this."
        )

    def test_max_model_len_eq_output_ceiling_is_not_sufficient_proof(self):
        """Equal output ceiling still blocks without tokenized prompt evidence."""
        suite = _load_suite()
        max_ceiling = max(
            b.get("generation_ceiling", 0)
            for b in suite["benchmarks"].values()
            if isinstance(b, dict)
        )
        adapter = _make_canonical_adapter()
        adapter["serving"]["max_model_len"] = max_ceiling
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b", suite=suite)
        assert any("tokenized prompt evidence" in e.lower() for e in errors), errors

    def test_missing_prompt_token_evidence_blocks_readiness_even_with_large_window(self):
        """A large window is not proof without measured tokenized prompt maxima."""
        suite = _load_suite()
        max_ceiling = max(
            b.get("generation_ceiling", 0)
            for b in suite["benchmarks"].values()
            if isinstance(b, dict)
        )
        adapter = _make_canonical_adapter()
        adapter["serving"]["max_model_len"] = max_ceiling * 10
        errors = validate_adapter_campaign_ready(adapter, "test-model-7b", suite=suite)
        assert any("tokenized prompt evidence" in e.lower() for e in errors), errors


# ---------------------------------------------------------------------------
# T8: validate_adapters_dir discovers all adapters and reports status
# ---------------------------------------------------------------------------

class TestT8_ValidateAdaptersDir:
    """validate_adapters_dir() correctly discovers adapters and summarizes status."""

    def test_adapters_dir_exists(self):
        assert ADAPTERS_DIR.is_dir(), (
            "adapters/ directory must exist (create it with at least the two adapter files)"
        )

    def test_validate_adapters_dir_on_real_adapters_dir(self):
        """validate_adapters_dir() must run without exception on the real adapters/ dir."""
        if not ADAPTERS_DIR.is_dir():
            pytest.skip("adapters/ not yet created")
        # Should not raise; returns list[str] of errors
        errors = validate_adapters_dir(_REPO, ADAPTERS_DIR)
        assert isinstance(errors, list)

    def test_validate_adapters_dir_returns_list(self, tmp_path):
        """validate_adapters_dir() must return list[str] even for empty directory."""
        errors = validate_adapters_dir(_REPO, tmp_path)
        assert isinstance(errors, list)

    def test_validate_adapters_dir_detects_schema_invalid(self, tmp_path):
        """validate_adapters_dir() must flag schema-invalid adapter files."""
        bad_adapter = {"adapter_schema_version": 99, "model": {}, "serving": {}}
        (tmp_path / "bad.yaml").write_text(yaml.dump(bad_adapter))
        errors = validate_adapters_dir(_REPO, tmp_path)
        assert errors, f"Schema-invalid adapter should produce errors; got: {errors}"

    def test_real_adapters_dir_has_both_adapters(self):
        """After Task 3, adapters/ must contain both expected YAML files."""
        if not ADAPTERS_DIR.is_dir():
            pytest.skip("adapters/ not yet created")
        adapter_files = sorted(p.name for p in ADAPTERS_DIR.glob("*.yaml"))
        assert "gpt-oss-120b.yaml" in adapter_files, (
            f"adapters/gpt-oss-120b.yaml missing; found: {adapter_files}"
        )
        assert "qwen3.6-35b-a3b.yaml" in adapter_files, (
            f"adapters/qwen3.6-35b-a3b.yaml missing; found: {adapter_files}"
        )

    def test_noncanonical_adapters_do_not_cause_schema_errors(self):
        """Noncanonical adapters must be schema-valid even though they block campaign launch."""
        if not ADAPTERS_DIR.is_dir():
            pytest.skip("adapters/ not yet created")
        errors = validate_adapters_dir(_REPO, ADAPTERS_DIR)
        # Schema errors would be in errors. Campaign-readiness issues are separate.
        # Both canonical and noncanonical checked-in adapters must remain schema-valid.
        assert errors == [], (
            f"Real adapter files should be schema-valid; got errors: {errors}"
        )
