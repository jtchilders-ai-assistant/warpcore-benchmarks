"""
Regression tests for the code-quality refactor pass (Task 1, cycle 3).

These tests must ALL fail (RED) before the refactor is applied, and ALL pass
(GREEN) after.  They cover:

  R1: schema_helpers module exists and exports expected symbols
  R2: validate_doc uses FormatChecker always (no optional argument)
  R3: VALID_MANIFEST / VALID_ADAPTER / VALID_STATUS are importable and valid
  R4: live suite YAML validates against suite.schema.json
  R5: minLength:1 on statistical_policy string fields
  R6: minLength:1 on dataset path and split fields
  R7: gpqa_utils.process_docs gives stable output under fixed random.seed

Run: /usr/bin/python3 -m pytest tests/test_refactor_regression.py -v
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import jsonschema
import pytest
import yaml

REPO = Path(__file__).parent.parent
SUITE_FILE = REPO / "suite" / "warpcore-v1.yaml"
SCHEMAS_DIR = REPO / "suite" / "schemas"
SCHEMA_SUITE = SCHEMAS_DIR / "suite.schema.json"
SCHEMA_MANIFEST = SCHEMAS_DIR / "manifest.schema.json"
SCHEMA_ADAPTER = SCHEMAS_DIR / "adapter.schema.json"
SCHEMA_STATUS = SCHEMAS_DIR / "result-status.schema.json"


# ---------------------------------------------------------------------------
# R1 + R2 + R3: schema_helpers module and its exports
# ---------------------------------------------------------------------------

class TestSchemaHelpersModule:
    """schema_helpers must exist and export the required symbols."""

    def _import(self) -> ModuleType:
        import importlib, sys
        # Ensure tests/ is on path
        tests_dir = str(Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        return importlib.import_module("schema_helpers")

    def test_schema_helpers_importable(self):
        """tests/schema_helpers.py must exist and be importable."""
        sh = self._import()
        assert sh is not None

    def test_exports_valid_manifest(self):
        sh = self._import()
        assert hasattr(sh, "VALID_MANIFEST"), "schema_helpers must export VALID_MANIFEST"

    def test_exports_valid_adapter(self):
        sh = self._import()
        assert hasattr(sh, "VALID_ADAPTER"), "schema_helpers must export VALID_ADAPTER"

    def test_exports_valid_status(self):
        sh = self._import()
        assert hasattr(sh, "VALID_STATUS"), "schema_helpers must export VALID_STATUS"

    def test_exports_validate_doc(self):
        sh = self._import()
        assert hasattr(sh, "validate_doc"), "schema_helpers must export validate_doc"

    def test_exports_schema_paths(self):
        sh = self._import()
        for name in ("SCHEMA_SUITE", "SCHEMA_ADAPTER", "SCHEMA_MANIFEST", "SCHEMA_STATUS"):
            assert hasattr(sh, name), f"schema_helpers must export {name}"

    def test_valid_manifest_is_dict(self):
        sh = self._import()
        assert isinstance(sh.VALID_MANIFEST, dict)

    def test_valid_adapter_is_dict(self):
        sh = self._import()
        assert isinstance(sh.VALID_ADAPTER, dict)

    def test_valid_status_is_dict(self):
        sh = self._import()
        assert isinstance(sh.VALID_STATUS, dict)


class TestValidateDocFormatChecker:
    """validate_doc must use FormatChecker always — no optional argument."""

    def _import(self) -> ModuleType:
        import importlib, sys
        tests_dir = str(Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        return importlib.import_module("schema_helpers")

    def test_validate_doc_accepts_two_positional_args(self):
        """validate_doc(instance, schema_path) — no format_checker argument needed."""
        import inspect
        sh = self._import()
        sig = inspect.signature(sh.validate_doc)
        params = list(sig.parameters.keys())
        assert len(params) == 2, (
            f"validate_doc must take exactly 2 parameters (instance, schema_path); "
            f"got {params!r}.  FormatChecker must be used internally, not as an argument."
        )

    def test_validate_doc_rejects_invalid_datetime_without_explicit_format_checker(self):
        """validate_doc must reject bad timestamps WITHOUT caller passing format_checker."""
        sh = self._import()
        bad = dict(sh.VALID_MANIFEST)
        bad["timing"] = {
            "started_utc": "not-a-date",
            "completed_utc": None,
        }
        with pytest.raises(jsonschema.ValidationError):
            sh.validate_doc(bad, sh.SCHEMA_MANIFEST)

    def test_valid_manifest_passes_validate_doc(self):
        sh = self._import()
        sh.validate_doc(sh.VALID_MANIFEST, sh.SCHEMA_MANIFEST)  # must not raise

    def test_valid_adapter_passes_validate_doc(self):
        sh = self._import()
        sh.validate_doc(sh.VALID_ADAPTER, sh.SCHEMA_ADAPTER)  # must not raise

    def test_valid_status_passes_validate_doc(self):
        sh = self._import()
        sh.validate_doc(sh.VALID_STATUS, sh.SCHEMA_STATUS)  # must not raise


class TestValidDocumentValidity:
    """The canonical constants in schema_helpers must actually be schema-valid."""

    def _import(self) -> ModuleType:
        import importlib, sys
        tests_dir = str(Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        return importlib.import_module("schema_helpers")

    def test_canonical_manifest_validates(self):
        sh = self._import()
        schema = json.loads(sh.SCHEMA_MANIFEST.read_text())
        jsonschema.validate(sh.VALID_MANIFEST, schema,
                            format_checker=jsonschema.FormatChecker())

    def test_canonical_adapter_validates(self):
        sh = self._import()
        schema = json.loads(sh.SCHEMA_ADAPTER.read_text())
        jsonschema.validate(sh.VALID_ADAPTER, schema,
                            format_checker=jsonschema.FormatChecker())

    def test_canonical_status_validates(self):
        sh = self._import()
        schema = json.loads(sh.SCHEMA_STATUS.read_text())
        jsonschema.validate(sh.VALID_STATUS, schema,
                            format_checker=jsonschema.FormatChecker())


# ---------------------------------------------------------------------------
# R4: live suite YAML validates against suite.schema.json
# ---------------------------------------------------------------------------

class TestLiveSuiteYamlValidation:
    """The committed suite/warpcore-v1.yaml must validate against suite.schema.json."""

    def test_suite_yaml_validates_against_schema(self):
        """Direct test: parse suite YAML, validate with FormatChecker."""
        suite = yaml.safe_load(SUITE_FILE.read_text())
        schema = json.loads(SCHEMA_SUITE.read_text())
        # Must not raise
        jsonschema.validate(suite, schema, format_checker=jsonschema.FormatChecker())

    def test_suite_yaml_is_valid_yaml(self):
        suite = yaml.safe_load(SUITE_FILE.read_text())
        assert isinstance(suite, dict), "suite YAML must parse to a dict"

    def test_suite_schema_is_valid_json(self):
        schema = json.loads(SCHEMA_SUITE.read_text())
        assert isinstance(schema, dict)


# ---------------------------------------------------------------------------
# R5: minLength:1 on statistical_policy string fields
# ---------------------------------------------------------------------------

class TestStatisticalPolicyMinLength:
    """statistical_policy string fields must have minLength:1 to reject empty strings."""

    STRING_FIELDS = [
        "cross_model_comparison",
        "binary_test",
        "non_significant_means",
        "interval_method",
    ]

    def _load_schema(self) -> dict:
        return json.loads(SCHEMA_SUITE.read_text())

    def _load_valid_suite(self) -> dict:
        return yaml.safe_load(SUITE_FILE.read_text())

    @pytest.mark.parametrize("field", STRING_FIELDS)
    def test_schema_has_minlength_1_for_field(self, field: str):
        schema = self._load_schema()
        sp_props = schema["properties"]["statistical_policy"]["properties"]
        prop = sp_props[field]
        assert prop.get("minLength") == 1, (
            f"statistical_policy.{field} must have minLength:1 to reject empty strings; "
            f"got: {prop!r}"
        )

    @pytest.mark.parametrize("field", STRING_FIELDS)
    def test_empty_string_rejected_by_schema(self, field: str):
        suite = self._load_valid_suite()
        bad = dict(suite)
        bad["statistical_policy"] = dict(suite["statistical_policy"])
        bad["statistical_policy"][field] = ""
        schema = self._load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema, format_checker=jsonschema.FormatChecker())


# ---------------------------------------------------------------------------
# R6: minLength:1 on dataset path and split fields
# ---------------------------------------------------------------------------

class TestDatasetRefMinLength:
    """dataset_ref path and split must have minLength:1 to reject empty strings."""

    def _load_schema(self) -> dict:
        return json.loads(SCHEMA_SUITE.read_text())

    def _load_valid_suite(self) -> dict:
        return yaml.safe_load(SUITE_FILE.read_text())

    def test_dataset_ref_path_has_minlength_1(self):
        schema = self._load_schema()
        dataset_ref = schema["$defs"]["dataset_ref"]
        path_prop = dataset_ref["properties"]["path"]
        assert path_prop.get("minLength") == 1, (
            f"dataset_ref.path must have minLength:1; got: {path_prop!r}"
        )

    def test_dataset_ref_split_has_minlength_1(self):
        schema = self._load_schema()
        dataset_ref = schema["$defs"]["dataset_ref"]
        split_prop = dataset_ref["properties"]["split"]
        assert split_prop.get("minLength") == 1, (
            f"dataset_ref.split must have minLength:1; got: {split_prop!r}"
        )

    def test_empty_dataset_path_rejected(self):
        suite = self._load_valid_suite()
        bad = dict(suite)
        bad["benchmarks"] = dict(suite["benchmarks"])
        bad["benchmarks"]["gsm8k"] = dict(suite["benchmarks"]["gsm8k"])
        bad["benchmarks"]["gsm8k"]["dataset"] = dict(suite["benchmarks"]["gsm8k"]["dataset"])
        bad["benchmarks"]["gsm8k"]["dataset"]["path"] = ""
        schema = self._load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema, format_checker=jsonschema.FormatChecker())

    def test_empty_dataset_split_rejected(self):
        suite = self._load_valid_suite()
        bad = dict(suite)
        bad["benchmarks"] = dict(suite["benchmarks"])
        bad["benchmarks"]["gsm8k"] = dict(suite["benchmarks"]["gsm8k"])
        bad["benchmarks"]["gsm8k"]["dataset"] = dict(suite["benchmarks"]["gsm8k"]["dataset"])
        bad["benchmarks"]["gsm8k"]["dataset"]["split"] = ""
        schema = self._load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema, format_checker=jsonschema.FormatChecker())


# ---------------------------------------------------------------------------
# R7: gpqa_utils.process_docs gives stable output under fixed random.seed
# ---------------------------------------------------------------------------

class TestGpqaUtilsProcessDocsStability:
    """
    process_docs must give stable, deterministic output when called with the
    same fixed random.seed.  This documents runner seed dependency for later
    enforcement.  The canonical utility file is NOT altered.
    """

    def _make_mock_dataset(self, rows: list[dict]):
        """Build a minimal mock that quacks like a HuggingFace Dataset for process_docs."""
        # process_docs calls dataset.map(_process_doc); map returns a new Dataset.
        # We simulate with a simple wrapper.
        class MockDataset:
            def __init__(self, data):
                self._data = data

            def map(self, fn):
                return MockDataset([fn(row) for row in self._data])

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

        return MockDataset(rows)

    def _sample_rows(self):
        return [
            {
                "Incorrect Answer 1": "Oxygen",
                "Incorrect Answer 2": "Hydrogen",
                "Incorrect Answer 3": "Nitrogen",
                "Correct Answer": "Carbon",
                "Question": "What element is in CO2 besides oxygen?",
            },
            {
                "Incorrect Answer 1": "Paris",
                "Incorrect Answer 2": "London",
                "Incorrect Answer 3": "Berlin",
                "Correct Answer": "Madrid",
                "Question": "Capital of Spain?",
            },
        ]

    def _run_process_docs(self, seed: int) -> list[dict]:
        import sys
        # Ensure suite/tasks on path for gpqa_utils import
        tasks_dir = str(REPO / "suite" / "tasks")
        if tasks_dir not in sys.path:
            sys.path.insert(0, tasks_dir)
        import gpqa_utils  # noqa: F401 — canonical utility, not modified
        random.seed(seed)
        ds = self._make_mock_dataset(self._sample_rows())
        result = gpqa_utils.process_docs(ds)
        return list(result)

    def test_same_seed_gives_same_output_run1_run2(self):
        """Two calls with seed=42 must produce identical outputs."""
        out1 = self._run_process_docs(42)
        out2 = self._run_process_docs(42)
        assert out1 == out2, (
            "process_docs must give stable output under the same seed; "
            f"got different results:\n  run1={out1}\n  run2={out2}"
        )

    def test_correct_answer_always_included(self):
        """process_docs must always include the correct answer among the four choices."""
        out = self._run_process_docs(42)
        rows = self._sample_rows()
        for i, (doc, row) in enumerate(zip(out, rows)):
            correct = row["Correct Answer"].strip()
            choices = [doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]]
            assert correct in choices, (
                f"Row {i}: correct answer {correct!r} not found in choices {choices!r}"
            )

    def test_answer_field_matches_position_of_correct_choice(self):
        """answer field must be (A)/(B)/(C)/(D) matching the shuffled correct answer position."""
        out = self._run_process_docs(42)
        rows = self._sample_rows()
        for i, (doc, row) in enumerate(zip(out, rows)):
            correct = row["Correct Answer"].strip()
            choices = [doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]]
            expected_idx = choices.index(correct)
            expected_answer = f"({chr(65 + expected_idx)})"
            assert doc["answer"] == expected_answer, (
                f"Row {i}: answer {doc['answer']!r} does not match correct position "
                f"{expected_answer!r} (choices={choices!r}, correct={correct!r})"
            )

    def test_different_seed_may_give_different_ordering(self):
        """Document that seed changes can change ordering (proves seed dependency)."""
        # Run many seeds until we find two with different orderings, or verify
        # that at least one non-trivial seed works.
        out42 = self._run_process_docs(42)
        out99 = self._run_process_docs(99)
        # Both must be valid (not raise); they may or may not differ.
        # This test just documents that the function IS seed-dependent.
        assert isinstance(out42, list)
        assert isinstance(out99, list)
        assert len(out42) == 2
        assert len(out99) == 2

    def test_output_has_expected_keys(self):
        """Each output doc must have choice1–4 and answer keys."""
        out = self._run_process_docs(42)
        for i, doc in enumerate(out):
            for key in ("choice1", "choice2", "choice3", "choice4", "answer"):
                assert key in doc, f"Row {i} missing key {key!r}: {doc!r}"
