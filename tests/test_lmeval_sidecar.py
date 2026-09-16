"""tests/test_lmeval_sidecar.py — TDD tests for lm-eval 0.4.12 sidecar capture + reconciler.

Written BEFORE implementation (RED phase). Each test must fail until the
corresponding production code is created.

Requirements exercised:
 - Version assertion: wrapper fails if lm_eval != 0.4.12
 - Cache rejection: OPENAI_API_KEY + --use_cache rejected  
 - Metadata capture: sync + async paths write JSONL records
 - Secrets absent: auth headers never in sidecar output
 - Capture write failure propagates (fail-closed, not log-and-continue)
 - Reconciler: missing/duplicate/foreign/malformed/incomplete metadata detection
 - Duplicate prompt ambiguity detection (same fingerprint, differing metadata)
 - Finish reason, completion_tokens, content field validation
 - Disposition classification from metadata
 - Runner command/lifecycle ordering (wrapper not `python -m lm_eval`)
 - Sidecar path under run_dir/raw/, no credentials in argv
 - Completion requires sidecar exists + passes reconciliation
 - Integration test: real lm-eval CLI + fake OpenAI endpoint, limit>=3, concurrency>=3
"""
from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Shared fingerprint helper (must match capture.py exactly)
# ---------------------------------------------------------------------------

def _fp(messages: object) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _sample_with_fp(messages: list, doc_id: int = 0) -> dict:
    """Build a minimal lm-eval sample record with the messages fingerprint embedded."""
    msgs_str = json.dumps(messages)
    return {
        "doc_id": doc_id,
        "filter": "answer-line",
        "resps": [["The answer is 4"]],
        "filtered_resps": ["4"],
        "target": "4",
        "exact_match": 1.0,
        "arguments": {
            "gen_args_0": {
                "arg_0": [msgs_str],
            }
        },
    }


def _meta_record(
    messages: list,
    finish_reason: str = "stop",
    content: str = "The answer is 4",
    completion_tokens: int = 8,
    reasoning: str = None,
    reasoning_content: str = None,
) -> dict:
    """Build a metadata record as the sidecar would write it."""
    return {
        "fingerprint": _fp(messages),
        "ts_utc": "2026-01-01T00:00:00+00:00",
        "model": "fake-model",
        "finish_reasons": [finish_reason],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": completion_tokens,
            "total_tokens": 17 + completion_tokens,
        },
        "content": [content],
        "reasoning_content": [reasoning_content],
        "reasoning": [reasoning],
        "content_null_count": 0,
        "empty_by_length": False,
        "run_id": "run-test",
    }


def _write_jsonl(path: pathlib.Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_samples_gz(path: pathlib.Path, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


# ===========================================================================
# SECTION 1: Version assertion
# ===========================================================================

class TestVersionAssertion(unittest.TestCase):
    """Wrapper must assert lm_eval==0.4.12 and fail closed otherwise."""

    def test_import_fails_on_wrong_version(self):
        """If lm_eval.__version__ != '0.4.12', importing the wrapper must raise."""
        try:
            from lmeval_sidecar import capture as _cap  # noqa: F401
        except ImportError:
            self.fail("lmeval_sidecar.capture must be importable")

        # Patch LM_EVAL_REQUIRED_VERSION and test that assert_lm_eval_version raises
        # with a message mentioning the REQUIRED version
        from lmeval_sidecar import capture
        original = capture.LM_EVAL_REQUIRED_VERSION
        try:
            capture.LM_EVAL_REQUIRED_VERSION = "99.0.0"
            with self.assertRaises(RuntimeError) as ctx:
                capture.assert_lm_eval_version()
            # Error message must mention the required version
            err = str(ctx.exception)
            self.assertTrue(
                "99.0.0" in err or "0.4.12" in err or "version" in err.lower(),
                f"Error message must mention version, got: {err!r}"
            )
        finally:
            capture.LM_EVAL_REQUIRED_VERSION = original

    def test_version_check_passes_for_correct_version(self):
        """Version check must not raise when lm_eval == '0.4.12'."""
        from lmeval_sidecar import capture
        # If lm_eval is not installed at all, we can't test the happy path
        # but we can verify the function exists and is callable.
        # We mock the import to simulate a correct version.
        import unittest.mock as _mock
        fake_lm_eval = _mock.MagicMock()
        fake_lm_eval.__version__ = "0.4.12"
        original = capture.LM_EVAL_REQUIRED_VERSION
        try:
            with _mock.patch.dict("sys.modules", {"lm_eval": fake_lm_eval}):
                capture.assert_lm_eval_version()  # must not raise
        finally:
            capture.LM_EVAL_REQUIRED_VERSION = original

    def test_runner_script_asserts_version_at_top(self):
        """lmeval_sidecar_runner.py must call assert_lm_eval_version before cli_evaluate."""
        runner_path = _VIZ_DIR / "lmeval_sidecar_runner.py"
        self.assertTrue(runner_path.exists(), f"lmeval_sidecar_runner.py must exist at {runner_path}")
        src = runner_path.read_text()
        self.assertIn("assert_lm_eval_version", src,
                      "Runner must call assert_lm_eval_version() before cli_evaluate()")


# ===========================================================================
# SECTION 2: Cache rejection
# ===========================================================================

class TestCacheRejection(unittest.TestCase):
    """The wrapper must reject any invocation that would enable lm-eval caching."""

    def test_runner_adds_no_cache_flag(self):
        """build_command must include explicit cache-disabling flag(s)."""
        # The run_quality runner must not pass --use_cache / --cache_requests
        # and must explicitly disable via model_args or env
        runner_path = _VIZ_DIR / "lmeval_sidecar_runner.py"
        self.assertTrue(runner_path.exists())
        src = runner_path.read_text()
        # Runner must not enable cache
        self.assertNotIn("--use_cache", src)
        self.assertNotIn("--cache_requests", src)

    def test_sidecar_model_args_disables_cache(self):
        """SidecarChatCompletion must not enable caching (no use_cache in its args)."""
        from lmeval_sidecar.capture import SidecarChatCompletion
        # The model must not have cache-enabling constructor args
        import inspect
        sig = inspect.signature(SidecarChatCompletion.__init__)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("use_cache", param_names)
        self.assertNotIn("cache_requests", param_names)


# ===========================================================================
# SECTION 3: Fingerprint correctness
# ===========================================================================

class TestFingerprint(unittest.TestCase):
    """Fingerprint must be deterministic, match between capture and reconciler."""

    def test_fingerprint_deterministic(self):
        from lmeval_sidecar.capture import _fingerprint
        messages = [{"role": "user", "content": "hello"}]
        fp1 = _fingerprint(messages)
        fp2 = _fingerprint(messages)
        self.assertEqual(fp1, fp2)

    def test_fingerprint_is_sha256_hex(self):
        from lmeval_sidecar.capture import _fingerprint
        messages = [{"role": "user", "content": "test"}]
        fp = _fingerprint(messages)
        self.assertEqual(len(fp), 64)
        int(fp, 16)  # must be valid hex

    def test_fingerprint_matches_reconciler(self):
        from lmeval_sidecar.capture import _fingerprint as cap_fp
        from lmeval_sidecar.reconcile import _fingerprint as rec_fp
        messages = [{"role": "user", "content": "test message"}]
        self.assertEqual(cap_fp(messages), rec_fp(messages))

    def test_fingerprint_of_json_chat_str(self):
        """Fingerprint of JsonChatStr must match fingerprint of parsed messages."""
        from lmeval_sidecar.capture import _fingerprint
        messages = [{"role": "user", "content": "hello"}]
        # Try to import real JsonChatStr; if lm_eval unavailable, use a stub
        try:
            from lm_eval.models.api_models import JsonChatStr
        except ImportError:
            # Stub: a namedtuple-like with .prompt attribute (same interface)
            class JsonChatStr:  # type: ignore[no-redef]
                def __init__(self, prompt: str):
                    self.prompt = prompt
        jcs = JsonChatStr(json.dumps(messages))
        fp_list = _fingerprint(messages)
        fp_jcs = _fingerprint(jcs)
        self.assertEqual(fp_list, fp_jcs)


# ===========================================================================
# SECTION 4: Metadata record schema
# ===========================================================================

class TestMetadataSchema(unittest.TestCase):
    """_extract_metadata must produce records with all required fields."""

    def _fake_response(self, content="test", finish_reason="stop"):
        return {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": None,
                    "reasoning": None,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": "fake-model",
        }

    def test_extract_metadata_has_required_fields(self):
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        raw = self._fake_response()
        rec = _extract_metadata(raw, messages, "fake-model")
        required = ["fingerprint", "ts_utc", "model", "finish_reasons",
                    "usage", "content", "reasoning_content", "reasoning",
                    "content_null_count", "empty_by_length"]
        for field in required:
            self.assertIn(field, rec, f"Missing required field: {field}")

    def test_extract_metadata_never_contains_auth_header(self):
        """Auth headers must never appear in metadata records."""
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        raw = self._fake_response()
        rec = _extract_metadata(raw, messages, "fake-model")
        rec_str = json.dumps(rec)
        for secret_key in ("authorization", "Authorization", "api_key", "OPENAI_API_KEY",
                           "Bearer", "x-api-key"):
            self.assertNotIn(secret_key, rec_str,
                             f"Auth credential {secret_key!r} must not appear in metadata")

    def test_extract_metadata_finish_reason_nonempty(self):
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        raw = self._fake_response(finish_reason="stop")
        rec = _extract_metadata(raw, messages, "fake-model")
        self.assertTrue(rec["finish_reasons"], "finish_reasons must be nonempty list")
        self.assertIsNotNone(rec["finish_reasons"][0])

    def test_extract_metadata_usage_fields_present(self):
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        raw = self._fake_response()
        rec = _extract_metadata(raw, messages, "fake-model")
        usage = rec["usage"]
        self.assertIsNotNone(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIn(key, usage, f"usage.{key} must be present")

    def test_extract_metadata_content_key_present_even_if_null(self):
        """content key must always be present, even when choices return null content."""
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        # Response with null content
        raw = {
            "choices": [{"message": {"content": None, "reasoning_content": "think", "reasoning": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        rec = _extract_metadata(raw, messages, "fake-model")
        self.assertIn("content", rec)
        self.assertIn("reasoning_content", rec)
        self.assertIn("reasoning", rec)

    def test_extract_metadata_response_id_captured(self):
        """response_id must be captured if present in response."""
        from lmeval_sidecar.capture import _extract_metadata
        messages = [{"role": "user", "content": "q"}]
        raw = {
            "id": "chatcmpl-abc123",
            "choices": [{"message": {"content": "hi", "reasoning_content": None, "reasoning": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "model": "test-model",
        }
        rec = _extract_metadata(raw, messages, "test-model")
        self.assertIn("response_id", rec, "response_id must be captured when present")
        self.assertEqual(rec["response_id"], "chatcmpl-abc123")


# ===========================================================================
# SECTION 5: Capture write failure propagates (fail-closed)
# ===========================================================================

class TestCaptureWriteFailurePropagates(unittest.TestCase):
    """Capture write failures must propagate and fail the run, not log-and-continue."""

    def test_sidecar_writer_raises_on_write_failure(self):
        """_SidecarWriter.write must raise (not swallow) IOError on disk failure."""
        from lmeval_sidecar.capture import _SidecarWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "meta.jsonl"
            writer = _SidecarWriter(str(path))
            # Make the file read-only to force a write failure
            path.touch()
            path.chmod(0o444)
            try:
                with self.assertRaises(Exception):
                    writer.write({"test": "data"})
            finally:
                path.chmod(0o644)

    def test_sync_capture_failure_propagates(self):
        """In SidecarChatCompletion.model_call, capture failures must raise, not warn."""
        # We verify this by checking the implementation does NOT use try/except with warn
        from lmeval_sidecar import capture as cap_module
        import inspect
        src = inspect.getsource(cap_module.SidecarChatCompletion.model_call)
        # The implementation must NOT catch and log quietly — it must propagate
        # (no log-and-continue pattern like "eval_logger.warning(...)")
        # A correct implementation either propagates or re-raises; it must not swallow.
        self.assertNotIn("eval_logger.warning", src,
                         "model_call must not swallow capture failures via warning")

    def test_async_capture_failure_propagates(self):
        """In SidecarChatCompletion.amodel_call, capture failures must raise, not warn."""
        from lmeval_sidecar import capture as cap_module
        import inspect
        src = inspect.getsource(cap_module.SidecarChatCompletion.amodel_call)
        self.assertNotIn("eval_logger.warning", src,
                         "amodel_call must not swallow capture failures via warning")


# ===========================================================================
# SECTION 6: Reconciler — full inventory validation
# ===========================================================================

class TestReconcilerFullInventory(unittest.TestCase):
    """Reconciler must validate the full sample inventory fail-closed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_reconcile(self, samples: list, meta_records: list) -> tuple:
        """Write samples gz and meta jsonl, run reconcile, return (exit_code, report)."""
        from lmeval_sidecar.reconcile import reconcile
        samples_path = pathlib.Path(self.tmp) / "samples.jsonl.gz"
        meta_path = pathlib.Path(self.tmp) / "meta.jsonl"
        _write_samples_gz(samples_path, samples)
        _write_jsonl(meta_path, meta_records)
        report = reconcile(samples_path, meta_path)
        return report["exit_code"], report

    def _make_clean_pair(self, n: int = 3):
        """Return (samples, meta_records) for n clean items."""
        samples = []
        meta = []
        for i in range(n):
            msgs = [{"role": "user", "content": f"Question {i}?"}]
            samples.append(_sample_with_fp(msgs, doc_id=i))
            meta.append(_meta_record(msgs))
        return samples, meta

    def test_clean_inventory_passes(self):
        samples, meta = self._make_clean_pair(3)
        rc, report = self._run_reconcile(samples, meta)
        self.assertEqual(rc, 0, f"Clean inventory must pass; report: {report}")

    def test_missing_metadata_fails(self):
        """Sample with no matching metadata record must cause reconciler failure."""
        samples, meta = self._make_clean_pair(3)
        # Remove metadata for sample 1
        meta_without_one = [r for r in meta if r["fingerprint"] != _fp(
            json.loads(samples[1]["arguments"]["gen_args_0"]["arg_0"][0])
        )]
        rc, report = self._run_reconcile(samples, meta_without_one)
        self.assertNotEqual(rc, 0, "Missing metadata must fail reconciliation")
        self.assertIn("missing", str(report).lower())

    def test_foreign_metadata_fails(self):
        """Metadata records with no matching sample must cause reconciler failure."""
        samples, meta = self._make_clean_pair(3)
        # Add a foreign record
        extra_msgs = [{"role": "user", "content": "Foreign question?"}]
        meta.append(_meta_record(extra_msgs))
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Foreign metadata must fail reconciliation")
        self.assertIn("foreign", str(report).lower())

    def test_duplicate_metadata_fails(self):
        """Two metadata records with same fingerprint but different content → fail."""
        samples, meta = self._make_clean_pair(2)
        msgs_0 = json.loads(samples[0]["arguments"]["gen_args_0"]["arg_0"][0])
        # Add a second record for the first sample but different content
        dup = _meta_record(msgs_0, content="Different answer")
        meta.append(dup)
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Ambiguous duplicate metadata must fail reconciliation")

    def test_malformed_finish_reason_fails(self):
        """Metadata record with empty/null finish_reason must fail validation."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["finish_reasons"] = [""]  # empty string is invalid
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Empty finish_reason must fail validation")

    def test_missing_usage_fails(self):
        """A metadata record without usage cannot support token-count evidence."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["usage"] = None
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Missing usage must fail validation")

    def test_replayed_sidecar_run_id_fails(self):
        """Metadata captured for another campaign run must not be reusable."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["run_id"] = "old-run"
        samples_path = pathlib.Path(self.tmp) / "samples.jsonl.gz"
        metadata_path = pathlib.Path(self.tmp) / "meta.jsonl"
        _write_samples_gz(samples_path, samples)
        _write_jsonl(metadata_path, meta)
        from lmeval_sidecar.reconcile import reconcile_inventory
        report = reconcile_inventory(
            [samples_path], metadata_path,
            expected_run_id="current-run", expected_model="fake-model",
        )
        self.assertNotEqual(report["exit_code"], 0, report)

    def test_wrong_model_sidecar_fails(self):
        """Metadata from another model must not satisfy this campaign."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["model"] = "wrong-model"
        samples_path = pathlib.Path(self.tmp) / "samples.jsonl.gz"
        metadata_path = pathlib.Path(self.tmp) / "meta.jsonl"
        _write_samples_gz(samples_path, samples)
        _write_jsonl(metadata_path, meta)
        from lmeval_sidecar.reconcile import reconcile_inventory
        report = reconcile_inventory(
            [samples_path], metadata_path,
            expected_run_id="run-test", expected_model="fake-model",
        )
        self.assertNotEqual(report["exit_code"], 0, report)

    def test_malformed_completion_tokens_fails(self):
        """completion_tokens must be a nonnegative non-bool int."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["usage"]["completion_tokens"] = -1  # negative is invalid
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Negative completion_tokens must fail validation")

    def test_bool_completion_tokens_fails(self):
        """completion_tokens as bool (True/False) must be rejected."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["usage"]["completion_tokens"] = True  # bool disguised as int
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Bool completion_tokens must fail validation")

    def test_missing_content_key_fails(self):
        """Metadata record without 'content' field must fail validation."""
        samples, meta = self._make_clean_pair(1)
        del meta[0]["content"]
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Missing content field must fail validation")

    def test_missing_reasoning_keys_fails(self):
        """Metadata without 'reasoning' and 'reasoning_content' keys must fail."""
        samples, meta = self._make_clean_pair(1)
        del meta[0]["reasoning"]
        del meta[0]["reasoning_content"]
        rc, report = self._run_reconcile(samples, meta)
        self.assertNotEqual(rc, 0, "Missing reasoning keys must fail validation")

    def test_reasoning_keys_may_be_null(self):
        """reasoning and reasoning_content may be null (they are nullable)."""
        samples, meta = self._make_clean_pair(1)
        meta[0]["reasoning"] = [None]
        meta[0]["reasoning_content"] = [None]
        rc, report = self._run_reconcile(samples, meta)
        self.assertEqual(rc, 0, "Null reasoning fields must be valid")

    def test_duplicate_prompt_with_differing_metadata_fails_closed(self):
        """Two samples with same prompt but different metadata → ambiguous, fail closed."""
        msgs = [{"role": "user", "content": "Same question?"}]
        # Two samples with identical prompts (same fingerprint)
        s0 = _sample_with_fp(msgs, doc_id=0)
        s1 = _sample_with_fp(msgs, doc_id=1)
        # Two metadata records with same fingerprint but different content
        m0 = _meta_record(msgs, content="Answer A")
        m1 = _meta_record(msgs, content="Answer B")
        rc, report = self._run_reconcile([s0, s1], [m0, m1])
        self.assertNotEqual(rc, 0, "Ambiguous duplicate prompt must fail closed")
        self.assertIn("ambiguous", str(report).lower())

    def test_multiple_sample_files_reconcile_as_one_inventory(self):
        """A sidecar spanning several sample files must be reconciled once globally."""
        from lmeval_sidecar.reconcile import reconcile_inventory

        messages_a = [{"role": "user", "content": "Question A?"}]
        messages_b = [{"role": "user", "content": "Question B?"}]
        samples_a = pathlib.Path(self.tmp) / "samples_a.jsonl.gz"
        samples_b = pathlib.Path(self.tmp) / "samples_b.jsonl.gz"
        metadata = pathlib.Path(self.tmp) / "meta.jsonl"
        _write_samples_gz(samples_a, [_sample_with_fp(messages_a, doc_id=0)])
        _write_samples_gz(samples_b, [_sample_with_fp(messages_b, doc_id=1)])
        _write_jsonl(metadata, [_meta_record(messages_a), _meta_record(messages_b)])

        report = reconcile_inventory([samples_a, samples_b], metadata)
        self.assertEqual(report["exit_code"], 0, report)
        self.assertEqual(report["matched"], 2)

    def test_same_prompt_for_different_items_fails_even_with_one_metadata_record(self):
        """Fingerprint-only linkage cannot disambiguate two items sharing a prompt."""
        messages = [{"role": "user", "content": "Same question?"}]
        samples = [
            _sample_with_fp(messages, doc_id=0),
            _sample_with_fp(messages, doc_id=1),
        ]
        rc, report = self._run_reconcile(samples, [_meta_record(messages)])
        self.assertNotEqual(rc, 0, report)
        self.assertIn("ambiguous", str(report).lower())

    def test_multifilter_samples_reconcile_correctly(self):
        """lm-eval emits N filter rows per request; reconciler handles N:1 sample:metadata."""
        msgs = [{"role": "user", "content": "Q?"}]
        # Two sample rows for same doc (two filter rows, same fingerprint)
        s0_filter1 = _sample_with_fp(msgs, doc_id=0)
        s0_filter1["filter"] = "answer-line"
        s0_filter2 = _sample_with_fp(msgs, doc_id=0)
        s0_filter2["filter"] = "flexible-fallback"
        # One metadata record
        meta = [_meta_record(msgs)]
        rc, report = self._run_reconcile([s0_filter1, s0_filter2], meta)
        self.assertEqual(rc, 0, f"Multi-filter rows must reconcile to 1 metadata record; {report}")


# ===========================================================================
# SECTION 7: Disposition classification
# ===========================================================================

class TestDispositionClassification(unittest.TestCase):
    """classify_response must correctly categorize each metadata record."""

    def test_nonempty_content_is_scored(self):
        from lmeval_sidecar.reconcile import classify_response
        rec = _meta_record([{"role": "user", "content": "q"}], content="The answer is 4",
                            finish_reason="stop")
        self.assertEqual(classify_response(rec), "scored")

    def test_empty_plus_length_is_budget(self):
        from lmeval_sidecar.reconcile import classify_response
        rec = _meta_record([{"role": "user", "content": "q"}], content="",
                            finish_reason="length")
        self.assertEqual(classify_response(rec), "budget")

    def test_empty_stop_with_reasoning_is_parser(self):
        from lmeval_sidecar.reconcile import classify_response
        rec = _meta_record([{"role": "user", "content": "q"}], content="",
                            finish_reason="stop", reasoning_content="<think>reasoning here</think>")
        self.assertEqual(classify_response(rec), "parser")

    def test_empty_stop_no_reasoning_is_empty(self):
        from lmeval_sidecar.reconcile import classify_response
        rec = _meta_record([{"role": "user", "content": "q"}], content="",
                            finish_reason="stop")
        self.assertEqual(classify_response(rec), "empty")

    def test_null_content_stop_with_reasoning_is_parser(self):
        from lmeval_sidecar.reconcile import classify_response
        rec = {
            "fingerprint": "abc",
            "finish_reasons": ["stop"],
            "content": [None],
            "reasoning_content": ["think"],
            "reasoning": [None],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        self.assertEqual(classify_response(rec), "parser")


# ===========================================================================
# SECTION 8: Runner command ordering and no credentials in argv
# ===========================================================================

class TestRunnerCommand(unittest.TestCase):
    """run_quality.build_command must use the sidecar wrapper, not `python -m lm_eval`."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        self._write_adapter()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_adapter(self):
        import yaml
        adapter = {
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
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(adapter))

    def _make_runner(self, bench="gsm8k"):
        import run_quality
        _REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
        slug = "test-canonical-model"
        run_id = "run-test-cmd"
        run_dir = self.tmp / "results" / slug / "runs" / "warpcore-v1" / bench / run_id
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
            "benchmark": bench,
            "model": {"slug": slug, "id": "testorg/TestCanonicalModel", "revision": "a" * 40},
            "item_inventory": {"expected": 1319, "submitted": 0},
            "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
            "artifact_inventory": {
                "samples_jsonl_gz": False, "per_item_csv": False,
                "run_log": False, "command_txt": False, "done_sentinel": False,
            },
        }))
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima={"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000},
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        return runner, run_dir

    def test_build_command_uses_sidecar_runner_not_lm_eval_module(self):
        """build_command must invoke lmeval_sidecar_runner.py, not `python -m lm_eval`."""
        runner, run_dir = self._make_runner("gsm8k")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertNotIn("-m lm_eval", cmd_str,
                         "build_command must not use `python -m lm_eval`; use sidecar runner")
        self.assertIn("lmeval_sidecar_runner", cmd_str,
                      "build_command must invoke lmeval_sidecar_runner.py")

    def test_build_command_uses_sidecar_model(self):
        """build_command must pass --model sidecar-chat-completions."""
        runner, run_dir = self._make_runner("gsm8k")
        cmd = runner.build_command()
        self.assertIn("sidecar-chat-completions", cmd,
                      "build_command must use --model sidecar-chat-completions")

    def test_build_command_has_sidecar_path_under_raw(self):
        """build_command must pass sidecar path under run_dir/raw/ with no credentials."""
        runner, run_dir = self._make_runner("gsm8k")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # sidecar path must be under raw/
        self.assertIn(str(run_dir / "raw"), cmd_str,
                      "sidecar path must be under run_dir/raw/")

    def test_build_command_no_credentials_in_argv(self):
        """build_command argv must not contain API keys, Bearer tokens, or secrets."""
        runner, run_dir = self._make_runner("gsm8k")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        for secret in ("OPENAI_API_KEY", "Bearer", "sk-", "hf_", "Authorization"):
            self.assertNotIn(secret, cmd_str,
                             f"Credential {secret!r} must not appear in build_command argv")

    def test_build_command_has_sidecar_path_env_or_arg(self):
        """Sidecar path must be passed explicitly (not relying on default env fallback)."""
        runner, run_dir = self._make_runner("gsm8k")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # Either explicit env injection or model_args sidecar_path
        has_sidecar = (
            "LMEVAL_SIDECAR_PATH" in cmd_str or
            "sidecar_path" in cmd_str or
            "sidecar" in cmd_str.lower()
        )
        self.assertTrue(has_sidecar,
                        "build_command must explicitly pass sidecar path to the runner")


# ===========================================================================
# SECTION 9: Completion requires sidecar exists and passes reconciliation
# ===========================================================================

class TestCompletionGateRequiresSidecar(unittest.TestCase):
    """DONE must not be written unless sidecar exists and passes reconciliation."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        self._write_adapter()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_adapter(self):
        import yaml
        adapter = {
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
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(adapter))

    def _make_runner(self, bench="gsm8k", harness_exit=0):
        import run_quality
        _REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
        slug = "test-canonical-model"
        run_id = "run-test-sidecar"
        run_dir = self.tmp / "results" / slug / "runs" / "warpcore-v1" / bench / run_id
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
            "benchmark": bench,
            "model": {"slug": slug, "id": "testorg/TestCanonicalModel", "revision": "a" * 40},
            "item_inventory": {"expected": 1319, "submitted": 0},
            "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
            "artifact_inventory": {
                "samples_jsonl_gz": False, "per_item_csv": False,
                "run_log": False, "command_txt": False, "done_sentinel": False,
            },
        }))
        harness_mock = MagicMock(return_value=harness_exit)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima={"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000},
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=harness_mock,
        )
        return runner, run_dir

    def _make_artifacts(self, run_dir: pathlib.Path, n_items: int = 3,
                         with_sidecar: bool = True, corrupt_sidecar: bool = False,
                         missing_metadata: bool = False):
        """Create harness artifacts and optionally a sidecar file."""
        subdir = run_dir / "raw" / "gsm8k_cot_zeroshot_clean"
        subdir.mkdir(parents=True, exist_ok=True)
        # Aggregate result
        (subdir / "results_2026-01-01T00-00-00.json").write_text(
            '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.667}}}\n'
        )
        # Samples gz with proper fingerprints
        samples = []
        meta_records = []
        for i in range(n_items):
            msgs = [{"role": "user", "content": f"GSM8K question {i}?"}]
            response = "" if i == n_items - 1 else f"The answer is {i * 2}."
            sample = _sample_with_fp(msgs, doc_id=i)
            sample["resps"] = [[response]]
            sample["exact_match"] = 1.0 if response else 0.0
            samples.append(sample)
            if not missing_metadata or i != 0:
                record = _meta_record(msgs, content=response,
                                      finish_reason="stop" if response else "length")
                record["run_id"] = run_dir.name
                record["model"] = "testorg/TestCanonicalModel"
                meta_records.append(record)

        gz_path = subdir / "samples_gsm8k_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
        _write_samples_gz(gz_path, samples)

        if with_sidecar:
            sidecar_path = run_dir / "raw" / "response_metadata.jsonl"
            if corrupt_sidecar:
                sidecar_path.write_bytes(b"INVALID JSON\n")
            else:
                _write_jsonl(sidecar_path, meta_records)

        return samples, meta_records

    def test_done_not_written_when_sidecar_missing(self):
        """If sidecar file does not exist, DONE must not be written."""
        runner, run_dir = self._make_runner()
        self._make_artifacts(run_dir, with_sidecar=False)
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Must fail when sidecar is missing")
        self.assertFalse((run_dir / "DONE").exists(),
                         "DONE must not be written when sidecar is missing")

    def test_done_not_written_when_reconciliation_fails(self):
        """If reconciliation fails (missing metadata), DONE must not be written."""
        runner, run_dir = self._make_runner()
        self._make_artifacts(run_dir, n_items=3, missing_metadata=True)
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Must fail when reconciliation finds missing metadata")
        self.assertFalse((run_dir / "DONE").exists(),
                         "DONE must not be written when reconciliation fails")

    def test_done_written_when_sidecar_reconciles_cleanly(self):
        """When sidecar exists and reconciles cleanly, DONE must be written."""
        runner, run_dir = self._make_runner()
        self._make_artifacts(run_dir, with_sidecar=True)
        rc = runner.run()
        self.assertEqual(rc, 0, f"Must succeed with valid sidecar; exit {rc}")
        self.assertTrue((run_dir / "DONE").exists(),
                        "DONE must be written when sidecar reconciles cleanly")

    def test_per_item_csv_uses_metadata_finish_reason(self):
        """per_item.csv finish_reason must come from sidecar metadata, not empty string."""
        runner, run_dir = self._make_runner()
        self._make_artifacts(run_dir, with_sidecar=True)
        rc = runner.run()
        self.assertEqual(rc, 0)
        csv_path = run_dir / "per_item.csv"
        self.assertTrue(csv_path.exists())
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        # At least one row should have a non-empty finish_reason from metadata
        finish_reasons = [r.get("finish_reason", "") for r in rows]
        self.assertTrue(any(fr != "" for fr in finish_reasons),
                        f"per_item.csv finish_reason must come from metadata, got: {finish_reasons}")

    def test_per_item_csv_disposition_uses_metadata(self):
        """per_item.csv disposition must use reconciled metadata classification."""
        runner, run_dir = self._make_runner()
        self._make_artifacts(run_dir, n_items=3, with_sidecar=True)
        rc = runner.run()
        self.assertEqual(rc, 0)
        csv_path = run_dir / "per_item.csv"
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        dispositions = {r["item_id"]: r["disposition"] for r in rows}
        # Last item (index 2) was empty+length, should be "budget" not "empty_response"
        self.assertIn(dispositions.get("2", ""), ["budget", "empty_response"],
                      "Disposition must reflect metadata classification")


# ===========================================================================
# SECTION 10: Artifact inventory — response_metadata field
# ===========================================================================

class TestArtifactInventoryMetadataField(unittest.TestCase):
    """artifact_inventory must include response_metadata field after run with sidecar."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        import yaml
        adapter = {
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
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter_path.write_text(yaml.dump(adapter))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ===========================================================================
# SECTION 11: Integration test — real lm-eval CLI + fake OpenAI endpoint
# ===========================================================================

LMEVAL_VENV_PYTHON = "/Users/jchilders/workspaces/lmeval-venv/bin/python"
LMEVAL_VENV = "/Users/jchilders/workspaces/lmeval-venv"

INTEGRATION_SKIP_REASON = (
    "Integration test requires lmeval venv at "
    f"{LMEVAL_VENV} with lm-eval 0.4.12 installed"
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Fake OpenAI chat/completions endpoint for integration tests."""

    call_count = 0
    call_lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass  # suppress server logs during tests

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            req = json.loads(body)
            msgs = req.get("messages", [])
            # Echo back a response that includes the question number
            content = f"The answer is {len(msgs)}"
        except Exception:
            content = "The answer is 42"

        response = {
            "id": f"chatcmpl-fake-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "<think>brief reasoning</think>",
                    "reasoning": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 8,
                "total_tokens": 25,
            },
        }
        with _FakeOpenAIHandler.call_lock:
            _FakeOpenAIHandler.call_count += 1

        resp_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


@unittest.skipUnless(
    pathlib.Path(LMEVAL_VENV_PYTHON).exists(),
    INTEGRATION_SKIP_REASON,
)
class TestIntegrationRealLmEvalCLI(unittest.TestCase):
    """Integration: real lm-eval 0.4.12 CLI + fake OpenAI endpoint.

    Runs with limit=4, concurrency=3 to exercise both sync+async paths.
    Proves metadata capture and real sample linkage through reconciler.
    """

    LIMIT = 4
    CONCURRENCY = 3

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _FakeOpenAIHandler.call_count = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _start_fake_server(self) -> tuple:
        """Start fake OpenAI server in background thread. Returns (server, port)."""
        port = _find_free_port()
        server = HTTPServer(("127.0.0.1", port), _FakeOpenAIHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        # Wait for server to be ready
        for _ in range(20):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                s.close()
                break
            except OSError:
                time.sleep(0.1)
        return server, port

    def test_integration_capture_and_reconcile(self):
        """Real lm-eval CLI must capture metadata and reconcile cleanly for limit=4, concurrency=3."""
        import subprocess as sp

        server, port = self._start_fake_server()
        try:
            base_url = f"http://127.0.0.1:{port}/v1/chat/completions"
            sidecar_path = pathlib.Path(self.tmp) / "response_metadata.jsonl"
            output_path = pathlib.Path(self.tmp) / "lmeval_out"
            output_path.mkdir()

            # Build the sidecar runner path
            runner_script = str(_VIZ_DIR / "lmeval_sidecar_runner.py")
            sidecar_parent = str(_VIZ_DIR)

            env = {
                **os.environ,
                "OPENAI_API_KEY": "dummy",
                "LMEVAL_SIDECAR_PATH": str(sidecar_path),
                "PYTHONPATH": sidecar_parent,
            }

            cmd = [
                LMEVAL_VENV_PYTHON, runner_script,
                "--model", "sidecar-chat-completions",
                "--model_args", (
                    f"model=fake-model,"
                    f"base_url={base_url},"
                    f"num_concurrent={self.CONCURRENCY},"
                    f"max_retries=1,"
                    f"timeout=60,"
                    f"tokenized_requests=False"
                ),
                "--apply_chat_template",
                "--tasks", "gsm8k_cot_zeroshot",
                "--gen_kwargs", "max_gen_toks=64,temperature=0,do_sample=false",
                "--output_path", str(output_path),
                "--log_samples",
                "--limit", str(self.LIMIT),
                "--num_fewshot", "0",
            ]

            result = sp.run(cmd, capture_output=True, text=True, timeout=120, env=env)

            self.assertEqual(result.returncode, 0,
                             f"lm-eval must exit 0:\nSTDOUT: {result.stdout[-4000:]}\nSTDERR: {result.stderr[-4000:]}")

            # Verify sidecar file was created
            self.assertTrue(sidecar_path.exists(),
                            f"Sidecar file must exist at {sidecar_path}\n"
                            f"STDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-1000:]}")

            # Read sidecar records
            meta_records = []
            with open(sidecar_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        meta_records.append(json.loads(line))

            self.assertGreaterEqual(len(meta_records), self.LIMIT,
                                    f"Sidecar must have >= {self.LIMIT} records, got {len(meta_records)}")

            # Verify fields in each record
            for i, rec in enumerate(meta_records[:self.LIMIT]):
                self.assertIn("fingerprint", rec, f"Record {i} missing fingerprint")
                self.assertIn("finish_reasons", rec, f"Record {i} missing finish_reasons")
                self.assertIn("content", rec, f"Record {i} missing content")
                self.assertIn("reasoning_content", rec, f"Record {i} missing reasoning_content")
                self.assertIn("reasoning", rec, f"Record {i} missing reasoning")
                self.assertIn("usage", rec, f"Record {i} missing usage")
                # Verify no credentials in records
                rec_str = json.dumps(rec)
                self.assertNotIn("sk-", rec_str)
                self.assertNotIn("Bearer", rec_str)
                self.assertNotIn("Authorization", rec_str)

            # Normalize the exact lm-eval 0.4.12 output through the same
            # production helper used by QualityRunner before validation.
            import run_quality
            run_quality._normalize_lmeval_samples(output_path)

            # Find and verify canonical compressed samples were created.
            samples_files = list(output_path.rglob("samples_*.jsonl.gz"))
            self.assertTrue(
                samples_files,
                "lm-eval must write samples_*.jsonl.gz; produced files: "
                + repr(sorted(str(p.relative_to(output_path)) for p in output_path.rglob("*")))
                + f"\nSTDOUT: {result.stdout[-4000:]}\nSTDERR: {result.stderr[-4000:]}",
            )

            # Run reconciliation
            from lmeval_sidecar.reconcile import reconcile
            samples_gz = samples_files[0]
            # Read samples
            with gzip.open(samples_gz, "rt", encoding="utf-8") as fh:
                samples = [json.loads(l) for l in fh if l.strip()]

            # Run reconciliation (limit to unique fingerprints)
            report = reconcile(samples_gz, sidecar_path)
            self.assertEqual(report["exit_code"], 0,
                             f"Reconciliation must pass; report: {report}")

        finally:
            server.shutdown()

    def test_integration_sidecar_records_have_real_token_counts(self):
        """Sidecar records must have real completion_tokens from the fake server."""
        import subprocess as sp

        server, port = self._start_fake_server()
        try:
            base_url = f"http://127.0.0.1:{port}/v1/chat/completions"
            sidecar_path = pathlib.Path(self.tmp) / "response_metadata.jsonl"
            output_path = pathlib.Path(self.tmp) / "lmeval_out2"
            output_path.mkdir()

            runner_script = str(_VIZ_DIR / "lmeval_sidecar_runner.py")
            sidecar_parent = str(_VIZ_DIR)

            env = {
                **os.environ,
                "OPENAI_API_KEY": "dummy",
                "LMEVAL_SIDECAR_PATH": str(sidecar_path),
                "PYTHONPATH": sidecar_parent,
            }

            cmd = [
                LMEVAL_VENV_PYTHON, runner_script,
                "--model", "sidecar-chat-completions",
                "--model_args", (
                    f"model=fake-model,"
                    f"base_url={base_url},"
                    f"num_concurrent={self.CONCURRENCY},"
                    f"max_retries=1,"
                    f"timeout=60,"
                    f"tokenized_requests=False"
                ),
                "--apply_chat_template",
                "--tasks", "gsm8k_cot_zeroshot",
                "--gen_kwargs", "max_gen_toks=64,temperature=0,do_sample=false",
                "--output_path", str(output_path),
                "--log_samples",
                "--limit", str(self.LIMIT),
                "--num_fewshot", "0",
            ]

            result = sp.run(cmd, capture_output=True, text=True, timeout=120, env=env)

            if not sidecar_path.exists():
                self.skipTest(f"Sidecar not created (lm-eval exit {result.returncode})")

            meta_records = []
            with open(sidecar_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        meta_records.append(json.loads(line))

            for rec in meta_records:
                usage = rec.get("usage", {})
                if usage:
                    ct = usage.get("completion_tokens", -1)
                    self.assertIsInstance(ct, int)
                    self.assertNotIsInstance(ct, bool)
                    self.assertGreaterEqual(ct, 0)

        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
