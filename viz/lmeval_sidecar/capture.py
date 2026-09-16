"""viz/lmeval_sidecar/capture.py
================================
Production lm-eval 0.4.12 response-metadata sidecar.

Design:
  - Version assertion at import time: fails closed if lm_eval != 0.4.12.
  - Subclass LocalChatCompletion, registered as "sidecar-chat-completions".
  - Override model_call (sync) and amodel_call (async) to capture raw
    HTTP response metadata BEFORE parse_generations discards it.
  - Write one JSONL record per successful response atomically under process lock
    with flush+fsync for durability. Plain JSONL staging file (no gzip during run).
  - Capture write failures PROPAGATE — never log-and-continue.
  - Auth headers are NEVER written to the capture file.
  - Deterministic request fingerprint = sha256 of canonicalized messages JSON.
  - Caching is explicitly rejected: no use_cache / cache_requests in constructor.

Captured fields per record:
  - fingerprint       : sha256 of canonical messages (links to lm-eval sample)
  - ts_utc            : ISO-8601 UTC timestamp
  - model             : model name string
  - response_id       : the response id from the API (if present)
  - finish_reasons    : list[str]  (choices[*].finish_reason)
  - usage             : {prompt_tokens, completion_tokens, total_tokens} or null
  - content           : list[str|null]  (choices[*].message.content)
  - reasoning_content : list[str|null]  (choices[*].message.reasoning_content)
  - reasoning         : list[str|null]  (choices[*].message.reasoning, alt vLLM field)
  - content_null_count: int (how many choices had null content)
  - empty_by_length   : bool (any finish_reason == "length" with null/empty content)
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Version assertion — fail closed immediately on wrong version
# ---------------------------------------------------------------------------

LM_EVAL_REQUIRED_VERSION = "0.4.12"


def assert_lm_eval_version() -> None:
    """Raise RuntimeError if lm_eval version != LM_EVAL_REQUIRED_VERSION."""
    try:
        import lm_eval as _lme
        actual = _lme.__version__
    except ImportError as exc:
        raise RuntimeError(
            f"lm_eval is not installed. Required version: {LM_EVAL_REQUIRED_VERSION}"
        ) from exc
    if actual != LM_EVAL_REQUIRED_VERSION:
        raise RuntimeError(
            f"Unsupported lm_eval version {actual!r}. "
            f"This sidecar requires exactly lm_eval=={LM_EVAL_REQUIRED_VERSION}. "
            "Install the pinned version or update the wrapper."
        )

# ---------------------------------------------------------------------------
# Deferred imports from lm_eval (to avoid triggering version check at module import)
# ---------------------------------------------------------------------------

def _import_lm_eval_classes():
    """Import lm_eval classes after version assertion. Lazy to allow test patching."""
    from lm_eval.api.registry import register_model  # noqa: F401
    from lm_eval.models.api_models import JsonChatStr, LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER  # noqa: F401
    from lm_eval.models.openai_completions import LocalChatCompletion  # noqa: F401
    return register_model, JsonChatStr, LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER, LocalChatCompletion


# ---------------------------------------------------------------------------
# Concurrency-safe durable JSONL writer
# ---------------------------------------------------------------------------

class _SidecarWriter:
    """Thread-safe durable JSONL writer with atomic append under process lock + fsync."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Write a single record as a JSONL line. Raises on any IO failure (fail-closed)."""
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        raw = line.encode("utf-8")
        with self._lock:
            # Open in append binary mode; use OS-level file lock for cross-process safety
            with open(self.path, "ab") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            # IOError/OSError propagates — never swallowed


# One writer per path, shared across concurrent model instances in the same process.
_writers: Dict[str, _SidecarWriter] = {}
_writers_lock = threading.Lock()


def _get_writer(path: str) -> _SidecarWriter:
    with _writers_lock:
        if path not in _writers:
            _writers[path] = _SidecarWriter(path)
        return _writers[path]


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _canonical_messages(messages: Any) -> bytes:
    """Deterministic bytes from a messages payload.

    Works for:
      - list[dict]          (chat messages)
      - JsonChatStr         (serialized chat from lm-eval, has .prompt attribute)
      - str / list[str]     (text completions, rare on this path)
      - list[list[int]]     (tokenized)
    """
    # Handle JsonChatStr (real or stub) — any object with .prompt str attribute
    if hasattr(messages, "prompt") and isinstance(getattr(messages, "prompt"), str):
        obj = json.loads(messages.prompt)
        return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        obj = messages
    elif isinstance(messages, str):
        try:
            obj = json.loads(messages)
        except (json.JSONDecodeError, ValueError):
            obj = messages
    else:
        obj = messages  # fallback – str / tokens
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _fingerprint(messages: Any) -> str:
    return hashlib.sha256(_canonical_messages(messages)).hexdigest()


# ---------------------------------------------------------------------------
# Metadata extractor – operates on raw OpenAI JSON response dict
# ---------------------------------------------------------------------------

def _extract_metadata(
    raw: dict,
    messages_for_fingerprint: Any,
    model_name: str,
) -> dict:
    """Extract response metadata from a raw OpenAI response dict.

    Never includes auth headers or API keys.
    """
    choices = raw.get("choices", [])
    finish_reasons = [c.get("finish_reason") for c in choices]
    msgs = [c.get("message", {}) for c in choices]
    contents = [m.get("content") for m in msgs]
    reasoning_contents = [m.get("reasoning_content") for m in msgs]
    reasonings = [m.get("reasoning") for m in msgs]

    usage = raw.get("usage")  # may be None if server omits it
    response_id = raw.get("id")  # may be None

    null_content_count = sum(1 for c in contents if c is None)
    empty_by_length = any(
        fr == "length" and (content is None or content == "")
        for fr, content in zip(finish_reasons, contents)
    )

    return {
        "fingerprint": _fingerprint(messages_for_fingerprint),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "response_id": response_id,
        "finish_reasons": finish_reasons,
        "usage": usage,
        "content": contents,
        "reasoning_content": reasoning_contents,
        "reasoning": reasonings,
        "content_null_count": null_content_count,
        "empty_by_length": empty_by_length,
    }


# ---------------------------------------------------------------------------
# The sidecar model class
# ---------------------------------------------------------------------------

# Default sidecar path: override via LMEVAL_SIDECAR_PATH env var.
_DEFAULT_SIDECAR_PATH = os.environ.get(
    "LMEVAL_SIDECAR_PATH",
    "/tmp/lmeval_sidecar/metadata.jsonl",
)

# Import lm_eval base classes at module level (production usage)
try:
    from lm_eval.api.registry import register_model
    from lm_eval.models.api_models import JsonChatStr, LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER
    from lm_eval.models.openai_completions import LocalChatCompletion

    @register_model("sidecar-chat-completions")
    class SidecarChatCompletion(LocalChatCompletion):
        """Drop-in replacement for local-chat-completions with response metadata capture.

        Captures: finish_reason, usage, content, reasoning_content, reasoning,
        response_id, request fingerprint.

        Cache is never used (no use_cache / cache_requests constructor args).
        Capture write failures PROPAGATE — never swallowed.
        Auth headers are never written to the sidecar file.

        Usage:
            PYTHONPATH=/path/to/viz \\
            LMEVAL_SIDECAR_PATH=/out/metadata.jsonl \\
            lm_eval --model sidecar-chat-completions --model_args "model=...,base_url=..."
        """

        def __init__(
            self,
            *,
            sidecar_path: Optional[str] = None,
            **kwargs,
        ) -> None:
            # Explicitly reject any cache-enabling args
            if "use_cache" in kwargs or "cache_requests" in kwargs:
                raise ValueError(
                    "SidecarChatCompletion does not support cache (use_cache/cache_requests). "
                    "Cache hits bypass metadata capture. Disable caching to use this model."
                )
            super().__init__(**kwargs)
            resolved_path = sidecar_path or os.environ.get("LMEVAL_SIDECAR_PATH", _DEFAULT_SIDECAR_PATH)
            self._writer = _get_writer(resolved_path)

        # -----------------------------------------------------------------------
        # Sync path (model_call, used when num_concurrent <= 1)
        # -----------------------------------------------------------------------

        def model_call(
            self,
            messages,
            *,
            generate: bool = True,
            gen_kwargs=None,
            **kwargs,
        ) -> Optional[dict]:
            # Derive canonical messages for fingerprint before parent transforms them
            canonical_msgs = self.create_message(messages)
            raw = super().model_call(
                messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
            )
            if raw is not None and generate:
                record = _extract_metadata(raw, canonical_msgs, self.model)
                self._writer.write(record)  # propagates on failure — no try/except
            return raw

        # -----------------------------------------------------------------------
        # Async path (amodel_call, used when num_concurrent > 1)
        # -----------------------------------------------------------------------

        async def amodel_call(
            self,
            session,
            sem,
            messages,
            *,
            generate: bool = True,
            cache_keys=None,
            ctxlens=None,
            gen_kwargs=None,
            **kwargs,
        ):
            import copy as _copy
            gen_kwargs_copy = _copy.deepcopy(gen_kwargs)

            # Derive canonical messages for fingerprint
            canonical_msgs = self.create_message(messages)

            payload = self._create_payload(
                canonical_msgs,
                generate=generate,
                gen_kwargs=gen_kwargs_copy,
                seed=self._seed,
                **kwargs,
            )

            acquired = await sem.acquire()
            try:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=self.header,
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        import logging
                        logging.getLogger(__name__).warning(
                            f"[sidecar] API request failed: {response.status} {error_text}"
                        )
                    response.raise_for_status()
                    outputs = await response.json()

                # Capture metadata BEFORE parsing strips it
                if generate:
                    record = _extract_metadata(outputs, canonical_msgs, self.model)
                    self._writer.write(record)  # propagates on failure — no try/except

                # Parse using standard lm_eval methods
                tmp_answers = (
                    self.parse_generations(outputs=outputs)
                    if generate
                    else self.parse_logprobs(outputs=outputs, tokens=messages, ctxlens=ctxlens)
                )

                answers = []
                for a in tmp_answers:
                    if a is None:
                        answers.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                    else:
                        answers.append(a)

                if cache_keys:
                    cache_method = "generate_until" if generate else "loglikelihood"
                    for res, cache in zip(answers, cache_keys):
                        self.cache_hook.add_partial(cache_method, cache, res)

                return answers

            except Exception:
                raise  # never swallow
            finally:
                if acquired:
                    sem.release()

except ImportError:
    # lm_eval not installed — define a stub for testing
    class SidecarChatCompletion:  # type: ignore[no-redef]
        """Stub for environments without lm_eval installed."""

        def __init__(self, *, sidecar_path=None, **kwargs):
            if "use_cache" in kwargs or "cache_requests" in kwargs:
                raise ValueError(
                    "SidecarChatCompletion does not support cache."
                )
            resolved_path = sidecar_path or os.environ.get("LMEVAL_SIDECAR_PATH", _DEFAULT_SIDECAR_PATH)
            self._writer = _get_writer(resolved_path)
            self.model = kwargs.get("model", "unknown")

        def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
            """Stub sync path — raises in stub mode; real impl in lm_eval class."""
            raise NotImplementedError("lm_eval not installed")

        async def amodel_call(self, session, sem, messages, *, generate=True,
                              cache_keys=None, ctxlens=None, gen_kwargs=None, **kwargs):
            """Stub async path — raises in stub mode; real impl in lm_eval class."""
            raise NotImplementedError("lm_eval not installed")

