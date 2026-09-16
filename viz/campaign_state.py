"""
viz/campaign_state.py — Transactional campaign state machine.

Public API:
    apply_transition(status, new_state, timestamp, lifecycle=None) -> dict
    write_status(dest, status, run_dir=None) -> None

Exceptions:
    InvalidTransitionError   — illegal state transition
    InvalidTimestampError    — malformed or non-UTC timestamp
    UnsafePathError          — symlink escapes run directory

State machine (append-only history):

    planned -> preflight_passed -> running -> completed -> validated -> published
                                   \\-> failed

Rules:
    - Transitions are forward-only; skips and rewrites are rejected.
    - failed is legal only from 'running'.
    - Terminal states ('published', 'failed') cannot be transitioned further.
    - 'published' requires lifecycle == 'current'.
    - A failed execution defaults to lifecycle='invalid'; caller may explicitly
      request lifecycle='diagnostic', but 'current' is never valid for a failed run.
    - Timestamps must be monotonically non-decreasing (new timestamp >= last history ts).
    - History tail (history[-1].state) must match execution_state before transition.
    - lifecycle override for non-failure transitions must be a valid lifecycle value.
    - apply_transition never mutates the input dict.
    - write_status writes atomically: sibling temp + flush/fsync + os.replace.
    - write_status refuses symlinks that resolve outside run_dir (when given).
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import re
import tempfile
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised when a state transition is illegal."""


class InvalidTimestampError(Exception):
    """Raised when a timestamp is malformed or not UTC."""


class UnsafePathError(Exception):
    """Raised when a path resolves outside the expected directory."""


# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------

# Ordered execution states (terminal nodes are at the ends of branches)
_ORDERED_STATES = [
    "planned",
    "preflight_passed",
    "running",
    "completed",
    "validated",
    "published",
]

# The only legal branch off the main chain
_FAILURE_FROM = "running"
_TERMINAL_STATES = frozenset({"published", "failed"})

# Legal (from_state, to_state) pairs
_LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    # Forward chain pairs
    [(_ORDERED_STATES[i], _ORDERED_STATES[i + 1]) for i in range(len(_ORDERED_STATES) - 1)]
    # Failure branch
    + [(_FAILURE_FROM, "failed")]
)

# Lifecycle states acceptable as explicit override for a failed run
_FAILED_LIFECYCLE_OVERRIDES = frozenset({"invalid", "diagnostic"})

# All valid lifecycle values (for non-failure override validation)
_VALID_LIFECYCLES = frozenset({
    "current", "historical", "superseded", "diagnostic", "replay", "invalid"
})

# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

_UTC_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)


def _validate_utc_timestamp(ts: str) -> None:
    """Raise InvalidTimestampError if *ts* is not a valid UTC ISO-8601 string.

    We require the 'Z' suffix (no offset notation) to enforce UTC-only policy.
    This also catches naive timestamps (no TZ info at all).
    """
    if not isinstance(ts, str) or not _UTC_ISO_RE.match(ts):
        raise InvalidTimestampError(
            f"Timestamp must be an ISO 8601 UTC string ending in 'Z' "
            f"(e.g. '2026-09-15T12:00:00Z'); got: {ts!r}"
        )
    # Double-check by parsing; this catches bad calendar values like 2026-02-30T00:00:00Z
    try:
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise InvalidTimestampError(f"Invalid UTC timestamp {ts!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Core state machine
# ---------------------------------------------------------------------------

def apply_transition(
    status: dict,
    new_state: str,
    timestamp: str,
    lifecycle: Optional[str] = None,
) -> dict:
    """Apply a state transition to *status* and return a new dict.

    The input *status* dict is never mutated.

    Parameters
    ----------
    status:
        Current campaign status document (must conform to result-status schema).
    new_state:
        Target execution state.
    timestamp:
        UTC ISO-8601 timestamp string (e.g. '2026-09-15T12:00:00Z').
    lifecycle:
        Optional explicit lifecycle override for terminal transitions.
        For 'failed' transitions, only 'invalid' and 'diagnostic' are accepted;
        'current' is never valid after a failure. For non-failed transitions,
        if provided it must be a valid lifecycle value.

    Returns
    -------
    dict
        A new status dict with the transition applied (history appended,
        execution_state updated, lifecycle updated if changed).

    Raises
    ------
    InvalidTimestampError
        If *timestamp* is malformed or not UTC.
    InvalidTransitionError
        If the transition is illegal (skipped state, rewrite, terminal, etc.).
    """
    _validate_utc_timestamp(timestamp)

    current_state = status.get("execution_state", "")
    current_lifecycle = status.get("lifecycle", "current")
    history = status.get("history", [])

    if not isinstance(history, list) or not history:
        raise InvalidTransitionError("Inconsistent status: history must be a non-empty list")

    # Validate the complete append-only chain, not only its tail.
    previous_state = None
    previous_ts = None
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise InvalidTransitionError(f"Invalid history entry at index {index}")
        state = entry.get("state", "")
        entry_ts = entry.get("timestamp", "")
        _validate_utc_timestamp(entry_ts)
        if index == 0 and state != "planned":
            raise InvalidTransitionError("Invalid history: first state must be 'planned'")
        if previous_state is not None and (previous_state, state) not in _LEGAL_TRANSITIONS:
            raise InvalidTransitionError(
                f"Invalid history transition at index {index}: {previous_state!r} -> {state!r}"
            )
        if previous_ts is not None and entry_ts < previous_ts:
            raise InvalidTimestampError(
                f"Non-monotonic history timestamp at index {index}: {entry_ts!r} "
                f"is earlier than {previous_ts!r}"
            )
        previous_state, previous_ts = state, entry_ts

    if previous_state != current_state:
        raise InvalidTransitionError(
            f"Inconsistent status: execution_state={current_state!r} but "
            f"history tail state={previous_state!r}. History tail must match execution_state."
        )
    if timestamp < previous_ts:
        raise InvalidTimestampError(
            f"Non-monotonic timestamp: new timestamp {timestamp!r} is earlier than "
            f"last history timestamp {previous_ts!r}. Timestamps must be non-decreasing."
        )

    # Terminal state check — nothing may follow 'published' or 'failed'
    if current_state in _TERMINAL_STATES:
        raise InvalidTransitionError(
            f"Cannot transition from terminal state {current_state!r} to {new_state!r}. "
            "Terminal states 'published' and 'failed' end the campaign lifecycle."
        )

    # Legal transition check
    if (current_state, new_state) not in _LEGAL_TRANSITIONS:
        raise InvalidTransitionError(
            f"Illegal transition: {current_state!r} -> {new_state!r}. "
            f"Legal transitions from {current_state!r}: "
            f"{sorted(t for (f, t) in _LEGAL_TRANSITIONS if f == current_state)}"
        )

    # Publication requires lifecycle == 'current'
    if new_state == "published" and current_lifecycle != "current":
        raise InvalidTransitionError(
            f"Cannot publish: lifecycle is {current_lifecycle!r} but must be 'current'. "
            "Only validated current runs may be published."
        )

    # Determine the new lifecycle value
    if new_state == "failed":
        # Default to 'invalid'; allow only 'invalid' or 'diagnostic' as override
        if lifecycle is None:
            new_lifecycle = "invalid"
        elif lifecycle in _FAILED_LIFECYCLE_OVERRIDES:
            new_lifecycle = lifecycle
        else:
            raise InvalidTransitionError(
                f"Invalid lifecycle override {lifecycle!r} for a failed execution. "
                f"Only {sorted(_FAILED_LIFECYCLE_OVERRIDES)} are accepted. "
                "'current' is never valid after failure."
            )
    else:
        # Non-failure transition: keep existing lifecycle unless caller provides one
        if lifecycle is not None and lifecycle not in _VALID_LIFECYCLES:
            raise InvalidTransitionError(
                f"Invalid lifecycle override {lifecycle!r} for transition to {new_state!r}. "
                f"Valid lifecycle values: {sorted(_VALID_LIFECYCLES)}"
            )
        new_lifecycle = lifecycle if lifecycle is not None else current_lifecycle

    # Build new entry for history
    new_entry: dict = {"state": new_state, "timestamp": timestamp}

    # Deep-copy to avoid mutating the caller's dict
    new_status = copy.deepcopy(status)
    new_status["execution_state"] = new_state
    new_status["lifecycle"] = new_lifecycle
    new_status["history"] = list(new_status["history"]) + [new_entry]

    return new_status


# ---------------------------------------------------------------------------
# Atomic status write
# ---------------------------------------------------------------------------

def write_status(
    dest: pathlib.Path,
    status: dict,
    run_dir: Optional[pathlib.Path] = None,
) -> None:
    """Write *status* to *dest* atomically.

    Uses a sibling temporary file + os.fdatasync/os.fsync + os.replace to
    ensure the destination is never half-written.

    Parameters
    ----------
    dest:
        Destination path for status.json.
    status:
        Campaign status dict.
    run_dir:
        If provided, the resolved path of *dest* must be inside *run_dir*.
        This catches symlinks that escape the run directory.

    Raises
    ------
    UnsafePathError
        If *dest* resolves outside *run_dir* (symlink attack).
    """
    dest = pathlib.Path(dest)

    # Symlink safety: if run_dir is given, verify dest resolves inside it
    if run_dir is not None:
        run_dir = pathlib.Path(run_dir).resolve()
        try:
            resolved = dest.resolve()
        except OSError:
            # dest may not exist yet; use parent resolution + name
            resolved = dest.parent.resolve() / dest.name

        try:
            resolved.relative_to(run_dir)
        except ValueError:
            raise UnsafePathError(
                f"Destination {dest} resolves to {resolved}, which is outside "
                f"run_dir {run_dir}. Refusing to write."
            )

    # Also check if the file is already a symlink (pre-existing symlink to outside)
    if dest.is_symlink():
        if run_dir is not None:
            try:
                dest.resolve().relative_to(run_dir)
            except ValueError:
                raise UnsafePathError(
                    f"{dest} is a symlink pointing outside run_dir {run_dir}."
                )
        # If no run_dir, still refuse writing to arbitrary symlinks for safety
        else:
            raise UnsafePathError(
                f"{dest} is a symlink. Refusing atomic write to a symlink without run_dir context."
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(status, indent=2, ensure_ascii=False) + "\n"

    # Write to a sibling temp file, fsync, then atomically rename
    fd, tmp_path = tempfile.mkstemp(
        dir=dest.parent,
        prefix=".tmp_status_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
    except BaseException:
        # Clean up the temp file if anything went wrong before the rename
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
