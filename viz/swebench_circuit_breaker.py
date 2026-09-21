#!/usr/bin/env python3
"""viz/swebench_circuit_breaker.py — fail-closed SWE-bench campaign abort control.

WHY THIS EXISTS
---------------
A SWE-bench campaign is ~100 instances of GPU-hours. Twice now the whole budget
was spent proving something the first few instances already proved:

  * 2026-09-17 Qwen: every instance recorded RuntimeError (unmapped LiteLLM cost
    metadata). mini-swe-agent still exited 0 with complete file coverage.
  * 2026-08-04 Qwen: 22/100 lost to a 120 s docker pull timeout on a cold cache.
    The model was never invoked on those instances.

This module watches the live mini-swe-agent progress artifact while generation is
still running and reports a Decision as soon as the evidence already in hand
demonstrates a systemic parser/transport/server/infrastructure failure.

WHAT IT MUST NOT DO
-------------------
Model and config operational outcomes — step limit, cost limit, context window,
a submitted-but-wrong patch, a plain format mistake — are the model's own result.
Design §10.2 keeps them in the full denominator, so they never trip the breaker.
`RepeatedFormatError` is the sharp edge here: viz/swebench_fair.py counts it as
infrastructure for denominator attribution, but that is a post-hoc judgement over
a whole campaign. To kill a live campaign this module demands trajectory evidence
of actual parser/tool-argument corruption, because "the model formatted an action
wrongly" and "the tool-call parser is broken" produce the same exit status.

EVIDENCE SAFETY
---------------
mini-swe-agent rewrites exit_statuses_<timestamp>.yaml as the campaign advances,
so any read can land mid-write. A half-written file must never be mistaken for
evidence, in either direction:

  * a snapshot is accepted only when two consecutive reads are byte-identical and
    the parsed structure is well formed; otherwise nothing is reported;
  * observations are reconciled monotonically, so a truncated rewrite can never
    erase already-observed healthy completions and manufacture a majority.

Every failure to read, parse or classify resolves to "no decision" — the campaign
keeps running. Fail-closed here means the *campaign* is failed and marked invalid
when the breaker does trip; it never means aborting on ambiguous evidence.

This module makes no process, state or artifact decisions of its own. The runner
(viz/run_swebench.py) owns termination, lifecycle transition and artifact writes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_VIZ_DIR = pathlib.Path(__file__).resolve().parent

POLICY_RELPATH = "suite/swebench/circuit_breaker_policy.yaml"
POLICY_SCHEMA_RELPATH = "suite/schemas/swebench-circuit-breaker-policy.schema.json"

# Classification labels recorded in circuit_breaker.json.
SYSTEMIC_INFRASTRUCTURE = "systemic_infrastructure"
SYSTEMIC_PARSER = "systemic_parser_corruption"
OPERATIONAL_MODEL = "operational_model_outcome"
UNPROVEN_FORMAT_ERROR = "format_error_without_parser_evidence"
UNCLASSIFIED = "unclassified"

RULE_FIRST_COMPLETION = "first_completion_systemic"
RULE_SYSTEMIC_MAJORITY = "systemic_majority"


class PolicyError(Exception):
    """The suite-owned policy is missing, unreadable, or does not validate."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    """The suite-owned policy, content-addressed by the bytes it was loaded from."""

    policy_id: str
    policy_version: int
    suite_id: str
    benchmark: str
    enabled: bool
    first_completion_systemic: bool
    min_completed: int
    ratio_threshold: float
    systemic_exit_statuses: frozenset
    conditional_systemic_exit_statuses: Dict[str, str]
    operational_exit_statuses: frozenset
    parser_corruption_markers: tuple
    malformed_tool_call_arguments_is_proof: bool
    poll_interval_s: float
    stable_read_attempts: int
    stable_read_settle_s: float
    termination_scope: str
    termination_signal: str
    grace_period_s: float
    escalate_signal: str
    policy_path: pathlib.Path
    policy_sha256: str

    def as_provenance(self) -> dict:
        """The policy identity recorded in circuit_breaker.json."""
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_file": POLICY_RELPATH,
            "policy_sha256": self.policy_sha256,
        }


def load_policy(repo: pathlib.Path) -> CircuitBreakerPolicy:
    """Load and validate the suite-owned policy from *repo*.

    Fail-closed: a missing, unreadable, or schema-invalid policy raises
    PolicyError rather than falling back to a built-in default. A campaign must
    never run under an unverifiable abort control.
    """
    import yaml

    repo = pathlib.Path(repo).resolve()
    path = repo / POLICY_RELPATH
    if not path.is_file():
        raise PolicyError(
            f"SWE-bench circuit-breaker policy not found at {path}. "
            "The policy is suite-owned and required; it has no built-in default."
        )

    raw = path.read_bytes()
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise PolicyError(f"Circuit-breaker policy {path} is not readable YAML: {exc}") from exc

    if not isinstance(doc, dict):
        raise PolicyError(
            f"Circuit-breaker policy {path} must be a mapping; "
            f"got {type(doc).__name__}."
        )

    schema_path = repo / POLICY_SCHEMA_RELPATH
    if not schema_path.is_file():
        # Tests and ad-hoc repos may carry the policy without the schema; the real
        # repo always has both and `make contract` enforces it.
        schema_path = _VIZ_DIR.parent / POLICY_SCHEMA_RELPATH
    if not schema_path.is_file():
        raise PolicyError(
            f"Circuit-breaker policy schema not found at {schema_path}. "
            "The policy cannot be validated and must not be trusted."
        )

    from contract import validate_json  # local import: viz sibling module

    schema_errors = validate_json(doc, schema_path)
    if schema_errors:
        raise PolicyError(
            f"Circuit-breaker policy {path} fails schema validation:\n"
            + "\n".join(f"  {e}" for e in schema_errors)
        )

    rules = doc["rules"]
    majority = rules["systemic_majority"]
    classification = doc["classification"]
    proof = doc["trajectory_parser_corruption"]
    monitor = doc["monitor"]
    termination = doc["termination"]

    conditional = dict(classification["conditional_systemic_exit_statuses"] or {})
    systemic = frozenset(classification["systemic_exit_statuses"])
    operational = frozenset(classification["operational_exit_statuses"])

    # An exit status in two buckets would make classification order-dependent.
    for name, bucket in (("systemic", systemic), ("conditional", frozenset(conditional))):
        overlap = bucket & operational
        if overlap:
            raise PolicyError(
                f"Circuit-breaker policy {path}: exit statuses {sorted(overlap)} are "
                f"listed as both {name} and operational. Classification must be unambiguous."
            )
    overlap = systemic & frozenset(conditional)
    if overlap:
        raise PolicyError(
            f"Circuit-breaker policy {path}: exit statuses {sorted(overlap)} are listed "
            "as both unconditionally and conditionally systemic."
        )

    return CircuitBreakerPolicy(
        policy_id=doc["policy_id"],
        policy_version=int(doc["policy_version"]),
        suite_id=doc["suite_id"],
        benchmark=doc["benchmark"],
        enabled=bool(doc["enabled"]),
        first_completion_systemic=bool(rules["first_completion_systemic"]),
        min_completed=int(majority["min_completed"]),
        ratio_threshold=float(majority["ratio_threshold"]),
        systemic_exit_statuses=systemic,
        conditional_systemic_exit_statuses=conditional,
        operational_exit_statuses=operational,
        parser_corruption_markers=tuple(proof["markers"]),
        malformed_tool_call_arguments_is_proof=bool(
            proof["malformed_tool_call_arguments_is_proof"]
        ),
        poll_interval_s=float(monitor["poll_interval_s"]),
        stable_read_attempts=int(monitor["stable_read_attempts"]),
        stable_read_settle_s=float(monitor["stable_read_settle_s"]),
        termination_scope=termination["scope"],
        termination_signal=termination["signal"],
        grace_period_s=float(termination["grace_period_s"]),
        escalate_signal=termination["escalate_signal"],
        policy_path=path,
        policy_sha256=hashlib.sha256(raw).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Progress snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressSnapshot:
    """A stable read of one mini-swe-agent progress YAML."""

    source: pathlib.Path
    id_to_status: Dict[str, str]


def _parse_progress(text: str) -> Optional[Dict[str, str]]:
    """Return {instance_id: exit_status}, or None when the document is not sound.

    Anything other than a fully formed `instances_by_exit_status` mapping of
    status -> list-of-string-ids is treated as a partial write.
    """
    import yaml

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None

    by_status = doc.get("instances_by_exit_status")
    if not isinstance(by_status, dict) or not by_status:
        return None

    id_to_status: Dict[str, str] = {}
    for status, ids in by_status.items():
        if not isinstance(status, str) or not isinstance(ids, list):
            return None
        for iid in ids:
            if not isinstance(iid, str) or not iid:
                return None
            id_to_status[iid] = status
    if not id_to_status:
        return None
    return id_to_status


def read_stable_progress(
    raw_dir: pathlib.Path,
    *,
    policy: CircuitBreakerPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[ProgressSnapshot]:
    """Read the newest progress YAML, but only once it has stopped changing.

    mini-swe-agent rewrites the file as instances finish, so a single read can
    catch a truncated document. Two consecutive byte-identical reads are required
    before the content counts as evidence. Returns None when no progress file
    exists, when it is still changing, or when it does not parse into a sound
    `instances_by_exit_status` mapping — never a partial result.
    """
    raw_dir = pathlib.Path(raw_dir)
    previous: Optional[bytes] = None

    for attempt in range(max(2, policy.stable_read_attempts)):
        if attempt:
            sleep(policy.stable_read_settle_s)

        try:
            candidates = sorted(
                raw_dir.glob("exit_statuses_*.yaml"),
                key=lambda p: (p.stat().st_mtime_ns, p.name),
            )
        except OSError:
            return None
        if not candidates:
            return None
        latest = candidates[-1]

        try:
            current = latest.read_bytes()
        except OSError:
            previous = None
            continue

        if previous is not None and current == previous:
            try:
                text = current.decode("utf-8")
            except UnicodeDecodeError:
                return None
            id_to_status = _parse_progress(text)
            if id_to_status is None:
                return None
            return ProgressSnapshot(source=latest, id_to_status=id_to_status)

        previous = current

    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceClassification:
    instance_id: str
    exit_status: str
    classification: str
    systemic: bool
    evidence: str

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "exit_status": self.exit_status,
            "classification": self.classification,
            "systemic": self.systemic,
            "evidence": self.evidence,
        }


def _load_trajectory(raw_dir: pathlib.Path, instance_id: str) -> Optional[dict]:
    """Load an instance trajectory from either the live or normalized layout.

    A trajectory that is absent, unreadable, or still half-written parses to None
    and therefore proves nothing.
    """
    candidates = [
        raw_dir / instance_id / f"{instance_id}.traj.json",
        raw_dir / "trajectories" / f"{instance_id}.traj",
        raw_dir / "trajectories" / f"{instance_id}.traj.json",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            return doc
    return None


def _malformed_tool_call_arguments(node) -> Optional[str]:
    """Return the offending arguments string when a persisted tool call is corrupt.

    A tool call the harness actually recorded carries an `arguments` string that
    the parser was supposed to produce as JSON. When that string is present and
    does not parse, the parser emitted corruption — which is exactly the failure
    this breaker exists to catch, and is not something a well-formed model reply
    can cause on its own.
    """
    if isinstance(node, list):
        for item in node:
            found = _malformed_tool_call_arguments(item)
            if found is not None:
                return found
        return None
    if not isinstance(node, dict):
        return None

    function = node.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                return arguments

    for value in node.values():
        found = _malformed_tool_call_arguments(value)
        if found is not None:
            return found
    return None


def _find_marker(node, markers: tuple) -> Optional[str]:
    """Return the first policy marker string found anywhere in the trajectory."""
    if isinstance(node, str):
        for marker in markers:
            if marker in node:
                return marker
        return None
    if isinstance(node, list):
        for item in node:
            found = _find_marker(item, markers)
            if found is not None:
                return found
        return None
    if isinstance(node, dict):
        for value in node.values():
            found = _find_marker(value, markers)
            if found is not None:
                return found
    return None


def prove_parser_corruption(
    raw_dir: pathlib.Path,
    instance_id: str,
    policy: CircuitBreakerPolicy,
) -> Optional[str]:
    """Return a human-readable proof of parser corruption, or None.

    Two accepted proofs, both of which must be persisted in the trajectory:
      1. an explicit parser diagnostic listed in the policy markers;
      2. a recorded tool call whose captured arguments are not valid JSON.
    """
    trajectory = _load_trajectory(raw_dir, instance_id)
    if trajectory is None:
        return None

    marker = _find_marker(trajectory, policy.parser_corruption_markers)
    if marker is not None:
        return f"trajectory records parser diagnostic {marker!r}"

    if policy.malformed_tool_call_arguments_is_proof:
        arguments = _malformed_tool_call_arguments(trajectory)
        if arguments is not None:
            excerpt = arguments if len(arguments) <= 120 else arguments[:117] + "..."
            return f"trajectory persists malformed tool-call arguments JSON: {excerpt!r}"

    return None


def classify_instance(
    instance_id: str,
    exit_status: str,
    raw_dir: pathlib.Path,
    policy: CircuitBreakerPolicy,
) -> InstanceClassification:
    """Classify one completed instance against the suite-owned policy."""
    status = str(exit_status)

    if status in policy.operational_exit_statuses:
        return InstanceClassification(
            instance_id, status, OPERATIONAL_MODEL, False,
            "model/config operational outcome; stays in the full denominator",
        )

    if status in policy.systemic_exit_statuses:
        return InstanceClassification(
            instance_id, status, SYSTEMIC_INFRASTRUCTURE, True,
            "exit status is in the suite systemic parser/transport/server/"
            "infrastructure allowlist",
        )

    if status in policy.conditional_systemic_exit_statuses:
        proof = prove_parser_corruption(raw_dir, instance_id, policy)
        if proof is not None:
            return InstanceClassification(
                instance_id, status, SYSTEMIC_PARSER, True, proof,
            )
        return InstanceClassification(
            instance_id, status, UNPROVEN_FORMAT_ERROR, False,
            "no persisted trajectory evidence of parser/tool-argument corruption; "
            "treated as a model format outcome",
        )

    return InstanceClassification(
        instance_id, status, UNCLASSIFIED, False,
        "exit status is not in any suite classification list; reported, not assumed systemic",
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    rule: str
    reason: str
    completed_count: int
    systemic_count: int
    systemic_ratio: float
    first_completion_batch: List[str]
    classifications: List[InstanceClassification]
    policy: CircuitBreakerPolicy

    def observed(self) -> dict:
        return {
            "completed_count": self.completed_count,
            "systemic_count": self.systemic_count,
            "systemic_ratio": round(self.systemic_ratio, 6),
            "first_completion_batch": list(self.first_completion_batch),
            "instances": [c.as_dict() for c in self.classifications],
        }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


@dataclass
class BreakerMonitor:
    """Accumulates progress observations and reports the first trip decision.

    Stateful on purpose: it remembers the earliest observed completion batch (the
    only sound basis for "the first completed instance") and every instance it has
    ever seen, so a truncated rewrite of the progress YAML cannot shrink the
    denominator it reasons over.
    """

    raw_dir: pathlib.Path
    policy: CircuitBreakerPolicy
    sleep: Callable[[float], None] = time.sleep
    _id_to_status: Dict[str, str] = field(default_factory=dict, init=False)
    _first_batch: List[str] = field(default_factory=list, init=False)
    _classifications: List[InstanceClassification] = field(default_factory=list, init=False)

    def classifications(self) -> List[InstanceClassification]:
        """Classifications as of the last observation, in stable ID order."""
        return list(self._classifications)

    def observe(self) -> Optional[Decision]:
        """Read one stable snapshot and return a Decision when the policy trips."""
        if not self.policy.enabled:
            return None

        snapshot = read_stable_progress(
            self.raw_dir, policy=self.policy, sleep=self.sleep
        )
        if snapshot is None:
            return None

        # Monotonic reconciliation: later snapshots refresh a status but never
        # remove an instance we have already observed completing.
        if not self._first_batch:
            self._first_batch = sorted(snapshot.id_to_status)
        self._id_to_status.update(snapshot.id_to_status)

        self._classifications = [
            classify_instance(iid, self._id_to_status[iid], self.raw_dir, self.policy)
            for iid in sorted(self._id_to_status)
        ]
        return self._evaluate()

    def _evaluate(self) -> Optional[Decision]:
        by_id = {c.instance_id: c for c in self._classifications}
        completed = len(by_id)
        if not completed:
            return None
        systemic = [c for c in self._classifications if c.systemic]
        ratio = len(systemic) / completed

        def decide(rule: str, reason: str) -> Decision:
            return Decision(
                rule=rule,
                reason=reason,
                completed_count=completed,
                systemic_count=len(systemic),
                systemic_ratio=ratio,
                first_completion_batch=list(self._first_batch),
                classifications=list(self._classifications),
                policy=self.policy,
            )

        if self.policy.first_completion_systemic and self._first_batch:
            batch = [by_id[i] for i in self._first_batch if i in by_id]
            if batch and all(c.systemic for c in batch):
                labels = ", ".join(
                    f"{c.instance_id}={c.exit_status}" for c in batch
                )
                return decide(
                    RULE_FIRST_COMPLETION,
                    "The first completed instance was a systemic "
                    f"parser/transport/server/infrastructure failure ({labels}). "
                    "Continuing the campaign would measure the defect, not the model.",
                )

        if completed >= self.policy.min_completed and ratio > self.policy.ratio_threshold:
            return decide(
                RULE_SYSTEMIC_MAJORITY,
                f"{len(systemic)} of {completed} completed instances are systemic "
                f"failures ({ratio:.0%} > {self.policy.ratio_threshold:.0%} over at least "
                f"{self.policy.min_completed} completions). The remaining instances would "
                "measure the same defect.",
            )

        return None
