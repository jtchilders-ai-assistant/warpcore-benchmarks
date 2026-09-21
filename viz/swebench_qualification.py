#!/usr/bin/env python3
"""viz/swebench_qualification.py — the executable SWE-bench qualification gate.

WHY THIS FILE EXISTS
--------------------
External cron/shell logic once promoted a gpt-oss smoke whose terminal status was
``RepeatedFormatError``.  The repository documented what a qualification meant but
could not enforce it, so nothing refused the promotion.  This module makes the
refusal repository policy: a canonical SWE-bench campaign may launch only when a
fresh, exactly-bound qualification record proves the production path produced 20
clean submissions that the official grader terminally dispositioned.

It is the single authoritative implementation.  ``viz/run_swebench.py`` consults it
before a launch and reuses :func:`check_raw_evidence_policy` at the end of a
qualification run; ``viz/contract.py`` reuses :func:`validate_suite_qualification_config`
for suite validation; ``make qualify-swebench``, ``make verify-swebench-qualification``,
and ``make ci`` run it.  Nothing re-implements a weaker copy.

The record is produced by ``run_swebench.py --qualification-run`` (``make
run-swebench-qualification``): the production runner with the suite-owned twenty
substituted for the frozen hundred.  That mode is deliberately not gated — it is what
produces the authorization — and writes no campaign state.

WHAT IT IS NOT
--------------
It is not campaign execution state.  ``status.json`` records what a run *did*;
this record states what a serving profile is *permitted* to start.  It is also
not a bridge, parser repair, retry, sanitizer, or scaffold change: it only reads
evidence and refuses.

QUALIFICATION POLICY (canonical launch)
---------------------------------------
1.  Production runner / config builder path only — the recorded
    ``production_scaffold_hash`` must equal what :func:`build_production_scaffold_config`
    produces now, and ``launch_path`` must name the production entry point.
2.  Exactly 20 frozen qualification IDs, supplied by suite-owned config
    (``suite/swebench/qualification-ids-v1.json``), never adapter- or CLI-selected.
3.  Every one terminal ``Submitted``.
4.  No RepeatedFormatError, RuntimeError, parser, transport, server, or other
    infrastructure disposition anywhere.
5.  Every ``preds.json`` ``model_patch`` nonempty and syntactically patch-like.
6.  Every trajectory present, nonempty, and free of malformed tool-call arguments.
7.  Every ID exactly once in the official SWE-bench terminal grading dispositions.
    Resolved OR unresolved is acceptable; a grading infrastructure error is not.
8.  Zero foreign, duplicate, or missing IDs anywhere in the evidence.
9.  Fresh, and every digest still matching at campaign launch.

EXIT CODES (CLI)
----------------
    0  qualified / valid
    1  diagnosed defect — do not launch
    2  usage error
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

_VIZ_DIR = pathlib.Path(__file__).resolve().parent
_REPO_DIR = _VIZ_DIR.parent
for _p in (str(_VIZ_DIR), str(_REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contract import load_yaml, sha256_file, validate_json  # noqa: E402

# ---------------------------------------------------------------------------
# Repository policy constants
# ---------------------------------------------------------------------------

QUALIFICATION_SCHEMA_VERSION = 1
QUALIFICATION_SCHEMA_PATH = _REPO_DIR / "suite" / "schemas" / "swebench-qualification.schema.json"

#: Repository policy, not a suite-tunable knob: a canonical qualification is
#: exactly twenty instances.  The suite names *which* twenty; it cannot shrink
#: the gate.
REQUIRED_QUALIFICATION_COUNT = 20

#: The only terminal generation disposition a qualification instance may carry.
REQUIRED_TERMINAL_STATUS = "Submitted"

#: The production launch path.  A qualification produced by anything else —
#: a shell smoke, a notebook, a hand-edited config — is not a qualification.
PRODUCTION_LAUNCH_PATH = "viz/run_swebench.py::build_production_scaffold_config"

#: Official SWE-bench schema-v2 terminal dispositions that count as a graded
#: outcome.  Resolved or unresolved both qualify: the gate proves the harness
#: reached a verdict, not that the model was right.
TERMINAL_GRADING_CATEGORIES = ("resolved_ids", "unresolved_ids")

#: Official categories that mean the harness never reached a verdict.
NON_TERMINAL_GRADING_CATEGORIES = ("empty_patch_ids", "error_ids", "incomplete_ids")

_ALL_GRADING_CATEGORIES = TERMINAL_GRADING_CATEGORIES + NON_TERMINAL_GRADING_CATEGORIES

#: Fallback freshness window when the suite does not declare one.
DEFAULT_MAX_AGE_HOURS = 24

_API_KEY_REDACTION = "__REDACTED__"
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Forbidden disposition classes.  Membership is by lowercase substring because
#: harness names drift across versions (``APIConnectionError`` today,
#: ``APIConnectionErrorRetryable`` tomorrow) and a gate that only knows exact
#: spellings silently reopens on the next upgrade.
_FORBIDDEN_DISPOSITION_CLASSES = (
    ("repeated_format_error", ("repeatedformaterror",)),
    ("format_error", ("formaterror", "format_error")),
    ("runtime_error", ("runtimeerror", "runtime_error")),
    ("parser", ("parse", "harmony")),
    ("transport", ("connection", "transport", "socket", "ssl", "eof", "broken pipe")),
    ("server", ("servererror", "internalserver", "badgateway", "serviceunavailable",
                "unavailable", "5xx", "apierror", "apistatus")),
    ("infrastructure", ("infra", "docker", "timeout", "timedout", "oom",
                        "network", "diskspace", "container")),
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class QualificationResult:
    """Outcome of the launch gate.  ``ok`` is the only thing that may authorize."""

    ok: bool
    errors: list = field(default_factory=list)
    artifact: Optional[dict] = None
    summary: str = ""

    def render(self) -> str:
        """Return a human-readable block suitable for a runner log or dry-run."""
        if self.ok:
            return f"QUALIFICATION: OK — {self.summary}"
        lines = [f"QUALIFICATION: BLOCKED — {self.summary or 'not qualified'}"]
        lines.extend(f"  - {e}" for e in self.errors)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic suite-owned qualification set
# ---------------------------------------------------------------------------


def _repository_of(instance_id: str) -> str:
    """Return the SWE-bench repository slug of *instance_id* (text before the last '-')."""
    return instance_id.rsplit("-", 1)[0]


def derive_qualification_instance_ids(
    frozen_ids: Iterable[str],
    count: int = REQUIRED_QUALIFICATION_COUNT,
) -> list:
    """Derive the qualification subset deterministically from the frozen n=100 set.

    Criterion (documented in the design and the RUNBOOK):

      1. Group the frozen seed-42 instance IDs by repository — the substring
         before the final ``-``.
      2. Order repositories lexicographically; within a repository keep the
         frozen-set order.
      3. Take instances round-robin, one per repository per round, until *count*
         are selected.
      4. Emit them in frozen-set order, so the file reads as a subsequence of
         the frozen set.

    Round 1 therefore covers every repository present in the frozen set, and
    later rounds deepen coverage in lexicographic order.  The result is a pure
    function of the frozen set: no adapter, CLI flag, or operator preference can
    steer it toward convenient cases.
    """
    frozen = list(frozen_ids)
    groups: dict = {}
    for iid in frozen:
        groups.setdefault(_repository_of(iid), []).append(iid)

    selected: list = []
    round_index = 0
    repos = sorted(groups)
    while len(selected) < count:
        progressed = False
        for repo in repos:
            bucket = groups[repo]
            if len(bucket) > round_index:
                selected.append(bucket[round_index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break  # frozen set smaller than count
        round_index += 1

    chosen = set(selected)
    return [iid for iid in frozen if iid in chosen]


# ---------------------------------------------------------------------------
# Production scaffold builder — the single definition of "the production path"
# ---------------------------------------------------------------------------


def build_production_scaffold_config(
    *,
    scaffold: dict,
    model_id: str,
    endpoint: str,
    api_key: str,
) -> dict:
    """Return the frozen scaffold with only the adapter model identity injected.

    This is the one and only production config builder.
    ``run_swebench.SwebenchRunner.build_scaffold_config`` delegates here so that a
    qualification and the campaign it authorizes cannot diverge.

    Injects exactly three fields:
      * ``model.model_name``               -> ``hosted_vllm/<model_id>``
      * ``model.model_kwargs.api_base``    -> endpoint
      * ``model.model_kwargs.api_key``     -> api_key

    Every suite-owned experiment control (step_limit, cost_limit,
    environment.timeout, pull_timeout, temperature, max_tokens, submit protocol)
    is preserved byte-for-byte from the frozen scaffold.
    """
    config = copy.deepcopy(scaffold)
    config.setdefault("model", {})
    config["model"]["model_name"] = f"hosted_vllm/{model_id}"
    config["model"].setdefault("model_kwargs", {})
    config["model"]["model_kwargs"]["api_base"] = endpoint
    config["model"]["model_kwargs"]["api_key"] = api_key
    return config


def production_scaffold_hash(config: dict) -> str:
    """Return ``sha256:<hex>`` over *config* with the API key redacted.

    Redaction keeps the digest stable across key rotation and keeps a secret out
    of a committed artifact, while every experiment control and the endpoint
    still participate in the identity.
    """
    redacted = copy.deepcopy(config)
    model = redacted.get("model")
    if isinstance(model, dict):
        kwargs = model.get("model_kwargs")
        if isinstance(kwargs, dict) and "api_key" in kwargs:
            kwargs["api_key"] = _API_KEY_REDACTION
    canonical = json.dumps(redacted, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Evidence predicates
# ---------------------------------------------------------------------------


def classify_forbidden_disposition(status: Any) -> Optional[str]:
    """Return the forbidden class name for *status*, or None when it is not forbidden.

    Matching is lowercase-substring so harness renames do not silently reopen the
    gate.  ``Submitted`` is never forbidden; everything else is still rejected by
    the required-terminal-status rule even when it matches no class here.
    """
    if not isinstance(status, str):
        return "non_string_disposition"
    needle = status.strip().lower()
    if needle == REQUIRED_TERMINAL_STATUS.lower():
        return None
    for class_name, markers in _FORBIDDEN_DISPOSITION_CLASSES:
        for marker in markers:
            if marker in needle:
                return class_name
    return None


def is_patch_like(text: Any) -> bool:
    """Return True when *text* is a nonempty, syntactically patch-like unified diff.

    Syntactic only — the gate never repairs, reformats, or re-derives a patch.
    A diff needs a file header (``diff --git`` or a ``---``/``+++`` pair) and at
    least one hunk header, which is exactly what distinguishes a real submission
    from an apology or a truncated fragment.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    lines = stripped.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    has_git_header = any(line.startswith("diff --git ") for line in lines)
    has_unified_header = (
        any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
    )
    return has_hunk and (has_git_header or has_unified_header)


def scan_tool_calls(obj: Any) -> tuple:
    """Walk *obj* and return ``(well_formed_count, problems)`` for every tool call.

    Retained response objects are nested differently across mini-swe-agent
    versions (``messages``, ``trajectory``, ``info.messages``, raw provider
    payloads), so the walk is structural rather than path-based: any mapping with
    a ``tool_calls`` list is inspected wherever it appears.

    A call is well-formed when its ``function.arguments`` is a nonempty string
    that parses as a JSON object.  Truncated, empty, non-string, or non-object
    arguments are exactly the RepeatedFormatError signature and are reported.
    """
    problems: list = []
    count = 0

    def walk(node: Any, path: str) -> None:
        nonlocal count
        if isinstance(node, dict):
            calls = node.get("tool_calls")
            if isinstance(calls, list):
                for idx, call in enumerate(calls):
                    here = f"{path}.tool_calls[{idx}]"
                    if not isinstance(call, dict):
                        problems.append(f"{here}: tool call is not an object")
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict):
                        problems.append(f"{here}: missing function object")
                        continue
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        problems.append(
                            f"{here}: arguments is {type(args).__name__}, expected a JSON string"
                        )
                        continue
                    if not args.strip():
                        problems.append(f"{here}: arguments is empty")
                        continue
                    try:
                        parsed = json.loads(args)
                    except json.JSONDecodeError as exc:
                        problems.append(f"{here}: arguments is not valid JSON ({exc.msg})")
                        continue
                    if not isinstance(parsed, dict):
                        problems.append(
                            f"{here}: arguments parsed to {type(parsed).__name__}, expected an object"
                        )
                        continue
                    count += 1
            for key, value in node.items():
                if key != "tool_calls":
                    walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                walk(item, f"{path}[{idx}]")

    walk(obj, "$")
    return count, problems


# ---------------------------------------------------------------------------
# Suite-owned configuration access
# ---------------------------------------------------------------------------


def _swebench_block(suite: dict) -> dict:
    return ((suite.get("benchmarks") or {}).get("swebench") or {})


def load_qualification_config(suite_path) -> dict:
    """Return the suite-owned ``benchmarks.swebench.qualification`` block."""
    suite = load_yaml(pathlib.Path(suite_path))
    return _swebench_block(suite).get("qualification") or {}


def qualification_max_age_hours(suite_path) -> int:
    """Return the suite-owned freshness window in hours."""
    value = load_qualification_config(suite_path).get("max_age_hours", DEFAULT_MAX_AGE_HOURS)
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_AGE_HOURS


def suite_qualification_ids(repo, suite_path) -> list:
    """Return the suite-owned frozen qualification instance IDs."""
    cfg = load_qualification_config(suite_path)
    ids_file = cfg.get("ids_file", "")
    if not ids_file:
        raise ValueError("suite swebench.qualification is missing 'ids_file'")
    path = (pathlib.Path(repo) / ids_file).resolve()
    return json.loads(path.read_text())


def validate_suite_qualification_config(repo, bench: dict) -> list:
    """Validate the suite's qualification block.  Returns [] or a list of errors.

    Called by ``contract.validate_suite`` so suite drift is a CI failure, exactly
    like task-file drift.  This is the only place the suite-owned qualification
    set is checked, so a convenience set can never enter through a second door.
    """
    errors: list = []
    repo = pathlib.Path(repo).resolve()
    cfg = bench.get("qualification")
    if not isinstance(cfg, dict) or not cfg:
        return [
            "swebench: missing 'qualification' block — a canonical SWE-bench suite must "
            "name its suite-owned qualification instance set"
        ]

    ids_file = cfg.get("ids_file", "")
    if not ids_file:
        return ["swebench: qualification.ids_file is missing"]

    ids_path = (repo / ids_file).resolve()
    try:
        ids_path.relative_to(repo)
    except ValueError:
        return [f"swebench: qualification.ids_file escapes repo root: {ids_file!r}"]
    if not ids_path.is_file():
        return [f"swebench: qualification.ids_file not found: {ids_path}"]

    declared_hash = cfg.get("ids_sha256", "")
    actual_hash = sha256_file(ids_path)
    if declared_hash and declared_hash != actual_hash:
        errors.append(
            f"swebench: qualification.ids_sha256 mismatch — declared {declared_hash!r}, "
            f"actual {actual_hash!r} ({ids_file})"
        )

    try:
        qual_ids = json.loads(ids_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [f"swebench: cannot parse qualification.ids_file: {exc}"]

    if not isinstance(qual_ids, list) or not all(isinstance(i, str) and i for i in qual_ids):
        return errors + ["swebench: qualification.ids_file must be a JSON array of nonempty strings"]

    expected_count = cfg.get("expected_count")
    if expected_count != REQUIRED_QUALIFICATION_COUNT:
        errors.append(
            f"swebench: qualification.expected_count must be "
            f"{REQUIRED_QUALIFICATION_COUNT}; got {expected_count!r}"
        )
    if len(qual_ids) != REQUIRED_QUALIFICATION_COUNT:
        errors.append(
            f"swebench: qualification set holds {len(qual_ids)} IDs but repository policy "
            f"requires exactly {REQUIRED_QUALIFICATION_COUNT}"
        )
    if len(set(qual_ids)) != len(qual_ids):
        errors.append("swebench: qualification set contains duplicate instance IDs")

    required_status = cfg.get("required_terminal_status")
    if required_status != REQUIRED_TERMINAL_STATUS:
        errors.append(
            f"swebench: qualification.required_terminal_status must be "
            f"{REQUIRED_TERMINAL_STATUS!r}; got {required_status!r}"
        )

    # Membership in, and derivability from, the frozen n=100 set.
    instance_set_file = bench.get("instance_set_file", "")
    frozen_path = (repo / instance_set_file).resolve() if instance_set_file else None
    if frozen_path is not None and frozen_path.is_file():
        try:
            frozen = json.loads(frozen_path.read_text())
        except (OSError, json.JSONDecodeError):
            frozen = None
        # A malformed frozen set (non-string or unhashable elements) is already a
        # diagnosed defect reported by the instance-set checks above. Skip the
        # membership and derivation comparisons rather than crashing on it —
        # exit 1 with a clear message, never a TypeError traceback.
        if isinstance(frozen, list) and all(isinstance(i, str) and i for i in frozen):
            foreign = [i for i in qual_ids if i not in set(frozen)]
            if foreign:
                errors.append(
                    "swebench: qualification set contains IDs that are not in the frozen "
                    f"instance set: {sorted(foreign)[:5]}"
                )
            derived = derive_qualification_instance_ids(frozen)
            if qual_ids != derived:
                errors.append(
                    "swebench: qualification set does not match the deterministic "
                    "round-robin-by-repository derivation from the frozen instance set. "
                    "Qualification IDs are suite-owned and derived, never hand-picked."
                )

    return errors


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def resolve_repo_sha(repo) -> Optional[str]:
    """Return the git HEAD SHA of *repo*, or None when it cannot be established.

    None is not success: the gate treats an unresolvable SHA as a blocking defect,
    because a qualification that cannot be bound to a commit binds to nothing.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def serving_profile_digest(adapter: dict) -> str:
    """Return the deterministic serving-profile digest for *adapter*.

    Delegates to the existing implementation in ``viz/create_campaign.py`` so the
    qualification record and the run manifest can never disagree about what a
    serving profile is.  Imported lazily to keep module import order simple.
    """
    import create_campaign  # local import: avoids an import cycle at module load
    return create_campaign._serving_profile_digest(adapter)


def swebench_suite_input_hashes(repo, suite_path) -> dict:
    """Return the repo-relative SHA-256 map of every suite input the gate binds.

    Covers the suite YAML, the frozen instance set, the suite-owned qualification
    set, and the scaffold.  The map is compared as a whole at launch: an extra or
    missing entry is drift, not a rounding error.
    """
    repo = pathlib.Path(repo).resolve()
    suite_path = pathlib.Path(suite_path).resolve()
    suite = load_yaml(suite_path)
    bench = _swebench_block(suite)
    cfg = bench.get("qualification") or {}

    hashes: dict = {}
    rels = [str(suite_path.relative_to(repo))]
    for rel in (
        bench.get("instance_set_file"),
        bench.get("scaffold_file"),
        cfg.get("ids_file"),
    ):
        if rel:
            rels.append(rel)

    for rel in rels:
        path = (repo / rel).resolve()
        path.relative_to(repo)  # containment; raises on escape
        hashes[rel] = sha256_file(path)
    return hashes


def default_artifact_path(root, suite_id: str, model_slug: str) -> pathlib.Path:
    """Return the canonical qualification artifact location under *root*."""
    return (
        pathlib.Path(root) / "results" / model_slug / "qualification"
        / suite_id / "swebench" / "qualification.json"
    )


# ---------------------------------------------------------------------------
# Safe relative-path resolution inside the qualification directory
# ---------------------------------------------------------------------------


def _resolve_inside(base: pathlib.Path, rel: str, label: str, errors: list):
    """Resolve *rel* under *base*, refusing absolute paths and traversal."""
    if not isinstance(rel, str) or not rel:
        errors.append(f"evidence.{label} is missing")
        return None
    candidate = pathlib.PurePosixPath(rel)
    if candidate.is_absolute() or rel.startswith("/") or ".." in candidate.parts:
        errors.append(
            f"evidence.{label}={rel!r} is absolute or escapes the qualification "
            "directory; evidence must live beside its artifact"
        )
        return None
    resolved = (base / rel).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        errors.append(
            f"evidence.{label}={rel!r} resolves outside the qualification directory "
            f"({resolved})"
        )
        return None
    return resolved


def _read_json_file(path: pathlib.Path, label: str, errors: list):
    if not path.is_file():
        errors.append(f"{label} not found at {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is not readable JSON: {exc}")
        return None


def _check_digest(path: pathlib.Path, declared: Any, label: str, errors: list) -> bool:
    if not path.is_file():
        errors.append(f"{label} not found at {path}")
        return False
    actual = sha256_file(path)
    if actual != declared:
        errors.append(
            f"{label} sha256 mismatch — artifact records {str(declared)[:16]}…, "
            f"file is {actual[:16]}…; the evidence changed after qualification"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Evidence policy checks
# ---------------------------------------------------------------------------


def _check_dispositions(statuses: Any, expected_ids: list, errors: list) -> dict:
    """Validate terminal generation dispositions.  Returns the parsed map."""
    if not isinstance(statuses, dict):
        errors.append(
            f"exit_statuses.json must be an object keyed by instance_id; "
            f"got {type(statuses).__name__}"
        )
        return {}

    expected = set(expected_ids)
    present = set(statuses)
    missing = expected - present
    foreign = present - expected
    if missing:
        errors.append(
            f"exit_statuses.json is missing {len(missing)} qualification instance(s): "
            f"{sorted(missing)[:5]}"
        )
    if foreign:
        errors.append(
            f"exit_statuses.json holds {len(foreign)} foreign instance ID(s) not in the "
            f"suite-owned qualification set: {sorted(foreign)[:5]}"
        )

    for iid in expected_ids:
        if iid not in statuses:
            continue
        status = statuses[iid]
        forbidden_class = classify_forbidden_disposition(status)
        if forbidden_class is not None:
            errors.append(
                f"{iid}: forbidden {forbidden_class} disposition {status!r} — a "
                f"{status!r} qualification instance can never authorize a canonical launch"
            )
            continue
        if status != REQUIRED_TERMINAL_STATUS:
            errors.append(
                f"{iid}: terminal disposition {status!r} is not the required "
                f"{REQUIRED_TERMINAL_STATUS!r}"
            )
    return statuses


def _check_predictions(preds: Any, expected_ids: list, errors: list) -> None:
    if not isinstance(preds, dict):
        errors.append(f"preds.json must be an object keyed by instance_id; got {type(preds).__name__}")
        return

    expected = set(expected_ids)
    present = set(preds)
    missing = expected - present
    foreign = present - expected
    if missing:
        errors.append(
            f"preds.json is missing {len(missing)} qualification instance(s): {sorted(missing)[:5]}"
        )
    if foreign:
        errors.append(
            f"preds.json holds {len(foreign)} foreign instance ID(s) not in the suite-owned "
            f"qualification set: {sorted(foreign)[:5]}"
        )

    for iid in expected_ids:
        entry = preds.get(iid)
        if entry is None:
            continue
        patch = entry.get("model_patch") if isinstance(entry, dict) else None
        if not isinstance(patch, str) or not patch.strip():
            errors.append(f"{iid}: preds.json model_patch is empty or absent")
            continue
        if not is_patch_like(patch):
            errors.append(
                f"{iid}: preds.json model_patch is not syntactically patch-like "
                "(needs a diff header and at least one @@ hunk)"
            )


def _check_trajectories(traj_dir: pathlib.Path, expected_ids: list, errors: list) -> None:
    if not traj_dir.is_dir():
        errors.append(f"trajectory directory not found at {traj_dir}")
        return
    for iid in expected_ids:
        path = traj_dir / f"{iid}.traj"
        if not path.is_file():
            errors.append(f"{iid}: trajectory evidence missing at {path}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"{iid}: trajectory evidence is empty at {path}")
            continue
        try:
            traj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{iid}: trajectory is not readable JSON: {exc}")
            continue

        info = traj.get("info") if isinstance(traj, dict) else None
        if isinstance(info, dict) and "exit_status" in info:
            status = info.get("exit_status")
            forbidden_class = classify_forbidden_disposition(status)
            if forbidden_class is not None:
                errors.append(
                    f"{iid}: trajectory records a forbidden {forbidden_class} exit status "
                    f"{status!r}"
                )
                continue
            if status != REQUIRED_TERMINAL_STATUS:
                errors.append(
                    f"{iid}: trajectory exit status {status!r} is not the required "
                    f"{REQUIRED_TERMINAL_STATUS!r}"
                )
                continue

        count, problems = scan_tool_calls(traj)
        if problems:
            errors.append(
                f"{iid}: malformed tool-call arguments in retained response objects: "
                f"{problems[:3]}"
            )
            continue
        if count == 0:
            errors.append(
                f"{iid}: trajectory retains no tool call — a Submitted instance must show "
                "at least one well-formed tool call"
            )


def _check_official_grading(report: Any, expected_ids: list, errors: list) -> dict:
    """Validate the official SWE-bench grader report.  Returns normalized categories."""
    normalized = {key: [] for key in _ALL_GRADING_CATEGORIES}
    if not isinstance(report, dict):
        errors.append(
            f"official grader report must be a JSON object; got {type(report).__name__}"
        )
        return normalized

    for key in _ALL_GRADING_CATEGORIES:
        value = report.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            errors.append(f"official grader report {key} must be a list; got {type(value).__name__}")
            return normalized
        normalized[key] = list(value)

    if not any(key in report for key in TERMINAL_GRADING_CATEGORIES):
        errors.append(
            "official grader report carries no resolved_ids/unresolved_ids keys — this is "
            "not an official SWE-bench schema-v2 report"
        )
        return normalized

    for key in NON_TERMINAL_GRADING_CATEGORIES:
        if normalized[key]:
            errors.append(
                f"official grading reports {len(normalized[key])} instance(s) in {key}: "
                f"{sorted(normalized[key])[:5]} — grading infrastructure error is not an "
                "acceptable qualification outcome"
            )

    seen: dict = {}
    for key in _ALL_GRADING_CATEGORIES:
        for iid in normalized[key]:
            seen[iid] = seen.get(iid, 0) + 1
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    if duplicates:
        errors.append(
            f"official grading holds duplicate dispositions for {len(duplicates)} "
            f"instance(s): {duplicates[:5]}"
        )

    expected = set(expected_ids)
    foreign = sorted(set(seen) - expected)
    if foreign:
        errors.append(
            f"official grading holds {len(foreign)} foreign instance ID(s): {foreign[:5]}"
        )

    terminal = set()
    for key in TERMINAL_GRADING_CATEGORIES:
        terminal.update(normalized[key])
    missing = sorted(expected - terminal)
    if missing:
        errors.append(
            f"{len(missing)} qualification instance(s) have no terminal official grading "
            f"disposition: {missing[:5]}"
        )

    return normalized


def check_raw_evidence_policy(raw_dir, expected_ids: list) -> list:
    """Apply the qualification evidence policy to a freshly produced ``raw/`` directory.

    Same predicates the launch gate uses, minus the digest and identity binding —
    those only mean something once a record has been sealed.  The qualification
    runner calls this the moment grading finishes, so a run that produced a
    ``RepeatedFormatError`` reports it there rather than at sealing time.

    Returns [] when the evidence satisfies the policy, else diagnosed errors.
    """
    raw_dir = pathlib.Path(raw_dir)
    errors: list = []
    ids = list(expected_ids)

    preds = _read_json_file(raw_dir / "preds.json", "preds.json", errors)
    if preds is not None:
        _check_predictions(preds, ids, errors)

    statuses = _read_json_file(raw_dir / "exit_statuses.json", "exit_statuses.json", errors)
    if statuses is not None:
        _check_dispositions(statuses, ids, errors)

    _check_trajectories(raw_dir / "trajectories", ids, errors)

    try:
        report_path = _discover_grader_report(raw_dir)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        report = _read_json_file(report_path, "official grader report", errors)
        if report is not None:
            _check_official_grading(report, ids, errors)

    return errors


# ---------------------------------------------------------------------------
# Artifact construction
# ---------------------------------------------------------------------------


def _utc(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _discover_grader_report(raw_dir: pathlib.Path) -> pathlib.Path:
    known = {"preds.json", "exit_statuses.json", "grading_results.json"}
    candidates = [p for p in sorted(raw_dir.glob("*.json")) if p.name not in known]
    if not candidates:
        raise ValueError(
            f"No official SWE-bench grader report found in {raw_dir}. The harness writes "
            "<model>.<run_id>.json under --report_dir; qualification requires it."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous official grader report in {raw_dir}: {[p.name for p in candidates]}. "
            "Exactly one report must be retained."
        )
    return candidates[0]


def build_qualification_artifact(
    *,
    repo,
    suite_path,
    adapter_path,
    endpoint: str,
    served_model_id: str,
    evidence_run_dir,
    artifact_path,
    api_key: str = "warpcore",
    now: Optional[datetime] = None,
    repo_sha: Optional[str] = None,
    qualification_id: Optional[str] = None,
) -> dict:
    """Build (but do not validate or write) the qualification record.

    The caller — normally the ``emit`` CLI — must verify the result before
    writing it; ``emit`` refuses to write an artifact that would not pass.
    """
    repo = pathlib.Path(repo).resolve()
    suite_path = pathlib.Path(suite_path).resolve()
    adapter_path = pathlib.Path(adapter_path).resolve()
    evidence_run_dir = pathlib.Path(evidence_run_dir).resolve()
    artifact_path = pathlib.Path(artifact_path).resolve()

    suite = load_yaml(suite_path)
    adapter = load_yaml(adapter_path)
    bench = _swebench_block(suite)
    cfg = bench.get("qualification") or {}

    resolved_sha = repo_sha or resolve_repo_sha(repo)
    if not resolved_sha:
        raise ValueError(
            f"Cannot resolve the git HEAD of {repo}; a qualification must bind to a commit."
        )

    qual_ids_rel = cfg.get("ids_file", "")
    qual_ids_path = (repo / qual_ids_rel).resolve()
    qual_ids = json.loads(qual_ids_path.read_text())

    scaffold_rel = bench.get("scaffold_file", "")
    scaffold_path = (repo / scaffold_rel).resolve()
    scaffold = load_yaml(scaffold_path)

    model = adapter.get("model") or {}
    model_id = model.get("id", "")

    built_config = build_production_scaffold_config(
        scaffold=scaffold, model_id=model_id, endpoint=endpoint, api_key=api_key
    )

    raw_dir = evidence_run_dir / "raw"
    preds_path = raw_dir / "preds.json"
    statuses_path = raw_dir / "exit_statuses.json"
    report_path = _discover_grader_report(raw_dir)

    artifact_dir = artifact_path.parent
    try:
        run_rel = evidence_run_dir.relative_to(artifact_dir).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Evidence directory {evidence_run_dir} must live beside the artifact "
            f"({artifact_dir}) so the qualification is self-contained."
        ) from exc

    statuses = json.loads(statuses_path.read_text())
    report = json.loads(report_path.read_text())
    grading = {key: list(report.get(key) or []) for key in _ALL_GRADING_CATEGORIES}

    slug = model.get("slug", "unknown")
    generated = _utc(now)
    return {
        "qualification_schema_version": QUALIFICATION_SCHEMA_VERSION,
        "qualification_id": qualification_id or f"qual-{slug}-{generated.replace(':', '')}",
        "generated_utc": generated,
        "repo_sha": resolved_sha,
        "suite_id": suite.get("suite_id", ""),
        "suite_schema_version": suite.get("suite_schema_version", 1),
        "suite_input_hashes": swebench_suite_input_hashes(repo, suite_path),
        "adapter_hash": sha256_file(adapter_path),
        "serving_profile_digest": serving_profile_digest(adapter),
        "model": {
            "slug": slug,
            "id": model_id,
            "revision": model.get("revision", ""),
        },
        "endpoint": {"base_url": endpoint, "served_model_id": served_model_id},
        "scaffold_sha256": sha256_file(scaffold_path),
        "production_scaffold_hash": production_scaffold_hash(built_config),
        "launch_path": PRODUCTION_LAUNCH_PATH,
        "qualification_instance_ids": list(qual_ids),
        "qualification_ids_sha256": sha256_file(qual_ids_path),
        "evidence": {
            "run_dir": run_rel,
            "preds_file": preds_path.relative_to(evidence_run_dir).as_posix(),
            "preds_sha256": sha256_file(preds_path),
            "exit_statuses_file": statuses_path.relative_to(evidence_run_dir).as_posix(),
            "exit_statuses_sha256": sha256_file(statuses_path),
            "grader_report_file": report_path.relative_to(evidence_run_dir).as_posix(),
            "grader_report_sha256": sha256_file(report_path),
            "trajectories_dir": "raw/trajectories",
            "grader": {
                "harness": "swebench.harness.run_evaluation",
                "version": str((suite.get("required_harness") or {}).get("swebench_version", "")),
            },
        },
        "dispositions": {iid: str(statuses.get(iid, "")) for iid in qual_ids},
        "grading": grading,
    }


# ---------------------------------------------------------------------------
# The launch gate
# ---------------------------------------------------------------------------


def verify_qualification_for_launch(
    *,
    repo,
    suite_path,
    adapter_path,
    endpoint: str,
    artifact_path,
    api_key: str = "warpcore",
    now: Optional[datetime] = None,
    repo_sha: Optional[str] = None,
) -> QualificationResult:
    """Decide whether a canonical SWE-bench campaign may launch.  Fail-closed.

    Every failure mode returns ``ok=False`` with diagnosed errors: a missing,
    stale, malformed, or foreign-bound record blocks launch exactly as a
    forbidden disposition does.  There is no inconclusive success.
    """
    errors: list = []
    repo = pathlib.Path(repo).resolve()
    suite_path = pathlib.Path(suite_path).resolve()
    adapter_path = pathlib.Path(adapter_path).resolve()
    artifact_path = pathlib.Path(artifact_path).resolve()

    # -- 1. Artifact readable -------------------------------------------------
    if not artifact_path.is_file():
        return QualificationResult(
            ok=False,
            errors=[
                f"qualification record not found at {artifact_path}. Run "
                "`make qualify-swebench` against a completed 20-instance qualification run."
            ],
            summary="no qualification record",
        )
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return QualificationResult(
            ok=False,
            errors=[f"qualification record at {artifact_path} is not readable JSON: {exc}"],
            summary="malformed qualification record",
        )

    # -- 2. Schema -----------------------------------------------------------
    if not QUALIFICATION_SCHEMA_PATH.is_file():
        return QualificationResult(
            ok=False,
            errors=[f"qualification schema not found at {QUALIFICATION_SCHEMA_PATH}"],
            summary="schema unavailable",
        )
    schema_errors = validate_json(artifact, QUALIFICATION_SCHEMA_PATH)
    if schema_errors:
        return QualificationResult(
            ok=False,
            errors=[f"qualification record fails schema validation: {e}" for e in schema_errors],
            artifact=artifact,
            summary="malformed qualification record",
        )

    # -- 3. Expected launch identity -----------------------------------------
    try:
        suite = load_yaml(suite_path)
        adapter = load_yaml(adapter_path)
    except Exception as exc:  # unreadable suite/adapter is a blocking defect
        return QualificationResult(
            ok=False,
            errors=[f"cannot load suite or adapter for qualification binding: {exc}"],
            artifact=artifact,
            summary="unreadable launch context",
        )

    bench = _swebench_block(suite)
    cfg = bench.get("qualification") or {}
    if not cfg:
        errors.append(
            "suite declares no swebench.qualification block; a suite without an owned "
            "qualification set cannot authorize a canonical launch"
        )

    expected_sha = repo_sha or resolve_repo_sha(repo)
    if not expected_sha:
        errors.append(
            f"cannot resolve the git HEAD of the repo at {repo}; a qualification that "
            "cannot be bound to a commit cannot authorize a launch"
        )
    elif artifact.get("repo_sha") != expected_sha:
        errors.append(
            f"repo_sha mismatch — qualification was sealed at {artifact.get('repo_sha')}, "
            f"launch is at {expected_sha}; re-qualify on this commit"
        )

    if artifact.get("suite_id") != suite.get("suite_id"):
        errors.append(
            f"suite_id mismatch — qualification is for {artifact.get('suite_id')!r}, "
            f"launch uses {suite.get('suite_id')!r}"
        )
    if artifact.get("suite_schema_version") != suite.get("suite_schema_version"):
        errors.append(
            f"suite_schema_version mismatch — qualification "
            f"{artifact.get('suite_schema_version')!r} vs suite "
            f"{suite.get('suite_schema_version')!r}"
        )

    try:
        expected_hashes = swebench_suite_input_hashes(repo, suite_path)
    except Exception as exc:
        expected_hashes = None
        errors.append(f"cannot hash suite inputs for qualification binding: {exc}")
    if expected_hashes is not None and artifact.get("suite_input_hashes") != expected_hashes:
        differing = sorted(
            set(expected_hashes) ^ set(artifact.get("suite_input_hashes") or {})
        ) or [
            k for k, v in expected_hashes.items()
            if (artifact.get("suite_input_hashes") or {}).get(k) != v
        ]
        errors.append(
            f"suite_input_hashes mismatch — suite inputs changed since qualification: "
            f"{differing[:5]}"
        )

    expected_adapter_hash = sha256_file(adapter_path)
    if artifact.get("adapter_hash") != expected_adapter_hash:
        errors.append(
            f"adapter_hash mismatch — qualification {str(artifact.get('adapter_hash'))[:16]}…, "
            f"launch adapter {expected_adapter_hash[:16]}…"
        )

    try:
        expected_profile = serving_profile_digest(adapter)
    except Exception as exc:
        expected_profile = None
        errors.append(f"cannot compute the serving-profile digest: {exc}")
    if expected_profile is not None and artifact.get("serving_profile_digest") != expected_profile:
        errors.append(
            f"serving_profile_digest mismatch — qualification "
            f"{str(artifact.get('serving_profile_digest'))[:23]}…, launch {expected_profile[:23]}…"
        )

    adapter_model = adapter.get("model") or {}
    art_model = artifact.get("model") or {}
    for field_name in ("slug", "id", "revision"):
        if art_model.get(field_name) != adapter_model.get(field_name):
            errors.append(
                f"model.{field_name} mismatch — qualification "
                f"{art_model.get(field_name)!r}, launch adapter {adapter_model.get(field_name)!r}"
            )

    art_endpoint = artifact.get("endpoint") or {}
    if art_endpoint.get("base_url") != endpoint:
        errors.append(
            f"endpoint mismatch — qualification ran against "
            f"{art_endpoint.get('base_url')!r}, launch targets {endpoint!r}"
        )
    if art_endpoint.get("served_model_id") != adapter_model.get("id"):
        errors.append(
            f"endpoint served_model_id mismatch — qualification observed "
            f"{art_endpoint.get('served_model_id')!r} at /v1/models, adapter declares "
            f"{adapter_model.get('id')!r}"
        )

    # -- 4. Scaffold and production path -------------------------------------
    scaffold_rel = bench.get("scaffold_file", "")
    scaffold_path = (repo / scaffold_rel).resolve() if scaffold_rel else None
    if scaffold_path is None or not scaffold_path.is_file():
        errors.append(f"suite scaffold_file not found: {scaffold_rel!r}")
    else:
        expected_scaffold_hash = sha256_file(scaffold_path)
        if artifact.get("scaffold_sha256") != expected_scaffold_hash:
            errors.append(
                f"scaffold_sha256 mismatch — qualification "
                f"{str(artifact.get('scaffold_sha256'))[:16]}…, production scaffold "
                f"{expected_scaffold_hash[:16]}…"
            )
        built = build_production_scaffold_config(
            scaffold=load_yaml(scaffold_path),
            model_id=adapter_model.get("id", ""),
            endpoint=endpoint,
            api_key=api_key,
        )
        expected_prod_hash = production_scaffold_hash(built)
        if artifact.get("production_scaffold_hash") != expected_prod_hash:
            errors.append(
                f"production_scaffold_hash mismatch — the qualification did not run the "
                f"config this launch will use (qualification "
                f"{str(artifact.get('production_scaffold_hash'))[:23]}…, launch "
                f"{expected_prod_hash[:23]}…)"
            )

    if artifact.get("launch_path") != PRODUCTION_LAUNCH_PATH:
        errors.append(
            f"launch_path {artifact.get('launch_path')!r} is not the production path "
            f"{PRODUCTION_LAUNCH_PATH!r}; only the production runner/config builder may qualify"
        )

    # -- 5. Suite-owned qualification instance set ----------------------------
    expected_ids: list = []
    ids_rel = cfg.get("ids_file", "")
    ids_path = (repo / ids_rel).resolve() if ids_rel else None
    if ids_path is None or not ids_path.is_file():
        errors.append(f"suite qualification ids_file not found: {ids_rel!r}")
    else:
        try:
            expected_ids = json.loads(ids_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read suite qualification ids_file: {exc}")
            expected_ids = []
        if artifact.get("qualification_ids_sha256") != sha256_file(ids_path):
            errors.append(
                "qualification_ids_sha256 mismatch — the suite-owned qualification set "
                "changed since this record was sealed"
            )
        if artifact.get("qualification_instance_ids") != expected_ids:
            errors.append(
                "qualification_instance_ids do not match the suite-owned qualification set; "
                "qualification instances are suite-owned and may not be adapter- or "
                "CLI-selected"
            )
        if len(expected_ids) != REQUIRED_QUALIFICATION_COUNT:
            errors.append(
                f"suite qualification set holds {len(expected_ids)} IDs; repository policy "
                f"requires exactly {REQUIRED_QUALIFICATION_COUNT}"
            )

    art_ids = artifact.get("qualification_instance_ids") or []
    if len(art_ids) != REQUIRED_QUALIFICATION_COUNT:
        errors.append(
            f"qualification record covers {len(art_ids)} instances; repository policy "
            f"requires exactly {REQUIRED_QUALIFICATION_COUNT}"
        )

    # -- 6. Freshness ---------------------------------------------------------
    now = now or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    generated_raw = artifact.get("generated_utc", "")
    if not _UTC_RE.match(str(generated_raw)):
        errors.append(
            f"generated_utc {generated_raw!r} is not a second-precision UTC instant "
            "(YYYY-MM-DDTHH:MM:SSZ)"
        )
    else:
        generated = datetime.strptime(generated_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        max_age = timedelta(hours=qualification_max_age_hours(suite_path))
        if generated > now + timedelta(minutes=1):
            errors.append(
                f"generated_utc {generated_raw} is in the future relative to {_utc(now)}; "
                "a qualification cannot pre-date its own launch clock"
            )
        elif now - generated > max_age:
            age_h = (now - generated).total_seconds() / 3600.0
            errors.append(
                f"qualification is stale: sealed {generated_raw} ({age_h:.1f} h ago), "
                f"suite freshness window is {max_age.total_seconds() / 3600:.0f} h"
            )

    # -- 7. Evidence ----------------------------------------------------------
    policy_ids = expected_ids or art_ids
    evidence = artifact.get("evidence") or {}
    artifact_dir = artifact_path.parent
    run_dir = _resolve_inside(artifact_dir, evidence.get("run_dir", ""), "run_dir", errors)

    statuses: dict = {}
    if run_dir is not None:
        if not run_dir.is_dir():
            errors.append(f"qualification evidence directory not found at {run_dir}")
        else:
            preds_path = _resolve_inside(run_dir, evidence.get("preds_file", ""), "preds_file", errors)
            statuses_path = _resolve_inside(
                run_dir, evidence.get("exit_statuses_file", ""), "exit_statuses_file", errors
            )
            report_path = _resolve_inside(
                run_dir, evidence.get("grader_report_file", ""), "grader_report_file", errors
            )
            traj_dir = _resolve_inside(
                run_dir, evidence.get("trajectories_dir", ""), "trajectories_dir", errors
            )

            if preds_path is not None and _check_digest(
                preds_path, evidence.get("preds_sha256"), "preds.json", errors
            ):
                _check_predictions(
                    _read_json_file(preds_path, "preds.json", errors), policy_ids, errors
                )
            if statuses_path is not None and _check_digest(
                statuses_path, evidence.get("exit_statuses_sha256"), "exit_statuses.json", errors
            ):
                parsed = _read_json_file(statuses_path, "exit_statuses.json", errors)
                if parsed is not None:
                    statuses = _check_dispositions(parsed, policy_ids, errors)
            if report_path is not None and _check_digest(
                report_path, evidence.get("grader_report_sha256"),
                "official grader report", errors,
            ):
                _check_official_grading(
                    _read_json_file(report_path, "official grader report", errors),
                    policy_ids, errors,
                )
            if traj_dir is not None:
                _check_trajectories(traj_dir, policy_ids, errors)

    # Grader identity must match the harness the suite pins.
    declared_grader_version = str(((evidence.get("grader") or {}).get("version", "")))
    suite_grader_version = str((suite.get("required_harness") or {}).get("swebench_version", ""))
    if suite_grader_version and declared_grader_version != suite_grader_version:
        errors.append(
            f"official grader version mismatch — qualification used "
            f"{declared_grader_version!r}, suite pins {suite_grader_version!r}"
        )

    # The artifact's own disposition copy must agree with the evidence: a
    # hand-edited record is a forgery, not a qualification.
    if statuses:
        recorded = artifact.get("dispositions") or {}
        disagreeing = sorted(
            iid for iid in policy_ids
            if str(recorded.get(iid, "")) != str(statuses.get(iid, ""))
        )
        if disagreeing:
            errors.append(
                f"recorded dispositions disagree with the evidence for "
                f"{len(disagreeing)} instance(s): {disagreeing[:5]}"
            )

    if errors:
        return QualificationResult(
            ok=False,
            errors=errors,
            artifact=artifact,
            summary=f"{len(errors)} blocking finding(s)",
        )

    return QualificationResult(
        ok=True,
        errors=[],
        artifact=artifact,
        summary=(
            f"{len(policy_ids)}/{REQUIRED_QUALIFICATION_COUNT} suite-owned instances terminal "
            f"{REQUIRED_TERMINAL_STATUS}, officially graded by "
            f"{(evidence.get('grader') or {}).get('harness', 'the SWE-bench harness')} "
            f"{declared_grader_version}, sealed {artifact.get('generated_utc')} at "
            f"repo {str(artifact.get('repo_sha'))[:12]}"
        ),
    )


# ---------------------------------------------------------------------------
# Offline self-test (no network, no fixtures)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    failures: list = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    frozen_path = _REPO_DIR / "suite" / "swebench" / "instances-seed42-n100.json"
    ids_path = _REPO_DIR / "suite" / "swebench" / "qualification-ids-v1.json"
    frozen = json.loads(frozen_path.read_text())
    committed = json.loads(ids_path.read_text())
    derived = derive_qualification_instance_ids(frozen)

    check(len(committed) == REQUIRED_QUALIFICATION_COUNT, "qualification set is not exactly 20")
    check(committed == derived, "committed qualification set != deterministic derivation")
    check(set(committed) <= set(frozen), "qualification set is not a subset of the frozen set")
    check(
        {_repository_of(i) for i in derived} == {_repository_of(i) for i in frozen},
        "derivation does not cover every repository",
    )

    check(classify_forbidden_disposition("RepeatedFormatError") is not None, "RepeatedFormatError not forbidden")
    check(classify_forbidden_disposition("RuntimeError") is not None, "RuntimeError not forbidden")
    check(classify_forbidden_disposition("InternalServerError") is not None, "server error not forbidden")
    check(classify_forbidden_disposition("APIConnectionError") is not None, "transport error not forbidden")
    check(classify_forbidden_disposition("Submitted") is None, "Submitted wrongly forbidden")

    check(is_patch_like("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"), "valid diff rejected")
    check(not is_patch_like(""), "empty patch accepted")
    check(not is_patch_like("sorry, I could not fix this"), "prose accepted as a patch")
    check(not is_patch_like("diff --git a/x b/x\n"), "hunkless diff accepted")

    good = {"tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    bad = {"tool_calls": [{"function": {"name": "bash", "arguments": '{"command":'}}]}
    check(scan_tool_calls(good) == (1, []), "well-formed tool call not recognized")
    check(bool(scan_tool_calls(bad)[1]), "malformed tool call not detected")

    check(QUALIFICATION_SCHEMA_PATH.is_file(), "qualification schema file missing")

    if failures:
        for f in failures:
            print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
        return 1
    print("swebench_qualification self-test OK "
          f"({REQUIRED_QUALIFICATION_COUNT} suite-owned IDs, forbidden-class classifier, "
          "patch and tool-call scanners)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return _self_test()

    ap = argparse.ArgumentParser(
        description="SWE-bench qualification gate (warpcore-v1)."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--artifact", required=True, type=pathlib.Path)
        p.add_argument("--suite", required=True, type=pathlib.Path)
        p.add_argument("--adapter", required=True, type=pathlib.Path)
        p.add_argument("--endpoint", required=True)
        p.add_argument("--repo", type=pathlib.Path, default=_REPO_DIR)
        p.add_argument("--api-key", default="warpcore")

    verify_p = sub.add_parser("verify", help="Verify a qualification record against a launch context.")
    add_common(verify_p)

    emit_p = sub.add_parser(
        "emit",
        help="Build and write a qualification record from a completed qualification run. "
             "Refuses to write a record that would not pass verification.",
    )
    add_common(emit_p)
    emit_p.add_argument("--evidence", required=True, type=pathlib.Path,
                        help="Qualification run directory (must sit beside --artifact).")
    emit_p.add_argument("--served-model-id", required=True,
                        help="Exact model id returned by GET <endpoint>/models during qualification.")

    args = ap.parse_args(argv)

    if args.command == "emit":
        try:
            artifact = build_qualification_artifact(
                repo=args.repo,
                suite_path=args.suite,
                adapter_path=args.adapter,
                endpoint=args.endpoint,
                served_model_id=args.served_model_id,
                evidence_run_dir=args.evidence,
                artifact_path=args.artifact,
                api_key=args.api_key,
            )
        except Exception as exc:
            print(f"ERROR: cannot build qualification record: {exc}", file=sys.stderr)
            return 1
        # Write to a scratch sibling, verify it, and only then publish it.
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        scratch = args.artifact.with_name(args.artifact.name + ".candidate")
        scratch.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        result = verify_qualification_for_launch(
            repo=args.repo, suite_path=args.suite, adapter_path=args.adapter,
            endpoint=args.endpoint, artifact_path=scratch, api_key=args.api_key,
        )
        scratch.unlink(missing_ok=True)
        if not result.ok:
            print(result.render(), file=sys.stderr)
            print("Refusing to write a qualification record that does not qualify.", file=sys.stderr)
            return 1
        args.artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(result.render())
        print(f"Wrote {args.artifact}")
        return 0

    result = verify_qualification_for_launch(
        repo=args.repo, suite_path=args.suite, adapter_path=args.adapter,
        endpoint=args.endpoint, artifact_path=args.artifact, api_key=args.api_key,
    )
    print(result.render(), file=sys.stderr if not result.ok else sys.stdout)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
