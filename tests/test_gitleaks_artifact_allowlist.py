"""Regression tests for exact reviewed gitleaks artifact allowlists."""
from __future__ import annotations

import pathlib
import subprocess
import sys
from unittest import mock

_TESTS_DIR = pathlib.Path(__file__).parent
_VIZ_DIR = _TESTS_DIR.parent / "viz"
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import validate_campaign  # noqa: E402


def test_secret_scan_passes_repo_exact_ignore_file(tmp_path):
    """The validator must pass the repo's exact fingerprint ignore to gitleaks."""
    run_dir = tmp_path / "repo" / "results" / "model" / "runs" / "suite" / "bench" / "run"
    run_dir.mkdir(parents=True)
    ignore_path = tmp_path / "repo" / ".gitleaksignore"
    ignore_path.write_text("artifact.json:generic-api-key:123\n")
    completed = subprocess.CompletedProcess(args=["gitleaks"], returncode=0, stdout="", stderr="")

    with mock.patch.object(validate_campaign, "_resolve_gitleaks_bin", return_value="/usr/bin/gitleaks"), \
         mock.patch.object(subprocess, "run", return_value=completed) as run:
        errors: list[str] = []
        validate_campaign._secret_scan(run_dir, errors)

    assert errors == []
    argv = run.call_args.args[0]
    assert argv == [
        "/usr/bin/gitleaks", "dir", "--redact", "--exit-code", "1",
        "--gitleaks-ignore-path", str(ignore_path),
        "results/model/runs/suite/bench/run",
    ]
    assert run.call_args.kwargs["cwd"] == tmp_path / "repo"


def test_secret_scan_does_not_inherit_ignore_outside_repo(tmp_path):
    """A parent-directory ignore file must not weaken a repository scan."""
    run_dir = tmp_path / "repo" / "results" / "model" / "runs" / "suite" / "bench" / "run"
    run_dir.mkdir(parents=True)
    (tmp_path / ".gitleaksignore").write_text("artifact.json:generic-api-key:123\n")
    completed = subprocess.CompletedProcess(args=["gitleaks"], returncode=0, stdout="", stderr="")

    with mock.patch.object(validate_campaign, "_resolve_gitleaks_bin", return_value="/usr/bin/gitleaks"), \
         mock.patch.object(subprocess, "run", return_value=completed) as run:
        errors: list[str] = []
        validate_campaign._secret_scan(run_dir, errors)

    assert errors == []
    assert "--gitleaks-ignore-path" not in run.call_args.args[0]
    assert run.call_args.kwargs["cwd"] is None
