"""Regression tests for bounded gitleaks campaign scans."""
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


def test_secret_scan_allows_realistic_bounded_runtime(tmp_path):
    completed = subprocess.CompletedProcess(args=["gitleaks"], returncode=0, stdout="", stderr="")

    with mock.patch.object(validate_campaign, "_resolve_gitleaks_bin", return_value="/usr/bin/gitleaks"), \
         mock.patch.object(subprocess, "run", return_value=completed) as run:
        errors: list[str] = []
        validate_campaign._secret_scan(tmp_path, errors)

    assert errors == []
    assert run.call_args.kwargs["timeout"] >= 120


def test_secret_scan_still_fails_closed_on_timeout(tmp_path):
    expired = subprocess.TimeoutExpired(cmd=["gitleaks"], timeout=120)

    with mock.patch.object(validate_campaign, "_resolve_gitleaks_bin", return_value="/usr/bin/gitleaks"), \
         mock.patch.object(subprocess, "run", side_effect=expired):
        errors: list[str] = []
        validate_campaign._secret_scan(tmp_path, errors)

    assert len(errors) == 1
    assert "could not be completed" in errors[0]
