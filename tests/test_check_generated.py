"""Tests for generated-artifact freshness checks."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))

import check_generated  # noqa: E402


def test_changed_reports_content_and_missing_file_changes(tmp_path: Path) -> None:
    stable = tmp_path / "stable.txt"
    edited = tmp_path / "edited.txt"
    removed = tmp_path / "removed.txt"
    created = tmp_path / "created.txt"
    stable.write_text("same\n")
    edited.write_text("before\n")
    removed.write_text("present\n")

    before = check_generated.snapshot([stable, edited, removed, created])
    edited.write_text("after\n")
    removed.unlink()
    created.write_text("new\n")
    after = check_generated.snapshot([stable, edited, removed, created])

    assert check_generated.changed(before, after) == [edited, removed, created]
