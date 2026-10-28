"""Repo-wide guard (2026-10-27): no source, test or script file - or directory on its path - is named by a date.

CODE scope over the WORKING TREE (no project-git reads; docs and root files are checked from the tracked list by
``naming_policy.py paths --tracked`` in CI and pre-push).  One detector, one traversal: scripts/naming_policy.py.
Bug class: draft "Date used as an artifact's identifier" (naming_runs/bc_draft.md).  Local-only handoffs/plans under
docs/ and ignored runtime dirs are deliberately out of this guard's scope.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tests.unit.test_test_file_naming_guard import APPROVED_EXCEPTIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("naming_policy", REPO_ROOT / "scripts" / "naming_policy.py")
np_ = importlib.util.module_from_spec(_SPEC)
sys.modules["naming_policy"] = np_
_SPEC.loader.exec_module(np_)


def test_real_tree_has_no_dated_source_names():
    bad = np_.check_tree(REPO_ROOT)
    assert bad == [], "date-named artifacts (name by content; see scripts/naming_policy.py):\n" + "\n".join(
        f"  {s}: {r}" for s, r in bad
    )


def test_real_tree_has_no_unclassified_top_level_source_dir():
    assert np_.unclassified_top_level_dirs(REPO_ROOT) == []


def test_guard_fails_on_a_planted_dated_file_and_directory(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "fix_oct27.py").write_text("")
    (tmp_path / "utils" / "sep30_pkg").mkdir(parents=True)
    (tmp_path / "utils" / "sep30_pkg" / "mod.py").write_text("")
    assert {s for s, _ in np_.check_tree(tmp_path)} == {"core/fix_oct27.py", "utils/sep30_pkg/mod.py"}


def test_guard_ignores_a_planted_file_under_data(tmp_path):
    for runtime_dir in sorted(np_.SKIP_TOP_LEVEL):
        (tmp_path / runtime_dir).mkdir()
        (tmp_path / runtime_dir / "backup_20261027.json").write_text("{}")
    (tmp_path / "core").mkdir()
    assert np_.check_tree(tmp_path) == []


def test_the_approved_exception_matches_the_tests_guard_and_exists():
    assert np_.APPROVED_PATHS == APPROVED_EXCEPTIONS
    for rel in APPROVED_EXCEPTIONS:
        assert (REPO_ROOT / rel).is_file()
