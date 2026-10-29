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


def test_walk_skips_runtime_output_inside_code_dirs_but_not_its_neighbours(tmp_path):
    # Incident 2026-10-28: a model-bakeoff report under gitignored eval/runs/ failed this guard on the
    # owner's checkout while CI (no such folder) stayed green.  class: BC-66, BC-104
    for sub in np_.RUNTIME_OUTPUT_SUBTREES:
        (tmp_path / sub / "run_20261024_105613").mkdir(parents=True)
        (tmp_path / sub / "run_20261024_105613" / "report.py").write_text("")
    (tmp_path / "eval" / "sweep_oct27.py").write_text("")
    assert {s for s, _ in np_.check_tree(tmp_path)} == {"eval/sweep_oct27.py"}


def test_tracked_mode_still_checks_a_committed_file_under_runtime_output(tmp_path):
    committed = f"{np_.RUNTIME_OUTPUT_SUBTREES[0]}/fix_oct27.py"
    assert [s for s, _ in np_.check_tree(tmp_path, file_list=[committed])] == [committed]


def _gitignored_dirs_under_code_tops():
    out = set()
    for line in (REPO_ROOT / ".gitignore").read_text().splitlines():
        entry = line.strip()
        if entry.endswith("/") and not entry.startswith(("#", "!")) and "*" not in entry:
            entry = entry.strip("/")
            if entry.split("/")[0] in np_.CODE_INCLUDE and "/" in entry:
                out.add(entry)
    return out


def test_every_runtime_output_subtree_is_gitignored():
    # An exclusion is justified only by the folder never being committed.
    assert set(np_.RUNTIME_OUTPUT_SUBTREES) <= _gitignored_dirs_under_code_tops()


def test_every_gitignored_folder_inside_a_code_dir_is_classified():
    # A new ignored output folder must be listed deliberately (or be a by-name skip such as dist/),
    # so the local walk and a fresh CI checkout agree.
    unclassified = {
        d for d in _gitignored_dirs_under_code_tops()
        if d not in np_.RUNTIME_OUTPUT_SUBTREES and not np_._skipped_dir_name(d.rsplit("/", 1)[-1])
    }
    assert unclassified == set(), "classify in scripts/naming_policy.py RUNTIME_OUTPUT_SUBTREES"
