"""scripts/naming_policy.py: names describe content, not dates (incident 2026-10-27, owner request).

Bug class: draft "Date used as an artifact's identifier" (naming_runs/bc_draft.md); BC-63 for the source-reading
tests that depend on the path moves.  The `range` tests use THROWAWAY git repos under tmp_path only (the project
repo's own git state is never read - tests/unit/test_no_git_state_in_tests.py).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "naming_policy", Path(__file__).resolve().parents[2] / "scripts" / "naming_policy.py"
)
np_ = importlib.util.module_from_spec(_SPEC)
sys.modules["naming_policy"] = np_
_SPEC.loader.exec_module(np_)


# ------------------------------------------------------------------------------------------ date_tokens matrix
@pytest.mark.parametrize("text, expected", [
    ("oct27", ["oct27"]),
    ("oct-27", ["oct-27"]),
    ("oct27b", ["oct27"]),
    ("fix/oct27c-followups", ["oct27"]),
    ("sept5", ["sept5"]),
    ("Oct 27 bundle", ["Oct 27"]),
    ("2026-10-27", ["2026-10-27"]),
    ("20261027", ["20261027"]),
    ("REVIEW_20260909_example.md", ["20260909"]),
    ("test_sep10_calendar.py", ["sep10"]),
    ("sep30HideSurfaces.test.tsx", ["sep30"]),
    ("testOct27", ["Oct27"]),
    ("a_dec01_b", ["dec01"]),
])
def test_dated_text_is_detected(text, expected):
    assert np_.date_tokens(text) == expected


@pytest.mark.parametrize("text", [
    "gpt-4o", "bge-small-en-v1.5", "MGT6203", "phase5d", "v2", "ISO8601", "#111", "port 8000", "127.0.0.1:8000",
    "python3.11.8", "1.2.3", "decode2", "separate", "december", "monitor27", "nov32", "20261345", "2026-13-01",
    "may 5 items", "CGR-20261014-002", "CGR-20260913-005-2", "kimi-3", "deepseek-v4.1-flash", "claude-fable-5.1",
    "test_user_profile_confirmation.py", "hideSurfaces.test.tsx", "apply_profile_and_error_memory_cleanup.sh",
])
def test_non_dates_do_not_match(text):
    assert np_.date_tokens(text) == []


def test_record_id_is_stripped_but_a_neighbouring_date_still_counts():
    assert np_.date_tokens("CGR-20261014-002 oct14e") == ["oct14"]


# ------------------------------------------------------------------------------------------ surface checks
def test_path_checks_every_component_including_directories():
    assert np_.check_path("tests/unit/test_user_profile_x.py") == []
    assert np_.check_path("scripts/apply_20260903_cleanup.sh")
    assert np_.check_path("core/oct27_helpers/thing.py")  # dated DIRECTORY
    assert np_.check_path("web/src/sep30HideSurfaces.test.tsx")
    assert np_.check_path("./scripts/apply_20260903_cleanup.sh")


def test_allow_rules_archival_folders_and_the_single_approved_test():
    for ok in ("docs/archive/REVIEW_20260909_example.md", "docs/audits/INDEPENDENT_BUG_SEARCH_20261023.md",
               "docs/execution/compaction_20260915/inventory.tsv", "tests/unit/test_sep10_calendar_forced_action_loop.py"):
        assert np_.check_path(ok) == [], ok
    assert np_.check_path("docs/REVIEW_20260909_example.md")
    assert np_.check_path("tests/unit/test_sep10_other.py")
    assert np_.APPROVED_PATHS == {"tests/unit/test_sep10_calendar_forced_action_loop.py"}


def test_commit_subject_branch_and_pr_title():
    assert np_.check_commit_subject("ledger: re-pin accepted-debt anchors after the oct27c wave-3b lanes")
    assert np_.check_commit_subject("ledger: re-pin accepted-debt anchors after the wave-3b lanes") == []
    assert np_.check_commit_subject("Merge pull request #113 from lukehalleran/fix/oct27-docs-drift") == []
    assert np_.check_commit_subject("") == []
    assert np_.check_branch("fix/oct27b-thing") and np_.check_branch("fix/naming-policy-guards") == []
    assert np_.check_pr_title("Oct-25c bundle: file route") and np_.check_pr_title("File route and provider log") == []
    assert np_.check_pr_title("Merge cleanup 2026-10-27")  # PR titles get no merge-commit pass


# ------------------------------------------------------------------------------------------ traversal contract
def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def test_walk_flags_a_dated_file_and_a_dated_directory(tmp_path):
    _tree(tmp_path, {"core/normalize_oct27.py": "", "utils/ok.py": "", "processing/oct27_pkg/mod.py": ""})
    bad = {v[0] for v in np_.check_tree(tmp_path)}
    assert bad == {"core/normalize_oct27.py", "processing/oct27_pkg/mod.py"}


def test_walk_ignores_data_runtime_hidden_and_build_dirs(tmp_path):
    runtime = {f"{d}/backup_20261027.json": "" for d in sorted(np_.SKIP_TOP_LEVEL)}  # every skipped runtime dir
    _tree(tmp_path, {
        **runtime,
        "core/__pycache__/x_oct27.pyc": "", "web/node_modules/p/oct27.js": "", ".github/oct27.yml": "",
        "assets/generated/oct27.png": "", "core/ok.py": "",
    })
    assert np_.check_tree(tmp_path) == []


def test_walk_covers_processing_integrations_agent_branch_eval_hooks_and_web(tmp_path):
    for top in ("processing", "integrations", "agent_branch", "eval", "hooks", "installer", "web", "config"):
        _tree(tmp_path, {f"{top}/x_oct27.py": ""})
    flagged = {v[0].split("/")[0] for v in np_.check_tree(tmp_path)}
    assert flagged == {"processing", "integrations", "agent_branch", "eval", "hooks", "installer", "web", "config"}


def test_unclassified_new_top_level_source_dir_fails(tmp_path):
    _tree(tmp_path, {"core/ok.py": "", "newtree/mod.py": ""})
    out = np_.check_tree(tmp_path)
    assert [v[0] for v in out] == ["newtree/"] and "not classified" in out[0][1]
    # a top-level dir with no source (images etc.) and a skipped data dir do not trip it
    runtime_dir = sorted(np_.SKIP_TOP_LEVEL)[0]
    clean = _tree(tmp_path / "b", {"core/ok.py": "", "pics/a.png": "", f"{runtime_dir}/x.py": ""})
    assert np_.check_tree(clean) == []


def test_tracked_list_scope_adds_docs_and_root_but_not_unrelated_dirs():
    runtime_dir = sorted(np_.SKIP_TOP_LEVEL)[0]
    files = ["docs/PLAN_20260909_x.md", "docs/archive/PLAN_20260909_x.md", "docs/execution/a_20260915/i.tsv",
             "docs/audits/INDEP_20261023.md", "release_oct27.txt", "core/ok.py", f"{runtime_dir}/b_20261027.csv",
             ".github/workflows/naming.yml", "assets/pics/a_oct27.png", "newtree/mod.py"]
    got = np_.iter_candidate_paths(".", file_list=files)
    assert "docs/PLAN_20260909_x.md" in got and "release_oct27.txt" in got
    assert f"{runtime_dir}/b_20261027.csv" not in got and ".github/workflows/naming.yml" not in got
    bad = [v[0] for v in np_.check_tree(".", file_list=files)]
    assert sorted(bad) == ["docs/PLAN_20260909_x.md", "newtree/", "release_oct27.txt"]


# ------------------------------------------------------------------------------------------ range (throwaway repos)
_ENV = {**os.environ, "GIT_AUTHOR_NAME": "Tamsin", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "Tamsin", "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": os.devnull}


def _g(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=_ENV)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _commit(repo: Path, name: str, subject: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _g(repo, "add", name)
    _g(repo, "commit", "-q", "-m", subject)
    return _g(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "src"
    r.mkdir()
    _g(r, "init", "-q", "-b", "main")
    return r


def test_range_clean_subjects_exit_0(repo, capsys):
    base = _commit(repo, "a.txt", "initial")
    _commit(repo, "b.txt", "add the thing")
    head = _commit(repo, "c.txt", "fix the other thing")
    assert np_.main(["--root", str(repo), "range", base, head]) == 0
    assert capsys.readouterr().out == ""


def test_range_dated_subject_exit_1_with_reason(repo, capsys):
    base = _commit(repo, "a.txt", "initial")
    _commit(repo, "b.txt", "ledger: re-pin after the oct27b lanes")
    head = _commit(repo, "c.txt", "ok subject")
    assert np_.main(["--root", str(repo), "range", base, head]) == 1
    out = capsys.readouterr().out
    assert "oct27" in out and "date in commit subject" in out


def test_range_skips_merge_commits_but_checks_their_parents(repo, capsys):
    base = _commit(repo, "a.txt", "initial")
    _g(repo, "checkout", "-q", "-b", "fix/oct27-docs-drift")
    _commit(repo, "b.txt", "clean feature commit")
    _g(repo, "checkout", "-q", "main")
    _commit(repo, "c.txt", "clean mainline commit")
    _g(repo, "merge", "-q", "--no-ff", "fix/oct27-docs-drift", "-m", "Merge branch 'fix/oct27-docs-drift'")
    head = _g(repo, "rev-parse", "HEAD")
    assert np_.main(["--root", str(repo), "range", base, head]) == 0  # merge subject names a dated branch: skipped
    assert [s for _, s in np_.commits_in_range(repo, base, head)] == ["clean mainline commit", "clean feature commit"] \
        or sorted(s for _, s in np_.commits_in_range(repo, base, head)) == ["clean feature commit", "clean mainline commit"]


def test_range_missing_base_sha_exits_2_with_git_message(repo, capsys):
    head = _commit(repo, "a.txt", "initial")
    assert np_.main(["--root", str(repo), "range", "0" * 40, head]) == 2
    assert "git error" in capsys.readouterr().err


def test_range_shallow_clone_without_base_exits_2(repo, tmp_path, capsys):
    base = _commit(repo, "a.txt", "initial")
    _commit(repo, "b.txt", "second")
    head = _commit(repo, "c.txt", "third")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)], check=True, env=_ENV)
    assert np_.main(["--root", str(shallow), "range", base, head]) == 2
    assert "git error" in capsys.readouterr().err


def test_range_shallow_clone_lacking_merge_base_exits_2(repo, tmp_path, capsys):
    base = _commit(repo, "a.txt", "initial")
    _commit(repo, "b.txt", "second")
    head = _commit(repo, "c.txt", "third")
    shallow = tmp_path / "shallow2"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)], check=True, env=_ENV)
    _g(shallow, "fetch", "-q", "--depth", "1", "origin", base)  # base present, but no history linking it to head
    assert np_.main(["--root", str(shallow), "range", base, head]) == 2


def test_paths_tracked_lists_files_from_a_throwaway_repo(repo, capsys):
    _commit(repo, "release_oct27.txt", "add notes")
    (repo / "docs").mkdir()
    (repo / "docs" / "ok.md").write_text("x")
    _g(repo, "add", "docs/ok.md")
    assert np_.main(["--root", str(repo), "paths", "--tracked"]) == 1
    assert "release_oct27.txt" in capsys.readouterr().out
    assert np_.main(["--root", str(repo / "nonexistent"), "paths", "--tracked"]) == 2


def test_cli_usage_errors_exit_2(capsys):
    assert np_.main([]) == 2
    assert np_.main(["bogus"]) == 2
    assert np_.main(["subject"]) == 2
    assert np_.main(["subject", "fine subject"]) == 0
    assert np_.main(["subject", "Oct-27 bundle"]) == 1
    assert np_.main(["branch", "fix/area-what"]) == 0
    assert np_.main(["pr-title", "title 2026-10-27"]) == 1
