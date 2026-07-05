"""Tests for GitMemoryExtractor and GitMemoryLoader."""

import os
import pytest
from knowledge.git_memory import GitMemoryExtractor


class TestGitMemoryExtractor:
    """Tests for git commit extraction."""

    def test_extract_commits_returns_list(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=5)
        assert isinstance(commits, list)
        assert len(commits) <= 5

    def test_commit_has_required_fields(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1)
        assert commits, "No commits found (is this a git repo?)"
        commit = commits[0]
        assert "id" in commit
        assert commit["id"].startswith("git-")
        assert "content" in commit
        assert "metadata" in commit
        md = commit["metadata"]
        assert md["memory_type"] == "procedural"
        assert md["source"] == "git"
        assert "commit_hash" in md
        assert "timestamp" in md
        assert "tags" in md
        assert "git-commit" in md["tags"]

    def test_content_starts_with_commit_prefix(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1)
        assert commits
        assert commits[0]["content"].startswith("Commit: ")

    def test_tags_are_comma_separated_string(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1)
        assert commits
        tags = commits[0]["metadata"]["tags"]
        assert isinstance(tags, str)

    def test_tag_extraction_feat(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("feat: add new feature")
        assert "feature" in tags
        assert "git-commit" in tags

    def test_tag_extraction_fix(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("fix: resolve bug")
        assert "bugfix" in tags

    def test_tag_extraction_refactor(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("refactor: clean up memory module")
        assert "refactor" in tags

    def test_tag_extraction_docs(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("docs: update README")
        assert "documentation" in tags

    def test_tag_extraction_wip(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("WIP: partial refactor")
        assert "work-in-progress" in tags

    def test_tag_extraction_breaking(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("feat!: BREAKING change to API")
        assert "breaking-change" in tags

    def test_tag_extraction_unknown_prefix(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("random commit message")
        assert tags == ["git-commit"]

    def test_extract_with_diffs(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1, include_diffs=True)
        assert commits
        assert "Changes:" in commits[0]["content"]

    def test_extract_with_since_date(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=10, since="2025-01-01")
        assert isinstance(commits, list)

    def test_invalid_repo_path_returns_empty(self):
        extractor = GitMemoryExtractor(repo_path="/nonexistent/path")
        commits = extractor.extract_commits(limit=1)
        assert commits == []

    def test_get_recent_since_hash_with_head(self):
        """HEAD~0 should return 0 new commits."""
        extractor = GitMemoryExtractor()
        # Get current HEAD hash
        commits = extractor.extract_commits(limit=1)
        assert commits
        head_hash = commits[0]["metadata"]["full_hash"]
        new = extractor.get_recent_since_hash(head_hash)
        assert new == []


class TestChangeType:
    """Tests for change_type metadata derivation."""

    def test_change_type_always_present(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=3)
        assert commits
        for c in commits:
            assert "change_type" in c["metadata"]
            assert isinstance(c["metadata"]["change_type"], str)

    def test_derive_change_type_feature(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("feat: add thing")
        assert extractor._derive_change_type(tags) == "feature"

    def test_derive_change_type_bugfix(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("fix: squash bug")
        assert extractor._derive_change_type(tags) == "bugfix"

    def test_derive_change_type_other_when_no_prefix(self):
        extractor = GitMemoryExtractor()
        tags = extractor._extract_tags("random commit message")
        assert extractor._derive_change_type(tags) == "other"


class TestStructuredDiffStats:
    """Tests for per-commit structured diff metadata (--numstat)."""

    def test_include_diffs_adds_structured_metadata(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1, include_diffs=True)
        assert commits
        md = commits[0]["metadata"]
        assert "files_changed" in md
        assert "files_changed_count" in md
        assert "lines_added" in md
        assert "lines_removed" in md
        assert isinstance(md["files_changed"], str)  # comma-joined for Chroma
        assert isinstance(md["files_changed_count"], int)
        assert isinstance(md["lines_added"], int)
        assert isinstance(md["lines_removed"], int)
        assert md["files_changed_count"] >= 1

    def test_no_structured_metadata_without_diffs(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1, include_diffs=False)
        assert commits
        assert "files_changed" not in commits[0]["metadata"]

    def test_diff_stats_dict_shape(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1)
        assert commits
        head = commits[0]["metadata"]["full_hash"]
        stats = extractor._get_commit_diff_stats(head, max_files=50)
        assert set(stats.keys()) == {
            "files_changed", "files_changed_count",
            "lines_added", "lines_removed", "summary_text",
        }
        assert isinstance(stats["files_changed"], list)
        assert stats["files_changed_count"] == len(stats["files_changed"]) or \
            stats["files_changed_count"] >= len(stats["files_changed"])

    def test_diff_stats_caps_file_list(self):
        extractor = GitMemoryExtractor()
        commits = extractor.extract_commits(limit=1)
        assert commits
        head = commits[0]["metadata"]["full_hash"]
        stats = extractor._get_commit_diff_stats(head, max_files=1)
        assert len(stats["files_changed"]) <= 1

    def test_rename_plain_path_passthrough(self):
        assert GitMemoryExtractor._rename_new_path("core/foo.py") == "core/foo.py"

    def test_rename_simple(self):
        assert GitMemoryExtractor._rename_new_path("old.py => new.py") == "new.py"

    def test_rename_brace_leading(self):
        assert GitMemoryExtractor._rename_new_path("{a => b}/file.py") == "b/file.py"

    def test_rename_brace_middle(self):
        assert GitMemoryExtractor._rename_new_path("dir/{old => new}/file.py") == "dir/new/file.py"

    def test_diff_stats_bad_commit_returns_empty(self):
        extractor = GitMemoryExtractor()
        stats = extractor._get_commit_diff_stats("deadbeefdeadbeef", max_files=50)
        assert stats["files_changed"] == []
        assert stats["files_changed_count"] == 0


class TestHotFiles:
    """Tests for the hot-files churn tracker."""

    def test_returns_list_of_dicts(self):
        extractor = GitMemoryExtractor()
        hot = extractor.get_hot_files(since_days=3650, limit=10)
        assert isinstance(hot, list)
        assert len(hot) <= 10
        if hot:
            entry = hot[0]
            assert set(entry.keys()) == {"path", "commits", "last_touched", "age_relative"}
            assert isinstance(entry["commits"], int)
            assert entry["commits"] >= 1

    def test_ranked_by_commit_count_desc(self):
        extractor = GitMemoryExtractor()
        hot = extractor.get_hot_files(since_days=3650, limit=25)
        counts = [e["commits"] for e in hot]
        assert counts == sorted(counts, reverse=True)

    def test_exclude_globs_filters(self):
        extractor = GitMemoryExtractor()
        hot = extractor.get_hot_files(
            since_days=3650, limit=100, exclude_globs=["data/", "venv/"]
        )
        assert all("data/" not in e["path"] and "venv/" not in e["path"] for e in hot)

    def test_invalid_repo_returns_empty(self):
        extractor = GitMemoryExtractor(repo_path="/nonexistent/path")
        assert extractor.get_hot_files() == []

    def test_zero_day_window_is_safe(self):
        extractor = GitMemoryExtractor()
        hot = extractor.get_hot_files(since_days=0, limit=10)
        assert isinstance(hot, list)
