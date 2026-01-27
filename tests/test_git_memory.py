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
