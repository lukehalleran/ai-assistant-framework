"""Deployed-function regression tests for CGR-20260913-006 (dm01_raw_substring,
BC-01) — GitMemoryExtractor._extract_tags matched 'wip', 'breaking' and
'hotfix' by bare substring against the lowered commit subject
(knowledge/git_memory.py lines 364/366/368 before this batch), so each fired
inside an unrelated longer word: 'wip' ⊂ "swipe", 'breaking' ⊂
"groundbreaking", 'hotfix' ⊂ "hotfixture"/"nonhotfix".

Anchors (docs/execution/class_guards/requests/CGR-20260913-006.md):
  #15 line 364 `if "wip" in subject_lower:`
  #16 line 366 `if "breaking" in subject_lower:`
  #17 line 368 `if "hotfix" in subject_lower:`

GitMemoryExtractor.__init__(repo_path=".") only stores repo_path; _extract_tags
does no subprocess/git work at all, so these tests never touch a real
repository or shell out to git (tmp_path repo_path, never used).
"""

import pytest

from knowledge.git_memory import GitMemoryExtractor


@pytest.fixture
def extractor(tmp_path):
    return GitMemoryExtractor(repo_path=str(tmp_path))


class TestWipTag:
    """Anchor #15: `if "wip" in subject_lower:` (line 364)."""

    @pytest.mark.parametrize(
        "subject",
        [
            "WIP: partial refactor",
            "wip: quick patch",
            "a wip feature",
            "WIP:\n  partial refactor",  # wrapped/indented form
        ],
    )
    def test_bare_word_tags(self, extractor, subject):
        assert "work-in-progress" in extractor._extract_tags(subject)

    @pytest.mark.parametrize(
        "subject",
        [
            "swipe to dismiss",  # 'wip' <- "swipe" (the request packet's example)
            "wipe the board",
            "unwip",
            "swipe\n  to dismiss",  # wrapped/indented containment counterexample
        ],
    )
    def test_containment_does_not_tag(self, extractor, subject):
        assert "work-in-progress" not in extractor._extract_tags(subject)


class TestBreakingTag:
    """Anchor #16: `if "breaking" in subject_lower:` (line 366)."""

    @pytest.mark.parametrize(
        "subject",
        [
            "breaking change to the parser",
            "feat!: BREAKING change to API",  # tests/test_git_memory.py control
            "breaking\n  change to the parser",
        ],
    )
    def test_bare_word_tags(self, extractor, subject):
        assert "breaking-change" in extractor._extract_tags(subject)

    @pytest.mark.parametrize(
        "subject",
        [
            "groundbreaking results",  # 'breaking' as a suffix of a longer word
            "unbreaking",
            "groundbreaking\n  results",
        ],
    )
    def test_containment_does_not_tag(self, extractor, subject):
        assert "breaking-change" not in extractor._extract_tags(subject)


class TestHotfixTag:
    """Anchor #17: `if "hotfix" in subject_lower:` (line 368)."""

    @pytest.mark.parametrize(
        "subject",
        [
            "hotfix for login",
            "a hotfix",
            "hotfixes",  # sense-preserving inflection the chokepoint accepts
            "hotfix\n  for login",
        ],
    )
    def test_bare_word_tags(self, extractor, subject):
        assert "hotfix" in extractor._extract_tags(subject)

    @pytest.mark.parametrize(
        "subject",
        [
            "hotfixture",  # longer word; not an inflection the chokepoint accepts
            "hotfixation",
            "nonhotfix",
        ],
    )
    def test_containment_does_not_tag(self, extractor, subject):
        assert "hotfix" not in extractor._extract_tags(subject)


class TestCombinedAndNoNegation:
    """Contract checks beyond the three anchors."""

    def test_all_three_tag_together(self, extractor):
        tags = extractor._extract_tags("wip: breaking hotfix all at once")
        assert "work-in-progress" in tags
        assert "breaking-change" in tags
        assert "hotfix" in tags

    def test_no_negation_semantics(self, extractor):
        """Tag metadata is not a request cue: plain boundary matching only,
        with no negation short-circuit (unlike utils.trigger_match.find_hits
        / has_non_negated_hit, which are deliberately not used here)."""
        tags = extractor._extract_tags("do not merge, this is not wip")
        assert "work-in-progress" in tags

    def test_unrelated_subject_untagged(self, extractor):
        assert extractor._extract_tags("random commit message") == ["git-commit"]
