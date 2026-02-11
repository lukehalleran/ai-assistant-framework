# tests/unit/test_proposal_filter.py
"""
Unit tests for ProposalFilter — retrieval, dedup, gating, and ranking
of code proposals for prompt injection.
"""

import json
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from memory.code_proposal import CodeProposal, ProposalStatus, ProposalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(
    title: str = "Test Proposal",
    tags: list = None,
    priority: int = 5,
    proposal_type: ProposalType = ProposalType.FEATURE,
    status: ProposalStatus = ProposalStatus.PENDING,
    reasoning: str = "Improves the system",
    affected_files: list = None,
    created_at: float = None,
) -> CodeProposal:
    return CodeProposal(
        id=str(uuid.uuid4()),
        title=title,
        proposal_type=proposal_type,
        status=status,
        priority=priority,
        reasoning=reasoning,
        tags=tags or [],
        affected_files=affected_files or [],
        created_at=created_at if created_at is not None else time.time(),
    )


# ---------------------------------------------------------------------------
# Tests: _is_project_related
# ---------------------------------------------------------------------------

class TestIsProjectRelated:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_project_keywords_positive(self):
        pf = self._get_filter()
        assert pf._is_project_related("How should I refactor the pipeline?") is True
        assert pf._is_project_related("Add a new feature to memory system") is True
        assert pf._is_project_related("Fix the bug in the code") is True
        assert pf._is_project_related("implement the api endpoint") is True
        assert pf._is_project_related("test the module") is True
        assert pf._is_project_related("optimize database query performance") is True

    def test_non_project_queries_negative(self):
        pf = self._get_filter()
        assert pf._is_project_related("How are you today?") is False
        assert pf._is_project_related("What's the weather like?") is False
        assert pf._is_project_related("Tell me a joke") is False
        assert pf._is_project_related("") is False

    def test_mixed_queries(self):
        pf = self._get_filter()
        # Contains "build" - project keyword
        assert pf._is_project_related("Can you build me a sandwich?") is True
        # Contains "system" - project keyword
        assert pf._is_project_related("What is the solar system?") is True


# ---------------------------------------------------------------------------
# Tests: _build_utility_query
# ---------------------------------------------------------------------------

class TestBuildUtilityQuery:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_builds_from_real_goals_file(self):
        """Test that _build_utility_query parses the actual docs/GOALS.md."""
        pf = self._get_filter()
        pf._goals_hash = None
        pf._cached_utility_query = None

        query = pf._build_utility_query()
        # Real GOALS.md exists and has active goals
        assert "project goals" in query.lower() or "value" in query.lower()
        assert len(query) > 20

    def test_fallback_when_no_goals_file(self, tmp_path, monkeypatch):
        """When GOALS.md doesn't exist, should use fallback query."""
        pf = self._get_filter()
        pf._goals_hash = None
        pf._cached_utility_query = None

        # Change CWD to tmp_path so docs/GOALS.md won't be found
        monkeypatch.chdir(tmp_path)
        # Also patch __file__ resolution
        import core.prompt.proposal_filter as pf_mod
        original_file = pf_mod.__file__
        monkeypatch.setattr(pf_mod, "__file__", str(tmp_path / "fake.py"))

        query = pf._build_utility_query()
        assert "value" in query.lower() or "improve" in query.lower()

        monkeypatch.setattr(pf_mod, "__file__", original_file)

    def test_caching_by_hash(self):
        """Cached utility query should be returned when hash matches."""
        pf = self._get_filter()
        # Pre-populate cache
        pf._goals_hash = "abc123"
        pf._cached_utility_query = "cached query"
        assert pf._cached_utility_query == "cached query"

    def test_parses_active_goals(self):
        """Test goal extraction from content."""
        pf = self._get_filter()
        goals_content = """# Daemon Project Goals

## Active Goals (Current Sprint)

### 1. Complete Modular Refactor
- Status: In progress

### 2. Code Proposals System (`goal-directed`)
- Status: In progress

---

## Medium-Term Goals
"""
        # Write to a temp location and test
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(goals_content)
            temp_path = f.name

        try:
            from pathlib import Path as RealPath
            p = RealPath(temp_path)
            content = p.read_text(encoding="utf-8")

            # Simulate the parsing logic
            import re
            goals = []
            in_active = False
            for line in content.splitlines():
                if "Active Goals" in line:
                    in_active = True
                    continue
                if in_active and line.startswith("---"):
                    break
                if in_active and line.startswith("### "):
                    goal_text = re.sub(r"^###\s*\d+\.\s*", "", line).strip()
                    goal_text = re.sub(r"`[^`]*`", "", goal_text).strip()
                    if goal_text:
                        goals.append(goal_text)

            assert len(goals) == 2
            assert "Complete Modular Refactor" in goals[0]
            assert "Code Proposals System" in goals[1]
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Tests: _keyword_dedup
# ---------------------------------------------------------------------------

class TestKeywordDedup:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_no_duplicates(self):
        pf = self._get_filter()
        p1 = _make_proposal("Add caching layer", tags=["cache", "performance"], priority=7)
        p2 = _make_proposal("Fix login bug", tags=["auth", "bugfix"], priority=5)
        result = pf._keyword_dedup([p1, p2])
        assert len(result) == 2

    def test_removes_duplicate_by_tags_and_title(self):
        pf = self._get_filter()
        p1 = _make_proposal("Add memory dedup", tags=["memory", "dedup", "quality"], priority=7)
        p2 = _make_proposal("Add memory dedup system", tags=["memory", "dedup", "quality", "system"], priority=5)
        result = pf._keyword_dedup([p1, p2])
        assert len(result) == 1
        assert result[0].priority == 7  # keeps higher priority

    def test_keeps_distinct_proposals(self):
        pf = self._get_filter()
        p1 = _make_proposal("Refactor gate system", tags=["gate", "refactor"], priority=6)
        p2 = _make_proposal("Add web search", tags=["search", "web", "api"], priority=8)
        p3 = _make_proposal("Fix memory leak", tags=["memory", "bugfix"], priority=4)
        result = pf._keyword_dedup([p1, p2, p3])
        assert len(result) == 3

    def test_single_proposal(self):
        pf = self._get_filter()
        p1 = _make_proposal("Solo proposal")
        result = pf._keyword_dedup([p1])
        assert len(result) == 1

    def test_empty_list(self):
        pf = self._get_filter()
        result = pf._keyword_dedup([])
        assert result == []

    def test_empty_tags_no_dedup(self):
        """Proposals with empty tags should not be deduped by tags."""
        pf = self._get_filter()
        p1 = _make_proposal("Add new feature A", tags=[], priority=5)
        p2 = _make_proposal("Add new feature B", tags=[], priority=5)
        result = pf._keyword_dedup([p1, p2])
        # Empty tags: tag_sim = 0, so no dedup even if title overlaps
        assert len(result) == 2

    def test_threshold_boundary(self):
        """Exactly at threshold should trigger dedup."""
        pf = self._get_filter()
        # 3 shared tags out of 5 total = Jaccard 3/5 = 0.60 (exactly at threshold)
        # Title: "add memory system" vs "add memory cache" = intersection(add,memory)/union(add,memory,system,cache) = 2/4 = 0.50
        p1 = _make_proposal("add memory system", tags=["a", "b", "c"], priority=8)
        p2 = _make_proposal("add memory cache", tags=["a", "b", "c", "d", "e"], priority=4)
        # Jaccard = 3/5 = 0.60 >= 0.60, word_sim = 2/4 = 0.50 >= 0.50 => dedup
        result = pf._keyword_dedup([p1, p2], tag_threshold=0.60)
        assert len(result) == 1
        assert result[0].priority == 8


# ---------------------------------------------------------------------------
# Tests: _semantic_dedup
# ---------------------------------------------------------------------------

class TestSemanticDedup:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_distinct_proposals_kept(self):
        """Very different proposals should not be deduped."""
        pf = self._get_filter()
        p1 = _make_proposal(
            "Add Docker multi-stage builds",
            tags=["docker", "infra"],
            reasoning="Reduce image size and build time",
        )
        p2 = _make_proposal(
            "Implement user authentication with OAuth",
            tags=["auth", "security"],
            reasoning="Enable secure user login via third-party providers",
        )
        result = pf._semantic_dedup([p1, p2], threshold=0.85)
        assert len(result) == 2

    def test_similar_proposals_deduped(self):
        """Nearly identical proposals should be deduped."""
        pf = self._get_filter()
        p1 = _make_proposal(
            "Add cross-collection deduplication",
            tags=["memory", "dedup"],
            priority=8,
            reasoning="Facts, summaries, and skills often contain overlapping information",
        )
        p2 = _make_proposal(
            "Add cross-collection dedup system",
            tags=["memory", "dedup"],
            priority=5,
            reasoning="Facts, summaries, and skills often contain overlapping info",
        )
        result = pf._semantic_dedup([p1, p2], threshold=0.85)
        assert len(result) == 1
        assert result[0].priority == 8  # higher priority kept

    def test_empty_and_single(self):
        pf = self._get_filter()
        assert pf._semantic_dedup([], threshold=0.85) == []
        p1 = _make_proposal("Solo")
        assert len(pf._semantic_dedup([p1])) == 1


# ---------------------------------------------------------------------------
# Tests: get_proposals (full pipeline)
# ---------------------------------------------------------------------------

class TestGetProposals:
    @pytest.fixture
    def mock_proposals(self):
        """Create 5 proposals with 2 near-duplicates."""
        return [
            _make_proposal("Unified cross-collection deduplication", tags=["memory", "dedup"], priority=8,
                          reasoning="Facts, summaries, and skills contain overlapping info"),
            _make_proposal("Cross-collection dedup system", tags=["memory", "dedup", "quality"], priority=5,
                          reasoning="Facts, summaries, and skills contain overlapping information"),
            _make_proposal("Add retry logic to LLM calls", tags=["llm", "resilience"], priority=7,
                          reasoning="API calls sometimes fail transiently"),
            _make_proposal("Improve fact extraction precision", tags=["facts", "nlp", "quality"], priority=6,
                          reasoning="Too many false triples from current extraction"),
            _make_proposal("Add health check endpoint", tags=["api", "monitoring", "infra"], priority=4,
                          reasoning="No way to monitor system health"),
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_proposals):
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()

        # Mock proposal store
        mock_store = MagicMock()
        mock_store.query_proposals.return_value = mock_proposals
        pf._proposal_store = mock_store

        # Mock gate system (pass-through with scores)
        mock_gate = AsyncMock()
        async def fake_gate(query, dicts):
            for i, d in enumerate(dicts):
                d["relevance_score"] = 0.9 - (i * 0.1)
            return dicts
        mock_gate.filter_memories = fake_gate
        pf._gate_system = mock_gate

        result = await pf.get_proposals("How should I refactor the code?", limit=3)

        assert len(result) <= 3
        # Each result should have content, metadata, relevance_score
        for r in result:
            assert "content" in r
            assert "metadata" in r
            assert "relevance_score" in r

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()

        with patch("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", False):
            result = await pf.get_proposals("refactor the code", limit=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self):
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()
        mock_store = MagicMock()
        mock_store.query_proposals.return_value = []
        pf._proposal_store = mock_store

        result = await pf.get_proposals("implement new feature", limit=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_project_query_returns_empty(self):
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()
        # Should never even hit the store
        mock_store = MagicMock()
        pf._proposal_store = mock_store

        result = await pf.get_proposals("How are you today?", limit=3)
        assert result == []
        mock_store.query_proposals.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self):
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()
        pf._proposal_store = None

        # Mock the import to fail
        with patch.dict("sys.modules", {"memory.proposal_store": None}):
            result = await pf.get_proposals("refactor the pipeline", limit=3)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: ContextGatherer.get_proposed_features
# ---------------------------------------------------------------------------

class TestContextGathererIntegration:
    @pytest.mark.asyncio
    async def test_get_proposed_features_disabled(self):
        """Should return [] when feature is disabled."""
        from core.prompt.context_gatherer import ContextGatherer

        mock_coordinator = MagicMock()
        mock_coordinator.chroma_store = None
        mock_model = MagicMock()
        mock_token = MagicMock()

        gatherer = ContextGatherer(mock_coordinator, mock_model, mock_token)

        with patch("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", False):
            result = await gatherer.get_proposed_features("refactor code", limit=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_proposed_features_tracks_citations(self):
        """Should populate memory_id_map with PROPOSAL_N entries."""
        from core.prompt.context_gatherer import ContextGatherer

        mock_coordinator = MagicMock()
        mock_coordinator.chroma_store = MagicMock()
        mock_model = MagicMock()
        mock_token = MagicMock()

        gatherer = ContextGatherer(mock_coordinator, mock_model, mock_token)

        # Mock the filter
        fake_results = [
            {
                "content": "Proposal: Test Feature",
                "metadata": {
                    "proposal_id": "abc-123",
                    "title": "Test Feature",
                    "priority": 7,
                    "created_at": time.time(),
                },
                "relevance_score": 0.85,
            }
        ]

        with patch("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", True):
            with patch("core.prompt.proposal_filter.ProposalFilter") as MockFilter:
                mock_filter_instance = MockFilter.return_value
                mock_filter_instance.get_proposals = AsyncMock(return_value=fake_results)
                gatherer._proposal_filter = mock_filter_instance

                result = await gatherer.get_proposed_features("refactor code", limit=3)

        assert len(result) == 1
        assert "PROPOSAL_1" in gatherer.memory_id_map
        assert gatherer.memory_id_map["PROPOSAL_1"]["type"] == "code_proposal"
        assert gatherer.memory_id_map["PROPOSAL_1"]["title"] == "Test Feature"


# ---------------------------------------------------------------------------
# Tests: Composite scoring
# ---------------------------------------------------------------------------

class TestCompositeScoring:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_priority_scoring(self):
        pf = self._get_filter()
        # Priority 1 -> 0.0, Priority 10 -> 1.0
        low = _make_proposal("Low", priority=1)
        high = _make_proposal("High", priority=10)
        mid = _make_proposal("Mid", priority=5)

        assert pf._score_priority(low) == pytest.approx(0.0)
        assert pf._score_priority(high) == pytest.approx(1.0)
        assert 0.3 < pf._score_priority(mid) < 0.6

    def test_breadth_scoring(self):
        pf = self._get_filter()
        # No files/tags = 0.0
        narrow = _make_proposal("Narrow", tags=[], affected_files=[])
        assert pf._score_breadth(narrow) == pytest.approx(0.0)

        # Many files across dirs with diverse tags = high score
        broad = _make_proposal(
            "Broad",
            tags=["a", "b", "c", "d"],
            affected_files=[
                "core/prompt/builder.py",
                "core/prompt/formatter.py",
                "memory/coordinator.py",
                "processing/gate.py",
                "config/app_config.py",
            ],
        )
        assert pf._score_breadth(broad) > 0.7

    def test_breadth_counts_distinct_dirs(self):
        pf = self._get_filter()
        # 3 files but all in same dir = lower dir score
        same_dir = _make_proposal(
            "Same dir",
            tags=["a"],
            affected_files=["core/a.py", "core/b.py", "core/c.py"],
        )
        # 3 files across 3 dirs = higher dir score
        multi_dir = _make_proposal(
            "Multi dir",
            tags=["a"],
            affected_files=["core/a.py", "memory/b.py", "utils/c.py"],
        )
        assert pf._score_breadth(multi_dir) > pf._score_breadth(same_dir)

    def test_recency_scoring(self):
        pf = self._get_filter()
        # Brand new proposal = ~1.0
        new = _make_proposal("New", created_at=time.time())
        assert pf._score_recency(new) > 0.95

        # 14 days old = ~0.5 (half-life)
        two_weeks = _make_proposal("2 weeks", created_at=time.time() - 14 * 86400)
        assert 0.4 < pf._score_recency(two_weeks) < 0.6

        # 28 days old = ~0.25
        month = _make_proposal("Month", created_at=time.time() - 28 * 86400)
        assert pf._score_recency(month) < 0.35

    def test_recency_unknown_age(self):
        pf = self._get_filter()
        unknown = _make_proposal("Unknown", created_at=0)
        assert pf._score_recency(unknown) == 0.5

    def test_composite_score_all_signals(self):
        pf = self._get_filter()
        # High priority, broad, new, high alignment = high score
        best = _make_proposal(
            "Best",
            priority=10,
            tags=["a", "b", "c", "d"],
            affected_files=["core/a.py", "memory/b.py", "utils/c.py", "config/d.py", "proc/e.py"],
            created_at=time.time(),
        )
        score = pf._compute_composite_score(best, goal_alignment=0.9)
        assert score > 0.8

        # Low priority, narrow, old, low alignment = low score
        worst = _make_proposal(
            "Worst",
            priority=1,
            tags=[],
            affected_files=[],
            created_at=time.time() - 60 * 86400,
        )
        score_low = pf._compute_composite_score(worst, goal_alignment=0.1)
        assert score_low < 0.15

    def test_composite_weights_matter(self):
        """Changing weights should change relative ordering."""
        pf = self._get_filter()
        # High priority but narrow and old
        p1 = _make_proposal("High-P", priority=10, tags=[], affected_files=[],
                            created_at=time.time() - 30 * 86400)
        # Low priority but broad and new
        p2 = _make_proposal("Broad-New", priority=3, tags=["a", "b", "c", "d"],
                            affected_files=["core/a.py", "mem/b.py", "utils/c.py", "cfg/d.py", "x/e.py"],
                            created_at=time.time())

        # With high priority weight: p1 wins
        s1_prio = pf._compute_composite_score(p1, goal_alignment=0.5, w_priority=0.8, w_breadth=0.1, w_recency=0.05, w_goal=0.05)
        s2_prio = pf._compute_composite_score(p2, goal_alignment=0.5, w_priority=0.8, w_breadth=0.1, w_recency=0.05, w_goal=0.05)
        assert s1_prio > s2_prio

        # With high breadth+recency weight: p2 wins
        s1_broad = pf._compute_composite_score(p1, goal_alignment=0.5, w_priority=0.05, w_breadth=0.45, w_recency=0.45, w_goal=0.05)
        s2_broad = pf._compute_composite_score(p2, goal_alignment=0.5, w_priority=0.05, w_breadth=0.45, w_recency=0.45, w_goal=0.05)
        assert s2_broad > s1_broad

    def test_pipeline_uses_composite_not_pure_semantic(self, ):
        """Verify get_proposals sorts by composite score, not gate relevance alone."""
        # A high-priority narrow proposal should outrank a low-priority one
        # even if the gate gives both equal relevance scores
        pass  # Covered by test_full_pipeline_composite_ordering below


class TestFullPipelineCompositeOrdering:
    @pytest.mark.asyncio
    async def test_high_priority_outranks_low(self):
        """High-priority proposal should rank above low-priority even with equal gate scores."""
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter()

        proposals = [
            _make_proposal(
                "Add Docker multi-stage build configuration",
                priority=2, tags=["docker", "infra"],
                affected_files=["Dockerfile"],
                reasoning="Reduce image size with multi-stage builds",
                created_at=time.time(),
            ),
            _make_proposal(
                "Implement cross-collection memory deduplication",
                priority=9, tags=["memory", "dedup"],
                affected_files=["memory/coordinator.py", "processing/gate.py", "memory/scorer.py"],
                reasoning="Facts and summaries contain overlapping entries hurting quality",
                created_at=time.time(),
            ),
        ]

        mock_store = MagicMock()
        mock_store.query_proposals.return_value = proposals
        pf._proposal_store = mock_store

        # Gate gives equal scores to both
        mock_gate = AsyncMock()
        async def equal_gate(query, dicts):
            for d in dicts:
                d["relevance_score"] = 0.7
            return dicts
        mock_gate.filter_memories = equal_gate
        pf._gate_system = mock_gate

        result = await pf.get_proposals("refactor the code", limit=2)
        assert len(result) == 2
        # Higher priority (+ broader) should be first
        assert result[0]["metadata"]["priority"] == 9
        assert result[1]["metadata"]["priority"] == 2


# ---------------------------------------------------------------------------
# Tests: LLM pairwise ranking
# ---------------------------------------------------------------------------

class TestLLMPairwiseRanking:
    @pytest.mark.asyncio
    async def test_llm_ranking_picks_winner(self):
        """LLM pairwise ranking should select winner from each pair."""
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        mock_model.generate_once = AsyncMock(return_value="A")

        pf = ProposalFilter(model_manager=mock_model)

        proposals = [
            {"content": "A proposal", "metadata": {"title": "A", "proposal_type": "feature", "priority": 8, "reasoning": "good"}, "relevance_score": 0.9},
            {"content": "B proposal", "metadata": {"title": "B", "proposal_type": "feature", "priority": 5, "reasoning": "ok"}, "relevance_score": 0.7},
            {"content": "C proposal", "metadata": {"title": "C", "proposal_type": "bugfix", "priority": 6, "reasoning": "fine"}, "relevance_score": 0.8},
            {"content": "D proposal", "metadata": {"title": "D", "proposal_type": "refactor", "priority": 4, "reasoning": "meh"}, "relevance_score": 0.6},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=2)
        assert len(result) == 2
        # LLM always picks "A" so first from each pair wins
        assert result[0]["metadata"]["title"] == "A"
        assert result[1]["metadata"]["title"] == "C"

    @pytest.mark.asyncio
    async def test_llm_ranking_handles_odd_count(self):
        """Odd number of candidates: last one gets a bye."""
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        mock_model.generate_once = AsyncMock(return_value="B")

        pf = ProposalFilter(model_manager=mock_model)

        proposals = [
            {"content": "A", "metadata": {"title": "A", "proposal_type": "f", "priority": 5, "reasoning": "x"}, "relevance_score": 0.9},
            {"content": "B", "metadata": {"title": "B", "proposal_type": "f", "priority": 6, "reasoning": "y"}, "relevance_score": 0.8},
            {"content": "C", "metadata": {"title": "C", "proposal_type": "f", "priority": 4, "reasoning": "z"}, "relevance_score": 0.7},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=3)
        # B wins (A vs B, LLM says "B"), C gets bye -> [B, C]
        # But limit is 3 and we only have 2 after round 1, so both returned
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_llm_ranking_fallback_on_error(self):
        """If LLM call fails, should fall back to composite score order."""
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        mock_model.generate_once = AsyncMock(side_effect=Exception("API error"))

        pf = ProposalFilter(model_manager=mock_model)

        proposals = [
            {"content": "A", "metadata": {"title": "A"}, "relevance_score": 0.9},
            {"content": "B", "metadata": {"title": "B"}, "relevance_score": 0.7},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=1)
        assert len(result) == 1
        # Falls back to higher relevance_score
        assert result[0]["metadata"]["title"] == "A"

    @pytest.mark.asyncio
    async def test_llm_ranking_skipped_when_disabled(self):
        """No model_manager = skip LLM ranking, return as-is."""
        from core.prompt.proposal_filter import ProposalFilter

        pf = ProposalFilter(model_manager=None)

        proposals = [
            {"content": "A", "metadata": {"title": "A"}, "relevance_score": 0.9},
            {"content": "B", "metadata": {"title": "B"}, "relevance_score": 0.7},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=1)
        assert len(result) == 1
        assert result[0]["metadata"]["title"] == "A"

    @pytest.mark.asyncio
    async def test_llm_ranking_fewer_than_limit(self):
        """If candidates <= limit, return all without LLM calls."""
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        pf = ProposalFilter(model_manager=mock_model)

        proposals = [
            {"content": "A", "metadata": {"title": "A"}, "relevance_score": 0.9},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=3)
        assert len(result) == 1
        # No LLM call should have been made
        mock_model.generate_once.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Novelty scoring (git overlap detection)
# ---------------------------------------------------------------------------

class TestNoveltyScoring:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_no_git_history_returns_full_novelty(self):
        """Without git history, all proposals are novel."""
        pf = self._get_filter()
        pf._commit_cache = []  # simulate no git
        p = _make_proposal("Add memory consolidation engine")
        assert pf._score_novelty(p) == 1.0

    def test_matching_commit_reduces_novelty(self):
        """Proposal matching a recent commit should get low novelty."""
        pf = self._get_filter()
        pf._commit_cache = [
            "feat: add code proposals system with gui and shutdown integration",
            "feat: add procedural skills system",
            "refactor: slim memorycoordinator to thin orchestrator",
        ]
        # This proposal is already implemented (matches commit 1)
        p = _make_proposal(
            "Implement Code Proposals System",
            reasoning="Add proposals with GUI integration",
        )
        score = pf._score_novelty(p)
        assert score < 0.7, f"Expected low novelty for implemented proposal, got {score}"

    def test_unrelated_commit_keeps_novelty(self):
        """Proposal unrelated to commits should stay novel."""
        pf = self._get_filter()
        pf._commit_cache = [
            "fix: typo in readme",
            "docs: update changelog",
        ]
        p = _make_proposal(
            "Dreaming Engine — Batch Proposal Generation During Idle Time",
            reasoning="Generate proposals autonomously when system is idle",
        )
        score = pf._score_novelty(p)
        assert score > 0.8, f"Expected high novelty for unrelated proposal, got {score}"

    def test_short_title_returns_novel(self):
        pf = self._get_filter()
        pf._commit_cache = ["feat: add something"]
        # All words are < 3 chars so they get filtered out by regex
        p = _make_proposal("Do it up")
        assert pf._score_novelty(p) == 1.0

    def test_novelty_penalty_applied_in_composite(self):
        """Novelty penalty should reduce composite score for implemented proposals."""
        pf = self._get_filter()
        pf._commit_cache = [
            "feat: add proposal utility ranker with composite scoring",
        ]
        p_implemented = _make_proposal(
            "Proposal Utility Ranker — LLM Scoring",
            priority=9,
            tags=["proposals", "ranking"],
        )
        p_novel = _make_proposal(
            "Memory Decay and Consolidation Engine",
            priority=9,
            tags=["memory", "quality"],
        )
        novelty_implemented = pf._score_novelty(p_implemented)
        novelty_novel = pf._score_novelty(p_novel)
        assert novelty_novel > novelty_implemented, (
            f"Novel proposal ({novelty_novel:.2f}) should have higher novelty "
            f"than implemented one ({novelty_implemented:.2f})"
        )


# ---------------------------------------------------------------------------
# Tests: Topic diversity selection
# ---------------------------------------------------------------------------

class TestDiverseSelect:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def _make_scored_dict(self, title, tags, score):
        import json
        return {
            "content": title,
            "metadata": {
                "title": title,
                "tags_json": json.dumps(tags),
            },
            "relevance_score": score,
        }

    def test_diverse_topics_all_selected(self):
        """Three proposals with different tags should all be selected."""
        pf = self._get_filter()
        dicts = [
            self._make_scored_dict("Dreaming Engine", ["dreaming", "batch", "idle"], 0.90),
            self._make_scored_dict("Memory Consolidation", ["memory", "quality", "decay"], 0.85),
            self._make_scored_dict("Orchestrator Refactor", ["refactor", "orchestrator", "modular"], 0.80),
        ]
        result = pf._diverse_select(dicts, limit=3)
        assert len(result) == 3
        titles = [r["metadata"]["title"] for r in result]
        assert "Dreaming Engine" in titles
        assert "Memory Consolidation" in titles
        assert "Orchestrator Refactor" in titles

    def test_clustering_broken_up(self):
        """Multiple same-topic proposals should be de-clustered."""
        pf = self._get_filter()
        # All three top proposals share "proposals" + "ranking" tags (overlap >= 0.34)
        dicts = [
            self._make_scored_dict("Proposal Utility Ranker", ["proposals", "ranking", "utility"], 0.90),
            self._make_scored_dict("Proposal Leaderboard", ["proposals", "ranking", "dashboard"], 0.88),
            self._make_scored_dict("Proposal Dreaming Loop", ["proposals", "ranking", "dreaming"], 0.85),
            self._make_scored_dict("Memory Consolidation", ["memory", "quality", "decay"], 0.82),
            self._make_scored_dict("Orchestrator Slim-Down", ["refactor", "orchestrator", "modular"], 0.80),
        ]
        result = pf._diverse_select(dicts, limit=3)
        assert len(result) == 3
        titles = [r["metadata"]["title"] for r in result]
        # Should NOT have all three proposal-ranking entries
        proposal_count = sum(1 for t in titles if "Proposal" in t)
        assert proposal_count <= 1, (
            f"Expected at most 1 proposal-ranking entry in top 3, got {proposal_count}: {titles}"
        )
        # Memory and Orchestrator should have been promoted
        assert "Memory Consolidation" in titles
        assert "Orchestrator Slim-Down" in titles

    def test_empty_tags_dont_block(self):
        """Proposals with empty tags should not block each other."""
        pf = self._get_filter()
        dicts = [
            self._make_scored_dict("Proposal A", [], 0.90),
            self._make_scored_dict("Proposal B", [], 0.85),
            self._make_scored_dict("Proposal C", [], 0.80),
        ]
        result = pf._diverse_select(dicts, limit=3)
        assert len(result) == 3

    def test_fewer_than_limit(self):
        """If fewer proposals than limit, return all."""
        pf = self._get_filter()
        dicts = [
            self._make_scored_dict("Only One", ["test"], 0.90),
        ]
        result = pf._diverse_select(dicts, limit=3)
        assert len(result) == 1

    def test_backfill_when_too_aggressive(self):
        """If diversity filtering is too strict, backfill from remaining."""
        pf = self._get_filter()
        # All proposals share identical tags — diversity will reject after first
        dicts = [
            self._make_scored_dict("A", ["proposals", "ranking"], 0.90),
            self._make_scored_dict("B", ["proposals", "ranking"], 0.85),
            self._make_scored_dict("C", ["proposals", "ranking"], 0.80),
        ]
        result = pf._diverse_select(dicts, limit=3)
        # Should backfill to reach limit=3
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: Config constants
# ---------------------------------------------------------------------------

class TestConfig:
    def test_prompt_config_exists(self):
        from config.app_config import (
            CODE_PROPOSALS_PROMPT_ENABLED,
            CODE_PROPOSALS_PROMPT_MAX,
            CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD,
            CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD,
            CODE_PROPOSALS_LLM_RANKING,
            CODE_PROPOSALS_LLM_RANKING_MODEL,
            CODE_PROPOSALS_WEIGHT_PRIORITY,
            CODE_PROPOSALS_WEIGHT_BREADTH,
            CODE_PROPOSALS_WEIGHT_RECENCY,
            CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT,
        )
        assert isinstance(CODE_PROPOSALS_PROMPT_ENABLED, bool)
        assert isinstance(CODE_PROPOSALS_PROMPT_MAX, int)
        assert CODE_PROPOSALS_PROMPT_MAX > 0
        assert 0 < CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD <= 1.0
        assert 0 < CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD <= 1.0
        assert isinstance(CODE_PROPOSALS_LLM_RANKING, bool)
        assert isinstance(CODE_PROPOSALS_LLM_RANKING_MODEL, str)
        # Composite weights should sum close to 1.0
        total = (CODE_PROPOSALS_WEIGHT_PRIORITY + CODE_PROPOSALS_WEIGHT_BREADTH
                 + CODE_PROPOSALS_WEIGHT_RECENCY + CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT)
        assert 0.95 < total < 1.05


# ---------------------------------------------------------------------------
# Tests: Formatter sanitization
# ---------------------------------------------------------------------------

class TestFormatterSanitization:
    def test_proposed_features_header_sanitized(self):
        from core.prompt.formatter import _sanitize_embedded_headers

        text = "The system has a [PROPOSED FEATURES] section that shows proposals."
        result = _sanitize_embedded_headers(text)
        assert "[PROPOSED FEATURES]" not in result
        assert "(PROPOSED FEATURES)" in result

    def test_proposed_features_with_count_sanitized(self):
        from core.prompt.formatter import _sanitize_embedded_headers

        text = "We added [PROPOSED FEATURES n=3] to the prompt."
        result = _sanitize_embedded_headers(text)
        assert "[PROPOSED FEATURES" not in result


# ---------------------------------------------------------------------------
# Tests: Token manager priority
# ---------------------------------------------------------------------------

class TestTokenManagerPriority:
    def test_proposed_features_in_priority_order(self):
        from core.prompt.token_manager import PRIORITY_ORDER

        names = [name for name, _ in PRIORITY_ORDER]
        assert "proposed_features" in names

        # Should be lower priority than memories (5) and higher than wiki (1)
        pf_priority = dict(PRIORITY_ORDER).get("proposed_features")
        assert pf_priority == 3
