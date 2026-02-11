# tests/unit/test_shutdown_pipeline_proposals.py
"""Tests for pipeline-enriched proposal generation in ShutdownProcessor."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.shutdown_processor import ShutdownProcessor


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_session_items(n=5):
    """Create N dummy session conversation items."""
    return [
        {"query": f"test query {i}", "response": f"test response {i}"}
        for i in range(n)
    ]


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="")
    return mm


@pytest.fixture
def mock_chroma_store():
    cs = MagicMock()
    cs.query_collection = MagicMock(return_value=[])
    return cs


@pytest.fixture
def mock_memory_coordinator():
    mc = MagicMock()
    mc.get_memories = AsyncMock(return_value=[
        {"content": "User has been working on the proposal system"},
        {"content": "Recent refactoring of memory coordinator"},
    ])
    mc.get_summaries_hybrid = MagicMock(return_value=[
        {"content": "Session focused on proposal architecture"},
    ])
    mc.get_reflections_hybrid = AsyncMock(return_value=[
        {"content": "Good progress on modularization"},
    ])
    mc.get_skills = AsyncMock(return_value=[
        {"metadata": {"trigger": "debugging async code", "action_pattern": "add logging first"}},
    ])
    mc.get_facts = AsyncMock(return_value=[
        {"content": "User prefers Pydantic BaseModel"},
    ])
    mc.gate_system = None
    mc.time_manager = None
    return mc


@pytest.fixture
def mock_user_profile():
    up = MagicMock()
    up.get_context_injection = MagicMock(return_value="Name: Luke | Interests: AI, Python")
    return up


@pytest.fixture
def shutdown_processor(mock_model_manager, mock_chroma_store, mock_memory_coordinator, mock_user_profile):
    return ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=mock_chroma_store,
        consolidator=MagicMock(),
        fact_extractor=MagicMock(),
        model_manager=mock_model_manager,
        user_profile=mock_user_profile,
        storage=MagicMock(),
        session_start=datetime.now(),
        memory_coordinator=mock_memory_coordinator,
    )


@pytest.fixture
def shutdown_processor_no_mc(mock_model_manager, mock_chroma_store, mock_user_profile):
    """ShutdownProcessor without memory_coordinator (fallback mode)."""
    return ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=mock_chroma_store,
        consolidator=MagicMock(),
        fact_extractor=MagicMock(),
        model_manager=mock_model_manager,
        user_profile=mock_user_profile,
        storage=MagicMock(),
        session_start=datetime.now(),
        memory_coordinator=None,
    )


# ------------------------------------------------------------------
# _gather_proposal_context tests
# ------------------------------------------------------------------


class TestGatherProposalContext:
    @pytest.mark.asyncio
    async def test_returns_string(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_includes_semantic_memories(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Relevant Memories" in result
        assert "proposal system" in result

    @pytest.mark.asyncio
    async def test_includes_summaries(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Session Summaries" in result
        assert "proposal architecture" in result

    @pytest.mark.asyncio
    async def test_includes_reflections(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Past Reflections" in result
        assert "modularization" in result

    @pytest.mark.asyncio
    async def test_includes_skills(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Problem-Solving Patterns" in result
        assert "debugging async code" in result

    @pytest.mark.asyncio
    async def test_includes_user_facts(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Known User Facts" in result
        assert "Pydantic BaseModel" in result

    @pytest.mark.asyncio
    async def test_includes_user_profile(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "User Profile" in result
        assert "Luke" in result

    @pytest.mark.asyncio
    async def test_includes_conversation(self, shutdown_processor):
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Recent Conversation" in result
        assert "test query" in result

    @pytest.mark.asyncio
    async def test_includes_git_commits(self, shutdown_processor):
        shutdown_processor.chroma_store.query_collection.return_value = [
            {"content": "feat: Add proposal system"},
        ]
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert "Recent Git Activity" in result
        assert "proposal system" in result

    @pytest.mark.asyncio
    async def test_handles_empty_memories_gracefully(self, shutdown_processor):
        shutdown_processor.memory_coordinator.get_memories = AsyncMock(return_value=[])
        shutdown_processor.memory_coordinator.get_summaries_hybrid = MagicMock(return_value=[])
        shutdown_processor.memory_coordinator.get_reflections_hybrid = AsyncMock(return_value=[])
        shutdown_processor.memory_coordinator.get_skills = AsyncMock(return_value=[])
        shutdown_processor.memory_coordinator.get_facts = AsyncMock(return_value=[])
        shutdown_processor.user_profile.get_context_injection.return_value = ""

        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert isinstance(result, str)
        # Should still have conversation section
        assert "Recent Conversation" in result

    @pytest.mark.asyncio
    async def test_handles_retrieval_exceptions(self, shutdown_processor):
        shutdown_processor.memory_coordinator.get_memories = AsyncMock(side_effect=Exception("DB error"))
        shutdown_processor.memory_coordinator.get_skills = AsyncMock(side_effect=Exception("DB error"))
        # Should not raise, just skip failed sections
        result = await shutdown_processor._gather_proposal_context(_make_session_items())
        assert isinstance(result, str)


# ------------------------------------------------------------------
# _generate_proposals pipeline vs cold fallback tests
# ------------------------------------------------------------------


class TestGenerateProposalsPipeline:
    @pytest.mark.asyncio
    @patch("memory.shutdown_processor.ShutdownProcessor._gather_proposal_context", new_callable=AsyncMock)
    async def test_uses_pipeline_when_coordinator_available(
        self, mock_gather, shutdown_processor
    ):
        """When memory_coordinator exists, should attempt pipeline generation."""
        mock_gather.return_value = "## Rich Context\n- memories"

        mock_gen = MagicMock()
        mock_gen.generate_proposals_with_context = AsyncMock(return_value=[
            MagicMock(title="Pipeline Proposal")
        ])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.return_value = None
        mock_store.store_proposal.return_value = "doc-123"

        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                await shutdown_processor._generate_proposals(_make_session_items())

                mock_gather.assert_called_once()
                mock_gen.generate_proposals_with_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_cold_without_coordinator(self, shutdown_processor_no_mc):
        """Without memory_coordinator, should use cold generation."""
        mock_gen = MagicMock()
        mock_gen.generate_proposals = AsyncMock(return_value=[
            MagicMock(title="Cold Proposal")
        ])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.return_value = None
        mock_store.store_proposal.return_value = "doc-456"

        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                await shutdown_processor_no_mc._generate_proposals(_make_session_items())

                mock_gen.generate_proposals.assert_called_once()

    @pytest.mark.asyncio
    @patch("memory.shutdown_processor.ShutdownProcessor._gather_proposal_context", new_callable=AsyncMock)
    async def test_falls_back_on_pipeline_failure(
        self, mock_gather, shutdown_processor
    ):
        """If pipeline context gathering fails, should fall back to cold."""
        mock_gather.side_effect = Exception("Pipeline error")

        mock_gen = MagicMock()
        mock_gen.generate_proposals = AsyncMock(return_value=[
            MagicMock(title="Fallback Proposal")
        ])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.return_value = None
        mock_store.store_proposal.return_value = "doc-789"

        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                # Should not raise
                await shutdown_processor._generate_proposals(_make_session_items())

                mock_gen.generate_proposals.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_if_too_few_sessions(self, shutdown_processor):
        """Should skip if fewer than 3 session items."""
        sess = _make_session_items(2)
        # Should return without error
        await shutdown_processor._generate_proposals(sess)
        # No LLM call should have been made
        shutdown_processor.model_manager.generate_once.assert_not_called()


# ------------------------------------------------------------------
# _generate_proposals_cold tests
# ------------------------------------------------------------------


class TestGenerateProposalsCold:
    @pytest.mark.asyncio
    async def test_builds_excerpts_from_session(self, shutdown_processor):
        generator = MagicMock()
        generator.generate_proposals = AsyncMock(return_value=[])

        await shutdown_processor._generate_proposals_cold(
            _make_session_items(), generator, ""
        )

        call_args = generator.generate_proposals.call_args
        extra = call_args[1]["extra_context"]
        assert "Recent Conversation" in extra
        assert "test query" in extra

    @pytest.mark.asyncio
    async def test_includes_dedup_context(self, shutdown_processor):
        generator = MagicMock()
        generator.generate_proposals = AsyncMock(return_value=[])

        await shutdown_processor._generate_proposals_cold(
            _make_session_items(), generator, "- [pending] Existing proposal"
        )

        call_args = generator.generate_proposals.call_args
        extra = call_args[1]["extra_context"]
        assert "Existing Proposals" in extra
        assert "Existing proposal" in extra

    @pytest.mark.asyncio
    async def test_empty_excerpts_returns_empty(self, shutdown_processor):
        generator = MagicMock()
        generator.generate_proposals = AsyncMock(return_value=[])

        # Items with no query or response
        empty_items = [{"query": "", "response": ""}] * 5
        result = await shutdown_processor._generate_proposals_cold(
            empty_items, generator, ""
        )
        assert result == []
        generator.generate_proposals.assert_not_called()
