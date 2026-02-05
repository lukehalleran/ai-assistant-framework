# tests/unit/test_proposal_generator.py
"""Unit tests for knowledge.proposal_generator."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.proposal_generator import GoalDirectedGenerator
from memory.code_proposal import CodeProposal, ProposalType


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_model_manager():
    """Mock model manager with async generate_once."""
    mm = MagicMock()
    mm.generate_once = AsyncMock()
    return mm


@pytest.fixture
def generator(mock_model_manager):
    return GoalDirectedGenerator(
        model_manager=mock_model_manager,
        repo_path=".",
        model_alias="test-model",
    )


# ------------------------------------------------------------------
# _parse_response tests
# ------------------------------------------------------------------


class TestParseResponse:
    def test_empty_input(self, generator):
        assert generator._parse_response("") == []

    def test_garbage_input(self, generator):
        assert generator._parse_response("this is not json at all") == []

    def test_single_json_object(self, generator):
        data = '{"title": "Test", "proposal_type": "feature"}'
        result = generator._parse_response(data)
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    def test_json_array(self, generator):
        data = json.dumps([
            {"title": "A", "proposal_type": "feature"},
            {"title": "B", "proposal_type": "refactor"},
        ])
        result = generator._parse_response(data)
        assert len(result) == 2
        assert result[0]["title"] == "A"
        assert result[1]["title"] == "B"

    def test_line_delimited_json(self, generator):
        data = (
            '{"title": "First", "proposal_type": "feature"}\n'
            '{"title": "Second", "proposal_type": "bugfix"}\n'
        )
        result = generator._parse_response(data)
        assert len(result) == 2

    def test_code_fenced_json(self, generator):
        data = (
            "Here are the proposals:\n"
            "```json\n"
            '[{"title": "Fenced", "proposal_type": "test"}]\n'
            "```\n"
        )
        result = generator._parse_response(data)
        assert len(result) == 1
        assert result[0]["title"] == "Fenced"

    def test_code_fenced_line_delimited(self, generator):
        data = (
            "```\n"
            '{"title": "A"}\n'
            '{"title": "B"}\n'
            "```\n"
        )
        result = generator._parse_response(data)
        assert len(result) == 2

    def test_mixed_valid_invalid_lines(self, generator):
        data = (
            "Some intro text\n"
            '{"title": "Valid"}\n'
            "not json\n"
            '{"title": "Also valid"}\n'
        )
        result = generator._parse_response(data)
        assert len(result) == 2


# ------------------------------------------------------------------
# _parse_proposal tests
# ------------------------------------------------------------------


class TestParseProposal:
    def test_valid_data(self, generator):
        data = {
            "title": "Add caching",
            "proposal_type": "feature",
            "priority": 8,
            "reasoning": "Speed improvement",
            "description": "Add Redis caching",
            "implementation_steps": [
                {"order": 1, "description": "Add redis client", "file_path": "core/cache.py", "action": "create"}
            ],
            "affected_files": ["core/cache.py"],
            "tags": ["performance"],
            "estimated_complexity": "medium",
            "requires_tests": True,
        }
        proposal = generator._parse_proposal(data)
        assert proposal is not None
        assert proposal.title == "Add caching"
        assert proposal.proposal_type == ProposalType.FEATURE
        assert proposal.priority == 8
        assert len(proposal.implementation_steps) == 1
        assert proposal.tags == ["performance"]

    def test_missing_fields_uses_defaults(self, generator):
        data = {"title": "Minimal proposal"}
        proposal = generator._parse_proposal(data)
        assert proposal is not None
        assert proposal.title == "Minimal proposal"
        assert proposal.proposal_type == ProposalType.FEATURE
        assert proposal.priority == 5

    def test_garbage_data_returns_none(self, generator):
        assert generator._parse_proposal({}) is None
        assert generator._parse_proposal({"title": ""}) is None
        assert generator._parse_proposal({"title": "ab"}) is None  # too short

    def test_invalid_proposal_type_defaults_to_feature(self, generator):
        data = {"title": "Test proposal", "proposal_type": "invalid_type"}
        proposal = generator._parse_proposal(data)
        assert proposal is not None
        assert proposal.proposal_type == ProposalType.FEATURE

    def test_invalid_complexity_defaults_to_medium(self, generator):
        data = {"title": "Test proposal", "estimated_complexity": "extreme"}
        proposal = generator._parse_proposal(data)
        assert proposal.estimated_complexity == "medium"

    def test_priority_clamped(self, generator):
        data = {"title": "Test proposal", "priority": 99}
        proposal = generator._parse_proposal(data)
        assert proposal.priority == 10

        data2 = {"title": "Test proposal", "priority": -5}
        proposal2 = generator._parse_proposal(data2)
        assert proposal2.priority == 1

    def test_tags_limited_to_10(self, generator):
        data = {"title": "Test proposal", "tags": [f"tag{i}" for i in range(20)]}
        proposal = generator._parse_proposal(data)
        assert len(proposal.tags) == 10


# ------------------------------------------------------------------
# gather_context tests
# ------------------------------------------------------------------


class TestGatherContext:
    def test_returns_dict_with_expected_keys(self, generator):
        context = generator.gather_context()
        assert "skeleton" in context
        assert "goals" in context
        assert "recent_commits" in context

    def test_handles_missing_files(self):
        gen = GoalDirectedGenerator(repo_path="/nonexistent/path")
        context = gen.gather_context()
        assert context["skeleton"] == ""
        assert context["goals"] == ""


# ------------------------------------------------------------------
# generate_proposals tests
# ------------------------------------------------------------------


class TestGenerateProposals:
    @pytest.mark.asyncio
    async def test_no_model_manager_returns_empty(self):
        gen = GoalDirectedGenerator(model_manager=None)
        result = await gen.generate_proposals()
        assert result == []

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_model_manager):
        response = json.dumps([
            {
                "title": "Add error handling",
                "proposal_type": "bugfix",
                "priority": 7,
                "reasoning": "Unhandled exceptions in API",
                "description": "Add try/except blocks",
                "implementation_steps": [
                    {"order": 1, "description": "Wrap API calls", "file_path": "core/api.py", "action": "modify"}
                ],
                "affected_files": ["core/api.py"],
                "tags": ["error-handling"],
                "estimated_complexity": "low",
                "requires_tests": True,
            }
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager,
            repo_path=".",
        )
        proposals = await gen.generate_proposals(extra_context="Test context")

        assert len(proposals) == 1
        assert proposals[0].title == "Add error handling"
        assert proposals[0].proposal_type == ProposalType.BUGFIX
        assert proposals[0].priority == 7
        mock_model_manager.generate_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_empty(self, mock_model_manager):
        mock_model_manager.generate_once.return_value = ""
        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        result = await gen.generate_proposals(extra_context="something")
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_returns_garbage(self, mock_model_manager):
        mock_model_manager.generate_once.return_value = "I don't know how to help with that"
        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        result = await gen.generate_proposals(extra_context="something")
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self, mock_model_manager):
        mock_model_manager.generate_once.side_effect = Exception("API error")
        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        result = await gen.generate_proposals(extra_context="something")
        assert result == []

    @pytest.mark.asyncio
    async def test_max_proposals_respected(self, mock_model_manager):
        response = json.dumps([
            {"title": f"Proposal {i}", "proposal_type": "feature"}
            for i in range(10)
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager,
            repo_path=".",
            max_proposals=3,
        )
        proposals = await gen.generate_proposals(extra_context="context")
        assert len(proposals) == 3
