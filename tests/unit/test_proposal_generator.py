# tests/unit/test_proposal_generator.py
"""Unit tests for knowledge.proposal_generator."""

import json
from unittest.mock import AsyncMock, MagicMock

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
        assert "claude_md" in context
        assert "quick_reference" in context

    def test_handles_missing_files(self):
        gen = GoalDirectedGenerator(repo_path="/nonexistent/path")
        context = gen.gather_context()
        assert context["skeleton"] == ""
        assert context["goals"] == ""
        assert context["claude_md"] == ""
        assert context["quick_reference"] == ""

    def test_loads_claude_md(self, tmp_path):
        """Should load CLAUDE.md from repo root."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Project\n## Architecture\nSome architecture info")
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "Architecture" in context["claude_md"]
        assert "Some architecture info" in context["claude_md"]

    def test_loads_quick_reference(self, tmp_path):
        """Should load QUICK_REFERENCE.md from docs/ subdirectory."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        quick_ref = docs_dir / "QUICK_REFERENCE.md"
        quick_ref.write_text("# Quick Reference\n## Core Entry Point\nclass Orchestrator")
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "Quick Reference" in context["quick_reference"]
        assert "Core Entry Point" in context["quick_reference"]

    def test_claude_md_not_truncated(self, tmp_path):
        """Should load full CLAUDE.md without truncation."""
        claude_md = tmp_path / "CLAUDE.md"
        content = "X" * 10000
        claude_md.write_text(content)
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert len(context["claude_md"]) == 10000

    def test_quick_reference_not_truncated(self, tmp_path):
        """Should load full QUICK_REFERENCE.md without truncation."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        quick_ref = docs_dir / "QUICK_REFERENCE.md"
        content = "Y" * 40000
        quick_ref.write_text(content)
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert len(context["quick_reference"]) == 40000

    def test_claude_md_fallback_to_docs(self, tmp_path):
        """Should check docs/CLAUDE.md if root CLAUDE.md doesn't exist."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        claude_md = docs_dir / "CLAUDE.md"
        claude_md.write_text("# Fallback CLAUDE.md")
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "Fallback CLAUDE.md" in context["claude_md"]

    def test_quick_reference_fallback_to_root(self, tmp_path):
        """Should check root QUICK_REFERENCE.md if docs/ version doesn't exist."""
        quick_ref = tmp_path / "QUICK_REFERENCE.md"
        quick_ref.write_text("# Root Quick Reference")
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "Root Quick Reference" in context["quick_reference"]

    def test_skeleton_filters_core_components(self, tmp_path):
        """Should remove 'Core Components' section from skeleton."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        skeleton = docs_dir / "PROJECT_SKELETON.md"
        skeleton.write_text(
            "## 1. Architecture Overview\nArch content here\n\n"
            "## 2. Core Components\nHuge per-method docs\nMore details\n\n"
            "## 3. Configuration\nConfig content here\n"
        )
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "Arch content here" in context["skeleton"]
        assert "Config content here" in context["skeleton"]
        assert "Huge per-method docs" not in context["skeleton"]
        assert "Core Components" not in context["skeleton"]

    def test_skeleton_no_filter_without_core_components(self, tmp_path):
        """Skeleton without 'Core Components' should be returned in full."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        skeleton = docs_dir / "PROJECT_SKELETON.md"
        content = "## 1. Overview\nAll content\n## 3. Config\nMore content"
        skeleton.write_text(content)
        gen = GoalDirectedGenerator(repo_path=str(tmp_path))
        context = gen.gather_context()
        assert "All content" in context["skeleton"]
        assert "More content" in context["skeleton"]


# ------------------------------------------------------------------
# _build_prompt tests
# ------------------------------------------------------------------


class TestBuildPrompt:
    def test_claude_md_in_prompt(self, generator):
        """CLAUDE.md content should appear in prompt under 'Project Conventions'."""
        context = {
            "skeleton": "",
            "goals": "",
            "recent_commits": "",
            "claude_md": "## Architecture\nUse async/await everywhere",
            "quick_reference": "",
        }
        prompt = generator._build_prompt(context)
        assert "Project Conventions (CLAUDE.md)" in prompt
        assert "Use async/await everywhere" in prompt

    def test_quick_reference_in_prompt(self, generator):
        """QUICK_REFERENCE.md content should appear in prompt under 'API Quick Reference'."""
        context = {
            "skeleton": "",
            "goals": "",
            "recent_commits": "",
            "claude_md": "",
            "quick_reference": "class MemoryCoordinator:\n    async def get_memories()",
        }
        prompt = generator._build_prompt(context)
        assert "API Quick Reference" in prompt
        assert "MemoryCoordinator" in prompt

    def test_section_order(self, generator):
        """CLAUDE.md and quick reference should appear before skeleton and goals."""
        context = {
            "skeleton": "SKELETON_MARKER",
            "goals": "GOALS_MARKER",
            "recent_commits": "COMMITS_MARKER",
            "claude_md": "CLAUDE_MD_MARKER",
            "quick_reference": "QUICK_REF_MARKER",
        }
        prompt = generator._build_prompt(context)
        # CLAUDE.md and quick ref should come before skeleton and goals
        claude_pos = prompt.index("CLAUDE_MD_MARKER")
        quick_pos = prompt.index("QUICK_REF_MARKER")
        skeleton_pos = prompt.index("SKELETON_MARKER")
        goals_pos = prompt.index("GOALS_MARKER")
        assert claude_pos < skeleton_pos
        assert quick_pos < skeleton_pos
        assert claude_pos < goals_pos

    def test_empty_context_omits_sections(self, generator):
        """Empty context values should not produce section headers."""
        context = {
            "skeleton": "",
            "goals": "",
            "recent_commits": "",
            "claude_md": "",
            "quick_reference": "",
        }
        prompt = generator._build_prompt(context)
        assert "Project Conventions" not in prompt
        assert "API Quick Reference" not in prompt
        assert "Project Architecture" not in prompt


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


# ------------------------------------------------------------------
# generate_proposals_with_context tests
# ------------------------------------------------------------------


class TestGenerateProposalsWithContext:
    @pytest.mark.asyncio
    async def test_no_model_manager_returns_empty(self):
        gen = GoalDirectedGenerator(model_manager=None)
        result = await gen.generate_proposals_with_context(
            pipeline_context="## Relevant Memories\n- User worked on memory system"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_pipeline_context_passed_to_prompt(self, mock_model_manager):
        """Pipeline context should appear in the prompt sent to the LLM."""
        mock_model_manager.generate_once.return_value = ""
        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager, repo_path="."
        )
        await gen.generate_proposals_with_context(
            pipeline_context="## Relevant Memories\n- Memory about refactoring"
        )
        call_args = mock_model_manager.generate_once.call_args
        prompt_text = call_args[0][0]
        assert "Relevant Memories" in prompt_text
        assert "Memory about refactoring" in prompt_text

    @pytest.mark.asyncio
    async def test_extra_context_merged(self, mock_model_manager):
        """Both pipeline_context and extra_context should appear in prompt."""
        mock_model_manager.generate_once.return_value = ""
        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager, repo_path="."
        )
        await gen.generate_proposals_with_context(
            pipeline_context="## Pipeline Data\n- item 1",
            extra_context="## Dedup\n- existing proposal",
        )
        call_args = mock_model_manager.generate_once.call_args
        prompt_text = call_args[0][0]
        assert "Pipeline Data" in prompt_text
        assert "Dedup" in prompt_text

    @pytest.mark.asyncio
    async def test_full_pipeline_with_context(self, mock_model_manager):
        response = json.dumps([{
            "title": "Pipeline-driven proposal",
            "proposal_type": "feature",
            "priority": 8,
            "reasoning": "Based on semantic memories",
            "description": "A feature informed by pipeline context",
            "implementation_steps": [],
            "affected_files": [],
            "tags": ["pipeline"],
            "estimated_complexity": "medium",
            "requires_tests": True,
        }])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager, repo_path="."
        )
        proposals = await gen.generate_proposals_with_context(
            pipeline_context="## Relevant Memories\n- User was working on memory system"
        )

        assert len(proposals) == 1
        assert proposals[0].title == "Pipeline-driven proposal"
        mock_model_manager.generate_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self, mock_model_manager):
        mock_model_manager.generate_once.side_effect = Exception("API error")
        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager, repo_path="."
        )
        result = await gen.generate_proposals_with_context(
            pipeline_context="## Context"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_max_proposals_respected(self, mock_model_manager):
        response = json.dumps([
            {"title": f"Proposal {i}", "proposal_type": "feature"}
            for i in range(10)
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(
            model_manager=mock_model_manager, repo_path=".", max_proposals=2,
        )
        proposals = await gen.generate_proposals_with_context(
            pipeline_context="## Context"
        )
        assert len(proposals) == 2
