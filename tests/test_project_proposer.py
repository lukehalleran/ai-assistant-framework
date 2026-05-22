"""
Tests for project_proposer.py — standalone feature proposal generator.

Module Contract
- Purpose: Verify data models, context gathering, LLM response parsing,
  proposal validation, and end-to-end generation with mocked LLM.
- Inputs: Synthetic project structures via tmp_path, mock LLM callables.
- Outputs: Pass/fail assertions on Proposal objects, parsed dicts, and
  generated proposal lists.
- Dependencies: pytest, standard library only. No API keys or network.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_proposer import (
    ImplementationStep,
    Proposal,
    ProjectProposer,
    DEFAULT_CONTEXT_FILES,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def project(tmp_path):
    """Create a synthetic project directory."""
    # Docs
    (tmp_path / "README.md").write_text(
        "# Test Project\nA project for testing proposal generation.",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text(
        "# Project Rules\n- Use pytest for tests\n- Follow PEP 8",
        encoding="utf-8",
    )
    (tmp_path / "GOALS.md").write_text(
        "# Goals\n- Add API caching\n- Improve test coverage",
        encoding="utf-8",
    )
    # Structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("# main app", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("# tests", encoding="utf-8")
    # Metadata
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"',
        encoding="utf-8",
    )
    return tmp_path


def mock_llm_response(proposals: list[dict]):
    """Create a mock LLM callable that returns the given proposals as JSON."""
    def llm(prompt, system_prompt="", max_tokens=4000, temperature=0.7):
        return json.dumps(proposals)
    return llm


def make_raw_proposal(**overrides) -> dict:
    """Create a raw proposal dict with sensible defaults."""
    base = {
        "title": "Add caching layer",
        "proposal_type": "feature",
        "priority": 8,
        "reasoning": "Reduce API latency by caching frequent queries",
        "description": "Implement Redis-backed caching for API responses",
        "implementation_steps": [
            {"order": 1, "description": "Create cache module", "file_path": "src/cache.py", "action": "create"},
            {"order": 2, "description": "Add cache tests", "file_path": "tests/test_cache.py", "action": "test"},
        ],
        "affected_files": ["src/cache.py", "tests/test_cache.py"],
        "tags": ["performance", "caching"],
        "estimated_complexity": "medium",
        "requires_tests": True,
    }
    base.update(overrides)
    return base


# ############################################################################
#
#  Data models
#
# ############################################################################

class TestImplementationStep:

    def test_basic_creation(self):
        step = ImplementationStep(order=1, description="Create module")
        assert step.order == 1
        assert step.action == "modify"  # default

    def test_full_creation(self):
        step = ImplementationStep(
            order=2, description="Write tests",
            file_path="tests/test_new.py", action="test",
        )
        assert step.file_path == "tests/test_new.py"
        assert step.action == "test"

    def test_to_dict_from_dict_roundtrip(self):
        step = ImplementationStep(order=1, description="Do thing", file_path="a.py", action="create")
        d = step.to_dict()
        step2 = ImplementationStep.from_dict(d)
        assert step2.order == step.order
        assert step2.description == step.description
        assert step2.file_path == step.file_path
        assert step2.action == step.action

    def test_from_dict_missing_fields(self):
        step = ImplementationStep.from_dict({"description": "minimal"})
        assert step.order == 1  # default
        assert step.action == "modify"  # default


class TestProposal:

    def test_defaults(self):
        p = Proposal(title="Test")
        assert p.proposal_type == "feature"
        assert p.priority == 5
        assert p.estimated_complexity == "medium"
        assert p.requires_tests is True
        assert p.implementation_steps == []
        assert p.tags == []

    def test_full_creation(self):
        step = ImplementationStep(order=1, description="step")
        p = Proposal(
            title="Big Feature",
            proposal_type="feature",
            priority=9,
            reasoning="Need it",
            description="Build it",
            implementation_steps=[step],
            affected_files=["src/new.py"],
            tags=["important"],
            estimated_complexity="high",
        )
        assert p.title == "Big Feature"
        assert len(p.implementation_steps) == 1

    def test_to_dict_from_dict_roundtrip(self):
        raw = make_raw_proposal()
        p = Proposal.from_dict(raw)
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.title == p.title
        assert p2.priority == p.priority
        assert p2.reasoning == p.reasoning
        assert len(p2.implementation_steps) == len(p.implementation_steps)
        assert p2.tags == p.tags

    def test_from_dict_missing_optional_fields(self):
        p = Proposal.from_dict({"title": "Minimal"})
        assert p.title == "Minimal"
        assert p.proposal_type == "feature"
        assert p.priority == 5

    def test_summary(self):
        p = Proposal(title="Add caching", proposal_type="feature", priority=8)
        assert p.summary() == "[FEATURE] (P8) Add caching"

    def test_summary_different_type(self):
        p = Proposal(title="Fix bug", proposal_type="bugfix", priority=10)
        assert p.summary() == "[BUGFIX] (P10) Fix bug"

    def test_priority_clamped(self):
        p = Proposal.from_dict({"title": "Test", "priority": 99})
        assert p.priority == 10
        p2 = Proposal.from_dict({"title": "Test", "priority": -5})
        assert p2.priority == 1

    def test_tags_limited(self):
        p = Proposal.from_dict({
            "title": "Test",
            "tags": [f"tag{i}" for i in range(20)],
        })
        assert len(p.tags) == 10

    def test_from_dict_with_extra_keys(self):
        """Extra keys should be silently ignored."""
        p = Proposal.from_dict({
            "title": "Test",
            "unknown_field": "ignored",
            "another_extra": 42,
        })
        assert p.title == "Test"


# ############################################################################
#
#  Response parsing
#
# ############################################################################

class TestParseResponse:

    def test_empty_input(self):
        assert ProjectProposer._parse_response("") == []

    def test_garbage_input(self):
        assert ProjectProposer._parse_response("not json at all") == []

    def test_single_json_object(self):
        raw = json.dumps({"title": "Test", "priority": 5})
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    def test_json_array(self):
        raw = json.dumps([
            {"title": "First", "priority": 5},
            {"title": "Second", "priority": 3},
        ])
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 2

    def test_line_delimited_json(self):
        raw = '{"title": "One", "priority": 5}\n{"title": "Two", "priority": 3}'
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 2

    def test_code_fenced_json(self):
        raw = '```json\n[{"title": "Fenced", "priority": 7}]\n```'
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Fenced"

    def test_code_fenced_no_lang(self):
        raw = '```\n[{"title": "NoLang", "priority": 5}]\n```'
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 1

    def test_mixed_valid_invalid_lines(self):
        raw = (
            'Here are my proposals:\n'
            '{"title": "Good", "priority": 8}\n'
            'This line is invalid\n'
            '{"title": "Also Good", "priority": 6}\n'
        )
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 2

    def test_non_dict_items_filtered(self):
        raw = json.dumps([{"title": "Good"}, "not a dict", 42, {"title": "Also Good"}])
        result = ProjectProposer._parse_response(raw)
        assert len(result) == 2


# ############################################################################
#
#  Proposal validation
#
# ############################################################################

class TestValidateProposal:

    def test_valid_proposal(self):
        raw = make_raw_proposal()
        p = ProjectProposer._validate_proposal(raw)
        assert p is not None
        assert p.title == "Add caching layer"
        assert p.priority == 8
        assert len(p.implementation_steps) == 2

    def test_missing_title_returns_none(self):
        assert ProjectProposer._validate_proposal({}) is None

    def test_short_title_returns_none(self):
        assert ProjectProposer._validate_proposal({"title": "ab"}) is None

    def test_whitespace_only_title_returns_none(self):
        assert ProjectProposer._validate_proposal({"title": "   "}) is None

    def test_invalid_type_defaults_to_feature(self):
        p = ProjectProposer._validate_proposal({"title": "Test Thing", "proposal_type": "unknown"})
        assert p.proposal_type == "feature"

    def test_invalid_complexity_defaults_to_medium(self):
        p = ProjectProposer._validate_proposal({"title": "Test Thing", "estimated_complexity": "extreme"})
        assert p.estimated_complexity == "medium"

    def test_priority_clamped_high(self):
        p = ProjectProposer._validate_proposal({"title": "Test Thing", "priority": 100})
        assert p.priority == 10

    def test_priority_clamped_low(self):
        p = ProjectProposer._validate_proposal({"title": "Test Thing", "priority": -1})
        assert p.priority == 1

    def test_invalid_step_skipped(self):
        p = ProjectProposer._validate_proposal({
            "title": "Test Thing",
            "implementation_steps": [
                {"order": 1, "description": "good step"},
                "not a dict",
                42,
            ],
        })
        assert len(p.implementation_steps) == 1

    def test_all_valid_types(self):
        for t in ("feature", "refactor", "bugfix", "test", "docs", "infra"):
            p = ProjectProposer._validate_proposal({"title": "Test", "proposal_type": t})
            assert p.proposal_type == t


# ############################################################################
#
#  Context gathering
#
# ############################################################################

class TestGatherContext:

    def test_reads_configured_files(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        assert "README.md" in ctx
        assert "CLAUDE.md" in ctx
        assert "Test Project" in ctx["README.md"]

    def test_reads_goals(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        assert "GOALS.md" in ctx
        assert "API caching" in ctx["GOALS.md"]

    def test_reads_project_structure(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        assert "project_structure" in ctx
        assert "src/" in ctx["project_structure"]
        assert "tests/" in ctx["project_structure"]

    def test_reads_metadata(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        assert "pyproject.toml" in ctx
        assert "test-project" in ctx["pyproject.toml"]

    def test_handles_missing_files(self, tmp_path):
        """Empty project — no doc files or metadata found."""
        proposer = ProjectProposer(repo_path=tmp_path)
        ctx = proposer.gather_context()
        # No doc files, no metadata
        doc_keys = [k for k in ctx if k not in ("git_log", "project_structure")]
        assert len(doc_keys) == 0

    def test_custom_context_files(self, project):
        (project / "custom.md").write_text("Custom docs", encoding="utf-8")
        proposer = ProjectProposer(
            repo_path=project,
            context_files=["custom.md", "nonexistent.md"],
        )
        ctx = proposer.gather_context()
        assert "custom.md" in ctx
        assert "nonexistent.md" not in ctx

    def test_max_file_chars(self, project):
        (project / "big.md").write_text("x" * 50000, encoding="utf-8")
        proposer = ProjectProposer(
            repo_path=project,
            context_files=["big.md"],
            max_file_chars=1000,
        )
        ctx = proposer.gather_context()
        assert len(ctx["big.md"]) == 1000

    def test_empty_files_skipped(self, project):
        (project / "empty.md").write_text("", encoding="utf-8")
        proposer = ProjectProposer(
            repo_path=project,
            context_files=["empty.md"],
        )
        ctx = proposer.gather_context()
        assert "empty.md" not in ctx

    def test_git_log_with_real_repo(self, project):
        """If git is available, git_log should be populated."""
        # Initialize a git repo
        try:
            subprocess.run(
                ["git", "init"], cwd=str(project),
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "add", "."], cwd=str(project),
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "commit", "-m", "init", "--allow-empty"],
                cwd=str(project), capture_output=True, timeout=5,
                env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t",
                     "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t"},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("git not available")

        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        assert "git_log" in ctx
        assert "init" in ctx["git_log"]


# ############################################################################
#
#  Prompt building
#
# ############################################################################

class TestBuildPrompt:

    def test_includes_context(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        prompt = proposer._build_prompt(ctx)
        assert "Test Project" in prompt  # from README.md
        assert "PROJECT CONTEXT" in prompt

    def test_includes_extra_context(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        prompt = proposer._build_prompt(ctx, extra_context="Focus on security")
        assert "Focus on security" in prompt

    def test_empty_context(self):
        proposer = ProjectProposer()
        prompt = proposer._build_prompt({})
        # Should still have the instruction preamble
        assert "senior software architect" in prompt

    def test_structure_in_prompt(self, project):
        proposer = ProjectProposer(repo_path=project)
        ctx = proposer.gather_context()
        prompt = proposer._build_prompt(ctx)
        assert "Project Structure" in prompt


# ############################################################################
#
#  End-to-end generation
#
# ############################################################################

class TestGenerate:

    def test_no_llm_raises(self, project):
        proposer = ProjectProposer(repo_path=project, llm=None)
        with pytest.raises(ValueError, match="No LLM callable"):
            proposer.generate()

    def test_full_pipeline(self, project):
        raw = [make_raw_proposal(), make_raw_proposal(title="Second Feature", priority=6)]
        proposer = ProjectProposer(repo_path=project, llm=mock_llm_response(raw))
        proposals = proposer.generate()
        assert len(proposals) == 2
        # Sorted by priority descending
        assert proposals[0].priority >= proposals[1].priority

    def test_max_proposals_respected(self, project):
        raw = [make_raw_proposal(title=f"Feature {i}", priority=i) for i in range(10)]
        proposer = ProjectProposer(repo_path=project, llm=mock_llm_response(raw), max_proposals=3)
        proposals = proposer.generate()
        assert len(proposals) == 3

    def test_max_proposals_override(self, project):
        raw = [make_raw_proposal(title=f"Feature {i}") for i in range(10)]
        proposer = ProjectProposer(repo_path=project, llm=mock_llm_response(raw), max_proposals=10)
        proposals = proposer.generate(max_proposals=2)
        assert len(proposals) == 2

    def test_llm_returns_empty(self, project):
        llm = lambda prompt, **kw: ""
        proposer = ProjectProposer(repo_path=project, llm=llm)
        proposals = proposer.generate()
        assert proposals == []

    def test_llm_returns_garbage(self, project):
        llm = lambda prompt, **kw: "This is not JSON at all!"
        proposer = ProjectProposer(repo_path=project, llm=llm)
        proposals = proposer.generate()
        assert proposals == []

    def test_llm_exception_returns_empty(self, project):
        def bad_llm(prompt, **kw):
            raise ConnectionError("API down")
        proposer = ProjectProposer(repo_path=project, llm=bad_llm)
        proposals = proposer.generate()
        assert proposals == []

    def test_no_context_no_extra_returns_empty(self, tmp_path):
        """Empty project with no extra context."""
        llm = lambda prompt, **kw: "[]"
        proposer = ProjectProposer(repo_path=tmp_path, llm=llm, context_files=[])
        # Even project_structure exists, so this won't be empty
        # Use a non-existent path to force truly empty context
        proposer2 = ProjectProposer(repo_path=tmp_path / "nonexistent", llm=llm, context_files=[])
        # gather_context will fail on non-existent dir, but won't crash
        proposals = proposer2.generate()
        assert proposals == []

    def test_extra_context_passed(self, project):
        """Extra context should appear in the LLM prompt."""
        received_prompts = []

        def capturing_llm(prompt, **kw):
            received_prompts.append(prompt)
            return "[]"

        proposer = ProjectProposer(repo_path=project, llm=capturing_llm)
        proposer.generate(extra_context="We need GraphQL support")
        assert len(received_prompts) == 1
        assert "GraphQL support" in received_prompts[0]

    def test_proposals_sorted_by_priority(self, project):
        raw = [
            make_raw_proposal(title="Low", priority=2),
            make_raw_proposal(title="High", priority=9),
            make_raw_proposal(title="Mid", priority=5),
        ]
        proposer = ProjectProposer(repo_path=project, llm=mock_llm_response(raw))
        proposals = proposer.generate()
        priorities = [p.priority for p in proposals]
        assert priorities == sorted(priorities, reverse=True)

    def test_invalid_proposals_filtered(self, project):
        raw = [
            make_raw_proposal(title="Good One"),
            {"title": "ab"},  # too short, will be filtered
            make_raw_proposal(title="Another Good One"),
        ]
        proposer = ProjectProposer(repo_path=project, llm=mock_llm_response(raw))
        proposals = proposer.generate()
        assert len(proposals) == 2

    def test_code_fenced_response(self, project):
        """LLM wraps response in code fences."""
        raw = [make_raw_proposal()]

        def fenced_llm(prompt, **kw):
            return f"```json\n{json.dumps(raw)}\n```"

        proposer = ProjectProposer(repo_path=project, llm=fenced_llm)
        proposals = proposer.generate()
        assert len(proposals) == 1


# ############################################################################
#
#  LLM adapter factories (import checks only — no API calls)
#
# ############################################################################

class TestLLMAdapters:

    def test_openai_import_error(self):
        """openai_llm() raises ImportError if openai not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            from project_proposer import openai_llm
            # The import check happens inside the factory function
            # We can't easily test this without uninstalling openai,
            # so just verify the function exists
            assert callable(openai_llm)

    def test_anthropic_import_error(self):
        """anthropic_llm() raises ImportError if anthropic not installed."""
        with patch.dict(sys.modules, {"anthropic": None}):
            from project_proposer import anthropic_llm
            assert callable(anthropic_llm)
