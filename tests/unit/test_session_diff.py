"""Tests for Session-Start Codebase Diff feature.

Tests the get_codebase_changes() method in ContextGatherer and the
[CODEBASE CHANGES SINCE LAST SESSION] prompt section.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gatherer():
    """Create a minimal ContextGatherer for testing."""
    from core.prompt.context_gatherer import ContextGatherer

    mc = MagicMock()
    mc.corpus_manager = MagicMock()
    mm = MagicMock()
    tm = MagicMock()
    gs = MagicMock()
    time_mgr = MagicMock()

    gatherer = ContextGatherer(
        memory_coordinator=mc,
        model_manager=mm,
        token_manager=tm,
        gate_system=gs,
        time_manager=time_mgr,
    )
    return gatherer


def _make_subprocess_result(stdout="", returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# Tests: get_codebase_changes()
# ---------------------------------------------------------------------------

class TestGetCodebaseChanges:
    """Tests for ContextGatherer.get_codebase_changes()."""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        """When SESSION_DIFF_ENABLED is False, returns empty dict."""
        gatherer = _make_gatherer()
        with patch("config.app_config.SESSION_DIFF_ENABLED", False):
            result = await gatherer.get_codebase_changes(datetime.now())
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_datetime_returns_empty(self):
        """When since_datetime is None, returns empty dict."""
        gatherer = _make_gatherer()
        with patch("config.app_config.SESSION_DIFF_ENABLED", True):
            result = await gatherer.get_codebase_changes(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_not_git_repo_returns_empty(self):
        """When git rev-parse fails, returns empty dict gracefully."""
        gatherer = _make_gatherer()
        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", return_value=_make_subprocess_result(returncode=128)):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))
        assert result == {}

    @pytest.mark.asyncio
    async def test_committed_changes_parsed(self):
        """Committed changes from git log are returned."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="abc1234 feat: Add login\ndef5678 fix: Bug fix\n")
            if "diff" in cmd:
                return _make_subprocess_result(stdout="")
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))

        assert len(result["committed"]) == 2
        assert "abc1234 feat: Add login" in result["committed"]

    @pytest.mark.asyncio
    async def test_uncommitted_modified_parsed(self):
        """Uncommitted modified files from git diff are returned."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="")
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout="core/orchestrator.py\nmemory/store.py\n")
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))

        assert len(result["uncommitted_modified"]) == 2
        assert "core/orchestrator.py" in result["uncommitted_modified"]

    @pytest.mark.asyncio
    async def test_untracked_files_parsed(self):
        """Untracked new files from git status are returned."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="")
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout="")
            if "status" in cmd:
                return _make_subprocess_result(stdout="?? memory/new_module.py\n?? tests/test_new.py\n M core/old.py\n")
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))

        assert len(result["uncommitted_new"]) == 2
        assert "memory/new_module.py" in result["uncommitted_new"]

    @pytest.mark.asyncio
    async def test_extension_filtering(self):
        """Files with excluded extensions (.pyc, etc.) are filtered out."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="")
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout="core/module.py\ncore/module.pyc\nvenv/lib/thing.py\n__pycache__/cache.py\n")
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))

        # Only core/module.py should pass — .pyc excluded by extension,
        # venv/ and __pycache__ excluded by path pattern
        assert result["uncommitted_modified"] == ["core/module.py"]

    @pytest.mark.asyncio
    async def test_max_limits_enforced(self):
        """Caps on committed and uncommitted results are respected."""
        gatherer = _make_gatherer()

        many_commits = "\n".join([f"abc{i:04d} commit {i}" for i in range(30)])
        many_files = "\n".join([f"file_{i}.py" for i in range(30)])

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout=many_commits)
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout=many_files)
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 5), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 3), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(datetime.now() - timedelta(hours=2))

        assert len(result["committed"]) == 5
        assert len(result["uncommitted_modified"]) == 3

    @pytest.mark.asyncio
    async def test_since_label_formatting_hours(self):
        """Human-readable since_label shows hours+minutes for recent changes."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="abc1234 some change\n")
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout="")
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        since = datetime.now() - timedelta(hours=3, minutes=15)
        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(since)

        assert "3h" in result["since_label"]

    @pytest.mark.asyncio
    async def test_since_label_formatting_days(self):
        """Human-readable since_label shows days+hours for older changes."""
        gatherer = _make_gatherer()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(stdout="abc1234 some change\n")
            if cmd[1] == "diff":
                return _make_subprocess_result(stdout="")
            if "status" in cmd:
                return _make_subprocess_result(stdout="")
            return _make_subprocess_result()

        since = datetime.now() - timedelta(days=2, hours=5)
        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(since)

        assert "2d" in result["since_label"]


class TestCodebaseChangesInPrompt:
    """Tests for the [CODEBASE CHANGES SINCE LAST SESSION] section in assembled prompt."""

    def test_section_in_assembled_prompt(self):
        """When codebase_changes is in context, the section appears in the prompt."""
        from core.prompt.builder import UnifiedPromptBuilder

        mc = MagicMock()
        mc.corpus_manager = MagicMock()
        mm = MagicMock()

        builder = UnifiedPromptBuilder(memory_coordinator=mc, model_manager=mm)

        context = {
            "recent_conversations": [],
            "memories": [],
            "user_profile": "",
            "narrative_state": "",
            "summaries": [],
            "recent_summaries": [],
            "semantic_summaries": [],
            "reflections": [],
            "recent_reflections": [],
            "semantic_reflections": [],
            "dreams": [],
            "semantic_chunks": [],
            "wiki": [],
            "personal_notes": [],
            "user_uploads": [],
            "git_commits": [],
            "procedural_skills": [],
            "proposed_features": [],
            "graph_context": [],
            "unresolved_threads": [],
            "proactive_insights": [],
            "web_search_results": None,
            "codebase_changes": {
                "committed": ["abc1234 feat: Add login"],
                "uncommitted_modified": ["core/orchestrator.py"],
                "uncommitted_new": ["memory/new_thing.py"],
                "since_label": "2h 30m ago",
            },
        }

        prompt = builder._assemble_prompt(context=context, user_input="hello")

        assert "[CODEBASE CHANGES SINCE LAST SESSION]" in prompt
        assert "abc1234 feat: Add login" in prompt
        assert "core/orchestrator.py" in prompt
        assert "memory/new_thing.py" in prompt
        assert "2h 30m ago" in prompt

    def test_no_section_when_empty(self):
        """When codebase_changes is empty, the section does not appear."""
        from core.prompt.builder import UnifiedPromptBuilder

        mc = MagicMock()
        mc.corpus_manager = MagicMock()
        mm = MagicMock()

        builder = UnifiedPromptBuilder(memory_coordinator=mc, model_manager=mm)

        context = {
            "recent_conversations": [],
            "memories": [],
            "user_profile": "",
            "narrative_state": "",
            "summaries": [],
            "recent_summaries": [],
            "semantic_summaries": [],
            "reflections": [],
            "recent_reflections": [],
            "semantic_reflections": [],
            "dreams": [],
            "semantic_chunks": [],
            "wiki": [],
            "personal_notes": [],
            "user_uploads": [],
            "git_commits": [],
            "procedural_skills": [],
            "proposed_features": [],
            "graph_context": [],
            "unresolved_threads": [],
            "proactive_insights": [],
            "web_search_results": None,
            "codebase_changes": {},
        }

        prompt = builder._assemble_prompt(context=context, user_input="hello")

        assert "[CODEBASE CHANGES SINCE LAST SESSION]" not in prompt
