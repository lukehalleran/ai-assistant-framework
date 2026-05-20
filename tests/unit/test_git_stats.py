"""
Tests for the git_stats agentic tool.

Covers:
- GitStatsManager: intent parsing, time window extraction, subprocess safety, output formatting
- Tool definition structure
- Protocol parsing (native + XML)
- Provenance classification
"""

import re
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.git_stats_manager import (
    GitStatsManager,
    _classify_intent,
    _parse_time_window,
    ALLOWED_SUBCOMMANDS,
)
from core.agentic.types import (
    AgenticSearchSession,
    GIT_STATS_TOOL_DEFINITION,
    SearchDecision,
)


# ── Intent Classification ──────────────────────────────────────────

class TestClassifyIntent:
    """Test _classify_intent keyword matching."""

    def test_commit_count_how_many(self):
        assert _classify_intent("how many commits this week") == "commit_count"

    def test_commit_count_number_of(self):
        assert _classify_intent("number of commits today") == "commit_count"

    def test_commit_count_total(self):
        assert _classify_intent("total commits this month") == "commit_count"

    def test_recent_commits(self):
        assert _classify_intent("show me recent commits") == "recent_commits"

    def test_latest_commits(self):
        assert _classify_intent("latest commits on this branch") == "recent_commits"

    def test_commit_history(self):
        assert _classify_intent("commit history") == "recent_commits"

    def test_files_changed(self):
        assert _classify_intent("files changed this week") == "files_changed"

    def test_what_changed(self):
        assert _classify_intent("what changed today") == "files_changed"

    def test_contributors(self):
        assert _classify_intent("who are the contributors this month") == "contributors"

    def test_authors(self):
        assert _classify_intent("top authors") == "contributors"

    def test_branches(self):
        assert _classify_intent("list all branches") == "branches"

    def test_status(self):
        assert _classify_intent("git status") == "status"

    def test_uncommitted(self):
        assert _classify_intent("any uncommitted changes") == "status"

    def test_diff_stat(self):
        assert _classify_intent("lines changed this week") == "diff_stat"

    def test_insertions_deletions(self):
        assert _classify_intent("insertions and deletions today") == "diff_stat"

    def test_fallback(self):
        assert _classify_intent("something random about the repo") == "fallback"


# ── Time Window Parsing ────────────────────────────────────────────

class TestParseTimeWindow:
    """Test _parse_time_window temporal phrase extraction."""

    def test_today(self):
        result = _parse_time_window("commits today")
        assert result is not None
        now = datetime.now()
        assert result.year == now.year
        assert result.month == now.month
        assert result.day == now.day
        assert result.hour == 0
        assert result.minute == 0

    def test_yesterday(self):
        result = _parse_time_window("files changed yesterday")
        assert result is not None
        yesterday = datetime.now() - timedelta(days=1)
        assert result.day == yesterday.day

    def test_this_week(self):
        result = _parse_time_window("commits this week")
        assert result is not None
        expected = datetime.now() - timedelta(weeks=1)
        # Should be within a second
        assert abs((result - expected).total_seconds()) < 2

    def test_last_n_days(self):
        result = _parse_time_window("commits last 7 days")
        assert result is not None
        expected = datetime.now() - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_last_n_weeks(self):
        result = _parse_time_window("activity last 2 weeks")
        assert result is not None
        expected = datetime.now() - timedelta(weeks=2)
        assert abs((result - expected).total_seconds()) < 2

    def test_this_month(self):
        result = _parse_time_window("commits this month")
        assert result is not None
        now = datetime.now()
        assert result.day == 1
        assert result.month == now.month

    def test_this_year(self):
        result = _parse_time_window("commits this year")
        assert result is not None
        now = datetime.now()
        assert result.month == 1
        assert result.day == 1

    def test_no_temporal_phrase(self):
        result = _parse_time_window("show me branches")
        assert result is None


# ── GitStatsManager ────────────────────────────────────────────────

class TestGitStatsManager:
    """Test GitStatsManager methods."""

    def test_allowed_subcommands_are_readonly(self):
        """Ensure only read-only git commands are allowed."""
        dangerous = {"push", "reset", "checkout", "clean", "rm", "rebase", "merge", "commit"}
        assert ALLOWED_SUBCOMMANDS.isdisjoint(dangerous)

    def test_run_git_rejects_disallowed(self):
        mgr = GitStatsManager()
        mgr._repo_root = "/tmp"
        mgr._available = True
        with pytest.raises(ValueError, match="not allowed"):
            mgr._run_git(["push", "origin", "main"])

    def test_run_git_rejects_empty(self):
        mgr = GitStatsManager()
        mgr._repo_root = "/tmp"
        with pytest.raises(ValueError, match="No git arguments"):
            mgr._run_git([])

    @patch("core.git_stats_manager.subprocess.run")
    def test_run_git_truncates_long_output(self, mock_run):
        mgr = GitStatsManager(max_output_lines=3)
        mgr._repo_root = "/tmp"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="line1\nline2\nline3\nline4\nline5\n"
        )
        result = mgr._run_git(["log", "--oneline"])
        assert "[...truncated, 2 more lines]" in result
        lines = result.split("\n")
        assert len(lines) == 4  # 3 lines + truncation message

    @patch("core.git_stats_manager.subprocess.run")
    def test_run_git_timeout_raises(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="git log", timeout=10)
        mgr = GitStatsManager()
        mgr._repo_root = "/tmp"
        with pytest.raises(RuntimeError, match="timed out"):
            mgr._run_git(["log"])

    @patch("core.git_stats_manager.subprocess.run")
    def test_is_available_caches(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="/home/user/repo\n")
        mgr = GitStatsManager()
        assert mgr.is_available() is True
        assert mgr._repo_root == "/home/user/repo"
        # Second call should not run subprocess again
        mock_run.reset_mock()
        assert mgr.is_available() is True
        mock_run.assert_not_called()

    @patch("core.git_stats_manager.subprocess.run")
    def test_is_available_false_outside_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        mgr = GitStatsManager()
        assert mgr.is_available() is False

    @pytest.mark.asyncio
    async def test_execute_query_not_available(self):
        mgr = GitStatsManager()
        mgr._available = False
        result = await mgr.execute_query("commits today")
        assert result["success"] is False
        assert "Not in a git repository" in result["summary"]

    @pytest.mark.asyncio
    @patch.object(GitStatsManager, "_run_git")
    async def test_execute_query_commit_count(self, mock_run_git):
        mgr = GitStatsManager()
        mgr._available = True
        mgr._repo_root = "/tmp"
        mock_run_git.side_effect = [
            "5",  # rev-list --count
            "abc1 Fix bug\ndef2 Add test\nghi3 Refactor\njkl4 Docs\nmno5 Style",  # log
        ]
        result = await mgr.execute_query("how many commits this week")
        assert result["success"] is True
        assert "5" in result["summary"]
        assert len(result["commands_run"]) == 2

    @pytest.mark.asyncio
    @patch.object(GitStatsManager, "_run_git")
    async def test_execute_query_branches(self, mock_run_git):
        mgr = GitStatsManager()
        mgr._available = True
        mgr._repo_root = "/tmp"
        mock_run_git.return_value = "* main\n  feature/foo\n  fix/bar"
        result = await mgr.execute_query("list branches")
        assert result["success"] is True
        assert "3 branches" in result["summary"]

    @pytest.mark.asyncio
    @patch.object(GitStatsManager, "_run_git")
    async def test_execute_query_status_clean(self, mock_run_git):
        mgr = GitStatsManager()
        mgr._available = True
        mgr._repo_root = "/tmp"
        mock_run_git.return_value = ""
        result = await mgr.execute_query("git status")
        assert result["success"] is True
        assert "clean" in result["summary"].lower()

    def test_format_for_prompt_success(self):
        mgr = GitStatsManager()
        result = {
            "success": True,
            "query": "commits today",
            "commands_run": ["git rev-list --count --since=2026-03-29 HEAD"],
            "output": "12",
            "summary": "12 commits since 2026-03-29",
        }
        formatted = mgr.format_for_prompt(result)
        assert "12 commits" in formatted
        assert "git rev-list" in formatted

    def test_format_for_prompt_failure(self):
        mgr = GitStatsManager()
        result = {
            "success": False,
            "query": "test",
            "commands_run": [],
            "output": "",
            "summary": "Not in a git repository",
        }
        formatted = mgr.format_for_prompt(result)
        assert "failed" in formatted.lower()


# ── Tool Definition ────────────────────────────────────────────────

class TestToolDefinition:
    """Test GIT_STATS_TOOL_DEFINITION structure."""

    def test_has_required_keys(self):
        assert GIT_STATS_TOOL_DEFINITION["type"] == "function"
        func = GIT_STATS_TOOL_DEFINITION["function"]
        assert func["name"] == "git_stats"
        assert "description" in func
        assert "parameters" in func

    def test_query_is_required(self):
        params = GIT_STATS_TOOL_DEFINITION["function"]["parameters"]
        assert "query" in params["required"]

    def test_has_reason_field(self):
        props = GIT_STATS_TOOL_DEFINITION["function"]["parameters"]["properties"]
        assert "reason" in props


# ── Protocol Parsing ───────────────────────────────────────────────

class TestNativeProtocolParsing:
    """Test NativeToolsHandler parses git_stats tool calls."""

    def test_parse_git_stats_native(self):
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler(git_stats_available=True)

        # Simulate a tool call response
        mock_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "git_stats"
        tool_call.function.arguments = '{"query": "commits this week", "reason": "user asked"}'
        mock_response.tool_calls = [tool_call]
        mock_response.content = None

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        assert decisions[0].wants_git_stats is True
        assert decisions[0].git_stats_query == "commits this week"
        assert decisions[0].git_stats_reason == "user asked"

    def test_parse_git_stats_empty_query_defaults(self):
        """Empty-arg git_stats should default to 'recent commits' instead of failing."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler(git_stats_available=True)
        mock_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "git_stats"
        tool_call.function.arguments = '{"query": ""}'
        mock_response.tool_calls = [tool_call]
        mock_response.content = None

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        assert decisions[0].wants_git_stats is True
        assert decisions[0].git_stats_query == "recent commits"

    def test_git_stats_in_tools_list(self):
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler(git_stats_available=True)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "git_stats" in tool_names

    def test_git_stats_not_in_tools_when_disabled(self):
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler(git_stats_available=False)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "git_stats" not in tool_names


class TestXMLProtocolParsing:
    """Test XMLMarkerHandler parses <git_stats> tags."""

    def test_parse_git_stats_xml(self):
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        text = "Let me check the repo. <git_stats>commits this week</git_stats>"

        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_git_stats is True
        assert decisions[0].git_stats_query == "commits this week"

    def test_parse_git_stats_xml_empty_defaults(self):
        """Empty XML git_stats should default to 'recent commits'."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        text = "<git_stats>   </git_stats>"

        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_git_stats is True
        assert decisions[0].git_stats_query == "recent commits"


# ── Provenance Classification ──────────────────────────────────────

class TestProvenanceClassification:
    """Test _classify_round_action handles git_stats prefix."""

    def test_classify_git_stats(self):
        assert AgenticSearchSession._classify_round_action("[Git Stats] commits today") == "git_stats"

    def test_classify_web_search_unchanged(self):
        assert AgenticSearchSession._classify_round_action("some web query") == "web_search"

    def test_classify_memory_search_unchanged(self):
        assert AgenticSearchSession._classify_round_action("[Memory: facts] test") == "memory_search"


# ── Factory Function ───────────────────────────────────────────────

class TestFactoryFunction:
    """Test get_protocol_handler passes git_stats_available."""

    def test_factory_passes_git_stats(self):
        from core.agentic.protocols import get_protocol_handler
        from core.agentic.types import SearchProtocol

        handler = get_protocol_handler(
            SearchProtocol.NATIVE_TOOLS,
            git_stats_available=True,
        )
        assert handler.git_stats_available is True

    def test_factory_defaults_false(self):
        from core.agentic.protocols import get_protocol_handler
        from core.agentic.types import SearchProtocol

        handler = get_protocol_handler(SearchProtocol.NATIVE_TOOLS)
        assert handler.git_stats_available is False
