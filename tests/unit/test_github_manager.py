"""
Tests for the GitHub API agentic tool.

Covers:
- GitHubManager: intent parsing, command validation, allowlist enforcement,
  output formatting, subprocess safety
- Tool definition structure
- Protocol parsing (native + XML)
- Provenance classification
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.github_manager import (
    GitHubManager,
    _classify_intent,
    _extract_number,
    _extract_search_terms,
    _validate_command,
    ALLOWED_COMMANDS,
    BLOCKED_FLAGS,
)
from core.agentic.types import (
    AgenticSearchSession,
    GITHUB_TOOL_DEFINITION,
    SearchDecision,
)


# ── Intent Classification ──────────────────────────────────────────

class TestClassifyIntent:
    """Test _classify_intent keyword matching."""

    def test_issues_open(self):
        assert _classify_intent("open issues") == "issues"

    def test_issues_bugs(self):
        assert _classify_intent("list all bugs") == "issues"

    def test_issues_closed(self):
        assert _classify_intent("closed issues") == "issues"

    def test_issues_labeled(self):
        assert _classify_intent("issues labeled enhancement") == "issues"

    def test_issue_detail_number(self):
        assert _classify_intent("issue #42") == "issue_detail"

    def test_issue_detail_no_hash(self):
        assert _classify_intent("issue 7") == "issue_detail"

    def test_prs_open(self):
        assert _classify_intent("open pull requests") == "prs"

    def test_prs_merged(self):
        assert _classify_intent("merged prs") == "prs"

    def test_pr_detail(self):
        assert _classify_intent("PR #12") == "pr_detail"

    def test_pr_detail_full(self):
        assert _classify_intent("pull request #5") == "pr_detail"

    def test_pr_diff(self):
        assert _classify_intent("pr diff for #3") == "pr_diff"

    def test_pr_checks(self):
        assert _classify_intent("pr checks for #3") == "pr_checks"

    def test_ci_status(self):
        assert _classify_intent("ci status") == "pr_checks"

    def test_actions(self):
        assert _classify_intent("recent github actions") == "actions"

    def test_workflow_runs(self):
        assert _classify_intent("workflow runs") == "actions"

    def test_action_detail(self):
        assert _classify_intent("run #12345") == "action_detail"

    def test_workflows(self):
        assert _classify_intent("list workflows") == "workflows"

    def test_releases(self):
        assert _classify_intent("latest release") == "releases"

    def test_releases_list(self):
        assert _classify_intent("all releases") == "releases"

    def test_search_code(self):
        assert _classify_intent("search code for authenticate") == "search_code"

    def test_search_issues(self):
        assert _classify_intent("search issues for memory leak") == "search_issues"

    def test_search_prs(self):
        assert _classify_intent("search prs for refactor") == "search_prs"

    def test_contributors(self):
        assert _classify_intent("top contributors") == "contributors"

    def test_labels(self):
        assert _classify_intent("available labels") == "labels"

    def test_milestones(self):
        assert _classify_intent("show milestones") == "milestones"

    def test_repo_info(self):
        assert _classify_intent("repo info") == "repo_info"

    def test_repo_stars(self):
        assert _classify_intent("how many stars") == "repo_info"

    def test_fallback(self):
        assert _classify_intent("random question") == "fallback"


# ── Number Extraction ──────────────────────────────────────────────

class TestExtractNumber:
    def test_hash_number(self):
        assert _extract_number("issue #42") == 42

    def test_bare_number(self):
        assert _extract_number("PR 7") == 7

    def test_no_number(self):
        assert _extract_number("open issues") is None


# ── Search Term Extraction ──────────────────────────────────────────

class TestExtractSearchTerms:
    def test_strips_intent_words(self):
        result = _extract_search_terms("search code for authenticate")
        assert "search" not in result.lower().split()
        assert "authenticate" in result.lower()

    def test_preserves_meaningful_terms(self):
        result = _extract_search_terms("memory leak in parser")
        assert "memory" in result.lower()
        assert "leak" in result.lower()


# ── Command Validation ──────────────────────────────────────────────

class TestValidateCommand:
    """Test _validate_command allowlist enforcement."""

    def test_allowed_issue_list(self):
        _validate_command(["issue", "list"])  # Should not raise

    def test_allowed_pr_view(self):
        _validate_command(["pr", "view", "42"])

    def test_allowed_run_list(self):
        _validate_command(["run", "list"])

    def test_allowed_search_code(self):
        _validate_command(["search", "code", "authenticate"])

    def test_allowed_api_get(self):
        _validate_command(["api", "repos/user/repo/issues"])

    def test_blocked_subcommand(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["gist", "create"])

    def test_blocked_issue_create(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["issue", "create", "--title", "test"])

    def test_blocked_pr_merge(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["pr", "merge", "42"])

    def test_blocked_pr_close(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["pr", "close", "42"])

    def test_blocked_release_create(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["release", "create", "v1.0"])

    def test_blocked_method_flag(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["api", "repos/user/repo/issues", "--method", "POST"])

    def test_blocked_X_flag(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["api", "repos/user/repo/issues", "-X", "DELETE"])

    def test_blocked_field_flag(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["api", "repos/user/repo/issues", "--field", "title=bad"])

    def test_blocked_input_flag(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["api", "repos/user/repo", "--input", "data.json"])

    def test_empty_args(self):
        with pytest.raises(ValueError, match="No gh arguments"):
            _validate_command([])

    def test_allowed_repo_view(self):
        _validate_command(["repo", "view"])

    def test_blocked_repo_create(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["repo", "create", "new-repo"])

    def test_blocked_issue_edit(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["issue", "edit", "42"])

    def test_blocked_issue_comment(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(["issue", "comment", "42"])

    def test_allowed_workflow_list(self):
        _validate_command(["workflow", "list"])

    def test_allowed_workflow_view(self):
        _validate_command(["workflow", "view", "test.yml"])

    def test_allowed_search_commits(self):
        _validate_command(["search", "commits", "fix bug"])


# ── GitHubManager ──────────────────────────────────────────────────

class TestGitHubManager:
    """Test GitHubManager methods."""

    def test_init_defaults(self):
        mgr = GitHubManager()
        assert mgr.repo is None
        assert mgr.timeout == 15
        assert mgr.max_output_lines == 80

    def test_init_custom(self):
        mgr = GitHubManager(repo="user/repo", timeout=30, max_output_lines=100)
        assert mgr.repo == "user/repo"
        assert mgr.timeout == 30
        assert mgr.max_output_lines == 100

    def test_get_repo_flag_with_repo(self):
        mgr = GitHubManager(repo="user/repo")
        assert mgr._get_repo_flag() == ["--repo", "user/repo"]

    def test_get_repo_flag_no_repo(self):
        mgr = GitHubManager()
        assert mgr._get_repo_flag() == []

    def test_get_repo_flag_detected(self):
        mgr = GitHubManager()
        mgr._detected_repo = "detected/repo"
        assert mgr._get_repo_flag() == ["--repo", "detected/repo"]

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="user/repo\n")
        mgr = GitHubManager()
        assert mgr.is_available() is True

    @patch("subprocess.run")
    def test_is_available_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        mgr = GitHubManager()
        assert mgr.is_available() is False

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_is_available_no_gh(self, mock_run):
        mgr = GitHubManager()
        assert mgr.is_available() is False

    @patch("subprocess.run")
    def test_is_available_cached(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="user/repo\n")
        mgr = GitHubManager()
        mgr.is_available()
        mgr.is_available()
        # auth check + repo detect on first call, nothing on cached second call
        assert mock_run.call_count == 2  # auth + detect

    @patch("subprocess.run")
    def test_run_gh_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="line1\nline2\nline3\n",
        )
        mgr = GitHubManager()
        result = mgr._run_gh(["issue", "list"])
        assert "line1" in result

    @patch("subprocess.run")
    def test_run_gh_blocked_command(self, mock_run):
        mgr = GitHubManager()
        with pytest.raises(ValueError, match="not allowed"):
            mgr._run_gh(["issue", "create", "--title", "bad"])

    @patch("subprocess.run")
    def test_run_gh_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="not found",
        )
        mgr = GitHubManager()
        with pytest.raises(RuntimeError, match="failed"):
            mgr._run_gh(["issue", "view", "99999"])

    @patch("subprocess.run")
    def test_run_gh_truncation(self, mock_run):
        long_output = "\n".join(f"line{i}" for i in range(200))
        mock_run.return_value = MagicMock(returncode=0, stdout=long_output)
        mgr = GitHubManager(max_output_lines=10)
        result = mgr._run_gh(["issue", "list"])
        assert "truncated" in result

    @patch("subprocess.run")
    def test_run_gh_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=15)
        mgr = GitHubManager()
        with pytest.raises(RuntimeError, match="timed out"):
            mgr._run_gh(["issue", "list"])


# ── execute_query integration ──────────────────────────────────────

class TestExecuteQuery:
    """Test execute_query end-to-end with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_not_available(self):
        mgr = GitHubManager()
        mgr._available = False
        result = await mgr.execute_query("open issues")
        assert result["success"] is False
        assert "not available" in result["summary"].lower()

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_issues_query(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="#1 Bug report\n#2 Feature request\n",
        )
        mgr = GitHubManager(repo="user/repo")
        mgr._available = True
        result = await mgr.execute_query("open issues")
        assert result["success"] is True
        assert "issues" in result["summary"].lower()

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_pr_detail_query(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="PR #42: Big refactor\nStatus: open\n",
        )
        mgr = GitHubManager(repo="user/repo")
        mgr._available = True
        result = await mgr.execute_query("PR #42")
        assert result["success"] is True
        assert "#42" in result["summary"]

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_actions_query(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Run #1 completed\nRun #2 failed\n",
        )
        mgr = GitHubManager(repo="user/repo")
        mgr._available = True
        result = await mgr.execute_query("recent github actions")
        assert result["success"] is True
        assert "workflow runs" in result["summary"].lower()

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_releases_query(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="v1.0.0  Latest\nv0.9.0\n",
        )
        mgr = GitHubManager(repo="user/repo")
        mgr._available = True
        result = await mgr.execute_query("releases")
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_error_handling(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error occurred")
        mgr = GitHubManager(repo="user/repo")
        mgr._available = True
        result = await mgr.execute_query("issue #9999")
        assert result["success"] is False
        assert "Error" in result["summary"]


# ── Output Formatting ──────────────────────────────────────────────

class TestFormatForPrompt:
    def test_success_format(self):
        mgr = GitHubManager()
        result = {
            "success": True,
            "query": "open issues",
            "commands_run": ["gh issue list --repo user/repo"],
            "output": "#1 Bug\n#2 Feature",
            "summary": "2 issues",
        }
        formatted = mgr.format_for_prompt(result)
        assert "Summary: 2 issues" in formatted
        assert "Commands:" in formatted
        assert "#1 Bug" in formatted

    def test_failure_format(self):
        mgr = GitHubManager()
        result = {
            "success": False,
            "summary": "gh CLI not available",
        }
        formatted = mgr.format_for_prompt(result)
        assert "failed" in formatted.lower()
        assert "not available" in formatted.lower()


# ── Tool Definition ────────────────────────────────────────────────

class TestToolDefinition:
    def test_tool_name(self):
        assert GITHUB_TOOL_DEFINITION["function"]["name"] == "github"

    def test_tool_has_query_param(self):
        params = GITHUB_TOOL_DEFINITION["function"]["parameters"]
        assert "query" in params["properties"]

    def test_tool_query_required(self):
        params = GITHUB_TOOL_DEFINITION["function"]["parameters"]
        assert "query" in params["required"]


# ── Protocol Parsing ───────────────────────────────────────────────

class TestNativeProtocol:
    """Test native tool parsing for github tool calls."""

    def test_parse_github_native(self):
        from core.agentic.protocols import NativeToolsHandler
        handler = NativeToolsHandler(github_available=True)
        response = MagicMock()
        response.tool_calls = [MagicMock()]
        response.tool_calls[0].function.name = "github"
        response.tool_calls[0].function.arguments = '{"query": "open issues", "reason": "checking bugs"}'
        decisions = handler.parse_response(response)
        assert len(decisions) == 1
        assert decisions[0].wants_github is True
        assert decisions[0].github_query == "open issues"
        assert decisions[0].github_reason == "checking bugs"

    def test_github_tool_in_tools_list(self):
        from core.agentic.protocols import NativeToolsHandler
        handler = NativeToolsHandler(github_available=True)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "github" in tool_names

    def test_github_tool_not_in_tools_when_unavailable(self):
        from core.agentic.protocols import NativeToolsHandler
        handler = NativeToolsHandler(github_available=False)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "github" not in tool_names

    def test_github_in_augmented_prompt(self):
        from core.agentic.protocols import NativeToolsHandler
        handler = NativeToolsHandler(github_available=True)
        prompt = handler.augment_system_prompt("base prompt", 5)
        assert "github" in prompt


class TestXMLProtocol:
    """Test XML marker parsing for github tool."""

    def test_parse_github_xml(self):
        from core.agentic.protocols import XMLMarkerHandler
        handler = XMLMarkerHandler()
        decisions = handler.parse_response("<github>open issues labeled bug</github>")
        assert len(decisions) == 1
        assert decisions[0].wants_github is True
        assert decisions[0].github_query == "open issues labeled bug"

    def test_parse_github_xml_multiline(self):
        from core.agentic.protocols import XMLMarkerHandler
        handler = XMLMarkerHandler()
        text = "Let me check.\n<github>PR #42</github>\nLooking..."
        decisions = handler.parse_response(text)
        github_decisions = [d for d in decisions if d.wants_github]
        assert len(github_decisions) == 1
        assert github_decisions[0].github_query == "PR #42"

    def test_parse_github_xml_with_other_tools(self):
        from core.agentic.protocols import XMLMarkerHandler
        handler = XMLMarkerHandler()
        text = "<search>python asyncio</search>\n<github>open issues</github>"
        decisions = handler.parse_response(text)
        assert len(decisions) == 2
        github_d = [d for d in decisions if d.wants_github]
        search_d = [d for d in decisions if d.wants_search]
        assert len(github_d) == 1
        assert len(search_d) == 1


# ── Provenance Classification ──────────────────────────────────────

class TestProvenanceClassification:
    def test_github_classification(self):
        assert AgenticSearchSession._classify_round_action("[GitHub] open issues") == "github"

    def test_non_github_classification(self):
        assert AgenticSearchSession._classify_round_action("some web query") == "web_search"


# ── Allowlist Coverage ─────────────────────────────────────────────

class TestAllowlistCoverage:
    """Ensure ALLOWED_COMMANDS covers the expected GitHub operations."""

    def test_issue_operations(self):
        assert "list" in ALLOWED_COMMANDS["issue"]
        assert "view" in ALLOWED_COMMANDS["issue"]
        assert "search" in ALLOWED_COMMANDS["issue"]
        assert "status" in ALLOWED_COMMANDS["issue"]

    def test_pr_operations(self):
        assert "list" in ALLOWED_COMMANDS["pr"]
        assert "view" in ALLOWED_COMMANDS["pr"]
        assert "diff" in ALLOWED_COMMANDS["pr"]
        assert "checks" in ALLOWED_COMMANDS["pr"]
        assert "status" in ALLOWED_COMMANDS["pr"]
        assert "search" in ALLOWED_COMMANDS["pr"]

    def test_run_operations(self):
        assert "list" in ALLOWED_COMMANDS["run"]
        assert "view" in ALLOWED_COMMANDS["run"]

    def test_workflow_operations(self):
        assert "list" in ALLOWED_COMMANDS["workflow"]
        assert "view" in ALLOWED_COMMANDS["workflow"]

    def test_release_operations(self):
        assert "list" in ALLOWED_COMMANDS["release"]
        assert "view" in ALLOWED_COMMANDS["release"]

    def test_repo_operations(self):
        assert "view" in ALLOWED_COMMANDS["repo"]

    def test_search_operations(self):
        assert "code" in ALLOWED_COMMANDS["search"]
        assert "issues" in ALLOWED_COMMANDS["search"]
        assert "prs" in ALLOWED_COMMANDS["search"]
        assert "repos" in ALLOWED_COMMANDS["search"]
        assert "commits" in ALLOWED_COMMANDS["search"]

    def test_no_mutating_operations(self):
        """Verify no write/mutating actions slipped into the allowlist."""
        all_actions = set()
        for actions in ALLOWED_COMMANDS.values():
            all_actions.update(actions)
        dangerous = {"create", "delete", "edit", "close", "merge", "comment", "reopen"}
        assert all_actions.isdisjoint(dangerous), f"Mutating actions found: {all_actions & dangerous}"

    def test_blocked_flags_present(self):
        assert "--method" in BLOCKED_FLAGS
        assert "-X" in BLOCKED_FLAGS
        assert "--input" in BLOCKED_FLAGS
        assert "--field" in BLOCKED_FLAGS
        assert "-f" in BLOCKED_FLAGS


# ── Formatter ──────────────────────────────────────────────────────

class TestFormatter:
    def test_format_github_context(self):
        from core.agentic.formatters import AgenticFormatter
        fmt = AgenticFormatter()
        result = fmt.format_github_context(1, "open issues", "Issue list here")
        assert "[GITHUB" in result
        assert "Round 1" in result
        assert "open issues" in result
        assert "Issue list here" in result
