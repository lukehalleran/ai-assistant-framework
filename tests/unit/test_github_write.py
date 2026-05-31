"""Tests for GitHub write actions (create issue, comment on PR).

Mirrors tests/unit/test_calendar_create.py. The github_write functions import their config
flag inside the function body, so patching config.app_config.* takes effect per-call. The
`gh` subprocess is mocked (no network / no real writes).
"""

import pytest
from unittest.mock import patch, MagicMock

from core.actions.github_write import (
    create_github_issue,
    comment_github_pr,
    _run_gh_write,
    _extract_pr_number,
    WRITE_ALLOWLIST,
)
from core.actions.types import ActionProposal, ActionType, CONFIRMATION_REQUIRED
from core.actions.executors import ActionExecutorRegistry


# --------------------------------------------------------------------------- helpers
def _issue_proposal(**overrides) -> ActionProposal:
    defaults = {
        "action_type": "github_create_issue",
        "params": {"subject": "Fix the thing", "message": "It is broken."},
        "summary": "Create issue: Fix the thing",
        "reasoning": "user asked",
    }
    defaults.update(overrides)
    return ActionProposal(**defaults)


def _pr_proposal(**overrides) -> ActionProposal:
    defaults = {
        "action_type": "github_comment_pr",
        "params": {"pr_number": 42, "message": "LGTM"},
        "summary": "Comment on PR #42",
        "reasoning": "user asked",
    }
    defaults.update(overrides)
    return ActionProposal(**defaults)


def _ok(stdout: str = "https://github.com/o/r/issues/1") -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = ""
    return m


def _fail(stderr: str = "permission denied") -> MagicMock:
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = stderr
    return m


_ENABLED = "config.app_config.INTERNET_ACTIONS_GITHUB_WRITE_ENABLED"
_AVAIL = "core.actions.github_write._gh_available"
_REPO = "core.actions.github_write._detect_repo"
_RUN = "core.actions.github_write.subprocess.run"


# --------------------------------------------------------------------------- confirmation
class TestConfirmationRequired:
    def test_both_require_confirmation(self):
        assert ActionType.GITHUB_CREATE_ISSUE in CONFIRMATION_REQUIRED
        assert ActionType.GITHUB_COMMENT_PR in CONFIRMATION_REQUIRED


# --------------------------------------------------------------------------- write allowlist
class TestWriteAllowlist:
    def test_allowlist_contents(self):
        assert WRITE_ALLOWLIST == frozenset({("issue", "create"), ("pr", "comment")})

    def test_rejects_non_allowlisted(self):
        with pytest.raises(ValueError):
            _run_gh_write(["repo", "delete"], 5)
        with pytest.raises(ValueError):
            _run_gh_write(["issue", "close", "1"], 5)
        with pytest.raises(ValueError):
            _run_gh_write(["issue"], 5)  # too short

    def test_allows_the_two_write_commands(self):
        with patch(_RUN, return_value=_ok("done")):
            assert _run_gh_write(["issue", "create", "--title", "x", "--body", ""], 5) == "done"
            assert _run_gh_write(["pr", "comment", "1", "--body", "x"], 5) == "done"

    def test_nonzero_exit_raises_with_stderr(self):
        with patch(_RUN, return_value=_fail("nope")):
            with pytest.raises(RuntimeError, match="nope"):
                _run_gh_write(["issue", "create", "--title", "x", "--body", ""], 5)


# --------------------------------------------------------------------------- pr-number parsing
class TestExtractPrNumber:
    def test_from_pr_number(self):
        assert _extract_pr_number({"pr_number": 7}) == "7"

    def test_from_recipient(self):
        assert _extract_pr_number({"recipient": "PR #99"}) == "99"

    def test_none_when_absent(self):
        assert _extract_pr_number({"subject": "no number here"}) is None


# --------------------------------------------------------------------------- create issue
class TestCreateIssue:
    @pytest.mark.asyncio
    async def test_disabled_returns_failure(self):
        with patch(_ENABLED, False):
            res = await create_github_issue(_issue_proposal())
        assert res.success is False
        assert "disabled" in res.message

    @pytest.mark.asyncio
    async def test_gh_unavailable(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=False):
            res = await create_github_issue(_issue_proposal())
        assert res.success is False
        assert "not available" in res.message

    @pytest.mark.asyncio
    async def test_missing_title(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True):
            res = await create_github_issue(_issue_proposal(params={"message": "body only"}))
        assert res.success is False
        assert "subject" in res.message

    @pytest.mark.asyncio
    async def test_success(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_ok("https://github.com/o/r/issues/5")) as run:
            res = await create_github_issue(_issue_proposal())
        assert res.success is True
        assert "issues/5" in res.message
        called = run.call_args[0][0]
        assert called[:3] == ["gh", "issue", "create"]
        assert "--title" in called and "--body" in called

    @pytest.mark.asyncio
    async def test_gh_failure_surfaced(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_fail("HTTP 403: Resource not accessible")):
            res = await create_github_issue(_issue_proposal())
        assert res.success is False
        assert "403" in res.message

    @pytest.mark.asyncio
    async def test_repo_flag_passed_when_known(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value="owner/repo"), \
             patch(_RUN, return_value=_ok("ok")) as run:
            await create_github_issue(_issue_proposal())
        called = run.call_args[0][0]
        assert "--repo" in called and "owner/repo" in called


# --------------------------------------------------------------------------- comment PR
class TestCommentPr:
    @pytest.mark.asyncio
    async def test_disabled_returns_failure(self):
        with patch(_ENABLED, False):
            res = await comment_github_pr(_pr_proposal())
        assert res.success is False
        assert "disabled" in res.message

    @pytest.mark.asyncio
    async def test_missing_pr_number(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True):
            res = await comment_github_pr(_pr_proposal(params={"message": "hi"}))
        assert res.success is False
        assert "pr_number" in res.message

    @pytest.mark.asyncio
    async def test_missing_body(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True):
            res = await comment_github_pr(_pr_proposal(params={"pr_number": 3}))
        assert res.success is False
        assert "message" in res.message

    @pytest.mark.asyncio
    async def test_success(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_ok("https://github.com/o/r/pull/42#issuecomment-1")) as run:
            res = await comment_github_pr(_pr_proposal())
        assert res.success is True
        assert "PR #42" in res.message
        called = run.call_args[0][0]
        assert called[:3] == ["gh", "pr", "comment"]
        assert "42" in called

    @pytest.mark.asyncio
    async def test_pr_number_parsed_from_recipient(self):
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_ok("ok")) as run:
            res = await comment_github_pr(
                _pr_proposal(params={"recipient": "pull request #17", "message": "hi"})
            )
        assert res.success is True
        called = run.call_args[0][0]
        assert "17" in called


# --------------------------------------------------------------------------- registry no longer stub
class TestToolHealthAdvertisesGithubWrites:
    """Regression for the GUI failure: tool-health must tell the model it can create issues.

    The first attempt failed because get_tool_health() said 'github: ... read-only' and the
    propose_action line omitted the github actions, so the model drafted instead of proposing.
    """

    def _tool_executor(self):
        from core.agentic.tools import ToolExecutor
        return ToolExecutor(
            model_manager=None, web_search_manager=None, formatter=None,
            github_manager=object(),  # truthy → github tool "AVAILABLE"
        )

    def test_propose_action_lists_github_writes_when_enabled(self):
        te = self._tool_executor()
        with patch(_ENABLED, True):
            health = te.get_tool_health()
        propose_line = next(l for l in health.splitlines() if l.startswith("propose_action:"))
        assert "github_create_issue" in propose_line
        assert "github_comment_pr" in propose_line
        # The github query tool should point writers to propose_action, not imply writes are impossible.
        gh_line = next(l for l in health.splitlines() if l.startswith("github:"))
        assert "propose_action" in gh_line
        assert "read-only" not in gh_line

    def test_propose_action_omits_github_writes_when_disabled(self):
        te = self._tool_executor()
        with patch(_ENABLED, False):
            health = te.get_tool_health()
        propose_line = next(l for l in health.splitlines() if l.startswith("propose_action:"))
        assert "github_create_issue" not in propose_line


class TestIssueFieldExtraction:
    """Deterministic backfill: when the model emits propose_action with blank subject/message,
    the controller extracts the title/body from the user's request so the proposal is complete.
    """

    def test_extracts_title_and_body_from_request(self):
        from core.agentic.controller import _extract_issue_fields_from_query
        q = ('Open a GitHub issue titled "Decompose _assemble_prompt god-method" — the body '
             'should explain that core/prompt/formatter.py is ~900 lines and should be split.')
        title, body = _extract_issue_fields_from_query(q)
        assert title == "Decompose _assemble_prompt god-method"
        assert "formatter.py" in body and "900 lines" in body

    def test_title_without_quotes(self):
        from core.agentic.controller import _extract_issue_fields_from_query
        title, _ = _extract_issue_fields_from_query("file a github issue titled Add dark mode")
        assert title == "Add dark mode"

    def test_no_match_returns_empty(self):
        from core.agentic.controller import _extract_issue_fields_from_query
        assert _extract_issue_fields_from_query("what does this function do?") == ("", "")


class TestControllerDispatchesAction:
    """Regression: the agentic controller's dispatch router must route a wants_action
    decision to propose_action. It previously had branches for every read tool but NO
    branch for wants_action, so a parsed propose_action decision fell through to the empty
    else and was silently dropped — no proposal, no approval card, for any model.
    """

    @pytest.mark.asyncio
    async def test_dispatch_single_inner_routes_propose_action(self):
        from unittest.mock import MagicMock
        from core.agentic.controller import AgenticSearchController
        from core.agentic.types import SearchDecision
        from core.agentic.tools import ToolExecutor

        ctrl = AgenticSearchController(model_manager=MagicMock(), web_search_manager=None)
        store = ToolExecutor._get_pending_actions_store()
        store.clear()

        decision = SearchDecision(
            wants_action=True,
            action_type="github_create_issue",
            action_params={"subject": "Test title", "message": "Test body"},
            action_summary="github_create_issue: Test title",
            action_reason="user asked",
        )
        await ctrl._dispatch_single_inner(
            decision, 1, session=MagicMock(), crisis_level=None, sandbox_session=None
        )
        pend = store.get_pending()
        assert pend is not None, "propose_action decision was dropped by the dispatch router"
        assert pend.action_type == ActionType.GITHUB_CREATE_ISSUE
        assert pend.params.get("subject") == "Test title"


class TestRegistryDelegates:
    @pytest.mark.asyncio
    async def test_create_issue_no_longer_stub(self):
        reg = ActionExecutorRegistry()
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_ok("https://github.com/o/r/issues/9")):
            res = await reg.execute(_issue_proposal())
        assert res.success is True
        assert "not yet implemented" not in res.message

    @pytest.mark.asyncio
    async def test_comment_pr_no_longer_stub(self):
        reg = ActionExecutorRegistry()
        with patch(_ENABLED, True), patch(_AVAIL, return_value=True), \
             patch(_REPO, return_value=None), \
             patch(_RUN, return_value=_ok("ok")):
            res = await reg.execute(_pr_proposal())
        assert res.success is True
        assert "not yet implemented" not in res.message
