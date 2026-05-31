"""
# core/actions/github_write.py

Module Contract
- Purpose: Execute GitHub write actions (create issue, comment on PR) via the `gh` CLI.
- Public interface:
  - create_github_issue(proposal: ActionProposal) -> ActionResult
  - comment_github_pr(proposal: ActionProposal) -> ActionResult
- Dependencies: subprocess (stdlib); config.app_config
  (INTERNET_ACTIONS_GITHUB_WRITE_ENABLED, GITHUB_API_REPO, GITHUB_API_TIMEOUT).
- Side effects: Creates a GitHub issue / posts a PR comment (write actions) via `gh`.
  Gated on INTERNET_ACTIONS_GITHUB_WRITE_ENABLED and `gh` auth.
- Safety: this is a SEPARATE path from the read-only core/github_manager.py (whose allowlist
  deliberately blocks writes). It enforces its own tiny write-allowlist — only
  ("issue","create") and ("pr","comment") may run — uses explicit arg lists (never shell=True,
  so title/body/comment content cannot inject), applies a subprocess timeout, and surfaces gh
  stderr on failure. Execution is human-gated upstream (GUI Approve → ActionExecutorRegistry).
"""

import logging
import re
import subprocess
from typing import List, Optional

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_github_write")

# Only these (subcommand, action) pairs may run through this module. Everything
# else is rejected before reaching subprocess.
WRITE_ALLOWLIST = frozenset({("issue", "create"), ("pr", "comment")})


def _gh_available() -> bool:
    """Check if the `gh` CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _detect_repo() -> Optional[str]:
    """Detect owner/repo from the current `gh` context, or None."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _repo_args(params: dict) -> List[str]:
    """Return ['--repo', 'owner/repo'] from params/config/detection, or [] (gh uses cwd remote)."""
    from config.app_config import GITHUB_API_REPO

    repo = (params.get("repo") or GITHUB_API_REPO or _detect_repo() or "").strip()
    return ["--repo", repo] if repo else []


def _run_gh_write(args: List[str], timeout: int) -> str:
    """Run an allow-listed `gh` write command.

    Args:
        args: gh arguments without the leading 'gh', e.g. ["issue", "create", "--title", ...].
        timeout: subprocess timeout in seconds.

    Returns:
        Command stdout (stripped).

    Raises:
        ValueError: if (args[0], args[1]) is not in WRITE_ALLOWLIST.
        RuntimeError: if the command times out or exits non-zero (stderr surfaced).
    """
    if len(args) < 2 or (args[0], args[1]) not in WRITE_ALLOWLIST:
        allowed = ", ".join(f"{a} {b}" for a, b in sorted(WRITE_ALLOWLIST))
        raise ValueError(
            f"gh write command not allowed: '{' '.join(args[:2])}'. Allowed: {allowed}"
        )

    try:
        result = subprocess.run(
            ["gh", *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gh {args[0]} {args[1]} timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(f"gh {args[0]} {args[1]} failed: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def _extract_pr_number(params: dict) -> Optional[str]:
    """Find the PR number from explicit keys, falling back to recipient/subject."""
    for key in ("pr_number", "number", "pr"):
        val = params.get(key)
        if val is not None and str(val).strip():
            m = re.search(r"\d+", str(val))
            if m:
                return m.group(0)
    for key in ("recipient", "subject"):
        val = params.get(key)
        if val:
            m = re.search(r"\d+", str(val))
            if m:
                return m.group(0)
    return None


async def create_github_issue(proposal: ActionProposal) -> ActionResult:
    """Create a GitHub issue via `gh issue create`.

    Expects proposal.params:
        - subject (str): issue title (required).
        - message (str): issue body (optional).
        - repo (str, optional): owner/repo; defaults to GITHUB_API_REPO or the detected repo.
    """
    from config.app_config import INTERNET_ACTIONS_GITHUB_WRITE_ENABLED, GITHUB_API_TIMEOUT

    if not INTERNET_ACTIONS_GITHUB_WRITE_ENABLED:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub write actions are disabled (set internet_actions.github_write_enabled: true).",
        )
    if not _gh_available():
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub CLI (gh) not available or not authenticated. Run 'gh auth login'.",
        )

    params = proposal.params or {}
    title = (params.get("subject") or params.get("title") or "").strip()
    body = (params.get("message") or params.get("body") or "").strip()
    if not title:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing required parameter: subject (issue title).",
        )

    args = ["issue", "create", "--title", title, "--body", body] + _repo_args(params)
    try:
        out = _run_gh_write(args, GITHUB_API_TIMEOUT)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"[GitHubWrite] issue create failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"GitHub issue creation failed: {e}",
        )

    logger.info(f"[GitHubWrite] issue created: {out}")
    return ActionResult(
        action_id=proposal.action_id,
        success=True,
        message=f"GitHub issue created: {out}" if out else "GitHub issue created.",
    )


async def comment_github_pr(proposal: ActionProposal) -> ActionResult:
    """Comment on a GitHub PR via `gh pr comment <number> --body <text>`.

    Expects proposal.params:
        - pr_number (int|str): PR number (required; also parsed from recipient/subject).
        - message (str): comment body (required).
        - repo (str, optional): owner/repo; defaults to GITHUB_API_REPO or the detected repo.
    """
    from config.app_config import INTERNET_ACTIONS_GITHUB_WRITE_ENABLED, GITHUB_API_TIMEOUT

    if not INTERNET_ACTIONS_GITHUB_WRITE_ENABLED:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub write actions are disabled (set internet_actions.github_write_enabled: true).",
        )
    if not _gh_available():
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub CLI (gh) not available or not authenticated. Run 'gh auth login'.",
        )

    params = proposal.params or {}
    pr_number = _extract_pr_number(params)
    body = (params.get("message") or params.get("body") or "").strip()
    if not pr_number:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing required parameter: pr_number (the PR to comment on).",
        )
    if not body:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing required parameter: message (comment body).",
        )

    args = ["pr", "comment", pr_number, "--body", body] + _repo_args(params)
    try:
        out = _run_gh_write(args, GITHUB_API_TIMEOUT)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"[GitHubWrite] pr comment failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"GitHub PR comment failed: {e}",
        )

    logger.info(f"[GitHubWrite] PR #{pr_number} commented: {out}")
    return ActionResult(
        action_id=proposal.action_id,
        success=True,
        message=f"Comment posted to PR #{pr_number}." + (f"\n{out}" if out else ""),
    )
