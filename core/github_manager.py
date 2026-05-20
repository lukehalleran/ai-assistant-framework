"""
GitHub Manager -- Read-only GitHub API access for the agentic loop.

Contract:
    - Provides GitHubManager for querying GitHub via `gh` CLI
    - Strictly read-only: ALLOWED_COMMANDS whitelist enforced before subprocess
    - Blocks all mutating operations (create, close, merge, edit, delete, comment, push)
    - Intent-based routing maps natural-language queries to `gh` subcommands
    - All subprocess calls use timeouts and output capping
    - Results returned as structured dicts, formatted for LLM prompt injection

Public Interface:
    - GitHubManager(repo, timeout, max_output_lines)
    - is_available() -> bool  (checks if `gh` CLI is installed and authenticated)
    - execute_query(query) -> Dict[str, Any]  (parse intent, run gh, return results)
    - format_for_prompt(result) -> str  (format result dict for LLM context)

Dependencies:
    - subprocess (stdlib)
    - config.app_config (GITHUB_API_ENABLED, GITHUB_API_TIMEOUT, etc.)
"""

import logging
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Safety: only these (subcommand, action) pairs are allowed.
# Everything else is blocked before reaching subprocess.
# -----------------------------------------------------------------------
ALLOWED_COMMANDS: dict[str, frozenset[str]] = {
    # gh issue <action>
    "issue": frozenset({"list", "view", "search", "status"}),
    # gh pr <action>
    "pr": frozenset({"list", "view", "diff", "checks", "status", "search"}),
    # gh run <action>  (GitHub Actions)
    "run": frozenset({"list", "view"}),
    # gh workflow <action>
    "workflow": frozenset({"list", "view"}),
    # gh release <action>
    "release": frozenset({"list", "view"}),
    # gh repo <action>
    "repo": frozenset({"view"}),
    # gh search <action>
    "search": frozenset({"code", "issues", "prs", "repos", "commits"}),
    # gh api -- only GET requests (enforced separately)
    "api": frozenset(),
}

# Flags that are never allowed (safety blocklist for injection attempts)
BLOCKED_FLAGS = frozenset({
    "--method", "-X",  # could change HTTP method on gh api
    "--input",         # could pipe data
    "--field", "-f",   # could add body fields
    "--raw-field", "-F",
})


def _validate_command(args: List[str]) -> None:
    """Validate that a gh command is read-only and allowed.

    Args:
        args: The gh arguments (without the leading 'gh'), e.g. ["issue", "list"]

    Raises:
        ValueError: if the command is not in the allowlist or contains blocked flags
    """
    if not args:
        raise ValueError("No gh arguments provided")

    subcmd = args[0]
    if subcmd not in ALLOWED_COMMANDS:
        raise ValueError(
            f"gh subcommand '{subcmd}' is not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )

    # Check for blocked flags anywhere in args
    for arg in args:
        if arg in BLOCKED_FLAGS:
            raise ValueError(f"Flag '{arg}' is not allowed (could mutate state)")

    # For 'api' subcommand, enforce extra restrictions
    if subcmd == "api":
        # No method flags means GET-only (default)
        # Already blocked --method/-X above, but double-check
        return

    # For other subcommands, check action is allowed
    allowed_actions = ALLOWED_COMMANDS[subcmd]
    if allowed_actions and len(args) > 1:
        action = args[1]
        # Skip flag-like arguments (e.g. --repo)
        if not action.startswith("-") and action not in allowed_actions:
            raise ValueError(
                f"gh {subcmd} {action} is not allowed. "
                f"Allowed actions: {', '.join(sorted(allowed_actions))}"
            )


def _classify_intent(query: str) -> str:
    """Classify the GitHub query intent from natural language.

    Returns one of: issues, issue_detail, prs, pr_detail, pr_diff,
    pr_checks, actions, action_detail, workflows, releases, repo_info,
    search_code, search_issues, search_prs, contributors, labels, milestones,
    fallback.
    """
    q = query.lower()

    # Issue detail (specific number)
    if re.search(r'issue\s*#?\d+', q) or re.search(r'#\d+.*issue', q):
        return "issue_detail"

    # PR detail (specific number)
    if re.search(r'(?:pr|pull request|pull)\s*#?\d+', q):
        return "pr_detail"

    # PR diff
    if any(w in q for w in ("pr diff", "pull request diff", "diff for pr", "diff for pull")):
        return "pr_diff"

    # PR checks / CI status
    if any(w in q for w in ("pr check", "ci status", "checks for", "pr status", "build status")):
        return "pr_checks"

    # Search intents (must come before general list intents)
    # Search code
    if any(w in q for w in ("search code", "find code", "code search", "grep github")):
        return "search_code"

    # Search issues
    if "search issue" in q or "find issue" in q:
        return "search_issues"

    # Search PRs
    if "search pr" in q or "search pull" in q or "find pr" in q:
        return "search_prs"

    # Issues list/search
    if any(w in q for w in ("issues", "bugs", "bug list", "open issues", "closed issues",
                             "issue list", "labeled")):
        return "issues"

    # PRs list
    if any(w in q for w in ("pull requests", "prs", "open prs", "merged prs",
                             "pr list", "pending review")):
        return "prs"

    # GitHub Actions / CI
    if any(w in q for w in ("actions", "workflow run", "ci run", "build",
                             "pipeline", "action run", "github action")):
        return "actions"

    # Action detail (specific run)
    if re.search(r'run\s*#?\d+', q) or "run detail" in q:
        return "action_detail"

    # Workflows
    if any(w in q for w in ("workflow", "workflows")):
        return "workflows"

    # Releases
    if any(w in q for w in ("release", "releases", "latest release", "version",
                             "tags", "changelog")):
        return "releases"

    # Contributors
    if any(w in q for w in ("contributor", "contributors", "who contributed",
                             "top contributor", "committers")):
        return "contributors"

    # Labels
    if any(w in q for w in ("labels", "label list", "available labels")):
        return "labels"

    # Milestones
    if any(w in q for w in ("milestone", "milestones")):
        return "milestones"

    # Repo info (general)
    if any(w in q for w in ("repo info", "repository info", "about the repo",
                             "repo description", "repo stats", "stars",
                             "forks", "language", "topics")):
        return "repo_info"

    return "fallback"


def _extract_number(query: str) -> Optional[int]:
    """Extract an issue/PR/run number from a query."""
    m = re.search(r'#?(\d+)', query)
    if m:
        return int(m.group(1))
    return None


def _extract_search_terms(query: str) -> str:
    """Extract search terms by stripping common intent words."""
    # Remove common preamble
    cleaned = re.sub(
        r'\b(search|find|look for|show|get|list|github|issues?|prs?|'
        r'pull requests?|code|in the repo)\b',
        '', query, flags=re.IGNORECASE
    ).strip()
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else query


class GitHubManager:
    """Read-only GitHub API access for the agentic loop."""

    def __init__(
        self,
        repo: Optional[str] = None,
        timeout: int = 15,
        max_output_lines: int = 80,
    ):
        """
        Args:
            repo: Optional owner/repo string (e.g. "lukehalleran/Daemon").
                  If None, uses the repo detected from the current git remote.
            timeout: Subprocess timeout in seconds.
            max_output_lines: Max lines of output before truncation.
        """
        self.repo = repo
        self.timeout = timeout
        self.max_output_lines = max_output_lines
        self._available: Optional[bool] = None
        self._detected_repo: Optional[str] = None

    def is_available(self) -> bool:
        """Check if gh CLI is installed and authenticated. Caches result."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True, timeout=5,
            )
            self._available = result.returncode == 0
            if self._available:
                # Try to detect repo from current directory
                self._detect_repo()
        except FileNotFoundError:
            self._available = False
        except Exception:
            self._available = False
        return self._available

    def _detect_repo(self) -> None:
        """Detect the current repo from git remotes."""
        if self.repo:
            self._detected_repo = self.repo
            return
        try:
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._detected_repo = result.stdout.strip()
        except Exception:
            pass

    def _get_repo_flag(self) -> List[str]:
        """Return ['--repo', 'owner/repo'] if we know the repo, else []."""
        repo = self.repo or self._detected_repo
        if repo:
            return ["--repo", repo]
        return []

    def _run_gh(self, args: List[str]) -> str:
        """Run a gh command safely with validation, timeout, and output capping.

        Args:
            args: gh arguments (without leading 'gh'), e.g. ["issue", "list"]

        Returns:
            Command output as string, truncated if needed.

        Raises:
            ValueError: if the command is not allowed
            RuntimeError: if the gh command fails
        """
        _validate_command(args)

        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[:300]
                raise RuntimeError(f"gh {args[0]} failed: {stderr}")

            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > self.max_output_lines:
                truncated = lines[:self.max_output_lines]
                remaining = len(lines) - self.max_output_lines
                truncated.append(f"[...truncated, {remaining} more lines]")
                return "\n".join(truncated)
            return output

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gh {args[0]} timed out after {self.timeout}s")

    # ------------------------------------------------------------------
    # High-level query execution
    # ------------------------------------------------------------------

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Parse natural-language query and run appropriate gh commands.

        Args:
            query: Natural language description, e.g. "open issues labeled bug"

        Returns:
            Dict with success, query, commands_run, output, summary keys.
        """
        if not self.is_available():
            return {
                "success": False,
                "query": query,
                "commands_run": [],
                "output": "",
                "summary": "GitHub CLI (gh) not available or not authenticated",
            }

        intent = _classify_intent(query)
        repo_flag = self._get_repo_flag()
        commands_run: List[str] = []
        output_parts: List[str] = []
        summary = ""

        try:
            if intent == "issues":
                output, cmds, summary = self._query_issues(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "issue_detail":
                output, cmds, summary = self._query_issue_detail(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "prs":
                output, cmds, summary = self._query_prs(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "pr_detail":
                output, cmds, summary = self._query_pr_detail(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "pr_diff":
                output, cmds, summary = self._query_pr_diff(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "pr_checks":
                output, cmds, summary = self._query_pr_checks(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "actions":
                output, cmds, summary = self._query_actions(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "action_detail":
                output, cmds, summary = self._query_action_detail(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "workflows":
                output, cmds, summary = self._query_workflows(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "releases":
                output, cmds, summary = self._query_releases(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "repo_info":
                output, cmds, summary = self._query_repo_info(repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "search_code":
                output, cmds, summary = self._query_search(query, "code", repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "search_issues":
                output, cmds, summary = self._query_search(query, "issues", repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "search_prs":
                output, cmds, summary = self._query_search(query, "prs", repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "contributors":
                output, cmds, summary = self._query_contributors(repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "labels":
                output, cmds, summary = self._query_labels(repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            elif intent == "milestones":
                output, cmds, summary = self._query_milestones(repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

            else:
                # Fallback: show repo overview + recent issues + PRs
                output, cmds, summary = self._query_fallback(query, repo_flag)
                output_parts.append(output)
                commands_run.extend(cmds)

        except (ValueError, RuntimeError) as e:
            return {
                "success": False,
                "query": query,
                "commands_run": commands_run,
                "output": str(e),
                "summary": f"Error: {e}",
            }

        return {
            "success": True,
            "query": query,
            "commands_run": commands_run,
            "output": "\n".join(output_parts),
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _query_issues(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        q = query.lower()
        args = ["issue", "list"] + repo_flag + ["--limit", "15"]

        if "closed" in q:
            args.extend(["--state", "closed"])
        elif "all" in q:
            args.extend(["--state", "all"])
        # else default is open

        # Label filter
        label_match = re.search(r'label[ed]*\s+["\']?([^"\']+)["\']?', q)
        if label_match:
            args.extend(["--label", label_match.group(1).strip()])

        # Assignee filter
        assignee_match = re.search(r'assign(?:ed)?\s+(?:to\s+)?(\S+)', q)
        if assignee_match:
            args.extend(["--assignee", assignee_match.group(1)])

        output = self._run_gh(args)
        cmd_str = f"gh {' '.join(args)}"
        line_count = len(output.split("\n")) if output else 0
        return output or "(no issues found)", [cmd_str], f"{line_count} issues"

    def _query_issue_detail(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        num = _extract_number(query)
        if not num:
            return "(could not parse issue number)", [], "No issue number found"
        args = ["issue", "view", str(num)] + repo_flag
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], f"Issue #{num}"

    def _query_prs(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        q = query.lower()
        args = ["pr", "list"] + repo_flag + ["--limit", "15"]

        if "merged" in q:
            args.extend(["--state", "merged"])
        elif "closed" in q:
            args.extend(["--state", "closed"])
        elif "all" in q:
            args.extend(["--state", "all"])

        # Label filter
        label_match = re.search(r'label[ed]*\s+["\']?([^"\']+)["\']?', q)
        if label_match:
            args.extend(["--label", label_match.group(1).strip()])

        output = self._run_gh(args)
        cmd_str = f"gh {' '.join(args)}"
        line_count = len(output.split("\n")) if output else 0
        return output or "(no pull requests found)", [cmd_str], f"{line_count} PRs"

    def _query_pr_detail(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        num = _extract_number(query)
        if not num:
            return "(could not parse PR number)", [], "No PR number found"
        args = ["pr", "view", str(num)] + repo_flag
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], f"PR #{num}"

    def _query_pr_diff(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        num = _extract_number(query)
        if not num:
            return "(could not parse PR number for diff)", [], "No PR number found"
        args = ["pr", "diff", str(num)] + repo_flag
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], f"PR #{num} diff"

    def _query_pr_checks(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        num = _extract_number(query)
        if not num:
            return "(could not parse PR number for checks)", [], "No PR number found"
        args = ["pr", "checks", str(num)] + repo_flag
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], f"PR #{num} checks"

    def _query_actions(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        args = ["run", "list"] + repo_flag + ["--limit", "10"]
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no runs found)", [f"gh {' '.join(args)}"], f"{line_count} workflow runs"

    def _query_action_detail(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        num = _extract_number(query)
        if not num:
            return "(could not parse run number)", [], "No run number found"
        args = ["run", "view", str(num)] + repo_flag
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], f"Run #{num}"

    def _query_workflows(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        args = ["workflow", "list"] + repo_flag
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no workflows found)", [f"gh {' '.join(args)}"], f"{line_count} workflows"

    def _query_releases(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        args = ["release", "list"] + repo_flag + ["--limit", "10"]
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no releases found)", [f"gh {' '.join(args)}"], f"{line_count} releases"

    def _query_repo_info(
        self, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        # gh repo view takes the repo as a positional arg, not --repo flag
        repo = self.repo or self._detected_repo
        args = ["repo", "view"]
        if repo:
            args.append(repo)
        output = self._run_gh(args)
        return output, [f"gh {' '.join(args)}"], "Repository info"

    def _query_search(
        self, query: str, search_type: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        terms = _extract_search_terms(query)
        repo = self.repo or self._detected_repo

        args = ["search", search_type, terms, "--limit", "10"]
        if repo and search_type in ("code", "issues", "prs", "commits"):
            args.extend(["--repo", repo])

        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return (
            output or f"(no {search_type} results)",
            [f"gh {' '.join(args)}"],
            f"{line_count} {search_type} results",
        )

    def _query_contributors(
        self, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        repo = self.repo or self._detected_repo
        if not repo:
            return "(no repo detected)", [], "Cannot query contributors without repo"
        args = ["api", f"repos/{repo}/contributors", "--jq",
                '.[] | "\\(.login) (\\(.contributions) commits)"']
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no contributors)", [f"gh {' '.join(args)}"], f"{line_count} contributors"

    def _query_labels(
        self, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        repo = self.repo or self._detected_repo
        if not repo:
            return "(no repo detected)", [], "Cannot query labels without repo"
        args = ["api", f"repos/{repo}/labels", "--jq", '.[] | "\\(.name): \\(.description // "")"']
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no labels)", [f"gh {' '.join(args)}"], f"{line_count} labels"

    def _query_milestones(
        self, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        repo = self.repo or self._detected_repo
        if not repo:
            return "(no repo detected)", [], "Cannot query milestones without repo"
        args = ["api", f"repos/{repo}/milestones", "--jq",
                '.[] | "\\(.title) [\\(.state)] (\\(.open_issues) open, \\(.closed_issues) closed)"']
        output = self._run_gh(args)
        line_count = len(output.split("\n")) if output else 0
        return output or "(no milestones)", [f"gh {' '.join(args)}"], f"{line_count} milestones"

    def _query_fallback(
        self, query: str, repo_flag: List[str]
    ) -> Tuple[str, List[str], str]:
        """Fallback: repo info + recent issues + PRs."""
        parts = []
        cmds = []

        try:
            _repo = self.repo or self._detected_repo
            _repo_args = ["repo", "view"] + ([_repo] if _repo else [])
            repo_output = self._run_gh(_repo_args)
            parts.append(f"=== REPOSITORY ===\n{repo_output}")
            cmds.append(f"gh {' '.join(_repo_args)}")
        except (ValueError, RuntimeError):
            pass

        try:
            issues_output = self._run_gh(["issue", "list"] + repo_flag + ["--limit", "5"])
            parts.append(f"\n=== RECENT ISSUES ===\n{issues_output or '(none)'}")
            cmds.append("gh issue list --limit 5")
        except (ValueError, RuntimeError):
            pass

        try:
            pr_output = self._run_gh(["pr", "list"] + repo_flag + ["--limit", "5"])
            parts.append(f"\n=== RECENT PRs ===\n{pr_output or '(none)'}")
            cmds.append("gh pr list --limit 5")
        except (ValueError, RuntimeError):
            pass

        return "\n".join(parts) or "(no data)", cmds, "Repository overview"

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format result dict into a text block for the LLM context."""
        if not result.get("success"):
            return f"GitHub query failed: {result.get('summary', 'unknown error')}"

        parts = [f"Summary: {result['summary']}"]
        if result.get("commands_run"):
            parts.append(f"Commands: {'; '.join(result['commands_run'])}")
        if result.get("output"):
            parts.append(f"\n{result['output']}")
        return "\n".join(parts)
