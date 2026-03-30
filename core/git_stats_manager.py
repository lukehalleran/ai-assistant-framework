"""
Git Stats Manager — Read-only git repository stats for the agentic loop.

Contract:
    - Provides GitStatsManager for querying local git repository activity
    - Read-only: only ALLOWED_SUBCOMMANDS may execute (no push, reset, checkout, etc.)
    - Keyword-based intent parsing maps natural-language queries to git commands
    - Temporal phrase extraction converts "this week", "today", etc. to --since dates
    - All subprocess calls use timeouts and output capping
    - Results returned as structured dicts, formatted for LLM prompt injection

Public Interface:
    - GitStatsManager(timeout, max_output_lines)
    - is_available() -> bool  (checks if cwd is inside a git repo)
    - execute_query(query) -> Dict[str, Any]  (parse intent, run git, return results)
    - format_for_prompt(result) -> str  (format result dict for LLM context)

Dependencies:
    - subprocess (stdlib)
    - config.app_config (GIT_STATS_ENABLED, GIT_STATS_TIMEOUT, GIT_STATS_MAX_OUTPUT_LINES)
"""

import logging
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Safety: only these git subcommands are allowed
ALLOWED_SUBCOMMANDS = frozenset({
    "log", "shortlog", "diff", "status", "branch", "rev-list",
    "rev-parse", "show", "describe", "tag", "stash",
})

# Temporal phrase patterns → timedelta offsets
_TEMPORAL_PHRASES: List[Tuple[re.Pattern, timedelta]] = [
    (re.compile(r'\btoday\b', re.I), timedelta(days=0)),
    (re.compile(r'\byesterday\b', re.I), timedelta(days=1)),
    (re.compile(r'\bthis week\b', re.I), timedelta(weeks=1)),
    (re.compile(r'\blast (\d+) days?\b', re.I), None),  # dynamic
    (re.compile(r'\blast (\d+) weeks?\b', re.I), None),  # dynamic
    (re.compile(r'\bthis month\b', re.I), timedelta(days=30)),
    (re.compile(r'\blast month\b', re.I), timedelta(days=60)),
    (re.compile(r'\bthis year\b', re.I), timedelta(days=365)),
]


def _parse_time_window(query: str) -> Optional[datetime]:
    """Extract a temporal window from a natural-language query.

    Returns a datetime representing the start of the window, or None
    if no temporal phrase is found.
    """
    now = datetime.now()

    # "today" → start of today
    if re.search(r'\btoday\b', query, re.I):
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # "yesterday"
    if re.search(r'\byesterday\b', query, re.I):
        yesterday = now - timedelta(days=1)
        return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    # "this week"
    if re.search(r'\bthis week\b', query, re.I):
        return now - timedelta(weeks=1)

    # "last N days"
    m = re.search(r'\blast (\d+) days?\b', query, re.I)
    if m:
        return now - timedelta(days=int(m.group(1)))

    # "last N weeks"
    m = re.search(r'\blast (\d+) weeks?\b', query, re.I)
    if m:
        return now - timedelta(weeks=int(m.group(1)))

    # "this month"
    if re.search(r'\bthis month\b', query, re.I):
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # "last month"
    if re.search(r'\blast month\b', query, re.I):
        return now - timedelta(days=60)

    # "this year"
    if re.search(r'\bthis year\b', query, re.I):
        return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    return None


def _classify_intent(query: str) -> str:
    """Classify the git stats intent from a natural-language query.

    Returns one of: commit_count, recent_commits, files_changed,
    contributors, branches, status, diff_stat, fallback.
    """
    q = query.lower()

    # Commit count queries
    if any(w in q for w in ("how many commits", "commit count", "number of commits", "total commits")):
        return "commit_count"

    # Files changed queries
    if any(w in q for w in ("files changed", "what changed", "what files", "changes to")):
        return "files_changed"

    # Contributors / authors
    if any(w in q for w in ("contributor", "author", "who committed", "who has been")):
        return "contributors"

    # Branch queries (but not "commits on this branch")
    if any(w in q for w in ("branch", "branches")) and "commit" not in q:
        return "branches"

    # Status / uncommitted
    if any(w in q for w in ("status", "uncommitted", "staged", "untracked", "working tree", "dirty")):
        return "status"

    # Diff stat
    if any(w in q for w in ("lines changed", "line count", "insertions", "deletions", "diff stat", "lines added", "lines removed")):
        return "diff_stat"

    # Recent commits (generic)
    if any(w in q for w in ("recent commits", "latest commits", "last commit", "commit history", "commit log", "commits")):
        return "recent_commits"

    return "fallback"


class GitStatsManager:
    """Read-only git repository stats for the agentic loop."""

    def __init__(self, timeout: int = 10, max_output_lines: int = 50):
        self.timeout = timeout
        self.max_output_lines = max_output_lines
        self._repo_root: Optional[str] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if we're inside a git repo. Caches result."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                self._repo_root = result.stdout.strip()
                self._available = True
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def _run_git(self, args: List[str]) -> str:
        """Run a git command safely with timeout and output capping.

        Args:
            args: git subcommand + arguments (e.g. ["log", "--oneline", "-10"])

        Returns:
            Command output as string, truncated if needed.

        Raises:
            ValueError: if the subcommand is not in ALLOWED_SUBCOMMANDS
            RuntimeError: if the git command fails
        """
        if not args:
            raise ValueError("No git arguments provided")

        subcommand = args[0]
        if subcommand not in ALLOWED_SUBCOMMANDS:
            raise ValueError(f"Git subcommand '{subcommand}' is not allowed (read-only commands only)")

        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=self._repo_root,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[:200]
                raise RuntimeError(f"git {subcommand} failed: {stderr}")

            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > self.max_output_lines:
                truncated = lines[:self.max_output_lines]
                remaining = len(lines) - self.max_output_lines
                truncated.append(f"[...truncated, {remaining} more lines]")
                return "\n".join(truncated)
            return output

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git {subcommand} timed out after {self.timeout}s")

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Parse natural-language query and run appropriate git commands.

        Args:
            query: Natural language description, e.g. "commits this week"

        Returns:
            Dict with success, query, commands_run, output, summary keys.
        """
        if not self.is_available():
            return {
                "success": False,
                "query": query,
                "commands_run": [],
                "output": "",
                "summary": "Not in a git repository",
            }

        intent = _classify_intent(query)
        since_dt = _parse_time_window(query)
        since_arg = f"--since={since_dt.isoformat()}" if since_dt else None
        since_label = since_dt.strftime("%Y-%m-%d") if since_dt else None

        commands_run = []
        output_parts = []
        summary = ""

        try:
            if intent == "commit_count":
                # Count + short list
                count_args = ["rev-list", "--count", "HEAD"]
                if since_arg:
                    count_args = ["rev-list", "--count", since_arg, "HEAD"]
                count_output = self._run_git(count_args)
                commands_run.append(f"git {' '.join(count_args)}")

                count = count_output.strip()
                time_desc = f"since {since_label}" if since_label else "total"
                summary = f"{count} commits {time_desc}"
                output_parts.append(f"Commit count ({time_desc}): {count}")

                # Also show recent ones for context
                log_args = ["log", "--oneline", "-10"]
                if since_arg:
                    log_args = ["log", "--oneline", since_arg]
                log_output = self._run_git(log_args)
                commands_run.append(f"git {' '.join(log_args)}")
                if log_output:
                    output_parts.append(f"\nRecent commits:\n{log_output}")

            elif intent == "recent_commits":
                log_args = ["log", "--oneline", "-15"]
                if since_arg:
                    log_args = ["log", "--oneline", since_arg]
                log_output = self._run_git(log_args)
                commands_run.append(f"git {' '.join(log_args)}")
                commit_count = len(log_output.split("\n")) if log_output else 0
                time_desc = f"since {since_label}" if since_label else "recent"
                summary = f"{commit_count} commits ({time_desc})"
                output_parts.append(log_output or "(no commits)")

            elif intent == "files_changed":
                if since_arg:
                    # Files changed in committed history
                    log_args = ["log", since_arg, "--name-only", "--pretty=format:"]
                    log_output = self._run_git(log_args)
                    commands_run.append(f"git {' '.join(log_args)}")
                    # Deduplicate file list
                    files = sorted(set(f.strip() for f in log_output.split("\n") if f.strip()))
                    output_parts.append(f"Files changed since {since_label} ({len(files)} files):")
                    output_parts.append("\n".join(files[:self.max_output_lines]))
                    summary = f"{len(files)} files changed since {since_label}"
                else:
                    # Uncommitted changes
                    diff_output = self._run_git(["diff", "--name-only"])
                    commands_run.append("git diff --name-only")
                    status_output = self._run_git(["status", "--short"])
                    commands_run.append("git status --short")
                    output_parts.append(f"Uncommitted changes:\n{status_output or '(clean)'}")
                    if diff_output:
                        output_parts.append(f"\nModified files:\n{diff_output}")
                    summary = "Current uncommitted changes"

            elif intent == "contributors":
                shortlog_args = ["shortlog", "-sn", "--no-merges"]
                if since_arg:
                    shortlog_args.append(since_arg)
                shortlog_output = self._run_git(shortlog_args)
                commands_run.append(f"git {' '.join(shortlog_args)}")
                time_desc = f"since {since_label}" if since_label else "all time"
                summary = f"Contributors ({time_desc})"
                output_parts.append(shortlog_output or "(no contributors)")

            elif intent == "branches":
                branch_output = self._run_git(["branch", "-a"])
                commands_run.append("git branch -a")
                branch_count = len([l for l in branch_output.split("\n") if l.strip()])
                summary = f"{branch_count} branches"
                output_parts.append(branch_output)

            elif intent == "status":
                status_output = self._run_git(["status", "--short"])
                commands_run.append("git status --short")
                if not status_output:
                    summary = "Working tree clean"
                    output_parts.append("Working tree clean — no uncommitted changes.")
                else:
                    line_count = len(status_output.split("\n"))
                    summary = f"{line_count} changed files"
                    output_parts.append(status_output)

            elif intent == "diff_stat":
                if since_arg:
                    # Use diff --stat against merge-base
                    # Find the oldest commit in the window
                    oldest_args = ["log", since_arg, "--pretty=format:%H", "--reverse"]
                    oldest_output = self._run_git(oldest_args)
                    commands_run.append(f"git {' '.join(oldest_args)}")
                    if oldest_output:
                        oldest_hash = oldest_output.split("\n")[0].strip()
                        diff_args = ["diff", "--stat", f"{oldest_hash}^...HEAD"]
                        try:
                            diff_output = self._run_git(diff_args)
                            commands_run.append(f"git {' '.join(diff_args)}")
                            output_parts.append(diff_output)
                            summary = f"Diff stat since {since_label}"
                        except RuntimeError:
                            # Fallback: just show the shortstat from log
                            stat_args = ["log", since_arg, "--shortstat", "--pretty=format:"]
                            stat_output = self._run_git(stat_args)
                            commands_run.append(f"git {' '.join(stat_args)}")
                            output_parts.append(stat_output or "(no changes)")
                            summary = f"Change stats since {since_label}"
                    else:
                        output_parts.append("(no commits in this time window)")
                        summary = "No commits in time window"
                else:
                    # Uncommitted diff stat
                    diff_output = self._run_git(["diff", "--stat"])
                    commands_run.append("git diff --stat")
                    output_parts.append(diff_output or "(no uncommitted changes)")
                    summary = "Uncommitted diff stat"

            else:
                # Fallback: recent log
                log_output = self._run_git(["log", "--oneline", "-10"])
                commands_run.append("git log --oneline -10")
                summary = "Last 10 commits"
                output_parts.append(log_output or "(no commits)")

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

    def format_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format result dict into a text block for the LLM context."""
        if not result.get("success"):
            return f"Git query failed: {result.get('summary', 'unknown error')}"

        parts = [f"Summary: {result['summary']}"]
        if result.get("commands_run"):
            parts.append(f"Commands: {'; '.join(result['commands_run'])}")
        if result.get("output"):
            parts.append(f"\n{result['output']}")
        return "\n".join(parts)
