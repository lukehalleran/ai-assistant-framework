"""
# knowledge/git_memory.py

Module Contract
- Purpose: Extract git commit history as structured dicts for PROCEDURAL memory population.
- Class: GitMemoryExtractor(repo_path)
- Key methods:
  - extract_commits(limit, since, include_diffs, diff_max_lines) -> List[Dict]
    Parses git log output into memory-ready dicts (hash, subject, body, author, timestamp,
    tags, change_type, optional structured diff stats). Returns newest first.
  - get_recent_since_hash(last_hash) -> List[Dict]
    Returns commits newer than given hash (for incremental sync).
  - get_hot_files(since_days, limit, exclude_globs) -> List[Dict]
    Ranks files by recent churn (commit frequency in a window) — the active dev
    frontier. Single read-only `git log --name-only` call; no LLM. Consumers can
    bias code proposals / context toward actively-evolving files.
  - _get_commit_diff_stats(commit_hash, max_files) -> Dict  [structured --numstat stats]
  - _derive_change_type(tags) -> str  [feature/bugfix/refactor/... or 'other']
  - _extract_tags(subject) -> List[str]  [conventional commit prefixes: feat, fix, etc.]
- Outputs:
  - extract_commits: list of dicts with keys id, content (formatted for embedding),
    and metadata. Metadata always carries commit_hash, full_hash, author, age_relative,
    timestamp, source, memory_type, tags, change_type; when include_diffs is set it also
    carries files_changed (comma-joined, post-rename), files_changed_count, lines_added,
    lines_removed — filterable per-commit fields, not just an embedded text blob.
  - get_hot_files: list of {path, commits, last_touched, age_relative} ranked by churn.
- Dependencies:
  - git CLI (subprocess calls to git log / git show, read-only)
- Side effects:
  - Subprocess calls to local git repo (read-only)
"""

import subprocess
import re
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Separator unlikely to appear in commit messages
_SEP = "|||"


class GitMemoryExtractor:
    """Extract git history as procedural memories."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def extract_commits(
        self,
        limit: int = 200,
        since: Optional[str] = None,
        include_diffs: bool = False,
        diff_max_lines: int = 50,
    ) -> List[Dict]:
        """
        Extract commit history as memory-ready dicts.

        Args:
            limit: Max commits to extract.
            since: Only commits after this date (e.g. "2025-01-01").
            include_diffs: Whether to include --stat diff summaries.
            diff_max_lines: Truncate diff stats longer than this.

        Returns:
            List of dicts ready for ChromaDB storage, newest first.
        """
        # Use %x00 as record separator so multi-line bodies don't break parsing
        format_str = f"%H{_SEP}%s{_SEP}%b{_SEP}%ar{_SEP}%aI{_SEP}%an%x00"
        cmd = [
            "git", "log",
            f"--pretty=format:{format_str}",
            f"-n{limit}",
        ]
        if since:
            cmd.append(f"--since={since}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Git command failed: {e}")
            return []

        if result.returncode != 0:
            logger.error(f"Git log failed: {result.stderr}")
            return []

        memories = []
        for record in result.stdout.split("\x00"):
            record = record.strip()
            if not record:
                continue

            parts = record.split(_SEP, maxsplit=5)
            if len(parts) < 6:
                continue

            hash_full, subject, body, age_relative, timestamp, author = parts
            hash_short = hash_full[:8]

            # Collapse any newlines in body to single spaces
            body_clean = " ".join(body.split())

            # Build content
            content = f"Commit: {subject}"
            if body_clean:
                content += f"\n\n{body_clean}"

            tags = self._extract_tags(subject)

            metadata = {
                "commit_hash": hash_short,
                "full_hash": hash_full,
                "author": author,
                "age_relative": age_relative,
                "timestamp": timestamp,
                "source": "git",
                "memory_type": "procedural",
                "tags": ",".join(tags),
                # Derived from the conventional-commit tag — free, no extra git call.
                "change_type": self._derive_change_type(tags),
            }

            # Optionally enrich with structured, filterable diff stats (one git call/commit)
            if include_diffs:
                stats = self._get_commit_diff_stats(hash_full, diff_max_lines)
                if stats["summary_text"]:
                    content += f"\n\nChanges:\n{stats['summary_text']}"
                metadata["files_changed"] = ",".join(stats["files_changed"])
                metadata["files_changed_count"] = stats["files_changed_count"]
                metadata["lines_added"] = stats["lines_added"]
                metadata["lines_removed"] = stats["lines_removed"]

            memory = {
                "id": f"git-{hash_short}",
                "content": content,
                "metadata": metadata,
            }
            memories.append(memory)

        logger.info(f"Extracted {len(memories)} commits")
        return memories

    # Conventional-commit tag -> single change_type label.
    _CHANGE_TYPE_TAGS = frozenset({
        "feature", "bugfix", "refactor", "documentation", "testing",
        "maintenance", "performance", "style", "build", "ci-cd",
    })

    def _derive_change_type(self, tags: List[str]) -> str:
        """Map the first recognised conventional-commit tag to a change_type.

        Returns 'other' when the subject carried no conventional prefix.
        """
        for tag in tags:
            if tag in self._CHANGE_TYPE_TAGS:
                return tag
        return "other"

    @staticmethod
    def _rename_new_path(path_field: str) -> str:
        """Resolve the post-rename path from a numstat path field.

        numstat encodes renames as 'old => new' or with a brace segment like
        'dir/{old => new}/file.py'. Return the new path; plain paths pass through.
        """
        path_field = path_field.strip()
        if "=>" not in path_field:
            return path_field
        if "{" in path_field and "}" in path_field:
            # dir/{old => new}/file.py  ->  dir/new/file.py
            pre, rest = path_field.split("{", 1)
            inner, post = rest.split("}", 1)
            new_part = inner.split("=>", 1)[1].strip()
            return f"{pre}{new_part}{post}".replace("//", "/")
        # old.py => new.py  ->  new.py
        return path_field.split("=>", 1)[1].strip()

    def _get_commit_diff_stats(self, commit_hash: str, max_files: int) -> Dict:
        """Parse `git show --numstat` into structured, filterable diff stats.

        Returns a dict with:
          files_changed:       list of (post-rename) paths, capped at max_files
          files_changed_count: true count (before capping)
          lines_added:         total insertions (binary files count 0)
          lines_removed:       total deletions
          summary_text:        human-readable per-file summary for embedding
        """
        empty = {
            "files_changed": [],
            "files_changed_count": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "summary_text": "",
        }
        cmd = ["git", "show", "--numstat", "--format=", commit_hash]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError):
            return empty
        if result.returncode != 0:
            return empty

        files: List[str] = []
        total_add = 0
        total_del = 0
        summary_lines: List[str] = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            add_s, del_s = parts[0], parts[1]
            path = self._rename_new_path("\t".join(parts[2:]))
            # Binary files report '-' instead of counts.
            add_n = int(add_s) if add_s.isdigit() else 0
            del_n = int(del_s) if del_s.isdigit() else 0
            total_add += add_n
            total_del += del_n
            files.append(path)
            summary_lines.append(f"{path} (+{add_n} -{del_n})")

        if not files:
            return empty

        if len(summary_lines) > max_files:
            hidden = len(summary_lines) - max_files
            summary_lines = summary_lines[:max_files] + [f"... ({hidden} more files)"]

        return {
            "files_changed": files[:max_files],
            "files_changed_count": len(files),
            "lines_added": total_add,
            "lines_removed": total_del,
            "summary_text": "\n".join(summary_lines),
        }

    def get_hot_files(
        self,
        since_days: int = 90,
        limit: int = 20,
        exclude_globs: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Rank files by recent churn (commit frequency) — the active dev frontier.

        Aggregates `git log --name-only` over the last `since_days` days into a
        per-file commit count + most-recent-touch timestamp. No LLM, no diff
        parsing; a single read-only git call. Consumers (e.g. the code-proposal
        generator) can bias toward actively-evolving files.

        Args:
            since_days: Look-back window in days.
            limit: Max files to return.
            exclude_globs: Optional substrings; a path containing any is skipped
                (e.g. 'data/', 'venv/').

        Returns:
            List of {path, commits, last_touched, age_relative} dicts, ranked by
            commit count (desc), then recency. Empty on error / non-repo.
        """
        marker = "__COMMIT__"
        fmt = f"{marker}%H|%aI|%ar"
        cmd = [
            "git", "log",
            f"--since={since_days} days ago",
            f"--pretty=format:{fmt}",
            "--name-only",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Git command failed: {e}")
            return []
        if result.returncode != 0:
            logger.error(f"git log (hot files) failed: {result.stderr}")
            return []

        counts: Dict[str, int] = {}
        last_iso: Dict[str, str] = {}
        last_rel: Dict[str, str] = {}
        cur_iso = ""
        cur_rel = ""
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(marker):
                bits = line[len(marker):].split("|", 2)
                cur_iso = bits[1] if len(bits) > 1 else ""
                cur_rel = bits[2] if len(bits) > 2 else ""
                continue
            path = line
            if exclude_globs and any(g in path for g in exclude_globs):
                continue
            counts[path] = counts.get(path, 0) + 1
            # Walking newest->oldest, the first sighting of a file is its latest touch.
            if path not in last_iso:
                last_iso[path] = cur_iso
                last_rel[path] = cur_rel

        ranked = sorted(
            counts.keys(),
            key=lambda p: (counts[p], last_iso.get(p, "")),
            reverse=True,
        )
        hot = [
            {
                "path": p,
                "commits": counts[p],
                "last_touched": last_iso.get(p, ""),
                "age_relative": last_rel.get(p, ""),
            }
            for p in ranked[:limit]
        ]
        logger.info(
            f"Hot files: {len(counts)} files churned in {since_days}d, top {len(hot)}"
        )
        return hot

    def _extract_tags(self, subject: str) -> List[str]:
        """Extract tags from conventional commit format."""
        tags = ["git-commit"]

        patterns = {
            r"^feat": "feature",
            r"^fix": "bugfix",
            r"^refactor": "refactor",
            r"^docs": "documentation",
            r"^test": "testing",
            r"^chore": "maintenance",
            r"^perf": "performance",
            r"^style": "style",
            r"^build": "build",
            r"^ci": "ci-cd",
        }

        subject_lower = subject.lower().strip()
        for pattern, tag in patterns.items():
            if re.match(pattern, subject_lower):
                tags.append(tag)
                break

        if "wip" in subject_lower:
            tags.append("work-in-progress")
        if "breaking" in subject_lower:
            tags.append("breaking-change")
        if "hotfix" in subject_lower:
            tags.append("hotfix")

        return tags

    def get_recent_since_hash(self, last_hash: str) -> List[Dict]:
        """Get commits since a specific hash (for incremental updates)."""
        cmd = [
            "git", "rev-list",
            "--count",
            f"{last_hash}..HEAD",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Git command failed: {e}")
            return []

        if result.returncode != 0:
            logger.error(f"git rev-list failed: {result.stderr}")
            return []

        new_count = int(result.stdout.strip()) if result.stdout.strip() else 0
        if new_count == 0:
            return []

        return self.extract_commits(limit=new_count)
