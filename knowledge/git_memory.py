"""
Git commit history extractor for PROCEDURAL memory population.

Extracts commit messages, diffs (optional), and metadata to provide
Daemon with visibility into project evolution and decision rationale.
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

            # Optionally add diff summary
            if include_diffs:
                diff = self._get_diff_summary(hash_full, diff_max_lines)
                if diff:
                    content += f"\n\nChanges:\n{diff}"

            tags = self._extract_tags(subject)

            memory = {
                "id": f"git-{hash_short}",
                "content": content,
                "metadata": {
                    "commit_hash": hash_short,
                    "full_hash": hash_full,
                    "author": author,
                    "age_relative": age_relative,
                    "timestamp": timestamp,
                    "source": "git",
                    "memory_type": "procedural",
                    "tags": ",".join(tags),
                },
            }
            memories.append(memory)

        logger.info(f"Extracted {len(memories)} commits")
        return memories

    def _get_diff_summary(self, commit_hash: str, max_lines: int) -> str:
        """Get abbreviated --stat diff for a commit."""
        cmd = [
            "git", "show",
            "--stat",
            "--format=",
            commit_hash,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError):
            return ""
        if result.returncode != 0:
            return ""

        lines = result.stdout.strip().split("\n")
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more files)"]
        return "\n".join(lines)

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
