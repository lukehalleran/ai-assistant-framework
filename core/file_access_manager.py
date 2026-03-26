"""
File Access Manager — Approved-Folder Read & Grep for Agentic Loop

Module Contract:
    - Purpose: Provides read-only file access (read, grep, list) restricted to
      a configurable whitelist of directories. Used by the agentic ReAct loop
      so the LLM can inspect source code, configs, logs, and notes on disk.
    - Security:
      - Path.resolve() + is_relative_to() prevents path traversal
      - Extension whitelist prevents binary reads
      - Size limits prevent context flooding
      - Symlinks resolved before validation (no escape via symlinks)
    - Inputs: File paths, grep patterns, directory paths
    - Outputs: Structured dicts with content, metadata, success/error
    - Dependencies: None (pure stdlib)
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from utils.logging_utils import get_logger

logger = get_logger("file_access_manager")


class FileAccessManager:
    """Read-only file access restricted to approved directories."""

    def __init__(
        self,
        approved_folders: List[str],
        max_read_bytes: int = 100_000,
        max_grep_results: int = 25,
        max_list_entries: int = 200,
        allowed_extensions: Optional[List[str]] = None,
    ):
        self.approved_folders = [Path(f).expanduser().resolve() for f in approved_folders]
        self.max_read_bytes = max_read_bytes
        self.max_grep_results = max_grep_results
        self.max_list_entries = max_list_entries
        self.allowed_extensions: Set[str] = set(allowed_extensions) if allowed_extensions else set()

    def is_available(self) -> bool:
        """Check if file access is configured with at least one approved folder."""
        return len(self.approved_folders) > 0

    def _validate_path(self, filepath: str) -> Path:
        """
        Resolve path and verify it's within an approved folder.

        Raises PermissionError if path is outside approved folders or
        has a disallowed extension.
        """
        resolved = Path(filepath).expanduser().resolve()

        if not any(
            resolved == folder or resolved.is_relative_to(folder)
            for folder in self.approved_folders
        ):
            raise PermissionError(f"Access denied: {filepath} not in approved folders")

        if (
            resolved.suffix
            and self.allowed_extensions
            and resolved.suffix.lower() not in self.allowed_extensions
        ):
            raise PermissionError(
                f"File type {resolved.suffix} not in allowed extensions"
            )

        return resolved

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read file contents with optional line range.

        Returns dict with: content, path, lines, truncated, success, error
        """
        try:
            resolved = self._validate_path(filepath)
        except PermissionError as e:
            return {"error": str(e), "success": False}

        if not resolved.is_file():
            return {"error": f"Not a file: {filepath}", "success": False}

        file_size = resolved.stat().st_size
        truncated = file_size > self.max_read_bytes

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                if start_line or end_line:
                    lines = f.readlines()
                    start = max(0, (start_line or 1) - 1)
                    end = end_line or len(lines)
                    selected = lines[start:end]
                    content = "".join(selected)
                    total_lines = len(lines)
                else:
                    content = f.read(self.max_read_bytes)
                    total_lines = content.count("\n") + (1 if content else 0)

            return {
                "content": content,
                "path": str(resolved),
                "lines": total_lines,
                "truncated": truncated,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"[FileAccess] Error reading {filepath}: {e}")
            return {"error": f"Read error: {e}", "success": False}

    async def grep_files(
        self,
        pattern: str,
        folder: Optional[str] = None,
        file_glob: str = "*",
        case_sensitive: bool = False,
        context_lines: int = 2,
    ) -> Dict[str, Any]:
        """
        Grep for pattern across approved folders.

        Uses subprocess grep for performance on large trees.
        Returns dict with: matches, total_matches, truncated, success, error
        """
        # Determine search root
        if folder:
            try:
                search_root = self._validate_path(folder)
            except PermissionError as e:
                return {"error": str(e), "success": False}
            if not search_root.is_dir():
                return {"error": f"Not a directory: {folder}", "success": False}
            search_dirs = [search_root]
        else:
            search_dirs = [f for f in self.approved_folders if f.is_dir()]

        if not search_dirs:
            return {"error": "No valid search directories", "success": False}

        all_matches = []
        for search_dir in search_dirs:
            try:
                cmd = ["grep", "-rn"]
                if not case_sensitive:
                    cmd.append("-i")
                if context_lines > 0:
                    cmd.append(f"-C{context_lines}")
                cmd.append(f"--include={file_glob}")
                # Exclude common non-useful dirs
                for excl in ["__pycache__", ".git", "venv", ".venv", "node_modules", "data"]:
                    cmd.append(f"--exclude-dir={excl}")
                cmd.extend([pattern, str(search_dir)])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode <= 1:  # 0=found, 1=not found
                    for line in result.stdout.splitlines():
                        if len(all_matches) >= self.max_grep_results * 3:
                            break
                        all_matches.append(line)
            except subprocess.TimeoutExpired:
                logger.warning(f"[FileAccess] Grep timed out in {search_dir}")
            except Exception as e:
                logger.warning(f"[FileAccess] Grep error in {search_dir}: {e}")

        # Parse and cap results
        total = len(all_matches)
        truncated = total > self.max_grep_results * 3
        capped = all_matches[: self.max_grep_results * 3]

        return {
            "matches": capped,
            "total_matches": total,
            "truncated": truncated,
            "success": True,
        }

    async def list_directory(
        self,
        dirpath: str,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        List files in an approved directory.

        Returns dict with: entries, total, truncated, success, error
        """
        try:
            resolved = self._validate_path(dirpath)
        except PermissionError as e:
            return {"error": str(e), "success": False}

        if not resolved.is_dir():
            return {"error": f"Not a directory: {dirpath}", "success": False}

        try:
            entries = []
            iterator = resolved.rglob("*") if recursive else resolved.iterdir()
            for p in sorted(iterator):
                # Skip hidden and common noise
                rel = p.relative_to(resolved)
                parts = rel.parts
                if any(
                    part.startswith(".") or part in ("__pycache__", "venv", ".venv", "node_modules")
                    for part in parts
                ):
                    continue

                kind = "dir" if p.is_dir() else "file"
                size = p.stat().st_size if p.is_file() else 0
                entries.append({
                    "path": str(rel),
                    "type": kind,
                    "size": size,
                })
                if len(entries) >= self.max_list_entries:
                    break

            total = len(entries)
            return {
                "entries": entries,
                "total": total,
                "truncated": total >= self.max_list_entries,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"[FileAccess] Error listing {dirpath}: {e}")
            return {"error": f"List error: {e}", "success": False}

    def format_read_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format a read result for LLM context."""
        if not result.get("success"):
            return f"[File read error: {result.get('error', 'unknown')}]"

        header = f"File: {result['path']} ({result['lines']} lines)"
        if result.get("truncated"):
            header += " [TRUNCATED]"
        return f"{header}\n\n{result['content']}"

    def format_grep_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format grep results for LLM context."""
        if not result.get("success"):
            return f"[Grep error: {result.get('error', 'unknown')}]"

        matches = result.get("matches", [])
        if not matches:
            return "[No matches found]"

        output = "\n".join(matches)
        if result.get("truncated"):
            output += f"\n\n[Showing partial results — {result['total_matches']} total matches]"
        return output

    def format_list_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format directory listing for LLM context."""
        if not result.get("success"):
            return f"[List error: {result.get('error', 'unknown')}]"

        entries = result.get("entries", [])
        if not entries:
            return "[Empty directory]"

        lines = []
        for e in entries:
            if e["type"] == "dir":
                lines.append(f"  {e['path']}/")
            else:
                size_kb = e["size"] / 1024
                lines.append(f"  {e['path']}  ({size_kb:.1f}KB)")

        output = "\n".join(lines)
        if result.get("truncated"):
            output += f"\n\n[Listing capped at {len(entries)} entries]"
        return output
