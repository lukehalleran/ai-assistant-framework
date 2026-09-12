"""Shared primitives for the repo-static bug-class scanners.

Standard library ONLY.  These run from ``hooks/pre-push`` and from CI before
any application import; pulling an app module (and with it torch, chromadb or
PyYAML) would make the gate slower than the suite it guards and couple a
static check to runtime state.

A finding is anchored by CONTENT, never by line number — the anchoring
lesson from ``tests/unit/test_ordered_slice_guard.py`` (push a690a91 went red
when unrelated hunks shifted allowlisted slices).  The anchor is
``(scanner_id, relpath, enclosing qualname, stripped source line)``: pure
drift keeps an entry valid, while editing the flagged line or moving it into
another function surfaces the entry as STALE for re-review.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

# Excluded at any depth of any walk.
EXCLUDED_DIR_NAMES = frozenset(
    {"venv", "data", "web", "node_modules", "__pycache__", "integration.bak", ".git"}
)

# Python source roots (directories and single files) relative to the repo root.
SOURCE_ROOTS: tuple[str, ...] = (
    "core",
    "memory",
    "knowledge",
    "utils",
    "gui",
    "api",
    "models",
    "processing",
    "config",
    "scripts",
    "main.py",
)

# Findings carry a source line; cap it so a baseline entry stays reviewable.
MAX_TEXT = 240


class ScannerError(RuntimeError):
    """A scanner could not complete (unreadable or unparseable input)."""


@dataclass(frozen=True)
class Finding:
    scanner_id: str
    class_ids: tuple[str, ...]
    path: str
    symbol: str
    line: int
    text: str

    def fingerprint(self) -> tuple[str, str, str, str]:
        """Content anchor: (scanner, relpath, enclosing qualname, line text)."""
        return (self.scanner_id, self.path, self.symbol, self.text)


@dataclass(frozen=True)
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    files_processed: int = 0


@dataclass(frozen=True)
class Scanner:
    id: str
    class_ids: tuple[str, ...]
    description: str
    mode: str  # "gate" (affects exit code) | "report" (printed only)
    run: Callable[[Path], ScanResult]

    def scan(self, root: Path) -> ScanResult:
        return self.run(root)


def clip(text: str) -> str:
    """Deterministic, reviewable one-line rendering of a source line."""
    stripped = " ".join(text.split())
    if len(stripped) > MAX_TEXT:
        return stripped[: MAX_TEXT - 1] + "…"
    return stripped


def relpath(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _walk_dir(base: Path) -> Iterator[Path]:
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError as exc:  # pragma: no cover - surfaced as a scanner error
        raise ScannerError(f"cannot list {base}: {exc}") from exc
    for entry in entries:
        if entry.name in EXCLUDED_DIR_NAMES or entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk_dir(entry)
        elif entry.suffix == ".py":
            yield entry


def iter_python_files(root: Path, roots: Sequence[str] = SOURCE_ROOTS) -> list[Path]:
    """Every ``*.py`` under the given roots, sorted by POSIX relative path."""
    found: list[Path] = []
    for name in roots:
        base = root / name
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix == ".py":
                found.append(base)
        else:
            found.extend(_walk_dir(base))
    return sorted(set(found), key=lambda p: relpath(root, p))


def read_source(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScannerError(f"cannot read {path}: {exc}") from exc


def parse_module(path: Path, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ScannerError(f"cannot parse {path}: {exc}") from exc


def function_spans(tree: ast.AST) -> list[tuple[int, int, str]]:
    """(start, end, dotted qualname) for every def, qualified through classes."""
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                spans.append((child.lineno, child.end_lineno or child.lineno, name))
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return spans


def scope_for(spans: Sequence[tuple[int, int, str]], lineno: int) -> str:
    """Innermost def containing ``lineno``; "<module>" at module level."""
    innermost: tuple[int, int, str] | None = None
    for start, end, name in spans:
        if start <= lineno <= end and (
            innermost is None or (end - start) < (innermost[1] - innermost[0])
        ):
            innermost = (start, end, name)
    return innermost[2] if innermost else "<module>"


def source_line(lines: Sequence[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return clip(lines[lineno - 1])
    return ""
