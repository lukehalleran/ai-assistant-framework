"""Shared primitives for the repo-static bug-class scanners.

Standard library ONLY.  These run from ``hooks/pre-push`` and from CI before
any application import; pulling an app module (and with it torch, chromadb or
PyYAML) would make the gate slower than the suite it guards and couple a
static check to runtime state.

Inputs are declared LEGS (contract v2, 2026-09-13).  A leg names its roots and
whether it is required; ``resolve_leg`` reports every root on its own, so a
nonzero total can never hide a root that was absent or empty.  The same leg
definitions appear in ``config/bug_class_policy.json`` and are compared to
the registry before any scanner runs.

Anchors (contract v2).  A finding's identity is
``(scanner_id, relpath, enclosing qualname, candidate kind, digest)`` where
``digest`` is the SHA-256 of a canonical rendering of the candidate's AST
(positions, contexts and comments excluded).  Each scanner keeps its legacy
occurrence granularity — one candidate per physical line for DM-01, DM-17's
test literals and DM-18; one per parameter for DM-31; one per file for DM-17
scripts — so baseline multiplicities migrate one-for-one.  Line numbers are
never identity; ``excerpt`` is a bounded rendering for humans only.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

ANCHOR_VERSION = "bug-class-anchor/1"

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

LEG_KINDS = ("python_tree", "python_flat", "file")

# Findings carry a source line; cap it so a baseline entry stays reviewable.
MAX_TEXT = 240


class ScannerError(RuntimeError):
    """A scanner could not complete (unreadable or unparseable input)."""


# --------------------------------------------------------------------- legs


@dataclass(frozen=True)
class Exemption:
    path: str
    reason: str


@dataclass(frozen=True)
class Leg:
    """One declared input: ``python_tree`` walks each root recursively,
    ``python_flat`` takes ``<root>/*.py`` only, ``file`` is one exact file."""

    id: str
    kind: str
    roots: tuple[str, ...]
    required: bool
    exempt: tuple[Exemption, ...] = ()

    @property
    def is_python(self) -> bool:
        if self.kind == "file":
            return all(root.endswith(".py") for root in self.roots)
        return True

    @property
    def exempt_paths(self) -> frozenset[str]:
        return frozenset(item.path for item in self.exempt)


PYTHON_SOURCE_LEG = Leg("python_source", "python_tree", SOURCE_ROOTS, True)


@dataclass(frozen=True)
class RootReceipt:
    path: str
    status: str  # available | missing | empty
    files: int

    def to_json(self) -> dict:
        return {"path": self.path, "status": self.status, "files": self.files}


@dataclass(frozen=True)
class LegReceipt:
    id: str
    required: bool
    status: str  # available | missing | empty | unavailable | not_selected
    files_processed: int
    unresolved: int
    roots: tuple[RootReceipt, ...]

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "required": self.required,
            "status": self.status,
            "available": self.available,
            "files_processed": self.files_processed,
            "unresolved": self.unresolved,
            "roots": [root.to_json() for root in self.roots],
        }


@dataclass(frozen=True)
class ResolvedLeg:
    leg: Leg
    files: tuple[Path, ...]
    roots: tuple[RootReceipt, ...]

    @property
    def status(self) -> str:
        if any(root.status == "missing" for root in self.roots):
            return "missing" if self.leg.required else "unavailable"
        if any(root.status == "empty" for root in self.roots):
            return "empty" if self.leg.required else "unavailable"
        return "available"

    def receipt(self, unresolved: int = 0) -> LegReceipt:
        return LegReceipt(
            self.leg.id, self.leg.required, self.status, len(self.files), unresolved, self.roots
        )


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
        elif entry.suffix == ".py" and entry.is_file():
            yield entry


def _flat_python(base: Path) -> list[Path]:
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError as exc:  # pragma: no cover - surfaced as a scanner error
        raise ScannerError(f"cannot list {base}: {exc}") from exc
    return [
        entry
        for entry in entries
        if entry.suffix == ".py" and entry.is_file() and not entry.is_symlink()
    ]


def resolve_leg(root: Path, leg: Leg) -> ResolvedLeg:
    """Every file a leg owns, with one receipt per declared root."""
    files: set[Path] = set()
    receipts: list[RootReceipt] = []
    for name in leg.roots:
        base = root / name
        if leg.kind == "file":
            found = [base] if base.is_file() else []
            status = "available" if found else "missing"
        elif not base.exists():
            found, status = [], "missing"
        elif base.is_file():
            found = [base] if leg.kind == "python_tree" and base.suffix == ".py" else []
            status = "available" if found else "empty"
        else:
            found = _flat_python(base) if leg.kind == "python_flat" else list(_walk_dir(base))
            status = "available" if found else "empty"
        receipts.append(RootReceipt(name, status, len(found)))
        files.update(found)
    ordered = tuple(sorted(files, key=lambda p: relpath(root, p)))
    return ResolvedLeg(leg, ordered, tuple(receipts))


def iter_python_files(root: Path, roots: Sequence[str] = SOURCE_ROOTS) -> list[Path]:
    """Every ``*.py`` under the given roots, sorted by POSIX relative path."""
    return list(resolve_leg(root, Leg("adhoc", "python_tree", tuple(roots), False)).files)


# ----------------------------------------------------------------- findings


@dataclass(frozen=True)
class Finding:
    scanner_id: str
    class_ids: tuple[str, ...]
    path: str
    symbol: str
    line: int
    kind: str
    digest: str
    excerpt: str
    leg: str = ""
    unresolved: bool = False
    # Contract v1 anchor parts, kept only so the reviewed baseline migration
    # can map every legacy occurrence to exactly one v2 anchor.
    legacy_symbol: str | None = None
    legacy_text: str | None = None
    # First and last source line of the candidate expression (review only).
    span: tuple[int, int] = (0, 0)

    def fingerprint(self) -> tuple[str, str, str, str, str]:
        """Content anchor: (scanner, relpath, qualname, kind, digest)."""
        return (self.scanner_id, self.path, self.symbol, self.kind, self.digest)

    def legacy_fingerprint(self) -> tuple[str, str, str, str]:
        """Contract v1 anchor: (scanner, relpath, qualname, clipped line)."""
        symbol = self.symbol if self.legacy_symbol is None else self.legacy_symbol
        text = self.excerpt if self.legacy_text is None else self.legacy_text
        return (self.scanner_id, self.path, symbol, text)

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "digest": self.digest,
            "line": self.line,
            "excerpt": self.excerpt,
            "span": list(self.span) if self.span != (0, 0) else [self.line, self.line],
            "leg": self.leg,
            "unresolved": self.unresolved,
        }


@dataclass(frozen=True)
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    legs: tuple[LegReceipt, ...] = ()

    @property
    def files_processed(self) -> int:
        return sum(leg.files_processed for leg in self.legs)

    @property
    def unresolved(self) -> int:
        return sum(1 for finding in self.findings if finding.unresolved)


@dataclass(frozen=True)
class Scanner:
    id: str
    class_ids: tuple[str, ...]
    description: str
    mode: str  # "gate" (affects exit code) | "report" (printed only)
    contract_version: int
    legs: tuple[Leg, ...]
    kinds: tuple[str, ...]
    run: Callable[[Path], ScanResult]

    def scan(self, root: Path) -> ScanResult:
        return self.run(root)


# ------------------------------------------------------------------ anchors


def canonical(node: object) -> str:
    """Deterministic rendering of an AST value, independent of positions.

    Contexts, type comments and the ``u`` string-kind marker are dropped, as
    are None and empty-list fields, so layout, comments and interpreter
    versions that add empty fields do not change the rendering.
    """
    if isinstance(node, ast.AST):
        parts = []
        for name in node._fields:
            if name in {"ctx", "type_comment", "kind"}:
                continue
            value = getattr(node, name, None)
            if value is None or value == []:
                continue
            parts.append(f"{name}={canonical(value)}")
        return f"{type(node).__name__}({', '.join(parts)})"
    if isinstance(node, list):
        return "[" + ", ".join(canonical(item) for item in node) + "]"
    return repr(node)


def digest_of(parts: Sequence[str]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ helpers


def clip(text: str) -> str:
    """Deterministic, reviewable one-line rendering of a source line."""
    stripped = " ".join(text.split())
    if len(stripped) > MAX_TEXT:
        return stripped[: MAX_TEXT - 1] + "…"
    return stripped


def relpath(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


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


def docstring_ids(tree: ast.AST) -> set[int]:
    """``id()`` of every module/class/function docstring Constant."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def line_group_findings(
    scanner_id: str,
    class_ids: tuple[str, ...],
    rel: str,
    tree: ast.AST,
    lines: Sequence[str],
    groups: dict[int, list[tuple[int, str, str, int, int]]],
    leg: str,
) -> list[Finding]:
    """One finding per physical line from
    ``{line: [(col, kind, canonical, first line, last line)]}``."""
    spans = function_spans(tree)
    findings = []
    for lineno in sorted(groups):
        members = sorted(groups[lineno])
        kinds = sorted({member[1] for member in members})
        findings.append(
            Finding(
                scanner_id,
                class_ids,
                rel,
                scope_for(spans, lineno),
                lineno,
                "+".join(kinds),
                digest_of([member[2] for member in members]),
                source_line(lines, lineno),
                leg,
                span=(min(m[3] for m in members), max(m[4] for m in members)),
            )
        )
    return findings
