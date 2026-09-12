"""DM-16 — YAML leaves with no reader outside ``config/`` (BC-12, BC-10).

BC-12 is a config key validated by the schema but never threaded to its
runtime reader: ``google_calendar_lookahead_days`` was never passed,
``features.rewrite_timeout_s: 0`` never reached ``ContextPipeline``, and
``PROMPT_TOKEN_BUDGET_DEFAULT`` was dead for API models for months.  Every one
looked configured and did nothing.

A leaf ``section.key`` counts as REACHED when either
  * its key string literal appears in ``config/app_config.py`` or
    ``config/schema.py`` AND the constant assigned on that ``app_config`` line
    is referenced from a module outside ``config/``; or
  * the key string appears directly outside ``config/``.

The YAML is read with a small indentation reader rather than PyYAML: this lane
is stdlib-only.  The reader tracks quoted/blocked multi-line scalars so their
continuation lines cannot be mistaken for keys, and skips list items.

REPORT-ONLY: the live count is well above the threshold at which a gate would
be honest (CM-04 — shadow mode until precision is measured), and an unreached
constant is a candidate for a human to judge, not an automatic failure.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import (
    Finding,
    ScanResult,
    clip,
    iter_python_files,
    read_source,
)

SCANNER_ID = "dm16_config_key_reachability"
CLASS_IDS = ("BC-12", "BC-10")

CONFIG_YAML = "config/config.yaml"
APP_CONFIG = "config/app_config.py"
SCHEMA = "config/schema.py"
CONSUMER_ROOTS = (
    "core",
    "memory",
    "knowledge",
    "utils",
    "gui",
    "api",
    "models",
    "processing",
    "scripts",
    "main.py",
)

_KEY_RE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*:(?P<rest>.*)$")
_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=")
_BLOCK_SCALAR = {"|", ">", "|-", ">-", "|+", ">+"}


def _opens_multiline_scalar(rest: str) -> bool:
    if rest in _BLOCK_SCALAR:
        return True
    for quote in ("'", '"'):
        if rest.startswith(quote) and not (len(rest) > 1 and rest.endswith(quote)):
            return True
    return False


def read_leaves(text: str) -> list[tuple[str, str, int, str]]:
    """(dotted path, leaf key, line number, source line) for every scalar leaf."""
    stack: list[tuple[int, str]] = []
    leaves: list[tuple[str, str, int, str]] = []
    in_scalar = False
    scalar_indent = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if in_scalar:
            if indent > scalar_indent:
                continue
            in_scalar = False
        if raw.lstrip().startswith("- "):
            continue
        match = _KEY_RE.match(raw)
        if not match:
            continue
        key = match.group("key")
        rest = match.group("rest").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if rest == "" or rest.startswith("#"):
            stack.append((indent, key))
            continue
        if _opens_multiline_scalar(rest):
            in_scalar = True
            scalar_indent = indent
        dotted = ".".join([name for _, name in stack] + [key])
        leaves.append((dotted, key, lineno, clip(raw)))
    return leaves


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_RE = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_.\-]*)['\"]")


def _constants_for(key: str, app_lines: list[str]) -> list[str]:
    quoted = (f'"{key}"', f"'{key}'")
    names = []
    for line in app_lines:
        if any(q in line for q in quoted):
            assigned = _ASSIGN_RE.match(line)
            if assigned:
                names.append(assigned.group("name"))
    return names


def scan(root: Path) -> ScanResult:
    yaml_path = root / CONFIG_YAML
    app_path = root / APP_CONFIG
    if not yaml_path.is_file() or not app_path.is_file():
        return ScanResult([], 0)
    leaves = read_leaves(read_source(yaml_path))
    app_lines = read_source(app_path).splitlines()
    schema_path = root / SCHEMA
    schema_src = read_source(schema_path) if schema_path.is_file() else ""

    consumers = iter_python_files(root, CONSUMER_ROOTS)
    # Index the consumer corpus ONCE: a per-key regex sweep over ~10 MB of
    # source cost 17 s of an 19 s gate run, which is how a static check stops
    # being run at all.
    consumer_identifiers: set[str] = set()
    consumer_strings: set[str] = set()
    for path in consumers:
        text = read_source(path)
        consumer_identifiers.update(_IDENT_RE.findall(text))
        consumer_strings.update(_QUOTED_RE.findall(text))

    app_src = "\n".join(app_lines)
    findings: list[Finding] = []
    for dotted, key, lineno, line in leaves:
        if key in consumer_strings:
            continue
        quoted = (f'"{key}"', f"'{key}'")
        declared = any(q in app_src or q in schema_src for q in quoted)
        if declared and any(
            name in consumer_identifiers for name in _constants_for(key, app_lines)
        ):
            continue
        findings.append(
            Finding(SCANNER_ID, CLASS_IDS, CONFIG_YAML, dotted, lineno, line)
        )
    findings.sort(key=lambda f: (f.line, f.symbol))
    return ScanResult(findings, len(consumers) + 2 + (1 if schema_src else 0))
