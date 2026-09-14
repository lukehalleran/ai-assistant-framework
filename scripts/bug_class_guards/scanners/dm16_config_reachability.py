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
Both are lexical approximations (an identifier or string token, not a traced
read), which is one reason this scanner is report-only.

The YAML is read with a small indentation reader rather than PyYAML: this lane
is stdlib-only.  The reader tracks quoted/blocked multi-line scalars so their
continuation lines cannot be mistaken for keys.  Contract v2 (2026-09-13):
anything the reader does not understand — anchors, aliases, tags, merge keys,
non-empty flow collections, quoted or complex keys, document markers, and
mappings inside list items — is surfaced as a ``config_yaml_unresolved``
candidate instead of being silently skipped.  The YAML file, ``app_config.py``
and every consumer root are required inputs; ``schema.py`` is optional
evidence.

REPORT-ONLY: the live count is well above the threshold at which a gate would
be honest (CM-04 — shadow mode until precision is measured), and an unreached
constant is a candidate for a human to judge, not an automatic failure.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import (
    Finding,
    Leg,
    ScanResult,
    clip,
    digest_of,
    read_source,
    resolve_leg,
)

SCANNER_ID = "dm16_config_key_reachability"
CLASS_IDS = ("BC-12", "BC-10")
CONTRACT_VERSION = 2

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

YAML_LEG = Leg("dm16_config_yaml", "file", (CONFIG_YAML,), True)
APP_CONFIG_LEG = Leg("dm16_app_config", "file", (APP_CONFIG,), True)
SCHEMA_LEG = Leg("dm16_schema", "file", (SCHEMA,), False)
CONSUMER_LEG = Leg("dm16_consumers", "python_tree", CONSUMER_ROOTS, True)
LEGS = (YAML_LEG, APP_CONFIG_LEG, SCHEMA_LEG, CONSUMER_LEG)

KIND_UNREACHED = "config_leaf_unreached"
KIND_UNRESOLVED = "config_yaml_unresolved"
KINDS = (KIND_UNREACHED, KIND_UNRESOLVED)

_KEY_RE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*:(?P<rest>.*)$")
_LIST_MAPPING_RE = re.compile(r"^\s*-\s+[A-Za-z_\"'][^:]*:(\s|$)")
_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=")
_BLOCK_SCALAR = {"|", ">", "|-", ">-", "|+", ">+"}
_EMPTY_FLOW = {"{}", "[]"}


def _opens_multiline_scalar(rest: str) -> bool:
    if rest in _BLOCK_SCALAR:
        return True
    for quote in ("'", '"'):
        if rest.startswith(quote) and not (len(rest) > 1 and rest.endswith(quote)):
            return True
    return False


def _value_unsupported(rest: str) -> bool:
    value = rest.split(" #", 1)[0].strip()
    if value in _EMPTY_FLOW:
        return False
    return value[:1] in {"&", "*", "!", "[", "{"}


def read_yaml(text: str):
    """(leaves, unresolved) where both are (dotted, key, line number, source line)."""
    stack: list[tuple[int, str]] = []
    leaves: list[tuple[str, str, int, str]] = []
    unresolved: list[tuple[str, str, int, str]] = []
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
        stripped = raw.lstrip()
        if stripped.startswith("- ") or stripped == "-":
            if _LIST_MAPPING_RE.match(raw):
                dotted = ".".join(name for _, name in stack) or "<list>"
                unresolved.append((dotted, "", lineno, clip(raw)))
            continue
        match = _KEY_RE.match(raw)
        if not match:
            unresolved.append((".".join(name for _, name in stack) or "<yaml>", "", lineno, clip(raw)))
            continue
        key = match.group("key")
        rest = match.group("rest").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([name for _, name in stack] + [key])
        if rest == "" or rest.startswith("#"):
            stack.append((indent, key))
            continue
        if _value_unsupported(rest):
            unresolved.append((dotted, key, lineno, clip(raw)))
            if rest.split(" #", 1)[0].strip().startswith("&") and len(rest.split()) == 1:
                stack.append((indent, key))  # anchored mapping: keep reading its keys
            continue
        if _opens_multiline_scalar(rest):
            in_scalar = True
            scalar_indent = indent
        leaves.append((dotted, key, lineno, clip(raw)))
    return leaves, unresolved


def read_leaves(text: str) -> list[tuple[str, str, int, str]]:
    """(dotted path, leaf key, line number, source line) for every scalar leaf."""
    return read_yaml(text)[0]


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
    yaml_leg = resolve_leg(root, YAML_LEG)
    app_leg = resolve_leg(root, APP_CONFIG_LEG)
    schema_leg = resolve_leg(root, SCHEMA_LEG)
    consumer_leg = resolve_leg(root, CONSUMER_LEG)
    if yaml_leg.status != "available" or app_leg.status != "available":
        return ScanResult(
            [],
            (yaml_leg.receipt(), app_leg.receipt(), schema_leg.receipt(), consumer_leg.receipt()),
        )
    leaves, unresolved_lines = read_yaml(read_source(yaml_leg.files[0]))
    app_lines = read_source(app_leg.files[0]).splitlines()
    schema_src = read_source(schema_leg.files[0]) if schema_leg.files else ""

    # Index the consumer corpus ONCE: a per-key regex sweep over ~10 MB of
    # source cost 17 s of an 19 s gate run, which is how a static check stops
    # being run at all.
    consumer_identifiers: set[str] = set()
    consumer_strings: set[str] = set()
    for path in consumer_leg.files:
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
            Finding(
                SCANNER_ID, CLASS_IDS, CONFIG_YAML, dotted, lineno, KIND_UNREACHED,
                digest_of([dotted]), line, YAML_LEG.id,
            )
        )
    for dotted, _key, lineno, line in unresolved_lines:
        findings.append(
            Finding(
                SCANNER_ID, CLASS_IDS, CONFIG_YAML, dotted, lineno, KIND_UNRESOLVED,
                digest_of([dotted, line]), line, YAML_LEG.id, unresolved=True,
            )
        )
    findings.sort(key=lambda f: (f.line, f.symbol))
    receipts = (
        yaml_leg.receipt(len(unresolved_lines)),
        app_leg.receipt(),
        schema_leg.receipt(),
        consumer_leg.receipt(),
    )
    return ScanResult(findings, receipts)
