"""DM-29 — the closure-by-phrase-append signature in the changelog (BC-76).

BC-76 is a PROCESS class: a vocabulary miss closed by appending one more
phrase to the same regex/list, instead of adopting one of this project's own
generalized remedies (chokepoint module, categorized-generic vocabulary with
anchors, seeds+learned channel).  The signature is not any single entry — it
is the SAME target widened in three or more dated batches.

The changelog's dated batches are its ``## 20yy-mm-dd …`` headings.  Within a
batch, a target is collected from the five widening shapes the catalog names
(``X gained``, ``added … to X``, ``X now also``, ``extended X``, ``X +=``); a
target appearing in three or more DISTINCT dated batches is one finding.

Heuristic by construction, so this scanner is report-only: it flags a family
for a human to judge against
``docs/GENERALIZATION_AUDIT_20260901.md`` §"Remedy patterns", which is the
closure BC-76 actually calls for.  It never fails a push on its own.

Its only input, ``CLAUDE_CHANGELOG.md``, is gitignored and local-only.  An
absent changelog is an UNAVAILABLE optional leg — never "0 files, clean" —
and cannot be cited as evidence that BC-76 was scanned (contract v2).
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import Finding, Leg, ScanResult, clip, digest_of, resolve_leg

SCANNER_ID = "dm29_phrase_append_signature"
CLASS_IDS = ("BC-76",)
CONTRACT_VERSION = 2

CHANGELOG = "CLAUDE_CHANGELOG.md"
LEG = Leg("dm29_changelog", "file", (CHANGELOG,), False)
LEGS = (LEG,)
KIND = "phrase_append_signature"
KINDS = (KIND,)
MIN_BATCHES = 3

_BATCH_HEADING_RE = re.compile(r"^##\s+(?P<date>20[0-9]{2}-[0-9]{2}-[0-9]{2})")

# A widened target: a CONSTANT/regex/list name, or a function reference.
_TARGET = r"(?P<target>[A-Z_]{3,}[A-Z0-9_]*|[a-z_][a-z0-9_]+\(\))"
_Q = r"[`'\"]?"

_SHAPE_RES = (
    re.compile(rf"{_Q}{_TARGET}{_Q}\s+gained\b"),
    re.compile(rf"\badded\b[^.]{{0,120}}?\bto\s+{_Q}{_TARGET}{_Q}"),
    re.compile(rf"{_Q}{_TARGET}{_Q}\s+now\s+also\b"),
    re.compile(rf"\bextended\s+{_Q}{_TARGET}{_Q}"),
    re.compile(rf"{_Q}{_TARGET}{_Q}\s*\+="),
)


def scan(root: Path) -> ScanResult:
    resolved = resolve_leg(root, LEG)
    if resolved.status != "available":
        return ScanResult([], (resolved.receipt(),))
    text = resolved.files[0].read_bytes().decode("utf-8", errors="replace")

    batch = ""
    # target -> {batch date: (last line number, last matching line text)}
    hits: dict[str, dict[str, tuple[int, str]]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        heading = _BATCH_HEADING_RE.match(line)
        if heading:
            batch = heading.group("date")
        if not batch:
            continue
        for shape in _SHAPE_RES:
            for match in shape.finditer(line):
                target = match.group("target")
                hits.setdefault(target, {})[batch] = (lineno, line)

    findings: list[Finding] = []
    for target, batches in hits.items():
        if len(batches) < MIN_BATCHES:
            continue
        lineno, _line = batches[max(batches)]
        findings.append(
            Finding(
                SCANNER_ID,
                CLASS_IDS,
                CHANGELOG,
                target,
                lineno,
                KIND,
                digest_of([target]),
                clip(f"{target} widened in {len(batches)} dated batches"),
                LEG.id,
            )
        )
    findings.sort(key=lambda f: (f.symbol, f.line))
    return ScanResult(findings, (resolved.receipt(),))
