"""Catalog admission gate (BC-71) — the B1 parser wired into the ratchet.

BC-71 is documentation and tool self-description drift: prose asserting a
state the code no longer has.  ``bug_class_guards.catalog`` already decides
whether ``docs/BUG_CLASSES.md`` is internally consistent (index vs body IDs,
index vs body status, required fields, method references that resolve); this
scanner turns each of its diagnostics into one baselined finding so a NEW
inconsistency fails the push while the known ones stay visible.

A ``CatalogParseError`` is a scanner ERROR, not a finding: an unparseable
catalog means the gate cannot answer the question, and "cannot check" must
never read like "nothing wrong" (BC-47).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..catalog import CatalogParseError, catalog_diagnostics, parse_catalog
from .common import Finding, ScanResult, ScannerError, clip

SCANNER_ID = "catalog"
CLASS_IDS = ("BC-71",)

CATALOG_PATH = "docs/BUG_CLASSES.md"
_ID_RE = re.compile(r"\b(BC-[0-9]+)\b")


def scan(root: Path) -> ScanResult:
    path = root / CATALOG_PATH
    if not path.is_file():
        raise ScannerError(f"catalog missing: {CATALOG_PATH}")
    try:
        catalog = parse_catalog(path)
    except CatalogParseError as exc:
        raise ScannerError(f"{CATALOG_PATH}: {exc}") from exc
    findings: list[Finding] = []
    for message in catalog_diagnostics(catalog):
        match = _ID_RE.search(message)
        symbol = match.group(1) if match else ""
        line = catalog.entries[symbol].line if symbol in catalog.entries else 1
        findings.append(
            Finding(SCANNER_ID, CLASS_IDS, CATALOG_PATH, symbol, line, clip(message))
        )
    findings.sort(key=lambda f: (f.symbol, f.text))
    return ScanResult(findings, 1)
