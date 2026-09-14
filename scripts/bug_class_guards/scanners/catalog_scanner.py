"""Catalog admission gate (BC-71) — the B1 parser wired into the ratchet.

BC-71 is documentation and tool self-description drift: prose asserting a
state the code no longer has.  ``bug_class_guards.catalog`` already decides
whether ``docs/BUG_CLASSES.md`` is internally consistent (index vs body IDs,
index vs body status, required fields, method references that resolve); this
scanner turns each of its diagnostics into one baselined finding so a NEW
inconsistency fails the push while the known ones stay visible.

It checks catalog STRUCTURE only.  It does not detect any product behavior
described by the 78 classes, and it never counts as coverage for them.

A missing catalog is a missing required input leg (the CLI fails the
coverage contract).  A ``CatalogParseError`` is a scanner ERROR, not a
finding: an unparseable catalog means the gate cannot answer the question,
and "cannot check" must never read like "nothing wrong" (BC-47).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..catalog import CatalogParseError, catalog_diagnostics, parse_catalog
from .common import Finding, Leg, ScanResult, ScannerError, clip, digest_of, resolve_leg

SCANNER_ID = "catalog"
CLASS_IDS = ("BC-71",)
CONTRACT_VERSION = 2

CATALOG_PATH = "docs/BUG_CLASSES.md"
LEG = Leg("catalog_document", "file", (CATALOG_PATH,), True)
LEGS = (LEG,)
KIND = "catalog_diagnostic"
KINDS = (KIND,)
CATALOG_SYMBOL = "<catalog>"

_ID_RE = re.compile(r"\b(BC-[0-9]+)\b")


def scan(root: Path) -> ScanResult:
    resolved = resolve_leg(root, LEG)
    if resolved.status != "available":
        return ScanResult([], (resolved.receipt(),))
    try:
        catalog = parse_catalog(resolved.files[0])
    except CatalogParseError as exc:
        raise ScannerError(f"{CATALOG_PATH}: {exc}") from exc
    findings: list[Finding] = []
    for message in catalog_diagnostics(catalog):
        match = _ID_RE.search(message)
        class_id = match.group(1) if match else ""
        line = catalog.entries[class_id].line if class_id in catalog.entries else 1
        findings.append(
            Finding(
                SCANNER_ID,
                CLASS_IDS,
                CATALOG_PATH,
                class_id or CATALOG_SYMBOL,
                line,
                KIND,
                digest_of([message]),
                clip(message),
                LEG.id,
                legacy_symbol=class_id,
            )
        )
    findings.sort(key=lambda f: (f.symbol, f.excerpt))
    return ScanResult(findings, (resolved.receipt(),))
