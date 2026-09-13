"""Repo-static bug-class scanners (``docs/BUG_CLASSES.md`` §Detection methods).

Each scanner mechanises one DM row: a deterministic, stdlib-only pass over its
declared input legs that lists CANDIDATES for one or more bug classes.
``scripts/check_bug_classes.py scan`` compares this registry to
``config/bug_class_policy.json`` before running anything, then ratchets the
gate scanners against a content-anchored baseline
(``config/bug_class_baseline.json``) and its reviewed dispositions: a NEW
finding fails, a STALE baseline occurrence fails, and an occurrence without a
current review fails.

A scanner in ``gate`` mode affects the exit code; a ``report`` scanner prints
and is excluded from both the baseline and the exit code (CM-04: shadow mode
until precision is measured).  The set is a scoped structural lane, not
coverage of every catalog class.
"""

from __future__ import annotations

from typing import Iterable

from .common import Finding, Leg, LegReceipt, Scanner, ScannerError, ScanResult
from . import catalog_scanner, dm01_raw_substring, dm16_config_reachability
from . import dm17_apply_without_guard, dm18_except_returns_empty, dm29_phrase_append
from . import dm31_live_state_default


def _entry(module, description: str, mode: str) -> Scanner:
    return Scanner(
        id=module.SCANNER_ID,
        class_ids=module.CLASS_IDS,
        description=description,
        mode=mode,
        contract_version=module.CONTRACT_VERSION,
        legs=module.LEGS,
        kinds=module.KINDS,
        run=module.scan,
    )


_SCANNER_LIST = (
    _entry(
        dm01_raw_substring,
        "raw-substring keyword tests against lowered text",
        "gate",
    ),
    # 87 live candidates: above the count at which a gate would be honest,
    # and each one needs a human to say "dead" vs "read another way".
    _entry(
        dm16_config_reachability,
        "config.yaml leaves with no reader outside config/",
        "report",
    ),
    _entry(
        dm17_apply_without_guard,
        "--apply scripts without executed daemon-guard evidence; data/ paths in tests",
        "gate",
    ),
    _entry(
        dm18_except_returns_empty,
        "broad except returning an empty result beside a store call",
        "gate",
    ),
    # BC-76's own Closure line says the judgment is human, not automatable.
    _entry(
        dm29_phrase_append,
        "changelog phrase-append signature (>=3 dated batches)",
        "report",
    ),
    # Gated on arrival: the whole tree yields six candidates, hand-reviewed
    # (two were the live 2026-09-11 defect and are fixed; the others are
    # caller-chosen sizes or component-local flags and sit in the baseline).
    _entry(
        dm31_live_state_default,
        "public function asserts live budget/toggle state via a default",
        "gate",
    ),
    _entry(
        catalog_scanner,
        "BUG_CLASSES.md index/body/method admission diagnostics (structure only)",
        "gate",
    ),
)


def build_registry(scanners: Iterable[Scanner]) -> dict[str, Scanner]:
    """Registry keyed by id; a duplicate id is an error, never a silent overwrite."""
    registry: dict[str, Scanner] = {}
    for scanner in scanners:
        if scanner.id in registry:
            raise ValueError(f"duplicate scanner id: {scanner.id}")
        registry[scanner.id] = scanner
    return registry


SCANNERS: dict[str, Scanner] = build_registry(_SCANNER_LIST)

__all__ = [
    "Finding",
    "Leg",
    "LegReceipt",
    "SCANNERS",
    "ScanResult",
    "Scanner",
    "ScannerError",
    "build_registry",
]
