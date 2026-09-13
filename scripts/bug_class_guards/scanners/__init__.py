"""Repo-static bug-class scanners (``docs/BUG_CLASSES.md`` §Detection methods).

Each scanner mechanises one DM row: a deterministic, stdlib-only pass over the
tree that lists CANDIDATES for one bug class.  ``scripts/check_bug_classes.py
scan`` runs them against a content-anchored baseline
(``config/bug_class_baseline.json``) and ratchets: a NEW finding fails, and a
STALE baseline entry fails too, so fixed debt must be REMOVED from the
baseline rather than laundered by leaving the entry behind.

A scanner in ``gate`` mode affects the exit code; a ``report`` scanner prints
and is excluded from both the baseline and the exit code (CM-04: shadow mode
until precision is measured).
"""

from __future__ import annotations

from .common import Finding, Scanner, ScannerError, ScanResult
from . import catalog_scanner, dm01_raw_substring, dm16_config_reachability
from . import dm17_apply_without_guard, dm18_except_returns_empty, dm29_phrase_append
from . import dm31_live_state_default

_SCANNER_LIST = (
    Scanner(
        id=dm01_raw_substring.SCANNER_ID,
        class_ids=dm01_raw_substring.CLASS_IDS,
        description="raw-substring keyword tests outside utils.trigger_match",
        mode="gate",
        run=dm01_raw_substring.scan,
    ),
    Scanner(
        id=dm16_config_reachability.SCANNER_ID,
        class_ids=dm16_config_reachability.CLASS_IDS,
        description="config.yaml leaves with no reader outside config/",
        # 87 live candidates: above the count at which a gate would be honest,
        # and each one needs a human to say "dead" vs "read another way".
        mode="report",
        run=dm16_config_reachability.scan,
    ),
    Scanner(
        id=dm17_apply_without_guard.SCANNER_ID,
        class_ids=dm17_apply_without_guard.CLASS_IDS,
        description="--apply scripts without a daemon guard; data/ paths in tests",
        mode="gate",
        run=dm17_apply_without_guard.scan,
    ),
    Scanner(
        id=dm18_except_returns_empty.SCANNER_ID,
        class_ids=dm18_except_returns_empty.CLASS_IDS,
        description="broad except returning an empty result beside a store call",
        mode="gate",
        run=dm18_except_returns_empty.scan,
    ),
    Scanner(
        id=dm29_phrase_append.SCANNER_ID,
        class_ids=dm29_phrase_append.CLASS_IDS,
        description="changelog phrase-append signature (>=3 dated batches)",
        # BC-76's own Closure line says the judgment is human, not automatable.
        mode="report",
        run=dm29_phrase_append.scan,
    ),
    Scanner(
        id=dm31_live_state_default.SCANNER_ID,
        class_ids=dm31_live_state_default.CLASS_IDS,
        description="public function asserts live budget/toggle state via a default",
        # Gated on arrival: the whole tree yields six candidates, hand-reviewed
        # (two were the live 2026-09-11 defect and are fixed; four are
        # caller-chosen sizes or component-local flags and sit in the
        # baseline), so a NEW one is worth a human look.
        mode="gate",
        run=dm31_live_state_default.scan,
    ),
    Scanner(
        id=catalog_scanner.SCANNER_ID,
        class_ids=catalog_scanner.CLASS_IDS,
        description="BUG_CLASSES.md index/body/method admission diagnostics",
        mode="gate",
        run=catalog_scanner.scan,
    ),
)

SCANNERS: dict[str, Scanner] = {scanner.id: scanner for scanner in _SCANNER_LIST}

__all__ = [
    "Finding",
    "SCANNERS",
    "ScanResult",
    "Scanner",
    "ScannerError",
]
