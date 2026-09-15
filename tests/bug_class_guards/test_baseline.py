"""The content-anchored baseline module and live-tree parity (schema 2).

The gate has to fail in BOTH directions.  A new occurrence of a class must go
red, and so must a baseline occurrence whose candidate is gone — otherwise
fixed debt is laundered by a row nobody deletes.  Those ratchet controls live
in ``test_cli_contract.py`` (through the deployed CLI) and ``test_scanners.py``
(anchor stability); this file covers the baseline module's own multiset,
rendering and legacy-migration properties, and the live parity smoke that goes
red when the committed baseline or its reviews drift from the tree.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import check_bug_classes
from bug_class_guards import baseline as baseline_mod
from bug_class_guards import dispositions as dispositions_mod
from bug_class_guards.scanners import SCANNERS

from fixtures import DM18_TWIN_HANDLERS, build_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_BASELINE = REPO_ROOT / baseline_mod.DEFAULT_BASELINE
COMMITTED_DISPOSITIONS = REPO_ROOT / dispositions_mod.DEFAULT_DISPOSITIONS

# The schema-1 baseline reviewed and migrated on 2026-09-13: 133 occurrences
# (129 unique keys + 4 duplicate occurrences). Its exact bytes are the
# reconciliation target for the ledger's legacy history.
LEGACY_SHA256 = "9bb56e6d3f072cce5becd97293aaac93f97f7a0f592b212f9579a5f6dc47a756"
LEGACY_OCCURRENCES = 133

GATE_KINDS = {sid: scanner.kinds for sid, scanner in SCANNERS.items() if scanner.mode == "gate"}
DM18 = "dm18_except_returns_empty"


def _anchor(digest_char: str) -> tuple[str, str, str, str, str]:
    return (DM18, "memory/store.py", "Store.get_rows", "broad_except_returns_empty", digest_char * 64)


class TestBaselineModule:
    def test_save_then_load_round_trips_multiplicity_and_bytes(self, tmp_path):
        build_tree(tmp_path, {"memory/store.py": DM18_TWIN_HANDLERS})
        findings = SCANNERS[DM18].scan(tmp_path).findings
        stored = baseline_mod.from_findings(findings)
        path = tmp_path / "baseline.json"
        baseline_mod.save(path, stored)
        loaded = baseline_mod.load(path, GATE_KINDS, SCANNERS)
        assert loaded.counter == stored.counter
        assert loaded.occurrences == 2 and len(loaded.counter) == 1
        assert baseline_mod.render(loaded) == path.read_text(encoding="utf-8")

    def test_compare_is_a_two_way_multiset_ratchet(self):
        first, second = _anchor("a"), _anchor("b")
        result = baseline_mod.compare(Counter({first: 3, second: 1}), Counter({first: 2, second: 2}))
        assert result.matched == 3
        assert result.new == [first]
        assert result.stale == [second]

    def test_subset_keeps_only_the_named_scanners(self):
        other = ("dm31_live_state_default", "u.py", "f", "live_state_literal_default", "c" * 64)
        counter = Counter({_anchor("a"): 2, other: 1})
        assert baseline_mod.subset(counter, [DM18]) == Counter({_anchor("a"): 2})

    def test_legacy_render_is_byte_identical_to_the_schema_one_writer(self, tmp_path):
        rows = [
            {"scanner": DM18, "path": "memory/a.py", "symbol": "A.get", "text": "return []"},
            {"scanner": DM18, "path": "memory/a.py", "symbol": "A.get", "text": "return []"},
            {"scanner": "dm17_apply_without_guard", "path": "scripts/x.py", "symbol": "", "text": "--apply …"},
        ]
        rows.sort(key=lambda row: (row["scanner"], row["path"], row["symbol"], row["text"]))
        text = json.dumps({"schema": 1, "entries": rows}, indent=2, ensure_ascii=False) + "\n"
        path = tmp_path / "legacy.json"
        path.write_text(text, encoding="utf-8")
        assert baseline_mod.legacy_render(baseline_mod.legacy_load(path)) == text


class TestLiveTree:
    """Parity smoke — the test that goes red when the baseline or its reviews drift.

    It runs the deployed CLI over the real repository with the COMMITTED
    policy, baseline and dispositions.  A failure means a new occurrence was
    introduced, debt was fixed and its row must be removed, or a source file
    with accepted debt changed and needs re-review.
    """

    def test_committed_baseline_and_dispositions_match_the_live_tree(self, capsys):
        code = check_bug_classes.main(["scan", "--root", str(REPO_ROOT)])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "bug-class scan: OK" in out

    def test_every_required_leg_of_every_scanner_has_files(self, capsys):
        check_bug_classes.main(["scan", "--root", str(REPO_ROOT), "--json"])
        payload = json.loads(capsys.readouterr().out)
        for name, data in payload["scanners"].items():
            for leg in data["legs"]:
                if leg["required"]:
                    assert leg["available"] and leg["files_processed"] > 0, (name, leg["id"])

    def test_committed_ledger_reconciles_every_legacy_occurrence(self):
        ledger = dispositions_mod.load(COMMITTED_DISPOSITIONS, GATE_KINDS, SCANNERS)
        assert (ledger.legacy_sha256, ledger.legacy_occurrences) == (LEGACY_SHA256, LEGACY_OCCURRENCES)
        history = Counter(record.legacy for record in ledger.records if record.legacy is not None)
        assert sum(history.values()) == LEGACY_OCCURRENCES
        rendered = baseline_mod.legacy_render(history).encode("utf-8")
        assert hashlib.sha256(rendered).hexdigest() == LEGACY_SHA256

    def test_active_debt_equals_the_committed_baseline(self):
        stored = baseline_mod.load(COMMITTED_BASELINE, GATE_KINDS, SCANNERS)
        ledger = dispositions_mod.load(COMMITTED_DISPOSITIONS, GATE_KINDS, SCANNERS)
        accepted = Counter(r.anchor for r in ledger.records if r.status == "accepted_debt")
        assert accepted == stored.counter
