"""Tests for scripts/purge_adaptive_exemplars.py matching logic.

2026-08-15: the gate-veto teacher poisoned web_search/no_search with an
explicit lookup command and a news share; the script removes curated entries
(dry-run default, pre-image backup, daemon-live guard on --apply).
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "purge_adaptive_exemplars",
    Path(__file__).resolve().parents[2] / "scripts" / "purge_adaptive_exemplars.py",
)
purge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(purge)


STORE = {
    "web_search": {
        "no_search": [
            {"text": "Ugh. Either I'm dying or super fucking constipated.", "source": "gate_veto", "ts": "t1"},
            {"text": 'The president "declared" the strait of hormuz to be us land lmfao', "source": "gate_veto", "ts": "t2"},
            {"text": "Look it up it's pretty funny", "source": "gate_veto", "ts": "t3"},
        ],
        "search_worthy": [
            {"text": "look it up somewhere else entirely", "source": "citation", "ts": "t4"},
        ],
    }
}


class TestFindMatches:
    def test_matches_case_insensitive_substring(self):
        matched, unmatched = purge.find_matches(
            STORE, "web_search", "no_search", ["strait of hormuz", "LOOK IT UP"]
        )
        assert [e["ts"] for e in matched] == ["t2", "t3"]
        assert unmatched == []

    def test_unmatched_term_reported(self):
        matched, unmatched = purge.find_matches(
            STORE, "web_search", "no_search", ["strait of hormuz", "no such text"]
        )
        assert [e["ts"] for e in matched] == ["t2"]
        assert unmatched == ["no such text"]

    def test_scoped_to_label(self):
        # "look it up" also appears under search_worthy — must not match there
        matched, _ = purge.find_matches(STORE, "web_search", "no_search", ["look it up"])
        assert [e["ts"] for e in matched] == ["t3"]

    def test_missing_domain_is_empty(self):
        matched, unmatched = purge.find_matches(STORE, "tone", "concern", ["x"])
        assert matched == []
        assert unmatched == ["x"]
