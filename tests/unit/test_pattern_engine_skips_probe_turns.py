"""Pattern analysis never counts owner [test] probe turns, in any session.

Incident 2026-10-18, class BC-58 (test-state isolation): ``_corpus_entries``
applied only the junk filter, so probe turns inflated session-rhythm and
content-type counts AND the per-bucket denominators, and the tone dimension
(which reads turn records, where a probe keeps its envelope in ``query``)
counted a probe's tone as the user's mood. Probes are excluded in every
session through THE predicate ``memory.fact_source.is_test_turn``; numerators
and denominators read the same ``_corpus_entries``, so they drop the same
turns. Invented content only; envelope forms clean and wrapped/indented.
"""

import json
from datetime import datetime, timedelta

import pytest

from memory.corpus_manager import CorpusManager
from memory.pattern_engine import PatternQuery, run_pattern_query

NOW = datetime(2026, 10, 18, 15, 0, 0)
BODY = "Rowan says the budget sheet needs a second tab, remember that?"
PROBE_FORMS = {
    "inline": f"[test]{BODY}[/test]",
    "inline_wrapped": f"   [test]{BODY}[/test]   ",
    "whole_line": f"[test]\n{BODY}\n[/test]",
    "whole_line_indented": "  [test]\n  Rowan says the budget sheet needs\n"
                           "  a second tab, remember that?\n  [/test]",
}
FORMS = pytest.mark.parametrize("probe", list(PROBE_FORMS.values()), ids=list(PROBE_FORMS))


def _corpus(tmp_path, probe):
    def e(ts, q):
        return {"query": q, "response": "ok", "timestamp": ts.isoformat()}

    entries = [
        e(NOW - timedelta(days=3, hours=1), "Tamsin moved the standup to ten."),
        e(NOW - timedelta(days=2, hours=1), "Rowan sent the lease paperwork."),
        e(NOW - timedelta(days=2), probe),                     # prior-session probe
        e(NOW - timedelta(days=1, hours=1), "The budget sheet is finished."),
        e(NOW - timedelta(minutes=20), probe),                 # current-session probe
        e(NOW - timedelta(hours=1), "Quick check on the lease paperwork."),
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return CorpusManager(corpus_file=str(path))


@FORMS
def test_session_rhythm_numerator_and_denominator_skip_probes(tmp_path, probe):
    res = run_pattern_query(
        PatternQuery(dimension="session_rhythm", window_days=7, now=NOW),
        corpus_manager=_corpus(tmp_path, probe))
    assert res.total == 4                  # 4 real turns, 2 probes excluded
    assert res.denominator_total == 4      # denominators exclude the same turns


@FORMS
def test_content_type_denominator_skips_probes(tmp_path, probe):
    res = run_pattern_query(
        PatternQuery(dimension="content_type", terms=["lyrics"], window_days=7, now=NOW),
        corpus_manager=_corpus(tmp_path, probe))
    assert res.denominator_total == 4


@FORMS
def test_tone_dimension_skips_probe_turn_records(tmp_path, probe):
    rows = [
        {"ts": (NOW - timedelta(days=2)).isoformat(), "tone_level": "CONCERN",
         "query": "I am worn out today"},
        {"ts": (NOW - timedelta(days=1)).isoformat(), "tone_level": "HIGH",
         "query": probe},
    ]
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = run_pattern_query(
        PatternQuery(dimension="tone", window_days=7, now=NOW),
        telemetry_path=str(path))
    assert res.total == 1
    assert {v for b in res.buckets for v in b.values} == {"concern"}


def test_topic_keyword_keeps_skipping_probes(tmp_path):
    """#91 made search_keyword skip probes; pinned here, not re-implemented."""
    res = run_pattern_query(
        PatternQuery(dimension="topic_keyword", terms=["budget"], window_days=7, now=NOW),
        corpus_manager=_corpus(tmp_path, PROBE_FORMS["inline"]))
    assert res.total == 1                  # only the real "budget sheet" turn
