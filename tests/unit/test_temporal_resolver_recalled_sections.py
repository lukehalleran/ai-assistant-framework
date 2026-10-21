"""Recalled summaries, reflections and "last night"/"this morning" words carry the date they meant.

Incident 2026-10-18 (BC-100 registered; #90/#92 anchored recalled conversations and
memories only) with residual coverage found 2026-10-20: [RECENT/SEMANTIC SUMMARIES]
and [RECENT/SEMANTIC REFLECTIONS] (prompt formatter AND the agentic evidence renderer)
still replayed "today" / "last night" as now-relative, and the resolver knew neither
"last night" nor "this morning|afternoon|evening" nor "earlier today".

Class: BC-100 (content-relative time words read as now-relative).
ONE resolver (_resolve) and ONE date-picking helper (anchor_recalled_item) serve both
renderers. Invented content only.
"""
from datetime import datetime

import pytest

import core.prompt.formatter as pf
from core.agentic.formatters import AgenticFormatter
from utils.temporal_resolver import (
    anchor_recalled_item,
    anchor_temporal_references,
    has_temporal_reference,
    resolve_temporal_references,
)

SAT = "2026-10-17T20:00:00"
NOW = datetime(2026, 10, 19, 9, 0)


def _wrap(t):
    return "\n".join("    " + p for p in t.split(". "))


def _both(text):
    return [text, _wrap(text)]


@pytest.mark.parametrize("src,want", [
    ("we talked last night about the quasar", "we talked last night [Fri 2026-10-16] about the quasar"),
    ("I slept badly this morning. Fine now.", "I slept badly this morning [Sat 2026-10-17]. Fine now."),
    ("gym this afternoon", "gym this afternoon [Sat 2026-10-17]"),
    ("call Rowan this evening", "call Rowan this evening [Sat 2026-10-17]"),
    ("I fixed it earlier today", "I fixed it earlier today [Sat 2026-10-17]"),
])
def test_new_expressions_annotated_clean_and_wrapped(src, want):
    for form_src, form_want in ((src, want), (_wrap(src), _wrap(want))):
        assert anchor_temporal_references(form_src, SAT, NOW) == form_want


def test_same_day_unchanged_and_idempotent():
    t = "last night and this morning, earlier today too"
    assert anchor_temporal_references(t, "2026-10-19T07:00:00", NOW) == t
    once = anchor_temporal_references(t, SAT, NOW)
    assert once.count("[") == 3
    assert anchor_temporal_references(once, SAT, NOW) == once


@pytest.mark.parametrize("text", [
    "nightlife is fun", "I like mornings", "a good night", "last nights are long",
    "this morning's plan", "this mornings", "burning night", "thisevening",
])
def test_word_boundaries_untouched(text):
    for t in _both(text):
        assert not has_temporal_reference(t)
        assert anchor_temporal_references(t, SAT, NOW) == t


def test_earlier_today_is_one_annotation_and_tonight_unaffected():
    out = anchor_temporal_references("earlier today and tonight", SAT, NOW)
    assert out == "earlier today [Sat 2026-10-17] and tonight [Sat 2026-10-17]"


def test_replace_mode_rewrites_new_expressions():
    ref = datetime(2026, 10, 17)
    assert resolve_temporal_references("met last night", ref) == "met Fri 2026-10-16"
    assert resolve_temporal_references("met this morning", ref) == "met Sat 2026-10-17"
    assert resolve_temporal_references("done earlier today", ref) == "done Sat 2026-10-17"


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(pf, "datetime", _Frozen)


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_day_summary_carries_bracket(frozen, wrapped):
    txt = "Rowan finished the draft last night. Plan is due today."
    txt = _wrap(txt) if wrapped else txt
    out = pf._format_summary_section(
        [{"content": txt, "timestamp": "2026-10-17T23:00:00", "covers": "2026-10-17"}],
        "RECENT SUMMARIES")
    assert "last night [Fri 2026-10-16]" in out
    assert "today [Sat 2026-10-17]" in out


def test_multi_day_and_spanless_summary_untouched(frozen):
    out = pf._format_summary_section(
        [{"content": "done today", "timestamp": "2026-10-17", "covers": "2026-10-15 – 2026-10-17"},
         {"content": "done today too", "timestamp": "2026-10-17"}],
        "RECENT SUMMARIES")
    assert "[Sat" not in out and "[Fri" not in out


def test_reflection_anchored_to_creation_day(frozen):
    out = pf._format_summary_section(
        [{"content": "Tamsin seemed tired this morning.", "timestamp": "2026-10-17T21:00:00"}],
        "RECENT REFLECTIONS", kind="reflection")
    assert "this morning [Sat 2026-10-17]" in out


def test_helper_rules():
    assert anchor_recalled_item("x today", {"covers": "2026-10-17"}, NOW, kind="summary") == "x today [Sat 2026-10-17]"
    assert anchor_recalled_item("x today", {"covers": "2026-10-16 – 2026-10-17"}, NOW, kind="summary") == "x today"
    assert anchor_recalled_item("x today", {"timestamp": SAT}, NOW, kind="summary") == "x today"
    assert anchor_recalled_item("x today", {"timestamp": SAT}, NOW, kind="reflection") == "x today [Sat 2026-10-17]"
    assert anchor_recalled_item("x today", {"timestamp": "garbage"}, NOW, kind="reflection") == "x today"


def test_agentic_renderers_use_same_rule():
    f = AgenticFormatter()
    s = f.format_summaries(
        [{"content": "shipped it last night", "timestamp": "2026-10-17", "covers": "2026-10-17"},
         {"content": "shipped it today", "timestamp": "2026-10-17", "covers": "2026-10-16 – 2026-10-17"}],
        now=NOW)
    assert "last night [Fri 2026-10-16]" in s
    assert "shipped it today" in s and "today [" not in s  # multi-day span untouched
    r = f.format_reflections([{"content": "calm this evening", "timestamp": SAT}], now=NOW)
    assert "this evening [Sat 2026-10-17]" in r


def test_last_exchange_never_renders_prior_day_text():
    """The [LAST EXCHANGE] skip rule (date differs => session boundary) means prior-day
    text never reaches it, so no anchoring is needed there."""
    assert pf._detect_session_boundary(datetime(2026, 10, 18, 23, 55), datetime(2026, 10, 19, 0, 5))
