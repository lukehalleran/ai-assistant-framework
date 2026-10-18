"""tests/unit/test_temporal_resolver_recalled_anchor.py

Incident 2026-10-18 14:04 (Sunday): yesterday's stored turns said, in their
own words, "finish HW 6 by end of day today" / "due today for them" /
"skipping the questions until today" (all meaning Saturday). The
recalled items were rendered with a timestamp but the relative words inside
the text stayed bare, and the reply written on Sunday copied every "today"
as if it meant Sunday.

Fix: ``utils.temporal_resolver.anchor_temporal_references`` ANNOTATES the
relative words of recalled text from an earlier day with the absolute date
they meant (``today [Sat 2026-10-17]``) — same resolution code path as
``resolve_temporal_references``, same-day / unparseable-timestamp text
untouched, idempotent, word-bounded. Applied at the recalled-conversation
render chokepoints: core/agentic/formatters.py (format_memories,
format_recent_conversations, via the controller wrappers) and
core/prompt/formatter.py (mem_parts for [RECENT CONVERSATION] /
[RELEVANT MEMORIES], and _format_memory).

class: BC-100 (content-relative time words read as now-relative;
draft in the lane's bc_draft.md).
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from utils.temporal_resolver import (
    anchor_temporal_references,
    resolve_temporal_references,
)

NOW = datetime(2026, 10, 18, 14, 4, 0)          # Sunday
SAT = "2026-10-17T16:20:00"                      # said Saturday
SAT_SPACE = "2026-10-17 16:20"
SUN = "2026-10-18T09:00:00"                      # said today


def _wrap(text):
    """Wrapped/indented variant of a live text (workflow section 3)."""
    return "\n    ".join(text.split(" "))


FIXTURES = [
    ("finish HW 6 by end of day today", "today [Sat 2026-10-17]"),
    ("HW 7 is due today for them", "today [Sat 2026-10-17]"),
    ("skipping the 5 Arena questions until today", "today [Sat 2026-10-17]"),
    ("I will be done tonight", "tonight [Sat 2026-10-17]"),
    ("the quiz is due tomorrow", "tomorrow [Sun 2026-10-18]"),
    ("I rested yesterday", "yesterday [Fri 2026-10-16]"),
    ("Rowan visits in 3 days", "in 3 days [Tue 2026-10-20]"),
    ("lab starts next Monday", "next Monday [Mon 2026-10-19]"),
]


class TestAnchorResolver:
    @pytest.mark.parametrize("text,expected", FIXTURES)
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_saturday_text_read_on_sunday(self, text, expected, wrapped):
        src = _wrap(text) if wrapped else text
        out = anchor_temporal_references(src, datetime.fromisoformat(SAT), NOW)
        want = _wrap(expected) if wrapped else expected
        # the original words survive, the bracket follows the expression
        assert out != src
        bracket = "[" + expected.split("[", 1)[1]
        assert bracket in out
        word = expected.split(" [")[0]
        assert (_wrap(word) if wrapped else word) in out
        assert out.replace(" " + bracket, "") == src

    def test_incident_sentence_exact(self):
        out = anchor_temporal_references(
            "finish HW 6 by end of day today", SAT, NOW)
        assert out == "finish HW 6 by end of day today [Sat 2026-10-17]"

    def test_string_timestamp_formats(self):
        for ts in (SAT, SAT_SPACE, "2026-10-17", datetime.fromisoformat(SAT)):
            assert anchor_temporal_references("due tomorrow", ts, NOW) == \
                "due tomorrow [Sun 2026-10-18]"

    def test_same_day_untouched(self):
        for ts in (SUN, datetime.fromisoformat(SUN)):
            assert anchor_temporal_references("due today", ts, NOW) == "due today"

    @pytest.mark.parametrize("ts", [None, "", "Unknown time", "garbage", 12345])
    def test_unparseable_timestamp_untouched(self, ts):
        assert anchor_temporal_references("due today", ts, NOW) == "due today"

    def test_future_dated_item_untouched(self):
        assert anchor_temporal_references("due today", "2026-10-19T08:00:00", NOW) == "due today"

    def test_idempotent(self):
        once = anchor_temporal_references("due today and tomorrow", SAT, NOW)
        assert anchor_temporal_references(once, SAT, NOW) == once
        assert once.count("[") == 2

    def test_word_boundaries(self):
        for text in ("todays plan", "Tomorrowland tickets", "yesterdays news", "nexts Monday"):
            assert anchor_temporal_references(text, SAT, NOW) == text

    def test_multiple_expressions_one_pass(self):
        out = anchor_temporal_references("yesterday I said I'd finish today, due tomorrow", SAT, NOW)
        assert "yesterday [Fri 2026-10-16]" in out
        assert "today [Sat 2026-10-17]" in out
        assert "tomorrow [Sun 2026-10-18]" in out

    def test_same_resolution_as_replace_mode(self):
        """One code path: the annotation is the replace-mode value."""
        text = "ship it tomorrow, then the following day review, next Friday retro"
        ref = datetime.fromisoformat(SAT)
        replaced = resolve_temporal_references(text, ref)
        annotated = anchor_temporal_references(text, ref, NOW)
        import re
        brackets = re.findall(r"\[([^\]]+)\]", annotated)
        for b in brackets:
            assert b in replaced

    def test_replace_mode_unchanged(self):
        assert resolve_temporal_references("due tomorrow", datetime(2026, 3, 12)) == \
            "due Fri 2026-03-13"


# --- renderers ---------------------------------------------------------------


def _agentic_formatter():
    from core.agentic.formatters import AgenticFormatter
    return AgenticFormatter()


class TestAgenticRenderers:
    def test_format_memories_old_vs_same_day(self):
        f = _agentic_formatter()
        out = f.format_memories([
            {"timestamp": SAT, "content": "finish HW 6 by end of day today"},
            {"timestamp": SUN, "content": "finish HW 7 by end of day today"},
        ], now=NOW)
        assert "finish HW 6 by end of day today [Sat 2026-10-17]" in out
        assert "finish HW 7 by end of day today\n" in out + "\n"
        assert out.count("[Sat 2026-10-17]") == 1

    def test_format_memories_response_and_wrapped(self):
        f = _agentic_formatter()
        out = f.format_memories([{
            "timestamp": SAT, "content": _wrap("it is due tomorrow"),
            "response": _wrap("I rested yesterday"),
        }], now=NOW)
        assert "tomorrow [Sun 2026-10-18]" in out
        assert "yesterday [Fri 2026-10-16]" in out

    def test_format_recent_conversations(self):
        f = _agentic_formatter()
        out = f.format_recent_conversations([
            {"timestamp": SAT, "query": "HW 7 is due today for them",
             "response": "Then finish it tonight."},
            {"timestamp": SUN, "query": "quiz is due today", "response": "ok"},
        ], now=NOW)
        assert "HW 7 is due today [Sat 2026-10-17] for them" in out
        assert "tonight [Sat 2026-10-17]" in out
        assert "quiz is due today\n" in out

    def test_annotation_precedes_clip(self):
        """Annotation runs BEFORE clip_text: an expression early in a long text
        keeps its bracket intact and the (explicit) truncation marker comes
        after it; nothing is annotated past the cut."""
        from core.agentic.formatters import TRUNCATION_MARKER
        f = _agentic_formatter()
        text = "due today " + "filler " * 120 + "and again today"
        out = f.format_memories([{"timestamp": SAT, "content": text}], now=NOW)
        assert "due today [Sat 2026-10-17] filler" in out
        assert TRUNCATION_MARKER in out
        assert out.index("[Sat 2026-10-17]") < out.index(TRUNCATION_MARKER)
        assert out.count("[Sat 2026-10-17]") == 1

    def test_controller_wrapper_uses_controller_clock(self, monkeypatch):
        from core.agentic.controller import AgenticSearchController

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 18, 9, 30, 0)

        manager = MagicMock()
        manager.api_models = {}
        c = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
        monkeypatch.setattr("core.agentic.controller.datetime", _Frozen)
        out = c._format_memories([{"timestamp": SAT, "content": "due today"}])
        assert "due today [Sat 2026-10-17]" in out
        out = c._format_recent_conversations([{"timestamp": SAT, "query": "due today", "response": ""}])
        assert "due today [Sat 2026-10-17]" in out


def _base_formatter():
    from core.prompt.formatter import PromptFormatter
    token_mgr = MagicMock()
    token_mgr.count_tokens = MagicMock(return_value=10)
    fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
    fmt._feature_inventory_cache = None
    return fmt


def _ctx(**kw):
    ctx = {k: [] for k in (
        "recent_conversations", "memories", "summaries", "reflections", "dreams",
        "semantic_chunks", "wiki", "personal_notes", "reference_docs", "user_uploads",
        "git_commits", "procedural_skills", "proposed_features", "graph_context",
        "unresolved_threads", "upcoming_schedule", "google_calendar", "proactive_insights")}
    ctx.update({"user_profile": "", "narrative_state": "", "web_search_results": None})
    ctx.update(kw)
    return ctx


@pytest.fixture
def frozen_formatter_clock(monkeypatch):
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 18, 14, 4, 0)
    monkeypatch.setattr("core.prompt.formatter.datetime", _Frozen)


class TestBaseFormatterRenderers:
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_recent_conversation_section(self, frozen_formatter_clock, wrapped):
        q = "HW 7 is due today for them"
        r = "finish HW 6 by end of day today"
        if wrapped:
            q, r = _wrap(q), _wrap(r)
        fmt = _base_formatter()
        out = fmt._assemble_prompt(_ctx(recent_conversations=[
            {"query": q, "response": r, "timestamp": SAT},
            {"query": "quiz is due today", "response": "ok", "timestamp": SUN},
        ]), "hello")
        sec = out.split("[RECENT CONVERSATION]")[1]
        assert sec.count("today [Sat 2026-10-17]") == 2
        assert "quiz is due today\n" in sec or "quiz is due today " in sec
        assert "quiz is due today [" not in sec

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_relevant_memories_section(self, frozen_formatter_clock, wrapped):
        c = "skipping the 5 Arena questions until today"
        if wrapped:
            c = _wrap(c)
        fmt = _base_formatter()
        out = fmt._assemble_prompt(_ctx(memories=[
            {"content": c, "timestamp": SAT},
            {"content": "same day today", "timestamp": SUN},
        ]), "hello")
        sec = out.split("[RELEVANT MEMORIES]")[1]
        assert "today [Sat 2026-10-17]" in sec
        assert "same day today" in sec and "same day today [" not in sec

    def test_format_memory_method(self, frozen_formatter_clock):
        fmt = _base_formatter()
        out = fmt._format_memory({"query": "due tomorrow", "response": "I rested yesterday",
                                  "timestamp": SAT})
        assert "tomorrow [Sun 2026-10-18]" in out
        assert "yesterday [Fri 2026-10-16]" in out
        same = fmt._format_memory({"query": "due tomorrow", "timestamp": SUN})
        assert "tomorrow [" not in same
