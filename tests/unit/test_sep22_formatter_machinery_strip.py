"""tests/unit/test_sep22_formatter_machinery_strip.py

A delivery notice (the appended "> ⚠️ ..." blockquote gui/handlers.py glues
onto a stored assistant reply AFTER it was generated, e.g. the calendar
backstop notice) is machinery, not something the assistant said. Because
display == storage, it used to re-enter later prompts as content:
[RECENT CONVERSATION] and [LAST EXCHANGE FOR CONTEXT] both quoted a stored
notice back into the next turn's prompt (2026-09-22, turn 3 — the reply
that produced the notice fed the "fair's Sunday" turn).

core/prompt/formatter.py now applies read_time_markers.strip_delivery_notices
to the Daemon segment at every render site (mem_parts / _format_memory /
[LAST EXCHANGE FOR CONTEXT]) BEFORE the existing unverified-action-claim
annotator runs, so a pre-existing "[unverified action claim]" marker line
and any mid-text blockquote are left untouched — only a TRAILING delivery
notice goes.

class: BC-75, BC-91. Plan: PLAN_20260922_turn_audit_guardfixes.md §7 (B4).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from utils.read_time_markers import UNVERIFIED_CLAIM_MARKER

# Exact live notice text (2026-09-22 finding A2/A3;
# gui/handlers.py's calendar backstop).
CALENDAR_NOTICE = (
    "\n\n> ⚠️ I don't see that on your calendar — nothing "
    "was created. Say \"add it\" and I'll queue a card."
)


def _get_formatter():
    from core.prompt.formatter import PromptFormatter

    token_mgr = MagicMock()
    token_mgr.count_tokens = MagicMock(return_value=10)
    fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
    fmt._feature_inventory_cache = None
    return fmt


def _make_context(recent_conversations=None, **overrides):
    ctx = {
        "recent_conversations": recent_conversations or [],
        "memories": [],
        "user_profile": "",
        "narrative_state": "",
        "summaries": [],
        "reflections": [],
        "dreams": [],
        "semantic_chunks": [],
        "wiki": [],
        "personal_notes": [],
        "reference_docs": [],
        "user_uploads": [],
        "git_commits": [],
        "procedural_skills": [],
        "proposed_features": [],
        "graph_context": [],
        "unresolved_threads": [],
        "upcoming_schedule": [],
        "google_calendar": [],
        "proactive_insights": [],
        "web_search_results": None,
    }
    ctx.update(overrides)
    return ctx


# --- [RECENT CONVERSATION] (mem_parts) --------------------------------------


class TestRecentConversationMachineryStrip:
    """[RECENT CONVERSATION] renders each entry through mem_parts()."""

    def test_trailing_calendar_notice_stripped_body_kept(self):
        fmt = _get_formatter()
        body = "Sounds like a rough morning — glad you got a little rest."
        old_ts = (datetime.now() - timedelta(hours=5)).isoformat()
        entry = {
            "query": "did you add it",
            "response": body + CALENDAR_NOTICE,
            "timestamp": old_ts,
        }
        ctx = _make_context(recent_conversations=[entry])
        result = fmt._assemble_prompt(ctx, "how's the schedule looking")
        assert body in result
        assert "> ⚠️" not in result

    def test_unverified_action_claim_marker_survives_notice_strip(self):
        """Proves ONLY the trailing notice is stripped — a marker line already
        present in the stored response (a prior render's annotation, glued
        back in because display == storage) is left exactly where it is."""
        fmt = _get_formatter()
        body = "Saved the note for you already."
        old_ts = (datetime.now() - timedelta(hours=5)).isoformat()
        response = body + "\n" + UNVERIFIED_CLAIM_MARKER + CALENDAR_NOTICE
        entry = {
            "query": "did that get saved",
            "response": response,
            "timestamp": old_ts,
        }
        ctx = _make_context(recent_conversations=[entry])
        result = fmt._assemble_prompt(ctx, "checking in on that")
        assert UNVERIFIED_CLAIM_MARKER in result
        assert "> ⚠️" not in result

    def test_midtext_blockquote_survives(self):
        """Only a TRAILING blockquote run starting with the delivery-notice
        prefix is a candidate; a blockquote sitting mid-text, followed by a
        normal paragraph, is not a delivery notice and stays."""
        fmt = _get_formatter()
        response = (
            "Intro paragraph.\n\n"
            "> ⚠️ heads up, mid text\n\n"
            "A normal paragraph follows and is the last line."
        )
        old_ts = (datetime.now() - timedelta(hours=5)).isoformat()
        entry = {
            "query": "what happened earlier",
            "response": response,
            "timestamp": old_ts,
        }
        ctx = _make_context(recent_conversations=[entry])
        result = fmt._assemble_prompt(ctx, "recap please")
        assert "> ⚠️ heads up, mid text" in result
        assert "A normal paragraph follows and is the last line." in result


# --- [LAST EXCHANGE FOR CONTEXT] --------------------------------------------


class TestLastExchangeMachineryStrip:
    """[LAST EXCHANGE FOR CONTEXT] is built from recent_conversations[0]'s
    assistant text when that entry is same-session (formatter.py ~1915)."""

    def test_last_exchange_strips_trailing_notice(self):
        fmt = _get_formatter()
        body = "I don't see anything on your calendar for that."
        recent_ts = (datetime.now() - timedelta(minutes=5)).isoformat()
        entry = {
            "query": "did you add it",
            "response": body + CALENDAR_NOTICE,
            "timestamp": recent_ts,
        }
        ctx = _make_context(recent_conversations=[entry])
        result = fmt._assemble_prompt(ctx, "so is it on there or not")
        assert "[LAST EXCHANGE FOR CONTEXT]" in result
        assert body in result
        assert "> ⚠️" not in result


# --- _format_memory (sibling Daemon-segment render site) --------------------


class TestFormatMemoryMachineryStrip:
    """_format_memory shares the annotate_unverified_action_claim call with
    mem_parts (see core/action_claim_guard.annotate_unverified_action_claim's
    docstring, which names both as the two conversation-render sites) and
    must strip a trailing delivery notice the same way."""

    def test_format_memory_strips_trailing_notice(self):
        fmt = _get_formatter()
        body = "Sounds like a rough morning — glad you got a little rest."
        mem = {
            "query": "did you add it",
            "response": body + CALENDAR_NOTICE,
            "timestamp": "2026-05-28T10:00:00",
        }
        result = fmt._format_memory(mem)
        assert body in result
        assert "> ⚠️" not in result
