"""2026-10-08 agentic turn: the decision prompt keeps the calendar (batch 5, lane H2).

Live incident (agentic T3): recent_conversations + session_timeline filled the
10,000-token budget, so the token manager emptied unresolved_threads and
google_calendar; the decision prompt carried only counts plus a 220-char digest
that clipped Daemon's own reply mid-word, and the model bound a salon on the
calendar to a professor.

Covered here (synthetic text only):
  H2.2  today's calendar titles reach the decision-prompt inventory verbatim
  H2.3  small time-sensitive sections are metered before history
  H2.4  the recent-turn digest labels Daemon's wording and clips on a word

H2.1 (reuse blocked on budget-emptied evidence) is NOT covered: the controller
never receives the pre-budget section outcomes (see handoff).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.prompt.token_manager import TokenManager


def _controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


def _iso(day_offset=0, hour=14):
    d = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
    return (d + timedelta(days=day_offset)).isoformat()


class TestCalendarTitlesInInventory:
    def test_todays_title_verbatim(self):
        ctx = {"google_calendar": [
            {"summary": "Service(s) scheduled at Rivera Hair Studio", "start": _iso(0), "end": _iso(0, 15)},
        ]}
        inv = _controller()._compute_context_inventory(ctx)
        assert "Service(s) scheduled at Rivera Hair Studio" in inv

    def test_other_days_excluded(self):
        ctx = {"google_calendar": [
            {"summary": "Office hours with Prof. Rivera", "start": _iso(1), "end": _iso(1, 15)},
            {"summary": "Rivera Hair Studio", "start": _iso(0), "end": _iso(0, 15)},
        ]}
        inv = _controller()._compute_context_inventory(ctx)
        assert "Rivera Hair Studio" in inv
        assert "Prof. Rivera" not in inv

    def test_no_calendar_no_line(self):
        c = _controller()
        assert "CALENDAR" not in c._compute_context_inventory({"memories": [{"content": "m"}]})
        only_tomorrow = {"google_calendar": [{"summary": "X", "start": _iso(2), "end": _iso(2, 15)}]}
        assert "CALENDAR" not in c._compute_context_inventory(only_tomorrow)

    def test_caps_events_and_chars(self):
        ctx = {"google_calendar": [
            {"summary": f"Appointment number {i} " + "word " * 20, "start": _iso(0), "end": _iso(0, 15)}
            for i in range(9)
        ]}
        line = _controller()._todays_calendar_line(ctx["google_calendar"])
        assert len(line) <= 300
        assert "number 5" not in line


class TestSmallSectionsMeteredBeforeHistory:
    class _MM:
        def get_active_model_name(self):
            return "test-model"

    class _Tok:
        def count_tokens(self, text, model_name=None):
            return len((text or "").split())

    def test_history_over_budget_keeps_calendar_and_thread(self):
        history = [{"content": " ".join(["hist%d" % i] * 50)} for i in range(20)]
        ctx = {
            "recent_conversations": history,
            "session_timeline": [{"content": " ".join(["tl"] * 50)} for _ in range(10)],
            "google_calendar": [
                {"content": "Rivera Hair Studio today 2pm"},
                {"content": "Office hours with Prof. Rivera tomorrow"},
            ],
            "unresolved_threads": [{"content": "follow up on the Rivera thread"}],
        }
        tm = TokenManager(self._MM(), self._Tok(), 300)
        out = tm._manage_token_budget(ctx)
        assert len(out["google_calendar"]) == 2
        assert len(out["unresolved_threads"]) == 1
        assert len(out["recent_conversations"]) < 20


class TestDigestLabelAndWordClip:
    def test_labelled_and_clipped_on_a_word(self):
        c = _controller()
        reply = ("word " * 43) + "Rivera" + " tail words that run past the limit"
        assert reply[219] != " "
        digest = c._compute_recent_conversation_digest({
            "recent_conversations": [{"query": "what is today", "response": reply}],
        })
        line = [l for l in digest.splitlines() if "Daemon" in l and "not a source" in l]
        assert line, digest
        body = line[0].split("Daemon (earlier, not a source): ", 1)[1]
        assert body.endswith("…")
        assert len(body) <= c._DIGEST_MSG_CHARS
        kept = body[:-1]
        assert reply.startswith(kept)
        assert reply[len(kept)] == " ", "clip landed mid-word"

    def test_short_reply_unchanged(self):
        digest = _controller()._compute_recent_conversation_digest({
            "recent_conversations": [{"query": "hi", "response": "Short reply."}],
        })
        assert "Daemon (earlier, not a source): Short reply." in digest
        line = [l for l in digest.splitlines() if "Short reply." in l][0]
        assert "…" not in line


# ---------------------------------------------------------------------------
# H2.1 reuse never fires on evidence the budget emptied
# ---------------------------------------------------------------------------
class _MM:
    def get_active_model_name(self):
        return "test-model"


class _Tok:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


class TestReuseBlockedOnBudgetEmptiedEvidence:
    def _session(self):
        from core.agentic.types import AgenticSearchSession
        return AgenticSearchSession(query="what is on my calendar today")

    def test_calendar_emptied_by_budget_blocks_reuse(self):
        ctx = {
            "recent_conversations": [{"content": "hello there"}],
            # one item larger than the whole budget: the first pass cannot keep it
            "google_calendar": [{"content": "Rivera Hair Studio " + "word " * 150}],
        }
        out = TokenManager(_MM(), _Tok(), 100)._manage_token_budget(ctx)
        assert not out["google_calendar"], "precondition: the budget emptied the calendar"
        assert out["_budget_emptied"] == ["google_calendar"]
        c = _controller()
        assert c._first_unmet_retrieval_key(out, self._session()) == "google_calendar"
        assert c._decision_saw_admitted_evidence(self._session(), out) is False

    def test_genuinely_empty_calendar_does_not_block_reuse(self):
        ctx = {"recent_conversations": [{"content": "hello there"}], "google_calendar": []}
        out = TokenManager(_MM(), _Tok(), 100)._manage_token_budget(ctx)
        assert "_budget_emptied" not in out
        c = _controller()
        assert c._decision_saw_admitted_evidence(self._session(), out) is True

    def test_underscore_key_is_not_metered_or_rendered(self):
        from core.prompt.token_manager import PRIORITY_ORDER
        assert "_budget_emptied" not in {n for n, _ in PRIORITY_ORDER}
        out = TokenManager(_MM(), _Tok(), 100)._manage_token_budget(
            {"google_calendar": [{"content": "x " * 300}]})
        assert out["_budget_emptied"] == ["google_calendar"]

    def test_builder_forwards_the_key(self):
        import inspect
        from core.prompt import builder
        assert '"_budget_emptied"' in inspect.getsource(builder.UnifiedPromptBuilder)


# ---------------------------------------------------------------------------
# H2.3 pass-2: small time-sensitive rows trim LAST among their priority ties
# ---------------------------------------------------------------------------
class TestPassTwoTrimsHistoryBeforeCalendar:
    def test_history_shrinks_before_calendar(self):
        holder = {}

        class _TripTok:
            """Counts words; the lowest-priority sentinel (counted last in the
            first pass) drops the budget so ONLY the second pass trims."""
            def count_tokens(self, text, model_name=None):
                if (text or "").strip() == "SENTINEL":
                    holder["tm"].token_budget = 70
                    return 1
                return len((text or "").split())

        tm = TokenManager(_MM(), _TripTok(), 200)
        holder["tm"] = tm
        ctx = {
            "recent_conversations": [{"content": " ".join(["h%d" % i] * 10)} for i in range(8)],
            "google_calendar": [{"content": "a b c"}, {"content": "d e f"}],
            "unresolved_threads": [{"content": "t u v"}],
            "wiki": "SENTINEL",
        }
        out = tm._manage_token_budget(ctx)
        # one history trim (8 -> 6) is enough to reach the budget: the calendar
        # and thread must still be whole.
        assert len(out["recent_conversations"]) == 6
        assert len(out["google_calendar"]) == 2
        assert len(out["unresolved_threads"]) == 1


# ---------------------------------------------------------------------------
# H2.5 light path resets the per-turn web decision
# ---------------------------------------------------------------------------
class TestLightPathResetsWebDecision:
    async def test_previous_turn_search_not_inherited(self):
        from unittest.mock import AsyncMock
        from core.prompt.builder import UnifiedPromptBuilder
        b = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
        gatherer = MagicMock()
        gatherer._get_recent_conversations = AsyncMock(return_value=[])
        gatherer.memory_id_map = {}
        gatherer.last_web_decision = {
            "triggered": True, "source": "llm", "reason": "search", "confidence": 0.9,
            "results": 3, "error": None,
        }
        b.context_gatherer = gatherer
        b.token_manager = TokenManager(_MM(), _Tok(), 1000)
        ctx = await b._build_lightweight_context("ok cool")
        assert gatherer.last_web_decision["triggered"] is False
        assert gatherer.last_web_decision["results"] is None
        assert gatherer.last_web_decision["source"] == "light_path"
        assert ctx["web_search_decision"]["triggered"] is False
