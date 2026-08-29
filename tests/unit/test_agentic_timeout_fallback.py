"""Deterministic tool dispatch on agentic decision-round timeout (2026-08-27).

Incident (drop-date thread, turn 3): the user explicitly asked "Can we do a
web search and attempt to confirm the specific drop date". The Tier-1 gate
routed to the tool loop, the 75s decision-round timeout fired on a busy model,
and the timeout handler returned wants_answer=True — the loop ended with ZERO
tools dispatched and spent ~280s synthesizing an answer it had no evidence
for (370s total turn).

Fix: SearchDecision.timed_out marks the timeout fallback decision; when it
arrives with nothing gathered on a tool-triggered session, the loop
substitutes deterministic web-search decisions from the trigger's own seed
terms (or the query itself), once. A SECOND timeout falls through to
synthesis as before.
"""
import pytest
from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision


LIVE_QUERY = "Can we do a web search and attempt to confirm the specific drop date"
LIVE_TERMS = ["college drop date August 2026", "school withdrawal deadline August 2026"]


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    web = MagicMock()
    web.is_available = MagicMock(return_value=True)
    return AgenticSearchController(model_manager=manager, web_search_manager=web)


def _timeout_decision():
    return [SearchDecision(wants_answer=True, timed_out=True)]


def _make_final_spy(captured):
    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["final_called"] = True
        yield "SYNTHESIZED ANSWER."
    return fake_final


def _make_dispatch_spy(captured):
    class _Res:
        start_events = []
        end_events = []
        round_data = MagicMock(duration_ms=1.0)
        formatted_context = "WEB: search results."
        memory_collection = None
        is_expand = False
        decision = SearchDecision()

    async def fake_dispatch(decision, *a, **k):
        captured.setdefault("dispatched", []).append(decision)
        return _Res()

    return fake_dispatch


async def _run(controller, query=LIVE_QUERY, terms=None):
    events = []
    async for ev in controller.run_agentic_search(
        query=query, system_prompt="sys", model_name="test-model",
        initial_search_terms=terms if terms is not None else [],
        skip_initial_search=True,
    ):
        events.append(ev)
    return events


# ===========================================================================
# Fallback substitution
# ===========================================================================

class TestTimeoutToolFallback:

    @pytest.mark.asyncio
    async def test_timeout_with_nothing_gathered_dispatches_seed_terms(
        self, controller, monkeypatch
    ):
        """Live-turn reproduction: decision timeout + zero rounds + seed terms
        available → the loop dispatches the trigger's seed searches instead of
        answering from context."""
        captured = {"final_called": False}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _timeout_decision()
            return [SearchDecision(is_done=True, done_reason="enough")]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        dispatched = captured.get("dispatched", [])
        assert len(dispatched) == 2
        assert all(d.wants_search for d in dispatched)
        assert [d.search_query for d in dispatched] == LIVE_TERMS
        assert all("timeout" in (d.search_reason or "") for d in dispatched)
        # The turn still synthesizes — now WITH gathered evidence.
        assert captured["final_called"] is True

    @pytest.mark.asyncio
    async def test_seed_terms_capped_at_two(self, controller, monkeypatch):
        captured = {}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _timeout_decision()
            return [SearchDecision(is_done=True)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(
            controller, "_generate_final_response", _make_final_spy(captured)
        )

        await _run(controller, terms=["a one", "b two", "c three", "d four"])

        assert [d.search_query for d in captured["dispatched"]] == ["a one", "b two"]

    @pytest.mark.asyncio
    async def test_no_seed_terms_derives_from_query(self, controller, monkeypatch):
        """The trigger sometimes routes without distilled terms — the fallback
        strips the search-request preamble from the raw query."""
        captured = {}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _timeout_decision()
            return [SearchDecision(is_done=True)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(
            controller, "_generate_final_response", _make_final_spy(captured)
        )

        await _run(controller, query=LIVE_QUERY, terms=[])

        dispatched = captured.get("dispatched", [])
        assert len(dispatched) == 1
        q = dispatched[0].search_query
        assert "drop date" in q
        assert "web search" not in q.lower()

    @pytest.mark.asyncio
    async def test_second_timeout_falls_through_to_synthesis(
        self, controller, monkeypatch
    ):
        """One-shot: after the fallback round, another timeout must end the
        loop (answer with gathered context), not loop forever."""
        captured = {"final_called": False}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            return _timeout_decision()  # times out EVERY round

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        # Fallback fired once (2 searches), then the second timeout ended the loop.
        assert len(captured.get("dispatched", [])) == 2
        assert calls["n"] == 2
        assert captured["final_called"] is True


# ===========================================================================
# Fallback must NOT fire
# ===========================================================================

class TestFallbackGuards:

    @pytest.mark.asyncio
    async def test_no_fallback_when_tools_already_ran(self, controller, monkeypatch):
        """A timeout AFTER a real tool round answers from gathered context —
        that is the original, correct backstop behavior."""
        captured = {"final_called": False}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return [SearchDecision(
                    wants_memory_search=True,
                    memory_query="x", memory_collection="conversations",
                )]
            return _timeout_decision()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        dispatched = captured.get("dispatched", [])
        # Only the round-1 memory search — no substituted web searches.
        assert len(dispatched) == 1
        assert dispatched[0].wants_memory_search
        assert not any(d.wants_search for d in dispatched)
        assert captured["final_called"] is True

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, controller, monkeypatch):
        monkeypatch.setattr(
            "config.app_config.AGENTIC_TIMEOUT_TOOL_FALLBACK", False
        )
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return _timeout_decision()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        assert captured.get("dispatched", []) == []
        assert captured["final_called"] is True

    @pytest.mark.asyncio
    async def test_no_fallback_without_web_search(self, monkeypatch):
        manager = MagicMock()
        manager.api_models = {}
        controller = AgenticSearchController(
            model_manager=manager, web_search_manager=None
        )
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return _timeout_decision()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        assert captured.get("dispatched", []) == []
        assert captured["final_called"] is True

    @pytest.mark.asyncio
    async def test_genuine_ready_to_answer_untouched(self, controller, monkeypatch):
        """A wants_answer WITHOUT timed_out is the model's own signal — the
        fallback must never hijack it."""
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return [SearchDecision(wants_answer=True)]  # timed_out=False

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _make_dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        await _run(controller, terms=LIVE_TERMS)

        assert captured.get("dispatched", []) == []
        assert captured["final_called"] is True


# ===========================================================================
# Timeout handler marks the decision
# ===========================================================================

class TestTimeoutMarking:

    @pytest.mark.asyncio
    async def test_timeout_decision_carries_timed_out_flag(self, controller, monkeypatch):
        """_get_model_decision's wait_for backstop must mark its fallback
        decision so the loop can tell it from a real ready-to-answer."""
        import asyncio

        monkeypatch.setattr("config.app_config.AGENTIC_ROUND_TIMEOUT_S", 0.05)

        async def never_returns(*a, **k):
            await asyncio.sleep(5.0)

        monkeypatch.setattr(controller, "_generate_decision_no_reasoning", never_returns)

        handler = MagicMock()
        session = MagicMock()
        session.protocol = "xml_markers"

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m",
            handler=handler, session=session,
        )

        assert decisions[0].wants_answer is True
        assert decisions[0].timed_out is True

    def test_default_decision_not_timed_out(self):
        assert SearchDecision(wants_answer=True).timed_out is False


# ===========================================================================
# Query-derived fallback terms
# ===========================================================================

class TestFallbackTermsFromQuery:

    def test_live_query_preamble_stripped(self):
        out = AgenticSearchController._fallback_terms_from_query(LIVE_QUERY)
        assert "drop date" in out
        assert "web search" not in out.lower()

    def test_plain_query_unchanged(self):
        q = "Georgia Tech fall 2026 withdrawal deadline"
        assert AgenticSearchController._fallback_terms_from_query(q) == q

    def test_short_remainder_falls_back_to_query(self):
        # Stripping would leave <2 words — keep the original query.
        out = AgenticSearchController._fallback_terms_from_query(
            "Can you do a web search?"
        )
        assert out  # never empty

    def test_empty_query_safe(self):
        assert AgenticSearchController._fallback_terms_from_query("") == ""
