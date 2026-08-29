"""Decision-answer reuse: the agentic loop must not pay a second full-context
synthesis call when the decision round already produced the final answer.

Incident (2026-07-15): a 32s decision call returned answer text with no tool
calls ("ready to answer implicit"); the text was discarded and a second 24s
call re-generated essentially the same answer. The controller now vets the
decision-round text (_usable_decision_answer) and, when it is a complete
substantive answer, yields it directly — _generate_final_response is skipped.
"""
import pytest
from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision


# A substantive, complete answer: >200 chars, terminal punctuation, no tool
# vocabulary (which would trip the round-1 narration nudge).
LONG_ANSWER = (
    "They really are among the roughest substances to develop a dependence on. "
    "Withdrawal can be medically dangerous — seizures and delirium are possible — "
    "which is why tapering plans exist and why short courses are the standard "
    "recommendation. The relief they provide is real, and that is exactly what "
    "makes them so difficult to stop."
)


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


def _make_final_spy(captured):
    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["final_called"] = True
        captured["session"] = session
        yield "SYNTHESIZED ANSWER."
    return fake_final


async def _run(controller, out=None):
    events = []
    async for ev in controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True,
    ):
        events.append(ev)
    if out is not None:
        out.extend(events)
    return "".join(c for c in events if isinstance(c, str))


# ===========================================================================
# _usable_decision_answer (vetting helper)
# ===========================================================================

class TestUsableDecisionAnswer:

    def test_substantive_answer_passes(self, controller):
        assert controller._usable_decision_answer(LONG_ANSWER) == LONG_ANSWER

    def test_short_answer_rejected(self, controller):
        assert controller._usable_decision_answer("Yes, that's right.") is None

    def test_empty_and_none_rejected(self, controller):
        assert controller._usable_decision_answer("") is None
        assert controller._usable_decision_answer(None) is None

    def test_truncated_answer_rejected(self, controller):
        # No terminal punctuation → treated as a capped generation.
        truncated = LONG_ANSWER.rstrip(".") + " and the other thing is"
        assert controller._usable_decision_answer(truncated) is None

    def test_promissory_narration_rejected(self, controller):
        narration = (
            "Let me check the available information on this topic first. " + LONG_ANSWER
        )
        assert controller._usable_decision_answer(narration) is None

    # --- 2026-08-28 regression: the live turn-8 narration shipped as the
    # final response. It defeated the old guard twice: the promissory check
    # scanned only the first 150 chars ("Let me aim…" sat just past the
    # window) AND 'aim' wasn't in the substring verb list.

    LIVE_NARRATION = (
        "The first round of results missed the mark — they're about government "
        "disclosure and privacy regulations, not patient-to-doctor disclosure "
        "behavior. Let me aim at the actual research on patients withholding "
        "or delaying disclosure of sensitive symptoms."
    )

    def test_live_midloop_narration_rejected(self, controller):
        assert controller._usable_decision_answer(self.LIVE_NARRATION) is None

    def test_promissory_past_head_window_rejected(self, controller):
        # Plan phrasing ANYWHERE in the text rejects — not just the opener.
        tail_plan = LONG_ANSWER + " I'll search for the specific numbers next."
        assert controller._usable_decision_answer(tail_plan) is None

    def test_loop_meta_opener_rejected(self, controller):
        meta = (
            "That round of results didn't return anything on-topic at all. "
            + LONG_ANSWER
        )
        assert controller._usable_decision_answer(meta) is None

    def test_let_me_know_closing_still_passes(self, controller):
        # Conversational closers are NOT tool-intent plans.
        friendly = LONG_ANSWER + " Let me know if you want the citations."
        assert controller._usable_decision_answer(friendly) == friendly

    def test_results_mentioned_in_body_still_passes(self, controller):
        # "results" deep in a real answer must not trip the head-anchored
        # loop-meta check.
        body = LONG_ANSWER + " The survey results support this interpretation overall."
        assert controller._usable_decision_answer(body) == body

    def test_reasoning_block_only_rejected(self, controller):
        # sanitize_for_storage strips the tagged block; nothing substantive remains.
        leak = f"<reasoning>{LONG_ANSWER}</reasoning>"
        assert controller._usable_decision_answer(leak) is None

    def test_reasoning_prefix_stripped_then_reused(self, controller):
        mixed = f"<thinking>internal deliberation here</thinking>\n{LONG_ANSWER}"
        result = controller._usable_decision_answer(mixed)
        assert result is not None
        assert "internal deliberation" not in result
        assert result.endswith("difficult to stop.")


# ===========================================================================
# Loop integration: reuse vs fallback
# ===========================================================================

class TestDecisionAnswerReuseInLoop:

    @pytest.mark.asyncio
    async def test_implicit_answer_reused_skips_final_call(self, controller, monkeypatch):
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return [SearchDecision(wants_answer=True, partial_response=LONG_ANSWER)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)
        assert LONG_ANSWER in text
        assert captured["final_called"] is False

    @pytest.mark.asyncio
    async def test_done_with_answer_text_reused(self, controller, monkeypatch):
        captured = {"final_called": False}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return [SearchDecision(
                    wants_memory_search=True,
                    memory_query="x", memory_collection="conversations",
                )]
            return [SearchDecision(is_done=True, partial_response=LONG_ANSWER)]

        class _Res:
            start_events = []
            end_events = []
            round_data = MagicMock(duration_ms=1.0)
            formatted_context = "MEMORY: relevant context."
            memory_collection = "conversations"
            is_expand = False
            decision = SearchDecision()

        async def fake_dispatch(*a, **k):
            return _Res()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)
        assert LONG_ANSWER in text
        assert captured["final_called"] is False

    @pytest.mark.asyncio
    async def test_short_answer_falls_back_to_synthesis(self, controller, monkeypatch):
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return [SearchDecision(wants_answer=True, partial_response="Yes.")]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)
        assert captured["final_called"] is True
        assert "SYNTHESIZED ANSWER." in text

    @pytest.mark.asyncio
    async def test_reuse_disabled_by_config_falls_back(self, controller, monkeypatch):
        captured = {"final_called": False}
        monkeypatch.setattr(
            "config.app_config.AGENTIC_REUSE_DECISION_ANSWER", False
        )

        async def fake_decision(*a, **k):
            return [SearchDecision(wants_answer=True, partial_response=LONG_ANSWER)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)
        assert captured["final_called"] is True
        assert "SYNTHESIZED ANSWER." in text

    @pytest.mark.asyncio
    async def test_action_dispatch_same_round_blocks_reuse(self, controller, monkeypatch):
        """An action dispatched in the closing round means the answer text was
        written BEFORE the action result existed — it must not be reused."""
        captured = {"final_called": False}

        async def fake_decision(*a, **k):
            return [
                SearchDecision(
                    wants_action=True, action_type="send_email",
                    action_params={"to": "x@example.com"},
                ),
                SearchDecision(is_done=True, partial_response=LONG_ANSWER),
            ]

        class _Res:
            start_events = []
            end_events = []
            round_data = MagicMock(duration_ms=1.0)
            formatted_context = "ACTION: proposed."
            memory_collection = None
            is_expand = False
            decision = SearchDecision()

        async def fake_dispatch(*a, **k):
            return _Res()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)
        assert captured["final_called"] is True
        assert "SYNTHESIZED ANSWER." in text
