"""A reused agentic answer must come from a reasoned decision round.

Incident 2026-10-18 (student coursework session): 14 of 20 agentic turns called
no tool and shipped the DECISION round's text as the answer
(``agentic_answer_call=decision_reuse``). That round is a tool-selection router
run with reasoning OFF on purpose, so it produced confidently wrong course
content (reversed Queue/Seize order, an invented "Smallest Number Busy"
definition, a self-contradiction). The enhanced path and the synthesis call run
with reasoning and were accurate.

Class: BC-27 (a shortcut path skips what the primary path does). The model
manager here is a fake that answers every decision call with a long, clean,
confidently wrong text and no tool call; the DEPLOYED loop,
``_get_model_decision`` and ``_generate_with_tools`` run unmodified.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import app_config
from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchProtocol

from tests.unit.test_agentic_decision_answer_reuse import LONG_ANSWER
from tests.unit.test_agentic_controller_decision_evidence import (
    _FrozenDatetime, _NoPending, _mock_tool_executor, _wrap,
)
from core.agentic.tools import ToolExecutor

WRONG_ANSWER = (
    "Rowan's entity only reaches the Queue block if the Seize is busy. "
    "That is how the chain works, and it is the whole story here."
)
SYNTH = "SYNTHESIZED ANSWER FROM THE REASONED CALL."


class _Manager:
    """Fake model manager recording how each decision call was made."""

    api_models = {}

    def __init__(self, text):
        self.text = text
        self.calls = []

    async def generate_once(self, **kw):
        self.calls.append(("generate_once", kw))
        return self.text

    async def generate_once_with_tools(self, **kw):
        self.calls.append(("generate_once_with_tools", kw))
        return SimpleNamespace(content=self.text, tool_calls=None)


def _make(monkeypatch, text, protocol):
    mm = _Manager(text)
    c = AgenticSearchController(model_manager=mm, web_search_manager=MagicMock())
    c._tool_executor = _mock_tool_executor()
    c.detect_protocol = lambda m: protocol
    monkeypatch.setattr("core.agentic.controller.datetime", _FrozenDatetime)
    monkeypatch.setattr(ToolExecutor, "_get_pending_actions_store",
                        staticmethod(lambda: _NoPending()))

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        yield SYNTH

    c._generate_final_response = fake_final
    return c, mm


async def _drive(c):
    out = []
    async for ev in c.run_agentic_search(
        query="Does the entity sit in the Queue before the Seize?",
        system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True, initial_context={},
    ):
        if isinstance(ev, str):
            out.append(ev)
    return "".join(out), c._last_session


@pytest.mark.parametrize("text", [WRONG_ANSWER + " " + LONG_ANSWER,
                                  _wrap(WRONG_ANSWER + " " + LONG_ANSWER)])
class TestDefaultsRouterNeverShips:
    @pytest.mark.asyncio
    async def test_native_default_uses_synthesis_and_records_reason(self, monkeypatch, text):
        assert app_config.AGENTIC_DECISION_REASONING is False
        c, mm = _make(monkeypatch, text, SearchProtocol.NATIVE_TOOLS)
        shown, sess = await _drive(c)
        assert shown == SYNTH
        assert sess.answer_call == "final_synthesis"
        assert sess.decision_reasoned is False
        assert sess.reuse_skipped_reason == "decision round ran without reasoning"
        name, kw = mm.calls[0]
        assert name == "generate_once_with_tools"
        assert kw["disable_reasoning"] is True
        assert kw["max_tokens"] == app_config.AGENTIC_DECISION_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_xml_default_uses_synthesis(self, monkeypatch, text):
        c, mm = _make(monkeypatch, text, SearchProtocol.XML_MARKERS)
        shown, sess = await _drive(c)
        assert shown == SYNTH
        assert sess.reuse_skipped_reason == "decision round ran without reasoning"
        assert mm.calls[0][1]["disable_reasoning"] is True


class TestReasonedDecisionRound:
    @pytest.mark.asyncio
    async def test_native_opt_in_reasons_and_reuses_same_instance(self, monkeypatch):
        text = WRONG_ANSWER + " " + LONG_ANSWER
        c, mm = _make(monkeypatch, text, SearchProtocol.NATIVE_TOOLS)
        # same controller instance, flipped live (BC-82)
        shown, sess = await _drive(c)
        assert shown == SYNTH
        monkeypatch.setattr(app_config, "AGENTIC_DECISION_REASONING", True)
        monkeypatch.setattr(app_config, "AGENTIC_DECISION_REASONING_MAX_TOKENS", 4321)
        shown2, sess2 = await _drive(c)
        assert shown2 == text.strip()
        assert sess2.answer_call == "decision_reuse"
        assert sess2.decision_reasoned is True
        _, kw = mm.calls[-1]
        assert kw["disable_reasoning"] is False
        assert kw["max_tokens"] == 4321

    @pytest.mark.asyncio
    async def test_xml_protocol_stays_unreasoned_even_when_enabled(self, monkeypatch):
        monkeypatch.setattr(app_config, "AGENTIC_DECISION_REASONING", True)
        c, mm = _make(monkeypatch, WRONG_ANSWER + " " + LONG_ANSWER, SearchProtocol.XML_MARKERS)
        shown, sess = await _drive(c)
        assert shown == SYNTH
        assert sess.decision_reasoned is False
        assert mm.calls[0][1]["disable_reasoning"] is True
        assert mm.calls[0][1]["max_tokens"] == app_config.AGENTIC_DECISION_MAX_TOKENS


def test_config_keys_are_declared():
    from config.schema import AgenticSearchSection
    sec = AgenticSearchSection()
    assert sec.decision_reasoning is False
    assert sec.decision_reasoning_max_tokens == 6000
