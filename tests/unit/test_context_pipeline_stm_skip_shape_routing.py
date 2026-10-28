"""Behavioural sibling: the context pipeline's STM stage honours ``stm_skip_shape`` (source-text row).

Incident date: 2026-09-03 ("Hey" after a 2 h gap ran STM, which summarized the PREVIOUS
turn as a restatement and injected the recall warning onto the greeting).
Class: BC-63 (a grep for ``stm_skip_shape(user_input`` stands in for the routing),
BC-58 (stale prior-turn state surfacing on an unrelated turn).

Drives THE deployed ``ContextPipeline.build`` (same harness as
test_context_pipeline_stm_timeout) with a recording ``_analyze_stm``: a bare greeting
or acknowledgement never reaches the analyzer, a substantive message does, and the
decision follows the live message shape on the SAME pipeline instance.
"""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from core.context_pipeline import ContextPipeline, ToneLevel


def _pipeline(monkeypatch):
    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        cp = ContextPipeline(model_manager=None, topic_manager=None,
                             config={"enable_query_rewrite": False})
    calls = []

    async def analyze(user_input, history):
        calls.append(user_input)
        return {"intent": "x", "topic": "t"}

    async def topics(user_input, last_exchange=None):
        return "general", ["general"]

    async def tone(user_input, history):
        return ToneLevel.CONVERSATIONAL, None

    async def thread():
        return None

    async def heavy(user_input, topics_):
        return False, [], NS(is_question=False, is_command=False, token_count=5)

    monkeypatch.setattr(cp, "_extract_topics", topics)
    monkeypatch.setattr(cp, "_detect_tone", tone)
    monkeypatch.setattr(cp, "_get_identity_context", lambda: ("", None))
    monkeypatch.setattr(cp, "_get_thread_context", thread)
    monkeypatch.setattr(cp, "_get_tone_instructions", lambda level: "")
    monkeypatch.setattr(cp, "_intent_classifier", None)
    monkeypatch.setattr(cp, "_check_heavy_topics", heavy)
    monkeypatch.setattr(cp, "_analyze_stm", analyze)
    monkeypatch.setattr(cp, "_should_run_stm", lambda history: True)
    return cp, calls


@pytest.mark.parametrize("text", [
    "Hey", "hey!", "Good morning", "ok cool", "thanks",
    "   Good\n   morning  ",
])
def test_greeting_or_acknowledgement_never_reaches_the_stm_analyzer(monkeypatch, text):
    cp, calls = _pipeline(monkeypatch)
    result = asyncio.run(cp.build(text))
    assert calls == []
    assert result.stm_summary is None


@pytest.mark.parametrize("text", [
    "Biscuit figured out fetch today, sort of.",
    "Hey so the doctor never responded and my dad is going the fraud route",
    "   Hey so the doctor never responded\n   and my dad is going the fraud route\n",
])
def test_substantive_message_reaches_the_stm_analyzer(monkeypatch, text):
    cp, calls = _pipeline(monkeypatch)
    result = asyncio.run(cp.build(text))
    assert len(calls) == 1
    assert result.stm_summary == {"intent": "x", "topic": "t"}


def test_the_decision_follows_each_messages_shape_on_one_pipeline(monkeypatch):
    cp, calls = _pipeline(monkeypatch)
    asyncio.run(cp.build("Hey"))
    asyncio.run(cp.build("Biscuit figured out fetch today, sort of."))
    asyncio.run(cp.build("thanks"))
    assert calls == ["Biscuit figured out fetch today, sort of."]
