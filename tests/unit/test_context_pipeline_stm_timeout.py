"""STM analysis cap is config (stm.timeout_s), read live, with a clean fallback.

Incident: oct27 wave 2 lane L1 (M-5), 2026-10-27. Class: BC-69 (unbounded /
hard-coded wait on the critical path), BC-82 (live config).

STM (p50 2 s, p95 3 s, max 10 s over 229 live calls) runs inside the
context-pipeline gather, so a stalled analyzer held the whole turn for the
hard-coded 10 s.  Drives THE deployed ``ContextPipeline.build``; only the
unrelated stages and the analyzer itself are fakes.  The fake analyzer is a
misbehaving model (sleeps far past the cap).
"""
import asyncio
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

import config.app_config as ac
from core.context_pipeline import ContextPipeline, ToneLevel

QUERY = ("please explain in detail how the memory gate scores its candidate "
         "documents before the reranker runs today?")
WRAPPED = "   " + QUERY.replace(" how ", "\n   how ") + "\n"


def _pipeline(monkeypatch, analyze_stm):
    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        cp = ContextPipeline(model_manager=None, topic_manager=None,
                             config={"enable_query_rewrite": False})

    async def topics(user_input, last_exchange=None):
        return "general", ["general"]

    async def tone(user_input, history):
        return ToneLevel.CONVERSATIONAL, None

    async def thread():
        return None

    async def heavy(user_input, topics_):
        return False, [], NS(is_question=True, is_command=False, token_count=20)

    monkeypatch.setattr(cp, "_extract_topics", topics)
    monkeypatch.setattr(cp, "_detect_tone", tone)
    monkeypatch.setattr(cp, "_get_identity_context", lambda: ("", None))
    monkeypatch.setattr(cp, "_get_thread_context", thread)
    monkeypatch.setattr(cp, "_get_tone_instructions", lambda level: "")
    monkeypatch.setattr(cp, "_intent_classifier", None)
    monkeypatch.setattr(cp, "_check_heavy_topics", heavy)
    monkeypatch.setattr(cp, "_analyze_stm", analyze_stm)
    monkeypatch.setattr(cp, "_should_run_stm", lambda history: True)
    return cp


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
def test_stalled_stm_falls_back_within_the_configured_cap(monkeypatch, text):
    async def stalled(user_input, history):
        await asyncio.sleep(30)
        return {"intent": "x", "topic": "t"}

    monkeypatch.setattr(ac, "STM_TIMEOUT_S", 0.2)
    cp = _pipeline(monkeypatch, stalled)
    t0 = time.perf_counter()
    result = asyncio.run(cp.build(text))
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, elapsed          # base: waits the hard-coded 10 s
    assert result.stm_summary is None      # deterministic fallback: no STM this turn


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
def test_fast_stm_still_lands_and_cap_is_read_live(monkeypatch, text):
    async def quick(user_input, history):
        await asyncio.sleep(0.05)
        return {"intent": "x", "topic": "t"}

    monkeypatch.setattr(ac, "STM_TIMEOUT_S", 0.2)
    cp = _pipeline(monkeypatch, quick)
    assert asyncio.run(cp.build(text)).stm_summary == {"intent": "x", "topic": "t"}
    # same instance, cap lowered below the analyzer's time -> now falls back
    monkeypatch.setattr(ac, "STM_TIMEOUT_S", 0.01)
    assert asyncio.run(cp.build(text)).stm_summary is None


def test_shipped_default_is_five_seconds():
    from config.app_config import STM_TIMEOUT_S
    assert STM_TIMEOUT_S == 5.0
