"""Behavioural sibling for ``TestFloorChainBudget.test_chain_counter_source_wiring``
in tests/unit/test_tone_floor_self_latch.py (BC-63: it grepped the source of
``ContextPipeline._detect_tone`` for the names ``TONE_FLOOR_CHAIN_MAX`` /
``_floor_chain`` and passed whether or not the budget worked).

Incident date: 2026-08-22 (distress-sticky-floor self-latch). Bug classes:
BC-63, BC-62. THE deployed ``_detect_tone`` is driven with a fake
``analyze_emotional_context`` that reports ``distress_sticky_floor`` every
turn; after TONE_FLOOR_CHAIN_MAX consecutive floor turns the pipeline must
withhold ``previous_tone`` and disable the floor stage, and an organic turn
must reset the counter.
"""
from types import SimpleNamespace

import pytest

from core.context_pipeline import ContextPipeline
from utils.tone_detector import CrisisLevel

QUERY = "ok sounds good"
WRAPPED = "    ok sounds\n    good\n   "


def _pipeline(tmp_path):
    pipe = object.__new__(ContextPipeline)
    pipe.memory_system = None
    pipe.model_manager = None
    pipe._last_tone_level = None
    pipe._floor_chain = 0
    pipe._TONE_STATE_PATH = str(tmp_path / "tone_state.json")
    pipe._should_reset_tone_stickiness = lambda recent: False
    return pipe


def _install_fake(monkeypatch, triggers):
    calls = []
    seq = iter(triggers)

    async def fake(message, conversation_history=None, model_manager=None,
                   previous_tone=None, allow_sticky_floor=True, **kw):
        calls.append({"previous_tone": previous_tone,
                      "allow_sticky_floor": allow_sticky_floor})
        trig = next(seq)
        return SimpleNamespace(crisis_level=CrisisLevel.CONCERN, tone_trigger=trig)

    import utils.emotional_context as ec
    monkeypatch.setattr(ec, "analyze_emotional_context", fake)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [QUERY, WRAPPED])
async def test_floor_chain_is_budgeted(monkeypatch, tmp_path, query):
    monkeypatch.setenv("TONE_FLOOR_CHAIN_MAX", "3")
    pipe = _pipeline(tmp_path)
    calls = _install_fake(monkeypatch, ["distress_sticky_floor"] * 5)
    for _ in range(5):
        await pipe._detect_tone(query)
    # turn 1 starts cold; turns 2-3 carry the floor's own output; the 4th turn
    # has a chain of 3, so previous_tone is withheld AND the floor stage is off.
    assert [c["allow_sticky_floor"] for c in calls[:3]] == [True, True, True]
    assert calls[1]["previous_tone"] is not None
    assert calls[3]["allow_sticky_floor"] is False
    assert calls[3]["previous_tone"] is None


@pytest.mark.asyncio
async def test_organic_turn_resets_the_chain(monkeypatch, tmp_path):
    monkeypatch.setenv("TONE_FLOOR_CHAIN_MAX", "3")
    pipe = _pipeline(tmp_path)
    calls = _install_fake(monkeypatch, [
        "distress_sticky_floor", "distress_sticky_floor", "semantic",
        "distress_sticky_floor", "distress_sticky_floor",
    ])
    for _ in range(5):
        await pipe._detect_tone(QUERY)
    assert pipe._floor_chain == 2
    assert all(c["allow_sticky_floor"] for c in calls)


@pytest.mark.asyncio
async def test_budget_follows_env_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("TONE_FLOOR_CHAIN_MAX", "1")
    pipe = _pipeline(tmp_path)
    calls = _install_fake(monkeypatch, ["distress_sticky_floor"] * 3)
    for _ in range(3):
        await pipe._detect_tone(QUERY)
    assert calls[1]["allow_sticky_floor"] is False
