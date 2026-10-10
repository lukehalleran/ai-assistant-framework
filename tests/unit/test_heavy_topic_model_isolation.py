"""B0: the heavy-topic LLM check names its model per call.

It used to switch the SHARED active model for the duration of the call and
restore it afterwards, so any concurrent generate_once() with no model_name
ran on the classifier model (and a restore could clobber a user switch).
These tests drive the REAL ``_classify_heavy_topic_llm`` against a fake
ModelManager with a two-event barrier (deterministic, no scheduler luck).

class: BC-30, BC-89
"""
import asyncio
import logging

import pytest

import utils.query_checker as qc

SYNTHETIC = "a long enough synthetic message about nothing heavy at all"


class FakeMM:
    """Fake ModelManager. ``active_model_name`` assignments are logged."""

    def __init__(self, api_models=None, active="active-chat"):
        self.api_models = (
            {qc.HEAVY_TOPIC_MODEL: "openai/" + qc.HEAVY_TOPIC_MODEL,
             "active-chat": "x/active-chat"}
            if api_models is None else api_models
        )
        self._active = active
        self.setter_log = []
        self.switch_calls = []
        self.switched = asyncio.Event()
        self.release = asyncio.Event()
        self.seen = []        # (kind, model used on entry)
        self.heavy_kwargs = None
        self.calls = 0

    @property
    def active_model_name(self):
        return self._active

    @active_model_name.setter
    def active_model_name(self, value):
        self.setter_log.append(value)
        self._active = value

    def get_active_model_name(self):
        return self._active

    def switch_model(self, name):
        self.switch_calls.append(name)
        self.active_model_name = name

    async def generate_once(self, prompt, model_name=None, **kw):
        self.calls += 1
        used = model_name or self._active
        if prompt != "x":  # the heavy-topic call
            self.seen.append(("heavy", used))
            self.heavy_kwargs = dict(kw, model_name=model_name)
            self.switched.set()
            await self.release.wait()
            return "NORMAL"
        self.seen.append(("unrelated", used))
        return "ok"


async def _one_run():
    mm = FakeMM()
    heavy = asyncio.create_task(qc._classify_heavy_topic_llm(SYNTHETIC, mm))
    await asyncio.wait_for(mm.switched.wait(), 2)
    await asyncio.wait_for(mm.generate_once("x"), 2)  # unrelated, no model_name
    mm.release.set()
    result = await asyncio.wait_for(heavy, 2)
    return mm, result


@pytest.mark.parametrize("rep", range(20))
def test_concurrent_unrelated_call_keeps_active_model(rep):
    mm, result = asyncio.run(_one_run())
    seen = dict(mm.seen)
    assert seen["unrelated"] == "active-chat"
    assert seen["heavy"] == qc.HEAVY_TOPIC_MODEL
    assert mm.setter_log == []
    assert mm.switch_calls == []
    assert mm.active_model_name == "active-chat"
    assert result is False


def test_heavy_call_disables_reasoning_and_names_model():
    """BC-89: small max_tokens budget on a possibly-reasoning model."""
    mm, _ = asyncio.run(_one_run())
    assert mm.heavy_kwargs["disable_reasoning"] is True
    assert mm.heavy_kwargs["model_name"] == qc.HEAVY_TOPIC_MODEL
    assert mm.heavy_kwargs["max_tokens"] == qc.HEAVY_TOPIC_MAX_TOKENS


# ---------------------------------------------------------------- resolver

def _resolve(monkeypatch, mm, configured):
    monkeypatch.setattr(qc, "HEAVY_TOPIC_MODEL", configured)
    return qc._resolve_heavy_topic_model(mm)


def test_resolver_alias_returns_itself(monkeypatch):
    mm = FakeMM(api_models={"gpt-4o-mini": "openai/gpt-4o-mini"})
    assert _resolve(monkeypatch, mm, "gpt-4o-mini") == "gpt-4o-mini"


def test_resolver_full_slug_returns_alias(monkeypatch):
    mm = FakeMM(api_models={"gpt-4o-mini": "openai/gpt-4o-mini"})
    assert _resolve(monkeypatch, mm, "openai/gpt-4o-mini") == "gpt-4o-mini"


def test_resolver_full_slug_picks_first_sorted_key(monkeypatch):
    mm = FakeMM(api_models={"zeta": "openai/m", "alpha": "openai/m"})
    assert _resolve(monkeypatch, mm, "openai/m") == "alpha"


def test_resolver_unknown_returns_none_and_warns_once(monkeypatch, caplog):
    mm = FakeMM(api_models={"gpt-4o-mini": "openai/gpt-4o-mini"})
    with caplog.at_level(logging.WARNING):
        assert _resolve(monkeypatch, mm, "no-such-model") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no-such-model" in warnings[0].getMessage()


def test_resolver_manager_without_api_models_returns_none(monkeypatch):
    class Bare:
        async def generate_once(self, *a, **k):
            return "NORMAL"
    assert _resolve(monkeypatch, Bare(), "gpt-4o-mini") is None


# ----------------------------------------------- unresolved -> heuristic

def test_unresolved_model_raises_and_never_uses_active_model(monkeypatch):
    monkeypatch.setattr(qc, "HEAVY_TOPIC_MODEL", "no-such-model")
    mm = FakeMM(api_models={"active-chat": "x/active-chat"})

    async def go():
        with pytest.raises(qc.HeavyTopicModelUnavailable):
            await asyncio.wait_for(qc._classify_heavy_topic_llm(SYNTHETIC, mm), 2)

    asyncio.run(go())
    assert mm.calls == 0
    assert mm.setter_log == []


def test_unresolved_model_analyze_query_async_keeps_heuristic(monkeypatch):
    monkeypatch.setattr(qc, "HEAVY_TOPIC_MODEL", "no-such-model")
    mm = FakeMM(api_models={"active-chat": "x/active-chat"})
    expected = qc.analyze_query(SYNTHETIC, model_manager=None)
    got = asyncio.run(asyncio.wait_for(qc.analyze_query_async(SYNTHETIC, mm), 2))
    assert mm.calls == 0
    assert got == expected
