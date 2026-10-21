"""Owner [test] probes never teach the adaptive exemplar store.

Incident 2026-10-18, class BC-58: ``Learned web_search/search_worthy exemplar
(source=citation...): "[test]Search the web for ..."`` -- record() had no probe
check. Callers that run AFTER the ingress unwraps the envelope (intent, tone,
need, gate, web trigger) hand record() the INNER text, so the whole turn is
marked at gui.handlers.handle_submit and record() refuses while it is marked.
"""

import asyncio

import pytest

import utils.adaptive_exemplars as ae

PROBE = "[test]Search the web for the weather forecast in Springfield this weekend[/test]"
INNER = "Search the web for the weather forecast in Springfield this weekend"
WRAPPED = "[test]Search the web for the weather\nforecast in Springfield this weekend[/test]"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = ae.AdaptiveExemplarStore(str(tmp_path / "ex.json"))
    monkeypatch.setattr(ae, "_store", s)
    return s


@pytest.mark.parametrize("text", [PROBE, WRAPPED])
def test_record_refuses_enveloped_text(store, text):
    assert store.record("web_search", "search_worthy", text, "citation") is False
    assert store.get_learned("web_search", "search_worthy") == []


def test_record_still_learns_ordinary_text(store):
    assert store.record("web_search", "search_worthy", INNER, "citation") is True


def test_record_refuses_stripped_text_on_a_marked_turn(store):
    token = ae.mark_probe_turn(True)
    try:
        assert store.record("web_search", "search_worthy", INNER, "citation") is False
    finally:
        ae.end_probe_turn(token)
    assert store.record("web_search", "search_worthy", INNER, "citation") is True


def test_citation_hook_teaches_nothing_from_a_probe(store):
    from core.orchestrator import PostResponseHookContext, _hook_search_worthy_teach

    class _Orch:
        _last_turn_signals = {"tone_level": "CONVERSATIONAL"}

    for text in (PROBE, WRAPPED):
        _hook_search_worthy_teach(PostResponseHookContext(
            orchestrator=_Orch(), user_input=text, response_text="See [WEB_1] for it."))
    assert store.get_learned("web_search", "search_worthy") == []
    _hook_search_worthy_teach(PostResponseHookContext(
        orchestrator=_Orch(), user_input=INNER, response_text="See [WEB_1] for it."))
    assert store.get_learned("web_search", "search_worthy") == [INNER]


def test_intent_teacher_with_stripped_text_is_refused_on_probe_turn(store):
    from core.intent_classifier import INTENT_EXEMPLARS, _learn_intent_exemplar
    label = next(iter(INTENT_EXEMPLARS))
    phrase = "what did we decide about the Tamsin budget spreadsheet last Tuesday"
    token = ae.mark_probe_turn(True)
    try:
        _learn_intent_exemplar(phrase, label, "regex")
    finally:
        ae.end_probe_turn(token)
    assert store.get_learned("intent", label) == []
    _learn_intent_exemplar(phrase, label, "regex")
    assert store.get_learned("intent", label) == [phrase]


def test_handle_submit_marks_probe_turns_and_restores(monkeypatch):
    import gui.handlers as h
    seen = []

    async def _inner(user_text, *a, **k):
        seen.append(ae._PROBE_TURN.get())
        yield {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(h, "_handle_submit_inner", _inner)

    async def _drive(text):
        async for _ in h.handle_submit(text, [], [], False, object()):
            pass

    asyncio.run(_drive(PROBE))
    asyncio.run(_drive("Rowan asked whether the library is open on Sundays"))
    assert seen == [True, False]
    assert ae._PROBE_TURN.get() is False
