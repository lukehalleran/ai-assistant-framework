"""STM fidelity fallback (2026-10-08, class: BC-08, BC-46).

The summary LLM sometimes describes the PREVIOUS exchange instead of the
current message. The deterministic check in STMAnalyzer.analyze replaces such
a summary's user_question with a verbatim quote. Drives THE deployed analyzer
with a stub model that echoes the prior turn. Test text is synthetic.
"""
import asyncio
import json
import logging

from core.stm_analyzer import STMAnalyzer


class _Model:
    def __init__(self, payload):
        self.payload = payload

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.payload)


_PRIOR_ECHO = {
    "topic": "Harbor festival comments",
    "user_question": "User is stating that most online comments defend the festival",
    "intent": "Discuss the earlier festival thread",
    "tone": "casual",
    "reference_type": "recall",
    "temporal_facts": [],
    "open_threads": [],
    "constraints": [],
}

_WINDOW = [{
    "timestamp": "2026-10-08T13:00:00",
    "query": "Most online comments defend the harbor festival",
    "response": "That tracks with the thread you linked.",
}]


def _run(payload, query, last_reply=None):
    analyzer = STMAnalyzer(_Model(payload))
    analyzer._get_recent_daily_notes_text = lambda *a, **k: ""
    return asyncio.run(analyzer.analyze(
        recent_memories=_WINDOW, user_query=query, last_assistant_response=last_reply,
    ))


def test_summary_about_the_previous_message_is_replaced_by_a_quote(caplog):
    with caplog.at_level(logging.INFO):
        out = _run(dict(_PRIOR_ECHO), "Senator one of them")
    assert out["user_question"] == 'User said: "Senator one of them"'
    assert out["reference_type"] == "unclear"
    assert out["stm_fidelity_override"] is True
    assert any("[STM] fidelity fallback" in r.getMessage() for r in caplog.records)


def test_imperative_with_no_overlap_is_replaced():
    out = _run(dict(_PRIOR_ECHO), "Yeah search it")
    assert out["user_question"] == 'User said: "Yeah search it"'
    assert out["reference_type"] == "unclear"


def test_faithful_summary_is_left_unchanged():
    payload = dict(_PRIOR_ECHO, user_question="User asks to search for the senator",
                   reference_type="new_event")
    out = _run(payload, "Senator one of them")
    assert out["user_question"] == "User asks to search for the senator"
    assert out["reference_type"] == "new_event"
    assert "stm_fidelity_override" not in out


def test_inflected_summary_counts_as_faithful():
    payload = dict(_PRIOR_ECHO, intent="Wants the assistant to be searching the web",
                   reference_type="new_event")
    out = _run(payload, "Yeah search it")
    assert "stm_fidelity_override" not in out


def test_quote_is_clipped_to_200_chars():
    from core.stm_analyzer import apply_fidelity_fallback
    long_q = "zyxwv " * 60
    parsed = dict(_PRIOR_ECHO)
    assert apply_fidelity_fallback(long_q, parsed, _WINDOW) is True
    quoted = parsed["user_question"][len('User said: "'):-1]
    assert len(quoted) == 200


def test_paraphrase_without_anchor_in_earlier_messages_is_left_alone():
    # "stimulant" summarised as "Medication check-in": no shared word, but the
    # summary's words appear in NO earlier user message -> not provably
    # anchored elsewhere -> unchanged.
    payload = dict(_PRIOR_ECHO, topic="Medication check-in",
                   user_question="User is restating their routine", intent="Share")
    out = _run(payload, "I took my stimulant at 10 AM")
    assert "stm_fidelity_override" not in out
    assert out["user_question"] == "User is restating their routine"


def test_pure_acknowledgment_is_exempt():
    payload = dict(_PRIOR_ECHO)
    out = _run(payload, "ok thanks")
    assert "stm_fidelity_override" not in out
    assert out["reference_type"] == "recall"


def test_empty_summary_and_tokenless_query_are_left_alone():
    from core.stm_analyzer import apply_fidelity_fallback
    assert apply_fidelity_fallback("Senator one of them", {"topic": "", "user_question": "", "intent": ""}, _WINDOW) is False
    assert apply_fidelity_fallback("it is so", dict(_PRIOR_ECHO), _WINDOW) is False
    assert apply_fidelity_fallback("", dict(_PRIOR_ECHO), _WINDOW) is False
    assert apply_fidelity_fallback("Senator one of them", dict(_PRIOR_ECHO), None) is False
