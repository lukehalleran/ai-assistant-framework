"""Spanless exemplar teachers now pass their supporting span (A06 contract).

Incident 2026-10-26 (H1 handoff 2c): core/agentic/gate.py (no_search teach) and
core/action_claim_guard.py (record_claim_exemplar) handed over-long text to the
exemplar store, which declines >300 chars without a span -- the label was
silently never learned. Fixed 2026-10-27. class: BC-58 (learning channel dead
at one end), BC-29 (label rests on clipped text). core/orchestrator.py's
citation teach is NOT covered: it has no clause to point at (see handoff).
Every case is asserted clean and wrapped/indented. Fixture text is invented.
"""
from unittest.mock import patch

import pytest

from core.action_claim_guard import record_claim_exemplar
from core.agentic.gate import AgenticDecision, _vent_support_span, apply_intent_veto
from utils.adaptive_exemplars import AdaptiveExemplarStore

_FILLER = ("The schedule for the week covers the garden notes, the oven timing, "
           "and the dishes planned for the table on Saturday. ")
_VENT = "I am so unhappy and embarrassed about my reaction earlier. "


def _wrap(text, width=50):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n    ".join(lines)


def _shape(text, shape):
    return _wrap(text) if shape == "wrapped" else text


def _decision():
    return AgenticDecision(
        should_trigger=True, modes=["memory"], search_terms=[], matched_entities=[],
        doc_gen_intent=None, self_note_intent=None, skip_initial_search=True,
        reason="triggered: memory", veto_exempt=False)


@pytest.fixture
def store(tmp_path):
    s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
    with patch("utils.adaptive_exemplars.get_store", return_value=s):
        yield s


@pytest.mark.parametrize("shape", ["clean", "wrapped"])
def test_long_vent_message_teaches_no_search_with_the_vent_window(store, shape):
    query = _shape(_FILLER * 3 + _VENT + _FILLER, shape)
    assert len(" ".join(query.split())) > 300
    apply_intent_veto(_decision(), {"intent_type": "general", "confidence": 0.0},
                      tone_level="light_support", query=query)
    learned = store.get_learned("web_search", "no_search")
    assert learned, "the vent sentence is the span; the long message must be learned as a window"
    assert all(len(t) <= 300 for t in learned) and any("unhappy" in t for t in learned)


@pytest.mark.parametrize("shape", ["clean", "wrapped"])
def test_support_span_points_at_the_first_person_sentence(shape):
    query = _shape(_FILLER + _VENT + _FILLER, shape)
    start, end = _vent_support_span(query)
    assert "unhappy" in " ".join(query[start:end].split())
    assert _vent_support_span("The mayor declared the bridge open.") is None
    assert _vent_support_span("") is None


@pytest.mark.parametrize("shape", ["clean", "wrapped"])
def test_long_claim_sentence_is_learned_as_a_window(store, shape):
    sentence = ("I have added the dentist appointment to your calendar for Thursday and "
                + "also " * 60 + "it is all set.")
    sentence = _shape(sentence, shape)
    assert len(" ".join(sentence.split())) > 300
    with patch("utils.adaptive_exemplars.get_store", return_value=store):
        assert record_claim_exemplar("calendar_state", sentence, "user_failure_report")
    learned = store.get_learned("action_claim", "calendar_state")
    assert learned and all(len(t) <= 300 for t in learned)


def test_short_claim_sentence_is_unchanged(store):
    with patch("utils.adaptive_exemplars.get_store", return_value=store):
        assert record_claim_exemplar("card_claim", "The card is waiting for your approval.", "t")
    assert store.get_learned("action_claim", "card_claim") == ["The card is waiting for your approval."]
