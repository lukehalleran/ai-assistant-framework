"""B5 (2026-10-12): pasted third-party text must not route insight mode or set
the user's tone (class: BC-58). Beta turn: "what am i looking at here:" + a
Wikipedia page -> insight pattern_temporal (53 s) and a LIGHT SUPPORT tip reply.

Original batch label: oct12_b5 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
from utils.request_span import request_span
from utils.web_search_trigger import _looks_like_pattern_candidate
from core.insight.detector import detect_insight_request
from utils.tone_detector import _recent_distress_from_history

_CHROME = ["WikipediaThe Free Encyclopedia", "", "Donate", "Create account", "Log in", "",
           "Mathematics", "", "Article", "Talk", "Read", "View source", "View history", ""]
_PARA = ("Mathematics is a field of knowledge. We study patterns and trends over time, and "
         "my favourite example is how number sequences correlate with growth. Some learners "
         "report mathematical anxiety, which can hurt their performance in school. ") * 4


def _wiki_paste(n_sections: int = 10) -> str:
    lines = list(_CHROME)
    for i in range(n_sections):
        lines += [f"Section {i}", "", _PARA, "", "See also", "References", "External links", ""]
    lines += ["Privacy policy", "Mobile view"]
    return "\n".join(lines)


PASTE = _wiki_paste()
BETA_TURN = "what am i looking at here: \n" + PASTE + (
    "\n\n[ACTIVE DOCUMENT NOTE] Several attached documents: a.pdf, b.md - ask which one.")

VENT = "\n\n".join(
    "I feel like everything is falling apart and I can't stop thinking about how my life "
    "keeps going wrong, I'm exhausted and hopeless and I don't know who to talk to. " * 3
    for _ in range(5))


def test_paste_is_big_enough():
    assert len(PASTE) > 6000


def test_request_span_keeps_only_the_request():
    span = request_span(BETA_TURN)
    assert span.startswith("what am i looking at here:")
    assert "Mathematics is a field" not in span
    assert len(span) < 400


def test_long_first_person_message_is_not_a_paste():
    assert len(VENT) > 1500
    assert request_span(VENT) == VENT
    assert request_span("short message") == "short message"


def test_short_first_line_then_long_prose_vent_not_split():
    text = "I can't do this anymore.\n" + VENT
    assert request_span(text) == text


def test_pattern_candidate_ignores_pasted_block():
    assert _looks_like_pattern_candidate(BETA_TURN) is False


def test_beta_turn_is_not_an_insight_request():
    assert detect_insight_request(BETA_TURN) is None


def test_genuine_insight_request_still_detects():
    req = "gather everything I've said about sleep over time"
    assert detect_insight_request(req) is not None
    # request first, article pasted after it: still the request span
    assert detect_insight_request(req + "\n" + PASTE) is not None
    # request after the paste (tail)
    assert detect_insight_request(PASTE + "\n\n" + req) is not None


def test_pasted_heavy_row_is_not_distress_evidence():
    assert _recent_distress_from_history([{"query": BETA_TURN, "is_heavy_topic": True}]) is False


def test_first_person_vent_row_still_distress_evidence():
    assert _recent_distress_from_history([{"query": VENT, "is_heavy_topic": True}]) is True
