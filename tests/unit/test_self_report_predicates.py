"""utils.query_checker.is_self_report / is_request_shaped (2026-09-06).

Closed grammatical sets only (pronouns, auxiliaries, imperative openers) —
the predicates must hold for any American-English user's status update or
request, never for a topic. All cases call the deployed functions.
"""

import pytest

from utils.query_checker import is_request_shaped, is_self_report

Q1 = ("I took my stimulant at 10 AM today and I'm just resting this afternoon, "
      "feels good honestly even though I got nothing done")
Q2 = ("Does my history actually support scheduling occasional rest days off the "
      "medication? Weigh both sides.")
Q3 = ("Give me a detailed analysis in a table of what my record can establish "
      "about medication gaps.")


class TestIsSelfReport:
    @pytest.mark.parametrize("text", [
        Q1,
        "ok I checked it out, kind of sucked",
        "im so tired today",
        "we finally moved the couch",
        "honestly I feel a lot better than yesterday",
        "Yeah, I've been sleeping badly all week",
    ])
    def test_first_person_statements(self, text):
        assert is_self_report(text) is True

    @pytest.mark.parametrize("text", [
        Q2, Q3,
        "Did I take it at 10?",
        "summarize my week",
        "please pull up the veto logic",
        "I want you to look at the log",
        "can you check the log",
        "what did I say yesterday",
        "The president declared the strait closed",
        "It was maybe 3 years of twice a week",
        "ok", "lol", "",
    ])
    def test_questions_requests_third_party_and_fragments_excluded(self, text):
        assert is_self_report(text) is False

    def test_paste_excluded_by_lines_and_length(self):
        assert is_self_report("Hi Morgan,\nThanks for the note.\nBest,\nLuke") is False
        assert is_self_report("I " + "walked and walked " * 30) is False

    def test_max_words_is_a_parameter(self):
        text = "I went to the store and " * 5 + "came home"
        assert is_self_report(text, max_words=10) is False
        assert is_self_report(text, max_words=60) is True


class TestIsRequestShaped:
    @pytest.mark.parametrize("text", [
        Q2, Q3,
        "please pull up the veto logic",
        "can you check the log",
        "weigh both sides for me",
        "I want you to look at the log",
        "what's the weather",
        "should I switch to postgres",
    ])
    def test_requests(self, text):
        assert is_request_shaped(text) is True

    @pytest.mark.parametrize("text", [
        Q1,
        "we finally moved the couch",
        "The president declared the strait closed",
        "ok", "",
    ])
    def test_non_requests(self, text):
        assert is_request_shaped(text) is False

    def test_self_report_and_request_are_disjoint_on_live_queries(self):
        for q in (Q1, Q2, Q3):
            assert not (is_self_report(q) and is_request_shaped(q))


def test_soft_wrapped_self_report_is_one_message():
    """Live 2026-09-06 15:10: the query arrived hard-wrapped from a code-block
    paste (3 lines, leading spaces) and the ≥3-lines paste guard rejected it —
    no retrieval trim, no decision-support exclusion. Soft wraps are one
    message; a paste is blank-line paragraphs or punctuation-terminated lines."""
    wrapped = ("I took my stimulant at 10 AM today and I'm just\n  resting this "
               "afternoon, feels good honestly even\n  though I got nothing done")
    assert is_self_report(wrapped) is True
    assert is_self_report("I went out.\n\nThen I came home.\n\nIt was fine.") is False
    assert is_self_report("Hi Morgan,\nThanks for the note.\nBest,\nLuke") is False
