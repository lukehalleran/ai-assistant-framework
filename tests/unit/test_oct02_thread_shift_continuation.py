"""[THREAD CONTEXT] must not assert a topic shift on short follow-up shapes.

class: BC-08. Drives the deployed core.orchestrator._thread_topic_shifted.
"""
import pytest

from core.orchestrator import _thread_topic_shifted as shifted

DAD_Q = "Does he need you there for the whole weekend, or just Saturday?"
OFFER = "I can put that on your calendar. Want me to?"
EMAIL = "I searched your inbox and found three threads. Anything else?"


@pytest.mark.parametrize(
    "query, prev_resp, thread, current",
    [
        ("Navient. Search that", EMAIL, "Email Search", "Student Loans"),
        ("yes please", OFFER, "Calendar Scheduling", "General Assistance"),
        ("No it's for me. I don't want to move", DAD_Q, "Dad Visit Plans", "Housing Affordability"),
    ],
)
def test_short_followups_never_assert_shift(query, prev_resp, thread, current):
    assert shifted(thread, current, query, last_assistant_response=prev_resp) is False


def test_answer_to_question_needs_the_pending_question():
    # Without a question pending, divergent labels on a statement still shift.
    q = "No it's for me. I don't want to move"
    assert shifted("Dad Visit Plans", "Housing Affordability", q,
                   last_assistant_response="Sounds like a long weekend.") is True
    assert shifted("Dad Visit Plans", "Housing Affordability", q) is True


def test_long_new_topic_after_question_still_shifts():
    q = ("Separately I need to figure out how to restructure my statistics "
         "homework schedule across the next three weeks before midterms start")
    assert shifted("Dad Visit Plans", "Homework Planning", q,
                   last_assistant_response=DAD_Q) is True
