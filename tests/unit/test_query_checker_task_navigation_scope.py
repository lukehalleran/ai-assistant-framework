"""is_task_navigation counts only a navigation REQUEST.

Incident 2026-10-18 (class: BC-04 classifier missing an anchor; BC-51):
a pasted write-up beginning "Problem 12 ..." and a statement "Next one due
Friday" were routed as active-document navigation and appended slide text /
"item 12 does not exist" to the user's message.

Drives THE deployed utils.query_checker.is_task_navigation, each fixture in
clean and wrapped/indented form.
"""
import pytest

from utils.query_checker import is_task_navigation


def _wrapped(text: str) -> str:
    # client soft line-wrap + indentation
    words = text.split(" ")
    mid = len(words) // 2
    return "  " + " ".join(words[:mid]) + "\n    " + " ".join(words[mid:])


REQUESTS = [
    "next question please",
    "ok go to question 3",
    "ok next q please",
    "show me the first question",
    "what's the next one",
    "Q3",
]
STATEMENTS = [
    "Next one due Friday",
    "part 2 is due friday",
    "the next one is due Friday and I have not started",
]


@pytest.mark.parametrize("text", REQUESTS)
def test_navigation_requests_match(text):
    assert is_task_navigation(text) is True
    assert is_task_navigation(_wrapped(text)) is True


@pytest.mark.parametrize("text", STATEMENTS)
def test_statements_about_next_item_do_not_match(text):
    assert is_task_navigation(text) is False
    assert is_task_navigation(_wrapped(text)) is False


def _writeup(position: str) -> str:
    body = " ".join(["the mean of the sample was computed from the table"] * 45)
    if position == "head":
        return "Problem 12 the Assign Module\n" + body + "\nwas i wrong?"
    if position == "middle":
        return body + "\nProblem 12 the Assign Module\n" + body + "\nwas i wrong?"
    return body + "\nwhat the screenshot in question 6 shows\n" + body


@pytest.mark.parametrize("position", ["head", "middle", "screenshot"])
def test_pasted_writeup_numbering_is_not_navigation(position):
    text = _writeup(position)
    assert len(text.split()) > 300
    assert is_task_navigation(text) is False
    assert is_task_navigation("\n".join("  " + ln for ln in text.split("\n"))) is False


def test_long_message_with_trailing_request_still_navigates():
    body = " ".join(["the mean of the sample was computed from the table"] * 45)
    text = body + "\n\nok go to question 3"
    assert is_task_navigation(text) is True
