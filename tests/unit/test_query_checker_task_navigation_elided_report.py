"""F1 leftover (measured 2026-10-26; wave 2 2026-10-27; class: BC-06, BC-58).

"did the next one already" (subject elided) still navigated: its past-tense
auxiliary opener read as an interrogative, so the clause counted as
request-shaped. A subject-less finite-past report is treated like the
first-person one ("I did the next one already"), unless the clause carries an
explicit request shape (imperative, address to the assistant, "can you").
"""
from __future__ import annotations

import pytest

from utils.query_checker import is_task_navigation


@pytest.mark.parametrize("text", [
    "did the next one already",
    "Did the next one already",
    "did the next one\nalready",
    "  did the next one already",
])
def test_elided_past_report_is_not_navigation(text):
    assert is_task_navigation(text) is False


@pytest.mark.parametrize("text", [
    "I did the next one already",
    "ok next question",
    "did you do the next one?",
    "please give me the next question",
    "next one please",
])
def test_requests_and_first_person_reports_unchanged(text):
    expected = text != "I did the next one already"
    assert is_task_navigation(text) is expected
