"""A first-person report of a finished act is not a navigation request.

Incident 2026-10-20: "I did the next one already" still counted as task
navigation (after #93, the terse stop-word rule accepted it because every word
besides the navigation phrase is a function word), so the active-document
route fired on a status report. Class: BC-06 (grammatical mood/tense must be
read, not word membership). Drives THE deployed ``is_task_navigation`` and
``is_casual_acknowledgment``.
"""
import pytest

from utils.query_checker import is_task_navigation


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 2) if wrapped else text


REPORTS = [
    "I did the next one already",
    "I've done the next one",
    "We did part 2 already.",
    "I finished q3 already.",
    "I did the next question already.",
]
REQUESTS = [
    "ok next q please",
    "next one",
    "what's the next one",
    "did I do the next one?",
    "show me the next one",
    "go on to the next one",
    "Q3",
    "next question",
]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("text", REPORTS)
def test_first_person_done_report_is_not_navigation(text, wrapped):
    assert is_task_navigation(_wrap(text, wrapped)) is False


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("text", REQUESTS)
def test_requests_stay_navigation(text, wrapped):
    assert is_task_navigation(_wrap(text, wrapped)) is True


def test_parser_missing_keeps_previous_behaviour(monkeypatch):
    import memory.fact_extractor as fe
    monkeypatch.setattr(fe, "_get_nlp", lambda: None)
    assert is_task_navigation("next one") is True
