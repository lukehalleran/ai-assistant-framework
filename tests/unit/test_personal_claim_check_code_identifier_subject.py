"""A capitalized code identifier is not a personal subject.

Incident 2026-10-20 (oct20b residual 1b), class: BC-84.  After the 10-21 fix an
unknown proper noun is no longer a person, but the elided-agent fail-closed
branch ("HW4 submitted.") still flagged module names such as "TaskQueue crashed."
and "The MemoryScorer finished.".  Decided by token SHAPE (CamelCase inner
capital, snake_case, dotted path), not a name list.  Drives THE deployed
``_is_personal_completion``; clean AND wrapped/indented forms.
"""
import pytest

from core.personal_claim_check import _is_code_shaped, _is_personal_completion


def _forms(text):
    return [text, "   " + text + "\n"]


@pytest.mark.parametrize("text", [
    "TaskQueue crashed.",
    "The MemoryScorer finished.",
    "MemoryScorer returned.",
    "RetrievalGate completed.",
    "memory_scorer finished.",
    "utils.safe_json finished.",
])
def test_code_identifier_subject_is_not_personal(text):
    for form in _forms(text):
        assert _is_personal_completion(form) is False, form


@pytest.mark.parametrize("text", [
    "Rowan finished.",
    "HW4 submitted.",
    "I finished.",
    "TaskQueue crashed and I finished the report.",
])
def test_person_or_assignment_subject_stays_personal(text):
    assert _is_personal_completion(text) is True, text


@pytest.mark.parametrize("token,expected", [
    ("TaskQueue", True), ("memory_scorer", True), ("utils.safe_json", True),
    ("run()", True), ("Arena", False), ("Rowan", False), ("HW4", False), ("", False),
])
def test_code_shape_predicate(token, expected):
    assert _is_code_shaped(token) is expected
