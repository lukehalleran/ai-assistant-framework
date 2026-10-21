"""Visual gate: the weak verbs see/look/watch carry imagery intent only when
the USER is the perceiver.

Incident 2026-10-17: "I think you will be able to see the files now?" (an
upload turn) passed the weak-verb arm of ``_query_wants_visual`` and ran the
visual-memory path; the user was asking about the ASSISTANT's access.
class: BC-06 (trigger over-fire on a non-request).
Drives THE deployed ``_query_wants_visual``; each fixture in clean, indented
and line-wrapped form.
"""

import pytest

from core.prompt.gatherer_knowledge import _query_wants_visual


def _forms(text):
    words = text.split()
    mid = max(1, len(words) // 2)
    wrapped = " ".join(words[:mid]) + "\n    " + " ".join(words[mid:])
    return [text, "\n    " + text + "\n", wrapped]


CASES = [
    # (text, intent, expected)
    ("I think you will be able to see the files now?", "general", False),
    ("can you see it now?", "general", False),
    ("did you see it yet?", "general", False),
    ("you'll be able to see it now", "general", False),
    ("you should be able to see the screenshot", "general", True),  # noun arm
    ("can you see the photo?", "general", True),  # noun arm unchanged
    ("let me see the cat", "general", True),
    ("can I see the pictures from the lake", "general", True),
    ("look at this", "general", True),
    ("what does my cat look like", "factual_recall", True),
    ("show me the folder", "general", False),  # document noun
    ("can you show me it", "general", True),  # request-shaped, exempt
    ("don't let me see it", "general", False),  # negated
    ("I don't want to look at it", "general", False),
    ("don't show me the photos", "general", False),
]


@pytest.mark.parametrize("text,intent,expected", CASES)
def test_perceiver_rule(text, intent, expected):
    for form in _forms(text):
        assert _query_wants_visual(form, intent) is expected, repr(form)


def test_look_at_object_from_you_is_still_a_request():
    for form in _forms("can you look at the diagram in my notes?"):
        assert _query_wants_visual(form, "general") is True, repr(form)
