"""A first-person past-tense verb outside the closed word set is a past cue.

Incident 2026-10-16 (class: BC-04 closed list standing in for a grammatical
class): ``_PAST_TENSE_CUE_RE`` is a closed word set that lacked "reviewed", so
``classify_claim_time("I reviewed chapters yesterday")`` was `unknown` instead
of a dated event. ``has_past_tense_cue`` keeps the regex as the fast path and
falls back to the parse (VBD verb with a first-person subject, directly or
through a conjunct). Every claim-time call site routes through it.
"""
from datetime import datetime
import inspect

import pytest

import memory.fact_source as fs

OBS = datetime(2026, 10, 21, 12, 0)


def _wrapped(t):
    return "   " + t.replace(" yesterday", "\n   yesterday")


@pytest.mark.parametrize("text", [
    "I reviewed chapters yesterday",
    "I reviewed chapters yesterday and edited the notes",
    "Yesterday I annotated the paper.",
])
def test_unlisted_past_verb_dates_the_event(text):
    for form in (text, _wrapped(text)):
        assert fs.has_past_tense_cue(form), form
        ct = fs.classify_claim_time(form, observed_at=OBS)
        assert ct.kind == "event", (form, ct)
    ct = fs.classify_claim_time("I reviewed chapters yesterday", observed_at=OBS)
    assert ct.event_date.isoformat() == "2026-10-20"


@pytest.mark.parametrize("text", [
    "Rowan reviewed chapters yesterday",   # third-person subject: not the user's act
    "I will review chapters tomorrow",
    "I review chapters every day",
    "The prescribed dose is on the label",
])
def test_non_user_or_non_past_is_not_a_past_cue(text):
    assert not fs.has_past_tense_cue(text), text


def test_listed_verbs_take_the_fast_path_without_the_parser(monkeypatch):
    monkeypatch.setattr("memory.fact_extractor._get_nlp",
                        lambda: (_ for _ in ()).throw(AssertionError("parser used")))
    fs._parsed_past_first_person_verb.cache_clear()
    assert fs.has_past_tense_cue("I skipped the gym yesterday")


def test_no_call_site_bypasses_the_helper():
    src = inspect.getsource(fs)
    body = src.replace("_PAST_TENSE_CUE_RE = re.compile(", "")
    uses = [l for l in body.splitlines() if "_PAST_TENSE_CUE_RE" in l and ".search(" in l]
    # the only direct .search is the helper's own fast path
    assert len(uses) == 1, uses
