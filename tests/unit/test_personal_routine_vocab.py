"""Owner-specific routine nouns come from the gitignored vocabulary, not this repo.

`is_personal_routine_question` keeps a first-person dosing/schedule question
("what time should I take <medication> tonight") away from the web-search
trigger. Its noun list used to carry one owner's medication by name. The generic
nouns stay in the module; a personal one is read live from
`user_profile.personal_vocabulary.category_tokens.health` (config.local.yaml).
Drives THE deployed predicate; the medication name here is synthetic.
"""
import pytest

import config.app_config as app_config
from utils.web_search_trigger import is_personal_routine_question

PERSONAL = "What time should I take lorvatin tonight"


@pytest.fixture
def health_tokens(monkeypatch):
    def _set(tokens):
        monkeypatch.setattr(app_config, "PROFILE_PERSONAL_CATEGORY_TOKENS", {"health": tokens})
    return _set


def test_a_fresh_clone_knows_no_personal_medication(health_tokens):
    health_tokens([])
    assert not is_personal_routine_question(PERSONAL)


@pytest.mark.parametrize("query", [PERSONAL, "when should I take LORVATIN", "should i take my lorvatin now"])
def test_a_configured_medication_is_a_routine_noun(health_tokens, query):
    health_tokens(["Lorvatin"])
    assert is_personal_routine_question(query)


def test_generic_nouns_need_no_configuration(health_tokens):
    health_tokens([])
    assert is_personal_routine_question("What time should I take melatonin to get to bed")


def test_the_cue_is_still_required(health_tokens):
    health_tokens(["lorvatin"])
    assert not is_personal_routine_question("lorvatin dosing guidelines for adults")


def test_a_token_matches_whole_words_only(health_tokens):
    health_tokens(["lor"])
    assert not is_personal_routine_question(PERSONAL)


def test_the_vocabulary_is_read_live_not_frozen_at_import(health_tokens):
    health_tokens([])
    assert not is_personal_routine_question(PERSONAL)
    health_tokens(["lorvatin"])
    assert is_personal_routine_question(PERSONAL)


@pytest.mark.parametrize("vocabulary", [None, {}, {"health": None}, {"hobbies": ["lorvatin"]}])
def test_missing_or_other_category_vocabulary_is_harmless(monkeypatch, vocabulary):
    monkeypatch.setattr(app_config, "PROFILE_PERSONAL_CATEGORY_TOKENS", vocabulary)
    assert not is_personal_routine_question(PERSONAL)
