"""
Tests for memory/relation_classifier.py — the single source of truth for
relation TTLs (standard-ephemeral vs health-transient vs durable).
"""

import importlib

import pytest

from memory import relation_classifier as rc


# --------------------------------------------------------------------------
# Health-transient tier
# --------------------------------------------------------------------------

HEALTH_TRANSIENT = [
    "post_illness_recovery",   # substring: illness + recover
    "recovery_status",         # substring: recover (also _status suffix)
    "recovery_time",
    "recent_illness",          # substring: illness (also recent_ prefix)
    "illness_start",
    "last_sick_day",           # substring: sick (also last_ prefix)
    "sick_days",
    "symptoms",
    "current_symptom",
    "health_status",           # exact health set (no substring)
    "health_condition",
    "current_condition",
    "recent_condition",
]


@pytest.mark.parametrize("rel", HEALTH_TRANSIENT)
def test_health_transient_gets_health_ttl(rel):
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS
    assert rc.ephemeral_ttl_hours(rel) == float(PROFILE_HEALTH_TRANSIENT_TTL_HOURS)
    assert rc.is_ephemeral_relation(rel) is True


def test_health_transient_beats_standard_ephemeral():
    """recovery_status matches both the _status suffix (24h) and the recover
    cue (health). Health wins so it gets the longer horizon, not 24h."""
    from config.app_config import (
        PROFILE_HEALTH_TRANSIENT_TTL_HOURS, PROFILE_EPHEMERAL_TTL_HOURS,
    )
    assert PROFILE_HEALTH_TRANSIENT_TTL_HOURS != PROFILE_EPHEMERAL_TTL_HOURS
    assert rc.ephemeral_ttl_hours("recovery_status") == float(PROFILE_HEALTH_TRANSIENT_TTL_HOURS)


# --------------------------------------------------------------------------
# Standard-ephemeral tier
# --------------------------------------------------------------------------

STANDARD_EPHEMERAL = [
    "current_activity",   # config exact list
    "current_feeling",
    "current_mood",
    "current_state",      # current_ prefix
    "woke_up_time",       # woke_ prefix
    "work_status",        # _status suffix
    "project_status",
    "energy_level",       # _EPHEMERAL_EXACT
    "meal",
]


@pytest.mark.parametrize("rel", STANDARD_EPHEMERAL)
def test_standard_ephemeral_gets_ephemeral_ttl(rel):
    from config.app_config import PROFILE_EPHEMERAL_TTL_HOURS
    assert rc.ephemeral_ttl_hours(rel) == float(PROFILE_EPHEMERAL_TTL_HOURS)
    assert rc.is_ephemeral_relation(rel) is True


# --------------------------------------------------------------------------
# Durable tier (no TTL)
# --------------------------------------------------------------------------

DURABLE = [
    "name", "birthday", "age", "brother_name", "location",
    "allergies", "takes_medication", "work_period",
    # Permanent conditions must NOT be treated as transient illness state:
    "disability", "chronic_condition", "diagnosis", "diagnosed_with",
]


@pytest.mark.parametrize("rel", DURABLE)
def test_durable_relations_have_no_ttl(rel):
    assert rc.ephemeral_ttl_hours(rel) is None
    assert rc.is_ephemeral_relation(rel) is False


# --------------------------------------------------------------------------
# Edge cases + config responsiveness
# --------------------------------------------------------------------------

def test_empty_and_whitespace():
    assert rc.ephemeral_ttl_hours("") is None
    assert rc.ephemeral_ttl_hours(None) is None
    # case + surrounding whitespace are normalized
    assert rc.ephemeral_ttl_hours("  Health_Status  ") is not None


def test_ttl_reads_config_live(monkeypatch):
    """TTL is read lazily so config changes are reflected without reimport."""
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "PROFILE_HEALTH_TRANSIENT_TTL_HOURS", 12)
    monkeypatch.setattr(cfg, "PROFILE_EPHEMERAL_TTL_HOURS", 3)
    assert rc.ephemeral_ttl_hours("post_illness_recovery") == 12.0
    assert rc.ephemeral_ttl_hours("current_activity") == 3.0


def test_disability_not_swept_by_value_independent_relation():
    """Regression guard for the user's explicit constraint: a disability is not
    an illness episode and must never get a finite TTL on relation name alone."""
    for rel in ("disability", "permanent_disability", "chronic_condition"):
        assert rc.is_ephemeral_relation(rel) is False


# --------------------------------------------------------------------------
# Free-text health-framing detection (narrative memories)
# --------------------------------------------------------------------------

HEALTH_TRANSIENT_TEXT = [
    "That post-viral drag is stubborn — you've been fighting it",
    "Still post-viral and grinding through homework",
    "You were fighting a virus like 3-4 days ago, man",
    "That's your body still recovering from the flu",
    "Post-viral fatigue hit hard this afternoon",
    "honestly worried I was sick or something",
    "I think I'm getting sick again",
    "feeling under the weather today",
]


@pytest.mark.parametrize("text", HEALTH_TRANSIENT_TEXT)
def test_health_transient_text_detected(text):
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS
    assert rc.health_transient_text_ttl_hours(text) == float(PROFILE_HEALTH_TRANSIENT_TTL_HOURS)
    assert rc.is_health_transient_text(text) is True


NOT_HEALTH_TEXT = [
    "that video went viral on twitter overnight",   # bare "viral" is not a cue
    "it was cold weather on the walk but otherwise fine",  # bare "cold"
    "my workout recovery time is 2-3 days",         # bare "recovery"
    "we shipped a retrieval overhaul and OAuth2 pipeline today",
    "the simulation homework is due soon",
    "",
]


@pytest.mark.parametrize("text", NOT_HEALTH_TEXT)
def test_non_health_text_not_detected(text):
    assert rc.health_transient_text_ttl_hours(text) is None
    assert rc.is_health_transient_text(text) is False


DURABLE_HEALTH_TEXT = [
    "I have a chronic illness so fatigue is just part of life",
    "diagnosed with an autoimmune condition years ago",
    "it's a permanent disability, not something I recover from",
    "managing a lifelong condition, was sick of explaining it",  # durable cue wins
]


@pytest.mark.parametrize("text", DURABLE_HEALTH_TEXT)
def test_durable_health_text_suppresses_transient(text):
    """Chronic / permanent condition language must NOT be treated as a transient
    episode even if a transient phrase also appears (mirrors _DURABLE_OVERRIDES
    for relation names — a disability is not an illness episode)."""
    assert rc.health_transient_text_ttl_hours(text) is None
    assert rc.is_health_transient_text(text) is False


def test_health_text_ttl_reads_config_live(monkeypatch):
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "PROFILE_HEALTH_TRANSIENT_TTL_HOURS", 12)
    assert rc.health_transient_text_ttl_hours("still post-viral and tired") == 12.0
