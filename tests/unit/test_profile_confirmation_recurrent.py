"""Confirming a previously-superseded value must make it current again.

Regression for 2026-08-05: the live profile had age='19' (junk, current) over
a historical age='32'/'33'. Re-stating the true value hit add_fact Case 1
(exact match → confidence boost only), which left is_current=False on the
true fact and the junk value current — no number of confirmations could ever
displace it. Same live failure with timezone='your time' over 'Central'.
"""
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from memory.user_profile import UserProfile


@pytest.fixture
def profile():
    with tempfile.TemporaryDirectory() as td:
        yield UserProfile(profile_path=os.path.join(td, "profile.json"))


def _facts(profile, relation):
    out = {}
    for facts in profile.profile["categories"].values():
        for f in facts:
            if isinstance(f, dict) and f.get("relation") == relation:
                out[f["value"]] = f
    return out


def test_confirmation_recurrents_superseded_value(profile):
    t0 = datetime.now() - timedelta(days=120)
    profile.add_fact(relation="age", value="33", confidence=0.9, timestamp=t0)
    # Junk value supersedes the true one (this is how the live profile got here).
    profile.add_fact(relation="age", value="19", confidence=0.7,
                     timestamp=t0 + timedelta(days=30))
    facts = _facts(profile, "age")
    assert facts["33"]["is_current"] is False
    assert facts["19"]["is_current"] is True

    # User re-states the true value → must become current again.
    profile.add_fact(relation="age", value="33", confidence=0.95)
    facts = _facts(profile, "age")
    assert facts["33"]["is_current"] is True
    assert facts["19"]["is_current"] is False


def test_confirmation_of_current_value_unchanged(profile):
    profile.add_fact(relation="timezone", value="Central", confidence=0.9)
    profile.add_fact(relation="timezone", value="Central", confidence=0.9)
    facts = _facts(profile, "timezone")
    assert facts["Central"]["is_current"] is True
    # Confirmation boosts confidence, no duplicate appended.
    assert len([v for v in facts.values()]) == 1
    assert facts["Central"]["confirmation_count"] == 1


def test_recurrent_applies_truth_correction_to_displaced_fact(profile):
    t0 = datetime.now() - timedelta(days=60)
    profile.add_fact(relation="timezone", value="Central", confidence=0.9, timestamp=t0)
    profile.add_fact(relation="timezone", value="your time", confidence=0.6,
                     timestamp=t0 + timedelta(days=10))
    before = _facts(profile, "timezone")["your time"]["truth_score"]
    profile.add_fact(relation="timezone", value="Central", confidence=0.95)
    facts = _facts(profile, "timezone")
    assert facts["Central"]["is_current"] is True
    assert facts["your time"]["is_current"] is False
    assert facts["your time"]["truth_score"] < before


def test_quick_profile_updates_regardless_of_category(profile):
    """2026-08-21: quick_profile only refreshed when the categorizer labeled
    the fact IDENTITY — the 08-05 timezone correction arrived as 'preferences',
    so quick_profile kept rendering the junk 'your time' in every prompt.
    _update_quick_profile filters by relation itself; the category gate is gone."""
    profile.add_fact(relation="timezone", value="your time", confidence=0.8)
    profile.add_fact(relation="timezone", value="Central", confidence=0.9)
    assert profile.get_quick_profile().get("timezone") == "Central"


def test_quick_profile_school_and_program(profile):
    """school/program are quick keys (2026-08-21): the program fact ('Online
    Master of Science in Analytics') rode the semantic fact lottery and the
    model assumed in-person campus classes even with the fact in-prompt —
    durable identity facts belong in the always-rendered Quick Profile."""
    profile.add_fact(relation="school", value="Georgia Tech", confidence=1.0)
    profile.add_fact(
        relation="program",
        value="Online Master of Science in Analytics (fully online)",
        confidence=0.9,
    )
    quick = profile.get_quick_profile()
    assert quick.get("school") == "Georgia Tech"
    assert "Online" in quick.get("program", "")


def test_quick_profile_non_quick_relations_ignored(profile):
    profile.add_fact(relation="favorite_food", value="tacos", confidence=0.9)
    assert "favorite_food" not in profile.get_quick_profile()
