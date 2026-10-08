"""Household relation variants fold to ONE canonical, single-valued relation.

Live failure (2026-10-02): living_arrangement="live on my own" (Feb) stayed
current beside living_with=<person> (Aug), and roommate / relationship_with_
roommate duplicated living_with. class: BC-76, BC-58
"""
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from memory.relation_classifier import is_multi_valued_relation
from memory.user_profile import UserProfile
from memory.user_profile_schema import canonicalize_profile_relation

VARIANTS = [
    "roommate", "roommates", "relationship_with_roommate", "lives_with",
    "living_arrangement", "household_members", "living_with",
]


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


@pytest.mark.parametrize("rel", VARIANTS)
def test_variants_canonicalize(rel):
    assert canonicalize_profile_relation(rel, "x") == "living_with"


def test_living_with_is_single_valued():
    assert not is_multi_valued_relation("living_with")


def test_new_canonical_supersedes_older_alias_fact(profile):
    t0 = datetime.now() - timedelta(days=200)
    profile.add_fact(relation="living_arrangement", value="live on my own",
                     confidence=0.8, timestamp=t0)
    profile.add_fact(relation="living_with", value="mom", confidence=0.8)
    cur = [f for fs in profile.profile["categories"].values() for f in fs
           if isinstance(f, dict) and f.get("relation") == "living_with"
           and f.get("is_current", True)]
    assert [f["value"] for f in cur] == ["mom"]
    assert all(f["value"] != "live on my own" for f in cur)


def test_alias_written_after_canonical_stays_one_current(profile):
    profile.add_fact(relation="living_with", value="mom", confidence=0.8)
    profile.add_fact(relation="roommate", value="dad", confidence=0.8)
    cur = [f["value"] for fs in profile.profile["categories"].values() for f in fs
           if isinstance(f, dict) and f.get("is_current", True)
           and canonicalize_profile_relation(f.get("relation", "")) == "living_with"]
    assert cur == ["dad"]
