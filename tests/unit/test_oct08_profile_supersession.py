"""2026-10-08 (class: BC-16, BC-58, BC-51, BC-55): profile supersession.

The 10-08 shutdown extraction retired one current medication when another
arrived, one food when another arrived, and stored a bare
medication_dose='200 mg' that any reader pairs with the current drug. These
tests drive THE deployed ``UserProfile.add_fact`` on a synthetic tmp profile
(never the live data/ profile).

Base-failure receipt (recorded in the handoff, not asserted here): on the base
tree the multi-valued tests fail because Case 2 supersedes every current value.
"""

from __future__ import annotations

import copy

import pytest

from memory import relation_classifier as rc
from memory.user_profile import UserProfile


@pytest.fixture
def profile(tmp_path):
    return UserProfile(profile_path=str(tmp_path / "profile.json"))


def _facts(profile, relation):
    out = []
    for facts in profile.profile["categories"].values():
        out.extend(f for f in facts if isinstance(f, dict) and f.get("relation") == relation)
    return out


def _current_values(profile, relation):
    return sorted(f["value"] for f in _facts(profile, relation) if f.get("is_current", True))


def _is_current(profile, relation, value):
    return [f for f in _facts(profile, relation) if f["value"] == value][0].get("is_current", True)


class TestMultiValuedAppend:
    def test_medication_name_both_current(self, profile):
        assert profile.add_fact("medication_name", "kavarin", 0.8, "took kavarin")
        assert profile.add_fact("medication_name", "Lorvatin", 0.8, "started Lorvatin")
        assert _current_values(profile, "medication_name") == ["Lorvatin", "kavarin"]

    def test_condition_both_current(self, profile):
        profile.add_fact("condition", "long covid", 0.8, "x")
        profile.add_fact("condition", "awful migraines", 0.8, "y")
        assert len(_current_values(profile, "condition")) == 2

    def test_eats_both_current(self, profile):
        profile.add_fact("eats", "sushi", 0.8, "x")
        profile.add_fact("eats", "one cracker a day", 0.8, "y")
        assert _current_values(profile, "eats") == ["one cracker a day", "sushi"]

    def test_multi_valued_append_has_no_supersedes_link(self, profile):
        profile.add_fact("medication_name", "kavarin", 0.8, "x")
        profile.add_fact("medication_name", "Lorvatin", 0.8, "y")
        new = [f for f in _facts(profile, "medication_name") if f["value"] == "Lorvatin"][0]
        assert not new.get("supersedes")

    def test_single_valued_relation_still_supersedes(self, profile):
        assert "lives_in" not in rc.MULTI_VALUED_RELATIONS
        profile.add_fact("lives_in", "Austin", 0.8, "x")
        profile.add_fact("lives_in", "Denver", 0.8, "y")
        assert _current_values(profile, "lives_in") == ["Denver"]
        assert _is_current(profile, "lives_in", "Austin") is False


class TestKeyedDoseSupersession:
    def test_same_drug_dose_revision_supersedes(self, profile):
        profile.add_fact("medication_dose", "kavarin 300 mg", 0.8, "x")
        profile.add_fact("medication_dose", "200 mg kavarin daily", 0.8, "y")
        assert _is_current(profile, "medication_dose", "kavarin 300 mg") is False
        assert _is_current(profile, "medication_dose", "200 mg kavarin daily") is True

    def test_other_drug_dose_stays_current(self, profile):
        profile.add_fact("medication_dose", "kavarin 300 mg", 0.8, "x")
        profile.add_fact("medication_dose", "200 mg kavarin daily", 0.8, "y")
        profile.add_fact("medication_dose", "Lorvatin 30 mg", 0.8, "z")
        assert _is_current(profile, "medication_dose", "Lorvatin 30 mg") is True
        assert _is_current(profile, "medication_dose", "200 mg kavarin daily") is True
        assert _is_current(profile, "medication_dose", "kavarin 300 mg") is False


    def test_bare_dose_rejected_and_profile_unchanged(self, profile):
        profile.add_fact("medication_name", "Lorvatin", 0.8, "x")
        before = copy.deepcopy(profile.profile)
        assert profile.add_fact("medication_dose", "200 mg", 0.8, "So I am taking 200 a day now") is False
        assert profile.profile == before

    def test_bare_dose_variants_rejected(self, profile):
        for v in ("200 mg", "5mg", "2 tablets daily", "once a day"):
            assert profile.add_fact("medication_dose", v, 0.8, "s") is False, v

class TestCase1Recurrent:
    def test_recurrent_of_one_medication_keeps_the_other(self, profile):
        profile.add_fact("medication_name", "kavarin", 0.8, "x")
        profile.add_fact("medication_name", "Lorvatin", 0.8, "y")
        # make kavarin historical, then confirm it again
        for f in _facts(profile, "medication_name"):
            if f["value"] == "kavarin":
                f["is_current"] = False
        profile.add_fact("medication_name", "kavarin", 0.8, "still on kavarin")
        assert _is_current(profile, "medication_name", "kavarin") is True
        assert _is_current(profile, "medication_name", "Lorvatin") is True

    def test_recurrent_dose_retires_only_same_key(self, profile):
        profile.add_fact("medication_dose", "kavarin 300 mg", 0.8, "x")
        profile.add_fact("medication_dose", "Lorvatin 30 mg", 0.8, "y")
        profile.add_fact("medication_dose", "kavarin 200 mg", 0.8, "z")  # retires 300
        assert _is_current(profile, "medication_dose", "kavarin 300 mg") is False
        profile.add_fact("medication_dose", "kavarin 300 mg", 0.8, "back to 300")
        assert _is_current(profile, "medication_dose", "kavarin 300 mg") is True
        assert _is_current(profile, "medication_dose", "kavarin 200 mg") is False
        assert _is_current(profile, "medication_dose", "Lorvatin 30 mg") is True


class TestClassifierHelpers:
    @pytest.mark.parametrize("value,key", [
        ("kavarin 300 mg", "kavarin"),
        ("200 mg kavarin daily", "kavarin"),
        ("30 mg by lorvtn", "lorvtn"),
        ("Lorvatin 30 mg twice a day", "lorvatin"),
        ("200 mg", None),
        ("", None),
    ])
    def test_supersession_key_dose(self, value, key):
        assert rc.supersession_key("medication_dose", value) == key

    def test_supersession_key_none_for_unkeyed_relation(self):
        assert rc.supersession_key("eats", "sushi") is None
        assert rc.supersession_key("lives_in", "Austin") is None

    def test_dose_value_lacks_referent(self):
        assert rc.dose_value_lacks_referent("medication_dose", "200 mg") is True
        assert rc.dose_value_lacks_referent("medication_dose", "Lorvatin 200 mg") is False
        assert rc.dose_value_lacks_referent("eats", "200 mg") is False
