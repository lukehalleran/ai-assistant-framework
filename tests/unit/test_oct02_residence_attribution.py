"""2026-10-02: a residence relation must be supported by a span where the USER
is the resident. A place possessed by another person ("my dad's, which is on
Bartlet") never supports user lives_in (live case: Bartlet via a visit to
the user's dad)."""
from __future__ import annotations

import pytest

from memory.fact_source import find_supporting_user_span
from memory.user_profile_schema import RESIDENCE_RELATIONS


def _t(obj, relation="lives_in"):
    return {"subject": "user", "relation": relation, "object": obj}


class TestResidenceAttribution:
    def test_live_dad_case_does_not_support_lives_in(self):
        msgs = ["I will have to go to my dad's first and then drive back to hang with Alex",
                "my dad's on Saturday which is on Bartlet"]
        assert find_supporting_user_span(_t("Bartlet"), msgs) is None

    def test_staying_at_dads_does_not_support_lives_in(self):
        # carries a residence cue ("stay") — the pre-fix code accepted this
        msgs = ["I'll stay at my dad's on Saturday which is on Bartlet"]
        assert find_supporting_user_span(_t("Bartlet"), msgs) is None

    def test_mom_place_in_city(self):
        assert find_supporting_user_span(_t("Dallas"), ["my mom's place where I'm staying is in Dallas"]) is None

    def test_proper_name_possessive(self):
        assert find_supporting_user_span(_t("Dallas"), ["staying at Alex's house in Dallas"]) is None

    def test_user_resident_still_supported(self):
        span = find_supporting_user_span(_t("Bartlett"), ["I live in Bartlett now"])
        assert span is not None and span.anchor == "first_person"

    def test_moved_still_supported(self):
        assert find_supporting_user_span(_t("Tulsa"), ["I moved to Tulsa last year"]) is not None

    def test_live_at_dads_with_explicit_resident_verb_supported(self):
        assert find_supporting_user_span(_t("Dallas"), ["I live at my dad's in Dallas"]) is not None

    def test_contraction_s_is_not_a_possessive(self):
        assert find_supporting_user_span(_t("Bartlett"), ["Let's see, I live in Bartlett"]) is not None

    def test_other_person_resident_still_rejected(self):
        assert find_supporting_user_span(_t("Dallas"), ["My dad lives in Dallas"]) is None

    @pytest.mark.parametrize("rel", ["lives_in", "living_with", "home_location"])
    def test_every_residence_relation_gated(self, rel):
        assert rel in RESIDENCE_RELATIONS
        assert find_supporting_user_span(
            _t("Dallas", rel), ["I'm at my mom's home in Dallas"]) is None

    def test_non_residence_relation_unaffected(self):
        # works_at near a possessive is NOT a residence claim
        assert find_supporting_user_span(
            {"subject": "user", "relation": "works_at", "object": "Acme"},
            ["I work at Acme, my dad's old company"]) is not None
