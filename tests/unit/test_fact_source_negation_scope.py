"""Negation scope + clause-level ownership (memory/fact_source.py, 2026-09-06).

Live incident: the quick profile stored `took=5 mg Zelphex at 11 AM` sourced
to the excerpt "I did not take Zelphex today." — the NEGATED sentence — while
the actual affirmative statement ("Had the extra 5 mg Zelphex at 11 and went
out to movie with a friend and his dad and had drinks.") was rejected by the
provenance layer because it contains the third-party pronoun "his" — in a
clause unrelated to the medication mention. Fixed via clause-level negation
scope + clause-level third-party rejection.
"""

from memory.fact_source import find_supporting_user_span


def _t(subject, relation, obj):
    return {"subject": subject, "relation": relation, "object": obj}


LIVE_MESSAGE = (
    "Yeah but honestly maybe expected this time. Had the extra 5 mg Zelphex "
    "at 11 and went out to movie with a friend and his dad and had drinks. So "
    "that makes sense to me. I did not take Zelphex today. I think I need to "
    "actually take a semi break today"
)


class TestLiveIncident:
    def test_supports_from_the_affirmative_clause_not_the_negated_sentence(self):
        ev = find_supporting_user_span(_t("user", "took", "5 mg Zelphex at 11 AM"), [LIVE_MESSAGE])
        assert ev is not None
        assert "Had the extra 5 mg Zelphex" in ev.text
        assert "did not take" not in ev.text

    def test_negated_sentence_alone_supports_nothing(self):
        assert find_supporting_user_span(
            _t("user", "took", "5 mg Zelphex at 11 AM"), ["I did not take Zelphex today."]
        ) is None


class TestNegationScope:
    def test_simple_negation_blocks_the_object(self):
        assert find_supporting_user_span(
            _t("user", "smokes", "cigarettes"), ["I never smoked cigarettes."]
        ) is None

    def test_contraction_negation_blocks_the_object(self):
        assert find_supporting_user_span(
            _t("user", "attends", "the gym"), ["I didn't go to the gym today."]
        ) is None

    def test_coordinated_clauses_isolate_negation_to_its_own_clause(self):
        text = "I took 5 mg Prozin at 10 but I didn't take the second dose."
        ok = find_supporting_user_span(_t("user", "took", "5 mg Prozin at 10"), [text])
        assert ok is not None
        assert "took 5 mg Prozin at 10" in ok.text
        bad = find_supporting_user_span(_t("user", "took", "second dose"), [text])
        assert bad is None

    def test_third_party_clause_is_not_rescued_by_a_later_first_person_clause(self):
        text = "Sam had 5 mg Prozin at 11 and I went home."
        assert find_supporting_user_span(_t("user", "took", "5 mg Prozin at 11"), [text]) is None

    def test_bare_absence_object_is_not_treated_as_negated(self):
        # "no patient portal" is an affirmative claim about an absence, not a
        # negated claim about the object — regression guard for the existing
        # care-team behavior (bare "no" must not become a negation governor).
        ev = find_supporting_user_span(
            _t("user", "doctor_communication", "no patient portal"),
            ["My psychiatrist has no patient portal at all"],
        )
        assert ev is not None


class TestClauseLevelOwnershipRegression:
    def test_plain_affirmative_first_person_still_supported(self):
        ev = find_supporting_user_span(_t("user", "uses", "WHOOP"), ["I use a WHOOP band to track sleep"])
        assert ev is not None
        assert ev.anchor == "first_person"

    def test_joint_clause_with_partner_still_supported_via_whole_sentence_i(self):
        # "I moved ... and my partner Sarah came too" — the object's own
        # clause ("my partner Sarah came too") is ambiguous on its own, but
        # the sentence's explicit "I" rescues it (existing regression guard).
        turn = "I moved to Atlanta last spring and my partner Sarah came too."
        ev = find_supporting_user_span(_t("user", "relationship", "Sarah"), [turn])
        assert ev is not None and "Sarah" in ev.text

    def test_relative_subject_alone_is_still_rejected(self):
        assert find_supporting_user_span(
            _t("user", "lives_in", "Boston"), ["My brother moved to Boston."]
        ) is None

    def test_implicit_first_person_fragment_still_supported(self):
        ev = find_supporting_user_span(
            _t("user", "takes_medication", "Lorvatin"), ["Started Lorvatin again today"]
        )
        assert ev is not None and ev.anchor == "implicit_first_person"


class TestSingleTokenOverlapMustBeTheHeadNoun:
    def test_non_head_token_overlap_is_not_evidence(self):
        # "blue" alone (a modifier, not the head noun "prius") must not
        # anchor the claim.
        assert find_supporting_user_span(
            _t("user", "owns_car", "blue Prius"),
            ["I really like the color blue for my desk"],
        ) is None

    def test_head_token_overlap_is_evidence(self):
        ev = find_supporting_user_span(
            _t("user", "owns_car", "blue Prius"),
            ["I drive a beat-up old Prius everywhere"],
        )
        assert ev is not None
