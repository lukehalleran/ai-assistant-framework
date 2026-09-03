"""Fact provenance layer (memory/fact_source.py) — 2026-09-02 audit.

Four live contamination classes drove this module:
  - WHOOP: a user turn QUOTING another model's profile comparison.
  - Mochi/Waffles as dogs: relation name asserted a species never stated.
  - lived_in=Atlanta evidence = first 200 chars of a long turn (lyrics).
  - Facts sourced from Daemon's own responses.
"""

from memory.fact_source import (
    EvidenceSpan,
    _span_is_user_owned,
    find_supporting_user_span,
    iter_user_messages,
    supporting_excerpt,
)


def _t(subject, relation, obj):
    return {"subject": subject, "relation": relation, "object": obj}


# ---------------------------------------------------------------------------
# Speaker boundary: assistant text and quoted/reported text never count
# ---------------------------------------------------------------------------

class TestSpeakerBoundary:
    def test_assistant_response_never_supports_a_fact(self):
        msgs = [{"query": "hey", "response": "You use a WHOOP band and play D&D."}]
        assert find_supporting_user_span(_t("user", "uses", "WHOOP"), msgs) is None
        assert find_supporting_user_span(_t("user", "plays", "D&D"), msgs) is None

    def test_role_prefixed_assistant_string_skipped(self):
        turns = list(iter_user_messages(["assistant: I live in Paris", "user: I live in Atlanta"]))
        assert turns == [(1, "I live in Atlanta", "")]

    def test_quoted_other_model_claim_is_not_testimony(self):
        # The live WHOOP fact: the user pasted another model's description.
        msgs = ["Funny, the export says you use WHOOP and do D&D and running."]
        assert find_supporting_user_span(_t("user", "uses", "WHOOP"), msgs) is None
        msgs = ["ChatGPT said I use WHOOP, which is wrong"]
        assert find_supporting_user_span(_t("user", "uses", "WHOOP"), msgs) is None

    def test_first_person_use_statement_is_supported(self):
        ev = find_supporting_user_span(_t("user", "uses", "WHOOP"), ["I use a WHOOP band to track sleep"])
        assert ev is not None
        assert "WHOOP" in ev.text
        assert ev.anchor == "first_person"
        assert ev.role == "user"

    def test_blockquote_and_code_fence_ignored(self):
        text = "> I live in Paris\n```\nI live in Denver\n```\nI live in Atlanta now."
        assert find_supporting_user_span(_t("user", "lives_in", "Paris"), [text]) is None
        assert find_supporting_user_span(_t("user", "lives_in", "Denver"), [text]) is None
        ev = find_supporting_user_span(_t("user", "lives_in", "Atlanta"), [text])
        assert ev is not None and ev.text == "I live in Atlanta now."

    def test_pasted_transcript_inside_user_turn(self):
        text = "here is what it wrote\nAssistant: you live in Denver\nanyway I live in Atlanta"
        assert find_supporting_user_span(_t("user", "lives_in", "Denver"), [text]) is None
        assert find_supporting_user_span(_t("user", "lives_in", "Atlanta"), [text]) is not None


# ---------------------------------------------------------------------------
# Relation-name claims need corroboration
# ---------------------------------------------------------------------------

class TestRelationClaims:
    def test_playing_with_pets_does_not_make_them_dogs(self):
        msgs = ["I was playing with Mochi and Waffles all afternoon"]
        assert find_supporting_user_span(_t("user", "has_dog", "Mochi"), msgs) is None
        assert find_supporting_user_span(_t("user", "has_cat", "Mochi"), msgs) is None

    def test_possessive_species_phrase_supports_pet_relation(self):
        ev = find_supporting_user_span(_t("user", "has_cat", "Biscuit"), ["My cat Biscuit is sitting here"])
        assert ev is not None and ev.anchor == "pet_ownership"
        ev = find_supporting_user_span(_t("user", "pet_name", "Biscuit"), ["My kitten Biscuit is being silly"])
        assert ev is not None

    def test_someone_elses_pet_is_not_the_users(self):
        assert find_supporting_user_span(_t("user", "has_cat", "Clover"), ["my mom's cat Clover is sick"]) is None
        # …but the entity fact about Clover is fine.
        ev = find_supporting_user_span(_t("clover", "species", "cat"), ["my mom's cat Clover is sick"])
        assert ev is not None and ev.anchor == "entity"

    def test_residence_needs_a_residence_cue(self):
        assert find_supporting_user_span(_t("user", "lives_in", "Atlanta"), ["I visited Atlanta for a concert"]) is None
        assert find_supporting_user_span(_t("user", "lives_in", "Atlanta"), ["Moved to Atlanta in June"]) is not None

    def test_preference_needs_a_preference_cue(self):
        assert find_supporting_user_span(_t("user", "likes", "Python"), ["I wrote some Python today"]) is None
        assert find_supporting_user_span(_t("user", "likes", "Python"), ["I really like Python"]) is not None

    def test_unknown_relation_needs_only_grounding_and_anchor(self):
        ev = find_supporting_user_span(_t("user", "knows", "Python"), ["I know Python"])
        assert ev is not None and ev.support == "exact_object"


# ---------------------------------------------------------------------------
# Who is the subject of the sentence?
# ---------------------------------------------------------------------------

class TestSubjectAttribution:
    def test_implicit_first_person_fragment(self):
        assert _span_is_user_owned("Started Lorvatin again today") == "implicit_first_person"
        ev = find_supporting_user_span(_t("user", "takes_medication", "Lorvatin"), ["Started Lorvatin again today"])
        assert ev is not None and ev.anchor == "implicit_first_person"

    def test_capitalised_starter_words_are_not_names(self):
        assert _span_is_user_owned("Today is day six of feeling stable") == "implicit_first_person"

    def test_proper_noun_subject_is_third_party(self):
        assert _span_is_user_owned("Biscuit is being a silly kitten today") is None
        assert find_supporting_user_span(_t("user", "pet_name", "Biscuit"), ["Biscuit is being a silly kitten today"]) is None
        ev = find_supporting_user_span(_t("biscuit", "species", "kitten"), ["Biscuit is being a silly kitten today"])
        assert ev is not None

    def test_reported_third_party_move(self):
        msgs = ["Sam told me he moved to Boston"]
        assert find_supporting_user_span(_t("user", "lives_in", "Boston"), msgs) is None
        ev = find_supporting_user_span(_t("sam", "lives_in", "Boston"), msgs)
        assert ev is not None and ev.anchor == "entity"

    def test_relative_as_subject(self):
        assert _span_is_user_owned("My brother moved to Boston") is None
        assert _span_is_user_owned("My brother and I moved to Boston") == "first_person"

    def test_possessive_proper_noun_marks_third_party(self):
        assert _span_is_user_owned("Sam's new job is at Google, nice for me") is None

    def test_scoped_referent_subject_is_user_owned(self):
        ev = find_supporting_user_span(
            _t("user's unnamed referent", "is", "abusive"), ["she is abusive and I am done"]
        )
        assert ev is not None and ev.anchor == "scoped_referent"


# ---------------------------------------------------------------------------
# Evidence window + provenance metadata
# ---------------------------------------------------------------------------

class TestEvidenceWindow:
    LYRICS = "Like the clouds, see color, drifting over the water, " * 12

    def test_excerpt_is_the_claim_sentence_not_the_head_of_the_turn(self):
        turn = self.LYRICS + " Anyway. I moved to Atlanta last spring and my partner Sarah came too."
        ev = find_supporting_user_span(_t("user", "lives_in", "Atlanta"), [turn])
        assert ev is not None
        assert "Atlanta" in ev.text
        assert len(ev.text) <= 200
        assert "clouds" not in ev.text
        ev = find_supporting_user_span(_t("user", "relationship", "Sarah"), [turn])
        assert ev is not None and "Sarah" in ev.text and "clouds" not in ev.text

    def test_regex_path_helper_crops_to_claim_span(self):
        turn = self.LYRICS + " Anyway. I moved to Atlanta last spring."
        excerpt = supporting_excerpt(turn, "Atlanta", limit=200)
        assert excerpt == "I moved to Atlanta last spring."
        # A run-on with no sentence punctuation: the window is anchored on the
        # claim (tail of the turn), never the head.
        excerpt = supporting_excerpt(self.LYRICS + " I moved to Atlanta last spring.", "Atlanta", limit=200)
        assert excerpt.endswith("I moved to Atlanta last spring.") and len(excerpt) <= 200
        # Object absent from any span: still bounded, never crashes.
        assert len(supporting_excerpt(turn, "Zanzibar", limit=120)) <= 120
        assert supporting_excerpt("", "x") == ""

    def test_turn_metadata_and_newest_tie_break(self):
        msgs = [
            {"query": "I like Python", "timestamp": "2026-09-01T10:00:00"},
            {"query": "I like Python a lot", "timestamp": "2026-09-02T10:00:00"},
        ]
        ev = find_supporting_user_span(_t("user", "likes", "Python"), msgs)
        assert ev == EvidenceSpan(
            text="I like Python a lot", turn_index=1, role="user",
            support="exact_object", turn_id="2026-09-02T10:00:00", anchor="first_person",
        )

    def test_empty_and_malformed_inputs(self):
        assert find_supporting_user_span(_t("user", "likes", "Python"), []) is None
        assert find_supporting_user_span({"subject": "user", "relation": "likes"}, ["I like Python"]) is None
        assert find_supporting_user_span(_t("user", "", "Python"), ["I like Python"]) is None
