"""A pronoun-led message that brings its own referent is fresh-classified
(2026-10-10, class BC-08/BC-46). 10-08 live: "This is evil holy shit people
defending this <cnn url>" inherited the previous topic "Meeting Confusion"
for three turns because it opened with "This".

Everything drives the DEPLOYED is_anaphoric_continuation / _extract_topics.
"""
import asyncio

from utils.query_checker import is_anaphoric_continuation

CORNELL = (
    "This is evil holy shit people defending this "
    "https://www.cnn.com/2026/10/08/us/cornell-jane-doe-threats-swatting"
)
PRIOR = "User: what time is the meeting\nAssistant: The meeting confusion is about Tuesday."


class _TM:
    def __init__(self, last_topic, fresh="Campus News"):
        self.last_topic = last_topic
        self.fresh = fresh
        self.fresh_calls = 0

    def get_primary_topic(self, text=None):
        self.fresh_calls += 1
        return self.fresh


def _pipeline(tm):
    from core.context_pipeline import ContextPipeline
    p = ContextPipeline.__new__(ContextPipeline)
    p.topic_manager = tm
    return p


class TestUrlBringsOwnReferent:
    def test_cornell_sentence_not_anaphoric(self):
        assert not is_anaphoric_continuation(CORNELL)
        assert not is_anaphoric_continuation(CORNELL, prior_text=PRIOR)

    def test_url_defeats_correction_marker_too(self):
        assert not is_anaphoric_continuation("No I mean this one https://example.com/a")

    def test_plain_fragment_still_anaphoric(self):
        q = "It was maybe 3 years of twice a week"
        assert is_anaphoric_continuation(q)
        assert is_anaphoric_continuation(q, prior_text=PRIOR)

    def test_expletive_it_unchanged(self):
        # "it's" is not in the opener set: False before and after this change.
        assert not is_anaphoric_continuation("it's raining")
        assert not is_anaphoric_continuation("it's raining", prior_text=PRIOR)


class TestNewEntityAgainstPriorText:
    def test_new_proper_noun_absent_from_prior_is_fresh(self):
        q = "It was Cornell that started all of this"
        assert not is_anaphoric_continuation(q, prior_text=PRIOR)

    def test_same_noun_in_prior_stays_continuation(self):
        q = "It was Cornell that started all of this"
        assert is_anaphoric_continuation(
            q, prior_text="We talked about Cornell and the tour."
        )

    def test_no_prior_text_keeps_old_behaviour(self):
        assert is_anaphoric_continuation("It was Cornell that started all of this")


class TestExtractTopicsPath:
    def test_cornell_link_is_fresh_classified(self):
        tm = _TM(last_topic="Meeting Confusion")
        exchange = {"query": "what time is the meeting",
                    "response": "The meeting confusion is about Tuesday."}
        primary, _ = asyncio.run(
            _pipeline(tm)._extract_topics(CORNELL, last_exchange=exchange)
        )
        assert primary == "Campus News"
        assert tm.fresh_calls == 1

    def test_true_continuation_still_inherits(self):
        tm = _TM(last_topic="Meeting Confusion")
        exchange = {"query": "what time is the meeting",
                    "response": "The meeting confusion is about Tuesday."}
        primary, _ = asyncio.run(
            _pipeline(tm)._extract_topics(
                "It was maybe 3 years of twice a week", last_exchange=exchange
            )
        )
        assert primary == "Meeting Confusion"
        assert tm.fresh_calls == 0

    def test_new_entity_via_pipeline_is_fresh(self):
        tm = _TM(last_topic="Meeting Confusion")
        exchange = {"query": "what time is the meeting",
                    "response": "The meeting confusion is about Tuesday."}
        primary, _ = asyncio.run(
            _pipeline(tm)._extract_topics(
                "It was Cornell that started all of this", last_exchange=exchange
            )
        )
        assert primary == "Campus News"
