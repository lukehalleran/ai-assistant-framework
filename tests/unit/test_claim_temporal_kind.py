"""B2 (2026-09-06): claim temporal kind + event date.

`memory.fact_source.classify_claim_time` classifies what KIND of claim a
span makes (event/state/habit/plan/unknown) and, for a past EVENT, resolves
the calendar date when possible. This closes the general case behind the
Zelphex incident: an object phrase can claim "today" while its only
supporting evidence describes YESTERDAY (see
test_extraction_replay_provenance.py) -- read-side consumers now use the
EVIDENCE's own temporal kind/date to decide what is "current", regardless of
what the object text says.

FAILED-before evidence (recorded in docs/HANDOFF_20260906_context_integrity_phaseAB.md,
NOT asserted here): before 2026-09-06 `classify_claim_time`/`ClaimTime` did not
exist and `UserProfile.add_fact` took no `claim_kind`/`event_date` kwarg. Tests
must never read git state (`git show HEAD:` was green only while the change was
uncommitted and went red the moment it was committed — CI run 34146393709);
`tests/unit/test_no_git_state_in_tests.py` guards the class.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from memory.fact_source import ClaimTime, classify_claim_time, find_supporting_user_span
from memory.llm_fact_extractor import LLMFactExtractor, _normalize_triple


# ---------------------------------------------------------------------------
# classify_claim_time — grammar-level kind classification
# ---------------------------------------------------------------------------

OBSERVED = datetime(2026, 9, 6, 12, 0, 0)  # a Sunday


class TestClassifyClaimTimeKinds:
    def test_habit_every(self):
        ct = classify_claim_time("I take 5mg of Zelphex every night.", observed_at=OBSERVED)
        assert ct.kind == "habit"
        assert ct.event_date is None

    def test_habit_each_week(self):
        ct = classify_claim_time("I go to the gym each week.", observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_habit_simple_present_frequency(self):
        ct = classify_claim_time("I take it twice a day.", observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_event_yesterday(self):
        ct = classify_claim_time("Took an extra 5mg of Zelphex yesterday.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 9, 5)
        assert ct.event_date_source == "relative"

    def test_event_days_ago(self):
        ct = classify_claim_time("Took an extra dose 3 days ago.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 9, 3)

    def test_event_last_night(self):
        ct = classify_claim_time("Had a rough night last night.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 9, 5)

    def test_event_this_morning(self):
        ct = classify_claim_time("Felt dizzy this morning.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 9, 6)

    def test_event_on_weekday_resolves_most_recent_occurrence(self):
        # OBSERVED is Sunday 2026-09-06; "on Monday" -> most recent Monday.
        ct = classify_claim_time("Took the extra dose on Monday.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 8, 31)

    def test_event_last_weekday(self):
        ct = classify_claim_time("Skipped it last Tuesday.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 9, 1)

    def test_event_ambiguous_last_week_has_no_resolvable_date(self):
        """"last week" cannot be a single day -- kind stays event, date is
        unresolved (per spec: unknown date -> event_date=None, kind still
        "event")."""
        ct = classify_claim_time("Took an extra dose last week.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date is None
        assert ct.event_date_source == "none"

    def test_event_explicit_date_uses_resolve_date_expression(self):
        """Uses THE deployed utils.temporal_resolver.resolve_date_expression,
        which is future-biased by design (scheduling callers): a month+day
        already passed this year rolls to NEXT year. A past-tense claim
        cannot describe a future date, so classify_claim_time rolls the
        resolved date back one year when it lies after the observation
        (referee fix, 2026-09-06)."""
        ct = classify_claim_time("Took the extra dose on September 5th.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date_source == "explicit"
        assert ct.event_date == date(2026, 9, 5)

    def test_event_explicit_date_earlier_this_year_is_not_rolled(self):
        ct = classify_claim_time("Started the new job on March 3rd.", observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date is not None
        assert ct.event_date <= OBSERVED.date()

    def test_past_tense_without_anchor_is_not_classified_as_event(self):
        """A past-tense verb with NO temporal anchor at all is under-fired to
        unknown rather than guessed as an event -- an anchor is required."""
        ct = classify_claim_time("Took an extra dose.", observed_at=OBSERVED)
        assert ct.kind != "event"

    def test_plan_going_to(self):
        ct = classify_claim_time("Going to take an extra dose tomorrow.", observed_at=OBSERVED)
        assert ct.kind == "plan"

    def test_plan_will(self):
        ct = classify_claim_time("I will take it tomorrow.", observed_at=OBSERVED)
        assert ct.kind == "plan"

    def test_state_present_progressive(self):
        ct = classify_claim_time("I am feeling better today.", observed_at=OBSERVED)
        assert ct.kind == "state"

    def test_state_ive_been(self):
        ct = classify_claim_time("I've been more tired lately.", observed_at=OBSERVED)
        assert ct.kind == "state"

    def test_unknown_no_recognized_cue(self):
        ct = classify_claim_time("Zelphex is a fictional medication.", observed_at=OBSERVED)
        assert ct.kind == "unknown"
        assert ct.event_date is None

    def test_empty_span_is_unknown(self):
        assert classify_claim_time("", observed_at=OBSERVED).kind == "unknown"

    def test_default_observed_at_is_now_when_omitted(self):
        # Doesn't raise, produces a real ClaimTime regardless of caller
        # supplying an anchor.
        ct = classify_claim_time("Took an extra dose yesterday.")
        assert isinstance(ct, ClaimTime)
        assert ct.kind == "event"
        assert ct.event_date == date.today() - timedelta(days=1)

    def test_habit_wins_over_event_anchor(self):
        """A habitual cue takes precedence even when a past-tense event verb
        and an anchor both happen to be present in the same span."""
        ct = classify_claim_time(
            "I always took my dose every night, including yesterday.",
            observed_at=OBSERVED,
        )
        assert ct.kind == "habit"


# ---------------------------------------------------------------------------
# Forwarding: find_supporting_user_span attaches claim_kind/event_date/
# observed_at to EvidenceSpan (the LLM shutdown path's join primitive).
# ---------------------------------------------------------------------------

class TestEvidenceSpanForwarding:
    def test_event_kind_and_date_attached(self):
        msg = "Took an extra 5mg of Zelphex yesterday."
        ev = find_supporting_user_span(
            {"subject": "user", "relation": "medication_dose", "object": "extra 5mg"}, [msg],
        )
        assert ev is not None
        assert ev.claim_kind == "event"
        assert ev.event_date  # non-empty ISO string
        assert ev.observed_at  # non-empty ISO string

    def test_habit_kind_has_no_event_date(self):
        msg = "I take 5mg of Zelphex every night."
        ev = find_supporting_user_span(
            {"subject": "user", "relation": "medication_dose", "object": "5mg"}, [msg],
        )
        assert ev is not None
        assert ev.claim_kind == "habit"
        assert ev.event_date == ""

    def test_observed_at_uses_turn_timestamp_when_available(self):
        msg = {"query": "Took an extra 5mg of Zelphex yesterday.",
               "timestamp": "2026-09-06T08:00:00"}
        ev = find_supporting_user_span(
            {"subject": "user", "relation": "medication_dose", "object": "extra 5mg"}, [msg],
        )
        assert ev is not None
        assert ev.observed_at.startswith("2026-09-06T08:00:00")
        assert ev.event_date == "2026-09-05"


class TestLLMPathForwarding:
    """The shutdown LLM path's provenance join (_attach_source_excerpts)
    forwards claim_kind/event_date/observed_at onto the surviving triple."""

    def test_event_fields_forwarded_onto_triple(self):
        raw = {"subject": "user", "relation": "medication_dose", "object": "extra 5mg",
               "confidence": 0.8}
        norm = _normalize_triple(raw)
        triples = [norm]
        LLMFactExtractor._attach_source_excerpts(
            triples, ["Took an extra 5mg of Zelphex yesterday."])
        assert triples
        assert triples[0]["claim_kind"] == "event"
        assert triples[0]["event_date"]
        assert triples[0]["observed_at"]

    def test_no_new_keys_when_kind_absent(self):
        """classify_claim_time never returns an empty claim_kind for a
        non-empty span (always at least "unknown"), so this documents that
        an "unknown" classification still forwards a claim_kind key (the
        read side treats "unknown" as untouched, same as no key at all)."""
        raw = {"subject": "user", "relation": "hobby", "object": "chess", "confidence": 0.8}
        norm = _normalize_triple(raw)
        triples = [norm]
        LLMFactExtractor._attach_source_excerpts(triples, ["I've been playing chess lately."])
        assert triples
        assert triples[0].get("claim_kind") in {"state", "unknown", "habit"}


# ---------------------------------------------------------------------------
# Regex path forwarding: FactExtractor._to_node attaches the same metadata.
# ---------------------------------------------------------------------------

class TestRegexPathForwarding:
    @pytest.fixture(scope="class")
    def extractor(self):
        from memory.fact_extractor import FactExtractor
        return FactExtractor(use_rebel=False, use_regex=True)

    def test_to_node_attaches_event_kind_and_date(self, extractor):
        node = extractor._to_node(
            "user", "medication_dose", "5mg", 0.8, "regex",
            "Took an extra 5mg of Zelphex yesterday. Didn't take any today.",
        )
        assert node.metadata["claim_kind"] == "event"
        assert node.metadata["event_date"]
        assert node.metadata["observed_at"]

    def test_to_node_attaches_habit_kind_with_no_date(self, extractor):
        node = extractor._to_node(
            "user", "medication_dose", "5mg", 0.8, "regex",
            "I take 5mg of Zelphex every night.",
        )
        assert node.metadata["claim_kind"] == "habit"
        assert node.metadata["event_date"] == ""


# ---------------------------------------------------------------------------
# Read side: user_profile.get_category stops projecting a past EVENT as
# current state, regardless of the fact VALUE's own wording.
# ---------------------------------------------------------------------------

class TestGetCategoryEventFiltering:
    @pytest.fixture
    def profile(self, tmp_path):
        from memory.user_profile import UserProfile
        return UserProfile(profile_path=str(tmp_path / "profile.json"))

    def test_past_event_fact_excluded_by_default(self, profile):
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact(
            "medication_dose", "5mg", 0.8,
            "Took an extra 5mg of Zelphex yesterday.",
            claim_kind="event", event_date="2026-09-05",
        )
        current = profile.get_category(ProfileCategory.HEALTH)
        assert current == []

    def test_past_event_fact_visible_with_include_historical(self, profile):
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact(
            "medication_dose", "5mg", 0.8,
            "Took an extra 5mg of Zelphex yesterday.",
            claim_kind="event", event_date="2026-09-05",
        )
        historical = profile.get_category(ProfileCategory.HEALTH, include_historical=True)
        assert len(historical) == 1
        assert historical[0]["claim_kind"] == "event"

    def test_habit_fact_is_not_filtered(self, profile):
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact(
            "exercise_routine", "daily jog", 0.8,
            "I go for a jog every morning.", claim_kind="habit",
        )
        current = profile.get_category(ProfileCategory.FITNESS)
        assert len(current) == 1

    def test_state_fact_is_not_filtered(self, profile):
        from memory.user_profile_schema import categorize_relation
        profile.add_fact(
            "mood", "content", 0.8, "I am feeling content today.", claim_kind="state",
        )
        cat = categorize_relation("mood")
        current = profile.get_category(cat)
        assert len(current) == 1

    def test_unknown_kind_fact_is_not_filtered(self, profile):
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact(
            "hobby", "chess", 0.8, "I like chess.", claim_kind="unknown",
        )
        current = profile.get_category(ProfileCategory.HOBBIES)
        assert len(current) == 1

    def test_legacy_fact_with_no_claim_kind_key_unaffected(self, profile):
        """A fact stored with no claim_kind at all (legacy, or the caller
        simply didn't pass one) behaves exactly as before -- get_category
        never treats absence as "filter it out"."""
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact("hobby", "painting", 0.8, "I like painting.")
        current = profile.get_category(ProfileCategory.HOBBIES)
        assert len(current) == 1
        assert "claim_kind" not in current[0]

    def test_future_dated_event_is_not_filtered(self, profile):
        """A past EVENT before today is filtered; an event whose resolved
        date is NOT before today (e.g. a same-day event) is not."""
        from memory.user_profile_schema import ProfileCategory
        profile.add_fact(
            "medication_dose", "5mg", 0.8, "Felt dizzy this morning.",
            claim_kind="event", event_date=date.today().isoformat(),
        )
        current = profile.get_category(ProfileCategory.HEALTH)
        assert len(current) == 1

    def test_add_facts_batch_forwards_claim_kind_and_event_date(self, profile):
        from memory.user_profile_schema import ProfileCategory
        added = profile.add_facts_batch([{
            "relation": "medication_dose", "value": "5mg", "confidence": 0.8,
            "source_excerpt": "Took an extra 5mg of Zelphex yesterday.",
            "claim_kind": "event", "event_date": "2026-09-05",
        }])
        assert added == 1
        assert profile.get_category(ProfileCategory.HEALTH) == []
        assert len(profile.get_category(ProfileCategory.HEALTH, include_historical=True)) == 1


# --- referee additions (2026-09-06): observed_at threading ------------------

class TestObservedAtThreading:
    """The regex extractor resolves relative anchors against the TURN's own
    timestamp when the caller passes one (shutdown replay), else now."""

    @pytest.mark.asyncio
    async def test_extract_facts_uses_caller_observed_at(self):
        """The regex extractor deliberately under-fires on many shapes, so the
        contract is checked at the seam: extract_facts(observed_at=) stores
        the anchor and _to_node resolves the excerpt's relative anchor
        against it (not against now)."""
        from datetime import datetime
        from memory.fact_extractor import FactExtractor
        fx = FactExtractor()
        turn_time = datetime(2026, 9, 1, 9, 0)
        await fx.extract_facts("I took my Zelphex yesterday morning.", "", observed_at=turn_time)
        node = fx._to_node(
            subject="user", relation="medication_taken", object="Zelphex",
            confidence=0.8, method="regex",
            source_text="I took my Zelphex yesterday morning.",
        )
        assert node.metadata.get("observed_at") == turn_time.isoformat()
        assert node.metadata.get("claim_kind") == "event"
        assert node.metadata.get("event_date") == "2026-08-31"

    @pytest.mark.asyncio
    async def test_extract_facts_without_observed_at_uses_now(self):
        from datetime import datetime
        from memory.fact_extractor import FactExtractor
        fx = FactExtractor()
        await fx.extract_facts("Skipped it this morning.", "")
        node = fx._to_node(
            subject="user", relation="medication_taken", object="Zelphex",
            confidence=0.8, method="regex", source_text="Skipped it this morning.",
        )
        assert node.metadata.get("event_date") == datetime.now().date().isoformat()
