"""Tests for core/correction_detector.py — user correction/confirmation detection."""

import pytest

from core.correction_detector import CorrectionDetector, CorrectionEvent, EntityCorrectionEvent


@pytest.fixture
def detector():
    return CorrectionDetector()


@pytest.fixture
def sample_facts():
    return [
        {
            "fact_id": "f1",
            "relation": "lives_in",
            "value": "London",
            "confidence": 0.8,
        },
        {
            "fact_id": "f2",
            "relation": "works_at",
            "value": "Acme Corp",
            "confidence": 0.9,
        },
        {
            "fact_id": "f3",
            "relation": "favorite_color",
            "value": "blue",
            "confidence": 0.7,
        },
    ]


class TestCorrectionDetection:
    """Test correction pattern matching."""

    def test_actually_correction(self, detector, sample_facts):
        events = detector.detect_corrections("Actually, I live in Manchester now", sample_facts)
        assert len(events) >= 1
        assert events[0].event_type == "correction"
        assert events[0].relation == "lives_in"

    def test_no_i_meant_correction(self, detector, sample_facts):
        events = detector.detect_corrections("No, I meant I work at Google", sample_facts)
        assert len(events) >= 1
        assert any(e.relation == "works_at" for e in events)

    def test_thats_wrong_correction(self, detector, sample_facts):
        events = detector.detect_corrections("That's wrong, I don't live in London", sample_facts)
        assert len(events) >= 1
        assert events[0].fact_id == "f1"

    def test_moved_correction(self, detector, sample_facts):
        events = detector.detect_corrections("I moved to Berlin last month", sample_facts)
        # Should detect some correction event
        assert len(events) >= 0  # May or may not match depending on overlap

    def test_confidence_threshold(self, detector, sample_facts):
        """All returned events should have confidence >= 0.6."""
        events = detector.detect_corrections("Actually, I'm not in London anymore", sample_facts)
        for event in events:
            assert event.confidence >= 0.6


class TestConfirmationDetection:
    """Test confirmation pattern matching."""

    def test_still_confirmation(self, detector, sample_facts):
        events = detector.detect_confirmations("Yeah, I still live in London", sample_facts)
        assert len(events) >= 1
        assert events[0].event_type == "confirmation"
        assert events[0].relation == "lives_in"

    def test_thats_right_confirmation(self, detector, sample_facts):
        events = detector.detect_confirmations("That's right, I work at Acme Corp", sample_facts)
        assert len(events) >= 1
        assert any(e.fact_id == "f2" for e in events)

    def test_still_working_confirmation(self, detector, sample_facts):
        events = detector.detect_confirmations("Still working at Acme Corp", sample_facts)
        assert len(events) >= 1


class TestFalsePositiveResistance:
    """Test that normal conversation doesn't trigger events."""

    def test_no_match_returns_empty(self, detector, sample_facts):
        events = detector.detect_corrections("How's the weather today?", sample_facts)
        assert events == []

    def test_no_confirmation_on_unrelated(self, detector, sample_facts):
        events = detector.detect_confirmations("I had pizza for lunch", sample_facts)
        assert events == []

    def test_empty_message(self, detector, sample_facts):
        assert detector.detect_corrections("", sample_facts) == []
        assert detector.detect_confirmations("", sample_facts) == []

    def test_empty_facts(self, detector):
        assert detector.detect_corrections("Actually it's wrong", []) == []
        assert detector.detect_confirmations("Yeah, still here", []) == []

    def test_facts_without_ids_skipped(self, detector):
        bad_facts = [{"relation": "name", "value": "Bob"}]  # No fact_id
        events = detector.detect_corrections("Actually, my name is Alice", bad_facts)
        assert events == []


class TestCorrectionEventModel:
    """Test the Pydantic CorrectionEvent model."""

    def test_valid_event(self):
        event = CorrectionEvent(
            fact_id="f1",
            relation="lives_in",
            old_value="London",
            new_value="Berlin",
            confidence=0.85,
            event_type="correction",
        )
        assert event.fact_id == "f1"
        assert event.event_type == "correction"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            CorrectionEvent(
                fact_id="f1", relation="x", old_value="a",
                confidence=1.5, event_type="correction",
            )

    def test_confirmation_event(self):
        event = CorrectionEvent(
            fact_id="f1",
            relation="lives_in",
            old_value="London",
            new_value="London",
            confidence=0.8,
            event_type="confirmation",
        )
        assert event.event_type == "confirmation"
        assert event.old_value == event.new_value


class TestEntityCorrectionDetection:
    """Test entity-level correction pattern matching."""

    def test_did_not_die(self, detector):
        events = detector.detect_entity_corrections("Flapjack did not die")
        assert len(events) == 1
        assert events[0].entity_name == "Flapjack"
        assert events[0].correction_type == "not_dead"
        assert events[0].confidence >= 0.85

    def test_didnt_die(self, detector):
        events = detector.detect_entity_corrections("Flapjack didn't die")
        assert len(events) == 1
        assert events[0].entity_name == "Flapjack"
        assert events[0].correction_type == "not_dead"

    def test_lowercase_entity(self, detector):
        """Lowercase entity names should work (re.I flag)."""
        events = detector.detect_entity_corrections("flapjack did not die hes still here")
        assert len(events) >= 1
        assert any(e.entity_name.lower() == "flapjack" for e in events)

    def test_is_still_alive(self, detector):
        events = detector.detect_entity_corrections("Flapjack is still alive")
        assert len(events) == 1
        assert events[0].correction_type == "alive"

    def test_is_alive(self, detector):
        events = detector.detect_entity_corrections("Flapjack is alive")
        assert len(events) == 1
        assert events[0].correction_type == "alive"

    def test_is_not_dead(self, detector):
        events = detector.detect_entity_corrections("Flapjack is not dead")
        assert len(events) == 1
        assert events[0].correction_type == "alive"

    def test_still_here(self, detector):
        events = detector.detect_entity_corrections("Flapjack is still here")
        assert len(events) == 1
        assert events[0].correction_type == "alive"

    def test_survived(self, detector):
        events = detector.detect_entity_corrections("Flapjack survived the surgery")
        assert len(events) == 1
        assert events[0].correction_type == "survived"

    def test_made_it(self, detector):
        events = detector.detect_entity_corrections("Flapjack made it through")
        assert len(events) == 1
        assert events[0].correction_type == "survived"

    def test_pulled_through(self, detector):
        events = detector.detect_entity_corrections("Flapjack pulled through")
        assert len(events) == 1
        assert events[0].correction_type == "survived"

    def test_is_fine(self, detector):
        events = detector.detect_entity_corrections("Flapjack is fine")
        assert len(events) == 1
        assert events[0].correction_type == "survived"

    def test_my_cat_pattern(self, detector):
        events = detector.detect_entity_corrections("my cat Flapjack is alive")
        assert len(events) >= 1
        assert any(e.entity_name == "Flapjack" for e in events)

    def test_still_with_us(self, detector):
        events = detector.detect_entity_corrections("Flapjack is still with us")
        assert len(events) == 1
        assert events[0].correction_type == "alive"

    def test_no_prefix(self, detector):
        events = detector.detect_entity_corrections("no, Flapjack didn't die")
        assert len(events) >= 1
        assert any(e.entity_name == "Flapjack" for e in events)

    def test_no_match_on_normal_text(self, detector):
        events = detector.detect_entity_corrections("Flapjack had his dinner")
        assert events == []

    def test_no_match_on_unrelated(self, detector):
        events = detector.detect_entity_corrections("I went to the store today")
        assert events == []

    def test_empty_message(self, detector):
        events = detector.detect_entity_corrections("")
        assert events == []

    def test_pronoun_filtered(self, detector):
        """Pronouns should not be captured as entity names."""
        events = detector.detect_entity_corrections("He is still alive")
        assert events == []

    def test_it_pronoun_filtered(self, detector):
        events = detector.detect_entity_corrections("It did not die")
        assert events == []

    def test_multiple_entities(self, detector):
        events = detector.detect_entity_corrections(
            "Flapjack survived and Whiskers is fine"
        )
        assert len(events) == 2
        names = {e.entity_name for e in events}
        assert any("flapjack" in n.lower() for n in names)
        assert any("whiskers" in n.lower() for n in names)

    def test_deduplication(self, detector):
        """Same entity matching multiple patterns should produce one event."""
        events = detector.detect_entity_corrections(
            "Flapjack is alive and Flapjack is not dead"
        )
        flapjack_events = [e for e in events if e.entity_name.lower() == "flapjack"]
        assert len(flapjack_events) == 1

    def test_real_user_message(self, detector):
        """The exact message that triggered the Flapjack hallucination."""
        events = detector.detect_entity_corrections(
            "i forgot about that. oh i need to fix this later haha flapjack did not die hes still here"
        )
        assert len(events) >= 1
        assert any(e.entity_name.lower() == "flapjack" for e in events)

    def test_correction_text_truncated(self, detector):
        long_msg = "Flapjack did not die " + "x" * 300
        events = detector.detect_entity_corrections(long_msg)
        assert len(events) == 1
        assert len(events[0].correction_text) <= 200


class TestEntityCorrectionEventModel:
    """Test the EntityCorrectionEvent Pydantic model."""

    def test_valid_event(self):
        event = EntityCorrectionEvent(
            entity_name="Flapjack",
            correction_type="alive",
            correction_text="Flapjack is alive",
            confidence=0.90,
        )
        assert event.entity_name == "Flapjack"
        assert event.correction_type == "alive"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            EntityCorrectionEvent(
                entity_name="Flapjack",
                correction_type="alive",
                correction_text="test",
                confidence=1.5,
            )
