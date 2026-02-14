"""
Tests for core/escalation_tracker.py — session-level emotional momentum tracking.

Covers:
- Initial state and defaults
- Strategy transitions based on consecutive elevated messages
- Engagement detection (positive and negative)
- De-escalation detection (calming down vs. intensity shift)
- Escalation velocity calculation
- Token budget overrides
- Strategy-specific instruction content
- Reset behavior
- Edge cases (mixed escalation, CONCERN not counted, etc.)
"""

import pytest
from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.context_pipeline import ToneLevel


class TestInitialState:
    """Test default initialization."""

    def test_default_strategy(self):
        tracker = EscalationTracker()
        assert tracker.current_strategy == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_default_counts(self):
        tracker = EscalationTracker()
        assert tracker.consecutive_elevated_count == 0
        assert tracker.consecutive_calm_count == 0
        assert tracker.ignored_suggestion_count == 0

    def test_empty_history(self):
        tracker = EscalationTracker()
        assert len(tracker.tone_history) == 0
        assert len(tracker.last_suggestions) == 0

    def test_custom_config(self):
        tracker = EscalationTracker(
            escalation_threshold=5,
            deescalation_window=3,
            max_history=20,
        )
        assert tracker.escalation_threshold == 5
        assert tracker.deescalation_window == 3
        assert tracker.max_history == 20


class TestBasicStrategyTransitions:
    """Test core strategy transition logic."""

    def setup_method(self):
        self.tracker = EscalationTracker(escalation_threshold=3)

    def test_conversational_stays_validate(self):
        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "hey how's it going")
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_single_elevated_stays_validate(self):
        strategy = self.tracker.update(ToneLevel.ELEVATED, "I'm not okay")
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST
        assert self.tracker.consecutive_elevated_count == 1

    def test_two_elevated_stays_validate(self):
        self.tracker.update(ToneLevel.ELEVATED, "I'm not okay")
        strategy = self.tracker.update(ToneLevel.CRISIS, "Everything is falling apart")
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST
        assert self.tracker.consecutive_elevated_count == 2

    def test_three_consecutive_crisis_shifts_to_grounding(self):
        self.tracker.update(ToneLevel.CRISIS, "I can't take this")
        self.tracker.update(ToneLevel.CRISIS, "Everything is falling apart")
        strategy = self.tracker.update(ToneLevel.CRISIS, "I'm spiraling")
        assert strategy == ResponseStrategy.GROUNDING_PRESENCE

    def test_three_mixed_elevated_shifts_to_grounding(self):
        """ELEVATED and CRISIS both count toward escalation."""
        self.tracker.update(ToneLevel.ELEVATED, "I'm stressed")
        self.tracker.update(ToneLevel.CRISIS, "I can't handle this")
        strategy = self.tracker.update(ToneLevel.ELEVATED, "This is too much")
        assert strategy == ResponseStrategy.GROUNDING_PRESENCE

    def test_concern_does_not_count_as_elevated(self):
        """CONCERN level should not count toward escalation threshold."""
        self.tracker.update(ToneLevel.CONCERN, "I'm a bit worried")
        self.tracker.update(ToneLevel.CONCERN, "Still worried")
        self.tracker.update(ToneLevel.CONCERN, "Worried more")
        assert self.tracker.consecutive_elevated_count == 0
        assert self.tracker.current_strategy == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_mixed_escalation_resets_count(self):
        """A conversational message between crisis messages resets the count."""
        self.tracker.update(ToneLevel.CRISIS, "bad")
        self.tracker.update(ToneLevel.CRISIS, "worse")
        self.tracker.update(ToneLevel.CONVERSATIONAL, "ok never mind")
        strategy = self.tracker.update(ToneLevel.CRISIS, "no actually bad again")
        assert self.tracker.consecutive_elevated_count == 1
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST


class TestIgnoredSuggestions:
    """Test strategy shift when suggestions are consistently ignored."""

    def setup_method(self):
        self.tracker = EscalationTracker(escalation_threshold=3)

    def test_ignored_suggestions_shift_to_quiet(self):
        """
        GROUNDING always precedes QUIET. At threshold → GROUNDING,
        then past threshold + 2 ignored suggestions → QUIET_COMPANIONSHIP.
        """
        # First response with suggestions
        self.tracker.record_response("Try going for a walk. Consider calling a friend.")

        # Elevated message ignoring suggestions
        self.tracker.update(ToneLevel.CRISIS, "I'm screaming")
        self.tracker.record_response("Your body is doing what it needs to. Try breathing.")

        # Another elevated message ignoring suggestions
        self.tracker.update(ToneLevel.CRISIS, "AHHHH")
        self.tracker.record_response("I hear you. Maybe try some water?")

        # Third elevated message — hits threshold → GROUNDING first
        strategy = self.tracker.update(ToneLevel.CRISIS, "MAKE IT STOP")
        assert strategy == ResponseStrategy.GROUNDING_PRESENCE

        # Grounding response has no suggestions
        self.tracker.record_response("I'm here. This is hard.")

        # Fourth elevated message — past threshold + 2 ignored → QUIET
        strategy = self.tracker.update(ToneLevel.CRISIS, "NOTHING HELPS")
        assert strategy == ResponseStrategy.QUIET_COMPANIONSHIP

    def test_engagement_reduces_ignored_count(self):
        """If user engages with suggestion, ignored count decreases."""
        self.tracker.record_response("Try going for a walk.")
        self.tracker.update(ToneLevel.CRISIS, "AAAA")  # ignored, count = 1

        self.tracker.record_response("Consider deep breathing.")
        self.tracker.update(ToneLevel.CRISIS, "STILL BAD")  # ignored, count = 2

        self.tracker.record_response("Maybe try calling someone.")
        # User engages with a suggestion
        self.tracker.update(ToneLevel.ELEVATED, "I tried calling my friend, didn't help")
        # Engaged → count reduced from 2 to 1
        assert self.tracker.ignored_suggestion_count == 1


class TestDeescalation:
    """Test de-escalation detection and strategy transitions."""

    def setup_method(self):
        self.tracker = EscalationTracker(escalation_threshold=3, deescalation_window=2)

    def test_deescalation_triggers_gentle_reengagement(self):
        """Dropping from crisis to conversational → GENTLE_REENGAGEMENT."""
        self.tracker.update(ToneLevel.CRISIS, "I can't do this")
        self.tracker.update(ToneLevel.CRISIS, "Everything hurts")
        self.tracker.update(ToneLevel.CRISIS, "I'm so angry")

        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "ok I feel a bit better now")
        assert strategy == ResponseStrategy.GENTLE_REENGAGEMENT

    def test_deescalation_with_perspective_returns_to_validate(self):
        """User shifting to analytical mode — skip gentle, go straight to validate."""
        self.tracker.update(ToneLevel.CRISIS, "I can't do this")
        self.tracker.update(ToneLevel.CRISIS, "Everything hurts")
        self.tracker.update(ToneLevel.CRISIS, "I'm so angry")

        # User shifts to PERSPECTIVE need type — they want engagement, not gentle handling
        strategy = self.tracker.update(
            ToneLevel.CONCERN,
            "actually let me think about why that happened",
            need_type="PERSPECTIVE",
        )
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_deescalation_with_presence_stays_gentle(self):
        """User calming but still emotional — stay gentle."""
        self.tracker.update(ToneLevel.CRISIS, "I can't do this")
        self.tracker.update(ToneLevel.CRISIS, "Everything hurts")
        self.tracker.update(ToneLevel.CRISIS, "I'm so angry")

        strategy = self.tracker.update(
            ToneLevel.CONCERN,
            "I'm just tired...",
            need_type="PRESENCE",
        )
        assert strategy == ResponseStrategy.GENTLE_REENGAGEMENT

    def test_gentle_reengagement_persists_during_window(self):
        """GENTLE_REENGAGEMENT stays for deescalation_window messages."""
        self.tracker.update(ToneLevel.CRISIS, "bad")
        self.tracker.update(ToneLevel.CRISIS, "worse")
        self.tracker.update(ToneLevel.CRISIS, "worst")

        # First calm message → GENTLE
        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "better now")
        assert strategy == ResponseStrategy.GENTLE_REENGAGEMENT

        # Second calm message → still GENTLE (window=2)
        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "yeah ok")
        assert strategy == ResponseStrategy.GENTLE_REENGAGEMENT

    def test_gentle_reengagement_ends_after_window(self):
        """After deescalation_window calm messages, the NEXT one returns to VALIDATE."""
        self.tracker.update(ToneLevel.CRISIS, "bad")
        self.tracker.update(ToneLevel.CRISIS, "worse")
        self.tracker.update(ToneLevel.CRISIS, "worst")

        # 1st calm → GENTLE (in window)
        self.tracker.update(ToneLevel.CONVERSATIONAL, "better")
        # 2nd calm → GENTLE (still in window, this IS the 2nd message)
        self.tracker.update(ToneLevel.CONVERSATIONAL, "ok")
        # 3rd calm → past window (consec_calm=3 > deescalation_window=2)
        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "let's talk about something else")
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_no_deescalation_without_prior_escalation(self):
        """Conversational messages without prior escalation stay VALIDATE."""
        self.tracker.update(ToneLevel.CONVERSATIONAL, "hi")
        strategy = self.tracker.update(ToneLevel.CONVERSATIONAL, "how are you")
        assert strategy == ResponseStrategy.VALIDATE_AND_SUGGEST


class TestEngagementDetection:
    """Test the engagement heuristic."""

    def setup_method(self):
        self.tracker = EscalationTracker()

    def test_explicit_engagement_phrase(self):
        result = self.tracker._detect_engagement(
            "I tried walking, it helped a bit",
            ["Try going for a walk."],
        )
        assert result is True

    def test_thanks_is_engagement(self):
        result = self.tracker._detect_engagement(
            "thanks for the suggestion",
            ["Consider taking a break."],
        )
        assert result is True

    def test_keyword_overlap_engagement(self):
        result = self.tracker._detect_engagement(
            "I went outside and took a walk around the block",
            ["Try going for a walk outside."],
        )
        assert result is True

    def test_no_engagement_when_venting(self):
        result = self.tracker._detect_engagement(
            "EVERYTHING IS TERRIBLE AND NOTHING WORKS",
            ["Try going for a walk."],
        )
        assert result is False

    def test_no_engagement_empty_suggestions(self):
        result = self.tracker._detect_engagement(
            "I'm doing okay now",
            [],
        )
        assert result is False

    def test_engagement_with_fair_point(self):
        result = self.tracker._detect_engagement(
            "fair point about the breathing thing",
            ["Try some breathing exercises."],
        )
        assert result is True


class TestSuggestionExtraction:
    """Test extraction of actionable suggestions from responses."""

    def setup_method(self):
        self.tracker = EscalationTracker()

    def test_try_pattern(self):
        suggestions = self.tracker._extract_suggestions(
            "I hear you. Try taking a few deep breaths."
        )
        assert len(suggestions) >= 1
        assert any("deep breaths" in s.lower() for s in suggestions)

    def test_consider_pattern(self):
        suggestions = self.tracker._extract_suggestions(
            "That sounds rough. Consider reaching out to someone you trust."
        )
        assert len(suggestions) >= 1

    def test_bullet_point_suggestions(self):
        suggestions = self.tracker._extract_suggestions(
            "Some things that might help:\n- Try a short walk\n- Consider journaling\n- Maybe call a friend"
        )
        assert len(suggestions) >= 2

    def test_no_suggestions_in_acknowledgment(self):
        suggestions = self.tracker._extract_suggestions(
            "I hear you. That sounds really hard. I'm here."
        )
        assert len(suggestions) == 0

    def test_you_might_pattern(self):
        suggestions = self.tracker._extract_suggestions(
            "You might find it helpful to talk to someone about this."
        )
        assert len(suggestions) >= 1


class TestEscalationVelocity:
    """Test escalation velocity calculation."""

    def setup_method(self):
        self.tracker = EscalationTracker()

    def test_no_history_zero_velocity(self):
        assert self.tracker.get_escalation_velocity() == 0.0

    def test_single_message_zero_velocity(self):
        self.tracker.update(ToneLevel.CRISIS, "bad")
        assert self.tracker.get_escalation_velocity() == 0.0

    def test_stable_tone_zero_velocity(self):
        self.tracker.update(ToneLevel.CONVERSATIONAL, "hi")
        self.tracker.update(ToneLevel.CONVERSATIONAL, "hey")
        self.tracker.update(ToneLevel.CONVERSATIONAL, "yo")
        assert self.tracker.get_escalation_velocity() == 0.0

    def test_rapid_escalation_high_velocity(self):
        self.tracker.update(ToneLevel.CONVERSATIONAL, "hi")
        self.tracker.update(ToneLevel.CONCERN, "hmm")
        self.tracker.update(ToneLevel.ELEVATED, "not good")
        self.tracker.update(ToneLevel.CRISIS, "terrible")
        velocity = self.tracker.get_escalation_velocity()
        assert velocity > 0.0
        assert velocity <= 1.0

    def test_deescalation_low_velocity(self):
        self.tracker.update(ToneLevel.CRISIS, "terrible")
        self.tracker.update(ToneLevel.ELEVATED, "still bad")
        self.tracker.update(ToneLevel.CONCERN, "getting better")
        self.tracker.update(ToneLevel.CONVERSATIONAL, "ok now")
        velocity = self.tracker.get_escalation_velocity()
        # Descending should be 0.0 (clamped at 0)
        assert velocity == 0.0


class TestStrategyInstructions:
    """Test that strategy instructions are appropriate."""

    def setup_method(self):
        self.tracker = EscalationTracker()

    def test_validate_no_override(self):
        """Default strategy produces no supplemental instructions."""
        instructions = self.tracker.get_strategy_instructions()
        assert instructions == ""

    def test_grounding_instructions(self):
        self.tracker.current_strategy = ResponseStrategy.GROUNDING_PRESENCE
        instructions = self.tracker.get_strategy_instructions()
        assert "GROUNDING PRESENCE" in instructions
        assert "2-3 sentences" in instructions
        assert "No advice" in instructions

    def test_quiet_instructions(self):
        self.tracker.current_strategy = ResponseStrategy.QUIET_COMPANIONSHIP
        instructions = self.tracker.get_strategy_instructions()
        assert "QUIET COMPANIONSHIP" in instructions
        assert "1-2 sentences" in instructions

    def test_reengagement_instructions(self):
        self.tracker.current_strategy = ResponseStrategy.GENTLE_REENGAGEMENT
        instructions = self.tracker.get_strategy_instructions()
        assert "GENTLE REENGAGEMENT" in instructions
        assert "ONE small" in instructions


class TestTokenBudgetOverride:
    """Test token budget override for each strategy."""

    def setup_method(self):
        self.tracker = EscalationTracker()

    def test_validate_no_override(self):
        assert self.tracker.get_token_budget_override() is None

    def test_grounding_reduced_budget(self):
        self.tracker.current_strategy = ResponseStrategy.GROUNDING_PRESENCE
        budget = self.tracker.get_token_budget_override()
        assert budget is not None
        assert budget < 1000  # Less than default crisis budget

    def test_quiet_minimal_budget(self):
        self.tracker.current_strategy = ResponseStrategy.QUIET_COMPANIONSHIP
        budget = self.tracker.get_token_budget_override()
        assert budget is not None
        assert budget <= 300  # Very short responses

    def test_reengagement_moderate_budget(self):
        self.tracker.current_strategy = ResponseStrategy.GENTLE_REENGAGEMENT
        budget = self.tracker.get_token_budget_override()
        assert budget is not None
        assert 500 <= budget <= 1000  # Moderate length


class TestReset:
    """Test reset clears all state."""

    def test_reset_clears_state(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CRISIS, "bad")
        tracker.update(ToneLevel.CRISIS, "worse")
        tracker.record_response("Try something.")
        tracker.ignored_suggestion_count = 3

        tracker.reset()

        assert tracker.consecutive_elevated_count == 0
        assert tracker.consecutive_calm_count == 0
        assert tracker.current_strategy == ResponseStrategy.VALIDATE_AND_SUGGEST
        assert len(tracker.tone_history) == 0
        assert len(tracker.last_suggestions) == 0
        assert tracker.ignored_suggestion_count == 0


class TestDebugInfo:
    """Test debug info output."""

    def test_debug_info_keys(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.ELEVATED, "stressed")
        info = tracker.get_debug_info()
        assert "strategy" in info
        assert "consecutive_elevated" in info
        assert "consecutive_calm" in info
        assert "ignored_suggestions" in info
        assert "velocity" in info
        assert "history_length" in info

    def test_debug_info_values(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CRISIS, "bad")
        tracker.update(ToneLevel.CRISIS, "worse")
        info = tracker.get_debug_info()
        assert info["strategy"] == "validate_and_suggest"
        assert info["consecutive_elevated"] == 2
        assert info["history_length"] == 2


class TestHistoryWindow:
    """Test sliding window behavior."""

    def test_history_respects_max(self):
        tracker = EscalationTracker(max_history=5)
        for i in range(10):
            tracker.update(ToneLevel.CONVERSATIONAL, f"message {i}")
        assert len(tracker.tone_history) == 5

    def test_history_keeps_recent(self):
        tracker = EscalationTracker(max_history=3)
        tracker.update(ToneLevel.CONVERSATIONAL, "first")
        tracker.update(ToneLevel.CONCERN, "second")
        tracker.update(ToneLevel.ELEVATED, "third")
        tracker.update(ToneLevel.CRISIS, "fourth")

        assert len(tracker.tone_history) == 3
        assert tracker.tone_history[0] == ToneLevel.CONCERN
        assert tracker.tone_history[-1] == ToneLevel.CRISIS


class TestEndToEndScenarios:
    """Integration-style tests for realistic conversation flows."""

    def test_spiral_scenario(self):
        """
        Simulates a user spiraling: calm → escalating → sustained crisis.
        Strategy progression: VALIDATE → GROUNDING → QUIET.
        GROUNDING always precedes QUIET.
        """
        tracker = EscalationTracker(escalation_threshold=3)

        # Normal conversation
        s = tracker.update(ToneLevel.CONVERSATIONAL, "hey what's up")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

        # Starting to get distressed
        s = tracker.update(ToneLevel.CONCERN, "I'm feeling off today")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

        # Escalating — 1st elevated
        s = tracker.update(ToneLevel.ELEVATED, "I'm really not okay")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

        # Record a response with suggestions
        tracker.record_response("That sounds hard. Try taking a break and getting some fresh air.")

        # Continued escalation, ignoring suggestion — 2nd elevated
        s = tracker.update(ToneLevel.CRISIS, "I don't want to go outside I want to scream")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

        tracker.record_response("I hear you. Your feelings are valid. Consider calling a friend.")

        # Still escalating, ignoring suggestion — 3rd elevated = threshold → GROUNDING
        s = tracker.update(ToneLevel.CRISIS, "NOBODY UNDERSTANDS")
        assert s == ResponseStrategy.GROUNDING_PRESENCE

        # Grounding response: pure acknowledgment, no suggestions
        tracker.record_response("I'm here with you. This is really hard.")

        # 4th elevated — past threshold + 2 ignored suggestions → QUIET
        s = tracker.update(ToneLevel.CRISIS, "I HATE EVERYTHING")
        assert s == ResponseStrategy.QUIET_COMPANIONSHIP

    def test_recovery_scenario(self):
        """
        User goes through crisis and then calms down.
        Tracker should shift to gentle reengagement for deescalation_window
        messages, then back to normal.
        """
        tracker = EscalationTracker(escalation_threshold=3, deescalation_window=2)

        # Crisis
        tracker.update(ToneLevel.CRISIS, "I can't do this")
        tracker.update(ToneLevel.CRISIS, "Everything is wrong")
        tracker.update(ToneLevel.CRISIS, "I'm falling apart")

        # Calming down — 1st calm message → GENTLE
        s = tracker.update(ToneLevel.CONCERN, "...ok. I'm breathing.")
        assert s == ResponseStrategy.GENTLE_REENGAGEMENT

        # 2nd calm message → still GENTLE (within window of 2)
        s = tracker.update(ToneLevel.CONVERSATIONAL, "sorry about that")
        assert s == ResponseStrategy.GENTLE_REENGAGEMENT

        # 3rd calm message → past window (3 > 2), back to normal
        s = tracker.update(ToneLevel.CONVERSATIONAL, "anyway, what were we talking about")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_intensity_shift_not_deescalation(self):
        """
        User shifts from grief to analytical processing (PERSPECTIVE need type).
        This is NOT de-escalation — they want engagement, not gentle handling.
        """
        tracker = EscalationTracker(escalation_threshold=3)

        # Sustained crisis
        tracker.update(ToneLevel.CRISIS, "I lost my job")
        tracker.update(ToneLevel.CRISIS, "I don't know what to do")
        tracker.update(ToneLevel.CRISIS, "Everything is falling apart")

        # User shifts to analytical: "let me think about this"
        s = tracker.update(
            ToneLevel.CONCERN,
            "actually wait, what are my options here",
            need_type="PERSPECTIVE",
        )
        # Should go to VALIDATE (engagement-ready) not GENTLE_REENGAGEMENT
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST
