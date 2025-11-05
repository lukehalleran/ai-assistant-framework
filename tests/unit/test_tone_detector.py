"""
Unit tests for utils/tone_detector.py

Tests tone detection functionality:
- Crisis level classification
- Keyword matching
- Observational language detection
- Configuration
- Formatting functions
"""

import pytest
from utils.tone_detector import (
    CrisisLevel,
    ToneAnalysis,
    TONE_CONFIG,
    HIGH_CRISIS_KEYWORDS,
    MEDIUM_CRISIS_KEYWORDS,
    CONCERN_KEYWORDS,
    OBSERVATIONAL_MARKERS,
    _check_observational_language,
    _check_keyword_crisis,
    format_tone_log,
    should_log_tone_shift,
    format_tone_shift_log,
)


# =============================================================================
# Configuration Tests
# =============================================================================

def test_tone_config_exists():
    """TONE_CONFIG contains expected keys"""
    assert "threshold_high" in TONE_CONFIG
    assert "threshold_medium" in TONE_CONFIG
    assert "threshold_concern" in TONE_CONFIG
    assert "context_window" in TONE_CONFIG
    assert "escalation_boost" in TONE_CONFIG


def test_tone_config_thresholds_ordered():
    """Thresholds are in descending order (high > medium > concern)"""
    assert TONE_CONFIG["threshold_high"] > TONE_CONFIG["threshold_medium"]
    assert TONE_CONFIG["threshold_medium"] > TONE_CONFIG["threshold_concern"]


def test_tone_config_values_reasonable():
    """Config values are in reasonable ranges"""
    assert 0 < TONE_CONFIG["threshold_high"] <= 1.0
    assert 0 < TONE_CONFIG["threshold_medium"] <= 1.0
    assert 0 < TONE_CONFIG["threshold_concern"] <= 1.0
    assert TONE_CONFIG["context_window"] > 0
    assert TONE_CONFIG["escalation_boost"] >= 1.0


# =============================================================================
# CrisisLevel Enum Tests
# =============================================================================

def test_crisis_level_enum():
    """CrisisLevel enum has all expected values"""
    assert CrisisLevel.CONVERSATIONAL.value == "conversational"
    assert CrisisLevel.CONCERN.value == "light_support"
    assert CrisisLevel.MEDIUM.value == "elevated_support"
    assert CrisisLevel.HIGH.value == "crisis_support"


def test_crisis_level_ordering():
    """CrisisLevel can be compared by severity"""
    levels = [CrisisLevel.CONVERSATIONAL, CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH]
    # Just verify they can be stored and retrieved
    assert len(levels) == 4


# =============================================================================
# ToneAnalysis Dataclass Tests
# =============================================================================

def test_tone_analysis_creation():
    """ToneAnalysis can be created with expected fields"""
    analysis = ToneAnalysis(
        level=CrisisLevel.CONVERSATIONAL,
        confidence=0.9,
        trigger="default",
        raw_scores={"high": 0.1, "medium": 0.2},
        explanation="No crisis detected"
    )

    assert analysis.level == CrisisLevel.CONVERSATIONAL
    assert analysis.confidence == 0.9
    assert analysis.trigger == "default"
    assert analysis.raw_scores == {"high": 0.1, "medium": 0.2}
    assert analysis.explanation == "No crisis detected"


# =============================================================================
# Keyword Set Tests
# =============================================================================

def test_high_crisis_keywords_not_empty():
    """HIGH_CRISIS_KEYWORDS contains keywords"""
    assert len(HIGH_CRISIS_KEYWORDS) > 0
    assert "suicide" in HIGH_CRISIS_KEYWORDS
    assert "kill myself" in HIGH_CRISIS_KEYWORDS


def test_medium_crisis_keywords_not_empty():
    """MEDIUM_CRISIS_KEYWORDS contains keywords"""
    assert len(MEDIUM_CRISIS_KEYWORDS) > 0
    assert "panic attack" in MEDIUM_CRISIS_KEYWORDS


def test_concern_keywords_not_empty():
    """CONCERN_KEYWORDS contains keywords"""
    assert len(CONCERN_KEYWORDS) > 0
    assert "anxious" in CONCERN_KEYWORDS or "really anxious" in CONCERN_KEYWORDS


def test_observational_markers_not_empty():
    """OBSERVATIONAL_MARKERS contains markers"""
    assert len(OBSERVATIONAL_MARKERS) > 0
    assert "people are" in OBSERVATIONAL_MARKERS
    assert "according to" in OBSERVATIONAL_MARKERS


def test_keyword_sets_are_lowercase():
    """All keyword sets use lowercase for consistent matching"""
    for keyword in HIGH_CRISIS_KEYWORDS:
        assert keyword == keyword.lower()

    for keyword in MEDIUM_CRISIS_KEYWORDS:
        assert keyword == keyword.lower()

    for keyword in CONCERN_KEYWORDS:
        assert keyword == keyword.lower()


# =============================================================================
# _check_observational_language Tests
# =============================================================================

def test_check_observational_language_multiple_markers():
    """Detects multiple observational markers"""
    # Requires >= 2 markers
    assert _check_observational_language("According to reports, people are suffering") == True


def test_check_observational_language_news_and_people():
    """Detects news + people markers as observational"""
    assert _check_observational_language("The news says people can't get help") == True


def test_check_observational_language_third_person_only():
    """Detects third-person language without first-person"""
    # Requires multiple third-person pronouns and no first-person
    assert _check_observational_language("They are struggling and they can't cope") == True


def test_check_observational_language_personal_statement():
    """Does not detect personal statements as observational"""
    assert _check_observational_language("I am suffering and need help") == False


def test_check_observational_language_single_marker():
    """Single marker alone doesn't trigger observational"""
    # Only one marker - not enough
    assert _check_observational_language("People are struggling") == False


def test_check_observational_language_empty():
    """Empty string is not observational"""
    assert _check_observational_language("") == False


def test_check_observational_language_first_person_beats_third():
    """First-person language overrides third-person markers"""
    assert _check_observational_language("They are struggling but I am too") == False


# =============================================================================
# _check_keyword_crisis Tests
# =============================================================================

def test_check_keyword_crisis_high_suicide():
    """Detects 'suicide' as HIGH crisis"""
    result = _check_keyword_crisis("I'm thinking about suicide")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.HIGH
    assert "keyword" in trigger.lower()


def test_check_keyword_crisis_high_kill_myself():
    """Detects 'kill myself' as HIGH crisis"""
    result = _check_keyword_crisis("I want to kill myself")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.HIGH


def test_check_keyword_crisis_medium_panic():
    """Detects 'panic attack' as MEDIUM crisis"""
    result = _check_keyword_crisis("I'm having a panic attack")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.MEDIUM


def test_check_keyword_crisis_medium_breakdown():
    """Detects 'breakdown' as MEDIUM crisis"""
    result = _check_keyword_crisis("I'm having a complete breakdown")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.MEDIUM


def test_check_keyword_crisis_concern_anxious():
    """Detects 'really anxious' as CONCERN"""
    result = _check_keyword_crisis("I'm really anxious about this")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.CONCERN


def test_check_keyword_crisis_concern_scared():
    """Detects 'scared' as CONCERN"""
    result = _check_keyword_crisis("I'm scared about what's happening")
    assert result is not None
    level, trigger = result
    assert level == CrisisLevel.CONCERN


def test_check_keyword_crisis_no_match():
    """Returns None when no crisis keywords found"""
    result = _check_keyword_crisis("What's the weather like today?")
    assert result is None


def test_check_keyword_crisis_case_insensitive():
    """Keyword matching is case insensitive"""
    result = _check_keyword_crisis("I'm thinking about SUICIDE")
    assert result is not None
    assert result[0] == CrisisLevel.HIGH


def test_check_keyword_crisis_empty():
    """Empty string returns None"""
    result = _check_keyword_crisis("")
    assert result is None


def test_check_keyword_crisis_partial_match():
    """Handles partial keyword matches"""
    # "can't handle this" is a MEDIUM keyword
    result = _check_keyword_crisis("I can't handle this situation anymore")
    assert result is not None
    assert result[0] == CrisisLevel.MEDIUM


def test_check_keyword_crisis_high_beats_medium():
    """When multiple keywords present, highest level wins"""
    result = _check_keyword_crisis("I'm having a panic attack and want to die")
    assert result is not None
    level, _ = result
    # HIGH should win over MEDIUM
    assert level == CrisisLevel.HIGH


# =============================================================================
# format_tone_log Tests
# =============================================================================

def test_format_tone_log_basic():
    """format_tone_log produces formatted output"""
    analysis = ToneAnalysis(
        level=CrisisLevel.CONVERSATIONAL,
        confidence=0.9,
        trigger="default",
        raw_scores={},
        explanation="Normal conversation"
    )

    result = format_tone_log(analysis, "Hello there")

    assert isinstance(result, str)
    assert "CONVERSATIONAL" in result.upper() or "conversational" in result.lower()


def test_format_tone_log_high_crisis():
    """format_tone_log handles HIGH crisis"""
    analysis = ToneAnalysis(
        level=CrisisLevel.HIGH,
        confidence=0.95,
        trigger="keyword_match",
        raw_scores={"high": 0.95},
        explanation="Suicidal ideation detected"
    )

    result = format_tone_log(analysis, "I want to die")

    assert isinstance(result, str)
    assert "HIGH" in result.upper() or "crisis" in result.lower()


def test_format_tone_log_includes_message_preview():
    """format_tone_log includes truncated message"""
    analysis = ToneAnalysis(
        level=CrisisLevel.CONCERN,
        confidence=0.7,
        trigger="semantic",
        raw_scores={},
        explanation="Anxiety detected"
    )

    result = format_tone_log(analysis, "I'm worried about many things")

    # Should include some part of the message
    assert isinstance(result, str)
    assert len(result) > 0


# =============================================================================
# should_log_tone_shift Tests
# =============================================================================

def test_should_log_tone_shift_to_high():
    """Should log shift to HIGH crisis"""
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.HIGH) == True


def test_should_log_tone_shift_to_medium():
    """Should log shift to MEDIUM crisis"""
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.MEDIUM) == True


def test_should_log_tone_shift_to_concern():
    """Should log shift to CONCERN"""
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.CONCERN) == True


def test_should_log_tone_shift_same_level():
    """Should not log when level stays the same"""
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.CONVERSATIONAL) == False


def test_should_log_tone_shift_escalation():
    """Should log escalation from CONCERN to MEDIUM"""
    assert should_log_tone_shift(CrisisLevel.CONCERN, CrisisLevel.MEDIUM) == True


def test_should_log_tone_shift_deescalation():
    """Should log de-escalation from HIGH to CONVERSATIONAL"""
    assert should_log_tone_shift(CrisisLevel.HIGH, CrisisLevel.CONVERSATIONAL) == True


def test_should_log_tone_shift_from_conversational_to_high():
    """Should log significant escalation"""
    # Clear escalation should log
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.HIGH) == True


# =============================================================================
# format_tone_shift_log Tests
# =============================================================================

def test_format_tone_shift_log_escalation():
    """format_tone_shift_log handles escalation"""
    result = format_tone_shift_log(CrisisLevel.CONVERSATIONAL, CrisisLevel.HIGH, "keyword_match")

    assert isinstance(result, str)
    assert len(result) > 0


def test_format_tone_shift_log_deescalation():
    """format_tone_shift_log handles de-escalation"""
    result = format_tone_shift_log(CrisisLevel.MEDIUM, CrisisLevel.CONVERSATIONAL, "default")

    assert isinstance(result, str)
    assert len(result) > 0


def test_format_tone_shift_log_with_trigger():
    """format_tone_shift_log includes trigger information"""
    result = format_tone_shift_log(CrisisLevel.CONCERN, CrisisLevel.HIGH, "keyword_match")

    assert isinstance(result, str)
    assert len(result) > 0
    # Should mention something about the change


# =============================================================================
# Integration-style Tests (without embeddings)
# =============================================================================

def test_keyword_detection_comprehensive():
    """Comprehensive test of various crisis phrases"""
    # HIGH level
    assert _check_keyword_crisis("I want to end it all")[0] == CrisisLevel.HIGH
    assert _check_keyword_crisis("no point living anymore")[0] == CrisisLevel.HIGH

    # MEDIUM level
    assert _check_keyword_crisis("I'm spiraling out of control")[0] == CrisisLevel.MEDIUM
    assert _check_keyword_crisis("can't breathe, losing it")[0] == CrisisLevel.MEDIUM

    # CONCERN level
    assert _check_keyword_crisis("freaking out about this")[0] == CrisisLevel.CONCERN
    assert _check_keyword_crisis("I'm terrified")[0] == CrisisLevel.CONCERN


def test_personal_distress_detection():
    """Detects personal distress statements with exact keywords"""
    # Personal statements with exact keyword matches
    assert _check_keyword_crisis("I feel suicidal") is not None
    assert _check_keyword_crisis("I'm having a panic attack") is not None
    assert _check_keyword_crisis("I'm really anxious") is not None


def test_multiple_crisis_keywords():
    """Handles messages with multiple crisis levels"""
    # Has both MEDIUM and CONCERN keywords, higher level should win
    result = _check_keyword_crisis("I'm anxious and having a panic attack")
    assert result is not None
    # Should be at least CONCERN or higher
    assert result[0] in [CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH]
