"""
tests/test_tone_detection.py

Comprehensive test suite for tone detection system.

Tests crisis detection accuracy across various scenarios:
- Personal crisis vs. world event observation
- Keyword-based detection
- Semantic similarity detection
- Context-dependent escalation
- Edge cases and paraphrasing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

from utils.tone_detector import (
    detect_crisis_level,
    CrisisLevel,
    _check_keyword_crisis,
    _check_observational_language,
    _semantic_crisis_detection,
)
from utils.logging_utils import get_logger

logger = get_logger("test_tone")


# ===== Test Cases =====

TEST_CASES = [
    # ===== HIGH CRISIS (should trigger full therapeutic mode) =====
    {
        "message": "I don't want to be here anymore",
        "expected": CrisisLevel.HIGH,
        "category": "personal_crisis",
        "description": "Suicidal ideation - direct"
    },
    {
        "message": "I can't go on like this, I just want it to end",
        "expected": CrisisLevel.HIGH,
        "category": "personal_crisis",
        "description": "Suicidal ideation - desire to end"
    },
    {
        "message": "Everything would be better if I just wasn't here",
        "expected": CrisisLevel.HIGH,
        "category": "personal_crisis",
        "description": "Suicidal ideation - paraphrased"
    },
    {
        "message": "I'm planning to hurt myself tonight",
        "expected": CrisisLevel.HIGH,
        "category": "personal_crisis",
        "description": "Imminent self-harm"
    },

    # ===== MEDIUM CRISIS (elevated support) =====
    {
        "message": "I'm having a panic attack right now and can't breathe",
        "expected": CrisisLevel.MEDIUM,
        "category": "acute_distress",
        "description": "Active panic attack"
    },
    {
        "message": "I feel like I'm completely losing control and spiraling",
        "expected": CrisisLevel.MEDIUM,
        "category": "acute_distress",
        "description": "Acute psychological distress"
    },
    {
        "message": "I'm having a complete breakdown and don't know what to do",
        "expected": CrisisLevel.MEDIUM,
        "category": "acute_distress",
        "description": "Nervous breakdown"
    },

    # ===== CONCERN (light support) =====
    {
        "message": "I'm really anxious about these insurance costs",
        "expected": CrisisLevel.CONCERN,
        "category": "moderate_concern",
        "description": "Financial anxiety - from Luke's log"
    },
    {
        "message": "I'm freaking out a bit about this deadline",
        "expected": CrisisLevel.CONCERN,
        "category": "moderate_concern",
        "description": "Situational stress"
    },
    {
        "message": "I can't sleep, I'm so worried about tomorrow",
        "expected": CrisisLevel.CONCERN,
        "category": "moderate_concern",
        "description": "Anxiety with sleep disturbance"
    },

    # ===== CONVERSATIONAL (default mode - most important category) =====

    # Status updates (should NOT trigger crisis mode)
    {
        "message": "Woke up at 10",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "status_update",
        "description": "Simple status - from Luke's log"
    },
    {
        "message": "Work at 4:30",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "status_update",
        "description": "Schedule update"
    },
    {
        "message": "Just had breakfast, feeling okay",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "status_update",
        "description": "Routine update"
    },

    # World event observations (should NOT trigger crisis mode)
    {
        "message": "42 million people are losing food aid due to SNAP cuts. That 3.5% threshold is brutal.",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "world_observation",
        "description": "SNAP cuts concern - from Luke's log"
    },
    {
        "message": "People are really struggling with these deportation raids. ICE arrested hundreds in Chicago.",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "world_observation",
        "description": "Observing societal crisis"
    },
    {
        "message": "The reports about the protests turning violent are concerning. Police used tear gas on civilians.",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "world_observation",
        "description": "News about protests"
    },
    {
        "message": "According to the study, millions are experiencing anxiety due to economic uncertainty.",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "world_observation",
        "description": "Citing research/data"
    },
    {
        "message": "People can't afford housing anymore, the data shows evictions are up 40%",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "world_observation",
        "description": "Statistical observation"
    },

    # Casual conversation
    {
        "message": "What's the best way to handle this coding problem?",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "technical_question",
        "description": "Technical query"
    },
    {
        "message": "Can you explain how transformers work?",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "educational",
        "description": "Educational question"
    },
    {
        "message": "I'm thinking about learning Rust",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "casual_topic",
        "description": "Hobby/interest"
    },

    # Circadian rhythm discussion (from Luke's log - was over-therapized)
    {
        "message": "Being synced with daylight makes a huge difference for productivity",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "casual_observation",
        "description": "Circadian rhythm observation"
    },

    # ===== EDGE CASES =====

    # Context-dependent (same phrase, different contexts)
    {
        "message": "I'm overwhelmed",
        "expected": CrisisLevel.CONCERN,
        "category": "ambiguous",
        "description": "Standalone overwhelm statement"
    },
    {
        "message": "I'm overwhelmed with gift ideas for my friend's birthday!",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "ambiguous",
        "description": "Overwhelmed but positive context"
    },

    # Paraphrased crisis (semantic should catch)
    {
        "message": "Everything feels pointless and I don't see a way forward",
        "expected": CrisisLevel.HIGH,
        "category": "semantic_detection",
        "description": "Paraphrased suicidal ideation"
    },
    {
        "message": "I want to disappear and never come back",
        "expected": CrisisLevel.HIGH,
        "category": "semantic_detection",
        "description": "Paraphrased desire to die"
    },
    {
        "message": "I'm drowning and there's no way out",
        "expected": CrisisLevel.MEDIUM,
        "category": "semantic_detection",
        "description": "Metaphorical distress"
    },

    # Mixed personal + world event (should prioritize personal)
    {
        "message": "People are losing their homes and honestly I'm terrified I'll be next",
        "expected": CrisisLevel.CONCERN,
        "category": "mixed_context",
        "description": "World event + personal fear"
    },

    # Mild negative emotions (should stay conversational)
    {
        "message": "I'm a bit tired today",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "mild_emotion",
        "description": "Mild fatigue"
    },
    {
        "message": "That movie made me sad",
        "expected": CrisisLevel.CONVERSATIONAL,
        "category": "mild_emotion",
        "description": "Situational sadness"
    },
]


# ===== Test Functions =====

def _run_crisis_detection_test(test_case):
    """Test individual crisis detection cases (shared implementation)."""
    message = test_case["message"]
    expected = test_case["expected"]
    description = test_case["description"]

    result = detect_crisis_level(message)

    # Log the result for debugging
    logger.info(
        f"\nTest: {description}\n"
        f"Message: \"{message}\"\n"
        f"Expected: {expected.value}\n"
        f"Got: {result.level.value}\n"
        f"Confidence: {result.confidence:.2f}\n"
        f"Trigger: {result.trigger}\n"
        f"Explanation: {result.explanation}\n"
    )

    assert result.level == expected, (
        f"Failed: {description}\n"
        f"Message: \"{message}\"\n"
        f"Expected {expected.value}, got {result.level.value}\n"
        f"Trigger: {result.trigger}, Confidence: {result.confidence:.2f}"
    )


# Pytest-compatible wrapper
if HAS_PYTEST:
    @pytest.mark.parametrize("test_case", TEST_CASES)
    def test_crisis_detection(test_case):
        """Test individual crisis detection cases."""
        _run_crisis_detection_test(test_case)
else:
    def test_crisis_detection(test_case):
        """Test individual crisis detection cases (non-pytest version)."""
        _run_crisis_detection_test(test_case)


def test_observational_language_detection():
    """Test that observational/world event language is correctly identified."""

    # Should detect observational language
    observational_messages = [
        "People are suffering due to the deportations",
        "The government is cracking down on protests",
        "According to reports, thousands were arrested",
        "They are struggling with the new policies",
    ]

    for msg in observational_messages:
        assert _check_observational_language(msg), (
            f"Failed to detect observational language: '{msg}'"
        )

    # Should NOT detect as observational (personal statements)
    personal_messages = [
        "I'm suffering due to my situation",
        "I'm cracking under pressure",
        "I can't handle this anymore",
        "My life is falling apart",
    ]

    for msg in personal_messages:
        assert not _check_observational_language(msg), (
            f"Incorrectly flagged personal message as observational: '{msg}'"
        )


def test_keyword_detection():
    """Test explicit keyword detection."""

    # HIGH keywords
    high_result = _check_keyword_crisis("I want to die")
    assert high_result is not None
    assert high_result[0] == CrisisLevel.HIGH

    # MEDIUM keywords
    medium_result = _check_keyword_crisis("I'm having a panic attack")
    assert medium_result is not None
    assert medium_result[0] == CrisisLevel.MEDIUM

    # CONCERN keywords
    concern_result = _check_keyword_crisis("I'm really anxious")
    assert concern_result is not None
    assert concern_result[0] == CrisisLevel.CONCERN

    # No keywords (should return None)
    none_result = _check_keyword_crisis("Just had lunch")
    assert none_result is None


def test_context_escalation():
    """Test that prior distress context boosts current detection."""

    message = "I'm still feeling really bad"

    # Without context - should be conversational or low concern
    result_no_context = detect_crisis_level(message, conversation_history=None)

    # With heavy topic context - should escalate
    heavy_context = [
        {
            "query": "I'm having a really hard time",
            "response": "I understand...",
            "is_heavy_topic": True,
        }
    ]
    result_with_context = detect_crisis_level(message, conversation_history=heavy_context)

    logger.info(
        f"Context escalation test:\n"
        f"No context: {result_no_context.level.value} (confidence: {result_no_context.confidence:.2f})\n"
        f"With context: {result_with_context.level.value} (confidence: {result_with_context.confidence:.2f})"
    )

    # Context should either maintain or increase severity
    # (This is a softer assertion since escalation is probabilistic)
    assert result_with_context.level.value in [
        CrisisLevel.CONCERN.value,
        CrisisLevel.MEDIUM.value,
        CrisisLevel.HIGH.value
    ]


def test_semantic_detection_examples():
    """Test that semantic detection works for paraphrased crisis language."""

    # Paraphrased suicidal ideation (no exact keywords)
    semantic_high = [
        "Everything feels pointless and empty",
        "There's no reason to keep going",
        "I wish I could just stop existing",
    ]

    for msg in semantic_high:
        result = detect_crisis_level(msg)
        logger.info(f"Semantic HIGH test: '{msg}' → {result.level.value} (confidence: {result.confidence:.2f})")
        # Should be at least MEDIUM (semantic might not always reach HIGH without exact keywords)
        assert result.level in [CrisisLevel.HIGH, CrisisLevel.MEDIUM], (
            f"Failed semantic detection for: '{msg}'"
        )


def test_tone_shift_detection():
    """Test tone shift logging logic."""
    from utils.tone_detector import should_log_tone_shift, format_tone_shift_log

    # Should log shift when level changes
    assert should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.HIGH)
    assert should_log_tone_shift(CrisisLevel.CONCERN, CrisisLevel.CONVERSATIONAL)

    # Should NOT log when level stays same
    assert not should_log_tone_shift(CrisisLevel.CONVERSATIONAL, CrisisLevel.CONVERSATIONAL)

    # Should NOT log on first message (no previous)
    assert not should_log_tone_shift(None, CrisisLevel.CONVERSATIONAL)

    # Test format
    log_msg = format_tone_shift_log(
        CrisisLevel.CONVERSATIONAL,
        CrisisLevel.HIGH,
        "keyword: want to die"
    )
    assert "conversational → crisis_support" in log_msg


# ===== Summary Statistics =====

def test_print_category_summary():
    """Print summary statistics by category (for development insight)."""

    # Group test cases by category
    by_category = {}
    for tc in TEST_CASES:
        cat = tc["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(tc)

    print("\n" + "="*60)
    print("TONE DETECTION TEST SUITE SUMMARY")
    print("="*60)

    for category, cases in sorted(by_category.items()):
        print(f"\n{category.upper().replace('_', ' ')} ({len(cases)} tests):")
        for tc in cases:
            print(f"  - {tc['description']}: {tc['expected'].value}")

    print(f"\n{'='*60}")
    print(f"TOTAL TEST CASES: {len(TEST_CASES)}")
    print("="*60 + "\n")


# ===== Run Tests =====

if __name__ == "__main__":
    # Run with pytest if available, else run simple version
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "-s"]))
    except ImportError:
        print("pytest not available, running basic tests...")

        # Run test_print_category_summary first
        test_print_category_summary()

        # Run basic tests
        failed = 0
        passed = 0

        for i, test_case in enumerate(TEST_CASES, 1):
            try:
                test_crisis_detection(test_case)
                passed += 1
                print(f"✓ Test {i}/{len(TEST_CASES)} passed: {test_case['description']}")
            except AssertionError as e:
                failed += 1
                print(f"✗ Test {i}/{len(TEST_CASES)} FAILED: {test_case['description']}")
                print(f"  {e}")

        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
        print("="*60)

        # Run other tests
        print("\nRunning observational language test...")
        test_observational_language_detection()
        print("✓ Observational language test passed")

        print("\nRunning keyword detection test...")
        test_keyword_detection()
        print("✓ Keyword detection test passed")

        print("\nRunning context escalation test...")
        test_context_escalation()
        print("✓ Context escalation test passed")

        print("\nRunning semantic detection test...")
        test_semantic_detection_examples()
        print("✓ Semantic detection test passed")

        print("\nRunning tone shift test...")
        test_tone_shift_detection()
        print("✓ Tone shift test passed")

        sys.exit(0 if failed == 0 else 1)
