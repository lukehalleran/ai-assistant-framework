"""
Comprehensive test suite for harm scoring and crisis detection.
"""
import asyncio
import logging
from utils.tone_detector import detect_crisis_level, CrisisLevel
from utils.logging_utils import configure_logging

# Configure logging
configure_logging(level=logging.INFO, console_level=logging.INFO)

# Test cases covering various scenarios
TEST_CASES = [
    # HIGH CRISIS - Suicidal ideation
    {
        "name": "Suicidal ideation",
        "message": "I want to die. There's no point in living anymore. Everyone would be better off without me.",
        "expected_level": CrisisLevel.HIGH,
        "expected_min_score": 20,
    },

    # HIGH CRISIS - Self-harm + crying (Dec 6 scenario)
    {
        "name": "Self-harm + crying + abuse",
        "message": "I'm sobbing. She was abusive. I want to peel off all my skin.",
        "expected_level": CrisisLevel.HIGH,
        "expected_min_score": 15,
    },

    # HIGH CRISIS - Multiple HIGH indicators
    {
        "name": "Hopelessness + suicidal + crying",
        "message": "I've lost all hope. Nothing matters anymore. I can't stop crying and I just want it all to end.",
        "expected_level": CrisisLevel.HIGH,
        "expected_min_score": 30,
    },

    # MEDIUM - Panic attack
    {
        "name": "Panic attack",
        "message": "I'm having a panic attack. I can't breathe. My heart is racing and I'm losing control.",
        "expected_level": CrisisLevel.MEDIUM,
        "expected_min_score": 8,
    },

    # MEDIUM - Abuse + distress
    {
        "name": "Abuse trauma with distress",
        "message": "My ex was gaslighting me for years. I'm breaking down thinking about it. She manipulated everything.",
        "expected_level": CrisisLevel.MEDIUM,
        "expected_min_score": 8,
    },

    # MEDIUM - Dissociation + trauma
    {
        "name": "Dissociation + flashbacks",
        "message": "I'm dissociating. Having flashbacks to the trauma. Nothing feels real right now.",
        "expected_level": CrisisLevel.MEDIUM,
        "expected_min_score": 8,
    },

    # MEDIUM - Substance relapse
    {
        "name": "Substance relapse crisis",
        "message": "I relapsed. I'm using again and I can't stop. Breaking down completely.",
        "expected_level": CrisisLevel.MEDIUM,
        "expected_min_score": 8,
    },

    # CONCERN - Anxiety
    {
        "name": "General anxiety",
        "message": "I'm really anxious about the presentation tomorrow. Can't sleep and worried sick.",
        "expected_level": CrisisLevel.CONCERN,
        "expected_min_score": 4,
    },

    # CONCERN - Feeling lonely
    {
        "name": "Loneliness",
        "message": "Feeling so lonely tonight. Everyone's busy and I'm just sitting here alone.",
        "expected_level": CrisisLevel.CONCERN,
        "expected_min_score": 2,
    },

    # CONCERN - Grief
    {
        "name": "Grief and loss",
        "message": "I miss her so much. Grieving is hard. Still hurts even after all this time.",
        "expected_level": CrisisLevel.CONCERN,
        "expected_min_score": 4,
    },

    # CONCERN - Work stress
    {
        "name": "Work burnout",
        "message": "I'm so burned out at work. The deadline pressure is killing me. Exhausted.",
        "expected_level": CrisisLevel.CONCERN,
        "expected_min_score": 4,
    },

    # CONVERSATIONAL - Casual mention
    {
        "name": "Casual conversation",
        "message": "Had a tough day but I'm okay. Just need to decompress.",
        "expected_level": CrisisLevel.CONVERSATIONAL,
        "expected_min_score": 0,
    },

    # CONVERSATIONAL - No distress
    {
        "name": "Normal update",
        "message": "Just finished my project! Pretty happy with how it turned out.",
        "expected_level": CrisisLevel.CONVERSATIONAL,
        "expected_min_score": 0,
    },

    # Edge case - World events (not personal crisis)
    {
        "name": "World events observational",
        "message": "What the fuck is happening in the world. People are suffering and the government isn't doing anything.",
        "expected_level": CrisisLevel.CONVERSATIONAL,  # Should detect observational language
        "expected_min_score": None,  # May have event distress keywords but observational check should catch it
    },

    # Edge case - Positive overwhelmed
    {
        "name": "Positive overwhelmed",
        "message": "I'm so overwhelmed by all the birthday gifts and amazing support!",
        "expected_level": CrisisLevel.CONVERSATIONAL,
        "expected_min_score": 0,
    },

    # Pattern multiplier test - Abuse + distress
    {
        "name": "Abuse + distress multiplier",
        "message": "She was abusive and controlling. I'm sobbing and can't stop crying. She gaslighted me.",
        "expected_level": CrisisLevel.HIGH,
        "expected_min_score": 15,
        "test_multiplier": True,
    },

    # Pattern multiplier test - Sleep deprivation + mental distress
    {
        "name": "Sleep deprivation + mental distress",
        "message": "Haven't slept in days. I'm losing my mind. Can't cope with this anymore.",
        "expected_level": CrisisLevel.MEDIUM,
        "expected_min_score": 8,
    },
]


async def run_tests():
    """Run all test cases and report results."""
    print("=" * 80)
    print("HARM SCORING TEST SUITE")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"[Test {i}/{len(TEST_CASES)}] {test['name']}")
        print(f"Message: \"{test['message'][:60]}...\"" if len(test['message']) > 60 else f"Message: \"{test['message']}\"")

        # Run detection
        result = await detect_crisis_level(test['message'], conversation_history=None, model_manager=None)

        # Check expected level
        level_match = result.level == test['expected_level']

        # Check minimum score (if specified)
        score_match = True
        if test.get('expected_min_score') is not None:
            # Extract score from trigger if it's a harm_score trigger
            if 'harm_score:' in result.trigger:
                score_str = result.trigger.split('harm_score:')[1].split()[0]
                actual_score = float(score_str)
                score_match = actual_score >= test['expected_min_score']
            elif test['expected_min_score'] == 0:
                # For conversational, we expect no harm score trigger
                score_match = 'harm_score' not in result.trigger
            else:
                score_match = False

        # Overall pass/fail
        test_passed = level_match and score_match

        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        print(f"  Detected: {result.level.value} (trigger: {result.trigger})")
        print(f"  Expected: {test['expected_level'].value}")
        print(f"  Result: {status}")

        if not level_match:
            print(f"  ⚠️  Level mismatch!")
        if not score_match and test.get('expected_min_score') is not None:
            print(f"  ⚠️  Score below minimum!")

        print()

        results.append({
            "test": test['name'],
            "passed": test_passed,
            "detected": result.level.value,
            "expected": test['expected_level'].value,
            "trigger": result.trigger,
        })

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed: {passed} ({passed/len(TEST_CASES)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(TEST_CASES)*100:.1f}%)")
    print()

    if failed > 0:
        print("Failed tests:")
        for r in results:
            if not r['passed']:
                print(f"  - {r['test']}: detected {r['detected']}, expected {r['expected']}")
        print()

    # Category breakdown
    print("Results by category:")
    categories = {
        CrisisLevel.HIGH: [],
        CrisisLevel.MEDIUM: [],
        CrisisLevel.CONCERN: [],
        CrisisLevel.CONVERSATIONAL: [],
    }

    for test, r in zip(TEST_CASES, results):
        categories[test['expected_level']].append(r['passed'])

    for level, passes in categories.items():
        if passes:
            passed_count = sum(passes)
            total_count = len(passes)
            print(f"  {level.value}: {passed_count}/{total_count} passed")

    print()
    print("=" * 80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(run_tests())
    exit(0 if failed == 0 else 1)
