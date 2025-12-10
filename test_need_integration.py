"""
Simulation script to test need-type classifier integration.
Tests the full flow without running the actual Daemon system.
"""

import asyncio
from utils.need_detector import detect_need_type, NeedType
from utils.tone_detector import detect_crisis_level, CrisisLevel
from utils.emotional_context import analyze_emotional_context, format_emotional_context_log


async def simulate_message(message: str):
    """Simulate processing a single message through the emotional context system."""
    print(f"\n{'='*80}")
    print(f"MESSAGE: {message}")
    print('='*80)

    # Step 1: Need detection (synchronous)
    need_analysis = detect_need_type(message)
    print(f"\n[NEED DETECTOR]")
    print(f"  Type: {need_analysis.need_type.value}")
    print(f"  Confidence: {need_analysis.confidence:.2f}")
    print(f"  Trigger: {need_analysis.trigger}")
    print(f"  Explanation: {need_analysis.explanation}")
    if need_analysis.raw_scores:
        print(f"  Raw scores: {need_analysis.raw_scores}")

    # Step 2: Tone detection (async)
    tone_analysis = await detect_crisis_level(message)
    print(f"\n[TONE DETECTOR]")
    print(f"  Level: {tone_analysis.level.value}")
    print(f"  Confidence: {tone_analysis.confidence:.2f}")
    print(f"  Trigger: {tone_analysis.trigger}")
    print(f"  Explanation: {tone_analysis.explanation}")
    if tone_analysis.raw_scores:
        print(f"  Raw scores: {tone_analysis.raw_scores}")

    # Step 3: Combined emotional context
    emotional_context = await analyze_emotional_context(message)
    print(f"\n[EMOTIONAL CONTEXT]")
    print(f"  Crisis Level: {emotional_context.crisis_level.value}")
    print(f"  Need Type: {emotional_context.need_type.value}")
    print(f"  Tone Confidence: {emotional_context.tone_confidence:.2f}")
    print(f"  Need Confidence: {emotional_context.need_confidence:.2f}")

    # Step 4: Formatted log (what orchestrator would log)
    log_msg = format_emotional_context_log(emotional_context, message)
    print(f"\n[ORCHESTRATOR LOG]")
    print(f"  {log_msg}")

    # Step 5: Response mode decision
    print(f"\n[RESPONSE MODE]")
    if emotional_context.crisis_level == CrisisLevel.HIGH:
        print(f"  → CRISIS SUPPORT (full therapeutic mode)")
    elif emotional_context.crisis_level == CrisisLevel.MEDIUM:
        if emotional_context.need_type == NeedType.PRESENCE:
            print(f"  → ELEVATED SUPPORT + PRESENCE MODE (warmth first, measured support)")
        elif emotional_context.need_type == NeedType.PERSPECTIVE:
            print(f"  → ELEVATED SUPPORT + PERSPECTIVE MODE (supportive engagement)")
        else:
            print(f"  → ELEVATED SUPPORT (base mode)")
    elif emotional_context.crisis_level == CrisisLevel.CONCERN:
        if emotional_context.need_type == NeedType.PRESENCE:
            print(f"  → LIGHT SUPPORT + PRESENCE MODE (brief acknowledgment, stay present)")
        elif emotional_context.need_type == NeedType.PERSPECTIVE:
            print(f"  → LIGHT SUPPORT + PERSPECTIVE MODE (light engagement, offer reframes)")
        else:
            print(f"  → LIGHT SUPPORT (base mode)")
    else:  # CONVERSATIONAL
        if emotional_context.need_type == NeedType.PRESENCE:
            print(f"  → CONVERSATIONAL + PRESENCE MODE (warm casual)")
        elif emotional_context.need_type == NeedType.PERSPECTIVE:
            print(f"  → CONVERSATIONAL + PERSPECTIVE MODE (engaged casual)")
        else:
            print(f"  → CONVERSATIONAL (default casual)")


async def main():
    """Run simulation on test messages from the debug log."""

    print("\n" + "="*80)
    print("NEED-TYPE CLASSIFIER INTEGRATION SIMULATION")
    print("="*80)

    # Test cases from the debug log in the plan
    test_messages = [
        # Expected: CONVERSATIONAL / NEUTRAL
        "Morning man!",

        # Expected: CONCERN / PERSPECTIVE (has "I think the reason")
        "I'm still feeling a little off. Not like yesterday more just frustration. Online dating sucks lol I get so many matches and I think the reason they don't go anywhere is my school/living situation",

        # Expected: CONVERSATIONAL / NEUTRAL or PERSPECTIVE
        "Like I mentioned before I live with my mom at the moment",

        # Expected: CONCERN / PERSPECTIVE (has "I feel like")
        "I lead with transitional info when it comes up. I feel like I am in a good position. In two years I will be making more than 90% of my matches",

        # Expected: CONCERN / PRESENCE (has "ugh")
        "No one can see it ugh. I will have to wait 2 years",

        # Expected: CONCERN / PRESENCE (has "I am lonely")
        "It feels very long. I am lonely",

        # Expected: CONCERN / PRESENCE (has "Ugh")
        "I'm not it's just. Ugh. I will be 34",
    ]

    for message in test_messages:
        try:
            await simulate_message(message)
        except Exception as e:
            print(f"\n❌ ERROR processing message: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
