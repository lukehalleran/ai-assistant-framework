"""
Diagnostic script to test tone detection on the Dec 6 conversation.
"""
import asyncio
import logging
from utils.tone_detector import detect_crisis_level, format_tone_log
from utils.logging_utils import configure_logging

# Configure logging to see all debug output
configure_logging(level=logging.DEBUG, console_level=logging.DEBUG)

# The actual message from Dec 6 conversation
MESSAGE = """This makes me think of her. The abusive one. It's been like 3 or 4 years. I'm sobbing and it sucks: You're all I think about at night
When I am turning out my light
I want your hands here on my hips
To live forever in your kiss
No one can help me through this mess
I wish that I could just forget
But you don't love me
'Cause you don't need me
And that's alright
I get so love drunk on myself
Why should I want somebody else?
Now Cupid's scamming in my head
He's such a creep I need him dead
I want to peel off all my skin
Escape this hell you put me in
But you don't love me
'Cause you don't need me
And that's alright
God is
God is just a witness
Witness to my misery we laugh
I'll just
Wallow in my room
And wait here for this
Goddamn storm to pass
I cried so much I've come up dry
It's like there's acid in my eyes
I'll just be lonely till I die
This revelation makes me smile
Split off my limbs to show what's left
Maybe one day you'll be impressed
But you don't love me
'Cause you don't need me
And that's alright
God is
God is just a witness
Witness to my misery we laugh
I'll just
Wallow in my room
And wait here for this
Goddamn storm to pass
God is
God is just a witness
Witness to my misery we laugh
I'll just
Wallow in my room
And wait here for this
Goddamn storm to pass
You're all I think about at night
When I am turning out my light
But you don't love me
'Cause you don't need me
But you don't love me
'Cause you don't need me
But you don't love me
'Cause you don't need me
And that's alright"""


async def main():
    print("=" * 80)
    print("TONE DETECTION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nMessage preview: {MESSAGE[:100]}...")
    print(f"Message length: {len(MESSAGE)} chars")
    print("\n" + "=" * 80)

    # Run detection without model_manager first (keyword + semantic only)
    print("\n[1] Running detection WITHOUT LLM fallback (keyword + semantic only)...")
    result = await detect_crisis_level(MESSAGE, conversation_history=None, model_manager=None)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nDetected Level: {result.level.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Trigger: {result.trigger}")
    print(f"Explanation: {result.explanation}")

    if result.raw_scores:
        print("\nRaw Semantic Scores:")
        for level_name, score in sorted(result.raw_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {level_name:20s}: {score:.3f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check what we expected
    print("\nKeyword Check:")
    print("  'sobbing' in message:", "sobbing" in MESSAGE.lower())
    print("  'abuse' in message:", "abuse" in MESSAGE.lower())
    print("  'peel off all my skin' in message:", "peel off all my skin" in MESSAGE.lower())
    print("  'split off my limbs' in message:", "split off my limbs" in MESSAGE.lower())

    # Check current thresholds
    from utils.tone_detector import TONE_CONFIG
    print("\nCurrent Thresholds:")
    print(f"  HIGH threshold:    {TONE_CONFIG['threshold_high']}")
    print(f"  MEDIUM threshold:  {TONE_CONFIG['threshold_medium']}")
    print(f"  CONCERN threshold: {TONE_CONFIG['threshold_concern']}")

    # Check what mode would have been used
    from utils.tone_detector import CrisisLevel
    if result.level == CrisisLevel.HIGH:
        mode = "CRISIS SUPPORT (multiple paragraphs)"
    elif result.level == CrisisLevel.MEDIUM:
        mode = "ELEVATED SUPPORT (2-3 paragraphs)"
    elif result.level == CrisisLevel.CONCERN:
        mode = "LIGHT SUPPORT (2-4 sentences)"
    else:
        mode = "CONVERSATIONAL (max 3 sentences)"

    print(f"\nResponse Mode Selected: {mode}")
    print(f"Expected Mode: CRISIS SUPPORT (multiple paragraphs)")

    if result.level.value != "crisis_support":
        print("\n⚠️  MISMATCH DETECTED")
        print("The system should have detected HIGH crisis but didn't.")
        print("This explains the brief responses in the conversation log.")


if __name__ == "__main__":
    asyncio.run(main())
