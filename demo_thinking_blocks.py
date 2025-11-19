#!/usr/bin/env python3
"""
Interactive demo of the thinking blocks feature.

This simulates what would happen with real LLM responses.
"""

from core.orchestrator import DaemonOrchestrator


def demo_scenario(title, user_question, llm_response):
    """Demonstrate one complete scenario."""

    print("\n" + "="*80)
    print(f"SCENARIO: {title}")
    print("="*80)

    print(f"\n👤 User asks: \"{user_question}\"")

    print("\n🤖 LLM generates (internally):")
    print("-" * 80)
    print(llm_response)
    print("-" * 80)

    # Parse the response
    thinking, answer = DaemonOrchestrator._parse_thinking_block(llm_response)

    if thinking:
        print("\n📝 Logged for debugging:")
        print("-" * 80)
        print(f"[THINKING BLOCK]")
        print(thinking)
        print("-" * 80)
    else:
        print("\n📝 No thinking block detected - response returned as-is")

    print("\n👁️ User sees:")
    print("-" * 80)
    print(answer)
    print("-" * 80)

    print("\n💾 Stored in memory:")
    print(f"Q: '{user_question}'")
    print(f"A: '{answer}'")


def main():
    print("\n" + "="*80)
    print("THINKING BLOCKS FEATURE - INTERACTIVE DEMO")
    print("="*80)

    # Scenario 1: Math question with detailed reasoning
    demo_scenario(
        title="Simple Math with Step-by-Step Reasoning",
        user_question="What is 15 * 7?",
        llm_response="""<thinking>
The user is asking for 15 multiplied by 7.
Let me break this down:
- 15 × 7 = (10 × 7) + (5 × 7)
- 10 × 7 = 70
- 5 × 7 = 35
- 70 + 35 = 105
So the answer is 105.
</thinking>

15 × 7 = 105"""
    )

    # Scenario 2: Complex reasoning
    demo_scenario(
        title="Complex Question Requiring Analysis",
        user_question="Should I invest in stocks or bonds right now?",
        llm_response="""<thinking>
This is a financial advice question. I need to be careful:
1. I cannot provide personalized financial advice
2. I should acknowledge the complexity
3. I should suggest considering multiple factors
4. I should recommend consulting a professional

I'll provide a balanced, educational response without making
specific recommendations.
</thinking>

I can't provide personalized financial advice, but here are key factors to consider:

**Stocks** tend to offer higher potential returns but with more volatility.
**Bonds** generally provide more stable income with lower risk.

Your choice depends on:
- Your risk tolerance
- Investment timeline
- Financial goals
- Current portfolio balance

I'd recommend consulting a financial advisor who can assess your specific
situation and create a personalized strategy."""
    )

    # Scenario 3: No thinking block (backward compatibility)
    demo_scenario(
        title="Response Without Thinking Block (Backward Compatible)",
        user_question="What's the capital of France?",
        llm_response="The capital of France is Paris."
    )

    # Scenario 4: Creative task with reasoning
    demo_scenario(
        title="Creative Task with Multi-Step Reasoning",
        user_question="Write a haiku about coffee",
        llm_response="""<thinking>
A haiku has the structure 5-7-5 syllables.
Theme: coffee
Let me think about coffee imagery:
- Morning ritual
- Dark, rich brew
- Warmth and energy
- Steam rising

Draft:
"Morning warmth arrives" (5)
"Dark brew fills my favorite cup" (7)
"Day begins anew" (5)

This captures the essence and follows the form.
</thinking>

Morning warmth arrives
Dark brew fills my favorite cup
Day begins anew"""
    )

    # Scenario 5: Error handling (malformed)
    demo_scenario(
        title="Malformed Thinking Block (Graceful Handling)",
        user_question="Test question",
        llm_response="<thinking>Oops, forgot to close the tag. Here's my answer anyway."
    )

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("""
Key Takeaways:
✓ Thinking blocks provide transparency into LLM reasoning
✓ Users only see polished final answers
✓ Debugging logs capture the thinking process
✓ Backward compatible - works with or without thinking tags
✓ Gracefully handles malformed responses
    """)


if __name__ == "__main__":
    main()
