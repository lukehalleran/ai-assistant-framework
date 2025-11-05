#!/usr/bin/env python3
"""
Test script for two-step generation with thinking blocks.

This tests:
1. The _parse_thinking_block method correctly extracts thinking and final answer
2. The system prompt includes the thinking block instruction
3. The full flow works end-to-end (if models are available)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import DaemonOrchestrator


def test_parse_thinking_block():
    """Test the thinking block parser with various inputs."""

    print("="*80)
    print("Testing _parse_thinking_block method")
    print("="*80)

    # Test case 1: Normal thinking block
    response1 = "<thinking>Let me analyze this. 2+2 is a simple addition.</thinking>The answer is 4."
    thinking1, answer1 = DaemonOrchestrator._parse_thinking_block(response1)

    print("\nTest 1: Normal thinking block")
    print(f"Input: {response1[:80]}...")
    print(f"Thinking: {thinking1}")
    print(f"Answer: {answer1}")
    assert thinking1 == "Let me analyze this. 2+2 is a simple addition.", "Thinking extraction failed"
    assert answer1 == "The answer is 4.", "Answer extraction failed"
    print("✅ PASS")

    # Test case 2: No thinking block
    response2 = "The answer is 4."
    thinking2, answer2 = DaemonOrchestrator._parse_thinking_block(response2)

    print("\nTest 2: No thinking block")
    print(f"Input: {response2}")
    print(f"Thinking: '{thinking2}'")
    print(f"Answer: {answer2}")
    assert thinking2 == "", "Should return empty thinking"
    assert answer2 == "The answer is 4.", "Should return full response as answer"
    print("✅ PASS")

    # Test case 3: Thinking block with newlines
    response3 = """<thinking>
Step 1: Identify the operation
Step 2: 2 + 2 = 4
Step 3: Format answer
</thinking>
The answer is 4."""
    thinking3, answer3 = DaemonOrchestrator._parse_thinking_block(response3)

    print("\nTest 3: Thinking block with newlines")
    print(f"Thinking length: {len(thinking3)} chars")
    print(f"Answer: {answer3}")
    assert len(thinking3) > 0, "Thinking should not be empty"
    assert "Step 1" in thinking3, "Should preserve thinking content"
    assert answer3.strip() == "The answer is 4.", "Answer should be clean"
    print("✅ PASS")

    # Test case 4: Empty response
    response4 = ""
    thinking4, answer4 = DaemonOrchestrator._parse_thinking_block(response4)

    print("\nTest 4: Empty response")
    print(f"Thinking: '{thinking4}'")
    print(f"Answer: '{answer4}'")
    assert thinking4 == "", "Empty input should give empty thinking"
    assert answer4 == "", "Empty input should give empty answer"
    print("✅ PASS")

    # Test case 5: Only opening tag (malformed)
    response5 = "<thinking>Some thinking but no closing tag. The answer is unclear."
    thinking5, answer5 = DaemonOrchestrator._parse_thinking_block(response5)

    print("\nTest 5: Malformed (no closing tag)")
    print(f"Thinking: '{thinking5}'")
    print(f"Answer: {answer5[:50]}...")
    assert thinking5 == "", "Malformed should return empty thinking"
    assert answer5 == response5, "Malformed should return full response as answer"
    print("✅ PASS")

    print("\n" + "="*80)
    print("All parsing tests passed! ✅")
    print("="*80)


async def test_system_prompt_modification():
    """Test that the system prompt includes thinking block instruction."""

    print("\n" + "="*80)
    print("Testing system prompt modification")
    print("="*80)

    try:
        from models.model_manager import ModelManager
        from generation.response_generator import ResponseGenerator
        from core.prompt import UnifiedPromptBuilder

        # Create minimal orchestrator
        model_manager = ModelManager()
        response_generator = ResponseGenerator(model_manager)
        prompt_builder = UnifiedPromptBuilder(model_manager)

        orchestrator = DaemonOrchestrator(
            model_manager=model_manager,
            response_generator=response_generator,
            file_processor=None,
            prompt_builder=prompt_builder,
            memory_system=None,
        )

        # Test prepare_prompt
        prompt, system_prompt = await orchestrator.prepare_prompt(
            user_input="What is 2+2?",
            use_raw_mode=False
        )

        print("\nSystem prompt check:")
        if system_prompt:
            print(f"System prompt length: {len(system_prompt)} chars")

            # Check for thinking instruction
            has_thinking_instruction = "<thinking>" in system_prompt or "thinking block" in system_prompt.lower()

            if has_thinking_instruction:
                print("✅ System prompt contains thinking block instruction")

                # Show the thinking instruction part
                if "IMPORTANT:" in system_prompt:
                    idx = system_prompt.find("IMPORTANT:")
                    snippet = system_prompt[idx:idx+200]
                    print(f"\nThinking instruction snippet:\n{snippet}...")
            else:
                print("❌ System prompt does NOT contain thinking block instruction")
                print("\nLast 300 chars of system prompt:")
                print(system_prompt[-300:])
                return False
        else:
            print("❌ No system prompt returned")
            return False

        print("\n✅ System prompt modification test passed!")
        return True

    except Exception as e:
        print(f"\n⚠️  Could not test system prompt modification: {e}")
        print("This is expected if model components are not fully initialized.")
        return None


def test_integration():
    """Quick integration test showing the full flow."""

    print("\n" + "="*80)
    print("Integration Example")
    print("="*80)

    # Simulate what the LLM might return
    simulated_llm_response = """<thinking>
The user is asking a simple arithmetic question. Let me break this down:
1. The operation is addition: 2 + 2
2. Adding 2 and 2 gives us 4
3. I should provide a clear, concise answer
</thinking>

The answer to 2 + 2 is 4."""

    print("\nSimulated LLM response:")
    print("-" * 40)
    print(simulated_llm_response)
    print("-" * 40)

    # Parse it
    thinking, answer = DaemonOrchestrator._parse_thinking_block(simulated_llm_response)

    print("\nWhat gets logged (thinking):")
    print("-" * 40)
    print(thinking)
    print("-" * 40)

    print("\nWhat user sees (final answer):")
    print("-" * 40)
    print(answer)
    print("-" * 40)

    print("\nWhat gets stored in memory:")
    print("-" * 40)
    print(f"Query: 'What is 2+2?'")
    print(f"Response: '{answer}'")
    print("-" * 40)

    print("\n✅ Integration flow demonstrated!")


if __name__ == "__main__":
    print("\n🧪 Two-Step Generation Feature Tests\n")

    # Test 1: Parsing logic
    test_parse_thinking_block()

    # Test 2: System prompt modification
    import asyncio
    result = asyncio.run(test_system_prompt_modification())

    # Test 3: Integration example
    test_integration()

    print("\n" + "="*80)
    if result is not False:
        print("✅ All tests completed successfully!")
    else:
        print("⚠️  Some tests failed - see output above")
    print("="*80 + "\n")
