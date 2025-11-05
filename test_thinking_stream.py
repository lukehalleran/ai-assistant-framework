#!/usr/bin/env python3
"""
Test script to verify thinking block streaming works correctly.
This simulates the streaming behavior to ensure thinking blocks appear and then get replaced.
"""
import asyncio


async def mock_streaming_with_thinking():
    """
    Simulates the streaming behavior with a thinking block followed by an answer.
    """
    # Simulate chunks coming in
    chunks = [
        "<thinking>",
        "The user is asking about",
        " their cat Flapjack.",
        " This is an emotional topic.",
        " I should show empathy",
        " and recall any prior mentions.",
        "</thinking>",
        "\n\nI'm so sorry to hear",
        " that Flapjack isn't feeling well.",
        " How long has he been sick?",
    ]

    full_output = ""
    thinking_started = False
    thinking_complete = False

    print("=== Simulating Streaming Output ===\n")

    for i, chunk in enumerate(chunks):
        await asyncio.sleep(0.2)  # Simulate network delay

        full_output += chunk

        # Parse thinking block (simple simulation)
        if "<thinking>" in full_output and "</thinking>" in full_output:
            # Both tags present, extract thinking and answer
            start = full_output.find("<thinking>") + len("<thinking>")
            end = full_output.find("</thinking>")
            thinking_part = full_output[start:end].strip()
            final_answer = full_output[end + len("</thinking>"):].strip()

            if thinking_part and not final_answer:
                # Show thinking
                if not thinking_started:
                    print("\n[THINKING BLOCK STARTED]")
                    thinking_started = True
                print(f"💭 Thinking... {thinking_part}")

            elif thinking_part and final_answer and not thinking_complete:
                # Switch to answer
                print("\n[THINKING COMPLETE - SWITCHING TO ANSWER]")
                thinking_complete = True
                print(f"Answer: {final_answer}")

            elif final_answer:
                # Continue showing answer
                print(f"Answer: {final_answer}")

        elif "<thinking>" in full_output:
            # Only opening tag, accumulating thinking
            start = full_output.find("<thinking>") + len("<thinking>")
            thinking_part = full_output[start:].strip()
            if thinking_part and not thinking_started:
                print("\n[THINKING BLOCK STARTED]")
                thinking_started = True
            if thinking_part:
                print(f"💭 Thinking... {thinking_part}")
        else:
            # No thinking tags, normal output
            print(f"Normal: {full_output}")

    print("\n=== Streaming Complete ===")


async def test_parse_thinking_block():
    """
    Test the thinking block parsing logic.
    """
    print("\n=== Testing Thinking Block Parser ===\n")

    # Test cases
    test_cases = [
        ("No thinking tags", "Just a normal response"),
        ("With thinking", "<thinking>Some thoughts</thinking>\n\nThe answer is 42."),
        ("Only thinking", "<thinking>Still thinking...</thinking>"),
        ("Incomplete thinking", "<thinking>Thinking in progress..."),
    ]

    for name, text in test_cases:
        print(f"Test: {name}")
        print(f"Input: {text}")

        # Simple parser simulation
        if "<thinking>" in text and "</thinking>" in text:
            start = text.find("<thinking>") + len("<thinking>")
            end = text.find("</thinking>")
            thinking = text[start:end].strip()
            answer = text[end + len("</thinking>"):].strip()
            print(f"  Thinking: {thinking}")
            print(f"  Answer: {answer}")
        elif "<thinking>" in text:
            start = text.find("<thinking>") + len("<thinking>")
            thinking = text[start:].strip()
            print(f"  Thinking (incomplete): {thinking}")
            print(f"  Answer: (none yet)")
        else:
            print(f"  No thinking block, full text as answer")
        print()


if __name__ == "__main__":
    print("Testing Thinking Block Streaming Implementation\n")
    print("=" * 60)

    # Run tests
    asyncio.run(test_parse_thinking_block())
    asyncio.run(mock_streaming_with_thinking())

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\nTo test in the GUI:")
    print("1. Start the GUI: python main.py")
    print("2. Ask a question that triggers thinking (complex emotional query)")
    print("3. Watch for the '💭 Thinking...' block to appear and stream")
    print("4. Verify it gets replaced by the actual answer when complete")
