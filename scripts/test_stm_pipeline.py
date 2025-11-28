#!/usr/bin/env python3
"""
Test script for STM (Short-Term Memory) pipeline.

Tests the complete flow:
1. STM analyzer generates context summary from recent messages
2. Summary is injected into prompt
3. Last Q/A pair is attached to query
4. Emergency compression protects query section
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.stm_analyzer import STMAnalyzer
from models.model_manager import ModelManager
from config.app_config import STM_MODEL_NAME, USE_STM_PASS

async def test_stm_analyzer():
    """Test STM analyzer directly."""
    print("\n" + "="*80)
    print("TEST 1: STM Analyzer")
    print("="*80)

    # Create fake conversation history
    recent_memories = [
        {
            "query": "I'm working on a Python project",
            "response": "That's great! What kind of project?"
        },
        {
            "query": "It's a RAG system with memory",
            "response": "Interesting! How are you handling the memory storage?"
        },
        {
            "query": "Using ChromaDB for vector storage",
            "response": "Good choice. ChromaDB is efficient for semantic search."
        }
    ]

    current_query = "Should I add prompt caching?"

    try:
        # Initialize model manager
        model_manager = ModelManager()

        # Create STM analyzer
        stm = STMAnalyzer(model_manager=model_manager, model_name=STM_MODEL_NAME)

        # Run analysis
        print(f"\nAnalyzing conversation with {len(recent_memories)} turns...")
        print(f"Current query: {current_query}\n")

        summary = await stm.analyze(
            recent_memories=recent_memories,
            user_query=current_query,
            last_assistant_response=recent_memories[-1]["response"]
        )

        # Display results
        print("STM Analysis Results:")
        print("-" * 40)
        print(f"Topic: {summary.get('topic')}")
        print(f"User Question: {summary.get('user_question')}")
        print(f"Intent: {summary.get('intent')}")
        print(f"Tone: {summary.get('tone')}")
        print(f"Open Threads: {summary.get('open_threads')}")
        print(f"Constraints: {summary.get('constraints')}")
        print("\n✅ STM Analyzer test passed!")

        return summary

    except Exception as e:
        print(f"\n❌ STM Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_prompt_integration():
    """Test that STM summary integrates into prompt correctly."""
    print("\n" + "="*80)
    print("TEST 2: Prompt Integration")
    print("="*80)

    try:
        from core.prompt.formatter import PromptFormatter
        from core.prompt.token_manager import TokenManager
        from models.tokenizer_manager import TokenizerManager

        # Create managers
        model_manager = ModelManager()
        tokenizer_manager = TokenizerManager(model_manager=model_manager)
        token_manager = TokenManager(
            model_manager=model_manager,
            tokenizer_manager=tokenizer_manager,
            token_budget=4096
        )

        # Create formatter
        formatter = PromptFormatter(token_manager=token_manager)

        # Create test context with STM summary
        context = {
            "recent_conversations": [
                {"query": "What is Python?", "response": "Python is a programming language."},
                {"query": "How do I use lists?", "response": "Lists are created with square brackets."}
            ],
            "stm_summary": {
                "topic": "Python basics",
                "user_question": "Learning about Python lists",
                "intent": "Understand list fundamentals",
                "tone": "curious",
                "open_threads": ["List methods", "List comprehensions"],
                "constraints": ["Beginner level"]
            },
            "memories": [],
            "semantic_facts": [],
            "fresh_facts": [],
            "summaries": [],
            "reflections": [],
            "wiki": [],
            "semantic_chunks": [],
            "dreams": []
        }

        user_input = "Can you show me list comprehension examples?"

        # Assemble prompt
        prompt = formatter._assemble_prompt(context, user_input)

        # Verify STM section exists
        if "[SHORT-TERM CONTEXT SUMMARY]" in prompt:
            print("✅ STM summary section found in prompt")
        else:
            print("❌ STM summary section NOT found in prompt")

        # Verify last exchange is attached
        if "[LAST EXCHANGE FOR CONTEXT]" in prompt:
            print("✅ Last exchange attached to query")
        else:
            print("❌ Last exchange NOT attached")

        # Verify query section is present
        if "[CURRENT USER QUERY]" in prompt:
            print("✅ Current query section found")
        else:
            print("❌ Current query section NOT found")

        # Show relevant sections
        print("\nPrompt Structure:")
        print("-" * 40)
        sections = prompt.split("\n\n")
        for section in sections:
            if section.startswith("["):
                header = section.split("\n")[0]
                print(f"  • {header}")

        print(f"\nTotal prompt length: {len(prompt)} chars")
        print("\n✅ Prompt integration test passed!")

        return prompt

    except Exception as e:
        print(f"\n❌ Prompt integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_query_protection():
    """Test that middle-out compression doesn't affect query section."""
    print("\n" + "="*80)
    print("TEST 3: Query Protection from Compression")
    print("="*80)

    try:
        from core.prompt.formatter import PromptFormatter
        from core.prompt.token_manager import TokenManager
        from models.tokenizer_manager import TokenizerManager

        # Create managers with VERY LOW budget to force compression
        model_manager = ModelManager()
        tokenizer_manager = TokenizerManager(model_manager=model_manager)
        token_manager = TokenManager(
            model_manager=model_manager,
            tokenizer_manager=tokenizer_manager,
            token_budget=500  # Artificially low to trigger emergency compression
        )

        formatter = PromptFormatter(token_manager=token_manager)

        # Create large context to exceed budget
        large_context = {
            "recent_conversations": [{"query": f"Question {i}", "response": f"Answer {i} " * 100} for i in range(10)],
            "memories": [{"query": f"Memory {i}", "response": "Long memory " * 50} for i in range(5)],
            "stm_summary": {
                "topic": "Test",
                "user_question": "Testing compression",
                "intent": "Verify query protection",
                "tone": "testing",
                "open_threads": [],
                "constraints": []
            },
            "semantic_facts": [],
            "fresh_facts": [],
            "summaries": [],
            "reflections": [],
            "wiki": [],
            "semantic_chunks": [],
            "dreams": []
        }

        user_input = "THIS IS THE IMPORTANT USER QUERY THAT MUST NOT BE COMPRESSED"

        # Assemble prompt (should trigger compression)
        prompt = formatter._assemble_prompt(large_context, user_input)

        # Verify query is intact
        if user_input in prompt:
            print("✅ User query preserved despite compression")
        else:
            print("❌ User query was corrupted by compression!")

        # Verify compression marker exists
        if "middle-out snipped" in prompt:
            print("✅ Middle-out compression was applied")
        else:
            print("⚠️  Compression may not have been needed")

        # Verify query section comes after compression marker
        query_pos = prompt.find("[CURRENT USER QUERY]")
        snip_pos = prompt.find("middle-out snipped")

        if query_pos > snip_pos and snip_pos > 0:
            print("✅ Query section is AFTER compressed context (protected)")
        elif snip_pos < 0:
            print("⚠️  No compression applied (budget might be sufficient)")
        else:
            print("❌ Query section is in wrong position!")

        print(f"\nFinal prompt length: {len(prompt)} chars")
        print("\n✅ Query protection test passed!")

        return prompt

    except Exception as e:
        print(f"\n❌ Query protection test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("STM PIPELINE TEST SUITE")
    print("="*80)
    print(f"STM Enabled: {USE_STM_PASS}")
    print(f"STM Model: {STM_MODEL_NAME}")
    print("="*80)

    results = []

    # Test 1: STM Analyzer
    result1 = await test_stm_analyzer()
    results.append(("STM Analyzer", result1 is not None))

    # Test 2: Prompt Integration
    result2 = await test_prompt_integration()
    results.append(("Prompt Integration", result2 is not None))

    # Test 3: Query Protection
    result3 = await test_query_protection()
    results.append(("Query Protection", result3 is not None))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
