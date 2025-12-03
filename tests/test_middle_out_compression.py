#!/usr/bin/env python3
"""Test script for hybrid middle-out compression."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.prompt.token_manager import TokenManager
from models.model_manager import ModelManager
from models.tokenizer_manager import TokenizerManager

def test_per_item_compression():
    """Test that individual large items get compressed."""
    print("=" * 60)
    print("Test 1: Per-Item Compression")
    print("=" * 60)

    # Create managers
    model_manager = ModelManager(api_key="test")
    model_manager.switch_model("gpt-4o")
    tokenizer_manager = TokenizerManager(model_manager)
    token_manager = TokenManager(model_manager, tokenizer_manager, token_budget=2000)

    # Create a very long memory item (way over 512 tokens)
    long_text = "This is a very long memory. " * 200  # ~1000+ tokens

    context = {
        "memories": [
            {"content": long_text, "timestamp": "2025-11-21"},
            {"content": "Short memory", "timestamp": "2025-11-21"}
        ]
    }

    print(f"Before compression:")
    print(f"  Memory 1: ~{token_manager.get_token_count(long_text, 'gpt-4o')} tokens")

    # Apply budget management (should compress the large item)
    result = token_manager._manage_token_budget(context)

    print(f"\nAfter compression:")
    if result.get("memories"):
        for i, mem in enumerate(result["memories"], 1):
            mem_text = mem.get("content", "")
            tokens = token_manager.get_token_count(mem_text, 'gpt-4o')
            print(f"  Memory {i}: {tokens} tokens")
            if "middle-out snipped" in mem_text:
                print(f"    ✅ Compressed!")

    print()

def test_whole_prompt_fallback():
    """Test that whole-prompt compression kicks in as fallback."""
    print("=" * 60)
    print("Test 2: Whole-Prompt Fallback Compression")
    print("=" * 60)

    # Create managers
    model_manager = ModelManager(api_key="test")
    model_manager.switch_model("gpt-4o")
    tokenizer_manager = TokenizerManager(model_manager)
    token_manager = TokenManager(model_manager, tokenizer_manager, token_budget=1000)

    # Create a massive prompt that will exceed budget even after per-item compression
    huge_prompt = "[SYSTEM PROMPT]\n" + ("Context line. " * 500) + "\n\n[USER QUERY]\nHello"

    tokens_before = token_manager.get_token_count(huge_prompt, 'gpt-4o')
    print(f"Huge prompt: {tokens_before} tokens (budget: 1000)")

    # Apply whole-prompt compression
    compressed = token_manager._middle_out(huge_prompt, 1000, force=True)
    tokens_after = token_manager.get_token_count(compressed, 'gpt-4o')

    print(f"After compression: {tokens_after} tokens")
    print(f"Reduction: {tokens_before - tokens_after} tokens")

    if "middle-out snipped" in compressed:
        print("✅ Whole-prompt compression applied!")
    else:
        print("❌ Compression not detected")

    # Verify structure preserved
    if "[SYSTEM PROMPT]" in compressed and "[USER QUERY]" in compressed:
        print("✅ Start and end preserved!")
    else:
        print("❌ Structure lost")

    print()

def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Hybrid Middle-Out Compression Test" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    try:
        test_per_item_compression()
        test_whole_prompt_fallback()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
