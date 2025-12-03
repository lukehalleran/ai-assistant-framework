#!/usr/bin/env python3
"""Test script for prompt caching implementation."""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_manager import ModelManager

def test_caching_support():
    """Test which models are detected as supporting caching."""
    print("Testing prompt caching support detection...\n")

    # Create ModelManager instance (without API key for testing)
    manager = ModelManager(api_key="test_key")

    test_models = [
        "claude-opus",
        "sonnet-4.5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-5",
        "gemini-3-pro",
        "deepseek-v3.1",
        "deepseek-r1",
        "glm-4.6",
    ]

    print("Model Caching Support:")
    print("-" * 60)
    for model in test_models:
        supports_caching = manager.supports_prompt_caching(model)
        full_name = manager.api_models.get(model, "N/A")
        status = "✅ YES" if supports_caching else "❌ NO"
        print(f"{model:20} -> {full_name:40} {status}")

    print("\n")

def test_message_formatting():
    """Test message formatting with cache_control."""
    print("Testing message formatting with cache_control...\n")

    manager = ModelManager(api_key="test_key")

    system_prompt = "You are a helpful assistant with extensive knowledge."
    user_prompt = "What is the capital of France?"

    messages = manager._format_messages_with_cache(system_prompt, user_prompt)

    print("Formatted messages structure:")
    print("-" * 60)
    for i, msg in enumerate(messages):
        print(f"Message {i}:")
        print(f"  Role: {msg['role']}")
        if isinstance(msg['content'], list):
            print(f"  Content type: list (with cache_control)")
            for j, part in enumerate(msg['content']):
                print(f"    Part {j}:")
                print(f"      Type: {part.get('type')}")
                print(f"      Text preview: {part.get('text', '')[:50]}...")
                if 'cache_control' in part:
                    print(f"      Cache control: {part['cache_control']}")
        else:
            print(f"  Content type: string")
            print(f"  Content preview: {msg['content'][:50]}...")
        print()

def main():
    print("=" * 60)
    print("Prompt Caching Implementation Test")
    print("=" * 60)
    print()

    try:
        test_caching_support()
        test_message_formatting()
        print("✅ All tests completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
