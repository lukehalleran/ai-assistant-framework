#!/usr/bin/env python3
"""Test cross-section deduplication in prompt builder."""

import asyncio
from core.prompt.builder import UnifiedPromptBuilder
from config.app_config import config
from core.dependencies import Dependencies

async def main():
    # Initialize dependencies
    deps = Dependencies(config)

    # Create builder
    builder = UnifiedPromptBuilder(
        memory_coordinator=deps.memory_coordinator,
        model_manager=deps.model_manager,
        tokenizer_manager=deps.tokenizer_manager,
        corpus_manager=deps.corpus_manager,
        wiki_manager=deps.wiki_manager,
        config=config
    )

    # Create test context with duplicates across sections
    test_context = {
        "recent_conversations": [
            {"content": "ICE is conducting operations in Logan Square with tear gas", "timestamp": "2025-10-08"},
            {"content": "The workout was great today", "timestamp": "2025-11-16"}
        ],
        "summaries": [
            {"content": "ICE is conducting operations in Logan Square with tear gas", "timestamp": "2025-10-08"},  # Duplicate
            {"content": "User discussed gym routine and bench press", "timestamp": "2025-11-15"}
        ],
        "semantic_summaries": [
            {"content": "ICE is conducting operations in Logan Square with tear gas and flash grenades", "timestamp": "2025-10-08"},  # Near-duplicate (first 200 chars match)
            {"content": "Discussion about protein intake and recovery", "timestamp": "2025-11-14"}
        ],
        "memories": [
            {"content": "The workout was great today", "timestamp": "2025-11-16"},  # Duplicate
            {"content": "Bench press progress: 135 lbs x 5", "timestamp": "2025-10-15"}
        ]
    }

    print("BEFORE cross-section dedup:")
    print(f"  recent_conversations: {len(test_context['recent_conversations'])} items")
    print(f"  summaries: {len(test_context['summaries'])} items")
    print(f"  semantic_summaries: {len(test_context['semantic_summaries'])} items")
    print(f"  memories: {len(test_context['memories'])} items")
    print(f"  TOTAL: {sum(len(v) for v in test_context.values())} items")
    print()

    # Apply hygiene and caps (which includes deduplication)
    cleaned = builder._hygiene_and_caps(test_context)

    print("AFTER cross-section dedup:")
    print(f"  recent_conversations: {len(cleaned['recent_conversations'])} items")
    print(f"  summaries: {len(cleaned['summaries'])} items")
    print(f"  semantic_summaries: {len(cleaned['semantic_summaries'])} items")
    print(f"  memories: {len(cleaned['memories'])} items")
    print(f"  TOTAL: {sum(len(v) for v in cleaned.values())} items")
    print()

    print("Expected: 4 items total (removed 4 duplicates)")
    print(f"Actual: {sum(len(v) for v in cleaned.values())} items")

    # Show what survived
    print("\nContent that survived deduplication:")
    for section in ["recent_conversations", "summaries", "semantic_summaries", "memories"]:
        if cleaned.get(section):
            print(f"\n{section}:")
            for item in cleaned[section]:
                content = item.get("content", "")
                print(f"  - {content[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())
