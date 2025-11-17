#!/usr/bin/env python3
"""Test tone detection to see if it's stuck on crisis mode."""

import asyncio
from utils.tone_detector import detect_crisis_level, CrisisLevel
from models.model_manager import ModelManager
from config.app_config import config

# Test queries that should NOT trigger crisis mode
test_queries = [
    "How much can I bench press?",
    "What's 2+2?",
    "Tell me about my workout routine",
    "What did we discuss yesterday?",
    "How do I feel about ICE",
    "What's for dinner?",
    "I'm a bit tired today",
    "I'm feeling tired",
]

async def main():
    # Initialize model manager
    model_manager = ModelManager(config)

    print("Testing tone detection on casual queries:\n")
    print("=" * 80)

    for query in test_queries:
        # Detect tone
        analysis = await detect_crisis_level(
            message=query,
            conversation_history=None,
            model_manager=model_manager
        )

        # Format output
        level_str = analysis.level.value
        status = "✓" if analysis.level == CrisisLevel.CONVERSATIONAL else "✗ CRISIS"

        print(f"{status} Query: {query}")
        print(f"  Level: {level_str} (confidence: {analysis.confidence:.3f})")
        print(f"  Trigger: {analysis.trigger}")
        if analysis.raw_scores:
            print(f"  Scores: high={analysis.raw_scores.get('high', 0):.3f}, "
                  f"medium={analysis.raw_scores.get('medium', 0):.3f}, "
                  f"concern={analysis.raw_scores.get('concern', 0):.3f}, "
                  f"conversational={analysis.raw_scores.get('conversational', 0):.3f}")
        print()

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
