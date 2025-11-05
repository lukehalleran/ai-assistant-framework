#!/usr/bin/env python3
"""
Quick fail-fast test: Compare baseline vs competitive selection on 2 questions.

This doesn't modify your system — it just shows what the prompts would look like.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.prompt import UnifiedPromptBuilder
from core.competitive_scorer import apply_competitive_selection
from models.model_manager import ModelManager


async def test_competitive_vs_baseline():
    """Run side-by-side comparison."""

    # Initialize minimal components
    print("🔧 Initializing components...")
    model_manager = ModelManager()

    # Create prompt builder (will auto-init memory coordinator etc)
    builder = UnifiedPromptBuilder(
        model_manager=model_manager,
    )

    # Test questions
    questions = [
        "What are the key facts about machine learning?",
        "Tell me about recent conversations we've had",
    ]

    for i, query in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {query}")
        print(f"{'='*80}\n")

        # Gather context (baseline path)
        print("📦 Gathering context...")
        context_baseline = await builder._gather_context(
            user_input=query,
            include_dreams=False,
            include_wiki=False,
            include_semantic=True,
            personality_config=None
        )

        # Apply gating if available (baseline does this)
        if builder.gate_system:
            try:
                context_baseline = await builder._apply_gating(query, context_baseline)
            except Exception as e:
                print(f"⚠️  Gating failed (ok for test): {e}")

        # Count items in baseline
        baseline_counts = {
            field: len(context_baseline.get(field, []))
            for field in ["memories", "facts", "semantic_chunks", "recent_conversations", "summaries"]
        }

        # Apply competitive selection
        print("🏆 Applying competitive selection...")
        context_competitive = apply_competitive_selection(
            context=dict(context_baseline),  # deep copy
            query=query,
            tokenizer_manager=builder.tokenizer_manager,
            model_name=model_manager.get_active_model_name(),
            budget=2000  # Conservative budget for comparison
        )

        # Count items after competitive selection
        competitive_counts = {
            field: len(context_competitive.get(field, []))
            for field in ["memories", "facts", "semantic_chunks", "recent_conversations", "summaries"]
        }

        # Show comparison
        print(f"\n📊 BASELINE field counts:")
        for field, count in baseline_counts.items():
            if count > 0:
                print(f"   {field}: {count}")

        print(f"\n🏆 COMPETITIVE field counts:")
        for field, count in competitive_counts.items():
            if count > 0:
                print(f"   {field}: {count}")

        # Show what got selected
        print(f"\n🔍 Top items selected by competitive scorer:")
        for field in ["facts", "semantic_chunks", "memories", "recent_conversations", "summaries"]:
            items = context_competitive.get(field, [])
            if items:
                print(f"\n   [{field.upper()}] ({len(items)} items)")
                for idx, item in enumerate(items[:2], 1):  # Show first 2
                    text = _get_preview(item)
                    print(f"      {idx}. {text[:100]}...")

        print(f"\n{'─'*80}\n")

    print("\n✅ Test complete! Review the field distributions above.")
    print("   If competitive selection shows better balance/relevance, we proceed.")
    print("   If it looks worse or same, we kill the experiment.\n")


def _get_preview(item):
    """Get preview text from item."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ["content", "text", "response", "query"]:
            if key in item:
                return str(item[key])
    return str(item)


if __name__ == "__main__":
    try:
        asyncio.run(test_competitive_vs_baseline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
