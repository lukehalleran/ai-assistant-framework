#!/usr/bin/env python3
"""
Manual test script for Agentic Search

Run with: python test_agentic_search_manual.py

This script tests the agentic search loop directly without the full Daemon stack.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see what's happening
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def test_agentic_search():
    """Test the agentic search controller directly."""

    print("\n" + "="*60)
    print("AGENTIC SEARCH TEST")
    print("="*60 + "\n")

    # Import components
    try:
        from models.model_manager import ModelManager
        from knowledge.web_search_manager import WebSearchManager
        from core.agentic import AgenticSearchController, ProgressEvent
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return

    # Initialize model manager
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ OPENAI_API_KEY not set")
        return
    print("✓ API key found")

    model_manager = ModelManager(api_key=api_key)
    model_manager.switch_model("deepseek-r1")  # Or your preferred model
    print(f"✓ Model manager initialized (active: {model_manager.get_active_model_name()})")

    # Initialize web search manager with Tavily API key from config
    import yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tavily_key = config.get("web_search", {}).get("api_key", "")

    web_search_manager = WebSearchManager(
        api_key=tavily_key,
        default_timeout=30.0,
        max_content_chars=10000,
    )
    print(f"✓ Web search manager initialized (API key: {'set' if tavily_key else 'missing'})")

    # Initialize agentic controller
    controller = AgenticSearchController(
        model_manager=model_manager,
        web_search_manager=web_search_manager,
        max_rounds=5,
        context_budget_tokens=8000,
        compression_model="gpt-4o-mini",
    )
    print("✓ Agentic controller initialized")

    # Test query
    test_query = "What's the K flu variant and how serious is it?"
    initial_terms = ["K flu variant 2026", "H5N1 bird flu current outbreak"]

    print(f"\n{'─'*60}")
    print(f"Query: {test_query}")
    print(f"Initial search terms: {initial_terms}")
    print(f"{'─'*60}\n")

    # Run agentic search
    full_response = ""
    events = []

    try:
        async for item in controller.run_agentic_search(
            query=test_query,
            system_prompt="You are a helpful assistant that provides accurate, well-sourced information.",
            model_name=model_manager.get_active_model_name(),
            initial_search_terms=initial_terms,
        ):
            if isinstance(item, ProgressEvent):
                events.append(item)
                icon = {
                    "searching": "🔍",
                    "found_results": "📄",
                    "synthesizing": "✨",
                    "done": "✅",
                    "error": "❌",
                }.get(item.event_type, "•")
                print(f"{icon} [{item.event_type}] {item.message}")
            else:
                # Response chunk
                full_response += item
                print(item, end="", flush=True)

        print("\n")

    except Exception as e:
        print(f"\n❌ Error during agentic search: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total rounds: {len([e for e in events if e.event_type == 'searching'])}")
    print(f"Response length: {len(full_response)} chars")

    # Show events
    print(f"\nEvents timeline:")
    for e in events:
        print(f"  - Round {e.round_number}: {e.event_type} - {e.message}")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(test_agentic_search())
