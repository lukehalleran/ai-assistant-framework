#!/usr/bin/env python3
"""Test script to check prompt sections and timestamps."""

import asyncio
import sys
import os
sys.path.append(os.path.abspath('.'))

from config.app_config import config
from models.model_manager import ModelManager
from memory.memory_coordinator import MemoryCoordinator
from core.prompt.builder import UnifiedPromptBuilder
from core.prompt.context_gatherer import ContextGatherer
from core.prompt.formatter import PromptFormatter
from core.prompt.summarizer import LLMSummarizer
from core.prompt.token_manager import TokenManager

async def test_prompt():
    """Test prompt building with sections and timestamps."""

    # Initialize components
    model_manager = ModelManager()
    memory_coordinator = MemoryCoordinator()
    await memory_coordinator.initialize()

    token_manager = TokenManager()
    context_gatherer = ContextGatherer(memory_coordinator, model_manager, token_manager)
    formatter = PromptFormatter(token_manager)
    summarizer = LLMSummarizer(model_manager, memory_coordinator)

    builder = UnifiedPromptBuilder(
        memory_coordinator=memory_coordinator,
        model_manager=model_manager,
        token_manager=token_manager,
        context_gatherer=context_gatherer,
        formatter=formatter,
        summarizer=summarizer
    )

    # Build prompt
    user_query = "Tell me about quantum computing and extract key facts."
    print("Building prompt...")

    context = await builder.build_prompt(user_query)

    print("\n=== PROMPT CONTEXT SECTIONS ===")
    for section, data in context.items():
        if isinstance(data, list):
            print(f"{section.upper()}: {len(data)} items")
            if data:
                # Show first item as example
                first_item = data[0]
                if isinstance(first_item, dict):
                    content_preview = str(first_item.get('content', first_item))[:60]
                    timestamp = first_item.get('timestamp', 'no timestamp')
                    if not timestamp:
                        metadata = first_item.get('metadata', {})
                        timestamp = metadata.get('timestamp', 'no metadata timestamp')
                    print(f"  Example: {content_preview}... | {timestamp}")
                else:
                    print(f"  Example: {str(first_item)[:60]}...")
        else:
            print(f"{section.upper()}: {type(data)}")

    # Test formatter
    print("\n=== FORMATTED PROMPT SECTIONS ===")
    formatted = formatter._assemble_prompt(context, user_query)

    # Find each section
    sections = ['[RECENT FACTS]', '[SEMANTIC FACTS]', '[SUMMARIES]', '[RECENT REFLECTIONS]', '[DREAMS]', '[RECENT CONVERSATION]', '[RELEVANT MEMORIES]']
    for section in sections:
        if section in formatted:
            print(f"✓ {section} found")
            # Extract a sample line
            start = formatted.find(section)
            end = formatted.find('\n\n', start)
            if end == -1:
                end = start + 200
            sample = formatted[start:end]
            print(f"  Sample: {sample[:150]}...")
        else:
            print(f"✗ {section} missing")

if __name__ == "__main__":
    asyncio.run(test_prompt())