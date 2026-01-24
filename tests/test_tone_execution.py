#!/usr/bin/env python3
"""
Test script to verify tone detection actually executes in orchestrator

NOTE: This is a manual integration test that requires full system initialization.
Run with: python tests/test_tone_execution.py
"""
import asyncio
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Skip when run via pytest - this is a manual integration script
pytestmark = pytest.mark.skip(reason="Manual integration test - run directly with python")

from utils.logging_utils import get_logger

async def test_tone_detection():
    """Test that tone detection runs when processing a query"""
    # Import here to avoid import errors during pytest collection
    from core.orchestrator import DaemonOrchestrator
    from models.model_manager import ModelManager
    from core.response_generator import ResponseGenerator
    from utils.file_processor import FileProcessor
    from core.prompt.builder import UnifiedPromptBuilder
    from core.prompt.token_manager import TokenManager

    logger = get_logger("test_tone")
    logger.info("=" * 60)
    logger.info("STARTING TONE DETECTION EXECUTION TEST")
    logger.info("=" * 60)

    # Create minimal orchestrator with required components
    model_manager = ModelManager()
    response_generator = ResponseGenerator(model_manager=model_manager)
    file_processor = FileProcessor()
    token_manager = TokenManager()
    prompt_builder = UnifiedPromptBuilder(
        memory_coordinator=None,
        model_manager=model_manager,
        token_manager=token_manager
    )

    orchestrator = DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder,
        config={}
    )

    test_inputs = [
        "hey! sorry lost connection",
        "I'm feeling overwhelmed",
        "Want to die",
        "Woke up at 10"
    ]

    for test_input in test_inputs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing input: '{test_input}'")
        logger.info("=" * 60)

        try:
            # We expect this to fail during response generation, but tone detection should run first
            response, debug_info = await orchestrator.process_user_query(
                user_input=test_input,
                use_raw_mode=True  # Skip some processing
            )
        except Exception as e:
            # We expect errors since we have no model
            logger.info(f"Expected error during generation: {type(e).__name__}: {e}")

        # Check if tone detection ran
        tone_level = getattr(orchestrator, 'current_tone_level', None)
        logger.info(f"Tone level after processing: {tone_level}")

        if tone_level:
            logger.info("✓ TONE DETECTION EXECUTED")
        else:
            logger.error("✗ TONE DETECTION DID NOT EXECUTE")

if __name__ == "__main__":
    asyncio.run(test_tone_detection())
