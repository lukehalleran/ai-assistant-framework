#!/usr/bin/env python3
"""
Test script to verify tone detection actually executes in orchestrator
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import DaemonOrchestrator
from utils.logging_utils import get_logger

async def test_tone_detection():
    """Test that tone detection runs when processing a query"""

    logger = get_logger("test_tone")
    logger.info("=" * 60)
    logger.info("STARTING TONE DETECTION EXECUTION TEST")
    logger.info("=" * 60)

    # Create minimal orchestrator
    orchestrator = DaemonOrchestrator(
        model_manager=None,  # We won't actually call the model
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
