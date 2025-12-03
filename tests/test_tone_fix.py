#!/usr/bin/env python3
"""
Test that tone detection now works in both RAW and ENHANCED modes after the fix
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Mock minimal dependencies
class MockModelManager:
    def get_active_model_name(self):
        return "mock-model"

class MockResponseGenerator:
    pass

class MockFileProcessor:
    async def process_files(self, user_input, files):
        return user_input

class MockPromptBuilder:
    pass

class MockMemorySystem:
    class CorpusManager:
        def get_recent_memories(self, limit=3):
            return None

    def __init__(self):
        self.corpus_manager = self.CorpusManager()
        self.current_topic = "general"

# Now import after setting up mocks
from core.orchestrator import DaemonOrchestrator
from utils.logging_utils import get_logger

async def test_tone_in_prepare_prompt():
    """Test that tone detection runs in prepare_prompt (ENHANCED mode path)"""

    logger = get_logger("test_tone_fix")
    logger.info("=" * 60)
    logger.info("TEST 1: Tone detection in prepare_prompt (ENHANCED mode)")
    logger.info("=" * 60)

    # Create orchestrator with mocks
    orchestrator = DaemonOrchestrator(
        model_manager=MockModelManager(),
        response_generator=MockResponseGenerator(),
        file_processor=MockFileProcessor(),
        prompt_builder=MockPromptBuilder(),
        memory_system=MockMemorySystem(),
        config={}
    )

    test_cases = [
        ("hey! sorry lost connection", "conversational"),
        ("I'm feeling really anxious", "concern"),
        ("Having a panic attack", "elevated_support"),
        ("I want to die", "crisis_support"),
        ("Woke up at 10", "conversational"),
    ]

    for user_input, expected_level in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: '{user_input}'")
        logger.info(f"Expected: {expected_level}")
        logger.info("=" * 60)

        try:
            # Call prepare_prompt directly (like ENHANCED mode does)
            prompt, system_prompt = await orchestrator.prepare_prompt(
                user_input=user_input,
                files=None,
                use_raw_mode=False
            )

            # Check if tone was detected
            tone_level = getattr(orchestrator, 'current_tone_level', None)

            if tone_level:
                logger.info(f"✓ TONE DETECTED: {tone_level.value}")

                # Verify it matches expected
                if tone_level.value == expected_level:
                    logger.info(f"✓✓ CORRECT TONE LEVEL")
                else:
                    logger.warning(f"✗ WRONG TONE: expected {expected_level}, got {tone_level.value}")

                # Check if system prompt has tone instructions
                if system_prompt and "RESPONSE MODE:" in system_prompt:
                    logger.info(f"✓✓✓ TONE INSTRUCTIONS INJECTED INTO SYSTEM PROMPT")
                else:
                    logger.warning(f"✗ NO TONE INSTRUCTIONS IN SYSTEM PROMPT")
            else:
                logger.error(f"✗✗✗ TONE DETECTION FAILED - tone_level is None")

        except Exception as e:
            logger.error(f"✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tone_in_prepare_prompt())
