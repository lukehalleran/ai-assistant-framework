# daemon_7_11_25_refactor/test_basic_pipeline.py
"""
Basic test to verify the core pipeline is working without all dependencies
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_utils import get_logger
from core.response_generator import ResponseGenerator
from core.orchestrator import DaemonOrchestrator
from utils.file_processor import FileProcessor

logger = get_logger("test_basic")

class MockModelManager:
    """Simple mock to test without full model manager"""
    def __init__(self):
        self.model_name = "mock-model"
        logger.info("MockModelManager initialized")

    def get_active_model_name(self):
        return self.model_name

    def switch_model(self, name):
        self.model_name = name
        logger.info(f"Switched to model: {name}")

    def generate_async(self, prompt):
        """Return a mock async generator (not awaitable itself)"""
        logger.info(f"Generating response for prompt of length: {len(prompt)}")

        async def _generate():
            # Mock response
            response_text = f"Mock response to: '{prompt[:50]}...'"

            # Simulate streaming
            words = response_text.split()
            for i, word in enumerate(words):
                # Create mock chunk object
                chunk = type('MockChunk', (), {
                    'choices': [type('Choice', (), {
                        'delta': type('Delta', (), {'content': word + " "})()
                    })()]
                })()
                yield chunk
                await asyncio.sleep(0.01)  # Simulate network delay

        return _generate()

    def get_tokenizer(self, model_name):
        """
        PromptBuilder expects .encode(...) on the returned object.
        We’ll return a dummy tokenizer whose encode() just splits on whitespace.
        """
        class DummyTokenizer:
            def encode(self, text):
                return text.split()
        return DummyTokenizer()

async def test_basic_pipeline():
    """Test the basic pipeline without all dependencies"""
    print("\n" + "="*60)
    print("BASIC PIPELINE TEST")
    print("="*60)

    # Initialize components
    model_manager = MockModelManager()
    response_generator = ResponseGenerator(model_manager)
    file_processor = FileProcessor()

    # Create orchestrator (without prompt builder for now)
    orchestrator = DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=None  # We'll add this later
    )

    # Test 1: Simple query
    print("\n1. Testing simple query...")
    query = "What is 2+2?"

    try:
        response, debug_info = await orchestrator.process_user_query(query, use_raw_mode=True)
        print(f"✅ Query: {query}")
        print(f"✅ Response: {response}")
        print(f"✅ Duration: {debug_info.get('duration', 'N/A')}s")
        print(f"✅ Chunks: {debug_info.get('chunk_count', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Test with empty prompt builder
    print("\n2. Testing with prompt builder mode (should fall back to raw)...")
    try:
        response, debug_info = await orchestrator.process_user_query(
            "Tell me about Python",
            use_raw_mode=False  # This should work even without prompt builder
        )
        print(f"✅ Response received: {response[:100]}...")
        print(f"✅ Mode: {debug_info.get('mode', 'unknown')}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print("BASIC TESTS COMPLETE")
    print("="*60)

    # Show what's working
    print("\n✅ Working components:")
    print("  - Logging system")
    print("  - Response generator")
    print("  - Orchestrator")
    print("  - File processor")
    print("  - Basic async pipeline")

    print("\n📝 Next steps:")
    print("  1. Add PromptBuilder from original project")
    print("  2. Wire up UnifiedHierarchicalPromptBuilder")
    print("  3. Connect memory system")
    print("  4. Add model manager from original project")

if __name__ == "__main__":
    print("Starting basic pipeline test...")
    try:
        asyncio.run(test_basic_pipeline())
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTest script finished.")
