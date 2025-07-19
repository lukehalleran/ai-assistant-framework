# daemon_7_11_25_refactor/main.py
import asyncio
import sys
import os
from utils.logging_utils import get_logger
from utils.time_manager import TimeManager
# Setup logging
logger = get_logger("main")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our components
from core.response_generator import ResponseGenerator
from core.orchestrator import DaemonOrchestrator
from utils.file_processor import FileProcessor
from memory.storage.chroma_store import MultiCollectionChromaStore
from gui.launch import launch_gui
from knowledge.topic_manager import TopicManager
from knowledge.WikiManager import WikiManager
from models.tokenizer_manager import TokenizerManager
# Import model manager
try:
    from models.model_manager import ModelManager
except ImportError:
    logger.warning("ModelManager not found, using mock")
    # Create a mock ModelManager for testing
    class ModelManager:
        def __init__(self):
            self.model_name = "gpt-4-turbo"

        def load_openai_model(self, name, alias):
            logger.info(f"Mock: Loading {name} as {alias}")

        def switch_model(self, name):
            logger.info(f"Mock: Switching to {name}")
            self.model_name = name

        def get_active_model_name(self):
            return self.model_name

        async def generate_async(self, prompt):
            # Mock response that simulates streaming
            async def mock_stream():
                words = "This is a mock response. The calculation 2+2 equals 4.".split()
                for word in words:
                    yield type('obj', (object,), {
                        'choices': [type('obj', (object,), {
                            'delta': type('obj', (object,), {'content': word + ' '})()
                        })()]
                    })()
            return mock_stream()

        def close(self):
            logger.info("Mock: Closing model manager")

# Import the prompt builder components
prompt_builder_instance = None
#Instantate Time Manager
time_manager = TimeManager()
topic_manager=TopicManager()
WikiManager = WikiManager()
tokenizer_manager=TokenizerManager()
try:
    from core.prompt import PromptBuilder
    from core.prompt_builder import UnifiedHierarchicalPromptBuilder
    from processing.gate_system import GatedPromptBuilder
    logger.info("Found prompt builders, will use enhanced mode")
except ImportError as e:
    logger.warning(f"Prompt builders not found: {e}, will use raw mode")
def build_orchestrator():
    """Builds and returns a configured orchestrator"""
    model_manager = ModelManager()
    model_manager.load_openai_model("gpt-4-turbo", "gpt-4-turbo")
    model_manager.switch_model("gpt-4-turbo")

    response_generator = ResponseGenerator(model_manager)
    file_processor = FileProcessor()

    prompt_builder_instance = None
    if 'UnifiedHierarchicalPromptBuilder' in globals() and 'GatedPromptBuilder' in globals():
        try:
            base_pb = PromptBuilder(
                tokenizer_manager=tokenizer_manager,
                model_name="gpt-4-turbo",
                wiki=WikiManager,
                model_manager=model_manager,
                topic_manager=topic_manager
            )
            gated_pb = GatedPromptBuilder(prompt_builder=base_pb, model_manager=model_manager)
            chroma_store = MultiCollectionChromaStore(persist_directory="daemon_memory")

            prompt_builder_instance = UnifiedHierarchicalPromptBuilder(
                prompt_builder=gated_pb,
                model_manager=model_manager,
                chroma_store=chroma_store
            )
        except Exception as e:
            logger.warning(f"Could not initialize prompt builder: {e}")
            import traceback
            traceback.print_exc()

    from personality.personality_manager import PersonalityManager
    return DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder_instance,
        personality_manager=PersonalityManager()
    )

async def test_orchestrator():
    orchestrator = build_orchestrator()
    print("\n" + "="*50)
    print("Testing Orchestrator")
    print("="*50)

    # Test 1: Simple query
    response, debug_info = await orchestrator.process_user_query("What's 2+2? Give a brief answer.")
    print(f"\nResponse: {response}")
    for k, v in debug_info.items():
        if k != 'prompt': print(f"  {k}: {v}")



if __name__ == "__main__":
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "gui"
        if mode == "cli":
            asyncio.run(test_orchestrator())
        else:
            orchestrator = build_orchestrator()
            launch_gui(orchestrator)


    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        if hasattr(ModelManager, 'close'):
            ModelManager.close()
