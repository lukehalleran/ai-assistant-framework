# daemon_7_11_25_refactor/main.py
import asyncio
import sys
import os
from utils.logging_utils import get_logger
from utils.time_manager import TimeManager
from datetime import datetime# Setup logging
logger = get_logger("main")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our components
from core.response_generator import ResponseGenerator
from core.orchestrator import DaemonOrchestrator
from utils.file_processor import FileProcessor
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.memory_interface import HierarchicalMemorySystem
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from processing.gate_system import MultiStageGateSystem
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

#Instantate Time Manager
time_manager = TimeManager()
topic_manager=TopicManager()
wiki_manager = WikiManager()
tokenizer_manager=TokenizerManager()
try:
    from core.prompt_builder_v2 import UnifiedPromptBuilder
    from processing.gate_system import GatedPromptBuilder
    logger.info("Found prompt builders, will use enhanced mode")
except ImportError as e:
    logger.warning(f"Prompt builders not found: {e}, will use raw mode")
def build_orchestrator():
    """Builds and returns a configured orchestrator"""
    model_manager = ModelManager()
    model_manager.load_openai_model("gpt-4-turbo", "gpt-4-turbo")
    model_manager.switch_model("gpt-4-turbo")
    logger.info(f"[ModelManager] Active model set to: {model_manager.get_active_model_name()}")
    response_generator = ResponseGenerator(model_manager)
    file_processor = FileProcessor()

    # Build the prompt builder directly (UnifiedPromptBuilder is now standalone)
    chroma_store = MultiCollectionChromaStore(persist_directory="daemon_memory")
    # ✅ Explicitly grab the semantic collection
    semantic_collection = chroma_store.collections.get("semantic")

    # ✅ Initialize hierarchical memory with correct collection
    hierarchical_memory = HierarchicalMemorySystem(
        model_manager=model_manager,
        chroma_store=chroma_store,
    )

    # ✅ MemoryCoordinator with working memory system
    memory_system = MemoryCoordinator(
        corpus_manager=CorpusManager(),
        chroma_store=chroma_store,
        hierarchical_memory=hierarchical_memory,
        gate_system=MultiStageGateSystem(model_manager)
    )

    logger.info(f"[Memory Boot] Chroma collection set: {semantic_collection is not None}")
    logger.info(f"[Memory Boot] Loaded hierarchical memory count: {len(hierarchical_memory.memories)}")

    prompt_builder = UnifiedPromptBuilder(
        model_manager=model_manager,
        memory_coordinator=memory_system,
        tokenizer_manager=TokenizerManager(),
        wiki_manager=wiki_manager,
        topic_manager=TopicManager(),
        gate_system=MultiStageGateSystem(model_manager)
    )

    from personality.personality_manager import PersonalityManager
    return DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder,
        personality_manager=PersonalityManager(),
        memory_system=memory_system
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



# Optional test helpers
async def test_prompt_with_summaries():
    orchestrator = build_orchestrator()
    corpus_manager = orchestrator.memory_system.corpus_manager

    if len(corpus_manager.get_summaries()) < 2:
        print("Creating summaries...")
        corpus_manager.create_summary_now(5)
        corpus_manager.create_summary_now(10)

    prompt = await orchestrator.prompt_builder.build_prompt(
        user_input="Summarize our previous conversations.",
        include_dreams=False,
        include_wiki=False,
        include_semantic=False,
        include_summaries=True
    )

    print("\n" + "=" * 60)
    print("PROMPT WITH SUMMARIES")
    print("=" * 60)
    if "[CONVERSATION SUMMARIES]" in prompt:
        start = prompt.find("[CONVERSATION SUMMARIES]")
        end = prompt.find("\n[", start + 1)
        print(prompt[start:end if end != -1 else None])
    else:
        print("❌ No summary section found in prompt!")
    print("\nTotal prompt length:", len(prompt))


async def inspect_summaries():
    orchestrator = build_orchestrator()
    memory_system = orchestrator.memory_system
    corpus_manager = orchestrator.memory_system.corpus_manager
    summaries = memory_system.corpus_manager.get_summaries(limit=10)

    print("\n" + "=" * 60)
    print("SUMMARY INSPECTION")
    print("=" * 60)

    if not summaries:
        print("No summaries found.")
        return

    for i, s in enumerate(summaries):
        timestamp = s.get("timestamp", "Unknown")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Summary {i+1}: [{timestamp}] {s.get('tags', [])}")
        print(f"  {s.get('response', '')[:200]}...\n{'-' * 40}")


if __name__ == "__main__":
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "gui"

        if mode == "cli":
            asyncio.run(test_orchestrator())
        elif mode == "test-summaries":
            from tests.test_summaries import test_summary_integration
            asyncio.run(test_summary_integration())
        elif mode == "inspect-summaries":
            asyncio.run(inspect_summaries())
        elif mode == "test-prompt-summaries":
            asyncio.run(test_prompt_with_summaries())
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
            ModelManager().close()
