"""
# main.py

Module Contract
- Purpose: Application entry point. Builds the orchestrator stack, launches GUI or runs small CLI tests, and coordinates graceful shutdown work (reflections + summaries/facts).
- Inputs:
  - CLI arg `mode`: "gui" (default), "cli", "test-summaries", "inspect-summaries", "test-prompt-summaries"
  - Environment: GRADIO_* networking flags; config/app_config.py settings (paths, memory, models)
- Outputs:
  - Starts a Gradio app (GUI) or runs test routines; at shutdown triggers memory_system tasks.
- Key functions/classes:
  - build_orchestrator() → DaemonOrchestrator fully wired with model_manager, prompt_builder, memory_system, etc.
  - test_orchestrator(), test_prompt_with_summaries(), inspect_summaries(): small helpers for ad‑hoc testing.
  - __main__ block: selects mode, launches, and on shutdown runs:
      • memory_system.run_shutdown_reflection(...)
      • memory_system.process_shutdown_memory() (summary blocks + facts)
- Important dependencies: core.orchestrator.DaemonOrchestrator, gui.launch.launch_gui, config.app_config constants, memory components, processing.gate_system.
- Side effects:
  - Launches local web server; writes to conversation logs and corpus/chroma via orchestrator.
- Threading/Async:
  - Uses asyncio to stream model output and to run shutdown tasks deterministically.
"""
import asyncio
import sys
import os
from datetime import datetime

from utils.logging_utils import get_logger, configure_logging
from utils.time_manager import TimeManager
# Setup logging early to avoid duplicate handlers
configure_logging()
logger = get_logger("main")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our components
from core.response_generator import ResponseGenerator
from core.orchestrator import DaemonOrchestrator
from utils.file_processor import FileProcessor
from utils.conversation_logger import get_conversation_logger

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from processing.gate_system import MultiStageGateSystem
from gui.launch import launch_gui
from utils.topic_manager import TopicManager
from knowledge.WikiManager import WikiManager
# during startup
from knowledge.WikiManager import WikiManager, _get_embedder
_ = _get_embedder("all-MiniLM-L6-v2")

from processing.gate_system import set_topic_resolver
from models.tokenizer_manager import TokenizerManager
from config.app_config import config, CHROMA_PATH, CORPUS_FILE
orchestrator = None  # module-scope
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


# main.py
try:
    from core.prompt_builder import UnifiedPromptBuilder as _UnifiedPromptBuilder
    HAS_UNIFIED_PROMPT = True
    logger.info("Found prompt builders, will use enhanced mode")
except Exception as e:  # catch *any* import-time failure, not just ImportError
    HAS_UNIFIED_PROMPT = False
    logger.warning(f"Prompt builders not available: {e}; falling back to raw mode")
# after HAS_UNIFIED_PROMPT is set
class _SimplePromptBuilder:
    async def build_prompt(self, user_input: str, **kwargs) -> str:
        # ultra-simple fallback: just return user_input as the "prompt"
        return user_input
# main.py (updated build_orchestrator function)
def build_orchestrator():
    """Builds and returns a configured orchestrator"""
    # Create model_manager FIRST
    model_manager = ModelManager()
    model_manager.load_openai_model("gpt-4-turbo", "gpt-4-turbo")
    model_manager.switch_model("claude-opus")
    logger.info(f"[ModelManager] Active model set to: {model_manager.get_active_model_name()}")

    # NOW create instances that depend on model_manager
    time_manager = TimeManager()
    tokenizer_manager = TokenizerManager(model_manager=model_manager)
    topic_manager = TopicManager()
    # IMPORTANT: get_primary_topic must return str | None (not (str, …))
    set_topic_resolver(topic_manager.get_primary_topic)
    wiki_manager = WikiManager()

    gate_system = MultiStageGateSystem(model_manager)
    response_generator = ResponseGenerator(model_manager, time_manager)
    file_processor = FileProcessor()

    chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    # Build shared corpus manager once
    corpus_manager = CorpusManager(corpus_file=CORPUS_FILE)

    # Single coordinator instance (use the SAME corpus manager)
    memory_coordinator = MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        gate_system=gate_system,
        topic_manager=topic_manager,   # will be honored if __init__ uses `or TopicManager()`
        model_manager=model_manager,
        time_manager=time_manager
    )
    # Keep a local reference; we'll pass it where needed
    coord = memory_coordinator

    if HAS_UNIFIED_PROMPT:
        prompt_builder = _UnifiedPromptBuilder(
            model_manager=model_manager,
            memory_coordinator=coord,  # <- same instance
            tokenizer_manager=tokenizer_manager,
            wiki_manager=wiki_manager,
            topic_manager=topic_manager,
            gate_system=gate_system,
    )
        mc = getattr(prompt_builder, "memory_coordinator", None)
        logger.info("[orchestrator] coord wired: type=%s has get_memories=%s get_facts=%s",
        type(mc), hasattr(mc, "get_memories"), hasattr(mc, "get_facts"))

    else:
        logger.warning("Using simple fallback prompt builder")
        prompt_builder = _SimplePromptBuilder()

    from personality.personality_manager import PersonalityManager
    return DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder,
        personality_manager=PersonalityManager(),
        memory_system=coord,
        topic_manager=topic_manager,
        wiki_manager=wiki_manager,
        tokenizer_manager=tokenizer_manager,
        conversation_logger=get_conversation_logger()
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

    if len(corpus_manager.get_summaries()) < 4:
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

        try:
            # Gather session buffers, if your logger tracks them
            session_convos = []
            session_summaries = []
            if orchestrator:
                # If your conversation logger exposes a buffer of [{'query','response'}, ...]
                try:
                    logger_obj = getattr(orchestrator, "conversation_logger", None)
                    if logger_obj and hasattr(logger_obj, "buffer"):
                        session_convos = list(logger_obj.buffer)
                except Exception:
                    pass

                # Pull any summaries collected in this run if you keep them
                try:
                    pb = getattr(orchestrator, "prompt_builder", None)
                    if pb and isinstance(getattr(pb, "_last_summaries", None), list):
                        session_summaries = list(pb._last_summaries)
                except Exception:
                    pass

                # Ensure we can call async at shutdown
                async def _do_shutdown_reflection():
                    try:
                        await orchestrator.memory_system.run_shutdown_reflection(
                            session_conversations=session_convos,
                            session_summaries=session_summaries
                        )
                    except Exception as e:
                        logger.error(f"Shutdown reflection failed: {e}")

                async def _do_shutdown_summaries_and_facts():
                    try:
                        await orchestrator.memory_system.process_shutdown_memory()
                    except Exception as e:
                        logger.error(f"Shutdown summary/fact processing failed: {e}")

                try:
                    import asyncio as _a
                    loop = None
                    try:
                        loop = _a.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        # schedule and give it a beat
                        loop.create_task(_do_shutdown_reflection())
                        loop.create_task(_do_shutdown_summaries_and_facts())
                    else:
                        _a.run(_do_shutdown_reflection())
                        _a.run(_do_shutdown_summaries_and_facts())
                except Exception:
                    pass
        finally:
            # close model manager cleanly (don’t instantiate a new one)
            try:
                if orchestrator and hasattr(orchestrator, "model_manager"):
                    orchestrator.model_manager.close()
            except Exception:
                pass
