"""
# main.py

Module Contract
- Purpose: Application entry point. Builds the orchestrator stack, launches GUI or runs small CLI tests,
  and coordinates graceful shutdown work (reflections + summaries/facts).
- Inputs:
  - CLI arg `mode`: "gui" (default), "cli", "test-summaries", "inspect-summaries", "test-prompt-summaries",
    "export-profile", "show-profile", "wizard", "embed-vault", "vault-stats", "clear-vault",
    "upload-doc", "list-docs", "delete-doc", "clear-docs" [NEW: reference docs commands]
  - Environment: GRADIO_* networking flags; config/app_config.py settings (paths, memory, models)
- Outputs:
  - Starts a Gradio app (GUI) or runs test routines; at shutdown triggers memory_system tasks.
  - "export-profile" writes data/user_profile_export.md; "show-profile" prints to console
- Key functions/classes:
  - build_orchestrator() → DaemonOrchestrator fully wired with model_manager, prompt_builder, memory_system
  - test_orchestrator(), test_prompt_with_summaries(), inspect_summaries(): small helpers for ad‑hoc testing
  - __main__ block: selects mode, launches, and on shutdown runs:
      • memory_system.run_shutdown_reflection(...)
      • memory_system.process_shutdown_memory() (summary blocks + facts)
- Important dependencies: core.orchestrator.DaemonOrchestrator, gui.launch.launch_gui, config.app_config,
  memory components, processing.gate_system, memory.user_profile, utils.bootstrap (frozen mode)
- Side effects:
  - Launches local web server; writes to conversation logs and corpus/chroma via orchestrator
  - Writes to data/user_profile.json at shutdown with categorized facts
  - [FROZEN MODE] Bootstrap runs at module level BEFORE other imports to set env vars
- Threading/Async:
  - Uses asyncio to stream model output and to run shutdown tasks deterministically

CRITICAL (Frozen Executable):
  Bootstrap MUST run before config.app_config import. In frozen mode, this sets CORPUS_FILE,
  CHROMA_PATH, etc. to platform-specific user directories (~/.daemon/ on Linux).
"""
import sys
import os
import logging

# Disable ChromaDB telemetry before any chromadb imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Suppress chromadb telemetry errors (known bug with posthog compatibility)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# =============================================================================
# CRITICAL: BOOTSTRAP MUST RUN BEFORE ANY OTHER IMPORTS
# =============================================================================
# In frozen mode, we need to set up environment variables BEFORE config loads.
# This sets CORPUS_FILE, CHROMA_PATH, etc. to point to ~/.daemon/ instead of ./data/
if getattr(sys, 'frozen', False):
    import multiprocessing
    multiprocessing.freeze_support()  # Prevent Windows fork-bomb
    sys.setrecursionlimit(sys.getrecursionlimit() * 2)
    # Set up paths before any config imports
    from utils.bootstrap import setup_environment, IS_FROZEN
    _user_data_dir = setup_environment()
    print(f"[Bootstrap] Frozen mode: user data at {_user_data_dir}")
else:
    IS_FROZEN = False

from dotenv import load_dotenv
if getattr(sys, 'frozen', False):
    # In frozen mode, load .env from user data directory
    env_path = os.path.join(os.environ.get('APPDATA', ''), 'Daemon', '.env')
    load_dotenv(env_path, override=True)
else:
    # override=True ensures .env takes precedence over shell environment variables
    load_dotenv(override=True)
print(f"[DEBUG] OPENAI_API_KEY loaded: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")


import asyncio
import signal
import threading
import time
from datetime import datetime

# DEBUG: Check environment at the very start of main.py
print(f"[DEBUG main.py START] SEM_INDEX_PATH = {os.environ.get('SEM_INDEX_PATH', 'NOT SET')}")
print(f"[DEBUG main.py START] SEM_META_PATH = {os.environ.get('SEM_META_PATH', 'NOT SET')}")

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
from knowledge.WikiManager import WikiManager, _get_embedder
# Preload embedder during startup
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
# RE-ENABLE UNIFIED PROMPT BUILDER NOW THAT MEMORY ISSUES ARE RESOLVED
HAS_UNIFIED_PROMPT = True



logger.info("✅ Unified prompt builder re-enabled - memory system working correctly")

try:
    from core.prompt_builder import UnifiedPromptBuilder as _UnifiedPromptBuilder
    HAS_UNIFIED_PROMPT = True
    logger.info("Found prompt builders, will use enhanced mode")
except Exception as e:  # catch *any* import-time failure, not just ImportError
    HAS_UNIFIED_PROMPT = False
    logger.warning(f"Prompt builders not available: {e}; falling back to raw mode")
# after HAS_UNIFIED_PROMPT is set
class _SimplePromptBuilder:
    def __init__(self, memory_coordinator=None):
        self.memory_coordinator = memory_coordinator

    async def build_prompt(self, user_input: str, include_memories=True, max_memories=10, **kwargs) -> str:
        """
        Simple fallback prompt builder.

        The orchestrator passes many kwargs (system_prompt, search_query, personality_config, etc.)
        but this simple builder ignores them since the orchestrator handles the system prompt separately.
        """
        # Build context-only prompt (system prompt is handled separately by orchestrator)
        context_parts = []

        # Add relevant memories
        if include_memories and self.memory_coordinator:
            try:
                # Use search_query if provided for better retrieval
                query_for_search = kwargs.get('search_query') or user_input

                memories = await self.memory_coordinator.get_memories(query_for_search, limit=max_memories)

                if memories:
                    logger.info(f"[_SimplePromptBuilder] Retrieved {len(memories)} memories for prompt")
                    context_parts.append("RELEVANT MEMORIES:")
                    for i, memory in enumerate(memories[:5]):  # Limit to top 5
                        # Memory field structure varies by source:
                        # - Hybrid retriever uses 'content' field
                        # - Corpus manager uses 'query'/'response' fields
                        # - Timestamps may be in root or metadata

                        content = memory.get('content', '')
                        if not content:
                            # Fallback to query/response format
                            query = memory.get('query', '')
                            response = memory.get('response', '')
                            content = f"Q: {query}\nA: {response}" if query or response else ''

                        # Get timestamp from metadata or root
                        timestamp = memory.get('metadata', {}).get('timestamp', memory.get('timestamp', 'No timestamp'))

                        # Truncate content to reasonable length
                        content_preview = content[:300] if content else '[Empty memory]'

                        context_parts.append(f"{i+1}. [{timestamp}]\n   {content_preview}\n")
                    context_parts.append("")
                else:
                    logger.warning(f"[_SimplePromptBuilder] No memories retrieved for query: {user_input[:50]}")
            except Exception as e:
                logger.error(f"[_SimplePromptBuilder] Failed to retrieve memories: {e}", exc_info=True)

        # Just return the user input - let the orchestrator handle system prompt
        if not context_parts:
            logger.debug(f"[_SimplePromptBuilder] No context parts, returning raw user input")
            return user_input

        # Return memories + current query
        context_parts.append(f"USER: {user_input}")
        return "\n".join(context_parts)
# main.py (updated build_orchestrator function)
def build_orchestrator():
    """Builds and returns a configured orchestrator"""
    # Create model_manager FIRST
    model_manager = ModelManager()
    # Register common OpenRouter model ids explicitly
    model_manager.load_openai_model("gpt-4-turbo", "openai/gpt-4-turbo")
    # Choose active model: prefer persisted config, else default to GPT‑5
    try:
        active_from_config = (config.get("models", {}) or {}).get("active")
    except (AttributeError, TypeError, KeyError):
        active_from_config = None
    target_model = (active_from_config or "gpt-5")
    model_manager.switch_model(target_model)
    logger.info(f"[ModelManager] Active model set to: {model_manager.get_active_model_name()}")
    try:
        from config.app_config import (
            BEST_OF_GENERATOR_MODELS as _BO_GENS,
            BEST_OF_SELECTOR_MODELS as _BO_SEL,
            BEST_OF_DUEL_MODE as _BO_DUEL,
        )
        if _BO_DUEL and isinstance(_BO_GENS, list) and len(_BO_GENS) == 2:
            logger.info(
                f"[DUEL] configured model_1={_BO_GENS[0]} model_2={_BO_GENS[1]} judge={( _BO_SEL[0] if isinstance(_BO_SEL, list) and _BO_SEL else 'N/A')}"
            )
        elif isinstance(_BO_GENS, list) and _BO_GENS:
            logger.info(f"[BESTOF] configured generators={_BO_GENS} selectors={( _BO_SEL if isinstance(_BO_SEL, list) else [])}")
    except (ImportError, AttributeError):
        pass

    # Register shared dependencies so modules (e.g., TopicManager) can resolve them
    try:
        from core.dependencies import deps
        deps.initialize(model_manager)
    except Exception as e:
        logger.debug(f"[build_orchestrator] deps.initialize failed or unavailable: {e}")

    # NOW create instances that depend on model_manager
    time_manager = TimeManager()
    tokenizer_manager = TokenizerManager(model_manager=model_manager)
    # Enable hybrid topic extraction (heuristics + optional LLM fallback)
    topic_manager = TopicManager()  # resolves model_manager via deps if available
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
            time_manager=time_manager,  # Pass shared time_manager instance
    )
        mc = getattr(prompt_builder, "memory_coordinator", None)
        logger.info("[orchestrator] coord wired: type=%s has get_memories=%s get_facts=%s",
        type(mc), hasattr(mc, "get_memories"), hasattr(mc, "get_facts"))

    else:
        logger.warning("Using simple fallback prompt builder")
        prompt_builder = _SimplePromptBuilder(memory_coordinator=coord)

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
        conversation_logger=get_conversation_logger(),
        config=config
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


# Globals for shutdown coordination
_shutdown_requested = False
_orchestrator_ref = None
_last_activity_time = time.time()
_idle_check_interval = int(os.getenv("IDLE_CHECK_INTERVAL_MINUTES", "30"))  # Default 30 minutes
_idle_timeout_minutes = int(os.getenv("IDLE_TIMEOUT_MINUTES", "60"))  # Default 1 hour


def _run_shutdown_tasks(orchestrator):
    """Run reflection and summary tasks - callable from signal handler or idle thread."""
    global _shutdown_requested
    if _shutdown_requested:
        return  # Already shutting down

    _shutdown_requested = True
    logger.info("[Shutdown] Running reflection and summary tasks...")

    # Mark session end for time tracking
    try:
        time_mgr = getattr(orchestrator, "time_manager", None)
        if time_mgr and hasattr(time_mgr, "mark_session_end"):
            time_mgr.mark_session_end()
            logger.info("[Shutdown] Session end time recorded")
    except Exception as e:
        logger.debug(f"[Shutdown] Could not mark session end: {e}")

    try:
        # Gather session data
        session_convos = []
        session_summaries = []

        try:
            logger_obj = getattr(orchestrator, "conversation_logger", None)
            if logger_obj and hasattr(logger_obj, "buffer"):
                session_convos = list(logger_obj.buffer)
        except (AttributeError, TypeError):
            pass

        try:
            pb = getattr(orchestrator, "prompt_builder", None)
            if pb and isinstance(getattr(pb, "_last_summaries", None), list):
                session_summaries = list(pb._last_summaries)
        except (AttributeError, TypeError):
            pass

        # Run shutdown tasks
        async def _do_shutdown():
            # Wait for any pending background storage tasks first
            try:
                from gui.handlers import wait_for_pending_storage
                await wait_for_pending_storage(timeout=10.0)
            except Exception as e:
                logger.warning(f"[Shutdown] wait_for_pending_storage failed: {e}")

            try:
                await orchestrator.memory_system.run_shutdown_reflection(
                    session_conversations=session_convos,
                    session_summaries=session_summaries
                )
                logger.info("[Shutdown] Reflection completed")
            except Exception as e:
                logger.error(f"[Shutdown] Reflection failed: {e}")

            try:
                await orchestrator.memory_system.process_shutdown_memory(
                    session_conversations=session_convos
                )
                logger.info("[Shutdown] Summary/fact processing completed")
            except Exception as e:
                logger.error(f"[Shutdown] Summary/fact processing failed: {e}")

        # Run in new event loop
        asyncio.run(_do_shutdown())

    except Exception as e:
        logger.error(f"[Shutdown] Task execution failed: {e}")


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    logger.info(f"[Signal] Received signal {signum}, triggering shutdown tasks...")
    if _orchestrator_ref:
        _run_shutdown_tasks(_orchestrator_ref)
    sys.exit(0)


def _idle_monitor_thread():
    """Background thread to detect GUI idle state and trigger shutdown tasks."""
    global _last_activity_time, _shutdown_requested

    while not _shutdown_requested:
        time.sleep(_idle_check_interval * 60)  # Check every N minutes

        if _shutdown_requested:
            break

        idle_seconds = time.time() - _last_activity_time
        idle_minutes = idle_seconds / 60

        if idle_minutes >= _idle_timeout_minutes:
            logger.info(f"[Idle Monitor] GUI idle for {idle_minutes:.1f} minutes, running shutdown tasks...")
            if _orchestrator_ref:
                _run_shutdown_tasks(_orchestrator_ref)
                # Reset activity time after shutdown tasks
                _last_activity_time = time.time()


def update_activity_timestamp():
    """Call this whenever user interacts with GUI to reset idle timer."""
    global _last_activity_time
    _last_activity_time = time.time()


if __name__ == "__main__":
    # ==========================================================================
    # NOTE: Bootstrap already ran at module level (for frozen mode)
    # ==========================================================================
    # In frozen mode, bootstrap ran before any imports to set up env vars.
    # Here we just import close_splash for use when showing GUI.
    try:
        from utils.bootstrap import close_splash
    except ImportError:
        close_splash = lambda: None

    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "gui"
        force_wizard = (mode == "wizard" or "--wizard" in sys.argv)

        if mode == "wizard":
            # Force wizard mode for testing
            print("[WIZARD MODE] Forcing wizard to launch regardless of first-run status")
            mode = "gui"  # Continue to GUI launch

        if mode == "cli":
            asyncio.run(test_orchestrator())
        elif mode == "test-summaries":
            from tests.test_summaries import test_summary_integration
            asyncio.run(test_summary_integration())
        elif mode == "inspect-summaries":
            asyncio.run(inspect_summaries())
        elif mode == "test-prompt-summaries":
            asyncio.run(test_prompt_with_summaries())
        elif mode == "export-profile":
            # Export user profile to markdown
            from memory.user_profile import UserProfile
            profile = UserProfile()
            md = profile.export_markdown()
            output_path = "data/user_profile_export.md"
            with open(output_path, 'w') as f:
                f.write(md)
            print(f"✓ Profile exported to {output_path}")
            print(f"  Total facts: {profile.get_fact_count()}")
            sys.exit(0)
        elif mode == "show-profile":
            # Print profile summary to console
            from memory.user_profile import UserProfile
            profile = UserProfile()
            print(f"\n{'='*60}")
            print(f"USER PROFILE ({profile.get_fact_count()} facts)")
            print(f"{'='*60}\n")
            print(profile.get_context_injection(max_tokens=1000))
            print(f"\n{'='*60}")
            sys.exit(0)
        elif mode == "embed-vault":
            # Index Obsidian vault to ChromaDB
            from knowledge.obsidian_manager import ObsidianManager
            from config.app_config import OBSIDIAN_VAULT_PATH

            print(f"\n{'='*60}")
            print("OBSIDIAN VAULT INDEXING")
            print(f"{'='*60}")
            print(f"Vault path: {OBSIDIAN_VAULT_PATH}")

            force = "--force" in sys.argv
            if force:
                print("Force reindex enabled - will re-embed all files")

            manager = ObsidianManager()
            result = manager.embed_vault(force_reindex=force)

            print(f"\nResults:")
            print(f"  Total files found: {result.total_files}")
            print(f"  Files embedded:    {result.embedded_files}")
            print(f"  Total chunks:      {result.total_chunks}")
            print(f"  Files skipped:     {result.skipped_files}")
            print(f"  Errors:            {len(result.errors)}")
            print(f"  Duration:          {result.duration_seconds:.1f}s")

            if result.errors:
                print(f"\nErrors (showing first 5):")
                for err in result.errors[:5]:
                    print(f"  - {err}")
                if len(result.errors) > 5:
                    print(f"  ... and {len(result.errors) - 5} more")

            sys.exit(0)
        elif mode == "vault-stats":
            # Show Obsidian vault indexing statistics
            from knowledge.obsidian_manager import ObsidianManager

            manager = ObsidianManager()
            stats = manager.get_vault_stats()

            print(f"\n{'='*60}")
            print("OBSIDIAN VAULT STATS")
            print(f"{'='*60}")
            print(f"Vault path:      {stats['vault_path']}")
            print(f"Vault exists:    {stats['vault_exists']}")
            print(f"Indexed chunks:  {stats['indexed_chunks']}")
            sys.exit(0)
        elif mode == "clear-vault":
            # Clear all indexed Obsidian notes
            from knowledge.obsidian_manager import ObsidianManager

            manager = ObsidianManager()
            print(f"\n{'='*60}")
            print("CLEARING OBSIDIAN VAULT INDEX")
            print(f"{'='*60}")

            if manager.clear_index():
                print("Index cleared successfully")
            else:
                print("Failed to clear index")
            sys.exit(0)

        # === Reference Documents CLI Commands ===
        elif mode == "upload-doc":
            # Upload a reference document to ChromaDB
            from knowledge.reference_docs_manager import ReferenceDocsManager

            if len(sys.argv) < 3:
                print("Usage: python main.py upload-doc <file_path> [title]")
                print("  file_path: Path to document file (.md, .txt, etc.)")
                print("  title: Optional title (defaults to filename)")
                sys.exit(1)

            file_path = sys.argv[2]
            title = sys.argv[3] if len(sys.argv) > 3 else None

            manager = ReferenceDocsManager()
            print(f"\n{'='*60}")
            print(f"UPLOADING REFERENCE DOCUMENT")
            print(f"{'='*60}")
            print(f"File: {file_path}")
            if title:
                print(f"Title: {title}")

            result = manager.upload_document(file_path, title)

            print(f"\n{'='*60}")
            if result.success:
                print(f"SUCCESS: '{result.title}'")
                print(f"  Chunks: {result.total_chunks}")
                print(f"  Type: {result.file_type}")
                print(f"  Duration: {result.duration_seconds:.1f}s")
            else:
                print(f"FAILED: {result.errors}")
            print(f"{'='*60}")
            sys.exit(0 if result.success else 1)

        elif mode == "list-docs":
            # List all uploaded reference documents
            from knowledge.reference_docs_manager import ReferenceDocsManager

            manager = ReferenceDocsManager()
            stats = manager.get_stats()

            print(f"\n{'='*60}")
            print("REFERENCE DOCUMENTS")
            print(f"{'='*60}")
            print(f"Total documents: {stats['document_count']}")
            print(f"Total chunks: {stats['total_chunks']}")

            if stats['documents']:
                print(f"\nDocuments:")
                for doc in stats['documents']:
                    print(f"  - {doc['title']} ({doc['file_type']}, {doc['chunk_count']} chunks)")
            else:
                print("\nNo documents uploaded yet.")
                print("Use: python main.py upload-doc <file_path>")
            print(f"{'='*60}")
            sys.exit(0)

        elif mode == "delete-doc":
            # Delete a reference document
            from knowledge.reference_docs_manager import ReferenceDocsManager

            if len(sys.argv) < 3:
                print("Usage: python main.py delete-doc <title>")
                print("  title: Title of document to delete (use list-docs to see titles)")
                sys.exit(1)

            title = sys.argv[2]
            manager = ReferenceDocsManager()

            print(f"\n{'='*60}")
            print(f"DELETING REFERENCE DOCUMENT: {title}")
            print(f"{'='*60}")

            if manager.delete_document(title):
                print("Document deleted successfully")
            else:
                print("Failed to delete document (may not exist)")
            sys.exit(0)

        elif mode == "clear-docs":
            # Clear all reference documents
            from knowledge.reference_docs_manager import ReferenceDocsManager

            manager = ReferenceDocsManager()
            print(f"\n{'='*60}")
            print("CLEARING ALL REFERENCE DOCUMENTS")
            print(f"{'='*60}")

            if manager.clear_all():
                print("All documents cleared successfully")
            else:
                print("Failed to clear documents")
            sys.exit(0)

        elif mode == "daily-note":
            # Generate daily note for today or specified date
            # Usage: python main.py daily-note [date] [--force]
            import asyncio
            from utils.daily_notes_generator import DailyNotesGenerator
            from datetime import date as date_type, datetime, timedelta

            # Parse arguments
            target_date = date_type.today()
            force = "--force" in sys.argv

            # Check for date argument (YYYY-MM-DD format)
            for arg in sys.argv[2:]:
                if arg.startswith("--"):
                    continue
                if arg == "yesterday":
                    target_date = date_type.today() - timedelta(days=1)
                else:
                    try:
                        target_date = datetime.strptime(arg, "%Y-%m-%d").date()
                    except ValueError:
                        print(f"Invalid date format: {arg} (use YYYY-MM-DD or 'yesterday')")
                        sys.exit(1)

            print(f"\n{'='*60}")
            print("GENERATING DAILY NOTE")
            print(f"{'='*60}")
            print(f"Date: {target_date}")
            print(f"Force: {force}")
            print()

            generator = DailyNotesGenerator()
            result = asyncio.run(generator.generate_for_date(target_date, force=force))

            if result.success:
                print(f"{'='*60}")
                print("SUCCESS")
                print(f"{'='*60}")
                print(f"  Output: {result.output_path}")
                print(f"  Conversations: {result.conversation_count}")
                print(f"  Duration: {result.duration_hours:.1f} hours")
                print(f"  Intensity: {result.intensity}/10")
            elif result.skipped_reason:
                print(f"SKIPPED: {result.skipped_reason}")
                if result.output_path:
                    print(f"  Existing note: {result.output_path}")
            else:
                print(f"FAILED: {result.error}")
            sys.exit(0)

        elif mode == "daily-note-catchup":
            # Generate yesterday's note if missing (for startup hooks)
            import asyncio
            from utils.daily_notes_generator import DailyNotesGenerator
            from models.model_manager import ModelManager

            print(f"\n{'='*60}")
            print("DAILY NOTE CATCH-UP")
            print(f"{'='*60}")

            # Create ModelManager with API key to ensure LLM is available
            model_manager = ModelManager()
            generator = DailyNotesGenerator(model_manager=model_manager)
            result = asyncio.run(generator.generate_yesterday_if_missing())

            if result is None:
                print("Yesterday's note already exists, nothing to do.")
            elif result.success:
                print(f"Generated yesterday's note:")
                print(f"  Output: {result.output_path}")
                print(f"  Conversations: {result.conversation_count}")
            elif result.skipped_reason:
                print(f"Skipped: {result.skipped_reason}")
            else:
                print(f"Failed: {result.error}")
            sys.exit(0)

        elif mode == "weekly-note":
            # Generate weekly summary for current or specified week
            # Usage: python main.py weekly-note [date] [--force]
            import asyncio
            from utils.weekly_notes_generator import WeeklyNotesGenerator
            from datetime import date as date_type, datetime, timedelta

            # Parse arguments
            target_date = date_type.today()
            force = "--force" in sys.argv

            # Check for date argument (YYYY-MM-DD format or 'last-week')
            for arg in sys.argv[2:]:
                if arg.startswith("--"):
                    continue
                if arg == "last-week":
                    target_date = date_type.today() - timedelta(days=7)
                else:
                    try:
                        target_date = datetime.strptime(arg, "%Y-%m-%d").date()
                    except ValueError:
                        print(f"Invalid date format: {arg} (use YYYY-MM-DD or 'last-week')")
                        sys.exit(1)

            print(f"\n{'='*60}")
            print("GENERATING WEEKLY NOTE")
            print(f"{'='*60}")
            print(f"Week containing: {target_date}")
            print(f"Force: {force}")
            print()

            generator = WeeklyNotesGenerator()
            result = asyncio.run(generator.generate_for_week(target_date, force=force))

            if result.success:
                print(f"{'='*60}")
                print("SUCCESS")
                print(f"{'='*60}")
                print(f"  Week: {result.week_num}, {result.year}")
                print(f"  Folder: {result.week_folder}")
                print(f"  Output: {result.output_path}")
                print(f"  Daily notes found: {result.daily_notes_found}")
                print(f"  Daily notes moved: {result.daily_notes_moved}")
                print(f"  Total conversations: {result.total_conversations}")
                print(f"  Average intensity: {result.avg_intensity}/10")
            elif result.skipped_reason:
                print(f"SKIPPED: {result.skipped_reason}")
                if result.output_path:
                    print(f"  Existing summary: {result.output_path}")
            else:
                print(f"FAILED: {result.error}")
            sys.exit(0)

        elif mode == "weekly-note-catchup":
            # Generate last week's summary if complete and missing
            import asyncio
            from utils.weekly_notes_generator import WeeklyNotesGenerator

            print(f"\n{'='*60}")
            print("WEEKLY NOTE CATCH-UP")
            print(f"{'='*60}")

            generator = WeeklyNotesGenerator()
            result = asyncio.run(generator.generate_last_week_if_complete())

            if result is None:
                print("Last week's summary already exists, nothing to do.")
            elif result.success:
                print(f"Generated last week's summary:")
                print(f"  Week: {result.week_num}, {result.year}")
                print(f"  Output: {result.output_path}")
                print(f"  Daily notes: {result.daily_notes_found}")
            elif result.skipped_reason:
                print(f"Skipped: {result.skipped_reason}")
            else:
                print(f"Failed: {result.error}")
            sys.exit(0)

        elif mode == "organize-weekly":
            # Organize daily notes into weekly folders without generating summary
            import asyncio
            from utils.weekly_notes_generator import WeeklyNotesGenerator
            from datetime import date as date_type, datetime, timedelta

            # Parse arguments
            target_date = date_type.today()

            for arg in sys.argv[2:]:
                if arg == "last-week":
                    target_date = date_type.today() - timedelta(days=7)
                else:
                    try:
                        target_date = datetime.strptime(arg, "%Y-%m-%d").date()
                    except ValueError:
                        print(f"Invalid date format: {arg} (use YYYY-MM-DD or 'last-week')")
                        sys.exit(1)

            print(f"\n{'='*60}")
            print("ORGANIZING WEEKLY FOLDER")
            print(f"{'='*60}")
            print(f"Week containing: {target_date}")
            print()

            generator = WeeklyNotesGenerator()
            folder, moved = asyncio.run(generator.organize_week(target_date))

            print(f"Folder: {folder}")
            print(f"Notes moved: {moved}")
            sys.exit(0)

        elif mode == "refresh-narrative":
            # Manually regenerate the narrative context from Obsidian notes + corpus summaries
            # Usage: python main.py refresh-narrative
            import asyncio
            from memory.corpus_manager import CorpusManager
            from memory.memory_consolidator import MemoryConsolidator
            from models.model_manager import ModelManager

            async def refresh_narrative_context():
                """Manually regenerate the narrative context from Obsidian notes + corpus."""
                print(f"\n{'='*60}")
                print("NARRATIVE CONTEXT REFRESH (Hybrid)")
                print(f"{'='*60}\n")

                # Initialize components
                corpus = CorpusManager()
                model_manager = ModelManager()
                consolidator = MemoryConsolidator(model_manager)

                # Check Obsidian sources
                print("Checking Obsidian vault...")
                obsidian_weeklies = consolidator._read_obsidian_weekly_summaries(limit=2)
                obsidian_dailies = consolidator._read_obsidian_daily_notes(limit=7)
                print(f"  - Obsidian weekly summaries: {len(obsidian_weeklies)}")
                print(f"  - Obsidian daily notes: {len(obsidian_dailies)}")

                # Get corpus summaries as fallback
                print("\nChecking corpus summaries (fallback)...")
                from datetime import datetime, timedelta
                all_summaries = corpus.get_summaries(count=50)

                now = datetime.now()
                weekly_cutoff = now - timedelta(weeks=4)
                monthly_cutoff = now - timedelta(days=60)

                corpus_weeklies = []
                corpus_monthlies = []

                for s in (all_summaries or []):
                    ts = s.get("timestamp")
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except:
                            continue
                    if isinstance(ts, datetime):
                        if ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                        if ts >= weekly_cutoff:
                            corpus_weeklies.append(s)
                        elif ts >= monthly_cutoff:
                            corpus_monthlies.append(s)

                # Limit counts
                corpus_weeklies = sorted(corpus_weeklies, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:4]
                corpus_monthlies = sorted(corpus_monthlies, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:2]
                print(f"  - Corpus weekly-range summaries: {len(corpus_weeklies)}")
                print(f"  - Corpus monthly-range summaries: {len(corpus_monthlies)}")

                # Check if we have any content
                if not obsidian_weeklies and not obsidian_dailies and not corpus_weeklies and not corpus_monthlies:
                    print("\n✗ No content available for synthesis.")
                    print("  Generate daily notes first: python main.py daily-note")
                    return False

                print("\nSynthesizing narrative context...")

                # The method reads Obsidian notes automatically, pass corpus as fallback
                narrative = await consolidator.generate_narrative_context(
                    recent_weeklies=corpus_weeklies,
                    recent_monthlies=corpus_monthlies
                )

                if narrative:
                    corpus.save_narrative_context(narrative)
                    print(f"\n✓ Narrative context updated successfully ({len(narrative)} chars)")
                    print(f"\n{'-'*50}")
                    print("Generated narrative:")
                    print(f"{'-'*50}")
                    print(narrative)
                    return True
                else:
                    print("\n✗ Failed to generate narrative context")
                    return False

            success = asyncio.run(refresh_narrative_context())
            sys.exit(0 if success else 1)

        else:
            print(f"[DEBUG] Building orchestrator (mode={mode}, force_wizard={force_wizard})...")
            orchestrator = build_orchestrator()
            print("[DEBUG] Orchestrator built successfully")

            # Store orchestrator reference for signal handlers and idle monitor
            _orchestrator_ref = orchestrator

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
            logger.info("[Startup] Registered signal handlers for SIGTERM and SIGINT")

            # Start idle monitoring thread
            idle_thread = threading.Thread(target=_idle_monitor_thread, daemon=True, name="IdleMonitor")
            idle_thread.start()
            logger.info(f"[Startup] Started idle monitor (check every {_idle_check_interval}m, timeout {_idle_timeout_minutes}m)")

            # Close splash screen before showing GUI
            close_splash()

            print(f"[DEBUG] About to call launch_gui(orchestrator, force_wizard={force_wizard})")
            launch_gui(orchestrator, force_wizard=force_wizard)
            print("[DEBUG] launch_gui() returned")

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
                except (AttributeError, TypeError):
                    pass

                # Pull any summaries collected in this run if you keep them
                try:
                    pb = getattr(orchestrator, "prompt_builder", None)
                    if pb and isinstance(getattr(pb, "_last_summaries", None), list):
                        session_summaries = list(pb._last_summaries)
                except (AttributeError, TypeError):
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
                        await orchestrator.memory_system.process_shutdown_memory(session_conversations=session_convos)
                    except Exception as e:
                        logger.error(f"Shutdown summary/fact processing failed: {e}")

                scheduled_shutdown_tasks = False
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
                        scheduled_shutdown_tasks = True
                    else:
                        _a.run(_do_shutdown_reflection())
                        _a.run(_do_shutdown_summaries_and_facts())
                except RuntimeError:
                    pass  # Event loop issues at shutdown
        finally:
            # close model manager cleanly (don’t instantiate a new one)
            try:
                # If tasks were scheduled onto a still-running loop, skip immediate close
                # to avoid tearing down clients while tasks finish (e.g., httpx/uvloop EOF noise).
                if orchestrator and hasattr(orchestrator, "model_manager"):
                    if 'scheduled_shutdown_tasks' in locals() and scheduled_shutdown_tasks:
                        logger.debug("[main] Skipping immediate model_manager.close(); shutdown tasks still scheduled")
                    else:
                        orchestrator.model_manager.close()
            except (AttributeError, RuntimeError):
                pass  # Model manager cleanup errors at shutdown
