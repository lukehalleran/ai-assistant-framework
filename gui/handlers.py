"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/enhanced/agentic flows, streams the response to the UI, and persists interaction + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?, is_progress?} as Gradio updates.
- Behavior:
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - AGENTIC: If agentic_search.enabled and query triggers search, computation, OR memory recall, route through AgenticSearchController
    - 3-tier agentic gate [ENHANCED 2026-03-15]:
      - Tier 1: Keyword heuristic (instant) — computation keywords OR memory keywords ("do you remember", "my notes", etc.)
      - Tier 2: Entity match (instant) — query mentions a known entity from the knowledge graph (e.g., "Flapjack", "Auggie")
      - Tier 3: LLM fallback — piggybacks on web search trigger LLM call; also returns needs_memory_search field
    - Casual skip filter only applies when no keyword/entity trigger fired
    - skip_initial_search=True for computation and memory queries (skips Round 1 web search)
  - ENHANCED: orchestrator.prepare_prompt → extract note_images → response_generator.generate_streaming_response(images=...) → store interaction
  - IMAGE SUPPORT [NEW 2026-01-30]: Extracts note_images from raw_context and passes to streaming for multimodal models
- Side effects:
  - Writes to conversation logger; stores to memory_system; updates debug_state for Debug Trace tab.
"""
import asyncio
import logging
from core.response_parser import ResponseParser
from utils.logging_utils import log_and_time
from utils.conversation_logger import get_conversation_logger
from utils.file_processor import FileProcessor, ProcessedFilesResult
import json
from config.app_config import load_system_prompt, ENABLE_BEST_OF, BEST_OF_N, BEST_OF_TEMPS, BEST_OF_MAX_TOKENS, BEST_OF_MIN_QUESTION, BEST_OF_MODEL, BEST_OF_MIN_TOKENS
from utils.query_checker import analyze_query
DEFAULT_SYSTEM_PROMPT = load_system_prompt()
logger = logging.getLogger("gradio_gui")

# Initialize FileProcessor for secure file handling
file_processor = FileProcessor()

# Track pending background storage tasks (for graceful shutdown)
_pending_storage_tasks: set = set()


async def _background_store_interaction(
    orchestrator,
    merged_input: str,
    response_to_store: str,
    tags: list,
    user_text: str,
    final_output: str,
    personality: str,
    file_names: list,
    conversation_logger
):
    """
    Store interaction in background to avoid blocking response delivery.

    This runs after the response is fully streamed to the user, so ~1.7s of
    LLM calls (topic extraction, etc.) don't add to perceived latency.
    """
    try:
        memory_id = await orchestrator.memory_system.store_interaction(
            query=merged_input,
            response=response_to_store,
            tags=tags
        )
        logger.info(f"[HANDLE_SUBMIT] Background storage complete, ID: {memory_id}")

        # Log conversation with db_id
        conversation_logger.log_interaction(
            user_input=user_text,
            assistant_response=final_output,
            metadata={
                'mode': 'enhanced',
                'files': file_names if file_names else None,
                'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                'topic': getattr(orchestrator, 'current_topic', None),
                'db_id': memory_id
            }
        )
    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Background storage failed: {e}")


async def wait_for_pending_storage(timeout: float = 10.0):
    """
    Wait for all pending background storage tasks to complete.

    Call this at app shutdown to ensure no interactions are lost.

    Args:
        timeout: Maximum seconds to wait (default 10s)
    """
    if not _pending_storage_tasks:
        return

    logger.info(f"[SHUTDOWN] Waiting for {len(_pending_storage_tasks)} pending storage tasks...")
    try:
        await asyncio.wait_for(
            asyncio.gather(*_pending_storage_tasks, return_exceptions=True),
            timeout=timeout
        )
        logger.info("[SHUTDOWN] All storage tasks completed")
    except asyncio.TimeoutError:
        logger.warning(f"[SHUTDOWN] Storage tasks timed out after {timeout}s, {len(_pending_storage_tasks)} may be incomplete")


async def _persist_uploads(orchestrator, files_result: ProcessedFilesResult):
    """
    Persist uploaded documents and images to ChromaDB reference_docs collection.

    Runs as a fire-and-forget background task so upload persistence doesn't
    block response delivery.
    """
    try:
        from knowledge.reference_docs_manager import ReferenceDocsManager

        # Get or create a ReferenceDocsManager
        ref_manager = None
        if hasattr(orchestrator, 'prompt_builder') and hasattr(orchestrator.prompt_builder, 'context_gatherer'):
            ref_manager = orchestrator.prompt_builder.context_gatherer.reference_docs_manager
        if not ref_manager:
            ref_manager = ReferenceDocsManager()

        # Persist text documents
        for doc in files_result.documents:
            if doc.content_text and not doc.error:
                try:
                    ref_manager.upload_text(
                        content=doc.content_text,
                        title=f"upload:{doc.filename}",
                        metadata_overrides={'type': 'user_upload'}
                    )
                    logger.info(f"[PERSIST] Stored document upload: {doc.filename}")
                except Exception as e:
                    logger.warning(f"[PERSIST] Failed to store document {doc.filename}: {e}")

        # Persist images (store a description text + image metadata)
        for img in files_result.images:
            if not img.error:
                try:
                    description = f"User uploaded image: {img.filename} ({img.media_type}, {img.file_size} bytes)"
                    overrides = {
                        'type': 'user_upload',
                        'is_image': True,
                        'media_type': img.media_type,
                    }
                    if img.file_path:
                        overrides['image_path'] = img.file_path
                    ref_manager.upload_text(
                        content=description,
                        title=f"upload:{img.filename}",
                        metadata_overrides=overrides
                    )
                    logger.info(f"[PERSIST] Stored image upload: {img.filename}")
                except Exception as e:
                    logger.warning(f"[PERSIST] Failed to store image {img.filename}: {e}")

    except Exception as e:
        logger.error(f"[PERSIST] Upload persistence failed: {e}")


def smart_join(prev: str, new: str) -> str:
    """
    Inserts a space between tokens unless the new chunk begins with punctuation or whitespace.
    Prevents jammed-together words while respecting formatting.
    """
    if not prev:
        return new
    if prev.endswith((' ', '\n')) or new.startswith((' ', '\n', '.', ',', '?', '!', "'", '"', ")", "’", "”")):
        return prev + new
    else:
        return prev + ' ' + new


@log_and_time("Handle Submit")
async def handle_submit(
    user_text,
    files,
    history,
    use_raw_gpt,
    orchestrator,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    force_summarize=False,
    include_summaries=True,
    personality=None,
    fast_mode=False
):
    logger.info(f"[Handle Submit] ENTRY - raw_mode={use_raw_gpt}, fast_mode={fast_mode}")
    logger.info(f"[Handle Submit] Query: {user_text[:100]}...")

    # Update activity timestamp for idle monitor
    try:
        import main
        if hasattr(main, 'update_activity_timestamp'):
            main.update_activity_timestamp()
    except (ImportError, AttributeError) as e:
        logger.debug(f"[Handlers] Could not update activity timestamp: {e}")

    # Get conversation logger
    conversation_logger = get_conversation_logger()

    if not user_text.strip():
        yield {"role": "assistant", "content": "⚠️ Empty input received."}
        return

    # Process files using security-hardened FileProcessor
    # Supports .txt, .csv, .py, .docx files and .png, .jpg, .jpeg, .gif, .webp images
    file_names = [file.name for file in files] if files else []
    files_result = await file_processor.process_files_structured(user_text, files or [])
    merged_input = files_result.text_content

    # Persist uploads to ChromaDB in background (fire-and-forget)
    if files_result.documents or files_result.images:
        persist_task = asyncio.create_task(_persist_uploads(orchestrator, files_result))
        _pending_storage_tasks.add(persist_task)
        persist_task.add_done_callback(_pending_storage_tasks.discard)

    # RAW MODE: go straight through orchestrator (personality hook is handled inside process_user_query)
    if use_raw_gpt:
        logger.info("[Handle Submit] RAW MODE ENABLED – skipping memory and prompt building.")

        # Send immediate progress to prevent mobile timeout
        yield {"role": "assistant", "content": "💭 Processing...", "is_progress": True}

        response_text, debug_info = await orchestrator.process_user_query(
            user_input=merged_input,
            files=None,
            use_raw_mode=True,
            personality=personality
        )

        # Log the raw mode conversation
        conversation_logger.log_interaction(
            user_input=user_text,  # Log original input without file content for clarity
            assistant_response=response_text,
            metadata={
                'mode': 'raw',
                'files': file_names if file_names else None,
                'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
            }
        )

        # Emit final chunk including debug record for UI tracing
        # Compute token counts for the raw prompt (no system prompt)
        try:
            model_for_tokens = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(merged_input, model_for_tokens)) if tm else None
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[Handlers] Token counting failed for raw mode: {e}")
            prompt_tokens = None

        debug_record = {
            'mode': 'raw',
            'query': user_text,
            'prompt': merged_input,
            'system_prompt': None,
            'response': response_text,
            'model': getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)(),
            'prompt_tokens': prompt_tokens,
            'system_tokens': 0,
            'total_tokens': (prompt_tokens or 0),
            'citations': [],  # Raw mode bypasses memory
            'citations_enabled': False,
        }
        yield {"role": "assistant", "content": response_text, "debug": debug_record}
        return

    # Check if agentic search might be used (need to know before calling prepare_prompt)
    _cfg = getattr(orchestrator, 'config', {}) or {}
    agentic_cfg = _cfg.get('agentic_search', {}) if isinstance(_cfg, dict) else {}
    agentic_enabled = bool(agentic_cfg.get('enabled', False))

    # ENHANCED MODE: build prompt first via orchestrator.prepare_prompt (do NOT pass personality here)
    # Always request raw context (needed for images in multimodal models and agentic search)

    # Send immediate progress to prevent mobile timeout during prompt preparation
    yield {"role": "assistant", "content": "💭 Thinking...", "is_progress": True}

    logger.info("[Handle Submit] >>> Starting prepare_prompt...")

    # Send periodic keepalive during slow prepare_prompt (prevents mobile timeout)

    # Apply Fast Mode limits BEFORE prepare_prompt starts
    _original_limits = {}
    if fast_mode:
        logger.warning("[Handle Submit] ⚡⚡⚡ FAST MODE ENABLED ⚡⚡⚡")
        import core.prompt.builder as builder_module
        # Override builder module constants (the REAL location of these limits)
        _original_limits['PROMPT_MAX_MEMS'] = builder_module.PROMPT_MAX_MEMS
        logger.warning(f"[Fast Mode] PROMPT_MAX_MEMS: {builder_module.PROMPT_MAX_MEMS} → 10")
        builder_module.PROMPT_MAX_MEMS = 10

        _original_limits['PROMPT_MAX_RECENT'] = builder_module.PROMPT_MAX_RECENT
        logger.warning(f"[Fast Mode] PROMPT_MAX_RECENT: {builder_module.PROMPT_MAX_RECENT} → 5")
        builder_module.PROMPT_MAX_RECENT = 5

        if hasattr(builder_module, 'PROMPT_MAX_SEMANTIC'):
            _original_limits['PROMPT_MAX_SEMANTIC'] = builder_module.PROMPT_MAX_SEMANTIC
            logger.warning(f"[Fast Mode] PROMPT_MAX_SEMANTIC: {builder_module.PROMPT_MAX_SEMANTIC} → 8")
            builder_module.PROMPT_MAX_SEMANTIC = 8

        # CRITICAL: Set fast mode flags to reduce expensive hybrid retrieval (2150 → ~40 candidates)
        if hasattr(orchestrator.prompt_builder, 'context_gatherer'):
            orchestrator.prompt_builder.context_gatherer._fast_mode = True
            logger.warning("[Fast Mode] Set context_gatherer._fast_mode = True")

        # Also set on hybrid_retriever via memory_coordinator
        if hasattr(orchestrator, 'memory_coordinator'):
            retriever = getattr(orchestrator.memory_coordinator, '_retriever', None)
            if retriever and hasattr(retriever, 'hybrid_retriever'):
                retriever.hybrid_retriever._fast_mode = True
                logger.warning("[Fast Mode] Set hybrid_retriever._fast_mode = True (2150 → ~40 candidates)")

    prepare_task = asyncio.create_task(orchestrator.prepare_prompt(
        user_input=user_text,
        files=files,
        use_raw_mode=False,  # enhanced mode
        return_context=True  # Always get raw context for images and agentic search
    ))

    # Yield progress every 2 seconds while waiting
    progress_messages = ["💭 Analyzing context...", "🔍 Searching memories...", "📚 Building prompt..."]
    progress_idx = 0
    while not prepare_task.done():
        await asyncio.sleep(2)
        if not prepare_task.done():
            yield {"role": "assistant", "content": progress_messages[progress_idx % len(progress_messages)], "is_progress": True}
            progress_idx += 1

    prep_result = await prepare_task

    # Unpack result - always expect 3 values now
    full_prompt, system_prompt, raw_context = prep_result
    raw_context = raw_context or {}

    # Extract images for multimodal models
    note_images = raw_context.get("note_images", [])
    if note_images:
        logger.warning(f"[Handle Submit] Extracted {len(note_images)} images from raw_context for multimodal generation")

    # Inject uploaded images into note_images for immediate multimodal use
    if files_result.images:
        for img in files_result.images:
            if img.base64_data and not img.error:
                note_images.append({
                    "note_index": 0,
                    "note_title": f"Upload: {img.filename}",
                    "note_section": "",
                    "filename": img.filename,
                    "media_type": img.media_type,
                    "data": img.base64_data,
                })
        raw_context["note_images"] = note_images
        logger.warning(f"[Handle Submit] Injected {len(files_result.images)} upload images, total note_images={len(note_images)}")

    logger.info(f"[Handle Submit] <<< prepare_prompt done, prompt_len={len(full_prompt)}")

    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")
    logger.debug(f"[Handle Submit] Agentic pre-check: enabled={agentic_enabled}")

    if agentic_enabled:
        _lower = user_text.lower().strip()
        _words = _lower.split()
        needs_computation = False
        needs_memory = False
        _matched_entities = set()

        # --- Tier 1: Keyword heuristics (instant, no LLM) ---
        _computation_keywords = [
            'calculate', 'compute', 'solve', 'integral', 'derivative', 'equation',
            'fibonacci', 'factorial', 'prime', 'mean', 'median', 'standard deviation',
            'matrix', 'vector', 'plot', 'graph', 'chart', 'numpy', 'pandas', 'sympy',
            'statistics', 'regression', 'correlation', 'sum of', 'product of',
            'evaluate', 'simplify', 'expand', 'factor', 'differentiate', 'integrate'
        ]
        needs_computation = any(kw in _lower for kw in _computation_keywords)

        _memory_keywords = [
            'documentation', 'daemon docs', 'architecture',
            'do you remember', 'did we talk', 'did we discuss',
            'did i tell you', 'did i mention', 'have i told you',
            'what do you know about me', 'what are my',
            'my notes', 'obsidian', 'in my vault',
            'past conversations', 'search your memory',
            'search memory', 'check your memory', 'look up',
            'my facts', 'what did i say',
        ]
        needs_memory = any(kw in _lower for kw in _memory_keywords)

        # --- Tier 2: Entity match (instant, no LLM) ---
        # If query mentions a known personal entity from the knowledge graph,
        # it's almost certainly a memory query. Runs before casual skip so
        # short entity queries like "tell me about Flapjack" aren't filtered out.
        if not needs_computation and not needs_memory:
            try:
                _resolver = getattr(getattr(orchestrator, 'memory_system', None), 'entity_resolver', None)
                if _resolver:
                    from memory.graph_utils import extract_graph_entities
                    _matched_entities = extract_graph_entities(user_text, _resolver)
                    _matched_entities.discard("user")  # "user" is not a meaningful match
                    if _matched_entities:
                        needs_memory = True
                        logger.debug(f"[Handle Submit] Agentic triggered - query mentions known entities: {_matched_entities}")
            except Exception as e:
                logger.debug(f"[Handle Submit] Entity match check failed (non-fatal): {e}")

        # Casual skip filter: only applies when no keyword/entity trigger fired
        _skip_patterns = [
            len(_words) < 5 and not any(w in _lower for w in ['search', 'news', 'latest', 'current', 'today', 'recent', '2026', '2025']),
            _lower.startswith(('nice', 'thanks', 'thank you', 'cool', 'great', 'awesome', 'got it', 'ok ', 'okay', 'yeah', 'yes', 'no ', 'nope', 'nah')),
            'working' in _lower and 'search' in _lower,  # meta-comments about search
            all(w in ['yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope', 'thanks', 'thank', 'you'] for w in _words),
        ]
        if not needs_computation and not needs_memory and any(_skip_patterns):
            logger.debug("[Handle Submit] Agentic skipped - casual/short message")
            should_use_agentic = False
            search_terms = []
        else:

            if needs_computation:
                logger.debug("[Handle Submit] Agentic triggered - computational query detected")
                should_use_agentic = True
                search_terms = []  # No web search needed, just computation
            elif needs_memory:
                logger.debug("[Handle Submit] Agentic triggered - memory search query detected")
                should_use_agentic = True
                search_terms = []  # No web search needed, memory tool handles it
            else:
                # Check if web search or memory search should be triggered using LLM-first trigger
                try:
                    from utils.web_search_trigger import analyze_for_web_search_llm
                    trigger_decision = await analyze_for_web_search_llm(
                        query=user_text,
                        model_manager=orchestrator.model_manager
                    )
                    # trigger_decision is a WebSearchDecision object with attributes
                    should_use_agentic = getattr(trigger_decision, 'should_search', False)
                    search_terms = getattr(trigger_decision, 'search_terms', []) or []

                    # LLM-based memory search fallback: if web search not needed but
                    # LLM detected memory/recall intent, enter agentic for search_memory tool
                    if not should_use_agentic and getattr(trigger_decision, 'needs_memory_search', False):
                        logger.debug("[Handle Submit] Agentic triggered - LLM detected memory search intent")
                        should_use_agentic = True
                        needs_memory = True  # skip initial web search
                        search_terms = []

                    logger.debug(f"[Handle Submit] Agentic trigger: should_search={should_use_agentic}, needs_memory={needs_memory}, terms={search_terms}")
                except Exception as e:
                    logger.warning(f"[Handle Submit] Agentic trigger check failed: {e}")
                    import traceback
                    traceback.print_exc()
                    should_use_agentic = False
                    search_terms = []

        if should_use_agentic:
            logger.warning("[Handle Submit] AGENTIC SEARCH MODE - routing through agentic controller")
            try:
                from core.agentic import AgenticSearchController, ProgressEvent

                # Get the agentic controller from orchestrator
                agentic_controller = orchestrator.agentic_controller
                model_name = orchestrator.model_manager.get_active_model_name()

                # Get initial search terms from the trigger decision we already have
                initial_terms = search_terms if search_terms else []
                logger.debug(f"[Handle Submit] Agentic initial terms: {initial_terms}")

                # Run agentic search loop with RAG context
                agentic_response = ""
                logger.debug(f"[Handle Submit] Starting agentic loop with RAG context keys: {list(raw_context.keys())}")
                async for item in agentic_controller.run_agentic_search(
                    query=user_text,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    initial_search_terms=initial_terms,
                    initial_context=raw_context,  # Pass RAG context to agentic controller
                    skip_initial_search=needs_computation or needs_memory,  # Skip web search for computation/memory-only queries
                ):
                    if isinstance(item, ProgressEvent):
                        # Yield progress events as status messages
                        status_icon = {
                            "thinking": "💭",
                            "searching": "🔍",
                            "searching_memory": "🧠",
                            "found_results": "📄",
                            "computing": "🔢",
                            "computed": "✓",
                            "executing_code": "🐍",
                            "code_executed": "✅",
                            "code_error": "⚠️",
                            "synthesizing": "✨",
                            "done": "✅",
                            "error": "❌",
                        }.get(item.event_type, "•")
                        status_msg = f"{status_icon} {item.message}"
                        logger.debug(f"[Handle Submit] Agentic progress: {item.event_type}")
                        yield {"role": "assistant", "content": status_msg, "is_progress": True}
                    else:
                        # Response chunk - accumulate and stream
                        agentic_response += item
                        yield {"role": "assistant", "content": agentic_response}

                # Final output from agentic search - strip thinking blocks
                final_output = agentic_response
                thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)
                display_output = final_answer if final_answer else final_output
                logger.debug(f"[Handle Submit] Agentic loop done, response_len={len(final_output)}, display_len={len(display_output)}")

                # Token counts for debug
                try:
                    tm = getattr(orchestrator, 'tokenizer_manager', None)
                    prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
                    system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
                    total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"[Handlers] Token counting failed for agentic mode: {e}")
                    prompt_tokens = None
                    system_tokens = None
                    total_tokens = None

                debug_record = {
                    'mode': 'agentic-search',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': final_output,
                    'model': model_name,
                    'prompt_tokens': prompt_tokens,
                    'system_tokens': system_tokens,
                    'total_tokens': total_tokens,
                    'citations': [],
                    'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                }
                # Yield the clean response first (without debug, to ensure it displays)
                logger.debug(f"[Handle Submit] Agentic yielding final response: {display_output[:100]}...")
                yield {"role": "assistant", "content": display_output}

                # Then yield with debug record for the debug trace
                yield {"role": "assistant", "content": display_output, "debug": debug_record}
                logger.debug("[Handle Submit] Agentic final response yielded")

                # Store interaction in memory
                try:
                    # Sanitize response before storage
                    try:
                        from core.prompt import _truncate_at_spurious_turns
                        final_output_sanitized = _truncate_at_spurious_turns(final_output)
                    except Exception as e:
                        logger.warning(f"[Handle Submit] Failed to sanitize agentic response: {e}")
                        final_output_sanitized = final_output

                    tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                    memory_id = await orchestrator.memory_system.store_interaction(
                        query=merged_input,
                        response=final_output_sanitized,
                        tags=tags
                    )
                    logger.info(f"[Handle Submit] Agentic interaction stored with ID: {memory_id}")

                    # Log conversation
                    conversation_logger.log_interaction(
                        user_input=user_text,
                        assistant_response=final_output,
                        metadata={
                            'mode': 'agentic-search',
                            'files': file_names if file_names else None,
                            'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                            'topic': getattr(orchestrator, 'current_topic', None),
                            'db_id': memory_id
                        }
                    )
                except Exception as e:
                    logger.error(f"[Handle Submit] Failed to store agentic interaction: {e}")

                return  # Exit after agentic search completes

            except Exception as e:
                logger.error(f"[Handle Submit] Agentic search failed, falling back to standard: {e}")
                import traceback
                logger.debug(f"[Agentic] Exception traceback:\n{traceback.format_exc()}")

    final_output = ""
    display_output = ""
    debug_emitted = False
    try:
        logger.debug(
            "[🔍 FINAL MESSAGE PAYLOAD TO OPENAI]:\n" +
            json.dumps(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': full_prompt}],
                indent=2
            )
        )

        # Respect runtime settings via orchestrator.config if available
        _cfg = getattr(orchestrator, 'config', {}) or {}
        _features = _cfg.get('features', {}) if isinstance(_cfg, dict) else {}
        _enable_bestof = bool(_features.get('enable_best_of', ENABLE_BEST_OF))
        # Force streaming in GUI unless explicitly disabled via env FORCE_STREAMING=0
        try:
            import os as _os
            if str(_os.getenv('FORCE_STREAMING', '1')).strip().lower() in {'1','true','yes','on'}:
                _enable_bestof = False
        except (AttributeError, TypeError):
            pass

        # Policy: always stream, except DUEL mode. Single-model best-of is disabled for streaming UX.
        use_bestof = False
        try:
            # Detect duel-mode configuration from runtime features
            GEN_MODELS = list(_features.get('best_of_generator_models', []))
            SEL_MODELS = list(_features.get('best_of_selector_models', []))
            _DUEL_MODE = bool(_features.get('best_of_duel_mode', False))
            duel_possible = bool(_DUEL_MODE and len(GEN_MODELS) >= 2 and len(SEL_MODELS) >= 1)
            use_bestof = duel_possible  # only enable best-of path when duel is configured
            logger.info(f"[GUI] Duel check: duel_mode={_DUEL_MODE}, gen={len(GEN_MODELS)}, sel={len(SEL_MODELS)}, use_bestof(duel-only)={use_bestof}")
        except Exception as e:
            use_bestof = False
            logger.info(f"[GUI] Duel check failed: {e}; default to streaming")

        model_name = orchestrator.model_manager.get_active_model_name()
        # Token counts for debug trace
        try:
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
            system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
            total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[Handlers] Token counting failed for enhanced mode: {e}")
            prompt_tokens = None
            system_tokens = None
            total_tokens = None
        bestof_model = BEST_OF_MODEL or model_name

        if use_bestof:
            # Best-of with optional latency budget and streaming fallback
            # If BEST_OF_TEMPS not explicitly configured, anchor around current runtime temperature
            try:
                _explicit_temps = tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else None
            except (TypeError, ValueError) as e:
                logger.debug(f"[Handlers] Could not parse BEST_OF_TEMPS: {e}")
                _explicit_temps = None

            if _explicit_temps and len(_explicit_temps) > 0:
                _temps_to_use = _explicit_temps
            else:
                # Derive two temps around current default_temperature to reflect the Settings slider
                try:
                    _t = float(getattr(orchestrator.model_manager, 'default_temperature', 0.7))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"[Handlers] Could not get default_temperature: {e}")
                    _t = 0.7
                # Slight spread while staying in [0.0, 1.5]
                _low = max(0.0, min(1.5, round(0.6 * _t, 2)))
                _hi = max(0.0, min(1.5, round(_t, 2)))
                _temps_to_use = (_low, _hi)

            # Read latency budget: runtime features override, else app_config default
            try:
                from config.app_config import BEST_OF_LATENCY_BUDGET_S as _DEF_BUDGET
            except ImportError:
                _DEF_BUDGET = 0.0
            try:
                _budget = float(_features.get('best_of_latency_budget_s', _DEF_BUDGET))
            except (TypeError, ValueError) as e:
                logger.debug(f"[Handlers] Could not parse latency budget: {e}")
                _budget = float(_DEF_BUDGET) if _DEF_BUDGET else 0.0

            # Check for duel mode configuration (two generators + one judge)
            try:
                from config.app_config import (
                    BEST_OF_GENERATOR_MODELS as DEF_GEN_MODELS,
                    BEST_OF_SELECTOR_MODELS as DEF_SEL_MODELS,
                    BEST_OF_SELECTOR_MAX_TOKENS as DEF_SEL_MAXTOK,
                    BEST_OF_DUEL_MODE as DEF_DUEL_MODE,
                )
            except ImportError:
                DEF_GEN_MODELS = []
                DEF_SEL_MODELS = []
                DEF_SEL_MAXTOK = 64
                DEF_DUEL_MODE = False

            GEN_MODELS = list(_features.get('best_of_generator_models', DEF_GEN_MODELS))
            SEL_MODELS = list(_features.get('best_of_selector_models', DEF_SEL_MODELS))
            SEL_MAXTOK = int(_features.get('best_of_selector_max_tokens', DEF_SEL_MAXTOK))
            _DUEL_MODE = bool(_features.get('best_of_duel_mode', DEF_DUEL_MODE))
            use_duel = bool(_DUEL_MODE and len(GEN_MODELS) == 2 and len(SEL_MODELS) >= 1)

            import asyncio as _a
            try:
                if _budget and _budget > 0:
                    # With latency budget
                    if use_duel:
                        # DUEL MODE: Two models (Claude Opus + GPT-5) with judge (GPT-4o-mini)
                        m1, m2 = list(GEN_MODELS)[:2]
                        judge = list(SEL_MODELS)[0]
                        logger.info(f"[GUI] DUEL MODE: {m1} vs {m2}, judge={judge}")
                        best_task = _a.create_task(
                            orchestrator.response_generator.generate_duel_and_judge(
                                prompt=full_prompt,
                                model_a=m1,
                                model_b=m2,
                                judge_model=judge,
                                system_prompt=system_prompt,
                                question_text=user_text,
                                context_hint=full_prompt,
                                max_tokens=BEST_OF_MAX_TOKENS,
                                temperature_a=(_temps_to_use[0] if len(_temps_to_use) > 0 else None),
                                temperature_b=(_temps_to_use[1] if len(_temps_to_use) > 1 else None),
                                judge_max_tokens=int(SEL_MAXTOK),
                            )
                        )
                    else:
                        # Single-model best-of
                        logger.info(f"[GUI] SINGLE-MODEL BEST-OF: {bestof_model}")
                        best_task = _a.create_task(
                            orchestrator.response_generator.generate_best_of(
                                prompt=full_prompt,
                                model_name=bestof_model,
                                system_prompt=system_prompt,
                                question_text=user_text,
                                context_hint=full_prompt,
                                n=BEST_OF_N,
                                temps=_temps_to_use,
                                max_tokens=BEST_OF_MAX_TOKENS,
                            )
                        )
                    best = await _a.wait_for(best_task, timeout=float(_budget))
                    logger.info(f"[GUI] Got best result: type={type(best)}, is_dict={isinstance(best, dict)}, keys={best.keys() if isinstance(best, dict) else 'N/A'}")
                    # Handle dict return from duel mode (with thinking blocks) or string from best-of
                    if isinstance(best, dict) and 'answer' in best:
                        final_output = best['answer']
                        display_output = final_output
                        # Yield thinking blocks for duel mode
                        thinking_data = {
                            'thinking_a': best.get('thinking_a', ''),
                            'thinking_b': best.get('thinking_b', ''),
                            'model_a': best.get('model_a', ''),
                            'model_b': best.get('model_b', ''),
                            'winner': best.get('winner', ''),
                            'scores': best.get('scores', {})
                        }
                        logger.info(f"[GUI] YIELDING THINKING DATA (with budget): models={thinking_data.get('model_a')}/{thinking_data.get('model_b')}, winner={thinking_data.get('winner')}")
                        yield {"role": "assistant", "content": "", "thinking": thinking_data}
                    else:
                        final_output = best
                        # Parse thinking block - only show final answer to user
                        _, final_answer = ResponseParser.parse_thinking_block(final_output)
                        display_output = final_answer if final_answer else final_output
                    # Create debug record for best-of with latency budget
                    try:
                        tm = getattr(orchestrator, 'tokenizer_manager', None)
                        model_for_tokens = bestof_model
                        prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                        system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                        total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug(f"[Handlers] Token counting failed for best-of with budget: {e}")
                        prompt_tokens2 = None
                        system_tokens2 = None
                        total_tokens2 = None
                    # Extract citations if enabled
                    citations = []
                    debug_output = final_output
                    if getattr(orchestrator, 'enable_citations', False):
                        try:
                            memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                            if memory_id_map:
                                debug_output, citations = orchestrator._extract_citations(final_output, memory_id_map)
                        except (AttributeError, KeyError) as e:
                            logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': debug_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
                        'citations': citations,
                        'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                    }
                    yield {"role": "assistant", "content": display_output, "debug": debug_record}
                    debug_emitted = True
                else:
                    # No budget: run to completion (non-streaming)
                    if use_duel:
                        # DUEL MODE: Two models (Claude Opus + GPT-5) with judge (GPT-4o-mini)
                        m1, m2 = list(GEN_MODELS)[:2]
                        judge = list(SEL_MODELS)[0]
                        logger.info(f"[GUI] DUEL MODE: {m1} vs {m2}, judge={judge}")
                        best = await orchestrator.response_generator.generate_duel_and_judge(
                            prompt=full_prompt,
                            model_a=m1,
                            model_b=m2,
                            judge_model=judge,
                            system_prompt=system_prompt,
                            question_text=user_text,
                            context_hint=full_prompt,
                            max_tokens=BEST_OF_MAX_TOKENS,
                            temperature_a=(_temps_to_use[0] if len(_temps_to_use) > 0 else None),
                            temperature_b=(_temps_to_use[1] if len(_temps_to_use) > 1 else None),
                            judge_max_tokens=int(SEL_MAXTOK),
                        )
                    else:
                        # Single-model best-of
                        logger.info(f"[GUI] SINGLE-MODEL BEST-OF: {bestof_model}")
                        best = await orchestrator.response_generator.generate_best_of(
                            prompt=full_prompt,
                            model_name=bestof_model,
                            system_prompt=system_prompt,
                            question_text=user_text,
                            context_hint=full_prompt,
                            n=BEST_OF_N,
                            temps=_temps_to_use,
                            max_tokens=BEST_OF_MAX_TOKENS,
                        )
                    # Handle dict return from duel mode (with thinking blocks) or string from best-of
                    if isinstance(best, dict) and 'answer' in best:
                        final_output = best['answer']
                        display_output = final_output
                        # Yield thinking blocks for duel mode
                        thinking_data = {
                            'thinking_a': best.get('thinking_a', ''),
                            'thinking_b': best.get('thinking_b', ''),
                            'model_a': best.get('model_a', ''),
                            'model_b': best.get('model_b', ''),
                            'winner': best.get('winner', ''),
                            'scores': best.get('scores', {})
                        }
                        logger.info(f"[GUI] YIELDING THINKING DATA (no budget): models={thinking_data.get('model_a')}/{thinking_data.get('model_b')}, winner={thinking_data.get('winner')}")
                        yield {"role": "assistant", "content": "", "thinking": thinking_data}
                    else:
                        final_output = best
                        # Parse thinking block - only show final answer to user
                        _, final_answer = ResponseParser.parse_thinking_block(final_output)
                        display_output = final_answer if final_answer else final_output
                    # Create debug record for best-of without latency budget
                    try:
                        tm = getattr(orchestrator, 'tokenizer_manager', None)
                        model_for_tokens = bestof_model
                        prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                        system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                        total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug(f"[Handlers] Token counting failed for best-of (no budget): {e}")
                        prompt_tokens2 = None
                        system_tokens2 = None
                        total_tokens2 = None
                    # Extract citations if enabled
                    citations = []
                    debug_output = final_output
                    if getattr(orchestrator, 'enable_citations', False):
                        try:
                            memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                            if memory_id_map:
                                debug_output, citations = orchestrator._extract_citations(final_output, memory_id_map)
                        except (AttributeError, KeyError) as e:
                            logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': debug_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
                        'citations': citations,
                        'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                    }
                    yield {"role": "assistant", "content": display_output, "debug": debug_record}
                    debug_emitted = True
            except Exception as e:
                # Timeout or error: fall back to streaming
                logger.warning(f"[GUI] Best-of/duel mode failed, falling back to streaming: {e}")
                import traceback
                logger.debug(f"[GUI] Exception traceback:\n{traceback.format_exc()}")
                try:
                    if 'best_task' in locals():
                        best_task.cancel()
                except (NameError, AttributeError) as e:
                    logger.debug(f"[Handlers] Could not cancel best_task: {e}")
                thinking_started = False
                thinking_complete = False
                async for chunk in orchestrator.response_generator.generate_streaming_response(
                    prompt=full_prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    images=note_images if note_images else None  # Pass images for multimodal models
                ):
                    final_output = smart_join(final_output, chunk)
                    # Parse in real-time to separate thinking from answer
                    thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)

                    # If we have thinking content and haven't shown the answer yet
                    if thinking_part and not final_answer:
                        # Stream thinking block with special marker
                        thinking_started = True
                        display_output = f"💭 **Thinking...**\n\n{thinking_part}"
                        yield {"role": "assistant", "content": display_output, "is_thinking": True}
                    elif thinking_part and final_answer and not thinking_complete:
                        # Thinking is complete, answer is starting - switch to answer
                        thinking_complete = True
                        # Remove XML-like wrappers (e.g., <result> … </result>) for GUI display
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_answer)
                        except (IndexError, AttributeError):
                            display_output = final_answer
                        yield {"role": "assistant", "content": display_output, "is_thinking": False}
                    elif final_answer:
                        # Continue streaming the answer
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_answer)
                        except (IndexError, AttributeError):
                            display_output = final_answer
                        yield {"role": "assistant", "content": display_output}
                    else:
                        # No thinking block detected, stream normally
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", (final_output or ""), flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_output)
                        except (IndexError, AttributeError):
                            display_output = final_output
                        yield {"role": "assistant", "content": display_output}
                # After fallback streaming completes, emit debug record
                thinking_part_fb, final_answer_fb = ResponseParser.parse_thinking_block(final_output)
                if thinking_part_fb:
                    final_output = final_answer_fb if final_answer_fb else final_output
                try:
                    tm = getattr(orchestrator, 'tokenizer_manager', None)
                    model_for_tokens = model_name
                    prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                    system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                    total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"[Handlers] Token counting failed for fallback streaming: {e}")
                    prompt_tokens2 = None
                    system_tokens2 = None
                    total_tokens2 = None
                # Ensure we have content to log even if no chunk set display_output yet
                _resp_for_debug = display_output or final_output

                # Truncate at spurious turn markers (training data leakage) in fallback
                try:
                    from core.prompt import _truncate_at_spurious_turns
                    _resp_for_debug = _truncate_at_spurious_turns(_resp_for_debug)
                except Exception as e:
                    logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns in fallback: {e}")

                # Extract citations if enabled
                citations = []
                if getattr(orchestrator, 'enable_citations', False):
                    try:
                        memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                        if memory_id_map:
                            _resp_for_debug, citations = orchestrator._extract_citations(_resp_for_debug, memory_id_map)
                    except Exception as e:
                        logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                debug_record = {
                    'mode': 'fallback-streaming',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': _resp_for_debug,
                    'model': model_name,
                    'prompt_tokens': prompt_tokens2,
                    'system_tokens': system_tokens2,
                    'total_tokens': total_tokens2,
                    'citations': citations,
                    'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                }
                yield {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
                debug_emitted = True
        else:
            # Default streaming path with thinking block visibility
            logger.info(f"[Handle Submit] >>> Starting streaming with model={model_name}")
            thinking_started = False
            thinking_complete = False
            chunk_count = 0
            async for chunk in orchestrator.response_generator.generate_streaming_response(
                prompt=full_prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                images=note_images if note_images else None  # Pass images for multimodal models
            ):
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 20 == 0:
                    logger.info(f"[Handle Submit] Chunk #{chunk_count}: {str(chunk)[:50]}...")
                final_output = smart_join(final_output, chunk)
                # Parse in real-time to separate thinking from answer
                thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)

                # If we have thinking content and haven't shown the answer yet
                if thinking_part and not final_answer:
                    # Stream thinking block with special marker
                    thinking_started = True
                    display_output = f"💭 **Thinking...**\n\n{thinking_part}"
                    yield {"role": "assistant", "content": display_output, "is_thinking": True}
                elif thinking_part and final_answer and not thinking_complete:
                    # Thinking is complete, answer is starting - switch to answer
                    thinking_complete = True
                    display_output = final_answer
                    yield {"role": "assistant", "content": display_output, "is_thinking": False}
                elif final_answer:
                    # Continue streaming the answer
                    try:
                        import re
                        # Strip ONLY outer wrapper tags at start/end (not tags mentioned in content)
                        # Use non-greedy match and ensure we capture everything between outer tags
                        m = re.match(r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                        display_output = (m.group(2).strip() if m else final_answer)
                    except (IndexError, AttributeError):
                        display_output = final_answer
                    yield {"role": "assistant", "content": display_output}
                else:
                    # No thinking block detected, stream normally
                    try:
                        import re
                        # Strip ONLY outer wrapper tags at start/end (not tags mentioned in content)
                        m = re.match(r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$", (final_output or ""), flags=re.IGNORECASE)
                        display_output = (m.group(2).strip() if m else final_output)
                    except (IndexError, AttributeError):
                        display_output = final_output
                    yield {"role": "assistant", "content": display_output}

            # After streaming completes, parse thinking block for logging and storage
            logger.info(f"[Handle Submit] <<< Streaming done, {chunk_count} chunks, output_len={len(final_output)}")

            # Handle empty response from API (model returned no content)
            if chunk_count == 0 or not final_output.strip():
                model_name_for_error = model_name or "unknown"
                error_msg = f"⚠️ Model `{model_name_for_error}` returned an empty response. This can happen when:\n• The model is temporarily unavailable\n• Rate limiting or quota issues\n• The model failed to process the request\n\nTry switching to a different model or retry your message."
                logger.warning(f"[Handle Submit] Empty response detected from {model_name_for_error}")
                yield {"role": "assistant", "content": error_msg}
                return

            thinking_part_stream, final_answer_stream = ResponseParser.parse_thinking_block(final_output)
            if thinking_part_stream:
                logger.debug(f"[HANDLE_SUBMIT][THINKING BLOCK FROM STREAM]\n{thinking_part_stream}")
                # Update final_output to only include the final answer for storage
                final_output = final_answer_stream if final_answer_stream else final_output

            # After streaming completes, emit a final debug record so the Debug Trace tab is populated
            try:
                tm = getattr(orchestrator, 'tokenizer_manager', None)
                model_for_tokens = model_name
                prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"[Handlers] Token counting failed for streaming debug: {e}")
                prompt_tokens2 = None
                system_tokens2 = None
                total_tokens2 = None

            # Ensure we have content to log even if no chunk set display_output yet
            _resp_for_debug = display_output or final_output

            # Truncate at spurious turn markers (training data leakage) for display
            try:
                from core.prompt import _truncate_at_spurious_turns
                _resp_for_debug = _truncate_at_spurious_turns(_resp_for_debug)
            except Exception as e:
                logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns in display: {e}")

            # Extract citations if enabled
            citations = []
            if getattr(orchestrator, 'enable_citations', False):
                try:
                    # Get memory_id_map from orchestrator if available
                    memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                    if memory_id_map:
                        _resp_for_debug, citations = orchestrator._extract_citations(_resp_for_debug, memory_id_map)
                except Exception as e:
                    logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

            debug_record = {
                'mode': 'enhanced',
                'query': user_text,
                'prompt': full_prompt,
                'system_prompt': system_prompt,
                'response': _resp_for_debug,
                'model': model_name,
                'prompt_tokens': prompt_tokens2,
                'system_tokens': system_tokens2,
                'total_tokens': total_tokens2,
                'citations': citations,
                'citations_enabled': getattr(orchestrator, 'enable_citations', False),
            }
            yield {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
            debug_emitted = True

    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Streaming error: {e}")
        error_message = f"⚠️ Streaming error: {str(e)}"

        # Log error conversation
        conversation_logger.log_interaction(
            user_input=user_text,
            assistant_response=error_message,
            metadata={
                'error': str(e),
                'mode': 'enhanced',
                'files': file_names if file_names else None
            }
        )

        yield {"role": "assistant", "content": error_message}

    finally:
        # Clean up fast mode flags (defensive try/except to never interfere with streaming)
        if fast_mode:
            try:
                if hasattr(orchestrator.prompt_builder, 'context_gatherer'):
                    orchestrator.prompt_builder.context_gatherer._fast_mode = False
                if hasattr(orchestrator, 'memory_coordinator'):
                    retriever = getattr(orchestrator.memory_coordinator, '_retriever', None)
                    if retriever and hasattr(retriever, 'hybrid_retriever'):
                        retriever.hybrid_retriever._fast_mode = False
                logger.warning("[Fast Mode] Flags cleared")
            except Exception as e:
                logger.error(f"[Fast Mode] Cleanup error (non-fatal): {e}")

        # Persist interaction and debug after streaming, but do not emit additional
        # assistant content here (avoid overwriting the last streamed UI state).
        # Skip storage if response is an error message (starts with error indicators)
        is_error_response = final_output.strip().startswith(('[Error:', '⚠️')) if final_output else True
        if final_output and len(user_text.strip()) > 0 and not is_error_response:
            # Store in memory system FIRST to get the db_id
            memory_id = None
            try:
                logger.info("[HANDLE_SUBMIT] Storing interaction in memory...")
                tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                # Ensure corpus capacity is generous during testing (override at runtime)
                try:
                    cm = getattr(getattr(orchestrator, "memory_system", None), "corpus_manager", None)
                    if cm and hasattr(cm, "max_entries"):
                        # Default test cap to 5000 if not set via env
                        import os as _os
                        cm.max_entries = int(_os.getenv("CORPUS_MAX_ENTRIES", "5000"))
                except (AttributeError, ValueError) as e:
                    logger.debug(f"[Handlers] Could not override corpus max_entries: {e}")

                # Parse thinking block from final_output before storing
                # Only store the final answer, not the thinking process
                thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)
                if thinking_part:
                    logger.debug(f"[HANDLE_SUBMIT][THINKING BLOCK]\n{thinking_part}")
                response_to_store = final_answer if final_answer else final_output

                # Truncate at spurious turn markers (training data leakage)
                try:
                    from core.prompt import _truncate_at_spurious_turns
                    response_to_store = _truncate_at_spurious_turns(response_to_store)
                except Exception as e:
                    logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns: {e}")

                # Sanitize any echoed prompt headers before storing
                try:
                    import re
                    header_patterns = [
                        r"^\s*\[TIME CONTEXT\]",
                        r"^\s*\[RECENT CONVERSATION[^\]]*\]",
                        r"^\s*\[RELEVANT INFORMATION\]",
                        r"^\s*\[RELEVANT MEMORIES\]",
                        r"^\s*\[FACTS[ ^\]]*\]",
                        r"^\s*\[RECENT FACTS\]",
                        r"^\s*\[CURRENT MESSAGE FACTS\]",
                        r"^\s*\[DIRECTIVES\]",
                        r"^\s*\[CURRENT USER QUERY[ ^\]]*\]",
                        r"^\s*\[USER INPUT\]",
                        r"^\s*\[BACKGROUND KNOWLEDGE\]",
                        r"^\s*\[CONVERSATION SUMMARIES[ ^\]]*\]",
                        r"^\s*\[RECENT REFLECTIONS[ ^\]]*\]",
                        r"^\s*\[SESSION REFLECTIONS[ ^\]]*\]",
                    ]
                    header_re = re.compile("(" + ")|(".join(header_patterns) + ")", re.IGNORECASE)
                    lines = []
                    skip = False
                    for line in (response_to_store.splitlines() if response_to_store else []):
                        if header_re.search(line):
                            skip = True
                            continue
                        if skip:
                            if not line.strip():
                                skip = False
                            continue
                        lines.append(line)
                    response_to_store = "\n".join(lines).strip()
                except (re.error, TypeError, AttributeError) as e:
                    logger.debug(f"[Handlers] Header sanitization failed: {e}")

                # Fire-and-forget background storage (saves ~1.7s of LLM calls)
                # Topic extraction and fact extraction run in background
                task = asyncio.create_task(_background_store_interaction(
                    orchestrator=orchestrator,
                    merged_input=merged_input,
                    response_to_store=response_to_store,
                    tags=tags,
                    user_text=user_text,
                    final_output=final_output,
                    personality=personality,
                    file_names=file_names,
                    conversation_logger=conversation_logger
                ))
                # Track task for graceful shutdown
                _pending_storage_tasks.add(task)
                task.add_done_callback(_pending_storage_tasks.discard)
                logger.info("[HANDLE_SUBMIT] Storage dispatched to background")

                # No mid-session consolidation: summaries are generated at shutdown
            except Exception as e:
                logger.error(f"[HANDLE_SUBMIT] Failed to dispatch storage: {e}")

            # Do not yield another assistant message here; the UI already
            # received the final content during streaming. If needed, a debug
            # record is captured in-stream above.

        # Restore original config limits if Fast Mode was enabled
        if fast_mode and '_original_limits' in locals():
            from config import app_config
            for key, value in _original_limits.items():
                setattr(app_config, key, value)
                logger.warning(f"[Fast Mode] Restored {key} = {value}")
            logger.warning("[Handle Submit] ⚡ Fast Mode limits RESTORED to normal")
