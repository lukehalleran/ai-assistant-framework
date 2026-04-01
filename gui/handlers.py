"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/duel/agentic/enhanced flows, streams the response to the UI, and persists interaction + provenance + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?, is_progress?} as Gradio updates.
- Behavior:
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - DUEL mode: If BEST_OF_DUEL_MODE enabled, two models compete + judge picks winner. Builds provenance with response_mode="best-of-duel". Runs BEFORE agentic check.
  - AGENTIC: If agentic_search.enabled and query triggers search, computation, memory recall, OR knowledge search, route through AgenticSearchController
    - 4-tier agentic gate [ENHANCED 2026-03-31]:
      - Tier 1: Keyword heuristic (instant) — computation keywords OR memory keywords ("do you remember", "my notes", etc.)
      - Tier 1b: Knowledge keywords (instant) [NEW 2026-03-31] — encyclopedic/wiki intent ("explain in depth", "how does", "consult wikipedia", etc.), 4+ words, no computation/memory trigger
      - Tier 2: Entity match (instant) — query mentions a known entity from the knowledge graph (e.g., "Flapjack", "Auggie")
      - Tier 3: LLM fallback — piggybacks on web search trigger LLM call; returns needs_memory_search and needs_knowledge_search fields
    - Casual skip filter only applies when no keyword/entity/knowledge trigger fired
    - skip_initial_search=True for computation, memory, and knowledge queries (skips Round 1 web search)
    - Streaming hides incomplete thinking blocks via has_incomplete_thinking_block()
    - Citations extracted from memory_id_map via _extract_citations()
  - ENHANCED: orchestrator.prepare_prompt → extract note_images → response_generator.generate_streaming_response(images=...) → store interaction
  - IMAGE SUPPORT [NEW 2026-01-30]: Extracts note_images from raw_context and passes to streaming for multimodal models
  - API error classification: [CREDITS EXHAUSTED], [RATE LIMITED], [AUTH ERROR], [MODEL NOT FOUND], [SERVER ERROR] with user-friendly messages
- Provenance [NEW 2026-03-26]:
  - All 5 response modes build provenance dicts (response_mode, model_name, thinking_block, cited_ids, prompt_hash, agentic_summary)
  - _background_store_interaction() accepts session_id, provenance, mode params and forwards to memory system
- Side effects:
  - Writes to conversation logger; stores to memory_system (with provenance metadata); updates debug_state for Debug Trace tab.
"""
import asyncio
import logging
from core.response_parser import ResponseParser
from utils.logging_utils import log_and_time
from utils.conversation_logger import get_conversation_logger
from utils.file_processor import FileProcessor, ProcessedFilesResult
import json
from config.app_config import load_system_prompt
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
    conversation_logger,
    session_id: str = None,
    provenance: dict = None,
    mode: str = "enhanced",
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
            tags=tags,
            session_id=session_id,
            provenance=provenance,
        )
        logger.info(f"[HANDLE_SUBMIT] Background storage complete, ID: {memory_id}")

        # Log conversation with db_id
        log_metadata = {
            'mode': mode,
            'files': file_names if file_names else None,
            'personality': personality or "default",
            'topic': getattr(orchestrator, 'current_topic', None),
            'db_id': memory_id,
        }
        if provenance:
            log_metadata['provenance'] = provenance
        conversation_logger.log_interaction(
            user_input=user_text,
            assistant_response=final_output,
            metadata=log_metadata,
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
                'personality': personality or "default",
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

    # ── DUEL MODE: Two models + judge, takes priority over agentic ──
    _cfg_duel = getattr(orchestrator, 'config', {}) or {}
    _features_duel = _cfg_duel.get('features', {}) if isinstance(_cfg_duel, dict) else {}
    _DUEL_ON = bool(_features_duel.get('best_of_duel_mode', False))
    _DUEL_GENS = list(_features_duel.get('best_of_generator_models', []))
    _DUEL_SELS = list(_features_duel.get('best_of_selector_models', []))
    duel_active = bool(_DUEL_ON and len(_DUEL_GENS) >= 2 and len(_DUEL_SELS) >= 1)
    logger.info(f"[Handle Submit] Duel check: on={_DUEL_ON}, gens={_DUEL_GENS}, sels={_DUEL_SELS}, active={duel_active}")

    if duel_active:
        logger.warning(f"[Handle Submit] DUEL MODE — {_DUEL_GENS[0]} vs {_DUEL_GENS[1]}, judge={_DUEL_SELS[0]}")
        yield {"role": "assistant", "content": "⚖️ Duel mode — generating two responses...", "is_progress": True}

        try:
            # Read temps from config
            try:
                from config.app_config import BEST_OF_TEMPS, BEST_OF_MAX_TOKENS, BEST_OF_SELECTOR_MAX_TOKENS
                _duel_temps = tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else (0.2, 0.7)
                _duel_max_tok = int(BEST_OF_MAX_TOKENS)
                _duel_judge_tok = int(BEST_OF_SELECTOR_MAX_TOKENS)
            except (ImportError, TypeError, ValueError):
                _duel_temps = (0.2, 0.7)
                _duel_max_tok = 512
                _duel_judge_tok = 64

            # Read latency budget
            try:
                from config.app_config import BEST_OF_LATENCY_BUDGET_S
                _duel_budget = float(_features_duel.get('best_of_latency_budget_s', BEST_OF_LATENCY_BUDGET_S))
            except (ImportError, TypeError, ValueError):
                _duel_budget = 0.0

            m1, m2 = _DUEL_GENS[0], _DUEL_GENS[1]
            judge = _DUEL_SELS[0]

            duel_coro = orchestrator.response_generator.generate_duel_and_judge(
                prompt=full_prompt,
                model_a=m1,
                model_b=m2,
                judge_model=judge,
                system_prompt=system_prompt,
                question_text=user_text,
                context_hint=full_prompt,
                max_tokens=_duel_max_tok,
                temperature_a=_duel_temps[0] if len(_duel_temps) > 0 else None,
                temperature_b=_duel_temps[1] if len(_duel_temps) > 1 else None,
                judge_max_tokens=_duel_judge_tok,
            )

            if _duel_budget > 0:
                best = await asyncio.wait_for(duel_coro, timeout=_duel_budget)
            else:
                best = await duel_coro

            # Unpack dict result from generate_duel_and_judge
            if isinstance(best, dict) and 'answer' in best:
                final_output = best['answer']
                display_output = final_output

                # Yield thinking data for GUI accordion
                thinking_data = {
                    'thinking_a': best.get('thinking_a', ''),
                    'thinking_b': best.get('thinking_b', ''),
                    'model_a': best.get('model_a', ''),
                    'model_b': best.get('model_b', ''),
                    'winner': best.get('winner', ''),
                    'scores': best.get('scores', {}),
                }
                logger.info(f"[DUEL] Winner: Model {thinking_data['winner']}, scores={thinking_data['scores']}")
                yield {"role": "assistant", "content": "", "thinking": thinking_data}
            else:
                final_output = str(best)
                _, final_answer = ResponseParser.parse_thinking_block(final_output)
                display_output = final_answer if final_answer else final_output

            # Token counts
            model_name = orchestrator.model_manager.get_active_model_name()
            try:
                tm = getattr(orchestrator, 'tokenizer_manager', None)
                prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
                system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
                total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
            except (AttributeError, TypeError, ValueError):
                prompt_tokens = len(full_prompt) // 4 if full_prompt else None
                system_tokens = len(system_prompt or '') // 4
                total_tokens = (prompt_tokens or 0) + (system_tokens or 0)

            # Citations
            citations = []
            if getattr(orchestrator, 'enable_citations', False):
                try:
                    memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                    if memory_id_map:
                        _, citations = orchestrator._extract_citations(final_output, memory_id_map)
                except (AttributeError, KeyError) as e:
                    logger.warning(f"[DUEL] Failed to extract citations: {e}")

            # Provenance
            _duel_session_id = getattr(orchestrator.memory_system, 'session_id', None) if hasattr(orchestrator, 'memory_system') else None
            _duel_prov = {
                "response_mode": "best-of-duel",
                "session_id": _duel_session_id or "",
                "model_name": f"{m1} vs {m2}",
                "cited_ids": [c['memory_id'] for c in citations] if citations else [],
            }
            if isinstance(best, dict):
                _duel_prov["thinking_a"] = best.get('thinking_a', '')
                _duel_prov["thinking_b"] = best.get('thinking_b', '')
                _duel_prov["model_a"] = best.get('model_a', '')
                _duel_prov["model_b"] = best.get('model_b', '')
                _duel_prov["winner"] = best.get('winner', '')

            debug_record = {
                'mode': 'best-of-duel',
                'query': user_text,
                'prompt': full_prompt,
                'system_prompt': system_prompt,
                'response': final_output,
                'model': f"{m1} vs {m2}",
                'prompt_tokens': prompt_tokens,
                'system_tokens': system_tokens,
                'total_tokens': total_tokens,
                'citations': citations,
                'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                'provenance': _duel_prov,
            }

            # Yield the response
            yield {"role": "assistant", "content": display_output, "debug": debug_record}

            # Store interaction in background
            tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
            task = asyncio.create_task(_background_store_interaction(
                orchestrator=orchestrator,
                merged_input=merged_input,
                response_to_store=final_output,
                tags=tags,
                user_text=user_text,
                final_output=final_output,
                personality=personality,
                file_names=file_names,
                conversation_logger=conversation_logger,
                session_id=_duel_session_id,
                provenance=_duel_prov,
                mode='best-of-duel',
            ))
            _pending_storage_tasks.add(task)
            task.add_done_callback(_pending_storage_tasks.discard)

            return  # Done — duel mode complete

        except asyncio.TimeoutError:
            logger.warning(f"[DUEL] Timed out after {_duel_budget}s, falling back to streaming")
        except Exception as e:
            logger.error(f"[DUEL] Failed, falling back to standard: {e}")
            import traceback
            logger.debug(f"[DUEL] Traceback:\n{traceback.format_exc()}")
        # Fall through to agentic/streaming on failure

    if agentic_enabled:
        _lower = user_text.lower().strip()
        _words = _lower.split()
        needs_computation = False
        needs_memory = False
        needs_knowledge = False
        _matched_entities = set()

        # --- Tier 1: Keyword heuristics (instant, no LLM) ---
        # Only unambiguous computation terms — avoid common English words
        # that trigger false positives (e.g. "I mean", "expand on that", "plot of the movie")
        _computation_keywords = [
            'calculate', 'compute', 'solve', 'integral', 'derivative', 'equation',
            'fibonacci', 'factorial', 'median', 'standard deviation',
            'matrix', 'numpy', 'pandas', 'sympy',
            'regression', 'correlation', 'sum of', 'product of',
            'simplify', 'differentiate', 'integrate',
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

        # Knowledge/wiki keywords: explicit wiki references or in-depth knowledge requests
        _knowledge_keywords = [
            'wikipedia', 'consult wikipedia', 'wiki ',
            'explain in depth', 'explain in detail', 'in depth',
            'how does ', 'how do ', 'what is the difference between',
            'compare and contrast', 'tell me about ',
            'what is a ', 'what are ', 'what causes ',
            'history of ', 'science behind', 'mechanism of ',
        ]
        # Only trigger for substantive queries (4+ words), not casual one-liners
        if len(_words) >= 4 and not needs_computation and not needs_memory:
            needs_knowledge = any(kw in _lower for kw in _knowledge_keywords)

        # --- Tier 2: Entity match (instant, no LLM) ---
        # If query mentions a known entity AND has a recall/question signal,
        # treat as memory query. Entity mention alone (casual chat) is not enough.
        if not needs_computation and not needs_memory:
            try:
                _resolver = getattr(getattr(orchestrator, 'memory_system', None), 'entity_resolver', None)
                if _resolver:
                    from memory.graph_utils import extract_graph_entities
                    _matched_entities = extract_graph_entities(user_text, _resolver)
                    _matched_entities.discard("user")  # "user" is not a meaningful match
                    if _matched_entities:
                        # Require a recall/question signal alongside entity mention
                        _has_recall_signal = (
                            '?' in user_text
                            or any(w in _lower for w in [
                                'what', 'when', 'where', 'who', 'how', 'why',
                                'tell me', 'remind', 'remember', 'know about',
                                'recall', 'anything about', 'details on',
                            ])
                        )
                        if _has_recall_signal:
                            needs_memory = True
                            logger.debug(f"[Handle Submit] Agentic triggered - entity {_matched_entities} + recall signal")
                        else:
                            logger.debug(f"[Handle Submit] Entity match {_matched_entities} but no recall signal — skipping agentic")
            except Exception as e:
                logger.debug(f"[Handle Submit] Entity match check failed (non-fatal): {e}")

        # Casual skip filter: only applies when no keyword/entity trigger fired
        _skip_patterns = [
            len(_words) < 5 and not any(w in _lower for w in ['search', 'news', 'latest', 'current', 'today', 'recent', '2026', '2025']),
            _lower.startswith(('nice', 'thanks', 'thank you', 'cool', 'great', 'awesome', 'got it', 'ok ', 'okay', 'yeah', 'yes', 'no ', 'nope', 'nah')),
            'working' in _lower and 'search' in _lower,  # meta-comments about search
            all(w in ['yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope', 'thanks', 'thank', 'you'] for w in _words),
        ]
        if not needs_computation and not needs_memory and not needs_knowledge and any(_skip_patterns):
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
            elif needs_knowledge:
                logger.debug("[Handle Submit] Agentic triggered - knowledge/wiki query detected via keywords")
                should_use_agentic = True
                search_terms = []  # No web search needed, search_memory(wiki_knowledge) handles it
            else:
                # Check if web search, memory search, or knowledge search should be triggered using LLM-first trigger
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

                    # LLM-based knowledge search fallback: encyclopedic/wiki queries
                    if not should_use_agentic and getattr(trigger_decision, 'needs_knowledge_search', False):
                        logger.debug("[Handle Submit] Agentic triggered - LLM detected knowledge search intent")
                        should_use_agentic = True
                        needs_knowledge = True  # skip initial web search
                        search_terms = []

                    logger.debug(f"[Handle Submit] Agentic trigger: should_search={should_use_agentic}, needs_memory={needs_memory}, needs_knowledge={needs_knowledge}, terms={search_terms}")
                except Exception as e:
                    logger.warning(f"[Handle Submit] Agentic trigger check failed: {e}")
                    import traceback
                    traceback.print_exc()
                    should_use_agentic = False
                    search_terms = []

        # --- Intent-based veto: skip agentic for casual/meta queries even if keyword matched ---
        if should_use_agentic:
            _intent_info = raw_context.get("intent") if raw_context else None
            if _intent_info:
                _intent_type = getattr(_intent_info, 'intent_type', None) if not isinstance(_intent_info, dict) else _intent_info.get('intent_type')
                _intent_conf = getattr(_intent_info, 'confidence', 0) if not isinstance(_intent_info, dict) else _intent_info.get('confidence', 0)
                _veto_intents = {'meta_conversational', 'casual_social'}
                _type_val = getattr(_intent_type, 'value', str(_intent_type)) if _intent_type else ''
                if _type_val in _veto_intents and _intent_conf >= 0.75:
                    logger.info(f"[Handle Submit] Agentic VETOED by intent classifier: {_intent_type} (conf={_intent_conf:.2f})")
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
                    skip_initial_search=needs_computation or needs_memory or needs_knowledge,  # Skip web search for computation/memory/knowledge-only queries
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
                            "reading_file": "📄",
                            "file_read": "✅",
                            "searching_files": "🔎",
                            "files_searched": "✅",
                            "listing_files": "📂",
                            "files_listed": "✅",
                            "expanding_memory": "🧠",
                            "memory_expanded": "✅",
                            "synthesizing": "✨",
                            "done": "✅",
                            "error": "❌",
                        }.get(item.event_type, "•")
                        # Override display message for specific event types
                        _display_msg = {
                            "computing": "Computing...",
                            "executing_code": "Coding...",
                        }.get(item.event_type, item.message)
                        status_msg = f"{status_icon} {_display_msg}"
                        logger.debug(f"[Handle Submit] Agentic progress: {item.event_type}")
                        yield {"role": "assistant", "content": status_msg, "is_progress": True}
                    else:
                        # Response chunk - accumulate and stream
                        agentic_response += item
                        # Hide incomplete thinking blocks during streaming
                        if ResponseParser.has_incomplete_thinking_block(agentic_response):
                            yield {"role": "assistant", "content": "💭 **Thinking...**", "is_thinking": True}
                        else:
                            # Strip any completed thinking block before display
                            _, clean_answer = ResponseParser.parse_thinking_block(agentic_response)
                            yield {"role": "assistant", "content": clean_answer or agentic_response}

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
                    # Fallback: rough estimate (~4 chars per token)
                    prompt_tokens = len(full_prompt) // 4 if full_prompt else None
                    system_tokens = len(system_prompt or '') // 4
                    total_tokens = (prompt_tokens or 0) + (system_tokens or 0)

                # Extract citations if enabled (instead of hardcoding [])
                citations = []
                if getattr(orchestrator, 'enable_citations', False):
                    try:
                        memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                        if memory_id_map:
                            _, citations = orchestrator._extract_citations(final_output, memory_id_map)
                    except (AttributeError, KeyError) as e:
                        logger.warning(f"[CITATIONS] Failed to extract agentic citations: {e}")

                # Build agentic provenance
                _agentic_session_id = getattr(orchestrator.memory_system, 'session_id', None) if hasattr(orchestrator, 'memory_system') else None
                _agentic_prov = {
                    "response_mode": "agentic-search",
                    "session_id": _agentic_session_id or "",
                    "model_name": model_name,
                    "thinking_block": thinking_part or "",
                    "cited_ids": [c['memory_id'] for c in citations] if citations else [],
                }
                # Attach agentic session details if available
                try:
                    _ac = getattr(orchestrator, 'agentic_controller', None)
                    _last = getattr(_ac, '_last_session', None) if _ac else None
                    if _last and hasattr(_last, 'get_provenance_summary'):
                        _ap = _last.get_provenance_summary()
                        _agentic_prov["agentic_rounds"] = _ap.get("agentic_rounds", [])
                        _agentic_prov["final_prompt_hash"] = _ap.get("final_prompt_hash", "")
                except Exception as e:
                    logger.debug(f"[Handlers] Could not get agentic provenance: {e}")

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
                    'citations': citations,
                    'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                    'provenance': _agentic_prov,
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
                        tags=tags,
                        session_id=_agentic_session_id,
                        provenance=_agentic_prov,
                    )
                    logger.info(f"[Handle Submit] Agentic interaction stored with ID: {memory_id}")

                    # Log conversation
                    conversation_logger.log_interaction(
                        user_input=user_text,
                        assistant_response=final_output,
                        metadata={
                            'mode': 'agentic-search',
                            'files': file_names if file_names else None,
                            'personality': personality or "default",
                            'topic': getattr(orchestrator, 'current_topic', None),
                            'db_id': memory_id,
                            'provenance': _agentic_prov,
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

        # Duel mode is handled above (before agentic check). This path is streaming only.
        model_name = orchestrator.model_manager.get_active_model_name()

        # Streaming path (duel mode handled above, old best-of code removed)
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

            # Detect incomplete thinking block (opening tag arrived, closing hasn't yet)
            if ResponseParser.has_incomplete_thinking_block(final_output):
                thinking_started = True
                display_output = "💭 **Thinking...**"
                yield {"role": "assistant", "content": display_output, "is_thinking": True}
                continue

            # Parse in real-time to separate thinking from answer
            thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)

            # If we have thinking content and haven't shown the answer yet
            if thinking_part and not final_answer:
                # Still in thinking block — show indicator only (don't leak content)
                thinking_started = True
                display_output = "💭 **Thinking...**"
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

        # Detect classified API errors from model_manager
        _stripped = final_output.strip()
        _API_ERROR_PREFIXES = {
            "[CREDITS EXHAUSTED]": "💳 **Out of API Credits**\n\n{msg}\n\nYou can add credits at your provider's billing page or switch models in the dropdown above.",
            "[RATE LIMITED]": "⏳ **Rate Limited**\n\n{msg}",
            "[AUTH ERROR]": "🔑 **Authentication Error**\n\n{msg}",
            "[MODEL NOT FOUND]": "❓ **Model Not Found**\n\n{msg}",
            "[SERVER ERROR]": "🔥 **Server Error**\n\n{msg}",
            "[API Error]": "⚠️ **API Error**\n\n{msg}",
            "[API unavailable]": "⚠️ **API Unavailable**\n\n{msg}",
        }
        for prefix, template in _API_ERROR_PREFIXES.items():
            if _stripped.startswith(prefix):
                friendly = template.format(msg=_stripped[len(prefix):].strip())
                logger.warning(f"[Handle Submit] API error detected: {prefix}")
                yield {"role": "assistant", "content": friendly}
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

        _enh_session_id = getattr(orchestrator.memory_system, 'session_id', None) if hasattr(orchestrator, 'memory_system') else None
        _enh_prov = {
            "response_mode": "enhanced",
            "session_id": _enh_session_id or "",
            "model_name": model_name,
            "thinking_block": thinking_part_stream or "",
            "cited_ids": [c['memory_id'] for c in citations] if citations else [],
        }
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
            'provenance': _enh_prov,
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

                # Build provenance from the debug_record emitted during streaming
                _store_prov = None
                _store_mode = "enhanced"
                _store_session_id = getattr(orchestrator.memory_system, 'session_id', None) if hasattr(orchestrator, 'memory_system') else None
                if debug_emitted and 'debug_record' in dir():
                    try:
                        _store_prov = debug_record.get('provenance') if isinstance(debug_record, dict) else None
                        _store_mode = debug_record.get('mode', 'enhanced') if isinstance(debug_record, dict) else 'enhanced'
                    except Exception:
                        pass
                if _store_prov is None:
                    _store_prov = {
                        "response_mode": _store_mode,
                        "model_name": model_name if 'model_name' in dir() else "",
                        "thinking_block": thinking_part or "" if 'thinking_part' in dir() else "",
                    }

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
                    conversation_logger=conversation_logger,
                    session_id=_store_session_id,
                    provenance=_store_prov,
                    mode=_store_mode,
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
