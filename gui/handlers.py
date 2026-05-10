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
    - Streaming hides thinking via has_incomplete_thinking_block() (tags) + likely_untagged_thinking() (heuristic)
    - Post-content ProgressEvents suppressed to prevent response pop-back
    - Citations extracted from memory_id_map via _extract_citations()
  - UNCERTAINTY FALLBACK [NEW 2026-04-27]: After standard streaming, UncertaintyDetector checks response for "I don't know" signals (keyword regex + semantic embedding). If uncertain + agentic enabled → silent retry via agentic search. Retry only accepted if word overlap with original < 70%. No progress messages or chunk streaming during retry.
  - POST-ANSWER REVIEW GATE: Checks response against ResponsePlan. Silent retry if confidence >= 0.90 (raised from 0.80). Skipped for responses < 120 chars. Same similarity guard (overlap < 70%).
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

        # CLIP-embed uploaded images for visual memory retrieval
        try:
            from config.app_config import VISUAL_MEMORY_ENABLED, VISUAL_MEMORY_INGEST_ON_UPLOAD
            if VISUAL_MEMORY_ENABLED and VISUAL_MEMORY_INGEST_ON_UPLOAD:
                from knowledge.clip_manager import get_clip_manager
                from knowledge.visual_memory_store import VisualMemoryStore
                from knowledge.visual_memory_pipeline import VisualMemoryPipeline

                clip = get_clip_manager()
                chroma = getattr(orchestrator, 'memory_coordinator', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                store = VisualMemoryStore(chroma_store=chroma_store)
                model_mgr = getattr(orchestrator, 'model_manager', None)
                resolver = getattr(chroma, 'entity_resolver', None) if chroma else None
                pipeline = VisualMemoryPipeline(clip, store, model_manager=model_mgr, entity_resolver=resolver)

                for img in files_result.images:
                    if not img.error and img.file_path:
                        try:
                            await pipeline.ingest_image(
                                img.file_path, source="upload", media_type=img.media_type or ""
                            )
                        except Exception as e:
                            logger.warning(f"[PERSIST] Visual memory ingest failed for {img.filename}: {e}")
        except ImportError:
            pass  # Visual memory deps not installed

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

    # Use merged_input (user text + file contents) so file content appears in the prompt.
    # Still pass files for context pipeline's file_context detection (has_files flag).
    prepare_task = asyncio.create_task(orchestrator.prepare_prompt(
        user_input=merged_input,
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
        # Allow knowledge + memory combinations (e.g., "tell me about my notes on physics")
        if len(_words) >= 4 and not needs_computation:
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
        _search_words = {'search', 'look', 'find', 'news', 'latest', 'current', 'today', 'recent', '2026', '2025', 'what is', 'who is', 'how does', 'tell me about'}
        _has_search_signal = any(w in _lower for w in _search_words) or '?' in user_text
        _skip_patterns = [
            len(_words) < 5 and not _has_search_signal,
            len(_words) < 10 and not _has_search_signal,
            # Casual starters only skip for SHORT messages without search signals
            len(_words) < 12 and not _has_search_signal and _lower.startswith((
                'nice', 'thanks', 'thank you', 'cool', 'great', 'awesome', 'got it',
                'ok ', 'okay', 'yeah', 'yes', 'no ', 'nope', 'nah', 'haha', 'lol',
                'true', 'fair', 'same', 'right', 'exactly', 'for sure', 'bet', 'word',
            )),
            all(w in ['yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope', 'thanks', 'thank', 'you', 'lol', 'haha', 'true', 'right', 'fair', 'same'] for w in _words),
        ]
        if not needs_computation and not needs_memory and not needs_knowledge and any(_skip_patterns):
            logger.debug("[Handle Submit] Agentic skipped - casual/short message")
            should_use_agentic = False
            search_terms = []
        else:

            if needs_computation or needs_memory or needs_knowledge:
                should_use_agentic = True
                search_terms = []  # No initial web search needed, tools handle it

                # Log all triggered modes (can be multiple)
                triggered_modes = []
                if needs_computation:
                    triggered_modes.append("computation")
                if needs_memory:
                    triggered_modes.append("memory")
                if needs_knowledge:
                    triggered_modes.append("knowledge")
                logger.debug(f"[Handle Submit] Agentic triggered - modes: {', '.join(triggered_modes)}")
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

                # Keepalive wrapper: if the agentic loop stalls for >8s without
                # yielding (e.g. waiting on a slow LLM API call mid-stream), emit
                # a heartbeat progress message so the browser WebSocket stays alive
                # and the final response is actually delivered to the UI.
                _agentic_gen = agentic_controller.run_agentic_search(
                    query=user_text,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    initial_search_terms=initial_terms,
                    initial_context=raw_context,
                    skip_initial_search=needs_computation or needs_memory or needs_knowledge,
                )

                async def _agentic_next():
                    try:
                        return await _agentic_gen.__anext__(), False
                    except StopAsyncIteration:
                        return None, True

                _KEEPALIVE_S = 8.0
                _keepalive_n = 0

                while True:
                    _task = asyncio.ensure_future(_agentic_next())
                    while True:
                        _done, _ = await asyncio.wait({_task}, timeout=_KEEPALIVE_S)
                        if _done:
                            break
                        _keepalive_n += 1
                        _elapsed = int(_keepalive_n * _KEEPALIVE_S)
                        yield {"role": "assistant", "content": f"🔄 Processing... ({_elapsed}s)", "is_progress": True}
                    item, _exhausted = _task.result()
                    if _exhausted:
                        break
                    if isinstance(item, ProgressEvent):
                        # Don't overwrite streamed response with late progress events
                        if agentic_response:
                            logger.debug(f"[Handle Submit] Skipping post-content progress: {item.event_type}")
                            continue
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
                        elif ResponseParser.likely_untagged_thinking(agentic_response):
                            # Heuristic: suppress untagged thinking during streaming
                            yield {"role": "assistant", "content": "💭 **Thinking...**", "is_thinking": True}
                        else:
                            # Strip any completed thinking block before display
                            thinking_detected, clean_answer = ResponseParser.parse_thinking_block(agentic_response)
                            # Only use clean_answer if non-empty; if thinking was detected but
                            # answer is empty, show indicator instead of falling back to raw
                            if thinking_detected and not clean_answer:
                                yield {"role": "assistant", "content": "💭 **Thinking...**", "is_thinking": True}
                            else:
                                yield {"role": "assistant", "content": clean_answer or agentic_response}

                # Final output from agentic search - strip thinking blocks
                final_output = agentic_response
                thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)
                display_output = final_answer if final_answer else final_output
                display_output = ResponseParser.strip_thinking_tag_leaks(display_output)

                # Append web sources footer if [WEB_N] citations present
                _web_map = getattr(agentic_controller, '_current_web_source_map', None) or {}
                if _web_map:
                    import re as _re
                    _cited_ids = set(_re.findall(r'\[WEB_(\d+)\]', display_output))
                    if _cited_ids:
                        _footer_lines = []
                        for _n in sorted(_cited_ids, key=int):
                            _key = f"WEB_{_n}"
                            _src = _web_map.get(_key)
                            if _src:
                                _footer_lines.append(f"[{_key}] [{_src.get('title', '')}]({_src.get('url', '')})")
                        if _footer_lines:
                            display_output += "\n\n---\n**Sources:**\n" + "\n".join(_footer_lines)
                    # Also set on orchestrator for provenance
                    orchestrator._web_source_map = _web_map

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
                # Yield final response with debug record (response was already streamed
                # chunk-by-chunk during the loop, so only one yield needed here)
                logger.debug(f"[Handle Submit] Agentic yielding final response: {display_output[:100]}...")
                yield {"role": "assistant", "content": display_output, "debug": debug_record}
                logger.debug("[Handle Submit] Agentic final response yielded")

                # Store interaction in background (fire-and-forget, same as enhanced path).
                # Avoids a ~5s blocking await after the final yield that kept the
                # Gradio generator open and could prevent the response from rendering.
                try:
                    from core.prompt import _truncate_at_spurious_turns
                    final_output_sanitized = _truncate_at_spurious_turns(final_output)
                except Exception as e:
                    logger.warning(f"[Handle Submit] Failed to sanitize agentic response: {e}")
                    final_output_sanitized = final_output

                tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                _store_task = asyncio.create_task(_background_store_interaction(
                    orchestrator=orchestrator,
                    merged_input=merged_input,
                    response_to_store=final_output_sanitized,
                    tags=tags,
                    user_text=user_text,
                    final_output=final_output,
                    personality=personality,
                    file_names=file_names,
                    conversation_logger=conversation_logger,
                    session_id=_agentic_session_id,
                    provenance=_agentic_prov,
                    mode='agentic-search',
                ))
                _pending_storage_tasks.add(_store_task)
                _store_task.add_done_callback(_pending_storage_tasks.discard)
                logger.info("[Handle Submit] Agentic storage dispatched to background")

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
                # Suppress untagged thinking that parse_thinking_block couldn't split yet.
                # Fire when heuristic detects thinking patterns and we haven't already
                # found/completed a tagged thinking block.
                _heuristic_thinks = (
                    not thinking_complete
                    and ResponseParser.likely_untagged_thinking(final_output)
                )
                if _heuristic_thinks:
                    display_output = "💭 **Thinking...**"
                    yield {"role": "assistant", "content": display_output, "is_thinking": True}
                else:
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
            "[MODEL NOT SUPPORTED]": "🚫 **Unsupported Input**\n\n{msg}\n\nTry switching to a multimodal model (e.g. GPT-4o, Claude) in the dropdown above.",
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
            # Sync display_output so final yield doesn't show stale thinking-polluted content
            display_output = final_output

        # ── Uncertainty Fallback: retry via agentic search if response is uncertain ──
        _uncertainty_retry_done = False
        if agentic_enabled and final_output:
            try:
                from config.app_config import (
                    UNCERTAINTY_FALLBACK_ENABLED,
                    UNCERTAINTY_SEMANTIC_THRESHOLD,
                    UNCERTAINTY_MAX_LENGTH,
                )
                if UNCERTAINTY_FALLBACK_ENABLED:
                    from core.uncertainty_detector import UncertaintyDetector

                    _uf_embedder = getattr(
                        getattr(orchestrator, 'model_manager', None), 'embed_model', None
                    )

                    _uf_result = UncertaintyDetector.detect(
                        response=final_output,
                        embedder=_uf_embedder,
                        semantic_threshold=UNCERTAINTY_SEMANTIC_THRESHOLD,
                        max_length=UNCERTAINTY_MAX_LENGTH,
                    )

                    if _uf_result.is_uncertain:
                        logger.warning(
                            f"[UNCERTAINTY FALLBACK] Detected uncertain response "
                            f"(trigger={_uf_result.trigger_type}, "
                            f"conf={_uf_result.confidence:.2f}, "
                            f"pattern={_uf_result.matched_pattern}). "
                            f"Retrying via agentic search."
                        )

                        # Retry silently — don't show progress, only swap if result is different
                        try:
                            from core.agentic import ProgressEvent

                            _retry_agentic = orchestrator.agentic_controller
                            _retry_hint = (
                                f'[MEMORY SEARCH RETRY] The user asked: "{user_text}" '
                                f"and the initial response could not find relevant "
                                f"information from context. Search memory deeply using "
                                f"the search_memory tool across conversations, "
                                f"summaries, and obsidian_notes collections. The "
                                f"information may exist but was not retrieved in the "
                                f"initial pass."
                            )
                            _retry_system = _retry_hint + "\n\n" + (system_prompt or "")

                            _retry_response = ""
                            async for item in _retry_agentic.run_agentic_search(
                                query=user_text,
                                system_prompt=_retry_system,
                                model_name=model_name,
                                initial_search_terms=[],
                                initial_context=raw_context,
                                skip_initial_search=True,
                            ):
                                if isinstance(item, ProgressEvent):
                                    pass  # Don't show retry progress to user
                                else:
                                    _retry_response += item

                            if _retry_response.strip():
                                _think_retry, _answer_retry = (
                                    ResponseParser.parse_thinking_block(_retry_response)
                                )
                                _retry_clean = (
                                    _answer_retry if _answer_retry else _retry_response
                                )
                                # Only replace if meaningfully different from original
                                _orig_words = set(final_output.lower().split())
                                _retry_words = set(_retry_clean.lower().split())
                                _overlap = len(_orig_words & _retry_words) / max(len(_orig_words | _retry_words), 1)
                                if _overlap < 0.7:
                                    final_output = _retry_clean
                                    display_output = final_output
                                    thinking_part_stream = (
                                        _think_retry or thinking_part_stream
                                    )
                                    _uncertainty_retry_done = True
                                    logger.info(
                                        f"[UNCERTAINTY FALLBACK] Agentic retry accepted "
                                        f"({len(final_output)} chars, overlap={_overlap:.2f})"
                                    )
                                else:
                                    logger.info(
                                        f"[UNCERTAINTY FALLBACK] Retry too similar "
                                        f"(overlap={_overlap:.2f}), keeping original"
                                    )
                            else:
                                logger.warning(
                                    "[UNCERTAINTY FALLBACK] Agentic retry returned "
                                    "empty, keeping original response"
                                )

                        except Exception as e:
                            logger.error(
                                f"[UNCERTAINTY FALLBACK] Agentic retry failed: {e}"
                            )
                            import traceback
                            logger.debug(
                                f"[UNCERTAINTY FALLBACK] Traceback:\n"
                                f"{traceback.format_exc()}"
                            )

            except ImportError as e:
                logger.debug(f"[UNCERTAINTY FALLBACK] Module not available: {e}")
            except Exception as e:
                logger.warning(
                    f"[UNCERTAINTY FALLBACK] Detection failed (non-fatal): {e}"
                )

        # ── Post-Answer Review Gate: check response against plan ──
        _review_retry_done = False
        # Skip review for short responses (casual/brief answers don't need quality gating)
        _review_min_len = 120
        if agentic_enabled and final_output and not _uncertainty_retry_done and len(final_output) >= _review_min_len:
            try:
                from config.app_config import (
                    RESPONSE_REVIEW_ENABLED,
                    RESPONSE_REVIEW_CONFIDENCE_THRESHOLD,
                )
                if RESPONSE_REVIEW_ENABLED:
                    _plan = getattr(orchestrator, '_current_response_plan', None)
                    _planner = getattr(orchestrator, 'response_planner', None)
                    if _plan is not None and _planner is not None:
                        _review = await _planner.review_answer(
                            plan=_plan,
                            response=final_output,
                            query=user_text,
                        )
                        if (
                            _review
                            and not _review.passes
                            and _review.confidence >= RESPONSE_REVIEW_CONFIDENCE_THRESHOLD
                        ):
                            logger.warning(
                                f"[REVIEW GATE] Response failed review "
                                f"(confidence={_review.confidence:.2f}, "
                                f"issues={_review.issues}). Retrying via agentic."
                            )

                            # Retry silently — don't show progress, only swap if result is different
                            try:
                                from core.agentic import ProgressEvent

                                _review_agentic = orchestrator.agentic_controller
                                _review_hint = (
                                    f'[RESPONSE REVIEW RETRY] The user asked: "{user_text}" '
                                    f"The initial response had these issues: "
                                    f"{'; '.join(_review.issues)}. "
                                    f"Suggestion: {_review.suggestion}. "
                                    f"Search memory and provide a better answer."
                                )
                                _review_system = _review_hint + "\n\n" + (system_prompt or "")

                                _review_response = ""
                                async for item in _review_agentic.run_agentic_search(
                                    query=user_text,
                                    system_prompt=_review_system,
                                    model_name=model_name,
                                    initial_search_terms=[],
                                    initial_context=raw_context,
                                    skip_initial_search=True,
                                ):
                                    if isinstance(item, ProgressEvent):
                                        pass  # Don't show retry progress to user
                                    else:
                                        _review_response += item

                                if _review_response.strip():
                                    _think_review, _answer_review = (
                                        ResponseParser.parse_thinking_block(_review_response)
                                    )
                                    _review_clean = (
                                        _answer_review if _answer_review else _review_response
                                    )
                                    # Only replace if meaningfully different from original
                                    _orig_words = set(final_output.lower().split())
                                    _rev_words = set(_review_clean.lower().split())
                                    _overlap = len(_orig_words & _rev_words) / max(len(_orig_words | _rev_words), 1)
                                    if _overlap < 0.7:
                                        final_output = _review_clean
                                        display_output = final_output
                                        thinking_part_stream = (
                                            _think_review or thinking_part_stream
                                        )
                                        _review_retry_done = True
                                        logger.info(
                                            f"[REVIEW GATE] Agentic retry accepted "
                                            f"({len(final_output)} chars, overlap={_overlap:.2f})"
                                        )
                                    else:
                                        logger.info(
                                            f"[REVIEW GATE] Retry too similar "
                                            f"(overlap={_overlap:.2f}), keeping original"
                                        )
                                else:
                                    logger.warning(
                                        "[REVIEW GATE] Agentic retry returned "
                                        "empty, keeping original response"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"[REVIEW GATE] Agentic retry failed: {e}"
                                )
                                import traceback
                                logger.debug(
                                    f"[REVIEW GATE] Traceback:\n"
                                    f"{traceback.format_exc()}"
                                )
                        elif _review:
                            logger.debug(
                                f"[REVIEW GATE] Response passed review "
                                f"(confidence={_review.confidence:.2f})"
                            )

            except ImportError as e:
                logger.debug(f"[REVIEW GATE] Module not available: {e}")
            except Exception as e:
                logger.warning(
                    f"[REVIEW GATE] Review failed (non-fatal): {e}"
                )

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

        # Final safety net: strip any thinking that leaked through all layers
        _think_final, _answer_final = ResponseParser.parse_thinking_block(_resp_for_debug)
        if _think_final and _answer_final:
            _resp_for_debug = _answer_final
        _resp_for_debug = ResponseParser.strip_thinking_tag_leaks(_resp_for_debug)

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
        _enh_mode = "uncertainty-fallback" if _uncertainty_retry_done else "enhanced"
        _enh_prov = {
            "response_mode": _enh_mode,
            "session_id": _enh_session_id or "",
            "model_name": model_name,
            "thinking_block": thinking_part_stream or "",
            "cited_ids": [c['memory_id'] for c in citations] if citations else [],
        }
        if _uncertainty_retry_done:
            try:
                _ac = getattr(orchestrator, 'agentic_controller', None)
                _last = getattr(_ac, '_last_session', None) if _ac else None
                if _last and hasattr(_last, 'get_provenance_summary'):
                    _ap = _last.get_provenance_summary()
                    _enh_prov["agentic_rounds"] = _ap.get("agentic_rounds", [])
                    _enh_prov["final_prompt_hash"] = _ap.get("final_prompt_hash", "")
            except Exception:
                pass
        debug_record = {
            'mode': _enh_mode,
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
