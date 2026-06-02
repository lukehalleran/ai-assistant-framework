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
  - AGENTIC: If agentic_search.enabled, delegates to core/agentic/gate.py:evaluate_agentic_gate()
    which runs the 4-tier gate (keyword → entity → doc/note intent → LLM fallback) and returns
    an AgenticDecision. If triggered, routes through AgenticSearchController.
    Uses merged_input (user text + file content) as query so uploaded file content is visible to the agentic loop.
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
- Structure [REFACTORED 2026-05-30]: handle_submit is now a thin (~150-line) async-generator
  dispatcher. It builds a SubmitContext (threaded state) and routes to per-mode handler
  generators, each of which yields the same chunk shapes the old inline blocks did and signals
  completion via ctx.handled (the dispatcher returns when set, else falls through):
  - _prepare_submit_context(ctx): shared prelude (fast-mode limits, prepare_prompt keepalive,
    image inject) for all non-raw paths.
  - _run_raw / _run_duel / _run_agentic_search / _run_enhanced: the 4 mutually-exclusive parent
    modes. _run_duel and _run_agentic_search leave ctx.handled False on bail to fall through.
  - _run_doc_generation / _run_self_note: agentic-gate bypasses (do their own store_interaction).
  - _run_enhanced owns the post-answer passes (uncertainty fallback, review gate) and the
    finally cleanup (fast-mode restore + storage). Its finally is enhanced-path-only by design
    (see "latent fast-mode-restore" note below) — do NOT hoist it to the dispatcher.
- Extracted helpers:
  - _safe_count_tokens(), _safe_extract_citations(), _build_debug_record(), _build_provenance(),
    _attach_agentic_provenance(), _sanitize_response_text(), _strip_echoed_headers(),
    _dispatch_storage(), _silent_agentic_retry(), _get_session_id(), _find_email_draft()
  - _strip_inline_tool_xml(text, full=): consolidates the leaked tool-call XML stripping
    (5-pattern full set; 3-pattern subset for the enhanced lookup_contact site).
  - _make_text_action_proposal(decision, store): shared propose+audit for text tool-calls.
  - _resolve_contact_and_propose_email(...): shared contact resolution + auto-email proposal
    for the agentic and enhanced lookup_contact paths (no_contacts_suffix keeps each path's
    exact not-found wording).
- KNOWN latent bug (preserved, NOT fixed here): under fast_mode, the duel/doc-gen/self-note/
  agentic-success paths return before the enhanced finally, so fast-mode flags + _original_limits
  are never restored on those paths. Pinned by test_fast_mode_agentic_leaves_flag_set.
- Side effects:
  - Writes to conversation logger; stores to memory_system (with provenance metadata); updates debug_state for Debug Trace tab.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
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


import re as _re

# Regex to strip leaked XML tool-call markers from enhanced-mode LLM output.
# Covers both agentic-style markers (<search>, <memory>, etc.) and hallucinated
# variants the LLM may produce (<search_memory>, <web_search>, etc.).
# Top-level agentic tool tag names (the outer wrappers the LLM emits)
_AGENTIC_OUTER_TAGS = (
    r'search|memory|wolfram|python|expand_memory|get_full_document|git_stats|github|'
    r'recall_image|search_memory|web_search|fetch_url|tool_call|function_call|'
    r'file_read|file_grep|file_list|done'
)
# All tool-related tags including inner ones (query, action, collection, etc.)
_ALL_TOOL_TAGS = _AGENTIC_OUTER_TAGS + r'|action|query|collection|limit'

# Pattern 1: Strip opening/closing tags only (preserves content between them)
_LEAKED_XML_TOOL_RE = _re.compile(
    rf'</?(?:{_ALL_TOOL_TAGS})(?:\s[^>]*)?>',
    _re.IGNORECASE
)
# Pattern 2: Strip entire tool blocks — matches <tag>...</tag> for each outer tool name
# Uses [\s\S] instead of . for newline crossing
_LEAKED_XML_TOOL_BLOCK_RE = _re.compile(
    rf'<({_AGENTIC_OUTER_TAGS})(?:\s[^>]*)?>'
    rf'[\s\S]*?'
    rf'</\1>',
    _re.IGNORECASE
)
# Pattern 3: Self-closing tags like <done/>, <file_list ... />
_LEAKED_XML_SELF_CLOSING_RE = _re.compile(
    rf'<(?:{_ALL_TOOL_TAGS})\s*/?>',
    _re.IGNORECASE
)


def _strip_leaked_xml_markers(text: str) -> str:
    """Remove leaked XML tool-call markers from enhanced-mode output.

    Used during streaming to strip tags but preserve surrounding text.
    """
    cleaned = _LEAKED_XML_TOOL_RE.sub('', text)
    cleaned = _LEAKED_XML_SELF_CLOSING_RE.sub('', cleaned)
    # Collapse runs of blank lines left after stripping
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _strip_leaked_xml_blocks(text: str) -> str:
    """Remove entire leaked XML tool blocks (tags + content) from non-agentic responses.

    More aggressive than _strip_leaked_xml_markers — removes everything between
    opening and closing tool tags. Used on final output when response was not
    generated in agentic mode.
    """
    if not text or '<' not in text:
        return text
    cleaned = _LEAKED_XML_TOOL_BLOCK_RE.sub('', text)
    cleaned = _LEAKED_XML_SELF_CLOSING_RE.sub('', cleaned)
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


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


# ── Extracted helpers for handle_submit ──────────────────────────────
# These reduce repetition across the 6 mode paths (raw, duel, agentic,
# enhanced, uncertainty fallback, review gate) without changing behavior.


def _get_session_id(orchestrator) -> str:
    """Get the current memory session ID, or empty string."""
    try:
        return getattr(orchestrator.memory_system, 'session_id', None) or ""
    except AttributeError:
        return ""


def _safe_count_tokens(prompt, system_prompt, model_name, orchestrator):
    """Count tokens for prompt and system_prompt.

    Returns (prompt_tokens, system_tokens, total_tokens).
    Falls back to char//4 estimate on failure.
    """
    try:
        tm = getattr(orchestrator, 'tokenizer_manager', None)
        if tm:
            p = int(tm.count_tokens(prompt, model_name))
            s = int(tm.count_tokens(system_prompt or '', model_name))
            return p, s, p + s
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"[Handlers] Token counting failed: {e}")
    p = len(prompt) // 4 if prompt else 0
    s = len(system_prompt or '') // 4
    return p, s, p + s


def _safe_extract_citations(response_text, orchestrator):
    """Extract memory citations from response if enabled.

    Returns (possibly_modified_response, citations_list).
    """
    if not getattr(orchestrator, 'enable_citations', False):
        return response_text, []
    try:
        memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
        if memory_id_map:
            modified, citations = orchestrator._extract_citations(
                response_text, memory_id_map,
            )
            return modified, citations
    except (AttributeError, KeyError) as e:
        logger.warning(f"[CITATIONS] Failed to extract citations: {e}")
    return response_text, []


def _build_provenance(mode, session_id, model_name, citations,
                      thinking_block="", **extra):
    """Build a provenance dict for any response mode."""
    prov = {
        "response_mode": mode,
        "session_id": session_id or "",
        "model_name": model_name,
        "cited_ids": [c['memory_id'] for c in citations] if citations else [],
    }
    if thinking_block:
        prov["thinking_block"] = thinking_block
    prov.update(extra)
    return prov


def _attach_agentic_provenance(provenance, orchestrator):
    """Attach agentic session details (rounds, prompt hash) to provenance."""
    try:
        ac = getattr(orchestrator, 'agentic_controller', None)
        last = getattr(ac, '_last_session', None) if ac else None
        if last and hasattr(last, 'get_provenance_summary'):
            ap = last.get_provenance_summary()
            provenance["agentic_rounds"] = ap.get("agentic_rounds", [])
            provenance["final_prompt_hash"] = ap.get("final_prompt_hash", "")
    except Exception as e:
        logger.debug(f"[Handlers] Could not get agentic provenance: {e}")


def _build_debug_record(
    mode, user_text, prompt, system_prompt, response, model,
    prompt_tokens, system_tokens, total_tokens,
    citations, orchestrator, provenance=None,
    phase_timings=None, task_timings=None, gather_elapsed=0.0,
):
    """Build a debug record dict for the Debug Trace tab."""
    return {
        'mode': mode,
        'query': user_text,
        'prompt': prompt,
        'system_prompt': system_prompt,
        'response': response,
        'model': model,
        'prompt_tokens': prompt_tokens,
        'system_tokens': system_tokens,
        'total_tokens': total_tokens,
        'citations': citations,
        'citations_enabled': getattr(orchestrator, 'enable_citations', False),
        'provenance': provenance,
        'phase_timings': phase_timings or {},
        'task_timings': (
            {k: round(v, 3) for k, v in task_timings.items()}
            if task_timings else {}
        ),
        'gather_elapsed': round(gather_elapsed, 3) if gather_elapsed else 0.0,
    }


def _find_email_draft(chat_history: list, fallback: str) -> str:
    """Search chat history for the most recent email draft content.

    Looks backward through assistant messages for substantial content that
    looks like an email draft (bullet points, summaries, multiple lines).
    Returns None if no suitable draft is found — callers should NOT
    auto-send with meta-commentary as the body.
    """
    if not chat_history:
        return None

    # Search recent assistant messages (last 10) for draft-like content
    assistant_msgs = []
    for msg in reversed(chat_history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and len(content) > 100:
                assistant_msgs.append(content)
        if len(assistant_msgs) >= 10:
            break

    for content in assistant_msgs:
        # Strip XML artifacts before checking
        import re as _re_draft
        clean = _re_draft.sub(r'<function_calls>.*?</function_calls>', '', content, flags=_re_draft.DOTALL)
        clean = _re_draft.sub(r'<function_calls>.*$', '', clean, flags=_re_draft.DOTALL)
        clean = _re_draft.sub(r'<invoke\s[^>]*>.*?</invoke>', '', clean, flags=_re_draft.DOTALL)
        clean = _re_draft.sub(r'<thinking>.*?</thinking>', '', clean, flags=_re_draft.DOTALL)
        clean = clean.strip()

        # Look for draft-like indicators: bullet points, multiple paragraphs, summary-like content
        has_bullets = '- **' in clean or '- ' in clean
        has_length = len(clean) > 200
        has_structure = clean.count('\n') >= 3

        if has_bullets and has_length and has_structure:
            # Found a draft — extract just the substantive content
            # Remove meta-commentary lines (first line is often "Let me..." or "Here's...")
            lines = clean.split('\n')
            # Find where the actual content starts (first bullet or substantial line)
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('- ') or line.strip().startswith('* '):
                    start_idx = i
                    break
            draft = '\n'.join(lines[start_idx:]).strip()
            if len(draft) > 100:
                return draft

    return None


def _sanitize_response_text(text):
    """Core response sanitization: thinking blocks, XML leaks, spurious turns."""
    if not text:
        return ""
    thinking, answer = ResponseParser.parse_thinking_block(text)
    if thinking and answer:
        text = answer
    elif thinking and not answer:
        return ""
    text = ResponseParser.strip_thinking_tag_leaks(text)
    text = _strip_leaked_xml_blocks(text)
    try:
        from core.prompt import _truncate_at_spurious_turns
        text = _truncate_at_spurious_turns(text)
    except Exception:
        pass
    return text


# Compiled regex for stripping echoed prompt headers from stored responses.
_ECHOED_HEADER_RE = _re.compile(
    r"(" + r")|(".join([
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
    ]) + r")",
    _re.IGNORECASE,
)


def _strip_echoed_headers(text):
    """Remove echoed prompt section headers from response text (for storage)."""
    if not text:
        return text
    lines = []
    skip = False
    for line in text.splitlines():
        if _ECHOED_HEADER_RE.search(line):
            skip = True
            continue
        if skip:
            if not line.strip():
                skip = False
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _dispatch_storage(
    orchestrator, merged_input, response_to_store, user_text,
    final_output, personality, file_names, conversation_logger,
    session_id, provenance, mode,
):
    """Create a background storage task and track it for graceful shutdown."""
    tags = [
        f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}",
        "topic:general",
    ]
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
        session_id=session_id,
        provenance=provenance,
        mode=mode,
    ))
    _pending_storage_tasks.add(task)
    task.add_done_callback(_pending_storage_tasks.discard)
    return task


async def _silent_agentic_retry(
    orchestrator, merged_input, system_prompt, model_name,
    raw_context, original_response, hint, log_prefix,
):
    """Run a silent agentic retry and compare against the original response.

    Returns (clean_response, thinking) if accepted (overlap < 0.7),
    else (None, None).
    """
    try:
        from core.agentic import ProgressEvent

        agentic = orchestrator.agentic_controller
        retry_system = hint + "\n\n" + (system_prompt or "")

        retry_response = ""
        async for item in agentic.run_agentic_search(
            query=merged_input,
            system_prompt=retry_system,
            model_name=model_name,
            initial_search_terms=[],
            initial_context=raw_context,
            skip_initial_search=True,
        ):
            if isinstance(item, ProgressEvent):
                pass
            else:
                retry_response += item

        if not retry_response.strip():
            logger.warning(
                f"[{log_prefix}] Agentic retry returned empty, keeping original"
            )
            return None, None

        think_retry, answer_retry = ResponseParser.parse_thinking_block(
            retry_response,
        )
        retry_clean = answer_retry if answer_retry else retry_response

        orig_words = set(original_response.lower().split())
        retry_words = set(retry_clean.lower().split())
        overlap = len(orig_words & retry_words) / max(
            len(orig_words | retry_words), 1,
        )

        if overlap < 0.7:
            logger.info(
                f"[{log_prefix}] Agentic retry accepted "
                f"({len(retry_clean)} chars, overlap={overlap:.2f})"
            )
            return retry_clean, think_retry
        else:
            logger.info(
                f"[{log_prefix}] Retry too similar "
                f"(overlap={overlap:.2f}), keeping original"
            )
            return None, None

    except Exception as e:
        logger.error(f"[{log_prefix}] Agentic retry failed: {e}")
        import traceback
        logger.debug(f"[{log_prefix}] Traceback:\n{traceback.format_exc()}")
        return None, None


# Tool-call XML blocks the model sometimes leaks into its final answer. Shared by the
# agentic + enhanced display-cleanup sites. The full set (5 patterns) is used at the
# final-answer and text-action sites; the 3-pattern subset is used at the enhanced
# lookup_contact site (which historically stripped only function_calls + invoke).
_TOOL_XML_STRIP_PATTERNS = [
    (_re.compile(r'<function_calls>.*?</function_calls>', _re.DOTALL), ''),
    (_re.compile(r'<function_calls>.*$', _re.DOTALL), ''),
    (_re.compile(r'<invoke\s[^>]*>.*?</invoke>', _re.DOTALL), ''),
    (_re.compile(r'<propose_action[^>]*>.*?</propose_action>', _re.DOTALL), ''),
    (_re.compile(r'<lookup_contact[^>]*>.*?</lookup_contact>', _re.DOTALL), ''),
]


def _strip_inline_tool_xml(text, *, full=True):
    """Strip leaked tool-call XML blocks from a display string.

    ``full=True`` strips function_calls/invoke/propose_action/lookup_contact (the 5-pattern
    set used at the agentic + enhanced final-answer and text-action sites). ``full=False``
    strips only function_calls(complete+unclosed)/invoke (the 3-pattern subset used at the
    enhanced lookup_contact site). Mirrors the per-substitution ``.strip()`` of the inline
    originals exactly.
    """
    patterns = _TOOL_XML_STRIP_PATTERNS if full else _TOOL_XML_STRIP_PATTERNS[:3]
    for pat, repl in patterns:
        text = pat.sub(repl, text).strip()
    return text


def _make_text_action_proposal(decision, store):
    """Create + audit an ActionProposal from a parsed text tool-call decision.

    Shared by the agentic and enhanced text-action paths (the ActionType guard + propose +
    audit block). Returns the new ``action_id``, or ``None`` if the action type is unknown
    (caller should ``break`` in that case, matching the originals).
    """
    from core.actions.types import ActionProposal, ActionType
    from core.actions.audit import ActionAuditLog
    from config.app_config import INTERNET_ACTIONS_AUDIT_LOG
    try:
        action_type = ActionType(decision.action_type)
    except ValueError:
        logger.warning(f"[Handle Submit] Unknown action type from text: {decision.action_type}")
        return None
    proposal = ActionProposal(
        action_type=action_type,
        params=decision.action_params or {},
        summary=decision.action_summary or f"{decision.action_type}: action",
        reasoning=decision.action_reason or "",
    )
    store.propose(proposal)
    ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG).log_proposal(proposal)
    return proposal.action_id


async def _resolve_contact_and_propose_email(
    contact_name, user_text, history, display_text, store, *, no_contacts_suffix="",
):
    """Resolve a contact name and, when email intent is present, auto-create a send_email
    proposal. Shared by the agentic and enhanced lookup_contact paths.

    Appends a contact / proposal card (or a not-found message) to ``display_text`` and
    returns ``(updated_display_text, action_id_or_None)``. The ``no_contacts_suffix`` lets
    each caller keep its exact not-found wording (agentic: ""; enhanced: " in Google
    Contacts or Gmail"). Callers do their own XML stripping and own the surrounding
    try/except, matching the originals.
    """
    from core.actions.google_contacts import resolve_contact

    contacts = await resolve_contact(contact_name, max_results=5)
    action_id = None
    if contacts:
        email = contacts[0]['email']
        name = contacts[0]['name']
        alt = ""
        if len(contacts) > 1:
            alts = [f"{c['name']} <{c['email']}>" for c in contacts[1:]]
            alt = f"\n*(Also found: {', '.join(alts)})*"
        email_intent = any(w in user_text.lower() for w in (
            'send', 'email', 'mail', 'draft', 'fire', 'message', 'try',
        ))
        if email_intent:
            from core.actions.types import ActionProposal, ActionType
            from core.actions.audit import ActionAuditLog
            from config.app_config import INTERNET_ACTIONS_AUDIT_LOG
            body = _find_email_draft(history, display_text)
            if body:
                proposal = ActionProposal(
                    action_type=ActionType.SEND_EMAIL,
                    params={
                        "recipient": email,
                        "message": body,
                        "subject": "Weekly Summary",
                    },
                    summary=f"send_email to {name} <{email}>",
                    reasoning=f"Resolved '{contact_name}' via contact search",
                )
                store.propose(proposal)
                ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG).log_proposal(proposal)
                action_id = proposal.action_id
                card = f"\n\n---\n**send_email** to {name} <{email}>\n"
                card += f"> {body[:300]}\n\n"
                if alt:
                    card += alt + "\n"
                display_text += card
                logger.info(f"[Handle Submit] Auto-created send_email proposal to {email}")
            else:
                display_text += (
                    f"\n\n**Contact found:** {name} <{email}>{alt}\n\nI found the email "
                    f"address but couldn't locate the draft in this session. Could you "
                    f"paste or describe what you'd like to send?"
                )
        else:
            display_text += f"\n\n**Contact found:** {name} <{email}>{alt}"
    else:
        display_text += f"\n\nNo contacts found for '{contact_name}'{no_contacts_suffix}."
    return display_text, action_id


@dataclass
class SubmitContext:
    """Threaded state for a single handle_submit() turn, passed to the per-mode handlers.

    Built once in the dispatcher; the prelude (_prepare_submit_context) fills the prompt
    fields and the agentic gate fills the routing fields. Mode handlers read from it and
    set the control signals (handled / storage_dispatched).
    """
    # --- immutable inputs ---
    user_text: str
    files: Any
    history: Any
    use_raw_gpt: bool
    orchestrator: Any
    personality: Any
    fast_mode: bool
    conversation_logger: Any
    file_names: list
    merged_input: str
    files_result: Any
    agentic_enabled: bool = False
    # --- set by _prepare_submit_context ---
    full_prompt: str = ""
    system_prompt: str = ""
    raw_context: dict = field(default_factory=dict)
    note_images: list = field(default_factory=list)
    original_limits: dict = field(default_factory=dict)
    t_prepare_start: float = 0.0
    t_prepare_elapsed: float = 0.0
    # --- set after the agentic gate (evaluate_agentic_gate) ---
    gate_decision: Any = None
    should_use_agentic: bool = False
    search_terms: list = field(default_factory=list)
    doc_gen_intent: Any = None
    self_note_intent: Any = None
    skip_initial_search: bool = False
    # --- control signals set by mode handlers ---
    handled: bool = False
    storage_dispatched: bool = False


async def _prepare_submit_context(ctx):
    """Enhanced-path prelude: apply Fast Mode limits, run prepare_prompt (yielding keepalive
    progress), then extract + inject multimodal images. Mutates ``ctx`` with
    full_prompt / system_prompt / raw_context / note_images / original_limits / prepare timings.

    Shared by the duel / agentic / enhanced paths (everything except raw mode). Yields the
    same progress chunks the inline prelude did, in the same order.
    """
    orchestrator = ctx.orchestrator

    # Send immediate progress to prevent mobile timeout during prompt preparation
    yield {"role": "assistant", "content": "💭 Thinking...", "is_progress": True}

    logger.info("[Handle Submit] >>> Starting prepare_prompt...")

    # Apply Fast Mode limits BEFORE prepare_prompt starts
    ctx.original_limits = {}
    if ctx.fast_mode:
        logger.warning("[Handle Submit] ⚡⚡⚡ FAST MODE ENABLED ⚡⚡⚡")
        import core.prompt.builder as builder_module
        # Override builder module constants (the REAL location of these limits)
        ctx.original_limits['PROMPT_MAX_MEMS'] = builder_module.PROMPT_MAX_MEMS
        logger.warning(f"[Fast Mode] PROMPT_MAX_MEMS: {builder_module.PROMPT_MAX_MEMS} → 10")
        builder_module.PROMPT_MAX_MEMS = 10

        ctx.original_limits['PROMPT_MAX_RECENT'] = builder_module.PROMPT_MAX_RECENT
        logger.warning(f"[Fast Mode] PROMPT_MAX_RECENT: {builder_module.PROMPT_MAX_RECENT} → 5")
        builder_module.PROMPT_MAX_RECENT = 5

        if hasattr(builder_module, 'PROMPT_MAX_SEMANTIC'):
            ctx.original_limits['PROMPT_MAX_SEMANTIC'] = builder_module.PROMPT_MAX_SEMANTIC
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
    import time as _time_mod
    ctx.t_prepare_start = _time_mod.perf_counter()
    prepare_task = asyncio.create_task(orchestrator.prepare_prompt(
        user_input=ctx.merged_input,
        files=ctx.files,
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
    ctx.t_prepare_elapsed = _time_mod.perf_counter() - ctx.t_prepare_start

    # Unpack result - always expect 3 values now
    full_prompt, system_prompt, raw_context = prep_result
    raw_context = raw_context or {}

    # Extract images for multimodal models
    note_images = raw_context.get("note_images", [])
    if note_images:
        logger.warning(f"[Handle Submit] Extracted {len(note_images)} images from raw_context for multimodal generation")

    # Inject uploaded images into note_images for immediate multimodal use
    if ctx.files_result.images:
        for img in ctx.files_result.images:
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
        logger.warning(f"[Handle Submit] Injected {len(ctx.files_result.images)} upload images, total note_images={len(note_images)}")

    ctx.full_prompt = full_prompt
    ctx.system_prompt = system_prompt
    ctx.raw_context = raw_context
    ctx.note_images = note_images

    logger.info(f"[Handle Submit] <<< prepare_prompt done, prompt_len={len(full_prompt)}")
    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")
    logger.debug(f"[Handle Submit] Agentic pre-check: enabled={ctx.agentic_enabled}")


async def _run_raw(ctx):
    """RAW mode: bypass memory + prompt building, stream a one-shot response.

    Yields a progress chunk then the final chunk (with debug record). Always sets
    ctx.handled (raw always services the request).
    """
    orchestrator = ctx.orchestrator
    logger.info("[Handle Submit] RAW MODE ENABLED – skipping memory and prompt building.")

    # Send immediate progress to prevent mobile timeout
    yield {"role": "assistant", "content": "💭 Processing...", "is_progress": True}

    response_text, debug_info = await orchestrator.process_user_query(
        user_input=ctx.merged_input,
        files=None,
        use_raw_mode=True,
        personality=ctx.personality
    )

    # Log the raw mode conversation
    ctx.conversation_logger.log_interaction(
        user_input=ctx.user_text,  # Log original input without file content for clarity
        assistant_response=response_text,
        metadata={
            'mode': 'raw',
            'files': ctx.file_names if ctx.file_names else None,
            'personality': ctx.personality or "default",
        }
    )

    # Emit final chunk including debug record for UI tracing
    _raw_model = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
    _raw_ptok, _, _raw_ttok = _safe_count_tokens(ctx.merged_input, None, _raw_model, orchestrator)
    debug_record = _build_debug_record(
        mode='raw', user_text=ctx.user_text, prompt=ctx.merged_input,
        system_prompt=None, response=response_text, model=_raw_model,
        prompt_tokens=_raw_ptok, system_tokens=0, total_tokens=_raw_ttok,
        citations=[], orchestrator=orchestrator,
    )
    yield {"role": "assistant", "content": response_text, "debug": debug_record}
    ctx.handled = True


async def _run_duel(ctx, gens, sels, features_duel):
    """DUEL mode: two generator models compete, a judge picks the winner.

    Yields a progress chunk (before the try, so it is emitted even on failure), optional
    duel-thinking, and the final chunk; dispatches storage and sets ctx.handled +
    ctx.storage_dispatched on success. On asyncio.TimeoutError or any Exception it logs and
    returns with ctx.handled still False, so the dispatcher falls through to agentic/enhanced.
    """
    orchestrator = ctx.orchestrator
    logger.warning(f"[Handle Submit] DUEL MODE — {gens[0]} vs {gens[1]}, judge={sels[0]}")
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
            _duel_budget = float(features_duel.get('best_of_latency_budget_s', BEST_OF_LATENCY_BUDGET_S))
        except (ImportError, TypeError, ValueError):
            _duel_budget = 0.0

        m1, m2 = gens[0], gens[1]
        judge = sels[0]

        duel_coro = orchestrator.response_generator.generate_duel_and_judge(
            prompt=ctx.full_prompt,
            model_a=m1,
            model_b=m2,
            judge_model=judge,
            system_prompt=ctx.system_prompt,
            question_text=ctx.user_text,
            context_hint=ctx.full_prompt,
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

        # Token counts, citations, provenance, debug record
        model_name = orchestrator.model_manager.get_active_model_name()
        prompt_tokens, system_tokens, total_tokens = _safe_count_tokens(
            ctx.full_prompt, ctx.system_prompt, model_name, orchestrator,
        )
        _, citations = _safe_extract_citations(final_output, orchestrator)

        _duel_session_id = _get_session_id(orchestrator)
        _duel_extra = {}
        if isinstance(best, dict):
            for _dk in ('thinking_a', 'thinking_b', 'model_a', 'model_b', 'winner'):
                _duel_extra[_dk] = best.get(_dk, '')
        _duel_prov = _build_provenance(
            "best-of-duel", _duel_session_id, f"{m1} vs {m2}",
            citations, **_duel_extra,
        )

        debug_record = _build_debug_record(
            mode='best-of-duel', user_text=ctx.user_text, prompt=ctx.full_prompt,
            system_prompt=ctx.system_prompt, response=final_output,
            model=f"{m1} vs {m2}", prompt_tokens=prompt_tokens,
            system_tokens=system_tokens, total_tokens=total_tokens,
            citations=citations, orchestrator=orchestrator,
            provenance=_duel_prov,
        )

        yield {"role": "assistant", "content": display_output, "debug": debug_record}

        _dispatch_storage(
            orchestrator, ctx.merged_input, final_output, ctx.user_text,
            final_output, ctx.personality, ctx.file_names, ctx.conversation_logger,
            _duel_session_id, _duel_prov, 'best-of-duel',
        )

        ctx.handled = True
        ctx.storage_dispatched = True
        return  # Done — duel mode complete

    except asyncio.TimeoutError:
        logger.warning(f"[DUEL] Timed out after {_duel_budget}s, falling back to streaming")
    except Exception as e:
        logger.error(f"[DUEL] Failed, falling back to standard: {e}")
        import traceback
        logger.debug(f"[DUEL] Traceback:\n{traceback.format_exc()}")
    # Fall through to agentic/streaming on failure (ctx.handled stays False)


async def _run_doc_generation(ctx):
    """Direct document-generation bypass (agentic gate doc_gen_intent).

    Yields a progress chunk + the result chunk; does its own store_interaction and sets
    ctx.handled on success. On exception, logs and returns with ctx.handled False so the
    dispatcher falls through to the agentic-search path.
    """
    orchestrator = ctx.orchestrator
    _doc_gen_intent = ctx.doc_gen_intent
    logger.warning(f"[Handle Submit] DIRECT DOCUMENT GENERATION: {_doc_gen_intent}")
    try:
        from knowledge.document_generator import DocumentGenerator

        # Resolve web_search_manager: same path the orchestrator uses
        _wsm = None
        _pb = getattr(orchestrator, 'prompt_builder', None)
        if _pb:
            _cg = getattr(_pb, 'context_gatherer', None)
            if _cg:
                _wsm = getattr(_cg, 'web_search_manager', None)

        # Resolve chroma_store
        _cs = None
        _ms = getattr(orchestrator, 'memory_system', None)
        if _ms:
            _cs = getattr(_ms, 'chroma_store', None)

        _dg = DocumentGenerator(
            model_manager=orchestrator.model_manager,
            web_search_manager=_wsm,
            chroma_store=_cs,
        )

        yield {"role": "assistant", "content": f"📝 Researching: {_doc_gen_intent['topic']}...", "is_progress": True}

        _doc_result = await _dg.generate(
            topic=_doc_gen_intent["topic"],
            doc_type=_doc_gen_intent["doc_type"],
            focus=_doc_gen_intent.get("focus"),
        )

        _doc_response = (
            f"Document saved: **{_doc_result.title}**\n\n"
            f"- **Path**: `{_doc_result.path}`\n"
            f"- **Type**: {_doc_result.doc_type}\n"
            f"- **Sources**: {len(_doc_result.sources)}\n"
            f"- **Sections**: {_doc_result.sections_count}\n"
            f"- **Words**: {_doc_result.word_count}\n"
        )
        logger.info(f"[Handle Submit] Document generated: {_doc_result.path}")

        # Store interaction
        if orchestrator.memory_system:
            try:
                await orchestrator.memory_system.store_interaction(
                    query=ctx.user_text,
                    response=_doc_response,
                    tags=["document_generation"],
                )
            except Exception:
                pass

        yield {"role": "assistant", "content": _doc_response}
        ctx.handled = True
        return

    except Exception as e:
        logger.error(f"[Handle Submit] Direct document generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall through to normal agentic/enhanced mode (ctx.handled stays False)


# ============================================================================
# Action guard: pending-proposal capture + claimed-action verification
# (anti-confabulation — see core/pending_proposal.py + core/action_claim_guard.py)
# ============================================================================


def _get_pending_proposal_store(orchestrator):
    """Lazily create + return the session-scoped pending-proposal store, or None."""
    try:
        from config.app_config import PENDING_PROPOSAL_ENABLED, PENDING_PROPOSAL_TTL_TURNS
        if not PENDING_PROPOSAL_ENABLED:
            return None
        store = getattr(orchestrator, "_pending_proposal_store", None)
        if store is None:
            from core.pending_proposal import PendingProposalStore
            store = PendingProposalStore(ttl_turns=PENDING_PROPOSAL_TTL_TURNS)
            orchestrator._pending_proposal_store = store
        return store
    except Exception as e:
        logger.debug(f"[ActionGuard] pending-proposal store unavailable: {e}")
        return None


def _summary_from_body(body: str) -> str:
    """A short (>=10 char) summary from a note body, or '' to defer to the LLM."""
    text = (body or "").strip()
    if not text:
        return ""
    snippet = " ".join(text.split())
    if len(snippet) > 240:
        snippet = snippet[:240].rsplit(" ", 1)[0] + "…"
    return snippet if len(snippet) >= 10 else ""


def _recent_conversation_text(orchestrator) -> str:
    """The last couple of assistant responses (e.g. a plan from a prior turn)."""
    try:
        cm = getattr(getattr(orchestrator, "memory_system", None), "corpus_manager", None)
        if cm and hasattr(cm, "get_recent_memories"):
            parts = []
            for r in cm.get_recent_memories(2) or []:
                resp = (r.get("response") or "").strip()
                if resp:
                    parts.append(resp)
            return "\n\n".join(parts)[:4000]
    except Exception as e:
        logger.debug(f"[ActionGuard] recent-conversation lookup failed: {e}")
    return ""


async def _save_daemon_note(ctx, *, title, body="", category="implementation", summary="", confidence="medium"):
    """Persist a daemon self-note + store_interaction; yields progress + result.

    Honest about partial saves: if the disk write succeeds but the ChromaDB embed
    or index update failed, the result message says so instead of claiming a clean
    save. Sets ctx.handled on success. Shared by _run_self_note, the affirmation
    follow-through, and claim self-repair.
    """
    orchestrator = ctx.orchestrator
    from knowledge.daemon_notes_manager import DaemonNotesManager

    _cs = getattr(getattr(orchestrator, "memory_system", None), "chroma_store", None)
    _dnm = DaemonNotesManager(model_manager=orchestrator.model_manager, chroma_store=_cs)

    title = (title or "").strip()[:100] or "Conversation note"
    yield {"role": "assistant", "content": f"🗒️ Saving note: {title}...", "is_progress": True}

    if not summary:
        summary = _summary_from_body(body) or await _dnm._generate_summary(title, orchestrator.model_manager)

    note = await _dnm.create_note(
        title=title, category=category, summary=summary, confidence=confidence,
        body=body or "",
    )

    _resp = (
        f"Self-note saved: **{note.title}**\n\n"
        f"- **Path**: `{note.path}`\n"
        f"- **Category**: {note.category}\n"
        f"- **ID**: {note.id}\n"
    )
    if not note.fully_persisted:
        _missing = []
        if not note.embedded:
            _missing.append("semantic search index")
        if not note.indexed:
            _missing.append("notes index")
        _resp += (
            f"\n> ⚠️ Saved to disk, but couldn't update the {', '.join(_missing)} — "
            f"it may not resurface automatically in future sessions."
        )
    logger.info(f"[ActionGuard] Note saved: {note.path} (fully_persisted={note.fully_persisted})")

    if orchestrator.memory_system:
        try:
            await orchestrator.memory_system.store_interaction(
                query=ctx.user_text, response=_resp, tags=["daemon_self_note"],
            )
        except Exception:
            pass

    yield {"role": "assistant", "content": _resp}
    ctx.handled = True


def _capture_proposal(orchestrator, response_text):
    """Detect a daemon-note OFFER in a response and stash it for the next turn.

    Only NOTE offers are captured here — external actions (email/calendar) already
    flow through the propose_action / PendingActionsStore card approval path.
    """
    try:
        from config.app_config import PENDING_PROPOSAL_ENABLED
        if not PENDING_PROPOSAL_ENABLED or not response_text:
            return
        from core.action_claim_guard import ActionKind, detect_proposals
        from core.pending_proposal import build_proposal_from_response
        props = [p for p in detect_proposals(response_text) if p.kind == ActionKind.NOTE]
        if not props:
            return
        store = _get_pending_proposal_store(orchestrator)
        if store is None:
            return
        proposal = build_proposal_from_response(
            response_text, props[-1], turn=store.turn,
            session_id=_get_session_id(orchestrator),
        )
        store.capture(proposal)
        logger.info(f"[ActionGuard] Captured pending note proposal: {proposal.title!r}")
    except Exception as e:
        logger.debug(f"[ActionGuard] Proposal capture failed (non-fatal): {e}")


async def _self_repair_note(ctx, detected):
    """Back an unbacked NOTE claim by actually saving a note. Returns DaemonNote|None."""
    orchestrator = ctx.orchestrator
    body, title, category = "", (detected.topic or ""), "implementation"

    store = _get_pending_proposal_store(orchestrator)
    if store is not None:
        from core.action_claim_guard import ActionKind
        p = store.peek()
        if p is not None and p.kind == ActionKind.NOTE:
            body, title, category = p.body, p.title, p.category
            store.clear()
    if not body:
        body = _recent_conversation_text(orchestrator)
    if not title:
        first = next((ln.strip() for ln in body.splitlines() if len(ln.strip()) >= 3), "")
        title = first[:80] or "Conversation note"

    try:
        from knowledge.daemon_notes_manager import DaemonNotesManager
        _cs = getattr(getattr(orchestrator, "memory_system", None), "chroma_store", None)
        _dnm = DaemonNotesManager(model_manager=orchestrator.model_manager, chroma_store=_cs)
        summary = _summary_from_body(body) or f"Auto-saved from conversation: {title}"
        note = await _dnm.create_note(
            title=title[:100], category=category, summary=summary,
            confidence="low", body=body or "",
        )
        logger.warning(f"[ActionGuard] Self-repaired unbacked note claim → {note.path}")
        return note
    except Exception as e:
        logger.error(f"[ActionGuard] Note self-repair failed: {e}")
        return None


async def _apply_action_guard(ctx, response_text, *, executed_kinds, proposed_kinds, self_repair):
    """Reconcile completion claims in a response against what actually ran.

    Always captures a fresh proposal for next turn. Returns a suffix string to
    append to the response: a confirmation when a note claim was self-repaired,
    and/or an honest correction when an external action was claimed but neither
    executed nor proposed. Never auto-executes external actions.
    """
    _capture_proposal(ctx.orchestrator, response_text)

    suffix = ""
    try:
        from config.app_config import ACTION_CLAIM_GUARD_ENABLED, ACTION_CLAIM_SELF_REPAIR_ENABLED
        if not ACTION_CLAIM_GUARD_ENABLED or not response_text:
            return suffix
        from core.action_claim_guard import (
            ActionKind, build_correction_notice, detect_completion_claims, verify_claims,
        )
        claims = detect_completion_claims(response_text)
        if not claims:
            return suffix
        rec = verify_claims(claims, executed_kinds=set(executed_kinds), proposed_kinds=set(proposed_kinds))
        if not rec.has_issue:
            return suffix

        if self_repair and ACTION_CLAIM_SELF_REPAIR_ENABLED:
            for a in rec.repairable:
                if a.kind == ActionKind.NOTE:
                    saved = await _self_repair_note(ctx, a)
                    if saved is not None:
                        suffix += f"\n\n> 🗒️ (I went ahead and actually saved that note: `{saved.path}`)"

        external = [a for a in rec.external_unbacked if a.kind not in set(proposed_kinds)]
        suffix += build_correction_notice(external)
    except Exception as e:
        logger.warning(f"[ActionGuard] Claim guard failed (non-fatal): {e}")
    return suffix


async def _run_self_note(ctx):
    """Direct daemon self-note bypass (agentic gate self_note_intent).

    Yields a progress chunk + the result chunk; does its own store_interaction and sets
    ctx.handled on success. On exception, logs and returns with ctx.handled False.
    """
    _self_note_intent = ctx.self_note_intent
    logger.warning(f"[Handle Submit] DIRECT SELF-NOTE CREATION: {_self_note_intent}")
    try:
        async for _c in _save_daemon_note(
            ctx,
            title=_self_note_intent["topic"],
            category=_self_note_intent.get("category", "implementation"),
        ):
            yield _c
        return
    except Exception as e:
        logger.error(f"[Handle Submit] Direct self-note creation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall through to normal agentic/enhanced mode (ctx.handled stays False)


async def _run_pending_proposal(ctx, proposal):
    """Execute a previously-captured action proposal after the user affirmed it.

    Currently handles NOTE proposals (the captured kind). Yields progress + result
    and sets ctx.handled on success; on failure, logs and leaves ctx.handled False
    so the dispatcher falls through to the normal flow.
    """
    from core.action_claim_guard import ActionKind
    logger.warning(
        f"[ActionGuard] Affirmation → executing pending {proposal.kind.value}: {proposal.title!r}"
    )
    try:
        if proposal.kind == ActionKind.NOTE:
            async for _c in _save_daemon_note(
                ctx, title=proposal.title, body=proposal.body, category=proposal.category,
            ):
                yield _c
            return
    except Exception as e:
        logger.error(f"[ActionGuard] Pending proposal execution failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall through to normal flow (ctx.handled stays False)


async def _run_agentic_search(ctx):
    """AGENTIC SEARCH mode: ReAct loop via the agentic controller.

    Yields keepalive/progress/streamed chunks then the final chunk (with optional
    pending_action_id); dispatches storage and sets ctx.handled + ctx.storage_dispatched
    on success. On exception it logs and returns with ctx.handled False, so the dispatcher
    falls through to enhanced streaming.
    """
    orchestrator = ctx.orchestrator
    _gate_decision = ctx.gate_decision
    full_prompt = ctx.full_prompt
    system_prompt = ctx.system_prompt
    raw_context = ctx.raw_context
    note_images = ctx.note_images
    search_terms = ctx.search_terms
    skip_initial_search = ctx.skip_initial_search
    merged_input = ctx.merged_input
    user_text = ctx.user_text
    history = ctx.history
    personality = ctx.personality
    file_names = ctx.file_names
    conversation_logger = ctx.conversation_logger
    _t_prepare_start = ctx.t_prepare_start
    _t_prepare_elapsed = ctx.t_prepare_elapsed
    import time as _time_mod
    logger.warning("[Handle Submit] AGENTIC SEARCH MODE - routing through agentic controller")
    try:
        from core.agentic import AgenticSearchController, ProgressEvent

        # Get the agentic controller from orchestrator
        agentic_controller = orchestrator.agentic_controller
        model_name = orchestrator.model_manager.get_active_model_name()

        # Get initial search terms from the trigger decision we already have
        initial_terms = search_terms if search_terms else []
        logger.debug(f"[Handle Submit] Agentic initial terms: {initial_terms}")

        # Extract URLs from the user message for direct fetch
        import re as _re_url
        _url_pattern = _re_url.compile(r'https?://[^\s<>"\')\]]+')
        _extracted_urls = _url_pattern.findall(user_text)

        # Run agentic search loop with RAG context
        agentic_response = ""
        logger.debug(f"[Handle Submit] Starting agentic loop with RAG context keys: {list(raw_context.keys())}")

        # Keepalive wrapper: if the agentic loop stalls for >8s without
        # yielding (e.g. waiting on a slow LLM API call mid-stream), emit
        # a heartbeat progress message so the browser WebSocket stays alive
        # and the final response is actually delivered to the UI.
        _agentic_gen = agentic_controller.run_agentic_search(
            query=merged_input,
            system_prompt=system_prompt,
            model_name=model_name,
            initial_search_terms=initial_terms,
            initial_context=raw_context,
            skip_initial_search=_gate_decision.skip_initial_search and not _extracted_urls,
            initial_urls=_extracted_urls if _extracted_urls else None,
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
                    "generating_document": "📝",
                    "document_generated": "✅",
                    "saving_note": "🗒️",
                    "note_saved": "✅",
                    "proposing_action": "📨",
                    "action_proposed": "✅",
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
                        _stream_display = _strip_leaked_xml_blocks(clean_answer or agentic_response)
                        yield {"role": "assistant", "content": _stream_display}

        # Final output from agentic search - strip thinking blocks
        final_output = agentic_response
        thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)
        # Also try untagged thinking detection
        if not thinking_part:
            untagged_thinking, untagged_answer = ResponseParser._detect_untagged_thinking(final_output)
            if untagged_thinking:
                thinking_part = untagged_thinking
                final_answer = untagged_answer
        display_output = final_answer if final_answer else final_output
        # If entire response was thinking (no answer), don't show it
        if thinking_part and not final_answer:
            display_output = ""
        display_output = ResponseParser.strip_thinking_tag_leaks(display_output)
        display_output = _strip_leaked_xml_blocks(display_output)

        # If agentic loop ran but no tools were actually dispatched (model
        # just narrated what it would do), strip bare tool-call-like lines
        # that leaked as plain text (e.g. "list_repos", "Lines added...")
        _agentic_session = getattr(
            getattr(orchestrator, '_agentic_controller', None),
            '_last_session', None
        )
        _had_real_rounds = (
            _agentic_session
            and hasattr(_agentic_session, 'rounds')
            and len(_agentic_session.rounds) > 0
        )
        if not _had_real_rounds and display_output:
            # Response is just narration — strip lines that look like
            # bare tool queries (short lines without sentence structure)
            # But preserve [propose_action] blocks for text parsing
            _cleaned_lines = []
            _in_action_block = False
            for _line in display_output.split('\n'):
                _stripped_line = _line.strip()
                # Track action JSON blocks — don't strip them
                if _stripped_line.startswith('[propose_action'):
                    _in_action_block = True
                if _in_action_block:
                    _cleaned_lines.append(_line)
                    if _stripped_line == '}':
                        _in_action_block = False
                    continue
                # Keep empty lines and lines with sentence structure
                if (not _stripped_line
                        or len(_stripped_line.split()) >= 4
                        or _stripped_line.endswith(('.', '!', '?', ':', ';', ','))
                        or _stripped_line.startswith(('#', '-', '*', '>', '{', '"'))):
                    _cleaned_lines.append(_line)
                else:
                    logger.debug(f"[Handle Submit] Stripped bare tool-call line: {_stripped_line!r}")
            display_output = '\n'.join(_cleaned_lines).strip()
            display_output = _re.sub(r'\n{3,}', '\n\n', display_output)

        # Append web sources footer if [WEB_N] citations present
        _web_map = getattr(agentic_controller, '_current_web_source_map', None) or {}
        if _web_map:
            import re as _re_cite
            _cited_ids = set(_re_cite.findall(r'\[WEB_(\d+)\]', display_output))
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

        # Parse text-based action proposals from the final response.
        # The model sometimes outputs [propose_action: send_email] {...}
        # as text in the final generation instead of calling the tool
        # during the agentic loop.
        try:
            from config.app_config import INTERNET_ACTIONS_ENABLED
            if INTERNET_ACTIONS_ENABLED and display_output:
                from core.agentic.tools import ToolExecutor
                _actions_store = ToolExecutor._get_pending_actions_store()
                if not _actions_store.get_pending():
                    # No action was proposed via tool call — check text
                    from core.agentic.protocols import NativeToolsHandler
                    _text_handler = NativeToolsHandler(actions_available=True)
                    _text_decisions = _text_handler._parse_text_tool_calls(display_output)
                    for _td in _text_decisions:
                        if _td.wants_action and _td.action_type:
                            logger.info(f"[Handle Submit] Parsed text action proposal: {_td.action_type}")
                            if _make_text_action_proposal(_td, _actions_store) is None:
                                break
                            # Strip the raw tool text + leaked XML blocks from display
                            import re as _re_action
                            display_output = _re_action.sub(
                                r'\[propose_action:\s*\w+\]\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}',
                                '', display_output, count=1,
                            ).strip()
                            display_output = _strip_inline_tool_xml(display_output)
                            break  # Only one action per turn
        except (ImportError, Exception) as e:
            logger.warning(f"[Handle Submit] Text action parse failed: {e}")

        # Check for pending action proposals → append card to display
        _pending_action_id = None
        try:
            from config.app_config import INTERNET_ACTIONS_ENABLED
            if INTERNET_ACTIONS_ENABLED:
                from core.agentic.tools import ToolExecutor
                _actions_store = ToolExecutor._get_pending_actions_store()
                _pending_proposal = _actions_store.get_pending()
                if _pending_proposal:
                    _pending_action_id = _pending_proposal.action_id
                    _recipient = _pending_proposal.params.get("recipient", "")
                    _subject = _pending_proposal.params.get("subject", "")
                    _msg = _pending_proposal.params.get("message", "")
                    _action_header = f"**{_pending_proposal.action_type.value}**"
                    if _recipient:
                        _action_header += f" to {_recipient}"
                    if _subject:
                        _action_header += f" — *{_subject}*"
                    _action_card = f"\n\n---\n{_action_header}\n"
                    if _msg:
                        _action_card += f"> {_msg[:300]}\n\n"
                    display_output += _action_card
        except ImportError:
            pass

        logger.debug(f"[Handle Submit] Agentic loop done, response_len={len(final_output)}, display_len={len(display_output)}")

        # Token counts, citations, provenance, debug record
        prompt_tokens, system_tokens, total_tokens = _safe_count_tokens(
            full_prompt, system_prompt, model_name, orchestrator,
        )
        _, citations = _safe_extract_citations(final_output, orchestrator)

        _agentic_session_id = _get_session_id(orchestrator)
        _agentic_prov = _build_provenance(
            "agentic-search", _agentic_session_id, model_name,
            citations, thinking_block=thinking_part or "",
        )
        _attach_agentic_provenance(_agentic_prov, orchestrator)

        _agentic_phase = getattr(orchestrator, '_last_phase_timings', {})
        _agentic_tasks = getattr(orchestrator, '_last_task_timings', {})
        _agentic_gather = getattr(orchestrator, '_last_gather_elapsed', 0.0)
        _agentic_handler_timings = {
            "prepare_prompt": round(_t_prepare_elapsed, 3),
            "agentic_loop": round(_time_mod.perf_counter() - _t_prepare_start - _t_prepare_elapsed, 3),
            "total_wall": round(_time_mod.perf_counter() - _t_prepare_start, 3),
        }
        if _agentic_phase:
            _agentic_handler_timings["context_pipeline"] = _agentic_phase.get("context_pipeline", 0.0)
            _agentic_handler_timings["prompt_build"] = _agentic_phase.get("prompt_build", 0.0)

        debug_record = _build_debug_record(
            mode='agentic-search', user_text=user_text, prompt=full_prompt,
            system_prompt=system_prompt, response=final_output,
            model=model_name, prompt_tokens=prompt_tokens,
            system_tokens=system_tokens, total_tokens=total_tokens,
            citations=citations, orchestrator=orchestrator,
            provenance=_agentic_prov,
            phase_timings=_agentic_handler_timings,
            task_timings=_agentic_tasks,
            gather_elapsed=_agentic_gather,
        )
        # Yield final response with debug record (response was already streamed
        # chunk-by-chunk during the loop, so only one yield needed here)
        # Strip any XML tool call artifacts the model emitted in its final answer
        display_output = _strip_inline_tool_xml(display_output)
        # If model emitted contact lookup in final answer, resolve inline + auto-propose
        if not _pending_action_id:
            try:
                from config.app_config import INTERNET_ACTIONS_ENABLED
                if INTERNET_ACTIONS_ENABLED:
                    from core.agentic.protocols import NativeToolsHandler
                    _ag_handler = NativeToolsHandler(actions_available=True)
                    _ag_decisions = _ag_handler._parse_text_tool_calls(final_output or display_output)
                    for _agd in _ag_decisions:
                        if _agd.wants_lookup_contact and _agd.lookup_contact_name:
                            from core.agentic.tools import ToolExecutor
                            _ag_store = ToolExecutor._get_pending_actions_store()
                            display_output, _ag_aid = await _resolve_contact_and_propose_email(
                                _agd.lookup_contact_name, user_text, history,
                                display_output, _ag_store,
                            )
                            if _ag_aid:
                                _pending_action_id = _ag_aid
                            break
            except Exception as _ag_err:
                logger.warning(f"[Handle Submit] Agentic contact resolution failed: {_ag_err}")

        # If display_output is empty (entire response was thinking/reasoning), show fallback
        if not display_output.strip():
            display_output = "I processed your request but my response was caught by the thinking filter. Let me try again — could you rephrase or retry?"
            logger.warning("[Handle Submit] Agentic response was entirely thinking content, showing fallback")

        # ── Action guard: capture note offers (e.g. "Want me to save this?") for
        # the next turn, and honestly correct external claims that weren't backed.
        # No note/doc self-repair here — the agentic loop may have genuinely run
        # those tools, so auto-saving would risk a duplicate. External actions are
        # human-in-the-loop (never auto-executed by the loop), so a bare "I sent
        # it" with no proposal is safe to correct.
        try:
            from core.action_claim_guard import EXTERNAL as _EXTERNAL_KINDS
            _ag_proposed = _EXTERNAL_KINDS if _pending_action_id else set()
            _ag_guard_suffix = await _apply_action_guard(
                ctx, display_output, executed_kinds=set(),
                proposed_kinds=_ag_proposed, self_repair=False,
            )
            if _ag_guard_suffix:
                display_output = display_output.rstrip() + _ag_guard_suffix
                final_output = (final_output or "").rstrip() + _ag_guard_suffix
        except Exception as _ag_guard_err:
            logger.warning(f"[Handle Submit] Agentic action guard failed (non-fatal): {_ag_guard_err}")

        logger.debug(f"[Handle Submit] Agentic yielding final response: {display_output[:100]}...")
        _final_chunk = {"role": "assistant", "content": display_output, "debug": debug_record}
        if _pending_action_id:
            _final_chunk["pending_action_id"] = _pending_action_id
        yield _final_chunk
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

        # Prevent context pollution: strip leaked XML/tool text
        final_output_sanitized = _strip_leaked_xml_blocks(final_output_sanitized)
        if len(final_output_sanitized.strip()) < 20 and display_output.strip():
            final_output_sanitized = display_output

        _dispatch_storage(
            orchestrator, merged_input, final_output_sanitized, user_text,
            final_output, personality, file_names, conversation_logger,
            _agentic_session_id, _agentic_prov, 'agentic-search',
        )
        logger.info("[Handle Submit] Agentic storage dispatched to background")

        ctx.handled = True
        ctx.storage_dispatched = True
        return  # Exit after agentic search completes

    except Exception as e:
        logger.error(f"[Handle Submit] Agentic search failed, falling back to standard: {e}")
        import traceback
        logger.debug(f"[Agentic] Exception traceback:\n{traceback.format_exc()}")


async def _run_enhanced(ctx):
    """ENHANCED (default) path: streaming generation + thinking detection, the
    post-answer passes (uncertainty fallback, review gate), action parsing, and the
    finally cleanup (fast-mode flag/limit restore + background storage dispatch).
    Terminal handler — always the last path tried.
    """
    orchestrator = ctx.orchestrator
    full_prompt = ctx.full_prompt
    system_prompt = ctx.system_prompt
    note_images = ctx.note_images
    raw_context = ctx.raw_context
    merged_input = ctx.merged_input
    user_text = ctx.user_text
    personality = ctx.personality
    file_names = ctx.file_names
    conversation_logger = ctx.conversation_logger
    history = ctx.history
    agentic_enabled = ctx.agentic_enabled
    fast_mode = ctx.fast_mode

    # Enhanced is a tool-less generation path. Tell the model so it doesn't claim
    # to have performed side effects it can't (the confabulation backstop — the
    # action guard below catches it after the fact, this prevents it up front).
    # NOTE: scoped to the streaming call via _stream_system_prompt — the
    # uncertainty/review agentic RETRIES below reuse `system_prompt` and DO have
    # tools, so they must not inherit the "no tools" claim.
    _stream_system_prompt = system_prompt
    try:
        from config.app_config import ACTION_CLAIM_GUARD_ENABLED
        if ACTION_CLAIM_GUARD_ENABLED:
            _stream_system_prompt = (system_prompt or "") + (
                "\n\n[ACTION HONESTY] You have no tools available this turn. Do NOT "
                "claim you saved, sent, created, scheduled, emailed, or added anything. "
                "If the user wants such an action, OFFER it (\"Want me to …?\") and the "
                "system will carry it out — never state it is already done."
            )
    except Exception:
        pass
    _original_limits = ctx.original_limits
    _t_prepare_start = ctx.t_prepare_start
    _t_prepare_elapsed = ctx.t_prepare_elapsed
    import time as _time_mod
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
        _t_stream_start = _time_mod.perf_counter()
        thinking_started = False
        thinking_complete = False
        chunk_count = 0
        async for chunk in orchestrator.response_generator.generate_streaming_response(
            prompt=full_prompt,
            model_name=model_name,
            system_prompt=_stream_system_prompt,
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
                    display_output = _strip_leaked_xml_blocks(display_output)
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
                display_output = _strip_leaked_xml_blocks(display_output)
                yield {"role": "assistant", "content": display_output}

        # After streaming completes, if we're still showing "Thinking..." the user
        # sees nothing. Parse the accumulated output and yield whatever we have.
        if thinking_started and not thinking_complete and final_output.strip():
            thinking_part, final_answer = ResponseParser.parse_thinking_block(final_output)
            # If entire response is thinking (no answer), don't leak it
            if thinking_part and not final_answer:
                logger.warning("[Handle Submit] Entire response was thinking — suppressing")
                final_answer = ""
            recovered = final_answer if final_answer else final_output
            # Strip thinking tags/content if they leaked
            recovered = ResponseParser.strip_thinking_tag_leaks(recovered)
            recovered = _strip_leaked_xml_blocks(recovered)
            if recovered.strip():
                display_output = recovered.strip()
                final_output = display_output
                logger.info(f"[Handle Submit] Recovered from stuck thinking state, output_len={len(display_output)}")
                yield {"role": "assistant", "content": display_output}

        # After streaming completes, parse thinking block for logging and storage
        _t_stream_elapsed = _time_mod.perf_counter() - _t_stream_start
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
            # Update final_output to only include the final answer for storage.
            # If entire response was thinking, don't fall back to raw thinking.
            if final_answer_stream:
                final_output = final_answer_stream
            elif not final_answer_stream:
                logger.warning("[Handle Submit] Post-stream: entire response was thinking — suppressing")
                final_output = ""
            # Sync display_output so final yield doesn't show stale thinking-polluted content
            display_output = final_output

        # Strip leaked XML tool blocks (LLM sometimes hallucinates tool-call XML
        # in standard mode when prior conversation mentioned agentic tools).
        # Use block-level stripping to remove entire <tool>content</tool> sequences.
        final_output = _strip_leaked_xml_blocks(final_output)
        display_output = _strip_leaked_xml_blocks(display_output)

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
                        _uf_hint = (
                            f'[MEMORY SEARCH RETRY] The user asked: "{user_text}" '
                            f"and the initial response could not find relevant "
                            f"information from context. Search memory deeply using "
                            f"the search_memory tool across conversations, "
                            f"summaries, and obsidian_notes collections. The "
                            f"information may exist but was not retrieved in the "
                            f"initial pass."
                        )
                        _uf_clean, _uf_think = await _silent_agentic_retry(
                            orchestrator, merged_input, system_prompt,
                            model_name, raw_context, final_output,
                            _uf_hint, "UNCERTAINTY FALLBACK",
                        )
                        if _uf_clean is not None:
                            final_output = _uf_clean
                            display_output = final_output
                            thinking_part_stream = _uf_think or thinking_part_stream
                            _uncertainty_retry_done = True

            except ImportError as e:
                logger.debug(f"[UNCERTAINTY FALLBACK] Module not available: {e}")
            except Exception as e:
                logger.warning(
                    f"[UNCERTAINTY FALLBACK] Detection failed (non-fatal): {e}"
                )

        # ── Post-Answer Review Gate: check response against plan ──
        _review_retry_done = False
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
                            plan=_plan, response=final_output, query=user_text,
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
                            _rg_hint = (
                                f'[RESPONSE REVIEW RETRY] The user asked: "{user_text}" '
                                f"The initial response had these issues: "
                                f"{'; '.join(_review.issues)}. "
                                f"Suggestion: {_review.suggestion}. "
                                f"Search memory and provide a better answer."
                            )
                            _rg_clean, _rg_think = await _silent_agentic_retry(
                                orchestrator, merged_input, system_prompt,
                                model_name, raw_context, final_output,
                                _rg_hint, "REVIEW GATE",
                            )
                            if _rg_clean is not None:
                                final_output = _rg_clean
                                display_output = final_output
                                thinking_part_stream = _rg_think or thinking_part_stream
                                _review_retry_done = True
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

        # After streaming completes, emit a final debug record
        prompt_tokens2, system_tokens2, total_tokens2 = _safe_count_tokens(
            full_prompt, system_prompt, model_name, orchestrator,
        )

        _resp_for_debug = _sanitize_response_text(display_output or final_output)
        _resp_for_debug, citations = _safe_extract_citations(
            _resp_for_debug, orchestrator,
        )

        _enh_session_id = _get_session_id(orchestrator)
        _enh_mode = "uncertainty-fallback" if _uncertainty_retry_done else "enhanced"
        _enh_prov = _build_provenance(
            _enh_mode, _enh_session_id, model_name, citations,
            thinking_block=thinking_part_stream or "",
        )
        if _uncertainty_retry_done:
            _attach_agentic_provenance(_enh_prov, orchestrator)

        _phase_timings = getattr(orchestrator, '_last_phase_timings', {})
        _task_timings = getattr(orchestrator, '_last_task_timings', {})
        _gather_elapsed = getattr(orchestrator, '_last_gather_elapsed', 0.0)
        _handler_timings = {
            "prepare_prompt": round(_t_prepare_elapsed, 3),
            "llm_streaming": round(_t_stream_elapsed, 3),
            "total_wall": round(_t_prepare_elapsed + _t_stream_elapsed, 3),
        }
        if _phase_timings:
            _handler_timings["context_pipeline"] = _phase_timings.get("context_pipeline", 0.0)
            _handler_timings["prompt_build"] = _phase_timings.get("prompt_build", 0.0)

        # Parse text-based action proposals from enhanced response.
        # Model may output [propose_action: send_email] {...} as text even
        # in non-agentic mode if it knows about the tool from context.
        _enh_pending_action_id = None
        try:
            from config.app_config import INTERNET_ACTIONS_ENABLED
            if INTERNET_ACTIONS_ENABLED and _resp_for_debug:
                from core.agentic.tools import ToolExecutor
                _enh_store = ToolExecutor._get_pending_actions_store()
                if not _enh_store.get_pending():
                    from core.agentic.protocols import NativeToolsHandler
                    _enh_handler = NativeToolsHandler(actions_available=True)
                    _enh_decisions = _enh_handler._parse_text_tool_calls(_resp_for_debug)
                    # Handle lookup_contact inline: resolve contact, auto-create email proposal if context indicates sending
                    for _etd in _enh_decisions:
                        if _etd.wants_lookup_contact and _etd.lookup_contact_name:
                            try:
                                # Strip XML tool artifacts from display (3-pattern subset)
                                _resp_for_debug = _strip_inline_tool_xml(_resp_for_debug, full=False)
                                _resp_for_debug, _lc_aid = await _resolve_contact_and_propose_email(
                                    _etd.lookup_contact_name, user_text, history,
                                    _resp_for_debug, _enh_store,
                                    no_contacts_suffix=" in Google Contacts or Gmail",
                                )
                                if _lc_aid:
                                    _enh_pending_action_id = _lc_aid
                                logger.info(f"[Handle Submit] Enhanced: resolved contact '{_etd.lookup_contact_name}' inline")
                            except Exception as _lc_err:
                                logger.warning(f"[Handle Submit] Enhanced: contact lookup failed: {_lc_err}")
                            break
                    for _etd in _enh_decisions:
                        if _etd.wants_action and _etd.action_type:
                            logger.info(f"[Handle Submit] Enhanced: parsed text action: {_etd.action_type}")
                            _ea_aid = _make_text_action_proposal(_etd, _enh_store)
                            if _ea_aid is None:
                                break
                            _enh_pending_action_id = _ea_aid
                            # Strip raw tool text + leaked XML and append proper action card
                            import re as _re_enh_action
                            _resp_for_debug = _re_enh_action.sub(
                                r'\[propose_action:\s*\w+\]\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}',
                                '', _resp_for_debug, count=1,
                            ).strip()
                            _resp_for_debug = _strip_inline_tool_xml(_resp_for_debug)
                            _pending = _enh_store.get_pending()
                            if _pending:
                                _enh_recip = _pending.params.get("recipient", "")
                                _enh_subj = _pending.params.get("subject", "")
                                _enh_msg = _pending.params.get("message", "")
                                _enh_header = f"**{_pending.action_type.value}**"
                                if _enh_recip:
                                    _enh_header += f" to {_enh_recip}"
                                if _enh_subj:
                                    _enh_header += f" — *{_enh_subj}*"
                                _card = f"\n\n---\n{_enh_header}\n"
                                if _enh_msg:
                                    _card += f"> {_enh_msg[:300]}\n\n"
                                _resp_for_debug += _card
                            break
        except (ImportError, Exception) as e:
            logger.warning(f"[Handle Submit] Enhanced text action parse failed: {e}")

        # ── Action guard: capture proposals + verify completion claims ──
        # Enhanced is a TOOL-LESS path, so any "Done — saved the note" claim is
        # unbacked. Self-repair note/doc claims; honestly correct external claims
        # that weren't even proposed. (proposed kinds suppressed via the card.)
        try:
            from core.action_claim_guard import EXTERNAL as _EXTERNAL_KINDS
            _enh_proposed = _EXTERNAL_KINDS if _enh_pending_action_id else set()
            _guard_suffix = await _apply_action_guard(
                ctx, _resp_for_debug, executed_kinds=set(),
                proposed_kinds=_enh_proposed, self_repair=True,
            )
            if _guard_suffix:
                _resp_for_debug = (_resp_for_debug or "").rstrip() + _guard_suffix
                final_output = (final_output or "").rstrip() + _guard_suffix
        except Exception as e:
            logger.warning(f"[Handle Submit] Enhanced action guard failed (non-fatal): {e}")

        debug_record = _build_debug_record(
            mode=_enh_mode, user_text=user_text, prompt=full_prompt,
            system_prompt=system_prompt, response=_resp_for_debug,
            model=model_name, prompt_tokens=prompt_tokens2,
            system_tokens=system_tokens2, total_tokens=total_tokens2,
            citations=citations, orchestrator=orchestrator,
            provenance=_enh_prov, phase_timings=_handler_timings,
            task_timings=_task_timings, gather_elapsed=_gather_elapsed,
        )
        _enh_final_chunk = {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
        if _enh_pending_action_id:
            _enh_final_chunk["pending_action_id"] = _enh_pending_action_id
        yield _enh_final_chunk
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

                # Sanitize response for storage
                response_to_store = _sanitize_response_text(final_output)
                response_to_store = _strip_echoed_headers(response_to_store)

                # Build provenance from the debug_record emitted during streaming
                _store_prov = None
                _store_mode = "enhanced"
                _store_session_id = _get_session_id(orchestrator)
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
                        "thinking_block": "",
                    }

                _dispatch_storage(
                    orchestrator, merged_input, response_to_store, user_text,
                    final_output, personality, file_names, conversation_logger,
                    _store_session_id, _store_prov, _store_mode,
                )
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
    # Supports .txt, .md, .json, .yaml, .yml, .log, .html, .xml, .csv, .py, .docx, .xlsx, .pdf files and .png, .jpg, .jpeg, .gif, .webp images
    file_names = [file.name for file in files] if files else []
    files_result = await file_processor.process_files_structured(user_text, files or [])
    merged_input = files_result.text_content

    # Persist uploads to ChromaDB in background (fire-and-forget)
    if files_result.documents or files_result.images:
        persist_task = asyncio.create_task(_persist_uploads(orchestrator, files_result))
        _pending_storage_tasks.add(persist_task)
        persist_task.add_done_callback(_pending_storage_tasks.discard)

    # Threaded state for this turn, shared by the per-mode handlers.
    ctx = SubmitContext(
        user_text=user_text,
        files=files,
        history=history,
        use_raw_gpt=use_raw_gpt,
        orchestrator=orchestrator,
        personality=personality,
        fast_mode=fast_mode,
        conversation_logger=conversation_logger,
        file_names=file_names,
        merged_input=merged_input,
        files_result=files_result,
    )

    # RAW MODE: go straight through orchestrator (personality hook is handled inside process_user_query)
    if use_raw_gpt:
        async for _c in _run_raw(ctx):
            yield _c
        return

    # ── Pending-proposal follow-through ──────────────────────────────────────
    # When the previous turn OFFERED an action ("Want me to save this as a note?")
    # and this turn is a short affirmation ("sure that makes sense"), execute the
    # captured proposal directly — BEFORE the agentic gate's casual/short skip can
    # route it into a tool-less mode where the action would never fire (and the
    # model would confabulate success). Also bumps the per-turn counter used for
    # proposal TTL.
    _pp_store = _get_pending_proposal_store(orchestrator)
    if _pp_store is not None:
        _pp_store.bump_turn()
        _affirmed = _pp_store.consume_if_affirmed(user_text)
        if _affirmed is not None:
            async for _c in _run_pending_proposal(ctx, _affirmed):
                yield _c
            if ctx.handled:
                return
            # else execution failed — fall through to the normal flow

    # Check if agentic search might be used (need to know before calling prepare_prompt)
    _cfg = getattr(orchestrator, 'config', {}) or {}
    agentic_cfg = _cfg.get('agentic_search', {}) if isinstance(_cfg, dict) else {}
    agentic_enabled = bool(agentic_cfg.get('enabled', False))

    # Build the enhanced-path prompt context (fast-mode limits, prepare_prompt, image inject).
    ctx.agentic_enabled = agentic_enabled
    async for _c in _prepare_submit_context(ctx):
        yield _c

    # Alias prelude results back to locals for the inline mode bodies not yet extracted.
    full_prompt = ctx.full_prompt
    system_prompt = ctx.system_prompt
    raw_context = ctx.raw_context
    note_images = ctx.note_images
    _original_limits = ctx.original_limits
    _t_prepare_start = ctx.t_prepare_start
    _t_prepare_elapsed = ctx.t_prepare_elapsed
    import time as _time_mod

    # ── DUEL MODE: Two models + judge, takes priority over agentic ──
    _cfg_duel = getattr(orchestrator, 'config', {}) or {}
    _features_duel = _cfg_duel.get('features', {}) if isinstance(_cfg_duel, dict) else {}
    _DUEL_ON = bool(_features_duel.get('best_of_duel_mode', False))
    _DUEL_GENS = list(_features_duel.get('best_of_generator_models', []))
    _DUEL_SELS = list(_features_duel.get('best_of_selector_models', []))
    duel_active = bool(_DUEL_ON and len(_DUEL_GENS) >= 2 and len(_DUEL_SELS) >= 1)
    logger.info(f"[Handle Submit] Duel check: on={_DUEL_ON}, gens={_DUEL_GENS}, sels={_DUEL_SELS}, active={duel_active}")

    if duel_active:
        async for _c in _run_duel(ctx, _DUEL_GENS, _DUEL_SELS, _features_duel):
            yield _c
        if ctx.handled:
            return
        # else duel bailed (timeout/exception) — fall through to agentic/streaming

    if agentic_enabled:
        from core.agentic.gate import evaluate_agentic_gate
        _gate_decision = await evaluate_agentic_gate(
            user_text=user_text,
            entity_resolver=getattr(getattr(orchestrator, 'memory_system', None), 'entity_resolver', None),
            model_manager=orchestrator.model_manager,
            corpus_manager=getattr(getattr(orchestrator, 'memory_system', None), 'corpus_manager', None),
            intent_info=raw_context.get("intent") if raw_context else None,
        )
        should_use_agentic = _gate_decision.should_trigger
        search_terms = _gate_decision.search_terms
        needs_computation = "computation" in _gate_decision.modes
        needs_memory = "memory" in _gate_decision.modes
        needs_knowledge = "knowledge" in _gate_decision.modes
        needs_web_search = "web_search" in _gate_decision.modes
        needs_tools = "tools" in _gate_decision.modes
        _matched_entities = _gate_decision.matched_entities
        _doc_gen_intent = _gate_decision.doc_gen_intent
        _self_note_intent = _gate_decision.self_note_intent

        # Populate the gate outputs on ctx for the agentic-path handlers.
        ctx.gate_decision = _gate_decision
        ctx.should_use_agentic = should_use_agentic
        ctx.search_terms = search_terms
        ctx.doc_gen_intent = _doc_gen_intent
        ctx.self_note_intent = _self_note_intent
        ctx.skip_initial_search = getattr(_gate_decision, 'skip_initial_search', False)

        # --- Direct document generation (bypasses agentic loop) ---
        if _doc_gen_intent and should_use_agentic:
            async for _c in _run_doc_generation(ctx):
                yield _c
            if ctx.handled:
                return
            # else doc-gen failed — fall through to self-note / agentic / enhanced

        # --- Direct daemon self-note creation (bypasses agentic loop) ---
        if _self_note_intent and should_use_agentic:
            async for _c in _run_self_note(ctx):
                yield _c
            if ctx.handled:
                return

        if should_use_agentic:
            async for _c in _run_agentic_search(ctx):
                yield _c
            if ctx.handled:
                return
            # else agentic failed — fall through to enhanced

    async for _c in _run_enhanced(ctx):
        yield _c


# ---------------------------------------------------------------------------
# Internet Actions — Approve / Reject handlers (called by GUI buttons)
# ---------------------------------------------------------------------------

async def execute_pending_action(action_id: str, chat_history: list, orchestrator=None):
    """Execute an approved internet action. Called by GUI Approve button.

    Does NOT go through submit_chat — directly modifies chat_history and returns.
    """
    import gradio as gr
    from core.actions.types import PendingActionsStore
    from core.actions.audit import ActionAuditLog
    from config.app_config import INTERNET_ACTIONS_AUDIT_LOG

    if not action_id:
        return chat_history, gr.update(value=None), gr.update(visible=False)

    # Load proposal from the global store
    from core.agentic.tools import ToolExecutor
    store = ToolExecutor._get_pending_actions_store()
    proposal = store.approve(action_id)

    if not proposal:
        chat_history.append({
            "role": "assistant",
            "content": "Action expired or not found. Ask me again if you still want this."
        })
        return chat_history, gr.update(value=None), gr.update(visible=False)

    # Audit: log approval
    audit = ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG)
    audit.log_decision(action_id, approved=True)

    # Execute via the executor registry
    try:
        from core.actions.executors import ActionExecutorRegistry
        executor = ActionExecutorRegistry()
        result = await executor.execute(proposal)
        audit.log_execution(action_id, result)

        if result.success:
            store.mark_executed(action_id, result.message)
            msg = f"[ACTION EXECUTED: {proposal.action_type.value}] {result.message}"
        else:
            store.mark_failed(action_id, result.message)
            msg = f"Action failed: {result.message}\n\nWant me to try something else?"
    except Exception as e:
        store.mark_failed(action_id, str(e))
        msg = f"Action failed with error: {e}\n\nWant me to try something else?"
        logger.error(f"[Actions] Execution failed for {action_id}: {e}")

    chat_history.append({"role": "assistant", "content": msg})
    return chat_history, gr.update(value=None), gr.update(visible=False)


async def reject_pending_action(action_id: str, chat_history: list, orchestrator=None):
    """Reject a pending internet action. Called by GUI Reject button."""
    import gradio as gr
    from core.actions.audit import ActionAuditLog
    from config.app_config import INTERNET_ACTIONS_AUDIT_LOG

    if not action_id:
        return chat_history, gr.update(value=None), gr.update(visible=False)

    from core.agentic.tools import ToolExecutor
    store = ToolExecutor._get_pending_actions_store()
    proposal = store.reject(action_id)

    audit = ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG)
    audit.log_decision(action_id, approved=False)

    if proposal:
        chat_history.append({
            "role": "assistant",
            "content": f"[ACTION REJECTED] Cancelled: {proposal.summary}"
        })
    else:
        chat_history.append({
            "role": "assistant",
            "content": "Action already expired or was not found."
        })

    return chat_history, gr.update(value=None), gr.update(visible=False)
