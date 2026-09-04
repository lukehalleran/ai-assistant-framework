"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/duel/agentic/enhanced flows, streams the response to the UI, and persists interaction + provenance + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?, is_progress?} as Gradio updates.
- Behavior:
  - Pacing ingress (2026-07-25): handle_submit calls time_manager.mark_query_time() at INGRESS
    (was post-prompt-build in ResponseGenerator, skipped entirely by agentic turns — [TIME
    CONTEXT] showed "53 m" for a 2-min gap).
  - Agentic gate veto forwarding (2026-08-02): apply_intent_veto receives tone_level AND the raw
    query so the tone-corroborated + tone-statement vetoes can fire; a veto teaches "no_search"
    to the adaptive web-search anchors.
  - Citation-outcome teacher (2026-08-03): _write_turn_telemetry(response_text=...) — a response
    citing [WEB_ markers teaches "search_worthy" (utils.adaptive_exemplars, domain "web_search");
    elevated-tone turns never teach. All 3 telemetry call sites (enhanced/agentic/duel) pass text.
  - Empty-thinking-shell suppression (2026-08-03): between the synthetic </thinking> marker and
    the first content token the stream buffer is exactly "<thinking></thinking>";
    is_empty_thinking_shell() keeps the 💭 indicator up instead of flashing the literal tags.
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - DUEL mode: If BEST_OF_DUEL_MODE enabled, two models compete + judge picks winner. Builds provenance with response_mode="best-of-duel". Runs BEFORE agentic check.
  - AGENTIC: If agentic_search.enabled, delegates to core/agentic/gate.py:evaluate_agentic_gate()
    which runs the 4-tier gate (keyword → entity → doc/note intent → LLM fallback) and returns
    an AgenticDecision. If triggered, routes through AgenticSearchController.
    Uses merged_input (user text + file content) as query so uploaded file content is visible to the agentic loop.
    - Streaming hides thinking via has_incomplete_thinking_block() (tags) + likely_untagged_thinking() (heuristic)
    - Storage uses _sanitize_response_text(final_output) — the raw accumulated stream
      contains synthetic <thinking></thinking> markers from response_generator and must
      never be persisted raw [FIXED 2026-06-10; backstop also in memory_storage]
    - Post-content ProgressEvents suppressed to prevent response pop-back
    - Citations extracted from memory_id_map via _extract_citations()
  - UNCERTAINTY FALLBACK [NEW 2026-04-27]: After standard streaming, UncertaintyDetector checks response for "I don't know" signals (keyword regex + semantic embedding). If uncertain + agentic enabled → silent retry via agentic search. Retry only accepted if word overlap with original < 70%. No progress messages or chunk streaming during retry.
  - POST-ANSWER REVIEW GATE: Checks response against ResponsePlan. Silent retry if confidence >= 0.90 (raised from 0.80). Skipped for responses < 120 chars. Same similarity guard (overlap < 70%).
  - ENHANCED: orchestrator.prepare_prompt → extract note_images → response_generator.generate_streaming_response(images=...) → store interaction
  - IMAGE SUPPORT [NEW 2026-01-30]: Extracts note_images from raw_context and passes to streaming for multimodal models
  - API error classification: [CREDITS EXHAUSTED], [RATE LIMITED], [AUTH ERROR], [MODEL NOT FOUND], [SERVER ERROR], [Streaming Error (2026-08-14)] with user-friendly messages + early return BEFORE storage
  - Stream-artifact display strip (2026-08-14): agentic/raw/duel paths apply strip_trailing_stream_artifact to yielded/recorded text (enhanced path sanitizes via _sanitize_response_text)
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
  - _run_insight_mode [NEW 2026-08-23]: insight/evidence-assembly turn-owner
    (gate_decision.insight_intent; dispatched BEFORE doc-gen). Facet decompose →
    ungated cross-store sweep (8s keepalive heartbeats) → provenance labeling →
    adversarial assessment (assessment kind) → streamed synthesis → optional
    DocumentGenerator.save_prewritten (doc only on agree/partial or un-assessed
    explicit request); stores with response_mode "insight-assembly"; exception →
    ctx.handled False (falls through). The dispatcher also arms the one-shot
    insight consent offer (gate.maybe_arm_insight_offer → [INSIGHT OFFER]
    system-prompt note) when the mode did NOT trigger.
  - _run_doc_generation / _run_self_note: agentic-gate bypasses (do their own store_interaction).
    Conversation-sourced docs (2026-08-24): _resolve_doc_source (LLM document_source
    declaration OR _DOC_CONVERSATION_SOURCE_RE deterministic backstop) routes
    "summarize these insights so I can text them to my therapist"-shaped requests
    to _build_conversation_source_material — ctx.history rendered as a transcript
    and passed as source_material (clears DOCUMENT_PROVIDED_MIN_CHARS, so the
    generator writes up the conversation instead of web-researching the topic).
  - _run_enhanced owns the post-answer passes (uncertainty fallback, review gate) and the
    finally cleanup (fast-mode restore + storage). Its finally is enhanced-path-only by design
    (see "latent fast-mode-restore" note below) — do NOT hoist it to the dispatcher.
- Turn telemetry [NEW 2026-07-03]: one JSONL line per completed turn (utils/turn_telemetry.py,
  logs/turn_records.jsonl). SubmitContext.telemetry accumulates gate decision (dispatcher) +
  uncertainty/review outcomes (_run_enhanced) + grounding-check outcomes (2026-08-28:
  grounding_prefilter_fired / grounding_verifier_fired / grounding_flagged /
  grounding_confidence / grounding_corrected, set by _apply_grounding_check on the
  enhanced AND agentic paths; 2026-09-04: grounding_mode ("log_only"/"correct") +
  grounding_verdict (redacted correction text, <=300 chars) so precision can be
  measured later — LOG-ONLY is the default, same class as the 08-28 review-gate
  LOG-ONLY fix; see GROUNDING_MODE in config.app_config); _write_turn_telemetry() merges those with
  orchestrator._last_turn_signals (intent/tone/plan, captured in build_full_prompt) and writes
  at the duel/agentic/enhanced storage-dispatch sites plus the doc-generation and
  self-note bypass paths (2026-07-05 — those turns previously vanished from the record). Never raises.
  Also hosts the post-response truth pipeline hook (2026-08-23; consolidated 2026-09-04
  into core.orchestrator.run_post_response_hooks()'s "post_response_detectors" entry,
  which calls orchestrator.run_post_response_detectors — corrections/confirmations →
  truth events + staleness cascade + narrative-stale flag; process_user_query drives
  the SAME registry via its own _run_post_response_hooks(), closing the class of
  GUI/process_user_query dead-wiring drift the escalation bug above also belonged to).
- Extracted helpers:
  - _safe_count_tokens(), _safe_extract_citations(), _build_debug_record(), _build_provenance(),
    _attach_agentic_provenance(), _sanitize_response_text(), _strip_echoed_headers(),
    _dispatch_storage(), _silent_agentic_retry(), _get_session_id(), _find_email_draft(),
    _write_turn_telemetry()
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
from utils.attachment_audit import audit_attachments, deadline_timezone_note
import json
from config.app_config import load_system_prompt
import re as _re_draft
import time as _time_mod
import time as _time
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
            # Log the same sanitized text that memory stores. The raw stream
            # can start with provider control tokens such as <|sep|>; logging
            # it raw made debug transcripts look polluted even when the UI and
            # corpus were clean.
            assistant_response=response_to_store,
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


def _gate_debug_summary(gate_decision) -> str:
    """One-line 'why this turn routed as it did' for the debug record — the
    gate's trigger modes / veto reason + veto-exempt/deferred flags. Added
    2026-09-02: a debug dump showed mode=agentic-search but not WHY (which
    tier fired, whether a tone veto was even considered), so a 129s agentic
    loop on an emotional check-in couldn't be diagnosed from the dump alone."""
    if gate_decision is None:
        return ""
    # A debug helper must never be able to take down the turn: it is called
    # inside both the agentic and enhanced paths, and (2026-09-02 evening) a
    # non-string `reason` raised inside the join, which killed the agentic
    # turn AND the enhanced fallback ("Streaming error"). Coerce + fail soft.
    try:
        parts = []
        reason = getattr(gate_decision, "reason", "") or ""
        if reason:
            parts.append(str(reason))
        modes = getattr(gate_decision, "modes", None)
        if modes:
            parts.append(f"modes={[str(m) for m in modes]}")
        if getattr(gate_decision, "veto_exempt", False):
            parts.append("veto_exempt")
        if getattr(gate_decision, "deferred_request", None):
            parts.append("deferred_request")
        return " | ".join(parts)
    except Exception:
        return ""


def _build_debug_record(
    mode, user_text, prompt, system_prompt, response, model,
    prompt_tokens, system_tokens, total_tokens,
    citations, orchestrator, provenance=None,
    phase_timings=None, task_timings=None, gather_elapsed=0.0,
    gate_reason=None,
):
    """Build a debug record dict for the Debug Trace tab."""
    # A leading EMPTY reasoning shell ("<thinking></thinking>Answer…") is a
    # stream artifact with zero diagnostic value — display and storage
    # already strip it; keep the record aligned with what the user actually
    # saw (2026-08-05: an agentic record's RESPONSE opened with the literal
    # shell, reading as a leak that never reached the user).
    if isinstance(response, str):
        response = _re.sub(
            r"^\s*<(thinking|think|reasoning|reason)>\s*</\1>\s*", "", response
        )
    # Preserve the exact instructions that were actually injected, not just
    # their count and tone.  Do not persist the raw planner response or the
    # context digest: the operative structured fields are sufficient to audit
    # a reversal, while hashes + section names establish context alignment
    # without multiplying copies of personal prompt content.
    plan_audit = None
    try:
        plan = getattr(orchestrator, "_current_response_plan", None)
        if plan is not None and hasattr(plan, "audit_record"):
            plan_audit = plan.audit_record()
            if isinstance(provenance, dict):
                provenance["response_plan"] = plan_audit
    except Exception as e:
        logger.debug(f"[Handlers] Could not attach response-plan audit: {e}")
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
        'gate_reason': gate_reason or '',
        'response_plan': plan_audit,
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
    # Canonical storage-grade strip: empty <thinking></thinking> stream markers,
    # leading/unclosed tagged blocks, stray tag fragments, reflection blocks
    text = ResponseParser.sanitize_for_storage(text)
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


# Friendly display templates for classified API-error payloads (the model
# layer yields these AS response text instead of raising). Single map for all
# display paths — enhanced streaming, agentic, and the mid-stream fail-fast.
_API_ERROR_DISPLAY = {
    "[CREDITS EXHAUSTED]": "💳 **Out of API Credits**\n\n{msg}\n\nYou can add credits at your provider's billing page or switch models in the dropdown above.",
    "[RATE LIMITED]": "⏳ **Rate Limited**\n\n{msg}",
    "[AUTH ERROR]": "🔑 **Authentication Error**\n\n{msg}",
    "[MODEL NOT SUPPORTED]": "🚫 **Unsupported Input**\n\n{msg}\n\nTry switching to a multimodal model (e.g. GPT-4o, Claude) in the dropdown above.",
    "[MODEL NOT FOUND]": "❓ **Model Not Found**\n\n{msg}",
    "[SERVER ERROR]": "🔥 **Server Error**\n\n{msg}",
    "[API Error]": "⚠️ **API Error**\n\n{msg}",
    "[API unavailable]": "⚠️ **API Unavailable**\n\n{msg}",
    "[OpenAI unavailable": "⚠️ **Provider Unavailable**\n\n{msg}\n\nThe model provider is down; retry in a moment or switch models in the dropdown above.",
    "[Streaming Error": "🔥 **Stream Interrupted**\n\n{msg}\n\nThe provider dropped the connection mid-response — retrying your message usually works.",
    "[Error: Model returned empty response": "⚠️ **Empty Response**\n\nThe model returned no usable answer after retrying without reasoning. Retry your message or switch models.",
}


def _friendly_api_error(text):
    """Return the friendly display string for a classified API-error response,
    or None if the text is not an error payload. Matches at the HEAD only —
    a real answer with an appended trailing marker is not converted here
    (the storage boundary strips the marker and keeps the partial answer)."""
    stripped = (text or "").strip()
    for prefix, template in _API_ERROR_DISPLAY.items():
        if stripped.startswith(prefix):
            msg = stripped[len(prefix):].strip()
            if prefix in ("[Streaming Error", "[OpenAI unavailable"):
                # These prefixes don't include the closing bracket (emit
                # shapes: "[Streaming Error: msg]" / "[Streaming Error] msg"
                # and "[OpenAI unavailable] msg") — strip the stray leading
                # "]" the naive slice above would otherwise leave in msg.
                msg = msg.lstrip(":]").strip().rstrip("]").strip()
            return template.format(msg=msg)
    return None


def _write_turn_telemetry(ctx, mode, session_id, model_name, response_len,
                          response_text=None):
    """Run the shared per-turn post-response hook registry (never raises).

    Delegates to core.orchestrator.run_post_response_hooks() — the single
    POST_RESPONSE_HOOKS registry also driven by
    DaemonOrchestrator._run_post_response_hooks() (process_user_query's own
    pipeline, reached by RAW mode / `python main.py cli`). Before
    2026-09-04 this function and process_user_query maintained separate,
    hand-written copies of this sequence, which is exactly how the 2026-08-21
    EscalationTracker and 2026-08-23 correction-pipeline dead-wiring bugs
    happened — see tests/unit/test_request_path_parity.py.

    Hooks run (in order): turn-telemetry write (merges
    orchestrator._last_turn_signals — intent/tone/plan, captured in
    build_full_prompt — with ctx.telemetry — gate + post-answer check
    fields — and this call's outcome fields; see utils/turn_telemetry.py);
    outcome-confirmed adaptive "search_worthy" teaching when response_text
    actually cited web results ([WEB_ markers, 2026-08-02; elevated-tone
    turns never teach); escalation_tracker.record_response (feeds next
    turn's engagement detection); orchestrator.run_post_response_detectors
    (corrections/confirmations -> truth events -> staleness cascade).
    """
    try:
        from core.orchestrator import PostResponseHookContext, run_post_response_hooks
        # getattr-defensive: some callers (e.g. _run_pending_proposal's
        # lightweight SimpleNamespace ctx) don't carry every SubmitContext
        # field. The pre-registry code wrapped each ctx.* read in its own
        # try/except; this preserves the same "never raises on a partial
        # ctx" contract in one place.
        hook_ctx = PostResponseHookContext(
            orchestrator=getattr(ctx, "orchestrator", None),
            user_input=getattr(ctx, "user_text", "") or "",
            response_text=response_text,
            mode=mode,
            session_id=session_id,
            model_name=model_name,
            response_len=response_len,
            telemetry=getattr(ctx, "telemetry", None) or {},
            t_prepare_elapsed=getattr(ctx, "t_prepare_elapsed", 0.0) or 0.0,
        )
        run_post_response_hooks(hook_ctx)
    except Exception as e:
        logger.debug(f"[Telemetry] post-response hooks skipped: {e}")


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
    if not store.propose(proposal):
        logger.warning("[Handle Submit] Pending action store rejected text proposal")
        return None
    ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG).log_proposal(proposal)
    return proposal.action_id


def _format_action_proposal_card(proposal) -> str:
    """Render enough detail for informed approval, including calendar batches."""
    params = proposal.params or {}
    action_name = proposal.action_type.value
    events = params.get("events") if isinstance(params.get("events"), list) else []
    if action_name == "calendar_create_event" and events:
        lines = [
            f"\n\n---\n**calendar_create_event** — {len(events)} events, one approval"
        ]
        for event in events:
            if not isinstance(event, dict):
                continue
            title = str(event.get("summary", "Untitled event"))
            start = str(event.get("start_time", "time missing"))
            all_day = event.get("all_day") is True or str(
                event.get("all_day", "")
            ).lower() in {"true", "1", "yes"}
            suffix = " [all day]" if all_day else ""
            lines.append(f"- **{title}** — {start}{suffix}")
        return "\n".join(lines) + "\n"

    if action_name == "calendar_create_event":
        title = str(params.get("summary", proposal.summary or "Untitled event"))
        start = str(params.get("start_time", "time missing"))
        all_day = params.get("all_day") is True or str(
            params.get("all_day", "")
        ).lower() in {"true", "1", "yes"}
        suffix = " [all day]" if all_day else ""
        return f"\n\n---\n**calendar_create_event** — **{title}** — {start}{suffix}\n"

    if action_name in ("calendar_update_event", "calendar_delete_event"):
        # Irreversible/modifying — the card must name EXACTLY which event.
        title = str(params.get("summary") or params.get("event_id") or "?")
        date = str(params.get("date", "date missing"))
        card = f"\n\n---\n**{action_name}** — **{title}** on {date}"
        changes = [
            f"{k[4:]} → {params[k]}"
            for k in ("new_summary", "new_start_time", "new_end_time",
                      "new_location", "new_description")
            if params.get(k)
        ]
        if changes:
            card += "\n> " + "; ".join(str(c)[:80] for c in changes)
        return card + "\n"

    recipient = params.get("recipient", "")
    subject = params.get("subject", "")
    message = params.get("message", "")
    header = f"**{action_name}**"
    if recipient:
        header += f" to {recipient}"
    if subject:
        header += f" — *{subject}*"
    card = f"\n\n---\n{header}\n"
    if message:
        card += f"> {str(message)[:300]}\n\n"
    return card


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
    # user_text plus any deterministic attachment/deadline notes (items 8-9,
    # 2026-09-04 homework-attachment turn audit) — what enhanced mode passes
    # as prepare_prompt's user_input (see _prepare_submit_context). Short:
    # ContextPipeline classification stages (topic/tone/intent/STM/query
    # rewrite) key off exactly this text before file content is merged in.
    analysis_text: str = ""
    agentic_enabled: bool = False
    # Agentic gate evaluated CONCURRENTLY with prepare_prompt (intent veto
    # applied post-hoc in the dispatcher once the context pipeline's intent
    # classification is available).
    gate_task: Any = None
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
    # --- per-turn telemetry accumulator (see utils/turn_telemetry.py) ---
    # Gate fields set in the dispatcher; uncertainty/review fields set in
    # _run_enhanced; merged with orchestrator._last_turn_signals and written
    # by _write_turn_telemetry() at each storage-dispatch site.
    telemetry: dict = field(default_factory=dict)


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
    from utils import turn_progress

    ctx.t_prepare_start = _time_mod.perf_counter()

    # Install the per-turn progress bus BEFORE prepare_prompt starts so the
    # prompt builder's live events (per-source retrieval completions, gating/
    # assembly milestones) stream to the UI instead of canned placeholders.
    _progress_q = turn_progress.begin_turn()
    try:
        prepare_task = asyncio.create_task(orchestrator.prepare_prompt(
            # user_input=ctx.user_text, NOT ctx.merged_input (2026-09-04,
            # homework-attachment turn audit items 1+2): ctx.merged_input is
            # already the file-processor's merged text (user text + every
            # attached file's content). Passing THAT plus files=ctx.files
            # made ContextPipeline.build()'s Stage 3 (_process_files)
            # re-run FileProcessor.process_files() on the SAME files against
            # the ALREADY-merged text, appending each attachment's content a
            # SECOND time — ~130K of a 265K-token turn was pure duplication.
            # Passing the raw user text + files instead lets Stage 3 do the
            # ONE canonical merge (context.file_context), which is what
            # build_full_prompt renders into [CURRENT QUERY] (has_files ?
            # file_context : original_query) — so the rendered prompt still
            # carries the full attachment content exactly once. It also
            # fixes the intent/tone/topic/STM/query-rewrite misclassification
            # class for free: every one of those stages keys off THIS
            # `user_input` parameter, evaluated BEFORE Stage 3 merges files
            # in, so they now see the user's own words instead of the
            # attachment blob (a "history/timeline/previous" hit inside a
            # lecture transcript no longer routes the turn to
            # temporal_recall).
            user_input=ctx.analysis_text or ctx.user_text,
            files=ctx.files,
            use_raw_mode=False,  # enhanced mode
            return_context=True  # Always get raw context for images and agentic search
        ))

        # Relay real pipeline events while waiting; fall back to a heartbeat
        # if the pipeline is quiet so mobile SSE connections stay alive.
        _quiet_polls = 0
        while not prepare_task.done():
            await asyncio.sleep(0.3)
            events = turn_progress.drain(_progress_q)
            if events:
                _quiet_polls = 0
                for _ev in events:
                    yield {"role": "assistant", "content": _ev, "is_progress": True}
            else:
                _quiet_polls += 1
                if _quiet_polls >= 20:  # ~6s of silence
                    _quiet_polls = 0
                    _elapsed_s = _time_mod.perf_counter() - ctx.t_prepare_start
                    yield {"role": "assistant", "content": f"💭 Working... ({_elapsed_s:.0f}s)", "is_progress": True}

        prep_result = await prepare_task
        # Flush any events emitted between the last poll and completion
        for _ev in turn_progress.drain(_progress_q):
            yield {"role": "assistant", "content": _ev, "is_progress": True}
    finally:
        turn_progress.end_turn()

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
    # kimi-3 lone-'e' stream artifact (display path — storage strips separately)
    response_text = ResponseParser.strip_trailing_stream_artifact(response_text)

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
        gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
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

        # kimi-3 lone-'e' stream artifact (display path — storage strips separately)
        final_output = ResponseParser.strip_trailing_stream_artifact(final_output)
        display_output = ResponseParser.strip_trailing_stream_artifact(display_output)

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
            gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
        )

        yield {"role": "assistant", "content": display_output, "debug": debug_record}

        _dispatch_storage(
            orchestrator, ctx.merged_input, final_output, ctx.user_text,
            final_output, ctx.personality, ctx.file_names, ctx.conversation_logger,
            _duel_session_id, _duel_prov, 'best-of-duel',
        )
        _write_turn_telemetry(
            ctx, 'best-of-duel', _duel_session_id, f"{m1} vs {m2}",
            len(final_output or ""),
            response_text=final_output,
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


# Conversation-sourced document backstop (2026-08-23): "Please summerize
# insight with direct evidence so I can text that to my therapist" ran
# RESEARCH mode — source_material was just the ~70-char request (below
# DOCUMENT_PROVIDED_MIN_CHARS) so the generator web-searched the topic
# instead of writing up the conversation. Deterministic shapes that mean
# "the source is THIS conversation": a summarize/write-up verb (misspelling
# tolerant) plus either a conversation/insight referent or a sharing cue
# ("so I can text that to my therapist").
_DOC_CONV_VERB = r"(?:summ[ae]r?i[sz]e|write\s+(?:\w+\s+)?up|put\s+together)"
_DOC_CONV_REFERENT = (
    r"(?:(?:this|our|the)\s+(?:conversation|discussion|chat|session)"
    r"|what\s+we(?:'ve)?\s+(?:just\s+)?(?:discussed|talked\s+about|covered)"
    r"|\binsights?\b"
    r"|so\s+i\s+can\s+(?:text|send|show|share|give|forward))"
)
_DOC_CONVERSATION_SOURCE_RE = _re.compile(
    rf"\b{_DOC_CONV_VERB}\b.{{0,120}}{_DOC_CONV_REFERENT}", _re.I | _re.S
)

_DOC_TRANSCRIPT_MAX_MESSAGES = 30
_DOC_TRANSCRIPT_MAX_CHARS = 8000


def _resolve_doc_source(doc_gen_intent, user_text) -> str:
    """'conversation' | 'research' for a doc_gen_intent.

    The LLM trigger's document_source declaration wins when it says
    "conversation"; the deterministic regex is the backstop (Tier-3 regex
    intents carry no source, and the LLM may omit the field).
    """
    if (doc_gen_intent or {}).get("source") == "conversation":
        return "conversation"
    if user_text and _DOC_CONVERSATION_SOURCE_RE.search(user_text):
        return "conversation"
    return "research"


def _build_conversation_source_material(history, user_text) -> str:
    """Render recent ctx.history as a transcript for DocumentGenerator.

    Newest messages win the char cap (trimmed from the front); the user's
    request rides along so the generator knows the framing.
    """
    lines = []
    for msg in (history or [])[-_DOC_TRANSCRIPT_MAX_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if not content or content.startswith("📝"):
            continue
        speaker = "User" if msg.get("role") == "user" else "Daemon"
        lines.append(f"{speaker}: {content}")
    transcript = "\n\n".join(lines)
    if len(transcript) > _DOC_TRANSCRIPT_MAX_CHARS:
        transcript = transcript[-_DOC_TRANSCRIPT_MAX_CHARS:]
    return f"[CONVERSATION TRANSCRIPT]\n{transcript}\n\n[USER REQUEST]\n{user_text or ''}".strip()


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

        # Conversation-sourced request ("summarize these insights so I can
        # text them to my therapist") → the transcript IS the material and
        # clears DOCUMENT_PROVIDED_MIN_CHARS, so web/wiki research is
        # suppressed. Otherwise pass the user's full message as primary
        # material so a "write a report evaluating this: <pasted content>"
        # request is grounded in that content rather than a generic web
        # search on the topic string.
        _doc_source = _resolve_doc_source(_doc_gen_intent, getattr(ctx, "user_text", None))
        if _doc_source == "conversation":
            _source_material = _build_conversation_source_material(
                getattr(ctx, "history", None), getattr(ctx, "user_text", None)
            )
            yield {"role": "assistant", "content": "📝 Summarizing our conversation...", "is_progress": True}
        else:
            _source_material = getattr(ctx, "user_text", None)
            yield {"role": "assistant", "content": f"📝 Researching: {_doc_gen_intent['topic']}...", "is_progress": True}

        _doc_result = await _dg.generate(
            topic=_doc_gen_intent["topic"],
            doc_type=_doc_gen_intent["doc_type"],
            focus=_doc_gen_intent.get("focus"),
            source_material=_source_material,
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

        # Debug record (2026-09-04): this bypass path used to yield the final
        # chunk with no "debug" key at all, so api/chat_service.py's
        # `is_final = "debug" in chunk` check never fired — a session whose
        # turns were all doc-gen never got a debug_records entry and
        # Provenance 404'd. Mirror _run_raw's pattern exactly (token counts
        # are zeroed rather than counted — this path never builds a metered
        # prompt/system_prompt the way the LLM-generation paths do).
        _doc_model = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
        debug_record = _build_debug_record(
            mode='doc-generation', user_text=ctx.user_text, prompt=_source_material,
            system_prompt=None, response=_doc_response, model=_doc_model,
            prompt_tokens=0, system_tokens=0, total_tokens=0,
            citations=[], orchestrator=orchestrator,
            gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
        )
        yield {"role": "assistant", "content": _doc_response, "debug": debug_record}
        _write_turn_telemetry(
            ctx, 'doc-generation', _get_session_id(orchestrator),
            _doc_model,
            len(_doc_response or ""),
        )
        ctx.handled = True
        return

    except Exception as e:
        logger.error(f"[Handle Submit] Direct document generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall through to normal agentic/enhanced mode (ctx.handled stays False)


def _interleave_phase_events(comparisons):
    """Round-robin scan events across phases (outcome events before proxies
    within each phase) so the downstream pattern-evidence cap samples every
    phase. Sequential phase-order appending let the earlier phases fill the
    cap: the 2026-08-31 sleep/functioning run had 25 stable-on + 29 taper
    outcome events, so the 50-item cap dropped ALL 62 post-cessation ("off")
    events — the synthesis had no quotable post-cessation statement while the
    manifest reported the phase counts.

    2026-09-04: delegates to utils.ordered_slice.round_robin_merge (same
    "don't let one group starve the others under a cap" fairness primitive
    core.insight.sweep.interleave_evidence_for_coverage uses for ISO-week
    buckets — here the grouping axis is PHASE, not time, so it calls the
    shared merge loop directly rather than window_fair_sample).
    """
    from utils.ordered_slice import round_robin_merge
    queues = [
        list(comparison.events) + list(comparison.proxy_events)
        for comparison in comparisons
    ]
    return round_robin_merge(queues)


def _window_scan_collection(chroma_store, collection_name, window, cap):
    """Chunks whose CONTENT date falls inside [start, end] (ISO strings).

    The date-range retrieval arm for windowed longitudinal specs: an explicit
    calendar window is a metadata question, not a similarity question. Scans
    the collection's metadata (small collections only — notes/facts, a few
    thousand chunks) and prefers note_date (content date) over index-time
    timestamps. Read-only; failures degrade to an empty list.

    2026-09-04: the implementation moved to
    ``core.insight.sweep.window_scan_collection`` (the theme-sweep date-range
    arm needed the SAME logic for conversations/summaries — single source of
    truth instead of a second copy); this name stays as a thin delegating
    alias so existing call sites/tests keep working unchanged.
    """
    from core.insight.sweep import window_scan_collection
    return window_scan_collection(chroma_store, collection_name, window, cap)


async def _run_insight_mode(ctx):
    """Insight / evidence-assembly mode (agentic gate insight_intent, 2026-08-23).

    Owns the turn: facet decomposition → UNGATED cross-store sweep (the memory
    gate's per-doc cosine test structurally cannot pass low-pairwise-similarity
    / high-collective-signal evidence sets) → provenance labeling → adversarial
    assessment (assessment kind only) → streamed synthesis → optional
    prewritten-document save. Does its own store_interaction dispatch and sets
    ctx.handled on success. On exception, logs and returns with ctx.handled
    False so the dispatcher falls through to agentic/enhanced.
    """
    orchestrator = ctx.orchestrator
    _intent_dict = dict(getattr(ctx.gate_decision, "insight_intent", None) or {})
    # Empty/malformed intent dict: fall through to the normal flow instead of
    # crashing InsightIntent validation ("2 validation errors for
    # InsightIntent: kind/theme Field required", 2026-08-28 logs).
    if not _intent_dict.get("kind") or not _intent_dict.get("theme"):
        logger.warning(
            f"[Handle Submit] INSIGHT MODE: intent missing kind/theme "
            f"({_intent_dict}) — falling through"
        )
        return
    logger.warning(f"[Handle Submit] INSIGHT MODE: {_intent_dict}")
    try:
        from core.agentic.gate import _tone_is_elevated
        from core.insight.assessor import assess
        from core.insight.facets import decompose
        from core.insight.provenance import label_evidence
        from core.insight.sweep import (
            exclude_current_request_evidence,
            interleave_evidence_for_coverage,
            run_sweep,
        )
        from core.insight.synthesizer import build_synthesis_prompts, synthesize_stream
        from core.insight.types import InsightIntent, EvidenceItem

        intent = InsightIntent(**_intent_dict)
        tone_level = (ctx.raw_context or {}).get("tone_level")
        tone_elevated = _tone_is_elevated(tone_level)

        _ms = getattr(orchestrator, "memory_system", None)
        _chroma = getattr(_ms, "chroma_store", None)
        _corpus = getattr(_ms, "corpus_manager", None)
        _graph = getattr(_ms, "graph_memory", None)
        _resolver = getattr(_ms, "entity_resolver", None)
        if _chroma is None:
            raise RuntimeError("insight mode requires a chroma_store")
        _expander = None
        try:
            from memory.memory_expander import MemoryExpander
            _expander = MemoryExpander(_chroma)
        except Exception:
            pass

        _is_pattern = intent.kind == "pattern_temporal"
        yield {"role": "assistant",
               "content": (
                   f"📈 Analyzing patterns across your history: {intent.theme}..."
                   if _is_pattern else
                   f"🔎 Sweeping your history for: {intent.theme}..."),
               "is_progress": True}

        # Keepalive wrapper: yield a heartbeat every 8s while a stage runs
        # (same discipline as the agentic loop — mobile clients time out on
        # silence, and the sweep alone may take tens of seconds).
        _KEEPALIVE_S = 8.0
        # Synthesis stream wall-clock ceiling: a healthy full report finishes
        # well under this; the 2026-08-31 degenerate stream ran 3.5 min and
        # was only stopped by the owner killing the process.
        _SYNTHESIS_STREAM_MAX_S = 240.0

        # (async generators can't yield from a helper — inline the loop per stage)
        async def _wait_stage(task):
            """Wait one keepalive interval; returns True when done."""
            _done, _ = await asyncio.wait({task}, timeout=_KEEPALIVE_S)
            return bool(_done)

        # Pattern stage (2026-08-29): deterministic engine FIRST — the counts
        # are computed, the LLM only narrates them. Read-only + sync → thread.
        _patterns = None
        _deliberation = None
        _pattern_evidence = []
        if _is_pattern:
            from core.insight.temporal import run_pattern_stage
            from core.insight.coordinator import (
                LongitudinalDeliberationCoordinator,
                normalize_chroma_rows,
            )
            _profile = getattr(_ms, "user_profile", None)

            # Email pattern prefetch (2026-09-01): the engine is sync and
            # never fetches — when the pattern theme carries an email cue,
            # fetch live headers HERE (async layer) and inject them as rows
            # (docs/EMAIL_INTEGRATION_DESIGN.md). Fail-soft: None = the
            # dimension reports "source not available" honestly.
            _email_rows = None
            try:
                import re as _re_email
                if _re_email.search(r"\b(?:e-?mails?|inbox|gmail|outlook)\b",
                                    intent.theme or "", _re_email.IGNORECASE):
                    from config.app_config import EMAIL_INTEGRATION_ENABLED
                    if EMAIL_INTEGRATION_ENABLED:
                        from core.email.service import get_email_service
                        _email_rows = await get_email_service().recent(
                            window_days=intent.window_days or 30, limit=200)
            except Exception as _e:
                logger.debug(f"[Insight Mode] email pattern prefetch skipped: {_e}")
                _email_rows = None
            _wsm = getattr(getattr(getattr(orchestrator, "prompt_builder", None), "context_gatherer", None), "web_search_manager", None)

            async def _web_adapter(q):
                if _wsm is None:
                    raise RuntimeError("web search manager unavailable")
                result = await _wsm.search(q, localize=False, max_results=5)
                if getattr(result, "error", None):
                    raise RuntimeError(str(result.error))
                return [{
                    "id": getattr(page, "url", ""),
                    "title": getattr(page, "title", ""),
                    "url": getattr(page, "url", ""),
                    "snippet": (getattr(page, "content", "") or getattr(page, "snippet", ""))[:1200],
                    "published_date": getattr(page, "published_date", None),
                    "source": getattr(page, "source", "web"),
                } for page in getattr(result, "pages", [])]

            async def _pubmed_adapter(q):
                from knowledge.pubmed_search import search_pubmed
                # Literature requests need enough breadth to expose adjacent
                # endpoints (aggression, agitation, behavior scales), not just
                # the first few keyword hits.
                return await search_pubmed(q, max_results=10)

            async def _arxiv_adapter(q):
                from knowledge.research_search import search_arxiv
                return await search_arxiv(q, max_results=5)

            async def _stackexchange_adapter(q):
                from knowledge.research_search import search_stackexchange
                return await search_stackexchange(q, max_results=5)

            async def _chroma_adapter(collection, channel, q, limit, window=None):
                rows = await asyncio.to_thread(
                    _chroma.query_collection, collection, q, limit,
                )
                normalized = normalize_chroma_rows(rows, channel=channel)
                if window:
                    # Date-range retrieval arm: semantic similarity is
                    # date-blind, so a windowed longitudinal spec also pulls
                    # chunks whose CONTENT date falls inside the frozen window
                    # (live 2026-08-31: 0 of 70 retrieved notes were in a
                    # six-month window while 200+ dated chunks existed).
                    dated = await asyncio.to_thread(
                        _window_scan_collection, _chroma, collection,
                        window, max(limit * 3, 30),
                    )
                    seen = {row.get("source_id") or row.get("id") for row in normalized}
                    for row in normalize_chroma_rows(dated, channel=channel):
                        key = row.get("source_id") or row.get("id")
                        if key not in seen:
                            seen.add(key)
                            normalized.append(row)
                return normalized

            async def _notes_adapter(q, window=None):
                return await _chroma_adapter("obsidian_notes", "notes", q, 20, window=window)

            async def _facts_adapter(q, window=None):
                return await _chroma_adapter("facts", "facts", q, 12, window=window)

            async def _wiki_adapter(q):
                return await _chroma_adapter("wiki_knowledge", "wiki", q, 6)

            async def _files_adapter(q):
                return await _chroma_adapter("reference_docs", "files", q, 8)

            _wolfram_manager = getattr(
                getattr(orchestrator, "_agentic_controller", None),
                "wolfram_manager", None,
            )
            if _wolfram_manager is None:
                try:
                    from config.app_config import WOLFRAM_ENABLED, WOLFRAM_APP_ID
                    if WOLFRAM_ENABLED and WOLFRAM_APP_ID:
                        from knowledge.wolfram_manager import WolframManager
                        _wolfram_manager = WolframManager()
                except Exception as _wolfram_init_error:
                    logger.debug(
                        f"[Insight] Wolfram adapter unavailable: {_wolfram_init_error}"
                    )

            async def _wolfram_adapter(q):
                result = await _wolfram_manager.query(q)
                if not result.success:
                    raise RuntimeError(result.error or "Wolfram query failed")
                return [{
                    "title": "Wolfram Alpha computation",
                    "text": _wolfram_manager.format_for_prompt(result),
                    "query": q,
                    "source": "Wolfram Alpha",
                }]

            _deliberation_adapters = {
                "notes": _notes_adapter,
                "facts": _facts_adapter,
                "pubmed": _pubmed_adapter,
                "web": _web_adapter,
                "wiki": _wiki_adapter,
                "files": _files_adapter,
                "arxiv": _arxiv_adapter,
                "stackexchange": _stackexchange_adapter,
            }
            if _wolfram_manager is not None:
                _deliberation_adapters["wolfram"] = _wolfram_adapter

            _coord = LongitudinalDeliberationCoordinator(
                corpus_manager=_corpus,
                adapters=_deliberation_adapters,
                model_manager=orchestrator.model_manager,
            )
            _deliberation_task = asyncio.ensure_future(_coord.run(intent.raw_query or intent.theme))
            _n = 0
            while not await _wait_stage(_deliberation_task):
                _n += 1
                yield {"role": "assistant",
                       "content": f"🔄 Planning and gathering the evidence set... ({int(_n * _KEEPALIVE_S)}s)",
                       "is_progress": True}
            _deliberation = _deliberation_task.result()

            # Secondary rolling aggregates use ONLY the already-frozen outcome
            # contract. If planning failed, do not guess terms from raw prose.
            if _deliberation.freeze.status == "ready" and _deliberation.freeze.spec is not None:
                if _deliberation.freeze.spec.analysis_kind == "time_series":
                    _pattern_task = asyncio.ensure_future(asyncio.to_thread(
                        run_pattern_stage, intent,
                        corpus_manager=_corpus, user_profile=_profile,
                        spec=_deliberation.freeze.spec,
                        email_rows=_email_rows,
                    ))
                    _n = 0
                    while not await _wait_stage(_pattern_task):
                        _n += 1
                        yield {"role": "assistant",
                               "content": f"🔄 Counting the frozen outcomes... ({int(_n * _KEEPALIVE_S)}s)",
                               "is_progress": True}
                    _patterns, _rolling_evidence = _pattern_task.result()
                    _pattern_evidence.extend(_rolling_evidence)
                else:
                    # Event/period comparisons are already computed over exact
                    # phase bounds by run_longitudinal_scan. A second rolling
                    # default window would select a different evidence set.
                    _patterns = []
            else:
                # Terminal fail-closed behavior applies to longitudinal
                # requests: never turn raw comparison prose into a computed
                # aggregate. Ordinary theme sweeps retain their established
                # locator behavior when no deliberation contract is needed.
                if intent.kind == "pattern_temporal":
                    _patterns = []
                else:
                    _fallback_pattern_task = asyncio.ensure_future(asyncio.to_thread(
                        run_pattern_stage, intent,
                        corpus_manager=_corpus, user_profile=_profile,
                        email_rows=_email_rows,
                    ))
                    _n = 0
                    while not await _wait_stage(_fallback_pattern_task):
                        _n += 1
                        yield {"role": "assistant",
                               "content": f"🔄 Recovering a broad pattern signal... ({int(_n * _KEEPALIVE_S)}s)",
                               "is_progress": True}
                    _patterns, _fallback_evidence = _fallback_pattern_task.result()
                    _pattern_evidence.extend(_fallback_evidence)

            if _deliberation.scan is not None:
                _seen_internal = set()
                for _event in _interleave_phase_events(
                        _deliberation.scan.comparisons):
                    _key = (_event.source_class, _event.source_id)
                    if _key in _seen_internal:
                        continue
                    _seen_internal.add(_key)
                    _pattern_evidence.append(EvidenceItem(
                        doc_id=_event.source_id,
                        text=_event.text,
                        date=_event.timestamp,
                        collection=("obsidian_notes" if _event.source_class in {"users-own-note", "assistant-summary"} else "corpus"),
                        speaker=_event.speaker,
                        stance_label=(
                            "users-own-note" if _event.source_class == "users-own-note"
                            else "assistant-inferred" if _event.source_class == "assistant-summary"
                            else "user-stated"
                        ),
                        facet="deliberation:phase-evidence",
                    ))
            # External research remains usable even when a fuzzy personal
            # anchor prevents the before/after phase scan from running.
            for _src in _deliberation.external_evidence:
                _pattern_evidence.append(EvidenceItem(
                    doc_id=str(_src.get("source_id") or _src.get("id") or _src.get("pmid") or ""),
                    text=str(
                        _src.get("snippet") or _src.get("abstract")
                        or _src.get("text") or _src.get("content")
                        or _src.get("document") or _src.get("title") or ""
                    )[:800],
                    date=_src.get("date") or _src.get("published_date"),
                    collection=str(_src.get("source_class") or "research"),
                    stance_label=(
                        "computed-evidence"
                        if _src.get("source_class") == "wolfram"
                        else "external-research"
                    ),
                    facet="deliberation:research",
                ))

        if (_deliberation is not None
                and _deliberation.freeze.status == "ready"
                and _deliberation.freeze.spec is not None):
            # The deliberation plan is the sole evidence selector. Convert its
            # frozen support/refute angles into the existing cross-store sweep
            # interface instead of decomposing the raw conversational request
            # a second time.
            from core.insight.types import FacetPlan, FacetQuery
            _spec = _deliberation.freeze.spec
            _facet_terms = (_spec.outcome_terms + _spec.behavioral_indicators)[:8]
            _facet_rows = []
            for _idx, _facet in enumerate(_spec.supporting_facets[:6]):
                _facet_rows.append(FacetQuery(
                    name=f"support-{_idx + 1}", query_text=_facet,
                    keywords=_facet_terms, entities=[],
                ))
            for _idx, _facet in enumerate(_spec.refuting_facets[:4]):
                _facet_rows.append(FacetQuery(
                    name=("counter-evidence" if _idx == 0 else f"refute-{_idx + 1}"),
                    query_text=_facet, keywords=_facet_terms, entities=[],
                ))
            plan = FacetPlan(
                facets=_facet_rows,
                claims=[claim.proposition for claim in _spec.claims],
                fallback=False,
            )
        elif _is_pattern and _deliberation is not None:
            # If the strict longitudinal planner is unavailable (for example,
            # a transient model timeout), keep the product useful by falling
            # back to the existing facet planner.  This is deliberately
            # marked as fallback: it can assemble evidence, but must not be
            # presented as a frozen phase/causal analysis.
            _fallback_plan_task = asyncio.ensure_future(
                decompose(intent, orchestrator.model_manager,
                          entity_resolver=_resolver)
            )
            _n = 0
            while not await _wait_stage(_fallback_plan_task):
                _n += 1
                yield {"role": "assistant",
                       "content": f"🔄 Recovering with a broad evidence sweep... ({int(_n * _KEEPALIVE_S)}s)",
                       "is_progress": True}
            plan = _fallback_plan_task.result()
            plan.fallback = True
        else:
            _plan_task = asyncio.ensure_future(
                decompose(intent, orchestrator.model_manager, entity_resolver=_resolver)
            )
            _n = 0
            while not await _wait_stage(_plan_task):
                _n += 1
                yield {"role": "assistant",
                       "content": f"🔄 Planning the sweep... ({int(_n * _KEEPALIVE_S)}s)",
                       "is_progress": True}
            plan = _plan_task.result()

        # Explicit ISO date window from the request ("from 2026-07-15 through
        # today") — theme-sweep only; pattern_temporal/deliberation own their
        # own windowing (window_days / the frozen phase spec) and never set
        # intent.date_window.
        _date_window = (
            tuple(intent.date_window) if len(intent.date_window) == 2 else None
        )
        if plan.facets:
            _sweep_task = asyncio.ensure_future(run_sweep(
                plan, chroma_store=_chroma, corpus_manager=_corpus,
                graph_memory=_graph, entity_resolver=_resolver,
                memory_expander=_expander,
                request_text=intent.raw_query or intent.theme,
                date_window=_date_window,
            ))
            _n = 0
            while not await _wait_stage(_sweep_task):
                _n += 1
                yield {"role": "assistant",
                       "content": f"🔄 Sweeping stores... ({int(_n * _KEEPALIVE_S)}s)",
                       "is_progress": True}
            evidence = _sweep_task.result()
        else:
            evidence = []
        evidence = label_evidence(evidence)
        # Self-reference exclusion: the sweep can surface the current
        # request's own turn or a prior exchange discussing it — those are
        # not history (2026-09-04 live incident: 7 of 37 rendered items were
        # the request itself / the reply about it; round 2 same day: the
        # PREVIOUS day's near-identical request turns also survived because
        # they overlapped below the any-day 60% bar — current_turn_date is
        # now threaded through so the same-day-tightened bar can engage).
        from datetime import datetime as _insight_now
        evidence = exclude_current_request_evidence(
            evidence, intent.raw_query or intent.theme,
            current_turn_date=_insight_now.now().isoformat(),
        )
        if _is_pattern:
            # Engine exemplars are already provenance-labeled; join after
            # label_evidence so their stance_labels are preserved. Put frozen
            # contract/recovery evidence first so the generic sweep cannot
            # crowd it out of the synthesis prompt's bounded evidence block.
            evidence = _pattern_evidence + evidence

        # The generic adapters and the cross-store sweep can surface the same
        # source. Keep one prompt item while preserving first-seen provenance.
        _deduped_evidence = []
        _seen_evidence = set()
        _query_tokens = {
            _token for _token in _re.findall(r"[a-z0-9']+", (intent.raw_query or "").lower())
            if len(_token) > 2
        }
        for _item in evidence:
            # Assistant-authored summaries can help locate originals, but are
            # not independent evidence for a personal longitudinal analysis.
            # Keeping them here caused prior runs to report the model's own
            # failed answers as if they were Luke's history.
            if _is_pattern and (
                _item.speaker == "assistant"
                or _item.stance_label == "assistant-inferred"
            ):
                continue
            # Conversation indexing can return clipped copies of the current
            # request itself. They are not historical observations. Suppress
            # only high-overlap, substantial excerpts so ordinary records
            # mentioning the same medication/topic remain eligible.
            if _is_pattern and _query_tokens:
                _item_tokens = {
                    _token for _token in _re.findall(r"[a-z0-9']+", (_item.text or "").lower())
                    if len(_token) > 2
                }
                _overlap = (
                    len(_query_tokens & _item_tokens) / max(1, len(_item_tokens))
                )
                if len((_item.text or "").split()) >= 35 and _overlap >= 0.72:
                    continue
            _evidence_key = (
                # Same excerpts often arrive through several collections and
                # with different IDs; content identity is the useful key.
                " ".join((_item.text or "").split()).lower()[:1200],
            )
            if _evidence_key in _seen_evidence:
                continue
            _seen_evidence.add(_evidence_key)
            _deduped_evidence.append(_item)
            if _is_pattern and len(_deduped_evidence) >= 50:
                break
        evidence = _deduped_evidence

        # Window-fair rendering (2026-09-04): reorder for the eventual
        # render_evidence_block cap so a long request window survives
        # truncation instead of collapsing to the newest few days. Theme
        # sweeps only — pattern_temporal narrates a computed aggregate and
        # deliberation owns its own frozen manifest ordering; reordering
        # either would fight the numbers they restate.
        if intent.kind == "theme_sweep":
            evidence = interleave_evidence_for_coverage(evidence)

        _stores = {e.collection for e in evidence}
        yield {"role": "assistant",
               "content": (f"🗂️ Found {len(evidence)} pieces of evidence across "
                           f"{len(_stores)} stores — assembling..."),
               "is_progress": True}

        assessment = None
        if intent.kind == "insight_assessment":
            _assess_task = asyncio.ensure_future(assess(
                plan.claims or [intent.theme], evidence, orchestrator.model_manager,
            ))
            _n = 0
            while not await _wait_stage(_assess_task):
                _n += 1
                yield {"role": "assistant",
                       "content": f"🔄 Checking the claim against the record... ({int(_n * _KEEPALIVE_S)}s)",
                       "is_progress": True}
            assessment = _assess_task.result()

        model_name = getattr(
            orchestrator.model_manager, 'get_active_model_name', lambda: None
        )()

        async def _synthesis_with_keepalive(stream):
            """Yield synthesis text while keeping the GUI connection alive.

            Runaway watchdog (2026-08-31): a degenerate kimi-3 stream looped
            garbage for ~3.5 min — chunks kept arriving so the keepalive
            timer never fired and nothing bounded it. Wall-clock ceiling +
            periodic degenerate-shape check; a tripped watchdog yields a
            terminal {"kind": "runaway"} event and stops consuming.
            """
            _stream_iter = stream.__aiter__()
            _elapsed = 0
            _accum = ""
            _checked_at = 0
            _started = _time.monotonic()
            while True:
                _next_task = asyncio.ensure_future(_stream_iter.__anext__())
                try:
                    while True:
                        _done, _ = await asyncio.wait({_next_task}, timeout=_KEEPALIVE_S)
                        if _done:
                            break
                        _elapsed += int(_KEEPALIVE_S)
                        yield {"kind": "progress", "seconds": _elapsed}
                    try:
                        _piece = _next_task.result()
                    except StopAsyncIteration:
                        break
                    _accum += _piece
                    if _time.monotonic() - _started > _SYNTHESIS_STREAM_MAX_S:
                        yield {"kind": "runaway", "reason": "duration"}
                        return
                    if len(_accum) - _checked_at > 2000:
                        _checked_at = len(_accum)
                        if ResponseParser.looks_degenerate_stream(_accum):
                            yield {"kind": "runaway", "reason": "degenerate"}
                            return
                    yield {"kind": "text", "value": _piece}
                finally:
                    if not _next_task.done():
                        _next_task.cancel()

        _buffer = ""
        _primary_stream = synthesize_stream(
            intent, evidence, assessment,
            model_manager=orchestrator.model_manager,
            model_name=model_name,
            tone_elevated=tone_elevated,
            patterns=_patterns,
            deliberation_manifest=(
                _deliberation.manifest if _deliberation is not None else None
            ),
            disable_reasoning=True,
        )
        _runaway_reason = None
        async for _event in _synthesis_with_keepalive(_primary_stream):
            if _event["kind"] == "progress":
                yield {"role": "assistant", "content":
                       f"🔄 Still assembling the report... ({_event['seconds']}s)",
                       "is_progress": True}
            elif _event["kind"] == "runaway":
                _runaway_reason = _event.get("reason") or "runaway"
                break
            else:
                _buffer += _event["value"]
                yield {"role": "assistant", "content":
                       ResponseParser.strip_trailing_stream_artifact(_buffer)}
        if _runaway_reason:
            logger.error(
                f"[Insight Mode] Synthesis stream aborted by watchdog "
                f"({_runaway_reason}); {len(_buffer)} chars discarded, nothing stored"
            )
            ctx.handled = True
            yield {"role": "assistant", "content":
                   "⚠️ I stopped the report mid-stream — the model's output "
                   "went off the rails ("
                   + ("it exceeded the time ceiling" if _runaway_reason == "duration"
                      else "it started repeating garbage")
                   + "). Nothing from this attempt was stored. "
                   "Please rerun the same query."}
            return

        final_text = _sanitize_response_text(_buffer).strip()
        if not final_text:
            # Reasoning-capable models can occasionally spend the entire
            # synthesis turn in reasoning and emit no visible text. Retry once
            # with the provider's explicit reasoning off-switch before letting
            # the dispatcher fall through to an unrelated agentic search.
            yield {"role": "assistant", "content":
                   "🔄 Synthesis returned no visible text; retrying without extended reasoning...",
                   "is_progress": True}
            _retry_buffer = ""
            _retry_stream = synthesize_stream(
                intent, evidence, assessment,
                model_manager=orchestrator.model_manager,
                model_name=model_name,
                tone_elevated=tone_elevated,
                patterns=_patterns,
                deliberation_manifest=(
                    _deliberation.manifest if _deliberation is not None else None
                ),
                disable_reasoning=True,
            )
            async for _event in _synthesis_with_keepalive(_retry_stream):
                if _event["kind"] == "progress":
                    yield {"role": "assistant", "content":
                           f"🔄 Still assembling the retry... ({_event['seconds']}s)",
                           "is_progress": True}
                elif _event["kind"] == "runaway":
                    logger.error(
                        f"[Insight Mode] Retry synthesis aborted by watchdog "
                        f"({_event.get('reason')}); nothing stored"
                    )
                    raise RuntimeError(
                        "insight synthesis retry aborted: runaway stream"
                    )
                else:
                    _retry_buffer += _event["value"]
                    yield {"role": "assistant", "content":
                           ResponseParser.strip_trailing_stream_artifact(_retry_buffer)}
            final_text = _sanitize_response_text(_retry_buffer).strip()
            if not final_text:
                raise RuntimeError("insight synthesis returned no visible content after reasoning-off retry")

        # Optional document save. Explicit wants_document requests save when no
        # assessment gates them; an assessment gates on agree/partial (fail-
        # honest: never hand the user a document the record disputes). An
        # assessment run without an explicit doc request also saves on
        # agreement when doc_on_agreement is enabled (goal 2's contract).
        from config.app_config import INSIGHT_DOC_ON_AGREEMENT
        _save_doc = (
            (intent.wants_document
             and (assessment is None or assessment.allows_document))
            or (assessment is not None and INSIGHT_DOC_ON_AGREEMENT
                and assessment.allows_document)
        )
        doc_line = ""
        if _save_doc:
            try:
                from knowledge.document_generator import DocumentGenerator
                _dg = DocumentGenerator(model_manager=orchestrator.model_manager)
                _doc = _dg.save_prewritten(
                    final_text,
                    topic=intent.theme,
                    doc_type="summary",
                    source_types=sorted(_stores),
                )
                doc_line = f"\n\n📄 Saved as **{_doc.title}** → `{_doc.path}`"
            except Exception as _doc_err:
                logger.warning(f"[Insight Mode] Document save failed: {_doc_err}")
                doc_line = "\n\n(I couldn't save the document to disk — the text above is the full content.)"
        elif intent.wants_document and assessment is not None and not assessment.allows_document:
            doc_line = (
                "\n\n(I held off on saving a document: the record doesn't "
                "support this strongly enough to put it in writing yet — "
                "the honest read is above.)"
            )

        _insight_session_id = _get_session_id(orchestrator)
        _prov = _build_provenance(
            "insight-assembly", _insight_session_id, model_name, [],
            insight_kind=intent.kind,
            insight_theme=intent.theme[:120],
            evidence_count=len(evidence),
            evidence_stores=sorted(_stores),
            assessment_overall=(assessment.overall if assessment else None),
            document_saved=bool(_save_doc and "Saved as" in doc_line),
        )
        # Audit F19 (2026-08-31): pass the same patterns/manifest kwargs the
        # real synthesize_stream calls pass — the debug record understated
        # the sent prompt by up to 14K chars without them.
        _syn_system, _syn_prompt = build_synthesis_prompts(
            intent, evidence, assessment, tone_elevated=tone_elevated,
            patterns=_patterns,
            deliberation_manifest=(
                _deliberation.manifest if _deliberation is not None else None
            ),
        )
        prompt_tokens, system_tokens, total_tokens = _safe_count_tokens(
            _syn_prompt, _syn_system, model_name, orchestrator,
        )
        debug_record = _build_debug_record(
            mode='insight-assembly', user_text=ctx.user_text, prompt=_syn_prompt,
            system_prompt=_syn_system, response=final_text, model=model_name,
            prompt_tokens=prompt_tokens, system_tokens=system_tokens,
            total_tokens=total_tokens, citations=[], orchestrator=orchestrator,
            provenance=_prov,
            gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
        )
        yield {"role": "assistant", "content": final_text + doc_line,
               "debug": debug_record}

        _dispatch_storage(
            orchestrator, ctx.merged_input, final_text, ctx.user_text,
            final_text, ctx.personality, ctx.file_names, ctx.conversation_logger,
            _insight_session_id, _prov, 'insight-assembly',
        )
        _write_turn_telemetry(
            ctx, 'insight-assembly', _insight_session_id, model_name,
            len(final_text or ""), response_text=final_text,
        )
        ctx.handled = True
        ctx.storage_dispatched = True
        return

    except Exception as e:
        logger.error(f"[Handle Submit] Insight mode failed: {e}")
        import traceback
        traceback.print_exc()
        # Insight requests are a distinct, evidence-sensitive workflow. Do
        # not silently replace a failed analysis with an unrelated agentic
        # answer; surface the bounded failure and keep provenance honest.
        ctx.handled = True
        yield {"role": "assistant", "content":
               "I couldn't complete the insight synthesis on this attempt. "
               "The retrieval work was not converted into a final report; "
               "please rerun this same query."}


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

    # Debug record (2026-09-04): same gap as _run_doc_generation — this
    # bypass path yielded the final chunk with no "debug" key, so a
    # self-note-only session never produced a debug_records entry.
    _note_model = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
    debug_record = _build_debug_record(
        mode='self-note', user_text=ctx.user_text, prompt=body,
        system_prompt=None, response=_resp, model=_note_model,
        prompt_tokens=0, system_tokens=0, total_tokens=0,
        citations=[], orchestrator=orchestrator,
        gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
    )
    yield {"role": "assistant", "content": _resp, "debug": debug_record}
    _write_turn_telemetry(
        ctx, 'self-note', _get_session_id(orchestrator),
        _note_model,
        len(_resp or ""),
    )
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


def _apply_web_citations(text, web_map, wiki_map=None):
    """Make [WEB_N]/[WIKI_N] citations clickable + append a Sources footer (display only).

    gr.Chatbot renders markdown, so each inline [WEB_N] is rewritten to
    `[[WEB_N](url)]` — literal brackets around a clickable "WEB_N" pointing at the
    source URL — and a `**Sources:**` list is appended. Markers with no URL in
    web_map are STRIPPED from the display (with their preceding space): on a
    turn with no web search the model can still imitate [WEB_N] from replayed
    history, and leaving it renders as literal bracket junk to the user.
    [WIKI_N] markers (agentic Wikipedia results) are handled identically via
    wiki_map — article-URL links + Sources entries. Applied to the DISPLAY
    string only; the stored response keeps the canonical markers.

    NOTE: do NOT use the `[\\[WEB_N\\]](url)` escaped-bracket form — this chatbot
    registers `\\[ ... \\]` as a LaTeX display-math delimiter (see gr.Chatbot
    latex_delimiters in gui/launch.py), so backslash-escaped brackets render as
    math, not a link. The `[[WEB_N](url)]` form has no backslashes and is safe.
    """
    if not text:
        return text
    # Idempotency guard: the linkified form [[WEB_N](url)] still contains the literal
    # substring [WEB_N], so a second pass would re-wrap it ([[[WEB_N](url)](url)]) and
    # re-append Sources. If any marker is already linkified, this text is done — return it.
    if _re.search(r'\[\[(?:WEB|WIKI)_\d+\]\(', text):
        return text

    out = text
    footer = []
    for prefix, src_map in (("WEB", web_map or {}), ("WIKI", wiki_map or {})):
        cited = sorted(set(_re.findall(r'\[' + prefix + r'_(\d+)\]', out)), key=int)
        if not cited:
            continue

        def _repl(m, _prefix=prefix, _map=src_map):
            key = f"{_prefix}_{m.group(2)}"
            url = ((_map.get(key) or {}).get("url") or "").strip()
            return f"{m.group(1)}[[{key}]({url})]" if url else ""

        out = _re.sub(r'( ?)\[' + prefix + r'_(\d+)\]', _repl, out)

        for _n in cited:
            key = f"{prefix}_{_n}"
            src = src_map.get(key)
            if src and (src.get("url") or "").strip():
                title = src.get("title") or src["url"]
                if prefix == "WIKI":
                    title = f"Wikipedia: {title}"
                footer.append(f"[{key}] [{title}]({src['url']})")
    if footer:
        out += "\n\n---\n**Sources:**\n" + "\n".join(footer)
    return out


# Map the action system's ActionType (user-intent classifier) to the guard's
# coarser ActionKind, so we can tell when the USER actually asked Daemon to
# perform an external action this turn.
def _user_requested_external_kinds(user_text):
    """External ActionKinds the user explicitly asked Daemon to perform this turn.

    Empty when the user merely narrated/reported their own action ("I sent it",
    "no bounce back yet") — the case where an external completion phrase in the
    reply is affirmation, not Daemon confabulating.
    """
    try:
        from core.actions.registry import detect_action_intent
        from core.actions.types import ActionType
        from core.action_claim_guard import ActionKind
        at = detect_action_intent(user_text or "")
        if at is None:
            return set()
        mapping = {
            ActionType.SEND_EMAIL: ActionKind.EMAIL,
            ActionType.CALENDAR_CREATE_EVENT: ActionKind.CALENDAR,
            ActionType.SEND_TELEGRAM: ActionKind.MESSAGE,
            ActionType.SEND_DISCORD: ActionKind.MESSAGE,
            ActionType.GITHUB_CREATE_ISSUE: ActionKind.GITHUB,
            ActionType.GITHUB_COMMENT_PR: ActionKind.GITHUB,
        }
        k = mapping.get(at)
        return {k} if k is not None else set()
    except Exception:
        return set()


def _pending_proposal_kinds(orchestrator):
    """ActionKinds with a prior-turn offer still pending ("Want me to email X?")."""
    try:
        store = _get_pending_proposal_store(orchestrator)
        p = store.peek() if store is not None else None
        return {p.kind} if p is not None else set()
    except Exception:
        return set()


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
            ActionKind, build_correction_notice, detect_completion_claims,
            is_first_person_claim, verify_claims,
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

        # Structural gate (anti false-positive): only correct an unsent EXTERNAL
        # action when Daemon was genuinely in a position to act — the user asked
        # for it this turn, an offer of that kind is pending, or the claim is a
        # first-person self-assertion ("I sent it"). A passive/ambiguous external
        # phrase with no such context ("the email's sent — you fixed the address")
        # is the user narrating their OWN action, not Daemon confabulating.
        actionable = _user_requested_external_kinds(ctx.user_text) | _pending_proposal_kinds(ctx.orchestrator)
        external = [
            a for a in rec.external_unbacked
            if a.kind not in set(proposed_kinds)
            and (a.kind in actionable or is_first_person_claim(a.matched_text))
        ]
        suffix += build_correction_notice(external)
    except Exception as e:
        logger.warning(f"[ActionGuard] Claim guard failed (non-fatal): {e}")
    return suffix


async def _apply_grounding_check(ctx, response_text, source_material: str = ""):
    """Factual-grounding floor (2026-08-28): deterministic claim-shape
    pre-filter → LLM verifier → correction.

    Runs on ALL tones — the plan/review gate is skipped on CONCERN+ (no plan
    → no review), which is exactly where the refrigerator-mother endorsement
    shipped. Fail-open everywhere: any failure returns (None, "") and never
    blocks the shown response beyond the final-chunk window.

    Returns (revised_text, suffix):
    - (revised, "")  — 2026-08-29 integration path: the correction is woven
      INTO the response by a bounded rewrite. Caller must replace BOTH the
      display text and final_output with `revised` (the final yield is a
      whole-bubble replacement on every path, so display == storage holds).
    - (None, suffix) — fallback: append suffix to display AND final_output.
    - (None, "")     — no action.
    - (response_text, "") — 2026-09-04 GROUNDING_MODE=="log_only" (the
      default): the full prefilter+verifier+demotion pipeline ran and a
      flagged verdict was recorded to telemetry (grounding_mode,
      grounding_verdict) and logged at WARNING, but the response is returned
      UNCHANGED — same class as the 2026-08-28 review-gate LOG-ONLY fix
      (>=9 documented false corrections, 0 documented true, in the window
      that motivated it). This is a deliberate no-op for the caller (the
      value equals what was passed in), never an integrated/suffixed
      correction. Set grounding_check.mode: correct in config to restore
      the pre-09-04 shipped-correction behavior.

    source_material: text the assistant retrieved while answering (agentic
    tool-round results). The verifier treats it like user-pasted material —
    without it, correct document-derived facts get flagged against the
    verifier model's own priors (live 2026-08-29: "Fall 2026" at conf 0.9).
    """
    _no_action = (None, "")
    try:
        from config.app_config import (
            GROUNDING_CHECK_ENABLED, GROUNDING_CHECK_MODEL,
            GROUNDING_CONFIDENCE_THRESHOLD, GROUNDING_TIMEOUT_S,
            GROUNDING_MAX_TOKENS, GROUNDING_MIN_RESPONSE_CHARS,
            GROUNDING_INTEGRATE_ENABLED, GROUNDING_INTEGRATE_TIMEOUT_S,
            GROUNDING_INTEGRATE_MAX_RESPONSE_CHARS,
        )
        if (not GROUNDING_CHECK_ENABLED or not response_text
                or len(response_text.strip()) < GROUNDING_MIN_RESPONSE_CHARS):
            return _no_action
        from core.grounding_check import (
            has_checkable_claims, verify_grounding, build_grounding_correction,
            integrate_grounding_correction,
        )
        if not has_checkable_claims(response_text, ctx.user_text or ""):
            return _no_action
        ctx.telemetry["grounding_prefilter_fired"] = True

        mm = getattr(ctx.orchestrator, "model_manager", None)
        if mm is None:
            return _no_action
        ctx.telemetry["grounding_verifier_fired"] = True
        # The runtime clock is source data, not something the verifier should
        # reconstruct from model priors. Put it FIRST so source truncation can
        # never drop it behind a long retrieved document.
        from datetime import datetime as _grounding_datetime
        _runtime_now = _grounding_datetime.now().astimezone()
        _runtime_source = (
            "[AUTHORITATIVE RUNTIME CLOCK]\n"
            f"Current time: {_runtime_now.strftime('%A, %Y-%m-%d %H:%M:%S %Z')}"
        )
        _grounding_source = _runtime_source
        if source_material:
            _grounding_source += "\n\n" + str(source_material)
        verdict = await verify_grounding(
            ctx.user_text or "", response_text, mm,
            model_name=GROUNDING_CHECK_MODEL,
            max_tokens=GROUNDING_MAX_TOKENS,
            timeout_s=GROUNDING_TIMEOUT_S,
            source_material=_grounding_source,
        )
        if verdict is None:
            return _no_action  # fail-open: timeout / call failure / unparseable
        ctx.telemetry["grounding_flagged"] = bool(verdict.false_claim_present)
        ctx.telemetry["grounding_confidence"] = round(float(verdict.confidence), 3)
        if not (verdict.false_claim_present
                and verdict.correction.strip()
                and verdict.confidence >= GROUNDING_CONFIDENCE_THRESHOLD):
            if verdict.false_claim_present:
                # Observability: without this line a flagged-but-suppressed
                # verdict leaves no trace of what the verifier wanted to say
                # or which gate stopped it (2026-09-01 live-verification gap).
                logger.info(
                    "[GroundingCheck] Flagged verdict suppressed "
                    f"(conf={verdict.confidence:.2f} < {GROUNDING_CONFIDENCE_THRESHOLD}"
                    f" or empty correction): {verdict.correction[:120]!r}"
                )
            return _no_action

        # Live-config doctrine: read at call time — GROUNDING_MODE is a
        # module attr tests monkeypatch, and a module-level `from` import
        # would freeze the pre-patch value.
        from config.app_config import GROUNDING_MODE
        from utils.privacy_redaction import redact_text

        ctx.telemetry["grounding_mode"] = GROUNDING_MODE
        _redacted_verdict = redact_text(verdict.correction)[:300]
        ctx.telemetry["grounding_verdict"] = _redacted_verdict

        if GROUNDING_MODE == "log_only":
            # 2026-09-04: same class as the 2026-08-28 review-gate LOG-ONLY
            # fix — telemetry over the window showed 42 verifier fires -> 27
            # flags -> 25 shipped corrections, >=9 documented false, 0
            # documented true. Run the full prefilter+verifier+demotion
            # pipeline (so precision can still be measured) but never touch
            # the shown/stored response.
            logger.warning(
                f"[Grounding] LOG-ONLY flagged conf={verdict.confidence:.2f} "
                f": {_redacted_verdict}"
            )
            return (response_text, "")

        from core.agentic.gate import _tone_is_elevated
        elevated = _tone_is_elevated((ctx.raw_context or {}).get("tone_level"))
        logger.warning(
            f"[GroundingCheck] Correcting false claim "
            f"(conf={verdict.confidence:.2f}, elevated={elevated}): "
            f"{verdict.claim[:120]!r}"
        )
        # Audit F24 (2026-08-31): grounding_corrected records what SHIPPED —
        # it is set only once a correction (integrated or suffix) is actually
        # returned, never before the integrate attempt.
        if GROUNDING_INTEGRATE_ENABLED:
            revised = await integrate_grounding_correction(
                response_text, verdict, mm,
                model_name=GROUNDING_CHECK_MODEL,
                timeout_s=GROUNDING_INTEGRATE_TIMEOUT_S,
                max_response_chars=GROUNDING_INTEGRATE_MAX_RESPONSE_CHARS,
            )
            if revised:
                ctx.telemetry["grounding_integrated"] = True
                ctx.telemetry["grounding_corrected"] = True
                return (revised, "")
        _suffix = build_grounding_correction(verdict.correction, elevated=elevated)
        if _suffix:
            ctx.telemetry["grounding_corrected"] = True
        return (None, _suffix)
    except Exception as e:
        logger.warning(f"[GroundingCheck] failed (non-fatal): {e}")
        return _no_action


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


def _retry_fetch_urls_from_context(user_text, chat_history) -> list[str]:
    """Recover a URL only for an explicit retry after the assistant failed to fetch."""
    from utils.query_checker import is_retry_continuation
    if not is_retry_continuation(user_text):
        return []
    # Both production callers append the CURRENT turn before this runs (SPA:
    # the user message; legacy Gradio: user + an "…" typing placeholder), so a
    # naive last-user/last-assistant read tests the retry message itself and
    # the placeholder (audit F1 — digest-order class). Work over a view with
    # the current turn and placeholders removed, and pair the failure reply
    # with the user message that PRECEDED it.
    current_norm = " ".join(str(user_text or "").split())
    messages = []
    for m in (chat_history or []):
        if not isinstance(m, dict):
            continue
        content = str(m.get("content", "")).strip()
        role = m.get("role")
        if role == "assistant" and content in {"", "…", "..."}:
            continue
        if role == "user" and " ".join(content.split()) == current_norm:
            continue
        messages.append(m)
    last_assistant_idx = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if messages[i].get("role") == "assistant"), None,
    )
    if last_assistant_idx is None:
        return []
    assistant_text = str(messages[last_assistant_idx].get("content", "")).lower()
    markers = (
        "won't load", "wont load", "blank page", "couldn't fetch", "can't fetch",
        "could not fetch", "failed to fetch", "can't open the link", "couldn't open the link",
    )
    if not any(marker in assistant_text for marker in markers):
        return []
    previous_user = next(
        (messages[i] for i in range(last_assistant_idx - 1, -1, -1)
         if messages[i].get("role") == "user"), None,
    )
    import re
    urls = re.findall(r'https?://[^\s<>"\')\]]+', str((previous_user or {}).get("content", "")))
    return urls


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
    logger.warning("[Handle Submit] AGENTIC SEARCH MODE - routing through agentic controller")
    try:
        from core.agentic import AgenticSearchController, ProgressEvent

        # Get the agentic controller from orchestrator
        agentic_controller = orchestrator.agentic_controller
        if agentic_controller is None:
            # Lazy construction can fail when an optional dependency is
            # unavailable.  Do not turn that into a misleading NoneType
            # traceback; leave ctx unhandled so the dispatcher can choose its
            # documented fallback and record the reason.
            logger.error(
                "[Handle Submit] Agentic controller unavailable; "
                "falling through with reason=controller_init_failed"
            )
            ctx.agentic_fallback_reason = "controller_init_failed"
            # Audit F22 (2026-08-31): land the reason in turn telemetry —
            # the attribute alone was write-only.
            ctx.telemetry["agentic_fallback_reason"] = "controller_init_failed"
            return
        model_name = orchestrator.model_manager.get_active_model_name()

        # Get initial search terms from the trigger decision we already have
        initial_terms = search_terms if search_terms else []
        logger.debug(f"[Handle Submit] Agentic initial terms: {initial_terms}")

        # Extract URLs from the user message for direct fetch
        import re as _re_url
        _url_pattern = _re_url.compile(r'https?://[^\s<>"\')\]]+')
        _url_in_current_msg = _url_pattern.findall(user_text)
        _retry_recovered_urls = _retry_fetch_urls_from_context(user_text, history)
        _extracted_urls = _url_in_current_msg or _retry_recovered_urls
        from utils.topic_manager import _TOPIC_URL_RE
        from core.actions.registry import detect_action_intent
        from config.app_config import AGENTIC_FETCH_FASTPATH
        _remainder_words = len(_TOPIC_URL_RE.sub("", user_text).split())
        _gate_modes = getattr(_gate_decision, "modes", []) or []
        _forced_action = detect_action_intent(user_text)
        _fastpath_ok = (
            AGENTIC_FETCH_FASTPATH
            and ((bool(_url_in_current_msg) and _remainder_words <= 12)
                 or (bool(_retry_recovered_urls) and _remainder_words <= 25))
            and not any(mode in _gate_modes for mode in ("memory", "computation", "knowledge"))
            and not getattr(_gate_decision, "doc_gen_intent", None)
            and not getattr(_gate_decision, "self_note_intent", None)
            and not getattr(_gate_decision, "insight_intent", None)
            and not _forced_action
        )

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
            fetch_fastpath=_fastpath_ok,
            gate_modes=_gate_modes,
        )

        async def _agentic_next():
            try:
                return await _agentic_gen.__anext__(), False
            except StopAsyncIteration:
                return None, True

        _KEEPALIVE_S = 8.0
        _keepalive_n = 0
        # Degenerate-stream watchdog (2026-09-01): shape check only — the
        # agentic loop legitimately runs minutes of rounds before the first
        # response chunk, so a wall-clock arm here would discard real answers.
        _degenerate_check_threshold = 2000
        _last_degenerate_check_len = 0

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

                # Degenerate-stream watchdog: abort if the stream entered a
                # repeating-garbage loop (same detection as enhanced mode).
                if len(agentic_response) - _last_degenerate_check_len > _degenerate_check_threshold:
                    _last_degenerate_check_len = len(agentic_response)
                    if ResponseParser.looks_degenerate_stream(agentic_response):
                        logger.error(
                            "[Agentic] Stream aborted by watchdog (degenerate); "
                            f"{len(agentic_response)} chars discarded, nothing stored"
                        )
                        yield {"role": "assistant", "content":
                               "⚠️ The agentic search's synthesis became repetitive and "
                               "incoherent. I stopped the generation — nothing was stored. "
                               "Please try again."}
                        ctx.handled = True
                        return

                # Fail fast on a classified API-error payload at the stream
                # head — never render raw error JSON into the bubble (same
                # guard as the enhanced path, added 2026-08-21).
                from models.model_manager import API_ERROR_PREFIXES as _api_err_prefixes
                if agentic_response.lstrip().startswith(_api_err_prefixes):
                    logger.warning("[Handle Submit] Agentic stream head is an API-error payload — suppressing raw display")
                    break
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

        # Classified API-error payload → friendly display, no storage dispatch
        # (2026-08-21; the storage-time guard would also skip it, but the
        # enhanced path's early-return semantics apply here too).
        _agentic_friendly = _friendly_api_error(final_output)
        if _agentic_friendly:
            logger.warning("[Handle Submit] Agentic API error detected — showing friendly message")
            yield {"role": "assistant", "content": _agentic_friendly}
            ctx.handled = True
            return

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
        display_output = ResponseParser.strip_trailing_stream_artifact(display_output)  # edge <|sep|> + trailing-'e' (2026-08-22)
        # kimi-3 lone-'e' stream artifact: storage strips it via
        # sanitize_for_storage, but this display path never did — the stray
        # 'e' showed in the chat bubble + debug record (seen live 2026-08-14
        # on an agentic turn; the enhanced path sanitizes, this one didn't).
        # final_output feeds the debug record, display_output the chat bubble.
        final_output = ResponseParser.strip_trailing_stream_artifact(final_output)
        display_output = ResponseParser.strip_trailing_stream_artifact(display_output)

        # If agentic loop ran but no tools were actually dispatched (model
        # just narrated what it would do), strip bare tool-call-like lines
        # that leaked as plain text (e.g. "list_repos", "Lines added...")
        # ── Narration-shaped final recovery (2026-08-29): the synthesis call
        # shipped "let me grab the full text back out of memory…" as the final
        # reply — a plan, not an answer (the 08-28 promissory guards covered
        # only the decision-answer REUSE path). One bounded no-reasoning
        # retry; the final chunk below is a whole-bubble replacement yield, so
        # a recovery replaces what the user briefly saw AND what gets stored.
        try:
            if agentic_controller.narration_shaped_final(display_output):
                logger.warning(
                    "[Handle Submit] Agentic final response is narration-shaped "
                    f"({len(display_output)} chars) — attempting recovery")
                _narr_recovered = await agentic_controller.regenerate_final_answer()
                if _narr_recovered:
                    display_output = _narr_recovered
                    final_output = _narr_recovered
                    ctx.telemetry["agentic_narration_recovered"] = True
        except Exception as _narr_err:
            logger.warning(f"[Handle Submit] Narration recovery failed (non-fatal): {_narr_err}")

        _agentic_session = getattr(
            getattr(orchestrator, '_agentic_controller', None),
            '_last_session', None
        )
        _had_real_rounds = (
            _agentic_session
            and hasattr(_agentic_session, 'rounds')
            and len(_agentic_session.rounds) > 0
        )
        if (not _had_real_rounds and display_output
                and not ctx.telemetry.get("agentic_narration_recovered")):
            # Response is just narration — strip lines that look like
            # bare tool queries (short lines without sentence structure)
            # But preserve [propose_action] blocks for text parsing
            # (a narration-recovered answer skips this: it's a vetted full
            # reply, and its markdown table rows must not be line-stripped)
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
                        or _stripped_line.startswith(('#', '-', '*', '>', '{', '"', '|'))):
                    _cleaned_lines.append(_line)
                else:
                    logger.debug(f"[Handle Submit] Stripped bare tool-call line: {_stripped_line!r}")
            display_output = '\n'.join(_cleaned_lines).strip()
            display_output = _re.sub(r'\n{3,}', '\n\n', display_output)

        # Make [WEB_N] citations clickable + append a Sources footer (display only).
        # The accumulated web-source map lives on the controller's ToolExecutor
        # (assign_web_ids/_merge_web_ids write it there across rounds), NOT on the
        # controller itself — read it from _tool_executor or the linkify no-ops and
        # [WEB_N] render as plain text. (Controller attr kept as a legacy/mock fallback.)
        _web_map = (
            getattr(getattr(agentic_controller, '_tool_executor', None),
                    '_current_web_source_map', None)
            or getattr(agentic_controller, '_current_web_source_map', None)
            or {}
        )
        _wiki_map = getattr(getattr(agentic_controller, '_tool_executor', None),
                            '_current_wiki_source_map', None) or {}
        if _web_map or _wiki_map:
            display_output = _apply_web_citations(display_output, _web_map, wiki_map=_wiki_map)
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
                _all_pending = _actions_store.get_all_pending()
                if _all_pending:
                    # Newest drives the approve button; EVERY pending card
                    # renders (2026-09-01: the older of a delete+create pair
                    # was invisible and could never be approved).
                    _pending_action_id = _all_pending[-1].action_id
                    for _pp in _all_pending:
                        display_output += _format_action_proposal_card(_pp)
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
            gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
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

        # If display_output is still empty after the controller's reasoning-only
        # recovery retry, the model returned no usable answer twice. Show an honest
        # fallback (last resort — recovery normally repopulates this).
        if not display_output.strip():
            display_output = "Hmm, the model came back empty on that one (its answer didn't make it out of the reasoning step). Mind sending that again?"
            logger.warning("[Handle Submit] Agentic response empty after recovery, showing fallback")

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

        # ── Factual-grounding floor (same as enhanced path): correct a
        # confirmably-false claim before the final yield. The verifier gets
        # the loop's tool-round results as SOURCE MATERIAL — on agentic turns
        # the response's facts come from retrieved documents the verifier
        # otherwise never sees (it flagged a correct "Fall 2026" against its
        # own date prior, live 2026-08-29). Integration path replaces the
        # whole text (final yield is a bubble replacement); suffix append is
        # the fallback. Either way display AND final_output stay identical.
        try:
            _ag_source_parts = []
            for _gc_round in (getattr(_agentic_session, 'rounds', None) or []):
                _gc_piece = getattr(_gc_round, 'summary', None)
                if not _gc_piece and getattr(_gc_round, 'results', None) is not None:
                    _gc_piece = str(_gc_round.results)
                if _gc_piece:
                    _ag_source_parts.append(str(_gc_piece)[:2000])
            _ag_source = "\n---\n".join(_ag_source_parts)[:6000]
            _ag_gc_revised, _ag_gc_suffix = await _apply_grounding_check(
                ctx, display_output, source_material=_ag_source)
            if _ag_gc_revised:
                display_output = _ag_gc_revised
                final_output = _ag_gc_revised
            elif _ag_gc_suffix:
                display_output = display_output.rstrip() + _ag_gc_suffix
                final_output = (final_output or "").rstrip() + _ag_gc_suffix
        except Exception as _ag_gc_err:
            logger.warning(f"[Handle Submit] Agentic grounding check failed (non-fatal): {_ag_gc_err}")

        # Audit F9 (2026-08-31): contact resolution, the action guard, and the
        # grounding check above mutate display_output AFTER debug_record was
        # built — on exactly the turns grounding changed facts, the Debug tab
        # showed the uncorrected text. Sync the record to what the user sees
        # (same leading-empty-shell strip _build_debug_record applies; the
        # enhanced path builds its record post-mutation).
        debug_record["response"] = _re.sub(
            r"^\s*<(thinking|think|reasoning|reason)>\s*</\1>\s*", "",
            display_output,
        )
        logger.debug(f"[Handle Submit] Agentic yielding final response: {display_output[:100]}...")
        _final_chunk = {"role": "assistant", "content": display_output, "debug": debug_record}
        if _pending_action_id:
            _final_chunk["pending_action_id"] = _pending_action_id
        yield _final_chunk
        logger.debug("[Handle Submit] Agentic final response yielded")

        # Store interaction in background (fire-and-forget, same as enhanced path).
        # Avoids a ~5s blocking await after the final yield that kept the
        # Gradio generator open and could prevent the response from rendering.
        # Full sanitization (thinking blocks + synthetic <thinking></thinking>
        # stream markers + XML leaks + spurious turns) — final_output is the RAW
        # accumulated stream, which historically persisted thinking artifacts
        # into the conversations collection (752 polluted docs as of 2026-06-10).
        try:
            final_output_sanitized = _sanitize_response_text(final_output)
        except Exception as e:
            logger.warning(f"[Handle Submit] Failed to sanitize agentic response: {e}")
            final_output_sanitized = final_output

        if len(final_output_sanitized.strip()) < 20 and display_output.strip():
            final_output_sanitized = display_output

        _dispatch_storage(
            orchestrator, merged_input, final_output_sanitized, user_text,
            final_output_sanitized, personality, file_names, conversation_logger,
            _agentic_session_id, _agentic_prov, 'agentic-search',
        )
        logger.info("[Handle Submit] Agentic storage dispatched to background")
        if _agentic_session is not None:
            ctx.telemetry["agentic_rounds"] = list(
                getattr(_agentic_session, "round_telemetry", [])
            )
            ctx.telemetry["agentic_reuse_fired"] = bool(
                getattr(_agentic_session, "decision_answer_reuse_fired", False)
            )
            ctx.telemetry["agentic_fastpath"] = bool(
                getattr(_agentic_session, "fetch_fastpath_fired", False)
            )
        _write_turn_telemetry(
            ctx, 'agentic-search', _agentic_session_id,
            model_name if 'model_name' in dir() else None,
            len(final_output_sanitized or ""),
            response_text=final_output_sanitized,
        )

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

    # Enhanced cannot invoke tools directly, but it must still know the
    # application's REAL backend status. The live calendar miss said "I don't
    # have calendar access" even though OAuth + write scope were healthy,
    # because this fallback carried only a blanket "no tools" sentence. Share
    # the registry-backed action status with both paths and distinguish
    # pass-local invocation from application-wide capability.
    # NOTE: scoped to the streaming call via _stream_system_prompt — the
    # uncertainty/review agentic RETRIES below reuse `system_prompt` and DO have
    # tools, so they must not inherit the "no tools" claim.
    _stream_system_prompt = system_prompt
    try:
        from core.actions.registry import get_runtime_action_health
        _action_health = get_runtime_action_health()
        _stream_system_prompt = (system_prompt or "") + (
            "\n\n[APPLICATION ACTION STATUS — AUTHORITATIVE]\n"
            f"{_action_health}\n"
            "Never contradict an AVAILABLE backend by claiming the application "
            "lacks access, authentication, or capability."
        )
    except Exception as _health_err:
        logger.warning(
            f"[Handle Submit] Action-status block build failed (non-fatal): {_health_err}"
        )
    try:
        from config.app_config import ACTION_CLAIM_GUARD_ENABLED
        if ACTION_CLAIM_GUARD_ENABLED:
            _stream_system_prompt = (_stream_system_prompt or "") + (
                "\n\n[ACTION HONESTY] This enhanced generation pass cannot invoke tools "
                "directly. Do NOT claim you "
                "saved, sent, created, scheduled, emailed, added, read, opened, fetched, "
                "or pulled up anything in this pass. Do not turn that pass-local limit "
                "into a false claim that an AVAILABLE backend is disconnected or "
                "unauthenticated. If you cannot read a file or retrieve a saved document "
                "right now, say so plainly — do NOT invent a reason for it "
                "(for example, never claim it's because you're \"on mobile\" or similar). "
                "If the user wants such an action, state that a confirmation proposal is "
                "required; never promise that merely sending another chat message will "
                "execute it. Never state it is already done."
            )
    except Exception:
        pass
    _original_limits = ctx.original_limits
    _t_prepare_start = ctx.t_prepare_start
    _t_prepare_elapsed = ctx.t_prepare_elapsed
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
        yield {"role": "assistant", "content": f"✨ Generating response ({model_name})…", "is_progress": True}
        _t_stream_start = _time_mod.perf_counter()
        thinking_started = False
        thinking_complete = False
        chunk_count = 0
        # Degenerate-stream watchdog (2026-09-01): if the model enters a loop
        # emitting repeated garbage, abort on degenerate-shape detection.
        # Shape check ONLY — no wall-clock arm (a slow endpoint streaming a
        # long legitimate answer must never be aborted on time; the insight
        # synthesis 240s ceiling is insight-only by design).
        _degenerate_check_threshold = 2000
        _last_degenerate_check_len = 0
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

            # Degenerate-stream watchdog: abort if the stream entered a
            # repeating-garbage loop (same detection as insight mode).
            if len(final_output) - _last_degenerate_check_len > _degenerate_check_threshold:
                _last_degenerate_check_len = len(final_output)
                if ResponseParser.looks_degenerate_stream(final_output):
                    logger.error(
                        "[Enhanced] Stream aborted by watchdog (degenerate); "
                        f"{len(final_output)} chars discarded, nothing stored"
                    )
                    yield {"role": "assistant", "content":
                           "⚠️ The model's output became repetitive and incoherent. "
                           "I stopped the generation — nothing was stored. "
                           "Please try again."}
                    return

            # Fail fast on classified API-error payloads. model_manager yields
            # the classified error string AS stream content; before 2026-08-21
            # it was rendered chunk-by-chunk into the chat bubble (a 402
            # streamed ~3.2K of raw error JSON mid-distress, 3× on 08-18) and
            # only converted to a friendly message after the stream ended.
            # Break immediately — the post-loop prefix detection below turns
            # the accumulated error into the friendly display and returns
            # before storage. Only the stream HEAD can be an error payload
            # (mid-stream errors append after real content and are handled by
            # the trailing-strip at the storage boundary), so stop checking
            # once real content is flowing.
            if chunk_count <= 5:
                from models.model_manager import API_ERROR_PREFIXES as _api_err_prefixes
                if final_output.lstrip().startswith(_api_err_prefixes):
                    logger.warning(
                        "[Handle Submit] API-error payload at stream head — "
                        "aborting stream display, converting to friendly error"
                    )
                    break

            # Detect incomplete thinking block (opening tag arrived, closing hasn't yet)
            if ResponseParser.has_incomplete_thinking_block(final_output):
                thinking_started = True
                display_output = "💭 **Thinking...**"
                yield {"role": "assistant", "content": display_output, "is_thinking": True}
                continue

            # Closed-but-empty synthetic shell: reasoning ended (</thinking>
            # marker arrived) but the first real content token hasn't yet.
            # parse_thinking_block returns ("", "") for "<thinking></thinking>",
            # so the fallthrough below would display the literal tags for the
            # gap between reasoning end and first token (observed 2026-08-03).
            if ResponseParser.is_empty_thinking_shell(final_output):
                thinking_started = True
                display_output = "💭 **Thinking...**"
                yield {"role": "assistant", "content": display_output, "is_thinking": True}
                continue

            # Empty shell FOLLOWED by content (2026-08-22): kimi-3 sent
            # literal "<thinking>"+"</thinking>" as its first content chunks,
            # then the answer. parse_thinking_block finds no thinking BODY in
            # that buffer, so the fallthrough displayed the raw tags on
            # screen until the end-of-stream recovery. Strip the shell from
            # the accumulated buffer once and treat the rest as the answer.
            if not thinking_complete:
                _shell_stripped = ResponseParser.strip_leading_empty_thinking_shell(final_output)
                if _shell_stripped != final_output:
                    final_output = _shell_stripped
                    thinking_complete = True

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
                    display_output = ResponseParser.strip_trailing_stream_artifact(display_output)  # edge <|sep|> + trailing-'e' (2026-08-22)
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
                display_output = ResponseParser.strip_trailing_stream_artifact(display_output)  # edge <|sep|> + trailing-'e' (2026-08-22)
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
        _friendly = _friendly_api_error(final_output)
        if _friendly:
            logger.warning("[Handle Submit] API error detected — showing friendly message")
            yield {"role": "assistant", "content": _friendly}
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
        display_output = ResponseParser.strip_trailing_stream_artifact(display_output)  # edge <|sep|> + trailing-'e' (2026-08-22)

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
                        ctx.telemetry.update({
                            "uncertainty_fired": True,
                            "uncertainty_trigger": str(_uf_result.trigger_type),
                            "uncertainty_confidence": round(float(_uf_result.confidence), 3),
                        })
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
                        ctx.telemetry["uncertainty_retry_accepted"] = bool(_uf_clean is not None)

            except ImportError as e:
                logger.debug(f"[UNCERTAINTY FALLBACK] Module not available: {e}")
            except Exception as e:
                logger.warning(
                    f"[UNCERTAINTY FALLBACK] Detection failed (non-fatal): {e}"
                )

        # ── Post-Answer Review Gate: LOG-ONLY (2026-08-28) ──
        # The gate previously ran a silent agentic retry and SWAPPED the
        # response before storage when review failed at high confidence. An
        # 8-week telemetry audit showed exactly ONE real-turn swap ever — a
        # 07-28 CONCERN emotional turn where what the user READ diverged from
        # what memory STORED (33 of 34 swaps were benchmark traffic), against
        # an 84% review-failure rate that says the plan-adherence criteria are
        # miscalibrated. Stored must always equal seen: the review still runs
        # and records pass/fail/confidence telemetry for recalibration, but
        # NEVER replaces the response.
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
                        if _review is not None:
                            ctx.telemetry.update({
                                "review_fired": True,
                                "review_passed": bool(_review.passes),
                                "review_confidence": round(float(_review.confidence), 3),
                            })
                        if (
                            _review
                            and not _review.passes
                            and _review.confidence >= RESPONSE_REVIEW_CONFIDENCE_THRESHOLD
                        ):
                            logger.warning(
                                f"[REVIEW GATE] Response failed review "
                                f"(confidence={_review.confidence:.2f}, "
                                f"issues={_review.issues}). Log-only — "
                                f"response NOT replaced."
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

        # After streaming completes, emit a final debug record
        prompt_tokens2, system_tokens2, total_tokens2 = _safe_count_tokens(
            full_prompt, system_prompt, model_name, orchestrator,
        )

        _resp_for_debug = _sanitize_response_text(display_output or final_output)
        # Extract citation METADATA from the marker-bearing text, but discard the
        # aggressively-cleaned text it returns (extract_citations strips ALL markers
        # — including [WEB_N] — AND collapses newlines). We clean the display
        # separately so [WEB_N] survive for end-of-turn linkification and multi-
        # paragraph markdown isn't flattened. (Mirrors the agentic path, which
        # linkifies display_output and extracts citations from a separate string.)
        _, citations = _safe_extract_citations(_resp_for_debug, orchestrator)
        from core.citation_extractor import strip_memory_citation_markers
        _resp_for_debug = strip_memory_citation_markers(_resp_for_debug)

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
                                _resp_for_debug += _format_action_proposal_card(_pending)
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

        # ── Factual-grounding floor: catch a confirmably-false claim the
        # response asserted/endorsed in its own voice. Runs on the FINAL text
        # (after uncertainty/review retries + action guard) so the verifier
        # sees exactly what ships. Integration path (2026-08-29) replaces the
        # text — the final chunk below is a whole-bubble replacement yield, so
        # display and storage stay identical; suffix append is the fallback.
        try:
            _gc_revised, _gc_suffix = await _apply_grounding_check(ctx, _resp_for_debug)
            if _gc_revised:
                _resp_for_debug = _gc_revised
                final_output = _gc_revised
            elif _gc_suffix:
                _resp_for_debug = (_resp_for_debug or "").rstrip() + _gc_suffix
                final_output = (final_output or "").rstrip() + _gc_suffix
        except Exception as e:
            logger.warning(f"[Handle Submit] Grounding check failed (non-fatal): {e}")

        # Make [WEB_N] citations from the standard web-search path clickable
        # (display only; stored response keeps the canonical markers).
        _resp_for_debug = _apply_web_citations(
            _resp_for_debug, getattr(orchestrator, '_web_source_map', None) or {}
        )

        debug_record = _build_debug_record(
            mode=_enh_mode, user_text=user_text, prompt=full_prompt,
            system_prompt=system_prompt, response=_resp_for_debug,
            model=model_name, prompt_tokens=prompt_tokens2,
            system_tokens=system_tokens2, total_tokens=total_tokens2,
            citations=citations, orchestrator=orchestrator,
            provenance=_enh_prov, phase_timings=_handler_timings,
            task_timings=_task_timings, gather_elapsed=_gather_elapsed,
            gate_reason=_gate_debug_summary(getattr(ctx, 'gate_decision', None)),
        )
        _enh_final_chunk = {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
        if _enh_pending_action_id:
            _enh_final_chunk["pending_action_id"] = _enh_pending_action_id
        yield _enh_final_chunk
        debug_emitted = True

    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Streaming error: {e}")
        error_message = f"⚠️ Streaming error: {str(e)}"

        # Even partial/failed turns must leave an inspectable trace. Without
        # this, the UI shows a partial answer while the Debug/Provenance tabs
        # remain blank, making the failure impossible to diagnose remotely.
        if not debug_emitted:
            try:
                _failure_debug = _build_debug_record(
                    mode="failed", user_text=user_text, prompt="",
                    system_prompt="", response=error_message,
                    model=locals().get("model_name", ""),
                    prompt_tokens=0, system_tokens=0, total_tokens=0,
                    citations=[], orchestrator=orchestrator,
                    provenance={
                        "response_mode": "failed",
                        "error": str(e),
                        "partial": bool(final_output),
                    },
                )
                debug_record = _failure_debug
                debug_emitted = True
            except Exception as _debug_error:
                logger.error(f"[HANDLE_SUBMIT] Failed to build error debug: {_debug_error}")

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

        yield {"role": "assistant", "content": error_message, "debug": debug_record if debug_emitted else {}}

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
                _write_turn_telemetry(
                    ctx, _store_mode, _store_session_id,
                    model_name if 'model_name' in dir() else None,
                    len(final_output or ""),
                    response_text=final_output,
                )

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


# ── Ingress guard (2026-08-28): duplicate-submit dedupe + client-error strip ──
# The SPA's "⚠️ Failed to fetch" resend path double-processed turns for weeks
# (duplicate gate evals + 30s builds + stored corpus duplicates on crisis days;
# telemetry shows x2/x3 records for the same heavy query since at least 08-18)
# and embedded the client error text inside resent messages (one stored query
# carries it permanently). The guard is IN-FLIGHT-ONLY by design: a resend
# after the first attempt finished may be the REAL turn (live case 08-28: the
# 14:42 attempt died mid-stream, the 14:46 resend produced the actual reply) —
# a completed-window dedupe would have blocked the good turn.

# normalized-text key → registration monotonic timestamp
_INFLIGHT_SUBMITS: dict = {}
_INFLIGHT_STALE_S = 600.0        # crashed turns never cleaned → expire
_INFLIGHT_MIN_CHARS = 20         # short repeats ("ugh", "hello") are legit

# Completed-turn resend window (2026-08-31): a mobile client that loses the
# SSE at the moment of completion resends the identical query minutes later
# (live case: a 19:59 resend of the completed-and-stored 19:56 insight turn
# re-ran a 4-minute pipeline). This does NOT violate the in-flight-only
# doctrine above — it never BLOCKS a resend, it SERVES the stored reply, and
# only when the first attempt verifiably completed (non-empty stored
# response). A mid-stream-death resend has no stored entry and runs normally.
_COMPLETED_RESEND_WINDOW_S = 300.0


def _resend_serve_appropriate(user_text, stored_reply, history) -> bool:
    """Serve the stored reply only for a genuine lost-reply resend
    (2026-09-01: the 08-31 dedupe served a stale reply to a DELIBERATE
    identical retest — twice — and the served "Queued — approve it" text
    referenced a proposal that had already expired).

    - If the client's own history already contains the stored reply, the
      user SAW the answer; an identical send is a deliberate retry → run.
    - Write-action requests always run fresh: proposals expire in minutes,
      so a served approval prompt is wrong by construction.
    """
    try:
        from core.actions.registry import detect_action_intent
        if detect_action_intent(user_text or "") is not None:
            return False
    except Exception:
        pass
    try:
        head = (stored_reply or "").strip()[:120]
        if head:
            for msg in list(history or [])[-8:]:
                if (isinstance(msg, dict) and msg.get("role") == "assistant"
                        and head in str(msg.get("content") or "")):
                    return False
    except Exception:
        pass
    return True


def _recent_completed_duplicate(orchestrator, norm_query: str):
    """Return the stored response of an identical turn completed within the
    resend window, else None. Read-only over the newest corpus entries."""
    try:
        from datetime import datetime as _dt
        corpus = getattr(
            getattr(orchestrator, "memory_system", None), "corpus_manager", None,
        )
        entries = list(getattr(corpus, "corpus", []) or [])[-5:]
        now = _dt.now()
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            stored_norm = " ".join(str(entry.get("query") or "").lower().split())
            if stored_norm != norm_query:
                continue
            response = str(entry.get("response") or "").strip()
            if not response:
                continue
            try:
                age = (now - _dt.fromisoformat(str(entry.get("timestamp")))).total_seconds()
            except (TypeError, ValueError):
                continue
            if 0 <= age < _COMPLETED_RESEND_WINDOW_S:
                return response
        return None
    except Exception:
        return None

# Line-anchored SPA/client error artifacts that leak into resent messages.
_CLIENT_ERROR_LINE_RE = _re.compile(
    r"^\s*(?:⚠️\s*)?Failed to fetch\s*$", _re.MULTILINE
)


def _strip_client_error_artifacts(text: str) -> str:
    """Remove line-anchored client-transport error artifacts from an incoming
    message. Only whole lines are removed — a user SAYING 'failed to fetch'
    mid-sentence is untouched."""
    if not text or "Failed to fetch" not in text:
        return text
    cleaned = _CLIENT_ERROR_LINE_RE.sub("", text)
    if cleaned != text:
        logger.warning("[Ingress] Stripped client error artifact from incoming message")
        cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _inflight_key(user_text: str, file_names) -> str:
    norm = " ".join((user_text or "").lower().split())
    files_part = "|".join(sorted(str(n) for n in (file_names or [])))
    return f"{norm}::{files_part}"


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
    """Ingress wrapper around the turn dispatcher: strips client error
    artifacts from the incoming text and rejects a duplicate submit of the
    SAME message while the first is still being processed. Transparent
    otherwise — all callers keep this entry point."""

    user_text = _strip_client_error_artifacts(user_text or "")

    _key = None
    _norm = " ".join(user_text.lower().split())
    if len(_norm) >= _INFLIGHT_MIN_CHARS:
        _key = _inflight_key(user_text, [getattr(f, "name", f) for f in (files or [])])
        _now = _time.monotonic()
        _seen = _INFLIGHT_SUBMITS.get(_key)
        if _seen is not None and (_now - _seen) < _INFLIGHT_STALE_S:
            logger.warning(
                f"[Ingress] Duplicate submit while original still in flight "
                f"— ignoring ({user_text[:60]!r})"
            )
            yield {
                "role": "assistant",
                "content": "⚠️ I'm already working on this message — "
                           "ignoring the duplicate submit.",
            }
            return
        _stored_reply = _recent_completed_duplicate(orchestrator, _norm)
        if _stored_reply is not None and not _resend_serve_appropriate(
                user_text, _stored_reply, history):
            logger.info(
                "[Ingress] Identical query but deliberate retry / action "
                "request — running fresh instead of serving the stored reply"
            )
            _stored_reply = None
        if _stored_reply is not None:
            logger.warning(
                f"[Ingress] Resend of a just-completed identical turn — "
                f"serving the stored reply ({user_text[:60]!r})"
            )
            yield {
                "role": "assistant",
                "content": "♻️ This looks like a resend of a message I just "
                           "answered (your connection may have dropped). "
                           "Here's that reply:\n\n" + _stored_reply,
            }
            return
        _INFLIGHT_SUBMITS[_key] = _now
        # opportunistic sweep of stale entries (crashed turns)
        for k in [k for k, t in _INFLIGHT_SUBMITS.items() if (_now - t) >= _INFLIGHT_STALE_S]:
            _INFLIGHT_SUBMITS.pop(k, None)

    try:
        async for _chunk in _handle_submit_inner(
            user_text, files, history, use_raw_gpt, orchestrator,
            system_prompt=system_prompt, force_summarize=force_summarize,
            include_summaries=include_summaries, personality=personality,
            fast_mode=fast_mode,
        ):
            yield _chunk
    finally:
        if _key is not None:
            _INFLIGHT_SUBMITS.pop(_key, None)


async def _handle_submit_inner(
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

    # Pacing metrics: mark the turn at INGRESS, before prompt build. This
    # previously happened inside ResponseGenerator (after the prompt — with
    # its [TIME CONTEXT] block — was already assembled) and never on agentic
    # turns, so "Time since last message" lagged one turn and froze across
    # agentic responses ("53 m" shown for a 2-minute gap, 2026-07-25).
    try:
        _tm = getattr(orchestrator, 'time_manager', None)
        if _tm is not None:
            _tm.mark_query_time()
    except Exception as e:
        logger.debug(f"[Handle Submit] mark_query_time failed: {e}")

    # Process files using security-hardened FileProcessor
    # Supports .txt, .md, .json, .yaml, .yml, .log, .html, .xml, .csv, .py, .docx, .xlsx, .pdf files and .png, .jpg, .jpeg, .gif, .webp images
    file_names = [file.name for file in files] if files else []
    files_result = await file_processor.process_files_structured(user_text, files or [])
    merged_input = files_result.text_content

    # Deterministic attachment audit + deadline-timezone notes (2026-09-04,
    # homework-attachment turn audit items 8-9). No LLM calls; silent when
    # nothing to flag. Appended once to the analysis/merge text so both the
    # rendered [CURRENT QUERY] (via merged_input / ContextPipeline Stage 3)
    # and enhanced-mode classification (via analysis_text) see it.
    _attachment_note = ""
    if files_result.documents:
        try:
            _notes = []
            _audit_note = audit_attachments(user_text, files, files_result.documents)
            if _audit_note:
                _notes.append(_audit_note)
            _deadline_source = user_text + "\n" + "\n".join(
                d.content_text or "" for d in files_result.documents
            )
            _dl_note = deadline_timezone_note(_deadline_source)
            if _dl_note:
                _notes.append(_dl_note)
            _attachment_note = "\n".join(_notes)
        except Exception as e:
            logger.debug(f"[Handle Submit] attachment/deadline audit failed: {e}")

    analysis_text = user_text
    if _attachment_note:
        merged_input += "\n\n" + _attachment_note
        analysis_text = user_text + "\n\n" + _attachment_note

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
        analysis_text=analysis_text,
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

    # Kick off the agentic gate CONCURRENTLY with prompt building. The gate
    # only needs the query + recent corpus; its Tier-4 LLM fallback (~2s) was
    # previously serialized after prepare_prompt. The intent veto needs the
    # context pipeline's classification, so it is applied post-hoc in the
    # dispatcher via gate.apply_intent_veto().
    if agentic_enabled:
        from core.agentic.gate import evaluate_agentic_gate
        ctx.gate_task = asyncio.create_task(evaluate_agentic_gate(
            user_text=user_text,
            entity_resolver=getattr(getattr(orchestrator, 'memory_system', None), 'entity_resolver', None),
            model_manager=orchestrator.model_manager,
            corpus_manager=getattr(getattr(orchestrator, 'memory_system', None), 'corpus_manager', None),
            intent_info=None,  # not classified yet — veto applied post-hoc
        ))

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
            _gt = getattr(ctx, 'gate_task', None)
            if _gt is not None:
                # Duel serviced the turn; gate unused. If the gate already
                # finished (possibly with an error), retrieve the exception so
                # asyncio doesn't log "Task exception was never retrieved".
                _gt.cancel()
                _gt.add_done_callback(
                    lambda t: None if t.cancelled() else t.exception()
                )
            return
        # else duel bailed (timeout/exception) — fall through to agentic/streaming

    if agentic_enabled:
        from core.agentic.gate import evaluate_agentic_gate, apply_intent_veto
        if getattr(ctx, 'gate_task', None) is not None:
            # Gate ran concurrently with prepare_prompt; its ~2s LLM fallback
            # is already paid for by now on all but the fastest prepares.
            _gate_decision = await ctx.gate_task
        else:
            _gate_decision = await evaluate_agentic_gate(
                user_text=user_text,
                entity_resolver=getattr(getattr(orchestrator, 'memory_system', None), 'entity_resolver', None),
                model_manager=orchestrator.model_manager,
                corpus_manager=getattr(getattr(orchestrator, 'memory_system', None), 'corpus_manager', None),
                intent_info=None,
            )
        # Post-hoc intent veto with the context pipeline's classification.
        _gate_decision = apply_intent_veto(
            _gate_decision,
            raw_context.get("intent") if raw_context else None,
            tone_level=raw_context.get("tone_level") if raw_context else None,
            query=user_text,
        )
        # Explicit insight requests own the turn.  The gate runs concurrently
        # with context preparation, so a veto or stale classifier result must
        # not downgrade a recognized pattern request into ordinary web search.
        if not getattr(_gate_decision, "insight_intent", None):
            try:
                from core.insight.detector import detect_insight_request
                _explicit_insight = detect_insight_request(user_text)
                if _explicit_insight is not None:
                    _gate_decision.insight_intent = _explicit_insight.model_dump()
                    _gate_decision.should_trigger = True
                    _gate_decision.veto_exempt = True
                    _gate_decision.modes = ["insight"]
                    _gate_decision.reason = "insight-mode: direct detector override"
                    logger.info("[Handle Submit] Direct insight detector override")
            except Exception as _insight_override_error:
                logger.debug("[Handle Submit] Insight override failed: %s", _insight_override_error)
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

        # Telemetry: record the gate's routing decision for this turn
        ctx.telemetry.update({
            "gate_triggered": bool(should_use_agentic),
            "gate_modes": list(_gate_decision.modes or []),
            "gate_reason": getattr(_gate_decision, "reason", ""),
        })

        # Tone-deferred request (2026-08-21): a tone arm stood the gate down
        # on a request-shaped (non-vent) query. Never fail silently — tell
        # the model so it acknowledges the request and offers to proceed; a
        # terse affirmation next turn re-runs the original query veto-exempt
        # (armed in gate.apply_intent_veto). Vent-shaped turns never get
        # this note (anti-excavation).
        _deferred_q = getattr(_gate_decision, "deferred_request", None)
        if _deferred_q:
            ctx.system_prompt = (ctx.system_prompt or "") + (
                "\n\n[DEFERRED REQUEST] The user asked you to do something "
                f"this turn ('{_deferred_q[:160]}') that needs tools, but tool "
                "use was held back because the conversation register is heavy. "
                "Do NOT pretend you did it or invent results. Briefly "
                "acknowledge you can do it and ask if they'd like you to go "
                "ahead — if they confirm, it will run next turn. Keep the "
                "offer to one short sentence; the person comes first."
            )
            ctx.telemetry["gate_deferred_request"] = True

        # Insight consent offer (2026-08-23): the user made an insight-SHAPED
        # first-person statement at non-elevated tone and the mode did NOT
        # trigger — arm the ONE-per-session offer slot and tell the model it
        # may offer, once, in one sentence, to check the insight against full
        # history. A terse affirmation next turn runs the assessment
        # (gate consumes the slot); anything else drops the offer permanently.
        if not getattr(_gate_decision, "insight_intent", None):
            try:
                from core.agentic.gate import maybe_arm_insight_offer
                if maybe_arm_insight_offer(
                    user_text,
                    raw_context.get("tone_level") if raw_context else None,
                ):
                    ctx.system_prompt = (ctx.system_prompt or "") + (
                        "\n\n[INSIGHT OFFER] The user just stated an insight "
                        "about themselves. You MAY offer — once, in one short "
                        "sentence at the end of your reply — to check it "
                        "against everything they've told you across sessions. "
                        "Do not push; if they decline or ignore it, never "
                        "bring the offer up again."
                    )
                    ctx.telemetry["insight_offer_armed"] = True
            except Exception as _io_err:
                logger.debug(f"[Handle Submit] Insight offer check failed: {_io_err}")

        # --- Insight / evidence-assembly mode (owns the turn) ---
        if getattr(_gate_decision, "insight_intent", None) and should_use_agentic:
            async for _c in _run_insight_mode(ctx):
                yield _c
            if ctx.handled:
                return
            # else insight mode failed — fall through to doc-gen / agentic / enhanced

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

def _chain_next_pending(store, outcome):
    """Approval chaining (2026-09-01): a delete+create turn left the older
    proposal invisible (newest-only card) — after any decision, surface the
    next still-pending proposal so it can be approved instead of silently
    expiring."""
    try:
        nxt = store.get_pending()
        # Type-strict: only chain a real proposal (a permissive store double
        # returning a truthy stub must not leak into the outcome fields).
        if nxt is not None and isinstance(getattr(nxt, "action_id", None), str):
            outcome.next_action_id = nxt.action_id
            outcome.next_summary = nxt.summary
            outcome.message = (
                outcome.message.rstrip()
                + _format_action_proposal_card(nxt)
                + "One more proposal from that turn is still pending — approve it too?"
            )
    except Exception as _chain_err:
        logger.warning(f"[Actions] Pending-chain check failed (non-fatal): {_chain_err}")
    return outcome


async def execute_pending_action_core(action_id: str, orchestrator=None):
    """Approve + execute a pending internet action; transport-agnostic core.

    Returns an ActionOutcome whose `message` is the assistant-styled chat line.
    Shared by the Gradio Approve button and the FastAPI approve route.
    """
    from core.actions.types import ActionOutcome
    from core.actions.audit import ActionAuditLog
    from config.app_config import INTERNET_ACTIONS_AUDIT_LOG

    if not action_id:
        return ActionOutcome(status="not_found",
                             message="Action expired or not found. Ask me again if you still want this.")

    # Load proposal from the global store
    from core.agentic.tools import ToolExecutor
    store = ToolExecutor._get_pending_actions_store()
    proposal = store.approve(action_id)

    if not proposal:
        return ActionOutcome(status="not_found",
                             message="Action expired or not found. Ask me again if you still want this.")

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
            return _chain_next_pending(store, ActionOutcome(
                status="executed",
                message=f"[ACTION EXECUTED: {proposal.action_type.value}] {result.message}",
                action_type=proposal.action_type.value,
                summary=proposal.summary,
            ))
        store.mark_failed(action_id, result.message)
        return _chain_next_pending(store, ActionOutcome(
            status="failed",
            message=f"Action failed: {result.message}\n\nWant me to try something else?",
            action_type=proposal.action_type.value,
            summary=proposal.summary,
        ))
    except Exception as e:
        store.mark_failed(action_id, str(e))
        logger.error(f"[Actions] Execution failed for {action_id}: {e}")
        return ActionOutcome(
            status="failed",
            message=f"Action failed with error: {e}\n\nWant me to try something else?",
            action_type=proposal.action_type.value,
            summary=proposal.summary,
        )


async def reject_pending_action_core(action_id: str, orchestrator=None):
    """Reject a pending internet action; transport-agnostic core (see execute_pending_action_core)."""
    from core.actions.types import ActionOutcome
    from core.actions.audit import ActionAuditLog
    from config.app_config import INTERNET_ACTIONS_AUDIT_LOG

    if not action_id:
        return ActionOutcome(status="not_found",
                             message="Action already expired or was not found.")

    from core.agentic.tools import ToolExecutor
    store = ToolExecutor._get_pending_actions_store()
    proposal = store.reject(action_id)

    audit = ActionAuditLog(INTERNET_ACTIONS_AUDIT_LOG)
    audit.log_decision(action_id, approved=False)

    if proposal:
        return _chain_next_pending(store, ActionOutcome(
            status="rejected",
            message=f"[ACTION REJECTED] Cancelled: {proposal.summary}",
            action_type=proposal.action_type.value,
            summary=proposal.summary,
        ))
    return ActionOutcome(status="not_found",
                         message="Action already expired or was not found.")


async def execute_pending_action(action_id: str, chat_history: list, orchestrator=None):
    """Execute an approved internet action. Called by GUI Approve button.

    Does NOT go through submit_chat — directly modifies chat_history and returns.
    """
    import gradio as gr

    outcome = await execute_pending_action_core(action_id, orchestrator)
    if action_id:  # legacy behavior: empty id appends nothing
        chat_history.append({"role": "assistant", "content": outcome.message})
    return chat_history, gr.update(value=None), gr.update(visible=False)


async def reject_pending_action(action_id: str, chat_history: list, orchestrator=None):
    """Reject a pending internet action. Called by GUI Reject button."""
    import gradio as gr

    outcome = await reject_pending_action_core(action_id, orchestrator)
    if action_id:  # legacy behavior: empty id appends nothing
        chat_history.append({"role": "assistant", "content": outcome.message})
    return chat_history, gr.update(value=None), gr.update(visible=False)
