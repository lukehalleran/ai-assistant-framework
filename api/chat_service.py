"""Transport adapter: gui.handlers.handle_submit → typed ChatEvent stream.

submit_stream() reuses THE deployed chat pipeline (dispatch, sanitization,
uncertainty fallback, review gate, action guard, storage, telemetry) — it only
translates the yielded dicts into SSE-ready ChatEvents and maintains the API
session (history append, debug records, pending_action_id). Content events are
CUMULATIVE full text (Gradio replace semantics), so the client replace-renders
the last assistant bubble; no delta reassembly.
"""

import asyncio
import re
from typing import AsyncGenerator

from api.schemas import ChatEvent, ChatRequest
from api.state import AppState
from utils.logging_utils import get_logger

logger = get_logger("api_chat")

# Emit a keepalive comment if the pipeline is silent this long (seconds).
KEEPALIVE_INTERVAL_S = 10.0

# Same safety net as gui/launch.py submit_chat: strip leaked XML tool markers
# from the final content if they slipped through the handlers.
_XML_ARTIFACT_PATTERNS = [
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL),
    re.compile(r"<function_calls>.*$", re.DOTALL),
    re.compile(r"<invoke\s[^>]*>.*?</invoke>", re.DOTALL),
    re.compile(r"<propose_action[^>]*>.*?</propose_action>", re.DOTALL),
    re.compile(r"<lookup_contact[^>]*>.*?</lookup_contact>", re.DOTALL),
]


def _strip_xml_artifacts(text: str) -> str:
    if not text or ("<function_calls>" not in text and "<invoke" not in text
                    and "<propose_action" not in text and "<lookup_contact" not in text):
        return text
    for pat in _XML_ARTIFACT_PATTERNS:
        text = pat.sub("", text).strip()
    return text


async def submit_stream(req: ChatRequest, state: AppState) -> AsyncGenerator[ChatEvent, None]:
    """Run one chat turn through handle_submit, yielding typed ChatEvents.

    Caller must hold state.session.stream_lock. The user message and final
    assistant message are appended to session.history here.
    """
    from gui.handlers import handle_submit

    session = state.session
    orchestrator = state.orchestrator
    files = state.resolve_uploads(req.file_ids)

    # Same per-turn citation-mode toggle the Gradio submit path sets
    orchestrator.enable_citations = req.enable_citations

    session.history.append({"role": "user", "content": req.text})

    final_content = ""
    completed = False

    agen = handle_submit(
        user_text=req.text,
        files=files or None,
        history=session.history,
        use_raw_gpt=req.raw_mode,
        orchestrator=orchestrator,
        personality=session.personality,
        fast_mode=req.fast_mode,
    )

    try:
        next_task = asyncio.ensure_future(agen.__anext__())
        while True:
            # Wait for the next chunk, emitting keepalives while the pipeline
            # is quiet (context build / slow LLM) so proxies don't stall.
            try:
                chunk = await asyncio.wait_for(
                    asyncio.shield(next_task), timeout=KEEPALIVE_INTERVAL_S
                )
            except asyncio.TimeoutError:
                yield ChatEvent(event="progress", data={"text": ""})
                continue
            except StopAsyncIteration:
                break

            next_task = asyncio.ensure_future(agen.__anext__())

            if not isinstance(chunk, dict):
                final_content = str(chunk)
                yield ChatEvent(event="message", data={"content": final_content})
                continue

            if "thinking" in chunk:
                yield ChatEvent(event="duel_thinking", data=dict(chunk["thinking"] or {}))
                continue

            is_final = "debug" in chunk
            content = chunk.get("content") or ""

            if chunk.get("is_thinking") and not is_final:
                yield ChatEvent(event="thinking", data={"text": content})
                continue
            if chunk.get("is_progress") and not is_final:
                yield ChatEvent(event="progress", data={"text": content})
                continue

            if content:
                final_content = content

            if is_final:
                debug = chunk.get("debug") or {}
                # Fallback mirror of submit_chat: recover swallowed answers
                if not final_content:
                    final_content = (debug.get("response") or "") if isinstance(debug, dict) else ""
                final_content = _strip_xml_artifacts(final_content)

                pending_action_id = chunk.get("pending_action_id")
                session.pending_action_id = pending_action_id
                if isinstance(debug, dict):
                    session.debug_records.append(debug)
                session.history.append({"role": "assistant", "content": final_content})
                completed = True

                yield ChatEvent(event="complete", data={
                    "content": final_content,
                    "pending_action_id": pending_action_id,
                    "debug": debug if isinstance(debug, dict) else None,
                    "turn_index": len(session.history) - 1,
                })
            elif content:
                yield ChatEvent(event="message", data={"content": final_content})
    except Exception as e:
        logger.error(f"[API] Chat stream failed: {type(e).__name__}: {e}", exc_info=True)
        if not completed:
            final_content = _strip_xml_artifacts(final_content)
            session.history.append({
                "role": "assistant",
                "content": final_content or f"⚠️ Something went wrong: {e}",
            })
        yield ChatEvent(event="error", data={"message": str(e)})
    else:
        if not completed:
            # Stream ended without a debug chunk (shouldn't happen; be robust)
            final_content = _strip_xml_artifacts(final_content)
            session.history.append({"role": "assistant", "content": final_content})
            yield ChatEvent(event="complete", data={
                "content": final_content,
                "pending_action_id": None,
                "debug": None,
                "turn_index": len(session.history) - 1,
            })
