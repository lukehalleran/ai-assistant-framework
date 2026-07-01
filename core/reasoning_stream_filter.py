"""Interleaved reasoning/content stream filter.

Reasoning models (glm-5.2, deepseek-v4, ... via OpenRouter) can interleave their
reasoning and visible content within a single streamed response, e.g.::

    <reason…>  "synthesis system."  <reason…>  "Let me check what you've been up to…"

The naive consumer ("yield every content delta, suppress reasoning-only chunks")
fused the discarded pre-answer draft onto the real answer with no separator and
produced the user-visible leak ``"synthesis system.Let me check…"``. None of the
thinking-block strippers caught it: the fragment is untagged and glued to the
answer with no separator, so it survived ``sanitize_for_storage`` and persisted
into both the displayed and stored message. (See conversation ``0f6d70c7`` /
``daemon_debug`` 2026-06-28 — the model was glm-5.2.)

This filter sits between the raw delta stream and the consumer. It:

  * suppresses reasoning-only chunks, emitting synthetic ``<thinking>`` /
    ``</thinking>`` markers around them (preserving the existing thinking-block
    contract that the GUI relies on);
  * holds the leading *content run* (contiguous content emitted before any real
    answer has been committed) until it is confirmed to be genuine:
        - a run that grows past ``draft_max_chars`` is committed and then streams
          live, so long real answers still stream token-by-token;
        - a run cut short because reasoning RESUMES is treated as a provisional
          draft and dropped — UNLESS nothing ever replaces it, in which case it
          is restored at :meth:`finish` so a genuinely short final answer is
          never lost;
  * stays completely inert until the first reasoning chunk is seen, so
    non-reasoning models stream exactly as before (no added latency, no
    behaviour change).

Once a content run is confirmed (the model has clearly started answering), all
subsequent content streams live — only the *leading* draft is protected. This
narrowly targets the observed "query-echo fragment glued to the front" failure
without risking the loss of mid-answer segments in a genuinely multi-part
interleaved answer.

Shared by ``core.agentic.controller._generate_final_response`` and
``core.response_generator.generate_streaming_response`` so the two streaming
paths cannot drift.

Usage::

    f = InterleavedReasoningFilter()
    for delta_reasoning, delta_content in stream:
        for kind, text in f.feed(delta_reasoning, delta_content):
            if kind == MARKER:
                yield text          # "<thinking>" / "</thinking>"
            else:                    # kind == CONTENT
                ...                  # committed visible content
    for kind, text in f.finish():    # flush anything still held
        ...
    if f.reasoning_seen and not f.content_emitted:
        ...                          # reasoning-only swallow → recover
"""
from __future__ import annotations

from typing import List, Tuple

# Output item kinds returned by feed()/finish().
MARKER = "marker"    # a synthetic <thinking>/</thinking> tag — yield verbatim
CONTENT = "content"  # confirmed visible content

THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"

# A leading content run shorter than this (after strip) that is cut off by
# reasoning resuming is treated as a discarded draft. 50 chars comfortably
# clears the observed 17-char leak ("synthesis system.") while keeping the
# false-drop window small; the held-fallback in finish() prevents any genuine
# short final answer from being lost outright.
DEFAULT_DRAFT_MAX_CHARS = 50


class InterleavedReasoningFilter:
    """Cleans an interleaved reasoning/content delta stream. See module docstring."""

    def __init__(self, draft_max_chars: int = DEFAULT_DRAFT_MAX_CHARS):
        self.draft_max_chars = draft_max_chars
        self._in_reasoning = False     # currently inside a <thinking> segment
        self._reasoning_seen = False   # any reasoning chunk ever arrived
        self._content_emitted = False  # any visible content ever committed
        self._pending = ""             # leading content run, not yet confirmed
        self._streaming = False        # leading run confirmed → pass content live
        self._held_draft = ""          # last dropped draft, restored if nothing replaces it

    # ── introspection (used by the reasoning-only recovery paths) ────────────
    @property
    def reasoning_seen(self) -> bool:
        return self._reasoning_seen

    @property
    def content_emitted(self) -> bool:
        return self._content_emitted

    @property
    def in_reasoning(self) -> bool:
        return self._in_reasoning

    # ── stream processing ────────────────────────────────────────────────────
    def feed(self, delta_reasoning: str, delta_content: str) -> List[Tuple[str, str]]:
        """Process one streamed delta; return a list of (kind, text) items to emit."""
        out: List[Tuple[str, str]] = []
        delta_reasoning = delta_reasoning or ""
        delta_content = delta_content or ""

        # Reasoning-only chunk: suppress the text, manage the <thinking> marker.
        if delta_reasoning and not delta_content:
            self._reasoning_seen = True
            if not self._in_reasoning:
                self._in_reasoning = True
                out.append((MARKER, THINKING_OPEN))
                # The leading content run (if any) just ended because reasoning
                # resumed → decide whether it was a draft or genuine content.
                self._on_reasoning_resume(out)
            return out

        # Transition out of reasoning on the first content delta.
        if self._in_reasoning and delta_content:
            self._in_reasoning = False
            out.append((MARKER, THINKING_CLOSE))
            # Real content followed the held draft → the draft was indeed junk.
            self._held_draft = ""

        if delta_content:
            if not self._reasoning_seen or self._streaming:
                # No reasoning involved (non-reasoning model) or the leading run
                # is already confirmed → pass content through live.
                self._content_emitted = True
                out.append((CONTENT, delta_content))
            else:
                # Buffer the leading run until it is confirmed non-draft.
                self._pending += delta_content
                if len(self._pending.strip()) > self.draft_max_chars:
                    out.append((CONTENT, self._pending))
                    self._content_emitted = True
                    self._pending = ""
                    self._streaming = True
        return out

    def _on_reasoning_resume(self, out: List[Tuple[str, str]]) -> None:
        """Resolve a buffered leading run when reasoning starts again."""
        if not self._pending:
            return
        if len(self._pending.strip()) <= self.draft_max_chars:
            # Short leading run cut off by resumed reasoning → provisional draft.
            # Hold it as a fallback rather than emitting it.
            self._held_draft = self._pending
        else:
            # Substantial run → genuine content; commit it.
            out.append((CONTENT, self._pending))
            self._content_emitted = True
            self._held_draft = ""
        self._pending = ""

    def finish(self) -> List[Tuple[str, str]]:
        """Flush any content still held. Call once after the stream ends."""
        out: List[Tuple[str, str]] = []
        flush = ""
        if self._pending:
            # Leading run ended the stream without reasoning resuming → it was
            # the real (short) answer all along.
            flush = self._pending
        elif not self._content_emitted and self._held_draft:
            # We dropped a run as a draft but nothing ever replaced it → restore
            # it; it was actually the final answer.
            flush = self._held_draft
        self._pending = ""
        self._held_draft = ""
        if flush:
            if self._in_reasoning:
                out.append((MARKER, THINKING_CLOSE))
                self._in_reasoning = False
            out.append((CONTENT, flush))
            self._content_emitted = True
        return out
