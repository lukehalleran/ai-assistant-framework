"""Single structural parser for the ``[test]...[/test]`` operator-marker
convention (2026-09-06, B3): an owner-typed probe wraps synthetic/replay
turns so they are never mistaken for live user speech or testimony.

Module Contract
- Purpose: recognise a COMPLETE ``[test]...[/test]`` pair — either the
  whole-line block form (opening tag alone on a line, closing tag alone on
  a later line) or the inline form (both tags on the same line, e.g.
  ``"[test]Navient. Search that[/test]"``) — and give callers the content
  with the envelope removed.
- Inputs: any str (may be None/empty; never raises).
- Outputs:
  - has_envelope(text) -> bool: at least one complete pair present.
  - inner_text(text) -> str: every complete envelope UNWRAPPED (tags
    dropped, inner content kept in place); text with no complete envelope
    is returned unchanged.
  - envelope_line_indices(text) -> set[int]: line indices spanned by any
    complete envelope (open tag's line through close tag's line,
    inclusive) — for callers that exclude envelope content at line
    granularity (memory.fact_source.quoted_correspondence_lines).
- Side effects: None (pure functions, stdlib only).

Why (2026-09-27, BC-58): the owner's inline probes were invisible to every
consumer that only matched the WHOLE-LINE form —
`memory/fact_source.py:contains_test_block` could mint a real profile fact
from a probe, and gui/handlers.py's ingress routed the raw marker-bearing
text into gate/intent/tone/STM classification instead of what the probe was
actually asking. Detected structurally by the bracket markers only —
nothing infers "test" from wording, repetition, or content.

Leaf module: stdlib only (`re`, `__future__`) — importable from anywhere,
including memory/fact_source.py's own leaf-module contract, without adding
a package dependency.
"""

from __future__ import annotations

import re

# One grammar for both forms: the whole-line block form is just this same
# pattern with the captured content spanning multiple lines (the opening
# and closing tags happen to sit alone on their own lines). Non-greedy so
# two separate envelopes in one text are matched as two pairs, not one.
_ENVELOPE_RE = re.compile(r"\[test\](.*?)\[/test\]", re.IGNORECASE | re.DOTALL)


def has_envelope(text: str) -> bool:
    """True when ``text`` contains at least one COMPLETE [test]...[/test]
    pair (inline or spanning whole lines). An unclosed/dangling ``[test]``
    is not an envelope."""
    if not text:
        return False
    return bool(_ENVELOPE_RE.search(text))


def inner_text(text: str) -> str:
    """Return ``text`` with every complete [test]...[/test] envelope
    UNWRAPPED — tags removed, inner content kept — so routing/classification
    judges what the probe actually said instead of the raw marker-bearing
    turn. Text with no complete envelope is returned unchanged (identity
    fallback: safe to call on ordinary, non-enveloped text everywhere)."""
    if not text:
        return text or ""
    if not _ENVELOPE_RE.search(text):
        return text
    return _ENVELOPE_RE.sub(lambda m: m.group(1), text)


def envelope_line_indices(text: str) -> set[int]:
    """Line indices spanned by any complete [test]...[/test] envelope —
    the tag lines themselves through every content line between them,
    inclusive. Empty set when no complete envelope exists."""
    if not text:
        return set()
    indices: set[int] = set()
    for m in _ENVELOPE_RE.finditer(text):
        start_line = text.count("\n", 0, m.start())
        end_line = text.count("\n", 0, max(m.end() - 1, m.start()))
        indices.update(range(start_line, end_line + 1))
    return indices
