"""
core/insight/provenance.py

Module Contract
- Purpose: Label every EvidenceItem with a provenance stance and render the
  numbered evidence block. Provenance is FIRST-CLASS here: the 3-hop
  pseudo-fact chain (user appraisal → assistant elaboration stored →
  re-elaboration → "fact") is broken by explicitly marking every
  Daemon-authored line as interpretation, not the user's words.
- Inputs: list[EvidenceItem] from the sweep.
- Outputs: labeled items (in place + returned); rendered text block with
  [E1..En] references, every line dated + source-attributed.
- Key behaviors:
  * Label mapping — corpus user side → user-stated; corpus assistant side →
    assistant-inferred; obsidian_notes → users-own-note; facts →
    extracted-fact; graph → graph-edge; summaries/reflections/threads →
    assistant-inferred (Daemon-authored); conversation docs keep their inline
    speaker markers and are labeled by which sides they contain.
  * is_appraisal via the shared stance core (memory/stance_classifier) —
    fact triples get classify_triple_stance (metadata first, text fallback);
    free text gets is_evaluative_text.
  * Appraisal lines render with an explicit "[value judgment — an appraisal,
    not an objective fact]" marker; assistant-inferred lines with
    "[assistant's interpretation, not your words]".
- Side effects: none (pure over its inputs).
"""

from __future__ import annotations

from core.insight.types import EvidenceItem
from memory.stance_classifier import (
    classify_triple_stance,
    effective_stance,
    is_evaluative_text,
)

_ASSISTANT_AUTHORED_COLLECTIONS = frozenset({"summaries", "reflections", "threads"})

_LABEL_DISPLAY = {
    "user-stated": "you said",
    "users-own-note": "your note",
    "assistant-inferred": "assistant's interpretation, not your words",
    "extracted-fact": "extracted fact",
    "graph-edge": "relationship record",
}


def _label_conversation_doc(item: EvidenceItem) -> str:
    """Conversation docs carry inline 'User:'/'Assistant:' markers. A doc with
    ONLY assistant text must never be labeled as the user's words."""
    text = item.text
    has_user = "User:" in text or "Q:" in text
    has_assistant = "Assistant:" in text or "Daemon:" in text or "A:" in text
    if has_assistant and not has_user:
        return "assistant-inferred"
    return "user-stated"


def label_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Assign stance_label + is_appraisal to every item (in place)."""
    for item in items:
        coll = item.collection

        if coll == "corpus":
            item.stance_label = (
                "assistant-inferred" if item.speaker == "assistant" else "user-stated"
            )
        elif coll == "obsidian_notes":
            item.stance_label = "users-own-note"
        elif coll == "facts":
            item.stance_label = "extracted-fact"
        elif coll == "graph":
            item.stance_label = "graph-edge"
        elif coll in _ASSISTANT_AUTHORED_COLLECTIONS:
            item.stance_label = "assistant-inferred"
        elif coll == "conversations":
            item.stance_label = _label_conversation_doc(item)
        else:
            item.stance_label = "user-stated"

        if coll == "facts":
            item.is_appraisal = _fact_is_appraisal(item)
        elif coll == "graph":
            # sweep already set it from edge metadata (write-time stance);
            # fall back to text inspection for legacy untagged edges
            item.is_appraisal = item.is_appraisal or is_evaluative_text(item.text)
        else:
            item.is_appraisal = item.is_appraisal or is_evaluative_text(item.text)

    return items


def _fact_is_appraisal(item: EvidenceItem) -> bool:
    """Facts: prefer explicit stored stance, then triple classification of the
    ``subject | relation | object`` doc shape, then plain text inspection."""
    # (metadata isn't carried on EvidenceItem; the sweep folds stored stance
    #  into is_appraisal for graph edges. For fact docs, classify the triple.)
    if item.is_appraisal:
        return True
    parts = [p.strip() for p in item.text.split("|")]
    if len(parts) == 3:
        return classify_triple_stance(parts[0], parts[1], parts[2]).is_appraisal
    return is_evaluative_text(item.text)


def render_evidence_block(items: list[EvidenceItem], max_chars: int = 12000) -> str:
    """Render numbered, dated, attributed evidence lines: ``[E1] date · source``.

    Stops adding lines at max_chars (the synthesizer's prompt budget)."""
    lines: list[str] = []
    used = 0
    for i, item in enumerate(items, start=1):
        date = (item.date or "undated")[:10]
        display = _LABEL_DISPLAY.get(item.stance_label, item.stance_label)
        source = item.collection or "memory"
        markers = ""
        if item.stance_label == "assistant-inferred":
            markers = " [assistant's interpretation, not your words]"
        elif item.is_appraisal:
            markers = " [value judgment — an appraisal, not an objective fact]"
        line = f'[E{i}] {date} · {source} · {display}: "{item.text}"{markers}'
        if used + len(line) > max_chars:
            lines.append(f"(… {len(items) - i + 1} further items omitted for space)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
