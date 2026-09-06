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

from datetime import datetime
import re

from core.insight.sweep import week_bucket_key
from core.insight.types import EvidenceItem
from memory.fact_source import quoted_correspondence_lines
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
    "external-research": "external research source",
    "computed-evidence": "computed evidence",
    "quoted-correspondence": "quoted third-party text inside your message",
}

_SPEAKER_RE = re.compile(r"(?:^|\n)[ \t]*(User|Human|Q|Assistant|Daemon|A):[ \t]*")


def _split_speakers(item: EvidenceItem) -> list[EvidenceItem]:
    """Separate full speaker spans BEFORE snippet clipping can erase labels."""
    if item.collection != "conversations" or item.speaker:
        return [item]
    if "user_text" in item.metadata and "response" in item.metadata:
        return [item.model_copy(update={
            "text": text, "speaker": speaker,
            "doc_id": f"{item.doc_id}:{speaker}:0" if item.doc_id else None,
        }) for speaker, text in (("user", item.metadata["user_text"]),
                                 ("assistant", item.metadata["response"]))
                if isinstance(text, str) and text.strip()]
    matches = list(_SPEAKER_RE.finditer(item.text))
    if not matches:
        return [item]
    if len(matches) == 1:
        item.speaker = "user" if matches[0].group(1) in {"User", "Human", "Q"} else "assistant"
        return [item]
    parts = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(item.text)
        text = item.text[match.end():end].strip()
        if not text:
            continue
        speaker = "user" if match.group(1) in {"User", "Human", "Q"} else "assistant"
        parts.append(item.model_copy(update={
            "text": text, "speaker": speaker,
            "doc_id": f"{item.doc_id}:{speaker}:{index}" if item.doc_id else None,
        }))
    return parts or [item]


def _split_quoted_correspondence(item: EvidenceItem) -> list[EvidenceItem]:
    """Split a user-authored item into (optional framing, quoted-block)
    pieces when its text contains a pasted-correspondence block (greeting →
    closing → signature, both ends required — see
    ``memory.fact_source.quoted_correspondence_lines``).

    A pasted email inside a user turn is reported/historical third-party
    text, not the user speaking in their own voice: rendering the whole item
    as "you said" let a third party correcting the user get counted as the
    user correcting the assistant (2026-09-04 live incident). The user's own
    framing lines, if any remain once the block is removed, stay "you said".
    """
    if item.stance_label == "quoted-correspondence":
        # Already split: label_evidence runs in sweep._finalize AND again in
        # handlers (engine/frozen-contract items join late) — a second pass
        # must not re-suffix the doc_id (":quoted:quoted").
        return [item]
    quoted_idx = quoted_correspondence_lines(item.text)
    if not quoted_idx:
        return [item]
    lines = (item.text or "").splitlines()
    framing = "\n".join(ln for i, ln in enumerate(lines) if i not in quoted_idx).strip()
    quoted = "\n".join(ln for i, ln in enumerate(lines) if i in quoted_idx).strip()
    out: list[EvidenceItem] = []
    if framing:
        out.append(item.model_copy(update={"text": framing}))
    if quoted:
        out.append(item.model_copy(update={
            "text": quoted,
            "stance_label": "quoted-correspondence",
            "is_appraisal": False,
            "doc_id": (f"{item.doc_id}:quoted" if item.doc_id else None),
        }))
    return out or [item]


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
    """Assign stance_label + is_appraisal to every item.

    May SPLIT a user-authored corpus/conversation item that embeds pasted
    correspondence into up to two items (framing + quoted block) — callers
    must use the RETURNED list, not assume in-place-only mutation.
    """
    expanded: list[EvidenceItem] = []
    for item in [part for original in items for part in _split_speakers(original)]:
        if item.collection in ("corpus", "conversations") and item.speaker != "assistant":
            expanded.extend(_split_quoted_correspondence(item))
        else:
            expanded.append(item)
    items = expanded

    for item in items:
        if item.stance_label == "quoted-correspondence":
            continue  # already resolved by the split above
        coll = item.collection

        if coll == "corpus":
            item.stance_label = (
                "assistant-inferred" if item.speaker == "assistant" else "user-stated"
            )
        elif coll == "obsidian_notes":
            # Deliberation events may carry explicit provenance from
            # frontmatter (e.g. Daemon-generated daily summaries). Preserve
            # that classification instead of treating every vault note as a
            # first-person user record.
            metadata = item.metadata
            if (str(metadata.get("author", "")).lower() == "daemon"
                    or metadata.get("source_type") == "daemon_daily_summary"):
                item.stance_label = "assistant-inferred"
            elif item.stance_label not in {"assistant-inferred", "extracted-fact"}:
                item.stance_label = "users-own-note"
        elif coll == "facts":
            item.stance_label = "extracted-fact"
        elif coll == "graph":
            item.stance_label = "graph-edge"
        elif coll in _ASSISTANT_AUTHORED_COLLECTIONS:
            item.stance_label = "assistant-inferred"
        elif coll == "conversations":
            item.stance_label = (
                "assistant-inferred" if item.speaker == "assistant"
                else _label_conversation_doc(item)
            )
        elif item.stance_label in {"external-research", "computed-evidence"}:
            # Adapters assign these explicitly; never relabel independent
            # evidence as the user's statement because its collection is novel.
            pass
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


def format_evidence_line(index: int, item: EvidenceItem) -> str:
    """Render a single numbered, dated, attributed evidence line:
    ``[E<index>] date · source · label: "text" [markers]``.

    Extracted from ``render_evidence_block`` (2026-09-06) so a layout pass
    can estimate an item's rendered cost without duplicating the format.
    """
    date = (item.date or "undated")[:10]
    display = _LABEL_DISPLAY.get(item.stance_label, item.stance_label)
    source = item.collection or "memory"
    markers = ""
    if item.stance_label == "assistant-inferred":
        markers = " [assistant's interpretation, not your words]"
    elif item.is_appraisal:
        markers = " [value judgment — an appraisal, not an objective fact]"
    return f'[E{index}] {date} · {source} · {display}: "{item.text}"{markers}'


def render_evidence_block(items: list[EvidenceItem], max_chars: int = 12000) -> str:
    """Render numbered, dated, attributed evidence lines: ``[E1] date · source``.

    Stops adding lines at max_chars (the synthesizer's prompt budget). Always
    states the date range actually rendered, and — when the cap forced
    omissions — how many items per omitted ISO week, so the synthesizer (and
    through it, the user) can say honestly what it did NOT see, instead of
    silently reporting only the newest slice of a long window (2026-09-04
    live incident: a seven-week evidence set truncated to the newest three
    days behind a bare "N further items omitted" note).
    """
    lines: list[str] = []
    used = 0
    rendered_dates: list[str] = []
    omitted: list[EvidenceItem] = []
    for i, item in enumerate(items, start=1):
        date = (item.date or "undated")[:10]
        line = format_evidence_line(i, item)
        if used + len(line) > max_chars:
            # Skip, don't stop: a single oversized item must not hide every
            # item behind it (2026-09-06). E-numbers keep the original index.
            omitted.append(item)
            continue
        lines.append(line)
        used += len(line) + 1
        if date != "undated":
            rendered_dates.append(date)
    if omitted:
        lines.append(_omission_note(omitted))
    if rendered_dates:
        lines.append(
            f"(Evidence rendered above spans {min(rendered_dates)} to "
            f"{max(rendered_dates)} — {len(rendered_dates)} of {len(items)} "
            f"items considered.)"
        )
    return "\n".join(lines)


def _omission_note(omitted: list[EvidenceItem]) -> str:
    """'(… N further items omitted for space)' plus a per-ISO-week breakdown
    so the model can name what it did not see instead of a bare count."""
    by_week: dict[tuple[int, int], int] = {}
    order: list[tuple[int, int]] = []
    undated_count = 0
    for item in omitted:
        key = week_bucket_key(item.date)
        if key is None:
            undated_count += 1
            continue
        if key not in by_week:
            by_week[key] = 0
            order.append(key)
        by_week[key] += 1
    order.sort(reverse=True)
    parts = []
    for year, week in order:
        try:
            week_start = datetime.fromisocalendar(year, week, 1).date().isoformat()
        except ValueError:
            week_start = f"{year}-W{week:02d}"
        parts.append(f"{by_week[(year, week)]} from week of {week_start}")
    if undated_count:
        parts.append(f"{undated_count} undated")
    detail = "; ".join(parts) if parts else "unspecified dates"
    return f"(… {len(omitted)} further items omitted for space — by week: {detail})"
