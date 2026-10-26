"""utils/read_time_markers.py

Single source of truth for the read-time machinery text that gets appended to
a stored assistant response AFTER it was generated: the personal-claim-check
marker, the unverified-action-claim marker, and the blockquote "delivery
notices" gui/handlers.py appends post-hoc (e.g. the calendar/upload/no-card/
grounding "> ⚠️ ..." lines). None of that text is something the assistant
said — it is read-time or delivery-time annotation glued onto the stored
string — but because display == storage, it re-enters later prompts and
digests as if it were content.

class: BC-91. Plan: ~/daemon_exec/followups_0922_runs/plans/PLAN_20260922_turn_audit_guardfixes.md §3 (B0).

Before this module, `utils/personal_claim_provenance.py` and
`core/action_claim_guard.py` each defined their own marker literal, and
nothing stripped delivery notices before hashing a response — a receipt
computed over the pre-suffix reply could never match the stored (suffixed)
response, and machinery text quoted a stored turn back into a later prompt
as if it were the assistant's own words. `strip_machinery()` is the one
function every consumer that digests, dedups, or re-renders a stored
assistant turn should call first.
"""

from __future__ import annotations

import re

PERSONAL_CLAIM_MARKER = (
    "[Personal-claim check: some claims the ASSISTANT added in this response "
    "were not found in the user's own words. This note is about the "
    "assistant's claims only; it says nothing against anything the user "
    "said, and the user's statements are not unverified.]"
)
# Pre-2026-10-12 wording (B4): still present in stored turns, so the strippers
# must keep recognising it. It read as if the USER's account were unverified
# ("insufficient or conflicting user evidence ... assistant interpretation") and
# a model took a stored assistant reply carrying it as "my own guess" about a
# fact the user had stated. class: BC-91, BC-46.
LEGACY_PERSONAL_CLAIM_MARKERS = (
    "[Personal-claim check: this assistant response contains claims with "
    "insufficient or conflicting user evidence. Treat it as assistant "
    "interpretation, not a user completion report.]",
)
# Span family (BC-97): names WHICH claim was flagged, so the note carries
# information (the generic marker above fired on nearly every stored turn).
PERSONAL_CLAIM_SPAN_OPEN = '[Personal-claim check: the assistant said "'
PERSONAL_CLAIM_SPAN_CLOSE = '" — not found in the user\'s own words; do not repeat it as fact.]'


def personal_claim_span_marker(span: str) -> str:
    """One marker line quoting an unverified assistant claim (newlines flattened)."""
    return f"{PERSONAL_CLAIM_SPAN_OPEN}{' '.join(str(span).split())}{PERSONAL_CLAIM_SPAN_CLOSE}"


def _is_span_marker_line(stripped: str) -> bool:
    return (stripped.startswith(PERSONAL_CLAIM_SPAN_OPEN)
            and stripped.endswith(PERSONAL_CLAIM_SPAN_CLOSE)
            and len(stripped) >= len(PERSONAL_CLAIM_SPAN_OPEN) + len(PERSONAL_CLAIM_SPAN_CLOSE))


# Legacy bare form (stored turns still carry it; also the fallback when a
# claim cannot be named). The quoted family (BC-97, 2026-10-15) names WHICH
# sentence was unverified: [unverified action claim: "<sentence>"; ... +N more].
UNVERIFIED_CLAIM_MARKER = "[unverified action claim]"
UNVERIFIED_CLAIM_OPEN = '[unverified action claim: "'
UNVERIFIED_CLAIM_CLOSE = "]"
UNVERIFIED_CLAIM_MAX_CLAIMS = 2
UNVERIFIED_CLAIM_MAX_CHARS = 240


def unverified_claim_marker(claims) -> str:
    """ONE marker line quoting up to 2 claim sentences (each capped, ellipsis
    on cut; newlines flattened); a third+ claim becomes ``+N more``. No usable
    claim text -> the legacy bare marker. Quote only the assistant's own text."""
    seen: list[str] = []
    for c in claims or ():
        text = " ".join(str(c).split()).replace('"', "'")
        if not text:
            continue
        if len(text) > UNVERIFIED_CLAIM_MAX_CHARS:
            text = text[:UNVERIFIED_CLAIM_MAX_CHARS - 1].rstrip() + "…"
        if text not in seen:
            seen.append(text)
    if not seen:
        return UNVERIFIED_CLAIM_MARKER
    shown = seen[:UNVERIFIED_CLAIM_MAX_CLAIMS]
    body = '"; "'.join(shown)
    more = len(seen) - len(shown)
    tail = f'" +{more} more' if more else '"'
    return f"[unverified action claim: \"{body}{tail}{UNVERIFIED_CLAIM_CLOSE}"


def is_unverified_marker_line(line: str) -> bool:
    """True for the legacy bare marker or a quoted-family marker line."""
    s = line.strip()
    return s == UNVERIFIED_CLAIM_MARKER or (
        s.startswith(UNVERIFIED_CLAIM_OPEN) and s.endswith(UNVERIFIED_CLAIM_CLOSE)
        and len(s) > len(UNVERIFIED_CLAIM_OPEN))


def contains_unverified_marker(text: str) -> bool:
    return isinstance(text, str) and (
        UNVERIFIED_CLAIM_MARKER in text
        or any(is_unverified_marker_line(ln) for ln in text.split("\n")))


def trailing_unverified_marker(text: str) -> str:
    """The marker line when ``text``'s LAST line is one, else ''."""
    if not isinstance(text, str) or "\n" not in text:
        return ""
    last = text.rsplit("\n", 1)[1]
    return last.strip() if is_unverified_marker_line(last) else ""


def compact_unverified_marker(marker: str, max_len: int) -> str:
    """Shrink a marker line to <= max_len (cut inside the quote, keep the
    closing form); falls back to the legacy bare marker when too tight."""
    if len(marker) <= max_len:
        return marker
    room = max_len - len(UNVERIFIED_CLAIM_OPEN) - len('…"]')
    if marker == UNVERIFIED_CLAIM_MARKER or room < 8:
        return UNVERIFIED_CLAIM_MARKER
    return UNVERIFIED_CLAIM_OPEN + marker[len(UNVERIFIED_CLAIM_OPEN):][:room].rstrip() + '…"]'


READ_TIME_MARKERS = (PERSONAL_CLAIM_MARKER, UNVERIFIED_CLAIM_MARKER, *LEGACY_PERSONAL_CLAIM_MARKERS)
DELIVERY_NOTICE_PREFIX = "> ⚠️"

# Every delivery-notice FAMILY the system can glue onto a reply is declared
# here, once, as its stable opening clause. Emitters never write the
# blockquote literal themselves: they call ``delivery_notice(<NOTICE_*>,
# detail)`` with one of these constants, so the opening a consumer strips is
# by construction the opening an emitter wrote (no second copy to drift —
# BC-58), and a NEW emitter cannot exist without registering its family
# (DM-38 rule 3 rejects a ``> ⚠️`` literal or an unregistered opening outside
# this module). The dynamic tail of a notice (a document date, an action
# label) follows the opening and is never part of the registry.
NOTICE_NO_CARD = "Heads up — there's no card to approve: nothing was actually queued"
NOTICE_NOT_ACTUALLY_DONE = "Heads up — I didn't actually"
NOTICE_CALENDAR_UNSEEN = "I don't see that on your calendar — nothing was created."
NOTICE_UPLOAD_STALE = "No file was uploaded this session; the document"
NOTICE_NOTE_PARTIAL = "Saved to disk, but couldn't update the"
NOTICE_WEB_BUDGET = "I couldn't run a fresh web search because today's search budget"
NOTICE_WEB_BUDGET_PARTIAL = "I couldn't run every web search this needed because today's search budget"
# 2026-09-27 (BC-91, BC-46, BC-44): a raw tool marker (e.g. "<email_search>
# ...</email_search>") that leaked into a delivered reply, unexecuted —
# see core.response_parser.strip_tool_markers.
NOTICE_TOOL_NOT_RUN = "Heads up — that search didn't actually run"
DELIVERY_NOTICE_TEXTS = (
    NOTICE_NO_CARD,
    NOTICE_NOT_ACTUALLY_DONE,
    NOTICE_CALENDAR_UNSEEN,
    NOTICE_UPLOAD_STALE,
    NOTICE_NOTE_PARTIAL,
    NOTICE_WEB_BUDGET,
    NOTICE_WEB_BUDGET_PARTIAL,
    NOTICE_TOOL_NOT_RUN,
)


def delivery_notice(opening: str, detail: str = "", *, separator: str = "\n\n") -> str:
    """Compose ONE delivery notice: ``<separator>> ⚠️ <opening><detail>``.

    ``opening`` must be a registered ``NOTICE_*`` constant — the only way a
    notice family enters the system. ``detail`` is the free tail (starts
    with its own leading space when it continues the sentence). A notice is
    one physical line so ``strip_delivery_notices`` can remove exactly it.
    An unregistered opening raises ValueError: that is a programming error
    caught by the unit tests and the DM-38 scanner, never a runtime state.
    """
    if opening not in DELIVERY_NOTICE_TEXTS:
        raise ValueError(f"unregistered delivery-notice opening: {opening!r}")
    body = f"{opening}{detail}".replace("\n", " ")
    return f"{separator}{DELIVERY_NOTICE_PREFIX} {body}"


# Recalled-time anchors (2026-10-18, BC-100): utils.temporal_resolver.anchor_temporal_references
# annotates a time word in RECALLED text with the date it meant ("today [Sat 2026-10-17]").
# The bracket is prompt machinery; a reply that copies it (live 2026-10-19: "Yesterday
# [Sun 2026-10-18] was a grind … today [Mon 2026-10-19]?") shows it to the user and stores
# it. ONE definition: the emitter's idempotence guard and the stripper both use these.
_WEEKDAY_ABBR = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_MONTH_NAME = (r"(?:January|February|March|April|May|June|July|August|September|October|"
               r"November|December)")
TIME_ANCHOR_BODY = rf"(?:week of )?{_WEEKDAY_ABBR} \d{{4}}-\d{{2}}-\d{{2}}|{_MONTH_NAME} \d{{4}}"
_TIME_ANCHOR_RE = re.compile(
    rf"[ \t]?\[(?:week of )?{_WEEKDAY_ABBR} \d{{4}}-\d{{2}}-\d{{2}}\]"
    rf"|(?<=\bmonth)[ \t]?\[{_MONTH_NAME} \d{{4}}\]",
    re.IGNORECASE,
)


def strip_time_anchors(text: str) -> str:
    """Remove recalled-time anchor brackets (``[Sun 2026-10-18]``,
    ``[week of Sat 2026-10-24]``, ``month [November 2026]``) from assistant
    text; everything else is byte-identical. A weekday-and-ISO-date bracket is
    this module's own format, never prose. Non-str or empty input is returned
    unchanged; this never raises."""
    try:
        if not isinstance(text, str) or not text or "[" not in text:
            return text
        return _TIME_ANCHOR_RE.sub("", text)
    except Exception:
        return text


def strip_read_time_markers(text: str) -> str:
    """Drop every line whose stripped form is exactly a read-time marker.

    Every other line is kept byte-identical (only the marker lines are
    removed; nothing else is rewritten). The joined result is rstripped.
    Non-str or empty input is returned unchanged; this never raises.
    """
    try:
        if not isinstance(text, str) or not text:
            return text
        lines = text.split("\n")
        kept = [line for line in lines if line.strip() not in READ_TIME_MARKERS
                and not _is_span_marker_line(line.strip())
                and not is_unverified_marker_line(line)]
        return "\n".join(kept).rstrip()
    except Exception:
        return text


def strip_delivery_notices(text: str) -> str:
    """Drop the trailing delivery-notice block, if the text ends with one.

    Remove registered notice suffixes from the end, one at a time. Each
    notice is emitted as a blockquote after the reply body; removal begins at
    the actual notice line, preserving any authored quote immediately before
    it. Current emitters use one physical line per notice. Unknown warning
    text is retained. Non-str or empty input is returned unchanged; this
    never raises.
    """
    try:
        if not isinstance(text, str) or not text:
            return text
        lines = text.rstrip().split("\n")
        cursor = len(lines) - 1
        notice_starts = []
        # Only a suffix made entirely of blank separators and registered
        # notices is machinery. A notice quoted earlier in the answer, or
        # followed by ordinary answer text, is content and stays untouched.
        while cursor >= 0:
            if not lines[cursor].strip():
                cursor -= 1
                continue
            line = lines[cursor].strip()
            if not line.startswith(DELIVERY_NOTICE_PREFIX):
                break
            body = line[len(DELIVERY_NOTICE_PREFIX):].strip()
            if not any(body.startswith(registered) for registered in DELIVERY_NOTICE_TEXTS):
                break
            notice_starts.append(cursor)
            cursor -= 1
        if not notice_starts:
            return text.rstrip()
        return "\n".join(lines[: min(notice_starts)]).rstrip()
    except Exception:
        return text


# ---------------------------------------------------------------------------
# User-side note family (2026-10-25, class: BC-91, BC-58)
# ---------------------------------------------------------------------------
# Notes Daemon's own deterministic audits append to the USER's message so the
# model sees them in [CURRENT QUERY]. They are machinery, not the user's words:
# they ride the prompt only (gui.handlers prompt-only suffix) and must never
# reach classifier input or stored query text. Producers
# (utils.attachment_audit, core.active_document) build their notes from these
# constants. Deliberately NOT part of strip_machinery/strip_delivery_notices:
# those are assistant-side strippers with other callers.
ATTACHMENT_NOTE_PREFIX = "[ATTACHMENT NOTE]"
DEADLINE_NOTE_PREFIX = "[DEADLINE NOTE]"
SELF_REFERENCE_NOTE_PREFIX = "[SELF-REFERENCE NOTE]"
ACTIVE_DOCUMENT_PASSAGE_PREFIX = "[ACTIVE DOCUMENT —"
ACTIVE_DOCUMENT_NOTE_PREFIX = "[ACTIVE DOCUMENT NOTE]"
USER_SIDE_NOTE_PREFIXES = (
    ATTACHMENT_NOTE_PREFIX,
    DEADLINE_NOTE_PREFIX,
    SELF_REFERENCE_NOTE_PREFIX,
    ACTIVE_DOCUMENT_PASSAGE_PREFIX,
    ACTIVE_DOCUMENT_NOTE_PREFIX,
)


def is_user_side_note_line(line: str) -> bool:
    """True when ``line`` (after leading whitespace) opens with a user-side note prefix."""
    if not isinstance(line, str):
        return False
    return line.lstrip().startswith(USER_SIDE_NOTE_PREFIXES)


def strip_machinery(text: str) -> str:
    """Strip read-time markers, recalled-time anchors and any trailing delivery notice.

    The canonical transform for anything that hashes, dedups, or re-renders
    a stored assistant turn: machinery text glued on after generation is not
    content the assistant produced. Non-str or empty input is returned
    unchanged; this never raises.
    """
    try:
        if not isinstance(text, str) or not text:
            return text
        return strip_delivery_notices(strip_read_time_markers(strip_time_anchors(text)))
    except Exception:
        return text
