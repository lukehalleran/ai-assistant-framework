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

PERSONAL_CLAIM_MARKER = (
    "[Personal-claim check: this assistant response contains claims with "
    "insufficient or conflicting user evidence. Treat it as assistant "
    "interpretation, not a user completion report.]"
)
UNVERIFIED_CLAIM_MARKER = "[unverified action claim]"
READ_TIME_MARKERS = (PERSONAL_CLAIM_MARKER, UNVERIFIED_CLAIM_MARKER)
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
DELIVERY_NOTICE_TEXTS = (
    NOTICE_NO_CARD,
    NOTICE_NOT_ACTUALLY_DONE,
    NOTICE_CALENDAR_UNSEEN,
    NOTICE_UPLOAD_STALE,
    NOTICE_NOTE_PARTIAL,
    NOTICE_WEB_BUDGET,
    NOTICE_WEB_BUDGET_PARTIAL,
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
        kept = [line for line in lines if line.strip() not in READ_TIME_MARKERS]
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


def strip_machinery(text: str) -> str:
    """Strip both read-time markers and any trailing delivery notice.

    The canonical transform for anything that hashes, dedups, or re-renders
    a stored assistant turn: machinery text glued on after generation is not
    content the assistant produced. Non-str or empty input is returned
    unchanged; this never raises.
    """
    try:
        if not isinstance(text, str) or not text:
            return text
        return strip_delivery_notices(strip_read_time_markers(text))
    except Exception:
        return text
