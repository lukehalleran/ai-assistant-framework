"""
Action Claim Guard — anti-confabulation for side-effecting actions.

Module Contract
- Purpose: Detect when an assistant response either (a) PROPOSES a side-effecting
  action ("Want me to save this as a note?") or (b) CLAIMS to have COMPLETED one
  ("Done — saving the 2-week plan as a note"), and reconcile completion claims
  against what actually executed during the turn. This closes the confabulation
  failure mode where the model narrates success for an action it never performed
  (e.g. a daemon self-note that was never written because the turn ran in a
  tool-less generation mode).
- Inputs:
  - detect_proposals(text) -> list[DetectedAction]      (offer/question framing)
  - detect_completion_claims(text) -> list[DetectedAction]  (assertive framing)
  - verify_claims(claims, executed_kinds, proposed_kinds=...) -> ClaimReconciliation
  - build_correction_notice(external_unbacked) -> str   (user-facing correction)
- Outputs: Pydantic models — ActionKind (enum), DetectedAction, ClaimReconciliation.
- Key behaviors:
  - Action taxonomy splits SELF_REPAIRABLE kinds (note, document — safe, internal,
    idempotent-ish) from EXTERNAL kinds (email, calendar, message, github). EXTERNAL
    claims are NEVER auto-executed by the guard; they are corrected/flagged only.
  - Proposal detection requires an offer marker ("want me to", "should I") or a
    trailing question mark. Completion detection requires an assertive completion
    cue (past/progressive verb, "done — …", "I'll …", "is saved") AND excludes
    proposals/questions.
  - A clause must mention an action *kind* keyword to be considered at all, which
    keeps generic prose ("note that you have a deadline") from matching.
- Side effects: NONE. This module is pure detection + classification. Actually
  executing or repairing an action is the caller's responsibility.
- Dependencies: stdlib re + pydantic. No LLM, no I/O.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================================
# Taxonomy
# ============================================================================


class ActionKind(str, Enum):
    """Kinds of side-effecting action a response may propose or claim."""

    NOTE = "note"            # daemon self-note — internal, self-repairable
    DOCUMENT = "document"    # generated markdown doc — internal, self-repairable
    EMAIL = "email"          # external — human-in-the-loop, never auto-executed
    CALENDAR = "calendar"    # external — calendar event / reminder
    MESSAGE = "message"      # external — telegram / discord / DM
    GITHUB = "github"        # external — issue / comment / PR
    GENERIC = "generic"      # an action claim we can't classify precisely


#: Kinds the guard may transparently self-repair (low-risk, internal artifacts).
SELF_REPAIRABLE: frozenset[ActionKind] = frozenset({ActionKind.NOTE, ActionKind.DOCUMENT})

#: Kinds that touch the outside world — never auto-executed; corrected only.
EXTERNAL: frozenset[ActionKind] = frozenset(
    {ActionKind.EMAIL, ActionKind.CALENDAR, ActionKind.MESSAGE, ActionKind.GITHUB}
)


def is_self_repairable(kind: ActionKind) -> bool:
    return kind in SELF_REPAIRABLE


# ============================================================================
# Data models
# ============================================================================


class DetectedAction(BaseModel):
    """A single proposal or completion claim found in a response."""

    kind: ActionKind
    matched_text: str = Field(..., description="The clause that triggered the match")
    topic: str = Field("", description="Best-effort extracted topic/title hint")
    is_proposal: bool = Field(False, description="True for offers/questions, False for claims")

    @property
    def is_self_repairable(self) -> bool:
        return is_self_repairable(self.kind)


class ClaimReconciliation(BaseModel):
    """Result of checking completion claims against what actually executed."""

    unbacked_claims: list[DetectedAction] = Field(default_factory=list)
    repairable: list[DetectedAction] = Field(default_factory=list)
    external_unbacked: list[DetectedAction] = Field(default_factory=list)

    @property
    def has_issue(self) -> bool:
        return bool(self.unbacked_claims)


# ============================================================================
# Patterns
# ============================================================================

# Kind keyword patterns, checked in priority order. External kinds first so a
# clause mentioning both an external target and a generic word resolves to the
# external (more consequential) kind; NOTE before DOCUMENT (notes are primary).
_KIND_PATTERNS: list[tuple[ActionKind, re.Pattern]] = [
    (ActionKind.EMAIL, re.compile(r"\b(e-?mail(?:s|ed|ing)?)\b", re.IGNORECASE)),
    (ActionKind.CALENDAR, re.compile(r"\b(calendar(?:\s+event)?|reminders?|remind(?:ing)?\s+you)\b", re.IGNORECASE)),
    (ActionKind.MESSAGE, re.compile(r"\b(telegram|discord|dm\s+you|message\s+you|text\s+you)\b", re.IGNORECASE)),
    (ActionKind.GITHUB, re.compile(r"\b(github\s+(?:issue|comment|pr|pull\s+request)|(?:open|file|create)\s+an?\s+issue)\b", re.IGNORECASE)),
    (ActionKind.NOTE, re.compile(r"\b(daemon\s+note|self-?notes?|notes?|memos?|note\s+to\s+self|jot\s+(?:this|it|that)\s+down|write\s+(?:this|it|that)\s+down)\b", re.IGNORECASE)),
    (ActionKind.DOCUMENT, re.compile(r"\b(documents?|write-?ups?|reports?|markdown\s+(?:doc|file))\b", re.IGNORECASE)),
]

# Offer/question framing → proposal, not a claim.
_PROPOSAL_MARKER = re.compile(
    r"\b(want me to|do you want me to|would you like me to|should i|shall i|"
    r"i can|i could|i'?d be happy to|let me know if you(?:'d| would)? like|"
    r"happy to .* if you)\b",
    re.IGNORECASE,
)

# An action verb the assistant could perform — gates question-only proposals so
# that a question merely *mentioning* a kind ("is that your notes?") isn't read
# as an offer to act.
_ACTION_VERB = re.compile(
    r"\b(save|saving|store|storing|create|creating|write|writing|add|adding|"
    r"send|sending|email|emailing|schedule|scheduling|drop|dropping|jot|"
    r"jotting|put|make|making|record|recording|log|logging|set up|put together)\b",
    re.IGNORECASE,
)

# Assertive completion cues. Any match (with a kind keyword present, and no
# proposal/question framing) marks the clause as a completion claim.
_COMPLETION_PATTERNS: list[re.Pattern] = [
    # "Done — saving the 2-week plan as a note"  /  "done, saved ..."
    re.compile(r"\b(?:done|all set|all done)\b[\s,.:;—–-]+\s*(?:saving|saved|creating|created|writing|wrote|adding|added|sending|sent|scheduling|scheduled|storing|stored|jotting|jotted|noting|noted|dropping|dropped)\b", re.IGNORECASE),
    # "I've saved" / "I have created" / "I just added" / "I made a note"
    re.compile(r"\b(?:i'?ve|i have|i)\s+(?:just\s+|already\s+)?(?:saved|stored|created|made|wrote|written|added|recorded|logged|sent|emailed|scheduled|jotted|noted|dropped|put)\b", re.IGNORECASE),
    # "saved the note" / "created a doc" / "added your event"
    re.compile(r"\b(?:saved|created|made|added|stored|recorded|logged|sent|emailed|scheduled|jotted|noted|dropped)\b[\w\s,'-]{0,40}\b(?:the|a|an|your|this|that|it)\b", re.IGNORECASE),
    # "saving X as a note" / "dropping this into a note"
    re.compile(r"\b(?:saving|creating|making|adding|storing|recording|sending|scheduling|dropping|putting|jotting)\b[\w\s,'-]{0,40}\b(?:as|into|to)\s+(?:a|an|your)\b", re.IGNORECASE),
    # "I'll save this as a note" — a tool-less promise that won't be kept this turn
    re.compile(r"\bi'?ll\s+(?:go ahead and\s+)?(?:save|store|create|write|add|record|log|send|email|schedule|jot|note|drop|put)\b", re.IGNORECASE),
    # "the note is saved" / "your event has been scheduled"
    re.compile(r"\b(?:is|has been|have been|'?s)\s+(?:now\s+)?(?:saved|created|added|stored|sent|emailed|scheduled|recorded|logged|written)\b", re.IGNORECASE),
]

# Sentence splitter — split on . ! ? and newlines ONLY. Deliberately NOT on
# em-dashes, so "Done — saving the note" stays a single clause with verb + kind.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _detect_kind(clause: str) -> ActionKind | None:
    for kind, pat in _KIND_PATTERNS:
        if pat.search(clause):
            return kind
    return None


def _extract_topic(clause: str) -> str:
    """Best-effort topic hint — only the explicit "about X" phrase, else "".

    Verb-stripping heuristics produced noisy titles, so we keep this narrow and
    let callers fall back to a body-derived title when no clean topic is found.
    """
    m = re.search(r"\b(?:about|regarding)\s+(.+)", clause, re.IGNORECASE)
    if not m:
        return ""
    cand = re.sub(r"\b(this|that|it|the|a|an)\b", " ", m.group(1), flags=re.IGNORECASE)
    cand = re.sub(r"\s{2,}", " ", cand).strip(" ,.;:—–-?!")
    return cand[:80]


# ============================================================================
# Public API
# ============================================================================


def detect_proposals(text: str) -> list[DetectedAction]:
    """Find clauses where the assistant OFFERS to perform an action.

    A proposal is a clause that mentions an action kind AND is framed as an offer
    (proposal marker) or a question (trailing '?'). These do not need backing —
    they await the user's confirmation.
    """
    out: list[DetectedAction] = []
    for sent in _split_sentences(text):
        kind = _detect_kind(sent)
        if kind is None:
            continue
        is_question = sent.rstrip().endswith("?")
        has_marker = bool(_PROPOSAL_MARKER.search(sent))
        # An offer marker is sufficient; a bare question must also carry an
        # action verb to count (so "is that your notes?" is excluded).
        if has_marker or (is_question and _ACTION_VERB.search(sent)):
            out.append(
                DetectedAction(
                    kind=kind,
                    matched_text=sent,
                    topic=_extract_topic(sent),
                    is_proposal=True,
                )
            )
    return out


def detect_completion_claims(text: str) -> list[DetectedAction]:
    """Find clauses where the assistant CLAIMS to have completed an action.

    A completion claim mentions an action kind, carries an assertive completion
    cue, and is NOT framed as a proposal/question. These require backing — proof
    that the action actually ran this turn.
    """
    out: list[DetectedAction] = []
    for sent in _split_sentences(text):
        kind = _detect_kind(sent)
        if kind is None:
            continue
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue  # it's an offer, not a claim
        if any(p.search(sent) for p in _COMPLETION_PATTERNS):
            out.append(
                DetectedAction(
                    kind=kind,
                    matched_text=sent,
                    topic=_extract_topic(sent),
                    is_proposal=False,
                )
            )
    return out


def verify_claims(
    claims: list[DetectedAction],
    executed_kinds: set[ActionKind] | frozenset[ActionKind],
    proposed_kinds: set[ActionKind] | frozenset[ActionKind] | None = None,
) -> ClaimReconciliation:
    """Reconcile completion claims against what actually executed this turn.

    Args:
        claims: completion claims found in the response.
        executed_kinds: action kinds that genuinely ran (note written, doc saved,
            email actually sent, etc.).
        proposed_kinds: action kinds that were merely *proposed* this turn (e.g. a
            pending email card awaiting GUI approval). A proposed-but-not-executed
            external action still makes a "I sent it" claim unbacked, but the
            caller may want to message it differently.

    Returns:
        ClaimReconciliation splitting unbacked claims into self-repairable vs
        external.
    """
    proposed_kinds = proposed_kinds or frozenset()
    rec = ClaimReconciliation()
    seen: set[tuple[ActionKind, str]] = set()
    for c in claims:
        if c.kind in executed_kinds:
            continue  # the claim is backed by a real execution
        key = (c.kind, c.matched_text)
        if key in seen:
            continue
        seen.add(key)
        rec.unbacked_claims.append(c)
        if c.is_self_repairable:
            rec.repairable.append(c)
        else:
            rec.external_unbacked.append(c)
    return rec


_KIND_LABEL = {
    ActionKind.EMAIL: "send that email",
    ActionKind.CALENDAR: "add that to your calendar",
    ActionKind.MESSAGE: "send that message",
    ActionKind.GITHUB: "make that GitHub change",
    ActionKind.GENERIC: "do that",
}


def build_correction_notice(external_unbacked: list[DetectedAction]) -> str:
    """Build a short, honest correction for external claims that didn't execute.

    The guard never auto-performs external actions, so the best it can do is keep
    the record honest and offer to actually do it.
    """
    if not external_unbacked:
        return ""
    kinds = []
    for a in external_unbacked:
        label = _KIND_LABEL.get(a.kind, _KIND_LABEL[ActionKind.GENERIC])
        if label not in kinds:
            kinds.append(label)
    joined = kinds[0] if len(kinds) == 1 else (", ".join(kinds[:-1]) + f" or {kinds[-1]}")
    return (
        f"\n\n> ⚠️ Heads up — I didn't actually {joined}. That needs an explicit "
        f"action step, which didn't run this turn. Want me to do it now?"
    )
