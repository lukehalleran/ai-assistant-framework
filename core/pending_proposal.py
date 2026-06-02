"""
Pending Proposal Store — deferred action follow-through.

Module Contract
- Purpose: When the assistant PROPOSES a side-effecting action and asks for
  confirmation ("Want me to drop this into a daemon note?"), capture the proposal
  together with the content it would persist. A later short affirmation ("sure",
  "yes", "go ahead") then executes it. This closes the proposal → affirm → execute
  loop that the agentic gate's casual/short-message skip otherwise drops on the
  floor (the affirmation turn runs in a tool-less mode and the action never fires).
- Inputs:
  - PendingProposalStore.bump_turn()                 (call once per user turn)
  - PendingProposalStore.capture(proposal)           (after a proposing response)
  - PendingProposalStore.consume_if_affirmed(text)   (at the start of the next turn)
  - is_affirmation(text) -> bool
  - build_proposal_from_response(response, detected, *, turn, session_id) -> PendingProposal
- Outputs: PendingProposal Pydantic model (or None when nothing is pending/affirmed).
- Key behaviors:
  - Single-slot, session-scoped (latest proposal wins). One-shot consume.
  - TTL in turns (PENDING_PROPOSAL_TTL_TURNS); an unaffirmed proposal older than
    the TTL is discarded rather than acted on much later out of context.
  - Affirmation detection is conservative: short utterance, an affirmation token,
    no negation. Avoids treating "no thanks" or a new instruction as a yes.
- Side effects: NONE beyond its own in-memory slot. Persisting the action is the
  caller's job.
- Dependencies: stdlib re/time + pydantic + core.action_claim_guard (ActionKind,
  DetectedAction). No LLM, no I/O.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from core.action_claim_guard import ActionKind, DetectedAction


# ============================================================================
# Affirmation detection
# ============================================================================

# Strong affirmation tokens/phrases. Matched at the start of (or as) a short
# utterance. "sure that makes sense" → starts with "sure" → affirmed.
_AFFIRMATIONS: tuple[str, ...] = (
    "yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "k", "kk",
    "do it", "do that", "go ahead", "go for it", "please do", "yes please",
    "sounds good", "sounds great", "that works", "works for me", "makes sense",
    "that makes sense", "good idea", "great", "perfect", "absolutely",
    "definitely", "of course", "for sure", "sure thing", "yes do it",
    "please", "alright", "all right", "fine", "let's do it", "lets do it",
)

# Negations / declines — veto an otherwise-affirmative-looking message.
_NEGATION = re.compile(
    r"\b(no|nope|nah|don'?t|do not|never mind|nevermind|nvm|not now|"
    r"not yet|hold off|wait|stop|cancel|skip|instead|actually no)\b",
    re.IGNORECASE,
)

# Normalize leading filler so "ok sure" / "yeah that makes sense" still match.
_LEADING_FILLER = re.compile(r"^(?:well|so|um|uh|hmm|okay|ok|yeah|oh)[\s,]+", re.IGNORECASE)

_MAX_AFFIRMATION_WORDS = 8


def is_affirmation(text: str) -> bool:
    """True if ``text`` is a short, unambiguous yes to a prior proposal."""
    if not text:
        return False
    t = text.strip().lower()
    t = t.strip(".!? \t\n")
    if not t:
        return False
    if _NEGATION.search(t):
        return False
    if len(t.split()) > _MAX_AFFIRMATION_WORDS:
        return False

    # Exact match.
    if t in _AFFIRMATIONS:
        return True
    # Starts-with match ("sure that makes sense", "yes please do it").
    for aff in _AFFIRMATIONS:
        if t == aff or t.startswith(aff + " ") or t.startswith(aff + ","):
            return True
    # Tolerate one leading filler token, then re-check the head.
    stripped = _LEADING_FILLER.sub("", t).strip()
    if stripped and stripped != t:
        for aff in _AFFIRMATIONS:
            if stripped == aff or stripped.startswith(aff + " ") or stripped.startswith(aff + ","):
                return True
    return False


# ============================================================================
# Data model
# ============================================================================


class PendingProposal(BaseModel):
    """A captured action proposal awaiting the user's confirmation."""

    kind: ActionKind
    title: str
    body: str = Field("", description="Content to persist on confirmation (e.g. the plan)")
    category: str = Field("implementation", description="Daemon-note category when kind=note")
    topic: str = ""
    created_turn: int = 0
    created_at: float = 0.0
    session_id: str = ""
    source_response: str = Field("", description="The response text the proposal came from")


# ============================================================================
# Store
# ============================================================================


class PendingProposalStore:
    """Single-slot, session-scoped store for the most recent action proposal."""

    def __init__(self, ttl_turns: int = 2) -> None:
        self._proposal: PendingProposal | None = None
        self._turn: int = 0
        self._ttl_turns: int = max(1, int(ttl_turns))

    @property
    def turn(self) -> int:
        return self._turn

    def bump_turn(self) -> int:
        """Advance the turn counter. Call once per user message."""
        self._turn += 1
        return self._turn

    def capture(self, proposal: PendingProposal) -> None:
        """Store a proposal (latest wins), stamping the current turn if unset."""
        if proposal.created_turn == 0:
            proposal.created_turn = self._turn
        self._proposal = proposal

    def peek(self) -> PendingProposal | None:
        return self._proposal

    def clear(self) -> None:
        self._proposal = None

    def consume_if_affirmed(self, user_text: str) -> PendingProposal | None:
        """Return + clear the pending proposal iff ``user_text`` affirms it.

        Returns None when there is no pending proposal, the message is not an
        affirmation, or the proposal has aged past its turn-based TTL (in which
        case the stale proposal is also cleared).
        """
        p = self._proposal
        if p is None:
            return None
        if self._turn - p.created_turn > self._ttl_turns:
            self.clear()
            return None
        if not is_affirmation(user_text):
            return None
        self.clear()
        return p


# ============================================================================
# Construction helper
# ============================================================================


def build_proposal_from_response(
    response: str,
    detected: DetectedAction,
    *,
    turn: int = 0,
    session_id: str = "",
    now: float | None = None,
) -> PendingProposal:
    """Build a PendingProposal from a proposing response + the detected action.

    The note/doc body is the proposing response itself (it contains the actual
    content the assistant just laid out, e.g. the 2-week plan), with the trailing
    offer/question stripped so the persisted artifact reads cleanly.
    """
    body = _strip_offer_tail(response, detected.matched_text)
    title = detected.topic.strip() or _title_from_body(body) or "Untitled note"
    category = _category_for(detected, body)
    return PendingProposal(
        kind=detected.kind,
        title=title[:100],
        body=body,
        category=category,
        topic=detected.topic,
        created_turn=turn,
        created_at=now if now is not None else time.time(),
        session_id=session_id,
        source_response=response,
    )


def _strip_offer_tail(response: str, offer_clause: str) -> str:
    """Remove the trailing offer sentence ("Want me to …?") from the body."""
    if offer_clause and offer_clause in response:
        idx = response.rfind(offer_clause)
        if idx > 0:
            return response[:idx].rstrip()
    return (response or "").strip()


_INTRO_LINE = re.compile(
    r"^(here'?s|here is|okay|ok|sure|alright|so|well|let'?s|let me|sounds good|"
    r"got it|by the way)\b",
    re.IGNORECASE,
)


def _title_from_body(body: str) -> str:
    """First substantive line of the body, skipping generic intro lines."""
    fallback = ""
    for raw in (body or "").splitlines():
        line = raw.strip().lstrip("#").strip(" -—–:")
        if len(line) < 3:
            continue
        if not fallback:
            fallback = line[:80]
        if _INTRO_LINE.match(line):
            continue
        return line[:80]
    return fallback


def _category_for(detected: DetectedAction, body: str) -> str:
    """Pick a daemon-note category from the proposal/body keywords."""
    text = f"{detected.matched_text}\n{body[:400]}"
    if re.search(r"\barchitect(?:ure|ural)\b", text, re.IGNORECASE):
        return "architecture"
    if re.search(r"\bresearch\b", text, re.IGNORECASE):
        return "research"
    if re.search(r"\b(decision|decided|deciding)\b", text, re.IGNORECASE):
        return "decisions"
    return "implementation"
