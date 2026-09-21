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
  - is_first_person_claim(text) -> bool   (assistant's own action vs passive/narration)
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
  - Completion detection also excludes narration of someone else's action: a clause
    where the verb is governed by a non-assistant subject ("He emailed his
    counselor", "you saved your note") OR a second-person SUBJECT doing anything
    ("the email's sent — you caught the address issue", "you did the thing") is the
    user/third party acting, not the assistant. A first-person marker ("I saved …",
    "we created …") overrides the exclusion; object-"you" ("sent you the draft") and
    possessive-"your" ("saved your note") stay real self-claims.
  - Before scanning, drafted/quoted regions are stripped (`--- … ---` fences,
    blockquotes, code fences): an email the assistant DRAFTS is written in the
    user's first person ("When I emailed her"), and must not read as an assistant
    self-claim.
  - The actionability GATE — only correct an unbacked EXTERNAL claim when the
    assistant was expected to act (user requested it / a proposal is pending / the
    claim is first-person via `is_first_person_claim`) — lives in the CALLER
    (`gui/handlers.py:_apply_action_guard`), since it needs user-query + pending-
    proposal state this pure module doesn't see.
- Side effects: NONE. This module is pure detection + classification. Actually
  executing or repairing an action is the caller's responsibility.
- Dependencies: stdlib re + pydantic + utils.trigger_match (leaf,
  whitespace normalization only). No LLM, no I/O for the core detectors.
  The A13 seeds+learned semantic channel (claims_pending_card/
  claims_calendar_state) lazily imports models.model_manager +
  utils.adaptive_exemplars only when the composed grammar misses — a
  missing/unavailable embedder degrades to grammar-only, never an error.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

import utils.temporal_resolver as temporal_resolver
from utils.trigger_match import normalize_ws


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
#
# CALENDAR is split into a STRONG tier (unambiguous calendar nouns — always
# wins) and a WEAK tier (2026-09-10: recurrence-cadence words that also show
# up in ordinary NOTE offers — "weekly note", "note from this session").
# _detect_kind only lets the weak tier resolve to CALENDAR when the clause
# carries no NOTE word too; see the 2026-09-10 referee-followup comment there.
_CALENDAR_STRONG_RE = re.compile(
    r"\b(calendar(?:\s+event)?|events?|appointments?|reminders?|remind(?:ing)?\s+you|"
    r"recurring|repeating|office\s+hours)\b", re.IGNORECASE)
# Bare "event"/"appointment" added 2026-09-01: "Re-queuing the event with
# the corrected date … Approve that one" carried no "calendar" word and
# the confabulated re-queue claim went kind-less. The downstream
# expected-to-act gate still suppresses no-context narration.
# 2026-09-10: "Queuing it now: **QRS 7310 TA Session — Saturdays 11:00
# AM–12:00 PM CT, weekly through December 12**" was kind-less (no
# calendar/event word) so the narrated queue claim went unguarded —
# recurrence words and a named session/office-hours slot are calendar.
_CALENDAR_WEAK_RE = re.compile(r"\b(weekly|sessions?)\b", re.IGNORECASE)
_NOTE_KIND_RE = re.compile(
    r"\b(daemon\s+note|self-?notes?|notes?|memos?|note\s+to\s+self|"
    r"jot\s+(?:this|it|that)\s+down|write\s+(?:this|it|that)\s+down)\b", re.IGNORECASE)
_KIND_PATTERNS: list[tuple[ActionKind, re.Pattern]] = [
    (ActionKind.EMAIL, re.compile(r"\b(e-?mail(?:s|ed|ing)?)\b", re.IGNORECASE)),
    (ActionKind.CALENDAR, _CALENDAR_STRONG_RE),
    (ActionKind.MESSAGE, re.compile(r"\b(telegram|discord|dm\s+you|message\s+you|text\s+you)\b", re.IGNORECASE)),
    (ActionKind.GITHUB, re.compile(r"\b(github\s+(?:issue|comment|pr|pull\s+request)|(?:open|file|create)\s+an?\s+issue)\b", re.IGNORECASE)),
    (ActionKind.NOTE, _NOTE_KIND_RE),
    (ActionKind.DOCUMENT, re.compile(r"\b(documents?|write-?ups?|reports?|markdown\s+(?:doc|file))\b", re.IGNORECASE)),
]

# Offer/question framing → proposal, not a claim.
_PROPOSAL_MARKER = re.compile(
    r"\b(want me to|do you want me to|would you like me to|should i|shall i|"
    r"i can|i could|i'?d be happy to|let me know if you(?:'d| would)? like|"
    r"happy to .* if you|"
    # 2026-09-07: "Confirm and I'll create it" / "say the word and I'll add
    # them" are offers awaiting a go-ahead, not promises made this turn.
    r"confirm and i'?ll|once you confirm|if you confirm|say the word|"
    r"give me the (?:go-?ahead|green light)|ready to (?:create|send|add|schedule|fire|queue))\b",
    re.IGNORECASE,
)

# An action verb the assistant could perform — gates question-only proposals so
# that a question merely *mentioning* a kind ("is that your notes?") isn't read
# as an offer to act.
_ACTION_VERB = re.compile(
    r"\b(save|saving|store|storing|create|creating|write|writing|add|adding|"
    r"send|sending|email|emailing|schedule|scheduling|drop|dropping|jot|"
    r"jotting|put|make|making|record|recording|log|logging|set up|put together|"
    # 2026-09-10: Daemon's own habitual offer verb ("Want me to queue it up
    # with that end date?") was missing, so the offer never crossed the turn
    # boundary and the user's "yes" had no route.
    r"queue|queuing|queueing|re-?queue|book|booking|set|setting|fire|firing)\b",
    re.IGNORECASE,
)

# Assertive completion cues. Any match (with a kind keyword present, and no
# proposal/question framing) marks the clause as a completion claim.
_COMPLETION_PATTERNS: list[re.Pattern] = [
    # "Done — saving the 2-week plan as a note"  /  "done, saved ..."
    # 2026-09-07: "Confirmed — creating the recurring event now" shipped with
    # no proposal; the go-ahead acknowledgers join the done-words.
    re.compile(r"\b(?:done|all set|all done|confirmed|on it|got it|sure thing|you got it)\b[\s,.:;—–-]+\s*(?:saving|saved|creating|created|writing|wrote|adding|added|sending|sent|scheduling|scheduled|storing|stored|jotting|jotted|noting|noted|dropping|dropped|queuing|queueing|booking|booked)\b", re.IGNORECASE),
    # "creating the recurring event now" / "adding it to your calendar now" —
    # a present-progressive action verb closed by "now" asserts an action in
    # flight this turn (an offer would carry a question or offer marker).
    re.compile(r"\b(?:saving|creating|making|adding|storing|recording|sending|scheduling|dropping|putting|booking|queuing|queueing)\b[\w\s,'’:\-–—]{0,60}\bnow\b", re.IGNORECASE),
    # "I've saved" / "I have created" / "I just added" / "I made a note"
    re.compile(r"\b(?:i'?ve|i have|i)\s+(?:just\s+|already\s+)?(?:saved|stored|created|made|wrote|written|added|recorded|logged|sent|emailed|scheduled|jotted|noted|dropped|put|queued|re-?queued)\b", re.IGNORECASE),
    # "Re-queuing the event with the corrected date" — a queue/proposal claim
    # (2026-09-01 live: an enhanced-path turn claimed "Re-queuing the event
    # ... Approve that one" with no backend proposal created that turn)
    re.compile(r"\b(?:re-?)?queu(?:e|ed|ing|eing)\b[\w\s,'-]{0,40}\b(?:the|a|an|your|this|that|it)\b", re.IGNORECASE),
    # "saved the note" / "created a doc" / "added your event"
    re.compile(r"\b(?:saved|created|made|added|stored|recorded|logged|sent|emailed|scheduled|jotted|noted|dropped)\b[\w\s,'-]{0,40}\b(?:the|a|an|your|this|that|it)\b", re.IGNORECASE),
    # "saving X as a note" / "dropping this into a note"
    re.compile(r"\b(?:saving|creating|making|adding|storing|recording|sending|scheduling|dropping|putting|jotting)\b[\w\s,'-]{0,40}\b(?:as|into|to)\s+(?:a|an|your)\b", re.IGNORECASE),
    # "I'll save this as a note" — a tool-less promise that won't be kept this turn
    re.compile(r"\bi'?ll\s+(?:go ahead and\s+)?(?:save|store|create|write|add|record|log|send|email|schedule|jot|note|drop|put)\b", re.IGNORECASE),
    # "the note is saved" / "your event has been scheduled"
    re.compile(r"\b(?:is|has been|have been|'?s)\s+(?:now\s+)?(?:saved|created|added|stored|sent|emailed|scheduled|recorded|logged|written|queued|re-?queued)\b", re.IGNORECASE),
]

# A completion verb governed by an explicit non-assistant subject ("he emailed",
# "she sent", "they created", "you saved") is narration about someone else's
# action — not the assistant claiming IT acted. The optional contraction covers
# "he's"/"you've"/"they'll"; up to two intervening words absorb adverbs ("they
# just created"). Note: \byou\b does NOT match the possessive "your" (no boundary
# before the trailing "r"), so "added that to your calendar" is unaffected.
_OTHER_SUBJECT_CLAIM = re.compile(
    r"\b(?:he|she|they|you)(?:'(?:s|d|ve|ll|re))?\s+(?:\w+\s+){0,2}?"
    r"(?:e-?mailed|e-?mails|e-?mailing|"
    r"sent|sends|sending|"
    r"saved|saves|saving|"
    r"stored|stores|storing|"
    r"created|creates|creating|"
    r"made|makes|making|"
    r"wrote|writes|writing|written|"
    r"added|adds|adding|"
    r"recorded|records|recording|"
    r"logged|logs|logging|"
    r"scheduled|schedules|scheduling|"
    r"jotted|jots|jotting|"
    r"noted|notes|noting|"
    r"dropped|drops|dropping|"
    r"texted|texts|texting|"
    r"messaged|messages|messaging|"
    r"put|puts|putting)\b",
    re.IGNORECASE,
)

# First-person self-claim marker. If present anywhere in the clause, the assistant
# IS asserting its own action even when a third party is also mentioned, so the
# third-person exclusion above must not suppress it.
_FIRST_PERSON_CLAIM = re.compile(r"\b(?:i|we)(?:'(?:ve|ll|d|m|re))?\b", re.IGNORECASE)

# Second-person SUBJECT performing an action ("you caught the address issue",
# "you did the thing", "you sent it") narrates the USER's action — not the
# assistant claiming it acted. We require "you" in a SUBJECT position: at clause
# start, or after a connector / dash / comma, followed by a token that is NOT an
# object pronoun or article. This deliberately does NOT match object "you"
# ("sent you the draft" — the assistant is the actor there), nor possessive
# "your" ("saved your note" stays a real self-claim).
_SECOND_PERSON_SUBJECT = re.compile(
    r"(?:^|[—–:;,\-]\s*|\b(?:and|but|so|or|then|because|since|once)\s+)"
    r"you\b\s+(?!the\b|a\b|an\b|it\b|me\b|us\b|them\b|him\b|her\b|your\b|guys\b)\w+",
    re.IGNORECASE,
)

# Sentence splitter — split on . ! ? and newlines ONLY. Deliberately NOT on
# em-dashes, so "Done — saving the note" stays a single clause with verb + kind.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Markdown horizontal rule / draft fence (--- , *** , ___ on their own line).
_HR = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _strip_quoted_and_drafts(text: str) -> str:
    """Drop drafted/quoted regions so the guard scans only the assistant's OWN voice.

    The assistant routinely composes drafts (emails, letters, messages) written in
    the USER's first person — "When I emailed her…" inside a drafted email is the
    user's past action, not the assistant claiming it sent mail. Such content lives
    in fenced blocks (``` … ```), blockquotes (> …), or between markdown horizontal
    rules (--- … ---). Remove those before completion-claim detection so quoted
    first-person verbs don't trip a spurious "I didn't actually send that" notice.
    """
    if not text:
        return text
    # 1) Strip code fences and blockquotes line-by-line.
    cleaned: list[str] = []
    in_fence = False
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if ln.lstrip().startswith(">"):
            continue  # blockquote (incl. a prior turn's correction notice)
        cleaned.append(ln)
    # 2) Strip content between PAIRED horizontal rules (a fenced draft block). An
    #    unpaired trailing rule (a lone section divider) is left untouched, so a
    #    real claim after a divider is never silently dropped.
    hr_idx = [i for i, ln in enumerate(cleaned) if _HR.match(ln)]
    drop: set[int] = set()
    for a, b in zip(hr_idx[0::2], hr_idx[1::2]):
        drop.update(range(a, b + 1))
    return "\n".join(ln for i, ln in enumerate(cleaned) if i not in drop)


def _is_third_party_narration(clause: str) -> bool:
    """True when a completion verb is driven by a non-assistant subject.

    The guard exists to catch the *assistant* confabulating that *it* performed an
    action. "He emailed his counselor" / "you saved your note" / "the email's sent —
    you fixed the address" narrate a third or second party's action and must not be
    read as an assistant self-claim. A first-person marker anywhere in the clause
    overrides this (the assistant is then asserting its own action, e.g. "I saved it
    after you emailed me").
    """
    if _FIRST_PERSON_CLAIM.search(clause):
        return False
    return bool(_OTHER_SUBJECT_CLAIM.search(clause) or _SECOND_PERSON_SUBJECT.search(clause))


def is_first_person_claim(text: str) -> bool:
    """True when the clause asserts the ASSISTANT's own action (first-person I/we).

    A first-person external claim ("I've sent the email", "Done — I emailed them")
    is high-confidence confabulation: the assistant is explicitly saying IT acted.
    Passive/subjectless external phrases ("the email's sent") are not — those need
    a corroborating action context (a user request or pending offer) before the
    guard treats them as a self-claim worth correcting.
    """
    return bool(_FIRST_PERSON_CLAIM.search(text or ""))


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def split_claim_sentences(text: str) -> list[str]:
    """Public wrapper: drafted/quoted regions stripped, then split into
    sentences — the exact per-sentence view claims_pending_card/
    claims_calendar_state/detect_completion_claims scan. Exposed for
    callers outside this module that need to locate WHICH sentence a claim
    lives in (2026-09-10, round 3: the adaptive-exemplar teacher records
    the specific claim sentence a user's failure report just discredited)."""
    return _split_sentences(_strip_quoted_and_drafts(text or ""))


def _detect_kind(clause: str) -> ActionKind | None:
    for kind, pat in _KIND_PATTERNS:
        if pat.search(clause):
            return kind
    # 2026-09-10 referee follow-up (class BC-06 over-fire): the CALENDAR
    # weak-cadence words ("weekly", "session(s)") also show up in ordinary
    # NOTE offers ("Want me to save a note from this session?", "add that to
    # your weekly note?"). They only resolve to CALENDAR when the clause
    # carries no NOTE word too — otherwise fall through to the NOTE pattern
    # already checked (and missed) above.
    if _CALENDAR_WEAK_RE.search(clause) and not _NOTE_KIND_RE.search(clause):
        return ActionKind.CALENDAR
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
    for sent in _split_sentences(_strip_quoted_and_drafts(text)):
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


def detect_offer_clauses(text: str) -> list[str]:
    """Offer-shaped clauses REGARDLESS of kind word.

    detect_proposals needs a kind word in the same clause; a follow-up offer
    often refers back anaphorically ("Want me to create just the professor
    one now?", "Confirm and I'll create it") and carries the kind only
    elsewhere in the reply. Callers pair this with detect_kind(text) over the
    whole response to resolve the kind (core.actions.registry.offer_action_type).
    """
    out: list[str] = []
    for sent in _split_sentences(_strip_quoted_and_drafts(text)):
        is_question = sent.rstrip().endswith("?")
        has_marker = bool(_PROPOSAL_MARKER.search(sent))
        if (has_marker or is_question) and _ACTION_VERB.search(sent):
            out.append(sent)
    return out


def has_offer_marker(text: str) -> bool:
    """True when ``text`` carries an assistant-offer framing ("want me to",
    "I can", "confirm and I'll"). A bare question with an action verb ("Did
    you add it to your calendar?") asks about the USER's action and must not
    be taken as an offer the user can accept — the forced-action route
    requires this marker."""
    return bool(_PROPOSAL_MARKER.search(text or ""))


def detect_kind(text: str) -> ActionKind | None:
    """Public kind detector over arbitrary text (priority order as _KIND_PATTERNS)."""
    return _detect_kind(_strip_quoted_and_drafts(text or ""))


def detect_completion_claims(text: str) -> list[DetectedAction]:
    """Find clauses where the assistant CLAIMS to have completed an action.

    A completion claim mentions an action kind, carries an assertive completion
    cue, and is NOT framed as a proposal/question. These require backing — proof
    that the action actually ran this turn.
    """
    out: list[DetectedAction] = []
    for sent in _split_sentences(_strip_quoted_and_drafts(text)):
        kind = _detect_kind(sent)
        if kind is None:
            continue
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue  # it's an offer, not a claim
        if _is_third_party_narration(sent):
            continue  # narrates someone else's action, not an assistant self-claim
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


# ---------------------------------------------------------------------------
# Categorized-generic composed claim grammar (2026-09-10, round 3, A13) —
# docs/GENERALIZATION_AUDIT_20260901.md remedy pattern #5, closing the
# docs/BUG_CLASSES.md CM-01 "new regex per phrasing" class for THIS specific
# vocabulary. A claim-narration sentence is very often (THING) + (STATE-VERB
# /MODAL) + (STATE) — "the card should be up", "it's already scheduled" — so
# that shape is now a template over three small tables, bridged by a bounded
# same-sentence gap (never crossing '.', '?', '!' — BC-03 — though we are
# already scoped to one sentence via _split_sentences). Extending coverage
# for a shape that fits the template (round-3 gap: "is already IN PLACE
# from earlier today") is a new row in _CLAIM_STATES, never a new
# hand-written alternative. Idioms that do not fit the template (button
# clicks, "confirm and it", bare "locked in") stay enumerated below exactly
# as before this batch — a rigid grammar covering those too would either
# miss them or over-match ordinary prose.
_CLAIM_THING_RE = r"(?:card|proposal|event|it|that|this)"
_CLAIM_MODAL_RE = (
    r"(?:is|'s|are|should\s+be|will\s+be|'ll\s+be|would\s+be|shall\s+be|"
    r"already|re-?queued|queued|locked\s+in)"
)
# state word/phrase -> claim families it counts toward ("card" and/or
# "calendar"). A state may serve both claims_pending_card AND
# claims_calendar_state.
_CLAIM_STATES: list[tuple[str, frozenset]] = [
    (r"up", frozenset({"card"})),
    # A16 (2026-09-11, round 4, docs/BUG_CLASSES.md BC-58): "is on there"
    # (a calendar-STATE claim, "the event ... is on there through December
    # 12") was matching the card family's bare "there" row, so a reply
    # asserting an existing calendar item wrongly tripped claims_pending_card
    # (-> NO_CARD_NOTICE) instead of the calendar-state backstop. The two
    # families are disambiguated by the same fixed-width lookbehind the
    # A10 ingress chokepoint already guarantees is exactly one space wide
    # (normalize_ws runs before every scan): "on there" -> calendar,
    # bare "there" (not preceded by "on ") -> card.
    (r"(?<!on\s)there", frozenset({"card"})),
    (r"on\s+there", frozenset({"calendar"})),
    (r"ready", frozenset({"card"})),
    (r"waiting", frozenset({"card"})),
    (r"showing", frozenset({"card"})),
    (r"in\s+place", frozenset({"calendar"})),
    (r"on\s+the\s+books", frozenset({"calendar"})),
    (r"on\s+(?:the|your)\s+calendar", frozenset({"calendar"})),
    (r"scheduled", frozenset({"calendar"})),
    (r"created", frozenset({"calendar"})),
    (r"added", frozenset({"calendar"})),
    (r"set\s+up", frozenset({"calendar"})),
]


def _compose_thing_modal_state(family: str) -> str:
    """THING + MODAL + (bounded gap) + STATE alternation for one claim
    family, assembled from the category tables above — a new state word
    for `family` extends both `claims_pending_card`/`claims_calendar_state`
    with zero regex edits at the two call sites below."""
    states = "|".join(pat for pat, families in _CLAIM_STATES if family in families)
    return (
        rf"\b{_CLAIM_THING_RE}\b[^.?!]{{0,45}}?\b{_CLAIM_MODAL_RE}\b"
        rf"[^.?!]{{0,20}}?\b(?:{states})\b"
    )


_CLAIM_CARD_TEMPLATE = _compose_thing_modal_state("card")
_CLAIM_CALENDAR_TEMPLATE = _compose_thing_modal_state("calendar")


# "Approve it and it'll land on your calendar" / "hit approve" — the reply
# directs the user at an approval card. With no proposal created this turn
# that card does not exist (2026-09-07 live: an enhanced-path reply said
# "Firing it again: … Approve it and it should land this time" after a retry
# request that had no tool route). Kind-independent: the bullets under such a
# line often carry no kind word at all ("Zoom link + Piazza-first note").
_APPROVAL_PROMPT_RE = re.compile(
    _CLAIM_CARD_TEMPLATE + r"|"
    # 2026-09-10: "You should see the approval card pop up" pointed at a card
    # that was never created; card-appearance phrasings join the list.
    r"\b(?:approval\s+card|(?:the\s+)?card\s+(?:will|should|'?ll|to)\s+(?:pop\s+up|appear|show(?:\s+up)?)|"
    r"see\s+(?:the|an?|your)\s+(?:approval\s+)?card|card\s+(?:is\s+)?(?:up|ready|waiting))\b|"
    r"\b(?:approve\s+(?:it|that|this|the\s+(?:card|proposal|event|action|request))|"
    r"(?:hit|tap|click|press)\s+approve|the\s+approve\s+button|"
    r"(?:card|proposal)\s+(?:below|above)|queued\s+up\s+and\s+ready|"
    # F12 (2026-09-09): "Queued the deletion: … Confirm and it's off" narrated
    # a card that was never created (a forced-round proposal was rejected and
    # never reached the store). Queue/confirm-directive shapes, word-bounded;
    # "confirm that"/"the queue is" stay excluded (ordinary, non-directive
    # prose) since neither matches "confirm and"/"confirm to"/"queued the".
    r"queued\s+the\s+\w+|confirm\s+and\s+it|confirm\s+to\s+(?:proceed|confirm|finalize|approve)|"
    r"waiting\s+for\s+your\s+(?:confirmation|approval))\b|"
    # A7 (2026-09-10, round 2): gerund/participle/future forms. Live:
    # "Locked in … Approving the card will put it on your calendar" shipped
    # with NO card ever created — "Locked in" and "Approving" are neither
    # a completion claim (no assertive completion cue) nor the original
    # approval-prompt vocabulary above.
    r"\b(?:approving\s+(?:the|that|this)\s+card|"
    r"once\s+you(?:'?ve)?\s+approve[d]?\b|"
    r"it'?s\s+queued|"
    r"will\s+(?:put|land|show\s+up)\s+(?:it\s+)?on\s+your\s+calendar)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Claim-sentence eligibility (2026-09-12, A22, docs/BUG_CLASSES.md BC-04/
# BC-58/BC-76) — a shared VOICE + REFERENT-ANCHOR gate for
# claims_pending_card / claims_calendar_state / annotate_unverified_action_
# claim. Two live over-fires exposed the composed grammar's blind spot: a
# THIRD-PARTY "approve" ("...when asked whether Congress would need to
# approve it, said...") read as though the OWNER were being told to approve
# something, and a bare-pronoun THING+MODAL+STATE hit ("it's there if the
# question resurfaces", "that's given up", "the cleaned-up version", "a good
# one to wake up to", "all there is to it") matched ordinary English with no
# approval-surface word anywhere nearby. Both share one root cause: the
# grammar never asked WHOSE voice a claim is in.
#
# VOICE has two closed-table checks, applied to every card/calendar
# candidate sentence (both the regex AND the seeds+learned semantic
# channel):
#   (1) reported content — a source noun or third party plus a reporting
#       verb ("Congress ... said", "the email says", "asked whether") is
#       someone ELSE's speech/writing being relayed, never Daemon's own
#       claim.
#   (2) a non-owner-directed "approve" mention (card family only) — the
#       bare `approve\s+(?:it|that|...)` alternative above has no subject
#       check, so a third-party approver ("the board will approve that")
#       must not read as an instruction to the addressee.
#
# REFERENT ANCHOR (regex hits only — see the semantic-channel note below),
# CARD FAMILY: a pronoun THING ("it"/"that"/"this") needs an
# approval-surface word in the SAME sentence or the IMMEDIATELY PRECEDING
# one — card/approval/confirm.../queue.../pending — UNAMBIGUOUS, always
# anchor; an eligible owner-directed "approve" mention — always an anchor;
# or "proposal(s)" — AMBIGUOUS, ordinary news/legislative/business
# vocabulary too, so it anchors only alongside a first/second-person
# reference — I/we/you/your — in the SAME sentence, see GAP 2 below. A
# non-pronoun approval-surface THING ("card") always satisfies this
# trivially — the anchor-word scan finds its own matched token — so no
# separate pronoun/non-pronoun branch is needed for the unambiguous
# vocabulary; one word-level scan covers both. As of A23 (below) this
# check is scoped to voice-ok REGIONS (a maximal run of consecutive
# voice-ok clauses within a sentence), not whole sentences, and "the
# immediately preceding one" means any voice-ok region of the preceding
# SENTENCE.
#
# CALENDAR FAMILY: a bare `_CALENDAR_STRONG_RE` anchor word was the WHOLE
# check here through A22 — "does a calendar-strong word appear nearby",
# never "whose calendar". A23 below (2026-09-12, docs/BUG_CLASSES.md BC-04)
# replaces this with full OWNER ATTRIBUTION (S1/S2/S3 direct surface + a D
# definite/anaphoric reference riding on an earlier established surface) —
# see the A23 module note further down for the complete rule. A third-
# party or hypothetical calendar sentence with no owner surface anywhere
# in the reply is no longer a claim just because a calendar noun is
# nearby.
#
# The anchor/attribution check does NOT apply to the A13 seeds+learned
# semantic channel:
# TestA13SeedsAndLearnedSemanticChannel.test_novel_phrasing_near_a_seed_is_
# caught_semantically ("Everything's set on my end — it'll be finalized the
# moment you give it a nod.") is caught ONLY by cosine similarity and names
# no card/proposal/queue/approve word at all, by design — a genuinely novel
# phrasing near a learned exemplar is exactly what the semantic channel
# exists to catch. The voice checks above still apply to semantic hits.
# ---------------------------------------------------------------------------

# 2026-09-12, GAP 1 (frontier adversarial probe): enumerating subjects
# (he/she/they/it/the <noun>) missed a possessive or bare-name subject
# ("Your professor says…", "My advisor said…", "Sam says…") — reported
# content is not about WHO the subject is, it is about whether the SPEAKER
# of this sentence is Daemon itself. Structural rule: a reporting verb whose
# immediately preceding token is any word OTHER than a first-person one
# ("i"/"we") frames reported content — Daemon can only assert its OWN claim
# in the first person; everything else (second- or third-person, a name, a
# possessive, "it") is a claim being relayed, not asserted. "that said" is a
# discourse idiom ("That said, the card is up.") and is carved out
# explicitly, never generalized as a phrase-list entry (this is the ONE
# named exception the rule needs, not a pattern to extend per-miss).
#
# This structural "any preceding word" rule applies ONLY to reporting verbs
# that are unambiguously verbs in ordinary English (says/said/mentions/
# mentioned/writes/wrote/asks/asked — "asks/asked" duplicates the "asked
# whether/if" frame below for the sentence-initial "when asked..." shape).
# reports/reported/notes/noted/states/stated/claims/claimed are ALSO common
# NOUNS ("that exact note", "insurance claims", "financial reports", "US
# states") — the generalized rule flagged "...save that exact note
# yesterday..." as reported content (word="exact", "verb"="note") purely
# because a noun landed next to an adjective. Those four keep the original,
# narrower subject list (he/she/they/it/"the <noun>") — a real subject
# pronoun/determiner before them is unambiguous, an arbitrary adjective is
# not.
_REPORTING_VERB_UNAMBIGUOUS_RE = (
    r"(?:says?|said|mentions?|mentioned|writes?|wrote|asks?|asked)"
)
_REPORTING_VERB_NOUN_AMBIGUOUS_RE = (
    r"(?:reports?|reported|notes?|noted|states?|stated|claims?|claimed)"
)
_REPORTED_CONTENT_RE = re.compile(
    r"\baccording\s+to\b|\basked\s+(?:whether|if)\b"
    rf"|\b(?!(?:i|we)\b)(?!that\s+said\b)\w+\s+{_REPORTING_VERB_UNAMBIGUOUS_RE}\b"
    rf"|\b(?:he|she|they|it|the\s+\w+)\s+{_REPORTING_VERB_NOUN_AMBIGUOUS_RE}\b",
    re.IGNORECASE,
)


def _is_reported_content(sentence: str) -> bool:
    """True when `sentence` frames its content as someone/something else's
    speech or writing being relayed (a news quote, "the email says...",
    "Your professor says...", "when asked whether Congress...") rather than
    Daemon's own first-person claim. "I said"/"we said" (Daemon's own
    voice) and the "that said" discourse idiom are excluded."""
    return bool(_REPORTED_CONTENT_RE.search(sentence))


_APPROVE_VERB_RE = re.compile(r"\bapprov(?:e|es|ed|ing)\b", re.IGNORECASE)
# Clause-boundary set mirrors _SECOND_PERSON_SUBJECT's own clause-start
# detection above (sentence start or a sentence-internal punctuation break)
# plus the ellipsis this module's own dash-joined test fixtures use in place
# of a period ("Locked in … Approving the card will put it on your
# calendar" is ONE _split_sentences unit — no period anywhere in it).
_APPROVE_CLAUSE_LEAD = r"(?:^|[.!?]|[—–:;,\-]|\.\.\.|…)\s*"
_APPROVE_OWNER_DIRECTED_RE = re.compile(
    _APPROVE_CLAUSE_LEAD
    + r"(?:(?:just|then|now|go\s+ahead\s+and|please)\s+)*approv(?:e|ing)\b"
    r"|\byou(?:'(?:ve|d|ll|re))?\s+(?:can\s+|could\s+|should\s+|just\s+|need\s+to\s+)*approve[d]?\b"
    r"|\b(?:once|if|after)\s+you(?:'(?:ve|d))?\s+approve[d]?\b"
    r"|\byou\b[^.?!]{0,20}?\bto\s+approve\b"
    r"|\b(?:hit|tap|click|press)\s+approve\b|\bthe\s+approve\s+button\b",
    re.IGNORECASE,
)
# claims_pending_card/claims_calendar_state normalize_ws the text before
# splitting (A10), which collapses a genuine markdown paragraph break (two
# newlines, no punctuation) to a single space just like an intra-sentence
# soft line-wrap — so a bulleted-list reply's blank line before "Approve it
# and it should land this time." leaves NO punctuation at all ahead of
# "Approve" for _APPROVE_CLAUSE_LEAD to anchor on (live:
# TestNoCardBackstop.test_live_reply_claims_a_card). Capitalization is the
# surviving signal: ordinary English capitalizes "Approve"/"Approving" only
# at the start of what was originally its own sentence, never mid-clause
# after a lowercase-governing subject ("Congress would need to approve it"
# is lowercase). This check is deliberately case-SENSITIVE, unlike the rest
# of this module's grammar.
_APPROVE_CAPITALIZED_RE = re.compile(r"\bApprov(?:e|ing)\b")


def _approve_is_owner_directed(sentence: str) -> bool:
    """True when an "approve" mention in `sentence` is directed AT the
    addressed owner — a clause-initial imperative/gerund ("Approve it",
    "Approving that card…"), a second-person subject ("you can approve",
    "once you approve", "for you to approve"), a click-directive ("hit
    approve", "the approve button"), or a capitalized "Approve"/"Approving"
    surviving from an originally separate sentence (see
    _APPROVE_CAPITALIZED_RE above). A third-party subject ("Congress would
    need to approve it") matches none of these."""
    if _APPROVE_CAPITALIZED_RE.search(sentence):
        return True
    return bool(_APPROVE_OWNER_DIRECTED_RE.search(sentence))


# Card-family anchor vocabulary — card/approval, the confirm-* family (the
# same approval-workflow surface as "Queued the deletion" / "waiting for
# your confirmation" already in the grammar above), and queue/re-queue/
# pending. These are UNAMBIGUOUS: none of them is ordinary vocabulary
# outside the approval-card domain, so any one of them anchors on its own.
_CARD_ANCHOR_UNAMBIGUOUS_RE = re.compile(
    r"\b(?:cards?|approvals?|confirm(?:ation|s|ed|ing)?)\b",
    re.IGNORECASE,
)
# 2026-09-21: queue/re-queue/pending were listed above as "never ordinary
# vocabulary outside the approval-card domain". They are ordinary English —
# live: "if you want the rest of the PR rundown later, it's still queued up"
# drew the NO_CARD_NOTICE on a turn with no action anywhere in it. Worse, the
# grammar's own STATE slot is "queued", so the matched word anchored ITSELF and
# the anchor check could never fail. They are SOFT anchors: enough only when
# the caller knows Daemon was expected to act this turn (the live-reply
# backstop, given action context); the context-free read-time annotation and a
# turn with no action context require a hard anchor.
_CARD_ANCHOR_SOFT_RE = re.compile(
    r"\b(?:queue(?:s|d)?|re-?queu(?:e|ed|ing)|pending)\b",
    re.IGNORECASE,
)
# 2026-09-12, GAP 2 (frontier adversarial probe): "proposal" is ALSO
# ordinary news/legislative/business vocabulary ("The proposal is up for a
# vote in the Senate", "Their proposal is already there") — the Congress
# class by another route, this time through the anchor rather than the
# approve-verb. It is AMBIGUOUS: it only anchors a card claim alongside a
# first/second-person reference (I/we/you/your — a closed grammatical
# category, never a name/third-party pronoun like "their") IN THE SAME
# sentence, since that is what marks the proposal as the OWNER's own rather
# than a news subject's. An owner-directed "approve" mention (checked
# separately below) is always a valid anchor regardless of this ambiguity.
_CARD_ANCHOR_AMBIGUOUS_RE = re.compile(r"\bproposals?\b", re.IGNORECASE)
_PERSON_REFERENCE_RE = re.compile(r"\b(?:i|we|you|your)\b", re.IGNORECASE)


def _card_anchor_present(sentence: str, *, hard_only: bool = False) -> bool:
    if _CARD_ANCHOR_UNAMBIGUOUS_RE.search(sentence):
        return True
    if not hard_only and _CARD_ANCHOR_SOFT_RE.search(sentence):
        return True
    if _APPROVE_VERB_RE.search(sentence) and _approve_is_owner_directed(sentence):
        return True
    if _CARD_ANCHOR_AMBIGUOUS_RE.search(sentence):
        return bool(_PERSON_REFERENCE_RE.search(sentence))
    return False


def _claim_voice_ok(sentence: str, family: str) -> bool:
    """VOICE check only (no anchor) — shared by the regex AND semantic
    detection channels for both families."""
    if _is_reported_content(sentence):
        return False
    if family == "card" and _APPROVE_VERB_RE.search(sentence) and not _approve_is_owner_directed(sentence):
        return False
    return True


# ---------------------------------------------------------------------------
# Clause splitting + voice-ok REGIONS (2026-09-12, A23, docs/BUG_CLASSES.md
# BC-04) — the sentence-level review-finding follow-up
# (/tmp/daemon_sep12_followup_review.md, finding 1). A22's VOICE check
# above operates on a whole SENTENCE, which has two failure modes: (1) a
# reporting frame anywhere in a sentence vetoes the ENTIRE sentence even
# when it also carries an independently-voiced clause ("The email says the
# meeting is already scheduled, and it is already on your calendar for
# Friday at 3 PM." lost the real "it is already on your calendar..." claim
# to the "email says" veto), and (2) neither family previously required
# the claim to be ATTRIBUTED to the addressed OWNER's own calendar at all
# — a third-party/hypothetical calendar sentence with no "your" anywhere
# near it ("The event is already scheduled for March.", "Their appointment
# is already scheduled...") still returned as a claim.
#
# ``_claim_clause_spans`` splits a sentence into independent-clause
# character spans on a CLOSED boundary grammar (semicolon; an em/en dash or
# space-padded hyphen; a comma+coordinator UNLESS followed by a
# complementizer that/whether/if, which continues reported speech; a bare
# coordinator with no comma, only when the next word is a closed set of
# subject pronouns). ``_voice_ok_regions``/`_voice_ok_regions_from`` then
# group consecutive voice-ok clauses (per the existing, UNCHANGED
# ``_claim_voice_ok``) into REGIONS — spanning from the first kept clause's
# start to the last kept clause's end in the ORIGINAL sentence, so text
# between kept clauses (coordinators included) survives inside the region
# and regex spans like "Approve it and it should land" still match. When
# EVERY clause is voice-ok the sole region is the WHOLE sentence —
# byte-identical to the pre-A23 text, so every existing whole-sentence
# equality assertion is unaffected.
#
# The CARD family (``claims_pending_card`` / the annotator's card branch,
# via the shared ``_card_claim_regions``) is rescoped from whole sentences
# to these voice-ok regions: ``_APPROVAL_PROMPT_RE`` searches each region,
# and ``_card_anchor_present`` is checked against that region or any
# voice-ok region of the PRECEDING sentence (unchanged semantics from A22's
# whole-sentence "i-1" check, just region-scoped).
#
# The CALENDAR family (``claims_calendar_state`` / the annotator's calendar
# branch, via the shared ``_calendar_claim_regions``) additionally requires
# OWNER ATTRIBUTION, replacing the old bare ``_calendar_anchor_present``
# (which only asked "does a calendar-strong word appear nearby", never
# "whose calendar"). A calendar-state candidate clause is attributed iff:
#   (a) the clause itself carries an OWNER-CALENDAR SURFACE — S1 (a direct
#       your/our/the/google-calendar mention, or "calendar event/invite/
#       entry/item", or bare "on calendar"), S2 ("your" + up to 4
#       non-genitive modifier words + a calendar-entry noun — "your
#       professor's office hours" is excluded by the genitive), or S3
#       ("you"/"we" have/had (got) + up to 4 modifier words + a
#       calendar-entry noun, unless guarded by a preceding if/unless/
#       whether/"in case" — "If you have an appointment..." is not a claim
#       about an EXISTING appointment); or
#   (b) the clause carries a definite/anaphoric REFERENCE (D — "the/this/
#       that (...) <entry noun>", a bare it/it's/that's/this, or "on
#       there") AND an EARLIER voice-ok clause of the SAME REPLY (an
#       earlier clause of the same sentence, or any clause of an earlier
#       sentence) already established surface. Clauses of skipped sentences
#       (questions / ``_PROPOSAL_MARKER``) and voice-ineligible (reported)
#       clauses never supply an antecedent.
# This lets a bare "it's on your calendar" open a claim outright (a),
# while a later bare "it" or "the event" rides on a surface established
# earlier in the same reply (b) — exactly the R6A/R2/R4/R5 replay pattern
# where a live TA-session calendar event is established once and then
# referred to anaphorically across several sentences.
# ---------------------------------------------------------------------------

_CLAUSE_SUBJECT_PRONOUN_RE = (
    r"(?:it|it['’]s|that['’]s|this|there|there['’]s|you|you['’]ve|you['’]re|your|"
    r"i|i['’]ve|i['’]m|we|we['’]ve|we['’]re|they|they['’]re|he|she)"
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r";"
    # A dash followed by a digit is a numeric range ("8–9 PM", "2 - 3"), not
    # a clause break: splitting there separates "Your appointment runs 2"
    # from "3 on Friday" and the per-clause A21 rule loses the claim.
    r"|\s*[—–](?!\s*\d)\s*"
    r"|\s-(?!\s*\d)\s"
    r"|,\s+(?:and|but|so|yet)\s+(?!(?:that|whether|if)\b)"
    r"|\s+(?:and|but|so|yet)\s+(?=" + _CLAUSE_SUBJECT_PRONOUN_RE + r"\b)",
    re.IGNORECASE,
)


def _claim_clause_spans(sentence: str) -> list[tuple[int, int]]:
    """Character spans of ``sentence``'s independent clauses, in order,
    boundaries excluded (see the closed grammar in the module note above),
    empty/whitespace-only spans dropped."""
    if not sentence:
        return []
    bounds = [(m.start(), m.end()) for m in _CLAUSE_BOUNDARY_RE.finditer(sentence)]
    cuts = [0]
    for s, e in bounds:
        cuts.append(s)
        cuts.append(e)
    cuts.append(len(sentence))
    spans: list[tuple[int, int]] = []
    for i in range(0, len(cuts), 2):
        start, end = cuts[i], cuts[i + 1]
        while start < end and sentence[start].isspace():
            start += 1
        while end > start and sentence[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _claim_sentence_clauses(sentence: str, family: str) -> tuple[list[tuple[int, int]], list[bool]]:
    """(clause spans, per-clause voice-ok flags) for ``sentence`` in
    ``family`` — the shared computation ``_voice_ok_regions_from`` and the
    per-family region builders below all key off."""
    spans = _claim_clause_spans(sentence)
    if not spans:
        spans = [(0, len(sentence))] if sentence.strip() else []
    ok = [_claim_voice_ok(sentence[s:e], family) for s, e in spans]
    return spans, ok


def _voice_ok_regions_from(
    sentence: str, spans: list[tuple[int, int]], ok: list[bool]
) -> list[tuple[int, int, list[int]]]:
    """(region_start, region_end, clause_indices) — maximal runs of
    consecutive voice-ok clauses. If every clause is voice-ok the only
    region is the WHOLE sentence (byte-identical to pre-A23 text)."""
    n = len(spans)
    if n == 0:
        return []
    if all(ok):
        return [(0, len(sentence), list(range(n)))]
    regions: list[tuple[int, int, list[int]]] = []
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        regions.append((spans[i][0], spans[j][1], list(range(i, j + 1))))
        i = j + 1
    return regions


def _voice_ok_regions(sentence: str, family: str) -> list[tuple[int, int]]:
    """Public-ish (contract #2) wrapper: (start, end) character spans of
    ``sentence``'s voice-ok regions in ``family``."""
    spans, ok = _claim_sentence_clauses(sentence, family)
    return [(s, e) for s, e, _ in _voice_ok_regions_from(sentence, spans, ok)]


# ---------------------------------------------------------------------------
# Calendar OWNER-ATTRIBUTION grammar (A23) — see the module note above for
# the full (a)/(b) attribution rule this vocabulary implements.
# ---------------------------------------------------------------------------

#: Exactly the calendar-entry nouns already in ``_CALENDAR_STRONG_RE`` /
#: ``_CALENDAR_WEAK_RE`` — a single constant so S1-S3 and D share one
#: vocabulary.
_CALENDAR_ENTRY_NOUN_RE = r"(?:events?|appointments?|reminders?|office\s+hours|sessions?)"

_S1_OWNER_CALENDAR_RE = re.compile(
    r"\b(?:your|our|the|google)\s+calendars?\b"
    r"|\bcalendar\s+(?:events?|invites?|entr(?:y|ies)|items?)\b"
    r"|\bon\s+calendar\b",
    re.IGNORECASE,
)
_GENITIVE_WORD_RE = re.compile(r"\b\w+['’]s\b")
_S2_YOUR_ENTRY_RE = re.compile(
    r"\byour\b((?:\s+[\w’'-]+){0,4})\s+" + _CALENDAR_ENTRY_NOUN_RE + r"\b",
    re.IGNORECASE,
)
_S3_LEAD_RE = re.compile(
    r"\b(?:you|we)(?:['’]ve|\s+have|\s+had)(?:\s+got)?"
    r"((?:\s+[\w’'-]+){0,4})\s+" + _CALENDAR_ENTRY_NOUN_RE + r"\b",
    re.IGNORECASE,
)
_S3_SUBJECT_RE = re.compile(r"\b(?:you|we)\b", re.IGNORECASE)
_WORD_TOKEN_RE = re.compile(r"[\w’']+")


def _has_surface_s2(text: str) -> bool:
    for m in _S2_YOUR_ENTRY_RE.finditer(text):
        modifiers = m.group(1) or ""
        if not _GENITIVE_WORD_RE.search(modifiers):
            return True
    return False


def _s3_preceding_words_reject(text: str, pos: int) -> bool:
    """True when the two words immediately before ``pos`` disqualify an S3
    match — a conditional guard ("if/unless/whether you have...") or the
    pair "in case"."""
    words = [w.lower() for w in _WORD_TOKEN_RE.findall(text[:pos])]
    tail = words[-2:]
    if any(w in ("if", "unless", "whether") for w in tail):
        return True
    return len(tail) == 2 and tail[0] == "in" and tail[1] == "case"


def _has_surface_s3(text: str) -> bool:
    for m in _S3_LEAD_RE.finditer(text):
        subj_m = _S3_SUBJECT_RE.search(text, m.start(), m.end())
        pos = subj_m.start() if subj_m else m.start()
        if _s3_preceding_words_reject(text, pos):
            continue
        return True
    return False


def _owner_calendar_surface_present(text: str) -> bool:
    """OWNER-CALENDAR SURFACE — S1 | S2 | S3 (see the module note above)."""
    if not text:
        return False
    return bool(_S1_OWNER_CALENDAR_RE.search(text)) or _has_surface_s2(text) or _has_surface_s3(text)


_D_DEFINITE_REFERENCE_RE = re.compile(
    r"\b(?:the|this|that)\s+(?:[\w:-]+\s+){0,6}?(?:calendar\s+)?"
    + _CALENDAR_ENTRY_NOUN_RE + r"\b"
    r"|\b(?:it|it['’]s|that['’]s|this)\b"
    r"|\bon\s+there\b",
    re.IGNORECASE,
)


def _clause_is_calendar_attributed(clause_text: str, surface_before: bool) -> bool:
    """A calendar claim living in ``clause_text`` is ATTRIBUTED iff (a) the
    clause itself carries an owner-calendar SURFACE (S1|S2|S3), or (b) it
    carries a definite/anaphoric REFERENCE (D) and ``surface_before`` says
    an earlier voice-ok clause of the same reply already established
    surface."""
    if _owner_calendar_surface_present(clause_text):
        return True
    return bool(surface_before and _D_DEFINITE_REFERENCE_RE.search(clause_text))


# ---------------------------------------------------------------------------
# Seeds+learned semantic channel (2026-09-10, round 3, A13) —
# docs/GENERALIZATION_AUDIT_20260901.md remedy pattern #2. A composed
# grammar still cannot cover every possible phrasing; per-user LEARNED
# exemplars (utils.adaptive_exemplars, domain "action_claim") grow from an
# INDEPENDENT confirmation channel only (a user failure report —
# registry.is_failure_report, wired at core/agentic/gate.py's
# _prior_turn_offer_action — or an explicit correction the next turn);
# NEVER from this module's own verdict (self-reinforcement guard,
# docs/BUG_CLASSES.md CM-09). record_claim_exemplar() is the only writer.
# Same embedding-cache pattern as utils.web_search_trigger._get_search_anchors.
# ---------------------------------------------------------------------------
_CLAIM_SEED_EXEMPLARS: dict[str, list[str]] = {
    "card_claim": [
        "Locked in. Approving the card will put it on your calendar.",
        "I've queued it. Once you approve, it'll be created.",
        "It's queued and ready for you.",
        "Once you hit approve, it will show up on your calendar.",
    ],
    "calendar_state": [
        "It's also already on your calendar as a recurring weekly event.",
        "The recurring calendar event is already in place from earlier today.",
    ],
}
_CLAIM_SEMANTIC_THRESHOLD = 0.85
_claim_exemplar_text_emb_cache: dict = {}
_claim_anchor_embs: dict = {}
_claim_anchor_version = None


def _get_claim_anchors() -> dict:
    """Lazily embed CLAIM seed+learned exemplars per label, keyed on the
    adaptive store's version (mirrors utils.web_search_trigger's anchor
    cache exactly)."""
    global _claim_anchor_embs, _claim_anchor_version
    try:
        from utils.adaptive_exemplars import get_store  # lazy import: startup-cost
        version = get_store().version
    except Exception:
        version = -1
    if _claim_anchor_embs and _claim_anchor_version == version:
        return _claim_anchor_embs
    try:
        from models.model_manager import ModelManager  # lazy import: startup-cost
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return {}
        from utils.adaptive_exemplars import encode_texts_cached, get_store  # lazy import: startup-cost
        out = {}
        for label, seeds in _CLAIM_SEED_EXEMPLARS.items():
            texts = list(seeds)
            try:
                texts += get_store().get_learned("action_claim", label)
            except Exception:  # degrades: learned claim exemplars skipped, seed-only anchors used
                pass
            out[label] = encode_texts_cached(
                embedder, texts, _claim_exemplar_text_emb_cache, normalize=True
            )
        _claim_anchor_embs = out
        _claim_anchor_version = version
        return out
    except Exception:  # degrades: claim exemplar cache stays empty, semantic claim-narration check disabled
        return {}


def _claim_semantic_hit(sentence: str, label: str) -> bool:
    """True when ``sentence`` is cosine >= 0.85 similar to a seed/learned
    exemplar for ``label`` ("card_claim" or "calendar_state")."""
    if not sentence:
        return False
    anchors = _get_claim_anchors()
    embs = anchors.get(label)
    if embs is None or len(embs) == 0:
        return False
    try:
        from models.model_manager import ModelManager  # lazy import: startup-cost
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return False
        import numpy as np  # lazy import: startup-cost
        q = embedder.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = embs @ q
        return bool(np.max(sims) >= _CLAIM_SEMANTIC_THRESHOLD)
    except Exception:  # degrades: semantic claim similarity skipped, sentence treated as no match
        return False


def record_claim_exemplar(label: str, text: str, source: str) -> bool:
    """Teach a confirmed claim-narration exemplar. INDEPENDENT-channel
    callers only (see module docstring above) — never call this from
    claims_pending_card/claims_calendar_state's own verdict."""
    try:
        from utils.adaptive_exemplars import get_store  # lazy import: startup-cost
        return get_store().record("action_claim", label, text, source)
    except Exception:  # degrades: confirmed claim exemplar not persisted, learning skipped
        return False


def _card_claim_regions(text: str, *, hard_only: bool = False) -> list[str]:
    """Region texts where the CARD family's composed grammar +
    REFERENT-ANCHOR check fire (2026-09-12, A23) — the ONE shared helper
    ``claims_pending_card`` and ``annotate_unverified_action_claim``'s card
    branch both call, so the two can never drift. Question/
    ``_PROPOSAL_MARKER`` sentences are skipped as candidates but their
    voice-ok regions are still computed and available to donate an anchor
    to the FOLLOWING sentence — the same "sentence i-1" semantics A22's
    whole-sentence ``_claim_anchor_ok`` had, just region-scoped. Regex-only
    — never calls ``_claim_semantic_hit`` (annotate's REGEX-ONLY contract
    relies on that)."""
    sentences = _split_sentences(normalize_ws(_strip_quoted_and_drafts(text or "")))
    all_region_texts: list[list[str]] = []
    for sent in sentences:
        spans, ok = _claim_sentence_clauses(sent, "card")
        regions = _voice_ok_regions_from(sent, spans, ok)
        all_region_texts.append([sent[s:e] for s, e, _ in regions])
    hits: list[str] = []
    for i, sent in enumerate(sentences):
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue
        prev_region_texts = all_region_texts[i - 1] if i > 0 else []
        for rt in all_region_texts[i]:
            if _APPROVAL_PROMPT_RE.search(rt) and (
                _card_anchor_present(rt, hard_only=hard_only)
                or any(_card_anchor_present(p, hard_only=hard_only) for p in prev_region_texts)
            ):
                hits.append(rt)
    return hits


def claims_pending_card(text: str) -> bool:
    """True when the reply directs the user to approve a proposal card.

    Offer-framed or question clauses are skipped ("want me to queue it so you
    can approve it?"); quoted/drafted blocks are stripped first (real
    newlines still intact, so fence/blockquote stripping is unaffected),
    THEN the stripped text is whitespace-normalized before sentence
    splitting (2026-09-10, round 3, A10 sibling) — a client-side soft line-
    wrap inside one narration sentence ("...calendar event\n  is already in
    place...") must not get mis-split into two harmless-looking fragments
    at the wrap's newline; a genuine multi-sentence reply splits identically
    either way since real sentence breaks already follow '.', '?', or '!'.

    Detection = the composed grammar OR a seeds+learned semantic hit (A13),
    both scoped to VOICE-OK REGIONS within each sentence (2026-09-12, A23:
    ``_card_claim_regions`` — a clause-level split so a reported clause
    elsewhere in the same sentence can no longer veto an independently
    owner-directed claim, e.g. "The email says the card is pending, and
    you can approve it below." still fires on its second clause) + the
    REFERENT-ANCHOR check (a third-party "approve", reported or not, and
    an unanchored pronoun THING+MODAL+STATE hit like "it's there"/"that's
    given up" are not card claims). The anchor half of the check is
    skipped for the semantic hit (see the module note above A23).
    """
    if _card_claim_regions(text):
        return True
    sentences = _split_sentences(normalize_ws(_strip_quoted_and_drafts(text or "")))
    for sent in sentences:
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue
        for s, e in _voice_ok_regions(sent, "card"):
            if _claim_semantic_hit(sent[s:e], "card_claim"):
                return True
    return False


def card_claim_needs_action_context(text: str) -> bool:
    """True when ``claims_pending_card(text)`` holds ONLY through a soft anchor
    (queue/pending wording) — no card/approval/confirm word, no owner-directed
    "approve", no semantic hit. Such a reply is a card claim only on a turn
    where Daemon was expected to act; the caller decides that."""
    if _card_claim_regions(text, hard_only=True):
        return False
    if not _card_claim_regions(text):
        return False
    sentences = _split_sentences(normalize_ws(_strip_quoted_and_drafts(text or "")))
    for sent in sentences:
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue
        for s, e in _voice_ok_regions(sent, "card"):
            if _claim_semantic_hit(sent[s:e], "card_claim"):
                return False
    return True


NO_CARD_NOTICE = (
    "\n\n> ⚠️ Heads up — there's no card to approve: nothing was actually queued "
    "this turn. Ask me again (or say \"try again\") and I'll queue it for real."
)


# Fresh-upload confabulation (2026-09-10, probe T4): "Can you take a look?"
# resolved the unnamed document to a PDF uploaded five days earlier via the
# upload roster/reuse pool, and the reply asserted "You uploaded
# Homework1-2.pdf today" — a session/recency claim the retrieval path never
# supports (it only knows the file EXISTS, not that this session attached
# it). Checked per-SENTENCE (via _split_sentences, whitespace-after-period
# so a filename's own "." like "Homework1-2.pdf" never splits mid-sentence)
# so the freshness word can be anywhere in the same sentence as the claim,
# without a naive fixed-width gap wrongly excluding filename periods.
_FRESH_UPLOAD_CLAIM_RE = re.compile(
    r"\byou\s+(?:just\s+)?uploaded\b"
    r"|\bthe\s+file\s+you\s+just\s+attached\b"
    r"|\byou\s+(?:just\s+)?attached\b",
    re.IGNORECASE,
)
_FRESH_UPLOAD_TIME_RE = re.compile(
    r"\b(?:today|just\s+now|this\s+session|a\s+(?:moment|minute)\s+ago)\b",
    re.IGNORECASE,
)
_FRESH_UPLOAD_JUST_RE = re.compile(r"\bjust\s+(?:uploaded|attached)\b", re.IGNORECASE)


def claims_fresh_upload(reply: str) -> bool:
    """True when the reply asserts the user uploaded/attached a file THIS
    session or today ("you uploaded ... today", "the file you just
    attached") — a claim the upload-reuse pool never actually supports (it
    only knows a matching file exists, not when THIS session attached it).
    A reply that names an actual past date ("uploaded Sept 5") is not this
    shape. Quoted/drafted blocks are stripped first."""
    for sent in _split_sentences(_strip_quoted_and_drafts(reply or "")):
        if not _FRESH_UPLOAD_CLAIM_RE.search(sent):
            continue
        if _FRESH_UPLOAD_JUST_RE.search(sent) or _FRESH_UPLOAD_TIME_RE.search(sent):
            return True
    return False


# Schedule-narration existence claim (2026-09-11, round 5, A20): "the
# recurring Saturday 11:00 AM CT calendar event runs through December 12
# with the Zoom link attached" carries no MODAL word ("is"/"already"/
# "should be"/...) — the THING+MODAL+STATE template above requires one — it
# narrates the event's ongoing SCHEDULE ("runs ... through December 12")
# rather than asserting a static state. Same category-table method as
# _compose_thing_modal_state (docs/GENERALIZATION_AUDIT_20260901.md remedy
# #5, closing docs/BUG_CLASSES.md's CM-01 "new regex per phrasing" class):
# THING + SCHEDULE-VERB + RANGE/CADENCE, bounded gaps that never cross a
# sentence boundary (belt-and-suspenders — _split_sentences already scopes
# each scan to one sentence). A new schedule verb or cadence word is a new
# table row, never a hand-written alternative.
_SCHEDULE_CAL_THING_RE = r"(?:calendar\s+event|recurring\s+event|series|event|it)"
_SCHEDULE_VERB_RE = r"(?:runs|repeats|recurs|continues|goes|is\s+set)"
_SCHEDULE_RANGE_RE = r"(?:through|until|till|every|weekly|each)"


def _compose_schedule_narration() -> str:
    """THING + SCHEDULE-VERB + RANGE/CADENCE alternation for the calendar
    family, assembled from the three tables above."""
    return (
        rf"\b{_SCHEDULE_CAL_THING_RE}\b[^.?!]{{0,30}}?\b{_SCHEDULE_VERB_RE}\b"
        rf"[^.?!]{{0,30}}?\b{_SCHEDULE_RANGE_RE}\b"
    )


_SCHEDULE_NARRATION_TEMPLATE = _compose_schedule_narration()


# Calendar STATE claim (2026-09-10, round 2, A8): "It's also already on your
# calendar as a recurring weekly event (through December 12, Zoom link
# attached)" for a TA session that was never created. This is a STATE claim
# ("it already exists"), not a completion claim ("I created it") —
# `detect_completion_claims`'s _COMPLETION_PATTERNS require an assertive
# create/save/send verb and deliberately never match "is already on your
# calendar" wording, so it needs its own detector. Offer/question clauses
# are skipped the same way as the other claim detectors here.
_CALENDAR_STATE_RE = re.compile(
    _CLAIM_CALENDAR_TEMPLATE + r"|"
    r"\balready\s+on\s+(?:your\s+)?calendar\b"
    r"|\bis\s+on\s+(?:your\s+)?calendar\b"
    r"|\bit'?s\s+on\s+the\s+calendar\b"
    r"|\byou'?ve\s+got\s+it\s+on\s+the\s+calendar\b"
    r"|\balready\s+scheduled\b"
    r"|" + _SCHEDULE_NARRATION_TEMPLATE,
    re.IGNORECASE,
)


# Entity-anchored calendar-state claims (2026-09-11, round 6, A21) — the
# PRIMARY detector for calendar existence claims from here on (the composed
# THING+MODAL+STATE / schedule-narration templates above are KEPT as a
# fallback — round 2-5 vocabulary like "in place", "on the books" carries
# no temporal anchor and would otherwise stop matching). Live R6a: "...and
# the recurring Saturday 11 AM CT calendar event through December 12" (a
# bare NOUN PHRASE inside a list, no verb) and "Zoom link's on the event"
# (a possessive-state form, no verb) — neither fits ANY verb/modal-anchored
# template. Three rounds of adding a new state/verb row each time this
# happened is BC-76 territory (docs/BUG_CLASSES.md): the generalizable axis
# here is ENTITY + TIME, not grammar. A DECLARATIVE sentence (not a
# question, not an offer, not conditional) that names a calendar-thing noun
# AND a temporal anchor is claiming a concrete scheduled thing exists,
# whether or not it uses a verb to say so. A21 alone is permissive by
# design — it is the CANDIDATE finder, not the final verdict. Two layers
# now keep it honest: the OWNER-ATTRIBUTION gate below (2026-09-12, A23 —
# a candidate clause only becomes a claim when it is attributed to the
# addressed owner's own calendar, never a third party's or a hypothetical
# one) run at the DETECTOR level, and — for a claim that DOES survive
# attribution — the handler-side matcher (gui.handlers.
# _calendar_claim_matches_event) verifies weekday+title agreement against
# [GOOGLE CALENDAR] and only fires the notice when nothing matches, so a
# TRUE claim about a real event is never corrected just because A21 is
# more permissive than a verb-anchored template.
_CONDITIONAL_RE = re.compile(
    r"\b(?:could|would|might|can|if\s+you(?:'d|\s+want|\s+like)|let\s+me\s+know)\b",
    re.IGNORECASE,
)

# Card-lifecycle narration (2026-09-11, round 7): "this exact event was
# already queued yesterday evening, re-queued after you said it failed, and
# queued again this morning at 10:12" names a calendar noun and a clock time
# but describes a PROPOSAL's history, not an event that exists — the A21
# entity rule read it as an existence claim and the handler appended
# "I don't see that on your calendar" under a real, just-minted card. A
# sentence whose calendar noun sits with a proposal-lifecycle verb belongs
# to the card family (`claims_pending_card`), never the calendar family.
_LIFECYCLE_NARRATION_RE = re.compile(
    r"\b(?:re-?queued|queued|proposed|minted|fired|approval\s+card|the\s+card)\b",
    re.IGNORECASE,
)

_calendar_temporal_anchor_re_cache: re.Pattern | None = None


def _calendar_temporal_anchor_re() -> re.Pattern:
    """Weekday name | ``through|until|till|every`` + a month or weekday —
    lazily composed from the existing closed category tables in
    ``core.actions.registry`` (``_WEEKDAY_NAMES``) and
    ``utils.temporal_resolver`` (``_MONTH_NAMES``), never a new
    hand-written phrase list. A clock-time anchor is checked separately via
    ``core.actions.registry._CLOCK_TOKEN_RE`` (that pattern already covers
    "5 pm"/"17:00"/"1700"/bare-hour-after-preposition/noon/midnight shapes
    — reusing it here keeps the vocabulary in ONE place)."""
    global _calendar_temporal_anchor_re_cache
    if _calendar_temporal_anchor_re_cache is not None:
        return _calendar_temporal_anchor_re_cache
    from core.actions.registry import _WEEKDAY_NAMES  # lazy import: cycle
    weekday_alt = "|".join(sorted(_WEEKDAY_NAMES.keys()))
    month_alt = "|".join(sorted(temporal_resolver._MONTH_NAMES.keys()))
    _calendar_temporal_anchor_re_cache = re.compile(
        rf"\b(?:{weekday_alt})\b"
        rf"|\b(?:through|until|till|every)\b[^.?!]{{0,20}}?\b(?:{month_alt}|{weekday_alt})\b",
        re.IGNORECASE,
    )
    return _calendar_temporal_anchor_re_cache


def _has_calendar_temporal_anchor(sent: str) -> bool:
    from core.actions.registry import _CLOCK_TOKEN_RE  # lazy import: cycle
    return bool(_CLOCK_TOKEN_RE.search(sent) or _calendar_temporal_anchor_re().search(sent))


def _is_entity_anchored_calendar_claim(sent: str) -> bool:
    """DECLARATIVE (checked by the caller: not "?", not ``_PROPOSAL_MARKER``)
    + not conditional (``_CONDITIONAL_RE``) + a calendar-thing noun
    (``_CALENDAR_STRONG_RE`` — "calendar event"/"event"/"appointment"/
    "reminder"/"recurring"/"office hours"; a bare "on the event"/"on your
    calendar" possessive-state already matches it via the bare "event"/
    "calendar" tokens, no separate alternative needed) + a temporal anchor
    (weekday name, clock time, or a through/until/till/every cadence word
    with a month or weekday)."""
    if _CONDITIONAL_RE.search(sent):
        return False
    if _LIFECYCLE_NARRATION_RE.search(sent):
        return False  # narrates a CARD's history — the card family judges it
    if not _CALENDAR_STRONG_RE.search(sent):
        return False
    return _has_calendar_temporal_anchor(sent)


def _calendar_claim_regions(
    reply: str, *, use_entity_anchor: bool, use_semantic: bool
) -> list[str]:
    """Core CALENDAR-family algorithm (2026-09-12, A23) shared by
    ``claims_calendar_state`` (``use_entity_anchor=True,
    use_semantic=True``) and ``annotate_unverified_action_claim``'s
    calendar branch (both False — regex-only contract, no A21 permissive
    entity rule). Per sentence: skip questions/``_PROPOSAL_MARKER``
    (contributing nothing, including as a surface antecedent); split into
    clauses and voice-ok REGIONS; within each region, find CANDIDATE
    clauses via ``_CALENDAR_STATE_RE`` (+ ``_is_entity_anchored_calendar_
    claim`` per clause when ``use_entity_anchor``); a region becomes a
    claim when any candidate clause is OWNER-ATTRIBUTED (module note above
    A23) given the surface established by earlier voice-ok clauses of the
    same reply; falling back to a semantic hit on the region (when
    ``use_semantic``) still requires SOME voice-ok clause in the region to
    be attributed — a genuinely novel phrasing still can't ride on zero
    owner-calendar surface anywhere in the reply."""
    out: list[str] = []
    sentences = _split_sentences(normalize_ws(_strip_quoted_and_drafts(reply or "")))
    surface_seen = False
    for sent in sentences:
        if sent.rstrip().endswith("?") or _PROPOSAL_MARKER.search(sent):
            continue
        spans, ok = _claim_sentence_clauses(sent, "calendar")
        clause_texts = [sent[s:e] for s, e in spans]
        n = len(clause_texts)
        surface_before: list[bool] = []
        acc = surface_seen
        for k in range(n):
            surface_before.append(acc)
            if ok[k] and _owner_calendar_surface_present(clause_texts[k]):
                acc = True
        regions = _voice_ok_regions_from(sent, spans, ok)
        for region_start, region_end, clause_idxs in regions:
            region_text = sent[region_start:region_end]
            candidates: list[int] = []
            for m in _CALENDAR_STATE_RE.finditer(region_text):
                m_start = region_start + m.start()
                m_end = region_start + m.end()
                # A match can legitimately span a clause boundary WITHIN a
                # merged (all-voice-ok) region — section 2's own example
                # ("Approve it and it should land") is exactly this shape
                # for the card family; the calendar-state template has the
                # same property ("Approve it and it's on your calendar." —
                # THING="it" starts in one clause, STATE="on your
                # calendar" lands in the next). Every clause the match
                # OVERLAPS is a candidate, not just the one containing the
                # match's start.
                for k in clause_idxs:
                    if spans[k][0] < m_end and spans[k][1] > m_start:
                        candidates.append(k)
            if use_entity_anchor:
                for k in clause_idxs:
                    if _is_entity_anchored_calendar_claim(clause_texts[k]):
                        candidates.append(k)
            seen_c: set[int] = set()
            candidates = [k for k in candidates if not (k in seen_c or seen_c.add(k))]
            if any(_clause_is_calendar_attributed(clause_texts[k], surface_before[k]) for k in candidates):
                out.append(region_text.strip())
                continue
            if use_semantic and _claim_semantic_hit(region_text, "calendar_state") and any(
                _clause_is_calendar_attributed(clause_texts[k], surface_before[k])
                for k in clause_idxs
            ):
                out.append(region_text.strip())
        surface_seen = acc
    return out


def claims_calendar_state(reply: str) -> list[str]:
    """Sentences/regions asserting a calendar event ALREADY exists
    ("already on your calendar", "already scheduled", "it's on the
    calendar", "is already in place") OR, per A21, any declarative clause
    naming a calendar-thing noun plus a temporal anchor (weekday/clock-
    time/cadence) with no verb needed at all — AND (2026-09-12, A23)
    OWNER-ATTRIBUTED to the addressed owner's own calendar, either
    directly (an owner-calendar SURFACE — "your calendar", "your
    appointment", "you have an event...") or anaphorically (a definite/
    pronoun REFERENCE riding on surface established by an EARLIER
    voice-ok clause of the same reply). A third-party or hypothetical
    calendar sentence with no owner surface anywhere in the reply ("The
    event is already scheduled for March.", "Their appointment is already
    scheduled...") is not a claim. Returns the matched region texts (not
    just True/False; a region can be a whole sentence or a sub-sentence
    clause run) so a caller can compare each one's title/weekday/time
    tokens against the turn's actual gathered calendar events. Quoted/
    drafted blocks are stripped first (real newlines still intact for
    fence/blockquote stripping), THEN the stripped text is whitespace-
    normalized before sentence splitting (2026-09-10, round 3, A10
    sibling) — see claims_pending_card's docstring for why.

    Sentence-level VOICE (reported content — "the email says...") no
    longer vetoes an entire multi-clause sentence: clauses are split
    (A23) and grouped into voice-ok REGIONS first, so an independently
    owner-attributed clause in the same sentence as a reported one still
    fires ("The email says the meeting is already scheduled, and it is
    already on your calendar for Friday at 3 PM." returns only the
    second, un-reported clause). See ``_calendar_claim_regions`` for the
    full algorithm; this is a thin wrapper enabling the A21 entity rule
    and the seeds+learned semantic channel (A13)."""
    return _calendar_claim_regions(reply, use_entity_anchor=True, use_semantic=True)


# Unverified-action-claim marker (2026-09-10, round 4, B12; BC-75 via the
# CONVERSATIONS store). A stored Daemon reply that once confabulated a
# completed/existing action re-enters every future turn's prompt context —
# self-notes ([DAEMON SELF-NOTES]), [RECENT CONVERSATION], and [RELEVANT
# MEMORIES] — and both the model and downstream deterministic checks treat
# it as settled fact unless it is flagged IN PLACE at render time. Live:
# "Done — note saved to daemon_notes/ta-sessions-schedule-2026-09-10.md.
# And it's doubly covered: the recurring Saturday 11:00 AM CT calendar
# event (through December 12, Zoom attached) is already in place from
# earlier today." — [RECENT CONVERSATION] item 5, unflagged — became the
# SOURCE the next turn's reply cited for "you had me save that exact note
# yesterday ... the recurring ... calendar event ... is on there".
_UNVERIFIED_CLAIM_MARKER = "[unverified action claim]"

#: Public alias (2026-09-11, round 5, B13 sibling) — a hard-char-capped
#: digest (core.agentic.controller._compute_recent_conversation_digest)
#: needs to recognize and preserve this exact marker through truncation
#: rather than importing the private name.
UNVERIFIED_CLAIM_MARKER = _UNVERIFIED_CLAIM_MARKER


def annotate_unverified_action_claim(text: str) -> str:
    """Append the exact existing marker line "[unverified action claim]"
    to `text` when it contains an unbacked action pending-card/state/
    completion claim — REGEX ONLY (the composed grammars
    ``_APPROVAL_PROMPT_RE`` + ``_CALENDAR_STATE_RE``, plus
    ``detect_completion_claims``). The ``_claim_semantic_hit`` embedding
    channel is deliberately NEVER consulted here: this runs once per
    RENDERED conversation/self-note item (potentially many per turn), not
    once per live reply like the turn-level guard checks in this module —
    an embedding call per rendered item would be an unbounded per-turn
    cost. Under-fires relative to the full live-reply checks by design;
    offer/question-framed sentences are skipped exactly as in
    ``claims_pending_card``/``claims_calendar_state``. Idempotent (a text
    that already carries the marker is returned unchanged). `text` is
    returned unmodified (not `None`) when nothing is found, on empty
    input, or on any internal error — this must never raise into a render
    path.

    Shared by the two conversation-render sites in
    ``core/prompt/formatter.py`` (``_format_memory``, ``mem_parts``) — applied
    to the DAEMON segment only, never the user's own text — and by
    ``core/prompt/gatherer_knowledge.py``'s self-note block, which now
    delegates to this single implementation instead of its own inline
    ``claims_calendar_state(...) or detect_completion_claims(...)`` check.

    Also gated by the clause-scoped VOICE + OWNER-ATTRIBUTION machinery
    (2026-09-12, A23) shared with ``claims_pending_card`` (via
    ``_card_claim_regions``) / ``claims_calendar_state`` (via
    ``_calendar_claim_regions`` with ``use_entity_anchor=False,
    use_semantic=False`` — no A21 permissive entity rule, no embedder
    call), so this stays within the REGEX-ONLY contract above and the
    detector/annotator can never drift on what counts as a claim.
    """
    if not text:
        return text
    if _UNVERIFIED_CLAIM_MARKER in text:
        return text
    try:
        if detect_completion_claims(text):
            return text.rstrip() + "\n" + _UNVERIFIED_CLAIM_MARKER
        if _card_claim_regions(text, hard_only=True):
            return text.rstrip() + "\n" + _UNVERIFIED_CLAIM_MARKER
        if _calendar_claim_regions(text, use_entity_anchor=False, use_semantic=False):
            return text.rstrip() + "\n" + _UNVERIFIED_CLAIM_MARKER
    except Exception:
        return text
    return text


# Content-field conversation items (2026-09-11, round 5, B13). The hybrid
# retriever's `content` field renders a whole turn as one string — "User:
# ...\nDaemon: ..." or "...\nAssistant: ..." — rather than the separate
# query/response fields `annotate_unverified_action_claim` is applied to
# directly at the two formatter render sites. Pointing that same annotator
# at the FULL content string would risk a false hit inside the user's own
# text (a user message that happens to read as a calendar-state claim);
# this wrapper locates the LAST assistant label and re-annotates only the
# text after it, leaving the User segment (and everything before the
# label) byte-identical.
_LAST_ASSISTANT_LABEL_RE = re.compile(
    r"^(?:Daemon|Assistant):[ \t]*", re.MULTILINE
)


def annotate_conversation_content(content: str) -> str:
    """Annotate a rendered "User: ...\\nDaemon: ..." / "...\\nAssistant:
    ..." content-field conversation item: find the LAST "Daemon:"/
    "Assistant:" label (line-start match — start of string or right after
    a newline; case as rendered, i.e. capitalized) and re-annotate only the
    segment AFTER that label via ``annotate_unverified_action_claim``. The
    segment before the label (the User portion, and any earlier turns) is
    returned byte-identical. No label found -> `content` returned
    unchanged. Idempotent — delegates to the already-idempotent
    ``annotate_unverified_action_claim`` — and never raises into a
    gather/render path (any internal error returns `content` unmodified).
    """
    if not content:
        return content
    try:
        matches = list(_LAST_ASSISTANT_LABEL_RE.finditer(content))
        if not matches:
            return content
        m = matches[-1]
        head, tail = content[: m.end()], content[m.end():]
        annotated_tail = annotate_unverified_action_claim(tail)
        if annotated_tail == tail:
            return content
        return head + annotated_tail
    except Exception:
        return content


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
