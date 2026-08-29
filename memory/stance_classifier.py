"""
Shared stance / epistemic classifier — single source of truth for whether a
stored claim is a world-fact or somebody's take on the world.

Motivation (2026-08-23): the graph held ``casey | is | evil`` with weight 1.0 and
truth_score 1.0 — a one-mention crisis-day value judgment stored with exactly
the same epistemic status as ``user | lives_in | chicago``. Nothing in the
pipeline could represent "this is the user's appraisal, not an objective fact",
so the appraisal leaked into query expansion, rendered as a bare assertion in
prompts, and assistant elaborations of it compounded across sessions into
pseudo-facts. This module gives every write path and every consumer ONE
deterministic vocabulary for that distinction.

Stances
-------
  objective   ordinary world-fact shape ("user lives_in chicago")
  appraisal   a value judgment by its author — thick evaluative terms about a
              person/thing ("casey is evil", "user is a failure"). True *as an
              appraisal held by the author*; never assertable in system voice.
  reported    second-hand content: the triple relays what someone else said
              ("casey said user is worthless") — the reporting is the fact.
  inferred    authored by the assistant, not stated by the user (model
              elaborations must never launder into user-stated facts).
  unknown     no stance recorded (legacy data). Consumers must treat unknown
              CONSERVATIVELY: suppression behaviors fire only on explicit
              appraisal/inferred; standing-granting behaviors (settledness)
              require explicit non-elevated evidence.

Public API
----------
  classify_triple_stance(subject, relation, obj, *, author="user") -> StanceResult
  classify_utterance_stance(text, *, speaker="user") -> StanceResult
  is_evaluative_text(text) -> bool
  scope_unresolved_referent(subject, obj, entity_resolver=None) -> Optional[str]
      Pronoun/role subject + evaluative object → user-scoped subject string
      ("user's last partner"). NEVER fuzzy-resolves to a named entity.
  classify_for_storage(subject, relation, obj, *, author, tone_level) -> dict
      Write-time convenience: {"stance": ..., "capture_tone": ...}.
  effective_stance(metadata) -> str
      Read a stored stance with legacy default ("unknown").

Notes
-----
  * Pure and deterministic — no LLM, no IO, no config reads. Sibling of
    ``memory/relation_classifier.py`` (same single-source-of-truth pattern).
  * The evaluative lexicon is deliberately THICK terms only (Bernard Williams'
    sense): words that fuse description and evaluation of a person. Thin or
    ambiguous words ("bad", "good", "hard", "cold") are excluded — they
    false-positive constantly in ordinary narrative.
  * An LLM extractor may propose a stance, but on any lexicon hit THIS
    classifier's verdict overrides (deterministic wins; the LLM only fills
    gaps the lexicon cannot see).
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

VALID_STANCES = ("objective", "appraisal", "reported", "inferred", "unknown")

#: Stance assumed for stored items with no stance metadata (legacy data).
LEGACY_STANCE_DEFAULT = "unknown"


class StanceResult(BaseModel):
    """Outcome of a stance classification, with human-readable reasons."""

    stance: str = Field(default="objective")
    reasons: list[str] = Field(default_factory=list)

    @property
    def is_appraisal(self) -> bool:
        return self.stance == "appraisal"

    @property
    def is_inferred(self) -> bool:
        return self.stance == "inferred"


# --------------------------------------------------------------------------
# Evaluative lexicon — thick evaluative terms about persons. Multi-word
# phrases allowed; matched word-boundary, case-insensitive.
# --------------------------------------------------------------------------

EVALUATIVE_LEXICON = frozenset({
    # moral / character condemnation
    "evil", "abusive", "abuser", "toxic", "manipulative", "manipulator",
    "cruel", "vicious", "monster", "monstrous", "predator", "predatory",
    "narcissist", "narcissistic", "sociopath", "psychopath", "psycho",
    "gaslighter", "gaslighting", "controlling", "vindictive", "heartless",
    "selfish", "dishonest", "liar", "cheater", "unfaithful", "creep",
    "creepy", "disgusting", "vile", "wicked", "rotten", "scum",
    # self-directed / person-directed deprecation
    "worthless", "pathetic", "useless", "hopeless", "failure", "loser",
    "stupid", "idiot", "idiotic", "moron", "dumb", "lazy", "coward",
    "crazy", "insane", "broken", "unlovable", "burden", "disappointment",
    "piece of shit", "waste of space",
    # addiction-framed judgments (the framing is the appraisal; a diagnosis
    # relation like diagnosed_with stays objective via the reporting/relation
    # rules upstream of the lexicon)
    "addict", "junkie", "alcoholic", "drunk",
    # strong positive thick terms (appraisals cut both ways — a stored
    # "X is wonderful" is just as much the author's take as "X is evil")
    "wonderful", "amazing", "incredible", "perfect", "angelic", "saintly",
    "brilliant", "genius", "terrible", "horrible", "awful", "dreadful",
})

_LEXICON_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(t) for t in sorted(EVALUATIVE_LEXICON, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)

# Copula-family relations: the triple asserts a property of the subject, so an
# evaluative object makes the whole triple an appraisal.
_COPULA_RELATIONS = frozenset({
    "is", "was", "are", "were", "is_a", "was_a", "being", "been",
    "seems", "seemed", "acts_like", "acted_like", "behaves_like",
})

# Reporting relations: the triple relays someone's speech — second-hand.
_REPORTING_RELATIONS = frozenset({
    "said", "says", "said_that", "told_me", "told_user", "tells",
    "claims", "claimed", "reported", "reports", "according_to",
    "insisted", "insists", "swears", "swore",
})

# Subject tokens that mean "the user themself".
_USER_SUBJECTS = frozenset({"user", "i", "me", "myself", "the user"})

# --------------------------------------------------------------------------
# Unresolved-referent detection: pronouns and role phrases that must never be
# fuzzy-bound to a named entity when carrying an evaluative object.
# --------------------------------------------------------------------------

_PRONOUN_SUBJECTS = frozenset({
    "she", "he", "they", "her", "him", "them",
    "this person", "that person", "this woman", "that woman",
    "this guy", "that guy", "this man", "that man",
})

# role → canonical scoped phrasing (subject becomes user-scoped, e.g.
# "user's ex"). Matched as the head noun of the subject phrase.
_ROLE_NOUNS = frozenset({
    "ex", "partner", "boyfriend", "girlfriend", "husband", "wife",
    "spouse", "fiance", "fiancee", "date", "roommate", "boss",
    "coworker", "co-worker", "friend", "neighbor", "landlord",
    "therapist", "doctor", "psychiatrist", "teacher", "professor",
})

_ROLE_SUBJECT_RE = re.compile(
    r"^(?:my|the|a|an)?\s*(?:last|first|previous|old|new|current|former)?\s*"
    r"(" + "|".join(re.escape(r) for r in sorted(_ROLE_NOUNS, key=len, reverse=True)) + r")s?$",
    re.IGNORECASE,
)


def is_evaluative_text(text: str) -> bool:
    """True when the text contains a thick evaluative term (word-boundary)."""
    if not text:
        return False
    return bool(_LEXICON_PATTERN.search(text))


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def classify_triple_stance(
    subject: str,
    relation: str,
    obj: str,
    *,
    author: str = "user",
) -> StanceResult:
    """
    Classify a (subject, relation, object) triple's epistemic stance.

    Rule order (first hit wins):
      1. assistant-authored → inferred (model elaborations never become
         user-stated facts, evaluative or not)
      2. reporting relation → reported (second-hand content)
      3. copula relation + evaluative object → appraisal
      4. user subject + evaluative object → appraisal (self-appraisals like
         ``user | feels_like | a failure`` regardless of relation shape)
      5. otherwise → objective
    """
    rel = _norm(relation)
    subj = _norm(subject)
    reasons: list[str] = []

    if _norm(author) == "assistant":
        return StanceResult(stance="inferred", reasons=["assistant-authored"])

    if rel in _REPORTING_RELATIONS:
        return StanceResult(stance="reported", reasons=[f"reporting relation '{rel}'"])

    obj_evaluative = is_evaluative_text(obj)
    if obj_evaluative and rel in _COPULA_RELATIONS:
        reasons.append(f"copula relation '{rel}' + evaluative object")
        return StanceResult(stance="appraisal", reasons=reasons)

    if obj_evaluative and subj in _USER_SUBJECTS:
        reasons.append("user subject + evaluative object")
        return StanceResult(stance="appraisal", reasons=reasons)

    return StanceResult(stance="objective", reasons=["no appraisal/report signal"])


def classify_utterance_stance(text: str, *, speaker: str = "user") -> StanceResult:
    """
    Read-time stance for a free-text utterance (conversation turn, note line).

    assistant speaker → inferred; user speaker with evaluative language →
    appraisal; else objective. Used by the insight sweep's provenance labeler.
    """
    if _norm(speaker) == "assistant":
        return StanceResult(stance="inferred", reasons=["assistant-authored text"])
    if is_evaluative_text(text):
        return StanceResult(stance="appraisal", reasons=["evaluative language in user text"])
    return StanceResult(stance="objective", reasons=["no evaluative language"])


def scope_unresolved_referent(
    subject: str,
    obj: str,
    entity_resolver=None,
) -> Optional[str]:
    """
    When an *evaluative* claim's subject is a pronoun or unnamed role phrase
    ("she", "my last partner"), return a user-scoped subject string
    ("user's last partner") so the claim can never fuzzy-bind to a named
    entity. Returns None when no rescoping is needed (named subject, or the
    object is not evaluative).

    ``entity_resolver`` is accepted for signature stability but deliberately
    NEVER used to resolve the referent to a named node — that binding is
    exactly the failure this function exists to prevent.
    """
    if not is_evaluative_text(obj):
        return None
    subj = _norm(subject)
    if not subj:
        return None
    if subj in _USER_SUBJECTS:
        return None  # self-appraisals keep the user subject

    if subj in _PRONOUN_SUBJECTS:
        return "user's unnamed referent"

    m = _ROLE_SUBJECT_RE.match(subj)
    if m:
        role = m.group(1).lower()
        # preserve the ordinal/temporal qualifier if present ("last partner")
        qual_m = re.search(
            r"\b(last|first|previous|old|new|current|former)\b", subj, re.IGNORECASE
        )
        if qual_m:
            return f"user's {qual_m.group(1).lower()} {role}"
        return f"user's {role}"

    return None


# --------------------------------------------------------------------------
# Write-time API (Phase B): stance + capture-tone for storage metadata.
# --------------------------------------------------------------------------

_ELEVATED_TONE_TOKENS = ("high", "medium", "concern", "crisis")
_NON_ELEVATED_TONE_TOKENS = ("conversational", "casual", "none", "low")


def capture_tone_from_level(tone_level) -> str:
    """Map a tone-level value (str/enum/None) → 'elevated' | 'non_elevated' | 'unknown'."""
    if tone_level is None:
        return "unknown"
    t = str(tone_level).lower()
    if any(tok in t for tok in _ELEVATED_TONE_TOKENS):
        return "elevated"
    if any(tok in t for tok in _NON_ELEVATED_TONE_TOKENS):
        return "non_elevated"
    return "unknown"


def classify_for_storage(
    subject: str,
    relation: str,
    obj: str,
    *,
    author: str = "user",
    tone_level=None,
) -> dict:
    """
    Write-time convenience: classify a triple and derive capture tone in one
    call. Returns ``{"stance": <str>, "capture_tone": <str>}`` suitable for
    merging into fact/edge metadata.
    """
    result = classify_triple_stance(subject, relation, obj, author=author)
    return {
        "stance": result.stance,
        "capture_tone": capture_tone_from_level(tone_level),
    }


def effective_stance(metadata: Optional[dict]) -> str:
    """
    Read a stored item's stance, defaulting legacy (untagged) data to
    ``unknown``. Consumers must treat ``unknown`` conservatively: suppression
    behaviors act only on explicit appraisal/inferred, and standing-granting
    behaviors require explicit non-elevated evidence.
    """
    if not metadata:
        return LEGACY_STANCE_DEFAULT
    stance = metadata.get("stance")
    if stance in VALID_STANCES:
        return stance
    return LEGACY_STANCE_DEFAULT
