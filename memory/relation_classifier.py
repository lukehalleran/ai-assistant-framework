"""
Shared relation-state classifier — single source of truth for relation TTLs.

Decides whether a profile/fact *relation* names a transient state (and for how
long it stays "current") versus a durable attribute. Previously three call
sites each carried their own divergent copy of this logic:

  * ``memory/fact_extractor.py``  — exact config list only (storage block)
  * ``memory/user_profile.py``    — exact list + prefix/suffix patterns (profile TTL)
  * ``memory/memory_retriever.py``— exact config list only (facts-collection TTL)

The divergence meant health/recovery relations expired in the profile section
(``health_status`` is caught by the ``_status`` suffix) but never in the facts
collection (the retriever only checked the exact list), and relations whose
names matched no pattern at all (``post_illness_recovery``, ``mood_energy``)
were treated as *durable identity facts* everywhere — so a "recovering from
illness" line surfaced as current indefinitely.

This module unifies the read-side classification and adds a second TTL tier:

  HEALTH_TRANSIENT   illness / recovery / sickness / symptom episode state
                     → ``PROFILE_HEALTH_TRANSIENT_TTL_HOURS`` (medium, ~days)
  STANDARD EPHEMERAL mood / activity / time-of-day state (current_*, woke_*, …)
                     → ``PROFILE_EPHEMERAL_TTL_HOURS`` (short, ~24h)
  (durable)          everything else → no TTL (never expires)

A permanent condition / disability (``disability``, ``chronic_condition``,
``diagnosis``, …) is NOT an illness episode and must never age out, even though
some such names match a transient pattern (``chronic_condition`` ends in the
``_condition`` suffix). Those are pinned durable via ``_DURABLE_OVERRIDES``,
checked before everything else.

Public API
----------
  ephemeral_ttl_hours(relation) -> Optional[float]
      TTL in hours after which a fact with this relation is stale, or None if
      the relation is durable.
  is_ephemeral_relation(relation) -> bool
      Convenience: ``ephemeral_ttl_hours(relation) is not None``.
  health_transient_text_ttl_hours(text) -> Optional[float]
      TTL for *free-text* narrative that frames a transient illness/recovery
      episode (no structured relation), or None. Shares the health-transient
      horizon so a "post-viral fatigue" line in a three-week-old reflection
      ages out of read-time context on the same clock as the structured
      ``health_status -> recovering_from_illness`` graph edge / profile fact.
  is_health_transient_text(text) -> bool
      Convenience: ``health_transient_text_ttl_hours(text) is not None``.

Notes
-----
  * Config TTL values are read lazily on each call so tests can monkeypatch
    ``config.app_config.PROFILE_*_TTL_HOURS`` without re-importing.
  * Health-transient is checked *before* standard-ephemeral so a relation that
    matches both (e.g. ``recovery_status`` matches the ``_status`` suffix and
    the ``recover`` cue) gets the longer health horizon, not the 24h one.
  * This governs *retrieval-time* TTL only. ``fact_extractor.py``'s storage
    block intentionally still uses the exact config list — health-transient
    facts should be *stored* (they're useful for a few days), just aged out on
    read.
  * The free-text detector is deliberately phrase-based and conservative: bare
    "viral" (viral video), "cold" (cold weather), "recovery" (workout recovery)
    are NOT cues — they false-positive in narrative. Only illness-anchored
    phrases match. Durable/chronic language suppresses the match entirely.
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------
# Standard-ephemeral patterns (lifted verbatim from user_profile so profile
# classification of non-health relations is unchanged).
# --------------------------------------------------------------------------

# Suffix patterns that indicate ephemeral/transient facts
_EPHEMERAL_SUFFIXES = (
    "_status", "_condition", "_concern", "_intent",
    "_taken", "_left", "_duration", "_activity",
    "_time", "_deadline", "_variant", "_feeling",
    "_experience", "_plans", "_event",
    "_appointment", "_meeting", "_reschedule",
    "_intake", "_consumption",
)

# Prefix patterns that indicate ephemeral/transient facts
_EPHEMERAL_PREFIXES = (
    "current_", "recent_", "upcoming_", "last_", "next_",
    "time_", "waiting_", "took_", "woke_",
    "scheduled_", "signed_up_", "needs_reschedule",
    "meeting_with_", "needs_meeting",
)

# Exact-match relations that are always ephemeral (one-time events)
_EPHEMERAL_EXACT = frozenset({
    "meeting", "activities", "meeting_with",
    "energy_level", "activity_preference",
    "meal", "meal_choice", "drank_alcohol",
})

# --------------------------------------------------------------------------
# Health-transient patterns (illness / recovery episode state).
# Substring match on the relation name — safe because no durable relation
# contains these tokens (the cue lives in the relation, not the value).
# --------------------------------------------------------------------------

_HEALTH_TRANSIENT_SUBSTRINGS = ("illness", "recover", "sick", "symptom")

# Relations naming a current health state that lack the substrings above.
_HEALTH_TRANSIENT_EXACT = frozenset({
    "health_status", "health_condition",
    "current_condition", "recent_condition",
    "current_health_status", "current_health_condition",
})

# Permanent conditions / disabilities. These must NEVER get a finite TTL even
# though some match the transient patterns (e.g. "chronic_condition" ends in the
# "_condition" suffix; "chronic_illness" contains the "illness" cue). A
# disability is not an illness episode — it does not age out. Checked first.
_DURABLE_OVERRIDES = frozenset({
    "disability", "disabilities", "permanent_disability",
    "chronic_condition", "chronic_conditions", "chronic_illness",
    "medical_condition", "diagnosis", "diagnosed_with",
})

# --------------------------------------------------------------------------
# Free-text health-framing patterns (narrative memories — conversations,
# reflections, summaries, notes). Narrative text has no structured predicate,
# so we match curated transient-illness *phrases*. The TTL horizon is the same
# health-transient horizon used for structured relations (single source of
# truth): a "post-viral fatigue" line in a weeks-old reflection should stop
# reading as present-tense exactly like the graph edge / profile fact already
# do. Conservative on purpose — illness-anchored phrases only.
# --------------------------------------------------------------------------

_HEALTH_TRANSIENT_TEXT_PHRASES = (
    "post-viral", "post viral", "postviral",
    "post-illness", "post illness",
    "recovering from", "still recovering",
    "fighting a virus", "fighting the virus", "fighting the flu",
    "fighting a bug", "fighting a cold", "fighting off a",
    "been sick", "was sick", "still sick", "feeling sick", "feel sick",
    "got sick", "getting sick", "so sick", "i'm sick", "im sick",
    "the flu", "stomach bug", "head cold", "under the weather",
)

# Durable / chronic health language that must NOT be read as a transient
# episode (mirrors _DURABLE_OVERRIDES for relation names). Presence of any of
# these in the text suppresses the transient classification entirely.
_DURABLE_HEALTH_TEXT_PHRASES = (
    "chronic", "disability", "disabled", "autoimmune",
    "diagnosed with", "diagnosis", "permanent", "lifelong",
)


def _config_exact_set() -> frozenset:
    """The canonical ephemeral relation list from config (lazy, uncached so
    test monkeypatching of the config list is respected)."""
    try:
        from config.app_config import PROFILE_EPHEMERAL_RELATIONS
        return frozenset(r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS)
    except ImportError:
        return frozenset()


def _standard_ephemeral_ttl_hours() -> float:
    try:
        from config.app_config import PROFILE_EPHEMERAL_TTL_HOURS
        return float(PROFILE_EPHEMERAL_TTL_HOURS)
    except ImportError:
        return 24.0


def _health_transient_ttl_hours() -> float:
    try:
        from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS
        return float(PROFILE_HEALTH_TRANSIENT_TTL_HOURS)
    except ImportError:
        return 96.0  # 4 days — "a few days", not weeks


def _is_health_transient(rel: str) -> bool:
    if rel in _HEALTH_TRANSIENT_EXACT:
        return True
    return any(tok in rel for tok in _HEALTH_TRANSIENT_SUBSTRINGS)


def _is_standard_ephemeral(rel: str) -> bool:
    if rel in _config_exact_set():
        return True
    if rel in _EPHEMERAL_EXACT:
        return True
    if any(rel.startswith(p) for p in _EPHEMERAL_PREFIXES):
        return True
    if any(rel.endswith(s) for s in _EPHEMERAL_SUFFIXES):
        return True
    return False


def ephemeral_ttl_hours(relation: str) -> Optional[float]:
    """
    Return the TTL in hours for a transient-state relation, or None if the
    relation is durable (never expires).

    Health-transient is checked first so relations that match both tiers get
    the longer health horizon.
    """
    if not relation:
        return None
    rel = relation.lower().strip()
    if rel in _DURABLE_OVERRIDES:  # permanent conditions / disabilities never expire
        return None
    if _is_health_transient(rel):
        return _health_transient_ttl_hours()
    if _is_standard_ephemeral(rel):
        return _standard_ephemeral_ttl_hours()
    return None


def is_ephemeral_relation(relation: str) -> bool:
    """True if the relation names any transient state (has a finite TTL)."""
    return ephemeral_ttl_hours(relation) is not None


def health_transient_text_ttl_hours(text: str) -> Optional[float]:
    """
    Return the health-transient TTL (hours) for *narrative text* that frames a
    transient illness/recovery episode, or None when the text has no such
    framing — or names a durable/chronic condition, which never ages out.

    Unlike :func:`ephemeral_ttl_hours` (which keys on a structured relation
    name), this scans free-text memory content (conversation turns, reflections,
    summaries, notes) for curated illness phrases. The horizon returned is the
    shared health-transient TTL, so narrative ages out of read-time context on
    the same clock as structured relations.
    """
    if not text:
        return None
    t = text.lower()
    if any(p in t for p in _DURABLE_HEALTH_TEXT_PHRASES):
        return None
    if any(p in t for p in _HEALTH_TRANSIENT_TEXT_PHRASES):
        return _health_transient_ttl_hours()
    return None


def is_health_transient_text(text: str) -> bool:
    """True if narrative text frames a transient illness/recovery state."""
    return health_transient_text_ttl_hours(text) is not None
