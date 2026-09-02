"""
core/insight/temporal.py

Module Contract
- Purpose: The pattern_temporal stage of insight mode — runs the DETERMINISTIC
  pattern engine (memory/pattern_engine.py) for a detected pattern request and
  converts its output into (a) PatternResult aggregates for the synthesizer's
  computed-numbers block and (b) EvidenceItem exemplars (REAL store excerpts —
  the engine never fabricates) that join the regular sweep's evidence.
- Inputs: InsightIntent (kind == "pattern_temporal"), live components.
- Outputs: (list[PatternResult], list[EvidenceItem]).
- Key behaviors:
  * Primary query: topic_keyword over the theme's content words (user-side
    mentions only — assistant echoes excluded).
  * Secondary overlays, deterministic: a TONE query when the theme is
    mood-shaped (the tone time series answers "is it getting worse" better
    than keyword counting), and a CONTENT_TYPE query when the theme is about
    songs/lyrics/music shares.
  * Everything is read-only and sync (caller wraps in a thread).
- On-demand only: this stage runs exclusively for an explicit user request
  routed by the insight detector (owner decision 2026-08-29).
- Side effects: none.
"""

from __future__ import annotations

from datetime import datetime
import re

from core.insight.types import EvidenceItem, InsightIntent
from memory.pattern_engine import PatternQuery, PatternResult, run_pattern_query
from utils.logging_utils import get_logger

logger = get_logger("insight_temporal")

_STOPWORDS = frozenset(
    "a an the my our your i me is are was were of about on for with and or "
    "to in it this that been being have has had do does did how often many "
    "times days getting worse better more less over time last past".split()
)
# When the longitudinal planner is unavailable, the fallback still needs a
# topic query.  These are request-framing terms, not user outcomes; allowing
# them through turns a medication question into searches for code branches,
# relationships, and generic "patterns".
# Categorized generic framing vocabulary — NOT one query's transcript (the
# original list was built from a single live medication request and stripped
# subject-capable words like "medication", "therapist", and "spectrum" from
# every theme). Only words that are ~never the SUBJECT of a personal-record
# question belong here; when in doubt, leave the word out (under-strip —
# a noisy keyword costs precision, a stripped subject kills the query).
_REQUEST_FRAMING_STOPWORDS = frozenset(
    # request verbs / analysis meta
    "use using compare comparing analyze analyse analyzing examine examining "
    "assess assessing evaluate evaluating check verify verified review report "
    "separate include weigh weighing weighed consider considered considering "
    "suggest suggested suggesting said raised point points needs need "
    "retrieve search searches searching matching aim formulations "
    # tool/source vocabulary
    "pattern tool tools pubmed web wikipedia wiki corpus notes keyword "
    "keywords semantic papers sources channels data research evidence "
    "result results "
    # discourse / hedges
    "please probably likely maybe clearly broadly deeply relevant related "
    "irrelevant direct available another both some one bit ago "
    # bare temporal framing units (dates and subjects survive)
    "total long term overall month months week weeks".split()
)

# Mood-shaped themes get a tone-series overlay.
_MOOD_THEME_RE = re.compile(
    r"\b(?:mood|moods|sad|sadness|depress\w*|anxious|anxiety|panic|crisis|"
    r"distress\w*|down|low|hopeless\w*|overwhelm\w*|crying|cried|angry|anger|"
    r"upset|stress\w*|bad\s+days?|hard\s+days?|dark|feeling|feelings|felt)\b",
    re.IGNORECASE,
)

# Song/lyrics themes get a content_type overlay.
_MUSIC_THEME_RE = re.compile(
    r"\b(?:songs?|lyrics?|music|playlists?|tracks?)\b", re.IGNORECASE)


def theme_keywords(theme: str, cap: int = 6) -> list[str]:
    """Content words of the theme in first-occurrence order (bge-order
    doctrine), stopwords removed."""
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", (theme or "").lower())
    out: list[str] = []
    for w in words:
        if (w not in _STOPWORDS and w not in _REQUEST_FRAMING_STOPWORDS
                and w not in out):
            out.append(w)
    return out[:cap]


def run_pattern_stage(
    intent: InsightIntent,
    *,
    corpus_manager=None,
    user_profile=None,
    telemetry_path=None,
    now=None,
    spec=None,
    email_rows=None,
) -> tuple[list[PatternResult], list[EvidenceItem]]:
    """Run the deterministic pattern queries for a pattern_temporal intent.
    Never raises — a failed query degrades to a noted empty result."""
    results: list[PatternResult] = []
    window = intent.window_days

    # A frozen time-series phase is the authority when the user's raw wording
    # did not name a simple "last N" window. This keeps the aggregate aligned
    # with the same evidence contract instead of silently using the engine's
    # unrelated default window.
    if not window and spec is not None:
        starts = []
        for phase in spec.phases:
            if not phase.start:
                continue
            try:
                starts.append(datetime.fromisoformat(phase.start).replace(tzinfo=None))
            except (TypeError, ValueError):
                continue
        if starts:
            effective_now = now or datetime.now()
            window = max(1, (effective_now - min(starts)).days + 1)

    keywords = list(
        spec.outcome_terms if spec is not None and spec.outcome_terms
        else theme_keywords(intent.theme)
    )
    if keywords:
        results.append(run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=keywords,
                         window_days=window, now=now),
            corpus_manager=corpus_manager, user_profile=user_profile,
            telemetry_path=telemetry_path,
        ))

    if intent.dimension:  # explicit engine-dimension hint wins
        results.append(run_pattern_query(
            PatternQuery(dimension=intent.dimension, terms=keywords,
                         relation=keywords[0] if keywords else "",
                         window_days=window, now=now),
            corpus_manager=corpus_manager, user_profile=user_profile,
            telemetry_path=telemetry_path, email_rows=email_rows,
        ))
    else:
        if _MOOD_THEME_RE.search(intent.theme):
            results.append(run_pattern_query(
                PatternQuery(dimension="tone", window_days=window, now=now),
                corpus_manager=corpus_manager, user_profile=user_profile,
                telemetry_path=telemetry_path,
            ))
            # Daily-note Emotional State / intensity series: an independent
            # per-day mood source (predates telemetry, no double-counting).
            results.append(run_pattern_query(
                PatternQuery(dimension="daily_notes", window_days=window, now=now),
                corpus_manager=corpus_manager, user_profile=user_profile,
                telemetry_path=telemetry_path,
            ))
        if email_rows is not None:
            # The async caller pre-fetched live email headers (the engine is
            # sync and never fetches) — rows present = email cue in the theme.
            results.append(run_pattern_query(
                PatternQuery(dimension="email", window_days=window, now=now),
                corpus_manager=corpus_manager, user_profile=user_profile,
                telemetry_path=telemetry_path, email_rows=email_rows,
            ))
        if _MUSIC_THEME_RE.search(intent.theme):
            results.append(run_pattern_query(
                PatternQuery(dimension="content_type", terms=["lyrics"],
                             window_days=window, now=now),
                corpus_manager=corpus_manager, user_profile=user_profile,
                telemetry_path=telemetry_path,
            ))

    evidence: list[EvidenceItem] = []
    for res in results:
        for bucket in res.buckets:
            for ex in bucket.exemplars:
                evidence.append(EvidenceItem(
                    doc_id=ex.doc_id,
                    text=ex.text,
                    date=ex.date,
                    collection="pattern",
                    speaker=ex.speaker,
                    stance_label=(
                        "user-stated" if ex.speaker == "user"
                        else "assistant-inferred"
                    ),
                    is_appraisal=ex.is_appraisal,
                    facet=f"pattern:{res.dimension}",
                ))

    logger.info(
        f"[Insight] Pattern stage: {len(results)} queries, "
        f"{sum(r.total for r in results)} total hits, "
        f"{len(evidence)} exemplar evidence items"
    )
    return results, evidence
