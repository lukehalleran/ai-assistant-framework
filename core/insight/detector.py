"""
core/insight/detector.py

Module Contract
- Purpose: Deterministic (no-LLM) detection of insight-mode requests and
  insight-shaped statements. Called by the agentic gate BEFORE the doc-gen
  Tier-3 check — a personal-theme document request ("write a summary of my
  pattern with X for my therapist") must route HERE, not to web research
  (the observed generate_document misfire treated "insight" as a research
  topic).
- Inputs: raw query text.
- Outputs:
    detect_insight_request(text)   -> Optional[InsightIntent]
    detect_insight_statement(text) -> bool   (consent-offer eligibility only —
                                              never triggers the mode itself)
- Key behaviors:
  * Theme-sweep shapes ("gather everything I've said about X over time"),
    personal-theme doc shapes (doc-noun + save-verb + personal marker →
    wants_document=True), explicit assessment shapes ("check this against
    what I've told you", "am I right that ...").
  * Incidental-trigger discipline copied from document_generator: in a long
    message the trigger must sit at the head or tail; buried mid-body =
    incidental, no trigger.
  * detect_insight_statement requires a first-person self-model claim shape
    and defers to THE deployed gate helpers (_is_info_seeking,
    _is_request_shaped) — imported lazily to avoid a circular import with
    core.agentic.gate, which lazily imports this module. Epistemic markers
    ("I think", "I'm starting to realize") are NOT stripped here: for an
    insight statement the hedge IS the shape.
- Side effects: none (pure).
"""

from __future__ import annotations

import re
from typing import Optional

from core.insight.types import InsightIntent

# Long-message discipline (mirrors document_generator's incidental guard).
_SHORT_MSG_WORDS = 25
_EDGE_CHARS = 160

# ---------------------------------------------------------------------------
# Theme-sweep shapes
# ---------------------------------------------------------------------------

_GATHER_VERBS = r"(?:gather|pull(?:\s+up)?|collect|assemble|compile|find|dig\s+up|round\s+up)"
_SAID_FORMS = r"(?:i(?:'ve|\s+have)?\s+(?:ever\s+)?(?:said|told\s+you|mentioned|talked\s+about|shared)|we(?:'ve|\s+have)?\s+(?:discussed|talked\s+about))"

_THEME_SWEEP_PATTERNS = [
    # "gather/pull up everything I've said about X"
    re.compile(
        rf"\b{_GATHER_VERBS}\s+(?:together\s+)?everything\s+(?:{_SAID_FORMS}\s+)?"
        rf"(?:about|on|regarding)\s+(?P<theme>.+)",
        re.IGNORECASE,
    ),
    # "everything you know/have about my X (across sessions / over time)"
    re.compile(
        rf"\b{_GATHER_VERBS}\s+everything\s+you\s+(?:know|have|remember)\s+"
        rf"(?:about|on)\s+(?P<theme>.+)",
        re.IGNORECASE,
    ),
    # "what have I said/told you about X over time"
    re.compile(
        rf"\bwhat\s+(?:all\s+)?have\s+i\s+(?:said|told\s+you|mentioned)\s+"
        rf"(?:about|on)\s+(?P<theme>.+?)\s+(?:over\s+time|across\s+sessions|over\s+the\s+(?:months|years))\b",
        re.IGNORECASE,
    ),
    # "look/search across (all) (your|our) (sessions|history|memory) for X"
    re.compile(
        r"\b(?:look|search|sweep)\s+(?:across|through)\s+(?:all\s+)?(?:your|our|my)\s+"
        r"(?:sessions?|history|memory|memories|notes?|conversations?)\s+"
        r"(?:for|about|on)\s+(?P<theme>.+)",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Personal-theme document shapes (must beat detect_document_intent)
# ---------------------------------------------------------------------------

_SAVE_VERBS = r"(?:save|write|prepare|create|generate|make|produce|draft|put\s+together)"
_DOC_NOUNS = r"(?:report|document|doc|summary|write[\s-]?up|overview|timeline|pattern\s+document)"

# Self-model nouns: themes about the user's own life/mind/behavior.
_SELF_MODEL_NOUNS = (
    r"(?:patterns?|behaviors?|behaviours?|habits?|tendenc(?:y|ies)|relationships?|"
    r"histor(?:y|ies)|sleep|anxiety|depression|moods?|triggers?|drinking|spending|"
    r"coping|trauma|recovery|health|symptoms?|medications?|struggles?|issues?|"
    r"problems?|fears?|self[\s-]?talk|thinking|feelings?|episodes?)"
)

_PERSONAL_MARKER_RE = re.compile(
    rf"\b(?:my|our)\s+(?:\w+\s+){{0,3}}{_SELF_MODEL_NOUNS}\b"
    rf"|\bfor\s+my\s+(?:therapist|psychiatrist|psychologist|counselor|counsellor|doctor|psych)\b"
    rf"|\bwhat\s+i(?:'ve|\s+have)\s+(?:been\s+through|told\s+you)\b",
    re.IGNORECASE,
)

_PERSONAL_DOC_RE = re.compile(
    rf"\b{_SAVE_VERBS}\b\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:\w+\s+){{0,2}}\b{_DOC_NOUNS}\b"
    rf"\s*(?:about|on|of|covering|for)?\s*(?P<theme>.*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Explicit assessment shapes
# ---------------------------------------------------------------------------

_ASSESS_PATTERNS = [
    # "check this/that/it/my theory against what I've told you / you know / my history"
    re.compile(
        r"\bcheck\s+(?:this|that|it|my\s+\w+(?:\s+\w+)?)\s+against\s+"
        r"(?:what\s+(?:i(?:'ve|\s+have)\s+told\s+you|you\s+know)|my\s+history|"
        r"everything|your\s+memory|the\s+record)",
        re.IGNORECASE,
    ),
    # "does that/this match/square with what I've told you (before)?"
    re.compile(
        r"\bdoes\s+(?:that|this)\s+(?:match|square|line\s+up|fit)\s+with\s+"
        r"what\s+i(?:'ve|\s+have)\s+(?:told|said)",
        re.IGNORECASE,
    ),
    # "am I right that/about ..."
    re.compile(r"\bam\s+i\s+right\s+(?:that|about)\s+(?P<theme>.+)", re.IGNORECASE),
    # "assess/evaluate/verify this insight/realization/theory (against ...)"
    re.compile(
        r"\b(?:assess|evaluate|verify|test)\s+(?:this|that|my)\s+"
        r"(?:insight|realization|realisation|theory|hypothesis|hunch|read)\b",
        re.IGNORECASE,
    ),
    # "is that borne out by / supported by (what you know / my history / the record)"
    re.compile(
        r"\bis\s+(?:that|this)\s+(?:borne\s+out|supported|backed\s+up)\s+by\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Insight-shaped statements (consent-offer eligibility)
# ---------------------------------------------------------------------------

_INSIGHT_STATEMENT_RE = re.compile(
    r"\b(?:"
    r"my\s+real\s+(?:problem|issue)\s+is"
    r"|i(?:'m|\s+am)\s+(?:starting|beginning)\s+to\s+(?:realize|realise|see|think|wonder\s+if)"
    r"|i\s+(?:just\s+)?realized?"
    r"|i\s+think\s+(?:my|the)\s+(?:real\s+)?(?:problem|issue|pattern)\s+(?:is|with)"
    r"|maybe\s+(?:my|the)\s+(?:real\s+)?(?:problem|issue|pattern)\s+is"
    r"|i\s+keep\s+(?:doing|repeating|falling\s+into)"
    r"|it(?:'s|\s+is)\s+the\s+same\s+pattern"
    r")\b",
    re.IGNORECASE,
)

_SELF_MODEL_NOUN_RE = re.compile(rf"\b{_SELF_MODEL_NOUNS}\b|\bmyself\b|\bi\b", re.IGNORECASE)

# Suffixes trimmed off extracted themes.
_THEME_TRIM_RE = re.compile(
    r"\s*(?:over\s+time|across\s+sessions?|over\s+the\s+(?:months|years)|please|for\s+me"
    r"|for\s+my\s+(?:therapist|psychiatrist|psychologist|counselor|counsellor|doctor|psych))\s*$",
    re.IGNORECASE,
)


def _clean_theme(raw: str) -> str:
    theme = (raw or "").strip().strip(".!?,;: \"'")
    prev = None
    while prev != theme:
        prev = theme
        theme = _THEME_TRIM_RE.sub("", theme).strip().strip(".!?,;: \"'")
    return theme


def _trigger_is_incidental(query: str, match_start: int) -> bool:
    """Long-message discipline: in a long message a genuine trigger sits at
    the head or the tail; one buried mid-body is incidental (quoted/narrated)."""
    if len(query.split()) <= _SHORT_MSG_WORDS:
        return False
    tail_start = max(0, len(query) - _EDGE_CHARS)
    return not (match_start < _EDGE_CHARS or match_start >= tail_start)


def detect_insight_request(text: str) -> Optional[InsightIntent]:
    """Detect an explicit insight-mode request. Returns None when the message
    isn't one (including personal-doc shapes that are really ordinary doc-gen,
    e.g. 'write a report about the roman empire')."""
    if not text or not text.strip():
        return None
    query = text.strip()

    # 1. Explicit assessment shapes (checked first: "check this against what
    #    I've told you" may also contain gather-ish vocabulary).
    for pat in _ASSESS_PATTERNS:
        m = pat.search(query)
        if m and not _trigger_is_incidental(query, m.start()):
            theme = _clean_theme(m.groupdict().get("theme") or query)
            return InsightIntent(
                kind="insight_assessment",
                theme=theme or query,
                wants_document=False,
                raw_query=query,
            )

    # 2. Theme-sweep shapes.
    for pat in _THEME_SWEEP_PATTERNS:
        m = pat.search(query)
        if m and not _trigger_is_incidental(query, m.start()):
            theme = _clean_theme(m.group("theme"))
            if theme:
                return InsightIntent(
                    kind="theme_sweep",
                    theme=theme,
                    wants_document=False,
                    raw_query=query,
                )

    # 3. Personal-theme document shapes → theme_sweep + wants_document.
    m = _PERSONAL_DOC_RE.search(query)
    if m and not _trigger_is_incidental(query, m.start()):
        if _PERSONAL_MARKER_RE.search(query):
            theme = _clean_theme(m.group("theme")) or _clean_theme(query)
            return InsightIntent(
                kind="theme_sweep",
                theme=theme,
                wants_document=True,
                raw_query=query,
            )

    return None


def detect_insight_statement(text: str) -> bool:
    """
    True when the message is an insight-SHAPED first-person statement — a
    self-model claim the consent offer may (once, at non-elevated tone) offer
    to check against full history. Never triggers the mode by itself.
    """
    if not text or not text.strip():
        return False
    query = text.strip()

    if not _INSIGHT_STATEMENT_RE.search(query):
        return False
    if not _SELF_MODEL_NOUN_RE.search(query):
        return False

    # Defer to THE deployed gate shape helpers (lazy: gate lazily imports us).
    try:
        # lazy import: cycle (gate imports insight.detector at call time)
        from core.agentic.gate import _is_info_seeking, _is_request_shaped
        if _is_info_seeking(query) or _is_request_shaped(query):
            return False
    except ImportError:
        pass

    return True
