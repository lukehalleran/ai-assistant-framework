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
from datetime import datetime
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

# Explicit "evidence sweep" cue (2026-09-04). None of the shapes above match
# "sweep" as a bare noun-phrase opener — only "sweep across/through my
# history ... for X". Live miss: "Evidence sweep request. From 2026-07-15
# through today, find every turn where I corrected something you said ..."
# matched nothing. Narrow and word-bounded (the 'sweep' substring inside
# "sweeper" must NOT match — same class as the 'solve'⊂"resolution" bug):
# either "evidence sweep"/"memory sweep" literally, or "sweep" followed
# within a short gap by an "our/my conversations/chats/history/memory"
# anchor. Still requires the SAME first-person/possessive anchor
# (gate._FIRST_PERSON_RE) detect_insight_statement already relies on, so a
# bystander mention ("the street sweeper came by") can't qualify.
_EVIDENCE_SWEEP_RE = re.compile(
    r"\b(?:evidence|memory)\s+sweep\b"
    r"|\bsweep\b[^.?!]{0,40}\b(?:our|my)\s+(?:conversations?|chats?|history|memory)\b",
    re.IGNORECASE,
)

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
    r"problems?|fears?|self[\s-]?talk|thinking|feelings?|episodes?|"
    r"procrastination|focus|attention|motivation|energy|pain|fatigue|sobriety|"
    r"cravings?|addictions?|grief|parenting|marriage|finances?|debts?|burnout|"
    r"confidence|self[\s-]?esteem|productivity|routines?)"
)

_PERSONAL_MARKER_RE = re.compile(
    rf"\b(?:my|our)\s+(?:\w+\s+){{0,3}}{_SELF_MODEL_NOUNS}\b"
    rf"|\bfor\s+my\s+(?:therapist|psychiatrist|psychologist|counselor|counsellor|doctor|psych|"
    rf"sponsor|coach|mentor|advisor|adviser|pastor|priest|rabbi|imam|lawyer|attorney|"
    rf"caseworker|social\s+worker|support\s+group)\b"
    rf"|\bwhat\s+i(?:'ve|\s+have)\s+(?:been\s+through|told\s+you)\b",
    re.IGNORECASE,
)

_PERSONAL_DOC_RE = re.compile(
    rf"\b{_SAVE_VERBS}\b\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:\w+\s+){{0,2}}\b{_DOC_NOUNS}\b"
    rf"\s*(?:about|on|of|covering|for)?\s*(?P<theme>.*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern-temporal shapes (2026-08-29) — "how often / how many times / is X
# getting worse / trend over time". Deterministic, UNDER-fires by design:
# ambiguous requests fall through to theme_sweep, which still works.
# ---------------------------------------------------------------------------

_PATTERN_TEMPORAL_PATTERNS = [
    # "how often have I / do I ... X"
    re.compile(
        r"\bhow\s+often\s+(?:have|do|did|am|was|is|are)\s+(?:i|my|we)\b\s*(?P<theme>.*)",
        re.IGNORECASE,
    ),
    # "how many times/days (have I) ... X"
    re.compile(
        r"\bhow\s+many\s+(?:times|days|nights|weeks|sessions)\b\s*(?P<theme>.*)",
        re.IGNORECASE,
    ),
    # "is/are my X getting worse/better/more/less (frequent)"
    re.compile(
        r"\b(?:is|are)\s+(?:my|the)\s+(?P<theme>.+?)\s+"
        r"(?:getting|becoming|growing)\s+(?:worse|better|more|less|stronger|weaker)\b",
        re.IGNORECASE,
    ),
    # "(show me / what's) the trend in/of/for MY X" — the personal anchor is
    # mandatory (audit F3: "What's the trend in AI regulation?" is a world
    # question, not a personal-record scan; UNDER-fires by design)
    re.compile(
        r"\btrends?\s+(?:in|of|for|with)\s+my\s+(?P<theme>.+)",
        re.IGNORECASE,
    ),
    # "my pattern with X over time / over the last N ..." (temporal cue REQUIRED
    # — bare "my pattern with X" stays a theme sweep)
    re.compile(
        r"\b(?:my|the)\s+patterns?\s+(?:with|of|in|around)\s+(?P<theme>.+?)\s+"
        r"(?:over\s+(?:time|the\s+(?:last|past)\s+\w+)|across\s+(?:the\s+)?(?:months|weeks|years))\b",
        re.IGNORECASE,
    ),
    # "track/chart/graph my X over ..."
    re.compile(
        r"\b(?:track|chart|graph|plot|count)\s+(?:my\s+)?(?P<theme>.+?)\s+"
        r"over\s+(?:time|the\s+(?:last|past)\s+\w+\s*\w*)\b",
        re.IGNORECASE,
    ),
]

# High-precision deliberation shapes whose subject cannot be extracted safely
# with a regex. These route the complete request to the generic planner, which
# freezes the actual outcomes, phases, claims, and tools before retrieval.
_EXPLICIT_PATTERN_TOOL_RE = re.compile(
    r"\b(?:use|run|invoke|with)\s+(?:the\s+)?pattern\s+"
    r"(?:tool|scan|engine|analysis)\b|\bpattern_scan\b",
    re.IGNORECASE,
)
_EXPLICIT_PATTERN_COMMAND_RE = re.compile(
    r"(?:^|[.!?\n]\s*)(?:please\s+)?(?:use|run|invoke)\s+"
    r"(?:the\s+)?pattern\s+(?:tool|scan|engine|analysis)\b|"
    r"(?:^|[.!?\n]\s*)pattern_scan\b",
    re.IGNORECASE,
)
_PERSONAL_RECORD_RE = re.compile(
    r"\b(?:my|our)\s+(?:history|record|data|notes?|timeline|messages?|"
    r"conversations?)\b|\beverything\s+i(?:'ve|\s+have)\s+(?:said|told\s+you|shared)\b|"
    r"\bacross\s+(?:my|our)\s+(?:history|notes?|record|conversations?)\b|"
    r"\b(?:my\s+theory|my\s+hypothesis|my\s+hunch).{0,80}\bthe\s+record\b|"
    # "re-run my Zelphex cessation analysis" — a possessive ANALYSIS request
    # is a personal-record cue even when every other possessive is dropped
    # (2026-08-31 16:11 turn: this head-anchored cue was the only personal
    # marker; the sole matching arm was notes→PubMed at char 225, which the
    # long-message incidental guard correctly suppressed, and the turn
    # misrouted to agentic knowledge mode). {0,3} intervening plain words;
    # possessive chains ("my friend's analysis") fail the \w-only word shape.
    r"\b(?:my|our)\s+(?:[\w-]+\s+){0,3}(?:analysis|analyses|comparison|assessment)\b",
    re.IGNORECASE,
)
_DELIBERATION_OPERATION_RE = re.compile(
    r"\b(?:compare|contrast|analy[sz]e|evaluate|assess|test|verify|weigh|"
    r"check|examine|investigate|determine|decide|correlat\w*|covar\w*|track|trend\w*|"
    r"changed?|changes?|different|theory|hypothesis|hunch|decision)\b|"
    r"\bbefore\s+(?:versus|vs\.?|and)\s+after\b|"
    r"\b(?:what|which)\s+tends?\s+to\s+happen\s+when\b",
    re.IGNORECASE,
)
_NON_INSIGHT_LOOKUP_RE = re.compile(
    r"^\s*(?:how\s+many\s+times\s+has\b|"
    r"is\s+my\s+\w+\s+before\b|"
    r"have\s+i\s+told\s+you\s+about\b)",
    re.IGNORECASE,
)
_IMPLICIT_PERSONAL_COMPARISON_RE = re.compile(
    r"\b(?:has|have|did|does|is|are|am)\s+(?:my|our|i|we)\b.{0,100}"
    r"\b(?:changed?|different|better|worse|more|less|since|after|before|"
    r"track(?:s|ed|ing)?|correlat\w*|covar\w*)\b|"
    r"\bcompare\b.{0,80}\b(?:i|me|my|we|our)\b.{0,100}"
    r"\b(?:before|after|between|since)\b|"
    r"\bwhat\s+tends?\s+to\s+happen\s+when\s+i\b",
    re.IGNORECASE | re.DOTALL,
)
_PERSONAL_PLUS_EXTERNAL_RE = re.compile(
    r"\b(?:history|record|notes?|data|everything\s+i(?:'ve|\s+have)\s+said)\b"
    r".{0,180}\b(?:pub\s*med|wikipedia|wiki|web|internet|research|stud(?:y|ies)|"
    r"literature|arxiv|wolfram)\b|"
    r"\b(?:pub\s*med|wikipedia|wiki|web|internet|research|stud(?:y|ies)|"
    r"literature|arxiv|wolfram)\b.{0,180}"
    r"\b(?:my\s+)?(?:history|record|notes?|data)\b",
    re.IGNORECASE | re.DOTALL,
)


def _detect_deliberation_shape(query: str) -> Optional[re.Match]:
    explicit = _EXPLICIT_PATTERN_COMMAND_RE.search(query)
    if explicit:
        return explicit
    implicit = _IMPLICIT_PERSONAL_COMPARISON_RE.search(query)
    if implicit:
        return implicit
    record = _PERSONAL_RECORD_RE.search(query)
    operation = _DELIBERATION_OPERATION_RE.search(query)
    if record and operation:
        return record if record.start() <= operation.start() else operation
    mixed = _PERSONAL_PLUS_EXTERNAL_RE.search(query)
    if mixed and _DELIBERATION_OPERATION_RE.search(query):
        return mixed
    return None

# Time-window parsing for pattern requests. Deterministic; default handled by
# the engine (PATTERN_DEFAULT_WINDOW_DAYS) when nothing matches.
_WINDOW_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_WINDOW_N_RE = re.compile(
    r"\b(?:last|past)\s+(\d+)\s+(day|week|month|year)s?\b", re.IGNORECASE)
_WINDOW_ONE_RE = re.compile(
    r"\b(?:last|past)\s+(day|week|month|year)\b", re.IGNORECASE)
_WINDOW_ALL_RE = re.compile(
    r"\b(?:all\s+time|ever|entire\s+history|over\s+the\s+years|since\s+we\s+started)\b",
    re.IGNORECASE,
)


def parse_window_days(text: str) -> int:
    """0 = no window named (engine applies its default); -1 = all history."""
    if not text:
        return 0
    if _WINDOW_ALL_RE.search(text):
        return -1
    m = _WINDOW_N_RE.search(text)
    if m:
        return int(m.group(1)) * _WINDOW_UNIT_DAYS[m.group(2).lower()]
    m = _WINDOW_ONE_RE.search(text)
    if m:
        return _WINDOW_UNIT_DAYS[m.group(1).lower()]
    if re.search(r"\bthis\s+year\b", text, re.IGNORECASE):
        return 365  # coarse YTD — the engine buckets by month at this size
    return 0


# Explicit ISO date windows for the theme-sweep evidence path (2026-09-04).
# Distinct from parse_window_days: that function returns a relative day
# COUNT for the pattern engine's rolling buckets; this returns an absolute
# (start, end) pair used to FILTER sweep evidence by content date. A live
# "from 2026-07-15 through today" request named a seven-week window that
# nothing enforced — the sweep returned everything and truncation collapsed
# it to the last few days. Deliberately narrow: only explicit ISO dates
# ("between 2026-07-15 and 2026-08-01", "from 2026-07-15 through today",
# "since 2026-07-15"); relative phrasing stays parse_window_days's job.
_ISO = r"\d{4}-\d{2}-\d{2}"
_DATE_WINDOW_BETWEEN_RE = re.compile(
    rf"\bbetween\s+({_ISO})\s+and\s+({_ISO})\b", re.IGNORECASE)
_DATE_WINDOW_RANGE_RE = re.compile(
    rf"\b(?:from\s+)?({_ISO})\s*(?:through|thru|to|until|-|–|—)\s*({_ISO}|today)\b",
    re.IGNORECASE)
_DATE_WINDOW_SINCE_RE = re.compile(rf"\bsince\s+({_ISO})\b", re.IGNORECASE)


def parse_date_window(text: str, *, now: Optional[datetime] = None) -> Optional[tuple[str, str]]:
    """Parse an explicit ISO date window from free text.

    Handles "between A and B", "from A through B", "from A through today",
    and "since A" (an open end defaults to today). Returns an inclusive
    ``(start_iso, end_iso)`` date-string pair, or None when no explicit ISO
    window is present.
    """
    if not text:
        return None
    today = (now or datetime.now()).date().isoformat()
    m = _DATE_WINDOW_BETWEEN_RE.search(text)
    if m:
        start, end = m.group(1), m.group(2)
    else:
        m = _DATE_WINDOW_RANGE_RE.search(text)
        if m:
            start = m.group(1)
            end = today if m.group(2).lower() == "today" else m.group(2)
        else:
            m = _DATE_WINDOW_SINCE_RE.search(text)
            if not m:
                return None
            start, end = m.group(1), today
    if start > end:
        start, end = end, start
    return (start, end)


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
    # Explicit ISO window ("from 2026-07-15 through today") for the
    # theme-sweep evidence path only; pattern_temporal keeps its own
    # relative-day window_days.
    _date_window = list(parse_date_window(query) or ())

    # Factual/calendar/memory lookups are not longitudinal self-analysis even
    # when they contain words such as "before" or "how many times".
    if _NON_INSIGHT_LOOKUP_RE.search(query):
        return None

    # The pattern-analysis kill switch governs every pattern_temporal arm
    # (audit F17 2026-08-31: it only gated the agentic pattern_scan tool —
    # the v1 primary surface ignored it). Assessment and theme-sweep are not
    # pattern analysis and stay live.
    try:
        # lazy import: live-config read (tests monkeypatch app_config attrs)
        from config.app_config import PATTERN_ANALYSIS_ENABLED as _pattern_enabled
    except ImportError:
        _pattern_enabled = True

    explicit_command = bool(_EXPLICIT_PATTERN_COMMAND_RE.search(query))

    # 1. Explicit assessment shapes — genuinely checked first (audit F10
    #    2026-08-31: the deliberation shape used to preempt these, routing
    #    "assess my theory against my history" away from the fail-honest
    #    adversarial assess() machinery). Deliberation still owns requests
    #    with deliberation-strength cues: an explicit pattern-tool command,
    #    mixed personal+external evidence ("use my notes and outside
    #    research to test my theory"), or an explicit time window ("am I
    #    right my anxiety is worse over the last month" is longitudinal).
    _deliberation_pull = _pattern_enabled and (
        explicit_command
        or bool(_PERSONAL_PLUS_EXTERNAL_RE.search(query))
        or parse_window_days(query) != 0
    )
    if not _deliberation_pull:
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

    # Generic longitudinal deliberation owns mixed internal/external requests.
    # The regex identifies only the operation shape; the LLM planner selects
    # the phenomenon and evidence contract from the complete request.
    deliberation_match = _detect_deliberation_shape(query) if _pattern_enabled else None
    if deliberation_match and (
        explicit_command or not _trigger_is_incidental(query, deliberation_match.start())
    ):
        return InsightIntent(kind="pattern_temporal", theme=_clean_theme(query),
                             wants_document=False, raw_query=query,
                             window_days=parse_window_days(query))

    # 1.5 Pattern-temporal shapes (before theme sweep: "my pattern with X
    #     over time" would otherwise match nothing and "how often" nothing).
    for pat in (_PATTERN_TEMPORAL_PATTERNS if _pattern_enabled else ()):
        m = pat.search(query)
        if m and not _trigger_is_incidental(query, m.start()):
            theme = _clean_theme(m.groupdict().get("theme") or "")
            if theme:
                return InsightIntent(
                    kind="pattern_temporal",
                    theme=theme,
                    wants_document=bool(
                        _PERSONAL_DOC_RE.search(query)
                        and _PERSONAL_MARKER_RE.search(query)
                    ),
                    raw_query=query,
                    window_days=parse_window_days(query),
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
                    date_window=_date_window,
                )

    # 2.5 Explicit "evidence sweep" cue — narrow addition (2026-09-04); see
    # _EVIDENCE_SWEEP_RE. Requires the same first-person/possessive anchor
    # detect_insight_statement already relies on (do not weaken it here).
    m = _EVIDENCE_SWEEP_RE.search(query)
    if m and not _trigger_is_incidental(query, m.start()):
        try:
            # lazy import: cycle (gate imports insight.detector at call time)
            from core.agentic.gate import _FIRST_PERSON_RE
        except ImportError:
            _FIRST_PERSON_RE = None
        if _FIRST_PERSON_RE is not None and _FIRST_PERSON_RE.search(query.lower()):
            return InsightIntent(
                kind="theme_sweep",
                theme=_clean_theme(query),
                wants_document=False,
                raw_query=query,
                date_window=_date_window,
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
                date_window=_date_window,
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
