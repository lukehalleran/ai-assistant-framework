
"""
# utils/query_checker.py

Module Contract
- Purpose: Lightweight query analysis heuristics — deictic detection, meta-conversational
  routing, temporal window extraction, heavy topic classification, thread continuity scoring,
  and unified QueryAnalysis dataclass.
- Data class: QueryAnalysis(is_deictic, is_question, is_command, is_meta_conversational,
    is_heavy_topic, temporal_window_days, keywords, thread_keywords)
- Key public functions:
  - analyze_query(q, model_manager) -> QueryAnalysis  [sync, runs all heuristics]
  - analyze_query_async(q, model_manager) -> QueryAnalysis  [async, adds LLM heavy-topic check]
  - is_deictic(query) -> bool  [deictic hints: "explain", "that", "it", "this", etc.]
  - is_anaphoric_continuation(q) -> bool  [pronoun-anchored fragment or referent
    correction ("No I mean...") whose subject lives in the PREVIOUS exchange —
    drives topic inheritance + thread-shift suppression, 2026-07-28]
  - is_deictic_followup(q) -> bool  [alias]
  - is_question(q) -> bool  [starts with question word or ends with ?]
  - is_command(q) -> bool  [starts with /, "please", "do", etc.]
  - is_meta_conversational(q) -> bool  [memory/recall markers: "do you recall", "we talked about"]
  - extract_temporal_window(q) -> int  [days from temporal markers: "yesterday"=1, "last week"=7]
  - keyword_tokens(q, min_len) -> List[str]  [salient words for gating]
  - extract_rare_proper_nouns(q, max_terms) -> List[str]  [name-shaped tokens a bge
    embedding can't anchor on — feeds the keyword-anchor retrieval fallback
    (memory_retriever) and the obsidian keyword proper-noun floor, 2026-08-26;
    sentence-initial tokens excluded, days/months/common capitals stoplisted,
    adjacent names merge ("Jordan Vale"), UNDER-fires by design]
  - heavy_keyword_hits(text) -> List[str]  [word-bounded/substring HEAVY_KEYWORDS hits,
    2026-09-08 — see _HEAVY_MATCHER]
  - _is_heavy_topic_heuristic(q) -> bool  [keyword-based heavy topic detection]
  - _classify_heavy_topic_llm(q, model_manager) -> bool  [async LLM-based heavy topic check]
  - extract_thread_keywords(text) -> Set[str]  [salient keywords for thread matching]
  - has_thread_break_marker(query) -> bool  [explicit topic-change signals]
  - calculate_thread_continuity_score(query, recent_context, ...) -> Dict  [thread coherence scoring]
  - belongs_to_thread(query, thread_queries, ...) -> bool  [thread membership test]
- Dependencies:
  - models.model_manager (LLM for async heavy topic classification) [optional]
- Side effects:
  - LLM API call only in analyze_query_async for heavy topic edge cases
"""

import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Set
from utils.logging_utils import get_logger
from utils.trigger_match import is_negated as _trigger_is_negated, compile_keyword_matcher, prefix_only_hits
from memory.fact_source import strip_quoted_correspondence
import utils.temporal_resolver as temporal_resolver
import re
from datetime import datetime

logger = get_logger("query_checker")

# Deliberate near-duplicate of web_search_trigger._RETRY_CUES.  That module
# imports this one, so importing its cues here would create a cycle.
_RETRY_CONTINUATION_PHRASES = (
    "try again", "try that again", "try it now", "one more time",
    "should work now", "restarted you",
)
_RETRY_WORD_RE = re.compile(r"\bretry\b|\bfixed it\b", re.IGNORECASE)


def is_retry_continuation(query: str, max_words: int = 25) -> bool:
    """Return whether a short query explicitly retries the prior operation."""
    text = _normalize(query)
    if len(text.split()) > max_words:
        return False
    return any(phrase in text for phrase in _RETRY_CONTINUATION_PHRASES) or bool(
        _RETRY_WORD_RE.search(text)
    )


DEICTIC_HINTS: tuple[str, ...] = (
    "explain", "that", "it", "this", "again", "another way",
    "different way", "more", "elaborate", "clarify", "what about",
    "those", "these", "there", "former", "latter"
)

QUESTION_LEADS: tuple[str, ...] = (
    "what", "who", "when", "where", "why", "how", "which"
)

COMMAND_SIGNS: tuple[str, ...] = (
    "/", "please ", "do ", "tell me to ", "create ", "generate ", "write ", "summarize ",
)

META_CONVERSATIONAL_MARKERS: tuple[str, ...] = (
    "do you recall", "do you remember", "did you", "did we",
    "we discussed", "we talked about", "last time", "the other day",
    "you said", "you mentioned", "you told me",
    "earlier you", "before you", "didn't we", "haven't we",
    "recall the", "remember when", "remember the",
    # Queries about the conversation/responses themselves
    "that response", "that off topic response", "your response", "your answer",
    "that message", "earlier message", "previous response", "last response",
    "what causes that", "why did you", "geared to", "seems like it was"
)

# Temporal markers for detecting time-based memory queries
TEMPORAL_MARKERS = {
    # Yesterday/recent
    "yesterday": 1,
    "last night": 1,
    "this morning": 1,
    "earlier today": 1,

    # Days ago
    "days ago": 3,
    "few days ago": 3,
    "couple days ago": 2,
    "other day": 2,

    # Last week
    "last week": 7,
    "week ago": 7,
    "few weeks ago": 14,
    "couple weeks ago": 14,

    # Longer periods
    "last month": 30,
    "month ago": 30,
    "while back": 14,
    "while ago": 14,
    "long time ago": 30,

    # Specific days
    "monday": 7,
    "tuesday": 7,
    "wednesday": 7,
    "thursday": 7,
    "friday": 7,
    "saturday": 7,
    "sunday": 7,
}


def _normalize(q: str) -> str:
    return (q or "").strip().lower()


# Openers that mark a terse acknowledgment / status remark ("ok", "hmm not
# working yet", "yeah makes sense") — turns that need conversational
# continuity, not the full retrieval apparatus.
ACK_STARTERS: frozenset = frozenset({
    "ok", "okay", "kk", "k", "cool", "nice", "sweet", "great", "awesome",
    "yeah", "yep", "yup", "ya", "sure", "right", "true", "word", "bet",
    "thanks", "thank", "thx", "ty", "lol", "lmao", "haha", "heh",
    "hmm", "hm", "hmmm", "huh", "ugh", "oof", "welp", "whew", "phew",
    "alright", "gotcha", "got", "makes", "sounds", "fair", "interesting",
    "damn", "dang", "sheesh", "wow", "oh", "ah", "aw", "meh", "nope", "nah",
})

# Session-opening greetings. Deliberately NOT in ACK_STARTERS: a greeting
# must keep the FULL builder path (so [UNRESOLVED THREADS] surface at session
# start), but it never inherits the previous turn's topic and never suppresses
# a thread-shift assertion (2026-09-03: "Hey" 13h after the last session was
# treated as a fragment continuation and rendered "message #3 in an ongoing
# conversation thread about <yesterday's topic>").
GREETING_STARTERS: frozenset = frozenset({
    "hey", "heyy", "hi", "hii", "hello", "yo", "sup", "hiya", "heya",
    "howdy", "greetings", "morning", "evening", "afternoon",
})
_GREETING_GOOD_SUFFIXES: frozenset = frozenset({
    "morning", "afternoon", "evening", "night",
})


def is_greeting_opener(q: str) -> bool:
    """True when the message OPENS with a greeting ("Hey", "hi there",
    "Good morning!"). Only the opener is inspected — "hey can you look up X"
    is still a greeting opener for topic-inheritance purposes (a greeting
    never continues the previous turn's fragment), and that is all this
    predicate decides. Missing/garbage input → False.
    """
    ql = _normalize(q)
    if not ql:
        return False
    words = ql.split()
    first = words[0].strip(".,!…:;'\"")
    if first in GREETING_STARTERS:
        return True
    if first == "good" and len(words) >= 2:
        second = words[1].strip(".,!…:;'\"")
        return second in _GREETING_GOOD_SUFFIXES
    return False


# Words that signal an actual information request even without a "?"
_REQUEST_MARKERS: tuple = (
    "can you", "could you", "would you", "will you", "do you", "should i",
    "should we", "help me", "show me", "give me", "explain",
)


# Task navigation (2026-09-08, live shape R8: "ok next q please") — moving
# between numbered items in an attached worksheet/document. Word-bounded so
# 'q' can't match inside another word; negation-guarded so "don't show the
# next question" doesn't route as navigation.
_TASK_NAV_PATTERNS: tuple = (
    re.compile(
        r"\b(?:next|first|previous|prev|last|following|second|third)\s+"
        r"(?:q|question|task|part|problem|step|section|page|one)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bq\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:question|task|part|problem)\s+\d+\b", re.IGNORECASE),
)


def is_task_navigation(text: str) -> bool:
    """True when the message asks to move between numbered items in an
    attached document/worksheet ("next question", "Q3", "part 2"). Such a
    message can open with an ack word ("ok next q please") without being a
    casual acknowledgment — it needs the active document/passage, not the
    light path. Negation-guarded: "don't show the next question" is False."""
    q = (text or "").strip()
    if not q:
        return False
    ql = q.lower()
    for pat in _TASK_NAV_PATTERNS:
        for m in pat.finditer(ql):
            if not _trigger_is_negated(ql, m.start()):
                return True
    return False


def is_casual_acknowledgment(q: str, max_words: int = 8) -> bool:
    """
    True for terse acknowledgment/status turns that don't need heavy
    retrieval ("ok", "hmm not working yet", "yeah makes sense", "thanks").

    Conservative by design — any question/command/request shape disqualifies,
    because a false positive means answering a real query without context.
    Feeds QueryAnalysis.is_small_talk, which routes the builder to
    _build_lightweight_context (recent turns only, no memories/wiki/git/web).
    That routing was DEAD until 2026-07-15: the builder read
    query_analysis.is_small_talk but no code ever set it, so a 7-word "Hmm
    not working yet" pulled a 23K-token full-apparatus prompt.
    """
    ql = _normalize(q)
    if not ql:
        return False
    words = ql.split()
    if len(words) > max_words:
        return False
    if "?" in ql or is_question(ql) or is_command(ql) or is_meta_conversational(ql):
        return False
    # Interrogative anywhere ("ok how do i fix this") = real question
    if any(w in QUESTION_LEADS for w in words):
        return False
    if any(m in ql for m in _REQUEST_MARKERS):
        return False
    # Task navigation ("ok next q please") is a request for the next item in
    # an attached document, not an acknowledgment, even though it can open
    # with an ack word (2026-09-08, live shape R8). A bare "please" anywhere
    # is the same signal in weaker form.
    if is_task_navigation(ql) or re.search(r"\bplease\b", ql):
        return False
    first = words[0].strip(".,!…:;'\"")
    if first not in ACK_STARTERS:
        return False
    # Ack-prefixed IMPERATIVES are requests, not acknowledgments (2026-08-22):
    # "Alright, check it out now" opened with an ack starter, passed the
    # command checks (which look at the raw opener), rode the light path to a
    # 1K-token context, and the gate had nothing to work with. The deployed
    # request-shape test tolerates leading discourse markers.
    try:
        from core.agentic.gate import _is_request_shaped  # lazy import: cycle
        if _is_request_shaped(q):
            return False
    except Exception:
        pass
    return True


def is_continuation_answer(q: str, last_assistant_response: str, max_words: int = 6) -> bool:
    """
    True when the current message is a SHORT reply directly answering the
    assistant's own immediately-preceding question — e.g. assistant asks
    "What was the error?" and the user replies "amplification".

    Such a turn must be interpreted from the immediate exchange, not free-
    associated across the whole corpus: a bare noun like "amplification" otherwise
    triggers corpus-wide semantic retrieval that surfaces a topically-matching but
    contextually-wrong old memory, which then hijacks the response (observed
    2026-07-21). Routes to the lightweight-context path (recent turns only).

    Deliberately narrow — requires the prior assistant turn to end in a question,
    a short reply, and NOT itself be a question/command/meta turn (a counter-
    question like "why?" is excluded). Heavy/crisis topics are filtered by the
    caller, not here.
    """
    ql = _normalize(q)
    lr = (last_assistant_response or "").strip()
    if not ql or not lr:
        return False
    # The assistant must have asked something at the END of its last turn.
    # Tolerate a trailing sentence after the question ("...what was the
    # error? You mentioned the last exchange was bad.") but not a "?"
    # buried anywhere (audit F29 2026-08-31: nearly every Daemon reply
    # contains one somewhere, so terse NEW statements were misframed as
    # answers to questions the user never saw as pending).
    if "?" not in lr[-240:]:
        return False
    words = ql.split()
    if len(words) > max_words:
        return False
    # A short reply that is itself a question/command/meta is not an answer.
    if "?" in ql or is_question(ql) or is_command(ql) or is_meta_conversational(ql):
        return False
    if any(w in QUESTION_LEADS for w in words):
        return False
    # A request ("show me the diff", "can you...") is an action, not a bare answer.
    if any(m in ql for m in _REQUEST_MARKERS):
        return False
    return True


# Markers that the user is repairing the assistant's reading of their OWN
# previous message ("No I mean of being sick I wasn't talking about working
# out") — the referent lives in the prior exchange, not in this message.
_REFERENT_CORRECTION_MARKERS: tuple = (
    "i mean", "i meant", "i didn't mean", "i didnt mean", "i don't mean",
    "i dont mean", "i wasn't talking about", "i wasnt talking about",
    "i was talking about", "not talking about", "that's not what i",
    "thats not what i", "i was referring to", "i'm referring to",
    "im referring to", "i was asking about", "not what i said",
)

# Bare referential openers whose subject is in the previous exchange.
_ANAPHORIC_OPENERS: frozenset = frozenset({
    "it", "that", "this", "they", "those", "these",
})


def is_anaphoric_continuation(q: str, max_words: int = 30) -> bool:
    """
    True when the message's SUBJECT lives in the previous exchange: it opens
    with a bare referential pronoun ("It was maybe 3 years of...") or repairs
    the assistant's reading of the user's own prior message ("No I mean of
    being sick I wasn't talking about working out").

    Fresh topic classification of such a message is unreliable by
    construction — the classifier sees only surface keywords and the referent
    isn't in the text. 2026-07-28 incident: mid long-covid conversation,
    "It was maybe 3 years of twice a week then the year and a half of like
    5 6 days a week" (illness frequency) was classified topic "Exercise
    Routine"; the [THREAD CONTEXT] injection then asserted a topic shift
    ("Follow the current query") and the response plan doubled down, so the
    model answered about the gym. The explicit correction on the next turn
    was classified as ANOTHER new topic ("Being Sick") and misread again.

    Consumers: ContextPipeline._extract_topics (inherit the previous turn's
    topic instead of fresh-classifying), orchestrator._thread_topic_shifted
    (never assert a topic shift on such a turn).

    Deliberately does NOT try to exclude expletive-"it" openers ("it's
    raining") — inheriting the prior topic there is a soft, low-cost hint,
    while a false topic-shift assertion on a real continuation was the
    demonstrated failure.
    """
    ql = _normalize(q)
    if not ql:
        return False
    words = ql.split()
    if len(words) > max_words:
        return False
    if any(m in ql for m in _REFERENT_CORRECTION_MARKERS):
        return True
    first = words[0].strip(".,!…:;'\"")
    return first in _ANAPHORIC_OPENERS


def is_fragment_continuation(q: str, max_words: int = 4) -> bool:
    """True for a bare noun-phrase FRAGMENT mid-conversation ("Tactical
    Taylors", "the other one") — no question shape, no request shape, not a
    casual ack, just a few words riffing on the ongoing exchange.

    2026-08-22 incident: mid Taylor-Greene/Taylor-Swift joke thread, the
    two-word riff "Tactical Taylors" was fresh-classified topic "Tactical
    Gear", STM INVENTED "User Question: What are Tactical Taylors?" (the
    user asked nothing), a new thread spawned, and the reply led with a
    MOLLE-gear brand encyclopedia answer. Same failure class as the 07-28
    pronoun-fragment incident, without the pronoun.

    Consumers treat it exactly like is_anaphoric_continuation: inherit the
    previous turn's topic, never assert a thread shift. Soft, low-cost
    inheritance — a genuinely fresh 3-word topic opener pays one turn of a
    stale topic label; a fresh-classified riff derails the whole reply.
    """
    ql = _normalize(q)
    if not ql:
        return False
    words = ql.split()
    if len(words) > max_words:
        return False
    if "?" in ql or is_question(ql) or is_command(ql):
        return False
    if any(w in QUESTION_LEADS for w in words):
        return False
    first = words[0].strip(".,!…:;'\"")
    if first in ACK_STARTERS:
        return False  # casual acks have their own (light-path) routing
    if is_greeting_opener(ql):
        return False  # greetings never inherit topic / suppress shift (2026-09-03)
    try:
        from core.agentic.gate import _is_request_shaped  # lazy import: cycle
        if _is_request_shaped(q):
            return False  # imperatives route to tools, not topic inheritance
    except Exception:
        pass
    # at least one substantive word (not all ack-type fillers)
    return not all(w.strip(".,!…:;'\"") in ACK_STARTERS for w in words)


def is_deictic(query: str) -> bool:
    """True if the query likely refers to earlier context (anaphora)."""
    if not query:
        return False
    ql = _normalize(query)

    # Short follow-ups with hints are often deictic (raised threshold from 6 to 10)
    if len(ql.split()) <= 10 and any(h in ql for h in DEICTIC_HINTS):
        return True

    # Pronouns/markers at beginning suggest reference
    if ql.startswith(("that", "this", "it", "they", "those", "these", "so", "and", "then")):
        return True

    # "watched it", "saw it", "read it" etc. are deictic follow-ups
    if re.search(r'\b(watched|saw|read|heard|looked at|checked|finished|started)\s+(it|that|this)\b', ql):
        return True

    return False


def is_deictic_followup(q: str) -> bool:
    """Softer check for follow-up phrasing used by wiki gating."""
    ql = _normalize(q)
    return any(h in ql for h in DEICTIC_HINTS)


def is_question(q: str) -> bool:
    ql = _normalize(q)
    return ql.endswith("?") or ql.startswith(QUESTION_LEADS)


def is_command(q: str) -> bool:
    ql = _normalize(q)
    return ql.startswith(COMMAND_SIGNS)


def is_meta_conversational(q: str) -> bool:
    """True if the query is asking about the conversation history itself."""
    if not q:
        return False
    ql = _normalize(q)
    return any(marker in ql for marker in META_CONVERSATIONAL_MARKERS)


# Closed grammatical sets for the self-report shape (2026-09-06). Pronouns
# and auxiliaries are grammar, not topic vocabulary — this predicate must hold
# for any American-English user's status update ("I took it at 10 and I'm
# resting", "we finally moved the couch", "im so tired today").
_FIRST_PERSON_OPENERS: frozenset = frozenset({
    "i", "i'm", "im", "i've", "ive", "i'd", "id", "i'll", "ill",
    "we", "we're", "we've", "we'd", "we'll", "my", "our", "me",
})
_FIRST_PERSON_VERB_RE = re.compile(
    r"\b(?:i|we)(?:'m|'ve|'d|'re|'ll"
    r"|\s+(?:am|are|was|were|have|had|been|just|already|finally|still|currently|kinda|kind\s+of|sorta|sort\s+of))?"
    r"\s+[a-z]+(?:ed|ing|t)?\b",
    re.IGNORECASE,
)
_ADDRESSING_ASSISTANT_RE = re.compile(
    r"\b(?:can|could|would|will|should)\s+you\b"
    r"|\b(?:want|need|like|'d\s+like)\s+you\s+to\b"
    r"|\byou\s+(?:should|need\s+to|could|can)\b"
    r"|\bfor\s+me\b\s*[.!]?$",
    re.IGNORECASE,
)
_SELF_REPORT_ACK_PREFIX_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|so|and|well|yeah|yep|also|now|then|honestly|anyway|update)[,:\s]+){0,3}",
    re.IGNORECASE,
)


# Generic imperative request openers — the gate's own _REQUEST_SHAPED_RE is
# deliberately limited to RETRIEVAL verbs (it routes tools); a decision or
# analysis request ("give me a breakdown", "weigh both sides", "walk me
# through it") is request-shaped without naming a tool. Grammar, not topic.
_IMPERATIVE_REQUEST_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|so|and|now|then|also|well|hey|yeah|please)[,\s]+){0,3}"
    r"(?:please\s+)?"
    r"(?:give|tell|write|make|draft|explain|help|analy[sz]e|assess|evaluate|weigh|compare|"
    r"outline|describe|walk|break|lay|put|plan|suggest|recommend|figure|work|think|go|talk|"
    r"remind|create|generate|build|calculate|estimate|rank|sort|prioriti[sz]e|translate|rewrite|"
    r"proofread|edit|fix|debug|design|map|track|log|note|save)\b",
    re.IGNORECASE,
)


def is_request_shaped(q: str) -> bool:
    """Public request-shape predicate: the agentic gate's retrieval-imperative
    / "can you …" shapes and info-seeking cues, plus a generic imperative
    opener ("give me…", "weigh both sides") and an explicit address to the
    assistant ("I want you to…"). Call-time import: gate imports this module
    lazily in function bodies, so a module-level import here would cycle."""
    text = (q or "").strip()
    if not text:
        return False
    from core.agentic.gate import _is_info_seeking, _is_request_shaped  # lazy import: cycle
    if _is_request_shaped(text) or _is_info_seeking(text):
        return True
    ql = _normalize(text)
    if _ADDRESSING_ASSISTANT_RE.search(ql):
        return True
    return bool(_IMPERATIVE_REQUEST_RE.match(ql))


# Fenced code blocks (```...```, DOTALL so a multi-line block collapses to one
# token). Applied before the line-level code-shape strip below.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

# Line-shaped prefixes that mark pasted code/console output rather than the
# user's own prose: '#' comments, '>' blockquote/paste markers, '$ ' shell
# prompts, '>>>' REPL prompts. Checked against the STRIPPED line.
_CODE_LINE_PREFIXES = ("#", ">", "$ ", ">>>")


def strip_code_shaped_lines(text: str) -> str:
    """Remove fenced code blocks and lines that read as pasted code/console
    output (comments, blockquote/shell/REPL prompts, `<-` assignment lines —
    the R assignment operator) from `text`. Pure; blank lines are preserved
    so downstream clause/paragraph splitting is unaffected.

    2026-09-08: a pasted R homework comment ("#read csv data file into data
    frame") and a `used_car_data <- read.csv(...)` assignment line both read
    as file-retrieval REQUESTS to the raw keyword/pattern matchers even
    though they are the user's own script, not something addressed to
    Daemon — this strip runs ahead of those matchers.
    """
    raw = text or ""
    raw = _CODE_FENCE_RE.sub(" ", raw)
    kept: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith(_CODE_LINE_PREFIXES):
            continue
        if "<-" in stripped:
            continue
        kept.append(line)
    return "\n".join(kept)


_CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+")


def request_clauses(text: str) -> List[str]:
    """Split `text` into non-empty clauses after removing pasted
    correspondence (memory.fact_source.strip_quoted_correspondence) and
    code-shaped lines (strip_code_shaped_lines). is_self_report checks EVERY
    clause for a request shape, not just the message's overall head/shape —
    "wait. lol i have it... please show me the first question..." is not a
    self-report just because its FIRST clause is an aside; a later clause
    still asks for something (2026-09-08)."""
    cleaned = strip_quoted_correspondence(text or "")
    cleaned = strip_code_shaped_lines(cleaned)
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(cleaned) if c.strip()]


def is_self_report(q: str, max_words: int = 40) -> bool:
    """A short first-person statement about the user's own state or action —
    "I took my stimulant at 10 AM today and I'm just resting" — that requests
    nothing. Under-fires by design: any question, command, meta remark,
    request shape, or address to the assistant disqualifies; a pasted block
    (≥3 lines or over ``max_words``) disqualifies. Consumers: the retrieval
    budget trim and the decision-support gate (2026-09-06)."""
    raw = (q or "").strip()
    if not raw:
        return False
    # Paste guard — but soft-wrapped chat text ("...I'm just\n  resting this
    # afternoon...", the live 2026-09-06 shape pasted from a code block) is
    # ONE message: a paste is blank-line paragraphs, or ≥3 lines of which ≥2
    # end in terminal punctuation (greeting/closing/signature lines).
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    paragraphs = [blk for blk in re.split(r"\n\s*\n", raw) if blk.strip()]
    if len(paragraphs) >= 2:
        return False
    if len(lines) >= 3 and sum(1 for ln in lines if ln[-1] in ".!?:,;") >= 2:
        return False
    ql = _normalize(" ".join(raw.split()))
    words = ql.split()
    if not words or len(words) > max_words:
        return False
    if "?" in ql or is_question(ql) or is_command(ql) or is_meta_conversational(ql):
        return False
    if _ADDRESSING_ASSISTANT_RE.search(ql) or is_request_shaped(ql):
        return False
    # Clause-level request detection (2026-09-08, live shape R2): a request
    # can arrive after an opening aside ("wait. lol i have it... please show
    # me the first question...") — the whole-message checks above only look
    # at the overall shape/head, so a request buried in a LATER clause slid
    # through as a "self-report".
    for clause in request_clauses(raw):
        cl = clause.lower()
        if (is_request_shaped(clause)
                or _ADDRESSING_ASSISTANT_RE.search(cl)
                or _IMPERATIVE_REQUEST_RE.match(cl)
                or cl.startswith("please")):
            return False
    stripped = _SELF_REPORT_ACK_PREFIX_RE.sub("", ql, count=1).strip()
    if not stripped:
        return False
    head = stripped.split()
    if head[0].strip(".,!") in _FIRST_PERSON_OPENERS:
        return True
    window = " ".join(head[:12])
    return bool(_FIRST_PERSON_VERB_RE.search(window))


# Completion-shaped verbs a status update opens with once its ACK opener is
# stripped ("Cool. Managed to push today...", "Nice, finally sent it").
# Deliberately a small closed set of PAST-COMPLETION verbs, not a general
# first-person-verb scan (that's is_self_report's job for an explicit
# pronoun subject) — the subject here is elided ("[I] managed to..."), which
# is exactly the shape is_self_report's contract does not cover.
_STATUS_REPORT_VERB_RE = re.compile(
    r"^(?:managed|got|finished|wrapped(?:\s+up)?|pushed|sent|submitted|"
    r"completed|updated|posted|uploaded|committed|shipped|fixed|solved|"
    r"finally\s+(?:got|finished|sent|managed|pushed|wrapped(?:\s+up)?)|"
    r"just\s+(?:finished|sent|pushed|submitted|wrapped(?:\s+up)?))\b",
    re.IGNORECASE,
)


def is_status_report(q: str, max_words: int = 40) -> bool:
    """A terse conversational ACK (``ACK_STARTERS`` — "Cool.", "Nice,")
    directly followed by an implied-first-person, completion-shaped status
    update — "Cool. Managed to push today and there is a new doc I think
    will be helpful" — that requests nothing (2026-09-10 probe T5, round 2:
    ``is_self_report`` returns False on this exact text because its subject
    is elided after the ack rather than restated as a pronoun, so the
    planner's self-report skip in ``ResponsePlanner.should_plan`` never
    fired; it confidently planned three points about "a new doctor" from a
    misread STM abbreviation expansion). Deliberately kept as a SEPARATE,
    narrow predicate rather than widening ``is_self_report``'s contract —
    that predicate's other consumers (the retrieval-trim gate, the
    decision-support gate) are calibrated to its first-person-pronoun
    shape specifically. Same disqualifiers as ``is_self_report``: paste
    guard, any question/command/meta shape, a request/address-to-assistant
    shape anywhere in the message (including a later clause). Under-fires
    by design — only the small ``_STATUS_REPORT_VERB_RE`` verb set counts.
    """
    raw = (q or "").strip()
    if not raw:
        return False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    paragraphs = [blk for blk in re.split(r"\n\s*\n", raw) if blk.strip()]
    if len(paragraphs) >= 2:
        return False
    if len(lines) >= 3 and sum(1 for ln in lines if ln[-1] in ".!?:,;") >= 2:
        return False
    ql = _normalize(" ".join(raw.split()))
    words = ql.split()
    if not words or len(words) > max_words:
        return False
    if "?" in ql or is_question(ql) or is_command(ql) or is_meta_conversational(ql):
        return False
    if _ADDRESSING_ASSISTANT_RE.search(ql) or is_request_shaped(ql):
        return False
    for clause in request_clauses(raw):
        cl = clause.lower()
        if (is_request_shaped(clause)
                or _ADDRESSING_ASSISTANT_RE.search(cl)
                or _IMPERATIVE_REQUEST_RE.match(cl)
                or cl.startswith("please")):
            return False
    first_word = words[0].strip(".,!?:;")
    if first_word not in ACK_STARTERS:
        return False
    rest = ql[len(words[0]):].lstrip(" .,!?:;")
    if not rest:
        return False
    return bool(_STATUS_REPORT_VERB_RE.match(rest))


def extract_temporal_window(q: str) -> int:
    """
    Extract the temporal window (in days) from a query based on time markers.

    This analyzes queries for temporal references like "yesterday", "last week",
    "few days ago" and returns an appropriate retrieval window in days.

    Args:
        q: Query text

    Returns:
        Number of days to look back. Returns 0 if no temporal markers found.
        Examples:
            "yesterday" -> 1
            "few days ago" -> 3
            "last week" -> 7
            "last month" -> 30
            "no temporal marker" -> 0
    """
    if not q:
        return 0

    ql = _normalize(q)

    # Check for temporal markers and find the largest window
    max_days = 0
    for marker, days in TEMPORAL_MARKERS.items():
        if marker in ql:
            max_days = max(max_days, days)

    # Also check for explicit date references (e.g., "Nov 1st", "November 1")

    # Pattern: Month name/abbreviation + day number (with optional suffixes like "st", "nd", "rd", "th")
    date_pattern = r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s*\d{1,2}(?:st|nd|rd|th)?\b'
    if re.search(date_pattern, ql):
        # If explicit date mentioned, assume up to 30 days back
        max_days = max(max_days, 30)

    # Pattern: "N days ago" where N is a number
    num_days_pattern = r'(\d+)\s+days?\s+ago'
    match = re.search(num_days_pattern, ql)
    if match:
        try:
            num_days = int(match.group(1))
            max_days = max(max_days, num_days)
        except ValueError:
            pass

    return max_days


def keyword_tokens(q: str, min_len: int = 3) -> List[str]:
    ql = _normalize(q)
    return [t for t in ql.split() if len(t) >= min_len]


# --- Rare-proper-noun extraction (2026-08-26) -------------------------------
# In bge space a rare name contributes almost nothing to a query embedding —
# "get appointment scheduled with Morgan for Friday" retrieved appointment-vibe
# docs, none containing Morgan, and the live memory path had no keyword channel
# (the corpus keyword scan existed only inside insight mode). This detector
# feeds the keyword-anchor fallback in memory_retriever.get_memories and the
# proper-noun floor in obsidian_manager._keyword_search.

_PROPER_NOUN_STOPWORDS = frozenset(w.lower() for w in (
    # days / months (the Morgan query itself contained "Friday")
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Today", "Tomorrow", "Yesterday",
    # conversational capitals / interjections
    "I", "Idk", "Ok", "Okay", "Lol", "Lmao", "Omg", "Btw", "Imo", "Tbh",
    "God", "Jesus", "Christ",
    # self-reference — anchoring "Daemon" would pull half the corpus
    "Daemon",
    # common capitalized words that aren't discriminative alone
    "American", "America", "English", "Internet", "Google", "YouTube", "Reddit",
))


def extract_rare_proper_nouns(q: str, max_terms: int = 3) -> List[str]:
    """Capitalized name-shaped tokens a semantic embedding can't anchor on.

    Rules (conservative — deliberately UNDER-fires, since a miss costs
    nothing while a false positive injects wrong-topic memories):
      - TitleCase / interior-capital token ("Morgan", "Ashdown"), len >= 3,
        not ALL-CAPS (emphasis: "SO much evidence")
      - possessives stripped ("Morgan's" -> "Morgan")
      - sentence-initial tokens excluded (no dictionary distinguishes
        "Morgan" from "Please" there) unless the same token also appears
        capitalized mid-sentence in the message
      - days/months/common capitalized words stoplisted
    Returns at most ``max_terms`` terms in order of first appearance.
    """
    if not q or not q.strip():
        return []

    token_re = re.compile(r"^[A-Z][a-zA-Z'’-]{2,}$")
    # "I'm" / "I'll" / "I've" / "I'd" pass the TitleCase shape (capital I +
    # apostrophe + letters) but are the pronoun, not a name — the STM novelty
    # override rendered "Note: the current message names I'm, which do not
    # appear in the short-term window" on a live turn (2026-09-05).
    i_contraction_re = re.compile(r"^I['’](?:m|ll|ve|d)$")
    # Walk word tokens plus sentence-boundary punctuation so we know which
    # tokens sit in a position where capitalization is expected anyway.
    pieces = re.findall(r"[A-Za-z'’-]+|[.!?\n]", q)
    # A period after a title abbreviation is not a sentence boundary —
    # otherwise "Dr. Goldsman" reads as sentence-initial and is dropped.
    _title_abbrevs = {"dr", "mr", "mrs", "ms", "prof", "st"}

    found: List[str] = []
    seen_lower: set = set()
    sentence_initial = True
    prev_word: str = ""
    prev_accepted = False  # previous piece was an accepted proper-noun token

    for tok in pieces:
        if tok in (".", "!", "?", "\n"):
            if not (tok == "." and prev_word.lower() in _title_abbrevs):
                sentence_initial = True
            prev_accepted = False
            continue
        is_initial = sentence_initial
        sentence_initial = False
        prev_word, was_prev_accepted = tok, prev_accepted
        prev_accepted = False

        surface = tok.removesuffix("'s").removesuffix("’s").rstrip("'’-")
        if not token_re.match(surface):
            continue
        if surface.isupper() or i_contraction_re.match(surface):
            continue
        low = surface.lower()
        if low in _PROPER_NOUN_STOPWORDS:
            continue
        # Adjacent proper nouns form one phrase ("Jordan Vale") — a single
        # word-boundary phrase scan beats two scans hitting the same docs.
        if was_prev_accepted and found:
            found[-1] = f"{found[-1]} {surface}"
            prev_accepted = True
            continue
        if low in seen_lower:
            continue
        if is_initial:
            continue
        seen_lower.add(low)
        found.append(surface)
        prev_accepted = True

    return found[:max_terms]


# ── Personal-document search detection (2026-08-29) ─────────────────────
# "please search for documents related to the ABC class I am currently
# enrolled in" fired the WEB trigger (heuristic "explicit search request"
# conf 0.80 → 3 Tavily sub-searches, one literally "Add dates and deadlines
# to Google Calendar") and the agentic gate's Tier-1 web arm — but the
# search TARGET is the user's own document corpus, not the internet.
# Deliberately UNDER-fires: search verb + personal-doc noun in proximity +
# a first-person/ownership anchor, short messages only (paste lesson), and
# an explicit web/internet mention disqualifies.

_DOC_SEARCH_VERB_NOUN_RE = re.compile(
    r"\b(?:search|find|locate|look|pull|dig|check)\w*"
    r"(?:\W+\w+){0,6}?\W+"
    r"(?:documents?|docs?|notes?|files?|uploads?|attachments?|materials?|syllab(?:us|i|uses))\b",
    re.IGNORECASE,
)
# "uploads"/"attachments" are self-anchoring (2026-09-07): nobody searches
# the web for "the uploads" — the noun itself names the user's own corpus
# ("can we look in the user uploads for the homework" carried no my/our/I
# anchor and stayed in tool-less chat).
# Course-material nouns are self-anchoring too (2026-09-10): "can we check
# course docs and see if there a weekend back up?" carried no my/our/I anchor
# and stayed unrouted — a student's "course docs"/"the syllabus"/"lecture
# notes" name their own uploaded corpus, never the web.
_PERSONAL_ANCHOR_RE = re.compile(
    r"\b(?:my|our)\b|\bi(?:'m|\s+am)\b|\bi\s+(?:have|uploaded|saved|gave|sent|shared)\b"
    r"|\b(?:enrolled|uploaded|saved)\b|\b(?:uploads?|attachments?)\b"
    r"|\b(?:course|class|lecture|homework|hw|assignment|semester)\s+"
    r"(?:docs?|documents?|notes?|files?|materials?|pdfs?|syllab(?:us|i))\b"
    r"|\bsyllab(?:us|i|uses)\b",
    re.IGNORECASE,
)
# "search the web/online for ..." is a real web request even if a doc noun
# follows. "Google calendar"/"Google Docs" must NOT disqualify — only the
# bare web/internet/online words and "google it/search".
_WEB_EXPLICIT_RE = re.compile(
    r"\bweb\b|\bonline\b|\binternet\b|\bgoogle\s+(?:it|search)\b|\bwww\b|https?://",
    re.IGNORECASE,
)
# Third-party product docs (audit F27 2026-08-31): "the FastAPI docs" is a
# reference-lookup target, not the user's corpus — a TitleCase word directly
# before the doc noun disqualifies unless it is itself possessive-anchored
# ("my Python notes" stays personal).
_THIRD_PARTY_DOC_RE = re.compile(
    r"\b([A-Za-z][\w.+-]*)\s+(?:docs?|documentation)\b"
)
_PERSONAL_DOC_SEARCH_MAX_WORDS = 60


def is_personal_doc_search(q: str) -> bool:
    """True when the message asks to search/find the USER'S OWN documents
    (notes, files, uploads, a syllabus) — internal-retrieval intent, never a
    web search. Conservative by design; a miss just means the old routing."""
    if not q or not q.strip():
        return False
    if len(q.split()) > _PERSONAL_DOC_SEARCH_MAX_WORDS:
        return False  # paste-sized message: incidental vocabulary, not a command
    if _WEB_EXPLICIT_RE.search(q):
        return False
    for m in _THIRD_PARTY_DOC_RE.finditer(q):
        qualifier = m.group(1)
        # Sentence-initial verbs/determiners capitalize too — only an
        # unexpected TitleCase word (a product name) disqualifies.
        if (qualifier[0].isupper()
                and qualifier.lower() not in (
                    "my", "our", "your", "the", "these", "those", "any", "all",
                    "some", "search", "find", "locate", "look", "pull", "dig",
                    "check")):
            prefix = q[:m.start()].rstrip().lower()
            if not prefix.endswith(("my", "our")):
                return False
    return bool(_DOC_SEARCH_VERB_NOUN_RE.search(q) and _PERSONAL_ANCHOR_RE.search(q))


# Note-save request (2026-09-10): "jot down a note for this session: TA
# sessions are Saturdays at 11 CT," found no route to create_daemon_note —
# the deployed detect_self_note_intent only recognizes Daemon's OWN
# "note to yourself/for future" phrasing, not a user asking Daemon to save a
# session/personal reminder. Head-anchored (an optional polite prefix is
# allowed) so a mid-message mention ("I jotted a note earlier") never
# matches, and deliberately excludes bare "note that X" (a common way to
# just INFORM Daemon of a fact, not a request to persist a note).
_NOTE_SAVE_NEGATION_RE = re.compile(
    r"^(?:don'?t|do\s+not|never|no\s+need\s+to|please\s+don'?t)\b", re.IGNORECASE,
)
_NOTE_SAVE_VERB_RE = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+)?"
    r"(?:jot\s+(?:down|this)\s+(?:a\s+)?note|"
    r"save\s+(?:this\s+)?(?:as\s+)?(?:a\s+)?note|"
    r"write\s+(?:this\s+)?(?:down\s+)?(?:as\s+)?(?:a\s+)?note|"
    r"make\s+(?:a\s+)?note|"
    r"remember\s+this(?:\s+for\s+(?:me|later|next\s+time))?|"
    r"note\s+to\s+self)\b",
    re.IGNORECASE,
)


def is_note_save_request(text: str) -> bool:
    """True for a head-anchored request that Daemon persist a session/
    personal note ("jot down a note for this session: …", "save this as a
    note", "remember this", "note to self …"). Deliberately NOT "note that
    the deadline moved" (informing Daemon of a fact — a distinct, common
    usage with no "note" object-noun verb of its own) nor past-tense
    narration ("I jotted a note earlier", which does not open the message).
    Negation-guarded and quote-guarded.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in "\"'“‘":
        return False
    if _NOTE_SAVE_NEGATION_RE.match(stripped):
        return False
    return bool(_NOTE_SAVE_VERB_RE.match(stripped))


def is_task_directive(q: str) -> bool:
    """True when the user is DIRECTING Daemon to do something — a head-
    clause imperative (``_IMPERATIVE_REQUEST_RE``), a note-save request
    (``is_note_save_request``), or an explicit action-registry intent
    (``core.actions.registry.detect_action_intent``) — rather than venting
    or asking a question (2026-09-10, round 4, B10).

    Live finding: `utils.tone_detector.detect_crisis_level`'s distress-
    sticky floor AND borderline backstop both floored "jot down a note for
    this session: TA sessions are Saturdays at 11 CT" to CONCERN off a
    borderline semantic score even though the arbiter itself said
    CONVERSATIONAL — the reply then carried LIGHT SUPPORT ("let them vent")
    onto a plain task instruction. Both stages stand down when the message
    is a task directive; the arbiter's/semantic verdict then stands.

    QUESTIONS are NEVER task directives ("what's the point of anything"
    keeps the backstop) even when a question happens to open with a
    directive-shaped verb — this predicate is consulted by tone-safety
    code, which must not lose a genuinely distress-shaped question to a
    coincidental grammatical match.
    """
    text = (q or "").strip()
    if not text:
        return False
    if "?" in text or is_question(text):
        return False
    clauses = request_clauses(text)
    head = _normalize(clauses[0]) if clauses else _normalize(text)
    if _IMPERATIVE_REQUEST_RE.match(head):
        return True
    if is_note_save_request(text):
        return True
    # lazy import: leaf parse-layer function (core.actions.registry has no
    # module-level dependency on this module) — avoids a module-load-time
    # cycle and stays a patch point for tests, matching the established
    # call-time-import convention used by is_request_shaped above.
    from core.actions.registry import detect_action_intent  # lazy import: cycle
    return detect_action_intent(text) is not None


@dataclass
class QueryAnalysis:
    text: str
    tokens: List[str]
    is_question: bool
    is_command: bool
    is_deictic: bool
    is_followup: bool
    token_count: int
    char_count: int
    intents: Set[str]
    is_heavy_topic: bool = False  # Crisis/sensitive topics requiring inline fact extraction
    is_meta_conversational: bool = False  # Query asking about conversation history itself
    is_small_talk: bool = False  # Terse acknowledgment → builder's lightweight-context path


def analyze_query(q: str, model_manager=None) -> QueryAnalysis:
    """
    Analyze query for various properties including heavy topic classification.

    Args:
        q: Query text
        model_manager: Optional model manager for LLM-based heavy topic classification.
                      If not provided, only heuristic classification is used.

    Returns:
        QueryAnalysis with all query properties
    """
    tokens = keyword_tokens(q)
    intents: Set[str] = set()
    q_is_question = is_question(q)
    q_is_command = is_command(q)
    q_is_deictic = is_deictic(q)
    q_is_follow = is_deictic_followup(q)
    q_is_meta = is_meta_conversational(q)

    if q_is_question:
        intents.add("question")
    if q_is_command:
        intents.add("command")
    if q_is_meta:
        intents.add("meta_conversational")
    if not intents:
        intents.add("statement")

    # Heavy topic classification (synchronous - uses heuristics only by default)
    q_is_heavy = _is_heavy_topic_heuristic(q)

    # Terse acknowledgments route to the lightweight prompt path — but never
    # for heavy/crisis topics, which need the full context apparatus.
    try:
        from config.app_config import LIGHT_PROMPT_MAX_WORDS as _lp_max  # lazy import: patch-point (tests/test_thread_surfacing.py:200)
    except Exception:
        _lp_max = 8
    q_is_small_talk = (not q_is_heavy) and is_casual_acknowledgment(q, max_words=_lp_max)

    return QueryAnalysis(
        text=q or "",
        tokens=tokens,
        is_question=q_is_question,
        is_command=q_is_command,
        is_deictic=q_is_deictic,
        is_followup=q_is_follow,
        token_count=len(tokens),
        char_count=len(q or ""),
        intents=intents,
        is_heavy_topic=q_is_heavy,
        is_meta_conversational=q_is_meta,
        is_small_talk=q_is_small_talk,
    )


async def analyze_query_async(q: str, model_manager=None) -> QueryAnalysis:
    """
    Async version of analyze_query that can use LLM for heavy topic classification.

    Args:
        q: Query text
        model_manager: Optional model manager for LLM classification

    Returns:
        QueryAnalysis with all query properties including LLM-based heavy topic result
    """
    # Run synchronous analysis first
    analysis = analyze_query(q, model_manager=None)

    # If heuristic already says it's heavy, skip LLM
    if analysis.is_heavy_topic:
        return analysis

    # Try LLM classification if available
    if model_manager is not None:
        try:
            is_heavy = await _classify_heavy_topic_llm(q, model_manager)
            # Update the analysis
            return QueryAnalysis(
                text=analysis.text,
                tokens=analysis.tokens,
                is_question=analysis.is_question,
                is_command=analysis.is_command,
                is_deictic=analysis.is_deictic,
                is_followup=analysis.is_followup,
                token_count=analysis.token_count,
                char_count=analysis.char_count,
                intents=analysis.intents,
                is_heavy_topic=is_heavy,
                is_meta_conversational=analysis.is_meta_conversational,
            )
        except Exception as e:
            logger.debug(f"[QueryChecker] LLM heavy topic classification failed: {e}")

    return analysis

# ===== Heavy Topic Classification =====

# Configuration
# Retired 2026-09-12: length alone no longer implies a heavy topic (see
# _is_heavy_topic_heuristic). Kept as a named constant only so the env
# variable does not silently become a no-op for anyone who set it; nothing
# reads it. Delete with the env var.
HEAVY_TOPIC_CHAR_THRESHOLD = int(os.getenv("HEAVY_TOPIC_CHAR_THRESHOLD", "2500"))
HEAVY_TOPIC_MODEL = os.getenv("HEAVY_TOPIC_MODEL", "gpt-4o-mini")
HEAVY_TOPIC_TIMEOUT = float(os.getenv("HEAVY_TOPIC_TIMEOUT", "2.0"))
HEAVY_TOPIC_MAX_TOKENS = int(os.getenv("HEAVY_TOPIC_MAX_TOKENS", "10"))

# Heavy topic keywords (crisis, violence, human rights, emotional/mental health)
HEAVY_KEYWORDS = {
    # Political violence & enforcement
    "raid", "raids", "ice", "deportation", "deported", "arrested", "arrest", "arrests",
    "military", "police", "protest", "protests", "riot", "riots", "violence", "violent",
    "shooting", "shot", "tear gas", "pepper spray", "detention", "detained", "detain",
    "federal agents", "national guard", "troops", "soldiers",
    "undocumented", "illegal", "immigration", "deport",

    # Crisis & trauma
    "crisis", "emergency", "disaster", "tragedy", "trauma", "traumatic",
    "killed", "dead", "death", "deaths", "casualties", "wounded", "injured",

    # Human rights & persecution
    "persecution", "discriminat*", "racism", "racist", "hate crime",
    "ethnic cleansing", "genocide", "war crime", "torture", "abuse",
    "refugee", "refugees", "asylum", "sanctuary",

    # Conflict & war
    "war", "warfare", "combat", "attack", "attacks", "bomb", "bombing",
    "terrorist", "terrorism", "insurgent", "militant",

    # Authoritarianism
    "authoritarian", "dictatorship", "oppression", "crackdown",
    "martial law", "curfew", "lockdown",

    # Mental health & emotional distress
    "depressed", "depression", "anxiety", "anxious", "panic", "panic attack",
    "ptsd", "mental health", "mental illness", "bipolar", "schizophrenia",
    "therapy", "therapist", "psychiatrist", "psychologist", "counseling",
    "medication", "antidepressant", "psychiatric", "psych ward",
    "suicidal", "suicide", "kill myself", "end my life", "self-harm", "self harm",
    "cutting", "overdose", "pills",

    # Emotional crisis states
    "breakdown", "nervous breakdown", "meltdown", "losing it",
    "can't take it", "can't cope", "overwhelmed", "hopeless", "helpless",
    "despair", "devastated", "heartbroken", "broken", "shattered",
    "lonely", "isolated", "alone", "abandoned", "worthless", "hate myself",
    "scared", "terrified", "afraid", "frightened", "fear",

    # Relationship distress & life crises
    "breakup", "broke up", "divorce", "divorcing", "separated", "separation",
    "cheating", "cheated", "affair", "betrayed", "betrayal",
    "miscarriage", "stillborn", "pregnancy loss", "lost the baby",
    "funeral", "mourning", "grieving", "grief", "loss",
    "fired", "laid off", "lost my job", "unemployment",
    "evicted", "eviction", "homeless", "foreclosure",

    # Anger & violence (personal)
    "angry", "furious", "rage", "enraged", "hate", "hatred",
    "want to hurt", "want to kill", "violence", "fight", "assault",
    "domestic violence", "abusive", "abuser",
}

# Word-boundary (bare single words) / substring (phrases) matcher over
# HEAVY_KEYWORDS — see utils/trigger_match.py. 2026-09-08: bare substring
# matching let "ice" fire inside "Price"/"office"/"notice"/"nice"/"device";
# a pasted R homework script whose response variable was `Price` (`model <-
# lm(Price ~ ., data = used_car_data)`) was stored `is_heavy_topic=True` on
# that alone, which then fed the tone sticky-floor's history scan (see
# tone_detector._recent_distress_from_history) into an unwarranted CONCERN
# floor on 11 homework turns. Phrases (containing a space, e.g. "tear gas")
# keep substring semantics; a deliberately truncated stem declares
# itself with a trailing "*" ("discriminat*" → discriminate/discrimination/
# discriminatory) now that bare words are bounded on BOTH sides
# (2026-09-12: 'dead' had fired HEAVY on "deadline"/"deadlift" and 'war' on
# "warning"/"warm" — 282 hits in the owner's corpus, and 17 rows stored
# is_heavy_topic=True with no valid heavy hit at all).
_HEAVY_MATCHER = compile_keyword_matcher(sorted(HEAVY_KEYWORDS))


def heavy_prefix_only_hits(text: str) -> List[str]:
    """HEAVY_KEYWORDS hits that only the pre-2026-09-12 left-boundary-only
    matcher would have produced (see `trigger_match.prefix_only_hits`).
    Used by `tone_detector._heavy_row_is_distress_evidence` to neutralize
    corpus rows stored `is_heavy_topic=True` on nothing but a boundary bug."""
    if not text or not isinstance(text, str):
        return []
    return prefix_only_hits(_HEAVY_MATCHER, text.lower())


def heavy_keyword_hits(text: str) -> List[str]:
    """Word-bounded (single words) / substring (phrases) HEAVY_KEYWORDS hits
    in `text`. Under-fires by design relative to the old bare-substring scan
    — see the 2026-09-08 note on `_HEAVY_MATCHER` above."""
    if not text or not isinstance(text, str):
        return []
    return [h.keyword for h in _HEAVY_MATCHER.iter_hits(text.lower())]


def _is_heavy_topic_heuristic(q: str) -> bool:
    """
    Fast heuristic check for heavy topics.

    Strategy: keyword matching against crisis/violence/rights/emotional/
    mental-health terms — heaviness is a property of the CONTENT.

    2026-09-12: a bare length check ("> HEAVY_TOPIC_CHAR_THRESHOLD chars =
    likely article/news") used to return True on its own, which made EVERY
    long paste a heavy topic. An 18,549-char R package-install log on
    2026-09-11 was stored `is_heavy_topic=True` on length alone, and
    `tone_detector._recent_distress_from_history` then armed the
    distress-sticky floor from it: nine consecutive package-debugging turns
    were answered under LIGHT SUPPORT ("let them vent, don't offer
    unsolicited advice") while the user was asking technical yes/no
    questions. Length is not a signal here; a genuinely heavy long paste
    carries heavy vocabulary, and `analyze_query_async` still consults
    `_classify_heavy_topic_llm` whenever this heuristic says no.

    Returns:
        True if heuristics suggest heavy topic (political, emotional, or mental health crisis)
    """
    if not q or not isinstance(q, str):
        return False

    # Keyword matching (word-bounded for bare words, substring for phrases)
    hits = len(heavy_keyword_hits(q))

    # If multiple heavy keywords appear, likely a heavy topic
    if hits >= 2:
        return True
    
    # Single keyword but with contextual markers (numbers, locations, quotes)
    if hits == 1:
        # Check for news article markers
        has_numbers = bool(re.search(r"\b\d{1,3}[,\s]?\d{0,3}\b", q))
        has_quotes = '"' in q or '"' in q or '"' in q
        has_locations = bool(re.search(
            r"\b(Chicago|Illinois|Texas|California|New York|Washington|D\.?C\.?)\b",
            q, re.IGNORECASE
        ))
        
        if (has_numbers and has_quotes) or (has_numbers and has_locations):
            return True
    
    return False


async def _classify_heavy_topic_llm(q: str, model_manager) -> bool:
    """
    Use LLM to classify topic as heavy/normal.
    
    Args:
        q: Query text
        model_manager: ModelManager instance
    
    Returns:
        True if heavy, False if normal
    
    Raises:
        asyncio.TimeoutError if classification times out
        Exception if LLM call fails
    """
    if not model_manager or not hasattr(model_manager, "generate_once"):
        return False
    
    # Build prompt
    prompt = _build_heavy_topic_prompt(q)
    
    # Preserve current model
    prev_model = None
    try:
        if hasattr(model_manager, "get_active_model_name"):
            prev_model = model_manager.get_active_model_name()
        
        # Switch to classifier model if registered
        if hasattr(model_manager, "switch_model"):
            if hasattr(model_manager, "api_models") and HEAVY_TOPIC_MODEL in model_manager.api_models:
                model_manager.switch_model(HEAVY_TOPIC_MODEL)
    except Exception as e:
        logger.debug(f"[QueryChecker] Model switch failed: {e}")
    
    try:
        # Call LLM with timeout
        response = await asyncio.wait_for(
            model_manager.generate_once(
                prompt,
                max_tokens=HEAVY_TOPIC_MAX_TOKENS
            ),
            timeout=HEAVY_TOPIC_TIMEOUT
        )
        
        # Parse response
        result = _parse_heavy_topic_response(response)
        logger.debug(f"[QueryChecker] LLM heavy topic result: {result}")
        return result
        
    finally:
        # Restore previous model
        try:
            if prev_model and hasattr(model_manager, "switch_model"):
                model_manager.switch_model(prev_model)
        except Exception:
            pass


def _build_heavy_topic_prompt(q: str) -> str:
    """Build LLM prompt for heavy topic classification."""
    # Truncate very long input
    truncated = q[:1000]
    if len(q) > 1000:
        truncated += "..."

    return f"""Classify this message as HEAVY or NORMAL.

HEAVY topics: political violence, protests, raids, arrests, deportation, war, terrorism, human rights crises, personal safety threats, trauma, discrimination, persecution, mental health crises (depression, anxiety, suicidal thoughts, self-harm, PTSD), emotional distress (grief, heartbreak, breakup, divorce, job loss), relationship crises (abuse, betrayal, domestic violence), personal emergencies.

NORMAL topics: coding, general knowledge, casual conversation, entertainment, hobbies, academic topics, advice, mild emotions (slightly happy/sad), everyday stress, general questions.

MESSAGE:
{truncated}

Respond with ONLY one word: HEAVY or NORMAL"""


def _parse_heavy_topic_response(response: str) -> bool:
    """
    Parse LLM response for heavy topic classification.
    
    Args:
        response: Raw LLM output
    
    Returns:
        True if HEAVY, False otherwise
    """
    if not response:
        return False
    
    normalized = response.strip().upper()
    
    # Explicit markers
    if "HEAVY" in normalized:
        return True
    if "NORMAL" in normalized:
        return False
    
    # Fuzzy matches
    heavy_indicators = ["CRISIS", "SENSITIVE", "SEVERE", "YES", "TRUE"]
    if any(indicator in normalized for indicator in heavy_indicators):
        return True
    
    return False


# ===== Conversation Thread Detection =====

# Configuration
THREAD_TIME_HARD_CUTOFF = int(os.getenv("THREAD_TIME_HARD_CUTOFF", "7200"))  # 2 hours
THREAD_TIME_CLOSE_BONUS = int(os.getenv("THREAD_TIME_CLOSE_BONUS", "300"))   # 5 min
THREAD_TIME_MEDIUM_BONUS = int(os.getenv("THREAD_TIME_MEDIUM_BONUS", "1800"))  # 30 min
THREAD_CONTINUITY_THRESHOLD = float(os.getenv("THREAD_CONTINUITY_THRESHOLD", "0.5"))

# Thread scoring weights
THREAD_WEIGHT_KEYWORDS = float(os.getenv("THREAD_WEIGHT_KEYWORDS", "0.5"))
THREAD_WEIGHT_TIME = float(os.getenv("THREAD_WEIGHT_TIME", "0.25"))
THREAD_WEIGHT_HEAVY = float(os.getenv("THREAD_WEIGHT_HEAVY", "0.15"))
THREAD_WEIGHT_TOPIC = float(os.getenv("THREAD_WEIGHT_TOPIC", "0.1"))
THREAD_WEIGHT_DEICTIC = float(os.getenv("THREAD_WEIGHT_DEICTIC", "0.25"))  # Deictic follow-ups

# Thread-breaking phrases
THREAD_BREAK_MARKERS = {
    "changing topics", "different topic", "switching gears",
    "on another note", "anyway,", "by the way,", "moving on",
    "different subject", "new question", "unrelated"
}
# Word-boundary matched via utils.trigger_match (dm01_raw_substring /
# CGR-20260913-005, anchor #28): bare 'unrelated' must not fire on
# containment inside an unrelated token ("unrelatedid"). Negation-INSENSITIVE
# (parent fix D5, 2026-09-14): a marker states the topic change itself, and
# the generic negation cues (never / stop / skip / rather than / instead of)
# routinely precede a real change ("never mind, moving on"). 'anyway,' /
# 'by the way,' keep raw-substring (comma) semantics — they fail the
# chokepoint's bare-word fullmatch, so no structural check is needed at this
# site. Built from the SAME set above — no vocabulary duplication (BC-76).
_THREAD_BREAK_MATCHER = compile_keyword_matcher(sorted(THREAD_BREAK_MARKERS))


def extract_thread_keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords for thread continuity detection.

    Args:
        text: Input text

    Returns:
        Set of lowercase keywords (min 3 chars, no stopwords)
    """

    # Extract alphanumeric words only (strips punctuation)
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]+\b', text_lower)

    # Filter by minimum length
    tokens = [w for w in words if len(w) >= 3]

    # Common stopwords to filter
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "about",
        "what", "when", "where", "which", "who", "why", "how",
        "can", "could", "would", "should", "will", "are", "were", "was",
        "have", "has", "had", "been", "being", "does", "did", "doing",
        "you", "your", "they", "their", "them", "his", "her", "him"
    }

    return set(t for t in tokens if t not in stopwords)


def has_thread_break_marker(query: str) -> bool:
    """
    Check if query contains explicit thread-breaking phrases.

    Negation-INSENSITIVE (parent fix D5, 2026-09-14): a marker states the
    topic change itself, and generic negation cues ("never mind", "stop",
    "skip", "rather than", "instead of") routinely precede a real change,
    so every word-bounded hit counts. A missed break would keep the
    previous thread's context attached (calculate_thread_continuity_score).

    Args:
        query: User query text

    Returns:
        True if query signals a topic switch
    """
    query_lower = query.lower()
    return _THREAD_BREAK_MATCHER(query_lower)


def calculate_thread_continuity_score(
    current_query: str,
    last_query: str,
    time_diff_seconds: float,
    both_heavy: bool = False,
    same_topic: bool = False,
    current_topic: Optional[str] = None,
    last_was_heavy: bool = False
) -> float:
    """
    Calculate a continuity score (0.0-1.0) indicating if current query continues last conversation.

    Args:
        current_query: Current user query
        last_query: Previous conversation query
        time_diff_seconds: Time elapsed since last conversation
        both_heavy: True if both queries are heavy topics
        same_topic: True if both have same detected topic
        current_topic: The current topic (to filter "general")
        last_was_heavy: True if last conversation was heavy

    Returns:
        Continuity score (0.0-1.0). Score >= 0.5 suggests thread continuity.
    """
    score = 0.0
    
    # Hard cutoff: too much time = auto-break
    if time_diff_seconds > THREAD_TIME_HARD_CUTOFF:
        return 0.0
    
    # Explicit break markers override everything
    if has_thread_break_marker(current_query):
        return 0.0
    
    # Factor 1: Keyword overlap
    last_keywords = extract_thread_keywords(last_query)
    curr_keywords = extract_thread_keywords(current_query)

    if last_keywords and curr_keywords:
        overlap = len(last_keywords & curr_keywords)
        union = len(last_keywords | curr_keywords)
        overlap_ratio = overlap / union if union > 0 else 0.0
        keyword_score = overlap_ratio * THREAD_WEIGHT_KEYWORDS
        score += keyword_score
        logger.debug(
            f"[Thread] Keyword overlap: last={last_keywords}, curr={curr_keywords}, "
            f"overlap={overlap}, union={union}, ratio={overlap_ratio:.3f}, score_contrib={keyword_score:.3f}"
        )
    else:
        logger.debug(f"[Thread] No keywords extracted from one or both queries")
    
    # Factor 2: Time proximity
    if time_diff_seconds < THREAD_TIME_CLOSE_BONUS:
        # Very recent (< 5 min): full time bonus
        score += THREAD_WEIGHT_TIME
    elif time_diff_seconds < THREAD_TIME_MEDIUM_BONUS:
        # Medium recent (< 30 min): decaying bonus
        decay = 1.0 - ((time_diff_seconds - THREAD_TIME_CLOSE_BONUS) / 
                       (THREAD_TIME_MEDIUM_BONUS - THREAD_TIME_CLOSE_BONUS))
        score += THREAD_WEIGHT_TIME * decay
    # else: no time bonus (but not auto-break unless > 2 hours)
    
    # Factor 3: Heavy topic continuity
    # Full bonus if both are heavy
    if both_heavy:
        score += THREAD_WEIGHT_HEAVY
    # Partial bonus if previous was heavy and we're discussing same specific topic
    # (follow-up questions about a crisis are likely to continue the thread)
    elif last_was_heavy and same_topic and current_topic and current_topic.lower() != "general":
        score += THREAD_WEIGHT_HEAVY * 0.5  # Half credit for heavy topic continuity
    
    # Factor 4: Same detected topic (exclude "general")
    # Give larger bonus for specific topic match (helps when keywords don't overlap semantically)
    if same_topic and current_topic and current_topic.lower() != "general":
        # Double the topic weight for non-general topics (0.2 instead of 0.1)
        # This helps when discussing the same specific topic with different vocabulary
        score += THREAD_WEIGHT_TOPIC * 2.0

    # Factor 5: Deictic reference bonus
    # Short follow-ups with "it", "that", "watched it" etc. strongly suggest continuation
    if is_deictic(current_query):
        score += THREAD_WEIGHT_DEICTIC
        logger.debug(f"[Thread] Deictic bonus: +{THREAD_WEIGHT_DEICTIC:.3f}")

    # Factor 6: Casual-time decay
    # When neither message is heavy/topical and the gap is beyond medium (30 min),
    # decay the score so that keyword overlap alone can't bridge the gap.
    # Important conversations (heavy topics, specific shared topics) are unaffected.
    is_casual_pair = (
        not both_heavy and not last_was_heavy
        and (not same_topic or not current_topic or current_topic.lower() == "general")
    )
    if is_casual_pair and time_diff_seconds > THREAD_TIME_MEDIUM_BONUS:
        casual_decay = max(0.0, 1.0 - (
            (time_diff_seconds - THREAD_TIME_MEDIUM_BONUS)
            / (THREAD_TIME_HARD_CUTOFF - THREAD_TIME_MEDIUM_BONUS)
        ))
        logger.debug(
            f"[Thread] Casual-time decay: {casual_decay:.3f} "
            f"(gap={time_diff_seconds:.0f}s, score {score:.3f} -> {score * casual_decay:.3f})"
        )
        score *= casual_decay

    return score


def topics_related(thread_topic: str, current_topic: str) -> bool:
    """Loose continuity check between the (previous turn's) thread topic and
    the current query topic. Related when either contains the other or they
    share a substantive (>3-char) word. Unclassified topics ("general", empty)
    give no signal → treated as related so the thread wording is preserved
    rather than asserting a shift we can't support.

    Moved here from core.orchestrator (2026-09-03) so storage-time thread
    depth (``belongs_to_thread``) and the read-time [THREAD CONTEXT] honesty
    branch judge topic continuity with ONE predicate — they used to disagree
    (exact-equality at storage vs. loose containment at read time).
    """
    a = (thread_topic or "").strip().lower()
    b = (current_topic or "").strip().lower()
    if not a or not b or a == "general" or b == "general":
        return True
    if a == b or a in b or b in a:
        return True
    a_words = {w for w in re.split(r"\W+", a) if len(w) > 3}
    b_words = {w for w in re.split(r"\W+", b) if len(w) > 3}
    return bool(a_words & b_words)


def belongs_to_thread(
    current_query: str,
    last_conversation: dict,
    current_topic: Optional[str] = None
) -> bool:
    """
    Determine if current query continues the immediate previous conversation thread.

    Threads are strictly consecutive - any topic switch breaks the thread.

    Args:
        current_query: Current user query
        last_conversation: Dict with keys: query, response, timestamp, is_heavy_topic, topic, thread_depth
        current_topic: Current detected topic

    Returns:
        True if current query continues the thread
    """

    # Calculate time difference
    last_time = last_conversation.get("timestamp")
    if isinstance(last_time, str):
        try:
            last_time = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
            time_diff = (datetime.now() - last_time).total_seconds()
        except Exception:
            # Can't parse timestamp, use conservative time bonus
            time_diff = 3600.0  # Assume 1 hour
    elif isinstance(last_time, datetime):
        time_diff = (datetime.now() - last_time).total_seconds()
    else:
        time_diff = 3600.0  # Default

    # Check if both are heavy topics
    last_heavy = last_conversation.get("is_heavy_topic", False)
    curr_heavy = _is_heavy_topic_heuristic(current_query)

    # Check if same topic. Loose match via the SAME predicate the read-time
    # [THREAD CONTEXT] honesty branch uses (2026-09-03) — exact equality made
    # every classifier relabel ("Playing Fetch" → "Playing Games") reset the
    # thread depth to 1 while the read side called the topics related.
    # "general"/empty carry no signal and never earn the bonus (topics_related
    # deliberately returns True for them, so exclude explicitly here).
    last_topic = last_conversation.get("topic", "general")
    same_topic = (
        bool(current_topic)
        and bool(last_topic)
        and str(last_topic).lower() != "general"
        and str(current_topic).lower() != "general"
        and topics_related(str(last_topic), str(current_topic))
    )

    # Build previous conversation text (query + response for better keyword matching)
    last_query = last_conversation.get("query", "")
    last_response = last_conversation.get("response", "")
    last_full_text = f"{last_query} {last_response}"

    # Calculate continuity score
    score = calculate_thread_continuity_score(
        current_query=current_query,
        last_query=last_full_text,  # Use full conversation text
        time_diff_seconds=time_diff,
        both_heavy=(last_heavy and curr_heavy),
        same_topic=same_topic,
        current_topic=current_topic,
        last_was_heavy=last_heavy  # Pass last_heavy for partial credit
    )

    # Thread momentum bonus: if already in a thread (depth >= 2), give a small bonus
    # This helps maintain longer threads even when keywords don't overlap perfectly
    last_depth = last_conversation.get("thread_depth", 1)
    if last_depth >= 2 and same_topic and current_topic and current_topic.lower() != "general":
        # Add 0.1 bonus for thread momentum (helps reach 0.5 threshold)
        momentum_bonus = 0.1
        score += momentum_bonus
        logger.debug(f"[Thread] Thread momentum bonus: +{momentum_bonus:.3f} (depth={last_depth})")

    # Debug logging
    logger.debug(
        f"[Thread] Continuity check: score={score:.3f}, threshold={THREAD_CONTINUITY_THRESHOLD}, "
        f"time_diff={time_diff:.1f}s, last_heavy={last_heavy}, curr_heavy={curr_heavy}, "
        f"same_topic={same_topic} (last={last_topic}, curr={current_topic}), depth={last_depth}"
    )

    return score >= THREAD_CONTINUITY_THRESHOLD


# --- New-data override support (2026-09-06) ---------------------------------
# The STM analyzer's LLM prompt is deliberately biased toward "recall" when
# in doubt, which mislabels two shapes: a REQUEST for analysis (not a report
# of an event at all) and a fresh present-tense self-report that carries a
# data point (clock time, date, dose, day count) absent from the short-term
# window. Both predicates below are closed grammatical/numeric sets — no
# topic vocabulary — per project doctrine.

# Closed deictic set: present-tense time anchors. Order matters only for
# readability; matching below is word-bounded so overlap (e.g. "now" inside
# "right now") is harmless.
PRESENT_DEICTIC_ANCHORS: tuple = (
    "today", "tonight", "right now", "now", "currently", "at the moment",
    "this morning", "this afternoon", "this evening", "so far today",
)
_PRESENT_DEICTIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in PRESENT_DEICTIC_ANCHORS) + r")\b",
    re.IGNORECASE,
)

# First-person present-progressive / present-state verb forms.
_PRESENT_PROGRESSIVE_RE = re.compile(
    r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+(?:\w+\s+){0,2}?\w+ing\b",
    re.IGNORECASE,
)
_PRESENT_STATE_VERB_RE = re.compile(
    r"\bi\s+(?:feel|am|have\s+been|'ve\s+been)\b",
    re.IGNORECASE,
)

# Data-shaped token families: clock times, dates, number+unit, "day N".
_CLOCK_TIME_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_MONTH_DAY_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_ORDINAL_DAY_RE = re.compile(r"\bday\s+\d+\b", re.IGNORECASE)
# Any number+unit-suffix shape — no named units (closed grammatical shape,
# not a topic vocabulary): "5 mg", "3 miles", "20%". The suffix must not be
# a function word (2026-09-08: "alarm got me at 930 and I got out of bed"
# produced the tokens "930and", "3or", "330but" — a clock shorthand glued to
# the conjunction after it — which the STM prompt then reported as novel
# details). The closed grammatical stoplist is not a topic vocabulary.
_UNIT_STOPWORDS = (
    "and|or|but|of|to|in|at|on|for|the|a|an|is|was|so|if|my|i|it|by|as|then|than|that|"
    "this|when|with|from|not|be|do|did|am|are|were|had|has|have|got|get|which|who|what|"
    "how|its|his|her|our|your|their|me|we|you|he|she|they|up|out|off|about|after|before|"
    "til|till|until|ish|maybe|like|just|also|only|even|still|yet"
)
_NUMBER_UNIT_RE = re.compile(
    rf"\b\d+(?:\.\d+)?\s*(?!(?:{_UNIT_STOPWORDS})\b)[a-z%]{{1,6}}\b", re.IGNORECASE)

_DATA_TOKEN_RE = re.compile(
    "|".join([
        _CLOCK_TIME_RE.pattern,
        _ISO_DATE_RE.pattern,
        _SLASH_DATE_RE.pattern,
        _MONTH_DAY_RE.pattern,
        _ORDINAL_DAY_RE.pattern,
        _NUMBER_UNIT_RE.pattern,
    ]),
    re.IGNORECASE,
)


def _canonicalize_clock_time(raw: str) -> Optional[str]:
    """Canonicalize a matched clock-time span to "HH:MM" via the deployed
    single-time parser (utils.temporal_resolver._parse_single_time) so "10
    AM" and "10:00" compare equal. None if unparseable."""
    cleaned = (raw or "").strip().lower()
    try:
        hour, minute, _has_explicit = temporal_resolver._parse_single_time(cleaned)
    except Exception:
        hour, minute = None, None
    if hour is None:
        return None
    return f"{hour:02d}:{minute:02d}"


def _canonicalize_data_token(raw: str) -> str:
    cleaned = (raw or "").strip()
    if _CLOCK_TIME_RE.fullmatch(cleaned):
        canon = _canonicalize_clock_time(cleaned)
        if canon:
            return canon
    return re.sub(r"\s+", "", cleaned.lower())


def extract_data_tokens(q: str) -> List[str]:
    """Data-shaped tokens (clock times, dates, number+unit, "day N") in
    ``q``, normalized so equivalent forms compare equal (canonical clock
    times; lowercase + internal-whitespace-stripped otherwise). Order of
    first appearance; missing/garbage input -> []."""
    text = q if isinstance(q, str) else ""
    if not text.strip():
        return []
    tokens: List[str] = []
    for m in _DATA_TOKEN_RE.finditer(text):
        canon = _canonicalize_data_token(m.group(0))
        if canon:
            tokens.append(canon)
    return tokens


def has_present_state_report(q: str) -> bool:
    """A first-person present-progressive/state form ("I'm resting", "I
    feel...") co-occurring with a present-tense deictic anchor ("today",
    "right now"...) — the shape of a fresh self-report, as opposed to a
    request or a report of a past event. Closed grammatical set; missing
    input -> False."""
    text = q if isinstance(q, str) else ""
    if not text.strip():
        return False
    ql = _normalize(text)
    if not _PRESENT_DEICTIC_RE.search(ql):
        return False
    return bool(_PRESENT_PROGRESSIVE_RE.search(ql) or _PRESENT_STATE_VERB_RE.search(ql))
