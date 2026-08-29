"""
# core/agentic/gate.py

Module Contract
- Purpose: 4-tier agentic gate — decides whether a user query should be
  routed to the agentic search loop (tools, web, memory, knowledge, etc.)
  or fall through to standard enhanced streaming.
- Public interface:
  - AgenticDecision (dataclass): structured gate result (incl. veto_exempt)
  - evaluate_agentic_gate(user_text, entity_resolver, model_manager,
    corpus_manager, intent_info) -> AgenticDecision
  - apply_intent_veto(decision, intent_info, tone_level=None, query=None) -> AgenticDecision
    (query enables the tone-statement veto: elevated tone + first-person
    vent-shaped statement stands the gate down regardless of intent,
    2026-08-02; narrowed to vent shape 2026-08-15 — third-party/news
    statements and pronoun-split lookups ("look it up") are never vetoed
    or taught as no_search)
    — the intent veto, extracted so handle_submit can run the gate CONCURRENTLY
    with prepare_prompt (intent_info=None at launch, veto applied post-hoc once
    the context pipeline's classification exists). evaluate_agentic_gate
    still applies it inline when intent_info is passed. tone_level enables the
    corroborated veto: emotional_support at the STM-refined floor (>=0.60,
    below the 0.75 hard veto floor) vetoes ONLY when the tone detector
    independently reads the turn as CONCERN or above — two weak signals
    agreeing that this is an emotional turn, not a search task.
- Dependencies:
  - memory.graph_utils.extract_graph_entities (Tier 2 entity match)
  - utils.web_search_trigger.analyze_for_web_search_llm (Tier 4 LLM fallback)
  - knowledge.document_generator.detect_document_intent (doc gen detection).
    Tier-4 doc_gen_intent also carries "source" (2026-08-24): the LLM trigger's
    document_source ("research" | "conversation") threads through so handlers can
    pass the conversation transcript as source_material when the user wants THIS
    conversation written up (Tier-3 regex intents omit the key; handlers'
    deterministic backstop covers them).
  - knowledge.daemon_notes_manager.detect_self_note_intent (self-note detection)
  All imports are lazy (inside the function) with try/except guards.
- Side effects: None. Pure decision logic + one optional async LLM call.
- Email-by-name patterns: Tier 1 TOOL_KEYWORDS includes contact lookup keywords
  ('look up contact', 'find email', "'s email", etc.) and email-by-name regex
  patterns (e.g. "email <name>", "send <name> an email") that trigger agentic
  routing for contact resolution + propose_action.
- File/document retrieval: Tier 1 routes file/saved-document requests to the
  agentic loop so the file_read / file_list / get_full_document tools are offered.
  Detection is three-layered: FILE_ACCESS_KEYWORDS (literal fast-path),
  FILE_ACCESS_PATTERNS (regex tolerant of inflection/intervening words and
  capability assertions like "you have the tool"), and a pronoun/affirmation
  continuation ("pull it", or "yes" after the model offered "Want me to pull
  that up?") gated on prior file/document context. Distinct from Tier 3 document
  *generation*. Also counts as an explicit request so the intent veto can't
  suppress it. The enhanced (tool-less) streaming path carries a matching
  [ACTION HONESTY] note so a gate miss degrades to an honest "I can't this turn"
  + offer, never a confabulated reason.
- Insight / evidence-assembly routing (2026-08-23): detection runs BEFORE the
  Tier-3 doc-gen check (a personal-theme document request — "write a summary
  of my pattern with X for my therapist" — must route to core/insight, not
  web research) via core.insight.detector.detect_insight_request; on hit the
  decision carries modes=["insight"], insight_intent (serialized
  InsightIntent), and is ALWAYS veto_exempt (explicit requests work
  mid-distress; the synthesizer handles elevated-tone framing). Consent
  offer: maybe_arm_insight_offer(query, tone_level) arms a one-shot slot
  (max ONE offer per session, in-memory) for insight-shaped statements at
  non-elevated tone; a terse affirmation on the next turn yields an
  insight_assessment decision; anything else consumes and drops the offer
  permanently (anti-excavation). _reset_insight_offer_state() is the test
  hook.
- Tone-deferral clarify loop (2026-08-21): a TONE arm vetoing a
  REQUEST-shaped, non-vent query (imperative without a lookup cue — "review
  the tuesday logs"; lookup-cue and interrogative shapes escape the veto
  instead) sets decision.deferred_request and arms a ONE-SHOT module slot.
  handlers append a [DEFERRED REQUEST] system-prompt note (acknowledge +
  offer, never confabulate), and a terse affirmation on the immediately
  following turn re-runs the ORIGINAL query veto-exempt (explicit consent).
  Vent-shaped turns never get the offer (anti-excavation), and
  request-shaped queries never teach no_search.
- Continuation override: a terse affirmation (≤ CONTINUATION_MAX_WORDS words
  containing a CONTINUATION_PHRASES entry) after an agentic turn bypasses the
  casual skip. "Previous turn was agentic" is read from the corpus entry's
  stored response_mode (ground truth, written by memory_storage from
  provenance); a word-boundary keyword fallback covers only legacy entries
  that predate the field. Long messages that merely contain "yeah"/"sure"
  are new statements, never continuations (2026-07-15 benzo-turn incident).
- Follow-up resolution: Tier 4 builds a compact recent-conversation digest from
  corpus_manager (_build_recent_context) and passes it as conversation_context to
  analyze_for_web_search_llm, so elliptical follow-ups ("check the news", "any
  updates on that") resolve to topic-specific search terms instead of generic
  ones. No corpus_manager / no history → None → legacy (context-free) behavior.
- Knowledge-search backstop [2026-07-24]: the Tier-4 LLM sometimes sets
  needs_knowledge_search=True on collaborative-task follow-ups that merely NAME a
  concept ("Yeah and I'll need it in ATS-friendly format") — the user is directing
  work on their OWN materials, not asking an encyclopedic question. Routing those
  into the agentic knowledge loop hung a turn ~2 min (kimi-3 slow-round incident).
  needs_knowledge_search is now honored only when the query carries an actual
  knowledge-QUESTION signal (KNOWLEDGE_QUESTION_SIGNAL — interrogative / "explain"
  / "tell me about" / "?"); else it falls back to normal generation (which still
  carries wiki-RAG context). Fails OPEN — only a genuinely signal-less task
  statement is suppressed. The web_search_trigger prompt's KNOWLEDGE criteria were
  tightened in parallel (prompt-only guards don't hold — belt and suspenders).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re as _re_gate
import re as _re_gate2

logger = logging.getLogger("agentic_gate")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AgenticDecision:
    """Structured result from the agentic gate evaluation."""
    should_trigger: bool
    modes: List[str] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)
    matched_entities: Set[str] = field(default_factory=set)
    doc_gen_intent: Optional[Dict[str, Any]] = None
    self_note_intent: Optional[Dict[str, Any]] = None
    skip_initial_search: bool = False
    reason: str = ""
    # True when the trigger is an explicit request (search keywords, URL,
    # file access, doc-gen, self-note) that the intent veto must never
    # suppress. Lets apply_intent_veto() run post-hoc without recomputing
    # gate internals.
    veto_exempt: bool = False
    # True when veto_exempt came from a BARE URL alone (no explicit request
    # keyword like "go to http", no files/doc-gen/self-note). 2026-08-28
    # retrospective: "I don't want to die I am fucking scared and hurt
    # https://en.wikip…" (MEDIUM, harm_score 12) ran an agentic loop because
    # the pasted link made the vent veto-exempt — a link inside an acute
    # first-person vent is shared context, not a fetch request. The acute
    # tone arm may pierce THIS exemption only; explicit-keyword URL requests
    # stay exempt.
    veto_exempt_url_only: bool = False
    # Set when a TONE arm vetoed a request-shaped (non-vent) query
    # (2026-08-21): handlers inject a "acknowledge + offer to proceed" note
    # into the system prompt, and a terse affirmation on the NEXT turn
    # re-runs the original query veto-exempt (see _arm_deferred_request).
    deferred_request: Optional[str] = None
    # Insight / evidence-assembly mode (2026-08-23): serialized InsightIntent
    # dict when the query is an explicit insight request (theme sweep,
    # personal-theme document, insight assessment) or an affirmation of the
    # one-shot consent offer. Handlers route to _run_insight_mode; always
    # veto_exempt (explicit requests work even mid-distress).
    insight_intent: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Keyword constants (extracted verbatim from gui/handlers.py)
# ---------------------------------------------------------------------------

COMPUTATION_KEYWORDS = [
    'calculate', 'compute', 'solve', 'integral', 'derivative', 'equation',
    'fibonacci', 'factorial', 'median', 'standard deviation',
    'matrix', 'numpy', 'pandas', 'sympy',
    'regression', 'correlation', 'sum of', 'product of',
    'simplify', 'differentiate', 'integrate',
]

WEB_SEARCH_KEYWORDS = [
    'web search', 'search the web', 'search for', 'search online',
    'google ', 'look it up', 'fetch the', 'fetch url',
    'go to http', 'check out http', 'visit http',
]

TOOL_KEYWORDS = [
    'github', 'git stats', 'git_stats', 'git stat',
    'wolfram', 'sandbox', 'execute python',
    'search memory', 'search_memory',
    'loc ', 'lines of code', 'lines added', 'lines changed',
    'workflow', 'pull request', 'open issues', 'closed issues',
    'actions', 'releases',
    # Internet actions (email, telegram, discord)
    'send email', 'send an email', 'email to ', 'email him', 'email her',
    'email them', 'send a message to', 'send message to',
    'send telegram', 'send discord', 'message on telegram',
    'message on discord', 'notify ', 'text him', 'text her',
    # Contact lookup / email by name (no @ required)
    # NOTE: no bare "what is " here — it matched any sentence containing the
    # bigram ("I will see what is said on Reddit…") and fired irrelevant
    # searches (2026-07-16). "'s email"/"'s contact" already cover
    # "what is <name>'s email".
    'look up contact', 'lookup contact', 'find email', 'find contact',
    "'s email", "'s contact",
]

MEMORY_KEYWORDS = [
    'documentation', 'daemon docs', 'architecture',
    'do you remember', 'did we talk', 'did we discuss',
    'did i tell you', 'did i mention', 'have i told you',
    'what do you know about me', 'what are my',
    'my notes', 'obsidian', 'in my vault',
    'past conversations', 'search your memory',
    'search memory', 'check your memory', 'look up',
    'my facts', 'what did i say',
]

KNOWLEDGE_KEYWORDS = [
    'wikipedia', 'consult wikipedia', 'wiki ',
    'explain in depth', 'explain in detail', 'in depth',
    'how does ', 'how do ', 'what is the difference between',
    'compare and contrast', 'tell me about ',
    'what is a ', 'what are ', 'what causes ',
    'history of ', 'science behind', 'mechanism of ',
]

# Deterministic backstop for the Tier-4 LLM knowledge-search decision. Every
# KNOWLEDGE_KEYWORDS entry and every prompt example is a QUESTION or explicit
# explain-request ("how does X work", "what is Y", "explain Z", "compare A and
# B", "tell me about W", "consult Wikipedia"). The LLM trigger over-fires
# needs_knowledge on collaborative-task follow-ups that merely NAME a concept
# ("Yeah and I'll need it in ATS-friendly format") — the user is directing work
# on their OWN materials, not asking an encyclopedic question. Routing those
# into the agentic knowledge loop cost a ~2-minute hang (2026-07-24, kimi-3
# slow-round incident). Before honoring needs_knowledge_search we require this
# signal; absent it the turn falls back to normal generation (which still
# carries wiki-RAG context). Deliberately FAIL OPEN — a generous match trusts
# the LLM; only a genuinely signal-less query (a bare task statement) is
# suppressed.
KNOWLEDGE_QUESTION_SIGNAL = re.compile(
    r'\?'                                              # any question mark
    r'|\b(?:what|whats|how|why|who|whom|when|where|which)\b'
    r'|\bexplain\b|\bdefine\b|\bdefinition\b|\bcompare\b|\bcontrast\b'
    r'|\btell me about\b|\bdifference between\b|\bhistory of\b'
    r'|\bconsult\b|\blook (?:it|this|that|them|these) up\b'
    r'|\bin[- ]depth\b|\bmechanism of\b|\bscience behind\b',
    re.IGNORECASE,
)

# File / saved-document RETRIEVAL intent (distinct from doc *generation* in
# Tier 3). These route to the agentic loop so the file_read / file_list /
# get_full_document tools are actually offered. Without them, requests like
# "pull and print the full document" fall through to enhanced mode — where no
# tools exist — and the model confabulates "I don't have file access". The
# tools themselves are wired and available; only the gate was missing them.
#
# NOTE: literal substrings alone proved too brittle — real phrasings inflect
# verbs and separate words ("pulling up and printing the document"), use
# pronouns ("pull it"), or assert the capability ("you have the tool"). The
# FILE_ACCESS_PATTERNS regexes below are the robust layer; the literal list
# stays as a cheap fast-path for noun-only phrases that have no retrieval verb.
FILE_ACCESS_KEYWORDS = [
    'full document', 'the full doc', 'whole document', 'entire document',
    'the saved file', 'the saved document', 'that document you', 'document you wrote',
    'document you saved', 'file you wrote', 'file you saved', 'file contents',
]

# Robust regex fallbacks — tolerate inflection, intervening words, and
# capability assertions. Validated against the actual failing transcripts.
FILE_ACCESS_PATTERNS = [
    # retrieval verb (any inflection) + file/doc/note object within a few words:
    # "read the file", "pulling up and printing the document", "show me the doc"
    re.compile(
        r'\b(?:pull|print|fetch|open|read|show|display|retriev|grab|load|cat|view|render|reprint|output|spit|bring)\w*'
        r'(?:\W+\w+){0,6}?\W+'
        r'(?:files?|docs?|documents?|notes?|pdfs?|markdown)\b',
        re.IGNORECASE,
    ),
    # explicit assertion that a tool exists / should be used:
    # "you have the tool", "use the file_read tool", "with your file access"
    re.compile(
        r'\b(?:you\s+(?:have|do\s+have|already\s+have|still\s+have|got)|use|using|via|invoke|call|with\s+your)\b'
        r'(?:\W+\w+){0,4}?\W+'
        r'(?:tools?|file[_\s-]?read|file\s+access|file[_\s-]?tool)\b',
        re.IGNORECASE,
    ),
    # explicit tool / capability name anywhere
    re.compile(r'\bfile[_\s-]?read\b|\bfile\s+access\b', re.IGNORECASE),
]

# Pronoun retrieval ("pull it", "print that", "grab it up") — ambiguous on its
# own ("pull it together"), so this only counts as file access when the PREVIOUS
# turn was file/document themed (see the continuation handling below).
FILE_RETRIEVAL_PRONOUN_PATTERN = re.compile(
    r'\b(?:pull|print|fetch|grab|open|show|display|load|retriev|render|reprint|spit|bring'
    r'|check|review|read|inspect|verify|examine|look)\w*'
    r'(?:\s+at)?\s+(?:it|that|this|them|those)(?:\s+(?:up|out|here|over|in|now))?\b',
    re.IGNORECASE,
)

# Markers that a previous turn was about a saved file/document — used to
# disambiguate the pronoun-retrieval pattern above.
# 'document' and 'file' moved to the word-bounded regex below (2026-08-27):
# as substrings they matched "documented medical circumstances",
# "documentation", "filed after the deadline", and even "profile" — for a
# user whose life admin constantly involves medical documentation, the
# prior-turn "file context" test was effectively always true.
FILE_DOC_CONTEXT_WORDS = (
    'doc ', ' doc.', '.md', 'markdown', 'pdf', '.txt',
    'saved', 'on disk', 'implementation plan', 'file_read', 'file access',
    'reconstruct', 'print',
    # Repo/project vocabulary (2026-08-22): "Pushed yesterday's work, docs are
    # updated — check it out" had NO context hit ('doc ' is not a substring of
    # "docs are"), so the pronoun-retrieval continuation never routed to tools.
    'repo', 'repositor', 'commit', 'codebase', 'docs', 'pushed',
)

# Word-bounded file/document markers: matches "the document"/"files" but not
# "documented"/"documentation"/"filed"/"profile".
FILE_DOC_CONTEXT_WORD_RE = re.compile(r"\b(?:documents?|files?)\b", re.IGNORECASE)

# Markers that the model OFFERED to read/pull a file last turn ("Want me to pull
# that up?"). A bare affirmation ("yes", "do it") after one of these routes to
# tools — this is what makes the enhanced-mode honesty offer actually get
# carried out on the follow-up turn.
FILE_OFFER_MARKERS = (
    'want me to pull', 'want me to read', 'want me to grab', 'want me to print',
    'want me to open', 'want me to fetch', 'want me to retrieve',
    'pull that up', 'pull it up', 'i can pull', 'i can read', 'i can print',
    'shall i pull', 'should i pull',
)

RECALL_SIGNAL_WORDS = [
    'what', 'when', 'where', 'who', 'how', 'why',
    'tell me', 'remind', 'remember', 'know about',
    'recall', 'anything about', 'details on',
]

SEARCH_SIGNAL_WORDS = {
    'search', 'look', 'find', 'news', 'latest', 'current',
    'today', 'recent', '2026', '2025', 'what is', 'who is',
    'how does', 'tell me about',
}

CASUAL_STARTERS = (
    'nice', 'thanks', 'thank you', 'cool', 'great', 'awesome', 'got it',
    'ok ', 'okay', 'yeah', 'yes', 'no ', 'nope', 'nah', 'haha', 'lol',
    'true', 'fair', 'same', 'right', 'exactly', 'for sure', 'bet', 'word',
)

FILLER_WORDS = [
    'yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope',
    'thanks', 'thank', 'you', 'lol', 'haha', 'true', 'right', 'fair', 'same',
]

CONTINUATION_PHRASES = (
    'try again', 'try that again', 'one more', 'do it',
    'go ahead', 'yes please', 'please do', 'go for it',
    'run it', "let's go", 'sure', 'yep',
    'yes', 'yeah', 'do that',
)


def _compile_keyword_matcher(keywords):
    """Left-word-boundary matching for bare-word keywords, substring for the rest.

    'solve' must not match "resolution"/"unresolved" — a memory-ingest paste
    titled "crisis resolution" keyword-routed to a 49s computation+tools loop
    on 2026-08-28 (same substring class as 'document'⊂"documented", fixed
    2026-08-27). Only the LEFT boundary is enforced so 'solve' still matches
    "solves"/"solving"; keywords containing spaces, apostrophes, or
    trailing-space sentinels keep their original substring semantics
    ('go to http' must still match "go to https://...").
    """
    word_pats = []
    substrings = []
    for kw in keywords:
        if re.fullmatch(r"[a-z][a-z0-9_]*", kw):
            word_pats.append(re.compile(rf"\b{re.escape(kw)}"))
        else:
            substrings.append(kw)

    def _hit(lower_text: str) -> bool:
        return (
            any(p.search(lower_text) for p in word_pats)
            or any(k in lower_text for k in substrings)
        )

    return _hit


_COMPUTATION_HIT = _compile_keyword_matcher(COMPUTATION_KEYWORDS)
_WEB_SEARCH_HIT = _compile_keyword_matcher(WEB_SEARCH_KEYWORDS)
_TOOL_HIT = _compile_keyword_matcher(TOOL_KEYWORDS)
_MEMORY_HIT = _compile_keyword_matcher(MEMORY_KEYWORDS)
_KNOWLEDGE_HIT = _compile_keyword_matcher(KNOWLEDGE_KEYWORDS)
# Tier-2's recall-signal test used bare substring — 'how' ⊂ "sHOWer" fired
# memory mode on "I am in bathroom with shower running…" (live 2026-08-29;
# 4th occurrence of the substring class after 'solve'⊂"resolution",
# 'document'⊂"documented", 'cat'⊂"catalog").
_RECALL_SIGNAL_HIT = _compile_keyword_matcher(RECALL_SIGNAL_WORDS)

# Tier-2 entity+recall arm is only trusted on short messages (2026-08-29:
# a lyrics paste with an embedded '?' ran a 151s memory loop).
TIER2_ENTITY_MAX_WORDS = int(os.getenv("TIER2_ENTITY_MAX_WORDS", "60"))


def _entity_mention_is_proper(user_text: str, entity_id: str) -> bool:
    """A Tier-2 entity anchor must look like a NAME in the user's text:
    TitleCase at the mention site, or an inherently multi-word entity id.
    Generic-word graph nodes ('normal', 'tie', 'bed') otherwise anchor the
    memory arm on ordinary prose (live 2026-08-29). Under-fires for names
    typed lowercase — those recall queries carry Tier-1 keywords anyway."""
    if "_" in entity_id or " " in entity_id:
        return True
    head = entity_id.split("_")[0]
    if not head:
        return False
    pat = r"\b" + re.escape(head[0].upper() + head[1:]) + r"\b"
    return re.search(pat, user_text) is not None


# Deterministic backstop for the Tier-4 LLM WEB trigger (2026-08-29): asked
# "is this expected, or potentially useful info for Monday" (deictic — his
# reactivity, his appointment), the trigger LLM proposed three searches of
# the shape "<generic noun> for Monday August 31 2026" and burned Tavily
# credits on Keene-NH construction news. A term whose content is ONLY
# time/date tokens plus generic filler names no searchable topic — if every
# proposed term is like that, the search cannot be about anything.
_TEMPORAL_GENERIC_TOKENS = frozenset({
    # time/date
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "today", "tomorrow", "yesterday", "week", "weekend", "month",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    # generic filler the trigger pads terms with
    "expected", "events", "useful", "information", "info", "news",
    "updates", "update", "things", "happening", "schedule", "plans",
    "upcoming", "potentially", "general", "relevant",
    # stopwords
    "for", "on", "in", "the", "a", "an", "of", "and", "or", "to", "about",
    "this", "that", "is", "are", "what",
})


def _terms_are_temporal_generic(terms) -> bool:
    """True when EVERY proposed search term reduces to time words + generic
    filler (no content-bearing token survives). Never fires on real topics:
    'GT drop date August 2026' keeps 'gt'/'drop'/'date'."""
    if not terms:
        return False
    for term in terms:
        tokens = re.findall(r"[a-zA-Z]+", str(term).lower())
        if any(t not in _TEMPORAL_GENERIC_TOKENS and not t.isdigit()
               for t in tokens):
            return False
    return True

# The email-action arms fire on an address + action-verb co-occurrence, which
# is trivially true for any long paste carrying an email SIGNATURE plus
# narration like "Emailed re: ..." (live 2026-08-28 ingest turn). A real send
# request is either terse ("email Morgan the update") or opens with a send
# imperative ("Send this to Morgan@...: <draft>"). Head-anchored (the 08-27
# unanchored-"can you" lesson); "can/could you email..." within the length
# cap passes via the word count.
EMAIL_ACTION_MAX_WORDS = 40
_EMAIL_COMMAND_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|cool|yeah|yes|sure|right|so|and|"
    r"now|then|also|well|hey|please)[,\s]+){0,3}"
    r"(?:please\s+)?(?:send|e-?mail|draft|write|compose|forward|shoot|fire\s+off)\b",
    re.IGNORECASE,
)

# A genuine continuation/affirmation is terse ("yes please", "ok try again").
# Longer messages that merely CONTAIN one of the phrases above are new
# statements, not continuations — "Yeah they seem like the worst drug to get
# addicted to" (11 words) matched 'yeah' as a substring and rode the
# continuation override into a pointless 60s agentic loop (2026-07-15).
CONTINUATION_MAX_WORDS = 6

# Fallback tool-intent inference for LEGACY corpus entries that predate the
# stored `response_mode` field. Word-boundary anchored and deliberately
# narrow: the old bare-substring list matched conversational text ('issues'
# inside "my sleep issues", 'loc' inside "local") and flagged emotional turns
# as agentic-intent. New entries carry response_mode and never reach this.
_PREV_AGENTIC_QUERY_PATTERN = re.compile(
    r'\b(?:search|look\s+up|github|git\s+stats?|pull\s+requests?|'
    r'open\s+issues|closed\s+issues|commits?|lines\s+of\s+code|'
    r'calculate|compute|wolfram)\b'
)

EXPLICIT_SEARCH_KEYWORDS = [
    'search', 'look up', 'fetch', 'check out', 'go to', 'visit', 'pull up',
]

VETO_INTENTS = {'meta_conversational', 'casual_social'}

# Elevated-tone markers for the corroborated emotional-support veto. Tone
# arrives in mixed encodings depending on the caller ("light_support" /
# "elevated_support" / "crisis_support" enum values, or "CrisisLevel.CONCERN"
# style strings) — match on uppercase substrings.
_ELEVATED_TONE_MARKERS = (
    "CONCERN", "MEDIUM", "HIGH",
    "LIGHT_SUPPORT", "ELEVATED_SUPPORT", "CRISIS_SUPPORT",
)


def _tone_is_elevated(tone_level) -> bool:
    """True when a tone/crisis level string indicates CONCERN or above."""
    if not tone_level:
        return False
    _t = str(getattr(tone_level, "value", tone_level)).upper()
    if "CONVERSATIONAL" in _t:
        return False
    return any(marker in _t for marker in _ELEVATED_TONE_MARKERS)


# Acute crisis markers: MEDIUM/HIGH (and their support-mode encodings), i.e.
# strictly above the CONCERN/light_support tier the sticky floor can hold for
# a whole session.
_ACUTE_TONE_MARKERS = ("MEDIUM", "HIGH", "ELEVATED_SUPPORT", "CRISIS_SUPPORT")


def _tone_is_acute(tone_level) -> bool:
    """True for MEDIUM/HIGH crisis levels — the tier where a confident
    retrieval intent must no longer defeat the safety veto (2026-08-21: a
    MEDIUM suicide-mention turn regex-classified temporal_recall@0.85 ran a
    49s agentic loop and was abandoned by the user)."""
    if not tone_level:
        return False
    _t = str(getattr(tone_level, "value", tone_level)).upper()
    return any(marker in _t for marker in _ACUTE_TONE_MARKERS)


def _build_recent_context(corpus_manager, max_turns: int = 2) -> Optional[str]:
    """Build a compact recent-conversation digest for the web-search trigger.

    Elliptical follow-ups ("check the news", "any updates on that") carry no
    topic of their own — the term-generating LLM in the trigger needs the prior
    turns to resolve what "the news" refers to. Without this it defaults to
    generic/world-news terms. Returns chronological (oldest→newest) text, or
    None when there is no corpus_manager / no history.
    """
    if corpus_manager is None:
        return None
    try:
        recent = corpus_manager.get_recent_memories(max_turns)
    except Exception as e:
        logger.debug(f"[Agentic Gate] recent-context build failed (non-fatal): {e}")
        return None
    if not recent:
        return None
    from core.agentic.formatters import clip_text
    lines: List[str] = []
    for mem in reversed(recent):  # get_recent_memories is newest-first
        q = (mem.get('query', '') or '').strip()
        r = (mem.get('response', '') or '').strip()
        if q:
            lines.append(f"User: {clip_text(q, 200)}")
        if r:
            lines.append(f"Assistant: {clip_text(r, 300)}")
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

async def evaluate_agentic_gate(
    user_text: str,
    entity_resolver=None,
    model_manager=None,
    corpus_manager=None,
    intent_info=None,
) -> AgenticDecision:
    """Evaluate whether a query should be routed to the agentic search loop.

    4-tier decision process:
        Tier 1: Keyword heuristics (instant, no LLM)
        Tier 2: Entity match from knowledge graph (instant)
        Tier 3: Document generation / self-note intent detection (instant)
        Tier 4: LLM fallback via web search trigger (async)
    Plus: casual skip filter, continuation override, intent-based veto.

    Args:
        user_text: The raw user message.
        entity_resolver: EntityResolver instance for graph entity matching,
            or None to skip Tier 2.
        model_manager: ModelManager instance for LLM fallback, or None to
            skip Tier 4.
        corpus_manager: CorpusManager for checking previous turn (continuation
            detection), or None to skip.
        intent_info: Intent classification result (dict or object with
            intent_type and confidence), or None.

    Returns:
        AgenticDecision with routing information.
    """
    _lower = user_text.lower().strip()
    _words = _lower.split()
    _has_url = 'http://' in _lower or 'https://' in _lower

    # ── Deferred-request affirmation (2026-08-21) ────────────────────
    # If the PREVIOUS turn tone-deferred a request-shaped query (the model
    # acknowledged and offered), a terse affirmation now re-runs the ORIGINAL
    # query veto-exempt. The slot is one-shot: consumed on every call, so
    # only the immediately-following turn can affirm — anything else drops it.
    _deferred_query = _consume_deferred_request()
    if _deferred_query:
        _is_affirm = (
            len(_words) <= CONTINUATION_MAX_WORDS
            and any(p in _lower for p in CONTINUATION_PHRASES)
        )
        if _is_affirm:
            logger.info(
                f"[Agentic Gate] Affirmation after tone-deferral — re-running "
                f"deferred request veto-exempt: '{_deferred_query[:60]}'"
            )
            _redo = await evaluate_agentic_gate(
                user_text=_deferred_query,
                entity_resolver=entity_resolver,
                model_manager=model_manager,
                corpus_manager=corpus_manager,
                intent_info=None,  # explicit user consent — no re-veto
            )
            _redo.veto_exempt = True
            if not _redo.should_trigger:
                # The original query triggered once (that's why it was
                # vetoed); if re-evaluation flakes, honor the user's consent.
                _redo.should_trigger = True
                if not _redo.modes:
                    _redo.modes = ["web_search", "memory"]
            _redo.reason = (
                f"deferred-request affirmation: '{_deferred_query[:80]}' "
                f"({_redo.reason or 'triggered'})"
            )
            return _redo

    # ── Insight consent-offer affirmation (2026-08-23) ───────────────
    # If the PREVIOUS turn armed the one-shot insight offer (an insight-shaped
    # statement at non-elevated tone; handlers injected "may I check this
    # against your history?"), a terse affirmation now runs the assessment on
    # the ORIGINAL statement, veto-exempt. Slot is one-shot: consumed on every
    # call — anything but an immediate affirmation drops it permanently
    # (a decline is never re-offered; anti-excavation).
    _offered_insight = _consume_insight_offer()
    if _offered_insight:
        _is_affirm = (
            len(_words) <= CONTINUATION_MAX_WORDS
            and any(p in _lower for p in CONTINUATION_PHRASES)
        )
        if _is_affirm:
            logger.info(
                f"[Agentic Gate] Insight-offer affirmed — assessing: "
                f"'{_offered_insight[:60]}'"
            )
            return AgenticDecision(
                should_trigger=True,
                modes=["insight"],
                reason=f"insight-offer affirmation: '{_offered_insight[:80]}'",
                veto_exempt=True,
                insight_intent={
                    "kind": "insight_assessment",
                    "theme": _offered_insight,
                    "wants_document": False,
                    "raw_query": _offered_insight,
                },
            )

    modes: List[str] = []
    search_terms: List[str] = []
    matched_entities: Set[str] = set()
    doc_gen_intent = None
    self_note_intent = None

    needs_computation = False
    needs_memory = False
    needs_knowledge = False
    needs_web_search = False
    needs_tools = False

    # ── Tier 1: Keyword heuristics (instant, no LLM) ─────────────────
    needs_computation = _COMPUTATION_HIT(_lower)

    if _has_url or _WEB_SEARCH_HIT(_lower):
        needs_web_search = True
        logger.debug("[Agentic Gate] Tier 1: explicit web search/URL keyword detected")

    if _TOOL_HIT(_lower):
        needs_tools = True

    # Email arms only run for a plausible SEND request: terse message or a
    # head-anchored send imperative. A long paste containing addresses in a
    # signature plus incidental "email"/"write" is narration, not a command.
    _email_action_plausible = (
        len(_words) <= EMAIL_ACTION_MAX_WORDS
        or bool(_EMAIL_COMMAND_RE.search(user_text.strip()))
    )

    # Email address + action verb → internet action intent
    if not needs_tools and _email_action_plausible:
        _has_email_addr = bool(_re_gate.search(r'\S+@\S+\.\S+', user_text))
        if _has_email_addr and any(w in _lower for w in ('email', 'send', 'message', 'write', 'mail', 'contact')):
            needs_tools = True

    # Email-by-name patterns: "email Meagan", "send Meagan an email", "email her about X"
    if not needs_tools and _email_action_plausible:
        # "email <name>" at start of message
        if _re_gate2.match(r'^email\s+[a-z]', _lower):
            needs_tools = True
        # "send <name> an email/message" or "send an email to <name>"
        elif _re_gate2.search(r'\bsend\b.*\b(email|message)\b', _lower):
            needs_tools = True
        # "email" as a verb anywhere + action-like context
        elif 'email' in _lower and any(w in _lower for w in ('send', 'draft', 'write', 'compose', 'fire off')):
            needs_tools = True
        if needs_tools:
            logger.debug("[Agentic Gate] Tier 1: email-by-name intent detected")

    # File / saved-document retrieval intent → route to agentic so file_read /
    # file_list / get_full_document are offered. Literal fast-path + robust regex.
    needs_files = (
        any(kw in _lower for kw in FILE_ACCESS_KEYWORDS)
        or any(p.search(_lower) for p in FILE_ACCESS_PATTERNS)
    )
    if needs_files:
        needs_tools = True
        logger.debug("[Agentic Gate] Tier 1: file/document access intent detected")

    needs_memory = _MEMORY_HIT(_lower)

    # Knowledge keywords require 4+ words and no computation trigger
    if len(_words) >= 4 and not needs_computation:
        needs_knowledge = _KNOWLEDGE_HIT(_lower)

    # ── Tier 2: Entity match (instant, no LLM) ───────────────────────
    # 2026-08-29 live-session hardening (three agentic loops on emotional
    # turns, 44-151s each):
    #   - LENGTH CAP: on a paste-sized message both halves of the test are
    #     unreliable — a '?' inside pasted lyrics ("am I just beaten so ?")
    #     is not a user question, and lyric words resolve to graph nodes
    #     ('tie', 'bed'). Genuine long recall requests hit Tier-1 keywords.
    #   - PROPER-MENTION FILTER: a matched entity must appear TitleCase in
    #     the raw text (or be multi-word) — generic-word graph nodes
    #     ('normal', 'tie', 'bed') fired this arm on therapy-processing
    #     replies. Same under-fire doctrine as extract_rare_proper_nouns.
    if (not needs_computation and not needs_memory and entity_resolver is not None
            and len(_words) <= TIER2_ENTITY_MAX_WORDS):
        try:
            from memory.graph_utils import extract_graph_entities
            matched_entities = extract_graph_entities(user_text, entity_resolver)
            matched_entities.discard("user")
            matched_entities = {
                e for e in matched_entities
                if _entity_mention_is_proper(user_text, e)
            }
            if matched_entities:
                _has_recall_signal = (
                    '?' in user_text
                    or _RECALL_SIGNAL_HIT(_lower)
                )
                if _has_recall_signal:
                    needs_memory = True
                    logger.debug(
                        f"[Agentic Gate] Tier 2: entity {matched_entities} + recall signal"
                    )
                else:
                    logger.debug(
                        f"[Agentic Gate] Tier 2: entity match {matched_entities} "
                        f"but no recall signal — skipping"
                    )
        except Exception as e:
            logger.debug(f"[Agentic Gate] Entity match check failed (non-fatal): {e}")

    # ── Casual skip filter ────────────────────────────────────────────
    _has_search_signal = (
        any(w in _lower for w in SEARCH_SIGNAL_WORDS)
        or '?' in user_text
        or _has_url
    )
    _skip_patterns = [
        len(_words) < 5 and not _has_search_signal,
        len(_words) < 10 and not _has_search_signal,
        len(_words) < 12 and not _has_search_signal and _lower.startswith(CASUAL_STARTERS),
        all(w in FILLER_WORDS for w in _words),
    ]

    # ── Context-aware continuation override ───────────────────────────
    _prev_was_agentic = False
    _request_continuation = False
    if any(_skip_patterns):
        # Terse affirmation ("yes please", "run it") — the classic shape.
        # OR (2026-08-22) an affirmative DIRECTIVE: "Sure check out the
        # learning loop stuff" is 7 words — one over the cap — and carries a
        # noun phrase no pronoun pattern can see, but it is request-shaped
        # with an affirmation opener. After a stored-agentic turn that is a
        # continuation, full stop. The 07-15 benzo-turn guard holds: "Yeah
        # they seem like the worst drug..." is not request-shaped.
        _request_continuation = (
            len(_words) <= 12 and _is_request_shaped(user_text)
        )
        _is_continuation = (
            len(_words) <= CONTINUATION_MAX_WORDS
            and any(p in _lower for p in CONTINUATION_PHRASES)
        ) or _request_continuation
        if _is_continuation and corpus_manager is not None:
            try:
                _recent = corpus_manager.get_recent_memories(2)
                for _prev in _recent:
                    _prev_query = (_prev.get('query', '') or '').lower()
                    _prev_response = (_prev.get('response', '') or '')[:800]
                    _prev_mode = (_prev.get('response_mode', '') or '').lower()
                    if _prev_mode:
                        # Ground truth: the stored mode of the previous turn.
                        _prev_had_signals = _prev_mode == 'agentic-search'
                    else:
                        # Legacy entry without response_mode — infer from query.
                        _prev_had_signals = bool(
                            _PREV_AGENTIC_QUERY_PATTERN.search(_prev_query)
                        )
                    _prev_mentioned_tools = any(w in _prev_response.lower() for w in (
                        'let me pull', 'let me grab', 'let me run',
                        'let me check', 'let me search', 'let me query',
                        "i'll hit", "i'll search",
                        'git_stats', 'github api', 'github tool',
                    ))
                    if _prev_had_signals or _prev_mentioned_tools:
                        _prev_was_agentic = True
                        if _request_continuation:
                            # An affirmative directive continuing an agentic
                            # session gets the tool loop directly — Tier 4's
                            # web-trigger LLM has no repo/file channel.
                            needs_tools = True
                        logger.debug(
                            f"[Agentic Gate] Continuation after agentic-intent turn — "
                            f"overriding casual skip (query_signals={_prev_had_signals}, "
                            f"response_tools={_prev_mentioned_tools}, "
                            f"request_continuation={_request_continuation})"
                        )
                        break
            except Exception as e:
                logger.debug(f"[Agentic Gate] Previous-turn check failed (non-fatal): {e}")

    # ── File retrieval continuation (pronoun or affirmation) ──────────
    # Two terse follow-up shapes route to tools when the prior turn was about a
    # saved file/document:
    #   • pronoun retrieval ("pull it", "print that") after any file/doc turn —
    #     catches "No. I mean can you pull it."
    #   • a bare affirmation ("yes", "do it") right after the model OFFERED to
    #     read/pull a file ("Want me to pull that up?") — makes the enhanced-mode
    #     honesty offer actually get carried out on the next turn.
    if not needs_files and corpus_manager is not None:
        # First-person self-reports ("I checked it out, kind of sucked") are
        # the user narrating, not requesting — the widened verb set
        # (check/review/read/look, 2026-08-22) would otherwise match them.
        _self_report = bool(re.search(
            r"\b(?:i|we)(?:'ve|'d|\s+(?:just|already|have|had|finally))?\s+"
            r"(?:check|review|read|look|open|pull)\w*\b", _lower))
        # Round 5 (2026-08-22): the pronoun requirement was the last brittle
        # link — "pull up the veto logic" is the canonical retrieval
        # imperative and has NO pronoun. A REQUEST-shaped message (imperative
        # retrieval verb, ack-prefix tolerant, self-reports excluded) counts
        # exactly like a pronoun retrieval; the prior-turn file/doc/repo
        # context gate below is what prevents over-fire either way.
        # Length cap (2026-08-27): retrieval continuations are terse. A long
        # pasted message can contain incidental matches for either shape
        # (an email's own "can you ..." matched request-shape and rode this
        # arm into a 106s tool loop).
        _is_pronoun_retrieval = (
            not _self_report
            and len(_words) <= REQUEST_CONTINUATION_MAX_WORDS
            and (
                bool(FILE_RETRIEVAL_PRONOUN_PATTERN.search(_lower))
                or _is_request_shaped(user_text)
            )
        )
        _is_affirmation = len(_words) <= CONTINUATION_MAX_WORDS and (
            any(p in _lower for p in CONTINUATION_PHRASES)
            or (bool(_words) and all(w in FILLER_WORDS for w in _words))
        )
        if _is_pronoun_retrieval or _is_affirmation:
            try:
                _recent = corpus_manager.get_recent_memories(2)
                for _prev in _recent:
                    _resp = (_prev.get('response', '') or '')[:800].lower()
                    _blob = (_prev.get('query', '') or '').lower() + ' ' + _resp
                    _prev_was_file = (
                        any(w in _blob for w in FILE_DOC_CONTEXT_WORDS)
                        or bool(FILE_DOC_CONTEXT_WORD_RE.search(_blob))
                    )
                    _prev_offered_file = any(o in _resp for o in FILE_OFFER_MARKERS)
                    # pronoun → needs file/doc context; affirmation → needs explicit offer
                    if ((_is_pronoun_retrieval and _prev_was_file)
                            or (_is_affirmation and _prev_offered_file)):
                        needs_files = True
                        needs_tools = True
                        logger.debug(
                            "[Agentic Gate] File retrieval continuation — routing to tools"
                        )
                        break
            except Exception as e:
                logger.debug(f"[Agentic Gate] File continuation check failed (non-fatal): {e}")

    # ── Insight / evidence-assembly requests (2026-08-23) ─────────────
    # Checked BEFORE Tier-3 doc-gen: a personal-theme document request
    # ("write a summary of my pattern with X for my therapist") must route to
    # the insight mode, not to web research — detect_document_intent treated
    # exactly that shape as a research topic. Always veto_exempt: an explicit
    # request works even mid-distress (therapist-doc framing is handled by
    # the synthesizer's elevated-tone tail, not by refusing the turn).
    try:
        from config.app_config import INSIGHT_MODE_ENABLED
        if INSIGHT_MODE_ENABLED:
            # lazy import: cycle (insight.detector imports gate at call time)
            from core.insight.detector import detect_insight_request
            _insight = detect_insight_request(user_text)
            if _insight:
                logger.info(
                    f"[Agentic Gate] Insight mode: {_insight.kind} "
                    f"theme='{_insight.theme[:60]}' doc={_insight.wants_document}"
                )
                return AgenticDecision(
                    should_trigger=True,
                    modes=["insight"],
                    reason=f"insight-mode: {_insight.kind}",
                    veto_exempt=True,
                    insight_intent=_insight.model_dump(),
                )
    except Exception as e:
        logger.warning(f"[Agentic Gate] Insight detection failed (non-fatal): {e}")

    # ── Tier 3: Document generation + self-note intent ────────────────
    try:
        from knowledge.document_generator import detect_document_intent
        doc_gen_intent = detect_document_intent(user_text)
        if doc_gen_intent:
            logger.warning(f"[Agentic Gate] Document generation detected: {doc_gen_intent}")
            needs_tools = True
    except Exception as e:
        logger.warning(f"[Agentic Gate] Document intent check failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        from knowledge.daemon_notes_manager import detect_self_note_intent
        self_note_intent = detect_self_note_intent(user_text)
        if self_note_intent:
            logger.warning(f"[Agentic Gate] Self-note detected: {self_note_intent}")
            needs_tools = True
    except Exception as e:
        logger.debug(f"[Agentic Gate] Self-note intent check failed: {e}")

    # ── Decision: skip, keyword trigger, or LLM fallback ─────────────
    should_trigger = False

    if (doc_gen_intent or self_note_intent or needs_computation or needs_memory
            or needs_knowledge or needs_web_search or needs_tools):
        should_trigger = True
    elif any(_skip_patterns) and not _prev_was_agentic:
        logger.debug("[Agentic Gate] Skipped — casual/short message")
        should_trigger = False
    else:
        # Tier 4: LLM fallback
        if model_manager is not None:
            try:
                from utils.web_search_trigger import analyze_for_web_search_llm
                # Carry the prior turns so elliptical follow-ups ("check the
                # news") resolve to topic-specific search terms instead of
                # generic ones.
                _recent_ctx = _build_recent_context(corpus_manager)
                trigger_decision = await analyze_for_web_search_llm(
                    query=user_text,
                    model_manager=model_manager,
                    conversation_context=_recent_ctx,
                )
                should_trigger = getattr(trigger_decision, 'should_search', False)
                search_terms = getattr(trigger_decision, 'search_terms', []) or []

                # Temporal-generic term guard (2026-08-29): if every proposed
                # term is time words + filler ("useful information for Monday
                # August 31 2026"), the LLM misread a personal schedule
                # reference as a news request — stand down instead of burning
                # searches on nothing.
                if should_trigger and search_terms and \
                        _terms_are_temporal_generic(search_terms):
                    logger.info(
                        "[Agentic Gate] LLM web trigger suppressed — all "
                        f"proposed terms are temporal-generic: {search_terms}")
                    should_trigger = False
                    search_terms = []

                if getattr(trigger_decision, 'needs_memory_search', False):
                    logger.debug("[Agentic Gate] LLM detected memory search intent")
                    should_trigger = True
                    needs_memory = True
                    search_terms = []
                elif getattr(trigger_decision, 'needs_knowledge_search', False):
                    # Deterministic backstop: honor knowledge search only when the
                    # query actually reads as a knowledge QUESTION. A collaborative-
                    # task follow-up ("Yeah and I'll need it in ATS format") has no
                    # such signal — suppress it to normal generation rather than
                    # spin up the agentic knowledge loop.
                    if KNOWLEDGE_QUESTION_SIGNAL.search(user_text):
                        if not should_trigger:
                            logger.debug("[Agentic Gate] LLM detected knowledge search intent")
                            should_trigger = True
                        needs_knowledge = True
                        search_terms = []
                    else:
                        logger.debug(
                            "[Agentic Gate] LLM knowledge-search suppressed — no "
                            "knowledge-question signal (collaborative-task/follow-up): "
                            f"{user_text[:60]!r}"
                        )
                elif getattr(trigger_decision, 'needs_document_generation', False):
                    logger.info("[Agentic Gate] LLM detected document generation intent")
                    should_trigger = True
                    needs_tools = True
                    search_terms = []
                    doc_gen_intent = {
                        "topic": getattr(trigger_decision, 'document_topic', '') or user_text,
                        "doc_type": getattr(trigger_decision, 'document_type', 'report') or 'report',
                        "focus": None,
                        # "conversation" = write up THIS conversation's content
                        # (handlers pass the transcript as source_material);
                        # "research"/None = research the topic externally.
                        "source": getattr(trigger_decision, 'document_source', '') or None,
                    }

                logger.debug(
                    f"[Agentic Gate] LLM trigger: should_search={should_trigger}, "
                    f"needs_memory={needs_memory}, needs_knowledge={needs_knowledge}, "
                    f"terms={search_terms}"
                )
            except Exception as e:
                logger.warning(f"[Agentic Gate] LLM trigger check failed: {e}")
                import traceback
                traceback.print_exc()
                should_trigger = False
                search_terms = []

    # Log triggered modes
    if should_trigger:
        triggered = []
        if needs_computation:
            triggered.append("computation")
        if needs_memory:
            triggered.append("memory")
        if needs_knowledge:
            triggered.append("knowledge")
        if needs_web_search:
            triggered.append("web_search")
        if needs_tools:
            triggered.append("tools")
        if triggered:
            logger.debug(f"[Agentic Gate] Triggered — modes: {', '.join(triggered)}")

    # ── Intent-veto exemption (explicit requests are never vetoed) ────
    _explicit_kw = any(kw in _lower for kw in EXPLICIT_SEARCH_KEYWORDS)
    _veto_exempt = (
        _explicit_kw or _has_url or needs_files
        or bool(doc_gen_intent) or bool(self_note_intent)
    )
    # Bare pasted link with NO request shape — the only exemption the acute
    # tone arm may pierce (see AgenticDecision.veto_exempt_url_only).
    _veto_exempt_url_only = _has_url and not (
        _explicit_kw or needs_files or bool(doc_gen_intent) or bool(self_note_intent)
    )

    # ── Build modes list ──────────────────────────────────────────────
    if needs_computation:
        modes.append("computation")
    if needs_memory:
        modes.append("memory")
    if needs_knowledge:
        modes.append("knowledge")
    if needs_web_search:
        modes.append("web_search")
    if needs_tools:
        modes.append("tools")

    # ── Compute skip_initial_search ───────────────────────────────────
    # Skip the blind Round-1 web search whenever we have no concrete search
    # terms to seed with. Without terms the controller would fall back to
    # searching the raw user message verbatim (filler words, pronouns, no
    # distilled intent) — almost always low quality, and it mislabelled a
    # casual message as a news query. With no terms, let the agentic loop
    # distill its own query and pick tools. This subsumes the old
    # (needs_web_search and not search_terms) clause.
    skip_initial = (
        needs_computation or needs_memory or needs_knowledge or needs_tools
        or not search_terms
    )

    # ── Build reason string ───────────────────────────────────────────
    if should_trigger:
        reason = f"triggered: {', '.join(modes) if modes else 'llm-fallback'}"
    elif any(_skip_patterns) and not _prev_was_agentic:
        reason = "casual/short message"
    else:
        reason = "no trigger"

    decision = AgenticDecision(
        should_trigger=should_trigger,
        modes=modes,
        search_terms=search_terms,
        matched_entities=matched_entities,
        doc_gen_intent=doc_gen_intent,
        self_note_intent=self_note_intent,
        skip_initial_search=skip_initial,
        reason=reason,
        veto_exempt=_veto_exempt,
        veto_exempt_url_only=_veto_exempt_url_only,
    )

    # Intent veto — applied here when intent_info was available at call time.
    # Callers that run the gate CONCURRENTLY with the context pipeline pass
    # intent_info=None and apply the veto post-hoc via apply_intent_veto().
    if intent_info is not None:
        decision = apply_intent_veto(decision, intent_info)

    return decision


# Interrogative/command shapes and explicit lookup cues. A query with any of
# these is info-seeking — the tone-statement veto below must never suppress it.
_INTERROGATIVE_OPENERS = (
    "what", "when", "where", "who", "why", "how", "which",
    "can you", "could you", "would you", "will you", "do you", "did ",
    "does ", "is ", "are ", "was ", "were ", "should", "tell me", "show me",
    "find ", "search", "look up", "remind me", "any idea", "help me",
)
_INFO_SEEKING_CUES = (
    "search", "look up", "google", "find out", "remember when", "what did",
    "recall", "look for", "look into", "research", "check the", "check my",
    # Confirmation-shaped lookups: "I would like to confirm it's this Friday
    # that's the drop date" has no "?", no interrogative opener, and no lookup
    # verb — it read as vent-shaped and the tone-veto killed the agentic gate
    # on a deadline question (2026-08-27).
    "confirm", "verify", "double-check", "double check", "check if",
    "check whether",
)
# Pronoun-split lookup commands: "look IT up", "pull THAT up". The contiguous
# "look up" cue missed them — "Look it up it's pretty funny" was vetoed AND
# learned as a no_search exemplar (2026-08-15).
_LOOKUP_CUE_RE = re.compile(
    r"\b(?:look|pull)\s+(?:(?:it|this|that|them|these|those)\s+)?up\b"
)


def _is_info_seeking(query: str) -> bool:
    """True when the query has question/command/lookup shape. Fail-open on
    empty input (no veto without evidence)."""
    q = (query or "").strip().lower()
    if not q:
        return True
    if "?" in q:
        return True
    if q.startswith(_INTERROGATIVE_OPENERS):
        return True
    if _LOOKUP_CUE_RE.search(q):
        return True
    return any(c in q for c in _INFO_SEEKING_CUES)


# First-person pronoun anywhere in the text — the marker that a non-info-seeking
# statement is about the USER'S OWN state (a vent) rather than the outside world.
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|id|me|my|mine|myself)\b")

# Epistemic-stance / reporting markers: first-person phrases that frame a claim
# about the OUTSIDE WORLD rather than express the speaker's own state. "I think
# people would be arrested…", "I mean it seems possible this was censored…",
# "i would say most people are unaware…" all carry a first-person pronoun but
# are opinions about third parties — during a sticky-CONCERN session (2026-08-18
# evening) seven such political/news statements passed the first-person-anywhere
# test, were vetoed, AND were taught as no_search exemplars (the 08-15 poisoning
# class through a new hole). Strip these markers first; vent shape then requires
# a first-person pronoun in what REMAINS ("I think I'm getting sick" stays a
# vent via the second "I'm"; "I think it's censored" does not).
_EPISTEMIC_MARKER_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:mean|think|thought|guess|suppose|bet|believe|reckon|figure|assume|"
    r"heard|read|saw|watched|imagine|swear|doubt)"
    r"|i\s+(?:do\s+not|don'?t)\s+(?:think|believe|know|get|understand|buy)"
    r"|i\s*(?:'d|\s+would)\s+(?:say|think|guess|bet|imagine|assume)"
    r"|i\s*(?:'m|\s+am)\s+(?:sure|positive|certain|pretty\s+sure)"
    r"|i\s*(?:'ve|\s+have)\s+(?:heard|read|seen)"
    r"|idk|imo|imho|afaik|iirc"
    r")\b"
)


def strip_epistemic_markers(text: str) -> str:
    """Remove first-person epistemic/reporting markers ("i think", "i mean",
    "i would say", "i heard", …) so first-person checks see only SUBSTANTIVE
    self-reference. Shared doctrine — utils.web_search_trigger's
    personal-state check imports this (lazily) too."""
    return _EPISTEMIC_MARKER_RE.sub(" ", (text or "").lower())


# Request shape (2026-08-21): an imperative ("check the docs for what
# changed", "pull up the logs") or a second-person ask ("can you run it")
# is a REQUEST even though it has no "?" / interrogative opener / lookup cue,
# so _is_info_seeking misses it and a tone arm can veto it. Such a veto
# should be a visible DEFERRAL (acknowledge + offer), never a silent decline
# — and never a no_search teaching event. The (?!,) guard keeps discourse
# markers ("Look, I'm just tired") out: comma after the verb = not a command.
# BOTH branches are head-anchored (2026-08-27): the second-person branch used
# to match "\bcan you\b" ANYWHERE, so a 700-word pasted email containing
# "...can you point me to the right process?" (a question addressed to the
# email's RECIPIENT, not to Daemon) counted as request-shaped and routed a
# status-update turn into a 106s agentic loop. A request to Daemon leads the
# message; "can you" buried mid-paste is quoted content.
_REQUEST_SHAPED_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|cool|yeah|yes|sure|right|so|and|now|then|also|well|hey)[,\s]+){0,3}"
    r"(?:(?:please\s+)?(?:check|look|pull|show|run|search|find|read|open|list|"
    r"verify|fetch|grab|review|summarize|summarise|scan|test|compare)\b(?!,)"
    r"|(?:please\s+)?(?:can|could|would|will)\s+you\b)",
    re.IGNORECASE,
)

# A retrieval continuation is TERSE by nature ("pull up the veto logic",
# "check it out now"). Anything longer is a substantive message that should
# route through the normal tiers — a pasted email that happens to open with
# an ack word must never ride the continuation shortcut.
REQUEST_CONTINUATION_MAX_WORDS = 30


def _is_request_shaped(text: str) -> bool:
    return bool(_REQUEST_SHAPED_RE.search((text or "").strip()))


# One-shot cross-turn slot for a tone-deferred request. Armed by
# apply_intent_veto, consumed (read + cleared) at the TOP of the next
# evaluate_agentic_gate call — so exactly the immediately-following turn can
# affirm it. Module-level state is fine here: single-user app, same pattern
# as the module embed caches.
_DEFERRED_REQUEST_SLOT: Dict[str, str] = {}


def _arm_deferred_request(query: str) -> None:
    _DEFERRED_REQUEST_SLOT.clear()
    _DEFERRED_REQUEST_SLOT["query"] = query


def _consume_deferred_request() -> Optional[str]:
    q = _DEFERRED_REQUEST_SLOT.get("query")
    _DEFERRED_REQUEST_SLOT.clear()
    return q


# One-shot cross-turn slot for the insight consent offer (2026-08-23).
# Armed by maybe_arm_insight_offer when the user makes an insight-SHAPED
# statement at non-elevated tone (handlers then inject a one-sentence
# "want me to check this against your history?" offer); consumed at the top
# of the next evaluate_agentic_gate call. Capped at ONE offer per session
# (in-memory counter — a restart forgetting a decline is the cheapest
# failure mode). A decline is never re-offered: anti-excavation.
_INSIGHT_OFFER_SLOT: Dict[str, str] = {}
_INSIGHT_OFFERS_THIS_SESSION: int = 0


def maybe_arm_insight_offer(query: str, tone_level: Optional[str] = None) -> bool:
    """Arm the one-shot insight consent offer if the query is an insight-shaped
    first-person statement, tone is non-elevated, and the per-session offer
    budget (1) is unspent. Returns True when armed (handlers inject the offer
    note only then)."""
    global _INSIGHT_OFFERS_THIS_SESSION
    try:
        from config.app_config import INSIGHT_MODE_ENABLED, INSIGHT_OFFER_ENABLED
        if not (INSIGHT_MODE_ENABLED and INSIGHT_OFFER_ENABLED):
            return False
    except Exception:
        return False
    if _INSIGHT_OFFERS_THIS_SESSION >= 1:
        return False
    if _tone_is_elevated(tone_level) or _tone_is_acute(tone_level):
        return False
    try:
        from core.insight.detector import detect_insight_statement
        if not detect_insight_statement(query):
            return False
    except Exception:
        return False
    _INSIGHT_OFFER_SLOT.clear()
    _INSIGHT_OFFER_SLOT["statement"] = query.strip()
    _INSIGHT_OFFERS_THIS_SESSION += 1
    logger.info(f"[Agentic Gate] Insight offer armed: '{query.strip()[:60]}'")
    return True


def _consume_insight_offer() -> Optional[str]:
    s = _INSIGHT_OFFER_SLOT.get("statement")
    _INSIGHT_OFFER_SLOT.clear()
    return s


def _reset_insight_offer_state() -> None:
    """Test helper: clear the slot and the per-session offer counter."""
    global _INSIGHT_OFFERS_THIS_SESSION
    _INSIGHT_OFFER_SLOT.clear()
    _INSIGHT_OFFERS_THIS_SESSION = 0


def _is_vent_shaped(query: str) -> bool:
    """True for a first-person, non-info-seeking statement — the shape of the
    emotional vents the tone-corroborated veto exists for.

    Third-party/news statements are deliberately NOT vent-shaped: mid-distress
    the sticky tone floor keeps every turn at CONCERN, and a topic shift to the
    outside world ("The president 'declared' the strait of hormuz to be us
    land lmfao") is a searchable share, not a vent — the old any-statement test
    vetoed it against the LLM trigger's should_search=0.8 (2026-08-15).
    First-person is checked ANYWHERE, not just as opener: real vents open with
    "Yeah...", "Ugh...", "It feels like..." as often as with "I". But an
    epistemic marker ("I think/I mean/i would say …") is NOT self-reference —
    it frames a claim about the world (2026-08-21 narrowing; see
    _EPISTEMIC_MARKER_RE)."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if _is_info_seeking(q):
        return False
    return bool(_FIRST_PERSON_RE.search(strip_epistemic_markers(q)))


# Confident retrieval-flavored intents are respected even on elevated-tone
# statements — the user is asking the system to go get something.
_RETRIEVAL_INTENTS = {
    "factual_recall", "temporal_recall", "technical_help", "project_work",
    "meta_conversational",
}


def apply_intent_veto(decision: AgenticDecision, intent_info, tone_level=None,
                      query: str = None) -> AgenticDecision:
    """Apply the intent-classifier veto to a gate decision (idempotent).

    Extracted so the gate can run concurrently with the context pipeline:
    the veto needs the pipeline's intent classification, which isn't
    available when the gate is launched early. Explicit requests (search
    keywords, URL, file access, doc-gen, self-note — recorded on
    decision.veto_exempt) are never suppressed.

    tone_level (optional): the pipeline's detected tone/crisis level. An
    emotional_support intent alone can't veto — STM-refined intents cap at
    0.60, under the 0.75 veto floor — but when the tone detector
    INDEPENDENTLY reads the turn as CONCERN or above, the two weak signals
    corroborate and the gate stands down: an emotional vent mid-distress
    should never pay for a multi-round search loop ("when I was moaning and
    crying in bed my mom ignored me" ran a 22s agentic loop, 2026-07-25).

    query (optional): the raw user text. When tone is elevated and the query
    is VENT-SHAPED (first-person, no info-seeking shape — question mark,
    interrogative opener, lookup cue), the gate stands down REGARDLESS of
    intent — the intent classifier read every 2026-08-02 distress vent as
    general@0.00, so the emotional_support corroboration above was starved
    exactly when needed ("I am embarrassed for how I reacted earlier … I am
    so unhappy" ran a 31s agentic memory loop). Confident retrieval intents
    (≥0.75) still win. 2026-08-15: narrowed from ANY non-info-seeking
    statement to vent shape — the sticky tone floor holds CONCERN across a
    long distress session, and a third-party news share mid-session was
    vetoed against the LLM trigger's should_search=0.8. Tone corroborates a
    vent; it must not disable search for statements about the outside world.
    """
    if decision is None or not decision.should_trigger or intent_info is None:
        return decision
    if decision.veto_exempt:
        # Bare-URL exemption pierce (2026-08-28): a pasted link inside an
        # ACUTE (MEDIUM/HIGH) first-person vent is shared context, not a
        # fetch request — the tone arms below may stand the gate down.
        # Every other exemption source (explicit keywords, files, doc-gen,
        # self-note) remains absolute.
        _pierce = (
            getattr(decision, "veto_exempt_url_only", False)
            and _tone_is_acute(tone_level)
            and query and _is_vent_shaped(query)
        )
        if not _pierce:
            return decision
        logger.info(
            "[Agentic Gate] Bare-URL exemption pierced: acute tone + "
            "vent-shaped message — tone arms may veto"
        )
    _intent_type = (
        getattr(intent_info, 'intent_type', None)
        if not isinstance(intent_info, dict)
        else intent_info.get('intent_type')
    )
    _intent_conf = (
        getattr(intent_info, 'confidence', 0)
        if not isinstance(intent_info, dict)
        else intent_info.get('confidence', 0)
    )
    _type_val = getattr(_intent_type, 'value', str(_intent_type)) if _intent_type else ''
    # Explicit info-seeking shape (question mark, interrogative opener, lookup
    # cue incl. pronoun-split "look it up") escapes every TONE-based arm: the
    # user directly asking the system to go get something wins over tone
    # corroboration. 2026-08-18 14:25: "Look it up on Wikipedia" was Tier-1
    # triggered and then killed by the emotional_support arm — the 08-15
    # info-seeking escape existed only inside _is_vent_shaped, so it never
    # reached this arm. (Confident VETO_INTENTS below are intent-driven, not
    # tone-driven, and keep their semantics.)
    _info_seeking = bool(query) and _is_info_seeking(query)
    # Captured BEFORE the veto clears them: trigger-proposed search terms are
    # conflicting evidence for the no_search teacher below.
    _proposed_terms = bool(decision.search_terms)
    _veto_reason = None
    _tone_driven = False
    if _type_val in VETO_INTENTS and _intent_conf >= 0.75:
        _veto_reason = f"intent-veto: {_type_val}@{_intent_conf:.2f}"
    elif _tone_is_acute(tone_level) and query and not _info_seeking:
        # Acute ceiling (2026-08-21): at MEDIUM/HIGH a confident retrieval
        # intent no longer defeats the veto — a MEDIUM suicide-mention turn
        # regex-classified temporal_recall@0.85 paid for a 49s decision round
        # the user abandoned. Explicit info-seeking shape and veto_exempt
        # (handled above) still win.
        _veto_reason = (
            f"tone-veto: acute tone ({tone_level}) — statement mid-crisis, "
            f"intent {_type_val or 'unknown'}@{_intent_conf:.2f} overridden"
        )
        _tone_driven = True
    elif (
        _type_val == 'emotional_support'
        and _intent_conf >= 0.60
        and _tone_is_elevated(tone_level)
        and not _info_seeking
    ):
        _veto_reason = (
            f"intent-veto: emotional_support@{_intent_conf:.2f} + elevated tone ({tone_level})"
        )
        _tone_driven = True
    elif (
        _tone_is_elevated(tone_level)
        and query
        and _is_vent_shaped(query)
        and not (_type_val in _RETRIEVAL_INTENTS and _intent_conf >= 0.75)
    ):
        _veto_reason = (
            f"tone-veto: elevated tone ({tone_level}) + first-person vent-shaped statement"
        )
    if _veto_reason:
        logger.info(f"[Agentic Gate] VETOED by intent classifier: {_veto_reason}")
        decision.should_trigger = False
        decision.search_terms = []
        decision.reason = _veto_reason
        # Visible deferral (2026-08-21): a TONE arm declining a REQUEST-shaped
        # (non-vent) query must not fail silently — mark the decision so
        # handlers tell the model to acknowledge + offer, and arm the one-shot
        # slot so a terse affirmation next turn re-runs the original query
        # veto-exempt. Vent-shaped turns never get the offer (anti-excavation:
        # don't advertise suppressed tooling mid-distress).
        if (
            _tone_driven and query and _is_request_shaped(query)
            and not _is_vent_shaped(query)
        ):
            decision.deferred_request = query
            _arm_deferred_request(query)
            logger.info(
                "[Agentic Gate] Tone-vetoed a request-shaped query — deferral "
                "armed for next-turn affirmation"
            )
        # Deterministic confirmation that this query should NOT search →
        # teach the web-trigger's negative anchors (adaptive store), so the
        # semantic boost stops pushing this user's vent phrasing toward
        # search BEFORE the gate has to veto it (2026-08-02). Vent shape is
        # required for TEACHING regardless of which arm vetoed: an explicit
        # lookup or third-party statement must never become a no_search
        # anchor (2026-08-15: "Look it up it's pretty funny" was learned).
        # 2026-08-21: additionally require that the trigger pipeline proposed
        # NO search terms — proposed terms mean an independent channel judged
        # the turn searchable, and a veto under conflicting evidence must not
        # become a durable anchor (08-18: medication-withdrawal phrasing with
        # live "kavarin withdrawal" terms was taught; under a sticky-CONCERN
        # session the learning channel is one-sided, so bad no_search anchors
        # can never be offset by search_worthy ones).
        if (
            query and _tone_is_elevated(tone_level) and _is_vent_shaped(query)
            and not _is_request_shaped(query) and not _proposed_terms
        ):
            try:
                from utils.adaptive_exemplars import get_store
                get_store().record(
                    "web_search", "no_search", query, "gate_veto"
                )
            except Exception as e:
                logger.debug(f"[Agentic Gate] no_search learning skipped: {e}")
    return decision
