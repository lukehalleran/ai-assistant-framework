"""
# core/agentic/gate.py

Module Contract
- Purpose: 4-tier agentic gate — decides whether a user query should be
  routed to the agentic search loop (tools, web, memory, knowledge, etc.)
  or fall through to standard enhanced streaming.
- Public interface:
  - AgenticDecision (dataclass): structured gate result
  - evaluate_agentic_gate(user_text, entity_resolver, model_manager,
    corpus_manager, intent_info) -> AgenticDecision
- Dependencies:
  - memory.graph_utils.extract_graph_entities (Tier 2 entity match)
  - utils.web_search_trigger.analyze_for_web_search_llm (Tier 4 LLM fallback)
  - knowledge.document_generator.detect_document_intent (doc gen detection)
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
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

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
    'look up contact', 'lookup contact', 'find email', 'find contact',
    "what is ", "'s email", "'s contact",
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
    r'\b(?:pull|print|fetch|grab|open|show|display|load|retriev|render|reprint|spit|bring)\w*'
    r'\s+(?:it|that|this|them|those)(?:\s+(?:up|out|here|over|in))?\b',
    re.IGNORECASE,
)

# Markers that a previous turn was about a saved file/document — used to
# disambiguate the pronoun-retrieval pattern above.
FILE_DOC_CONTEXT_WORDS = (
    'document', 'doc ', ' doc.', 'file', '.md', 'markdown', 'pdf', '.txt',
    'saved', 'on disk', 'implementation plan', 'file_read', 'file access',
    'reconstruct', 'print',
)

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

EXPLICIT_SEARCH_KEYWORDS = [
    'search', 'look up', 'fetch', 'check out', 'go to', 'visit', 'pull up',
]

VETO_INTENTS = {'meta_conversational', 'casual_social'}


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
    needs_computation = any(kw in _lower for kw in COMPUTATION_KEYWORDS)

    if _has_url or any(kw in _lower for kw in WEB_SEARCH_KEYWORDS):
        needs_web_search = True
        logger.debug("[Agentic Gate] Tier 1: explicit web search/URL keyword detected")

    if any(kw in _lower for kw in TOOL_KEYWORDS):
        needs_tools = True

    # Email address + action verb → internet action intent
    if not needs_tools:
        import re as _re_gate
        _has_email_addr = bool(_re_gate.search(r'\S+@\S+\.\S+', user_text))
        if _has_email_addr and any(w in _lower for w in ('email', 'send', 'message', 'write', 'mail', 'contact')):
            needs_tools = True

    # Email-by-name patterns: "email Meagan", "send Meagan an email", "email her about X"
    if not needs_tools:
        import re as _re_gate2
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

    needs_memory = any(kw in _lower for kw in MEMORY_KEYWORDS)

    # Knowledge keywords require 4+ words and no computation trigger
    if len(_words) >= 4 and not needs_computation:
        needs_knowledge = any(kw in _lower for kw in KNOWLEDGE_KEYWORDS)

    # ── Tier 2: Entity match (instant, no LLM) ───────────────────────
    if not needs_computation and not needs_memory and entity_resolver is not None:
        try:
            from memory.graph_utils import extract_graph_entities
            matched_entities = extract_graph_entities(user_text, entity_resolver)
            matched_entities.discard("user")
            if matched_entities:
                _has_recall_signal = (
                    '?' in user_text
                    or any(w in _lower for w in RECALL_SIGNAL_WORDS)
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
    if any(_skip_patterns):
        _is_continuation = any(p in _lower for p in CONTINUATION_PHRASES)
        if _is_continuation and corpus_manager is not None:
            try:
                _recent = corpus_manager.get_recent_memories(2)
                for _prev in _recent:
                    _prev_query = (_prev.get('query', '') or '').lower()
                    _prev_response = (_prev.get('response', '') or '')[:800]
                    _prev_had_signals = (
                        '?' in _prev_query
                        or any(w in _prev_query for w in (
                            'search', 'find', 'look', 'github', 'git',
                            'issues', 'pull request', 'pr ', 'commits',
                            'stats', 'loc', 'lines', 'code',
                            'calculate', 'compute', 'wolfram',
                        ))
                    )
                    _prev_mentioned_tools = any(w in _prev_response.lower() for w in (
                        'let me pull', 'let me grab', 'let me run',
                        'let me check', 'let me search', 'let me query',
                        "i'll hit", "i'll search",
                        'git_stats', 'github api', 'github tool',
                    ))
                    if _prev_had_signals or _prev_mentioned_tools:
                        _prev_was_agentic = True
                        logger.debug(
                            f"[Agentic Gate] Continuation after agentic-intent turn — "
                            f"overriding casual skip (query_signals={_prev_had_signals}, "
                            f"response_tools={_prev_mentioned_tools})"
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
        _is_pronoun_retrieval = bool(FILE_RETRIEVAL_PRONOUN_PATTERN.search(_lower))
        _is_affirmation = (
            any(p in _lower for p in CONTINUATION_PHRASES)
            or (bool(_words) and all(w in FILLER_WORDS for w in _words))
        )
        if _is_pronoun_retrieval or _is_affirmation:
            try:
                _recent = corpus_manager.get_recent_memories(2)
                for _prev in _recent:
                    _resp = (_prev.get('response', '') or '')[:800].lower()
                    _blob = (_prev.get('query', '') or '').lower() + ' ' + _resp
                    _prev_was_file = any(w in _blob for w in FILE_DOC_CONTEXT_WORDS)
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
                trigger_decision = await analyze_for_web_search_llm(
                    query=user_text,
                    model_manager=model_manager,
                )
                should_trigger = getattr(trigger_decision, 'should_search', False)
                search_terms = getattr(trigger_decision, 'search_terms', []) or []

                if getattr(trigger_decision, 'needs_memory_search', False):
                    logger.debug("[Agentic Gate] LLM detected memory search intent")
                    should_trigger = True
                    needs_memory = True
                    search_terms = []
                elif getattr(trigger_decision, 'needs_knowledge_search', False):
                    if not should_trigger:
                        logger.debug("[Agentic Gate] LLM detected knowledge search intent")
                        should_trigger = True
                    needs_knowledge = True
                    search_terms = []
                elif getattr(trigger_decision, 'needs_document_generation', False):
                    logger.info("[Agentic Gate] LLM detected document generation intent")
                    should_trigger = True
                    needs_tools = True
                    search_terms = []
                    doc_gen_intent = {
                        "topic": getattr(trigger_decision, 'document_topic', '') or user_text,
                        "doc_type": getattr(trigger_decision, 'document_type', 'report') or 'report',
                        "focus": None,
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

    # ── Intent-based veto ─────────────────────────────────────────────
    _has_explicit_search = (
        any(kw in _lower for kw in EXPLICIT_SEARCH_KEYWORDS) or _has_url or needs_files
    )
    if should_trigger and not _has_explicit_search and not doc_gen_intent and not self_note_intent:
        if intent_info is not None:
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
            if _type_val in VETO_INTENTS and _intent_conf >= 0.75:
                logger.info(
                    f"[Agentic Gate] VETOED by intent classifier: "
                    f"{_intent_type} (conf={_intent_conf:.2f})"
                )
                should_trigger = False
                search_terms = []

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
    skip_initial = (
        needs_computation or needs_memory or needs_knowledge or needs_tools
        or (needs_web_search and not search_terms)
    )

    # ── Build reason string ───────────────────────────────────────────
    if should_trigger:
        reason = f"triggered: {', '.join(modes) if modes else 'llm-fallback'}"
    elif any(_skip_patterns) and not _prev_was_agentic:
        reason = "casual/short message"
    else:
        reason = "no trigger"

    return AgenticDecision(
        should_trigger=should_trigger,
        modes=modes,
        search_terms=search_terms,
        matched_entities=matched_entities,
        doc_gen_intent=doc_gen_intent,
        self_note_intent=self_note_intent,
        skip_initial_search=skip_initial,
        reason=reason,
    )
