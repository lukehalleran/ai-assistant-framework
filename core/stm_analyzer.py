"""
# core/stm_analyzer.py

Module Contract
- Purpose: Short-Term Memory analyzer that generates concise JSON summaries of recent conversation context
- Inputs:
  - recent_memories: List of recent conversation dicts (query/response pairs).
    Caller is expected to time-window these (typically last 24h via
    CorpusManager.get_recent_within_hours), capped by STM_MAX_RECENT_MESSAGES.
  - user_query: Current user input
  - last_assistant_response: Optional last assistant message for context
- Side inputs (read internally, not passed):
  - Last STM_INJECT_DAILY_NOTES_DAYS daily notes from the Obsidian vault, read
    via utils.daily_notes_generator.read_daily_note(). Used to give STM
    cross-day recall disambiguation. Gracefully degrades when notes are
    missing (e.g. session starts before catch-up has run).
- Outputs:
  - Dict with: topic, user_question, intent, tone, reference_type, temporal_facts, open_threads, constraints
  - reference_type ∈ {new_event, recall, clarification, correction, unclear} — disambiguates whether the current message is a new report or a restatement of prior context. Defaults to "unclear" when uncertain.
  - Disambiguation rule 5 [2026-07-28]: bare-pronoun openers ("It was...") and referent
    corrections ("No I mean...") resolve from the IMMEDIATELY PRECEDING exchange — the topic
    continues; never re-derived from surface keywords (pairs with
    query_checker.is_anaphoric_continuation topic inheritance in ContextPipeline).
  - Disambiguation rule 6 [2026-08-05]: substances/medications/proper nouns are named EXACTLY
    as in the CURRENT message — never substituted from earlier context (a live STM summary
    rewrote "900 mg lorvatin" as "900 mg of kavarin daily" because kavarin dominated the
    surrounding turns).
  - Rule 3 extension [2026-08-05]: no invented DURATIONS — "stopped the medication 3 days
    ago" is a timeline marker, not a symptom duration; similar older episodes are PREVIOUS
    episodes (pairs with the episode-boundary block in tone_instructions session headers).
  - temporal_facts: normalized facts about user's current state, with collapse-toward-fewer-events disambiguation rule applied.
- Key pieces:
  - analyze(): Main async method that calls LLM to analyze context
  - _format_memories(): Converts memory dicts to readable conversation text with relative day labels
  - _parse_json(): Robust JSON parser with fallback handling
- Side effects:
  - None beyond LLM API call and logging
"""
import json
import re
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger
from datetime import date, timedelta
from datetime import datetime
from utils.query_checker import (
    _PROPER_NOUN_STOPWORDS,
    extract_rare_proper_nouns,
    extract_data_tokens,
    has_present_state_report,
    is_request_shaped,
    keyword_tokens,
)

logger = get_logger("stm_analyzer")

# Graph entity types whose sentence-initial mention is name-shaped enough to
# count as a novel referent (a person or a pet — not a place/concept/project,
# whose TitleCase tokens are far more often ordinary sentence-initial words).
_NOVELTY_ENTITY_TYPES = frozenset({"animal", "pet", "person"})
_SENTENCE_INITIAL_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z'’-]{2,}$")


def _word_in_text(word: str, text: str) -> bool:
    """Case-insensitive word-bounded containment; garbage → False."""
    if not word or not text:
        return False
    try:
        return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.IGNORECASE) is not None
    except re.error:
        return False


def _sentence_initial_candidates(user_query: str) -> List[str]:
    """TitleCase tokens in sentence-INITIAL position — the ones
    ``extract_rare_proper_nouns`` deliberately drops. Mirrors its tokenizer,
    stoplist, possessive strip and ALL-CAPS exclusion; returns surfaces in
    order of first appearance."""
    if not user_query or not user_query.strip():
        return []
    pieces = re.findall(r"[A-Za-z'’-]+|[.!?\n]", user_query)
    _title_abbrevs = {"dr", "mr", "mrs", "ms", "prof", "st"}
    found: List[str] = []
    seen: set = set()
    sentence_initial = True
    prev_word = ""
    for tok in pieces:
        if tok in (".", "!", "?", "\n"):
            if not (tok == "." and prev_word.lower() in _title_abbrevs):
                sentence_initial = True
            continue
        is_initial = sentence_initial
        sentence_initial = False
        prev_word = tok
        if not is_initial:
            continue
        surface = tok.removesuffix("'s").removesuffix("’s").rstrip("'’-")
        if not _SENTENCE_INITIAL_TOKEN_RE.match(surface):
            continue
        if surface.isupper():
            continue
        low = surface.lower()
        if low in _PROPER_NOUN_STOPWORDS or low in seen:
            continue
        seen.add(low)
        found.append(surface)
    return found


def _resolve_graph_node(name: str, graph_memory=None, entity_resolver=None):
    """Resolve a mention to a graph node via the deployed alias index
    (``GraphMemory.resolve_entity`` → ``get_entity``); optional
    ``entity_resolver`` (memory_coordinator.entity_resolver) tried first.
    Any failure or a non-string resolution → None (Mock-safe)."""
    eid = None
    try:
        if entity_resolver is not None and hasattr(entity_resolver, "resolve"):
            cand = entity_resolver.resolve(name)
            if isinstance(cand, str) and cand.strip():
                eid = cand
    except Exception:
        eid = None
    if eid is None and graph_memory is not None:
        try:
            cand = graph_memory.resolve_entity(name)
            if isinstance(cand, str) and cand.strip():
                eid = cand
        except Exception:
            eid = None
    if not isinstance(eid, str) or not eid.strip():
        return None
    try:
        node = graph_memory.get_entity(eid) if graph_memory is not None else None
    except Exception:
        node = None
    return node


def _node_type(node) -> str:
    et = getattr(node, "entity_type", None)
    return et.strip().lower() if isinstance(et, str) else ""


def _node_surfaces(node, name: str) -> List[str]:
    """Every string the node could appear as in the window: the mention, its
    id/display name, aliases, and metadata['original_name']."""
    surfaces = [name]
    for attr in ("entity_id", "display_name"):
        v = getattr(node, attr, None)
        if isinstance(v, str) and v.strip():
            surfaces.append(v.strip())
    aliases = getattr(node, "aliases", None)
    if isinstance(aliases, (list, tuple, set)):
        surfaces.extend(a.strip() for a in aliases if isinstance(a, str) and a.strip())
    md = getattr(node, "metadata", None)
    if isinstance(md, dict):
        orig = md.get("original_name")
        if isinstance(orig, str) and orig.strip():
            surfaces.append(orig.strip())
    return surfaces


def novel_named_entities(
    user_query: str,
    window_text: str,
    graph_memory=None,
    entity_resolver=None,
) -> List[str]:
    """Names in the current message that appear NOWHERE in the short-term
    window (2026-09-03 STM novelty override).

    The STM prompt biases reference_type toward "recall" when in doubt, and
    the formatter then injects "the current message restates an event already
    in context" — which was wrong for a message naming an entity absent from
    the whole window. Candidates:
      * mid-sentence rare proper nouns (``extract_rare_proper_nouns``), always;
      * sentence-initial TitleCase tokens ONLY when a graph is given and the
        token resolves to a person/pet/animal node (no dictionary separates
        "Biscuit caught a moth" from "Please pass the salt" otherwise).
    A candidate is novel only if neither the name nor any alias / original
    name of its resolved node appears word-bounded in ``window_text``.
    Under-fires: missing input, resolver failures, Mock objects → [].
    """
    if not isinstance(user_query, str) or not user_query.strip():
        return []
    window = window_text if isinstance(window_text, str) else ""
    try:
        candidates = list(extract_rare_proper_nouns(user_query))
    except Exception:
        candidates = []
    ordered: List[tuple] = [(c, False) for c in candidates]
    if graph_memory is not None or entity_resolver is not None:
        try:
            for tok in _sentence_initial_candidates(user_query):
                if tok.lower() not in {c.lower() for c, _ in ordered}:
                    ordered.append((tok, True))
        except Exception:
            pass

    novel: List[str] = []
    seen: set = set()
    for name, needs_graph in ordered:
        if not isinstance(name, str) or not name.strip():
            continue
        if name.lower() in seen:
            continue
        node = None
        if graph_memory is not None or entity_resolver is not None:
            node = _resolve_graph_node(name, graph_memory, entity_resolver)
        if needs_graph:
            if node is None or _node_type(node) not in _NOVELTY_ENTITY_TYPES:
                continue
        surfaces = _node_surfaces(node, name) if node is not None else [name]
        try:
            if any(_word_in_text(surf, window) for surf in surfaces):
                continue
        except Exception:
            continue
        seen.add(name.lower())
        novel.append(name)
    return novel


def novel_data_tokens(user_query: str, window_text: str) -> List[str]:
    """Data-shaped tokens (clock times, dates, dose/quantity, "day N") in the
    current message that appear NOWHERE in the short-term window — the
    numeric analogue of ``novel_named_entities`` (2026-09-06 new-data
    override). Comparison is on the CANONICAL form (a query's "10 AM"
    matches a window's "10:00"). Order of first appearance in the query;
    under-fires on missing input or any failure -> []."""
    if not isinstance(user_query, str) or not user_query.strip():
        return []
    window = window_text if isinstance(window_text, str) else ""
    try:
        query_tokens = extract_data_tokens(user_query)
        window_tokens = set(extract_data_tokens(window))
    except Exception:
        return []
    novel: List[str] = []
    seen: set = set()
    for tok in query_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok not in window_tokens:
            novel.append(tok)
    return novel


def new_data_override(user_query: str, window_text: str) -> Dict[str, Any]:
    """Deterministic post-check (2026-09-06): the STM prompt biases "recall"
    when in doubt, which mislabels two shapes it never accounts for —
    (a) a REQUEST for analysis/action, not a report of an event at all, and
    (b) a fresh present-tense self-report carrying a data point (clock time,
    date, dose, day count) the short-term window has no record of. Ladder,
    first hit wins:
      1. request-shaped message                                -> "request"
      2. a data-shaped token in the query absent from the window
                                                                 -> "new_data"
      3. a present-tense self-report whose salient (len>=4) keyword
         vocabulary is majority-absent from the window          -> "new_data"
    Deterministic, under-fires by design; {} on no hit or any failure.
    Companion to ``novel_named_entities`` — apply only when the novelty
    override above did NOT already demote this turn."""
    try:
        text = user_query if isinstance(user_query, str) else ""
        if not text.strip():
            return {}
        window = window_text if isinstance(window_text, str) else ""
        if is_request_shaped(text):
            return {"reason": "request"}
        novel = novel_data_tokens(text, window)
        if novel:
            return {"reason": "new_data", "novel_data": novel}
        if has_present_state_report(text):
            raw_kws = keyword_tokens(text, min_len=4)
            kws: List[str] = []
            seen_kw: set = set()
            for w in raw_kws:
                clean = w.strip(".,!?;:'\"()")
                if clean and clean not in seen_kw:
                    seen_kw.add(clean)
                    kws.append(clean)
            if kws:
                absent = [w for w in kws if not _word_in_text(w, window)]
                if len(absent) / len(kws) >= 0.5:
                    return {"reason": "new_data", "novel_data": absent[:5]}
        return {}
    except Exception:
        return {}


class STMAnalyzer:
    """
    Analyzes short-term conversation context using a lightweight LLM pass.

    Returns structured JSON summaries to help the main model understand
    immediate conversation state without re-reading full message history.
    """

    def __init__(self, model_manager, model_name: str = "gpt-4o-mini"):
        """
        Initialize STM analyzer.

        Args:
            model_manager: ModelManager instance for LLM calls
            model_name: Model to use for analysis (default: gpt-4o-mini for speed/cost)
        """
        self.model_manager = model_manager
        self.model_name = model_name
        logger.info(f"[STMAnalyzer] Initialized with model: {model_name}")

    async def analyze(
        self,
        recent_memories: List[Dict[str, Any]],
        user_query: str,
        last_assistant_response: Optional[str] = None,
        graph_memory=None,
    ) -> Dict[str, Any]:
        """
        Analyze short-term conversation context.

        Args:
            recent_memories: Recent conversation turns (list of dicts with 'query'/'response')
            user_query: Current user input
            last_assistant_response: Optional last assistant message for additional context
            graph_memory: Optional GraphMemory — enables the sentence-initial
                name allow-gate of the novelty override (a known pet/person
                named at sentence start counts as a novel referent when
                absent from the window)

        Returns:
            Dict with fields:
                - topic: Current conversation topic (brief)
                - user_question: What user is asking now
                - intent: What they really want
                - tone: Emotional tone
                - open_threads: Unresolved questions (list)
                - constraints: Any constraints noted (list)
        """
        conversation_text = self._format_memories(recent_memories)
        daily_notes_text = self._get_recent_daily_notes_text()

        immediate_section = ""
        if last_assistant_response:
            immediate = self._head_tail(str(last_assistant_response), 2400)
            immediate_section = (
                "\n\nIMMEDIATELY PRECEDING ASSISTANT REPLY (authoritative for "
                "short answers and option selections; inspect its final question):\n"
                f"{immediate}\n"
            )

        # Daily notes section is included only when at least one note was found.
        # When sessions start before catch-up has run, this gracefully degrades
        # to the old (recent-conversation-only) behavior.
        notes_section = ""
        if daily_notes_text:
            notes_section = (
                "\n\nRecent daily notes (Daemon-generated EOD summaries — use these "
                "to identify whether the current message restates an event that has "
                "already happened on a prior day):\n"
                f"{daily_notes_text}\n"
            )

        # Build STM analysis prompt
        prompt = f"""Analyze this conversation's SHORT-TERM context only.

Recent conversation:
{conversation_text}
{immediate_section}
{notes_section}
Current user query: {user_query}

Return ONLY valid JSON with these fields:
- "topic": current conversation topic (brief, 2-5 words)
- "user_question": what user is asking now, paraphrased neutrally (one sentence). If they are restating something, paraphrase the act of restatement, not the underlying claim.
- "intent": what they really want from this turn (one sentence)
- "tone": emotional tone (one word: neutral/casual/concerned/frustrated/excited)
- "reference_type": ONE of:
    * "new_event"     — user is reporting something not present in recent conversation
    * "recall"        — user is restating, re-emphasizing, or returning to an event already in recent conversation
    * "clarification" — user is adding a small detail to an existing topic
    * "correction"    — user is contradicting a claim the assistant made
    * "unclear"       — cannot tell from context (DEFAULT when uncertain)
  Treating a recall as a new event is the more dangerous error. Default to "recall" or "unclear" when in doubt. If the user names actors, locations, or events you do NOT see in recent conversation, classify as "unclear" — you may simply lack the source memory.
- "temporal_facts": list of normalized facts about the user's current state with explicit time anchors. Resolve ambiguous references conservatively: collapse toward fewer events, not more. (list of strings, may be empty)
- "open_threads": unresolved questions or topics from conversation (list of strings)
- "constraints": any constraints mentioned (time, tools, safety, etc.) (list of strings)

CRITICAL DISAMBIGUATION RULES:
1. If the current message names the same actors + action + outcome as something in recent conversation, classify as "recall" — not a new event.
2. Resolve ambiguous temporal phrases by collapsing toward fewer events. "Did not sleep" + "fight mode all night" on the same morning = ONE bad night, not two.
3. Do NOT invent a count, pattern, or DURATION. If you only have evidence of one occurrence of something, say so explicitly in temporal_facts. Never state how long a symptom/situation has lasted unless the user said it for THIS episode — "stopped the medication 3 days ago" is a timeline marker, not a symptom duration; a similar episode in older context is a PREVIOUS episode, not the current one.
4. When the user uses phrases like "told them what happened", "this situation", "that thing" without naming a new event, assume they are referencing something already known — classify as "recall" or "unclear".
5. If the current message opens with a bare pronoun ("It was...", "That's...") or corrects your reading ("No I mean...", "I wasn't talking about X"), resolve the pronoun from the IMMEDIATELY PRECEDING exchange — the topic CONTINUES that exchange's topic. Do not re-derive the topic from surface keywords: "It was 3 years of twice a week" mid-illness-conversation is about the illness, not exercise. A correction re-scopes the user's own previous message; it is NOT a new event or a new topic.
6. Name substances, medications, and proper nouns EXACTLY as the user did in the CURRENT message. Never substitute a different drug/entity from earlier context: if the user says "900 mg of lorvatin" but earlier turns discussed kavarin, the fact is about Lorvatin. When the current message names no substance and the referent is ambiguous, write "the medication" rather than guessing a name.
7. If the current message is a SHORT FRAGMENT (a few words, no verb, no question mark), do NOT invent a "user_question" or reframe it as an information request — it is almost always a riff or continuation of the immediately preceding exchange. Describe it as a continuation (e.g. "User is continuing the joke about X") and set reference_type to "recall" or "clarification", not "new_event".

Example (new event):
{{
  "topic": "Python debugging",
  "user_question": "How to fix the timeout error",
  "intent": "Get practical solution to immediate technical problem",
  "tone": "frustrated",
  "reference_type": "new_event",
  "temporal_facts": ["user is currently debugging a timeout error"],
  "open_threads": ["Performance optimization", "Error handling strategy"],
  "constraints": ["Limited to standard library", "Production environment"]
}}

Example (recall — user re-emphasizing something already discussed):
{{
  "topic": "Police response",
  "user_question": "User is re-emphasizing that police declined to act on the prior incident",
  "intent": "Express continued distress about the same event, not report a new one",
  "tone": "concerned",
  "reference_type": "recall",
  "temporal_facts": ["one prior incident of police inaction is in recent context; no evidence of a second"],
  "open_threads": [],
  "constraints": []
}}

Return JSON only, no markdown or extra text:"""

        try:
            logger.debug(f"[STMAnalyzer] Running analysis for query: {user_query[:50]}...")

            # Call model manager using generate_once for non-streaming response
            content = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=self.model_name,
                system_prompt="You are a context analyzer that returns only valid JSON.",
                max_tokens=300,
                temperature=0.3  # Lower temp for more structured output
            )

            # Parse and return
            parsed = self._parse_json(content)
            # Deterministic continuity floor: a short non-question directly
            # following the assistant's question is an answer/selection, never
            # a new information request. The live "Day of please thank you"
            # turn was otherwise rewritten as "what day is today?".
            try:
                from utils.query_checker import is_continuation_answer
                if is_continuation_answer(user_query, last_assistant_response or ""):
                    parsed["reference_type"] = "clarification"
                    parsed["user_question"] = (
                        "User is answering the assistant's immediately preceding "
                        f"question with: {user_query.strip()}"
                    )
                    parsed["intent"] = (
                        "Apply the user's selected option to the immediately "
                        "preceding request without asking for it again"
                    )
            except Exception:
                pass
            # Novelty override (2026-09-03): "recall" asserts the message
            # restates something already in context, but the prompt biases
            # toward it when in doubt. A message naming an entity that appears
            # NOWHERE in the STM window (conversation + daily notes + the
            # immediately preceding reply) cannot be a restatement of that
            # window — demote to "unclear" so the formatter's verify-first
            # warning replaces the "do not count it as a separate occurrence"
            # one. Deterministic, under-fires (no names → no-op).
            try:
                if parsed.get("reference_type") == "recall":
                    window = "\n".join(filter(None, [
                        conversation_text,
                        daily_notes_text,
                        str(last_assistant_response or ""),
                    ]))
                    novel = novel_named_entities(user_query, window, graph_memory)
                    if novel:
                        parsed["reference_type"] = "unclear"
                        parsed["novelty_override"] = True
                        parsed["novel_entities"] = novel
                        logger.debug(
                            f"[STMAnalyzer] Novelty override: recall → unclear "
                            f"(names absent from window: {novel})"
                        )
            except Exception:
                pass
            # New-data override (2026-09-06): runs only when the novelty
            # step above did NOT already demote this turn (still "recall").
            # Covers two more shapes the recall-biased prompt gets wrong:
            # a REQUEST for analysis/action (not a report at all), and a
            # fresh present-tense self-report carrying a data point (clock
            # time, date, dose, day count) absent from the same window.
            try:
                if parsed.get("reference_type") == "recall":
                    window = "\n".join(filter(None, [
                        conversation_text,
                        daily_notes_text,
                        str(last_assistant_response or ""),
                    ]))
                    override = new_data_override(user_query, window)
                    if override:
                        parsed["reference_type"] = "unclear"
                        parsed["new_data_override"] = override["reason"]
                        if override.get("novel_data"):
                            parsed["novel_data"] = override["novel_data"]
                        logger.debug(
                            f"[STMAnalyzer] New-data override: recall → unclear "
                            f"(reason={override['reason']})"
                        )
            except Exception:
                pass
            logger.debug(f"[STMAnalyzer] Analysis complete: topic={parsed.get('topic')}, tone={parsed.get('tone')}")
            return parsed

        except Exception as e:
            logger.error(f"[STMAnalyzer] Analysis failed: {e}")
            return self._empty_summary()

    def _get_recent_daily_notes_text(self, num_days: Optional[int] = None) -> str:
        """Read the last N daily notes from the Obsidian vault and format them
        as a single text block for STM injection.

        Returns "" if the feature is disabled, no notes are available, or the
        helper module can't be imported. Pure read-only — never triggers
        generation or narrative refresh.
        """
        try:
            from config.app_config import STM_INJECT_DAILY_NOTES_DAYS
        except ImportError:
            STM_INJECT_DAILY_NOTES_DAYS = 0

        n = num_days if num_days is not None else STM_INJECT_DAILY_NOTES_DAYS
        if n <= 0:
            return ""

        try:
            from utils.daily_notes_generator import read_daily_note
        except ImportError:
            return ""

        today = date.today()
        parts: List[str] = []

        for offset in range(1, n + 1):
            target = today - timedelta(days=offset)
            text = read_daily_note(target)
            if not text:
                continue
            label = "yesterday" if offset == 1 else f"{offset} days ago"
            parts.append(
                f"--- Daily note for {target.isoformat()} ({label}) ---\n{text.strip()}"
            )

        if not parts:
            logger.debug("[STMAnalyzer] No daily notes available for injection")
            return ""

        logger.debug(f"[STMAnalyzer] Injecting {len(parts)} daily note(s) into STM input")
        return "\n\n".join(parts)

    def _format_memories(self, memories: List[Dict]) -> str:
        """
        Convert memory dicts to readable conversation text with temporal markers.

        Args:
            memories: List of conversation dicts with 'query', 'response', and 'timestamp' keys

        Returns:
            Formatted conversation string with relative day labels
        """
        from utils.time_manager import format_relative_timestamp

        # Corpus contract is newest-first. Render chronologically so the final
        # exchange really is the immediate predecessor; sort by timestamp when
        # all timestamps are parseable, otherwise reverse the contract order.
        ordered = list(memories or [])
        parsed_times = []
        for mem in ordered:
            raw = mem.get("timestamp", "") if isinstance(mem, dict) else ""
            try:
                parsed_times.append(
                    raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
                )
            except (ValueError, TypeError):
                parsed_times.append(None)
        if ordered and all(ts is not None for ts in parsed_times):
            try:
                ordered = [
                    mem for _, mem in sorted(
                        zip(parsed_times, ordered),
                        key=lambda pair: pair[0].timestamp(),
                    )
                ]
            except (ValueError, OSError, OverflowError):
                ordered.reverse()
        else:
            ordered.reverse()

        lines = []
        for mem in ordered:
            query = mem.get('query', '').strip()
            response = mem.get('response', '').strip()

            # Extract timestamp for temporal context
            ts = mem.get('timestamp', '')
            ts_prefix = ""
            if ts:
                try:
                    if isinstance(ts, datetime):
                        ts_prefix = f"[{format_relative_timestamp(ts)}] "
                    elif isinstance(ts, str):
                        ts_prefix = f"[{format_relative_timestamp(datetime.fromisoformat(ts))}] "
                except (ValueError, TypeError):
                    pass

            if query:
                # Preserve both the opening context and closing ask. The old
                # head-only 200 chars routinely deleted the choice question.
                query_short = self._head_tail(query, 500)
                lines.append(f"{ts_prefix}User: {query_short}")

            if response:
                response_short = self._head_tail(response, 700)
                lines.append(f"Assistant: {response_short}")

        # Keep last 10 lines max (5 exchanges)
        return '\n'.join(lines[-10:])

    @staticmethod
    def _head_tail(text: str, limit: int) -> str:
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        head = limit // 2
        return value[:head] + "\n[…snipped…]\n" + value[-(limit - head):]

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Robust JSON parser with fallback handling.

        Handles markdown-wrapped JSON, extra text, and malformed output.

        Args:
            raw: Raw LLM response text

        Returns:
            Parsed dict or empty summary on failure
        """
        try:
            # Try to extract JSON if wrapped in markdown
            if '```json' in raw:
                start = raw.find('```json') + 7
                end = raw.find('```', start)
                raw = raw[start:end].strip()
            elif '```' in raw:
                start = raw.find('```') + 3
                end = raw.find('```', start)
                raw = raw[start:end].strip()

            # Try direct parsing
            parsed = json.loads(raw)

            # Validate required fields
            required = ['topic', 'user_question', 'intent', 'tone', 'open_threads', 'constraints']
            for field in required:
                if field not in parsed:
                    logger.warning(f"[STMAnalyzer] Missing field '{field}' in JSON, using empty summary")
                    return self._empty_summary()

            # Backfill optional fields added later (graceful degradation if older
            # model output omits them)
            parsed.setdefault('reference_type', 'unclear')
            parsed.setdefault('temporal_facts', [])

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"[STMAnalyzer] Failed to parse JSON: {e}. Raw: {raw[:100]}...")
            return self._empty_summary()

    def _empty_summary(self) -> Dict[str, Any]:
        """
        Fallback summary when analysis fails.

        Returns:
            Empty but valid STM summary dict
        """
        return {
            "topic": "unknown",
            "user_question": "",
            "intent": "",
            "tone": "neutral",
            "reference_type": "unclear",
            "temporal_facts": [],
            "open_threads": [],
            "constraints": []
        }
