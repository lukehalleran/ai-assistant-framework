"""
# memory/llm_fact_extractor.py

Additive LLM-assisted fact extractor used at shutdown to augment regex/spaCy/REBEL facts.
ENHANCED: Now extracts facts with category classification for user profile building.
ENHANCED (2026-03): Extracts entity facts (non-user subjects) with user_connection metadata.
ENHANCED (2026-04): Accepts existing profile facts so LLM reuses relation names for updates/cancellations.

Contract
- Inputs: list of recent messages — either plain strings (user-only) or dicts with
  {"query": str, "response": str} (conversation pairs). model_manager, optional existing_facts list
- Behavior: calls a compact LLM prompt to extract SRO triples as strict JSON with category metadata
  - When existing_facts provided, prompt instructs LLM to reuse relation names for updates;
    explicit-only guard prevents inferring updates from absence of mentions
  - User facts: subject="user" for personal facts (pronouns normalized)
  - Entity facts: subject=entity name for people/places/orgs/pets discussed in either
    user messages OR assistant responses. This allows extracting structured facts about
    entities that Daemon knows about from conversation history (e.g., pet details, family info)
    Entity facts include user_connection field (e.g., "user's boss", "user's mom's cat")
- Output: list of dict triples (subject, relation, object, value, category, confidence, fact_scope, source_excerpt)
  - fact_scope: "user" or "entity" — indicates whether fact is about the user or a third-party entity
  - source_excerpt: keyword-matched user message that sourced the fact (200-char truncated);
    falls back to last message if no keyword match found
"""

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import json
import re
import os

from utils.logging_utils import get_logger
from memory.user_profile_schema import ProfileCategory, categorize_relation

logger = get_logger("llm_facts")


def _snake(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")


def _is_ephemeral_relation(rel: str) -> bool:
    """Check if a relation is ephemeral and should not be stored as a durable fact."""
    try:
        from config.app_config import PROFILE_EPHEMERAL_RELATIONS
        ephemeral = frozenset(r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS)
    except ImportError:
        ephemeral = frozenset()
    return rel.lower().strip() in ephemeral


def _is_boolean_noise(obj: str) -> bool:
    """Reject facts where the object is just 'true'/'false'/'yes'/'no'."""
    return obj.strip().lower() in {"true", "false", "yes", "no"}


def _normalize_triple(t: Dict[str, Any]) -> Dict[str, str] | None:
    """Normalize and categorize extracted triple."""
    # Coerce to strings defensively; models sometimes emit numbers/bools
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or t.get("value") or "").strip()

    if not subj or not rel or not obj:
        return None

    # Pronouns → user (keep named entities as-is)
    if subj.lower() in {"i", "me", "my", "we", "us", "you", "user"}:
        subj = "user"

    # Relation snake_case
    rel = _snake(rel)

    # Block ephemeral predicates — transient state, not durable facts
    if _is_ephemeral_relation(rel):
        logger.debug(f"[LLM Facts] Blocked ephemeral relation: {subj}|{rel}|{obj}")
        return None

    # Block boolean-only values — no informational content
    if _is_boolean_noise(obj):
        logger.debug(f"[LLM Facts] Blocked boolean noise: {subj}|{rel}|{obj}")
        return None

    # Auto-categorize
    category = categorize_relation(rel)

    # Get confidence if provided
    confidence = float(t.get("confidence", 0.7))

    # Determine fact scope
    is_user = (subj.lower() == "user")
    fact_scope = "user" if is_user else "entity"

    result = {
        "subject": subj.lower(),
        "relation": rel,
        "object": obj.strip().strip(". "),
        "value": obj.strip().strip(". "),  # Alias for compatibility
        "category": category.value,
        "confidence": confidence,
        "fact_scope": fact_scope,
    }

    # Forward user_connection if the LLM provided one
    user_connection = str(t.get("user_connection") or "").strip()
    if user_connection:
        result["user_connection"] = user_connection

    return result


class LLMFactExtractor:
    def __init__(self, model_manager, *,
                 model_alias: str = None,
                 max_input_chars: int = 4000,
                 max_triples: int = 15):  # Increased from 10
        self.mm = model_manager
        self.model_alias = model_alias or "gpt-4o-mini"
        self.max_input_chars = int(max_input_chars)
        self.max_triples = int(max_triples)

    @staticmethod
    def _build_alias_hints(existing_facts: List[Dict[str, Any]]) -> str:
        """Build compact alias hints from SAFE_RELATION_ALIASES for relations the user has."""
        try:
            from memory.user_profile_schema import SAFE_RELATION_ALIASES
        except ImportError:
            return ""
        # Invert: canonical → [variants]
        from collections import defaultdict
        canonical_to_variants = defaultdict(set)
        for variant, canonical in SAFE_RELATION_ALIASES.items():
            canonical_to_variants[canonical].add(variant)
        # Only show aliases where user has a matching fact
        user_rels = {f.get("relation", "") for f in (existing_facts or [])}
        lines = []
        for canonical, variants in sorted(canonical_to_variants.items()):
            if canonical in user_rels or variants & user_rels:
                bad = ", ".join(sorted(variants - {canonical}))
                if bad:
                    lines.append(f"- {canonical}, NOT: {bad}")
        if not lines:
            return ""
        return (
            "\nRELATION ALIASES — always use canonical form:\n"
            + "\n".join(lines[:15]) + "\n"  # cap at 15 to limit token cost
        )

    def _build_prompt(self, user_messages: List, existing_facts: List[Dict[str, Any]] = None) -> str:
        msgs = []
        total = 0
        # Accept either plain strings or conversation pair dicts
        for m in user_messages[-50:]:  # hard cap safety
            if isinstance(m, dict):
                # Conversation pair: include both query and response
                q = (m.get("query") or "").strip()
                r = (m.get("response") or "").strip()
                # Skip API error responses
                if r.startswith("[API Error]"):
                    r = ""
                if not q:
                    continue
                if r:
                    entry = f"User: {q}\nDaemon: {r}"
                else:
                    entry = f"User: {q}"
            else:
                entry = (m or "").strip()
                if not entry:
                    continue
                # Strip role prefixes if user text contained them
                entry = re.sub(r"^(?:user|assistant)\s*:\s*", "", entry, flags=re.I)
            if total + len(entry) + 10 > self.max_input_chars:
                break
            msgs.append(entry)
            total += len(entry) + 10

        joined = "\n".join(f"- {m}" for m in msgs)

        # Build existing facts section if provided
        existing_facts_section = ""
        if existing_facts:
            lines = []
            for f in existing_facts[:60]:  # cap to avoid prompt bloat
                rel = f.get("relation", "")
                val = f.get("value", "")
                cat = f.get("category", "")
                if rel and val:
                    lines.append(f"  - {rel}={val} [{cat}]")
            if lines:
                existing_facts_section = (
                    "\n\nEXISTING PROFILE FACTS (current beliefs about the user):\n"
                    + "\n".join(lines)
                    + "\n\nUPDATE RULES:\n"
                    "- When the user's messages UPDATE, CANCEL, RESCHEDULE, or CHANGE something "
                    "already in the existing facts, you MUST reuse the SAME relation name with "
                    "the new value. This ensures the old fact gets properly superseded.\n"
                    "- Example: existing fact is date_planned=date in Algonquin at noon on Sat. "
                    "User says 'date got cancelled'. Output: "
                    '{{"subject": "user", "relation": "date_planned", "object": "cancelled — was Algonquin Sat, needs reschedule", "category": "goals", "confidence": 0.95}}\n'
                    "- Example: existing fact is gym_schedule=MWF mornings. "
                    "User says 'switching to evenings'. Output: "
                    '{{"subject": "user", "relation": "gym_schedule", "object": "MWF evenings", "category": "fitness", "confidence": 0.9}}\n'
                    "- Do NOT invent a new relation name when an existing one covers the same topic.\n"
                    "- Do NOT infer updates from absence. If the user does not mention a topic, "
                    "leave the existing fact unchanged — do not extract a contradicting fact.\n"
                    "- Only extract an update when the user EXPLICITLY states a change "
                    '(e.g., "I got a new job", "we broke up", "I moved to Boston").\n'
                )
                # Add relation alias hints from SAFE_RELATION_ALIASES
                alias_hints = self._build_alias_hints(existing_facts)
                if alias_hints:
                    existing_facts_section += alias_hints

        # Enhanced extraction prompt with category awareness and few-shot examples
        prompt_template = """You extract factual information about the user from conversation messages.

CATEGORIES to classify facts into:
- identity: name, age, birthday, location, timezone
- education: degrees, schools, courses, GPA, certifications
- career: jobs, roles, skills, companies, experience
- projects: personal projects, tech stack, status
- health: conditions, medications, supplements, symptoms, sleep
- fitness: lifts, body metrics, workout routines, goals
- preferences: likes, dislikes, favorites, formatting preferences
- hobbies: games, interests, pets, social activities
- study: exams, study hours, strategies, weak points
- finance: insurance, budgeting, financial plans
- relationships: family, friends, colleagues
- goals: plans, intentions, aspirations

OUTPUT FORMAT (strict JSON array):
[
  {{"subject": "user", "relation": "snake_case_relation", "object": "value", "category": "category_name", "confidence": 0.0-1.0}}
]

EXAMPLES:
Input: "I'm Sarah, a software developer from Seattle"
Output: [
  {{"subject": "user", "relation": "name", "object": "Sarah", "category": "identity", "confidence": 0.95}},
  {{"subject": "user", "relation": "occupation", "object": "software developer", "category": "career", "confidence": 0.95}},
  {{"subject": "user", "relation": "lives_in", "object": "Seattle", "category": "identity", "confidence": 0.95}}
]

Input: "I built this app and I'm testing it now"
Output: [
  {{"subject": "user", "relation": "role", "object": "app builder", "category": "projects", "confidence": 0.85}}
]

Input: "My name is Luke, I created you"
Output: [
  {{"subject": "user", "relation": "name", "object": "Luke", "category": "identity", "confidence": 0.95}},
  {{"subject": "user", "relation": "role", "object": "creator", "category": "projects", "confidence": 0.9}}
]

ENTITY FACTS (in addition to user facts):
- Extract facts about people, pets, places, topics discussed in EITHER user OR Daemon messages
- Subject should be the entity name (NOT "user")
- Add "user_connection" field explaining relation to user (if known)
- Only extract when clearly stated, not hypothetical
- Pay special attention to pets, family members, and recurring people — these are high-value entities
- Example: "My boss Oliver moved from London"
  → {{"subject": "Oliver", "relation": "moved_from", "object": "London", "category": "relationships", "confidence": 0.75, "user_connection": "user's boss"}}
- Example: Daemon says "Poppy is your mom's black cat, male, with long fur"
  → {{"subject": "Poppy", "relation": "species", "object": "cat, black, long fur, male", "category": "hobbies", "confidence": 0.85, "user_connection": "user's mom's cat"}}

RULES:
- Subject is "user" for personal facts, or the entity name for entity facts
- Relation must be snake_case (e.g., "squat_max", "lives_in", "favorite_beer")
- Extract DURABLE facts — things that are true across sessions, not just right now
- Confidence: 0.9+ for direct statements, 0.7-0.8 for inferred, <0.7 for uncertain
- Do NOT extract questions or hypotheticals
- IMPORTANT: If user introduces themselves or describes their role/activity, extract those as facts
- TEMPORAL: Today's date is {today}. When the user mentions relative dates ("tomorrow", "next Monday", "the following day"), resolve them to absolute dates in the object field. Example: "I work tomorrow" on 2026-03-12 → object: "work on Thu 2026-03-13"
- Do NOT extract transient/ephemeral state that changes constantly:
  * current_activity, current_mood, current_feeling, feeling, feels
  * woke_at, walked_to, showered, tidied, will_drive_to, greeting
  * Generic "is", "was", "has", "thinks", "plans", "wants", "needs" predicates
  * Boolean-value facts like "showered=true" or "has_energy=true"
  * One-time actions: "asked_about", "tested", "walked_to"
  These pollute the fact store. Only extract facts that remain true over time.
{existing_facts}
MESSAGES (conversation pairs, newest last):
{messages}

JSON:"""
        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        prompt = prompt_template.format(
            messages=joined,
            today=today_str,
            existing_facts=existing_facts_section,
        )

        # Always log the messages being processed (helps debug empty extractions)
        logger.info(f"[LLM Facts] Building prompt with {len(msgs)} messages, total chars={len(prompt)}")
        logger.info(f"[LLM Facts] Input messages: {joined[:300]}{'...' if len(joined) > 300 else ''}")
        if existing_facts:
            logger.info(f"[LLM Facts] Injected {len(existing_facts)} existing profile facts for relation reuse")

        return prompt

    async def extract_triples(self, user_messages: List,
                              existing_facts: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        if not user_messages:
            return []

        prompt = self._build_prompt(user_messages, existing_facts=existing_facts)

        try:
            text = await self.mm.generate_once(
                prompt=prompt,
                model_name=self.model_alias,
                system_prompt="You output only strict JSON arrays. No prose, no explanation.",
                max_tokens=600,  # Increased from 400 for more facts
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as e:
            logger.warning(f"[LLM Facts] generate_once failed: {e}")
            return []

        if not isinstance(text, str) or not text.strip():
            logger.warning("[LLM Facts] generate_once returned empty or non-string response")
            return []

        # Log the raw response for debugging
        logger.info(f"[LLM Facts] Raw LLM response: {text[:500]}")

        # Attempt to parse a JSON array (robust to leading/trailing junk)
        raw = text.strip()

        # Check for stub response indicating API client issue
        if raw.startswith("[API unavailable]"):
            logger.warning(f"[LLM Facts] API client not available - got stub response: {raw[:100]}")
            return []

        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            else:
                logger.warning(f"[LLM Facts] No JSON array found in response: {text[:200]}")
                return []
            data = json.loads(raw)
            if not isinstance(data, list):
                logger.warning(f"[LLM Facts] JSON parsed but not a list: {type(data)}")
                return []
        except Exception as e:
            logger.warning(f"[LLM Facts] JSON parse failed: {e} - Response was: {text[:200]}")
            return []

        triples: List[Dict[str, str]] = []
        seen = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            norm = _normalize_triple(item)
            if not norm:
                continue
            key = (norm["subject"], norm["relation"], norm["object"])
            if key in seen:
                continue
            seen.add(key)
            triples.append(norm)
            if len(triples) >= self.max_triples:
                break

        # Attach source_excerpt by matching each triple back to its source message
        if triples:
            self._attach_source_excerpts(triples, user_messages)
            logger.info(f"[LLM Facts] SUCCESS: extracted={len(triples)} facts using model={self.model_alias}")
            for t in triples:
                logger.info(f"[LLM Facts]   -> {t['relation']}: {t['object']} (conf={t['confidence']})")
        else:
            logger.warning(f"[LLM Facts] WARNING: extracted=0 facts from LLM response. Model returned empty array.")

        return triples

    @staticmethod
    def _attach_source_excerpts(triples: List[Dict[str, str]],
                                user_messages: List) -> None:
        """Match each triple to its most likely source message via keyword overlap."""
        if not user_messages:
            return
        # Clean messages once — handle both plain strings and conversation pair dicts
        cleaned = []
        for m in user_messages:
            if isinstance(m, dict):
                q = (m.get("query") or "").strip()
                r = (m.get("response") or "").strip()
                # Combine both for keyword matching
                text = q
                if r and not r.startswith("[API Error]"):
                    text = f"{q} {r}"
            else:
                text = re.sub(r"^(?:user|assistant)\s*:\s*", "", (m or "").strip(), flags=re.I)
            cleaned.append(text)
        for triple in triples:
            # Build keyword set from object + relation words
            keywords = set()
            for field in ("object", "relation"):
                val = triple.get(field, "")
                # Split snake_case and regular words
                for word in re.split(r"[_\s]+", val.lower()):
                    if len(word) > 2:
                        keywords.add(word)
            # Score each message by keyword overlap
            best_msg = ""
            best_score = 0
            for msg in cleaned:
                if not msg:
                    continue
                msg_lower = msg.lower()
                score = sum(1 for kw in keywords if kw in msg_lower)
                if score > best_score:
                    best_score = score
                    best_msg = msg
            # Fallback to last non-empty message if no keyword match
            if not best_msg:
                for msg in reversed(cleaned):
                    if msg:
                        best_msg = msg
                        break
            triple["source_excerpt"] = best_msg[:200]
