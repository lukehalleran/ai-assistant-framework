"""
# memory/llm_fact_extractor.py

Additive LLM-assisted fact extractor used at shutdown to augment regex/spaCy/REBEL facts.
ENHANCED: Now extracts facts with category classification for user profile building.
ENHANCED (2026-03): Extracts entity facts (non-user subjects) with user_connection metadata.
ENHANCED (2026-04): Accepts existing profile facts so LLM reuses relation names for updates/cancellations.

Contract
- Inputs: list of recent user-only messages (strings), model_manager, optional existing_facts list
- Behavior: calls a compact LLM prompt to extract SRO triples as strict JSON with category metadata
  - When existing_facts provided, prompt instructs LLM to reuse relation names for updates
  - User facts: subject="user" for personal facts (pronouns normalized)
  - Entity facts: subject=entity name for people/places/orgs the user discusses
    Entity facts include user_connection field (e.g., "user's boss")
- Output: list of dict triples (subject, relation, object, value, category, confidence, fact_scope)
  - fact_scope: "user" or "entity" — indicates whether fact is about the user or a third-party entity
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

    def _build_prompt(self, user_messages: List[str], existing_facts: List[Dict[str, Any]] = None) -> str:
        msgs = []
        total = 0
        # Build from newest last; enforce char budget
        for m in user_messages[-50:]:  # hard cap safety
            m = (m or "").strip()
            if not m:
                continue
            # Strip role prefixes if user text contained them
            m = re.sub(r"^(?:user|assistant)\s*:\s*", "", m, flags=re.I)
            if total + len(m) + 10 > self.max_input_chars:
                break
            msgs.append(m)
            total += len(m) + 10

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
                )

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
  {{"subject": "user", "relation": "role", "object": "app builder", "category": "projects", "confidence": 0.85}},
  {{"subject": "user", "relation": "current_activity", "object": "testing", "category": "projects", "confidence": 0.8}}
]

Input: "My name is Luke, I created you"
Output: [
  {{"subject": "user", "relation": "name", "object": "Luke", "category": "identity", "confidence": 0.95}},
  {{"subject": "user", "relation": "role", "object": "creator", "category": "projects", "confidence": 0.9}}
]

ENTITY FACTS (in addition to user facts):
- Also extract facts about people, places, topics the user discusses
- Subject should be the entity name (NOT "user")
- Add "user_connection" field explaining relation to user (if known)
- Only extract when clearly stated, not hypothetical
- Example: "My boss Oliver moved from London"
  → {{"subject": "Oliver", "relation": "moved_from", "object": "London", "category": "relationships", "confidence": 0.75, "user_connection": "user's boss"}}

RULES:
- Subject is "user" for personal facts, or the entity name for entity facts
- Relation must be snake_case (e.g., "squat_max", "lives_in", "favorite_beer")
- Extract facts the user states about themselves - be generous, not restrictive
- Confidence: 0.9+ for direct statements, 0.7-0.8 for inferred, <0.7 for uncertain
- Do NOT extract questions or hypotheticals
- IMPORTANT: If user introduces themselves or describes their role/activity, extract those as facts
- TEMPORAL: Today's date is {today}. When the user mentions relative dates ("tomorrow", "next Monday", "the following day"), resolve them to absolute dates in the object field. Example: "I work tomorrow" on 2026-03-12 → object: "work on Thu 2026-03-13"
{existing_facts}
MESSAGES (user only, newest last):
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

    async def extract_triples(self, user_messages: List[str],
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

        if triples:
            logger.info(f"[LLM Facts] SUCCESS: extracted={len(triples)} facts using model={self.model_alias}")
            for t in triples:
                logger.info(f"[LLM Facts]   -> {t['relation']}: {t['object']} (conf={t['confidence']})")
        else:
            logger.warning(f"[LLM Facts] WARNING: extracted=0 facts from LLM response. Model returned empty array.")

        return triples
