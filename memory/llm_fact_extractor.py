"""
# memory/llm_fact_extractor.py

Additive LLM-assisted fact extractor used at shutdown to augment regex/spaCy/REBEL facts.
ENHANCED: Now extracts facts with category classification for user profile building.

Contract
- Inputs: list of recent user-only messages (strings), model_manager
- Behavior: calls a compact LLM prompt to extract SRO triples as strict JSON with category metadata
- Output: list of dict triples (subject, relation, object, value, category, confidence)
"""

from __future__ import annotations

from typing import List, Dict, Any
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

    # Pronouns → user
    if subj.lower() in {"i", "me", "my", "we", "us", "you", "user"}:
        subj = "user"

    # Relation snake_case
    rel = _snake(rel)

    # Auto-categorize
    category = categorize_relation(rel)

    # Get confidence if provided
    confidence = float(t.get("confidence", 0.7))

    return {
        "subject": subj.lower(),
        "relation": rel,
        "object": obj.strip().strip(". "),
        "value": obj.strip().strip(". "),  # Alias for compatibility
        "category": category.value,
        "confidence": confidence,
    }


class LLMFactExtractor:
    def __init__(self, model_manager, *,
                 model_alias: str = None,
                 max_input_chars: int = 4000,
                 max_triples: int = 15):  # Increased from 10
        self.mm = model_manager
        self.model_alias = model_alias or "gpt-4o-mini"
        self.max_input_chars = int(max_input_chars)
        self.max_triples = int(max_triples)

    def _build_prompt(self, user_messages: List[str]) -> str:
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

        # Enhanced extraction prompt with category awareness
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

RULES:
- Subject is always "user" for personal facts
- Relation must be snake_case (e.g., "squat_max", "lives_in", "favorite_beer")
- Only extract EXPLICIT facts stated in the text
- Confidence: 0.9+ for direct statements, 0.7-0.8 for inferred, <0.7 for uncertain
- Do NOT extract opinions, speculation, or questions
- Do NOT extract facts about other people unless relation to user is clear

MESSAGES (user only, newest last):
{messages}

JSON:"""
        prompt = prompt_template.format(messages=joined)

        # Optional debug of input size/content (off by default)
        try:
            dbg = os.getenv("LLM_FACTS_LOG_INPUT", "0").strip().lower() not in ("0","false","no","off")
        except Exception:
            dbg = False
        if dbg:
            logger.debug(
                f"[LLM Facts][Input] msgs={len(msgs)} budget={self.max_input_chars} prompt_chars={len(prompt)}"
            )
        return prompt

    async def extract_triples(self, user_messages: List[str]) -> List[Dict[str, str]]:
        if not user_messages:
            return []

        prompt = self._build_prompt(user_messages)

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
            logger.debug(f"[LLM Facts] generate_once failed: {e}")
            return []

        if not isinstance(text, str) or not text.strip():
            return []

        # Attempt to parse a JSON array (robust to leading/trailing junk)
        raw = text.strip()
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
        except Exception:
            logger.debug("[LLM Facts] JSON parse failed")
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

        logger.info(f"[LLM Facts] extracted={len(triples)} model={self.model_alias}")
        return triples
