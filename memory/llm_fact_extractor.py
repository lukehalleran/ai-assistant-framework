"""
# memory/llm_fact_extractor.py

Additive LLM-assisted fact extractor used at shutdown to augment regex/spaCy/REBEL facts.

Contract
- Inputs: list of recent user-only messages (strings), model_manager
- Behavior: calls a compact LLM prompt to extract SRO triples as strict JSON
- Output: list of dict triples (subject, relation, object)
"""

from __future__ import annotations

from typing import List, Dict, Any
import json
import re
import os

from utils.logging_utils import get_logger

logger = get_logger("llm_facts")


def _snake(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")


def _normalize_triple(t: Dict[str, Any]) -> Dict[str, str] | None:
    # Coerce to strings defensively; models sometimes emit numbers/bools
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or "").strip()
    if not subj or not rel or not obj:
        return None
    # Pronouns → user
    if subj.lower() in {"i", "me", "my", "we", "us", "you"}:
        subj = "user"
    # Relation snake_case
    rel = _snake(rel)
    # Lowercase object for stability, but keep short
    obj = obj.strip().strip(". ")
    return {"subject": subj.lower(), "relation": rel, "object": obj.lower()}


class LLMFactExtractor:
    def __init__(self, model_manager, *,
                 model_alias: str = None,
                 max_input_chars: int = 4000,
                 max_triples: int = 10):
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
        sys = (
            "You convert user messages into factual triples only. No speculation. "
            "Subject pronouns referring to the user become 'user'. Output ONLY strict JSON array."
        )
        rules = (
            "Return ONLY strict JSON (array of objects), no prose.\n"
            "Rules:\n- Use objects with keys: subject, relation, object\n"
            "- relation in snake_case; short strings; no reasoning\n"
            "- Only include facts present in the text; ignore opinions\n"
        )
        prompt = (
            f"{sys}\n\n{rules}\n\nMESSAGES (user-only, newest last):\n{joined}\n\nJSON:"
        )
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
                system_prompt="You output only strict JSON arrays of triples.",
                max_tokens=400,
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
