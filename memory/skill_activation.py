# memory/skill_activation.py
"""
Procedural skill activation policy and cooldown tracking.

Takes candidate procedural skills from existing retrieval and applies
intent-aware filtering, relevance thresholding, and cooldown to surface
only the skills that are useful for the current conversational context.

Inputs:
    - candidates: List[Dict] from MemoryRetriever.get_skills()
    - intent_type: Optional[str] (e.g. "TECHNICAL_HELP", "EMOTIONAL_SUPPORT")
    - stm_topics: Optional[List[str]] from STM analysis

Outputs:
    - List[Dict] — filtered/reranked skills, capped to max_skills

Side effects:
    - SkillCooldownStore persists shown skill IDs to data/skill_cooldown.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from utils.logging_utils import get_logger

logger = get_logger("skill_activation")

# Intent types that should never receive procedural skills.
# Surfacing workflows during emotional distress or casual greetings is tone-deaf.
_SUPPRESSED_INTENTS: Set[str] = {"EMOTIONAL_SUPPORT", "CASUAL_SOCIAL"}


class SkillCooldownStore:
    """JSON-backed TTL store tracking which skills were recently surfaced.

    Modeled after memory/surfacing_history.py.  Each entry stores the
    skill's last-surfaced ISO timestamp and a running count.
    """

    def __init__(self, persist_path: str = "data/skill_cooldown.json"):
        self.persist_path = persist_path
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_cooling_down(self, skill_id: str, cooldown_hours: float = 48.0) -> bool:
        """Return True if *skill_id* was surfaced within the cooldown window."""
        entry = self._entries.get(skill_id)
        if not entry:
            return False
        try:
            last = datetime.fromisoformat(entry["last_surfaced"])
            return datetime.now() - last < timedelta(hours=cooldown_hours)
        except (KeyError, ValueError):
            return False

    def record_surfaced(self, skill_id: str) -> None:
        """Record that a skill was surfaced to the user."""
        entry = self._entries.get(skill_id, {"count": 0})
        entry["last_surfaced"] = datetime.now().isoformat()
        entry["count"] = entry.get("count", 0) + 1
        self._entries[skill_id] = entry
        self.save()

    def cleanup_old(self, max_age_days: int = 30) -> int:
        """Remove entries older than *max_age_days*.  Returns count removed."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        to_remove: List[str] = []
        for key, entry in self._entries.items():
            try:
                last = datetime.fromisoformat(entry.get("last_surfaced", ""))
                if last < cutoff:
                    to_remove.append(key)
            except (ValueError, TypeError):
                to_remove.append(key)

        for key in to_remove:
            del self._entries[key]

        if to_remove:
            self.save()
            logger.debug(f"[SkillCooldownStore] Cleaned up {len(to_remove)} stale entries")
        return len(to_remove)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load cooldown state from disk."""
        if not os.path.exists(self.persist_path):
            self._entries = {}
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[SkillCooldownStore] Failed to load {self.persist_path}: {e}")
            self._entries = {}

    def save(self) -> None:
        """Persist cooldown state to disk."""
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"[SkillCooldownStore] Failed to save {self.persist_path}: {e}")


class SkillActivationPolicy:
    """Post-retrieval filter that decides which procedural skills to surface.

    Designed to sit between the existing ChromaDB retrieval and prompt
    injection — not a new pipeline stage.

    Filtering order:
        1. Intent suppression  (EMOTIONAL_SUPPORT / CASUAL_SOCIAL → [])
        2. Minimum relevance score threshold
        3. STM topic bonus (rerank)
        4. Cooldown filter
        5. Cap to max_skills
    """

    def __init__(
        self,
        cooldown_store: Optional[SkillCooldownStore] = None,
        min_score: float = 0.25,
        cooldown_hours: float = 48.0,
        max_skills: int = 3,
        stm_bonus: float = 0.10,
        enabled: bool = True,
    ):
        self.cooldown_store = cooldown_store
        self.min_score = min_score
        self.cooldown_hours = cooldown_hours
        self.max_skills = max_skills
        self.stm_bonus = stm_bonus
        self.enabled = enabled

    def filter(
        self,
        candidates: List[Dict[str, Any]],
        intent_type: Optional[str] = None,
        stm_topics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter and rerank candidate procedural skills.

        Args:
            candidates: Raw results from MemoryRetriever.get_skills()
            intent_type: Current intent (string value of IntentType enum)
            stm_topics: Topics extracted by STM analysis

        Returns:
            Filtered list of skill dicts, at most self.max_skills items.
        """
        if not self.enabled:
            return []
        if not candidates:
            return []

        # 1. Intent suppression
        if intent_type and intent_type in _SUPPRESSED_INTENTS:
            logger.debug(f"[SkillActivation] Suppressed: intent={intent_type}")
            return []

        # 2. Minimum relevance score threshold
        scored = []
        for skill in candidates:
            score = skill.get("relevance_score", 0.0)
            if score < self.min_score:
                continue
            scored.append((score, skill))

        if not scored:
            logger.debug("[SkillActivation] No candidates above min_score threshold")
            return []

        # 3. STM topic bonus — boost skills whose tags or trigger match STM topics
        if stm_topics:
            topics_lower = {t.lower() for t in stm_topics if t}
            boosted = []
            for score, skill in scored:
                bonus = self._compute_stm_bonus(skill, topics_lower)
                boosted.append((score + bonus, skill))
            scored = boosted

        # Sort by adjusted score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # 4. Cooldown filter
        filtered: List[Dict[str, Any]] = []
        for _score, skill in scored:
            if len(filtered) >= self.max_skills:
                break
            skill_id = skill.get("id", "")
            if skill_id and self.cooldown_store and self.cooldown_store.is_cooling_down(
                skill_id, self.cooldown_hours
            ):
                logger.debug(f"[SkillActivation] Cooldown skip: {skill_id}")
                continue
            filtered.append(skill)

        # 5. Record surfaced skills in cooldown store
        if self.cooldown_store:
            for skill in filtered:
                skill_id = skill.get("id", "")
                if skill_id:
                    self.cooldown_store.record_surfaced(skill_id)

        logger.debug(
            f"[SkillActivation] {len(candidates)} candidates → {len(filtered)} activated "
            f"(intent={intent_type}, stm_topics={len(stm_topics or [])})"
        )
        return filtered

    def _compute_stm_bonus(
        self, skill: Dict[str, Any], topics_lower: set
    ) -> float:
        """Compute relevance bonus based on STM topic overlap with skill metadata."""
        if not topics_lower:
            return 0.0

        meta = skill.get("metadata", {})
        trigger = (meta.get("trigger", "") or "").lower()
        tags_raw = meta.get("tags_json", "")
        content = (skill.get("content", "") or "").lower()

        # Parse tags
        tags: List[str] = []
        if tags_raw:
            try:
                parsed = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                tags = [t.lower() for t in parsed] if isinstance(parsed, list) else []
            except Exception:
                pass

        # Check for any topic match in trigger, tags, or content
        match_fields = " ".join([trigger] + tags + [content])
        for topic in topics_lower:
            if topic in match_fields:
                return self.stm_bonus

        return 0.0
