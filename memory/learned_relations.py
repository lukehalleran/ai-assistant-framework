"""
# memory/learned_relations.py

Module Contract
- Purpose: Auto-growing relation vocabulary for the LLM fact extractor. When
  the extractor invents a non-core relation and the triple SURVIVES the
  normalization/junk gates, its usage is recorded here; relations that recur
  across enough distinct days are PROMOTED into the extractor prompt's
  preferred-relations list automatically — no owner vocab edits required.
  (2026-08-05: `doctor_communication`-class facts were never captured partly
  because the fixed core vocabulary had no slot for them; hand-adding vocab
  after each observed miss doesn't scale.)
- Class: LearnedRelationStore(path=None)
  - record(relation, obj="", when=None) -> bool — normalize + shape-guard the
    relation, then count the usage (per distinct day). Returns True if recorded.
  - promoted(min_days=None, cap=MAX_PROMOTED) -> List[str] — relations seen on
    >= min_days distinct days, most-established first.
- Poisoning/junk-floodgate guards (the 08-02 finding was 630/696 stored
  relations being single-use inventions — promotion must not reopen that):
  - Only relations whose triples PASSED the deployed extraction gates are
    recorded (caller hooks post-normalization in llm_fact_extractor).
  - entity_resolver.normalize_relation() runs first — a relation that
    collapses into the core vocabulary is never tracked separately.
  - Shape guard: snake_case, <= 3 tokens, letters/digits only.
  - Ephemeral relations (relation_classifier) are never tracked — transient
    state must not become vocabulary.
  - Promotion needs recurrence on >= LEARNED_RELATIONS_MIN_DAYS distinct days
    (default 3), so one chatty session can't promote anything.
  - Promoted list capped at MAX_PROMOTED; tracked map capped at MAX_TRACKED
    (least-recently-seen evicted).
- Persistence: data/learned_relations.json — derived, self-healing state:
  atomic write (utils.safe_json.atomic_write_json), lenient load (corrupt or
  missing file = cold start, never an abort).
- Env: LEARNED_RELATIONS_ENABLED (default 1), LEARNED_RELATIONS_MIN_DAYS
  (default 3).
- Tests: tests/unit/test_learned_relations.py (store sandboxed via
  tests/conftest.py autouse fixture, same pattern as adaptive_exemplars).
"""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from utils.logging_utils import get_logger
from utils.safe_json import atomic_write_json

logger = get_logger("learned_relations")

_STORE_PATH = "data/learned_relations.json"

MAX_PROMOTED = 15
MAX_TRACKED = 200
_MAX_DAYS_KEPT = 30
_MAX_EXAMPLES = 3

_SHAPE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+){0,2}$")


def _min_days() -> int:
    try:
        return max(2, int(os.getenv("LEARNED_RELATIONS_MIN_DAYS", "3")))
    except (ValueError, TypeError):
        return 3


def _enabled() -> bool:
    return os.getenv("LEARNED_RELATIONS_ENABLED", "1") == "1"


class LearnedRelationStore:
    def __init__(self, path: Optional[str] = None):
        # Resolve the module global at call time (tests monkeypatch _STORE_PATH)
        self._path = Path(path or _STORE_PATH)
        self._data = {"version": 1, "relations": {}}
        try:
            if self._path.exists():
                import json
                raw = json.loads(self._path.read_text())
                if isinstance(raw, dict) and isinstance(raw.get("relations"), dict):
                    self._data = raw
        except Exception as e:  # lenient load: corrupt file = cold start
            logger.warning(f"[LearnedRelations] cold start (unreadable store): {e}")

    def _save(self) -> None:
        try:
            atomic_write_json(str(self._path), self._data)
        except Exception as e:  # best-effort persistence, never raises
            logger.warning(f"[LearnedRelations] save failed: {e}")

    @staticmethod
    def _acceptable(relation: str) -> Optional[str]:
        """Normalize + guard; return the trackable relation name or None."""
        rel = (relation or "").strip().lower()
        if not rel or not _SHAPE_RE.match(rel):
            return None
        try:
            from memory.entity_resolver import normalize_relation
            rel = normalize_relation(rel) or rel
        except Exception:
            pass
        # lazy import: cycle (llm_fact_extractor imports this store at call time)
        from memory.llm_fact_extractor import CORE_RELATIONS
        if rel in CORE_RELATIONS:
            return None
        try:
            from memory.relation_classifier import is_ephemeral_relation
            if is_ephemeral_relation(rel):
                return None
        except Exception:
            pass
        if not _SHAPE_RE.match(rel):  # normalization may have reshaped it
            return None
        return rel

    def record(self, relation: str, obj: str = "", when: Optional[datetime] = None) -> bool:
        if not _enabled():
            return False
        rel = self._acceptable(relation)
        if rel is None:
            return False
        day = (when or datetime.now()).date().isoformat()
        rels = self._data["relations"]
        row = rels.setdefault(rel, {"count": 0, "days": [], "last_seen": day, "examples": []})
        row["count"] = int(row.get("count", 0)) + 1
        days = row.get("days", [])
        if day not in days:
            days.append(day)
            row["days"] = days[-_MAX_DAYS_KEPT:]
        row["last_seen"] = day
        obj = (obj or "").strip()[:80]
        if obj and obj not in row.get("examples", []):
            row["examples"] = (row.get("examples", []) + [obj])[-_MAX_EXAMPLES:]
        # Evict least-recently-seen beyond the tracking cap.
        if len(rels) > MAX_TRACKED:
            by_seen = sorted(rels.items(), key=lambda kv: kv[1].get("last_seen", ""))
            for k, _ in by_seen[: len(rels) - MAX_TRACKED]:
                del rels[k]
        self._save()
        return True

    def promoted(self, min_days: Optional[int] = None, cap: int = MAX_PROMOTED) -> List[str]:
        if not _enabled():
            return []
        need = min_days if min_days is not None else _min_days()
        rows = [
            (rel, row) for rel, row in self._data["relations"].items()
            if len(row.get("days", [])) >= need
        ]
        rows.sort(key=lambda kv: (len(kv[1].get("days", [])), kv[1].get("last_seen", "")), reverse=True)
        return [rel for rel, _ in rows[:cap]]


_store: Optional[LearnedRelationStore] = None


def get_learned_relation_store() -> LearnedRelationStore:
    global _store
    if _store is None:
        _store = LearnedRelationStore()
    return _store
