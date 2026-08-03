# utils/adaptive_exemplars.py
"""
AdaptiveExemplarStore — seed + learned exemplar phrases for semantic checks.

Module Contract:
- Purpose: end the hand-maintained-keyword-list treadmill. Semantic checks
  (tone detection first; need detection / gate cues can adopt the same store)
  ship with SEED exemplars in code and GROW per-user from confirmed live
  classifications: when a deterministic channel (keyword stage) or the LLM
  arbiter confirms a label for a message, that message becomes an exemplar,
  so the semantic channel catches future PARAPHRASES that share no keywords
  (2026-08-02: three borderline tone misses in one day were each fixed by
  hand-adding exemplars — this store makes that loop automatic).
- Storage: data/adaptive_exemplars.json —
  {domain: {label: [{"text": str, "ts": iso, "source": str}]}}.
  Derived, self-healing state: atomic writes (utils.safe_json), LENIENT load
  (corrupt/missing file = empty store, never a startup abort — unlike real
  user-data stores).
- API:
  - get_store() -> AdaptiveExemplarStore (process singleton)
  - store.get_learned(domain, label) -> list[str]
  - store.record(domain, label, text, source, embedder=None, seed_texts=None)
      -> bool — applies quality gates before storing:
      * length bounds (min 12 chars, clipped to 300)
      * exact-duplicate check vs seeds + learned (case-folded)
      * near-duplicate check vs seeds + learned when an embedder is provided
        (cosine >= 0.92 rejected — repeats must not dominate the prototype)
      * per-label cap (40) — oldest evicted
  - store.version — increments on every accepted record; callers cache
    derived artifacts (exemplar prototype embeddings) keyed on it.
- Poisoning guard (callers' contract): only record labels confirmed by a
  deterministic or independent channel. Never record from the check's own
  semantic verdict (self-reinforcement) and never record "safe"/negative
  labels from silence (a missed detection would then TEACH the miss).
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("adaptive_exemplars")

_STORE_PATH = "data/adaptive_exemplars.json"

_MIN_CHARS = 12
_CLIP_CHARS = 300
_PER_LABEL_CAP = 40
_NEAR_DUP_COSINE = 0.92


class AdaptiveExemplarStore:
    def __init__(self, path: Optional[str] = None):
        # Resolve the module global at call time (tests monkeypatch _STORE_PATH)
        self._path = Path(path or _STORE_PATH)
        self._data: Dict[str, Dict[str, List[dict]]] = {}
        self.version = 0
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        try:
            import json
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                if isinstance(raw, dict):
                    self._data = raw
                    n = sum(len(v) for d in raw.values() for v in d.values())
                    logger.info(f"[AdaptiveExemplars] Loaded {n} learned exemplars")
        except Exception as e:
            logger.warning(f"[AdaptiveExemplars] Load skipped (starting empty): {e}")
            self._data = {}

    def _save(self) -> None:
        try:
            from utils.safe_json import atomic_write_json
            atomic_write_json(str(self._path), self._data)
        except Exception as e:
            logger.warning(f"[AdaptiveExemplars] Save failed (non-fatal): {e}")

    # -- reads ------------------------------------------------------------
    def get_learned(self, domain: str, label: str) -> List[str]:
        return [e["text"] for e in self._data.get(domain, {}).get(label, [])]

    # -- writes -----------------------------------------------------------
    def record(
        self,
        domain: str,
        label: str,
        text: str,
        source: str,
        embedder=None,
        seed_texts: Optional[List[str]] = None,
    ) -> bool:
        """Store a confirmed exemplar after quality gates. Returns True if kept."""
        t = " ".join((text or "").split()).strip()
        if len(t) < _MIN_CHARS:
            return False
        t = t[:_CLIP_CHARS]

        existing = self.get_learned(domain, label)
        pool = [*(seed_texts or []), *existing]
        low = t.lower()
        if any(low == p.lower() for p in pool):
            return False

        if embedder is not None and pool:
            try:
                vecs = embedder.encode([t] + pool, convert_to_numpy=True)
                q = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
                others = vecs[1:]
                others = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-9)
                if float(np.max(others @ q)) >= _NEAR_DUP_COSINE:
                    return False
            except Exception as e:
                logger.debug(f"[AdaptiveExemplars] near-dup check skipped: {e}")

        entries = self._data.setdefault(domain, {}).setdefault(label, [])
        entries.append({"text": t, "ts": datetime.now().isoformat(), "source": source})
        if len(entries) > _PER_LABEL_CAP:
            del entries[: len(entries) - _PER_LABEL_CAP]
        self.version += 1
        self._save()
        logger.info(
            f"[AdaptiveExemplars] Learned {domain}/{label} exemplar "
            f"(source={source}, n={len(entries)}): {t[:60]!r}"
        )
        return True


_store: Optional[AdaptiveExemplarStore] = None


def get_store() -> AdaptiveExemplarStore:
    global _store
    if _store is None:
        _store = AdaptiveExemplarStore()
    return _store
