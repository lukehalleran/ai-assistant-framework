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
  - store.record(domain, label, text, source, embedder=None, seed_texts=None,
                 support_span=None)
      -> bool — applies quality gates before storing:
      * length bounds (min 12 chars; over 300 chars a window of <=300 chars
        containing `support_span` — the (start, end) offsets, in the text as
        passed, of the text the teacher's verdict rested on — is stored;
        an over-long text with no usable span is DECLINED, never head-clipped:
        2026-10-23 A06, a neutral head would be taught under a distress label)
      * exact-duplicate check vs seeds + learned (case-folded)
      * near-duplicate check vs seeds + learned when an embedder is provided
        (cosine >= 0.92 rejected — repeats must not dominate the prototype)
      * per-label cap (40) — oldest evicted
  - store.version — increments on every accepted record; callers cache
    derived artifacts (exemplar prototype embeddings) keyed on it.
  - encode_texts_cached(embedder, texts, cache, normalize=False) — shared
    per-text embedding cache (2026-08-21): adopters re-encode only NEW texts
    on a version bump instead of the whole seed+learned set. The cache dict is
    owned by the caller, one per module (never share across embedding spaces).
- Poisoning guard (callers' contract): only record labels confirmed by a
  deterministic or independent channel. Never record from the check's own
  semantic verdict (self-reinforcement) and never record "safe"/negative
  labels from silence (a missed detection would then TEACH the miss).
"""

import contextvars
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import utils.safe_json as safe_json
import utils.test_envelope as test_envelope
from utils.bootstrap import store_path
from utils.logging_utils import get_logger

logger = get_logger("adaptive_exemplars")

_STORE_PATH = store_path("data/adaptive_exemplars.json")

_MIN_CHARS = 12
_CLIP_CHARS = 300
_PER_LABEL_CAP = 40
_NEAR_DUP_COSINE = 0.92


# Probe-turn flag (2026-10-18, BC-58): handlers ingress unwraps an owner
# "[test]...[/test]" probe BEFORE the classifiers run, so intent/tone/need/
# gate/web-trigger teachers hand record() the INNER text and the envelope
# check on the text alone cannot see it. The ingress marks the whole turn
# instead; record() -- the one write chokepoint -- refuses while it is set.
_PROBE_TURN: contextvars.ContextVar = contextvars.ContextVar(
    "adaptive_exemplars_probe_turn", default=False
)


def mark_probe_turn(active: bool) -> contextvars.Token:
    """Mark (or unmark) the current turn context as an owner probe."""
    return _PROBE_TURN.set(bool(active))


def end_probe_turn(token: contextvars.Token) -> None:
    """Restore the flag set by mark_probe_turn; never raises (a generator
    closed from another task cannot reset a foreign context's token)."""
    try:
        _PROBE_TURN.reset(token)
    except ValueError:  # degrades: flag stays set in a context that is discarded with the turn
        pass


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
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                if isinstance(raw, dict):
                    self._data = self._validated(raw)
                    n = sum(len(v) for d in self._data.values() for v in d.values())
                    logger.info(f"[AdaptiveExemplars] Loaded {n} learned exemplars")
        except Exception as e:
            logger.warning(f"[AdaptiveExemplars] Load skipped (starting empty): {e}")
            self._data = {}

    @staticmethod
    def _validated(raw: dict) -> Dict[str, Dict[str, List[dict]]]:
        """Keep only well-shaped entries (A09, 2026-10-23): the store is
        derived, self-healing state, so a malformed domain/label/entry is
        dropped (debug log) instead of poisoning get_learned() with a KeyError."""
        out: Dict[str, Dict[str, List[dict]]] = {}
        dropped = 0
        for domain, labels in raw.items():
            if not isinstance(domain, str) or not isinstance(labels, dict):
                dropped += 1
                continue
            for label, entries in labels.items():
                if not isinstance(label, str) or not isinstance(entries, list):
                    dropped += 1
                    continue
                good = []
                for e in entries:
                    if (
                        isinstance(e, dict)
                        and isinstance(e.get("text"), str)
                        and e["text"].strip()
                    ):
                        good.append(e)
                    else:
                        dropped += 1
                if good:
                    out.setdefault(domain, {})[label] = good
        if dropped:
            logger.debug(f"[AdaptiveExemplars] Dropped {dropped} malformed entries on load")
        return out

    def _save(self) -> None:
        try:
            safe_json.atomic_write_json(str(self._path), self._data)
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
        support_span: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Store a confirmed exemplar after quality gates. Returns True if kept."""
        # Owner probes teach nothing: a "[test]...[/test]" envelope in the
        # text, or a turn the ingress marked as a probe (stripped text).
        if _PROBE_TURN.get() or test_envelope.has_envelope(text or ""):
            return False
        t = " ".join((text or "").split()).strip()
        if len(t) < _MIN_CHARS:
            return False
        if len(t) > _CLIP_CHARS:
            _full_len = len(t)
            t = _support_window(text or "", support_span)
            if t is None:
                # Logged at INFO (was DEBUG, 2026-10-26): a teacher that keeps
                # passing long spanless text is a wiring gap the owner should be
                # able to see, not a silent decline.
                logger.info(
                    f"[AdaptiveExemplars] {domain}/{label} not learned: "
                    f"{_full_len}-char text exceeds {_CLIP_CHARS} with no usable "
                    f"supporting span (source={source})"
                )
                return False
            if len(t) < _MIN_CHARS:
                return False

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


def _support_window(text: str, span: Optional[Tuple[int, int]]) -> Optional[str]:
    """Whitespace-normalised window of <= _CLIP_CHARS chars, on word
    boundaries, centred on / containing `span` (character offsets into `text`
    as given). None when there is no usable span. A span wider than the clip
    keeps its own start (the window cannot contain it whole)."""
    try:
        s0, e0 = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (0 <= s0 < e0 <= len(text)):
        return None
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    idx = [i for i, (a, b, _) in enumerate(words) if a < e0 and b > s0]
    if not idx:
        return None
    lo, hi = idx[0], idx[-1]

    def width(l: int, h: int) -> int:
        return sum(len(w[2]) for w in words[l:h + 1]) + (h - l)

    while hi > lo and width(lo, hi) > _CLIP_CHARS:
        hi -= 1  # span wider than the clip: keep its start
    take_left = True
    while True:
        grew = False
        for side in ("L", "R") if take_left else ("R", "L"):
            if side == "L" and lo > 0 and width(lo - 1, hi) <= _CLIP_CHARS:
                lo -= 1
                grew = True
            elif side == "R" and hi < len(words) - 1 and width(lo, hi + 1) <= _CLIP_CHARS:
                hi += 1
                grew = True
        if not grew:
            break
    out = " ".join(w[2] for w in words[lo:hi + 1])
    return out[:_CLIP_CHARS] if out else None


_store: Optional[AdaptiveExemplarStore] = None


def encode_texts_cached(embedder, texts, cache: dict, normalize: bool = False):
    """Encode `texts`, re-using per-text cached vectors — only NEW texts hit
    the encoder.

    Adopters' prototype caches are keyed on store.version, so every accepted
    exemplar used to recompute the WHOLE seed+learned embedding set (the
    "one-time setup" log line ran on every learning event); with a per-text
    cache a version bump re-encodes only the newly learned texts (2026-08-21).

    cache is a dict OWNED BY THE CALLER, one per module — different adopters
    embed in different spaces (normalized vs raw, potentially different
    models); never share a cache across embedders. Returns a
    (len(texts), dim) numpy array in input order.
    """
    if not texts:
        return np.zeros((0, 384))
    missing = [t for t in texts if t not in cache]
    if missing:
        if normalize:
            embs = embedder.encode(
                missing, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            embs = embedder.encode(missing, convert_to_numpy=True)
        for t, e in zip(missing, embs):
            cache[t] = e
    return np.stack([cache[t] for t in texts])


def get_store() -> AdaptiveExemplarStore:
    global _store
    if _store is None:
        _store = AdaptiveExemplarStore()
    return _store
