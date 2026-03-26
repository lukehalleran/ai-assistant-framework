# memory/fact_verification.py
"""
Fact Verification Gate — intercepts newly extracted facts before ChromaDB
storage, checks against existing facts for contradictions, and returns a
verdict: STORE, STORE_AND_FLAG (marks old fact as superseded), REJECT, or
SKIP (ephemeral relation).

No auto-deletion — conflicting old facts get ``superseded_by`` metadata.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("fact_verification")


# ── Enums & Models ──────────────────────────────────────────────────

class FactVerdict(str, Enum):
    """Outcome of the verification gate."""
    STORE = "store"                  # No conflict — proceed normally
    STORE_AND_FLAG = "store_and_flag"  # Store new fact, mark old as superseded
    REJECT = "reject"                # Likely extraction error — discard
    SKIP = "skip"                    # Ephemeral relation — store without checking


class ConflictCandidate(BaseModel):
    """An existing fact that may conflict with the new one."""
    doc_id: str = ""
    content: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.0
    timestamp: str = ""


class VerificationResult(BaseModel):
    """Result of the fact verification gate."""
    verdict: FactVerdict = FactVerdict.STORE
    confidence: float = 1.0
    reason: str = ""
    conflicting_candidates: List[ConflictCandidate] = Field(default_factory=list)
    metadata_updates: dict = Field(default_factory=dict)


# ── Verifier ────────────────────────────────────────────────────────

class FactVerifier:
    """Pre-storage gate that checks new facts against existing ones.

    Flow::

        verify(subject, predicate, object_val, ...) →
          1. Ephemeral relation? → SKIP
          2. Query ChromaDB for candidates with matching subject+predicate
          3. No candidates? → STORE (fast path)
          4. Candidates have same object? → STORE (re-confirmation)
          5. User-stated source + confidence >= threshold? → STORE_AND_FLAG
          6. Entity-scope + low confidence? → REJECT
          7. LLM adjudication (if available) → A/B/C
          8. Fallback → STORE_AND_FLAG (trust newer)
    """

    def __init__(self, chroma_store, model_manager=None):
        self._chroma = chroma_store
        self._model_manager = model_manager
        self._ephemeral: Optional[frozenset] = None  # lazy-loaded

    # ── public API ──────────────────────────────────────────────────

    async def verify(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        fact_text: str,
        source: str = "conversation",
        confidence: float = 0.7,
        fact_scope: str = "user",
    ) -> VerificationResult:
        """Verify a single fact before storage.

        Returns a :class:`VerificationResult` whose *verdict* tells the
        caller whether to store, flag, reject, or skip.
        """
        # Normalise inputs
        subject = (subject or "").strip().lower()
        predicate = (predicate or "").strip().lower()
        object_val = (object_val or "").strip()
        source = (source or "conversation").strip().lower()

        # Fast-path: ephemeral, no triple, etc.
        fast = self._fast_path_verdict(
            subject, predicate, object_val, fact_text, source, confidence, fact_scope,
        )
        if fast is not None:
            return fast

        # Find conflicting candidates in ChromaDB
        candidates = self._find_candidates(subject, predicate, fact_text)

        # No existing facts → safe to store
        if not candidates:
            return VerificationResult(
                verdict=FactVerdict.STORE,
                confidence=1.0,
                reason="no_existing_candidates",
            )

        # Check for confirmation (same object)
        for c in candidates:
            if self._objects_match(object_val, c.object):
                return VerificationResult(
                    verdict=FactVerdict.STORE,
                    confidence=1.0,
                    reason="confirmation_of_existing",
                )

        # ── Conflict detected ───────────────────────────────────────

        # User-stated / corrected facts with high confidence → trust user
        from config.app_config import FACT_VERIFICATION_USER_TRUST_THRESHOLD
        user_trust_sources = {"user_stated", "user_corrected", "correction"}
        if source in user_trust_sources and confidence >= FACT_VERIFICATION_USER_TRUST_THRESHOLD:
            return VerificationResult(
                verdict=FactVerdict.STORE_AND_FLAG,
                confidence=confidence,
                reason="user_trust_override",
                conflicting_candidates=candidates,
                metadata_updates={"superseded_reason": "user_correction"},
            )

        # Entity-scope facts with low confidence → reject
        if fact_scope == "entity" and confidence < 0.6:
            return VerificationResult(
                verdict=FactVerdict.REJECT,
                confidence=confidence,
                reason="entity_scope_low_confidence",
                conflicting_candidates=candidates,
            )

        # LLM adjudication (when available and enabled)
        from config.app_config import FACT_VERIFICATION_LLM_ENABLED
        if self._model_manager and FACT_VERIFICATION_LLM_ENABLED:
            return await self._adjudicate_conflict(
                subject, predicate, object_val, fact_text,
                source, confidence, candidates,
            )

        # Fallback: trust newer fact
        return VerificationResult(
            verdict=FactVerdict.STORE_AND_FLAG,
            confidence=confidence,
            reason="fallback_trust_newer",
            conflicting_candidates=candidates,
            metadata_updates={"superseded_reason": "newer_fact"},
        )

    async def verify_batch(self, facts: List[dict]) -> List[VerificationResult]:
        """Verify a batch of facts. Returns results in the same order."""
        results = []
        for f in facts:
            r = await self.verify(
                subject=f.get("subject", ""),
                predicate=f.get("predicate", f.get("relation", "")),
                object_val=f.get("object", ""),
                fact_text=f.get("fact_text", ""),
                source=f.get("source", "conversation"),
                confidence=float(f.get("confidence", 0.7)),
                fact_scope=f.get("fact_scope", "user"),
            )
            results.append(r)
        return results

    # ── internal helpers ────────────────────────────────────────────

    def _get_ephemeral_relations(self) -> frozenset:
        """Load ephemeral relation names (lazy, cached)."""
        if self._ephemeral is None:
            try:
                from config.app_config import PROFILE_EPHEMERAL_RELATIONS
                self._ephemeral = frozenset(
                    r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS
                )
            except ImportError:
                self._ephemeral = frozenset()
        return self._ephemeral

    def _fast_path_verdict(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        fact_text: str,
        source: str,
        confidence: float,
        fact_scope: str,
    ) -> Optional[VerificationResult]:
        """Return a verdict immediately when no DB query is needed."""
        # No parseable triple → skip verification
        if not subject or not predicate:
            return VerificationResult(
                verdict=FactVerdict.SKIP,
                confidence=1.0,
                reason="no_parseable_triple",
            )

        # Ephemeral relations (mood, activity, etc.) → skip
        if predicate in self._get_ephemeral_relations():
            return VerificationResult(
                verdict=FactVerdict.SKIP,
                confidence=1.0,
                reason="ephemeral_relation",
            )

        return None

    def _find_candidates(
        self, subject: str, predicate: str, fact_text: str,
    ) -> List[ConflictCandidate]:
        """Query ChromaDB facts collection for potential conflicts."""
        from config.app_config import FACT_VERIFICATION_MAX_CANDIDATES

        candidates: List[ConflictCandidate] = []
        try:
            coll = self._chroma.collections.get("facts")
            if coll is None or coll.count() == 0:
                return []

            # Semantic search using the fact text
            results = self._chroma.query_collection(
                "facts",
                query_text=fact_text,
                n_results=min(FACT_VERIFICATION_MAX_CANDIDATES, coll.count()),
            )

            from memory.cross_deduplicator import CrossCollectionDeduplicator

            for doc in results:
                doc_meta = doc.get("metadata", {})
                doc_content = doc.get("content", "")

                # Extract triple from existing doc
                s, p, o = CrossCollectionDeduplicator._extract_triple({
                    "content": doc_content,
                    "metadata": doc_meta,
                })

                s_lower = s.lower().strip()
                p_lower = p.lower().strip()

                # Only keep candidates with matching subject AND predicate
                if s_lower == subject and p_lower == predicate:
                    candidates.append(ConflictCandidate(
                        doc_id=doc.get("id", ""),
                        content=doc_content,
                        subject=s,
                        predicate=p,
                        object=o,
                        confidence=float(doc_meta.get("confidence", 0.5)),
                        timestamp=doc_meta.get("timestamp", ""),
                    ))

        except Exception as e:
            logger.warning(f"[FactVerifier] Candidate search failed: {e}")

        return candidates

    async def _adjudicate_conflict(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        fact_text: str,
        source: str,
        confidence: float,
        candidates: List[ConflictCandidate],
    ) -> VerificationResult:
        """Use a minimal LLM prompt to decide A (update), B (coexist), C (reject)."""
        from config.app_config import FACT_VERIFICATION_MODEL

        existing_lines = []
        for c in candidates[:3]:  # limit context
            existing_lines.append(
                f"- {c.subject} | {c.predicate} | {c.object} "
                f"(confidence={c.confidence:.2f})"
            )

        prompt = (
            "EXISTING FACTS:\n"
            + "\n".join(existing_lines)
            + f"\n\nNEW FACT: {subject} | {predicate} | {object_val} "
            f"(confidence={confidence:.2f}, source={source})\n\n"
            "Is the new fact:\n"
            "A) An UPDATE that supersedes the old fact\n"
            "B) COMPLEMENTARY information that should coexist\n"
            "C) IMPLAUSIBLE or likely an extraction error\n\n"
            "Reply with exactly one letter: A, B, or C"
        )

        try:
            raw = await self._model_manager.generate_once(
                prompt,
                model_name=FACT_VERIFICATION_MODEL,
                system_prompt="You are a fact verification assistant. Reply with exactly one letter.",
                max_tokens=4,
                temperature=0.0,
            )
            answer = (raw or "").strip().upper()[:1]
        except Exception as e:
            logger.warning(f"[FactVerifier] LLM adjudication failed: {e}")
            answer = ""

        if answer == "A":
            return VerificationResult(
                verdict=FactVerdict.STORE_AND_FLAG,
                confidence=confidence,
                reason="llm_update",
                conflicting_candidates=candidates,
                metadata_updates={"superseded_reason": "llm_adjudication_update"},
            )
        elif answer == "B":
            return VerificationResult(
                verdict=FactVerdict.STORE,
                confidence=confidence,
                reason="llm_complementary",
                conflicting_candidates=candidates,
            )
        elif answer == "C":
            return VerificationResult(
                verdict=FactVerdict.REJECT,
                confidence=confidence,
                reason="llm_implausible",
                conflicting_candidates=candidates,
            )
        else:
            # Unparseable response → trust newer
            return VerificationResult(
                verdict=FactVerdict.STORE_AND_FLAG,
                confidence=confidence,
                reason="llm_unparseable_fallback",
                conflicting_candidates=candidates,
                metadata_updates={"superseded_reason": "newer_fact"},
            )

    @staticmethod
    def _objects_match(new_obj: str, existing_obj: str) -> bool:
        """Check if two object values refer to the same thing."""
        a = new_obj.strip().lower()
        b = existing_obj.strip().lower()
        if not a or not b:
            return False
        return a == b
