"""
# memory/synthesis_memory.py

Module Contract
- Purpose: Persistent storage for synthesis results with convergence tracking.
  Uses the 'synthesis_results' ChromaDB collection via MultiCollectionChromaStore.
- Class: SynthesisMemory(chroma_store, similarity_threshold)
- Key methods:
  - find_similar(connection_claim, threshold, limit) -> List[Tuple[SynthesisResult, float]]
  - store_result(result) -> str  [returns doc_id; deduplicates via convergence update]
  - get_recurring(min_paths, min_sources) -> List[SynthesisResult]
  - get_stats() -> dict
- Key behaviors:
  - Uses MultiCollectionChromaStore.query_collection() (flat dict results)
  - Convergence: same insight from different paths merges unique_paths/unique_sources
  - Promotes to CONVERGING status when paths >= 3 and sources >= 2
- Side effects: Reads/writes 'synthesis_results' ChromaDB collection
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from knowledge.synthesis_models import (
    CandidateStatus,
    SynthesisResult,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SynthesisMemory:
    """Persistent storage for synthesis results with convergence tracking.

    Uses ChromaDB collection 'synthesis_results' for vector storage and
    metadata-based convergence detection.
    """

    COLLECTION_NAME = "synthesis_results"
    DEFAULT_SIMILARITY_THRESHOLD = 0.85  # above this = same insight

    def __init__(self, chroma_store, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.store = chroma_store
        self.similarity_threshold = similarity_threshold

    def find_similar(
        self,
        connection_claim: str,
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> List[Tuple[SynthesisResult, float]]:
        """Find existing results similar to a connection claim.

        Returns list of (SynthesisResult, similarity_score) sorted by similarity desc.
        """
        threshold = threshold or self.similarity_threshold
        try:
            results = self.store.query_collection(
                collection_name=self.COLLECTION_NAME,
                query_text=connection_claim,
                n_results=limit,
            )
            if not results:
                return []

            matches = []
            for item in results:
                score = item.get("relevance_score")
                if score is None:
                    continue
                # query_collection returns relevance_score = 1/(1+distance)
                # Convert back: distance = (1/score) - 1, similarity = 1 - distance
                distance = (1.0 / score) - 1.0 if score > 0 else 1.0
                similarity = 1.0 - distance
                if similarity >= threshold:
                    doc = item.get("content", "")
                    metadata = item.get("metadata", {})
                    result = SynthesisResult.from_metadata(metadata, doc)
                    matches.append((result, similarity))

            return sorted(matches, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error finding similar synthesis results: {e}")
            return []

    def store_result(self, result: SynthesisResult) -> str:
        """Store a new synthesis result. Returns the document ID.

        Before storing, checks for existing similar results. If found,
        updates convergence tracking instead of creating a duplicate.
        """
        similar = self.find_similar(result.candidate.connection_claim)

        if similar:
            existing_result, similarity = similar[0]
            logger.info(
                f"Synthesis insight already exists (similarity={similarity:.3f}). "
                f"Updating convergence tracking instead of creating duplicate."
            )
            return self._update_convergence(existing_result, result)

        # New insight -- store it
        result.unique_paths.add(result.candidate.path_hash)
        source_key = f"{result.candidate.concept_a}|{result.candidate.concept_b}"
        result.unique_sources.add(source_key)
        result.convergence_strength = len(result.unique_paths) * len(result.unique_sources)

        doc_text = result.candidate.connection_claim
        metadata = result.to_metadata()

        try:
            doc_id = self.store.add_to_collection(
                name=self.COLLECTION_NAME,
                text=doc_text,
                metadata=metadata,
            )
            logger.info(f"Stored new synthesis result: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to store synthesis result: {e}")
            raise

    def _update_convergence(self, existing: SynthesisResult, new: SynthesisResult) -> str:
        """Update convergence tracking on an existing result with evidence from a new discovery."""
        existing.unique_paths.add(new.candidate.path_hash)
        source_key = f"{new.candidate.concept_a}|{new.candidate.concept_b}"
        existing.unique_sources.add(source_key)
        existing.convergence_strength = len(existing.unique_paths) * len(existing.unique_sources)

        # Promote status if convergence is strong
        was_converging = existing.status == CandidateStatus.CONVERGING
        if len(existing.unique_paths) >= 3 and len(existing.unique_sources) >= 2:
            existing.status = CandidateStatus.CONVERGING
            if not was_converging:
                logger.warning(
                    f"[SYNTH CONVERGING] "
                    f"concepts={existing.candidate.concept_a}<->{existing.candidate.concept_b} | "
                    f"paths={len(existing.unique_paths)} sources={len(existing.unique_sources)} | "
                    f"claim={existing.candidate.connection_claim[:120]}"
                )

        # Find the existing doc ID and update metadata
        try:
            search = self.store.query_collection(
                collection_name=self.COLLECTION_NAME,
                query_text=existing.candidate.connection_claim,
                n_results=1,
            )
            if search:
                doc_id = search[0].get("id", "")
                if doc_id:
                    self.store.update_metadata(
                        collection_name=self.COLLECTION_NAME,
                        doc_id=doc_id,
                        metadata_updates=existing.to_metadata(),
                    )
                    logger.info(
                        f"Updated convergence for {doc_id}: "
                        f"paths={len(existing.unique_paths)}, "
                        f"sources={len(existing.unique_sources)}, "
                        f"strength={existing.convergence_strength}"
                    )
                    return doc_id
        except Exception as e:
            logger.error(f"Failed to update convergence: {e}")

        return ""

    def get_recurring(self, min_paths: int = 3, min_sources: int = 2) -> List[SynthesisResult]:
        """Get synthesis results that show strong convergence signal."""
        try:
            results = self.store.query_collection(
                collection_name=self.COLLECTION_NAME,
                query_text="cross-domain connection",  # broad semantic query
                n_results=100,
            )
            if not results:
                return []

            converging = []
            for item in results:
                metadata = item.get("metadata", {})
                if metadata.get("status") != CandidateStatus.CONVERGING.value:
                    continue
                doc = item.get("content", "")
                result = SynthesisResult.from_metadata(metadata, doc)
                if len(result.unique_paths) >= min_paths and len(result.unique_sources) >= min_sources:
                    converging.append(result)

            return sorted(converging, key=lambda r: r.convergence_strength, reverse=True)
        except Exception as e:
            logger.error(f"Error fetching recurring results: {e}")
            return []

    def get_stats(self) -> dict:
        """Return summary stats about synthesis memory."""
        try:
            stats = self.store.get_collection_stats()
            coll_stats = stats.get(self.COLLECTION_NAME, {})
            count = coll_stats.get("count", 0)
            converging = len(self.get_recurring())
            return {
                "total_insights": count,
                "converging_insights": converging,
                "collection": self.COLLECTION_NAME,
            }
        except Exception:
            return {"total_insights": 0, "converging_insights": 0, "collection": self.COLLECTION_NAME}
