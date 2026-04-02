"""
# memory/synthesis_memory.py

Module Contract
- Purpose: Persistent storage for synthesis results with convergence tracking
  and human audit queue for ground-truth grading.
  Uses the 'synthesis_results' ChromaDB collection via MultiCollectionChromaStore.
- Class: SynthesisMemory(chroma_store, similarity_threshold)
- Key methods:
  - find_similar(connection_claim, threshold, limit) -> List[Tuple[SynthesisResult, float]]
  - store_result(result) -> str  [returns doc_id; deduplicates via convergence update]
  - store_rejected_for_audit(result) -> str  [stores composite-rejected for FN review]
  - get_recurring(min_paths, min_sources) -> List[SynthesisResult]
  - grade_result(doc_id, grade, notes) -> bool  [human audit: valid/invalid/should_pass/correct_reject]
  - get_ungraded(status_filter, limit) -> List[Tuple[str, SynthesisResult]]  [audit queue items]
  - get_graded(limit) -> List[Tuple[str, SynthesisResult]]  [graded history]
  - get_audit_stats() -> dict  [FP rate, FN rate, total graded, auto-halt status]
  - get_stats() -> dict
- Key behaviors:
  - Uses MultiCollectionChromaStore.query_collection() (flat dict results)
  - Convergence: same insight from different paths merges unique_paths/unique_sources
  - Promotes to CONVERGING status when paths >= 3 and sources >= 2
  - Audit: stores composite-rejected candidates alongside accepted for review
  - Auto-halt: if FP rate > threshold with sufficient graded data, flags for halt
- Side effects: Reads/writes 'synthesis_results' ChromaDB collection
"""

from __future__ import annotations

from datetime import datetime
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

    def create_bridge_edge(
        self,
        result: SynthesisResult,
        graph_memory,
        entity_resolver,
    ) -> str:
        """Create a provisional bridge edge in the knowledge graph for an accepted result.

        Provisional bridges start at weight=0.0 and only mature when
        independently rediscovered (convergence tracking detects new path
        to same claim, incrementing weight via add_relation dedup).

        Returns edge key if created, empty string on failure.
        """
        from config.app_config import SYNTHESIS_BRIDGE_ON_ACCEPT, SYNTHESIS_BRIDGE_RELATION

        if not SYNTHESIS_BRIDGE_ON_ACCEPT or not graph_memory or not entity_resolver:
            return ""

        concept_a = result.candidate.concept_a
        concept_b = result.candidate.concept_b

        try:
            # Resolve personal entity (concept_a)
            source_id = entity_resolver.resolve_or_create(
                mention=concept_a,
                entity_type="concept",
                display_name=concept_a,
            )

            # Resolve/create wiki entity (concept_b)
            target_id = entity_resolver.resolve(concept_b.lower().strip())
            if not target_id:
                from memory.graph_models import GraphNode
                target_id = concept_b.lower().strip().replace(" ", "_")
                graph_memory.add_entity(GraphNode(
                    entity_id=target_id,
                    display_name=concept_b,
                    entity_type="concept",
                    metadata={
                        "source": "synthesis_discovery",
                    },
                ))

            # Provisional bridge: weight=0.0, matures on rediscovery
            from memory.graph_models import GraphEdge
            from datetime import datetime

            edge = GraphEdge(
                source_id=source_id,
                relation=SYNTHESIS_BRIDGE_RELATION,
                target_id=target_id,
                weight=0.0,
                truth_score=result.coherence_level.value if result.coherence_level else 0.5,
                metadata={
                    "source": "synthesis_discovery",
                    "status": "provisional",
                    "coherence_level": result.coherence_level.name if result.coherence_level else "UNKNOWN",
                    "composite_score": result.composite_score,
                    "connection_claim": result.candidate.connection_claim[:200],
                    "discovery_session": datetime.now().isoformat(),
                },
            )

            graph_memory.add_relation(edge)
            logger.info(
                f"[SYNTH BRIDGE] Provisional: {source_id} --[{SYNTHESIS_BRIDGE_RELATION}]--> {target_id}"
            )
            return edge.edge_key()

        except Exception as e:
            logger.warning(f"[SYNTH BRIDGE] Failed (non-fatal): {e}")
            return ""

    def store_rejected_for_audit(self, result: SynthesisResult) -> str:
        """Store a composite-rejected candidate for audit queue review.

        These candidates passed coherence but failed composite scoring.
        Human review determines if composite was too aggressive (false negative).
        """
        doc_text = result.candidate.connection_claim
        metadata = result.to_metadata()

        try:
            doc_id = self.store.add_to_collection(
                name=self.COLLECTION_NAME,
                text=doc_text,
                metadata=metadata,
            )
            logger.debug(f"Stored composite reject for audit: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to store rejected result for audit: {e}")
            raise

    def grade_result(self, doc_id: str, grade: str, notes: str = "") -> bool:
        """Apply a human grade to a synthesis result.

        Args:
            doc_id: ChromaDB document ID
            grade: "1"-"5" (structural rubric) or legacy "valid"/"invalid"/etc.
            notes: Optional reviewer notes

        Returns True on success.
        """
        valid_grades = {"1", "2", "3", "4", "5", "valid", "invalid", "should_pass", "correct_reject"}
        if grade not in valid_grades:
            logger.error(f"Invalid grade '{grade}', must be one of {valid_grades}")
            return False

        try:
            self.store.update_metadata(
                collection_name=self.COLLECTION_NAME,
                doc_id=doc_id,
                metadata_updates={
                    "human_grade": grade,
                    "graded_at": datetime.now().isoformat(),
                    "grade_notes": notes,
                },
            )
            logger.info(f"[SYNTH AUDIT] Graded {doc_id} as '{grade}'")
            return True
        except Exception as e:
            logger.error(f"Failed to grade result {doc_id}: {e}")
            return False

    def get_all_results(self, limit: int = 200) -> List[Tuple[str, SynthesisResult]]:
        """Fetch all synthesis results (accepted + rejected) with doc IDs.

        Returns list of (doc_id, SynthesisResult).
        """
        try:
            results = self.store.query_collection(
                collection_name=self.COLLECTION_NAME,
                query_text="cross-domain connection structural mechanism",
                n_results=limit,
            )
            if not results:
                return []

            items = []
            for item in results:
                doc_id = item.get("id", "")
                doc = item.get("content", "")
                metadata = item.get("metadata", {})
                if not doc_id or not metadata:
                    continue
                result = SynthesisResult.from_metadata(metadata, doc)
                items.append((doc_id, result))
            return items
        except Exception as e:
            logger.error(f"Error fetching all synthesis results: {e}")
            return []

    def get_ungraded(
        self,
        status_filter: str = "",
        limit: int = 50,
    ) -> List[Tuple[str, SynthesisResult]]:
        """Fetch ungraded synthesis results for the audit queue.

        Args:
            status_filter: "accepted", "rejected", or "" for both
            limit: max results

        Returns list of (doc_id, SynthesisResult) with no human_grade set.
        """
        all_results = self.get_all_results(limit=limit * 2)
        ungraded = []
        for doc_id, result in all_results:
            if result.human_grade:
                continue
            if status_filter == "accepted" and result.status != CandidateStatus.ACCEPTED:
                if result.status != CandidateStatus.CONVERGING:
                    continue
            if status_filter == "rejected" and result.status != CandidateStatus.REJECTED:
                continue
            ungraded.append((doc_id, result))
            if len(ungraded) >= limit:
                break
        return ungraded

    def get_graded(self, limit: int = 50) -> List[Tuple[str, SynthesisResult]]:
        """Fetch previously graded synthesis results.

        Returns list of (doc_id, SynthesisResult) sorted by graded_at desc.
        """
        all_results = self.get_all_results(limit=limit * 2)
        graded = [
            (doc_id, result) for doc_id, result in all_results
            if result.human_grade
        ]
        graded.sort(
            key=lambda x: x[1].graded_at or "",
            reverse=True,
        )
        return graded[:limit]

    @staticmethod
    def _grade_is_valid(grade: str) -> bool:
        """Grade 4-5 or legacy 'valid'/'should_pass' = structurally valid."""
        if grade in ("valid", "should_pass"):
            return True
        try:
            return int(grade) >= 4
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _grade_is_invalid(grade: str) -> bool:
        """Grade 1-3 or legacy 'invalid'/'correct_reject' = reject."""
        if grade in ("invalid", "correct_reject"):
            return True
        try:
            return 1 <= int(grade) <= 3
        except (ValueError, TypeError):
            return False

    def get_audit_stats(self) -> dict:
        """Compute audit statistics for the synthesis pipeline.

        Grades 4-5 = valid (structural isomorphism or better).
        Grades 1-3 = invalid (hallucination, surface metaphor, or trivial).
        FP rate = fraction of accepted insights graded 1-3.
        Auto-halt triggers when FP rate > threshold with sufficient data.
        """
        from config.app_config import (
            SYNTHESIS_AUDIT_FP_HALT_THRESHOLD,
            SYNTHESIS_AUDIT_MIN_GRADED,
        )

        all_results = self.get_all_results(limit=500)

        valid_count = 0
        invalid_count = 0
        ungraded_accepted = 0
        ungraded_rejected = 0
        grade_sum = 0
        grade_count = 0

        for _, result in all_results:
            grade = result.human_grade
            if not grade:
                if result.status in (CandidateStatus.ACCEPTED, CandidateStatus.CONVERGING):
                    ungraded_accepted += 1
                elif result.status == CandidateStatus.REJECTED:
                    ungraded_rejected += 1
                continue

            if self._grade_is_valid(grade):
                valid_count += 1
            elif self._grade_is_invalid(grade):
                invalid_count += 1

            try:
                gn = int(grade)
                grade_sum += gn
                grade_count += 1
            except (ValueError, TypeError):
                pass

        total_graded = valid_count + invalid_count

        fp_rate = (
            invalid_count / total_graded
            if total_graded > 0 else 0.0
        )

        avg_grade = round(grade_sum / grade_count, 2) if grade_count > 0 else 0.0

        auto_halt = (
            fp_rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD
            and total_graded >= SYNTHESIS_AUDIT_MIN_GRADED
        )

        return {
            "total_graded": total_graded,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "fp_rate": round(fp_rate, 3),
            "avg_grade": avg_grade,
            "auto_halt": auto_halt,
            "ungraded_accepted": ungraded_accepted,
            "ungraded_rejected": ungraded_rejected,
            "fp_halt_threshold": SYNTHESIS_AUDIT_FP_HALT_THRESHOLD,
            "min_graded_for_halt": SYNTHESIS_AUDIT_MIN_GRADED,
        }

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
