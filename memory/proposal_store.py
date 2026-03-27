# memory/proposal_store.py
"""
ChromaDB-backed storage and retrieval for code proposals.

Module Contract
- Purpose: Persists CodeProposal objects in the 'proposals' ChromaDB collection
  with semantic search, status filtering, deduplication, and implementation tracking metadata.
- Class: ProposalStore(chroma_store)
- Key methods:
  - store_proposal(proposal) -> Optional[str]  [returns doc_id]
  - query_proposals(query, limit, status_filter) -> List[CodeProposal]  [semantic search + status filter]
  - get_proposal(proposal_id) -> Optional[CodeProposal]  [by ID lookup]
  - get_pending() -> List[CodeProposal]  [status=pending_review]
  - get_pending_and_approved() -> List[CodeProposal]  [pending_review + approved]
  - update_status(proposal_id, new_status, notes) -> bool  [delete-and-re-add]
  - check_similarity(proposal, threshold) -> Tuple[bool, float, Optional[str]]  [dedup check]
  - update_tracking_metadata(proposal_id, detection_result) -> bool  [implementation tracking fields]
  - get_for_dedup(limit) -> str  [serialized context string for dedup during proposal generation]
- Key behaviors:
  - Embedding text is title + reasoning (problem match, not solution match)
  - Full proposal data stored in metadata via to_metadata()
  - Status updates via delete-and-re-add (ChromaDB lacks native update)
  - Dedup uses cosine similarity from query_collection results
- Dependencies:
  - memory.storage.multi_collection_chroma_store (vector storage)
  - memory.code_proposal (CodeProposal, ProposalStatus data models)
  - config.app_config (thresholds, feature flag)
"""

import logging
from typing import Dict, List, Optional

from memory.code_proposal import CodeProposal, ProposalStatus
from utils.logging_utils import get_logger

logger = get_logger("proposal_store")

COLLECTION_NAME = "proposals"


class ProposalStore:
    """
    ChromaDB-backed store for code proposals.

    Uses MultiCollectionChromaStore.add_to_collection() and
    query_collection() for all vector operations.
    """

    def __init__(self, chroma_store=None):
        self.chroma_store = chroma_store

    def _ensure_collection(self) -> bool:
        """Ensure the proposals collection exists. Returns False if unavailable."""
        if not self.chroma_store:
            return False

        collections = getattr(self.chroma_store, "collections", None)
        if collections is None:
            return False

        if COLLECTION_NAME not in collections or collections[COLLECTION_NAME] is None:
            if hasattr(self.chroma_store, "create_collection"):
                try:
                    self.chroma_store.create_collection(COLLECTION_NAME)
                except Exception as e:
                    logger.error(f"[ProposalStore] Failed to create collection: {e}")
                    return False
            else:
                return False

        return True

    def store_proposal(self, proposal: CodeProposal) -> Optional[str]:
        """
        Store a proposal in ChromaDB.

        Args:
            proposal: CodeProposal to store

        Returns:
            Document ID if stored, None if failed
        """
        if not self._ensure_collection():
            logger.warning("[ProposalStore] ChromaDB not available, cannot store")
            return None

        try:
            embedding_text = proposal.to_embedding_text()
            metadata = proposal.to_metadata()

            doc_id = self.chroma_store.add_to_collection(
                COLLECTION_NAME, embedding_text, metadata
            )

            logger.info(
                f"[ProposalStore] Stored proposal {doc_id}: "
                f"'{proposal.title}' ({proposal.proposal_type.value})"
            )
            return doc_id

        except Exception as e:
            logger.error(f"[ProposalStore] Failed to store proposal: {e}")
            return None

    def query_proposals(
        self,
        query: str,
        n_results: int = 10,
        status_filter: Optional[List[str]] = None,
    ) -> List[CodeProposal]:
        """
        Semantic search for proposals matching a query.

        Args:
            query: Search text
            n_results: Maximum results to return
            status_filter: Optional list of status values to include
                (e.g., ["pending", "approved"])

        Returns:
            List of CodeProposal objects, ranked by relevance
        """
        if not self._ensure_collection():
            return []

        try:
            # Fetch extra results if filtering, since we filter post-query
            fetch_count = n_results * 3 if status_filter else n_results

            coll = self.chroma_store.collections.get(COLLECTION_NAME)
            if coll is None or coll.count() == 0:
                return []

            results = self.chroma_store.query_collection(
                COLLECTION_NAME,
                query_text=query,
                n_results=min(fetch_count, coll.count()),
            )

            proposals = []
            for match in results:
                meta = match.get("metadata") or {}
                if not meta.get("proposal_id"):
                    continue

                try:
                    proposal = CodeProposal.from_metadata(meta)
                except Exception as e:
                    logger.debug(f"[ProposalStore] Failed to deserialize proposal: {e}")
                    continue

                if status_filter and proposal.status.value not in status_filter:
                    continue

                proposals.append(proposal)
                if len(proposals) >= n_results:
                    break

            return proposals

        except Exception as e:
            logger.error(f"[ProposalStore] Query failed: {e}")
            return []

    def get_proposal(self, proposal_id: str) -> Optional[CodeProposal]:
        """
        Retrieve a single proposal by its ID.

        Since ChromaDB doesn't have a direct get-by-metadata-field,
        we list all and filter. For small collections this is acceptable.
        """
        if not self._ensure_collection():
            return None

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("proposal_id") == proposal_id:
                    return CodeProposal.from_metadata(meta)
            return None
        except Exception as e:
            logger.error(f"[ProposalStore] get_proposal failed: {e}")
            return None

    def get_pending(self) -> List[CodeProposal]:
        """Get all proposals with PENDING status."""
        if not self._ensure_collection():
            return []

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            proposals = []
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("status") == ProposalStatus.PENDING.value:
                    try:
                        proposals.append(CodeProposal.from_metadata(meta))
                    except Exception:
                        continue
            return proposals
        except Exception as e:
            logger.error(f"[ProposalStore] get_pending failed: {e}")
            return []

    def update_status(
        self,
        proposal_id: str,
        status: ProposalStatus,
        reason: str = "",
        commit_hash: str = "",
    ) -> bool:
        """
        Update a proposal's status.

        Uses delete-then-re-add since ChromaDB lacks native update.

        Args:
            proposal_id: ID of the proposal to update
            status: New status
            reason: Rejection reason (for REJECTED/FAILED)
            commit_hash: Commit hash (for COMPLETED)

        Returns:
            True if updated successfully
        """
        if not self._ensure_collection():
            return False

        try:
            # Find the proposal and its ChromaDB document ID
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            target_item = None
            target_doc_id = None

            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("proposal_id") == proposal_id:
                    target_item = item
                    target_doc_id = item.get("id")
                    break

            if not target_item or not target_doc_id:
                logger.warning(f"[ProposalStore] Proposal {proposal_id} not found for update")
                return False

            # Reconstruct proposal and apply status change
            proposal = CodeProposal.from_metadata(target_item["metadata"])

            if status == ProposalStatus.APPROVED:
                proposal.mark_approved()
            elif status == ProposalStatus.REJECTED:
                proposal.mark_rejected(reason)
            elif status == ProposalStatus.COMPLETED:
                proposal.mark_completed(commit_hash)
            elif status == ProposalStatus.FAILED:
                proposal.mark_failed(reason)
            else:
                proposal.status = status

            # Delete old entry
            coll = self.chroma_store.collections.get(COLLECTION_NAME)
            if coll:
                coll.delete(ids=[target_doc_id])

            # Re-store with updated metadata
            self.store_proposal(proposal)

            logger.info(f"[ProposalStore] Updated proposal {proposal_id} to {status.value}")
            return True

        except Exception as e:
            logger.error(f"[ProposalStore] update_status failed: {e}")
            return False

    def check_similarity(
        self,
        proposal: CodeProposal,
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        Check if a similar proposal already exists.

        Args:
            proposal: Proposal to check for duplicates
            threshold: Similarity threshold (0-1, higher = more similar).
                Defaults to config value.

        Returns:
            ID of the similar existing proposal if found, None otherwise
        """
        if not self._ensure_collection():
            return None

        if threshold is None:
            try:
                from config.app_config import CODE_PROPOSALS_DEDUP_THRESHOLD
                threshold = CODE_PROPOSALS_DEDUP_THRESHOLD
            except ImportError:
                threshold = 0.70

        try:
            coll = self.chroma_store.collections.get(COLLECTION_NAME)
            if coll is None or coll.count() == 0:
                return None

            embedding_text = proposal.to_embedding_text()
            results = self.chroma_store.query_collection(
                COLLECTION_NAME,
                query_text=embedding_text,
                n_results=min(5, coll.count()),
            )

            new_title_words = set(proposal.title.lower().split())

            for match in results:
                score = match.get("relevance_score", 0.0)
                meta = match.get("metadata") or {}
                existing_id = meta.get("proposal_id", "")
                existing_title = meta.get("title", "")

                # Semantic similarity check
                if score and score >= threshold:
                    logger.info(
                        f"[ProposalStore] Duplicate detected (score={score:.2f}): "
                        f"'{proposal.title[:60]}' ~ '{existing_title[:60]}'"
                    )
                    return existing_id

                # Title word overlap check — catch near-dupes that
                # rephrase the same idea with different embedding scores
                if existing_title and new_title_words:
                    existing_words = set(existing_title.lower().split())
                    union = new_title_words | existing_words
                    inter = new_title_words & existing_words
                    if union and len(inter) / len(union) >= 0.60:
                        logger.info(
                            f"[ProposalStore] Duplicate detected (title overlap "
                            f"{len(inter)}/{len(union)}): "
                            f"'{proposal.title[:60]}' ~ '{existing_title[:60]}'"
                        )
                        return existing_id

            return None

        except Exception as e:
            logger.error(f"[ProposalStore] check_similarity failed: {e}")
            return None

    def update_tracking_metadata(self, proposal_id: str, detection_result) -> bool:
        """
        Update a proposal's implementation tracking fields via metadata merge.

        Args:
            proposal_id: ID of the proposal to update
            detection_result: DetectionResult with confidence/status/evidence

        Returns:
            True if updated successfully
        """
        if not self._ensure_collection():
            return False

        try:
            import time as _time
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("proposal_id") == proposal_id:
                    doc_id = item.get("id")
                    if not doc_id:
                        return False

                    updates = {
                        "implementation_confidence": detection_result.confidence,
                        "implementation_status": detection_result.status,
                        "implementation_evidence": (detection_result.evidence or "")[:500],
                        "last_tracked_at": _time.time(),
                    }
                    self.chroma_store.update_metadata(COLLECTION_NAME, doc_id, updates)
                    logger.info(
                        f"[ProposalStore] Updated tracking for {proposal_id}: "
                        f"{detection_result.status} ({detection_result.confidence:.0%})"
                    )
                    return True

            logger.warning(f"[ProposalStore] Proposal {proposal_id} not found for tracking update")
            return False

        except Exception as e:
            logger.error(f"[ProposalStore] update_tracking_metadata failed: {e}")
            return False

    def get_pending_and_approved(self) -> List[CodeProposal]:
        """Get all proposals with PENDING or APPROVED status."""
        if not self._ensure_collection():
            return []

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            proposals = []
            target_statuses = {ProposalStatus.PENDING.value, ProposalStatus.APPROVED.value}
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("status") in target_statuses:
                    try:
                        proposals.append(CodeProposal.from_metadata(meta))
                    except Exception:
                        continue
            return proposals
        except Exception as e:
            logger.error(f"[ProposalStore] get_pending_and_approved failed: {e}")
            return []

    def get_for_dedup(self, limit: int = 25) -> str:
        """
        Get a formatted string of recent proposals for LLM deduplication context.

        Returns a text block that can be included in generation prompts to
        avoid generating proposals that already exist.
        """
        if not self._ensure_collection():
            return ""

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            if not all_items:
                return ""

            # Sort by created_at descending
            def _created_at(item):
                meta = item.get("metadata") or {}
                return float(meta.get("created_at", 0))

            all_items.sort(key=_created_at, reverse=True)

            lines = []
            for item in all_items[:limit]:
                meta = item.get("metadata") or {}
                title = meta.get("title", "Untitled")
                status = meta.get("status", "unknown")
                ptype = meta.get("proposal_type", "unknown")
                lines.append(f"- [{status}] ({ptype}) {title}")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"[ProposalStore] get_for_dedup failed: {e}")
            return ""
