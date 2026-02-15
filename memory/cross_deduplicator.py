# memory/cross_deduplicator.py
"""
Cross-collection memory deduplication service.

Detects near-duplicate documents across ChromaDB collections and
resolves fact contradictions (same subject+predicate, different object).
Runs as a maintenance pass on shutdown, on schedule, or via GUI.
"""

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import app_config as cfg
from memory.dedup_models import (
    ContradictionCluster,
    DedupAction,
    DedupPlan,
    DedupReason,
    DuplicatePair,
)
from utils.logging_utils import get_logger

logger = get_logger("cross_deduplicator")

# Collections that should never have entries deleted automatically
_PROTECTED_COLLECTIONS = frozenset({"conversations", "obsidian_notes", "reference_docs", "wiki_knowledge"})


class CrossCollectionDeduplicator:
    """
    Unified deduplication across multiple ChromaDB collections.

    Two detection modes:
    1. **Cross-duplicate detection** — finds semantically identical content
       across different collections (e.g., a fact already captured in a summary).
    2. **Fact contradiction detection** — finds facts with matching
       subject+predicate but differing objects, keeps the most recent.

    Usage::

        dedup = CrossCollectionDeduplicator(chroma_store)
        plan = dedup.run(dry_run=True)   # preview
        plan = dedup.run(dry_run=False)  # execute deletions
    """

    def __init__(self, chroma_store):
        """
        Args:
            chroma_store: MultiCollectionChromaStore instance.
        """
        self.chroma_store = chroma_store
        self.duplicate_threshold: float = cfg.CROSS_DEDUP_DUPLICATE_THRESHOLD
        self.contradiction_threshold: float = cfg.CROSS_DEDUP_CONTRADICTION_THRESHOLD
        self.max_docs: int = cfg.CROSS_DEDUP_MAX_DOCS_PER_COLLECTION
        self.collections: List[str] = [
            c for c in cfg.CROSS_DEDUP_COLLECTIONS
            if c not in _PROTECTED_COLLECTIONS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = True) -> DedupPlan:
        """
        Execute a full dedup pass: load docs, find duplicates and
        contradictions, optionally delete.

        Args:
            dry_run: If True, report only; if False, execute deletions.

        Returns:
            DedupPlan with findings and actions taken.
        """
        plan = DedupPlan(dry_run=dry_run)
        t0 = time.time()

        # 1. Load documents from all target collections
        all_docs, collection_doc_map = self._load_all_documents()
        plan.collections_scanned = list(collection_doc_map.keys())
        plan.total_docs_scanned = len(all_docs)

        if len(all_docs) < 2:
            logger.info("[CrossDedup] Fewer than 2 documents, nothing to deduplicate")
            return plan

        # 2. Compute embeddings for all documents
        texts = [doc["content"] for doc in all_docs]
        embeddings = self._compute_embeddings(texts)

        if embeddings is None:
            plan.errors.append("Failed to compute embeddings")
            return plan

        # 3. Find cross-collection duplicates
        duplicate_pairs = self._find_cross_duplicates(all_docs, embeddings)
        plan.duplicate_pairs = duplicate_pairs
        plan.duplicates_found = len(duplicate_pairs)

        # 4. Find fact contradictions
        fact_docs = [d for d in all_docs if d["collection"] == "facts"]
        contradiction_clusters = self._find_fact_contradictions(fact_docs)
        plan.contradiction_clusters = contradiction_clusters
        plan.contradictions_found = len(contradiction_clusters)

        # 5. Execute deletions if not dry_run
        if not dry_run:
            deleted = self._execute_plan(plan)
            plan.deletions_executed = deleted
        else:
            # Count proposed deletions
            plan.deletions_executed = (
                sum(1 for p in duplicate_pairs if p.delete_id)
                + sum(len(c.delete_ids) for c in contradiction_clusters)
            )

        elapsed = time.time() - t0
        logger.info(
            "[CrossDedup] Completed in %.2fs: %d docs scanned, %d duplicates, "
            "%d contradictions, %d deletions %s",
            elapsed, plan.total_docs_scanned, plan.duplicates_found,
            plan.contradictions_found, plan.deletions_executed,
            "(dry run)" if dry_run else "(executed)",
        )
        return plan

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_all_documents(self) -> Tuple[List[dict], Dict[str, List[dict]]]:
        """Load documents from all target collections.

        Returns:
            (flat_list, {collection_name: [docs]})
            Each doc dict has keys: id, content, metadata, collection.
        """
        all_docs = []
        collection_map: Dict[str, List[dict]] = {}

        for coll_name in self.collections:
            try:
                items = self.chroma_store.list_all(coll_name)
                if not items:
                    continue

                # Apply per-collection cap
                if len(items) > self.max_docs:
                    # Keep most recent by timestamp
                    items.sort(
                        key=lambda x: x.get("metadata", {}).get("timestamp", ""),
                        reverse=True,
                    )
                    items = items[: self.max_docs]

                docs = []
                for item in items:
                    content = (item.get("content") or "").strip()
                    if not content:
                        continue
                    doc = {
                        "id": item.get("id"),
                        "content": content,
                        "metadata": item.get("metadata", {}),
                        "collection": coll_name,
                    }
                    docs.append(doc)

                collection_map[coll_name] = docs
                all_docs.extend(docs)

            except Exception as e:
                logger.warning("[CrossDedup] Error loading collection %s: %s", coll_name, e)

        logger.info(
            "[CrossDedup] Loaded %d documents from %d collections: %s",
            len(all_docs),
            len(collection_map),
            {k: len(v) for k, v in collection_map.items()},
        )
        return all_docs, collection_map

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def _compute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Compute embeddings for a list of texts using the shared ChromaDB embedder.

        Returns:
            (N, D) numpy array of L2-normalized embeddings, or None on failure.
        """
        if not texts:
            return None

        try:
            # ChromaDB's SentenceTransformerEmbeddingFunction is callable
            raw = self.chroma_store.embedding_fn(texts)
            vecs = np.array(raw, dtype=np.float32)
            # L2-normalize for cosine similarity via dot product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
            return vecs
        except Exception as e:
            logger.error("[CrossDedup] Embedding computation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Cross-duplicate detection
    # ------------------------------------------------------------------

    def _find_cross_duplicates(
        self, docs: List[dict], embeddings: np.ndarray
    ) -> List[DuplicatePair]:
        """Find near-duplicate pairs across (or within) collections.

        Uses pairwise cosine similarity on pre-normalized embeddings.
        Only pairs exceeding self.duplicate_threshold are flagged.

        Ephemeral fact pairs (both from "facts" with an ephemeral predicate)
        are skipped — those represent temporal state at different points in
        time and their history is meaningful.
        """
        n = len(docs)
        if n < 2:
            return []

        ephemeral = self._get_ephemeral_relations()

        # Compute full pairwise similarity matrix via dot product
        # (embeddings are L2-normalized, so dot = cosine)
        sim_matrix = embeddings @ embeddings.T

        pairs: List[DuplicatePair] = []
        # Track which doc IDs have already been marked for deletion
        # to avoid cascading (deleting A because of B, then B because of C)
        marked_for_delete = set()

        for i in range(n):
            for j in range(i + 1, n):
                score = min(float(sim_matrix[i, j]), 1.0)  # clamp fp32 overshoot
                if score < self.duplicate_threshold:
                    continue

                doc_a = docs[i]
                doc_b = docs[j]

                # Skip if either already marked for deletion
                if doc_a["id"] in marked_for_delete or doc_b["id"] in marked_for_delete:
                    continue

                # Skip ephemeral fact pairs — "tired" yesterday vs "fine" today
                # are not duplicates, they're temporal snapshots
                if self._is_ephemeral_fact_pair(doc_a, doc_b, ephemeral):
                    continue

                # Determine which to keep: prefer by collection priority,
                # then by recency
                keep, delete = self._pick_keep_delete(doc_a, doc_b)

                reason = (
                    DedupReason.CROSS_DUPLICATE
                    if doc_a["collection"] != doc_b["collection"]
                    else DedupReason.WITHIN_DUPLICATE
                )

                pair = DuplicatePair(
                    doc_id_a=doc_a["id"],
                    doc_id_b=doc_b["id"],
                    collection_a=doc_a["collection"],
                    collection_b=doc_b["collection"],
                    similarity=score,
                    content_preview_a=doc_a["content"][:200],
                    content_preview_b=doc_b["content"][:200],
                    reason=reason,
                    action=DedupAction.DELETE,
                    keep_id=keep["id"],
                    delete_id=delete["id"],
                    confidence=min(1.0, score),
                )
                pairs.append(pair)
                marked_for_delete.add(delete["id"])

        logger.info("[CrossDedup] Found %d duplicate pairs above threshold %.3f", len(pairs), self.duplicate_threshold)
        return pairs

    def _pick_keep_delete(self, doc_a: dict, doc_b: dict) -> Tuple[dict, dict]:
        """Decide which document to keep and which to delete.

        Priority rules:
        1. Summaries > facts (a summary subsumes its constituent facts).
        2. More recent document wins among equal-priority collections.
        """
        priority = {
            "summaries": 5,
            "reflections": 4,
            "procedural_skills": 3,
            "proposals": 2,
            "facts": 1,
        }
        pa = priority.get(doc_a["collection"], 0)
        pb = priority.get(doc_b["collection"], 0)

        if pa != pb:
            # Higher priority collection = keep
            return (doc_a, doc_b) if pa > pb else (doc_b, doc_a)

        # Same priority: keep the more recent one
        ts_a = self._extract_timestamp(doc_a)
        ts_b = self._extract_timestamp(doc_b)
        return (doc_a, doc_b) if ts_a >= ts_b else (doc_b, doc_a)

    @staticmethod
    def _extract_timestamp(doc: dict) -> float:
        """Extract a comparable timestamp from document metadata."""
        meta = doc.get("metadata", {})
        ts = meta.get("timestamp") or meta.get("created_at") or meta.get("date") or ""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return 0.0

    # ------------------------------------------------------------------
    # Fact contradiction detection
    # ------------------------------------------------------------------

    def _find_fact_contradictions(
        self, fact_docs: List[dict]
    ) -> List[ContradictionCluster]:
        """Find facts with matching subject+predicate but different objects.

        Groups facts by (subject, predicate) extracted from metadata or
        content, then flags groups where objects differ.

        Ephemeral relations (current_feeling, current_activity, etc.) are
        skipped — those are temporal facts where historical values carry
        meaning (e.g. "you were tired yesterday").
        """
        if not fact_docs:
            return []

        # Load ephemeral relations to exclude from contradiction detection
        ephemeral = self._get_ephemeral_relations()

        # Group facts by (subject, predicate)
        groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

        for doc in fact_docs:
            subj, pred, obj = self._extract_triple(doc)
            if subj and pred:
                pred_norm = pred.lower().strip()
                # Skip ephemeral predicates — history matters
                if pred_norm in ephemeral:
                    continue
                key = (subj.lower().strip(), pred_norm)
                groups[key].append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "collection": doc["collection"],
                    "object": obj,
                    "timestamp": self._extract_timestamp(doc),
                })

        clusters = []
        for (subj, pred), entries in groups.items():
            if len(entries) < 2:
                continue

            # Check if objects actually differ
            objects = [e.get("object", "").lower().strip() for e in entries]
            unique_objects = set(o for o in objects if o)
            if len(unique_objects) <= 1:
                # Same object = true duplicates, handled by cross-dedup
                continue

            # Sort by timestamp descending — keep the most recent
            entries.sort(key=lambda e: e["timestamp"], reverse=True)
            keep_id = entries[0]["id"]
            delete_ids = [e["id"] for e in entries[1:]]

            cluster = ContradictionCluster(
                subject=subj,
                predicate=pred,
                entries=entries,
                keep_id=keep_id,
                delete_ids=delete_ids,
            )
            clusters.append(cluster)

        logger.info(
            "[CrossDedup] Found %d fact contradiction clusters (skipped %d ephemeral relations)",
            len(clusters), len(ephemeral),
        )
        return clusters

    @staticmethod
    def _get_ephemeral_relations() -> frozenset:
        """Load the set of ephemeral relation names that should be excluded
        from contradiction detection (their history is meaningful)."""
        try:
            from config.app_config import PROFILE_EPHEMERAL_RELATIONS
            return frozenset(r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS)
        except ImportError:
            return frozenset()

    @classmethod
    def _is_ephemeral_fact_pair(cls, doc_a: dict, doc_b: dict, ephemeral: frozenset) -> bool:
        """Return True if either doc is a fact with an ephemeral predicate.

        Ephemeral facts (current_feeling, current_activity, etc.) record
        temporal state — "tired" yesterday vs "fine" today are not duplicates.
        Checks metadata fields first, falls back to parsing content.
        """
        if doc_a["collection"] != "facts" or doc_b["collection"] != "facts":
            return False

        for doc in (doc_a, doc_b):
            # Try metadata first
            meta = doc.get("metadata", {})
            pred = (
                meta.get("predicate")
                or meta.get("relation")
                or ""
            ).lower().strip()

            # Fall back to parsing content (e.g. "user | current_time | 1:30")
            if not pred:
                _, pred, _ = cls._extract_triple(doc)
                pred = pred.lower().strip()

            if pred and pred in ephemeral:
                return True

        return False

    @staticmethod
    def _extract_triple(doc: dict) -> Tuple[str, str, str]:
        """Extract (subject, predicate, object) from a fact document.

        Tries metadata fields first, falls back to parsing content
        patterns like "subject predicate object" or "subject - predicate - object".
        """
        meta = doc.get("metadata", {})
        subj = meta.get("subject") or meta.get("entity") or ""
        pred = meta.get("predicate") or meta.get("relation") or ""
        obj = meta.get("object") or meta.get("value") or ""

        if subj and pred:
            return (subj, pred, obj)

        # Fallback: parse from content
        content = doc.get("content", "")
        # Pattern: "Subject | Predicate | Object" or "Subject - Predicate - Object"
        for sep in [" | ", " - ", " — "]:
            parts = content.split(sep)
            if len(parts) >= 3:
                return (parts[0].strip(), parts[1].strip(), sep.join(parts[2:]).strip())
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip(), "")

        # Pattern: "Subject predicate: Object"
        m = re.match(r"^(.+?)\s+(is|has|was|likes|dislikes|prefers|uses|works|lives)\s+(.+)$", content, re.IGNORECASE)
        if m:
            return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())

        return ("", "", "")

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _execute_plan(self, plan: DedupPlan) -> int:
        """Execute deletions from the plan.

        Returns:
            Number of successful deletions.
        """
        deleted = 0

        # Delete duplicate pairs
        for pair in plan.duplicate_pairs:
            if not pair.delete_id:
                continue
            try:
                coll_name = (
                    pair.collection_b
                    if pair.delete_id == pair.doc_id_b
                    else pair.collection_a
                )
                coll = self.chroma_store.collections.get(coll_name)
                if coll:
                    coll.delete(ids=[pair.delete_id])
                    deleted += 1
                    logger.debug(
                        "[CrossDedup] Deleted %s from %s (dup of %s in %s, sim=%.3f)",
                        pair.delete_id, coll_name, pair.keep_id,
                        pair.collection_a if pair.delete_id == pair.doc_id_b else pair.collection_b,
                        pair.similarity,
                    )
            except Exception as e:
                msg = f"Failed to delete {pair.delete_id}: {e}"
                logger.warning("[CrossDedup] %s", msg)
                plan.errors.append(msg)

        # Delete contradicting facts (keep most recent) + apply truth penalties
        for cluster in plan.contradiction_clusters:
            # Apply truth score penalties for contradictions
            self._apply_contradiction_truth_penalties(cluster)

            for del_id in cluster.delete_ids:
                try:
                    coll = self.chroma_store.collections.get("facts")
                    if coll:
                        coll.delete(ids=[del_id])
                        deleted += 1
                        logger.debug(
                            "[CrossDedup] Deleted contradicting fact %s "
                            "(subject=%s, pred=%s, keeping %s)",
                            del_id, cluster.subject, cluster.predicate, cluster.keep_id,
                        )
                except Exception as e:
                    msg = f"Failed to delete contradiction {del_id}: {e}"
                    logger.warning("[CrossDedup] %s", msg)
                    plan.errors.append(msg)

        return deleted

    def _apply_contradiction_truth_penalties(self, cluster: ContradictionCluster) -> None:
        """Apply truth score penalties when contradictions are resolved.

        The older (deleted) facts get a contradiction penalty; the kept
        (newest) fact gets a small boost for surviving.
        """
        try:
            from memory.truth_scorer import TruthScorer

            # Penalize older contradicting facts
            for del_id in cluster.delete_ids:
                try:
                    self.chroma_store.update_metadata("facts", del_id, {
                        "truth_score": TruthScorer.apply_contradiction(0.7),
                    })
                except Exception:
                    pass  # Best-effort; fact may be about to be deleted anyway

            # Small boost for the kept (most recent) fact
            if cluster.keep_id:
                try:
                    coll = self.chroma_store.collections.get("facts")
                    if coll:
                        existing = coll.get(ids=[cluster.keep_id], include=["metadatas"])
                        if existing and existing.get("metadatas"):
                            old_truth = float(existing["metadatas"][0].get("truth_score", 0.7))
                            new_truth = min(1.0, old_truth + 0.03)
                            self.chroma_store.update_metadata("facts", cluster.keep_id, {
                                "truth_score": new_truth,
                            })
                except Exception as e:
                    logger.debug("[CrossDedup] Failed to boost kept fact %s: %s", cluster.keep_id, e)

        except ImportError:
            logger.debug("[CrossDedup] TruthScorer not available; skipping contradiction penalties")
