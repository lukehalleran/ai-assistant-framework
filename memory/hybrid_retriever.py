"""
memory/hybrid_retriever.py
Hybrid retrieval combining query rewriting, semantic search, and keyword matching.
Improves retrieval quality for casual/slang queries.

Module Contract:
- Purpose: Enhance semantic search quality with 3-tier approach
- Inputs: User query, retrieval limits
- Outputs: Ranked list of memories with hybrid scores
- Key collaborators: MultiCollectionChromaStore, query rewriter, keyword matcher
- Side effects: None (read-only retrieval)
- Async behavior: Main methods are async for ChromaDB queries
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from utils.logging_utils import get_logger
from utils.query_rewriter import rewrite_query, extract_keywords
from utils.keyword_matcher import calculate_keyword_score
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CHROMA_PATH

logger = get_logger("hybrid_retriever")

class HybridRetriever:
    """
    Hybrid retrieval system that combines:
    1. Query rewriting - expands casual queries with synonyms
    2. Semantic search - ChromaDB embedding similarity
    3. Keyword matching - exact term boosting
    """

    def __init__(self, chroma_store: Optional[MultiCollectionChromaStore] = None):
        """Initialize hybrid retriever with components."""
        self.chroma_store = chroma_store or MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

        # Hybrid scoring weights
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3

    async def retrieve(self, query: str, limit: int = 30) -> List[Dict]:
        """
        Main retrieval method combining all three approaches.

        Args:
            query: Original user query
            limit: Number of results to return

        Returns:
            List of memories with hybrid scores
        """
        logger.info(f"[HybridRetriever] Retrieving for query: '{query[:50]}...'")

        # Step 1: Query rewriting
        expanded_query = rewrite_query(query)
        logger.debug(f"[HybridRetriever] Expanded query: '{expanded_query}'")

        # Step 2: Semantic search with expanded query
        semantic_results = await self._semantic_search(expanded_query, limit * 3)  # Get more candidates

        # Step 3: Keyword matching
        keyword_results = self._keyword_match(query, semantic_results)

        # Step 4: Hybrid scoring and ranking
        hybrid_results = self._hybrid_score(semantic_results, keyword_results, query)

        # Step 5: Return top results
        final_results = hybrid_results[:limit]

        logger.info(f"[HybridRetriever] Returning {len(final_results)} results (from {len(semantic_results)} candidates)")
        return final_results

    async def _semantic_search(self, query: str, n_results: int = 90) -> List[Dict]:
        """
        Perform semantic search across multiple ChromaDB collections.

        Args:
            query: Query text (may be expanded)
            n_results: Number of results per collection

        Returns:
            List of semantic search results
        """
        memories = []
        collections_to_query = ['conversations', 'summaries', 'reflections']

        try:
            # Batch query all collections
            batch_results = await self.chroma_store.query_multiple_collections(
                collections_to_query,
                query_text=query,
                n_results=n_results
            )

            for collection_name, results in batch_results.items():
                if not results:
                    continue

                logger.debug(f"[HybridRetriever] Processing {len(results)} from {collection_name}")

                for item in results:
                    if item:
                        if not isinstance(item, dict):
                            item = {"content": str(item), "id": str(hash(str(item)))}

                        # Normalize result format
                        memory = {
                            "id": item.get("id", str(hash(str(item)))),
                            "content": item.get("content", ""),
                            "query": item.get("query", ""),
                            "response": item.get("response", ""),
                            "metadata": item.get("metadata", {}),
                            "collection": collection_name,
                            "semantic_score": 1.0 - item.get("distance", 1.0),  # Convert distance to similarity
                            "distance": item.get("distance", 1.0)
                        }
                        memories.append(memory)

        except Exception as e:
            logger.error(f"[HybridRetriever] Semantic search failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"[HybridRetriever] Semantic search found {len(memories)} candidates")
        return memories

    def _keyword_match(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Score candidates with keyword matching.

        Args:
            query: Original query
            candidates: Semantic search results

        Returns:
            Candidates with keyword scores added
        """
        logger.debug(f"[HybridRetriever] Keyword matching for {len(candidates)} candidates")

        # Extract keywords from query
        keywords = extract_keywords(query)
        logger.debug(f"[HybridRetriever] Extracted keywords: {keywords}")

        # Score each candidate
        for candidate in candidates:
            # Calculate keyword score using functional approach
            keyword_score = calculate_keyword_score(keywords, candidate)
            candidate["keyword_score"] = keyword_score

        return candidates

    def _hybrid_score(self, semantic_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
        """
        Combine semantic and keyword scores into hybrid score.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Same results with keyword scores added
            query: Original query for scoring context

        Returns:
            Results ranked by hybrid score
        """
        logger.debug(f"[HybridRetriever] Calculating hybrid scores for {len(keyword_results)} results")

        scored_results = []

        for result in keyword_results:
            semantic_score = result.get("semantic_score", 0.0)
            keyword_score = result.get("keyword_score", 0.0)

            # Hybrid score is weighted combination
            hybrid_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )

            result["hybrid_score"] = hybrid_score
            result["final_score"] = hybrid_score  # For compatibility with existing code

            # Add scoring details for debugging
            result["scoring"] = {
                "semantic": semantic_score,
                "keyword": keyword_score,
                "hybrid": hybrid_score,
                "weights": {
                    "semantic": self.semantic_weight,
                    "keyword": self.keyword_weight
                }
            }

            scored_results.append(result)

        # Sort by hybrid score descending
        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Log top few results for debugging
        for i, result in enumerate(scored_results[:5]):
            logger.debug(f"[HybridRetriever] Top {i+1}: hybrid={result['hybrid_score']:.3f} "
                        f"(semantic={result['semantic_score']:.3f}, "
                        f"keyword={result['keyword_score']:.3f})")

        return scored_results

    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """
        Adjust hybrid scoring weights.

        Args:
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            keyword_weight: Weight for keyword matching (0.0-1.0)
        """
        total = semantic_weight + keyword_weight
        if total > 0:
            self.semantic_weight = semantic_weight / total
            self.keyword_weight = keyword_weight / total
            logger.info(f"[HybridRetriever] Updated weights: semantic={self.semantic_weight:.2f}, "
                       f"keyword={self.keyword_weight:.2f}")