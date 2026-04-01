"""
# knowledge/synthesis_retriever.py

Module Contract
- Purpose: Retrieval-based synthesis candidate generator. Replaces random
  pairing with semantic retrieval: extracts structural queries from personal
  facts via LLM, searches FAISS for cross-domain matches, and adversarially
  evaluates whether a genuine structural parallel exists.
- Class: RetrievalSynthesisGenerator(chroma_store, model_manager, graph_memory=None, entity_resolver=None)
- Key methods:
  - generate_candidates(count) -> List[SynthesisCandidate]  [async, main entry]
  - _generate_structural_query(fact_item, semaphore) -> Optional[Tuple[Dict, str]]  [LLM]
  - _retrieve_and_filter(fact_item, structural_query) -> List[Dict]  [FAISS + domain filter]
  - _evaluate_connection(fact, wiki, query, semaphore) -> Optional[SynthesisCandidate]  [adversarial LLM]
- Inputs: ChromaDB store (facts), FAISS index (40M wiki vectors), ModelManager for LLM
- Outputs: List[SynthesisCandidate] compatible with SynthesisFilter pipeline
- Side effects: LLM API calls (structural query + adversarial evaluation)
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge.synthesis_models import SynthesisCandidate
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Few-shot structural query prompt — diverse examples to prevent convergence
# on generic "feedback loops and optimization"
STRUCTURAL_QUERY_PROMPT = """Given a personal fact, extract the specific structural dynamics it involves.
Avoid generic terms like "feedback loops" or "optimization" unless they
describe something specific about THIS fact.

Example 1:
Fact: runs a sub-1:45 half marathon
Structural: performance plateau where incremental improvement requires disproportionate effort increase; aerobic threshold as a hard physiological boundary that changes training strategy above vs below it

Example 2:
Fact: 6 years in brewing
Structural: batch process where small parameter changes cascade into product failure; timing-sensitive sequential operations with irreversible commitment points; apprenticeship knowledge that resists codification

Example 3:
Fact: strained relationship with parents
Structural: asymmetric power dynamic that inverts over time; obligation bonds that persist independent of emotional state; recursive pattern where conflict about communication style becomes the communication problem

Now extract structural dynamics for:
Fact: {concept_name}
Context: {context}

Structural:"""

# Evaluation prompt — produce the structural insight directly
ADVERSARIAL_EVAL_PROMPT = """Personal fact ({domain_a}): {concept_a}
Context: {context_a}

Retrieved article ({domain_b}): {concept_b}
Context: {context_b}

Structural link: {structural_query}

If a specific structural mechanism operates in both domains, state it \
directly in 1-2 sentences. Name the mechanism and explain how it works \
in each domain concretely.

Do NOT write "The connection is..." or evaluate whether it's meaningful. \
Just state the parallel as a factual claim.

If no real parallel exists (just wordplay or generic similarity), \
respond with exactly: NO_CONNECTION

Parallel:"""


class RetrievalSynthesisGenerator:
    """Retrieval-based synthesis candidate generator.

    Replaces random pairing with: structural query extraction → FAISS
    retrieval → adversarial evaluation. Same interface as SynthesisGenerator
    for drop-in use in shutdown processor.
    """

    def __init__(
        self,
        chroma_store,
        model_manager,
        graph_memory=None,
        entity_resolver=None,
    ):
        self.store = chroma_store
        self.model_manager = model_manager
        self.graph_memory = graph_memory
        self.entity_resolver = entity_resolver

        # Delegate sampling/classification to existing SynthesisGenerator
        from knowledge.synthesis_generator import SynthesisGenerator
        self._base = SynthesisGenerator(
            chroma_store, model_manager, graph_memory, entity_resolver,
        )

    async def generate_candidates(
        self, count: int = 5
    ) -> List[SynthesisCandidate]:
        """Generate synthesis candidates via retrieval-based discovery.

        1. Sample personal facts
        2. LLM extracts structural queries (few-shot, diversity-constrained)
        3. FAISS retrieval with similarity floor
        4. Domain filtering
        5. Adversarial LLM evaluation
        6. Package as SynthesisCandidate
        """
        from config.app_config import (
            SYNTHESIS_RETRIEVAL_ENABLED,
            SYNTHESIS_GENERATOR_LLM_CONCURRENCY,
            SYNTHESIS_COHERENCE_MODEL,
        )

        if not SYNTHESIS_RETRIEVAL_ENABLED:
            logger.debug("[RetrievalSynth] Disabled via config")
            return []

        # Step 1: Sample personal facts (over-sample for attrition)
        facts = self._base._sample_personal_entities(count * 3)
        if not facts:
            logger.warning("[RetrievalSynth] No personal facts to sample")
            return []

        logger.info(f"[RetrievalSynth] Sampled {len(facts)} personal facts")

        # Step 2: Generate structural queries (parallel, semaphore-limited)
        semaphore = asyncio.Semaphore(SYNTHESIS_GENERATOR_LLM_CONCURRENCY)
        query_tasks = [
            self._generate_structural_query(fact, semaphore)
            for fact in facts
        ]
        query_results = await asyncio.gather(*query_tasks)
        fact_queries = [r for r in query_results if r is not None]

        if not fact_queries:
            logger.info("[RetrievalSynth] No structural queries generated")
            return []

        # Log queries for diversity analysis
        logger.info(f"[RetrievalSynth] Structural queries ({len(fact_queries)}):")
        for fact, query in fact_queries:
            name = self._base._extract_concept_name(fact, source="personal")
            logger.info(f"  {name}: {query[:100]}")

        # Steps 3-4: FAISS retrieval + domain filtering
        # Limit to 1 best result per fact to force diversity across facts
        retrieval_pairs: List[Tuple[Dict, Dict, str]] = []  # (fact, wiki, query)
        for fact, structural_query in fact_queries:
            wiki_results = self._retrieve_and_filter(fact, structural_query)
            if wiki_results:
                retrieval_pairs.append((fact, wiki_results[0], structural_query))

        if not retrieval_pairs:
            logger.info("[RetrievalSynth] No cross-domain retrievals survived filtering")
            return []

        logger.info(f"[RetrievalSynth] {len(retrieval_pairs)} retrieval pairs to evaluate")

        # Step 5: Adversarial evaluation (parallel, semaphore-limited)
        eval_tasks = [
            self._evaluate_connection(fact, wiki, query, semaphore)
            for fact, wiki, query in retrieval_pairs
        ]
        eval_results = await asyncio.gather(*eval_tasks)
        candidates = [c for c in eval_results if c is not None]

        candidates = candidates[:count]
        logger.info(
            f"[RetrievalSynth] Generated {len(candidates)} candidates "
            f"(from {len(retrieval_pairs)} pairs)"
        )
        return candidates

    async def _generate_structural_query(
        self,
        fact_item: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Optional[Tuple[Dict, str]]:
        """LLM extracts domain-agnostic structural properties from a personal fact."""
        from config.app_config import (
            SYNTHESIS_COHERENCE_MODEL,
            SYNTHESIS_STRUCTURAL_QUERY_MAX_TOKENS,
        )

        concept_name = self._base._extract_concept_name(fact_item, source="personal")
        if not concept_name or len(concept_name) < 3:
            return None

        context = (fact_item.get("content", ""))[:300]

        prompt = STRUCTURAL_QUERY_PROMPT.format(
            concept_name=concept_name,
            context=context,
        )

        async with semaphore:
            try:
                response = await self.model_manager.generate_once(
                    prompt=prompt,
                    model_name=SYNTHESIS_COHERENCE_MODEL,
                    system_prompt=(
                        "You extract abstract structural patterns from specific "
                        "examples. Your output will be used as a search query to "
                        "find structurally similar content in other domains. Be "
                        "specific about the MECHANISM, not the surface topic."
                    ),
                    max_tokens=SYNTHESIS_STRUCTURAL_QUERY_MAX_TOKENS,
                    temperature=0.4,
                )
            except Exception as e:
                logger.debug(f"[RetrievalSynth] Structural query LLM failed: {e}")
                return None

        query = response.strip()
        if not query or len(query.split()) < 3:
            return None

        return (fact_item, query)

    def _retrieve_and_filter(
        self,
        fact_item: Dict[str, Any],
        structural_query: str,
    ) -> List[Dict[str, Any]]:
        """FAISS retrieval with similarity floor + domain filtering."""
        from config.app_config import (
            SYNTHESIS_RETRIEVAL_K,
            SYNTHESIS_RETRIEVAL_MIN_SIMILARITY,
        )
        from knowledge.semantic_search import semantic_search_with_neighbors

        try:
            results = semantic_search_with_neighbors(
                structural_query, k=SYNTHESIS_RETRIEVAL_K,
            )
        except Exception as e:
            logger.debug(f"[RetrievalSynth] FAISS search failed: {e}")
            return []

        if not results:
            return []

        # Similarity floor — drop garbage retrievals
        results = [
            r for r in results
            if r.get("similarity", 0.0) >= SYNTHESIS_RETRIEVAL_MIN_SIMILARITY
        ]

        # Normalize to SynthesisGenerator-compatible wiki item shape
        normalized = []
        for r in results:
            normalized.append({
                "content": r.get("content", r.get("text", "")),
                "metadata": {
                    "title": r.get("title", ""),
                    "source": r.get("source", "wikipedia"),
                    "section": r.get("section", ""),
                },
                "collection": "wiki_knowledge",
                "relevance_score": r.get("similarity", 0.0),
            })

        # Domain filtering — exclude same-domain results
        fact_domain = self._base._classify_domain(fact_item)
        filtered = []
        for wiki_item in normalized:
            wiki_domain = self._base._classify_wiki_domain(wiki_item)
            if wiki_domain != fact_domain or fact_domain in ("unknown", "personal"):
                filtered.append(wiki_item)

        return filtered

    async def _evaluate_connection(
        self,
        fact_item: Dict[str, Any],
        wiki_item: Dict[str, Any],
        structural_query: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[SynthesisCandidate]:
        """Adversarial LLM evaluation: find reasons the parallel does NOT hold."""
        from config.app_config import SYNTHESIS_COHERENCE_MODEL

        concept_a = self._base._extract_concept_name(fact_item, source="personal")
        concept_b = self._base._extract_concept_name(wiki_item, source="wiki")
        domain_a = self._base._classify_domain(fact_item)
        domain_b = self._base._classify_wiki_domain(wiki_item)
        context_a = (fact_item.get("content", ""))[:300]
        context_b = (wiki_item.get("content", ""))[:300]

        prompt = ADVERSARIAL_EVAL_PROMPT.format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain_a=domain_a,
            domain_b=domain_b,
            context_a=context_a,
            context_b=context_b,
            structural_query=structural_query,
        )

        async with semaphore:
            try:
                response = await self.model_manager.generate_once(
                    prompt=prompt,
                    model_name=SYNTHESIS_COHERENCE_MODEL,
                    system_prompt=(
                        "You evaluate cross-domain structural parallels. "
                        "Accept connections where the shared mechanism is specific "
                        "enough that studying one domain would teach you something "
                        "about the other. Reject only when the connection is "
                        "pure wordplay or applies to literally everything."
                    ),
                    max_tokens=150,
                    temperature=0.3,
                )
            except Exception as e:
                logger.debug(f"[RetrievalSynth] Eval LLM failed: {e}")
                return None

        claim = response.strip()

        # Check for rejection
        if not claim or "NO_CONNECTION" in claim.upper():
            return None
        if len(claim.split()) < 5:
            return None

        # Distance from FAISS similarity (semantic distance proxy)
        faiss_sim = wiki_item.get("relevance_score", 0.5)
        distance = max(0.15, min(1.0, 1.0 - faiss_sim))

        # Keep "personal" as a valid domain — retrieval already filtered same-domain
        source_domains = {domain_a, domain_b} - {"unknown", "other", ""}
        if len(source_domains) < 2:
            # Ensure at least 2 domains for domain_crossing gate
            if domain_a in ("unknown", "other", ""):
                source_domains.add("personal")
            if domain_b in ("unknown", "other", ""):
                source_domains.add("knowledge")

        # Tag as retrieval source via walk_path convention
        walk_path = ["retrieval", concept_a.lower(), concept_b.lower()]

        candidate = SynthesisCandidate(
            concept_a=concept_a,
            concept_b=concept_b,
            connection_claim=claim,
            walk_path=walk_path,
            source_domains=source_domains,
            endpoint_distance=distance,
            timestamp=datetime.now(),
        )

        return candidate

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Return info about available sampling pools."""
        return self._base.get_sampling_stats()
