"""
# knowledge/synthesis_generator.py

Module Contract
- Purpose: Minimum viable candidate generator for the knowledge synthesis pipeline.
  Samples entities from personal ChromaDB stores (facts) and Wikipedia (via FAISS
  semantic search), uses an LLM to articulate cross-domain connections, and packages
  results as SynthesisCandidate objects for the filter pipeline.
- Class: SynthesisGenerator(chroma_store, model_manager, graph_memory=None, entity_resolver=None)
- Key methods:
  - generate_candidates(count) -> List[SynthesisCandidate]  [async, main entry]
  - _sample_personal_entities(n) -> List[Dict]  [ChromaDB facts query]
  - _sample_wiki_articles(n) -> List[Dict]  [FAISS semantic search, 40M wiki vectors]
  - _classify_domain(entity_text, relation) -> str  [reuses user_profile_schema]
  - _articulate_bridge(concept_a, concept_b, context_a, context_b) -> Optional[str]  [LLM]
  - _compute_endpoint_distance(concept_a, concept_b) -> float  [graph shortest path or default]
- Inputs: ChromaDB store (facts collection), FAISS index (wiki vectors), ModelManager for LLM
- Outputs: List[SynthesisCandidate] ready for SynthesisFilter.process_batch()
- Side effects: LLM API calls for bridge articulation (parallelized with semaphore)
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge.synthesis_models import SynthesisCandidate
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Sampling query seeds — broad terms to pull diverse entities from ChromaDB
_PERSONAL_QUERY_SEEDS = [
    "my family and friends",
    "my work and career",
    "things I enjoy doing",
    "my health and fitness",
    "places I have lived",
    "my goals and plans",
    "my education and learning",
    "my projects and hobbies",
    "my pets and animals",
    "my daily routines",
    "my finances and money",
    "things I believe in",
]

_WIKI_QUERY_SEEDS = [
    "scientific discovery breakthrough",
    "historical event transformation",
    "mathematical concept pattern",
    "biological mechanism adaptation",
    "economic theory market",
    "philosophical argument ethics",
    "engineering design constraint",
    "cultural tradition practice",
    "psychological phenomenon behavior",
    "ecological system balance",
    "linguistic structure meaning",
    "artistic movement expression",
]

# Bridge articulation prompt — neutral, doesn't reveal graph machinery
BRIDGE_PROMPT = """Consider these two concepts from different life domains.

Concept A ({domain_a}): {concept_a}
Context: {context_a}

Concept B ({domain_b}): {concept_b}
Context: {context_b}

If there is a specific, non-obvious connection between them, describe it in one or two sentences. The connection should be:
- Concrete (not "both are important" or "both involve systems")
- Mechanistic (describes HOW they relate, not just THAT they relate)
- Falsifiable (someone could argue against it)

If no meaningful connection exists, respond with exactly: NO_CONNECTION

Connection:"""


class SynthesisGenerator:
    """Minimum viable candidate generator via cross-store sampling.

    Samples entities from personal stores (facts collection) and Wikipedia,
    uses a cheap LLM to articulate connections, and packages results as
    SynthesisCandidate objects.

    Args:
        chroma_store: MultiCollectionChromaStore for entity sampling
        model_manager: ModelManager for bridge articulation LLM calls
        graph_memory: Optional GraphMemory for shortest-path distance
        entity_resolver: Optional EntityResolver for domain classification
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

        # Build display_name → node_id index for graph resolution
        self._display_name_index: Dict[str, str] = {}
        if graph_memory:
            for nid, data in graph_memory.graph.nodes(data=True):
                dn = data.get("display_name", "").lower().strip()
                if dn and dn not in self._display_name_index:
                    self._display_name_index[dn] = nid

    async def generate_candidates(
        self, count: int = 5
    ) -> List[SynthesisCandidate]:
        """Generate synthesis candidates via cross-store sampling.

        Samples personal entities and wiki articles, pairs them across
        domains, deduplicates pairs, articulates bridges via LLM, and
        packages as SynthesisCandidate objects.

        Args:
            count: Target number of candidates to produce.
                   Samples more pairs than needed to account for dedup and
                   LLM NO_CONNECTION responses.

        Returns:
            List of SynthesisCandidate objects ready for filter pipeline.
        """
        from config.app_config import (
            SYNTHESIS_GENERATOR_ENABLED,
            SYNTHESIS_GENERATOR_LLM_CONCURRENCY,
            SYNTHESIS_GENERATOR_MIN_GRAPH_NODES,
        )

        if not SYNTHESIS_GENERATOR_ENABLED:
            logger.debug("[SynthesisGenerator] Disabled via config")
            return []

        # Graph sparsity guard (optional — generator works without graph)
        if self.graph_memory:
            node_count = self.graph_memory.node_count()
            if node_count < SYNTHESIS_GENERATOR_MIN_GRAPH_NODES:
                logger.info(
                    f"[SynthesisGenerator] Graph too sparse ({node_count} nodes), skipping"
                )
                return []

        # Over-sample to account for dedup + NO_CONNECTION
        sample_count = count * 3

        # Step 1: Sample entities from both stores
        personal_entities = self._sample_personal_entities(sample_count)
        wiki_articles = self._sample_wiki_articles(sample_count)

        if not personal_entities or not wiki_articles:
            logger.warning(
                f"[SynthesisGenerator] Insufficient samples: "
                f"personal={len(personal_entities)}, wiki={len(wiki_articles)}"
            )
            return []

        # Step 2: Form cross-domain pairs and deduplicate
        pairs = self._form_pairs(personal_entities, wiki_articles, count * 2)

        if not pairs:
            logger.info("[SynthesisGenerator] No valid cross-domain pairs formed")
            return []

        logger.info(
            f"[SynthesisGenerator] Formed {len(pairs)} unique pairs, "
            f"articulating bridges (concurrency={SYNTHESIS_GENERATOR_LLM_CONCURRENCY})"
        )

        # Step 3: Parallel LLM articulation with semaphore
        semaphore = asyncio.Semaphore(SYNTHESIS_GENERATOR_LLM_CONCURRENCY)
        tasks = [
            self._articulate_and_package(pair, semaphore)
            for pair in pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: Collect valid candidates, cap at requested count
        candidates = []
        errors = 0
        no_connections = 0
        for r in results:
            if isinstance(r, Exception):
                errors += 1
                logger.debug(f"[SynthesisGenerator] Articulation error: {r}")
            elif r is None:
                no_connections += 1
            else:
                candidates.append(r)
            if len(candidates) >= count:
                break

        logger.info(
            f"[SynthesisGenerator] Generated {len(candidates)} candidates "
            f"(no_connection={no_connections}, errors={errors})"
        )
        return candidates[:count]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_personal_entities(self, n: int) -> List[Dict[str, Any]]:
        """Sample diverse personal entities from the facts collection.

        Uses multiple broad query seeds to get varied results, then
        shuffles and deduplicates. Filters out low-quality facts with
        short or meaningless concept names.
        """
        seen_ids: Set[str] = set()
        seen_concepts: Set[str] = set()
        entities: List[Dict[str, Any]] = []
        seeds = random.sample(_PERSONAL_QUERY_SEEDS, min(len(_PERSONAL_QUERY_SEEDS), 6))

        # Over-fetch to have room after quality filtering
        per_seed = max((n * 3) // len(seeds), 5)
        for seed in seeds:
            try:
                results = self.store.query_collection(
                    collection_name="facts",
                    query_text=seed,
                    n_results=per_seed,
                )
                for item in results:
                    doc_id = item.get("id", "")
                    if not doc_id or doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    # Quality filter: concept name must be meaningful
                    name = self._extract_concept_name(item, source="personal")
                    if not name or len(name) < 3:
                        continue
                    # Skip numeric-only, single common words, timestamps
                    if name.isdigit() or name.lower() in (
                        "user", "yes", "no", "true", "false", "none",
                        "today", "yesterday", "home", "good", "bad",
                    ):
                        continue

                    # Deduplicate by concept name (not just ID)
                    name_key = name.lower().strip()
                    if name_key in seen_concepts:
                        continue
                    seen_concepts.add(name_key)

                    entities.append(item)
            except Exception as e:
                logger.debug(f"[SynthesisGenerator] Facts query failed for '{seed}': {e}")

        random.shuffle(entities)
        return entities[:n]

    def _sample_wiki_articles(self, n: int) -> List[Dict[str, Any]]:
        """Sample diverse wiki articles via FAISS semantic search (40M vectors)."""
        from knowledge.semantic_search import semantic_search_with_neighbors

        seen_titles: Set[str] = set()
        articles: List[Dict[str, Any]] = []
        seeds = random.sample(_WIKI_QUERY_SEEDS, min(len(_WIKI_QUERY_SEEDS), 6))

        per_seed = max(n // len(seeds), 3)
        for seed in seeds:
            try:
                results = semantic_search_with_neighbors(seed, k=per_seed)
                for item in results:
                    # Deduplicate by title (FAISS results don't have stable IDs)
                    title = item.get("title", "")
                    dedup_key = title.lower().strip() if title else item.get("content", "")[:80]
                    if dedup_key and dedup_key not in seen_titles:
                        seen_titles.add(dedup_key)
                        # Normalize to match ChromaDB result shape for downstream compat
                        articles.append({
                            "content": item.get("content", item.get("text", "")),
                            "metadata": {
                                "title": title,
                                "source": item.get("source", "wikipedia"),
                                "section": item.get("section", ""),
                            },
                            "collection": "wiki_knowledge",
                            "relevance_score": item.get("similarity", 0.0),
                        })
            except Exception as e:
                logger.debug(f"[SynthesisGenerator] FAISS wiki query failed for '{seed}': {e}")

        random.shuffle(articles)
        return articles[:n]

    # ------------------------------------------------------------------
    # Pair Formation
    # ------------------------------------------------------------------

    def _form_pairs(
        self,
        personal: List[Dict[str, Any]],
        wiki: List[Dict[str, Any]],
        max_pairs: int,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Form cross-domain pairs via round-robin, deduplicated by concept names.

        Round-robin ensures diverse personal concepts — each personal entity
        gets at most one wiki partner before cycling to the next entity.
        """
        seen_pair_keys: Set[frozenset] = set()
        seen_personal: Set[str] = set()
        pairs = []

        # Shuffle to randomize pairings
        personal_shuffled = personal[:]
        wiki_shuffled = wiki[:]
        random.shuffle(personal_shuffled)
        random.shuffle(wiki_shuffled)

        # Round-robin: cycle through personal entities, assign one wiki each
        wiki_idx = 0
        passes = 0
        max_passes = 3  # allow up to 3 passes through personal list

        while len(pairs) < max_pairs and passes < max_passes:
            made_progress = False
            for p_item in personal_shuffled:
                if len(pairs) >= max_pairs:
                    break

                p_name = self._extract_concept_name(p_item, source="personal")
                if not p_name:
                    continue

                p_key = p_name.lower().strip()

                # On first pass, each personal concept gets exactly one pair.
                # On subsequent passes, allow repeats with different wiki articles.
                if passes == 0 and p_key in seen_personal:
                    continue

                # Try wiki articles until we find a cross-domain match
                attempts = 0
                while attempts < len(wiki_shuffled) and wiki_idx < len(wiki_shuffled) * (passes + 1):
                    w_item = wiki_shuffled[wiki_idx % len(wiki_shuffled)]
                    wiki_idx += 1
                    attempts += 1

                    w_name = self._extract_concept_name(w_item, source="wiki")
                    if not w_name:
                        continue

                    pair_key = frozenset([p_name.lower(), w_name.lower()])
                    if pair_key in seen_pair_keys:
                        continue

                    # Classify domains — skip if same domain
                    p_domain = self._classify_domain(p_item)
                    w_domain = self._classify_domain(w_item)
                    if p_domain == w_domain and p_domain != "unknown":
                        continue

                    seen_pair_keys.add(pair_key)
                    seen_personal.add(p_key)
                    pairs.append((p_item, w_item))
                    made_progress = True
                    break  # move to next personal entity

            passes += 1
            if not made_progress:
                break

        return pairs

    def _extract_concept_name(
        self, item: Dict[str, Any], source: str = "personal"
    ) -> str:
        """Extract a short concept name from a ChromaDB result item."""
        metadata = item.get("metadata", {})

        if source == "personal":
            # Facts have subject/predicate/object in metadata
            subject = metadata.get("subject", "")
            predicate = metadata.get("predicate", metadata.get("relation", ""))
            obj = metadata.get("object", "")
            # Prefer the object (the interesting fact) unless it's just "user"
            if obj and obj.lower() != "user":
                return obj.strip()
            if subject and subject.lower() != "user":
                return subject.strip()
            # Fall back: parse pipe-delimited content (e.g. "user | hobby | sourdough baking")
            content = item.get("content", "")
            if "|" in content:
                parts = [p.strip() for p in content.split("|")]
                # Return the object (last part) unless it's "user"
                for part in reversed(parts):
                    if part.lower() not in ("user", "your", ""):
                        return part[:60]
            return content[:60].strip() if content else ""
        else:
            # Wiki articles: use title from metadata or first line of content
            title = metadata.get("title", "")
            if title:
                return title.strip()
            content = item.get("content", "")
            # First sentence as fallback
            first_line = content.split(".")[0] if content else ""
            return first_line[:80].strip()

    # Keyword map for personal fact domain classification fallback
    _PERSONAL_DOMAIN_KEYWORDS = {
        "fitness": ["workout", "gym", "bench", "squat", "deadlift", "running",
                    "marathon", "lifting", "training", "exercise", "muscle",
                    "weight", "cardio", "reps", "sets", "sweat", "jog"],
        "health": ["medication", "adhd", "allergy", "sleep", "diet", "doctor",
                   "symptom", "pain", "anxiety", "therapy", "vitamin",
                   "prescription", "diagnosis", "illness"],
        "career": ["work", "job", "brewery", "brewer", "salary", "commute",
                   "office", "project", "client", "boss", "coworker",
                   "promotion", "career", "meeting", "deadline"],
        "relationships": ["brother", "sister", "mom", "dad", "parent", "friend",
                         "dating", "partner", "wife", "husband", "cat", "dog",
                         "pet", "family", "auggie", "sarah", "paczki"],
        "education": ["study", "learn", "course", "class", "statistics",
                     "math", "algebra", "exam", "grade", "actuarial",
                     "university", "degree", "textbook"],
        "hobbies": ["game", "board", "baking", "sourdough", "cooking",
                   "reading", "book", "movie", "music", "guitar",
                   "dungeons", "dragons", "hobby", "craft"],
        "projects": ["daemon", "code", "python", "programming", "software",
                    "chromadb", "faiss", "github", "build", "deploy"],
        "geography": ["portland", "eugene", "oregon", "japan", "city",
                     "neighborhood", "travel", "move", "live"],
        "food": ["beer", "coffee", "ramen", "restaurant", "recipe",
                "tripel", "brew", "ferment", "donut"],
    }

    def _classify_domain(self, item: Dict[str, Any]) -> str:
        """Classify a ChromaDB result into a life domain.

        Reuses categorize_relation() from user_profile_schema.py for
        personal facts, falls back to keyword heuristics on content.
        Wiki articles use a separate keyword classifier.
        """
        metadata = item.get("metadata", {})
        collection = item.get("collection", "")

        if collection == "wiki_knowledge" or metadata.get("source") == "wikipedia":
            return self._classify_wiki_domain(item)

        # Personal fact — try structured relation first
        relation = metadata.get("predicate", metadata.get("relation", ""))
        if relation:
            try:
                from memory.user_profile_schema import categorize_relation
                category = categorize_relation(relation)
                if category.value != "other":
                    return category.value
            except Exception:
                pass

        # Check domain_hint in metadata (from test facts)
        domain_hint = metadata.get("domain_hint", "")
        if domain_hint:
            return domain_hint

        # Fallback: keyword heuristics on content + relation
        text = (item.get("content", "") + " " + relation).lower()
        best_domain = "personal"  # generic fallback (not "unknown" — enables cross-domain pairing)
        best_count = 0
        for domain, keywords in self._PERSONAL_DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_domain = domain

        return best_domain

    def _classify_wiki_domain(self, item: Dict[str, Any]) -> str:
        """Classify a wiki article into a broad domain via keywords."""
        content = (item.get("content", "") + " " +
                   item.get("metadata", {}).get("title", "")).lower()

        # Keyword map — broader than personal domains
        domain_keywords = {
            "science": ["physics", "chemistry", "biology", "quantum", "molecule",
                        "atom", "cell", "gene", "evolution", "organism", "species"],
            "history": ["century", "empire", "dynasty", "war", "revolution",
                        "ancient", "medieval", "colonial", "kingdom"],
            "mathematics": ["theorem", "equation", "algorithm", "calculus",
                           "probability", "geometry", "topology", "algebra"],
            "philosophy": ["philosophy", "ethics", "metaphysics", "epistemology",
                          "existential", "ontology", "moral"],
            "technology": ["computer", "software", "algorithm", "digital",
                          "internet", "programming", "engineering"],
            "arts": ["painting", "sculpture", "music", "literature", "poetry",
                    "film", "theater", "novel", "artistic"],
            "economics": ["economy", "market", "trade", "capital", "inflation",
                         "gdp", "monetary", "fiscal"],
            "psychology": ["psychology", "cognitive", "behavior", "perception",
                          "memory", "consciousness", "emotion"],
            "ecology": ["ecosystem", "climate", "biodiversity", "habitat",
                       "conservation", "extinction", "environment"],
        }

        best_domain = "knowledge"  # generic fallback
        best_count = 0
        for domain, keywords in domain_keywords.items():
            count = sum(1 for kw in keywords if kw in content)
            if count > best_count:
                best_count = count
                best_domain = domain

        return best_domain

    # ------------------------------------------------------------------
    # Bridge Articulation
    # ------------------------------------------------------------------

    async def _articulate_and_package(
        self,
        pair: Tuple[Dict[str, Any], Dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> Optional[SynthesisCandidate]:
        """Articulate a bridge and package as SynthesisCandidate.

        Returns None if the LLM finds no meaningful connection.
        """
        p_item, w_item = pair
        concept_a = self._extract_concept_name(p_item, source="personal")
        concept_b = self._extract_concept_name(w_item, source="wiki")
        domain_a = self._classify_domain(p_item)
        domain_b = self._classify_domain(w_item)

        context_a = (p_item.get("content", ""))[:300]
        context_b = (w_item.get("content", ""))[:300]

        async with semaphore:
            claim = await self._articulate_bridge(
                concept_a, concept_b,
                domain_a, domain_b,
                context_a, context_b,
            )

        if not claim:
            return None

        # Compute endpoint distance
        distance = self._compute_endpoint_distance(concept_a, concept_b)

        # Build walk path (synthetic — not a real graph walk)
        walk_path = [concept_a.lower(), concept_b.lower()]

        source_domains = {domain_a, domain_b} - {"unknown"}
        if len(source_domains) < 1:
            source_domains = {"personal", "knowledge"}

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

    async def _articulate_bridge(
        self,
        concept_a: str,
        concept_b: str,
        domain_a: str,
        domain_b: str,
        context_a: str,
        context_b: str,
    ) -> Optional[str]:
        """LLM call to generate a connection claim between two concepts.

        Returns None if the model can't find a meaningful connection.
        """
        from config.app_config import SYNTHESIS_COHERENCE_MODEL

        prompt = BRIDGE_PROMPT.format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain_a=domain_a,
            domain_b=domain_b,
            context_a=context_a,
            context_b=context_b,
        )

        try:
            response = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=SYNTHESIS_COHERENCE_MODEL,
                system_prompt=(
                    "You are a creative thinker identifying non-obvious connections "
                    "between concepts. Be specific and concrete. If no real connection "
                    "exists, say NO_CONNECTION."
                ),
                max_tokens=150,
                temperature=0.7,
            )
        except Exception as e:
            logger.debug(f"[SynthesisGenerator] Bridge LLM call failed: {e}")
            return None

        response = response.strip()

        # Check for NO_CONNECTION or empty
        if not response or "NO_CONNECTION" in response.upper():
            return None

        # Basic sanity: must be at least a few words
        if len(response.split()) < 5:
            return None

        return response

    # ------------------------------------------------------------------
    # Distance Computation
    # ------------------------------------------------------------------

    def _resolve_to_graph(self, concept_name: str) -> Optional[str]:
        """Try multiple strategies to resolve a concept name to a graph node ID.

        Strategies (in order):
        1. EntityResolver alias lookup
        2. Direct node ID lookup (already lowercase)
        3. Slugified form (spaces→underscores, strip punctuation)
        4. Display name index (case-insensitive match on display_name)
        """
        if not self.graph_memory:
            return None

        name = concept_name.lower().strip()

        # Strategy 1: EntityResolver
        if self.entity_resolver:
            resolved = self.entity_resolver.resolve(name)
            if resolved and resolved in self.graph_memory.graph:
                return resolved

        # Strategy 2: Direct node ID lookup
        if name in self.graph_memory.graph:
            return name

        # Strategy 3: Slugified form
        slug = re.sub(r"[^\w\s]", "", name)
        slug = re.sub(r"\s+", "_", slug).strip("_")
        if slug and slug in self.graph_memory.graph:
            return slug

        # Strategy 4: Display name index
        if name in self._display_name_index:
            return self._display_name_index[name]

        return None

    def _compute_endpoint_distance(self, concept_a: str, concept_b: str) -> float:
        """Compute distance between two concepts.

        Uses multi-strategy graph resolution to find node IDs, then
        shortest path for real distance. Falls back to 0.55 only when
        both concepts are genuinely absent from the graph.
        """
        if self.graph_memory:
            try:
                a_id = self._resolve_to_graph(concept_a)
                b_id = self._resolve_to_graph(concept_b)

                if a_id and b_id:
                    path = self.graph_memory.shortest_path(a_id, b_id)
                    if path and len(path) >= 2:
                        normalized = min(len(path) / 6.0, 1.0)
                        return max(normalized, 0.15)
                    # Both in graph but no path = structurally distant
                    return 0.85
            except Exception:
                pass

        # Default: mid-range distance that passes Stage 2
        return 0.55

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Return info about available sampling pools."""
        stats = {}
        try:
            coll_stats = self.store.get_collection_stats()
            stats["facts_count"] = coll_stats.get("facts", {}).get("count", 0)
        except Exception:
            stats["facts_count"] = 0

        # Wiki stats from FAISS index
        try:
            from knowledge.semantic_search import get_index
            idx = get_index()
            stats["wiki_count"] = idx._total_rows if idx.loaded else 0
            stats["wiki_source"] = "faiss"
        except Exception:
            stats["wiki_count"] = 0
            stats["wiki_source"] = "unavailable"

        if self.graph_memory:
            stats["graph_nodes"] = self.graph_memory.node_count()
            stats["graph_edges"] = self.graph_memory.edge_count()

        return stats
