# knowledge/graph_walk_generator.py
"""
Graph walk synthesis candidate generator.

Generates SynthesisCandidate objects via biased Markov random walks
on the knowledge graph.  Walks start from personal entity nodes and
must cross the personal→wikidata boundary, ending on a different
personal entity.

The walk articulation prompt narrates a real multi-hop path instead
of asking the LLM to invent a connection from scratch.

Inputs:  GraphMemory (with personal + wikidata nodes), EntityResolver, ModelManager
Outputs: List[SynthesisCandidate] compatible with SynthesisFilter pipeline
"""

import asyncio
import hashlib
import random
from datetime import datetime
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("graph_walk_generator")

# Walk narration prompt — LLM interprets a real path, not invents one
WALK_PROMPT = """You are given two personal concepts connected through a chain of general knowledge.

Concept A ({domain_a}): {concept_a}
Concept B ({domain_b}): {concept_b}
Knowledge path: {path_str}

Each arrow represents a real relationship in a knowledge graph.

Describe the specific structural connection this path reveals between Concept A \
and Concept B. Focus on the shared mechanism, feedback loop, or structural \
isomorphism — not surface-level analogy. If the path doesn't reveal a meaningful \
connection, respond with NO_CONNECTION.

Connection:"""


class GraphWalkGenerator:
    """Generates synthesis candidates via biased Markov walks on the knowledge graph."""

    def __init__(self, graph_memory, entity_resolver, model_manager):
        self.graph = graph_memory
        self.resolver = entity_resolver
        self.model_manager = model_manager

    async def generate_candidates(self, count: int = 10) -> list:
        """Generate synthesis candidates via graph walks.

        1. Select personal seed nodes with mixed neighborhoods
        2. Run multiple random walks per seed
        3. Filter walks that cross the personal→wikidata boundary
        4. LLM-articulate bridges for qualifying walks
        5. Return as SynthesisCandidate objects
        """
        from config.app_config import (
            GRAPH_WALK_WALKS_PER_SEED,
            GRAPH_WALK_MAX_LENGTH,
            GRAPH_WALK_MAX_CANDIDATES,
            SYNTHESIS_COHERENCE_MODEL,
            SYNTHESIS_GENERATOR_LLM_CONCURRENCY,
        )

        seeds = self._select_seed_nodes(count * 2)
        if not seeds:
            logger.info("[GraphWalk] No suitable seed nodes found")
            return []

        # Collect qualifying walks
        qualifying = []
        for seed in seeds:
            for _ in range(GRAPH_WALK_WALKS_PER_SEED):
                walk = self._random_walk(seed, GRAPH_WALK_MAX_LENGTH)
                if self._qualifies_as_candidate(walk):
                    qualifying.append(walk)

            if len(qualifying) >= count * 3:
                break

        if not qualifying:
            logger.info("[GraphWalk] No qualifying walks found (need boundary crossing)")
            return []

        # Deduplicate by endpoint pair
        seen_pairs = set()
        unique_walks = []
        for walk in qualifying:
            pair = frozenset([walk[0], walk[-1]])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_walks.append(walk)

        logger.info(
            f"[GraphWalk] {len(qualifying)} qualifying walks, "
            f"{len(unique_walks)} unique endpoint pairs"
        )

        # Articulate bridges via LLM (parallel with semaphore)
        semaphore = asyncio.Semaphore(SYNTHESIS_GENERATOR_LLM_CONCURRENCY)
        tasks = [
            self._articulate_and_package(walk, semaphore)
            for walk in unique_walks[:count * 2]
        ]
        results = await asyncio.gather(*tasks)
        candidates = [c for c in results if c is not None]

        # Cap
        candidates = candidates[:min(count, GRAPH_WALK_MAX_CANDIDATES)]
        logger.info(f"[GraphWalk] Generated {len(candidates)} candidates from graph walks")
        return candidates

    def _select_seed_nodes(self, count: int) -> list[str]:
        """Select personal entity nodes as walk starting points.

        Prefers nodes with mixed neighborhoods (have both personal and
        wikidata neighbors), which are more likely to produce
        boundary-crossing walks.

        Hub dampening: nodes with degree > HUB_DEGREE_THRESHOLD get a
        log-scale penalty to prevent high-degree nodes (like "states")
        from dominating seed selection.
        """
        import math
        from config.app_config import GRAPH_WALK_HUB_DEGREE_THRESHOLD

        candidates = []
        for nid, data in self.graph.graph.nodes(data=True):
            source = data.get("metadata", {}).get("source", "personal")
            if source != "personal":
                continue
            if nid == "user":
                continue

            # Score by neighborhood diversity
            neighbors = list(self.graph.graph.successors(nid)) + \
                        list(self.graph.graph.predecessors(nid))
            if not neighbors:
                continue

            has_wikidata_neighbor = any(
                self.graph.graph.nodes.get(n, {}).get("metadata", {}).get("source") == "wikidata"
                for n in neighbors
            )
            degree = len(neighbors)

            # Hub dampening: penalize high-degree nodes logarithmically
            if degree > GRAPH_WALK_HUB_DEGREE_THRESHOLD:
                effective_degree = GRAPH_WALK_HUB_DEGREE_THRESHOLD + math.log2(
                    degree - GRAPH_WALK_HUB_DEGREE_THRESHOLD + 1
                )
            else:
                effective_degree = degree

            # Prioritize nodes near bridge edges
            score = effective_degree
            if has_wikidata_neighbor:
                score += 100

            candidates.append((nid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Mix: top-scored + some random for diversity
        top = [c[0] for c in candidates[:count // 2]]
        remaining = [c[0] for c in candidates[count // 2:]]
        if remaining:
            random_picks = random.sample(remaining, min(count // 2, len(remaining)))
        else:
            random_picks = []

        seeds = top + random_picks
        random.shuffle(seeds)
        return seeds[:count]

    def _random_walk(self, start: str, max_length: int) -> list[str]:
        """Biased Markov chain random walk.

        Treats the DiGraph bidirectionally for traversal (follows both
        in-edges and out-edges).  Once the walk enters wikidata territory,
        inflates transition probabilities toward personal nodes by
        personal_return_bias.
        """
        from config.app_config import (
            GRAPH_WALK_RESTART_PROB,
            GRAPH_WALK_PERSONAL_RETURN_BIAS,
        )

        walk = [start]
        current = start
        crossed_boundary = False

        for _ in range(max_length - 1):
            # Restart probability
            if random.random() < GRAPH_WALK_RESTART_PROB:
                current = start
                walk = [start]
                crossed_boundary = False
                continue

            # Get all neighbors (bidirectional)
            neighbors = set(self.graph.graph.successors(current)) | \
                        set(self.graph.graph.predecessors(current))
            # Don't revisit immediately
            if len(walk) > 1:
                neighbors.discard(walk[-2])
            # Don't visit "user" hub — too central, makes walks boring
            neighbors.discard("user")

            if not neighbors:
                break

            # Weight edges
            weighted = []
            current_source = self._get_node_source(current)

            for n in neighbors:
                # Base weight from edge
                edge_data = self.graph.graph.edges.get((current, n), {})
                if not edge_data:
                    edge_data = self.graph.graph.edges.get((n, current), {})
                w = edge_data.get("weight", 1.0)

                # Bridge confidence weighting
                bridge_conf = edge_data.get("metadata", {}).get("bridge_confidence", 1.0)
                w *= bridge_conf

                # Node2Vec-style return bias: once in wikidata territory,
                # inflate weights toward personal nodes
                n_source = self._get_node_source(n)
                if crossed_boundary and current_source == "wikidata" and n_source == "personal":
                    w *= GRAPH_WALK_PERSONAL_RETURN_BIAS

                weighted.append((n, w))

            if not weighted:
                break

            # Sample proportional to weights
            total_w = sum(w for _, w in weighted)
            r = random.random() * total_w
            cumulative = 0.0
            chosen = weighted[0][0]
            for n, w in weighted:
                cumulative += w
                if cumulative >= r:
                    chosen = n
                    break

            walk.append(chosen)
            current = chosen

            # Track boundary crossing
            if not crossed_boundary:
                chosen_source = self._get_node_source(chosen)
                if chosen_source != "personal":
                    crossed_boundary = True

        return walk

    def _qualifies_as_candidate(self, walk: list[str]) -> bool:
        """Check if a walk path qualifies as a synthesis candidate.

        Requirements:
          - Starts at a personal node
          - Ends at a personal node (different from start)
          - Passes through at least one non-personal node
          - Length >= min_path_length
          - Walk touches ≥ GRAPH_WALK_MIN_DOMAINS distinct domain categories
            (counts both endpoint domains + wikidata domain_category metadata)
        """
        from config.app_config import GRAPH_WALK_MIN_PATH, GRAPH_WALK_MIN_DOMAINS

        if len(walk) < GRAPH_WALK_MIN_PATH:
            return False

        start_source = self._get_node_source(walk[0])
        end_source = self._get_node_source(walk[-1])

        if start_source != "personal" or end_source != "personal":
            return False

        if walk[0] == walk[-1]:
            return False

        # Must cross boundary at least once
        has_non_personal = False
        for node in walk[1:-1]:
            if self._get_node_source(node) != "personal":
                has_non_personal = True
                break

        if not has_non_personal:
            return False

        # Cross-domain constraint: collect domains from all walk nodes
        domains = set()
        for node in walk:
            data = self.graph.graph.nodes.get(node, {})
            # Wikidata nodes have domain_category in metadata
            domain_cat = data.get("metadata", {}).get("domain_category", "")
            if domain_cat:
                domains.add(domain_cat)
            else:
                # Personal nodes: classify via user edge relations
                domain = self._classify_endpoint_domain(node)
                if domain and domain not in ("personal", "unknown", "other"):
                    domains.add(domain)

        if len(domains) < GRAPH_WALK_MIN_DOMAINS:
            return False

        return True

    def _get_node_source(self, entity_id: str) -> str:
        """Returns 'personal', 'wikidata', or 'wiki_retrieved'."""
        data = self.graph.graph.nodes.get(entity_id, {})
        return data.get("metadata", {}).get("source", "personal")

    def _get_display_name(self, entity_id: str) -> str:
        """Get the display name for a node, falling back to entity_id."""
        data = self.graph.graph.nodes.get(entity_id, {})
        return data.get("display_name", entity_id)

    def _compute_walk_distance(self, walk_path: list[str]) -> float:
        """Compute endpoint distance from actual path length."""
        from config.app_config import GRAPH_WALK_MAX_LENGTH
        normalized = len(walk_path) / GRAPH_WALK_MAX_LENGTH
        return max(0.15, min(1.0, normalized))

    def _classify_endpoint_domain(self, entity_id: str) -> str:
        """Classify a personal entity into a domain for the candidate."""
        # Check outgoing relations from user to this entity
        for src, _, edata in self.graph.graph.in_edges(entity_id, data=True):
            if src == "user":
                rel = edata.get("relation", "")
                try:
                    from memory.user_profile_schema import categorize_relation
                    cat = categorize_relation(rel)
                    if cat.value != "other":
                        return cat.value
                except Exception:
                    pass

        # Check the node's own domain_category metadata
        data = self.graph.graph.nodes.get(entity_id, {})
        domain = data.get("metadata", {}).get("domain_category", "")
        if domain:
            return domain

        return "personal"

    async def _articulate_and_package(self, walk: list[str], semaphore) -> Optional[object]:
        """Articulate the walk path via LLM and package as SynthesisCandidate."""
        from config.app_config import SYNTHESIS_COHERENCE_MODEL
        from knowledge.synthesis_models import SynthesisCandidate

        concept_a = self._get_display_name(walk[0])
        concept_b = self._get_display_name(walk[-1])
        domain_a = self._classify_endpoint_domain(walk[0])
        domain_b = self._classify_endpoint_domain(walk[-1])

        # Build display path
        display_path = [self._get_display_name(n) for n in walk]
        path_str = " → ".join(display_path)

        prompt = WALK_PROMPT.format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain_a=domain_a,
            domain_b=domain_b,
            path_str=path_str,
        )

        async with semaphore:
            try:
                response = await self.model_manager.generate_once(
                    prompt=prompt,
                    model_name=SYNTHESIS_COHERENCE_MODEL,
                    system_prompt=(
                        "You identify structural connections between concepts "
                        "by interpreting knowledge graph paths. Be specific and "
                        "concrete. If no real connection exists, say NO_CONNECTION."
                    ),
                    max_tokens=200,
                    temperature=0.5,
                )
            except Exception as e:
                logger.debug(f"[GraphWalk] Bridge LLM call failed: {e}")
                return None

        claim = response.strip()

        # Filter NO_CONNECTION and too-short responses
        if not claim or "NO_CONNECTION" in claim.upper():
            return None
        if len(claim.split()) < 5:
            return None

        # Build candidate — collect domains from all walk nodes for richer signal
        source_domains = {domain_a, domain_b}
        for node in walk[1:-1]:
            data = self.graph.graph.nodes.get(node, {})
            dc = data.get("metadata", {}).get("domain_category", "")
            if dc:
                source_domains.add(dc)
        source_domains -= {"unknown", "personal", "other", ""}
        if len(source_domains) < 1:
            source_domains = {"personal", "knowledge"}

        distance = self._compute_walk_distance(walk)

        candidate = SynthesisCandidate(
            concept_a=concept_a,
            concept_b=concept_b,
            connection_claim=claim,
            walk_path=walk,  # real multi-hop path
            source_domains=source_domains,
            endpoint_distance=distance,
            timestamp=datetime.now(),
        )

        return candidate
