# memory/context_surfacer.py
"""
Proactive cross-domain context surfacing engine.

Classifies knowledge graph entities into life domains (HEALTH, CAREER, etc.),
detects which domains the current conversation touches, and uses a single LLM
call to synthesize insights that bridge across domains.

Leverages the star topology of the production graph (most entities connect
directly to "user") by classifying user-adjacent entities via the existing
categorize_relation() from user_profile_schema.py.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Set

from utils.logging_utils import get_logger

logger = get_logger("context_surfacer")

# Relation overrides for edges that categorize_relation() misclassifies.
# These take precedence over the schema's heuristic fallbacks.
_SURFACING_RELATION_OVERRIDES: Dict[str, str] = {
    "pet": "hobbies",
    "pet_of": "hobbies",
    "pet_name": "hobbies",
    "cat_name": "hobbies",
    "dog_name": "hobbies",
    "sibling_of": "relationships",
    "brother_name": "relationships",
    "sister_name": "relationships",
    "parent_of": "relationships",
    "child_of": "relationships",
    "mother_name": "relationships",
    "father_name": "relationships",
    "partner": "relationships",
    "spouse": "relationships",
    "friend": "relationships",
    "roommate": "relationships",
}

# Keyword heuristics for domain detection when entity resolution fails.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "health": ["health", "medication", "medicine", "symptom", "condition", "sleep",
               "illness", "doctor", "therapy", "anxiety", "depression", "adhd",
               "diagnosis", "treatment", "chronic", "pain", "fatigue"],
    "career": ["work", "job", "career", "boss", "coworker", "company", "salary",
               "promotion", "office", "manager", "employed", "shift"],
    "fitness": ["gym", "workout", "lift", "squat", "bench", "deadlift", "weight",
                "bulk", "cut", "macro", "protein", "cardio", "exercise", "training"],
    "education": ["school", "class", "course", "degree", "study", "exam", "grade",
                  "university", "college", "homework", "professor", "lecture"],
    "relationships": ["family", "brother", "sister", "mom", "dad", "parent",
                      "friend", "partner", "wife", "husband", "girlfriend",
                      "boyfriend", "roommate", "daughter", "son"],
    "hobbies": ["game", "gaming", "hobby", "music", "art", "reading", "cooking",
                "pet", "cat", "dog", "travel", "movie", "show"],
    "finance": ["money", "budget", "savings", "debt", "loan", "rent", "income",
                "expense", "invest", "insurance", "tax"],
    "goals": ["goal", "plan", "resolution", "ambition", "aspiration", "dream",
              "want to", "going to", "planning to"],
    "projects": ["project", "app", "build", "code", "develop", "feature",
                 "deploy", "release", "launch", "repo"],
}


# Merge per-user personal domain terms from config (keeps source general).
# config keys are lowercase category names; only those matching an existing
# domain are merged. See config/app_config.py PROFILE_PERSONAL_CATEGORY_TOKENS.
def _merge_personal_domain_keywords() -> None:
    try:
        from config.app_config import PROFILE_PERSONAL_CATEGORY_TOKENS
    except Exception:
        return
    for domain, tokens in (PROFILE_PERSONAL_CATEGORY_TOKENS or {}).items():
        key = str(domain).lower()
        if key in _DOMAIN_KEYWORDS:
            for t in (tokens or []):
                tl = str(t).lower()
                if tl not in _DOMAIN_KEYWORDS[key]:
                    _DOMAIN_KEYWORDS[key].append(tl)


_merge_personal_domain_keywords()


class ContextSurfacer:
    """Cross-domain insight generation from knowledge graph."""

    def __init__(self, graph_memory, entity_resolver, model_manager):
        self._graph_memory = graph_memory
        self._entity_resolver = entity_resolver
        self._model_manager = model_manager
        self._session_insights: Optional[List] = None  # Cache per session
        self._domain_cache: Optional[Dict] = None
        self._domain_cache_edge_count: int = 0

        # Lazy-init history
        self._history = None

    def _get_history(self):
        """Lazy-load surfacing history."""
        if self._history is None:
            from config.app_config import PROACTIVE_SURFACING_HISTORY_PATH
            from memory.surfacing_history import SurfacingHistory
            self._history = SurfacingHistory(persist_path=PROACTIVE_SURFACING_HISTORY_PATH)
        return self._history

    async def generate_insights(
        self,
        query: str,
        max_insights: int = 2,
    ) -> List[str]:
        """Main entry point. Returns list of insight text strings.

        Results are cached per session — the LLM call only fires on the first
        invocation.  Subsequent calls return the cached insight texts.
        """
        # Return cache if already generated this session
        if self._session_insights is not None:
            return [i.insight_text for i in self._session_insights]

        try:
            from config.app_config import (
                PROACTIVE_SURFACING_ENABLED,
                PROACTIVE_SURFACING_MIN_GRAPH_NODES,
                PROACTIVE_SURFACING_MIN_GRAPH_EDGES,
            )
            if not PROACTIVE_SURFACING_ENABLED:
                self._session_insights = []
                return []
        except ImportError:
            self._session_insights = []
            return []

        # Graph sparsity check
        if not self._graph_memory:
            self._session_insights = []
            return []

        node_count = self._graph_memory.node_count()
        edge_count = self._graph_memory.edge_count()
        if node_count < PROACTIVE_SURFACING_MIN_GRAPH_NODES or edge_count < PROACTIVE_SURFACING_MIN_GRAPH_EDGES:
            logger.debug(
                f"[ContextSurfacer] Graph too sparse ({node_count} nodes, {edge_count} edges), skipping"
            )
            self._session_insights = []
            return []

        try:
            # Build domain clusters
            clusters = self._classify_user_edges()
            if len(clusters) < 2:
                logger.debug("[ContextSurfacer] Fewer than 2 domains, no bridges possible")
                self._session_insights = []
                return []

            # Identify active domains from query
            active_domains = self._identify_active_domains(query, clusters)
            if not active_domains:
                logger.debug("[ContextSurfacer] No active domains detected in query")
                self._session_insights = []
                return []

            # Select bridge candidates
            candidates = self._select_bridge_candidates(active_domains, clusters, max_candidates=3)
            if not candidates:
                logger.debug("[ContextSurfacer] No bridge candidates after filtering")
                self._session_insights = []
                return []

            # Synthesize via LLM
            insights = await self._synthesize_insights_batch(candidates, max_insights)

            # Record in history
            history = self._get_history()
            for insight in insights:
                history.record_surfaced(insight.novelty_key())

            self._session_insights = insights
            logger.info(f"[ContextSurfacer] Generated {len(insights)} proactive insights")
            return [i.insight_text for i in insights]

        except Exception as e:
            logger.warning(f"[ContextSurfacer] Insight generation failed: {e}")
            self._session_insights = []
            return []

    def _classify_user_edges(self) -> Dict[str, "DomainCluster"]:
        """Walk all edges from 'user' and classify targets by domain.

        Uses a dirty-check cache: rebuilds only if edge_count changed.
        """
        current_edge_count = self._graph_memory.edge_count()
        if self._domain_cache is not None and self._domain_cache_edge_count == current_edge_count:
            return self._domain_cache

        from memory.surfacing_models import DomainCluster, DomainEntity
        from memory.user_profile_schema import categorize_relation

        edges = self._graph_memory.get_relations("user", direction="out")
        clusters: Dict[str, DomainCluster] = {}

        for edge in edges:
            target_id = edge.target_id
            if target_id == "user":
                continue

            relation = edge.relation
            # Override check first
            domain = _SURFACING_RELATION_OVERRIDES.get(relation)
            if domain is None:
                # Fall back to schema's categorize_relation
                category = categorize_relation(relation)
                domain = category.value

            # Get display name
            node = self._graph_memory.get_entity(target_id)
            display_name = node.display_name if node else target_id

            entity = DomainEntity(
                entity_id=target_id,
                display_name=display_name,
                domain=domain,
                relation=relation,
                edge_weight=edge.weight,
            )

            if domain not in clusters:
                clusters[domain] = DomainCluster(domain=domain)
            clusters[domain].entities.append(entity)

        # Enrich clusters with fact sentences
        for domain, cluster in clusters.items():
            sentences = []
            for ent in cluster.entities[:5]:  # Cap per-entity sentences to keep prompt small
                ent_sentences = self._graph_memory.get_context_sentences(
                    ent.entity_id, depth=1, max_sentences=3
                )
                sentences.extend(ent_sentences)
            cluster.fact_sentences = sentences[:10]  # Cap total per cluster

        self._domain_cache = clusters
        self._domain_cache_edge_count = current_edge_count
        logger.debug(
            f"[ContextSurfacer] Classified {sum(len(c.entities) for c in clusters.values())} "
            f"entities into {len(clusters)} domains"
        )
        return clusters

    def _identify_active_domains(
        self,
        query: str,
        clusters: Dict[str, "DomainCluster"],
    ) -> Set[str]:
        """Detect which domains the current query touches.

        Strategy:
        1. Extract entities from query via alias resolution → look up their domain
        2. Fallback: keyword heuristics for domain terms
        """
        active: Set[str] = set()

        # Strategy 1: entity resolution
        if self._entity_resolver:
            from memory.graph_utils import extract_graph_entities
            entity_ids = extract_graph_entities(query, self._entity_resolver)
            # Map entity IDs to their domain
            for domain, cluster in clusters.items():
                cluster_eids = {e.entity_id for e in cluster.entities}
                if entity_ids & cluster_eids:
                    active.add(domain)

        # Strategy 2: keyword fallback
        query_lower = query.lower()
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if domain in clusters:
                for kw in keywords:
                    if kw in query_lower:
                        active.add(domain)
                        break

        return active

    def _select_bridge_candidates(
        self,
        active_domains: Set[str],
        clusters: Dict[str, "DomainCluster"],
        max_candidates: int = 3,
    ) -> List["CrossDomainCandidate"]:
        """Score and select bridge candidates from non-active domains.

        Scoring: entity_count(0.3) + avg_weight(0.3) + lateral_edges(0.4)
        Filters out recently shown insights via SurfacingHistory.
        """
        from config.app_config import PROACTIVE_SURFACING_COOLDOWN_HOURS
        from memory.surfacing_models import CrossDomainCandidate

        history = self._get_history()
        candidates = []

        for active_domain in active_domains:
            active_cluster = clusters.get(active_domain)
            if not active_cluster:
                continue

            for bridged_domain, bridged_cluster in clusters.items():
                if bridged_domain in active_domains:
                    continue
                if not bridged_cluster.entities:
                    continue

                # Check novelty — build a rough novelty key for this pair
                pair_eids = sorted(
                    [e.entity_id for e in active_cluster.entities[:3]]
                    + [e.entity_id for e in bridged_cluster.entities[:3]]
                )
                pair_domains = sorted([active_domain, bridged_domain])
                rough_key = "|".join(pair_domains) + "|" + ",".join(pair_eids)
                if history.was_recently_shown(rough_key, PROACTIVE_SURFACING_COOLDOWN_HOURS):
                    continue

                # Score this bridge
                entity_count_score = min(len(bridged_cluster.entities) / 5.0, 1.0)
                avg_weight = (
                    sum(e.edge_weight for e in bridged_cluster.entities)
                    / len(bridged_cluster.entities)
                    if bridged_cluster.entities
                    else 0.0
                )
                weight_score = min(avg_weight / 3.0, 1.0)

                # Count lateral edges (edges between entities, not through "user")
                lateral_count = 0
                for ent in bridged_cluster.entities[:5]:
                    rels = self._graph_memory.get_relations(ent.entity_id, direction="both")
                    for r in rels:
                        other = r.target_id if r.source_id == ent.entity_id else r.source_id
                        if other != "user":
                            lateral_count += 1
                lateral_score = min(lateral_count / 10.0, 1.0)

                relevance = 0.3 * entity_count_score + 0.3 * weight_score + 0.4 * lateral_score

                candidates.append(CrossDomainCandidate(
                    active_domain=active_domain,
                    bridged_domain=bridged_domain,
                    active_cluster=active_cluster,
                    bridged_cluster=bridged_cluster,
                    relevance_score=relevance,
                ))

        # Sort by relevance, take top N
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:max_candidates]

    async def _synthesize_insights_batch(
        self,
        candidates: List["CrossDomainCandidate"],
        max_insights: int = 2,
    ) -> List["ProactiveInsight"]:
        """Single LLM call to synthesize insights from all candidates.

        Returns parsed ProactiveInsight objects. Gracefully handles malformed
        JSON or null entries in the response.
        """
        from memory.surfacing_models import ProactiveInsight

        if not candidates or not self._model_manager:
            return []

        # Build the prompt
        candidate_blocks = []
        for i, cand in enumerate(candidates, 1):
            active_facts = "\n".join(
                f"  - {s}" for s in (cand.active_cluster.fact_sentences or ["(no facts)"])[:5]
            )
            bridged_facts = "\n".join(
                f"  - {s}" for s in (cand.bridged_cluster.fact_sentences or ["(no facts)"])[:5]
            )
            candidate_blocks.append(
                f"CANDIDATE {i}: [{cand.active_domain.upper()}] <-> [{cand.bridged_domain.upper()}]\n"
                f"{cand.active_domain.upper()} facts:\n{active_facts}\n"
                f"{cand.bridged_domain.upper()} facts:\n{bridged_facts}"
            )

        candidates_text = "\n\n".join(candidate_blocks)

        prompt = (
            "Given facts from different areas of the user's life, identify meaningful connections "
            "that the user might not have noticed. Only surface genuinely useful insights.\n\n"
            f"{candidates_text}\n\n"
            "Output a JSON array with one entry per candidate. Each entry is either an object "
            "or null (if no meaningful connection exists):\n"
            '[{{"insight": "one sentence connection", "confidence": 0.0-1.0, '
            '"entity_ids": ["id1","id2"]}}, null, ...]\n'
            "Rules:\n"
            "- Only connections that are practically useful, not obvious\n"
            "- Keep each insight to one sentence\n"
            "- confidence 0.0-1.0 based on how meaningful the connection is\n"
            "- Output ONLY the JSON array, no other text"
        )

        system_prompt = (
            "You are a perceptive assistant that finds non-obvious connections "
            "across different areas of someone's life. Be concise and practical."
        )

        try:
            from config.app_config import PROACTIVE_SURFACING_MODEL
            model_name = PROACTIVE_SURFACING_MODEL or None

            response = await self._model_manager.generate_once(
                prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=400,
                temperature=0.3,
            )

            if not response:
                return []

            insights = self._parse_llm_response(response, candidates, max_insights)
            return insights

        except Exception as e:
            logger.warning(f"[ContextSurfacer] LLM synthesis failed: {e}")
            return []

    def _parse_llm_response(
        self,
        response: str,
        candidates: List["CrossDomainCandidate"],
        max_insights: int,
    ) -> List["ProactiveInsight"]:
        """Parse LLM JSON array response into ProactiveInsight objects."""
        from memory.surfacing_models import ProactiveInsight

        # Extract JSON array from response (handle potential text wrapping)
        text = response.strip()
        # Try to find JSON array in response
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning("[ContextSurfacer] No JSON array found in LLM response")
            return []

        json_text = text[start:end + 1]
        try:
            items = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"[ContextSurfacer] JSON parse failed: {e}")
            return []

        if not isinstance(items, list):
            return []

        insights = []
        now = datetime.now()

        for i, item in enumerate(items):
            if item is None or i >= len(candidates):
                continue
            if not isinstance(item, dict):
                continue

            insight_text = item.get("insight", "").strip()
            if not insight_text:
                continue

            confidence = float(item.get("confidence", 0.5))
            entity_ids = item.get("entity_ids", [])
            if not isinstance(entity_ids, list):
                entity_ids = []

            cand = candidates[i]
            insights.append(ProactiveInsight(
                insight_text=insight_text,
                active_domain=cand.active_domain,
                bridged_domain=cand.bridged_domain,
                entity_ids=entity_ids,
                confidence=confidence,
                generated_at=now,
            ))

            if len(insights) >= max_insights:
                break

        return insights
