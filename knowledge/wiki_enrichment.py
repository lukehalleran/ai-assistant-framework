# knowledge/wiki_enrichment.py
"""
Shutdown enrichment: add Wikipedia articles encountered during a session
to the knowledge graph.

For each tracked wiki article not already in the graph:
  1. Create an entity node (source="wiki_retrieved")
  2. Find mentions of existing graph entities in the article text
  3. Add "mentioned_alongside" edges at reduced weight (0.5)

This grows the graph organically from usage without LLM calls.
"""

import re
from datetime import datetime

from memory.graph_models import GraphEdge, GraphNode
from utils.logging_utils import get_logger

logger = get_logger("wiki_enrichment")


def _slugify(title: str) -> str:
    """Convert a Wikipedia article title to a graph entity ID."""
    s = title.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


class WikiGraphEnricher:
    """Enriches the knowledge graph with Wikipedia articles accessed during a session."""

    def __init__(self, graph_memory, entity_resolver):
        self.graph = graph_memory
        self.resolver = entity_resolver

    async def enrich_from_session(self) -> dict:
        """Main entry point.  Called from shutdown processor.

        Returns dict with keys: articles_added, relations_added, skipped.
        """
        from config.app_config import (
            WIKI_ENRICHMENT_MAX_PER_SESSION,
            WIKI_ENRICHMENT_MIN_TEXT,
            WIKI_ENRICHMENT_EDGE_RELATION,
            WIKI_ENRICHMENT_EDGE_WEIGHT,
        )
        from knowledge.wiki_tracker import WikiArticleTracker

        tracker = WikiArticleTracker.get_instance()
        tracked = tracker.get_tracked()

        if not tracked:
            return {"articles_added": 0, "relations_added": 0, "skipped": 0}

        articles_added = 0
        relations_added = 0
        skipped = 0

        # Pre-compute the set of entities that exist from conversations
        # (non-wiki sources). Only these are worth linking.
        convo_entities = self._get_conversation_entities()
        logger.debug(f"[WikiEnrichment] {len(convo_entities)} conversation entities in graph")

        for title, text in tracked.items():
            if articles_added >= WIKI_ENRICHMENT_MAX_PER_SESSION:
                break

            # Skip if text is too short to be useful
            if len(text) < WIKI_ENRICHMENT_MIN_TEXT:
                skipped += 1
                continue

            # Gate 1: Only create a wiki node if the article title matches
            # an entity already in the graph from conversations. This prevents
            # orphan concept clusters that dilute personal knowledge.
            slug = _slugify(title)
            resolved = self.resolver.resolve(title)
            title_entity = resolved or slug
            title_in_convo = title_entity in convo_entities

            if title_in_convo and not self._title_exists_in_graph(title):
                # Article matches a conversation entity — create the node
                entity_id = self._create_wiki_entity(title, text)
                articles_added += 1
            elif self._title_exists_in_graph(title):
                entity_id = slug if self.graph.graph.has_node(slug) else resolved
                skipped += 1
            else:
                # Article doesn't match any conversation entity — skip node creation
                # but still look for cross-entity bridges in the text
                entity_id = None
                skipped += 1

            # Gate 2: Only link to entities that exist from conversations
            # (not other wiki nodes). Filter through junk detection.
            edges = self._link_conversation_entities(
                entity_id, text, convo_entities,
                relation=WIKI_ENRICHMENT_EDGE_RELATION,
                weight=WIKI_ENRICHMENT_EDGE_WEIGHT,
            )
            relations_added += edges

        # Clear tracker for next session
        tracker.clear()

        logger.info(
            f"[WikiEnrichment] Added {articles_added} articles, "
            f"{relations_added} edges, skipped {skipped}"
        )

        return {
            "articles_added": articles_added,
            "relations_added": relations_added,
            "skipped": skipped,
        }

    def _title_exists_in_graph(self, title: str) -> bool:
        """Check if a Wikipedia article title already has a graph node."""
        slug = _slugify(title)
        if self.graph.graph.has_node(slug):
            return True
        # Also check via alias resolution
        if self.resolver.resolve(title) is not None:
            return True
        return False

    def _create_wiki_entity(self, title: str, text: str) -> str:
        """Create a graph node for a Wikipedia article.  Returns entity_id."""
        slug = _slugify(title)
        now = datetime.now()
        node = GraphNode(
            entity_id=slug,
            display_name=title,
            entity_type="concept",
            aliases=[],
            first_seen=now,
            last_seen=now,
            mention_count=1,
            metadata={
                "source": "wiki_retrieved",
                "text_preview": text[:200],
            },
        )
        entity_id = self.graph.add_entity(node)
        # Register the title as an alias for lookup
        self.resolver.learn_alias(title.lower(), entity_id, context="wiki_enrichment")
        return entity_id

    def _get_conversation_entities(self) -> set:
        """Return entity IDs that exist in the graph from conversation sources (not wiki)."""
        convo = set()
        for edge in self.graph.graph.edges(data=True):
            src, tgt, data = edge
            source = data.get("metadata", {}).get("source", "unknown") if isinstance(data.get("metadata"), dict) else "unknown"
            if source != "wiki_enrichment":
                convo.add(src)
                convo.add(tgt)
        convo.discard("user")  # user is a mega-hub, not useful
        return convo

    def _link_conversation_entities(
        self, wiki_entity_id: str | None, text: str,
        convo_entities: set,
        relation: str = "mentioned_alongside",
        weight: float = 0.5,
    ) -> int:
        """Find conversation entities mentioned in wiki text and link them.

        Only creates edges between entities that already exist from conversations.
        If wiki_entity_id is provided and is a conversation entity, also links
        it to other mentioned conversation entities.

        Filters out junk entities (short, numeric, stopwords).
        Returns count of edges added.
        """
        from memory.graph_utils import extract_graph_entities

        mentioned = extract_graph_entities(text, self.resolver)

        # Filter to conversation entities only + junk filter
        valid = set()
        for eid in mentioned:
            if eid in convo_entities and len(eid) > 2 and not eid.isdigit():
                valid.add(eid)

        if wiki_entity_id:
            valid.discard(wiki_entity_id)

        # If we have a wiki entity that's in the conversation graph, link it
        # to mentioned conversation entities
        if wiki_entity_id and wiki_entity_id in convo_entities:
            return self._add_edges_from(wiki_entity_id, valid, relation, weight)

        # Otherwise, create bridge edges between pairs of conversation entities
        # found in this article (they co-occur in the wiki text)
        if len(valid) < 2:
            return 0

        edges_added = 0
        valid_list = sorted(valid)
        for i, src in enumerate(valid_list):
            for tgt in valid_list[i + 1:]:
                # Only add if edge doesn't already exist
                if not self.graph.graph.has_edge(src, tgt):
                    edges_added += self._add_edges_from(src, {tgt}, relation, weight)
        return edges_added

    def _add_edges_from(self, source_id: str, targets: set,
                        relation: str, weight: float) -> int:
        """Add edges from source_id to each target. Returns count added."""
        edges_added = 0
        now = datetime.now()
        for target_id in targets:
            if not self.graph.graph.has_node(target_id):
                continue
            edge = GraphEdge(
                source_id=source_id,
                relation=relation,
                target_id=target_id,
                weight=weight,
                truth_score=0.8,
                first_seen=now,
                last_seen=now,
                source_fact_ids=[],
                metadata={"source": "wiki_enrichment"},
            )
            self.graph.add_relation(edge, fact_id="")
            edges_added += 1
        return edges_added

    def _link_to_existing_entities(
        self, wiki_entity_id: str, text: str,
        relation: str = "mentioned_alongside",
        weight: float = 0.5,
    ) -> int:
        """Legacy method — kept for backward compatibility. Use _link_conversation_entities instead."""
        convo = self._get_conversation_entities()
        return self._link_conversation_entities(wiki_entity_id, text, convo, relation, weight)
