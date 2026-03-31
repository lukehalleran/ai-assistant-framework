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

        for title, text in tracked.items():
            if articles_added >= WIKI_ENRICHMENT_MAX_PER_SESSION:
                break

            # Skip if already in graph
            if self._title_exists_in_graph(title):
                skipped += 1
                continue

            # Skip if text is too short to be useful
            if len(text) < WIKI_ENRICHMENT_MIN_TEXT:
                skipped += 1
                continue

            # Create entity node
            entity_id = self._create_wiki_entity(title, text)

            # Link to existing entities
            edges = self._link_to_existing_entities(
                entity_id, text,
                relation=WIKI_ENRICHMENT_EDGE_RELATION,
                weight=WIKI_ENRICHMENT_EDGE_WEIGHT,
            )
            relations_added += edges
            articles_added += 1

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

    def _link_to_existing_entities(
        self, wiki_entity_id: str, text: str,
        relation: str = "mentioned_alongside",
        weight: float = 0.5,
    ) -> int:
        """Find existing graph entities mentioned in text and add edges.

        Uses extract_graph_entities() from graph_utils for alias-based
        entity extraction.  Each match creates a bidirectional
        "mentioned_alongside" edge at reduced weight.

        Returns count of edges added.
        """
        from memory.graph_utils import extract_graph_entities

        mentioned = extract_graph_entities(text, self.resolver)

        # Remove self-reference
        mentioned.discard(wiki_entity_id)
        # Don't link to "user" — not useful for synthesis walks
        mentioned.discard("user")

        edges_added = 0
        now = datetime.now()

        for entity_id in mentioned:
            # Only link if the target node actually exists
            if not self.graph.graph.has_node(entity_id):
                continue

            edge = GraphEdge(
                source_id=wiki_entity_id,
                relation=relation,
                target_id=entity_id,
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
