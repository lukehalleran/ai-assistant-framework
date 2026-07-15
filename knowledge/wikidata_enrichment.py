# knowledge/wikidata_enrichment.py
"""
Anchored Wikidata typed-edge enrichment for the knowledge graph.

Contract
- Inputs: GraphMemory + EntityResolver, plus the offline Wikidata cache
  (data/wikidata_cache.json, built by scripts/extract_wikidata.py — 30K
  entities, ~7K typed relations across 10 domains).
- Behavior: for PERSONAL entities already in the graph (conversation-sourced,
  never wiki/wikidata nodes, never "user"), find an exact label/alias match in
  the cache, then add that entity's whitelisted typed relations as graph edges
  (instance_of, subclass_of, part_of, ...). Counterpart nodes are created on
  demand (entity_type="concept", source="wikidata") — 1 hop only, never
  expanded FROM. Matched personal nodes get `wikidata_qid` metadata, which
  also makes re-runs skip them (idempotent).
- Output: stats dict {matched, edges_added, nodes_created, skipped_existing}.
- Side effects: graph mutations only (nodes/edges/aliases); caller saves.

Deliberately NOT the mass import (scripts/import_wikidata_to_graph.py, built
for the retired graph-walk tier): that adds 30K unanchored nodes — the exact
orphan pollution the 2026-07-14 graph cleanup removed. This enricher is
anchored: every edge it adds touches an entity the user actually talked about,
and fan-out is bounded by the relation whitelist + per-entity/per-run caps.
"""

import json
import os
from datetime import datetime
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("wikidata_enrichment")

# Node metadata sources that mark a node as NOT personal (never enriched from)
_NON_PERSONAL_SOURCES = frozenset({
    "wikidata", "wikidata_enrichment", "wiki_retrieved", "wiki_enrichment",
})

# Taxonomic relations are only followed FORWARD (personal entity as source).
# Reverse taxonomic fan-in enumerates the MEMBERS of a category — a personal
# "beer" node would pull in every obscure beer brand in the cache as a new
# node (live preview: oesterstout, ambetanterik, caracu, ...). Compositional
# relations (part_of, has_part, uses, ...) stay bidirectional.
_FORWARD_ONLY_RELATIONS = frozenset({"instance_of", "subclass_of"})


def _slugify(text: str) -> str:
    import re
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


class WikidataGraphEnricher:
    """Adds whitelisted Wikidata typed edges to personal graph entities."""

    def __init__(self, graph_memory, entity_resolver,
                 cache_path: str = "", cache: Optional[dict] = None):
        self.graph = graph_memory
        self.resolver = entity_resolver
        self.cache_path = cache_path
        self._cache = cache  # injectable for tests; else lazy-loaded

    # ------------------------------------------------------------------

    def enrich(self) -> dict:
        """Main entry point. Returns stats dict; never raises."""
        from config.app_config import (
            WIKIDATA_ENRICHMENT_RELATION_WHITELIST,
            WIKIDATA_ENRICHMENT_MAX_EDGES_PER_ENTITY,
            WIKIDATA_ENRICHMENT_MAX_NEW_NODES,
            WIKIDATA_ENRICHMENT_MAX_EDGES_PER_RUN,
        )

        stats = {"matched": 0, "edges_added": 0, "nodes_created": 0,
                 "skipped_existing": 0}

        cache = self._load_cache()
        if not cache:
            return stats
        entities = cache.get("entities", {})
        relations = cache.get("relations", [])
        if not entities or not relations:
            return stats

        whitelist = set(WIKIDATA_ENRICHMENT_RELATION_WHITELIST)

        # label/alias (lowercase) -> qid
        label_index: dict[str, str] = {}
        for qid, ent in entities.items():
            label = (ent.get("label") or "").lower().strip()
            if label and label not in label_index:
                label_index[label] = qid
            for alias in ent.get("aliases", []) or []:
                a = alias.lower().strip()
                if a and a not in label_index:
                    label_index[a] = qid

        # qid -> whitelisted relations touching it (either direction)
        rels_by_qid: dict[str, list[dict]] = {}
        for r in relations:
            if r.get("relation_label") not in whitelist:
                continue
            rels_by_qid.setdefault(r.get("source_qid", ""), []).append(r)
            rels_by_qid.setdefault(r.get("target_qid", ""), []).append(r)

        now = datetime.now()
        edges_this_run = 0
        nodes_this_run = 0

        for node_id in list(self.graph.graph.nodes()):
            if edges_this_run >= WIKIDATA_ENRICHMENT_MAX_EDGES_PER_RUN:
                logger.info("[WikidataEnrichment] per-run edge cap reached")
                break

            data = self.graph.graph.nodes[node_id]
            if not self._is_personal(node_id, data):
                continue
            md = data.get("metadata") or {}
            if md.get("wikidata_qid"):
                stats["skipped_existing"] += 1
                continue

            qid = self._match_qid(node_id, data, label_index)
            if not qid:
                continue

            stats["matched"] += 1
            # Stamp the match on the personal node (also idempotency marker)
            self._stamp_qid(node_id, data, qid)

            added = 0
            for rel in rels_by_qid.get(qid, []):
                if added >= WIKIDATA_ENRICHMENT_MAX_EDGES_PER_ENTITY:
                    break
                if edges_this_run >= WIKIDATA_ENRICHMENT_MAX_EDGES_PER_RUN:
                    break
                is_forward = rel.get("source_qid") == qid
                if not is_forward and rel.get("relation_label") in _FORWARD_ONLY_RELATIONS:
                    continue
                other_qid = rel["target_qid"] if is_forward else rel["source_qid"]
                other_ent = entities.get(other_qid)
                if not other_ent or not other_ent.get("label"):
                    continue

                other_id, created = self._ensure_counterpart(
                    other_qid, other_ent, now,
                    nodes_this_run, WIKIDATA_ENRICHMENT_MAX_NEW_NODES,
                )
                if other_id is None:
                    continue  # node cap hit and counterpart doesn't exist yet
                if created:
                    nodes_this_run += 1
                    stats["nodes_created"] += 1

                src_id = node_id if is_forward else other_id
                tgt_id = other_id if is_forward else node_id
                self._add_edge(src_id, rel["relation_label"], tgt_id,
                               rel.get("property_id", ""), now)
                added += 1
                edges_this_run += 1
                stats["edges_added"] += 1

        logger.info(
            "[WikidataEnrichment] matched %d entities, added %d edges, "
            "created %d nodes, skipped %d already-enriched",
            stats["matched"], stats["edges_added"],
            stats["nodes_created"], stats["skipped_existing"],
        )
        return stats

    # ------------------------------------------------------------------

    def _load_cache(self) -> Optional[dict]:
        if self._cache is not None:
            return self._cache
        if not self.cache_path or not os.path.exists(self.cache_path):
            logger.debug("[WikidataEnrichment] no cache at %r", self.cache_path)
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            return self._cache
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[WikidataEnrichment] cache load failed: %s", e)
            return None

    @staticmethod
    def _is_personal(node_id: str, data: dict) -> bool:
        if node_id == "user":
            return False
        md = data.get("metadata") or {}
        return md.get("source") not in _NON_PERSONAL_SOURCES

    def _match_qid(self, node_id: str, data: dict,
                   label_index: dict[str, str]) -> Optional[str]:
        """Exact label/alias match only — no embeddings at shutdown."""
        candidates = [data.get("display_name", ""), node_id.replace("_", " ")]
        for alias in data.get("aliases", []) or []:
            candidates.append(alias)
        for cand in candidates:
            c = (cand or "").lower().strip()
            if c and c in label_index:
                return label_index[c]
        return None

    def _stamp_qid(self, node_id: str, data: dict, qid: str) -> None:
        from memory.graph_models import GraphNode
        self.graph.add_entity(GraphNode(
            entity_id=node_id,
            display_name=data.get("display_name", node_id),
            entity_type=data.get("entity_type", "other"),
            metadata={"wikidata_qid": qid},
        ))

    def _ensure_counterpart(self, qid: str, ent: dict, now,
                            nodes_this_run: int, max_new_nodes: int):
        """Resolve or create the wikidata-side node. Returns (entity_id|None,
        created: bool). None when the node cap blocks creation."""
        from memory.graph_models import GraphNode

        label = ent["label"]
        slug = _slugify(label)
        if self.graph.graph.has_node(slug):
            return slug, False
        resolved = self.resolver.resolve(label)
        if resolved:
            return resolved, False
        if nodes_this_run >= max_new_nodes:
            return None, False

        self.graph.add_entity(GraphNode(
            entity_id=slug,
            display_name=label,
            entity_type="concept",
            first_seen=now,
            last_seen=now,
            mention_count=0,
            metadata={
                "source": "wikidata_enrichment",
                "wikidata_qid": qid,
                "wikidata_description": (ent.get("description") or "")[:200],
                "domain_category": ent.get("domain_category", ""),
            },
        ))
        self.resolver.learn_alias(label.lower(), slug)
        return slug, True

    def _add_edge(self, src_id: str, relation: str, tgt_id: str,
                  property_id: str, now) -> None:
        from memory.graph_models import GraphEdge
        from memory.entity_resolver import normalize_relation
        self.graph.add_relation(GraphEdge(
            source_id=src_id,
            relation=normalize_relation(relation),
            target_id=tgt_id,
            weight=1.0,
            truth_score=0.95,
            first_seen=now,
            last_seen=now,
            metadata={"source": "wikidata_enrichment", "property_id": property_id},
        ))
