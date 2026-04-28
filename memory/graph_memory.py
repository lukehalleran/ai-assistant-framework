# memory/graph_memory.py
"""
Persistent knowledge graph wrapping NetworkX.

Provides CRUD operations, alias resolution, BFS traversal, and
JSON serialization.  Uses a DiGraph (directed, single edge per
source+relation+target pair) with weight strengthening on repeated
mentions.

Persistence uses a dirty-flag so saves only happen when the graph
has actually changed (not on every insertion).  Uses orjson for
fast serialization (falls back to stdlib json if unavailable).

Stats helpers: count_by_source() for provenance counts,
count_bridge_edges() for cross-provenance edge counts.
"""

import os
from collections import deque
from datetime import datetime
from typing import Optional

try:
    import orjson

    def _json_load(f):
        return orjson.loads(f.read())

    def _json_dump(payload, f):
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
except ImportError:
    import json

    def _json_load(f):
        return json.load(f)

    def _json_dump(payload, f):
        json.dump(payload, f, indent=2, ensure_ascii=False)

import networkx as nx

from memory.graph_models import GraphEdge, GraphNode
from utils.logging_utils import get_logger

logger = get_logger("graph_memory")

# Default persist path (overridden by config)
_DEFAULT_PERSIST_PATH = os.path.join("data", "knowledge_graph.json")


class GraphMemory:
    """Persistent knowledge graph wrapping NetworkX DiGraph."""

    def __init__(self, persist_path: str = _DEFAULT_PERSIST_PATH):
        self.graph = nx.DiGraph()
        self.persist_path = persist_path
        # alias -> canonical entity_id  (lowered alias -> lowered entity_id)
        self._alias_index: dict[str, str] = {}
        # edge_key -> GraphEdge  (for fast duplicate detection)
        self._edge_index: dict[str, GraphEdge] = {}
        self._dirty = False
        self._modification_count = 0
        self._bulk_mode = False
        # Auto-save threshold: save after this many modifications
        self._auto_save_threshold = 50
        self.load()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_entity(self, node: GraphNode) -> str:
        """Add or update an entity node.  Returns the entity_id."""
        eid = node.entity_id.lower().strip()
        now = datetime.now()

        if self.graph.has_node(eid):
            # Update existing node
            existing = self.graph.nodes[eid]
            existing["mention_count"] = existing.get("mention_count", 0) + 1
            existing["last_seen"] = now.isoformat()
            # Merge new aliases
            cur_aliases = set(existing.get("aliases", []))
            for alias in node.aliases:
                a_lower = alias.lower().strip()
                if a_lower and a_lower != eid:
                    cur_aliases.add(a_lower)
                    self._alias_index[a_lower] = eid
            existing["aliases"] = list(cur_aliases)
            # Merge metadata
            if node.metadata:
                existing_meta = existing.get("metadata", {})
                existing_meta.update(node.metadata)
                existing["metadata"] = existing_meta
        else:
            # New node
            aliases_lower = []
            for a in node.aliases:
                a_lower = a.lower().strip()
                if a_lower and a_lower != eid:
                    aliases_lower.append(a_lower)
                    self._alias_index[a_lower] = eid
            self._alias_index[eid] = eid  # self-reference
            self.graph.add_node(eid, **{
                "display_name": node.display_name,
                "entity_type": node.entity_type,
                "aliases": aliases_lower,
                "first_seen": (node.first_seen or now).isoformat(),
                "last_seen": (node.last_seen or now).isoformat(),
                "mention_count": max(node.mention_count, 1),
                "metadata": node.metadata,
            })

        self._mark_dirty()
        return eid

    def add_relation(self, edge: GraphEdge, fact_id: str = "") -> None:
        """Add or strengthen an edge.  Duplicate (src, rel, tgt) increments weight."""
        src = edge.source_id.lower().strip()
        tgt = edge.target_id.lower().strip()
        rel = edge.relation.lower().strip()
        now = datetime.now()

        # Ensure both nodes exist (create minimal stubs if not)
        if not self.graph.has_node(src):
            self.add_entity(GraphNode(entity_id=src, display_name=src))
        if not self.graph.has_node(tgt):
            self.add_entity(GraphNode(entity_id=tgt, display_name=tgt))

        ekey = f"{src}|{rel}|{tgt}"

        if ekey in self._edge_index:
            # Strengthen existing edge
            existing = self._edge_index[ekey]
            existing.weight += 1.0
            existing.last_seen = now
            if fact_id and fact_id not in existing.source_fact_ids:
                existing.source_fact_ids.append(fact_id)
            # Update NetworkX edge data
            self.graph[src][tgt]["weight"] = existing.weight
            self.graph[src][tgt]["last_seen"] = now.isoformat()
            self.graph[src][tgt]["source_fact_ids"] = existing.source_fact_ids
        else:
            # New edge
            edge_copy = GraphEdge(
                source_id=src,
                relation=rel,
                target_id=tgt,
                weight=edge.weight,
                truth_score=edge.truth_score,
                first_seen=edge.first_seen or now,
                last_seen=edge.last_seen or now,
                source_fact_ids=edge.source_fact_ids[:],
                metadata=edge.metadata.copy(),
            )
            if fact_id and fact_id not in edge_copy.source_fact_ids:
                edge_copy.source_fact_ids.append(fact_id)
            self._edge_index[ekey] = edge_copy
            self.graph.add_edge(src, tgt, **{
                "relation": rel,
                "weight": edge_copy.weight,
                "truth_score": edge_copy.truth_score,
                "first_seen": edge_copy.first_seen.isoformat() if edge_copy.first_seen else now.isoformat(),
                "last_seen": edge_copy.last_seen.isoformat() if edge_copy.last_seen else now.isoformat(),
                "source_fact_ids": edge_copy.source_fact_ids,
                "metadata": edge_copy.metadata,
            })

        # Touch node timestamps
        for nid in (src, tgt):
            if self.graph.has_node(nid):
                self.graph.nodes[nid]["last_seen"] = now.isoformat()

        self._mark_dirty()

    def get_entity(self, entity_id: str) -> Optional[GraphNode]:
        """Look up an entity by its canonical ID."""
        eid = entity_id.lower().strip()
        if not self.graph.has_node(eid):
            return None
        data = dict(self.graph.nodes[eid])
        return GraphNode.from_dict(eid, data)

    def get_relations(self, entity_id: str, direction: str = "both") -> list[GraphEdge]:
        """Get all edges involving an entity.

        direction: 'out' (entity is source), 'in' (entity is target), 'both'.
        """
        eid = entity_id.lower().strip()
        if not self.graph.has_node(eid):
            return []

        edges = []
        if direction in ("out", "both"):
            for _, tgt, data in self.graph.out_edges(eid, data=True):
                ekey = f"{eid}|{data.get('relation', '')}|{tgt}"
                if ekey in self._edge_index:
                    edges.append(self._edge_index[ekey])
                else:
                    edges.append(GraphEdge.from_dict({**data, "source_id": eid, "target_id": tgt}))
        if direction in ("in", "both"):
            for src, _, data in self.graph.in_edges(eid, data=True):
                ekey = f"{src}|{data.get('relation', '')}|{eid}"
                if ekey in self._edge_index:
                    edges.append(self._edge_index[ekey])
                else:
                    edges.append(GraphEdge.from_dict({**data, "source_id": src, "target_id": eid}))
        return edges

    # ------------------------------------------------------------------
    # Alias Resolution
    # ------------------------------------------------------------------

    def resolve_entity(self, mention: str) -> Optional[str]:
        """Resolve a mention to a canonical entity_id via alias index.

        Returns None if no match found.
        """
        key = mention.lower().strip()
        return self._alias_index.get(key)

    def register_alias(self, alias: str, entity_id: str) -> None:
        """Register an alias pointing to an existing entity."""
        a_lower = alias.lower().strip()
        eid = entity_id.lower().strip()
        if not a_lower or not eid:
            return
        self._alias_index[a_lower] = eid
        # Also store on the node
        if self.graph.has_node(eid):
            cur = set(self.graph.nodes[eid].get("aliases", []))
            cur.add(a_lower)
            self.graph.nodes[eid]["aliases"] = list(cur)
            self._mark_dirty()

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def neighbors(self, entity_id: str, depth: int = 1) -> dict[str, list[GraphEdge]]:
        """BFS traversal to depth N.  Returns {entity_id: [edges_from_that_entity]}."""
        eid = entity_id.lower().strip()
        if not self.graph.has_node(eid):
            return {}

        visited: set[str] = set()
        result: dict[str, list[GraphEdge]] = {}
        queue: deque[tuple[str, int]] = deque([(eid, 0)])

        while queue:
            current, d = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            edges = self.get_relations(current, direction="both")
            if edges:
                result[current] = edges

            if d < depth:
                # Enqueue neighbors
                for e in edges:
                    next_id = e.target_id if e.source_id == current else e.source_id
                    if next_id not in visited:
                        queue.append((next_id, d + 1))

        return result

    def subgraph_around(self, entity_id: str, depth: int = 2) -> list[GraphEdge]:
        """Return all edges within depth hops of an entity (flat list)."""
        neighborhood = self.neighbors(entity_id, depth=depth)
        seen_keys: set[str] = set()
        edges: list[GraphEdge] = []
        for edge_list in neighborhood.values():
            for e in edge_list:
                k = e.edge_key()
                if k not in seen_keys:
                    seen_keys.add(k)
                    edges.append(e)
        return edges

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Shortest path between two entities (list of entity IDs), or [] if no path."""
        src = source.lower().strip()
        tgt = target.lower().strip()
        try:
            return nx.shortest_path(self.graph, src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------
    # Natural Language Context (for prompt injection)
    # ------------------------------------------------------------------

    def get_context_sentences(self, entity_id: str, depth: int = 2, max_sentences: int = 15, with_attribution: bool = False) -> list[str]:
        """Return natural language sentences about an entity's neighborhood.

        Sorted by edge weight (strongest relationships first).

        Args:
            entity_id: Entity to get context for
            depth: Graph traversal depth
            max_sentences: Maximum sentences to return
            with_attribution: If True, append derivation markers to sentences
        """
        edges = self.subgraph_around(entity_id, depth=depth)
        # Sort by weight descending
        edges.sort(key=lambda e: e.weight, reverse=True)

        sentences = []
        for e in edges[:max_sentences]:
            src_node = self.get_entity(e.source_id)
            tgt_node = self.get_entity(e.target_id)
            src_name = src_node.display_name if src_node else e.source_id
            tgt_name = tgt_node.display_name if tgt_node else e.target_id
            sentences.append(e.to_natural_language(src_name, tgt_name, with_attribution=with_attribution))
        return sentences

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save graph to JSON.  Only writes if dirty."""
        if not self._dirty:
            return

        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)

        # Serialize nodes
        nodes = {}
        for nid, data in self.graph.nodes(data=True):
            nodes[nid] = {
                "display_name": data.get("display_name", nid),
                "entity_type": data.get("entity_type", "other"),
                "aliases": data.get("aliases", []),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
                "mention_count": data.get("mention_count", 0),
                "metadata": data.get("metadata", {}),
            }

        # Serialize edges from index (authoritative)
        edges = [e.to_dict() for e in self._edge_index.values()]

        payload = {"nodes": nodes, "edges": edges}

        try:
            # Atomic write: write to temp file, then rename.
            # Prevents data loss if the process is killed mid-write.
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                _json_dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.persist_path)
            self._dirty = False
            self._modification_count = 0
            logger.info(f"[GraphMemory] Saved {len(nodes)} nodes, {len(edges)} edges to {self.persist_path}")
        except Exception as e:
            logger.error(f"[GraphMemory] Save failed: {e}")

    def load(self) -> None:
        """Load graph from JSON file.  No-op if file doesn't exist."""
        if not os.path.exists(self.persist_path):
            logger.info(f"[GraphMemory] No graph file at {self.persist_path}, starting fresh")
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                payload = _json_load(f)
        except (ValueError, OSError) as e:
            logger.error(f"[GraphMemory] Failed to load {self.persist_path}: {e}")
            return

        # Load nodes
        for nid, data in payload.get("nodes", {}).items():
            self.graph.add_node(nid, **data)
            # Rebuild alias index
            self._alias_index[nid] = nid
            for alias in data.get("aliases", []):
                self._alias_index[alias.lower().strip()] = nid

        # Load edges
        for edata in payload.get("edges", []):
            edge = GraphEdge.from_dict(edata)
            src = edge.source_id
            tgt = edge.target_id
            ekey = edge.edge_key()
            self._edge_index[ekey] = edge
            self.graph.add_edge(src, tgt, **{
                "relation": edge.relation,
                "weight": edge.weight,
                "truth_score": edge.truth_score,
                "first_seen": edata.get("first_seen"),
                "last_seen": edata.get("last_seen"),
                "source_fact_ids": edge.source_fact_ids,
                "metadata": edge.metadata,
            })

        self._dirty = False
        self._modification_count = 0
        logger.info(
            f"[GraphMemory] Loaded {self.node_count()} nodes, "
            f"{self.edge_count()} edges from {self.persist_path}"
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return len(self._edge_index)

    def count_by_source(self, source: str) -> int:
        """Count nodes with a given metadata source value (e.g. 'wikidata', 'wiki_retrieved')."""
        count = 0
        for _nid, data in self.graph.nodes(data=True):
            if data.get("metadata", {}).get("source") == source:
                count += 1
        return count

    def count_bridge_edges(self) -> int:
        """Count edges where source and target nodes have different provenance."""
        count = 0
        for edge in self._edge_index.values():
            src_data = self.graph.nodes.get(edge.source_id, {})
            tgt_data = self.graph.nodes.get(edge.target_id, {})
            src_source = src_data.get("metadata", {}).get("source", "personal")
            tgt_source = tgt_data.get("metadata", {}).get("source", "personal")
            if src_source != tgt_source:
                count += 1
        return count

    def prune_garbage_bridges(self, dry_run: bool = True) -> dict:
        """Remove low-quality bridge edges between personal and wikidata nodes.

        Targets five categories of noise:
        1. mentioned_alongside edges from wiki_retrieved enrichment (session noise)
        2. wikidata_bridge edges with bridge_confidence < 0.80 (low-sim embedding matches)
        3. Fact-ingestion edges where personal relations (pets, dad, etc.) wrongly
           target wikidata entities (entity resolution collisions)
        4. Edges where either endpoint is a short common word that collided with
           a wikidata acronym (cats→CATS, ice→ICE)
        5. Unstructured fact predicates crossing to wikidata (verb-phrase relations
           like upcoming_project, believes_X, terminal_shows, etc.)

        Returns dict with counts of edges removed per category.
        """
        # Personal-fact relations that should never cross to wikidata
        _PERSONAL_ONLY_RELATIONS = frozenset({
            "pets", "pet", "dad", "mom", "parent", "brother", "sister",
            "likes_man", "fixed", "ask_about_feelings", "expresses_emotion",
            "feels_productive", "feels_better", "got_up_earlier",
            "perception_of_snap_cuts", "user_connection",
        })
        # Known structural wikidata relations (these are legitimate cross-provenance)
        _STRUCTURAL_RELATIONS = frozenset({
            "has_part", "part_of", "main_subject", "instance_of", "subclass_of",
            "related_to", "located_in", "country", "field_of_work",
        })

        to_remove = []
        categories = {
            "mentioned_alongside": 0, "low_confidence_bridge": 0,
            "misrouted_personal": 0, "short_name_collision": 0,
            "unstructured_predicate": 0,
        }

        for ekey, edge in list(self._edge_index.items()):
            src_data = self.graph.nodes.get(edge.source_id, {})
            tgt_data = self.graph.nodes.get(edge.target_id, {})
            src_source = src_data.get("metadata", {}).get("source", "personal")
            tgt_source = tgt_data.get("metadata", {}).get("source", "personal")

            # Only look at cross-provenance edges
            if src_source == tgt_source:
                continue
            is_bridge = (src_source == "wikidata") != (tgt_source == "wikidata")
            if not is_bridge:
                continue

            reason = None

            # Category 1: mentioned_alongside noise
            if edge.relation == "mentioned_alongside":
                reason = "mentioned_alongside"

            # Category 2: low-confidence embedding bridges
            if not reason:
                emeta = edge.metadata or {}
                if emeta.get("source") == "wikidata_bridge":
                    conf = emeta.get("bridge_confidence", 1.0)
                    if conf < 0.80:
                        reason = "low_confidence_bridge"

            # Category 3: personal relations that crossed to wikidata
            if not reason:
                if edge.relation in _PERSONAL_ONLY_RELATIONS:
                    reason = "misrouted_personal"

            # Category 4: short/common entity name collisions (acronym false positives)
            if not reason:
                personal_id = edge.source_id if src_source != "wikidata" else edge.target_id
                _ACRONYM_COLLISIONS = frozenset({
                    "cats", "cat", "ice", "man", "today", "crazy", "think",
                    "thought", "six", "one", "part", "area", "time", "track",
                })
                if len(personal_id) < 4 or personal_id in _ACRONYM_COLLISIONS:
                    reason = "short_name_collision"

            # Category 5: unstructured fact predicates crossing to wikidata
            if not reason:
                emeta = edge.metadata or {}
                has_bridge_meta = emeta.get("source") in ("wikidata_bridge", "wiki_retrieved")
                if not has_bridge_meta and edge.relation not in _STRUCTURAL_RELATIONS:
                    reason = "unstructured_predicate"

            if reason:
                to_remove.append((ekey, edge, reason))
                categories[reason] = categories.get(reason, 0) + 1

        if dry_run:
            logger.info(
                f"[GraphMemory] Bridge cleanup DRY RUN: would remove "
                f"{len(to_remove)} edges — {categories}"
            )
            for ekey, edge, reason in to_remove[:10]:
                s_dn = self.graph.nodes.get(edge.source_id, {}).get("display_name", edge.source_id)
                t_dn = self.graph.nodes.get(edge.target_id, {}).get("display_name", edge.target_id)
                logger.info(f"  [{reason}] {s_dn} --[{edge.relation}]--> {t_dn}")
        else:
            for ekey, edge, reason in to_remove:
                del self._edge_index[ekey]
                if self.graph.has_edge(edge.source_id, edge.target_id):
                    self.graph.remove_edge(edge.source_id, edge.target_id)
            self._mark_dirty()
            logger.info(
                f"[GraphMemory] Bridge cleanup: removed {len(to_remove)} edges — {categories}"
            )

        return {"removed": len(to_remove), "categories": categories, "dry_run": dry_run}

    def most_connected(self, n: int = 10) -> list[tuple[str, int]]:
        """Top N entities by total degree (in + out edges)."""
        degrees = [(nid, self.graph.degree(nid)) for nid in self.graph.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:n]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._modification_count += 1
        if not self._bulk_mode and self._modification_count >= self._auto_save_threshold:
            self.save()

    def bulk_import(self):
        """Context manager to suppress auto-saves during large imports.

        Usage:
            with graph_memory.bulk_import():
                for entity in entities:
                    graph_memory.add_entity(entity)
            # save() called once on exit
        """
        return _BulkImportContext(self)


class _BulkImportContext:
    def __init__(self, graph_memory: GraphMemory):
        self._gm = graph_memory

    def __enter__(self):
        self._gm._bulk_mode = True
        return self._gm

    def __exit__(self, *exc):
        self._gm._bulk_mode = False
        if self._gm._dirty:
            self._gm.save()
