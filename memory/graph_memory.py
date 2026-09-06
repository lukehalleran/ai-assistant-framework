# memory/graph_memory.py
"""
Persistent knowledge graph wrapping NetworkX.

Provides CRUD operations, alias resolution, BFS traversal, and
JSON serialization, with weight strengthening on repeated mentions.

Storage model [2026-09-03]: the NetworkX DiGraph is a TOPOLOGY index
(degree / shortest_path / successors) holding ONE edge per (source, target)
pair; the edge attributes on it are advisory only (they mirror the FIRST
relation stored for the pair).  ``_edge_index`` (keyed ``src|rel|tgt``) plus
the ``_out_keys`` / ``_in_keys`` adjacency maps are the source of truth for
relations — every relation-level read (get_relations → neighbors →
subgraph_around → get_context_sentences) goes through them.  Before this,
reads rebuilt the key from the nx pair's single ``relation`` attribute, so a
second relation on the same pair was invisible: 146 of 982 live edges were
never returned (``user→<pet>`` carried 8 relations, 7 of them shadowed), and
strengthening one relation overwrote the nx attrs of another.  The on-disk
JSON always stored the full relation-keyed list, so no schema change.

Persistence uses a dirty-flag so saves only happen when the graph
has actually changed (not on every insertion).  Uses orjson for
fast serialization (falls back to stdlib json if unavailable).
Saves are atomic (temp + fsync + os.replace); loads are strict — an
existing-but-corrupt graph file is quarantined and raises
CorruptStoreError instead of silently starting empty [2026-07-14].

Stats helpers: count_by_source() for provenance counts,
count_bridge_edges() for cross-provenance edge counts.

Read-side staleness: get_context_sentences() (the prompt-injection path) drops
transient-state edges past their per-relation TTL via the shared
relation_classifier, so illness/recovery/mood/activity relationships age out of
graph context on the same horizon as the user profile and the ChromaDB `facts`
collection. The graph has no `is_current` flag; this read-time filter is what
keeps a once-stored "currently sick" edge from surfacing as present-tense
indefinitely. Edges are never deleted — they just stop being injected once stale
(a fresh re-mention refreshes last_seen and brings them back).

Stance metadata + settledness [2026-08-23]: add_relation merges incoming
stance/capture_tone into existing-edge metadata and tracks appraisal edges'
distinct-ISO-day observation lists (appraisal_days/appraisal_tones,
_track_appraisal_settledness); an appraisal restated on
>= APPRAISAL_SETTLED_MIN_DAYS (3) distinct NON-elevated days gains
metadata["settled"]=True. Elevated/unknown-tone days never count —
settledness deliberately under-fires (a crisis-week spiral must not mint a
"settled" view). Rendering (GraphEdge.to_natural_language) attributes
explicit appraisal/inferred edges instead of asserting them.
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
from memory.graph_utils import (
    _DEFAULT_HUB_DEGREE,
    _DEFAULT_MIN_MENTIONS,
    is_temporal_deictic,
    relation_species_conflict,
)

logger = get_logger("graph_memory")

# Code-level STRUCTURAL relations (co-occurrence bookkeeping written by the
# ingestion path, weight up to ~2.5 live) that must never render into prompt
# graph context nor route its traversal (2026-09-03): "Python mentioned
# alongside <name>" is not a fact about either entity.  Storage/expansion
# scoring untouched — read-time render/traversal filter only.
_NO_RENDER_RELATIONS = frozenset({"mentioned_alongside"})

# On-disk schema version for knowledge_graph.json. Files without the field
# are v1 (pre-versioning). Bump when the payload shape changes and add a
# migration in load(); loads REFUSE files from newer versions (StoreVersionError).
GRAPH_SCHEMA_VERSION = 1

# Default persist path (overridden by config)
_DEFAULT_PERSIST_PATH = os.path.join("data", "knowledge_graph.json")

# Bare generic nouns must never BIND as an alias of a specific entity — the
# auto-learned alias "project" on phase_change_heat_exchanger_project made
# every "group project" mention resolve to that node (its stored notes PHOTO
# was attached to an unrelated memory-ingest turn, 2026-08-28; a second junk
# binding 'meeting'→gòu existed in the same scan). Multi-word possessives
# ("my project") still bind. Enforced at every alias-index write site
# INCLUDING load-time rebuild, so historical junk aliases in node data are
# neutralized without a store repair. Kept in sync in spirit with the
# broader single-word extraction stoplist in graph_utils._COMMON_WORDS.
_GENERIC_ALIAS_WORDS = frozenset({
    "project", "projects", "note", "notes", "email", "emails",
    "meeting", "meetings", "appointment", "appointments",
    "calendar", "syllabus", "screenshot", "screenshots",
    "photo", "photos", "image", "images", "picture", "pictures",
    "document", "documents", "file", "files", "folder", "folders",
    "app", "system", "class", "course", "homework",
})


def _alias_bindable(alias: str, eid: str) -> bool:
    """A generic bare word may self-reference but never alias another entity."""
    return alias == eid or alias not in _GENERIC_ALIAS_WORDS


class GraphMemory:
    """Persistent knowledge graph wrapping NetworkX DiGraph."""

    def __init__(self, persist_path: str = _DEFAULT_PERSIST_PATH):
        self.graph = nx.DiGraph()
        self.persist_path = persist_path
        # alias -> canonical entity_id  (lowered alias -> lowered entity_id)
        self._alias_index: dict[str, str] = {}
        # edge_key -> GraphEdge  (authoritative relation store; save() writes it)
        self._edge_index: dict[str, GraphEdge] = {}
        # Relation-level adjacency (2026-09-03): node -> ordered set of edge
        # keys (dict-as-ordered-set — insertion order is the read order).
        # The nx DiGraph collapses multi-relation pairs; these maps do not.
        self._out_keys: dict[str, dict[str, None]] = {}
        self._in_keys: dict[str, dict[str, None]] = {}
        # Number of edge keys currently indexed in the adjacency maps.  A
        # mismatch with len(_edge_index) (a script mutated the index
        # directly) triggers rebuild_edge_indexes() on the next read.
        self._adj_count = 0
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
                if a_lower and a_lower != eid and _alias_bindable(a_lower, eid):
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
                if a_lower and a_lower != eid and _alias_bindable(a_lower, eid):
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
            # Merge incoming stance metadata (2026-08-23): a re-mention that
            # carries a stance tag updates the stored one (write-time
            # classification is deterministic, so repeats agree; a legacy
            # edge gains its tag on first re-mention post-backfill).
            if edge.metadata:
                for _k in ("stance", "capture_tone"):
                    if edge.metadata.get(_k):
                        existing.metadata[_k] = edge.metadata[_k]
            self._track_appraisal_settledness(existing, edge.metadata or {}, now)
            # Mirror onto the advisory nx attrs ONLY when the pair's nx edge
            # holds THIS relation — writing unconditionally cross-contaminated
            # a sibling relation's weight/metadata on multi-relation pairs.
            if (self.graph.has_edge(src, tgt)
                    and self.graph[src][tgt].get("relation") == rel):
                self.graph[src][tgt]["weight"] = existing.weight
                self.graph[src][tgt]["last_seen"] = now.isoformat()
                self.graph[src][tgt]["source_fact_ids"] = existing.source_fact_ids
                self.graph[src][tgt]["metadata"] = existing.metadata
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
            self._track_appraisal_settledness(edge_copy, edge.metadata or {}, now)
            self._edge_index[ekey] = edge_copy
            self._index_edge(edge_copy)
            # The FIRST relation stored for a pair owns the nx pair's advisory
            # attrs; a later relation on the same pair must not overwrite them
            # (the DiGraph keeps one edge per pair — topology only).
            if not self.graph.has_edge(src, tgt):
                self.graph.add_edge(src, tgt, **self._nx_edge_attrs(edge_copy))

        # Touch node timestamps
        for nid in (src, tgt):
            if self.graph.has_node(nid):
                self.graph.nodes[nid]["last_seen"] = now.isoformat()

        self._mark_dirty()

    # Settledness thresholds (2026-08-23 stance layer): an appraisal repeated
    # on >= this many DISTINCT days, each at non-elevated capture tone, is
    # marked settled — a stable stated view rather than a crisis-day one-off.
    # Elevated/unknown-tone days never count (conservative: settledness
    # deliberately under-fires). Mirrors learned_relations' distinct-days
    # promotion pattern.
    APPRAISAL_SETTLED_MIN_DAYS = 3
    _APPRAISAL_DAYS_CAP = 30  # bound the stored day lists

    def _track_appraisal_settledness(self, edge: GraphEdge, incoming_md: dict,
                                     now: datetime) -> None:
        """Record a distinct-ISO-date observation for an appraisal edge and
        promote to settled at >= APPRAISAL_SETTLED_MIN_DAYS non-elevated days."""
        try:
            stance = (incoming_md or {}).get("stance") or edge.metadata.get("stance")
            if stance != "appraisal":
                return
            day = now.strftime("%Y-%m-%d")
            tone = (incoming_md or {}).get("capture_tone") or "unknown"
            days = edge.metadata.setdefault("appraisal_days", [])
            tones = edge.metadata.setdefault("appraisal_tones", [])
            if day not in days:
                days.append(day)
                tones.append(tone)
                # keep lists bounded and aligned
                if len(days) > self._APPRAISAL_DAYS_CAP:
                    del days[0:len(days) - self._APPRAISAL_DAYS_CAP]
                    del tones[0:len(tones) - len(days)]
            elif tone == "non_elevated":
                # A later same-day non-elevated restatement upgrades that
                # day's tone (elevated→non_elevated never downgrades).
                idx = days.index(day)
                if idx < len(tones) and tones[idx] != "non_elevated":
                    tones[idx] = "non_elevated"
            non_elevated_days = sum(1 for t in tones if t == "non_elevated")
            if non_elevated_days >= self.APPRAISAL_SETTLED_MIN_DAYS:
                edge.metadata["settled"] = True
        except Exception:
            pass

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

        # Relation-level reads come from the authoritative index via the
        # adjacency maps (2026-09-03) — never from the nx pair's single
        # 'relation' attribute, which hid every second relation on a pair.
        if self._adj_count != len(self._edge_index):
            self.rebuild_edge_indexes()

        edges: list[GraphEdge] = []
        index = self._edge_index
        if direction in ("out", "both"):
            for key in self._out_keys.get(eid, ()):
                e = index.get(key)
                if e is not None:
                    edges.append(e)
        if direction in ("in", "both"):
            for key in self._in_keys.get(eid, ()):
                e = index.get(key)
                if e is not None:
                    edges.append(e)
        return edges

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity node and all its incident edges, keeping the alias
        index and the authoritative ``_edge_index`` consistent.

        Returns True if the node existed and was removed, False otherwise.

        NOTE: destructive and does no confirmation/backup — callers (e.g. the
        ``scripts/graph_junk_cleanup.py`` maintenance tool) are responsible for
        dry-run gating and backups.  ``save()`` must be called to persist.
        """
        eid = entity_id.lower().strip()
        if not self.graph.has_node(eid):
            return False
        # Drop incident edges from the authoritative edge index (save() writes
        # edges from this index, so stale entries would resurrect dangling edges).
        stale = [(k, e) for k, e in self._edge_index.items()
                 if e.source_id == eid or e.target_id == eid]
        for k, e in stale:
            del self._edge_index[k]
            self._unindex_edge(e, key=k)
        # The node's own adjacency rows must not linger (a dangling key would
        # make the far endpoint report an edge to a node that no longer exists).
        for maps in (self._out_keys, self._in_keys):
            leftover = maps.pop(eid, None)
            if leftover:
                self._adj_count -= len(leftover)
        # Drop alias entries pointing at this entity
        self._alias_index = {a: t for a, t in self._alias_index.items() if t != eid}
        # Remove the node (networkx drops its incident graph edges too)
        self.graph.remove_node(eid)
        self._mark_dirty()
        return True

    # ------------------------------------------------------------------
    # Relation-level adjacency index (2026-09-03)
    # ------------------------------------------------------------------

    @staticmethod
    def _nx_edge_attrs(edge: GraphEdge) -> dict:
        """Advisory attribute mirror written onto the nx pair edge."""
        return {
            "relation": edge.relation,
            "weight": edge.weight,
            "truth_score": edge.truth_score,
            "first_seen": edge.first_seen.isoformat() if edge.first_seen else None,
            "last_seen": edge.last_seen.isoformat() if edge.last_seen else None,
            "source_fact_ids": edge.source_fact_ids,
            "metadata": edge.metadata,
        }

    def _index_edge(self, edge: GraphEdge, key: Optional[str] = None) -> None:
        """Record ``edge`` in both adjacency maps (idempotent per key)."""
        key = key or edge.edge_key()
        out = self._out_keys.setdefault(edge.source_id, {})
        if key not in out:
            self._adj_count += 1
        out[key] = None
        self._in_keys.setdefault(edge.target_id, {})[key] = None

    def _unindex_edge(self, edge: GraphEdge, key: Optional[str] = None) -> None:
        """Forget ``edge`` in both adjacency maps (no-op if absent)."""
        key = key or edge.edge_key()
        out = self._out_keys.get(edge.source_id)
        if out is not None and key in out:
            del out[key]
            self._adj_count -= 1
            if not out:
                del self._out_keys[edge.source_id]
        inn = self._in_keys.get(edge.target_id)
        if inn is not None and key in inn:
            del inn[key]
            if not inn:
                del self._in_keys[edge.target_id]

    def rebuild_edge_indexes(self) -> None:
        """Rebuild the adjacency maps from ``_edge_index`` (insertion order).

        Public so maintenance scripts that mutate ``_edge_index`` directly
        (relation canonicalization, calibration purges) can re-sync before
        save().  Also restores any nx pair the topology index is missing —
        ``graph.add_edge`` is called only when the pair is absent, so existing
        advisory attrs are never overwritten.  Never marks the graph dirty.
        """
        self._out_keys = {}
        self._in_keys = {}
        for key, edge in self._edge_index.items():
            # The dict key is authoritative (scripts re-key entries in place).
            self._out_keys.setdefault(edge.source_id, {})[key] = None
            self._in_keys.setdefault(edge.target_id, {})[key] = None
            if not self.graph.has_edge(edge.source_id, edge.target_id):
                self.graph.add_edge(edge.source_id, edge.target_id,
                                    **self._nx_edge_attrs(edge))
        self._adj_count = len(self._edge_index)

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
        if not _alias_bindable(a_lower, eid):
            logger.debug(
                f"[GraphMemory] Refusing generic-word alias binding: "
                f"{a_lower!r} -> {eid!r}"
            )
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

    def neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        *,
        hub_barrier: bool = False,
        hub_degree: Optional[int] = None,
        skip_relations: frozenset = frozenset(),
    ) -> dict[str, list[GraphEdge]]:
        """BFS traversal to depth N.  Returns {entity_id: [edges_from_that_entity]}.

        Default behaviour is a plain BFS (graph_utils' related-name/expansion
        consumers depend on that reach).  ``hub_barrier`` (2026-09-03, prompt
        injection only) applies the expansion doctrine from
        ``graph_utils.rank_expansion_candidates``: the seed always expands, but
        a NON-seed node is never expanded THROUGH when it is the ``user`` star
        hub or has ``hub_degree`` (default GRAPH_EXPANSION_HUB_DEGREE) or more
        edges — its own edges are still collected (reached), its neighbours are
        not enqueued.  Before this, a pet seed at depth 2 fanned out through
        ``user`` (degree 770) into python/gym edges.  ``skip_relations`` are
        neither traversed nor returned (``mentioned_alongside`` is structural).
        """
        eid = entity_id.lower().strip()
        if not self.graph.has_node(eid):
            return {}
        limit = hub_degree if hub_degree is not None else _DEFAULT_HUB_DEGREE
        if hub_barrier:
            try:
                from config.app_config import GRAPH_EXPANSION_HUB_DEGREE  # lazy import: live-config read
                if hub_degree is None:
                    limit = int(GRAPH_EXPANSION_HUB_DEGREE)
            except Exception:
                pass

        visited: set[str] = set()
        result: dict[str, list[GraphEdge]] = {}
        queue: deque[tuple[str, int]] = deque([(eid, 0)])

        while queue:
            current, d = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            edges = self.get_relations(current, direction="both")
            if skip_relations:
                edges = [e for e in edges if e.relation not in skip_relations]
            if edges:
                result[current] = edges

            if d < depth:
                if hub_barrier and current != eid and (
                    current == "user" or len(edges) >= limit
                ):
                    continue  # reached, never expanded through
                # Enqueue neighbors
                for e in edges:
                    next_id = e.target_id if e.source_id == current else e.source_id
                    if next_id not in visited:
                        queue.append((next_id, d + 1))

        return result

    def subgraph_around(self, entity_id: str, depth: int = 2, **traversal_kwargs) -> list[GraphEdge]:
        """Return all edges within depth hops of an entity (flat list).
        ``traversal_kwargs`` pass through to ``neighbors`` (hub_barrier, …)."""
        neighborhood = self.neighbors(entity_id, depth=depth, **traversal_kwargs)
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

    @staticmethod
    def _edge_is_stale_transient(edge: GraphEdge, now: Optional[datetime] = None) -> bool:
        """True if `edge` names a transient state past its per-relation TTL.

        Mirrors the read-side TTL already applied to the user profile
        (``user_profile.get_category``) and the ChromaDB ``facts`` collection
        (``memory_retriever``) so a once-stored "currently sick" / "recovering
        from illness" relationship does not keep surfacing as present-tense
        indefinitely. The graph had no ``is_current`` flag and no TTL, so these
        edges were emitted at full weight every time the query resolved to the
        ``user`` node — the gap behind "the agent still thinks I'm sick".

        Uses the shared single source of truth (``relation_classifier``):
        illness/recovery/symptom relations age out on the health horizon
        (~days), mood/activity state on the standard ephemeral horizon (~24h),
        durable identity relations never age out. Age is measured from
        ``last_seen`` (refreshed on every re-mention), so a fresh illness
        mention surfaces again and only ages out once it stops being mentioned.
        Edges with no/unparseable timestamp are kept (can't judge age).
        """
        from memory.relation_classifier import ephemeral_ttl_hours
        ttl_hours = ephemeral_ttl_hours(edge.relation)
        if ttl_hours is None or ttl_hours <= 0:
            return False  # durable relation — never ages out
        ts = edge.last_seen or edge.first_seen
        if ts is None:
            return False
        try:
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            age_hours = ((now or datetime.now()) - ts).total_seconds() / 3600.0
        except (TypeError, AttributeError):
            return False
        return age_hours > ttl_hours

    def edge_is_suppressed(self, edge: GraphEdge) -> bool:
        """Read-side neutralizer for an edge that must never render (2026-09-03).

        Two deterministic cases, both "never wrong > always active":
          * ``metadata.curation_quarantined`` — the reversible flag the
            curation ladder prefers over deletion (``scripts/
            quarantine_graph_edges.py``); the edge stays on disk, never
            surfaces.
          * species conflict — a species-typed relation (``has_dog``) whose
            TARGET node carries curated ``species`` metadata naming a
            different animal. The ingestion guard (memory_storage, 2026-08-28)
            only protects NEW edges; ``user|has_dog|Biscuit`` and
            ``user|has_dog|Mochi`` (shutdown LLM, 2026-08-18) pre-dated it and
            rendered "User has dog Mochi" into every turn — the planner then
            planned "common in dogs" and the reply called the cat a dog.
        Nodes without species metadata are never blocked (under-fires).
          * temporal-fragment endpoint (2026-09-05) — an edge whose source or
            target node is a bare when-word (`user|dad|today`,
            `user|feels_better|today`, `auggie|texted|on_thursday`; the
            live `today` node carried 11 such in-edges) renders as nonsense
            ("User dad today") and pre-dates the ingestion junk guard.
            Read-time neutralize is the ladder's first instrument: the
            edges stay on disk for the owner's graph_junk_cleanup review.
        """
        try:
            md = getattr(edge, "metadata", None) or {}
            if md.get("curation_quarantined"):
                return True
            if is_temporal_deictic(str(edge.source_id or "")) or is_temporal_deictic(
                str(edge.target_id or "")
            ):
                return True
            # Node-level quarantine (2026-09-05, curation graph adapter): a
            # flagged node hides every edge touching it — reversible, on-disk.
            for nid in (edge.source_id, edge.target_id):
                node = self.get_entity(str(nid or ""))
                if node is not None and (getattr(node, "metadata", None) or {}).get("curation_quarantined"):
                    return True
            tgt = self.get_entity(edge.target_id)
            if tgt is not None and relation_species_conflict(
                edge.relation, getattr(tgt, "metadata", None) or {}
            ):
                return True
        except Exception:
            return False
        return False

    def get_context_sentences(self, entity_id: str, depth: int = 2, max_sentences: int = 15, with_attribution: bool = False) -> list[str]:
        """Return natural language sentences about an entity's neighborhood.

        Sorted by edge weight (strongest relationships first). Stale transient
        edges (illness/recovery/mood/activity past their per-relation TTL) are
        dropped first so the graph respects the same read-side staleness as the
        profile and the ``facts`` collection — see ``_edge_is_stale_transient``.

        Args:
            entity_id: Entity to get context for
            depth: Graph traversal depth
            max_sentences: Maximum sentences to return
            with_attribution: If True, append derivation markers to sentences
        """
        seed = entity_id.lower().strip()
        # A when-word is never an entity (2026-09-05): even if a caller hands
        # one in, don't walk from it into the user hub and render the hub's
        # unrelated edges as "context" for "today".
        if is_temporal_deictic(seed):
            return []
        edges = self.subgraph_around(
            entity_id, depth=depth, hub_barrier=True, skip_relations=_NO_RENDER_RELATIONS
        )
        # Drop transient-state edges past their TTL (illness/recovery/mood/...)
        now = datetime.now()
        edges = [
            e for e in edges
            if e.relation not in _NO_RENDER_RELATIONS
            and not self._edge_is_stale_transient(e, now)
            and not self.edge_is_suppressed(e)
        ]
        # Ordering (2026-09-03): edges incident to the seed first, by weight
        # (depth-1 behaviour unchanged); then reached-through edges, by weight,
        # with a FAR node mentioned fewer than GRAPH_EXPANSION_MIN_MENTIONS times
        # sorted last — deprioritized, never dropped ("never wrong > always active").
        try:
            from config.app_config import GRAPH_EXPANSION_MIN_MENTIONS  # lazy import: live-config read
            min_mentions = int(GRAPH_EXPANSION_MIN_MENTIONS)
        except Exception:
            min_mentions = _DEFAULT_MIN_MENTIONS

        def _mentions(nid: str) -> int:
            node = self.get_entity(nid)
            try:
                return int(getattr(node, "mention_count", 0) or 0)
            except (TypeError, ValueError):
                return 0

        def _rank(e: GraphEdge):
            # tier 0: incident to the seed (weight order — depth-1 contract)
            # tier 1: reached-through edge whose endpoints are well-mentioned
            # tier 2: reached-through edge touching a rarely-mentioned node
            # tier 3: an edge of the ``user`` star hub reached through a pet/
            #         project seed — legitimate, but never ahead of the seed's
            #         own neighbourhood (the pre-fix "Python mentioned alongside
            #         name" class)
            if e.source_id == seed or e.target_id == seed:
                return (0, -e.weight)
            if seed != "user" and "user" in (e.source_id, e.target_id):
                return (3, -e.weight)
            # both endpoints are non-seed here; the lower-mention endpoint is
            # the "far" one (a one-off object like ``huge`` or ``possibly``)
            far_mentions = min(_mentions(e.source_id), _mentions(e.target_id))
            return (1 if far_mentions >= min_mentions else 2, -e.weight)

        edges.sort(key=_rank)

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

        payload = {"schema_version": GRAPH_SCHEMA_VERSION, "nodes": nodes, "edges": edges}

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
        """Load graph from JSON file.  No-op if file doesn't exist.

        An existing-but-corrupt file raises CorruptStoreError (with a
        quarantined copy) instead of starting empty — an empty graph would
        overwrite the user's accumulated graph on the next save.
        """
        if not os.path.exists(self.persist_path):
            logger.info(f"[GraphMemory] No graph file at {self.persist_path}, starting fresh")
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                payload = _json_load(f)
        except (ValueError, OSError) as e:
            from utils.safe_json import corrupt_store
            raise corrupt_store(self.persist_path, "Knowledge graph", e) from e

        # Refuse files written by a NEWER build (missing version = v1).
        from utils.safe_json import check_schema_version
        check_schema_version(payload, current=GRAPH_SCHEMA_VERSION,
                             path=self.persist_path, label="Knowledge graph")

        # Load nodes
        for nid, data in payload.get("nodes", {}).items():
            self.graph.add_node(nid, **data)
            # Rebuild alias index (generic-word junk aliases in historical
            # node data are skipped here — neutralized without a store repair)
            self._alias_index[nid] = nid
            for alias in data.get("aliases", []):
                a_lower = alias.lower().strip()
                if _alias_bindable(a_lower, nid):
                    self._alias_index[a_lower] = nid

        # Load edges — every relation lands in the index + adjacency maps;
        # the nx pair is created once (first relation in file order owns the
        # advisory attrs).  Pre-2026-09-03 this add_edge ran unconditionally,
        # so JSON order decided which relation survived a multi-relation pair.
        for edata in payload.get("edges", []):
            edge = GraphEdge.from_dict(edata)
            src = edge.source_id
            tgt = edge.target_id
            ekey = edge.edge_key()
            self._edge_index[ekey] = edge
            self._index_edge(edge, key=ekey)
            if not self.graph.has_edge(src, tgt):
                self.graph.add_edge(src, tgt, **self._nx_edge_attrs(edge))

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
                self._edge_index.pop(ekey, None)
                self._unindex_edge(edge, key=ekey)
                src, tgt = edge.source_id, edge.target_id
                if not self.graph.has_edge(src, tgt):
                    continue
                # The nx pair stays while ANY relation on it survives; if the
                # removed relation owned the advisory attrs, re-point them.
                survivor = None
                for k in self._out_keys.get(src, ()):
                    cand = self._edge_index.get(k)
                    if cand is not None and cand.target_id == tgt:
                        survivor = cand
                        break
                if survivor is None:
                    self.graph.remove_edge(src, tgt)
                elif self.graph[src][tgt].get("relation") == edge.relation:
                    self.graph[src][tgt].update(self._nx_edge_attrs(survivor))
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
