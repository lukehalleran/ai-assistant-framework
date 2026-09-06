# memory/graph_utils.py
"""
Shared helpers for graph-boosted scoring and graph-driven query expansion.

Provides entity extraction from text (via alias resolution) and neighbor
display-name lookups, shared by memory_scorer (graph boost) and
context_gatherer (query expansion).

Query expansion (rank_expansion_candidates) uses a hub-aware BFS: a node in
skip_ids or with degree >= hub_degree may be reached but is never expanded
*through*, so a high-degree star hub (e.g. "user") can't fan its whole
neighbourhood into a query.  Seed entities are exempt and always expand.
Expansion also honours read-time TTL: stale transient edges (past their
per-relation horizon, via GraphMemory._edge_is_stale_transient →
relation_classifier) are dropped from traversal and scoring, so aged-out
mood/activity/illness states stop surfacing without any deletion.

Stance filter [2026-08-23]: edges whose metadata carries an EXPLICIT
"appraisal" or "inferred" stance (memory/stance_classifier.effective_stance)
never route expansion or score — the user's value judgment (casey--is-->evil)
is their take, not a topical bridge; it had leaked "evil" into unrelated
queries' expansion terms. Legacy untagged edges ("unknown") are unchanged.
"""

import os
import re
from collections import deque
from datetime import datetime
from typing import List, Optional, Set

from utils.logging_utils import get_logger

logger = get_logger("graph_utils")

# Default degree at/above which a node is treated as a traversal *hub*: it may
# be reached during expansion, but the walk does not fan out *through* it.
# This stops a high-degree node (e.g. the "user" star hub) from dumping its
# entire neighbourhood into the query whenever any token incidentally links to
# it.  Overridable per-call via the ``hub_degree`` argument.
_DEFAULT_HUB_DEGREE = 30

# Minimum evidence bar for expansion candidates (2026-08-23): a node the graph
# has only seen ONCE is one extraction event, not an established concept — a
# single incidental mention shouldn't inject its display name into unrelated
# queries.  Nodes whose mention_count is below this bar are skipped during
# candidate scoring.  Env `GRAPH_EXPANSION_MIN_MENTIONS` overrides (0 disables);
# nodes without a mention_count attribute (mocks, legacy stores) are exempt.
_DEFAULT_MIN_MENTIONS = 2

# Stopwords to skip during entity extraction (same set used by context_gatherer)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "do", "does",
    "did", "have", "has", "had", "what", "who", "where", "when",
    "how", "why", "about", "with", "from", "for", "and", "or",
    "but", "not", "to", "in", "on", "at", "of", "my", "your",
    "i", "me", "you", "we", "they", "it", "this", "that", "can",
    "will", "would", "should", "could", "tell", "know", "think",
})

# Pronouns and generic words that should never be graph entities
_JUNK_ENTITIES = frozenset({
    # Pronouns
    "him", "her", "them", "his", "their", "its", "he", "she", "we",
    # Quantifiers / adjectives
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "much", "most", "more", "some", "other", "each", "both", "all",
    "possible", "also", "well", "first", "last", "new", "good", "bad", "best", "worst",
    "really", "very", "just", "only", "still", "even", "now", "then",
    # Generic words
    "said", "which", "way", "area", "method", "type", "kind",
    "thing", "stuff", "lot", "bit", "part", "point", "end",
    # Meta/conversational
    "response", "question", "answer", "content", "output", "error",
    "result", "process", "system", "data", "information",
})


_TEMPORAL_RE = re.compile(
    r"^\d+(\.\d+)?\s*(years?|months?|weeks?|days?|hours?)", re.IGNORECASE
)
# Bare temporal deictics / calendar names (2026-09-05). "today" and "tomorrow"
# in a casual plans update resolved to live graph nodes `today` (11 in-edges:
# `user|dad|today`, `user|project|today`, `user|feels_better|today` …) and
# `tomorrow`, and [KNOWLEDGE GRAPH] rendered nine junk lines; the memory
# ranker's graph boost pulled 29 "related names" through the same nodes.
# A deictic is never an entity: not a seed, not a subject, not an object.
# Full weekday/month names only (abbreviations like "sun"/"mar"/"may" collide
# with real words); an optional on/this/next/last/every prefix is absorbed so
# the normalized node id `on_thursday` is caught too.
_TEMPORAL_DEICTIC_RE = re.compile(
    r"^(?:(?:on|this|next|last|every|by|until|till|since)\s+)?"
    r"(?:today|tomorrow|tmrw|yesterday|tonight|now|later|soon|earlier|recently|lately|"
    r"morning|afternoon|evening|night|noon|midnight|weekend|weekends|weekday|weekdays|"
    r"monday|tuesday|tues|wednesday|thursday|thurs|friday|saturday|sunday|"
    r"january|february|march|april|june|july|august|september|sept|october|november|december|"
    r"(?:this|next|last|the|past|coming)\s+(?:week|month|year|weekend|semester|term|day)|"
    r"(?:a\s+)?(?:few|couple(?:\s+of)?)\s+(?:days|weeks|months|years)(?:\s+ago)?)$",
    re.IGNORECASE,
)


def is_temporal_deictic(name: str) -> bool:
    """True for a bare when-word ("today", "on thursday", "next week") — a
    graph node id may carry underscores ("on_thursday"); both forms match."""
    n = (name or "").strip().lower().replace("_", " ")
    n = re.sub(r"\s+", " ", n)
    return bool(n) and bool(_TEMPORAL_DEICTIC_RE.match(n))
_FREQUENCY_RE = re.compile(
    r"^(once|twice|three\s+times)\s+a\s+", re.IGNORECASE
)
_MEASUREMENT_RE = re.compile(
    r"^\d[\d'\".,]*(lbs?|iu|mg|mcg|kg|oz|ft|in)?\s*$", re.IGNORECASE
)

_VERB_STEMS = frozenset({
    "stopped", "started", "went", "came", "used", "began",
    "finished", "understands", "feels", "thinks", "strained",
    "relying", "protecting", "finishing", "moved", "working",
    "planning", "trying", "wanting", "becoming", "living",
    "dealing", "struggling", "considering", "taking",
})


def is_junk_entity(name: str) -> bool:
    """Return True if *name* is a stopword, pronoun, measurement, or other
    non-entity that should never be a graph node.

    Used by:
    - Graph ingestion (_ingest_fact_to_graph) to filter subjects AND objects
    - Query expansion (rank_expansion_candidates) to filter candidates
    - Wiki enrichment (_link_conversation_entities) to filter matches
    - extract_graph_entities() to filter extraction results
    """
    n = name.strip().lower()
    if not n or len(n) <= 2:
        return True
    if n in _STOPWORDS or n in _JUNK_ENTITIES:
        return True
    if len(n.split()) >= 4:
        return True
    if n[0].isdigit():
        return True
    if _TEMPORAL_RE.match(n) or _FREQUENCY_RE.match(n) or _MEASUREMENT_RE.match(n):
        return True
    if is_temporal_deictic(n):
        return True
    first_word = n.split()[0]
    if first_word in _VERB_STEMS:
        return True
    return False


def _is_expansion_junk(name: str) -> bool:
    """Return True if *name* looks like a junk phrase, not a real entity."""
    return is_junk_entity(name)


# Species words a relation can embed ("has_dog", "adopted_kitten"), mapped to
# their canonical species for comparison against node metadata. Juvenile forms
# collapse to the adult species.
_RELATION_SPECIES = {
    "dog": "dog", "puppy": "dog",
    "cat": "cat", "kitten": "cat",
    "bird": "bird", "fish": "fish", "hamster": "hamster",
    "rabbit": "rabbit", "bunny": "rabbit", "horse": "horse",
    "snake": "snake", "lizard": "lizard", "ferret": "ferret",
    "turtle": "turtle", "guinea_pig": "guinea_pig",
}

_RELATION_SPECIES_RE = re.compile(
    r"(?:^|_)(" + "|".join(sorted(_RELATION_SPECIES, key=len, reverse=True)) + r")s?(?:_|$)"
)


def relation_species_conflict(relation: str, node_metadata: Optional[dict]) -> bool:
    """True when a relation embeds an animal species that contradicts the
    target node's own ``species`` metadata.

    2026-08-18: the shutdown LLM extractor invented ``user | has_dog |
    Daisy`` from an excerpt that never mentions a dog; the graph node
    already carried curated ``species: cat`` metadata and an older
    ``has_cat`` edge — and the wrong edge fed a junk proactive insight.
    Curated node metadata is ground truth here: a single extraction naming
    a conflicting species is an extraction error, not new information.
    Fires ONLY when the node explicitly declares a species (under-fires by
    design); nodes without species metadata are never blocked.
    """
    if not node_metadata:
        return False
    declared = str(node_metadata.get("species", "") or "").strip().lower()
    if not declared:
        return False
    m = _RELATION_SPECIES_RE.search(str(relation or "").strip().lower())
    if not m:
        return False
    claimed = _RELATION_SPECIES[m.group(1)]
    # The declared value may be descriptive ("cat, black, big golden eyes") —
    # word-boundary containment, not equality. Multi-word species carry an
    # underscore in relation form but a space in declared metadata
    # ("guinea_pig" vs "guinea pig") — accept either separator.
    claimed_pat = re.escape(claimed).replace("_", r"[_\s]")
    return not re.search(r"\b" + claimed_pat + r"s?\b", declared)


def rank_expansion_candidates(
    entity_ids: Set[str],
    graph_memory,
    depth: int = 2,
    skip_ids: Optional[Set[str]] = None,
    max_terms: int = 8,
    hub_degree: Optional[int] = None,
    min_mentions: Optional[int] = None,
) -> List[str]:
    """Rank expansion candidates by connectivity (non-hub edge count).

    Walks *depth* hops from *entity_ids* with a **hub-aware** BFS: a hub node
    (anything in *skip_ids*, or any node whose degree is >= *hub_degree*) may
    be *reached*, but the walk never fans out *through* it.  Seed entities are
    exempt — a seed always expands, even if it is itself a hub.  This keeps the
    intended relation-bridging (``"my brother"`` → seed ``user`` → ``drew``)
    while preventing the failure where an incidental token resolves to a leaf
    that links to the ``user`` star hub and the 2-hop walk then dumps every one
    of the user's neighbours (pets, project terms, ...) into the query.

    Stale transient edges (mood/activity/illness past their per-relation TTL,
    per ``GraphMemory._edge_is_stale_transient`` → ``relation_classifier``) are
    dropped from both traversal and scoring, so an aged-out state neither routes
    expansion nor inflates a candidate's rank — the same read-time staleness the
    profile, facts collection and graph-context already apply.

    Candidates are scored by how many lateral (non-hub) edges they have and the
    top *max_terms* display names are returned.

    Args:
        entity_ids: Set of canonical entity IDs to start from (seeds)
        graph_memory: GraphMemory instance (needs .get_entity(), .get_relations())
        depth: BFS traversal depth (default 2 for star topologies)
        skip_ids: Entity IDs that are hubs / excluded from results (e.g. {"user"})
        max_terms: Maximum names to return
        hub_degree: Degree threshold for auto-detecting hubs (default
            ``_DEFAULT_HUB_DEGREE``).  Non-seed nodes at/above this degree act
            as traversal barriers even if not named in *skip_ids*.
        min_mentions: Evidence bar — candidates whose ``mention_count`` is
            below this are skipped (default ``_DEFAULT_MIN_MENTIONS``, env
            ``GRAPH_EXPANSION_MIN_MENTIONS``; 0 disables).  Nodes without the
            attribute are exempt.

    Returns:
        Ordered list of display names, best candidates first
    """
    if not entity_ids or not graph_memory:
        return []

    skip = skip_ids or set()
    seeds = {e.lower() for e in entity_ids}
    threshold = _DEFAULT_HUB_DEGREE if hub_degree is None else hub_degree
    if min_mentions is None:
        try:
            min_mentions = int(
                os.getenv("GRAPH_EXPANSION_MIN_MENTIONS", _DEFAULT_MIN_MENTIONS))
        except (TypeError, ValueError):
            min_mentions = _DEFAULT_MIN_MENTIONS

    # Read-time TTL: reuse GraphMemory's staleness check (single source of truth
    # in relation_classifier) so a transient edge (mood/activity/illness past its
    # per-relation horizon) neither routes expansion nor scores — same read-side
    # staleness the profile, facts collection and graph-context already apply.
    # getattr-guarded so mocks / stores without the method simply keep all edges.
    _now = datetime.now()
    _stale_fn = getattr(graph_memory, "_edge_is_stale_transient", None)

    def _live(edge) -> bool:
        # Stance filter (2026-08-23): EXPLICIT appraisal/inferred edges never
        # route expansion or score — the user's value judgment (casey--is-->evil)
        # is their take, not a topical bridge, and it leaked "evil" into
        # unrelated queries' expansion terms. Legacy edges without a stance
        # tag ("unknown") are unchanged — suppression acts only on explicit
        # tags (conservative missing-field semantics).
        try:
            from memory.stance_classifier import effective_stance
            if effective_stance(getattr(edge, "metadata", None)) in (
                    "appraisal", "inferred"):
                return False
        except Exception:
            pass
        if _stale_fn is None:
            return True
        try:
            return not _stale_fn(edge, _now)
        except Exception:
            return True

    def _neighbor_ids(nid: str) -> Set[str]:
        out: Set[str] = set()
        try:
            for edge in graph_memory.get_relations(nid, direction="both"):
                if not _live(edge):
                    continue
                other = edge.target_id if edge.source_id == nid else edge.source_id
                if other:
                    out.add(other)
        except Exception:
            pass
        return out

    def _is_hub(nid: str) -> bool:
        # Seeds always expand (intentional relation expansion), even if a hub.
        if nid in seeds:
            return False
        if nid in skip:
            return True
        try:
            return len(graph_memory.get_relations(nid, direction="both")) >= threshold
        except Exception:
            return False

    # Hub-aware BFS: collect candidates reachable from seeds without expanding
    # *through* a hub.  A hub may be reached (then dropped via skip/scoring) but
    # its neighbours are never enqueued — that is what stops the star-hub fan-out.
    candidate_ids: Set[str] = set()
    visited: Set[str] = set(seeds)
    queue: deque = deque((s, 0) for s in seeds)
    while queue:
        cur, d = queue.popleft()
        if cur not in seeds and cur not in skip:
            candidate_ids.add(cur)
        if d >= depth or _is_hub(cur):
            continue
        for nb in _neighbor_ids(cur):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1))

    # Score each candidate
    scored: list[tuple[float, str]] = []  # (score, display_name)
    for cid in candidate_ids:
        node = graph_memory.get_entity(cid)
        if not node or not node.display_name:
            continue
        name = node.display_name.strip()
        if not name or name.lower() in _STOPWORDS:
            continue
        if _is_expansion_junk(name):
            continue
        # Evidence bar: a once-mentioned node is a single extraction event,
        # not an established concept — don't let it steer query expansion.
        mentions = getattr(node, "mention_count", None)
        if mentions is not None and min_mentions > 0 and mentions < min_mentions:
            continue

        # Count non-hub edges (edges where the other end is NOT in skip_ids).
        # Skip stale transient edges so an aged-out state doesn't inflate rank.
        non_hub_edges = 0
        try:
            relations = graph_memory.get_relations(cid, direction="both")
            for edge in relations:
                if not _live(edge):
                    continue
                other = edge.target_id if edge.source_id == cid else edge.source_id
                if other not in skip:
                    non_hub_edges += 1
        except Exception:
            pass

        score = min(non_hub_edges * 0.3, 1.0)
        # Prefer concise names
        word_count = len(name.split())
        if word_count == 1:
            score += 0.1
        elif word_count >= 3:
            score -= 0.1

        scored.append((score, name))

    # Sort descending by score, then alphabetically for stability
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, name in scored[:max_terms]]


def extract_graph_entities(text: str, resolver, graph_memory=None) -> Set[str]:
    """Extract entity IDs from text using alias resolution.

    Checks trigrams, bigrams, then single words against the resolver's
    alias index.  Returns a set of canonical entity IDs.

    Single-word matches are filtered: generic concept/wikidata nodes are
    excluded to prevent common English words (e.g. "speed", "test") from
    polluting query expansion with irrelevant entities.

    Args:
        text: Input text to scan for entity mentions
        resolver: EntityResolver (or any object with a .resolve(str)->Optional[str] method)
        graph_memory: Optional GraphMemory — if provided, single-word matches
            are filtered to only include personal entity types (person, pet, etc.)

    Returns:
        Set of canonical entity IDs found in the text
    """
    if not text or not resolver:
        return set()

    # Entity types that are bulk-imported concepts — single-word matches
    # against these are almost always false positives (e.g. "speed", "test").
    _CONCEPT_TYPES = frozenset({"concept", "wikidata_entity"})

    # Common English words that should never resolve as entities on their own.
    # These cause false positives in query expansion (e.g. "speed test" →
    # the node "speed" → user's pet, coworker, etc. via graph walk).
    # Named entities (Whiskers, Max, Denver) aren't in this set.
    _COMMON_WORDS = _STOPWORDS | frozenset({
        "test", "speed", "light", "water", "fire", "time", "work",
        "part", "point", "line", "end", "run", "set", "day", "way",
        "back", "long", "high", "old", "new", "big", "good", "bad",
        "first", "last", "next", "little", "much", "right", "left",
        "same", "well", "also", "just", "only", "still", "even",
        "here", "there", "now", "then", "over", "down", "after",
        "before", "between", "under", "look", "give", "take", "make",
        "come", "going", "doing", "thing", "really", "getting",
        "feel", "feeling", "trying", "keep", "start", "stop",
        "need", "want", "like", "love", "hate", "mean", "kind",
        "sort", "type", "form", "plan", "code", "data", "system",
        "model", "graph", "node", "edge", "query", "search",
        # Common nouns/participles that are not entities but had been ingested
        # as graph nodes and pulled the user hub into unrelated queries.
        "video", "videos", "done", "homework", "stuff",
        # Generic possession/admin nouns that get auto-learned as ALIASES of
        # specific personal entities ("my project" → possessive alias "project"
        # on phase_change_heat_exchanger_project). On 2026-08-28 the bare word
        # "project" in "group project" resolved to that node and its stored
        # notes PHOTO was attached to an unrelated memory-ingest turn. These
        # words are constant in school/medical-admin vocabulary and must never
        # resolve as single-word entity mentions (n-gram matches still can).
        "project", "projects", "note", "notes", "email", "emails",
        "meeting", "meetings", "appointment", "appointments",
        "calendar", "syllabus", "screenshot", "screenshots",
        "photo", "photos", "image", "images", "picture", "pictures",
        "document", "documents", "file", "files", "folder", "folders",
    })

    # Strip punctuation from each word for cleaner matching
    words = [re.sub(r"[^\w\s]", "", w) for w in text.lower().split()]
    words = [w for w in words if w]  # remove empty strings
    entity_ids: Set[str] = set()

    # Check n-grams (trigram, bigram) first for multi-word entities
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            eid = resolver.resolve(phrase)
            if eid:
                entity_ids.add(eid)

    # Then single words (skip stopwords, short tokens, and common English words)
    for w in words:
        if w in _COMMON_WORDS or len(w) <= 2:
            continue
        eid = resolver.resolve(w)
        if eid:
            # Skip bulk-imported concept nodes (wikidata)
            if graph_memory:
                node = graph_memory.get_entity(eid)
                etype = getattr(node, 'entity_type', None) if node else None
                if etype in _CONCEPT_TYPES:
                    continue
            entity_ids.add(eid)

    # Final junk filter — catch anything that slipped through n-gram matching
    entity_ids = {eid for eid in entity_ids if not is_junk_entity(eid)}

    return entity_ids


def get_related_display_names(
    entity_ids: Set[str],
    graph_memory,
    depth: int = 1,
    skip_ids: Optional[Set[str]] = None,
) -> Set[str]:
    """Get display names of entities within `depth` hops of given entities.

    Args:
        entity_ids: Set of canonical entity IDs to start from
        graph_memory: GraphMemory instance
        depth: BFS traversal depth (default 1-hop)
        skip_ids: Entity IDs to exclude from results (e.g. "user")

    Returns:
        Set of display names for neighbor entities
    """
    if not entity_ids or not graph_memory:
        return set()

    skip = skip_ids or set()
    display_names: Set[str] = set()

    for eid in entity_ids:
        neighborhood = graph_memory.neighbors(eid, depth=depth)
        for neighbor_id, edges in neighborhood.items():
            if neighbor_id in skip or neighbor_id in entity_ids:
                continue
            node = graph_memory.get_entity(neighbor_id)
            if node and node.display_name:
                name = node.display_name.strip()
                # Skip generic/short names
                if len(name) > 1 and name.lower() not in _STOPWORDS:
                    display_names.add(name)

    return display_names


def get_related_entity_ids(
    entity_ids: Set[str],
    graph_memory,
    depth: int = 1,
) -> Set[str]:
    """Get entity IDs within `depth` hops of given entities.

    Args:
        entity_ids: Set of canonical entity IDs to start from
        graph_memory: GraphMemory instance
        depth: BFS traversal depth

    Returns:
        Set of neighboring entity IDs (excludes the input entity_ids)
    """
    if not entity_ids or not graph_memory:
        return set()

    neighbor_ids: Set[str] = set()
    for eid in entity_ids:
        neighborhood = graph_memory.neighbors(eid, depth=depth)
        for neighbor_id in neighborhood:
            if neighbor_id not in entity_ids:
                neighbor_ids.add(neighbor_id)

    return neighbor_ids
