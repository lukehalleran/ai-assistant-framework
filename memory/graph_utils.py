# memory/graph_utils.py
"""
Shared helpers for graph-boosted scoring and graph-driven query expansion.

Provides entity extraction from text (via alias resolution) and neighbor
display-name lookups, shared by memory_scorer (graph boost) and
context_gatherer (query expansion).
"""

import re
from typing import List, Optional, Set

from utils.logging_utils import get_logger

logger = get_logger("graph_utils")

# Stopwords to skip during entity extraction (same set used by context_gatherer)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "do", "does",
    "did", "have", "has", "had", "what", "who", "where", "when",
    "how", "why", "about", "with", "from", "for", "and", "or",
    "but", "not", "to", "in", "on", "at", "of", "my", "your",
    "i", "me", "you", "we", "they", "it", "this", "that", "can",
    "will", "would", "should", "could", "tell", "know", "think",
})


_TEMPORAL_RE = re.compile(
    r"^\d+(\.\d+)?\s*(years?|months?|weeks?|days?|hours?)", re.IGNORECASE
)
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


def _is_expansion_junk(name: str) -> bool:
    """Return True if *name* looks like a junk phrase, not a real entity."""
    n = name.strip()
    if not n or len(n) <= 2:
        return True
    if len(n.split()) >= 4:
        return True
    if n[0].isdigit():
        return True
    if _TEMPORAL_RE.match(n) or _FREQUENCY_RE.match(n) or _MEASUREMENT_RE.match(n):
        return True
    first_word = n.split()[0].lower()
    if first_word in _VERB_STEMS:
        return True
    return False


def rank_expansion_candidates(
    entity_ids: Set[str],
    graph_memory,
    depth: int = 2,
    skip_ids: Optional[Set[str]] = None,
    max_terms: int = 8,
) -> List[str]:
    """Rank expansion candidates by connectivity (non-hub edge count).

    Walks *depth* hops from *entity_ids*, collects candidate neighbor nodes,
    scores them by how many lateral (non-hub) edges they have, and returns
    the top *max_terms* display names.

    Args:
        entity_ids: Set of canonical entity IDs to start from
        graph_memory: GraphMemory instance (needs .neighbors(), .get_entity(), .get_relations())
        depth: BFS traversal depth (default 2 for star topologies)
        skip_ids: Entity IDs to exclude (e.g. {"user"})
        max_terms: Maximum names to return

    Returns:
        Ordered list of display names, best candidates first
    """
    if not entity_ids or not graph_memory:
        return []

    skip = skip_ids or set()
    # Collect unique candidate IDs (exclude inputs and skipped)
    candidate_ids: Set[str] = set()
    for eid in entity_ids:
        neighborhood = graph_memory.neighbors(eid, depth=depth)
        for neighbor_id in neighborhood:
            if neighbor_id not in skip and neighbor_id not in entity_ids:
                candidate_ids.add(neighbor_id)

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

        # Count non-hub edges (edges where the other end is NOT in skip_ids)
        non_hub_edges = 0
        try:
            relations = graph_memory.get_relations(cid, direction="both")
            for edge in relations:
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


def extract_graph_entities(text: str, resolver) -> Set[str]:
    """Extract entity IDs from text using alias resolution.

    Checks trigrams, bigrams, then single words against the resolver's
    alias index.  Returns a set of canonical entity IDs.

    Args:
        text: Input text to scan for entity mentions
        resolver: EntityResolver (or any object with a .resolve(str)->Optional[str] method)

    Returns:
        Set of canonical entity IDs found in the text
    """
    if not text or not resolver:
        return set()

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

    # Then single words (skip stopwords and short tokens)
    for w in words:
        if w not in _STOPWORDS and len(w) > 2:
            eid = resolver.resolve(w)
            if eid:
                entity_ids.add(eid)

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
