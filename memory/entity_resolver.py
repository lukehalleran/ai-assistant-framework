# memory/entity_resolver.py
"""
Entity resolution and relation normalization for the knowledge graph.

EntityResolver maps raw mentions ("my cat", "the project", "Spain")
to canonical graph node IDs via an alias index.  Resolution strategy
is deliberately simple (exact match + alias table) — fuzzy and semantic
matching can be added later without changing the interface.

Relation normalization maps raw relation strings ("lives in",
"resides in") to canonical forms ("lives_in").
"""

import json
import os
import re
from typing import Optional

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from utils.logging_utils import get_logger

logger = get_logger("entity_resolver")


# ------------------------------------------------------------------
# Relation normalization
# ------------------------------------------------------------------

RELATION_SYNONYMS: dict[str, list[str]] = {
    "lives_in": ["lives in", "resides in", "located in", "based in", "living in"],
    "works_on": ["works on", "building", "developing", "working on"],
    "created": ["created", "built", "made", "authored", "wrote"],
    "works_at": ["works at", "works for", "employed at", "employed by"],
    "wants_to_move_to": ["wants to move to", "plans to move to", "moving to"],
    "wants_to": ["wants to", "plans to", "intends to", "hopes to"],
    "studies_at": ["studies at", "enrolled in", "attending", "enrolled at"],
    "studies": ["studies", "studying", "learning", "taking"],
    "likes": ["likes", "enjoys", "loves", "fond of", "appreciates"],
    "dislikes": ["dislikes", "hates", "doesn't like", "avoids"],
    "owns": ["owns", "has", "possesses"],
    "is_a": ["is a", "is an"],
    "part_of": ["part of", "member of", "belongs to"],
    "related_to": ["related to", "connected to", "associated with"],
    "friend_of": ["friend of", "friends with"],
    "sibling_of": ["sibling of", "brother of", "sister of"],
    "parent_of": ["parent of", "father of", "mother of"],
    "child_of": ["child of", "son of", "daughter of"],
    "spouse_of": ["spouse of", "married to", "partner of"],
    "pet_of": ["pet of", "owned by"],
    "speaks": ["speaks", "fluent in", "learning to speak"],
    "born_in": ["born in", "from", "originally from", "native of"],
    "interested_in": ["interested in", "curious about", "passionate about"],
    "skilled_at": ["skilled at", "good at", "proficient in", "experienced in"],
}

# Build reverse lookup for fast matching
_RELATION_REVERSE: dict[str, str] = {}
for canonical, synonyms in RELATION_SYNONYMS.items():
    _RELATION_REVERSE[canonical] = canonical
    for syn in synonyms:
        _RELATION_REVERSE[syn.lower()] = canonical


def normalize_relation(raw: str) -> str:
    """Map raw relation strings to canonical forms.

    Examples:
        "lives in" -> "lives_in"
        "resides in" -> "lives_in"
        "favorite_color" -> "favorite_color"  (pass-through)
    """
    raw_lower = raw.lower().strip()
    if raw_lower in _RELATION_REVERSE:
        return _RELATION_REVERSE[raw_lower]
    # Pass through, replacing spaces with underscores
    return re.sub(r"\s+", "_", raw_lower)


# ------------------------------------------------------------------
# Possessive alias patterns  ("my cat" -> entity, "my boss" -> entity)
# ------------------------------------------------------------------

_POSSESSIVE_RE = re.compile(
    r"\bmy\s+(cat|dog|pet|kitten|puppy|boss|manager|wife|husband|partner|"
    r"friend|brother|sister|mom|dad|mother|father|son|daughter|"
    r"project|app|system|car|apartment|house|home)\b",
    re.IGNORECASE,
)


def extract_possessive_aliases(text: str) -> list[tuple[str, str]]:
    """Extract possessive phrases that could be aliases.

    Returns list of (alias_phrase, entity_type_hint) tuples.
    E.g. "my cat Flapjack" -> [("my cat", "pet")]
    """
    type_map = {
        "cat": "pet", "dog": "pet", "pet": "pet", "kitten": "pet", "puppy": "pet",
        "boss": "person", "manager": "person", "wife": "person", "husband": "person",
        "partner": "person", "friend": "person", "brother": "person", "sister": "person",
        "mom": "person", "dad": "person", "mother": "person", "father": "person",
        "son": "person", "daughter": "person",
        "project": "project", "app": "project", "system": "project",
        "car": "other", "apartment": "place", "house": "place", "home": "place",
    }
    results = []
    for m in _POSSESSIVE_RE.finditer(text):
        noun = m.group(1).lower()
        alias = m.group(0).lower()
        etype = type_map.get(noun, "other")
        results.append((alias, etype))
    return results


# ------------------------------------------------------------------
# EntityResolver
# ------------------------------------------------------------------

class EntityResolver:
    """Resolves raw entity mentions to canonical graph node IDs.

    Resolution is exact-match only (alias index lookup).
    Loads additional aliases from a JSON file for bootstrapping.
    """

    def __init__(self, graph_memory: GraphMemory, aliases_path: str = ""):
        self.graph = graph_memory
        self.aliases_path = aliases_path
        self._load_external_aliases()

    def _load_external_aliases(self) -> None:
        """Load aliases from external JSON file if it exists.

        Format: {"entity_id": ["alias1", "alias2", ...], ...}
        """
        if not self.aliases_path or not os.path.exists(self.aliases_path):
            return
        try:
            with open(self.aliases_path, "r", encoding="utf-8") as f:
                aliases_data = json.load(f)
            count = 0
            for entity_id, aliases in aliases_data.items():
                eid = entity_id.lower().strip()
                for alias in aliases:
                    a_lower = alias.lower().strip()
                    if a_lower:
                        self.graph.register_alias(a_lower, eid)
                        count += 1
            logger.info(f"[EntityResolver] Loaded {count} external aliases from {self.aliases_path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[EntityResolver] Failed to load aliases from {self.aliases_path}: {e}")

    def resolve(self, mention: str) -> Optional[str]:
        """Resolve a mention to a canonical entity_id.

        Returns the entity_id or None if not found.
        """
        return self.graph.resolve_entity(mention)

    def resolve_or_create(self, mention: str, entity_type: str = "other",
                          display_name: str = "") -> str:
        """Resolve a mention, creating a new entity if not found.

        Returns the canonical entity_id.
        """
        existing = self.resolve(mention)
        if existing:
            return existing

        # Create new entity
        eid = _normalize_id(mention)
        node = GraphNode(
            entity_id=eid,
            display_name=display_name or mention,
            entity_type=entity_type,
        )
        return self.graph.add_entity(node)

    def learn_alias(self, alias: str, entity_id: str, context: str = "") -> None:
        """Register a new alias, optionally learning from context.

        Also auto-detects possessive aliases from context text.
        """
        self.graph.register_alias(alias, entity_id)

        # Auto-learn possessive aliases if context mentions them
        if context:
            possessives = extract_possessive_aliases(context)
            for poss_alias, _ in possessives:
                # Only bind if the possessive phrase isn't already claimed
                if not self.graph.resolve_entity(poss_alias):
                    self.graph.register_alias(poss_alias, entity_id)
                    logger.debug(f"[EntityResolver] Auto-learned possessive alias: '{poss_alias}' -> '{entity_id}'")

    def save_external_aliases(self) -> None:
        """Save current alias state to external JSON file for bootstrapping."""
        if not self.aliases_path:
            return
        # Group aliases by entity
        by_entity: dict[str, list[str]] = {}
        for alias, eid in self.graph._alias_index.items():
            if alias != eid:  # skip self-references
                by_entity.setdefault(eid, []).append(alias)

        os.makedirs(os.path.dirname(self.aliases_path) or ".", exist_ok=True)
        try:
            with open(self.aliases_path, "w", encoding="utf-8") as f:
                json.dump(by_entity, f, indent=2, ensure_ascii=False)
            logger.info(f"[EntityResolver] Saved aliases to {self.aliases_path}")
        except OSError as e:
            logger.warning(f"[EntityResolver] Failed to save aliases: {e}")


def _normalize_id(mention: str) -> str:
    """Normalize a mention string into a valid entity ID.

    "Georgia Tech" -> "georgia_tech"
    "My Cat" -> "my_cat"
    """
    s = mention.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s
