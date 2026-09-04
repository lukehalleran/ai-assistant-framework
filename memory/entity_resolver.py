# memory/entity_resolver.py
"""
Entity resolution and relation normalization for the knowledge graph.

EntityResolver maps raw mentions ("my cat", "the project", "Spain")
to canonical graph node IDs via an alias index.  Resolution strategy
is deliberately simple (exact match + alias table) — fuzzy and semantic
matching can be added later without changing the interface.

Relation normalization maps raw relation strings ("lives in",
"resides in") to canonical forms ("lives_in") via three layers:
exact synonym table (RELATION_SYNONYMS), hygiene (spaces/hyphens →
underscores, punctuation stripped, repeats collapsed), then conservative
family-collapse patterns ("asked_about_wakeup_time" → "asked_about") —
per-fact specifics belong in the object, not the relation name.
Extended 2026-08-02 (630/696 stored relations were single-use inventions):
synonym table grew (drinks/attends/completed/likes_*/dislikes_*/...) and the
family patterns cover talked_about_*/mentioned_* — with a negative lookahead
excluding `mentioned_alongside`, a code-level structural relation used by
wiki enrichment. Historical edges predating a pattern can be re-canonicalized
with scripts/graph_relation_normalize.py (dry-run-first; --apply run 08-02,
24 edges renamed).

Persistence (entity_aliases.json): atomic write (utils.safe_json) and
strict load — an existing-but-corrupt file is quarantined and raises
CorruptStoreError instead of silently starting empty [2026-07-14].
"""

import json
import os
import re
from typing import Optional

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from utils.logging_utils import get_logger
from utils.safe_json import atomic_write_json, load_critical_json

logger = get_logger("entity_resolver")


# ------------------------------------------------------------------
# Relation normalization
# ------------------------------------------------------------------

RELATION_SYNONYMS: dict[str, list[str]] = {
    "lives_in": ["lives in", "resides in", "located in", "based in", "living in"],
    "works_on": ["works on", "building", "developing", "working on"],
    "created": ["created", "built", "made", "authored", "wrote"],
    "works_at": ["works at", "works for", "employed at", "employed by"],
    # 2026-09-03: `role` was an invented relation minted from activities
    # ("bug fixer", "PR approver"); folding it into occupation gives profile
    # canonicalization, graph ingest AND learned-relation hygiene one chokepoint.
    "occupation": ["role", "job_title", "job title"],
    "wants_to_move_to": ["wants to move to", "plans to move to", "moving to"],
    "wants_to": ["wants to", "plans to", "intends to", "hopes to", "desire",
                 "desires", "aspiration", "aspires_to", "hopes_to", "hoping_to"],
    "studies_at": ["studies at", "enrolled in", "attending", "enrolled at"],
    "studies": ["studies", "studying", "learning", "taking", "course",
                "course_enrollment", "coursework"],
    "likes": ["likes", "enjoys", "loves", "fond of", "appreciates",
              "appreciation", "enjoyed", "enjoying"],
    "dislikes": ["dislikes", "hates", "doesn't like", "avoids", "dislike",
                 "event_disinterest"],
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
    "asked_about": ["asked about", "asks about", "inquired about", "inquire_about",
                    "inquired_about", "inquires_about"],
    "concerned_about": ["concerned about", "worried about", "worried_about",
                        "worries_about"],
    # 2026-08-02 long-tail collapse (696 distinct relations over 904 edges,
    # 630 singletons — LLM-invented variants). Only synonyms where the object
    # carries the specifics.
    "drinks": ["drank_alcohol", "drinks_alcohol", "drink_preference", "beverage"],
    # (bare "attending" is already claimed by studies_at above — not repeated
    # here, since a later duplicate synonym would silently override it)
    "attends": ["attended", "attends_activity", "attending_event",
                "attends_event", "attending_activity"],
    "completed": ["completes", "finished", "completed_course",
                  "completed_activity", "completed_tasks"],
    "celebrates": ["celebrated", "celebration"],
    "cleans_up": ["cleaned_up", "cleaning_up"],
}

# Build reverse lookup for fast matching
_RELATION_REVERSE: dict[str, str] = {}
for canonical, synonyms in RELATION_SYNONYMS.items():
    _RELATION_REVERSE[canonical] = canonical
    for syn in synonyms:
        _RELATION_REVERSE[syn.lower()] = canonical


# Family-collapse patterns, applied after exact synonym lookup.  These fold
# LLM-invented per-fact variants ("asked_about_wakeup_time", "class_start_date",
# "classes_started_on") into one canonical predicate — the specifics live in
# the fact's object/value, not the relation name.  Keep this list conservative:
# only families where the collapsed form loses nothing the object doesn't carry.
_RELATION_FAMILY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^ask(s|ed)?_about_.+$"), "asked_about"),
    (re.compile(r"^inquire[sd]?_about_.+$"), "asked_about"),
    (re.compile(r"^concern(s|ed)?_about_.+$"), "concerned_about"),
    (re.compile(r"^worrie[sd]_about_.+$"), "concerned_about"),
    (re.compile(r"^class(es)?_start(s|ed)?(_(date|time|on|day))?$"), "classes_start"),
    # 2026-08-02 additions — same conservative rule: the stripped tail must be
    # per-fact specifics the object/value already carries.
    (re.compile(r"^like[sd]?_.+$"), "likes"),
    (re.compile(r"^enjoy(s|ed)?_.+$"), "likes"),
    (re.compile(r"^dislike[sd]?_.+$"), "dislikes"),
    (re.compile(r"^hate[sd]?_.+$"), "dislikes"),
    (re.compile(r"^interested_in_.+$"), "interested_in"),
    (re.compile(r"^excited_(about|for)_.+$"), "excited_about"),
    (re.compile(r"^talk(s|ed)?_about_.+$"), "talked_about"),
    # negative lookahead: mentioned_alongside is a code-level structural
    # relation (wiki_enrichment/graph_memory) and must never collapse
    (re.compile(r"^mention(s|ed)?_(?!alongside\b).+$"), "mentioned"),
    (re.compile(r"^complet(es|ed)_.+$"), "completed"),
    (re.compile(r"^attend(s|ed|ing)_.+$"), "attends"),
]


def normalize_relation(raw: str) -> str:
    """Map raw relation strings to canonical forms.

    Examples:
        "lives in" -> "lives_in"
        "resides in" -> "lives_in"
        "asked_about_wakeup_time" -> "asked_about"  (family collapse)
        "favorite_color" -> "favorite_color"  (pass-through)
    """
    raw_lower = raw.lower().strip()
    if raw_lower in _RELATION_REVERSE:
        return _RELATION_REVERSE[raw_lower]
    # Hygiene: spaces/hyphens -> underscores, drop stray punctuation, collapse repeats
    rel = re.sub(r"[\s\-]+", "_", raw_lower)
    rel = re.sub(r"[^a-z0-9_]", "", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    if not rel:  # punctuation-only input — fall back to space-joined form
        return re.sub(r"\s+", "_", raw_lower)
    if rel in _RELATION_REVERSE:
        return _RELATION_REVERSE[rel]
    for pattern, canonical in _RELATION_FAMILY_PATTERNS:
        if pattern.match(rel):
            return canonical
    return rel


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
    E.g. "my cat Whiskers" -> [("my cat", "pet")]
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
        # Corrupt existing file raises CorruptStoreError (quarantined copy kept)
        # instead of silently continuing empty and overwriting on next save.
        aliases_data = load_critical_json(self.aliases_path, "Entity alias store")
        if aliases_data is None:
            return
        count = 0
        # Use bulk_import to suppress auto-saves while rebuilding in-memory
        # alias index from the persisted file (no graph changes to flush).
        with self.graph.bulk_import():
            for entity_id, aliases in aliases_data.items():
                eid = entity_id.lower().strip()
                for alias in aliases:
                    a_lower = alias.lower().strip()
                    if a_lower:
                        self.graph.register_alias(a_lower, eid)
                        count += 1
        # Clear dirty flag — we just loaded what's already on disk.
        self.graph._dirty = False
        self.graph._modification_count = 0
        logger.info(f"[EntityResolver] Loaded {count} external aliases from {self.aliases_path}")

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

        try:
            atomic_write_json(self.aliases_path, by_entity)
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
