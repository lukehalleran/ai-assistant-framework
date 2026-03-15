#!/usr/bin/env python3
"""
Migrate existing ChromaDB facts into the knowledge graph.

Reads all facts from the 'facts' collection, parses S-R-O triples from
the text (format: "subject | relation | object"), resolves entities,
and populates the graph.

Applies quality filters:
  1. Skip ephemeral relations (feelings, moods, work status, etc.)
  2. Skip garbage subjects (pronouns, meta-phrases)
  3. Skip low-quality objects (booleans, single chars, very short)

Usage:
    python scripts/migrate_facts_to_graph.py [--dry-run]
"""

import argparse
import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import (
    CHROMA_PATH,
    KNOWLEDGE_GRAPH_PERSIST_PATH,
    KNOWLEDGE_GRAPH_ALIASES_PATH,
    KNOWLEDGE_GRAPH_MIN_CONFIDENCE,
    PROFILE_EPHEMERAL_RELATIONS,
)
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge
from memory.entity_resolver import EntityResolver, normalize_relation
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

logger = get_logger("migrate_facts_to_graph")

# ---------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------

# 1. Ephemeral relations — temporal state, not durable knowledge
_EPHEMERAL_RELATIONS: frozenset[str] = frozenset(
    r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS
)
# Extra relations to skip that aren't in the profile list but are noisy
_EXTRA_SKIP_RELATIONS: frozenset[str] = frozenset({
    "feeling", "feels", "mood", "inquiry", "testing", "tidied",
    "helped", "thanks", "got_it", "cannot_get_it", "hoping",
    "hopes", "crying_feeling", "diarrhea", "saw_sleep_doctor",
    "stm", "blanks", "comments", "lines", "files", "depth",
    "week", "time", "worked", "working", "food_eaten",
    "took_nap", "internet", "got", "said",
})

def _is_ephemeral_relation(rel: str) -> bool:
    """Check if a relation is ephemeral / transient."""
    r = rel.lower().strip().replace(" ", "_")
    if r in _EPHEMERAL_RELATIONS or r in _EXTRA_SKIP_RELATIONS:
        return True
    # Catch prefixed variants: current_*, recent_*, last_*
    if r.startswith(("current_", "recent_", "last_")):
        return True
    return False


# 2. Garbage subjects — pronouns, meta-phrases, error artifacts
_GARBAGE_SUBJECTS: frozenset[str] = frozenset({
    "who", "he", "him", "she", "her", "it", "they", "them",
    "said", "your", "my", "we", "you", "i", "me",
    "relevant memories", "last response", "error", "error_message",
    "pool session",
})

def _is_garbage_subject(subj: str) -> bool:
    s = subj.lower().strip()
    if s in _GARBAGE_SUBJECTS:
        return True
    # Skip subjects that are file paths or error artifacts
    if s.startswith(("/", "error")) or ".docx" in s or ".json" in s:
        return True
    return False


# 3. Low-quality objects — booleans, very short, purely numeric
_BOOLEAN_OBJECTS: frozenset[str] = frozenset({
    "true", "false", "yes", "no", "none", "null", "n/a",
})

def _is_low_quality_object(obj: str) -> bool:
    o = obj.lower().strip()
    if o in _BOOLEAN_OBJECTS:
        return True
    # Too short (single char or empty)
    if len(o) < 2:
        return True
    # Purely numeric (timestamps, counts, etc.)
    if re.match(r"^[\d.]+$", o):
        return True
    return False


# ---------------------------------------------------------------
# Migration
# ---------------------------------------------------------------

def migrate_facts(dry_run: bool = False):
    """Pull all existing S-R-O facts from ChromaDB and insert into graph."""

    # Initialize ChromaDB store
    chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

    # Initialize graph
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph_memory=graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    # Query all facts (use a broad query to get everything)
    all_facts = chroma_store.query_collection("facts", query_text="user", n_results=5000)

    print(f"Found {len(all_facts)} facts in ChromaDB\n")

    added = 0
    skipped = 0
    skip_reasons: dict[str, int] = {
        "bad_format": 0,
        "empty_field": 0,
        "low_confidence": 0,
        "ephemeral_relation": 0,
        "garbage_subject": 0,
        "low_quality_object": 0,
    }

    for fact in all_facts:
        content = fact.get("content", "")
        metadata = fact.get("metadata", {})
        doc_id = fact.get("id", "")

        # Parse S-R-O from "subject | relation | object" format
        parts = content.split("|")
        if len(parts) != 3:
            skipped += 1
            skip_reasons["bad_format"] += 1
            continue

        subj = parts[0].strip()
        rel = parts[1].strip()
        obj = parts[2].strip()

        if not subj or not rel or not obj:
            skipped += 1
            skip_reasons["empty_field"] += 1
            continue

        confidence = float(metadata.get("confidence", 0.6))
        if confidence < KNOWLEDGE_GRAPH_MIN_CONFIDENCE:
            skipped += 1
            skip_reasons["low_confidence"] += 1
            continue

        # --- Quality filters ---
        canon_rel = normalize_relation(rel)

        if _is_ephemeral_relation(canon_rel) or _is_ephemeral_relation(rel):
            skipped += 1
            skip_reasons["ephemeral_relation"] += 1
            continue

        if _is_garbage_subject(subj):
            skipped += 1
            skip_reasons["garbage_subject"] += 1
            continue

        if _is_low_quality_object(obj):
            skipped += 1
            skip_reasons["low_quality_object"] += 1
            continue

        # --- Passed all filters ---
        entity_type = metadata.get("entity_type", "")
        subj_type = "person" if subj.lower() == "user" else (entity_type or "other")
        subj_display = "User" if subj.lower() == "user" else subj

        if dry_run:
            print(f"  [DRY RUN] {subj_display} --{canon_rel}--> {obj}")
            added += 1
            continue

        # Resolve or create entities
        subj_id = resolver.resolve_or_create(subj, entity_type=subj_type, display_name=subj_display)
        obj_id = resolver.resolve_or_create(obj, display_name=obj)

        # Add relation
        graph.add_relation(
            GraphEdge(source_id=subj_id, relation=canon_rel, target_id=obj_id),
            fact_id=doc_id,
        )
        added += 1

    if not dry_run:
        graph.save()
        resolver.save_external_aliases()

    print(f"\n{'='*60}")
    print(
        f"Migration {'(DRY RUN) ' if dry_run else ''}complete: "
        f"{added} facts added, {skipped} skipped"
    )
    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"\nSkip breakdown:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {reason}: {count}")

    return added, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate ChromaDB facts to knowledge graph")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    migrate_facts(dry_run=args.dry_run)
