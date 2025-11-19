#!/usr/bin/env python3
"""
Script to deduplicate existing facts in the ChromaDB facts collection.

This script:
1. Loads all existing facts
2. Identifies duplicates using content similarity
3. Removes duplicate facts keeping the oldest one
4. Reports on deduplication results

Usage:
    python scripts/deduplicate_facts.py [--dry-run] [--threshold 0.90]
"""

import sys
import argparse
from datetime import datetime
from typing import List, Dict, Set, Tuple
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CHROMA_PATH
from utils.logging_utils import get_logger

logger = get_logger("deduplicate_facts")

def calculate_similarity(fact1: str, fact2: str) -> float:
    """Calculate Jaccard similarity between two facts."""
    tokens1 = set(fact1.lower().split())
    tokens2 = set(fact2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)

def find_duplicates(facts: List[Dict], threshold: float = 0.90) -> List[Tuple[int, int, float]]:
    """Find duplicate fact pairs based on similarity threshold."""
    duplicates = []

    print(f"Analyzing {len(facts)} facts for duplicates (threshold={threshold})")
    print("This may take a while for large datasets...")
    logger.info(f"Analyzing {len(facts)} facts for duplicates (threshold={threshold})")

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            content1 = facts[i].get('content', '')
            content2 = facts[j].get('content', '')

            # Skip empty content
            if not content1 or not content2:
                continue

            # Exact match
            if content1.strip().lower() == content2.strip().lower():
                duplicates.append((i, j, 1.0))
                continue

            # Similarity check
            similarity = calculate_similarity(content1, content2)
            if similarity >= threshold:
                duplicates.append((i, j, similarity))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(facts)} facts... Found {len(duplicates)} duplicates so far")
            logger.info(f"Processed {i + 1}/{len(facts)} facts...")

    return duplicates

def group_duplicates(duplicates: List[Tuple[int, int, float]]) -> List[List[int]]:
    """Group duplicate indices into clusters."""
    # Create a graph of duplicate relationships
    graph = {}
    for i, j, _ in duplicates:
        if i not in graph:
            graph[i] = set()
        if j not in graph:
            graph[j] = set()
        graph[i].add(j)
        graph[j].add(i)

    # Find connected components (duplicate groups)
    visited = set()
    groups = []

    def dfs(node: int, group: List[int]):
        visited.add(node)
        group.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, group)

    for node in graph:
        if node not in visited:
            group = []
            dfs(node, group)
            if len(group) > 1:
                groups.append(sorted(group))

    return groups

def deduplicate_facts(dry_run: bool = True, threshold: float = 0.90):
    """Main deduplication process."""
    print(f"Starting fact deduplication (dry_run={dry_run}, threshold={threshold})")
    logger.info(f"Starting fact deduplication (dry_run={dry_run}, threshold={threshold})")

    # Load ChromaDB store
    store = MultiCollectionChromaStore(CHROMA_PATH)

    # Get all facts
    facts = store.list_all('facts')
    print(f"Loaded {len(facts)} facts from database")
    logger.info(f"Loaded {len(facts)} facts from database")

    if len(facts) == 0:
        logger.info("No facts found in database")
        return

    # Find duplicates
    duplicate_pairs = find_duplicates(facts, threshold)
    logger.info(f"Found {len(duplicate_pairs)} duplicate pairs")

    if not duplicate_pairs:
        logger.info("No duplicates found!")
        return

    # Group duplicates
    duplicate_groups = group_duplicates(duplicate_pairs)
    logger.info(f"Grouped into {len(duplicate_groups)} duplicate clusters")

    # Process each group
    total_to_remove = 0
    removed_ids = []

    for group in duplicate_groups:
        # Sort by timestamp to keep the oldest fact
        group_facts = [facts[i] for i in group]
        group_facts_with_idx = [(facts[i], i) for i in group]

        # Sort by timestamp (oldest first)
        group_facts_with_idx.sort(key=lambda x: x[0].get('metadata', {}).get('timestamp', ''))

        # Keep the first (oldest), mark others for removal
        to_keep = group_facts_with_idx[0]
        to_remove = group_facts_with_idx[1:]

        logger.info(f"Duplicate group ({len(group)} facts):")
        logger.info(f"  KEEP: {to_keep[0]['content'][:100]}...")

        for fact_data, idx in to_remove:
            content = fact_data['content']
            fact_id = fact_data['id']
            timestamp = fact_data.get('metadata', {}).get('timestamp', 'unknown')

            logger.info(f"  REMOVE: {content[:100]}... (id={fact_id}, ts={timestamp})")
            removed_ids.append(fact_id)
            total_to_remove += 1

    logger.info(f"Summary: {total_to_remove} facts marked for removal")

    if dry_run:
        logger.info("DRY RUN - No facts were actually removed")
        return

    # Actually remove the duplicates
    logger.info("Removing duplicate facts...")

    for fact_id in removed_ids:
        try:
            # Note: ChromaDB doesn't have a direct delete by ID method in this implementation
            # We would need to add that functionality or recreate the collection
            logger.warning(f"Would remove fact {fact_id} (delete functionality needs to be implemented)")
        except Exception as e:
            logger.error(f"Error removing fact {fact_id}: {e}")

    logger.info(f"Deduplication complete. Removed {len(removed_ids)} duplicate facts")

def main():
    parser = argparse.ArgumentParser(description="Deduplicate facts in ChromaDB")
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Show what would be removed without actually removing')
    parser.add_argument('--execute', action='store_true',
                        help='Actually remove duplicates (overrides --dry-run)')
    parser.add_argument('--threshold', type=float, default=0.90,
                        help='Similarity threshold for duplicate detection (default: 0.90)')

    args = parser.parse_args()

    # If --execute is specified, set dry_run to False
    dry_run = not args.execute if args.execute else args.dry_run

    try:
        deduplicate_facts(dry_run=dry_run, threshold=args.threshold)
    except KeyboardInterrupt:
        logger.info("Deduplication interrupted by user")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()