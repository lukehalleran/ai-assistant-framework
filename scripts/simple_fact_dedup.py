#!/usr/bin/env python3
"""
Simple exact-match fact deduplication script.
Removes facts with identical content, keeping the oldest one.
"""

import sys
import os
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CHROMA_PATH

def simple_deduplicate(dry_run=True):
    """Remove exact duplicate facts."""
    print(f"Starting simple exact-match deduplication (dry_run={dry_run})")

    store = MultiCollectionChromaStore(CHROMA_PATH)
    facts = store.list_all('facts')

    print(f"Loaded {len(facts)} facts")

    # Group by exact content
    content_groups = defaultdict(list)
    for i, fact in enumerate(facts):
        content = fact.get('content', '').strip()
        if content:
            content_groups[content].append(fact)

    # Find duplicates
    duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}
    total_to_remove = sum(len(group) - 1 for group in duplicates.values())

    print(f"Found {len(duplicates)} duplicate groups")
    print(f"Total facts to remove: {total_to_remove}")
    print(f"Facts remaining: {len(facts) - total_to_remove}")

    if not duplicates:
        print("No duplicates found!")
        return

    print("\nTop duplicates:")
    sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
    for content, group in sorted_dups[:10]:
        print(f"  {len(group):2d}x: {content[:60]}...")

    if dry_run:
        print("\nDRY RUN - No facts removed")
        return

    # Actually remove duplicates
    print("\nRemoving duplicates...")
    removed = 0

    for content, group in duplicates.items():
        # Sort by timestamp, keep oldest
        group.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''))
        to_remove = group[1:]  # Remove all but first (oldest)

        print(f"Removing {len(to_remove)} duplicates of: {content[:50]}...")

        for fact in to_remove:
            fact_id = fact.get('id')
            if fact_id:
                try:
                    success = store.delete_fact(fact_id)
                    if success:
                        removed += 1
                        print(f"  ✓ Removed {fact_id}")
                    else:
                        print(f"  ✗ Failed to remove {fact_id}")
                except Exception as e:
                    print(f"  ✗ Error removing {fact_id}: {e}")

    print(f"\nSuccessfully removed {removed} duplicate facts!")

    # Verify results
    final_facts = store.list_all('facts')
    print(f"Facts remaining: {len(final_facts)} (was {len(facts)})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Actually remove duplicates')
    args = parser.parse_args()

    simple_deduplicate(dry_run=not args.execute)