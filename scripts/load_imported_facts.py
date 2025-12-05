#!/usr/bin/env python3
"""
load_imported_facts.py

Load previously imported ChatGPT facts into Daemon's user profile system.

Usage:
    python load_imported_facts.py --input data/imported_chatgpt_facts.json
"""

import json
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.user_profile import UserProfile


def load_facts(input_path: str, profile_path: str = None, dry_run: bool = False) -> dict:
    """
    Load imported facts into user profile.

    Args:
        input_path: Path to imported facts JSON file
        profile_path: Optional custom path for user profile (default: data/user_profile.json)
        dry_run: If True, show what would be imported without actually importing

    Returns:
        Dictionary with import statistics
    """

    # Load imported facts
    print(f"Loading facts from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    facts = data.get("facts", [])
    metadata = data.get("metadata", {})

    print(f"Found {len(facts)} facts to import")
    print(f"Source: {metadata.get('source', 'unknown')}")
    print(f"Original memories: {metadata.get('total_memories', 0)}")

    if dry_run:
        print("\n[DRY RUN] Would import the following facts:")
        category_preview = {}
        for fact in facts[:10]:  # Show first 10
            cat = fact.get('category', 'unknown')
            category_preview[cat] = category_preview.get(cat, 0) + 1
            print(f"  - [{cat}] {fact.get('relation', '???')}: {fact.get('object', '???')[:60]}...")

        if len(facts) > 10:
            print(f"  ... and {len(facts) - 10} more")

        print(f"\nCategory breakdown:")
        for cat, count in sorted(metadata.get('category_counts', {}).items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        return {
            "dry_run": True,
            "total_facts": len(facts),
            "would_add": len(facts)
        }

    # Initialize user profile
    print(f"\nLoading user profile...")
    profile = UserProfile(profile_path=profile_path)

    # Get current fact count before import
    before_count = profile.get_fact_count()
    print(f"Current profile has {before_count} facts")

    # Import facts
    print(f"\nImporting {len(facts)} facts...")
    added = profile.add_facts_batch(facts)

    # Get new fact count
    after_count = profile.get_fact_count()

    print(f"\n✓ Successfully added {added} facts")
    print(f"  Profile now has {after_count} total facts")
    print(f"  ({after_count - before_count} new entries in raw log)")

    # Show category breakdown
    print(f"\nCategory breakdown:")
    for cat, count in sorted(metadata.get('category_counts', {}).items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return {
        "dry_run": False,
        "total_facts": len(facts),
        "facts_added": added,
        "before_count": before_count,
        "after_count": after_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load imported ChatGPT facts into Daemon user profile"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/imported_chatgpt_facts.json",
        help="Path to imported facts JSON file"
    )
    parser.add_argument(
        "--profile", "-p",
        default=None,
        help="Custom path for user profile (default: data/user_profile.json)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be imported without actually importing"
    )

    args = parser.parse_args()

    try:
        stats = load_facts(args.input, args.profile, args.dry_run)

        if not stats["dry_run"]:
            print(f"\n✓ Import complete!")
            print(f"  Added {stats['facts_added']} facts to user profile")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
