#!/usr/bin/env python3
"""Simple test of cross-section deduplication logic."""

# Simulate the deduplication logic
def test_cross_dedup():
    # Test data with duplicates across sections
    context = {
        "recent_conversations": [
            {"content": "ICE is conducting operations in Logan Square with tear gas", "timestamp": "2025-10-08"},
            {"content": "The workout was great today", "timestamp": "2025-11-16"}
        ],
        "summaries": [
            {"content": "ICE is conducting operations in Logan Square with tear gas", "timestamp": "2025-10-08"},  # Exact duplicate
            {"content": "User discussed gym routine and bench press", "timestamp": "2025-11-15"}
        ],
        "semantic_summaries": [
            {"content": "ICE is conducting operations in Logan Square with tear gas and flash grenades targeting children and families in the middle of the night", "timestamp": "2025-10-08"},  # Near-duplicate (first 200 chars match)
            {"content": "Discussion about protein intake and recovery", "timestamp": "2025-11-14"}
        ],
        "memories": [
            {"content": "The workout was great today", "timestamp": "2025-11-16"},  # Exact duplicate
            {"content": "Bench press progress: 135 lbs x 5", "timestamp": "2025-10-15"}
        ]
    }

    print("BEFORE cross-section dedup:")
    print(f"  recent_conversations: {len(context['recent_conversations'])} items")
    print(f"  summaries: {len(context['summaries'])} items")
    print(f"  semantic_summaries: {len(context['semantic_summaries'])} items")
    print(f"  memories: {len(context['memories'])} items")
    total_before = sum(len(v) for v in context.values())
    print(f"  TOTAL: {total_before} items\n")

    # Apply cross-section deduplication
    seen_content = set()
    cross_dedup_sections = [
        "recent_conversations", "memories",
        "summaries", "recent_summaries", "semantic_summaries",
        "reflections", "recent_reflections", "semantic_reflections"
    ]

    for section in cross_dedup_sections:
        items = context.get(section, [])
        if not items or not isinstance(items, list):
            continue

        deduplicated = []
        for item in items:
            # Extract content for dedup check
            if isinstance(item, dict):
                content = item.get("content", "")
                if not content:
                    content = str(item.get("response", "") + item.get("query", ""))
            else:
                content = str(item)

            # Create a normalized key for dedup (first 200 chars, lowercased, stripped)
            dedup_key = content.strip().lower()[:200]

            if dedup_key and dedup_key not in seen_content:
                seen_content.add(dedup_key)
                deduplicated.append(item)
            else:
                if dedup_key:
                    print(f"  SKIPPED DUPLICATE in {section}: {content[:60]}...")

        original_count = len(items)
        if len(deduplicated) < original_count:
            print(f"  {section}: {original_count} -> {len(deduplicated)} items (removed {original_count - len(deduplicated)})")
            context[section] = deduplicated

    print("\nAFTER cross-section dedup:")
    print(f"  recent_conversations: {len(context['recent_conversations'])} items")
    print(f"  summaries: {len(context['summaries'])} items")
    print(f"  semantic_summaries: {len(context['semantic_summaries'])} items")
    print(f"  memories: {len(context['memories'])} items")
    total_after = sum(len(v) for v in context.values())
    print(f"  TOTAL: {total_after} items")
    print(f"\nRemoved {total_before - total_after} duplicates")

    print("\nContent that survived deduplication:")
    for section in ["recent_conversations", "summaries", "semantic_summaries", "memories"]:
        if context.get(section):
            print(f"\n{section}:")
            for item in context[section]:
                content = item.get("content", "")
                print(f"  - {content[:80]}")

if __name__ == "__main__":
    test_cross_dedup()
