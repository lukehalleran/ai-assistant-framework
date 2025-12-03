#!/usr/bin/env python3
"""
Test meta-conversational bonus implementation.
"""

import logging
from datetime import datetime
from memory.memory_scorer import MemoryScorer

# Enable DEBUG logging to populate debug dict
logging.basicConfig(level=logging.DEBUG)

def test_meta_conversational_bonus():
    """Test that meta-conversational queries boost episodic memories."""
    import copy
    scorer = MemoryScorer()

    now = datetime.now()

    # Create test memories with different types
    base_memories = [
        {
            'content': 'User asked about previous conversations',
            'query': 'What did we talk about yesterday?',
            'response': 'We discussed memory systems',
            'relevance_score': 0.7,
            'timestamp': now,
            'memory_type': 'EPISODIC',
            'collection': 'episodic',
            'metadata': {}
        },
        {
            'content': 'Summary of previous conversations',
            'relevance_score': 0.7,
            'timestamp': now,
            'memory_type': 'SUMMARY',
            'collection': 'summaries',
            'metadata': {}
        },
        {
            'content': 'Meta-level reflection about conversation patterns',
            'relevance_score': 0.7,
            'timestamp': now,
            'memory_type': 'META',
            'collection': 'meta',
            'metadata': {}
        },
        {
            'content': 'Semantic fact about Python programming',
            'relevance_score': 0.7,
            'timestamp': now,
            'memory_type': 'SEMANTIC',
            'collection': 'semantic',
            'metadata': {}
        }
    ]

    print("\n=== Test 1: Non-meta query (no bonus) ===")
    query_normal = "Tell me about Python"
    memories_normal = copy.deepcopy(base_memories)
    ranked_normal = scorer.rank_memories(
        memories_normal,
        current_query=query_normal,
        current_topic="programming",
        is_meta_conversational=False
    )

    print("\nRanked results (non-meta query):")
    for i, mem in enumerate(ranked_normal, 1):
        mem_type = mem.get('memory_type', 'UNKNOWN')
        score = mem.get('final_score', 0.0)
        meta_bonus = mem.get('debug', {}).get('meta_bonus', 0.0)
        print(f"  {i}. {mem_type}: score={score:.3f}, meta_bonus={meta_bonus:.2f}")

    print("\n=== Test 2: Meta-conversational query (with bonus) ===")
    query_meta = "What did we talk about yesterday?"
    memories_meta = copy.deepcopy(base_memories)
    ranked_meta = scorer.rank_memories(
        memories_meta,
        current_query=query_meta,
        current_topic="conversation_history",
        is_meta_conversational=True
    )

    print("\nRanked results (meta-conversational query):")
    for i, mem in enumerate(ranked_meta, 1):
        mem_type = mem.get('memory_type', 'UNKNOWN')
        score = mem.get('final_score', 0.0)
        meta_bonus = mem.get('debug', {}).get('meta_bonus', 0.0)
        print(f"  {i}. {mem_type}: score={score:.3f}, meta_bonus={meta_bonus:.2f}")

    # Verify bonuses
    print("\n=== Verification ===")
    episodic_mem = next((m for m in ranked_meta if m.get('memory_type') == 'EPISODIC'), None)
    summary_mem = next((m for m in ranked_meta if m.get('memory_type') == 'SUMMARY'), None)
    meta_mem = next((m for m in ranked_meta if m.get('memory_type') == 'META'), None)
    semantic_mem = next((m for m in ranked_meta if m.get('memory_type') == 'SEMANTIC'), None)

    if episodic_mem:
        episodic_bonus = episodic_mem.get('debug', {}).get('meta_bonus', 0.0)
        assert episodic_bonus == 0.15, f"Expected EPISODIC bonus 0.15, got {episodic_bonus}"
        print(f"✓ EPISODIC bonus: {episodic_bonus:.2f}")

    if summary_mem:
        summary_bonus = summary_mem.get('debug', {}).get('meta_bonus', 0.0)
        assert summary_bonus == 0.10, f"Expected SUMMARY bonus 0.10, got {summary_bonus}"
        print(f"✓ SUMMARY bonus: {summary_bonus:.2f}")

    if meta_mem:
        meta_bonus_val = meta_mem.get('debug', {}).get('meta_bonus', 0.0)
        assert meta_bonus_val == 0.12, f"Expected META bonus 0.12, got {meta_bonus_val}"
        print(f"✓ META bonus: {meta_bonus_val:.2f}")

    if semantic_mem:
        semantic_bonus = semantic_mem.get('debug', {}).get('meta_bonus', 0.0)
        assert semantic_bonus == 0.0, f"Expected SEMANTIC bonus 0.0, got {semantic_bonus}"
        print(f"✓ SEMANTIC bonus: {semantic_bonus:.2f} (no bonus)")

    # Verify non-meta query has no bonuses
    print("\n=== Non-meta query verification ===")
    for mem in ranked_normal:
        debug = mem.get('debug', {})
        meta_bonus = debug.get('meta_bonus', 0.0)
        mem_type = mem.get('memory_type', 'UNKNOWN')
        print(f"  {mem_type}: debug keys={list(debug.keys())}, meta_bonus={meta_bonus}")
        assert meta_bonus == 0.0, f"{mem_type} in non-meta query should have no bonus, got {meta_bonus}"
    print("✓ All memories in non-meta query have 0.0 meta_bonus")

    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_meta_conversational_bonus()
