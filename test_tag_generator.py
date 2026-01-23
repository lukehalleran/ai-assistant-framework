#!/usr/bin/env python3
"""
Quick test script for TagGenerator.

Usage:
    python test_tag_generator.py
"""

import asyncio
from utils.tag_generator import TagGenerator


async def test_tag_generator():
    """Test tag generation with sample note content."""

    # Initialize generator
    print("Initializing TagGenerator...")
    tag_gen = TagGenerator()

    # Sample daily note content
    sample_daily_content = """
## Summary
Today was a productive day focused on coding and learning. Worked on implementing
a new RAG system with hierarchical memory and spent time debugging some tricky
async issues. Also had a good workout session in the evening.

## Main Quest: Building RAG System
- Implemented tag generation for Obsidian notes
- Added LLM-based tag extraction with fallback vocabulary
- Integrated into daily and weekly note generators
- Debugged async/await pipeline issues
- All tests passing

## Life Events
- **Work**: 6 hours - productive coding session, made good progress
- **Study**: 2 hours - learned about async patterns in Python
- **Exercise**: 1 hour - gym workout, felt great
- **Sleep**: Not discussed today.

## Emotional State
Feeling focused and motivated. A bit tired in the evening but overall good energy.
Happy with the progress on the project.

## Key Decisions
- Decided to use LLM-first approach for tag generation rather than pure heuristics
- Chose to make tag generation optional with graceful degradation

## Intensity: 8/10
High intensity day with 16 conversations over 6 active hours. Complex technical work.
"""

    sample_weekly_content = """
## Week at a Glance
This week was highly productive with consistent focus on the RAG system project.
Made significant progress on memory architecture and tag generation. Maintained
good work-life balance with regular exercise.

## Main Quests This Week
- **RAG System Development** (5 days): Completed - implemented tag generation,
  improved memory retrieval, fixed async bugs
- **Learning Python Async** (3 days): Ongoing - made good progress understanding
  patterns
- **Health Focus** (4 days): Completed - consistent exercise routine

## Life Events Summary
- **Work**: 5 days mentioned, ~30 hours total, very productive week
- **Study**: 4 days, ~10 hours learning async patterns and RAG techniques
- **Exercise**: 4 days, consistent gym routine established
- **Sleep**: Good quality mentioned on 3 days

## Recurring Themes
- Deep focus on coding and system architecture
- Continuous learning and skill development
- Balancing productivity with wellness
- Problem-solving complex technical challenges

## Mood Arc
Started energized Monday, maintained good focus throughout week. Slight fatigue
Thursday but recovered. Overall optimistic and motivated.

## Intensity: 7/10
Consistently productive week with good pacing. High technical complexity but
well-managed workload.
"""

    # Test daily note tag generation
    print("\n" + "="*80)
    print("Testing DAILY note tag generation:")
    print("="*80)

    daily_metadata = {
        'main_quest': 'Building RAG System',
        'intensity': 8,
        'conversations': 16,
        'active_hours': 6.0,
    }

    daily_result = await tag_gen.generate_tags(
        sample_daily_content,
        note_type="daily",
        metadata=daily_metadata
    )

    print(f"\nGenerated {daily_result.tag_count} tags:")
    print(f"  Tags: {', '.join(daily_result.tags)}")
    print(f"  Known: {daily_result.known_tags}, Custom: {daily_result.custom_tags}")
    if daily_result.skipped_tags:
        print(f"  Skipped: {', '.join(daily_result.skipped_tags)}")
    if daily_result.error:
        print(f"  Error: {daily_result.error}")

    # Test weekly note tag generation
    print("\n" + "="*80)
    print("Testing WEEKLY note tag generation:")
    print("="*80)

    weekly_metadata = {
        'main_topic': 'Week 3 summary',
        'intensity': 7,
        'conversations': 85,
        'duration_hours': 28.5,
        'main_quests': 'RAG System Development, Learning Python Async, Health Focus',
    }

    weekly_result = await tag_gen.generate_tags(
        sample_weekly_content,
        note_type="weekly",
        metadata=weekly_metadata
    )

    print(f"\nGenerated {weekly_result.tag_count} tags:")
    print(f"  Tags: {', '.join(weekly_result.tags)}")
    print(f"  Known: {weekly_result.known_tags}, Custom: {weekly_result.custom_tags}")
    if weekly_result.skipped_tags:
        print(f"  Skipped: {', '.join(weekly_result.skipped_tags)}")
    if weekly_result.error:
        print(f"  Error: {weekly_result.error}")

    # Show tag vocabulary
    print("\n" + "="*80)
    print("Tag Vocabulary Stats:")
    print("="*80)
    vocab = tag_gen.get_tag_categories()
    all_tags = tag_gen.get_all_known_tags()
    print(f"\nTotal known tags: {len(all_tags)}")
    print("\nTags by category:")
    for category, tags in vocab.items():
        print(f"  {category.capitalize()}: {len(tags)} tags")
        print(f"    Examples: {', '.join(sorted(tags)[:5])}")


if __name__ == "__main__":
    asyncio.run(test_tag_generator())
