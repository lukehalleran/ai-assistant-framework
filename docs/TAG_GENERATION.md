# Tag Generation System

## Overview

The Tag Generation system automatically generates contextual tags for Obsidian notes (daily/weekly summaries and future .md-based memories) using LLM analysis. This improves discoverability, enables better organization, and supports future advanced filtering.

**Created:** 2026-01-22
**Module:** `utils/tag_generator.py`

## Features

- **LLM-based tag extraction**: Analyzes note content to identify relevant themes, topics, emotions, and activities
- **Consistent vocabulary**: Maintains a predefined tag vocabulary (100+ tags) organized by category
- **Smart validation**: Filters out low-quality tags and normalizes formats
- **Graceful degradation**: Falls back to heuristic tags if LLM fails
- **Future-ready**: Designed to support .md-based memories with tags as a 4th filtering stage

## Tag Categories

The system maintains 6 categories of tags:

1. **Life domains**: work, study, health, exercise, sleep, social, family, relationships, etc.
2. **Activities**: coding, programming, learning, reading, writing, debugging, gaming, etc.
3. **Emotions**: stress, anxiety, happy, motivated, tired, focused, overwhelmed, etc.
4. **Productivity**: productive, deep-work, flow-state, breakthrough, blocked, etc.
5. **Topics**: ai, programming, python, math, science, philosophy, psychology, etc.
6. **Meta**: crisis, decision, reflection, planning, insight, important, follow-up, etc.

Total vocabulary: **100+ known tags** with support for custom tags suggested by LLM.

## Integration Points

### Daily Notes Generator
- File: `utils/daily_notes_generator.py`
- Generates 3-10 tags per daily note
- Tags based on: main quest, conversations, emotional state, life events
- Example frontmatter:
```yaml
---
date: 2026-01-22
usage_intensity: 8
conversations: 16
tags: ["daily", "daemon-generated", "coding", "programming", "productive", "learning", "exercise", "health", "focused", "deep-work"]
---
```

### Weekly Notes Generator
- File: `utils/weekly_notes_generator.py`
- Generates 5-10 tags per weekly summary
- Tags based on: aggregated main quests, recurring themes, mood patterns
- Example frontmatter:
```yaml
---
week: 3
year: 2026
tags: ["weekly", "daemon-generated", "programming", "ai", "productive", "learning", "health", "exercise", "motivated", "achievement"]
---
```

### Future: .md Memory Files
- Planned migration from JSON to .md for memory storage
- Tags will serve as a 4th filtering stage alongside semantic/keyword/temporal
- Each memory chunk gets contextual tags for granular retrieval

## Configuration

### config.yaml
```yaml
tag_generation:
  enabled: true          # Enable/disable tag generation
  model: gpt-4o-mini    # LLM model for tag extraction
  max_tags: 10          # Maximum tags per note
  min_tags: 3           # Minimum tags per note
```

### app_config.py
```python
TAG_GENERATION_ENABLED: bool = True
TAG_GENERATION_MODEL: str = "gpt-4o-mini"
TAG_GENERATION_MAX_TAGS: int = 10
TAG_GENERATION_MIN_TAGS: int = 3
```

### Environment Variables
```bash
export TAG_GENERATION_ENABLED=1  # 1 or 0
```

## Usage

### Standalone Usage
```python
from utils.tag_generator import TagGenerator

# Initialize
tag_gen = TagGenerator()

# Generate tags for content
result = await tag_gen.generate_tags(
    content="Your note content here...",
    note_type="daily",  # or "weekly", "memory"
    metadata={
        'main_quest': 'Building RAG system',
        'intensity': 8,
        'conversations': 16,
        'active_hours': 6.0,
    }
)

# Access results
print(f"Tags: {result.tags}")
print(f"Count: {result.tag_count}")
print(f"Known: {result.known_tags}, Custom: {result.custom_tags}")
```

### Integrated Usage
Tag generation is automatically enabled in:
- Daily note generation (called on startup if yesterday's note missing)
- Weekly note generation (called on startup if last week complete)

No manual intervention needed - tags are generated automatically when notes are created.

## Testing

Run the test script:
```bash
python test_tag_generator.py
```

This demonstrates:
- Daily note tag generation
- Weekly note tag generation
- Tag vocabulary overview
- LLM fallback behavior

## Tag Format

Tags follow Obsidian conventions:
- Lowercase
- Hyphens instead of spaces (e.g., `deep-work`, `mental-health`)
- No special characters
- Quoted in YAML frontmatter to handle hyphens

## Examples

### Example 1: Productive Coding Day
**Content:** Deep focus session on implementing new features, debugged complex async issues, felt motivated.

**Generated tags:** `coding, programming, productive, deep-work, focused, motivated, debugging, learning`

### Example 2: Stressful Work Week
**Content:** Heavy workload, multiple deadlines, felt overwhelmed but pushed through. Skipped exercise.

**Generated tags:** `work, stress, overwhelmed, deadline, time-management, pressure, tired`

### Example 3: Wellness-Focused Week
**Content:** Established morning routine, consistent exercise, good sleep, reduced screen time.

**Generated tags:** `health, wellness, exercise, sleep, routine, balance, self-care, progress`

## Future Enhancements

1. **Tag relationships**: Build a tag hierarchy/ontology (e.g., `coding` → `programming` → `work`)
2. **Tag trends**: Track tag frequency over time to identify patterns
3. **Tag-based search**: Use tags as a fast prefilter before semantic search
4. **Custom vocabulary**: Allow users to add personal tags to the vocabulary
5. **Multi-language tags**: Support tags in multiple languages
6. **Tag suggestions**: Suggest new tags based on recent note patterns

## Architecture

### LLM Prompt Strategy
1. Provide note content + metadata (main quest, intensity, etc.)
2. Show known vocabulary to encourage consistency
3. Allow custom tags for new concepts
4. Request 5-10 relevant tags in comma-separated format

### Validation Pipeline
1. Parse LLM response (handles comma/newline/numbered formats)
2. Normalize tags (lowercase, hyphenate, remove special chars)
3. Validate quality (length, stopwords, duplicates)
4. Categorize as known vs custom
5. Enforce min/max limits

### Fallback Behavior
If LLM fails:
1. Use heuristic tags based on note type
2. Extract keywords from main quest/topic
3. Map intensity to productivity tags
4. Ensure minimum tag count

## Performance

- **Tag generation time**: ~1-2 seconds (LLM call)
- **Impact on note generation**: Minimal (runs async, non-blocking)
- **Failure mode**: Graceful (continues with fallback tags)
- **Cost**: ~$0.0001 per note (using gpt-4o-mini)

## Related Files

- `utils/tag_generator.py` - Main implementation
- `utils/daily_notes_generator.py` - Daily note integration
- `utils/weekly_notes_generator.py` - Weekly note integration
- `config/app_config.py` - Configuration constants
- `config/config.yaml` - YAML configuration
- `test_tag_generator.py` - Test script
- `docs/TAG_GENERATION.md` - This documentation
