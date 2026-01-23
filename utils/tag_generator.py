# utils/tag_generator.py
"""
TagGenerator - Generate contextual tags for Obsidian notes using LLM analysis.

Module Contract:
- Purpose: Analyze note content and generate relevant Obsidian tags for better discoverability
- Inputs:
  - generate_tags(content: str, note_type: str, metadata: dict) -> List[str]: Generate tags
- Outputs:
  - List of 5-10 relevant tag strings (e.g., ["work", "coding", "productivity"])
- Behavior:
  - Uses LLM to analyze content and extract thematic tags
  - Validates against predefined tag vocabulary for consistency
  - Allows custom tags when LLM suggests relevant new ones
  - Filters duplicates and normalizes tag format
- Dependencies:
  - models.model_manager (LLM generation)
  - config.app_config (tag vocabulary and settings)
- Future-ready:
  - Designed to support .md-based memories with tags as 4th filtering stage
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Predefined tag vocabulary organized by category
# This ensures consistency across notes and provides LLM with a reference
TAG_VOCABULARY = {
    # Life domains
    'life': {
        'work', 'study', 'health', 'exercise', 'sleep', 'social', 'family',
        'relationships', 'dating', 'friends', 'hobbies', 'finances', 'career',
        'education', 'wellness', 'mental-health', 'physical-health'
    },

    # Activities & skills
    'activities': {
        'coding', 'programming', 'learning', 'reading', 'writing', 'research',
        'debugging', 'building', 'designing', 'planning', 'gaming', 'travel',
        'cooking', 'art', 'music', 'sports', 'meditation', 'journaling'
    },

    # Emotional states & moods
    'emotions': {
        'stress', 'anxiety', 'happy', 'sad', 'frustrated', 'excited', 'calm',
        'motivated', 'tired', 'energized', 'confused', 'confident', 'worried',
        'optimistic', 'pessimistic', 'grateful', 'angry', 'peaceful', 'lonely',
        'content', 'overwhelmed', 'focused', 'distracted'
    },

    # Productivity & states
    'productivity': {
        'productive', 'unproductive', 'deep-work', 'flow-state', 'procrastinating',
        'efficient', 'struggling', 'breakthrough', 'blocked', 'progress',
        'achievement', 'setback', 'milestone', 'deadline', 'time-management'
    },

    # Topics & domains (technical/intellectual)
    'topics': {
        'ai', 'machine-learning', 'programming', 'python', 'javascript', 'web-dev',
        'data-science', 'algorithms', 'math', 'science', 'physics', 'biology',
        'history', 'philosophy', 'psychology', 'economics', 'politics', 'technology',
        'linguistics', 'literature', 'art-history', 'music-theory', 'engineering'
    },

    # Meta/system tags
    'meta': {
        'crisis', 'decision', 'reflection', 'planning', 'goal-setting', 'review',
        'brainstorming', 'problem-solving', 'question', 'insight', 'realization',
        'important', 'follow-up', 'unresolved', 'completed', 'archived'
    },

    # Life events
    'events': {
        'appointment', 'meeting', 'interview', 'presentation', 'deadline',
        'celebration', 'milestone', 'trip', 'move', 'change', 'transition',
        'accident', 'emergency', 'illness', 'recovery', 'achievement'
    }
}

# Flatten vocabulary for quick lookup
ALL_KNOWN_TAGS = set()
for category in TAG_VOCABULARY.values():
    ALL_KNOWN_TAGS.update(category)


@dataclass
class TagGenerationResult:
    """Result of tag generation."""
    tags: List[str] = field(default_factory=list)
    tag_count: int = 0
    known_tags: int = 0  # Count of tags from vocabulary
    custom_tags: int = 0  # Count of LLM-suggested new tags
    skipped_tags: List[str] = field(default_factory=list)  # Invalid/filtered tags
    error: Optional[str] = None


# LLM prompt for tag generation
TAG_GENERATION_PROMPT = '''You are a tag extraction expert for Obsidian note-taking system.

CONTENT TO TAG:
{content}

METADATA:
- Note type: {note_type}
- Main topic: {main_topic}
- Intensity: {intensity}/10
{additional_context}

KNOWN TAG VOCABULARY (prefer these for consistency):
{tag_vocabulary}

INSTRUCTIONS:
1. Analyze the content and extract 5-10 relevant tags
2. Prefer tags from the known vocabulary above
3. You can suggest new tags if they're clearly relevant and not covered by existing tags
4. Focus on:
   - Life domains discussed (work, study, health, social, etc.)
   - Activities mentioned (coding, learning, exercise, etc.)
   - Emotional states (stress, happy, focused, etc.)
   - Main topics/subjects (ai, programming, philosophy, etc.)
   - Meta aspects (crisis, decision, reflection, etc.)
5. Use lowercase, hyphenated format (e.g., "deep-work", "mental-health")
6. DO NOT include generic tags like "daily", "daemon-generated" (system adds those)
7. Return ONLY a comma-separated list of tags, nothing else

EXAMPLES:
- For coding session: coding, programming, python, deep-work, productive, learning
- For stressful work day: work, stress, overwhelmed, deadline, time-management
- For casual chat: social, reflection, philosophy, question
- For health discussion: health, exercise, wellness, sleep, energy

OUTPUT (comma-separated tags only):'''


class TagGenerator:
    """Generate contextual tags for Obsidian notes."""

    def __init__(self, model_manager=None, tag_model: str = "gpt-4o-mini",
                 max_tags: int = 10, min_tags: int = 3):
        """
        Initialize TagGenerator.

        Args:
            model_manager: ModelManager instance (lazy-loaded if None)
            tag_model: LLM model to use for tag generation
            max_tags: Maximum number of tags to generate
            min_tags: Minimum number of tags to generate
        """
        self._model_manager = model_manager
        self.tag_model = tag_model
        self.max_tags = max_tags
        self.min_tags = min_tags

        # Load config overrides if available
        try:
            from config.app_config import (
                TAG_GENERATION_MODEL,
                TAG_GENERATION_MAX_TAGS,
                TAG_GENERATION_MIN_TAGS,
            )
            self.tag_model = TAG_GENERATION_MODEL
            self.max_tags = TAG_GENERATION_MAX_TAGS
            self.min_tags = TAG_GENERATION_MIN_TAGS
        except ImportError:
            pass  # Use defaults

        logger.debug(f"[TagGenerator] Initialized: model={self.tag_model}, max_tags={self.max_tags}")

    @property
    def model_manager(self):
        """Lazy-load ModelManager."""
        if self._model_manager is None:
            try:
                from models.model_manager import ModelManager
                self._model_manager = ModelManager()
                logger.debug("[TagGenerator] ModelManager lazy-loaded")
            except Exception as e:
                logger.error(f"[TagGenerator] Failed to load ModelManager: {e}")
                raise
        return self._model_manager

    def _format_tag_vocabulary(self, max_per_category: int = 10) -> str:
        """Format tag vocabulary for LLM prompt (limit length)."""
        formatted = []
        for category, tags in TAG_VOCABULARY.items():
            # Take first N tags from each category to keep prompt reasonable
            sample_tags = sorted(tags)[:max_per_category]
            formatted.append(f"  {category.capitalize()}: {', '.join(sample_tags)}")
        return '\n'.join(formatted)

    def _normalize_tag(self, tag: str) -> str:
        """
        Normalize tag to Obsidian format.

        - Lowercase
        - Replace spaces/underscores with hyphens
        - Remove special characters
        - Strip # prefix if present
        """
        normalized = tag.strip().lower()
        normalized = normalized.lstrip('#')  # Remove # if present
        normalized = re.sub(r'[_\s]+', '-', normalized)  # Spaces/underscores to hyphens
        normalized = re.sub(r'[^a-z0-9-]', '', normalized)  # Remove special chars
        normalized = re.sub(r'-+', '-', normalized)  # Collapse multiple hyphens
        normalized = normalized.strip('-')  # Trim leading/trailing hyphens
        return normalized

    def _validate_tag(self, tag: str) -> bool:
        """
        Validate tag quality.

        Returns True if tag is valid:
        - At least 2 characters
        - Not a stopword
        - Not a generic word
        """
        if len(tag) < 2:
            return False

        # Stopwords that shouldn't be tags
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'note', 'today',
            'day', 'week', 'thing', 'stuff', 'etc', 'something', 'anything'
        }

        if tag in stopwords:
            return False

        return True

    def _parse_llm_tags(self, llm_response: str) -> List[str]:
        """Parse comma-separated tags from LLM response."""
        # Handle both comma-separated and newline-separated lists
        # Also handle numbered lists like "1. coding\n2. learning"

        # Remove common prefixes/formatting
        response = llm_response.strip()
        response = re.sub(r'^\s*tags?:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^\s*output:\s*', '', response, flags=re.IGNORECASE)

        # Try comma-separated first
        if ',' in response:
            raw_tags = response.split(',')
        else:
            # Try newline-separated or numbered list
            lines = response.split('\n')
            raw_tags = []
            for line in lines:
                # Remove numbering like "1. " or "- "
                cleaned = re.sub(r'^\s*[\d\-\*]+[\.\)]\s*', '', line)
                if cleaned.strip():
                    raw_tags.append(cleaned)

        # Normalize and validate
        tags = []
        for tag in raw_tags:
            normalized = self._normalize_tag(tag)
            if normalized and self._validate_tag(normalized):
                tags.append(normalized)

        return tags

    async def generate_tags(self, content: str, note_type: str = "daily",
                           metadata: Optional[Dict[str, Any]] = None) -> TagGenerationResult:
        """
        Generate tags for note content.

        Args:
            content: The note content to analyze
            note_type: Type of note (daily, weekly, memory, etc.)
            metadata: Optional metadata dict with keys like:
                - main_topic/main_quest: Primary subject
                - intensity: 1-10 score
                - conversations: Number of exchanges
                - emotional_state: Mood description
                - Any other contextual info

        Returns:
            TagGenerationResult with generated tags and stats
        """
        result = TagGenerationResult()

        if not content or len(content.strip()) < 50:
            result.error = "Content too short for tag generation"
            logger.warning(f"[TagGenerator] {result.error}")
            return result

        # Extract metadata
        metadata = metadata or {}
        main_topic = metadata.get('main_quest') or metadata.get('main_topic', 'General conversation')
        intensity = metadata.get('intensity', 5)

        # Build additional context from metadata
        additional_context_parts = []
        if 'conversations' in metadata:
            additional_context_parts.append(f"- Conversations: {metadata['conversations']}")
        if 'emotional_state' in metadata:
            additional_context_parts.append(f"- Emotional state: {metadata['emotional_state']}")
        if 'duration_hours' in metadata or 'active_hours' in metadata:
            hours = metadata.get('active_hours') or metadata.get('duration_hours', 0)
            additional_context_parts.append(f"- Active duration: {hours:.1f} hours")

        additional_context = '\n'.join(additional_context_parts) if additional_context_parts else '- No additional context'

        # Truncate content to keep prompt reasonable (take beginning and end)
        max_content_chars = 2000
        if len(content) > max_content_chars:
            # Take first 1500 chars and last 500 chars
            content = content[:1500] + "\n\n[...]\n\n" + content[-500:]

        # Format tag vocabulary for prompt
        vocab_str = self._format_tag_vocabulary()

        # Build prompt
        prompt = TAG_GENERATION_PROMPT.format(
            content=content,
            note_type=note_type,
            main_topic=main_topic,
            intensity=intensity,
            additional_context=additional_context,
            tag_vocabulary=vocab_str,
        )

        # Call LLM with fallback models
        fallback_models = [
            "gpt-4o-mini",
            "deepseek-v3.1",
            "gpt-4o",
            "sonnet-4.5",
        ]
        models_to_try = [self.tag_model] + [m for m in fallback_models if m != self.tag_model]

        llm_response = None
        last_error = None

        for model in models_to_try:
            try:
                logger.debug(f"[TagGenerator] Generating tags using {model}")
                llm_response = await self.model_manager.generate_once(
                    prompt,
                    max_tokens=100,
                    model_name=model,
                    temperature=0.3,  # Some creativity but mostly consistent
                )

                # Validate response
                if llm_response and not llm_response.startswith("["):
                    # Success
                    logger.debug(f"[TagGenerator] Got response from {model}")
                    break
                else:
                    last_error = f"Model {model} returned error or empty response"
                    llm_response = None
                    continue

            except Exception as e:
                logger.warning(f"[TagGenerator] Model {model} failed: {e}")
                last_error = str(e)
                llm_response = None
                continue

        # If all models failed, return fallback tags based on note type
        if not llm_response:
            result.error = f"LLM tag generation failed: {last_error}"
            logger.warning(f"[TagGenerator] {result.error}, using fallback tags")
            result.tags = self._generate_fallback_tags(note_type, metadata)
            result.tag_count = len(result.tags)
            return result

        # Parse tags from LLM response
        raw_tags = self._parse_llm_tags(llm_response)

        # Categorize tags: known vs custom
        final_tags = []
        for tag in raw_tags:
            if tag in ALL_KNOWN_TAGS:
                final_tags.append(tag)
                result.known_tags += 1
            elif len(tag) >= 3:  # Accept custom tags if reasonable length
                final_tags.append(tag)
                result.custom_tags += 1
            else:
                result.skipped_tags.append(tag)

        # Ensure we have min/max tags
        if len(final_tags) < self.min_tags:
            # Add fallback tags to meet minimum
            fallback = self._generate_fallback_tags(note_type, metadata)
            for tag in fallback:
                if tag not in final_tags and len(final_tags) < self.min_tags:
                    final_tags.append(tag)
                    result.known_tags += 1

        # Limit to max_tags
        final_tags = final_tags[:self.max_tags]

        result.tags = final_tags
        result.tag_count = len(final_tags)

        logger.info(f"[TagGenerator] Generated {result.tag_count} tags: {result.known_tags} known, {result.custom_tags} custom")
        logger.debug(f"[TagGenerator] Tags: {', '.join(final_tags)}")

        return result

    def _generate_fallback_tags(self, note_type: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Generate fallback tags when LLM fails.

        Uses simple heuristics based on note type and metadata.
        """
        tags = []

        # Always include note type
        if note_type == "daily":
            tags.append("reflection")
        elif note_type == "weekly":
            tags.append("review")

        # Intensity-based tags
        intensity = metadata.get('intensity', 5)
        if intensity >= 7:
            tags.append("productive")
        elif intensity <= 3:
            tags.append("quiet")

        # Check main topic/quest for keywords
        main_topic = str(metadata.get('main_quest', '') or metadata.get('main_topic', '')).lower()

        # Map common keywords to tags
        keyword_map = {
            'work': 'work',
            'job': 'work',
            'code': 'coding',
            'coding': 'coding',
            'programming': 'programming',
            'study': 'study',
            'learning': 'learning',
            'learn': 'learning',
            'exercise': 'exercise',
            'workout': 'exercise',
            'health': 'health',
            'sleep': 'sleep',
            'stress': 'stress',
            'anxious': 'anxiety',
            'anxiety': 'anxiety',
            'happy': 'happy',
            'sad': 'sad',
            'project': 'planning',
            'plan': 'planning',
        }

        for keyword, tag in keyword_map.items():
            if keyword in main_topic and tag not in tags:
                tags.append(tag)
                if len(tags) >= self.min_tags:
                    break

        # Ensure we have at least min_tags
        default_tags = ['conversation', 'general', 'reflection', 'thinking', 'question']
        for tag in default_tags:
            if tag not in tags and len(tags) < self.min_tags:
                tags.append(tag)

        return tags[:self.max_tags]

    def get_tag_categories(self) -> Dict[str, Set[str]]:
        """Return the full tag vocabulary organized by category."""
        return TAG_VOCABULARY.copy()

    def get_all_known_tags(self) -> Set[str]:
        """Return set of all known tags."""
        return ALL_KNOWN_TAGS.copy()
