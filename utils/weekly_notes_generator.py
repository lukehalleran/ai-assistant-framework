# utils/weekly_notes_generator.py
"""
WeeklyNotesGenerator - Generate weekly summary notes from daily notes.

Module Contract:
- Purpose: Organize daily notes into weekly folders and generate weekly summaries
           by aggregating the daily notes using LLM with contextual tag generation.
- Inputs:
  - generate_for_week(date: date) -> WeeklyGenerationResult: Generate for week containing date
  - generate_last_week_if_complete() -> Optional[WeeklyGenerationResult]: Startup catch-up
  - week_summary_exists(date: date) -> bool: Check if summary already exists
- Outputs:
  - Weekly folder: ~/Documents/Luke Notes/Vault/Daily Notes and To Do's/Week 3 Jan 2026/
  - Weekly summary: Week 3 Jan 2026 Summary.md
  - WeeklyGenerationResult with success status, paths, stats
- Behavior:
  - Finds daily notes for the week (Mon-Sun)
  - Moves daily notes into weekly folder
  - Reads and parses daily notes (frontmatter + content)
    - Backward compatible: reads both old (intensity, duration_hours) and new
      (usage_intensity, span_hours, active_hours) field names [UPDATED 2026-01-18]
  - Calls LLM to generate weekly summary:
    - Main Quests, Recurring Themes, Emotional Arc, etc.
    - Life Events Summary: aggregates Work/Study/Sleep/Exercise across week [NEW 2026-01-18]
  - Generates contextual tags using TagGenerator (5-10 tags) [NEW 2026-01-22]
  - Writes atomic markdown with YAML frontmatter:
    - avg_usage_intensity (was: avg_intensity) [RENAMED 2026-01-18]
    - total_active_hours (was: total_duration_hours) [RENAMED 2026-01-18]
    - tags: system tags + content tags [NEW 2026-01-22]
  - Idempotent: skips if summary already exists (unless force=True)
- Dependencies:
  - models.model_manager (LLM generation)
  - utils.tag_generator (tag generation for summaries) [NEW 2026-01-22]
  - config.app_config (paths and settings)
- Side effects:
  - Creates weekly folders
  - Moves daily note files
  - Writes summary files
  - LLM API calls (summary generation + tag generation)
  - Logging
"""

import os
import re
import logging
import shutil
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WeeklyGenerationResult:
    """Result of weekly note generation."""
    success: bool = False
    week_num: int = 0
    year: int = 0
    week_folder: Optional[Path] = None
    output_path: Optional[Path] = None
    daily_notes_found: int = 0
    daily_notes_moved: int = 0
    total_conversations: int = 0
    avg_intensity: float = 0.0
    skipped_reason: Optional[str] = None  # "already_exists", "no_daily_notes", "disabled"
    error: Optional[str] = None


# LLM prompt template for weekly summaries
WEEKLY_NOTES_PROMPT = '''You are Daemon, an AI companion writing a weekly summary of your conversations with Luke.

DAILY NOTES FROM WEEK {week_num}, {year}:

{formatted_daily_notes}

WEEK STATISTICS:
- Days with activity: {active_days}/7
- Total conversations: {total_conversations}
- Average intensity: {avg_intensity:.1f}/10
- Date range: {start_date} to {end_date}

Write a weekly summary in this EXACT structure:

## Week at a Glance
3-4 sentences capturing the week's overarching theme and trajectory.

## Main Quests This Week
List main quests from each day, noting which completed vs carried over:
- **Quest Name** (days active): Status - brief outcome

## Life Events Summary
Aggregate life activity patterns from the daily notes. For each category, summarize what was discussed across the week:

- **Work**: Days mentioned, total hours if noted, overall pattern/how it went. If not discussed any day: "Not discussed this week."
- **Study**: Subjects covered, total time if noted, progress made. If not discussed: "Not discussed this week."
- **Sleep**: Overall patterns, any issues noted. If not discussed: "Not discussed this week."
- **Exercise/Health**: Activities done, frequency. If not discussed: "Not discussed this week."
- **Other Events**: Notable life events aggregated across the week.

Note: "Not discussed" means Luke didn't mention it - NOT that it didn't happen.

## Recurring Themes
Topics/patterns that appeared across multiple days.

## Mood Arc
How Luke's emotional state evolved through the week.

## Key Decisions (Aggregated)
Major decisions consolidated from the week.

## Knowledge Gained (Aggregated)
New concepts/insights, deduplicated across days.

## Open Threads
Unresolved items carrying into next week.

## Week Stats
Summary of activity metrics.

## Intensity: X/10
Overall week assessment based on cumulative activity.

IMPORTANT:
- Write from YOUR perspective as Daemon ("This week we...", "Luke seemed...")
- Synthesize patterns across days, don't just list each day sequentially
- Note evolution and progress, not just static facts
- If a quest resolved, celebrate it; if it carried over, note why
- For Life Events: Only report what was explicitly discussed. "Not discussed" ≠ "didn't happen"
- Keep it concise but insightful
- Do NOT include any preamble or meta-commentary, just the note content
'''


class WeeklyNotesGenerator:
    """Generate weekly summary notes and organize daily notes into folders."""

    def __init__(self, model_manager=None, vault_path: str = None, tag_generator=None):
        """
        Initialize WeeklyNotesGenerator.

        Args:
            model_manager: ModelManager instance (lazy-loaded if None)
            vault_path: Path to Obsidian vault (defaults to config)
            tag_generator: TagGenerator instance (lazy-loaded if None)
        """
        self._model_manager = model_manager
        self._tag_generator = tag_generator

        # Load config
        try:
            from config.app_config import (
                OBSIDIAN_VAULT_PATH,
                DAILY_NOTES_ENABLED,
                DAILY_NOTES_FOLDER,
                WEEKLY_NOTES_ENABLED,
                WEEKLY_NOTES_MODEL,
                WEEKLY_NOTES_MAX_TOKENS,
                TAG_GENERATION_ENABLED,
            )
            self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH).expanduser()
            self.daily_enabled = DAILY_NOTES_ENABLED
            self.enabled = WEEKLY_NOTES_ENABLED
            self.daily_folder = DAILY_NOTES_FOLDER
            self.model_name = WEEKLY_NOTES_MODEL
            self.max_tokens = WEEKLY_NOTES_MAX_TOKENS
            self.tag_generation_enabled = TAG_GENERATION_ENABLED
        except ImportError:
            self.vault_path = Path(vault_path or "~/Documents/Luke Notes").expanduser()
            self.daily_enabled = True
            self.enabled = True
            self.daily_folder = "Daily"
            self.model_name = "gpt-4o-mini"
            self.max_tokens = 1200
            self.tag_generation_enabled = True

        self.output_dir = self.vault_path / self.daily_folder
        logger.debug(f"[WeeklyNotes] Initialized: vault={self.vault_path}, folder={self.daily_folder}")

    @property
    def model_manager(self):
        """Lazy-load ModelManager."""
        if self._model_manager is None:
            try:
                from models.model_manager import ModelManager
                self._model_manager = ModelManager()
                logger.debug("[WeeklyNotes] ModelManager lazy-loaded")
            except Exception as e:
                logger.error(f"[WeeklyNotes] Failed to load ModelManager: {e}")
                raise
        return self._model_manager

    @property
    def tag_generator(self):
        """Lazy-load TagGenerator."""
        if self._tag_generator is None:
            try:
                from utils.tag_generator import TagGenerator
                self._tag_generator = TagGenerator(model_manager=self.model_manager)
                logger.debug("[WeeklyNotes] TagGenerator lazy-loaded")
            except Exception as e:
                logger.warning(f"[WeeklyNotes] Failed to load TagGenerator: {e}")
                # Non-critical, can continue without tag generation
                self._tag_generator = None
        return self._tag_generator

    def _get_week_folder_name(self, target_date: date) -> str:
        """Format folder name: 'Week 3 Jan 2026'."""
        # Use ISO week number from Monday of that week
        monday = target_date - timedelta(days=target_date.weekday())
        week_num = monday.isocalendar()[1]
        return f"Week {week_num} {monday.strftime('%b %Y')}"

    def _get_week_date_range(self, target_date: date) -> Tuple[date, date]:
        """Get Monday-Sunday range for the week containing target_date."""
        monday = target_date - timedelta(days=target_date.weekday())
        sunday = monday + timedelta(days=6)
        return monday, sunday

    def _format_daily_filename(self, target_date: date) -> str:
        """Format daily note filename: 'M D YY Daily Note.md'."""
        return f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    def _find_daily_notes_for_week(self, start: date, end: date) -> List[Tuple[date, Path]]:
        """
        Find all daily notes in date range.
        Searches both flat directory and weekly folders.
        Returns list of (date, path) tuples.
        """
        notes = []
        current = start

        while current <= end:
            filename = self._format_daily_filename(current)

            # Check flat directory first
            flat_path = self.output_dir / filename
            if flat_path.exists():
                notes.append((current, flat_path))
                current += timedelta(days=1)
                continue

            # Check in weekly folder
            week_folder = self.output_dir / self._get_week_folder_name(current)
            folder_path = week_folder / filename
            if folder_path.exists():
                notes.append((current, folder_path))
                current += timedelta(days=1)
                continue

            # Note doesn't exist for this day
            current += timedelta(days=1)

        return notes

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        frontmatter = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except Exception:
                    pass
                body = parts[2].strip()

        return frontmatter, body

    def _read_daily_note(self, path: Path) -> Dict[str, Any]:
        """Read daily note and parse frontmatter and content."""
        try:
            content = path.read_text(encoding='utf-8')
            frontmatter, body = self._parse_frontmatter(content)

            return {
                'path': path,
                'frontmatter': frontmatter,
                'content': body,
                'date': frontmatter.get('date'),
                'intensity': frontmatter.get('usage_intensity', frontmatter.get('intensity', 0)),  # Support both old and new field names
                'conversations': frontmatter.get('conversations', 0),
                'main_quest': frontmatter.get('main_quest', 'Unknown'),
                'duration_hours': frontmatter.get('active_hours', frontmatter.get('duration_hours', 0)),  # Support both old and new
            }
        except Exception as e:
            logger.error(f"[WeeklyNotes] Failed to read {path}: {e}")
            return {
                'path': path,
                'frontmatter': {},
                'content': '',
                'date': None,
                'intensity': 0,
                'conversations': 0,
                'main_quest': 'Unknown',
                'duration_hours': 0,
            }

    def _move_daily_notes_to_folder(self, notes: List[Tuple[date, Path]], folder: Path) -> int:
        """Move daily notes into weekly folder. Returns count moved."""
        folder.mkdir(parents=True, exist_ok=True)
        moved = 0

        for note_date, note_path in notes:
            # Skip if already in the target folder
            if note_path.parent == folder:
                continue

            target_path = folder / note_path.name
            try:
                shutil.move(str(note_path), str(target_path))
                logger.info(f"[WeeklyNotes] Moved {note_path.name} to {folder.name}/")
                moved += 1
            except Exception as e:
                logger.error(f"[WeeklyNotes] Failed to move {note_path}: {e}")

        return moved

    def _format_daily_notes_for_prompt(self, notes_data: List[Dict]) -> str:
        """Format daily notes for LLM summarization."""
        formatted = []

        for note in sorted(notes_data, key=lambda x: x.get('date') or ''):
            date_str = note.get('date', 'Unknown date')
            if isinstance(date_str, date):
                date_str = date_str.strftime('%A, %B %d')

            intensity = note.get('intensity', '?')
            convos = note.get('conversations', '?')
            main_quest = note.get('main_quest', 'Unknown')

            # Extract key sections from content
            content = note.get('content', '')

            # Truncate content to keep prompt reasonable
            max_content = 1500
            if len(content) > max_content:
                content = content[:max_content] + "..."

            formatted.append(f"""### {date_str}
**Main Quest:** {main_quest}
**Intensity:** {intensity}/10 | **Conversations:** {convos}

{content}
""")

        return "\n---\n".join(formatted)

    def _calculate_week_stats(self, notes_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate stats from daily notes."""
        if not notes_data:
            return {
                'active_days': 0,
                'total_conversations': 0,
                'avg_intensity': 0.0,
                'total_duration': 0.0,
                'peak_day': None,
                'peak_intensity': 0,
            }

        total_convos = sum(n.get('conversations', 0) for n in notes_data)
        intensities = [n.get('intensity', 0) for n in notes_data if n.get('intensity')]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        total_duration = sum(n.get('duration_hours', 0) for n in notes_data)

        # Find peak day
        peak_note = max(notes_data, key=lambda x: x.get('intensity', 0), default=None)
        peak_day = None
        peak_intensity = 0
        if peak_note and peak_note.get('date'):
            d = peak_note['date']
            if isinstance(d, str):
                try:
                    d = datetime.fromisoformat(d).date()
                except ValueError:
                    d = None
            if d:
                peak_day = d.strftime('%A')
            peak_intensity = peak_note.get('intensity', 0)

        return {
            'active_days': len(notes_data),
            'total_conversations': total_convos,
            'avg_intensity': round(avg_intensity, 1),
            'total_duration': round(total_duration, 1),
            'peak_day': peak_day,
            'peak_intensity': peak_intensity,
        }

    def _build_weekly_frontmatter(self, week_num: int, year: int, start: date, end: date,
                                   stats: Dict[str, Any], content_tags: List[str] = None) -> str:
        """Generate YAML frontmatter for weekly note."""
        # System tags are always included
        system_tags = ['weekly', 'daemon-generated']

        # Add content tags if provided
        if content_tags:
            all_tags = system_tags + content_tags
        else:
            all_tags = system_tags

        # Format tags for YAML (quoted strings to handle hyphens)
        tags_str = ', '.join(f'"{tag}"' for tag in all_tags)

        return f"""---
week: {week_num}
year: {year}
start_date: {start.isoformat()}
end_date: {end.isoformat()}
total_conversations: {stats['total_conversations']}
avg_usage_intensity: {stats['avg_intensity']}
days_with_activity: {stats['active_days']}
total_active_hours: {stats['total_duration']}
tags: [{tags_str}]
generated: {datetime.now().isoformat()}
---
"""

    def _format_summary_filename(self, target_date: date) -> str:
        """Format summary filename: 'Week 3 Jan 2026 Summary.md'."""
        folder_name = self._get_week_folder_name(target_date)
        return f"{folder_name} Summary.md"

    def week_summary_exists(self, target_date: date) -> bool:
        """Check if weekly summary already exists."""
        week_folder = self.output_dir / self._get_week_folder_name(target_date)
        summary_path = week_folder / self._format_summary_filename(target_date)
        return summary_path.exists()

    def _write_summary(self, folder: Path, filename: str, content: str) -> Path:
        """Atomic write of weekly summary."""
        folder.mkdir(parents=True, exist_ok=True)

        summary_path = folder / filename
        temp_path = summary_path.with_suffix(".md.tmp")

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, summary_path)
            logger.info(f"[WeeklyNotes] Written: {summary_path}")
            return summary_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def generate_for_week(self, target_date: date, force: bool = False) -> WeeklyGenerationResult:
        """
        Generate weekly summary for the week containing target_date.

        Args:
            target_date: Any date within the target week
            force: If True, overwrite existing summary

        Returns:
            WeeklyGenerationResult with success status and details
        """
        # Get week info
        monday, sunday = self._get_week_date_range(target_date)
        week_num = monday.isocalendar()[1]
        year = monday.year

        result = WeeklyGenerationResult(
            week_num=week_num,
            year=year,
        )

        # Check if disabled
        if not self.enabled:
            result.skipped_reason = "disabled"
            logger.info(f"[WeeklyNotes] Skipped Week {week_num}: feature disabled")
            return result

        # Check if summary exists
        if not force and self.week_summary_exists(target_date):
            result.skipped_reason = "already_exists"
            week_folder = self.output_dir / self._get_week_folder_name(target_date)
            result.week_folder = week_folder
            result.output_path = week_folder / self._format_summary_filename(target_date)
            logger.info(f"[WeeklyNotes] Skipped Week {week_num}: summary already exists")
            return result

        # Find daily notes
        daily_notes = self._find_daily_notes_for_week(monday, sunday)
        result.daily_notes_found = len(daily_notes)

        if not daily_notes:
            result.skipped_reason = "no_daily_notes"
            logger.info(f"[WeeklyNotes] Skipped Week {week_num}: no daily notes found")
            return result

        # Create week folder and move notes
        week_folder = self.output_dir / self._get_week_folder_name(monday)
        result.week_folder = week_folder
        result.daily_notes_moved = self._move_daily_notes_to_folder(daily_notes, week_folder)

        # Re-find notes after moving (paths changed)
        daily_notes = self._find_daily_notes_for_week(monday, sunday)

        # Read and parse daily notes
        notes_data = [self._read_daily_note(path) for _, path in daily_notes]

        # Calculate stats
        stats = self._calculate_week_stats(notes_data)
        result.total_conversations = stats['total_conversations']
        result.avg_intensity = stats['avg_intensity']

        # Format for LLM
        formatted_notes = self._format_daily_notes_for_prompt(notes_data)

        # Build prompt
        prompt = WEEKLY_NOTES_PROMPT.format(
            week_num=week_num,
            year=year,
            formatted_daily_notes=formatted_notes,
            active_days=stats['active_days'],
            total_conversations=stats['total_conversations'],
            avg_intensity=stats['avg_intensity'],
            start_date=monday.strftime('%B %d'),
            end_date=sunday.strftime('%B %d, %Y'),
        )

        # Call LLM with fallback models
        # Expanded list includes Claude, Gemini, and newer GPT models for better reliability
        fallback_models = [
            "gpt-4o-mini",       # Fast, cheap OpenAI
            "deepseek-v3.1",    # DeepSeek
            "gpt-4o",           # Standard OpenAI
            "sonnet-4.5",       # Anthropic Claude (fast)
            "claude-opus-4.5",  # Anthropic Claude (best)
            "gemini-3-pro",     # Google Gemini
            "gpt-5",            # Newer OpenAI
            "deepseek-r1",      # DeepSeek reasoning
            "glm-4.6",          # GLM
        ]
        models_to_try = [self.model_name] + [m for m in fallback_models if m != self.model_name]

        llm_response = None
        last_error = None

        for model in models_to_try:
            try:
                logger.info(f"[WeeklyNotes] Generating summary for Week {week_num} ({stats['active_days']} days) using {model}")
                llm_response = await self.model_manager.generate_once(
                    prompt,
                    max_tokens=self.max_tokens,
                    model_name=model,
                )

                # Check for API error responses
                if llm_response and llm_response.startswith("[OpenAI unavailable]"):
                    logger.warning(f"[WeeklyNotes] Model {model} unavailable, trying next...")
                    last_error = f"Model {model} unavailable"
                    llm_response = None
                    continue

                # Check for empty response
                if llm_response and llm_response.startswith("[Error:"):
                    logger.warning(f"[WeeklyNotes] Model {model} returned error, trying next...")
                    last_error = f"Model {model} returned error"
                    llm_response = None
                    continue

                # Validate response length
                if not llm_response or len(llm_response.strip()) < 200:
                    logger.warning(f"[WeeklyNotes] Model {model} returned too-short response, trying next...")
                    last_error = f"Model {model} returned too-short response"
                    llm_response = None
                    continue

                # Success
                logger.info(f"[WeeklyNotes] Successfully generated summary using {model}")
                break

            except Exception as e:
                logger.warning(f"[WeeklyNotes] Model {model} failed: {e}, trying next...")
                last_error = str(e)
                llm_response = None
                continue

        if not llm_response:
            result.error = f"All LLM models failed. Last error: {last_error}"
            logger.error(f"[WeeklyNotes] All models failed for Week {week_num}: {last_error}")
            return result

        # Generate contextual tags
        content_tags = []
        if self.tag_generation_enabled and self.tag_generator:
            try:
                # Aggregate main quests from notes_data for context
                main_quests = [n.get('main_quest', '') for n in notes_data if n.get('main_quest')]
                tag_metadata = {
                    'main_topic': f"Week {week_num} summary",
                    'intensity': stats['avg_intensity'],
                    'conversations': stats['total_conversations'],
                    'duration_hours': stats['total_duration'],
                    'main_quests': ', '.join(main_quests[:3]),  # First 3 quests for context
                }
                tag_result = await self.tag_generator.generate_tags(
                    llm_response,
                    note_type="weekly",
                    metadata=tag_metadata
                )
                content_tags = tag_result.tags
                logger.info(f"[WeeklyNotes] Generated {len(content_tags)} tags: {', '.join(content_tags)}")
            except Exception as e:
                logger.warning(f"[WeeklyNotes] Tag generation failed: {e}, continuing without tags")

        # Build full note
        frontmatter = self._build_weekly_frontmatter(week_num, year, monday, sunday, stats, content_tags=content_tags)
        header = f"\n# Weekly Summary - Week {week_num}, {monday.strftime('%B %Y')}\n\n"
        full_content = frontmatter + header + llm_response.strip() + "\n"

        # Write summary
        try:
            summary_filename = self._format_summary_filename(monday)
            result.output_path = self._write_summary(week_folder, summary_filename, full_content)
            result.success = True
            logger.info(f"[WeeklyNotes] Success: {result.output_path}")
        except Exception as e:
            result.error = str(e)
            logger.error(f"[WeeklyNotes] Failed to write summary: {e}")

        return result

    async def generate_last_week_if_complete(self) -> Optional[WeeklyGenerationResult]:
        """
        Startup catch-up: generate last week's summary if the week is complete.

        Only generates if:
        - Today is Monday or later (full week has passed)
        - Summary doesn't exist
        - At least one daily note exists for the week

        Returns:
            WeeklyGenerationResult if generated, None if skipped
        """
        today = date.today()

        # Calculate last week's Monday
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)

        # Check if summary exists
        if self.week_summary_exists(last_monday):
            logger.debug(f"[WeeklyNotes] Last week's summary exists, skipping catch-up")
            return None

        logger.info(f"[WeeklyNotes] Catch-up: generating last week's summary (Week {last_monday.isocalendar()[1]})")
        return await self.generate_for_week(last_monday)

    async def organize_week(self, target_date: date) -> Tuple[Path, int]:
        """
        Just organize daily notes into weekly folder without generating summary.

        Returns:
            Tuple of (week_folder_path, notes_moved_count)
        """
        monday, sunday = self._get_week_date_range(target_date)
        week_folder = self.output_dir / self._get_week_folder_name(monday)

        daily_notes = self._find_daily_notes_for_week(monday, sunday)
        moved = self._move_daily_notes_to_folder(daily_notes, week_folder)

        return week_folder, moved
