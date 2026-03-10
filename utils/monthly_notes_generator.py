# utils/monthly_notes_generator.py
"""
MonthlyNotesGenerator - Generate monthly roll-up notes from daily notes.

Module Contract:
- Purpose: Roll up all daily notes within a calendar month into a single monthly summary,
           and migrate legacy weekly folders into monthly parent folders.
- Inputs:
  - generate_for_month(date) -> MonthlyGenerationResult: Generate monthly summary
  - generate_last_month_if_complete() -> Optional[MonthlyGenerationResult]: Startup catch-up
  - month_summary_exists(date) -> bool: Check if summary already exists
  - migrate_weekly_folders_to_monthly() -> int: Move Week folders into monthly parents
- Outputs:
  - Markdown files: Monthly/January 2026 Summary.md
  - MonthlyGenerationResult with success status, paths, stats
- Behavior:
  - Scans all daily notes whose filename date falls in the target calendar month
    (searches flat, weekly-at-root, and monthly/weekly paths)
  - Calls LLM to synthesize monthly themes, mood trajectory, growth, events
  - Generates contextual tags using TagGenerator
  - Writes atomic markdown with YAML frontmatter
  - Idempotent: skips if summary already exists (unless force=True)
  - Migration: moves Week folders into monthly parent folders (idempotent)
- Dependencies:
  - models.model_manager (LLM generation)
  - utils.tag_generator (tag generation)
  - config.app_config (paths and settings)
- Side effects:
  - Writes files to Obsidian vault
  - Moves folders during migration
  - LLM API calls
  - Logging
"""

import os
import re
import logging
import shutil
import calendar
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MonthlyGenerationResult:
    """Result of monthly note generation."""
    success: bool = False
    month: int = 0
    year: int = 0
    month_name: str = ""
    output_path: Optional[Path] = None
    daily_notes_found: int = 0
    total_conversations: int = 0
    avg_intensity: float = 0.0
    skipped_reason: Optional[str] = None  # "already_exists", "no_daily_notes", "disabled"
    error: Optional[str] = None


# LLM prompt template for monthly summaries
MONTHLY_NOTES_PROMPT = '''You are Daemon, an AI companion writing a monthly summary of your conversations with Luke.

DAILY NOTES FROM {month_name} {year}:

{formatted_daily_notes}

MONTH STATISTICS:
- Days with activity: {active_days}/{days_in_month}
- Total conversations: {total_conversations}
- Average intensity: {avg_intensity:.1f}/10
- Total active hours: {total_active_hours:.1f}

Write a monthly summary in this EXACT structure:

## Month at a Glance
4-5 sentences capturing the month's overarching theme, trajectory, and significance.

## Major Themes
The 3-5 biggest threads that defined this month. For each:
- **Theme Name**: What it was about, how it evolved, where it ended up

## Life Patterns
Aggregate life activity patterns across the entire month:

- **Work**: Overall pattern, total days mentioned, notable projects or shifts. If not discussed: "Not discussed this month."
- **Study**: Subjects covered, progress made, learning trajectory. If not discussed: "Not discussed this month."
- **Sleep**: Overall patterns, any recurring issues. If not discussed: "Not discussed this month."
- **Exercise/Health**: Activities, frequency, any changes. If not discussed: "Not discussed this month."
- **Social/Other**: Notable events, social activities, life changes.

Note: "Not discussed" means Luke didn't mention it - NOT that it didn't happen.

## Mood Trajectory
How Luke's emotional state evolved across the month. Note any turning points, sustained moods, or patterns.

## Growth & Progress
What changed between the start and end of the month? Skills developed, problems solved, goals advanced.

## Key Decisions
Major decisions made this month and their outcomes (if known).

## Unresolved Threads
Items carrying into next month.

## Month Stats
Summary of activity metrics and notable numbers.

## Intensity: X/10
Overall month assessment.

IMPORTANT:
- Write from YOUR perspective as Daemon ("This month we...", "Luke seemed...")
- Synthesize patterns across the entire month - don't just list weeks sequentially
- Identify trajectory and evolution, not just static summaries
- For Life Patterns: Only report what was explicitly discussed
- Be insightful about patterns that emerge over a month-long timescale
- Do NOT include any preamble or meta-commentary, just the note content
'''


class MonthlyNotesGenerator:
    """Generate monthly roll-up notes from daily notes and migrate folder structure."""

    def __init__(self, model_manager=None, vault_path: str = None, tag_generator=None):
        """
        Initialize MonthlyNotesGenerator.

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
                DAILY_NOTES_FOLDER,
                MONTHLY_NOTES_ENABLED,
                MONTHLY_NOTES_MODEL,
                MONTHLY_NOTES_MAX_TOKENS,
                TAG_GENERATION_ENABLED,
            )
            self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH).expanduser()
            self.enabled = MONTHLY_NOTES_ENABLED
            self.daily_folder = DAILY_NOTES_FOLDER
            self.model_name = MONTHLY_NOTES_MODEL
            self.max_tokens = MONTHLY_NOTES_MAX_TOKENS
            self.tag_generation_enabled = TAG_GENERATION_ENABLED
        except ImportError:
            self.vault_path = Path(vault_path or "~/Documents/Luke Notes").expanduser()
            self.enabled = True
            self.daily_folder = "Daily"
            self.model_name = "sonnet-4.5"
            self.max_tokens = 2000
            self.tag_generation_enabled = True

        self.output_dir = self.vault_path / self.daily_folder
        logger.debug(f"[MonthlyNotes] Initialized: vault={self.vault_path}, folder={self.daily_folder}")

    @property
    def model_manager(self):
        """Lazy-load ModelManager."""
        if self._model_manager is None:
            try:
                from models.model_manager import ModelManager
                self._model_manager = ModelManager()
                logger.debug("[MonthlyNotes] ModelManager lazy-loaded")
            except Exception as e:
                logger.error(f"[MonthlyNotes] Failed to load ModelManager: {e}")
                raise
        return self._model_manager

    @property
    def tag_generator(self):
        """Lazy-load TagGenerator."""
        if self._tag_generator is None:
            try:
                from utils.tag_generator import TagGenerator
                self._tag_generator = TagGenerator(model_manager=self.model_manager)
                logger.debug("[MonthlyNotes] TagGenerator lazy-loaded")
            except Exception as e:
                logger.warning(f"[MonthlyNotes] Failed to load TagGenerator: {e}")
                self._tag_generator = None
        return self._tag_generator

    def _get_month_folder_name(self, target_date: date) -> str:
        """Format monthly folder name: 'January 2026'."""
        return f"{target_date.strftime('%B %Y')}"

    def _get_week_folder_name(self, target_date: date) -> str:
        """Format weekly folder name: 'Week 3 Jan 2026'."""
        monday = target_date - timedelta(days=target_date.weekday())
        week_num = monday.isocalendar()[1]
        return f"Week {week_num} {monday.strftime('%b %Y')}"

    def _format_daily_filename(self, target_date: date) -> str:
        """Format daily note filename: 'M D YY Daily Note.md'."""
        return f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    def _parse_date_from_filename(self, filename: str) -> Optional[date]:
        """Parse date from daily note filename like '1 13 26 Daily Note.md'."""
        match = re.match(r'^(\d{1,2}) (\d{1,2}) (\d{2}) Daily Note\.md$', filename)
        if not match:
            return None
        try:
            month = int(match.group(1))
            day = int(match.group(2))
            year = 2000 + int(match.group(3))
            return date(year, month, day)
        except (ValueError, OverflowError):
            return None

    def _find_daily_notes_for_month(self, year: int, month: int) -> List[Tuple[date, Path]]:
        """
        Find all daily notes whose filename date falls in the target calendar month.
        Searches flat directory, weekly-at-root folders, and monthly/weekly folders.

        Returns list of (date, path) tuples sorted by date.
        """
        notes = {}  # date -> path (dedup by date)

        # 1. Check flat directory
        if self.output_dir.exists():
            for f in self.output_dir.glob("*Daily Note.md"):
                d = self._parse_date_from_filename(f.name)
                if d and d.year == year and d.month == month:
                    notes[d] = f

        # 2. Check weekly folders at root (legacy layout)
        if self.output_dir.exists():
            for folder in self.output_dir.iterdir():
                if folder.is_dir() and folder.name.startswith("Week "):
                    for f in folder.glob("*Daily Note.md"):
                        d = self._parse_date_from_filename(f.name)
                        if d and d.year == year and d.month == month and d not in notes:
                            notes[d] = f

        # 3. Check monthly/weekly folders (new layout)
        if self.output_dir.exists():
            for month_dir in self.output_dir.iterdir():
                if month_dir.is_dir() and not month_dir.name.startswith("Week "):
                    for week_dir in month_dir.iterdir():
                        if week_dir.is_dir() and week_dir.name.startswith("Week "):
                            for f in week_dir.glob("*Daily Note.md"):
                                d = self._parse_date_from_filename(f.name)
                                if d and d.year == year and d.month == month and d not in notes:
                                    notes[d] = f

        return sorted(notes.items(), key=lambda x: x[0])

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
                'intensity': frontmatter.get('usage_intensity', frontmatter.get('intensity', 0)),
                'conversations': frontmatter.get('conversations', 0),
                'main_quest': frontmatter.get('main_quest', 'Unknown'),
                'active_hours': frontmatter.get('active_hours', frontmatter.get('duration_hours', 0)),
                'span_hours': frontmatter.get('span_hours', 0),
            }
        except Exception as e:
            logger.error(f"[MonthlyNotes] Failed to read {path}: {e}")
            return {
                'path': path,
                'frontmatter': {},
                'content': '',
                'date': None,
                'intensity': 0,
                'conversations': 0,
                'main_quest': 'Unknown',
                'active_hours': 0,
                'span_hours': 0,
            }

    def _calculate_month_stats(self, notes_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate stats from daily notes for the month."""
        if not notes_data:
            return {
                'active_days': 0,
                'total_conversations': 0,
                'avg_intensity': 0.0,
                'total_active_hours': 0.0,
                'peak_day': None,
                'peak_intensity': 0,
            }

        total_convos = sum(n.get('conversations', 0) for n in notes_data)
        intensities = [n.get('intensity', 0) for n in notes_data if n.get('intensity')]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        total_active_hours = sum(n.get('active_hours', 0) for n in notes_data)

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
            if d and isinstance(d, date):
                peak_day = d.strftime('%B %d')
            peak_intensity = peak_note.get('intensity', 0)

        return {
            'active_days': len(notes_data),
            'total_conversations': total_convos,
            'avg_intensity': round(avg_intensity, 1),
            'total_active_hours': round(total_active_hours, 1),
            'peak_day': peak_day,
            'peak_intensity': peak_intensity,
        }

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

            content = note.get('content', '')

            # Truncate content to keep prompt reasonable
            max_content = 1200
            if len(content) > max_content:
                content = content[:max_content] + "..."

            formatted.append(f"""### {date_str}
**Main Quest:** {main_quest}
**Intensity:** {intensity}/10 | **Conversations:** {convos}

{content}
""")

        return "\n---\n".join(formatted)

    def _build_monthly_frontmatter(self, year: int, month: int, stats: Dict[str, Any],
                                    content_tags: List[str] = None) -> str:
        """Generate YAML frontmatter for monthly note."""
        system_tags = ['monthly', 'daemon-generated']

        if content_tags:
            all_tags = system_tags + content_tags
        else:
            all_tags = system_tags

        tags_str = ', '.join(f'"{tag}"' for tag in all_tags)
        month_name = calendar.month_name[month]
        days_in_month = calendar.monthrange(year, month)[1]

        return f"""---
month: {month}
year: {year}
month_name: "{month_name}"
total_conversations: {stats['total_conversations']}
avg_usage_intensity: {stats['avg_intensity']}
days_with_activity: {stats['active_days']}
days_in_month: {days_in_month}
total_active_hours: {stats['total_active_hours']}
tags: [{tags_str}]
generated: {datetime.now().isoformat()}
---
"""

    def month_summary_exists(self, target_date: date) -> bool:
        """Check if monthly summary already exists."""
        month_folder = self._get_month_folder_name(target_date)
        summary_filename = f"{month_folder} Summary.md"
        return (self.output_dir / month_folder / summary_filename).exists()

    def migrate_weekly_folders_to_monthly(self) -> int:
        """
        Move legacy Week folders from root into monthly parent folders.

        Only migrates folders matching 'Week N Mon YYYY' pattern that are
        directly under output_dir (not already inside a monthly folder).
        Idempotent: skips if already nested or target exists.

        Returns:
            Count of folders moved.
        """
        if not self.output_dir.exists():
            return 0

        moved = 0
        week_pattern = re.compile(r'^Week \d+ \w{3} \d{4}$')

        for folder in sorted(self.output_dir.iterdir()):
            if not folder.is_dir():
                continue
            if not week_pattern.match(folder.name):
                continue

            # Parse the month from the folder name (e.g., "Week 3 Jan 2026" -> January 2026)
            match = re.match(r'^Week \d+ (\w{3}) (\d{4})$', folder.name)
            if not match:
                continue

            month_abbr = match.group(1)
            year_str = match.group(2)

            # Convert abbreviated month to full name
            try:
                month_num = list(calendar.month_abbr).index(month_abbr)
                month_full = calendar.month_name[month_num]
            except (ValueError, IndexError):
                logger.warning(f"[MonthlyNotes] Could not parse month from '{folder.name}', skipping")
                continue

            month_folder_name = f"{month_full} {year_str}"
            month_folder = self.output_dir / month_folder_name
            target_path = month_folder / folder.name

            # Skip if target already exists
            if target_path.exists():
                logger.debug(f"[MonthlyNotes] Skip migration: {folder.name} already in {month_folder_name}/")
                continue

            # Create monthly parent and move
            try:
                month_folder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(target_path))
                logger.info(f"[MonthlyNotes] Migrated {folder.name} -> {month_folder_name}/{folder.name}")
                moved += 1
            except Exception as e:
                logger.error(f"[MonthlyNotes] Failed to migrate {folder.name}: {e}")

        return moved

    async def generate_for_month(self, target_date: date, force: bool = False) -> MonthlyGenerationResult:
        """
        Generate monthly summary for the month containing target_date.

        Args:
            target_date: Any date within the target month
            force: If True, overwrite existing summary

        Returns:
            MonthlyGenerationResult with success status and details
        """
        year = target_date.year
        month = target_date.month
        month_name = calendar.month_name[month]
        days_in_month = calendar.monthrange(year, month)[1]

        result = MonthlyGenerationResult(
            month=month,
            year=year,
            month_name=month_name,
        )

        # Check if disabled
        if not self.enabled:
            result.skipped_reason = "disabled"
            logger.info(f"[MonthlyNotes] Skipped {month_name} {year}: feature disabled")
            return result

        # Check if summary exists
        if not force and self.month_summary_exists(target_date):
            result.skipped_reason = "already_exists"
            logger.info(f"[MonthlyNotes] Skipped {month_name} {year}: summary already exists")
            return result

        # Find daily notes for this month
        daily_notes = self._find_daily_notes_for_month(year, month)
        result.daily_notes_found = len(daily_notes)

        if not daily_notes:
            result.skipped_reason = "no_daily_notes"
            logger.info(f"[MonthlyNotes] Skipped {month_name} {year}: no daily notes found")
            return result

        # Read and parse daily notes
        notes_data = [self._read_daily_note(path) for _, path in daily_notes]

        # Calculate stats
        stats = self._calculate_month_stats(notes_data)
        result.total_conversations = stats['total_conversations']
        result.avg_intensity = stats['avg_intensity']

        # Format for LLM
        formatted_notes = self._format_daily_notes_for_prompt(notes_data)

        # Build prompt
        prompt = MONTHLY_NOTES_PROMPT.format(
            month_name=month_name,
            year=year,
            formatted_daily_notes=formatted_notes,
            active_days=stats['active_days'],
            days_in_month=days_in_month,
            total_conversations=stats['total_conversations'],
            avg_intensity=stats['avg_intensity'],
            total_active_hours=stats['total_active_hours'],
        )

        # Call LLM with fallback models
        fallback_models = [
            "sonnet-4.5",       # Anthropic Claude (fast)
            "gpt-4o-mini",      # Fast, cheap OpenAI
            "deepseek-v3.1",    # DeepSeek
            "gpt-4o",           # Standard OpenAI
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
                logger.info(f"[MonthlyNotes] Generating summary for {month_name} {year} ({stats['active_days']} days) using {model}")
                llm_response = await self.model_manager.generate_once(
                    prompt,
                    max_tokens=self.max_tokens,
                    model_name=model,
                )

                # Check for API error responses
                if llm_response and llm_response.startswith("[API unavailable]"):
                    logger.warning(f"[MonthlyNotes] Model {model} unavailable, trying next...")
                    last_error = f"Model {model} unavailable"
                    llm_response = None
                    continue

                if llm_response and llm_response.startswith("[Error:"):
                    logger.warning(f"[MonthlyNotes] Model {model} returned error, trying next...")
                    last_error = f"Model {model} returned error"
                    llm_response = None
                    continue

                # Validate response length
                if not llm_response or len(llm_response.strip()) < 200:
                    logger.warning(f"[MonthlyNotes] Model {model} returned too-short response, trying next...")
                    last_error = f"Model {model} returned too-short response"
                    llm_response = None
                    continue

                # Success
                logger.info(f"[MonthlyNotes] Successfully generated summary using {model}")
                break

            except Exception as e:
                logger.warning(f"[MonthlyNotes] Model {model} failed: {e}, trying next...")
                last_error = str(e)
                llm_response = None
                continue

        if not llm_response:
            result.error = f"All LLM models failed. Last error: {last_error}"
            logger.error(f"[MonthlyNotes] All models failed for {month_name} {year}: {last_error}")
            return result

        # Generate contextual tags
        content_tags = []
        if self.tag_generation_enabled and self.tag_generator:
            try:
                main_quests = [n.get('main_quest', '') for n in notes_data if n.get('main_quest')]
                tag_metadata = {
                    'main_topic': f"{month_name} {year} monthly summary",
                    'intensity': stats['avg_intensity'],
                    'conversations': stats['total_conversations'],
                    'duration_hours': stats['total_active_hours'],
                    'main_quests': ', '.join(main_quests[:5]),
                }
                tag_result = await self.tag_generator.generate_tags(
                    llm_response,
                    note_type="monthly",
                    metadata=tag_metadata
                )
                content_tags = tag_result.tags
                logger.info(f"[MonthlyNotes] Generated {len(content_tags)} tags: {', '.join(content_tags)}")
            except Exception as e:
                logger.warning(f"[MonthlyNotes] Tag generation failed: {e}, continuing without tags")

        # Build full note
        frontmatter = self._build_monthly_frontmatter(year, month, stats, content_tags=content_tags)
        header = f"\n# Monthly Summary - {month_name} {year}\n\n"
        full_content = frontmatter + header + llm_response.strip() + "\n"

        # Write summary
        try:
            month_folder_name = self._get_month_folder_name(target_date)
            month_folder = self.output_dir / month_folder_name
            month_folder.mkdir(parents=True, exist_ok=True)

            summary_filename = f"{month_folder_name} Summary.md"
            summary_path = month_folder / summary_filename
            temp_path = summary_path.with_suffix(".md.tmp")

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            os.replace(temp_path, summary_path)

            result.output_path = summary_path
            result.success = True
            logger.info(f"[MonthlyNotes] Success: {summary_path}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"[MonthlyNotes] Failed to write summary: {e}")

        return result

    async def generate_last_month_if_complete(self) -> Optional[MonthlyGenerationResult]:
        """
        Startup catch-up: generate last month's summary if today >= 2nd of month
        and last month's summary is missing.

        Returns:
            MonthlyGenerationResult if generated, None if skipped
        """
        today = date.today()

        # Only generate after the 1st (give time for last day's daily note)
        if today.day < 2:
            logger.debug("[MonthlyNotes] Too early in month for catch-up (day < 2)")
            return None

        # Calculate last month
        if today.month == 1:
            last_month_date = date(today.year - 1, 12, 1)
        else:
            last_month_date = date(today.year, today.month - 1, 1)

        # Check if summary exists
        if self.month_summary_exists(last_month_date):
            logger.debug(f"[MonthlyNotes] Last month's summary exists, skipping catch-up")
            return None

        month_name = calendar.month_name[last_month_date.month]
        logger.info(f"[MonthlyNotes] Catch-up: generating {month_name} {last_month_date.year} summary")
        return await self.generate_for_month(last_month_date)
