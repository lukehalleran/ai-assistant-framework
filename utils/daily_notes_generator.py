# utils/daily_notes_generator.py
"""
DailyNotesGenerator - Generate daily summary notes from Daemon conversations.

Module Contract:
- Purpose: Automatically generate structured daily notes from conversation history,
           written from Daemon's perspective in YAML + bullet format.
- Inputs:
  - generate_for_date(date: date) -> GenerationResult: Generate note for specific date
  - generate_yesterday_if_missing() -> Optional[GenerationResult]: Startup catch-up
  - note_exists(date: date) -> bool: Check if note already exists
- Outputs:
  - Markdown files in Obsidian vault: ~/Documents/Luke Notes/Daily/YYYY-MM-DD.md
  - GenerationResult with success status, path, stats
- Behavior:
  - Filters corpus by timestamp to get day's conversations
  - Calls LLM to generate structured summary (Main Quest, Side Quests, etc.)
  - Writes atomic markdown with YAML frontmatter
  - Idempotent: skips if note already exists (unless force=True)
- Dependencies:
  - memory.corpus_manager (conversation retrieval)
  - models.model_manager (LLM generation)
  - config.app_config (paths and settings)
- Side effects:
  - Writes files to Obsidian vault
  - LLM API calls
  - Logging
"""

import os
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of daily note generation."""
    success: bool = False
    date: Optional[date] = None
    output_path: Optional[Path] = None
    conversation_count: int = 0
    intensity: int = 0
    duration_hours: float = 0.0
    skipped_reason: Optional[str] = None  # "already_exists", "no_conversations", "disabled"
    error: Optional[str] = None


# LLM prompt template
DAILY_NOTES_PROMPT = '''You are Daemon, an AI companion writing a daily note about your conversations with Luke today.

CONVERSATIONS FROM {date}:
{formatted_conversations}

STATISTICS:
- Conversations: {count}
- Time span: {first_time} to {last_time} ({duration_hours:.1f} hours)

Write a daily note in this EXACT structure:

## Summary
2-3 sentences capturing the day's theme from your perspective as Daemon.

## Main Quest: [Primary Focus]
The main thing we worked on today. 3-5 bullets about progress, challenges, outcomes.

## Side Quests
Other topics discussed. Format: **Topic**: one-line description. If only one main topic, write "None today - fully focused on the Main Quest."

## Emotional State
Luke's mood throughout the day. Note any shifts. Be specific and observant.

## Key Decisions
Explicit choices made today. Bullet list. If none, write "None today."

## Knowledge Gained
New facts, concepts, or insights that came up. Bullet list. If none, write "None today."

## Open Threads
Unresolved questions or things to follow up on. Bullet list. If none, write "All threads resolved."

## Intensity: X/10
One line: count, duration, and complexity assessment.

IMPORTANT:
- Write from YOUR perspective as Daemon ("Today we...", "Luke seemed...")
- Be factual - only include what actually happened in the conversations
- If a section has nothing relevant, acknowledge it briefly
- Keep it concise but informative
- Do NOT include any preamble or meta-commentary, just the note content
'''


class DailyNotesGenerator:
    """Generate daily summary notes from Daemon conversations."""

    def __init__(self, corpus_manager=None, model_manager=None, vault_path: str = None):
        """
        Initialize DailyNotesGenerator.

        Args:
            corpus_manager: CorpusManager instance (lazy-loaded if None)
            model_manager: ModelManager instance (lazy-loaded if None)
            vault_path: Path to Obsidian vault (defaults to config)
        """
        self._corpus_manager = corpus_manager
        self._model_manager = model_manager

        # Load config
        try:
            from config.app_config import (
                OBSIDIAN_VAULT_PATH,
                DAILY_NOTES_ENABLED,
                DAILY_NOTES_FOLDER,
                DAILY_NOTES_MODEL,
                DAILY_NOTES_MAX_TOKENS,
            )
            self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH).expanduser()
            self.enabled = DAILY_NOTES_ENABLED
            self.daily_folder = DAILY_NOTES_FOLDER
            self.model_name = DAILY_NOTES_MODEL
            self.max_tokens = DAILY_NOTES_MAX_TOKENS
        except ImportError:
            self.vault_path = Path(vault_path or "~/Documents/Luke Notes").expanduser()
            self.enabled = True
            self.daily_folder = "Daily"
            self.model_name = "gpt-4o-mini"
            self.max_tokens = 800

        self.output_dir = self.vault_path / self.daily_folder
        logger.debug(f"[DailyNotes] Initialized: vault={self.vault_path}, folder={self.daily_folder}")

    @property
    def corpus_manager(self):
        """Lazy-load CorpusManager."""
        if self._corpus_manager is None:
            try:
                from memory.corpus_manager import CorpusManager
                from config.app_config import CORPUS_FILE
                self._corpus_manager = CorpusManager(CORPUS_FILE)
                logger.debug("[DailyNotes] CorpusManager lazy-loaded")
            except Exception as e:
                logger.error(f"[DailyNotes] Failed to load CorpusManager: {e}")
                raise
        return self._corpus_manager

    @property
    def model_manager(self):
        """Lazy-load ModelManager."""
        if self._model_manager is None:
            try:
                from models.model_manager import ModelManager
                self._model_manager = ModelManager()
                logger.debug("[DailyNotes] ModelManager lazy-loaded")
            except Exception as e:
                logger.error(f"[DailyNotes] Failed to load ModelManager: {e}")
                raise
        return self._model_manager

    def _format_filename(self, target_date: date) -> str:
        """Format filename to match existing convention: 'M D YY Daily Note.md'."""
        return f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    def note_exists(self, target_date: date) -> bool:
        """Check if note already exists for date."""
        note_path = self.output_dir / self._format_filename(target_date)
        return note_path.exists()

    def _get_conversations_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Filter corpus entries by date."""
        conversations = []

        for entry in self.corpus_manager.corpus:
            ts = entry.get("timestamp")
            if ts is None:
                continue

            # Handle both datetime objects and ISO strings
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            elif not isinstance(ts, datetime):
                continue

            # Compare dates
            if ts.date() == target_date:
                conversations.append(entry)

        # Sort chronologically
        conversations.sort(key=lambda x: x.get("timestamp", datetime.min))
        return conversations

    def _format_conversations(self, convos: List[Dict[str, Any]]) -> str:
        """Format conversations for LLM prompt."""
        formatted = []

        for i, conv in enumerate(convos, 1):
            query = conv.get("query", "").strip()
            response = conv.get("response", "").strip()
            ts = conv.get("timestamp")

            # Format timestamp
            if isinstance(ts, datetime):
                time_str = ts.strftime("%H:%M")
            elif isinstance(ts, str):
                try:
                    time_str = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
                except ValueError:
                    time_str = "??:??"
            else:
                time_str = "??:??"

            # Truncate long messages
            q_max, a_max = 300, 400
            if len(query) > q_max:
                query = query[:q_max] + "..."
            if len(response) > a_max:
                response = response[:a_max] + "..."

            formatted.append(f"[{time_str}] Exchange {i}:\nUser: {query}\nDaemon: {response}")

        return "\n\n".join(formatted)

    def _calculate_intensity(self, convos: List[Dict[str, Any]], duration_hours: float) -> int:
        """
        Calculate intensity score 1-10 based on:
        - Conversation count
        - Duration
        - Average message length (complexity proxy)
        """
        if not convos:
            return 0

        count = len(convos)

        # Average message length as complexity proxy
        total_chars = sum(
            len(c.get("query", "")) + len(c.get("response", ""))
            for c in convos
        )
        avg_chars = total_chars / count if count > 0 else 0

        # Scoring components
        count_score = min(count / 20, 1.0) * 4  # Max 4 points for 20+ conversations
        duration_score = min(duration_hours / 4, 1.0) * 3  # Max 3 points for 4+ hours
        complexity_score = min(avg_chars / 1000, 1.0) * 3  # Max 3 points for avg 1000+ chars

        intensity = int(round(count_score + duration_score + complexity_score))
        return max(1, min(10, intensity))  # Clamp to 1-10

    def _extract_main_quest(self, llm_response: str) -> str:
        """Extract main quest title from LLM response for frontmatter."""
        # Look for "## Main Quest: [Title]"
        import re
        match = re.search(r'## Main Quest:\s*(.+?)(?:\n|$)', llm_response)
        if match:
            return match.group(1).strip()
        return "General Conversation"

    def _build_frontmatter(self, target_date: date, count: int, duration: float,
                           intensity: int, main_quest: str) -> str:
        """Generate YAML frontmatter."""
        return f"""---
date: {target_date.isoformat()}
intensity: {intensity}
conversations: {count}
duration_hours: {duration:.1f}
main_quest: "{main_quest}"
tags: [daily, daemon-generated]
generated: {datetime.now().isoformat()}
---
"""

    def _write_note(self, target_date: date, content: str) -> Path:
        """Atomic write to Daily/M D YY Daily Note.md."""
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        note_path = self.output_dir / self._format_filename(target_date)
        temp_path = note_path.with_suffix(".md.tmp")

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, note_path)
            logger.info(f"[DailyNotes] Written: {note_path}")
            return note_path
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def generate_for_date(self, target_date: date, force: bool = False) -> GenerationResult:
        """
        Generate daily note for specific date.

        Args:
            target_date: Date to generate note for
            force: If True, overwrite existing note

        Returns:
            GenerationResult with success status and details
        """
        result = GenerationResult(date=target_date)

        # Check if disabled
        if not self.enabled:
            result.skipped_reason = "disabled"
            logger.info(f"[DailyNotes] Skipped {target_date}: feature disabled")
            return result

        # Check if already exists
        if not force and self.note_exists(target_date):
            result.skipped_reason = "already_exists"
            result.output_path = self.output_dir / f"{target_date.isoformat()}.md"
            logger.info(f"[DailyNotes] Skipped {target_date}: note already exists")
            return result

        # Get conversations for date
        convos = self._get_conversations_for_date(target_date)
        result.conversation_count = len(convos)

        if not convos:
            result.skipped_reason = "no_conversations"
            logger.info(f"[DailyNotes] Skipped {target_date}: no conversations")
            return result

        # Calculate stats
        first_ts = convos[0].get("timestamp")
        last_ts = convos[-1].get("timestamp")

        if isinstance(first_ts, str):
            first_ts = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))

        duration_hours = (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else 0
        result.duration_hours = duration_hours
        result.intensity = self._calculate_intensity(convos, duration_hours)

        # Format for LLM
        formatted = self._format_conversations(convos)
        first_time = first_ts.strftime("%H:%M") if first_ts else "??:??"
        last_time = last_ts.strftime("%H:%M") if last_ts else "??:??"

        # Build prompt
        prompt = DAILY_NOTES_PROMPT.format(
            date=target_date.strftime("%B %d, %Y"),
            formatted_conversations=formatted,
            count=len(convos),
            first_time=first_time,
            last_time=last_time,
            duration_hours=duration_hours,
        )

        # Call LLM
        try:
            logger.info(f"[DailyNotes] Generating note for {target_date} ({len(convos)} conversations)")
            llm_response = await self.model_manager.generate_once(
                prompt,
                max_tokens=self.max_tokens,
                model_name=self.model_name,
            )
        except Exception as e:
            result.error = str(e)
            logger.error(f"[DailyNotes] LLM call failed: {e}")
            return result

        # Extract main quest for frontmatter
        main_quest = self._extract_main_quest(llm_response)

        # Build full note
        frontmatter = self._build_frontmatter(
            target_date, len(convos), duration_hours, result.intensity, main_quest
        )
        header = f"\n# Daily Note - {target_date.strftime('%B %d, %Y')}\n\n"
        full_content = frontmatter + header + llm_response.strip() + "\n"

        # Write to vault
        try:
            result.output_path = self._write_note(target_date, full_content)
            result.success = True
            logger.info(f"[DailyNotes] Success: {result.output_path}")
        except Exception as e:
            result.error = str(e)
            logger.error(f"[DailyNotes] Failed to write note: {e}")

        return result

    async def generate_yesterday_if_missing(self) -> Optional[GenerationResult]:
        """
        Startup catch-up: generate yesterday's note if missing.

        Returns:
            GenerationResult if generated, None if skipped
        """
        yesterday = date.today() - timedelta(days=1)

        if self.note_exists(yesterday):
            logger.debug(f"[DailyNotes] Yesterday's note exists, skipping catch-up")
            return None

        logger.info(f"[DailyNotes] Catch-up: generating yesterday's note ({yesterday})")
        return await self.generate_for_date(yesterday)

    async def generate_date_range(self, start_date: date, end_date: date,
                                   force: bool = False) -> List[GenerationResult]:
        """
        Generate notes for a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            force: If True, overwrite existing notes

        Returns:
            List of GenerationResult for each date
        """
        results = []
        current = start_date

        while current <= end_date:
            result = await self.generate_for_date(current, force=force)
            results.append(result)
            current += timedelta(days=1)

        return results
