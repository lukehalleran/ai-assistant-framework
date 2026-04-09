# utils/daily_notes_generator.py
"""
DailyNotesGenerator - Generate daily summary notes from Daemon conversations.

Module Contract:
- Purpose: Automatically generate structured daily notes from conversation history,
           written from Daemon's perspective in YAML + bullet format with LLM-based tag generation.
- Inputs:
  - generate_for_date(date: date) -> GenerationResult: Generate note for specific date
  - generate_yesterday_if_missing() -> Optional[GenerationResult]: Startup catch-up (called by GUI on launch)
  - note_exists(date: date) -> bool: Check if note already exists
- Outputs:
  - Markdown files in Obsidian vault: ~/Documents/Luke Notes/Daily/M D YY Daily Note.md
  - GenerationResult with success status, path, stats
- Behavior:
  - Filters corpus by timestamp to get day's conversations
  - Calculates active duration (estimated actual usage time, not wall-clock span) [NEW 2026-01-18]
  - Calls LLM to generate structured summary:
    - Main Quest, Side Quests, Emotional State, Key Decisions, etc.
    - Life Events section: Work, Study, Sleep, Exercise, Other [NEW 2026-01-18]
  - Generates contextual tags using TagGenerator (3-10 tags) [NEW 2026-01-22]
  - Writes atomic markdown with YAML frontmatter:
    - usage_intensity (was: intensity) [RENAMED 2026-01-18]
    - span_hours (wall-clock), active_hours (estimated usage) [NEW 2026-01-18]
    - tags: system tags + content tags [NEW 2026-01-22]
  - Idempotent: skips if note already exists (unless force=True)
  - Triggers narrative context refresh after daily note creation [NEW 2026-01-17]
- Key Methods:
  - _calculate_active_duration(convos): Estimate actual usage time [NEW 2026-01-18]
    - Reading time: ~200 words/min for responses
    - Typing time: ~40 words/min for queries
    - Gap time: Capped at 30 seconds (idle time excluded)
  - _calculate_intensity(convos, active_hours): 1-10 score based on count/active duration/complexity
- Dependencies:
  - memory.corpus_manager (conversation retrieval, narrative context persistence)
  - models.model_manager (LLM generation)
  - utils.tag_generator (tag generation for notes) [NEW 2026-01-22]
  - memory.memory_consolidator (narrative synthesis) [NEW 2026-01-17]
  - config.app_config (paths and settings)
- Side effects:
  - Writes files to Obsidian vault
  - LLM API calls (note generation + tag generation)
  - Logging
  - _trigger_narrative_refresh(): Regenerates narrative context after daily note creation [NEW 2026-01-17]
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


# ---------------------------------------------------------------------------
# Module-level path resolvers (shared by DailyNotesGenerator and STM injection)
# ---------------------------------------------------------------------------
# These exist as module-level functions so callers (e.g. STMAnalyzer) can read
# daily notes without instantiating DailyNotesGenerator (which would lazy-load
# corpus_manager + model_manager). Both helpers are read-only and never trigger
# generation or narrative refresh.

def get_daily_note_path(target_date: date, vault_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve the actual on-disk path for a daily note, checking all known
    layouts (flat, weekly-at-root, monthly/weekly).

    Returns None if no file exists for the given date.
    """
    if vault_path is None:
        try:
            from config.app_config import OBSIDIAN_VAULT_PATH, DAILY_NOTES_FOLDER
            base = Path(OBSIDIAN_VAULT_PATH).expanduser() / DAILY_NOTES_FOLDER
        except ImportError:
            return None
    else:
        base = Path(vault_path).expanduser()

    filename = f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    # Layout 1: flat
    p = base / filename
    if p.exists():
        return p

    # Layout 2: weekly folder at root (legacy)
    monday = target_date - timedelta(days=target_date.weekday())
    week_num = monday.isocalendar()[1]
    week_folder = f"Week {week_num} {monday.strftime('%b %Y')}"
    p = base / week_folder / filename
    if p.exists():
        return p

    # Layout 3: monthly/weekly folder (current)
    month_folder = f"{monday.strftime('%B %Y')}"
    p = base / month_folder / week_folder / filename
    if p.exists():
        return p

    return None


def read_daily_note(target_date: date, vault_path: Optional[Path] = None) -> Optional[str]:
    """Read the markdown text for a daily note, or None if missing or unreadable."""
    p = get_daily_note_path(target_date, vault_path)
    if p is None:
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[DailyNotes] Failed to read {p}: {e}")
        return None


# LLM prompt template
DAILY_NOTES_PROMPT = '''You are Daemon, an AI companion writing a daily note about your conversations with Luke today.

CONVERSATIONS FROM {date}:
{formatted_conversations}

STATISTICS:
- Conversations: {count}
- Time span: {first_time} to {last_time} ({span_hours:.1f} hours wall-clock)
- Active time: {active_hours:.1f} hours (estimated actual usage)

Write a daily note in this EXACT structure:

## Summary
2-3 sentences capturing the day's theme from your perspective as Daemon.

## Main Quest: [Primary Focus]
The main thing we worked on today. 3-5 bullets about progress, challenges, outcomes.

## Side Quests
Other topics discussed. Format: **Topic**: one-line description. If only one main topic, write "None today - fully focused on the Main Quest."

## Life Events
Track these specific life activities ONLY if mentioned in conversations. Do NOT assume they didn't happen if not mentioned - just note what was or wasn't discussed:

- **Work**: If mentioned: duration, what was done, how it went. If not mentioned: "Not discussed today."
- **Study**: If mentioned: what subject/material, duration, how it went, any progress. If not mentioned: "Not discussed today."
- **Sleep**: If mentioned: quality, duration, any issues. If not mentioned: "Not discussed today."
- **Exercise/Health**: If mentioned: what activity, how it went. If not mentioned: "Not discussed today."
- **Other Events**: Any other notable life events mentioned (social activities, appointments, errands, etc.). If none mentioned: "None mentioned."

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
- For Life Events: ONLY report what was explicitly mentioned. Never assume activities happened or didn't happen. If Luke didn't mention work/study/sleep, say "Not discussed today" - do NOT say "Luke didn't work today."
- If a section has nothing relevant, acknowledge it briefly
- Keep it concise but informative
- Do NOT include any preamble or meta-commentary, just the note content
'''


class DailyNotesGenerator:
    """Generate daily summary notes from Daemon conversations."""

    def __init__(self, corpus_manager=None, model_manager=None, vault_path: str = None, tag_generator=None):
        """
        Initialize DailyNotesGenerator.

        Args:
            corpus_manager: CorpusManager instance (lazy-loaded if None)
            model_manager: ModelManager instance (lazy-loaded if None)
            vault_path: Path to Obsidian vault (defaults to config)
            tag_generator: TagGenerator instance (lazy-loaded if None)
        """
        self._corpus_manager = corpus_manager
        self._model_manager = model_manager
        self._tag_generator = tag_generator

        # Load config
        try:
            from config.app_config import (
                OBSIDIAN_VAULT_PATH,
                DAILY_NOTES_ENABLED,
                DAILY_NOTES_FOLDER,
                DAILY_NOTES_MODEL,
                DAILY_NOTES_MAX_TOKENS,
                TAG_GENERATION_ENABLED,
            )
            self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH).expanduser()
            self.enabled = DAILY_NOTES_ENABLED
            self.daily_folder = DAILY_NOTES_FOLDER
            self.model_name = DAILY_NOTES_MODEL
            self.max_tokens = DAILY_NOTES_MAX_TOKENS
            self.tag_generation_enabled = TAG_GENERATION_ENABLED
        except ImportError:
            self.vault_path = Path(vault_path or "~/Documents/Luke Notes").expanduser()
            self.enabled = True
            self.daily_folder = "Daily"
            self.model_name = "sonnet-4.5"
            self.max_tokens = 800
            self.tag_generation_enabled = True

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

    @property
    def tag_generator(self):
        """Lazy-load TagGenerator."""
        if self._tag_generator is None:
            try:
                from utils.tag_generator import TagGenerator
                self._tag_generator = TagGenerator(model_manager=self.model_manager)
                logger.debug("[DailyNotes] TagGenerator lazy-loaded")
            except Exception as e:
                logger.warning(f"[DailyNotes] Failed to load TagGenerator: {e}")
                # Non-critical, can continue without tag generation
                self._tag_generator = None
        return self._tag_generator

    def _format_filename(self, target_date: date) -> str:
        """Format filename to match existing convention: 'M D YY Daily Note.md'."""
        return f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    def _get_week_folder_name(self, target_date: date) -> str:
        """Format weekly folder name: 'Week 3 Jan 2026'."""
        monday = target_date - timedelta(days=target_date.weekday())
        week_num = monday.isocalendar()[1]
        return f"Week {week_num} {monday.strftime('%b %Y')}"

    def _get_month_folder_name(self, target_date: date) -> str:
        """Format monthly folder name based on Monday of the week: 'January 2026'."""
        monday = target_date - timedelta(days=target_date.weekday())
        return f"{monday.strftime('%B %Y')}"

    def _get_note_path(self, target_date: date) -> Path:
        """Get note path - uses monthly/weekly folder structure."""
        filename = self._format_filename(target_date)
        month_folder = self._get_month_folder_name(target_date)
        week_folder = self._get_week_folder_name(target_date)
        return self.output_dir / month_folder / week_folder / filename

    def note_exists(self, target_date: date) -> bool:
        """Check if note already exists for date (flat, weekly-at-root, or monthly/weekly)."""
        filename = self._format_filename(target_date)

        # Check flat directory
        if (self.output_dir / filename).exists():
            return True

        # Check weekly folder at root (legacy layout)
        week_folder = self._get_week_folder_name(target_date)
        if (self.output_dir / week_folder / filename).exists():
            return True

        # Check monthly/weekly folder (new layout)
        month_folder = self._get_month_folder_name(target_date)
        if (self.output_dir / month_folder / week_folder / filename).exists():
            return True

        return False

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

    def _calculate_active_duration(self, convos: List[Dict[str, Any]]) -> float:
        """
        Calculate active usage duration in hours.

        For each conversation:
        - Estimate reading time: ~200 words/min for response
        - Estimate typing time: ~40 words/min for query
        - Cap gap between conversations at 30 seconds (idle time doesn't count)

        Returns:
            Active duration in hours
        """
        if not convos:
            return 0.0

        total_seconds = 0.0
        MAX_GAP_SECONDS = 30  # Cap idle time between exchanges

        for i, conv in enumerate(convos):
            query = conv.get("query", "")
            response = conv.get("response", "")

            # Estimate time for this exchange
            # Reading response: ~200 words/min = ~1000 chars/min
            read_time = len(response) / 1000 * 60  # seconds
            # Typing query: ~40 words/min = ~200 chars/min
            type_time = len(query) / 200 * 60  # seconds
            # Minimum 10 seconds per exchange
            exchange_time = max(10, read_time + type_time)

            total_seconds += exchange_time

            # Add capped gap time to next conversation
            if i < len(convos) - 1:
                curr_ts = conv.get("timestamp")
                next_ts = convos[i + 1].get("timestamp")

                if curr_ts and next_ts:
                    # Parse timestamps
                    if isinstance(curr_ts, str):
                        curr_ts = datetime.fromisoformat(curr_ts.replace("Z", "+00:00"))
                    if isinstance(next_ts, str):
                        next_ts = datetime.fromisoformat(next_ts.replace("Z", "+00:00"))

                    if isinstance(curr_ts, datetime) and isinstance(next_ts, datetime):
                        gap = (next_ts - curr_ts).total_seconds()
                        # Only count up to MAX_GAP_SECONDS of idle time
                        total_seconds += min(gap, MAX_GAP_SECONDS)

        return total_seconds / 3600  # Convert to hours

    def _calculate_intensity(self, convos: List[Dict[str, Any]], active_hours: float) -> int:
        """
        Calculate intensity score 1-10 based on:
        - Conversation count
        - Active duration (not wall-clock time)
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
        duration_score = min(active_hours / 2, 1.0) * 3  # Max 3 points for 2+ active hours
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

    def _build_frontmatter(self, target_date: date, count: int, span_hours: float,
                           active_hours: float, intensity: int, main_quest: str,
                           content_tags: List[str] = None) -> str:
        """Generate YAML frontmatter."""
        # System tags are always included
        system_tags = ['daily', 'daemon-generated']

        # Add content tags if provided
        if content_tags:
            all_tags = system_tags + content_tags
        else:
            all_tags = system_tags

        # Format tags for YAML (quoted strings to handle hyphens)
        tags_str = ', '.join(f'"{tag}"' for tag in all_tags)

        return f"""---
date: {target_date.isoformat()}
usage_intensity: {intensity}
conversations: {count}
span_hours: {span_hours:.1f}
active_hours: {active_hours:.1f}
main_quest: "{main_quest}"
tags: [{tags_str}]
generated: {datetime.now().isoformat()}
---
"""

    def _write_note(self, target_date: date, content: str) -> Path:
        """Atomic write to Daily/M D YY Daily Note.md (or weekly folder if exists)."""
        # Get the appropriate path (weekly folder or flat)
        note_path = self._get_note_path(target_date)

        # Ensure parent directory exists
        note_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Wall-clock span (for display: "10:00 to 20:00")
        span_hours = (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else 0
        # Active duration (estimated actual usage time)
        active_hours = self._calculate_active_duration(convos)
        result.duration_hours = active_hours  # Use active duration for stats
        result.intensity = self._calculate_intensity(convos, active_hours)

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
            span_hours=span_hours,
            active_hours=active_hours,
        )

        # Call LLM with fallback models
        # Try primary model first, then fallback to alternatives if it fails
        # Expanded list includes Claude, Gemini, and newer GPT models for better reliability
        fallback_models = [
            "sonnet-4.5",       # Anthropic Claude (fast)
            "gpt-4o-mini",       # Fast, cheap OpenAI
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
                logger.info(f"[DailyNotes] Generating note for {target_date} ({len(convos)} conversations) using {model}")
                llm_response = await self.model_manager.generate_once(
                    prompt,
                    max_tokens=self.max_tokens,
                    model_name=model,
                )

                # Check for API error responses (model_manager returns these as content, not exceptions)
                if llm_response and llm_response.startswith("[API unavailable]"):
                    logger.warning(f"[DailyNotes] Model {model} unavailable, trying next...")
                    last_error = f"Model {model} unavailable"
                    llm_response = None
                    continue

                # Check for empty response (like glm-4.6 returning nothing)
                if llm_response and llm_response.startswith("[Error:"):
                    logger.warning(f"[DailyNotes] Model {model} returned error, trying next...")
                    last_error = f"Model {model} returned error"
                    llm_response = None
                    continue

                # Validate we got a reasonable response (not empty or just the prompt echoed back)
                if not llm_response or len(llm_response.strip()) < 100:
                    logger.warning(f"[DailyNotes] Model {model} returned too-short response ({len(llm_response) if llm_response else 0} chars), trying next...")
                    last_error = f"Model {model} returned too-short response"
                    llm_response = None
                    continue

                # Success! Break out of the loop
                logger.info(f"[DailyNotes] Successfully generated note using {model}")
                break

            except Exception as e:
                logger.warning(f"[DailyNotes] Model {model} failed: {e}, trying next...")
                last_error = str(e)
                llm_response = None
                continue

        # If all models failed, return error
        if not llm_response:
            result.error = f"All LLM models failed. Last error: {last_error}"
            logger.error(f"[DailyNotes] All models failed for {target_date}: {last_error}")
            return result

        # Extract main quest for frontmatter
        main_quest = self._extract_main_quest(llm_response)

        # Generate contextual tags
        content_tags = []
        if self.tag_generation_enabled and self.tag_generator:
            try:
                tag_metadata = {
                    'main_quest': main_quest,
                    'intensity': result.intensity,
                    'conversations': len(convos),
                    'active_hours': active_hours,
                }
                tag_result = await self.tag_generator.generate_tags(
                    llm_response,
                    note_type="daily",
                    metadata=tag_metadata
                )
                content_tags = tag_result.tags
                logger.info(f"[DailyNotes] Generated {len(content_tags)} tags: {', '.join(content_tags)}")
            except Exception as e:
                logger.warning(f"[DailyNotes] Tag generation failed: {e}, continuing without tags")

        # Build full note
        frontmatter = self._build_frontmatter(
            target_date, len(convos), span_hours, active_hours, result.intensity, main_quest,
            content_tags=content_tags
        )
        header = f"\n# Daily Note - {target_date.strftime('%B %d, %Y')}\n\n"
        full_content = frontmatter + header + llm_response.strip() + "\n"

        # Write to vault
        try:
            result.output_path = self._write_note(target_date, full_content)
            result.success = True
            logger.info(f"[DailyNotes] Success: {result.output_path}")

            # Trigger narrative context refresh after new daily note
            await self._trigger_narrative_refresh()

        except Exception as e:
            result.error = str(e)
            logger.error(f"[DailyNotes] Failed to write note: {e}")

        return result

    async def _trigger_narrative_refresh(self) -> None:
        """
        Trigger narrative context refresh after daily note creation.

        This is a non-blocking operation - failures are logged but don't
        affect the daily note generation result.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED
            if not NARRATIVE_CONTEXT_ENABLED:
                return

            from memory.memory_consolidator import MemoryConsolidator
            consolidator = MemoryConsolidator(self.model_manager)

            # Generate narrative from Obsidian notes (no corpus fallback needed for fresh trigger)
            narrative = await consolidator.generate_narrative_context()

            if narrative:
                self.corpus_manager.save_narrative_context(narrative)
                logger.info("[DailyNotes] Narrative context refreshed after daily note creation")

        except Exception as e:
            # Non-critical: log and continue
            logger.warning(f"[DailyNotes] Failed to refresh narrative context: {e}")

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
