"""
# memory/memory_consolidator.py

Module Contract
- Purpose: Generate concise conversation summaries from recent exchanges. Used by shutdown summarizer and optionally mid‑session consolidation. Also synthesizes narrative context from Obsidian monthly/weekly/daily notes.
- Inputs:
  - consolidation_threshold (N): target block size for summaries
  - maybe_consolidate(corpus_manager): returns True if a new summary node is stored
  - generate_narrative_context(weeklies?, monthlies?, max_tokens?) -> str [NEW 2026-01-17]
- Outputs:
  - Creates a new summary node via corpus_manager.add_summary when due.
  - Synthesized narrative context string (temporal grounding) [NEW 2026-01-17]
- Dependencies:
  - models.model_manager for generate_once
  - config.app_config for OBSIDIAN_VAULT_PATH, NARRATIVE_SYNTHESIS_MODEL [NEW 2026-01-17]
- Side effects:
  - Writes summary nodes to corpus (and by caller pathways, to Chroma).
  - Reads Obsidian monthly/weekly/daily notes from vault [NEW 2026-01-17]
  - LLM API calls for narrative synthesis [NEW 2026-01-17]

Narrative Context System (2026-01-17, updated 2026-03-10 for 3-tier):
  - _get_obsidian_notes_path() -> Optional[str]: Discover vault daily notes folder
  - _read_obsidian_monthly_summaries(limit) -> List[Dict]: Parse <Month YYYY> Summary.md files
  - _read_obsidian_weekly_summaries(limit) -> List[Dict]: Parse Week * Summary.md files
  - _read_obsidian_daily_notes(limit) -> List[Dict]: Parse *Daily Note.md from Week * folders
  - generate_narrative_context(): Synthesize "Current Life State" narrative from notes
  - Output sections: Current Chapter, Active Threads, Emotional Trajectory, Recurring Themes
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger("memory_consolidator")

SUMMARY_EVERY_N = int(os.getenv("SUMMARY_EVERY_N", "20"))
SUMMARY_LOOKBACK = int(os.getenv("SUMMARY_LOOKBACK", "40"))
SUMMARY_MIN_GAP_MIN = int(os.getenv("SUMMARY_MIN_GAP_MIN", "20"))
SUMMARY_MODEL_ALIAS = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "220"))


def _format_recent_for_summary(recent: List[Dict[str, Any]],
                               q_max: int = 240,
                               a_max: int = 300) -> List[str]:
    """
    Build short conversation slices from recent corpus entries for summarization.
    Each slice is of the form:
      "User: <q>\nAssistant: <a>"
    with conservative clipping to avoid overly long prompts.
    """
    out: List[str] = []
    if not recent:
        return out
    for e in recent:
        try:
            q = (e.get("query") or "").strip()
            a = (e.get("response") or "").strip()
            if not (q or a):
                continue
            if len(q) > q_max:
                q = q[:q_max]
            if len(a) > a_max:
                a = a[:a_max]
            out.append(f"User: {q}\nAssistant: {a}")
        except Exception:
            # best-effort; skip malformed entries
            continue
    return out

class MemoryConsolidator:
    def __init__(
        self,
        model_manager,
        *,
        consolidation_threshold: int | None = None,
        lookback: int | None = None,
        min_gap_minutes: int | None = None,
        **_unused,  # tolerate extra kwargs from older callers
    ):
        """
        consolidation_threshold: how many exchanges between stored summaries (default from env SUMMARY_EVERY_N)
        lookback:                how many recent exchanges to compress (default from env SUMMARY_LOOKBACK)
        min_gap_minutes:         minimum minutes between stored summaries (default from env SUMMARY_MIN_GAP_MIN)
        """
        self.model_manager = model_manager

        # Prefer explicit args from caller; fall back to env defaults
        self.consolidation_threshold = consolidation_threshold or SUMMARY_EVERY_N
        self.lookback = lookback or SUMMARY_LOOKBACK
        self.min_gap_minutes = min_gap_minutes or SUMMARY_MIN_GAP_MIN

        logger.debug(
            "[Consolidator] init threshold=%s lookback=%s min_gap_min=%s",
            self.consolidation_threshold, self.lookback, self.min_gap_minutes
        )

    async def maybe_consolidate(self, corpus_manager) -> bool:
        """
        Check counters/time and, if due, generate+store a summary memory node.
        Returns True if a new summary was stored.
        """
        try:
            # 1) Throttle by time
            last = getattr(corpus_manager, "get_last_summary_meta", lambda: None)()
            if last and isinstance(last, dict):
                ts = last.get("timestamp")
                if ts and isinstance(ts, datetime):
                    if datetime.now() - ts < timedelta(minutes=self.min_gap_minutes):
                        logger.debug("[Consolidation] Skipped (min gap not elapsed)")
                        return False

            # 2) Count since last summary
            num_since_fn = getattr(corpus_manager, "count_exchanges_since_last_summary", None)
            if callable(num_since_fn):
                since = num_since_fn()
                if since < self.consolidation_threshold:
                    logger.debug(
                        "[Consolidation] Skipped (%s/%s since last summary)",
                        since, self.consolidation_threshold
                    )
                    return False
            else:
                # Fallback: use recent length as proxy
                recent_probe = corpus_manager.get_recent_memories(self.lookback)
                if len(recent_probe) < self.consolidation_threshold:
                    logger.debug(
                        "[Consolidation] Skipped (proxy %s/%s)",
                        len(recent_probe), self.consolidation_threshold
                    )
                    return False

            # 3) Gather the recent exchanges to compress
            recent = corpus_manager.get_recent_memories(self.lookback)
            convo_slices = _format_recent_for_summary(recent)
            if not convo_slices:
                logger.debug("[Consolidation] No material to summarize")
                return False

            # 4) LLM summarize (bounded by your model_manager’s timeout policies)
            excerpts = "\n\n".join(convo_slices)
            prompt = (
                "You are an extractive note-taker. Using ONLY the EXCERPTS below, write 3–5 factual bullets. "
                "Do NOT infer or invent anything not present. If information is minimal, output 1–2 bullets that "
                "quote or paraphrase the text. No headers, just bullets.\n\nEXCERPTS:\n" + excerpts + "\n\nBullets:"
            )
            if not hasattr(self.model_manager, "generate_once"):
                logger.debug("[Consolidation] No generate_once on model_manager")
                return False

            text = await self.model_manager.generate_once(prompt, max_tokens=SUMMARY_MAX_TOKENS)
            text = (text or "").strip()
            if not text:
                logger.debug("[Consolidation] LLM returned empty text")
                return False

            # 5) Persist as a summary node
            add_summary = getattr(corpus_manager, "add_summary", None)
            if not callable(add_summary):
                logger.debug("[Consolidation] corpus_manager.add_summary not available")
                return False

            node = {
                "content": text,
                "timestamp": datetime.now(),
                "type": "summary",
                "tags": ["summary:consolidated", "source:consolidator"],
            }
            add_summary(node)
            logger.info("[Consolidation] Stored new summary node")
            return True

        except Exception as e:
            logger.debug(f"[Consolidation] Error: {e}")
            return False

    # --- Narrative Context Synthesis ---
    # Uses Obsidian daily/weekly notes as primary source, corpus summaries as fallback
    NARRATIVE_SYNTHESIS_PROMPT = """You are synthesizing the user's recent life context from monthly, weekly, and daily notes.

MONTHLY SUMMARY (broadest context — life arc over past month):
{monthly_summaries}

WEEKLY SUMMARIES (medium context — active threads, most recent first):
{weekly_summaries}

DAILY NOTES (immediate context — last few days, most recent first):
{daily_notes}

CORPUS SUMMARIES (fallback, if available):
{corpus_summaries}

Synthesize a 250-300 word "Current Life State" narrative covering:
1. CURRENT CHAPTER: What life phase is the user in? Use the monthly summary for arc. (1-2 sentences)
2. ACTIVE THREADS: What's actively in progress this week? Use weekly summaries. (bullet list, 3-5 items)
3. EMOTIONAL TRAJECTORY: How is mood/stress trending? Use daily notes for recent shifts. (2-3 sentences)
4. RECURRING THEMES: Patterns visible across all three time scales.

TODAY'S DATE: {today}

TEMPORAL ACCURACY RULES — these are critical:
- Each daily note has a date in its frontmatter. Use those dates as ground truth.
- When a note references events from EARLIER days ("yesterday was rough", "on Saturday I..."),
  attribute those events to the correct calendar date, not the note's own date.
- For multi-day events (illness, travel, projects), state the full date range explicitly
  (e.g. "sick May 12-14", "started homework at 2:30 on May 15") — never collapse to a single day.
- ALWAYS use absolute dates (e.g. "May 18-20"), never vague relative terms like "recently" or "a few days ago".
- Use day-of-week names anchored to actual dates, not vague relative terms.

Write in third person ("The user is..."). Be specific and grounded in the data.
Prioritize recent daily notes for current state, weekly for active threads, monthly for trajectory.
Do NOT make up information not present in the summaries."""

    def _get_obsidian_notes_path(self) -> Optional[str]:
        """Get the path to Obsidian daily notes folder."""
        try:
            from config.app_config import OBSIDIAN_VAULT_PATH
            from pathlib import Path
            vault_path = Path(OBSIDIAN_VAULT_PATH).expanduser()
            notes_path = vault_path / "Vault" / "Daily Notes and To Do's"
            if notes_path.exists():
                return str(notes_path)
            # Fallback: try without "Vault" subfolder
            notes_path = vault_path / "Daily Notes and To Do's"
            if notes_path.exists():
                return str(notes_path)
            return None
        except Exception as e:
            logger.debug(f"[NarrativeSynthesis] Could not get Obsidian path: {e}")
            return None

    def _read_obsidian_weekly_summaries(self, limit: int = 2) -> List[Dict]:
        """
        Read recent weekly summaries from Obsidian vault.

        Returns list of dicts with 'content' and 'timestamp' keys.
        """
        from pathlib import Path
        import re

        notes_path = self._get_obsidian_notes_path()
        if not notes_path:
            return []

        try:
            notes_dir = Path(notes_path)
            weekly_summaries = []

            # Find all "Week N Mon YYYY" folders (at root and inside monthly parents)
            week_folders = []
            for folder in notes_dir.iterdir():
                if folder.is_dir() and folder.name.startswith("Week "):
                    # Legacy: weekly folder at root
                    week_folders.append(folder)
                elif folder.is_dir() and not folder.name.startswith("Week "):
                    # New layout: monthly parent containing weekly folders
                    for subfolder in folder.iterdir():
                        if subfolder.is_dir() and subfolder.name.startswith("Week "):
                            week_folders.append(subfolder)

            # Sort by folder name (most recent first based on modification time)
            week_folders.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for folder in week_folders[:limit]:
                # Look for summary file: "Week N Mon YYYY Summary.md"
                summary_files = list(folder.glob("*Summary.md"))
                if summary_files:
                    summary_file = summary_files[0]
                    content = summary_file.read_text(encoding='utf-8')

                    # Extract timestamp from YAML frontmatter if present
                    timestamp = ""
                    if content.startswith("---"):
                        try:
                            end_idx = content.find("---", 3)
                            if end_idx > 0:
                                frontmatter = content[3:end_idx]
                                for line in frontmatter.split("\n"):
                                    if line.startswith("generated:"):
                                        timestamp = line.split(":", 1)[1].strip()
                                        break
                                    elif line.startswith("start_date:"):
                                        timestamp = line.split(":", 1)[1].strip()
                        except Exception:
                            pass

                    weekly_summaries.append({
                        "content": content,
                        "timestamp": timestamp or folder.name,
                        "source": "obsidian_weekly"
                    })

            logger.debug(f"[NarrativeSynthesis] Found {len(weekly_summaries)} Obsidian weekly summaries")
            return weekly_summaries

        except Exception as e:
            logger.debug(f"[NarrativeSynthesis] Error reading weekly summaries: {e}")
            return []

    def _read_obsidian_monthly_summaries(self, limit: int = 1) -> List[Dict]:
        """
        Read recent monthly summaries from Obsidian vault.

        Looks for files like "February 2026/February 2026 Summary.md" inside
        the daily-notes root. Returns list of dicts with 'content' and
        'timestamp' keys, sorted by mtime (most recent first).
        """
        from pathlib import Path

        notes_path = self._get_obsidian_notes_path()
        if not notes_path:
            return []

        try:
            notes_dir = Path(notes_path)
            monthly_summaries = []

            for folder in notes_dir.iterdir():
                if not folder.is_dir() or folder.name.startswith("Week "):
                    continue
                # Monthly parent folder (e.g. "February 2026")
                summary_files = list(folder.glob("*Summary.md"))
                if not summary_files:
                    continue
                summary_file = summary_files[0]
                content = summary_file.read_text(encoding="utf-8")

                # Extract timestamp from YAML frontmatter if present
                timestamp = ""
                if content.startswith("---"):
                    try:
                        end_idx = content.find("---", 3)
                        if end_idx > 0:
                            frontmatter = content[3:end_idx]
                            for line in frontmatter.split("\n"):
                                if line.startswith("generated:"):
                                    timestamp = line.split(":", 1)[1].strip()
                                    break
                                elif line.startswith("start_date:"):
                                    timestamp = line.split(":", 1)[1].strip()
                    except Exception:
                        pass

                monthly_summaries.append({
                    "content": content,
                    "timestamp": timestamp or folder.name,
                    "source": "obsidian_monthly",
                    "_mtime": summary_file.stat().st_mtime,
                })

            # Sort by mtime (most recent first), then trim
            monthly_summaries.sort(key=lambda s: s.get("_mtime", 0), reverse=True)
            for s in monthly_summaries:
                s.pop("_mtime", None)

            logger.debug(f"[NarrativeSynthesis] Found {len(monthly_summaries)} Obsidian monthly summaries")
            return monthly_summaries[:limit]

        except Exception as e:
            logger.debug(f"[NarrativeSynthesis] Error reading monthly summaries: {e}")
            return []

    def _read_obsidian_daily_notes(self, limit: int = 7) -> List[Dict]:
        """
        Read recent daily notes from Obsidian vault.

        Only reads from Week * folders (new auto-generated system created 2026-01-15).
        Older notes in root folder are excluded - they're searchable via personal_notes.

        Returns list of dicts with 'content' and 'timestamp' keys.
        """
        from pathlib import Path

        notes_path = self._get_obsidian_notes_path()
        if not notes_path:
            return []

        try:
            notes_dir = Path(notes_path)
            daily_notes = []

            # Collect daily notes from Week * folders (at root and inside monthly parents)
            all_daily_files = []

            for folder in notes_dir.iterdir():
                if folder.is_dir() and folder.name.startswith("Week "):
                    # Legacy: weekly folder at root
                    for f in folder.glob("*Daily Note.md"):
                        all_daily_files.append(f)
                elif folder.is_dir() and not folder.name.startswith("Week "):
                    # New layout: monthly parent containing weekly folders
                    for subfolder in folder.iterdir():
                        if subfolder.is_dir() and subfolder.name.startswith("Week "):
                            for f in subfolder.glob("*Daily Note.md"):
                                all_daily_files.append(f)

            # Sort by modification time (most recent first)
            all_daily_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for daily_file in all_daily_files[:limit]:
                content = daily_file.read_text(encoding='utf-8')

                # Extract date from YAML frontmatter if present
                timestamp = ""
                if content.startswith("---"):
                    try:
                        end_idx = content.find("---", 3)
                        if end_idx > 0:
                            frontmatter = content[3:end_idx]
                            for line in frontmatter.split("\n"):
                                if line.startswith("date:"):
                                    timestamp = line.split(":", 1)[1].strip()
                                    break
                    except Exception:
                        pass

                daily_notes.append({
                    "content": content,
                    "timestamp": timestamp or daily_file.stem,
                    "source": "obsidian_daily"
                })

            logger.debug(f"[NarrativeSynthesis] Found {len(daily_notes)} Obsidian daily notes")
            return daily_notes

        except Exception as e:
            logger.debug(f"[NarrativeSynthesis] Error reading daily notes: {e}")
            return []

    async def generate_narrative_context(
        self,
        recent_weeklies: List[Dict] = None,
        recent_monthlies: List[Dict] = None,
        max_tokens: int = 400
    ) -> str:
        """
        Synthesize monthly/weekly/daily notes into a 'Current Life State' narrative.

        Uses Obsidian monthly/weekly/daily notes as primary source (3-tier),
        with corpus summaries as fallback for historical data.

        Args:
            recent_weeklies: Corpus weekly summaries (fallback, optional)
            recent_monthlies: Corpus monthly summaries (fallback, optional)
            max_tokens: Max tokens for output (default 400, ~300 words)

        Returns:
            Synthesized narrative string, or empty string on failure
        """
        try:
            from config.app_config import (
                NARRATIVE_SYNTHESIS_MODEL, NARRATIVE_MONTHLIES_COUNT,
                NARRATIVE_WEEKLIES_COUNT, NARRATIVE_DAILIES_COUNT,
            )

            # Primary: Read from Obsidian vault (3-tier)
            obsidian_monthlies = self._read_obsidian_monthly_summaries(limit=NARRATIVE_MONTHLIES_COUNT)
            obsidian_weeklies = self._read_obsidian_weekly_summaries(limit=NARRATIVE_WEEKLIES_COUNT)
            obsidian_dailies = self._read_obsidian_daily_notes(limit=NARRATIVE_DAILIES_COUNT)

            # Format Obsidian monthly summaries
            if obsidian_monthlies:
                monthly_text = "\n\n---\n\n".join([
                    f"[{s.get('timestamp', 'Unknown')}]\n{s.get('content', '')}"
                    for s in obsidian_monthlies
                ])
            else:
                monthly_text = "(No monthly summaries available)"

            # Format Obsidian weekly summaries
            if obsidian_weeklies:
                weekly_text = "\n\n---\n\n".join([
                    f"[{s.get('timestamp', 'Unknown')}]\n{s.get('content', '')}"
                    for s in obsidian_weeklies
                ])
            else:
                weekly_text = "(No weekly summaries available)"

            # Format Obsidian daily notes
            if obsidian_dailies:
                daily_text = "\n\n---\n\n".join([
                    f"[{s.get('timestamp', 'Unknown')}]\n{s.get('content', '')}"
                    for s in obsidian_dailies
                ])
            else:
                daily_text = "(No daily notes available)"

            # Fallback: Format corpus summaries
            corpus_text = "(No corpus summaries)"
            if recent_weeklies or recent_monthlies:
                corpus_parts = []
                for s in (recent_weeklies or [])[:3]:
                    corpus_parts.append(f"[{s.get('timestamp', 'Unknown')}]\n{s.get('content', s.get('response', ''))}")
                for s in (recent_monthlies or [])[:2]:
                    corpus_parts.append(f"[{s.get('timestamp', 'Unknown')}]\n{s.get('content', s.get('response', ''))}")
                if corpus_parts:
                    corpus_text = "\n\n".join(corpus_parts)

            # Skip if no meaningful content from any source
            if not obsidian_monthlies and not obsidian_weeklies and not obsidian_dailies and not recent_weeklies and not recent_monthlies:
                logger.debug("[NarrativeSynthesis] No content available for synthesis")
                return ""

            # Build prompt from template
            from datetime import date as _date
            prompt = self.NARRATIVE_SYNTHESIS_PROMPT.format(
                today=_date.today().strftime("%A, %B %d, %Y"),
                monthly_summaries=monthly_text,
                weekly_summaries=weekly_text,
                daily_notes=daily_text,
                corpus_summaries=corpus_text
            )

            # Generate via LLM
            if not hasattr(self.model_manager, "generate_once"):
                logger.warning("[NarrativeSynthesis] model_manager.generate_once not available")
                return ""

            # Use fast, cheap model for synthesis
            narrative = await self.model_manager.generate_once(
                prompt,
                model_name=NARRATIVE_SYNTHESIS_MODEL,
                max_tokens=max_tokens,
                temperature=0.3  # Low creativity, high fidelity
            )

            narrative = (narrative or "").strip()
            if not narrative:
                logger.debug("[NarrativeSynthesis] LLM returned empty result")
                return ""

            sources = []
            if obsidian_monthlies:
                sources.append(f"{len(obsidian_monthlies)} monthly")
            if obsidian_weeklies:
                sources.append(f"{len(obsidian_weeklies)} weekly")
            if obsidian_dailies:
                sources.append(f"{len(obsidian_dailies)} daily")
            if recent_weeklies or recent_monthlies:
                sources.append("corpus fallback")

            logger.info(f"[NarrativeSynthesis] Generated narrative ({len(narrative)} chars) from: {', '.join(sources)}")
            return narrative

        except Exception as e:
            logger.warning(f"[NarrativeSynthesis] Failed to generate narrative: {e}")
            return ""
