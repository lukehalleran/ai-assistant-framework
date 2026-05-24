"""
# core/prompt/formatter.py

Module Contract
- Purpose: Text formatting and final prompt assembly from context dict into LLM-ready string.
  Contains both the legacy simple assembler and the full-featured assembler moved from builder.py.
- Class: PromptFormatter(token_manager, time_manager)
- Key methods:
  - _assemble_prompt(context, user_input, directives, system_prompt, **kwargs) -> str
    Full-featured prompt assembler (moved from UnifiedPromptBuilder). Assembles all sections
    into final prompt string with numbered entries, timestamp-first formatting, web citation IDs,
    feature inventory, codebase changes, STM summary, and eval snapshot capture.
    Section order: RECENT CONVERSATION → RELEVANT MEMORIES → RECENT SUMMARIES →
    SEMANTIC SUMMARIES → RECENT REFLECTIONS → SEMANTIC REFLECTIONS → BACKGROUND KNOWLEDGE →
    WEB SEARCH RESULTS → RELEVANT INFORMATION → DREAMS → USER'S PERSONAL NOTES →
    USER UPLOADED ITEMS → VISUAL MEMORIES → DAEMON DOCUMENTATION → PROJECT COMMIT HISTORY →
    ADAPTIVE WORKFLOWS → PROPOSED FEATURES → KNOWLEDGE GRAPH → UNRESOLVED THREADS →
    PROACTIVE INSIGHTS → USER PROFILE → ACTIVE FEATURES → CODEBASE CHANGES →
    TIME CONTEXT → TEMPORAL GROUNDING → STM SUMMARY → CURRENT USER QUERY.
  - _assemble_prompt_legacy(context, user_input, directives) -> str
    Original simpler assembler (may be deprecated). Uses emergency middle-out compression.
  - _build_feature_inventory(context) -> str
    Generates [ACTIVE FEATURES] section from config flags and context counts.
  - _format_memory(mem) -> str  [single memory → "timestamp: User: Q / Daemon: A" with tags; uses format_relative_timestamp for day labels]
  - _format_web_search_results(web_search_result, max_chars) -> str  [WebSearchResult → [WEB SEARCH RESULTS] section]
  - _load_directives() -> str  [loads core/system_prompt.txt with header stripping]
  - _get_time_context() -> str  [current time + time_manager deltas]
- Module-level utilities (imported by other prompt modules):
  - _parse_bool(s, default) -> bool
  - _dedupe_keep_order(items, key_fn) -> List
  - _truncate_list(items, limit) -> List
  - _sanitize_embedded_headers(text) -> str  [escapes [] prompt headers in memory content to ()]
  - _truncate_at_spurious_turns(text) -> str  [truncates at training data leakage markers]
  - _strip_prompt_artifacts  [alias to ResponseParser.strip_prompt_artifacts]
  - _staleness_prefix(item) -> str  [staleness/resolution prefix for summaries/reflections]
  - _is_multimodal_model(model_id) -> bool  [check if model supports multimodal input]
  - _load_upload_image(image_path) -> Optional[dict]  [load persisted upload image as base64]
- Dependencies:
  - core.response_parser.ResponseParser (strip_prompt_artifacts)
  - .token_manager.TokenManager (emergency middle-out compression)
  - utils.time_manager.TimeManager (time deltas) [optional]
  - utils.time_manager.format_relative_timestamp (relative day labels on timestamps)
  - knowledge.web_search_manager.assign_web_ids (web citation ID assignment)
- Side effects:
  - Logging of formatting actions and content statistics
  - Emergency whole-prompt compression when over token budget (protects [CURRENT USER QUERY])
  - Eval snapshot capture (gated by DAEMON_EVAL_CAPTURE env var)
"""

import os
import re
import base64
import json
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime
from pathlib import Path
from utils.logging_utils import get_logger
from core.response_parser import ResponseParser

logger = get_logger("prompt_formatter")

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    """Parse boolean from string, with fallback."""
    if not s:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")

def _as_summary_dict(text: str, tags: list[str], source: str, timestamp: Optional[str] = None) -> dict:
    """Convert summary text to standardized dict format."""
    return {
        "content": text,
        "tags": tags or [],
        "source": source,
        "timestamp": timestamp or datetime.now().isoformat()
    }

def _dedupe_keep_order(items: Iterable[Any], key_fn=lambda x: str(x).strip().lower()) -> List[Any]:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for item in items:
        key = key_fn(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

def _truncate_list(items: List[Any], limit: int) -> List[Any]:
    """Truncate list to limit, keeping most recent items."""
    if limit <= 0:
        return []
    return items[-limit:] if len(items) > limit else items

# Alias for backward compatibility - delegates to ResponseParser
_strip_prompt_artifacts = ResponseParser.strip_prompt_artifacts


# Obsidian image loading for multimodal models
try:
    from config.app_config import (
        MULTIMODAL_MODELS,
    )
except ImportError:
    MULTIMODAL_MODELS = ["opus-4", "claude-3", "sonnet-4", "gpt-4o", "gemini"]


def _staleness_prefix(item) -> str:
    """Return a staleness or resolution prefix for summaries/reflections.

    Priority:
    1. If a resolution_note exists → [RESOLVED — <note>]
    2. If staleness_ratio >= STALENESS_HISTORICAL_THRESHOLD → [HISTORICAL — PARTIALLY OUTDATED]
    """
    try:
        from config.app_config import STALENESS_ENABLED, STALENESS_HISTORICAL_THRESHOLD
        if not STALENESS_ENABLED:
            return ""
        if isinstance(item, dict):
            md = item.get("metadata", {}) or {}
            # Check for explicit resolution note first (entity corrections)
            resolution_note = md.get("resolution_note", "") or item.get("resolution_note", "")
            if resolution_note:
                return f"[RESOLVED — {resolution_note}] "
            # Fall back to generic staleness prefix
            ratio = float(md.get("staleness_ratio", 0) or item.get("staleness_ratio", 0) or 0)
        else:
            return ""
        if ratio >= STALENESS_HISTORICAL_THRESHOLD:
            return "[HISTORICAL — PARTIALLY OUTDATED] "
    except Exception:
        pass
    return ""


def _format_summary_section(items: list, header: str, apply_staleness: bool = True) -> str | None:
    """Format a list of summary/reflection items into a numbered prompt section.

    Shared formatter for recent_summaries, semantic_summaries,
    recent_reflections, and semantic_reflections — all of which follow the
    same dict→content/timestamp extraction + optional staleness prefix
    pattern.  Returns the formatted section string, or None if empty.
    """
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            content = item.get("content", "") or str(item)
            ts = item.get("timestamp", "")
        else:
            content = str(item)
            ts = ""
        if content:
            content = _sanitize_embedded_headers(content)
            prefix = _staleness_prefix(item) if apply_staleness else ""
            if ts:
                lines.append(f"{i}) {ts}: {prefix}{content}")
            else:
                lines.append(f"{i}) {prefix}{content}")
    if not lines:
        return None
    return f"[{header}] n={len(lines)}\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Session boundary detection for conversation rendering
# ---------------------------------------------------------------------------

def _detect_session_boundary(
    ts_prev: Optional[datetime], ts_current: Optional[datetime], gap_hours: float = 2.0
) -> bool:
    """
    Detect whether two consecutive conversation entries cross a session boundary.

    A boundary exists when:
    - ts_prev is None (first entry starts a session)
    - Dates differ (different calendar day)
    - Time gap exceeds gap_hours

    Used to insert visual session markers in [RECENT CONVERSATION] so the LLM
    can distinguish which content belongs to which session.
    """
    if ts_prev is None:
        return True
    if ts_current is None:
        return False
    try:
        # Normalize to naive datetimes for comparison
        prev = ts_prev.replace(tzinfo=None) if hasattr(ts_prev, 'tzinfo') and ts_prev.tzinfo else ts_prev
        curr = ts_current.replace(tzinfo=None) if hasattr(ts_current, 'tzinfo') and ts_current.tzinfo else ts_current
        # Different calendar day = new session
        if prev.date() != curr.date():
            return True
        # Large time gap = new session
        gap = abs((prev - curr).total_seconds()) / 3600.0
        if gap >= gap_hours:
            return True
    except (AttributeError, TypeError):
        return False
    return False


def _format_session_header(ts: datetime) -> str:
    """
    Format a session boundary header with relative day label.

    Examples:
        --- Session: Today (Sat, May 17) ---
        --- Session: Yesterday (Fri, May 16) ---
        --- Session: 3 days ago (Wed, May 14) ---
    """
    try:
        from utils.time_manager import format_relative_timestamp
        rel = format_relative_timestamp(ts)
        # rel looks like "2026-05-17 11:41 (today)" — extract the day label
        # and the date for the header
        day_name = ts.strftime("%a, %b %-d")
        if "(today)" in rel.lower():
            return f"--- Session: Today ({day_name}) ---"
        elif "(yesterday)" in rel.lower():
            return f"--- Session: Yesterday ({day_name}) ---"
        else:
            # Extract "N days ago" from the relative label
            import re
            m = re.search(r'\((\d+\s+days?\s+ago)\)', rel, re.IGNORECASE)
            if m:
                return f"--- Session: {m.group(1).capitalize()} ({day_name}) ---"
            return f"--- Session: {day_name} ---"
    except Exception:
        return f"--- Session: {ts.strftime('%Y-%m-%d') if ts else 'Unknown'} ---"


def _parse_entry_timestamp(mem: dict) -> Optional[datetime]:
    """Extract a datetime from a conversation entry's timestamp field."""
    ts = mem.get("timestamp", "")
    if not ts:
        ts = (mem.get("metadata") or {}).get("timestamp", "")
    if isinstance(ts, datetime):
        return ts
    if ts:
        try:
            return datetime.fromisoformat(str(ts))
        except (ValueError, TypeError):
            pass
    return None


def _is_multimodal_model(model_id: str) -> bool:
    """Check if a model ID corresponds to a multimodal-capable model."""
    if not model_id:
        return False
    model_lower = model_id.lower()
    return any(pattern.lower() in model_lower for pattern in MULTIMODAL_MODELS)


def _load_upload_image(image_path: str) -> Optional[dict]:
    """
    Load a persisted upload image from disk as base64 for multimodal API calls.

    Args:
        image_path: Path to the image file on disk

    Returns:
        Dict with 'data', 'media_type', 'filename' keys, or None if loading fails
    """
    try:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.debug(f"[_load_upload_image] File not found: {image_path}")
            return None

        # Determine media type from extension
        ext = path.suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(ext, 'application/octet-stream')

        file_bytes = path.read_bytes()

        # Compress if image exceeds API limit (5MB)
        API_IMAGE_LIMIT = 4_500_000
        if len(file_bytes) > API_IMAGE_LIMIT:
            from utils.file_processor import FileProcessor
            fp = FileProcessor()
            file_bytes = fp._compress_image(file_bytes, ext, API_IMAGE_LIMIT)
            # Compression may change format to JPEG
            if ext not in ('.png',):
                media_type = 'image/jpeg'
            logger.info(f"[_load_upload_image] Compressed {image_path}: {path.stat().st_size//1024}KB → {len(file_bytes)//1024}KB")

        data = base64.b64encode(file_bytes).decode('utf-8')
        return {
            'data': data,
            'media_type': media_type,
            'filename': path.name,
        }
    except Exception as e:
        logger.warning(f"[_load_upload_image] Failed to load {image_path}: {e}")
        return None


def _truncate_at_spurious_turns(text: str) -> str:
    """
    Truncate response at spurious chat turn markers that indicate training data leakage.

    LLMs can sometimes output leaked training data containing markers like:
    - "Human:" or "User:" mid-response (not at the start)
    - Control characters like \\x05 (ENQ) or \\x06 (ACK)
    - File path patterns after turn markers

    This function truncates the response at the first spurious turn marker,
    preserving only the legitimate response content.
    """
    if not text:
        return text

    try:
        import re

        # First, strip any control characters (ASCII 0x00-0x1F except newline/tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Patterns that indicate training data leakage mid-response
        # These should NOT appear in a normal assistant response after content
        # Note: Control chars are stripped first, so these patterns match post-strip
        spurious_patterns = [
            r'\n\s*Human:\s*End\s*File',          # Human: End File (after control char strip)
            r'\n\s*User:\s*End\s*File',           # User: End File (after control char strip)
            r'\n\s*Human:\s*#\s*[/\w]',           # Human: followed directly by file comment
            r'\n\s*User:\s*#\s*[/\w]',            # User: followed directly by file comment
            r'\n\s*Human:\s*\n\s*#\s*[/\w]',      # Human: + newline + file comment
            r'\n\s*User:\s*\n\s*#\s*[/\w]',       # User: + newline + file comment
            r'End\s*File\s*#\s*\w+/',             # End File # path/... pattern
            r'\n\s*```\s*\n\s*#\s*[A-Za-z/]',     # Code block followed by file path
            r'\n\s*Human:\s*"""',                 # Human: followed by docstring
            r'\n\s*User:\s*"""',                  # User: followed by docstring
        ]

        combined = '(' + '|'.join(spurious_patterns) + ')'
        match = re.search(combined, text)

        if match:
            # Truncate at the spurious marker
            truncated = text[:match.start()].rstrip()
            logger.warning(f"[SANITIZE] Truncated response at spurious turn marker (pos {match.start()})")
            return truncated

        return text

    except Exception as e:
        logger.warning(f"[SANITIZE] Failed to check for spurious turns: {e}")
        return text


def _sanitize_embedded_headers(text: str) -> str:
    """
    Sanitize embedded section headers in memory content to prevent prompt pollution.

    When conversations discuss prompt structure, they may contain literal text like
    "[RECENT CONVERSATION]" which can confuse the LLM into thinking there are
    multiple sections. This function escapes such headers by replacing brackets.

    Example: "[RECENT CONVERSATION]" -> "(RECENT CONVERSATION)"
    """
    if not text:
        return text

    try:
        # List of section headers that should be escaped when found mid-text
        section_headers = [
            r'\[RECENT CONVERSATION[^\]]*\]',
            r'\[RELEVANT MEMORIES[^\]]*\]',
            r'\[RELEVANT INFORMATION[^\]]*\]',
            r'\[BACKGROUND KNOWLEDGE[^\]]*\]',
            r'\[USER PROFILE[^\]]*\]',
            r'\[TIME CONTEXT[^\]]*\]',
            r'\[CURRENT USER QUERY[^\]]*\]',
            r'\[CURRENT QUERY[^\]]*\]',
            r'\[LAST EXCHANGE[^\]]*\]',
            r'\[WEB SEARCH RESULTS[^\]]*\]',
            r'\[SUMMARIES[^\]]*\]',
            r'\[RECENT SUMMARIES[^\]]*\]',
            r'\[SEMANTIC SUMMARIES[^\]]*\]',
            r'\[RECENT REFLECTIONS[^\]]*\]',
            r'\[SEMANTIC REFLECTIONS[^\]]*\]',
            r'\[DREAMS[^\]]*\]',
            r'\[SHORT-TERM CONTEXT[^\]]*\]',
            r'\[STM[^\]]*\]',
            r'\[SEMANTIC FACTS[^\]]*\]',
            r'\[RECENT FACTS[^\]]*\]',
            r'\[FACTS[^\]]*\]',
            r'\[DIRECTIVES[^\]]*\]',
            r'\[ADAPTIVE WORKFLOWS[^\]]*\]',
            r'\[PROPOSED FEATURES[^\]]*\]',
            r'\[PROJECT COMMIT HISTORY[^\]]*\]',
        ]

        # Combine into one pattern
        combined_pattern = '(' + '|'.join(section_headers) + ')'

        def replace_brackets(match):
            """Replace [] with () to neutralize the header."""
            matched_text = match.group(0)
            # Replace opening [ with ( and closing ] with )
            return '(' + matched_text[1:-1] + ')'

        sanitized = re.sub(combined_pattern, replace_brackets, text, flags=re.IGNORECASE)

        if sanitized != text:
            logger.debug(f"[SANITIZE] Escaped embedded section headers in content")

        return sanitized
    except Exception as e:
        logger.warning(f"[SANITIZE] Failed to sanitize embedded headers: {e}")
        return text


class PromptFormatter:
    """Handles text formatting and prompt assembly for prompt building."""

    def __init__(self, token_manager, time_manager=None):
        self.token_manager = token_manager
        self.time_manager = time_manager

    def _load_directives(self) -> str:
        """Load directives from file with fallback to empty string."""
        try:
            directives_path = os.path.join("core", "system_prompt.txt")
            if os.path.isfile(directives_path):
                with open(directives_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                # Strip header comments
                lines = content.split("\n")
                cleaned = []
                for line in lines:
                    if line.strip().startswith("#") and not any(x in line.lower() for x in ["personality", "directive", "instruction"]):
                        continue
                    cleaned.append(line)
                return "\n".join(cleaned).strip()
        except (IOError, OSError, FileNotFoundError):
            pass  # Personality file not found or unreadable
        return ""

    def _get_time_context(self) -> str:
        """Get current time context for the prompt."""
        now = datetime.now()
        lines = [f"Current time: {now.strftime('%A, %Y-%m-%d %H:%M:%S')}"]

        # Add time deltas if time_manager is available
        if self.time_manager:
            time_since_msg = self.time_manager.time_since_previous_message()
            time_since_session = self.time_manager.elapsed_since_last_session()
            lines.append(f"Time since last message: {time_since_msg}")
            lines.append(f"Time since last session: {time_since_session}")

        return "\n".join(lines)

    def _format_memory(self, mem: Dict[str, Any]) -> str:
        """Format a single memory for display."""
        try:
            # Debug: Log what we're getting
            logger.debug(f"[FORMATTING] Formatting memory with keys: {list(mem.keys()) if isinstance(mem, dict) else type(mem)}")

            # Handle case where mem is already a string (shouldn't happen but defensive)
            if isinstance(mem, str):
                logger.debug("[FORMATTING] Memory is already string, returning as-is")
                return mem

            # Handle case where mem is already formatted properly
            if isinstance(mem, dict) and 'formatted' in mem:
                logger.debug("[FORMATTING] Memory has 'formatted' key, returning that")
                return mem['formatted']

            # Extract query and response
            query = mem.get("query", "")
            response = mem.get("response", "")
            logger.debug(f"[FORMATTING] Got query='{str(query)[:50]}...', response='{str(response)[:50]}...'")

            # Strip any bracketed prompt artifacts accidentally stored in memory
            if query:
                query = _strip_prompt_artifacts(query)
                query = _sanitize_embedded_headers(query)
            if response:
                response = _strip_prompt_artifacts(response)
                response = _sanitize_embedded_headers(response)

            # Get timestamp - check multiple possible locations and format as datetime
            timestamp = mem.get("timestamp", "")
            if not timestamp:
                meta = mem.get("metadata", {})
                timestamp = meta.get("timestamp", "")

            # Format timestamp with relative day label to prevent temporal hallucinations
            from utils.time_manager import format_relative_timestamp
            if isinstance(timestamp, datetime):
                datetime_str = format_relative_timestamp(timestamp)
            elif timestamp:
                try:
                    datetime_str = format_relative_timestamp(datetime.fromisoformat(str(timestamp)))
                except (ValueError, TypeError):
                    datetime_str = str(timestamp)
            else:
                datetime_str = "Unknown time"

            # Get tags - check multiple possible locations
            tags = mem.get("tags", [])
            if not tags:
                meta = mem.get("metadata", {})
                tags = meta.get("tags", [])

            # Ensure tags is a list and convert to string representation
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            elif not tags:
                tags = []

            # Format tags string
            tags_str = ", ".join(str(tag) for tag in tags) if tags else ""

            # Format based on content availability
            formatted_content = ""
            if query and response:
                formatted_content = f"{datetime_str}: User: {query.strip()}\nDaemon: {response.strip()}"
            elif response:
                formatted_content = f"{datetime_str}: Daemon: {response.strip()}"
            elif query:
                formatted_content = f"{datetime_str}: User: {query.strip()}"
            else:
                # Fallback to content field if available
                content = mem.get("content", "")
                if content:
                    content = _sanitize_embedded_headers(content)
                    formatted_content = f"{datetime_str}: {content.strip()}"
                else:
                    # Last resort - show memory id and basic info
                    mem_id = mem.get("id", "unknown")
                    formatted_content = f"{datetime_str}: Memory {mem_id}"

            # Add tags if available
            if tags_str and formatted_content:
                formatted_content += f"\nTags: {{{tags_str}}}"

            logger.debug(f"[FORMATTING] Final formatted content: {formatted_content[:100]}...")
            return formatted_content

        except Exception as e:
            logger.error(f"[FORMATTING] Error formatting memory: {e}")
            logger.error(f"[FORMATTING] Memory type: {type(mem)}, keys: {list(mem.keys()) if isinstance(mem, dict) else 'not dict'}")
            import traceback
            logger.debug(f"[FORMATTING] Exception traceback: {traceback.format_exc()}")
            # Return a safe fallback format
            try:
                mem_id = mem.get("id", "unknown") if isinstance(mem, dict) else "unknown"
                return f"Memory {mem_id} (formatting error)"
            except Exception:
                return "Memory (formatting error)"

    def _format_web_search_results(self, web_search_result: Any, max_chars: int = 10000) -> str:
        """
        Format web search results for prompt injection.

        Args:
            web_search_result: WebSearchResult object or None
            max_chars: Maximum characters for the section

        Returns:
            Formatted string with [WEB SEARCH RESULTS] header, or empty string
        """
        if web_search_result is None:
            return ""

        try:
            # Handle both WebSearchResult object and dict format
            if hasattr(web_search_result, 'has_results'):
                # It's a WebSearchResult object
                if not web_search_result.has_results:
                    return ""

                pages = web_search_result.pages
                from_cache = web_search_result.from_cache
                query = web_search_result.query
            elif isinstance(web_search_result, dict):
                # It's a dict (fallback)
                pages = web_search_result.get('pages', [])
                from_cache = web_search_result.get('from_cache', False)
                query = web_search_result.get('query', '')
                if not pages:
                    return ""
            else:
                return ""

            # Build formatted content
            content_parts = []
            total_chars = 0

            for page in pages:
                if total_chars >= max_chars:
                    break

                # Extract page info (handle both object and dict)
                if hasattr(page, 'title'):
                    title = page.title
                    url = page.url
                    page_content = page.content or page.snippet
                else:
                    title = page.get('title', '')
                    url = page.get('url', '')
                    page_content = page.get('content', '') or page.get('snippet', '')

                if not page_content:
                    continue

                # Format single result
                entry = f"**{title}** ({url})\n{page_content}"

                # Check if we can fit it
                if total_chars + len(entry) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        entry = entry[:remaining] + "..."
                        content_parts.append(entry)
                    break

                content_parts.append(entry)
                total_chars += len(entry) + 2  # +2 for newlines

            if not content_parts:
                return ""

            # Build the section
            cache_note = " (cached)" if from_cache else ""
            section = f"[WEB SEARCH RESULTS] n={len(content_parts)}{cache_note}\n"
            section += "\n\n".join(content_parts)

            logger.debug(f"[FORMATTING] Web search results: {len(content_parts)} pages, {len(section)} chars")
            return section

        except Exception as e:
            logger.warning(f"[FORMATTING] Error formatting web search results: {e}")
            return ""

    def _assemble_prompt_legacy(self, context: Dict[str, Any], user_input: str, directives: str = "") -> str:
        """
        Assemble final prompt from context and user input (legacy simple version).

        This is the original simple assembler. For the full-featured version with
        numbered entries, web citation IDs, feature inventory, etc., see _assemble_prompt().

        Section order (optimized for attention and token efficiency):
        1. Recent conversations (baseline context)
        2. Relevant memories (semantic hits)
        3. User profile (personalization) [MOVED from early position]
        4. Summaries (compressed history)
        5. Reflections (meta insights)
        6. Wiki content (background knowledge)
        7. Semantic chunks (relevant information)
        8. Dreams (if enabled)
        9. Time context (temporal grounding) [MOVED from first position]
        10. User input (always last)
        """
        sections = []

        # Recent conversations
        recent = context.get("recent_conversations", [])
        recent_text = []
        logger.debug(f"[ASSEMBLY] Processing {len(recent or [])} recent conversations")
        for i, mem in enumerate(recent or [], 1):
            logger.debug(f"[ASSEMBLY] Formatting recent memory #{i}, type: {type(mem)}")
            formatted = self._format_memory(mem)
            if formatted:
                recent_text.append(f"{i}) {formatted}")
            else:
                logger.warning(f"[ASSEMBLY] Recent memory #{i} formatted to empty string")
        # Always include header to keep structure stable
        sections.append(
            f"[RECENT CONVERSATION] n={len(recent_text)}" + (
                "\n" + "\n\n".join(recent_text) if recent_text else ""
            )
        )

        # Relevant memories
        memories = context.get("memories", [])
        memory_text = []
        logger.debug(f"[ASSEMBLY] Processing {len(memories or [])} relevant memories")
        for i, mem in enumerate(memories or [], 1):
            logger.debug(f"[ASSEMBLY] Formatting relevant memory #{i}, type: {type(mem)}")
            formatted = self._format_memory(mem)
            if formatted:
                memory_text.append(f"{i}) {formatted}")
            else:
                logger.warning(f"[ASSEMBLY] Relevant memory #{i} formatted to empty string")
        # Always include header
        sections.append(
            f"[RELEVANT MEMORIES] n={len(memory_text)}" + (
                "\n" + "\n\n".join(memory_text) if memory_text else ""
            )
        )

        # Summaries
        summaries = context.get("summaries", [])
        summary_text = []
        for summ in (summaries or []):
            if isinstance(summ, dict):
                content = summ.get("content", "")
                timestamp = summ.get("timestamp", "")
                if content:
                    if timestamp:
                        summary_text.append(f"{content} | {timestamp}")
                    else:
                        summary_text.append(content)
            else:
                summary_text.append(str(summ))
        sections.append(
            f"[SUMMARIES] n={len(summary_text)}" + (
                "\n" + "\n\n".join(summary_text) if summary_text else ""
            )
        )

        # Reflections
        reflections = context.get("reflections", [])
        reflection_text = []
        for refl in (reflections or []):
            if isinstance(refl, dict):
                content = refl.get("content", "")
                timestamp = refl.get("timestamp", "")
                if content:
                    if timestamp:
                        reflection_text.append(f"{content} | {timestamp}")
                    else:
                        reflection_text.append(content)
            else:
                reflection_text.append(str(refl))
        sections.append(
            f"[RECENT REFLECTIONS] n={len(reflection_text)}" + (
                "\n" + "\n\n".join(reflection_text) if reflection_text else ""
            )
        )

        # Wiki content
        wiki = context.get("wiki", [])
        wiki_text = []
        for w in (wiki or []):
            if isinstance(w, dict):
                content = w.get("content", "")
                title = w.get("title", "")
                if content:
                    if title:
                        wiki_text.append(f"**{title}**\n{content}")
                    else:
                        wiki_text.append(content)
            else:
                wiki_text.append(str(w))
        sections.append(
            f"[BACKGROUND KNOWLEDGE] n={len(wiki_text)}" + (
                "\n" + "\n\n".join(wiki_text) if wiki_text else ""
            )
        )

        # Web search results (real-time web content)
        web_search = context.get("web_search_results")
        if web_search is not None:
            web_text = self._format_web_search_results(web_search)
            if web_text:
                sections.append(web_text)

        # Semantic chunks
        semantic_chunks = context.get("semantic_chunks", [])
        semantic_text = []
        for chunk in (semantic_chunks or []):
            if isinstance(chunk, dict):
                content = chunk.get("filtered_content", "") or chunk.get("content", "")
                title = chunk.get("title", "")
                if content:
                    if title:
                        semantic_text.append(f"**{title}**\n{content}")
                    else:
                        semantic_text.append(content)
            else:
                semantic_text.append(str(chunk))
        sections.append(
            f"[RELEVANT INFORMATION] n={len(semantic_text)}" + (
                "\n" + "\n\n".join(semantic_text) if semantic_text else ""
            )
        )

        # Dreams
        dreams = context.get("dreams", [])
        if dreams:
            dream_text = []
            for dream in dreams:
                if isinstance(dream, dict):
                    content = dream.get("content", "")
                    timestamp = dream.get("timestamp", "")
                    if content:
                        if timestamp:
                            dream_text.append(f"{content} | {timestamp}")
                        else:
                            dream_text.append(content)
                else:
                    dream_text.append(str(dream))
            if dream_text:
                sections.append(f"[DREAMS] n={len(dream_text)}\n" + "\n\n".join(dream_text))

        # User Profile (replaces semantic_facts + fresh_facts)
        # MOVED: Placed here (after bulk knowledge, before query) for high attention with low token cost
        user_profile = context.get("user_profile", "")
        if user_profile and isinstance(user_profile, str):
            # Count facts (each fact ends with [timestamp])
            fact_count = user_profile.count('[20')  # Count timestamp brackets starting with [20xx
            # Profile is already formatted from UserProfile.get_context_injection()
            sections.append(
                f"[USER PROFILE] n={fact_count}\n"
                "Stored facts — reference naturally but do not add names, apps, or details not written here.\n"
                f"{user_profile}")

        # Time context
        # MOVED: Placed here (right before query) for temporal grounding with high attention
        time_ctx = self._get_time_context()
        if time_ctx:
            sections.append(f"[TIME CONTEXT]\n{time_ctx}")

        # User input with last Q/A pair for coherence
        if user_input:
            query_section = f"[CURRENT USER QUERY]\n"

            # Attach last Q/A pair for maximum coherence (high attention area)
            recent = context.get("recent_conversations", [])
            if recent and len(recent) > 0:
                last_exchange = recent[0]  # First item is most recent (list ordered newest-first)
                last_q = last_exchange.get("query", "")
                last_a = last_exchange.get("response", "")
                if last_q and last_a:
                    query_section += f"[LAST EXCHANGE FOR CONTEXT]\n"
                    query_section += f"User: {last_q}\n"
                    query_section += f"Assistant: {last_a}\n\n"

            query_section += f"[CURRENT QUERY]\n{user_input}"
            sections.append(query_section)

        # Join all sections with double newlines
        full_prompt = "\n\n".join(sections)

        # Emergency whole-prompt compression if still over budget after per-item compression
        # CRITICAL: Protect [CURRENT USER QUERY] section from compression
        if hasattr(self.token_manager, 'token_budget') and hasattr(self.token_manager, 'model_manager'):
            try:
                model_name = self.token_manager.model_manager.get_active_model_name()
                token_count = self.token_manager.get_token_count(full_prompt, model_name)
                budget = self.token_manager.token_budget

                if token_count > budget:
                    logger.warning(f"[EMERGENCY MIDDLE-OUT] Prompt STILL exceeds budget after per-item compression: {token_count} > {budget} tokens")
                    logger.warning(f"[EMERGENCY MIDDLE-OUT] Applying whole-prompt compression as last resort...")

                    # Split off [CURRENT USER QUERY] section to protect it
                    query_marker = "[CURRENT USER QUERY]"
                    if query_marker in full_prompt:
                        parts = full_prompt.split(query_marker, 1)
                        before_query = parts[0]
                        query_section = query_marker + parts[1]

                        # Compress only the context BEFORE the query
                        compressed_before = self.token_manager._middle_out(before_query, budget - 500, force=True)
                        full_prompt = compressed_before + "\n\n" + query_section

                        new_count = self.token_manager.get_token_count(full_prompt, model_name)
                        logger.warning(f"[EMERGENCY MIDDLE-OUT] Compressed context (protected query): {token_count} → {new_count} tokens")
                    else:
                        # Fallback: compress entire prompt if no query marker found
                        full_prompt = self.token_manager._middle_out(full_prompt, budget, force=True)
                        new_count = self.token_manager.get_token_count(full_prompt, model_name)
                        logger.warning(f"[EMERGENCY MIDDLE-OUT] Compressed entire prompt: {token_count} → {new_count} tokens")
                else:
                    logger.debug(f"[PROMPT] Final token count: {token_count}/{budget} (within budget)")
            except Exception as e:
                logger.warning(f"[EMERGENCY MIDDLE-OUT] Compression failed: {e}, using full prompt")

        return full_prompt

    def _build_feature_inventory(self, context: Dict[str, Any]) -> str:
        """Build a compact feature inventory showing which systems are active and what they returned.

        Reads config flags and counts results from the context dict.
        No retrieval needed — purely reads config flags and context dict counts.

        Returns:
            Compact multi-line string grouped by category, or empty string.
        """
        try:
            from config import app_config as cfg

            def _on_off(flag: bool) -> str:
                return "ON" if flag else "OFF"

            def _count(key: str, fallback=None) -> str:
                """Get count annotation from context, e.g. '(3)' or ''."""
                val = context.get(key, fallback)
                if val is None:
                    return ""
                if isinstance(val, list):
                    return f"({len(val)})" if val else "(0)"
                if isinstance(val, str):
                    return f"({len(val.split(chr(10)))})" if val.strip() else "(0)"
                if isinstance(val, dict):
                    total = sum(len(v) for v in val.values() if isinstance(v, list))
                    return f"({total})" if total else "(0)"
                return ""

            lines = []

            # Memory category
            mem_parts = []
            kg_enabled = getattr(cfg, 'KNOWLEDGE_GRAPH_ENABLED', False)
            kg_ctx = context.get("graph_context", []) or []
            mem_parts.append(f"knowledge_graph={_on_off(kg_enabled)}{f'({len(kg_ctx)} edges)' if kg_ctx else ''}")
            mem_parts.append(f"fact_verification={_on_off(getattr(cfg, 'FACT_VERIFICATION_ENABLED', False))}")
            mem_parts.append(f"truth_scorer={_on_off(getattr(cfg, 'TRUTH_SCORER_ENABLED', True))}")
            mem_parts.append(f"dedup={_on_off(getattr(cfg, 'CROSS_DEDUP_ENABLED', False))}")
            lines.append("Memory: " + " | ".join(mem_parts))

            # Knowledge category
            know_parts = []
            git = context.get("git_commits", []) or []
            know_parts.append(f"git_commits={_on_off(getattr(cfg, 'GIT_MEMORY_ENABLED', False))}{f'({len(git)})' if git else ''}")
            notes = context.get("personal_notes", []) or []
            know_parts.append(f"obsidian={_on_off(bool(notes))}{f'({len(notes)} notes)' if notes else ''}")
            ref_docs = context.get("reference_docs", []) or []
            know_parts.append(f"reference_docs={_on_off(getattr(cfg, 'REFERENCE_DOCS_AUTO_SEED', False))}{f'({len(ref_docs)})' if ref_docs else ''}")
            web = context.get("web_search_results")
            know_parts.append(f"web_search={_on_off(getattr(cfg, 'WEB_SEARCH_ENABLED', False))}{_count('web_search_results')}")
            lines.append("Knowledge: " + " | ".join(know_parts))

            # Proactive category
            pro_parts = []
            threads = context.get("unresolved_threads", []) or []
            pro_parts.append(f"threads={_on_off(getattr(cfg, 'THREAD_SURFACING_ENABLED', False))}{f'({len(threads)} open)' if threads else ''}")
            insights = context.get("proactive_insights", []) or []
            pro_parts.append(f"insights={_on_off(getattr(cfg, 'PROACTIVE_SURFACING_ENABLED', False))}{f'({len(insights)})' if insights else ''}")
            pro_parts.append(f"narrative={_on_off(getattr(cfg, 'NARRATIVE_CONTEXT_ENABLED', True) if hasattr(cfg, 'NARRATIVE_CONTEXT_ENABLED') else bool(context.get('narrative_state')))}")
            lines.append("Proactive: " + " | ".join(pro_parts))

            # Analysis category
            ana_parts = []
            ana_parts.append(f"intent={_on_off(getattr(cfg, 'INTENT_ENABLED', False))}")
            ana_parts.append(f"escalation={_on_off(getattr(cfg, 'ESCALATION_ENABLED', False))}")
            skills = context.get("procedural_skills", []) or []
            ana_parts.append(f"skills={_on_off(getattr(cfg, 'PROCEDURAL_SKILLS_ENABLED', False))}{f'({len(skills)})' if skills else ''}")
            lines.append("Analysis: " + " | ".join(ana_parts))

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"[PromptFormatter] Feature inventory failed: {e}")
            return ""

    def _assemble_prompt(self, context: Dict[str, Any] = None, user_input: str = "",
                        directives: str = "", system_prompt: str = "", **kwargs) -> str:
        """
        Assemble final prompt string from context with numbering and timestamp-first entries.

        Section order (optimized for attention and token efficiency):
        1. Recent conversations (baseline context)
        2. Relevant memories (semantic hits)
        3. Recent summaries (compressed recent history)
        4. Semantic summaries (relevant compressed history)
        5. Background knowledge (wiki)
        6. Relevant information (semantic chunks)
        7. Recent reflections (meta insights)
        8. Semantic reflections (relevant meta insights)
        9. Dreams (if enabled)
        10. User profile (cheap, high attention, personalization)
        11. Time context (cheap, high attention, temporal grounding)
        12. STM summary (short-term context, maximum attention)
        13. Current user query (always last)

        Note: User profile and time context moved to positions 10-11 (from 4 and 1) to leverage
        recency bias in LLM attention while keeping token cost minimal.
        """
        if context is None:
            context = {}
        if system_prompt and not directives:
            directives = system_prompt

        from datetime import datetime
        logger.warning(f"PROMPT ASSEMBLY START: context has {len(context)} keys: {list(context.keys())}")
        logger.warning(f"PROMPT ASSEMBLY START: recent_summaries={len(context.get('recent_summaries', []))}, semantic_summaries={len(context.get('semantic_summaries', []))}")
        logger.warning(f"PROMPT ASSEMBLY START: stm_summary present = {context.get('stm_summary') is not None}, value = {context.get('stm_summary')}")

        def mem_parts(mem: Dict[str, Any]) -> tuple[str, str]:
            try:
                # Memory field structure varies by source:
                # - Hybrid retriever uses 'content' field
                # - Corpus manager uses 'query'/'response' fields
                # Try content field first (from hybrid retriever)
                content_field = mem.get("content", "")

                if content_field:
                    # Content field has full conversation text
                    content = content_field.strip()
                else:
                    # Fallback to query/response format
                    q = str(mem.get("query", ""))
                    r = str(mem.get("response", ""))

                    # Build the content
                    if q and r:
                        content = f"User: {q.strip()}\nDaemon: {r.strip()}"
                    elif r:
                        content = f"Daemon: {r.strip()}"
                    elif q:
                        content = f"User: {q.strip()}"
                    else:
                        content = str(mem)

                # Get timestamp (may be in root or metadata)
                ts = mem.get("timestamp", "")
                if not ts:
                    ts = mem.get("metadata", {}).get("timestamp", "")

                # Get tags
                tags = mem.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                elif not tags:
                    tags = []
                tags_str = ", ".join(str(tag) for tag in tags) if tags else ""

                # Format timestamp with relative day label to prevent temporal hallucinations
                from utils.time_manager import format_relative_timestamp
                if isinstance(ts, datetime):
                    ts_str = format_relative_timestamp(ts)
                elif ts:
                    # Try parsing ISO string for relative formatting
                    try:
                        ts_str = format_relative_timestamp(datetime.fromisoformat(str(ts)))
                    except (ValueError, TypeError):
                        ts_str = str(ts)
                else:
                    ts_str = ""

                # Add tags
                if tags_str and content:
                    content += f"\nTags: {{{tags_str}}}"

                # Render shared content objects with provenance markers
                md = mem.get("metadata", {}) or {}
                content_type = md.get("content_type", "") or mem.get("content_type", "")
                if content_type:
                    type_labels = {
                        "poem": "Poem", "lyrics": "Song lyrics",
                        "code": "Code", "quote": "Quote",
                        "message": "Shared message", "dream": "Dream",
                    }
                    label = type_labels.get(content_type, content_type.capitalize())
                    title = md.get("content_title", "") or mem.get("content_title", "")
                    attr = md.get("content_attribution", "") or mem.get("content_attribution", "")
                    parts = [f"[SHARED — {label}"]
                    if title:
                        parts[0] += f' "{title}"'
                    if ts_str:
                        parts[0] += f", {ts_str}"
                    if attr:
                        parts[0] += f", by {attr}"
                    parts[0] += "]"
                    content = parts[0] + "\n" + content

                # Sanitize any embedded section headers to prevent prompt pollution
                content = _sanitize_embedded_headers(content)

                return content, ts_str
            except (AttributeError, TypeError, KeyError):
                return str(mem), ""

        sections: list[str] = []

        # Recent conversations — with session boundary markers
        recent = context.get("recent_conversations", []) or []
        logger.debug(f"[DEBUG RECENT] _assemble_prompt: Got {len(recent)} items in recent_conversations")
        recent_lines: list[str] = []
        prev_ts: Optional[datetime] = None
        for i, mem in enumerate(recent, start=1):
            content, ts_str = mem_parts(mem)
            if i <= 3 or i > len(recent) - 3:
                logger.debug(f"[DEBUG RECENT] Item {i}: ts={ts_str}, content_preview={content[:100] if content else 'EMPTY'}...")

            # Insert session boundary marker when sessions change
            entry_ts = _parse_entry_timestamp(mem)
            if _detect_session_boundary(prev_ts, entry_ts):
                if entry_ts:
                    recent_lines.append(_format_session_header(entry_ts))
            if entry_ts:
                prev_ts = entry_ts

            recent_lines.append(f"{i}) {ts_str}: {content}" if ts_str else f"{i}) {content}")
        if recent_lines:
            logger.debug(f"[DEBUG RECENT] Adding [RECENT CONVERSATION] section with {len([l for l in recent_lines if not l.startswith('---')])} formatted entries")
            sections.append(f"[RECENT CONVERSATION] n={len([l for l in recent_lines if not l.startswith('---')])}\n" + "\n\n".join(recent_lines))

        # Relevant memories
        memories = context.get("memories", []) or []
        logger.warning(f"PROMPT BUILD: FINAL COUNT - Got {len(memories)} memories from context BEFORE ASSEMBLY")
        memory_lines: list[str] = []
        for i, mem in enumerate(memories, start=1):
            content, ts = mem_parts(mem)
            memory_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if memory_lines:
            sections.append(f"[RELEVANT MEMORIES] n={len(memory_lines)}\n" + "\n\n".join(memory_lines))
            logger.warning(f"PROMPT BUILD: FINAL COUNT - [RELEVANT MEMORIES] section will contain {len(memory_lines)} memories")
        else:
            logger.warning("PROMPT BUILD: FINAL COUNT - No memories to display in [RELEVANT MEMORIES] section")

        # Summaries and reflections (4 sections, shared format via _format_summary_section)
        recent_summaries = context.get("recent_summaries", []) or []
        logger.warning(f"PROMPT ASSEMBLY: Got {len(recent_summaries)} recent summaries")
        _sec = _format_summary_section(recent_summaries, "RECENT SUMMARIES")
        if _sec:
            sections.append(_sec)
            logger.warning(f"PROMPT ASSEMBLY: Added recent summaries section")
        else:
            logger.warning("PROMPT ASSEMBLY: No recent summaries to add")

        _sec = _format_summary_section(context.get("semantic_summaries", []) or [], "SEMANTIC SUMMARIES")
        if _sec:
            sections.append(_sec)

        _sec = _format_summary_section(context.get("recent_reflections", []) or [], "RECENT REFLECTIONS")
        if _sec:
            sections.append(_sec)

        _sec = _format_summary_section(context.get("semantic_reflections", []) or [], "SEMANTIC REFLECTIONS")
        if _sec:
            sections.append(_sec)

        # Wiki content
        wiki = context.get("wiki", []) or []
        wiki_lines: list[str] = []
        for i, w in enumerate(wiki, start=1):
            if isinstance(w, dict):
                content = w.get("content", "")
                title = w.get("title", "")
                block = f"**{title}**\n{content}" if title and content else (content or str(w))
            else:
                block = str(w)
            wiki_lines.append(f"{i}) {block}")
        if wiki_lines:
            sections.append(f"[BACKGROUND KNOWLEDGE] n={len(wiki_lines)}\n" + "\n\n".join(wiki_lines))

        # Web search results (real-time web content) — with [WEB_N] source IDs
        web_search = context.get("web_search_results")
        if web_search is not None:
            try:
                # Handle WebSearchResult object
                if hasattr(web_search, 'has_results') and web_search.has_results:
                    from knowledge.web_search_manager import assign_web_ids
                    pages = web_search.pages
                    from_cache = web_search.from_cache
                    # Assign stable WEB_N IDs after dedupe
                    numbered_sources, web_source_map = assign_web_ids(pages)
                    # Store map for citation validation downstream
                    context["_web_source_map"] = web_source_map
                    ws_lines: list[str] = []
                    for src in numbered_sources[:8]:  # Limit to 8 results
                        content = src.content
                        if content:
                            if len(content) > 2000:
                                content = content[:2000] + "..."
                            ws_lines.append(f"[{src.source_id}] **{src.title}** ({src.url})\n{content}")
                    if ws_lines:
                        cache_note = " (cached)" if from_cache else ""
                        citation_instruction = (
                            "Cite web sources using [WEB_N] markers (e.g., 'According to Reuters [WEB_1]...'). "
                            "Every factual claim from web sources MUST include a citation.\n"
                        )
                        sections.append(
                            f"[WEB SEARCH RESULTS] n={len(ws_lines)}{cache_note}\n"
                            f"{citation_instruction}"
                            + "\n\n".join(ws_lines)
                        )
                        logger.info(f"[PROMPT ASSEMBLY] Added web search section with {len(ws_lines)} results, {len(web_source_map)} source IDs")
            except Exception as e:
                logger.warning(f"[PROMPT ASSEMBLY] Failed to format web search results: {e}")

        # Semantic chunks
        chunks = context.get("semantic_chunks", []) or []
        sc_lines: list[str] = []
        for i, c in enumerate(chunks, start=1):
            if isinstance(c, dict):
                content = c.get("filtered_content", "") or c.get("content", "")
                title = c.get("title", "")
                block = f"**{title}**\n{content}" if title and content else (content or str(c))
            else:
                block = str(c)
            sc_lines.append(f"{i}) {block}")
        if sc_lines:
            sections.append(f"[RELEVANT INFORMATION] n={len(sc_lines)}\n" + "\n\n".join(sc_lines))

        # Dreams
        dreams = context.get("dreams", []) or []
        dr_lines: list[str] = []
        for i, d in enumerate(dreams, start=1):
            if isinstance(d, dict):
                content = d.get("content", "") or str(d)
                ts = d.get("timestamp", "")
            else:
                content = str(d)
                ts = ""
            dr_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if dr_lines:
            sections.append(f"[DREAMS] n={len(dr_lines)}\n" + "\n\n".join(dr_lines))

        # Personal Notes from Obsidian vault
        personal_notes = context.get("personal_notes", []) or []
        pn_lines: list[str] = []
        note_images: list[dict] = []  # Collect images for multimodal models

        for i, note in enumerate(personal_notes, start=1):
            if isinstance(note, dict):
                title = note.get("metadata", {}).get("title", "")
                section = note.get("metadata", {}).get("section", "")
                tags = note.get("metadata", {}).get("tags", "")
                content = note.get("content", "")
                image_data = note.get("image_data", [])  # Base64 encoded images
                # Sanitize content to prevent embedded headers
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, section, tags, content = "", "", "", str(note)
                image_data = []

            if content:
                # Build header: **Title** (Section) #tag1 #tag2
                header_parts = []
                if title:
                    header_parts.append(f"**{title}**")
                if section:
                    header_parts.append(f"({section})")
                if tags:
                    # Convert comma-separated tags to hashtag format
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    if tag_list:
                        header_parts.append(" ".join(f"#{t}" for t in tag_list))

                # Add relevance score so the LLM can see match strength
                relevance = note.get("relevance_score", 0.0)
                if relevance > 0:
                    header_parts.append(f"[relevance: {relevance:.2f}]")

                # Add image indicator if images are present
                if image_data:
                    header_parts.append(f"[{len(image_data)} image(s) attached]")
                    # Collect images with context about which note they belong to
                    for img in image_data:
                        note_images.append({
                            "note_index": i,
                            "note_title": title,
                            "note_section": section,
                            "filename": img.get("filename", ""),
                            "media_type": img.get("media_type", ""),
                            "data": img.get("data", ""),
                        })

                header = " ".join(header_parts) if header_parts else ""
                pn_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if pn_lines:
            pn_header = f"[USER'S PERSONAL NOTES] n={len(pn_lines)}\n"
            if note_images:
                pn_header += (
                    "Note: Images below were retrieved from the user's personal notes as context. "
                    "Do NOT comment on, describe, or ask about these images unless the user's "
                    "current message explicitly asks about them or their subject matter.\n\n"
                )
            sections.append(pn_header + "\n\n".join(pn_lines))

        # Store images in context for multimodal API calls
        if note_images:
            context["note_images"] = note_images
            total_data_size = sum(len(img.get("data", "")) for img in note_images)
            logger.warning(f"[PromptBuilder] IMAGE DEBUG: {len(note_images)} images collected, total base64 size={total_data_size//1024}KB")
        else:
            # Check why no images
            total_image_data = sum(len(note.get("image_data", [])) for note in personal_notes if isinstance(note, dict))
            logger.warning(f"[PromptBuilder] IMAGE DEBUG: No images in note_images list. personal_notes has {len(personal_notes)} notes, total image_data entries={total_image_data}")

        # User Uploaded Items (files and images uploaded during sessions)
        user_uploads = context.get("user_uploads", []) or []
        uu_lines: list[str] = []
        upload_images: list[dict] = []  # Collect images for multimodal models

        for i, upload in enumerate(user_uploads, start=1):
            if isinstance(upload, dict):
                meta = upload.get("metadata", {})
                title = meta.get("title", "")
                is_image = meta.get("is_image", False)
                media_type = meta.get("media_type", "")
                image_path = meta.get("image_path", "")
                content = upload.get("content", "")
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, is_image, media_type, image_path, content = "", False, "", "", str(upload)

            if content:
                header_parts = []
                if title:
                    # Strip "upload:" prefix for cleaner display
                    display_title = title[7:] if title.startswith("upload:") else title
                    header_parts.append(f"**{display_title}**")
                if is_image:
                    header_parts.append(f"[image: {media_type}]")
                    # Load persisted image for multimodal API calls
                    if image_path:
                        img_data = _load_upload_image(image_path)
                        if img_data:
                            upload_images.append({
                                "note_index": 0,
                                "note_title": f"Upload: {title}",
                                "note_section": "",
                                "filename": img_data.get("filename", ""),
                                "media_type": img_data.get("media_type", ""),
                                "data": img_data.get("data", ""),
                            })
                header = " ".join(header_parts) if header_parts else ""
                uu_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if uu_lines:
            sections.append(f"[USER UPLOADED ITEMS] n={len(uu_lines)}\n" + "\n\n".join(uu_lines))

        # Merge upload images into note_images for multimodal API calls
        if upload_images:
            existing_images = context.get("note_images", [])
            existing_images.extend(upload_images)
            context["note_images"] = existing_images
            logger.debug(f"[PromptBuilder] Merged {len(upload_images)} upload images into note_images")

        # Visual Memories (CLIP-matched images from personal collection)
        visual_mems = context.get("visual_memories", {})
        vm_text_results = visual_mems.get("text_results", []) if isinstance(visual_mems, dict) else []
        vm_images = visual_mems.get("images", []) if isinstance(visual_mems, dict) else []

        vm_lines: list[str] = []
        for i, result in enumerate(vm_text_results, start=1):
            caption = result.get("caption", "")
            source = result.get("source", "")
            score = result.get("score", 0.0)
            entities = result.get("entity_ids", [])
            header_parts = [f"[{source}]"]
            if entities:
                header_parts.append(f"entities: {', '.join(entities)}")
            header_parts.append(f"[relevance: {score:.2f}]")
            if caption:
                vm_lines.append(f"{i}) {' '.join(header_parts)}\n{_sanitize_embedded_headers(caption)}")

        if vm_lines:
            vm_instruction = (
                "These images were retrieved from your visual memory. "
                "IMPORTANT: Do NOT mention, describe, or comment on any image unless the user's query "
                "explicitly asks about photos, images, or the specific subject shown. "
                "A casual greeting or unrelated question means IGNORE all images completely. "
                "When an image IS directly relevant to what the user asked, naturally reference what "
                "you observe and connect it to the conversation."
            )
            sections.append(
                f"[VISUAL MEMORIES] n={len(vm_lines)}\n{vm_instruction}\n\n" + "\n\n".join(vm_lines)
            )

        # Merge visual memory images into note_images for multimodal API calls
        if vm_images:
            existing_images = context.get("note_images", [])
            existing_images.extend(vm_images)
            context["note_images"] = existing_images
            logger.debug(f"[PromptBuilder] Merged {len(vm_images)} visual memory images into note_images")

        # Reference Documents (system docs, project outlines, etc.)
        reference_docs = context.get("reference_docs", []) or []
        rd_lines: list[str] = []
        for i, doc in enumerate(reference_docs, start=1):
            if isinstance(doc, dict):
                title = doc.get("metadata", {}).get("title", "")
                section = doc.get("metadata", {}).get("section", "")
                file_type = doc.get("metadata", {}).get("file_type", "")
                content = doc.get("content", "")
                # Sanitize content to prevent embedded headers
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, section, file_type, content = "", "", "", str(doc)

            if content:
                # Build header: **Title** (Section) [type]
                header_parts = []
                if title:
                    header_parts.append(f"**{title}**")
                if section:
                    header_parts.append(f"({section})")
                if file_type:
                    header_parts.append(f"[{file_type}]")
                header = " ".join(header_parts) if header_parts else ""
                rd_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if rd_lines:
            sections.append(f"[DAEMON DOCUMENTATION] n={len(rd_lines)}\n" + "\n\n".join(rd_lines))

        # Git commit history (procedural memory)
        git_commits = context.get("git_commits", []) or []
        gc_lines: list[str] = []
        for i, commit in enumerate(git_commits, start=1):
            if isinstance(commit, dict):
                content = commit.get("content", "")
                meta = commit.get("metadata", {})
                commit_hash = meta.get("commit_hash", "")
                author = meta.get("author", "")
                age = meta.get("age_relative", "")
                tags = meta.get("tags", "")
            else:
                content = str(commit)
                commit_hash, author, age, tags = "", "", "", ""

            if content:
                header_parts = []
                if commit_hash:
                    header_parts.append(f"[{commit_hash}]")
                if author:
                    header_parts.append(f"by {author}")
                if age:
                    header_parts.append(f"({age})")
                if tags:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip() and t.strip() != "git-commit"]
                    if tag_list:
                        header_parts.append(" ".join(f"#{t}" for t in tag_list))
                header = " ".join(header_parts) if header_parts else ""
                gc_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if gc_lines:
            sections.append(f"[PROJECT COMMIT HISTORY] n={len(gc_lines)}\n" + "\n\n".join(gc_lines))

        # Procedural skills (adaptive workflows)
        proc_skills = context.get("procedural_skills", []) or []
        sk_lines: list[str] = []
        for i, skill in enumerate(proc_skills, start=1):
            if isinstance(skill, dict):
                meta = skill.get("metadata", {})
                trigger = meta.get("trigger", "")
                action = meta.get("action_pattern", "")
                category = meta.get("category", "")
                confidence = meta.get("confidence", "")
                tags_raw = meta.get("tags_json", "")
                created_at = meta.get("created_at", 0)
            else:
                trigger = str(skill)
                action, category, confidence, tags_raw, created_at = "", "", "", "", 0

            if trigger and action:
                parts = []
                if category:
                    parts.append(f"[{category}]")
                # Relative age from created_at epoch
                if created_at:
                    try:
                        import time as _time
                        age_secs = _time.time() - float(created_at)
                        if age_secs < 3600:
                            age_str = f"{int(age_secs / 60)} minutes ago"
                        elif age_secs < 86400:
                            age_str = f"{int(age_secs / 3600)} hours ago"
                        else:
                            age_str = f"{int(age_secs / 86400)} days ago"
                        parts.append(f"({age_str})")
                    except (ValueError, TypeError):
                        pass
                if confidence:
                    try:
                        parts.append(f"(conf={float(confidence):.0%})")
                    except (ValueError, TypeError):
                        pass
                if tags_raw:
                    try:
                        tag_list = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                        if tag_list:
                            parts.append(" ".join(f"#{t}" for t in tag_list))
                    except Exception:
                        pass
                header = " ".join(parts) if parts else ""
                entry = f"{i}) {header}\nWHEN: {trigger}\nTHEN: {action}" if header else f"{i}) WHEN: {trigger}\nTHEN: {action}"
                sk_lines.append(entry)

        if sk_lines:
            sections.append(f"[ADAPTIVE WORKFLOWS] n={len(sk_lines)}\n" + "\n\n".join(sk_lines))

        # Proposed Features (code proposals surfaced for project-related queries)
        proposed_features = context.get("proposed_features", [])
        logger.info(f"[PROPOSED_FEATURES] _assemble_prompt: {len(proposed_features)} proposals in context")
        pf_lines = []
        for i, pf in enumerate(proposed_features, 1):
            meta = pf.get("metadata", {})
            title = meta.get("title", "Untitled")
            ptype = meta.get("proposal_type", "feature")
            priority = meta.get("priority", 5)
            tags_raw = meta.get("tags_json", "[]")
            reasoning = meta.get("reasoning", "")

            try:
                tag_list = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
            except Exception:
                tag_list = []

            tag_str = " ".join(f"#{t}" for t in tag_list) if tag_list else ""
            header = f"[{ptype}] P{priority}"
            if tag_str:
                header += f" {tag_str}"
            header += f" **{title}**"

            entry = f"{i}) {header}"
            if reasoning:
                entry += f"\n   Rationale: {reasoning[:200]}"
            pf_lines.append(entry)

        if pf_lines:
            sections.append(f"[PROPOSED FEATURES] n={len(pf_lines)}\n" + "\n\n".join(pf_lines))

        # Knowledge Graph context (entity relationships)
        graph_sentences = context.get("graph_context", []) or []
        if graph_sentences:
            graph_block = "\n".join(f"- {s}" for s in graph_sentences)
            try:
                from config.app_config import ENABLE_GRAPH_ATTRIBUTION
                if ENABLE_GRAPH_ATTRIBUTION:
                    sections.append(f"[KNOWLEDGE GRAPH] n={len(graph_sentences)} (derived relationships)\n{graph_block}")
                else:
                    sections.append(f"[KNOWLEDGE GRAPH] n={len(graph_sentences)}\n{graph_block}")
            except ImportError:
                sections.append(f"[KNOWLEDGE GRAPH] n={len(graph_sentences)}\n{graph_block}")

        # Unresolved threads (proactive surfacing)
        unresolved_threads = context.get("unresolved_threads", []) or []
        if unresolved_threads:
            thread_lines = []
            for t in unresolved_threads:
                ttype = t.get("thread_type", "unfinished")
                topic = t.get("topic", "")
                summary = t.get("summary", "")
                deadline = t.get("deadline_date")
                line = f"- [{ttype}] {topic}: {summary}"
                if deadline:
                    line += f" (deadline: {deadline})"
                thread_lines.append(line)
            sections.append(f"[UNRESOLVED THREADS] n={len(thread_lines)}\n" + "\n".join(thread_lines))

        # Upcoming schedule (gated — only present when retrieval was triggered)
        upcoming_schedule = context.get("upcoming_schedule", []) or []
        if upcoming_schedule:
            sched_lines = []
            for evt in upcoming_schedule:
                display_date = evt.get("display_date", "")
                kind = evt.get("schedule_kind", "")
                event_type = evt.get("event_type", kind)
                start = evt.get("schedule_start", "")
                end = evt.get("schedule_end", "")
                scope = evt.get("schedule_scope", "")
                confidence = float(evt.get("parser_confidence", 1.0))
                basis = evt.get("resolution_basis", "")

                # Build display line
                label = event_type.replace("_", " ").title()
                if start and end:
                    # Format times for display
                    try:
                        from datetime import datetime as _dt
                        s_dt = _dt.strptime(start, "%H:%M")
                        e_dt = _dt.strptime(end, "%H:%M")
                        time_str = f"{s_dt.strftime('%I:%M %p').lstrip('0')} – {e_dt.strftime('%I:%M %p').lstrip('0')}"
                    except (ValueError, TypeError):
                        time_str = f"{start} – {end}"
                    line = f"• {display_date}: {label} {time_str}"
                elif kind == "shift_pattern":
                    shift_val = evt.get("object", event_type)
                    line = f"• {label}: {shift_val}"
                elif kind == "day_off":
                    line = f"• {display_date}: Day off"
                elif kind == "exam_date":
                    line = f"• {display_date}: Exam"
                else:
                    line = f"• {display_date}: {label}"

                # Confidence qualifier for heuristic resolutions
                if confidence < 0.75 and basis not in ("explicit_ampm", "explicit_24h", "explicit_named"):
                    line += f" (inferred from context)"
                if scope == "one_off":
                    line += " (one-time)"

                sched_lines.append(line)

            sections.append(
                f"[UPCOMING SCHEDULE] n={len(sched_lines)}\n" + "\n".join(sched_lines)
            )

        # Daemon self-notes (working context from prior sessions)
        daemon_self_notes = context.get("daemon_self_notes", []) or []
        if daemon_self_notes:
            note_lines = []
            for note in daemon_self_notes:
                meta = note.get("metadata", {}) or {}
                category = meta.get("category", "general")
                confidence = meta.get("confidence", "medium")
                created = meta.get("created", "")[:10]
                content = note.get("content", "")
                note_lines.append(
                    f"[DAEMON SELF-NOTE | {category} | confidence: {confidence} | {created}]\n{content}"
                )
            sections.append(
                f"[DAEMON SELF-NOTES] n={len(note_lines)}\n"
                "These are Daemon's own working notes from prior sessions — not user-authored facts.\n"
                "Do not cite as established facts. Use as working context only.\n\n"
                + "\n\n".join(note_lines)
            )

        # Proactive cross-domain insights
        proactive_insights = context.get("proactive_insights", []) or []
        if proactive_insights:
            insight_block = "\n".join(f"- {s}" for s in proactive_insights)
            sections.append(
                f"[PROACTIVE INSIGHTS] n={len(proactive_insights)}\n"
                "These are AI-generated insights based on relationship analysis. "
                "Reference naturally when relevant, but clearly distinguish from user-provided facts.\n"
                f"{insight_block}")

        # User Profile (replaces semantic_facts + fresh_facts)
        # MOVED: Placed here (after bulk knowledge, before query) for high attention with low token cost
        user_profile = context.get("user_profile", "")
        if user_profile and isinstance(user_profile, str):
            # Count facts (each fact ends with [timestamp])
            fact_count = user_profile.count('[20')  # Count timestamp brackets starting with [20xx
            sections.append(
                f"[USER PROFILE] n={fact_count}\n"
                "Stored facts — reference naturally but do not add names, apps, or details not written here.\n"
                f"{user_profile}")

        # Active Features Inventory (always present, compact)
        feature_inventory = self._build_feature_inventory(context)
        if feature_inventory:
            sections.append(f"[ACTIVE FEATURES]\n{feature_inventory}")

        # Codebase changes since last session (first message only)
        codebase_changes = context.get("codebase_changes", {})
        if codebase_changes:
            cc_lines = []
            since_label = codebase_changes.get("since_label", "last session")
            committed = codebase_changes.get("committed", [])
            uncommitted_mod = codebase_changes.get("uncommitted_modified", [])
            uncommitted_new = codebase_changes.get("uncommitted_new", [])
            if committed:
                cc_lines.append(f"Committed ({len(committed)}):")
                for c in committed:
                    cc_lines.append(f"  - {c}")
            if uncommitted_mod:
                cc_lines.append(f"Modified uncommitted ({len(uncommitted_mod)}):")
                for f in uncommitted_mod:
                    cc_lines.append(f"  - {f}")
            if uncommitted_new:
                cc_lines.append(f"New untracked ({len(uncommitted_new)}):")
                for f in uncommitted_new:
                    cc_lines.append(f"  - {f}")
            if cc_lines:
                total = len(committed) + len(uncommitted_mod) + len(uncommitted_new)
                sections.append(
                    f"[CODEBASE CHANGES SINCE LAST SESSION] n={total} (since {since_label})\n"
                    + "\n".join(cc_lines))

        # Time context
        # MOVED: Placed here (right before STM and query) for temporal grounding with high attention
        time_ctx = self._get_time_context()
        if time_ctx:
            sections.append(f"[TIME CONTEXT]\n{time_ctx}")

        # Temporal Grounding (Narrative Context) - synthesized life state for trajectory awareness
        narrative_state = context.get("narrative_state", "")
        if narrative_state and isinstance(narrative_state, str) and narrative_state.strip():
            sections.append(f"[TEMPORAL GROUNDING]\n{narrative_state}")
            logger.debug(f"[PROMPT ASSEMBLY] Added temporal grounding section ({len(narrative_state)} chars)")

        # STM (Short-Term Memory) Summary - placed right before query for maximum attention
        stm_summary = context.get("stm_summary")
        logger.warning(f"STM RENDERING CHECK: stm_summary = {stm_summary}")
        if stm_summary:
            logger.warning("STM RENDERING: Rendering STM section before query")
            stm_lines = []
            stm_lines.append(f"Topic: {stm_summary.get('topic', 'unknown')}")
            stm_lines.append(f"User Question: {stm_summary.get('user_question', '')}")
            stm_lines.append(f"Intent: {stm_summary.get('intent', '')}")
            stm_lines.append(f"Tone: {stm_summary.get('tone', 'neutral')}")

            ref_type = stm_summary.get('reference_type', 'unclear')
            stm_lines.append(f"Reference Type: {ref_type}")
            if ref_type == 'recall':
                stm_lines.append(
                    "WARNING: The current message restates an event already in context. "
                    "Do NOT count it as a separate occurrence and do NOT fabricate a "
                    "pattern from a single underlying event."
                )
            elif ref_type == 'clarification':
                stm_lines.append(
                    "WARNING: The current message adds detail to an existing topic. "
                    "Do NOT treat it as a new event."
                )
            elif ref_type == 'correction':
                stm_lines.append(
                    "WARNING: The user is correcting a prior assistant claim. "
                    "Update your understanding; do not double down."
                )
            elif ref_type == 'unclear':
                stm_lines.append(
                    "WARNING: Reference is ambiguous. Before treating the current message "
                    "as a new event, verify it is not already present in [RELEVANT MEMORIES] "
                    "or [RECENT CONVERSATION]. Do NOT invent counts or patterns from a "
                    "single underlying occurrence."
                )

            temporal_facts = stm_summary.get('temporal_facts', [])
            if temporal_facts:
                stm_lines.append(f"Resolved State: {' | '.join(temporal_facts)}")

            open_threads = stm_summary.get('open_threads', [])
            if open_threads:
                stm_lines.append(f"Open Threads: {', '.join(open_threads)}")

            constraints = stm_summary.get('constraints', [])
            if constraints:
                stm_lines.append(f"Constraints: {', '.join(constraints)}")

            sections.append(f"[SHORT-TERM CONTEXT SUMMARY]\n" + "\n".join(stm_lines))
            logger.warning(f"STM RENDERING: Added STM section before query")
        else:
            logger.warning("STM RENDERING: No stm_summary in context, skipping section")

        # Disambiguation notes (injected just before user query for maximum attention)
        disambiguation = context.get("disambiguation_notes", []) or []
        if disambiguation:
            sections.append("\n".join(disambiguation))

        # User input with last Q/A pair for coherence
        if user_input:
            query_section = f"[CURRENT USER QUERY]\n"

            # Attach last Q/A pair for maximum coherence (high attention area)
            recent = context.get("recent_conversations", [])
            if recent and len(recent) > 0:
                last_exchange = recent[0]  # First item is most recent (list ordered newest-first)
                last_q = last_exchange.get("query", "")
                last_a = last_exchange.get("response", "")
                if last_q and last_a:
                    query_section += f"[LAST EXCHANGE FOR CONTEXT]\n"
                    query_section += f"User: {last_q}\n"
                    query_section += f"Assistant: {last_a}\n\n"

            query_section += f"[CURRENT QUERY]\n{user_input}"
            sections.append(query_section)

        # DEBUG: Check for duplicate section headers before returning
        section_headers = [s.split('\n')[0] for s in sections if s]
        header_counts = {}
        for header in section_headers:
            if header.startswith('['):
                header_counts[header] = header_counts.get(header, 0) + 1

        duplicates = {h: c for h, c in header_counts.items() if c > 1}
        if duplicates:
            logger.error(f"[DEBUG RECENT] DUPLICATE SECTIONS DETECTED: {duplicates}")
            logger.error(f"[DEBUG RECENT] Total sections: {len(sections)}, section headers: {section_headers}")
        else:
            logger.debug(f"[DEBUG RECENT] No duplicate sections. Total sections: {len(sections)}")

        final_prompt = "\n\n".join(sections)

        # Count how many times "[RECENT CONVERSATION]" appears in final assembled prompt
        recent_conv_count = final_prompt.count("[RECENT CONVERSATION]")
        if recent_conv_count > 1:
            logger.error(f"[DEBUG RECENT] FINAL PROMPT HAS {recent_conv_count} [RECENT CONVERSATION] HEADERS!")
            # Find positions
            matches = [(m.start(), m.end()) for m in re.finditer(r'\[RECENT CONVERSATION\]', final_prompt)]
            logger.error(f"[DEBUG RECENT] Found at positions: {matches}")
            for i, (start, end) in enumerate(matches):
                context_start = max(0, start - 50)
                context_end = min(len(final_prompt), end + 200)
                logger.error(f"[DEBUG RECENT] Match {i+1} context: ...{final_prompt[context_start:context_end]}...")

        # --- Eval snapshot hook (gated, read-only) ---
        # Lazy import to avoid circular dependency (function lives in builder.py)
        try:
            from .builder import _maybe_capture_eval_snapshot
            _maybe_capture_eval_snapshot(context, user_input, sections, final_prompt)
        except ImportError:
            pass

        return final_prompt
