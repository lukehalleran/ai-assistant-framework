"""
# core/prompt/formatter.py

Module Contract
- Purpose: Text formatting and assembly for final prompt rendering.
- Inputs:
  - render_prompt_sections(context: Dict[str, Any]) -> str
  - format_memories(memories: List[Dict]) -> str
  - format_conversations(conversations: List[Dict]) -> str
  - deduplicate_content(content: List[str]) -> List[str]
- Outputs:
  - Formatted prompt sections ready for model consumption
  - Structured text with proper headers and spacing
  - Deduplicated and truncated content within limits
- Behavior:
  - Assembles context sections into readable prompt format
  - Applies consistent formatting to memories, facts, conversations
  - Handles time context and metadata display
  - Deduplicates similar content and applies truncation
  - Maintains prompt structure and readability
- Dependencies:
  - utils.logging_utils (logging)
  - datetime (time formatting)
- Side effects:
  - Logging of formatting actions and content statistics
"""

import os
import re
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime
from utils.logging_utils import get_logger

logger = get_logger("prompt_formatter")

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except Exception:
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except Exception:
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

def _strip_prompt_artifacts(text: str) -> str:
    """Remove known bracketed prompt headers if the model echoes them."""
    if not text:
        return text
    try:
        header_patterns = [
            r"^\s*\[TIME CONTEXT\]",
            r"^\s*\[RECENT CONVERSATION[^\]]*\]",
            r"^\s*\[RELEVANT INFORMATION\]",
            r"^\s*\[RELEVANT MEMORIES\]",
            r"^\s*\[FACTS[ ^\]]*\]",
            r"^\s*\[SEMANTIC FACTS\]",
            r"^\s*\[RECENT FACTS\]",
            r"^\s*\[CURRENT MESSAGE FACTS\]",
            r"^\s*\[DIRECTIVES\]",
            r"^\s*\[CURRENT USER QUERY[ ^\]]*\]",
            r"^\s*\[USER INPUT\]",
            r"^\s*\[BACKGROUND KNOWLEDGE\]",
            r"^\s*\[CONVERSATION SUMMARIES[ ^\]]*\]",
            r"^\s*\[RECENT REFLECTIONS[ ^\]]*\]",
            r"^\s*\[SESSION REFLECTIONS[ ^\]]*\]",
        ]
        header_re = re.compile("(" + ")|(".join(header_patterns) + ")", re.IGNORECASE)
        lines = []
        skip_block = False
        for line in (text.splitlines() or []):
            if header_re.search(line):
                skip_block = True
                continue
            if skip_block:
                if not line.strip():
                    skip_block = False
                continue
            lines.append(line)
        return "\n".join(lines).strip()
    except Exception:
        return text


class PromptFormatter:
    """Handles text formatting and prompt assembly for prompt building."""

    def __init__(self, token_manager):
        self.token_manager = token_manager

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
        except Exception:
            pass
        return ""

    def _get_time_context(self) -> str:
        """Get current time context for the prompt."""
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

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
            if response:
                response = _strip_prompt_artifacts(response)

            # Get timestamp - check multiple possible locations and format as datetime
            timestamp = mem.get("timestamp", "")
            if not timestamp:
                meta = mem.get("metadata", {})
                timestamp = meta.get("timestamp", "")

            # Format timestamp as datetime string if it's a datetime object
            if isinstance(timestamp, datetime):
                datetime_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif timestamp:
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
            except:
                return "Memory (formatting error)"

    def _assemble_prompt(self, context: Dict[str, Any], user_input: str, directives: str = "") -> str:
        """
        Assemble final prompt from context and user input.

        This is the final step that converts the context dict into a formatted string
        ready for the LLM. The prompt structure follows a consistent format:
        - Time context
        - Recent conversations
        - Relevant memories
        - Facts and recent facts
        - Summaries
        - Reflections
        - Wiki content
        - Semantic chunks
        - Dreams
        - Directives
        - User input
        """
        sections = []

        # Time context
        time_ctx = self._get_time_context()
        if time_ctx:
            sections.append(f"[TIME CONTEXT]\n{time_ctx}")

        # STM (Short-Term Memory) Summary
        stm_summary = context.get("stm_summary")
        if stm_summary:
            stm_lines = []
            stm_lines.append(f"Topic: {stm_summary.get('topic', 'unknown')}")
            stm_lines.append(f"User Question: {stm_summary.get('user_question', '')}")
            stm_lines.append(f"Intent: {stm_summary.get('intent', '')}")
            stm_lines.append(f"Tone: {stm_summary.get('tone', 'neutral')}")

            open_threads = stm_summary.get('open_threads', [])
            if open_threads:
                stm_lines.append(f"Open Threads: {', '.join(open_threads)}")

            constraints = stm_summary.get('constraints', [])
            if constraints:
                stm_lines.append(f"Constraints: {', '.join(constraints)}")

            sections.append(f"[SHORT-TERM CONTEXT SUMMARY]\n" + "\n".join(stm_lines))

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

        # Semantic facts (query-relevant facts)
        semantic_facts = context.get("semantic_facts", [])
        if semantic_facts:
            fact_strs = []
            for fact in semantic_facts:
                if isinstance(fact, dict):
                    # Structured fact
                    subj = fact.get("subject", "")
                    rel = fact.get("relation", "")
                    obj = fact.get("object", "")
                    # Check both fact-level timestamp and metadata timestamp
                    timestamp = fact.get("timestamp", "")
                    if not timestamp:
                        metadata = fact.get("metadata", {})
                        timestamp = metadata.get("timestamp", "")
                    if subj and rel and obj:
                        fact_line = f"{subj} | {rel} | {obj}"
                        if timestamp:
                            fact_line += f" | {timestamp}"
                        fact_strs.append(fact_line)
                    else:
                        content = str(fact.get("content", fact))
                        if timestamp:
                            content += f" | {timestamp}"
                        fact_strs.append(content)
                else:
                    fact_strs.append(str(fact))
            if fact_strs:
                sections.append(f"[SEMANTIC FACTS] n={len(fact_strs)}\n" + "\n".join(fact_strs))

        # Recent facts
        recent_facts = context.get("fresh_facts", [])
        if recent_facts:
            recent_fact_strs = []
            for fact in recent_facts:
                if isinstance(fact, dict):
                    # Structured fact with timestamp support
                    subj = fact.get("subject", "")
                    rel = fact.get("relation", "")
                    obj = fact.get("object", "")
                    # Check both fact-level timestamp and metadata timestamp
                    timestamp = fact.get("timestamp", "")
                    if not timestamp:
                        metadata = fact.get("metadata", {})
                        timestamp = metadata.get("timestamp", "")

                    if subj and rel and obj:
                        fact_line = f"{subj} | {rel} | {obj}"
                        if timestamp:
                            fact_line += f" | {timestamp}"
                        recent_fact_strs.append(fact_line)
                    else:
                        content = fact.get("content", "") or str(fact)
                        if timestamp:
                            content += f" | {timestamp}"
                        recent_fact_strs.append(content)
                else:
                    recent_fact_strs.append(str(fact))
            if recent_fact_strs:
                sections.append(f"[RECENT FACTS] n={len(recent_fact_strs)}\n" + "\n".join(recent_fact_strs))

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
