"""
# core/prompt/base.py

Module Contract
- Purpose: Base utilities and fallback classes for the prompt building system.
- Utility functions:
  - _cfg_int(key, default) -> int  [config value with fallback]
  - _parse_bool(s, default) -> bool  [string to bool parsing]
  - _as_summary_dict(text, tags, source, timestamp) -> dict  [standardized summary format]
  - _dedupe_keep_order(items, key_fn) -> List  [dedup preserving insertion order]
  - _truncate_list(items, limit) -> List  [keep most recent N items]
  - _strip_prompt_artifacts(text) -> str  [remove bracketed prompt headers from echoed text]
- Fallback classes (for testing when real dependencies unavailable):
  - _FallbackCorpusManager: in-memory corpus with add_entry, get_recent_memories, get_summaries
  - _FallbackMemoryCoordinator: wraps _FallbackCorpusManager with async store_interaction,
    get_memories, retrieve_relevant_memories, get_summaries, get_dreams, get_facts
- Dependencies:
  - config.app_config (optional, graceful fallback to empty dict)
- Side effects:
  - None; pure utility functions and test stubs
"""

import os
import re
from typing import List, Any, Dict, Optional, Iterable
from datetime import datetime

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}

def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    """Parse boolean from string, with fallback."""
    if not s:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")

def _cfg_int(key: str, default_val: int) -> int:
    """Get integer config value with fallback."""
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

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
            r"^\\s*\\[TIME CONTEXT\\]",
            r"^\\s*\\[RECENT CONVERSATION[^\\]]*\\]",
            r"^\\s*\\[RELEVANT INFORMATION\\]",
            r"^\\s*\\[RELEVANT MEMORIES\\]",
            r"^\\s*\\[FACTS[ ^\\]]*\\]",
            r"^\\s*\\[RECENT FACTS\\]",
            r"^\\s*\\[CURRENT MESSAGE FACTS\\]",
            r"^\\s*\\[DIRECTIVES\\]",
            r"^\\s*\\[CURRENT USER QUERY[ ^\\]]*\\]",
            r"^\\s*\\[USER INPUT\\]",
            r"^\\s*\\[BACKGROUND KNOWLEDGE\\]",
            r"^\\s*\\[CONVERSATION SUMMARIES[ ^\\]]*\\]",
            r"^\\s*\\[RECENT REFLECTIONS[ ^\\]]*\\]",
            r"^\\s*\\[SESSION REFLECTIONS[ ^\\]]*\\]",
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
        return "\\n".join(lines).strip()
    except (re.error, TypeError, AttributeError):
        return text


class _FallbackCorpusManager:
    """Minimal corpus manager for testing when real one unavailable."""

    def __init__(self) -> None:
        self._entries = []

    def add_entry(self, query: str, response: str, tags=None, timestamp=None):
        """Add entry to in-memory corpus."""
        self._entries.append({
            "query": query,
            "response": response,
            "tags": tags or [],
            "timestamp": timestamp or datetime.now()
        })

    def get_recent_memories(self, count: int = 3):
        """Get recent entries."""
        return self._entries[-count:] if count > 0 else []

    def get_summaries(self, _count: int = 3):
        """Return empty summaries (fallback)."""
        return []

    def add_summary(self, *_, **__):
        """No-op summary addition."""
        pass


class _FallbackMemoryCoordinator:
    """Minimal memory coordinator for testing when real one unavailable."""

    def __init__(self) -> None:
        self.corpus_manager = _FallbackCorpusManager()

    async def store_interaction(self, query: str, response: str, tags=None):
        """Store interaction in fallback corpus."""
        self.corpus_manager.add_entry(query, response, tags)

    async def get_memories(self, _query: str, limit: int = 20, topic_filter: str | None = None):
        """Get memories from fallback corpus."""
        recents = self.corpus_manager.get_recent_memories(limit)
        return [{
            "query": item.get("query", ""),
            "response": item.get("response", ""),
            "metadata": {"source": "recent", "final_score": 1.0}
        } for item in recents]

    async def retrieve_relevant_memories(self, query: str, config=None):
        """Retrieve relevant memories with fallback behavior."""
        limit = (config or {}).get("recent_count", 5)
        memories = await self.get_memories(query, limit=limit)
        return {
            "memories": memories,
            "recent_conversations": memories[:3],
            "facts": [],
            "fresh_facts": []
        }

    def get_summaries(self, limit: int = 3):
        """Get summaries (fallback returns empty)."""
        return []

    def get_dreams(self, _limit: int = 2):
        """Get dreams (fallback returns empty)."""
        return []

    async def get_facts(self, *_, **__):
        """Get facts (fallback returns empty)."""
        return []