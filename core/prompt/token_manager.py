"""
# core/prompt/token_manager.py

Module Contract
- Purpose: Token management and budget control for prompt building system.
- Inputs:
  - extract_text_from_context(context: Dict[str, Any]) -> str
  - trim_to_budget(context: Dict[str, Any], budget: int) -> Dict[str, Any]
  - count_tokens(text: str) -> int
- Outputs:
  - Extracted text content from structured context
  - Budget-compliant context with trimmed content
  - Token counts for text validation
- Behavior:
  - Extracts text from various context sections (conversations, memories, facts, etc.)
  - Applies priority-based trimming when content exceeds token budget
  - Uses middle-out compression for large text blocks
  - Preserves most important content while respecting limits
- Dependencies:
  - models.tokenizer_manager.TokenizerManager (token counting)
  - utils.logging_utils (logging)
- Side effects:
  - Logging of token usage and trimming actions
"""

import os
from typing import Dict, Any
from utils.logging_utils import get_logger

logger = get_logger("prompt_token_manager")

# Constants for token management
PRIORITY_ORDER = [
    ("recent_conversations", 7),
    ("semantic_chunks",      6),
    ("memories",             5),
    ("facts",                4),
    ("summaries",            3),
    ("reflections", 2),  # below summaries; adjust if you want them stickier
    ("wiki",                 1),
    ("dreams",               2),   # still included; trimmed early if needed
]

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

def _parse_bool(s: str, default: bool = False) -> bool:
    """Parse boolean from string, with fallback."""
    if not s:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")

# Token limits and configuration
ENABLE_MIDDLE_OUT = _parse_bool(os.getenv("ENABLE_MIDDLE_OUT", "1"))
USER_INPUT_MAX_TOKENS = int(os.getenv("USER_INPUT_MAX_TOKENS", "4096"))
MEMORY_ITEM_MAX_TOKENS = int(os.getenv("MEMORY_ITEM_MAX_TOKENS", "512"))
SEMANTIC_ITEM_MAX_TOKENS = int(os.getenv("SEMANTIC_ITEM_MAX_TOKENS", "800"))


class TokenManager:
    """Handles token counting, budgets, and text compression for prompt building."""

    def __init__(self, model_manager, tokenizer_manager, token_budget: int):
        self.model_manager = model_manager
        self.tokenizer_manager = tokenizer_manager
        self.token_budget = token_budget
        self._prompt_token_usage = 0

    def get_token_count(self, text: str, model_name: str) -> int:
        """Delegate to tokenizer_manager; keeps compatibility with your models."""
        return self.tokenizer_manager.count_tokens(text or "", model_name)

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats for token counting."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("content", "text", "response", "filtered_content"):
                if key in item and item[key]:
                    return str(item[key])
            return str(item)
        return str(item)

    def _middle_out(self, text: str, max_tokens: int, head_ratio: float = 0.6, force: bool = False) -> str:
        """Compress text by keeping the head and tail, trimming the middle.

        Uses tokenizer_manager to decide if compression is needed, but slices by characters
        to avoid requiring a full encode/decode path. Good enough for budget safety.

        Args:
            text: Text to potentially compress
            max_tokens: Maximum tokens for this item
            head_ratio: Ratio of head to tail (default 0.6 = 60% head, 40% tail)
            force: If True, apply compression regardless of prompt size. If False (default),
                   only compress when total prompt would exceed 20K tokens.
        """
        if not ENABLE_MIDDLE_OUT:
            return text

        # Only apply middle-out if we're above the token budget
        # unless explicitly forced
        if not force and hasattr(self, '_prompt_token_usage'):
            if self._prompt_token_usage < self.token_budget:
                return text

        try:
            model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else "default"
            toks = self.get_token_count(text or "", model_name)
        except Exception:
            toks = len((text or "").split())
        if toks <= max_tokens:
            return text
        s = text or ""
        # Roughly map tokens to characters; assume ~4 chars per token as a conservative heuristic
        approx_chars = max_tokens * 4
        head_chars = int(approx_chars * head_ratio)
        tail_chars = max(0, approx_chars - head_chars)
        if len(s) <= approx_chars:
            return s
        head = s[:head_chars]
        tail = s[-tail_chars:] if tail_chars > 0 else ""
        snip = f"\\n… [middle-out snipped {len(s) - (head_chars + tail_chars)} chars] …\\n"
        return head + snip + tail

    def _manage_token_budget(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trim sections in increasing priority order until we fit within token budget.
        For lists, we remove items from the tail in conservative chunks (25%) to
        avoid over-trimming; for strings, we blank the whole section if needed.
        """
        model_name = self.model_manager.get_active_model_name()
        trimmed = dict(context)
        current_tokens = 0

        # Helper to count tokens for any item
        def _item_tokens(item: Any) -> int:
            text = self._extract_text(item)
            return self.get_token_count(text, model_name)

        # First pass: optimistic inclusion respecting soft caps already applied
        for name, _prio in PRIORITY_ORDER:
            val = trimmed.get(name)
            if not val:
                continue
            logger.warning(f"[TOKEN BUDGET] Processing section '{name}' with {len(val) if isinstance(val, list) else 1} items, current_tokens={current_tokens}")
            if isinstance(val, list):
                kept = []
                for i, item in enumerate(val):
                    t = _item_tokens(item)
                    if current_tokens + t <= self.token_budget:
                        kept.append(item)
                        current_tokens += t
                    else:
                        if name == "memories":
                            logger.warning(f"[TOKEN BUDGET] Stopped adding memories at item {i}/{len(val)}: budget={self.token_budget}, current={current_tokens}, item_tokens={t}")
                        break
                if name == "memories":
                    logger.warning(f"[TOKEN BUDGET] Kept {len(kept)}/{len(val)} memories, budget={self.token_budget}, used={current_tokens}")
                trimmed[name] = kept
            else:
                t = self.get_token_count(str(val), model_name)
                if current_tokens + t <= self.token_budget:
                    current_tokens += t
                else:
                    # We'll consider dropping this later in the second pass.
                    pass

        # If we're still over (due to some large strings), trim by priority
        def _total_tokens(ctx: Dict[str, Any]) -> int:
            total = 0
            for name, _ in PRIORITY_ORDER:
                v = ctx.get(name)
                if not v:
                    continue
                if isinstance(v, list):
                    for it in v:
                        total += _item_tokens(it)
                else:
                    total += self.get_token_count(str(v), model_name)
            return total

        usage = _total_tokens(trimmed)
        logger.debug(f"[PROMPT] Token budget (pre-trim check): {usage}/{self.token_budget}")

        if usage > self.token_budget:
            # Second pass: trim from lowest priority upward
            for name, prio in sorted(PRIORITY_ORDER, key=lambda x: x[1]):  # low → high
                v = trimmed.get(name)
                if not v:
                    continue

                if isinstance(v, list) and v:
                    # Drop a conservative slice from the tail
                    drop_n = max(1, int(len(v) * 0.25))
                    trimmed[name] = v[:-drop_n]
                elif isinstance(v, str) and v:
                    trimmed[name] = ""

                usage = _total_tokens(trimmed)
                if usage <= self.token_budget:
                    break

        logger.debug(f"[PROMPT] Token budget: {usage}/{self.token_budget}")
        self._prompt_token_usage = usage
        return trimmed