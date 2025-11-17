"""
# core/prompt/__init__.py

Module Contract
- Purpose: Core prompt building module providing unified interface to refactored components.
- Inputs:
  - Public classes: UnifiedPromptBuilder, ContextGatherer, PromptFormatter, etc.
  - Utility functions: _parse_bool, _cfg_int, formatting helpers
- Outputs:
  - Clean public API for prompt building functionality
  - Organized access to all prompt building components
  - Utility functions for configuration and data processing
- Behavior:
  - Provides single import point for all prompt building functionality
  - Exposes main classes (UnifiedPromptBuilder, TokenManager, etc.)
  - Makes utility functions and fallback classes available
  - Maintains backward compatibility through aliases
- Dependencies:
  - All submodules: builder, context_gatherer, formatter, summarizer, token_manager, base
- Side effects:
  - None; pure module organization and interface definition

Package Structure:
- builder: Main UnifiedPromptBuilder orchestration class
- token_manager: Token counting and budget management
- context_gatherer: Data collection (memories, facts, wiki)
- formatter: Text formatting and assembly
- summarizer: LLM summarization and reflection generation
- base: Utilities and fallback classes
"""

# Main public interface
from .builder import UnifiedPromptBuilder, PromptBuilder
from .context_gatherer import ContextGatherer
from .formatter import PromptFormatter
from .summarizer import LLMSummarizer
from .token_manager import TokenManager

# Utility functions and fallback classes
from .base import (
    _parse_bool,
    _cfg_int,
    _FallbackCorpusManager,
    _FallbackMemoryCoordinator
)

from .formatter import (
    _as_summary_dict,
    _dedupe_keep_order,
    _truncate_list,
    _strip_prompt_artifacts
)

__all__ = [
    'UnifiedPromptBuilder',
    'PromptBuilder',
    'ContextGatherer',
    'PromptFormatter',
    'LLMSummarizer',
    'TokenManager',
    '_parse_bool',
    '_cfg_int',
    '_as_summary_dict',
    '_dedupe_keep_order',
    '_truncate_list',
    '_strip_prompt_artifacts',
    '_FallbackCorpusManager',
    '_FallbackMemoryCoordinator'
]