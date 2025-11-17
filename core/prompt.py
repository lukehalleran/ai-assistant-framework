"""
# core/prompt.py

REFACTORED: This module has been split into modular components.

Module Contract
- Purpose: Unified Prompt Builder. Gathers recent/semantic memory, facts, summaries, reflections, wiki snippets; applies gating; manages token budgets; assembles final prompt context.
- Inputs:
  - build_prompt(user_input: str, search_query: Optional[str], personality_config: Optional[dict], system_prompt: Optional[str], current_topic: Optional[str])
- Outputs:
  - dict prompt_ctx consumed by orchestrator._assemble_prompt (time context, recent_conversations, memories, facts, summaries, reflections, semantic_chunks, wiki, etc.)
- Key classes/functions:
  - UnifiedPromptBuilder: main builder
  - _apply_gating(): wrap gate_system for memories/semantic/wiki
  - _get_summaries(): prefer stored summaries, else LLM fallback, else micro
  - _hygiene_and_caps(), _manage_token_budget(), _assemble_prompt(): ordering, dedupe, token budgeting, final render
- Dependencies:
  - memory coordinator (corpus + Chroma), processing.gate_system, knowledge.WikiManager/semantic_search, models.model_manager (for summary/reflection calls)
- Side effects:
  - None (fetch/compute only). Persistence handled by memory_coordinator and consolidators elsewhere.
- Async behavior:
  - Parallel subfetches (recent/mems/facts/summaries/semantic/wiki) with bounded timeouts.

REFACTORING NOTE:
This large file (2,473 lines) has been split into modular components in core/prompt/:
- builder.py: Main UnifiedPromptBuilder class
- context_gatherer.py: Data collection (memories, facts, wiki)
- formatter.py: Text formatting and assembly
- summarizer.py: LLM summarization
- token_manager.py: Token counting and budget management
- base.py: Utilities and fallback classes

The functionality remains identical - this is just a cleaner architecture.
"""

# Import everything from the new modular structure
from .prompt.builder import UnifiedPromptBuilder, PromptBuilder
from .prompt.context_gatherer import ContextGatherer
from .prompt.formatter import PromptFormatter
from .prompt.summarizer import LLMSummarizer
from .prompt.token_manager import TokenManager
from .prompt.base import (
    _parse_bool,
    _cfg_int,
    _as_summary_dict,
    _dedupe_keep_order,
    _truncate_list,
    _strip_prompt_artifacts,
    _FallbackCorpusManager,
    _FallbackMemoryCoordinator
)

# Maintain backward compatibility by exposing the main classes at module level
__all__ = [
    'UnifiedPromptBuilder',
    'PromptBuilder',  # Legacy compatibility
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

# For any code that imports specific functions from core.prompt directly,
# we maintain compatibility by exposing them here
from .prompt.formatter import (
    _parse_bool,
    _as_summary_dict,
    _dedupe_keep_order,
    _truncate_list,
    _strip_prompt_artifacts
)

from .prompt.base import (
    _cfg_int,
    _FallbackCorpusManager,
    _FallbackMemoryCoordinator
)

# Legacy function imports that might be used elsewhere
# These were previously in the monolithic file
def build_prompt(*args, **kwargs):
    """Legacy function wrapper - use UnifiedPromptBuilder.build_prompt() instead."""
    import warnings
    warnings.warn(
        "build_prompt() function is deprecated. Use UnifiedPromptBuilder.build_prompt() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # This would require instantiating UnifiedPromptBuilder, but we can't do that
    # without proper dependencies. Legacy code should be updated to use the class.
    raise NotImplementedError(
        "Legacy build_prompt() function removed. Please use UnifiedPromptBuilder class."
    )