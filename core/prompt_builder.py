from __future__ import annotations
"""
# core/prompt_builder.py

Module Contract
- Purpose: Compatibility shims that expose legacy builder names while delegating to the refreshed UnifiedPromptBuilder.
- Classes:
  - UnifiedHierarchicalPromptBuilder: thin adapter that forwards build_prompt/build_gated_prompt calls to the provided delegate builder; preserves consolidator reference.
  - UnifiedPromptBuilder (alias): re-export of core.prompt.UnifiedPromptBuilder.
- Inputs/Outputs:
  - Same signature/return shape as underlying builders (async build_prompt yielding context dict).
- Side effects:
  - None. Pure delegation.
"""
"""Compatibility builders bridging legacy imports to the refreshed pipeline."""



import inspect
from typing import Any

from core.prompt import UnifiedPromptBuilder as _UnifiedPromptBuilder, PromptBuilder

UnifiedPromptBuilder = _UnifiedPromptBuilder


class UnifiedHierarchicalPromptBuilder:
    """Thin wrapper that adapts legacy constructor signatures to the new builders."""

    def __init__(
        self,
        *,
        prompt_builder: Any,
        model_manager: Any,
        chroma_store: Any = None,
        **_: Any,
    ) -> None:
        self._delegate = prompt_builder
        self.model_manager = model_manager
        self.chroma_store = chroma_store
        self.consolidator = getattr(prompt_builder, "consolidator", None)

    async def build_prompt(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._delegate, "build_prompt"):
            result = self._delegate.build_prompt(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        if hasattr(self._delegate, "build_gated_prompt"):
            result = self._delegate.build_gated_prompt(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        raise AttributeError("Underlying prompt builder does not support build_prompt")

    def __getattr__(self, item: str) -> Any:
        return getattr(self._delegate, item)


__all__ = ["UnifiedPromptBuilder", "UnifiedHierarchicalPromptBuilder", "PromptBuilder"]
