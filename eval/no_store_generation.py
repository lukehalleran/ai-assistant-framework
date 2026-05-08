"""
Side-effect-free LLM generation for eval.

EvalGenerator calls ModelManager.generate_once() directly — never the orchestrator.
It does not store interactions, extract facts, update the graph, or trigger any
post-generation processing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from eval.schema import EvalGenerationResult, compute_prompt_hash


@dataclass
class EvalGenerationConfig:
    """Safety config that rejects any persistence flag set to True.

    Every flag defaults to False. __post_init__ raises ValueError if any
    field is True, making it structurally impossible to accidentally enable
    persistence during eval generation.
    """
    store_interaction: bool = False
    update_graph: bool = False
    extract_facts: bool = False
    update_threads: bool = False
    run_shutdown_processing: bool = False
    allow_web_search: bool = False
    update_summaries: bool = False
    update_reflections: bool = False
    update_stm: bool = False
    trigger_skills: bool = False

    def __post_init__(self):
        for field_name, value in vars(self).items():
            if value is True:
                raise ValueError(
                    f"EvalGenerationConfig.{field_name} must be False for eval. "
                    f"Eval generation must never trigger persistence side effects."
                )


class EvalGenerator:
    """Generates LLM responses without any persistence side effects.

    Uses ModelManager.generate_once() — a pure LLM call that does not
    touch memory, graph, threads, summaries, reflections, or storage.
    """

    def __init__(
        self,
        config: Optional[EvalGenerationConfig] = None,
        model_manager: Any = None,
    ):
        self._config = config or EvalGenerationConfig()
        self._model_manager = model_manager

    async def generate(
        self,
        assembled_prompt: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_message: Optional[str] = None,
    ) -> EvalGenerationResult:
        """Generate a response from a frozen prompt. No side effects.

        Args:
            assembled_prompt: The full context prompt text (from snapshot replay).
            model: Model name to use for generation.
            temperature: Sampling temperature (low for reproducibility).
            max_tokens: Max response tokens.
            system_message: Optional system prompt (passed separately to LLM).

        Returns:
            EvalGenerationResult with response text and metadata.

        Raises:
            RuntimeError: If no model_manager is configured.
        """
        if self._model_manager is None:
            raise RuntimeError(
                "EvalGenerator requires a model_manager. "
                "Pass one to __init__ or use a test stub."
            )

        prompt_hash = compute_prompt_hash(assembled_prompt)
        prompt_token_count = max(1, len(assembled_prompt) // 4)  # rough estimate

        t_start = time.perf_counter()

        response_text = await self._model_manager.generate_once(
            prompt=assembled_prompt,
            model_name=model,
            system_prompt=system_message or "You are a helpful assistant.",
            max_tokens=max_tokens,
            temperature=temperature,
        )

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        response_token_count = max(1, len(response_text) // 4)

        return EvalGenerationResult(
            response_text=response_text,
            model=model,
            prompt_hash=prompt_hash,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            generation_time_ms=elapsed_ms,
            temperature=temperature,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
