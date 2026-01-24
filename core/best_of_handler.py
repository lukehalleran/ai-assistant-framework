"""
BestOfHandler - Orchestrates best-of-N response generation.

Extracted from DaemonOrchestrator to reduce god object complexity.
Handles decision logic, mode selection (duel/ensemble/single), and fallback.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


@dataclass
class BestOfResult:
    """Result from best-of generation."""
    response: str
    used_best_of: bool
    mode: str  # "duel", "ensemble", "single", "fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BestOfHandler:
    """
    Orchestrates best-of-N response generation with multi-model and duel support.

    Delegates actual generation to ResponseGenerator methods:
    - generate_best_of() for single-model N candidates
    - generate_best_of_ensemble() for multi-model with judges
    - generate_duel_and_judge() for two models + judge
    """

    def __init__(
        self,
        response_generator: "ResponseGenerator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BestOfHandler.

        Args:
            response_generator: ResponseGenerator instance for actual generation
            config: Runtime config dict (typically orchestrator.config)
        """
        self.response_generator = response_generator
        self.config = config or {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default config from app_config with fallbacks."""
        try:
            from config.app_config import (
                ENABLE_BEST_OF,
                BEST_OF_N,
                BEST_OF_TEMPS,
                BEST_OF_MAX_TOKENS,
                BEST_OF_MIN_QUESTION,
                BEST_OF_MIN_TOKENS,
                BEST_OF_MODEL,
                BEST_OF_LATENCY_BUDGET_S,
                BEST_OF_GENERATOR_MODELS,
                BEST_OF_SELECTOR_MODELS,
                BEST_OF_SELECTOR_MAX_TOKENS,
                BEST_OF_SELECTOR_WEIGHTS,
                BEST_OF_SELECTOR_TOP_K,
                BEST_OF_DUEL_MODE,
            )
            self._defaults = {
                "enable": ENABLE_BEST_OF,
                "n": BEST_OF_N,
                "temps": BEST_OF_TEMPS,
                "max_tokens": BEST_OF_MAX_TOKENS,
                "min_question": BEST_OF_MIN_QUESTION,
                "min_tokens": BEST_OF_MIN_TOKENS,
                "model": BEST_OF_MODEL,
                "latency_budget_s": BEST_OF_LATENCY_BUDGET_S,
                "generator_models": BEST_OF_GENERATOR_MODELS,
                "selector_models": BEST_OF_SELECTOR_MODELS,
                "selector_max_tokens": BEST_OF_SELECTOR_MAX_TOKENS,
                "selector_weights": BEST_OF_SELECTOR_WEIGHTS,
                "selector_top_k": BEST_OF_SELECTOR_TOP_K,
                "duel_mode": BEST_OF_DUEL_MODE,
            }
        except ImportError:
            self._defaults = {
                "enable": True,
                "n": 2,
                "temps": (0.2, 0.7),
                "max_tokens": 512,
                "min_question": True,
                "min_tokens": 8,
                "model": None,
                "latency_budget_s": 2.0,
                "generator_models": [],
                "selector_models": [],
                "selector_max_tokens": 64,
                "selector_weights": {"heuristic": 1.0, "llm": 0.0},
                "selector_top_k": 0,
                "duel_mode": False,
            }

    def _get_runtime_config(self) -> Dict[str, Any]:
        """Get merged config (defaults + runtime overrides)."""
        features = {}
        if isinstance(self.config, dict):
            features = self.config.get("features", {}) or {}

        return {
            "enable": bool(features.get("enable_best_of", self._defaults["enable"])),
            "n": self._defaults["n"],
            "temps": tuple(self._defaults["temps"]) if isinstance(self._defaults["temps"], (list, tuple)) else (0.2, 0.7),
            "max_tokens": int(features.get("best_of_max_tokens", self._defaults["max_tokens"])),
            "min_question": self._defaults["min_question"],
            "min_tokens": self._defaults["min_tokens"],
            "model": self._defaults["model"],
            "latency_budget_s": float(features.get("best_of_latency_budget_s", self._defaults["latency_budget_s"])),
            "generator_models": list(features.get("best_of_generator_models", self._defaults["generator_models"])),
            "selector_models": list(features.get("best_of_selector_models", self._defaults["selector_models"])),
            "selector_max_tokens": int(features.get("best_of_selector_max_tokens", self._defaults["selector_max_tokens"])),
            "selector_weights": dict(features.get("best_of_selector_weights", self._defaults["selector_weights"])) if isinstance(features.get("best_of_selector_weights"), dict) else self._defaults["selector_weights"],
            "selector_top_k": int(features.get("best_of_selector_top_k", self._defaults["selector_top_k"])),
            "duel_mode": bool(features.get("best_of_duel_mode", self._defaults["duel_mode"])),
        }

    def should_use_best_of(self, user_input: str, use_raw_mode: bool = False) -> bool:
        """
        Determine if query qualifies for best-of generation.

        Args:
            user_input: The user's query text
            use_raw_mode: If True, skip best-of (raw streaming mode)

        Returns:
            True if best-of should be used, False otherwise
        """
        if use_raw_mode:
            return False

        cfg = self._get_runtime_config()
        if not cfg["enable"]:
            return False

        try:
            from utils.query_checker import analyze_query
            qinfo = analyze_query(user_input)
            return bool(
                (qinfo.is_question and qinfo.token_count >= cfg["min_tokens"])
                or (not cfg["min_question"])
            )
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[BestOfHandler] Query analysis failed: {e}, using enable flag")
            return cfg["enable"]

    async def generate(
        self,
        prompt: str,
        user_input: str,
        system_prompt: str,
        model_name: str,
        response_max_tokens: Optional[int] = None
    ) -> BestOfResult:
        """
        Execute best-of generation with automatic mode selection.

        Selects mode based on config:
        - Duel: 2 generators + 1 judge (when duel_mode=True and 2 generators configured)
        - Ensemble: Multiple generators + multiple judges
        - Single: One model, N candidates with temperature variation

        Falls back to streaming on timeout or error.

        Args:
            prompt: The full prompt to send to the model
            user_input: Original user query (for context/judging)
            system_prompt: System prompt for generation
            model_name: Default model name (used for single mode)
            response_max_tokens: Max tokens for fallback streaming

        Returns:
            BestOfResult with response, mode used, and metadata
        """
        cfg = self._get_runtime_config()

        # Determine mode
        gen_models = cfg["generator_models"]
        sel_models = cfg["selector_models"]
        use_duel = bool(cfg["duel_mode"] and len(gen_models) == 2 and len(sel_models) >= 1)
        use_ensemble = bool(gen_models) and not use_duel

        mode = "duel" if use_duel else ("ensemble" if use_ensemble else "single")
        temps = cfg["temps"]
        max_tokens = cfg["max_tokens"]

        logger.info(
            f"[BestOfHandler] mode={mode} gens={gen_models} selectors={sel_models} "
            f"budget={cfg['latency_budget_s']}s max_tokens={max_tokens}"
        )

        try:
            budget = cfg["latency_budget_s"]

            if use_duel:
                result = await self._run_with_budget(
                    self._generate_duel(prompt, user_input, system_prompt, gen_models, sel_models, temps, cfg),
                    budget
                )
                # Duel returns dict with metadata
                if isinstance(result, dict):
                    return BestOfResult(
                        response=result.get("answer", ""),
                        used_best_of=True,
                        mode="duel",
                        metadata={"raw": result}
                    )
                return BestOfResult(response=str(result), used_best_of=True, mode="duel")

            elif use_ensemble:
                result = await self._run_with_budget(
                    self._generate_ensemble(prompt, user_input, system_prompt, gen_models, sel_models, temps, cfg),
                    budget
                )
                return BestOfResult(response=str(result), used_best_of=True, mode="ensemble")

            else:
                result = await self._run_with_budget(
                    self._generate_single(prompt, user_input, system_prompt, model_name, temps, cfg),
                    budget
                )
                return BestOfResult(response=str(result), used_best_of=True, mode="single")

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.debug(f"[BestOfHandler] Timeout/cancelled, falling back to streaming: {type(e).__name__}")
            return await self._fallback_streaming(prompt, model_name, system_prompt, response_max_tokens)
        except Exception as e:
            logger.warning(f"[BestOfHandler] Error in best-of, falling back to streaming: {e}")
            return await self._fallback_streaming(prompt, model_name, system_prompt, response_max_tokens)

    async def _run_with_budget(self, coro, budget: float):
        """Run coroutine with optional timeout budget."""
        if budget > 0:
            return await asyncio.wait_for(coro, timeout=budget)
        return await coro

    async def _generate_duel(
        self,
        prompt: str,
        user_input: str,
        system_prompt: str,
        gen_models: List[str],
        sel_models: List[str],
        temps: Tuple[float, ...],
        cfg: Dict[str, Any]
    ):
        """Execute duel mode: two models + judge."""
        m1, m2 = gen_models[:2]
        judge = sel_models[0]
        return await self.response_generator.generate_duel_and_judge(
            prompt=prompt,
            model_a=m1,
            model_b=m2,
            judge_model=judge,
            system_prompt=system_prompt,
            question_text=user_input,
            context_hint=prompt,
            max_tokens=cfg["max_tokens"],
            temperature_a=temps[0] if len(temps) > 0 else None,
            temperature_b=temps[1] if len(temps) > 1 else None,
            judge_max_tokens=cfg["selector_max_tokens"],
        )

    async def _generate_ensemble(
        self,
        prompt: str,
        user_input: str,
        system_prompt: str,
        gen_models: List[str],
        sel_models: List[str],
        temps: Tuple[float, ...],
        cfg: Dict[str, Any]
    ):
        """Execute ensemble mode: multiple generators + judges."""
        return await self.response_generator.generate_best_of_ensemble(
            prompt=prompt,
            generator_models=gen_models,
            system_prompt=system_prompt,
            question_text=user_input,
            context_hint=prompt,
            n_total=cfg["n"],
            temps=temps,
            max_tokens=cfg["max_tokens"],
            selector_models=sel_models,
            selector_max_tokens=cfg["selector_max_tokens"],
            weight_heuristic=float(cfg["selector_weights"].get("heuristic", 0.5)),
            weight_llm=float(cfg["selector_weights"].get("llm", 0.5)),
            judge_top_k=cfg["selector_top_k"],
        )

    async def _generate_single(
        self,
        prompt: str,
        user_input: str,
        system_prompt: str,
        model_name: str,
        temps: Tuple[float, ...],
        cfg: Dict[str, Any]
    ):
        """Execute single-model best-of: N candidates with temperature variation."""
        return await self.response_generator.generate_best_of(
            prompt=prompt,
            model_name=cfg["model"] or model_name,
            system_prompt=system_prompt,
            question_text=user_input,
            context_hint=prompt,
            n=cfg["n"],
            temps=temps,
            max_tokens=cfg["max_tokens"],
        )

    async def _fallback_streaming(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str,
        max_tokens: Optional[int]
    ) -> BestOfResult:
        """Fall back to streaming on error/timeout."""
        full_response = ""
        async for chunk in self.response_generator.generate_streaming_response(
            prompt, model_name, system_prompt=system_prompt, max_tokens=max_tokens
        ):
            full_response += (chunk + " ")
        return BestOfResult(
            response=full_response.strip(),
            used_best_of=False,
            mode="fallback"
        )
