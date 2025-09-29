"""
# core/response_generator.py

Module Contract
- Purpose: Stream model responses with timing/telemetry, ensuring the system prompt is always included for chat models.
- Inputs:
  - generate_streaming_response(prompt: str, model_name: str, system_prompt: Optional[str])
- Outputs:
  - Async generator yielding content chunks (words or deltas), suitable for live UI updates.
- Behavior:
  - Switches to the requested model via model_manager; wraps generate_async; yields words; logs first‑token latency + total duration.
  - Ensures a non‑empty system prompt is sent (falls back to config SYSTEM_PROMPT if blank/None).
- Dependencies:
  - models.model_manager.ModelManager (generate_async/generate_once)
- Side effects:
  - Logging only; no persistence.
"""
from utils.logging_utils import get_logger
import time
from typing import AsyncGenerator, List, Tuple
from datetime import datetime
from utils.time_manager import TimeManager
from utils.query_checker import keyword_tokens
from config.app_config import (
    DEFAULT_TEMPERATURE,
)
logger = get_logger("response_generator")


class ResponseGenerator:
    """Handles response generation and streaming"""

    def __init__(self, model_manager, time_manager: TimeManager = None):
        self.model_manager = model_manager
        self.time_manager = time_manager or TimeManager()
        self.logger = logger

    async def generate_streaming_response(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate response with streaming support
        """
        self.logger.debug(f"[GENERATE] Starting async generation with model: {model_name}")
        start_time = time.time()
        self.time_manager.mark_query_time()
        self.logger.debug(f"[TIME] Since last query: {self.time_manager.elapsed_since_last()}")
        self.logger.debug(f"[TIME] Previous response time: {self.time_manager.last_response()}")

        first_token_time = None

        try:
            if model_name:
                self.model_manager.switch_model(model_name)
                self.logger.info(
                    "[ModelManager] Active model set to: %s",
                    self.model_manager.get_active_model_name()
                )

            # Ensure a non-empty system prompt is always sent.
            # Falls back to config default if None/blank is provided.
            try:
                from config.app_config import SYSTEM_PROMPT as DEFAULT_SP  # local import to avoid hard dep at import time
            except Exception:
                DEFAULT_SP = "You are Daemon, a helpful assistant with memory and RAG. Be direct, truthful, concise."

            effective_sp = (system_prompt or "").strip() or DEFAULT_SP

            # Always include the system message; "raw" only controls upstream prompt building
            response_generator = await self.model_manager.generate_async(
                prompt,
                system_prompt=effective_sp,
                raw=False,
            )

            # Streaming path
            if hasattr(response_generator, "__aiter__"):
                buffer = ""
                async for chunk in response_generator:
                    try:
                        # Extract content from different possible streaming chunk shapes
                        delta_content = ""
                        # OpenAI-style ChatCompletionChunk
                        if hasattr(chunk, "choices") and len(getattr(chunk, "choices", [])) > 0:
                            delta = chunk.choices[0].delta
                            delta_content = getattr(delta, "content", "") or ""
                        # Plain string chunk (e.g., stub/local streams)
                        elif isinstance(chunk, str):
                            delta_content = chunk
                        # Dict-like chunk with direct content
                        elif isinstance(chunk, dict):
                            delta_content = (chunk.get("content") or chunk.get("text") or "")

                        if delta_content:
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now
                                self.logger.debug(
                                    "[STREAMING] First token arrived after %.2f seconds",
                                    now - start_time
                                )

                            buffer += delta_content

                            # Yield word-by-word, keep the last partial word in buffer
                            if " " in buffer:
                                words = buffer.split(" ")
                                for word in words[:-1]:
                                    if word:
                                        yield word
                                buffer = words[-1] if words[-1] else ""

                    except Exception as e:
                        self.logger.error(f"[STREAMING] Error processing chunk: {e}")
                        continue

                # Yield any remaining buffer
                if buffer.strip():
                    yield buffer.strip()

                end_time = time.time()
                duration = self.time_manager.measure_response(
                    datetime.fromtimestamp(start_time),
                    datetime.fromtimestamp(end_time)
                )
                self.logger.info(f"[TIMING] Full response duration: {duration}")

            else:
                # Non-streaming fallback: simulate streaming by words
                self.logger.debug("[GENERATE] Non-streaming response, handling in fallback mode.")

                if hasattr(response_generator, "choices") and len(response_generator.choices) > 0:
                    if hasattr(response_generator.choices[0], "message"):
                        content = response_generator.choices[0].message.content
                    else:
                        content = str(response_generator)
                else:
                    content = str(response_generator)

                for word in content.split():
                    yield word

        except Exception as e:
            self.logger.error(f"[GENERATE] Error: {type(e).__name__}: {str(e)}")
            yield f"[Streaming Error] {e}"

    async def generate_full(self, prompt: str, model_name: str, system_prompt: str = None, temperature: float = None) -> str:
        """Generate a full, non-streamed response using generate_once."""
        try:
            if model_name:
                self.model_manager.switch_model(model_name)
            # Fallback system prompt if missing
            try:
                from config.app_config import SYSTEM_PROMPT as DEFAULT_SP
            except Exception:
                DEFAULT_SP = "You are Daemon, a helpful assistant with memory and RAG. Be direct, truthful, concise."
            effective_sp = (system_prompt or "").strip() or DEFAULT_SP
            text = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=self.model_manager.get_active_model_name(),
                system_prompt=effective_sp,
                temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            )
            return (text or "").strip()
        except Exception as e:
            self.logger.error(f"[GENERATE_FULL] Error: {e}")
            return f"[Generation error] {e}"

    # ---------------- Best-of-N (non-streaming) ----------------
    @staticmethod
    def _coverage_score(answer: str, question: str, context_hint: str = "") -> float:
        ans = (answer or "").lower()
        q_tokens = set(keyword_tokens(question, 3))
        if not q_tokens:
            return 0.0
        hit = sum(1 for t in q_tokens if t in ans)
        base = hit / max(1, len(q_tokens))
        # light context boost
        if context_hint:
            ctx_tokens = set(keyword_tokens(context_hint, 5))
            if ctx_tokens:
                ctx_hit = sum(1 for t in ctx_tokens if t in ans)
                base = 0.7 * base + 0.3 * (ctx_hit / max(1, len(ctx_tokens)))
        return max(0.0, min(1.0, base))

    @staticmethod
    def _length_score(answer: str, min_w: int = 40, max_w: int = 220) -> float:
        n = len((answer or "").split())
        if n <= 0:
            return 0.0
        if n < min_w:
            return max(0.0, n / float(min_w))
        if n > max_w:
            # linearly decay beyond max
            return max(0.0, 1.0 - (n - max_w) / float(max_w))
        return 1.0

    @staticmethod
    def _repetition_penalty(answer: str) -> float:
        words = [w for w in (answer or "").lower().split() if w]
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / float(len(words))
        # higher penalty when many repeats (unique_ratio small)
        return max(0.0, 1.0 - unique_ratio)

    @staticmethod
    def _hallucination_penalty(answer: str, context_hint: str = "") -> float:
        if not context_hint:
            return 0.0
        a = (answer or "").lower()
        c = (context_hint or "").lower()
        # penalize numbers not in context
        import re
        nums = re.findall(r"\b\d+[\w\-]*\b", a)
        if not nums:
            return 0.0
        missing = sum(1 for n in nums if n not in c)
        return min(1.0, missing / float(len(nums)))

    def _score_answer(self, answer: str, question: str, context_hint: str = "") -> float:
        cov = self._coverage_score(answer, question, context_hint)
        le = self._length_score(answer)
        rep = self._repetition_penalty(answer)
        hal = self._hallucination_penalty(answer, context_hint)
        score = 0.45 * cov + 0.25 * le - 0.10 * rep - 0.20 * hal
        self.logger.debug(
            f"[BESTOF] scores cov={cov:.2f} len={le:.2f} rep_pen={rep:.2f} hall_pen={hal:.2f} -> {score:.3f}"
        )
        return score

    async def generate_best_of(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str,
        question_text: str,
        context_hint: str = "",
        n: int = 2,
        temps: Tuple[float, ...] = (0.2, 0.7),
        max_tokens: int = 512,
    ) -> str:
        """Generate N candidates concurrently, score, and return the best."""
        # temperatures list length dictates N if provided
        if temps and len(temps) >= 1:
            n = min(n, len(temps))
        else:
            temps = (DEFAULT_TEMPERATURE,) * n

        import asyncio
        tasks = []
        for i in range(n):
            t = temps[i if i < len(temps) else -1]
            tasks.append(
                self.model_manager.generate_once(
                    prompt=prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=t,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: List[str] = []
        for idx, res in enumerate(results, start=1):
            if isinstance(res, Exception):
                self.logger.error(f"[BESTOF] candidate {idx}/{n} error: {res}")
                candidates.append("")
            else:
                text = (res or "").strip()
                candidates.append(text)
                self.logger.debug(f"[BESTOF] candidate {idx}/{n} len={len(text)}")

        # Score candidates
        scored = [(self._score_answer(ans, question_text, context_hint), ans) for ans in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_answer = scored[0]
        self.logger.info(f"[BESTOF] selected best candidate score={best_score:.3f}")
        return best_answer
