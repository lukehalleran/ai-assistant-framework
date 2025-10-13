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
from typing import AsyncGenerator, List, Tuple, Optional, Sequence, Dict, Any
from datetime import datetime
from utils.time_manager import TimeManager
from utils.query_checker import keyword_tokens
# No config constants imported here; defaults are managed by ModelManager
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
                # Let ModelManager apply its current default temperature when None
                temperature=temperature if temperature is not None else None,
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
            temps = (getattr(self.model_manager, 'default_temperature', 0.7),) * n

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

        if self.logger:
            try:
                self.logger.info(f"[BESTOF] single-model path model={model_name} n={n} temps={tuple(temps)} max_tokens={max_tokens}")
            except Exception:
                pass
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: List[str] = []
        raw_candidates: List[str] = []
        for idx, res in enumerate(results, start=1):
            if isinstance(res, Exception):
                self.logger.error(f"[BESTOF] candidate {idx}/{n} error: {res}")
                candidates.append("")
                raw_candidates.append("")
            else:
                text_raw = (res or "").strip()
                raw_candidates.append(text_raw)

                # Parse thinking block from each candidate
                from core.orchestrator import DaemonOrchestrator
                thinking, text_final = DaemonOrchestrator._parse_thinking_block(text_raw)

                # Log thinking block for this candidate
                if thinking and self.logger:
                    try:
                        self.logger.debug(f"[BESTOF] candidate {idx}/{n} [THINKING BLOCK]\n{thinking}")
                    except Exception:
                        pass

                candidates.append(text_final)
                self.logger.debug(f"[BESTOF] candidate {idx}/{n} raw_len={len(text_raw)} final_len={len(text_final)}")

        # Score candidates based on final answers (without thinking blocks)
        scored = [(self._score_answer(ans, question_text, context_hint), ans) for ans in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_answer = scored[0]
        self.logger.info(f"[BESTOF] selected best candidate score={best_score:.3f}")
        return best_answer

    # ---------------- Ensemble Best-of (multi-model + multi-selector) ----------------
    @staticmethod
    def _minmax_normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin <= 1e-9:
            return [5.0 for _ in values]
        return [10.0 * (v - vmin) / (vmax - vmin) for v in values]

    async def _llm_judge_score(
        self,
        judge_model: str,
        question_text: str,
        answer_text: str,
        context_hint: str = "",
        max_tokens: int = 64,
    ) -> float:
        """Ask a lightweight model to score answer 0-10. Robustly parse the number."""
        rubric = (
            "You are a strict evaluator. Score the candidate answer 0-10 for how well it answers the user question,"
            " avoiding hallucinations and staying consistent with the provided context."
            " Return ONLY a JSON object like {\"score\": <0-10 number>, \"justification\": \"...\"}."
        )
        user_block = (
            f"Question:\n{question_text}\n\n"
            f"Context (may be partial):\n{(context_hint or '')[:2000]}\n\n"
            f"Candidate Answer:\n{answer_text[:4000]}\n"
            "Respond with a single JSON object only."
        )
        try:
            text = await self.model_manager.generate_once(
                prompt=user_block,
                model_name=judge_model,
                system_prompt=rubric,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = (text or "").strip()
            # Try parse JSON first
            import json, re
            score: Optional[float] = None
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "score" in obj:
                    score = float(obj["score"])  # may raise ValueError
            except Exception:
                pass
            if score is None:
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
                if m:
                    score = float(m.group(1))
            if score is None:
                self.logger.debug(f"[JUDGE:{judge_model}] parse failed; raw=\n{raw}")
                return 0.0
            return float(max(0.0, min(10.0, score)))
        except Exception as e:
            self.logger.error(f"[JUDGE:{judge_model}] error: {e}")
            return 0.0

    async def _llm_judge_compare(
        self,
        judge_model: str,
        question_text: str,
        answer_a: str,
        answer_b: str,
        context_hint: str = "",
        max_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Ask a judge to choose between A and B. Returns dict with winner, scores and reason.
        Fallback: if parsing fails, score A and B independently and pick higher.
        """
        rubric = (
            "You are a strict evaluator. Compare two candidate answers (A and B) to the same question."
            " Decide the better answer for correctness, relevance, and consistency with the provided context."
            " Respond ONLY a JSON object like {\"winner\": \"A|B\", \"score_A\": <0-10>, \"score_B\": <0-10>, \"reason\": \"...\"}."
        )
        user_block = (
            f"Question:\n{question_text}\n\n"
            f"Context (may be partial):\n{(context_hint or '')[:2000]}\n\n"
            f"Answer A:\n{(answer_a or '')[:4000]}\n\n"
            f"Answer B:\n{(answer_b or '')[:4000]}\n"
            "Respond with a single JSON object only."
        )
        try:
            text = await self.model_manager.generate_once(
                prompt=user_block,
                model_name=judge_model,
                system_prompt=rubric,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = (text or "").strip()
            import json, re
            result: Dict[str, Any] = {}
            try:
                result = json.loads(raw)
            except Exception:
                # Fuzzy parse winner token
                m = re.search(r"\bwinner\b.*?([AB])", raw, re.IGNORECASE | re.DOTALL)
                if m:
                    result["winner"] = m.group(1).upper()
            if not isinstance(result, dict) or result.get("winner") not in ("A", "B"):
                # Fallback: independent scoring
                sc_a = await self._llm_judge_score(judge_model, question_text, answer_a, context_hint, max_tokens)
                sc_b = await self._llm_judge_score(judge_model, question_text, answer_b, context_hint, max_tokens)
                result = {
                    "winner": "A" if sc_a >= sc_b else "B",
                    "score_A": sc_a,
                    "score_B": sc_b,
                    "reason": "fallback: independent scoring",
                }
            return result
        except Exception as e:
            self.logger.error(f"[JUDGE/{judge_model}] compare error: {e}")
            # Hard fallback: heuristic coverage
            sa = self._score_answer(answer_a, question_text, context_hint)
            sb = self._score_answer(answer_b, question_text, context_hint)
            return {"winner": "A" if sa >= sb else "B", "score_A": sa, "score_B": sb, "reason": "fallback: heuristic"}

    async def generate_duel_and_judge(
        self,
        prompt: str,
        model_a: str,
        model_b: str,
        judge_model: str,
        system_prompt: str,
        question_text: str,
        context_hint: str = "",
        max_tokens: int = 512,
        temperature_a: Optional[float] = None,
        temperature_b: Optional[float] = None,
        judge_max_tokens: int = 64,
    ) -> str:
        """Generate once with two models, judge both, and return the winner.

        Thinking blocks: Both models may include <thinking>...</thinking> blocks.
        These are logged but stripped before sending to the judge, and only the
        final answer from the winning model is returned.
        """
        import asyncio
        if self.logger:
            try:
                self.logger.info(f"[DUEL] model_1={model_a} model_2={model_b} judge={judge_model}")
            except Exception:
                pass
        t1 = asyncio.create_task(
            self.model_manager.generate_once(
                prompt=prompt,
                model_name=model_a,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature_a,
            )
        )
        t2 = asyncio.create_task(
            self.model_manager.generate_once(
                prompt=prompt,
                model_name=model_b,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature_b,
            )
        )
        res1, res2 = await asyncio.gather(t1, t2, return_exceptions=True)
        a_text_raw = ("" if isinstance(res1, Exception) else (res1 or "")).strip()
        b_text_raw = ("" if isinstance(res2, Exception) else (res2 or "")).strip()

        # Parse thinking blocks from both responses
        from core.orchestrator import DaemonOrchestrator
        thinking_a, a_text = DaemonOrchestrator._parse_thinking_block(a_text_raw)
        thinking_b, b_text = DaemonOrchestrator._parse_thinking_block(b_text_raw)

        # Log thinking blocks for debugging
        if thinking_a and self.logger:
            try:
                self.logger.debug(f"[DUEL][{model_a}][THINKING BLOCK]\n{thinking_a}")
            except Exception:
                pass
        if thinking_b and self.logger:
            try:
                self.logger.debug(f"[DUEL][{model_b}][THINKING BLOCK]\n{thinking_b}")
            except Exception:
                pass

        if self.logger:
            try:
                self.logger.info(f"[DUEL] raw lengths a={len(a_text_raw)} b={len(b_text_raw)}, final lengths a={len(a_text)} b={len(b_text)}")
            except Exception:
                pass

        # Judge sees only final answers (without thinking blocks)
        result = await self._llm_judge_compare(
            judge_model=judge_model,
            question_text=question_text,
            answer_a=a_text,
            answer_b=b_text,
            context_hint=context_hint,
            max_tokens=judge_max_tokens,
        )
        winner = result.get("winner", "A")
        if self.logger:
            try:
                self.logger.info(f"[DUEL] winner={winner} scores A={result.get('score_A')} B={result.get('score_B')}")
            except Exception:
                pass

        # Return the final answer from the winner plus metadata about both thinking processes
        final_answer = a_text if winner == "A" else b_text

        # Return a dict with answer and thinking metadata for UI display
        return {
            'answer': final_answer,
            'thinking_a': thinking_a,
            'thinking_b': thinking_b,
            'model_a': model_a,
            'model_b': model_b,
            'winner': winner,
            'scores': {'A': result.get('score_A'), 'B': result.get('score_B')}
        }

    async def generate_best_of_ensemble(
        self,
        prompt: str,
        generator_models: Sequence[str],
        system_prompt: str,
        question_text: str,
        context_hint: str = "",
        n_total: int = 2,
        temps: Tuple[float, ...] = (0.2, 0.7),
        max_tokens: int = 512,
        selector_models: Optional[Sequence[str]] = None,
        selector_max_tokens: int = 64,
        weight_heuristic: float = 0.5,
        weight_llm: float = 0.5,
        judge_top_k: int = 0,
    ) -> str:
        """Multi-model candidate generation + optional multi-judge selection.

        - Distributes n_total roughly evenly across generator_models.
        - Uses internal heuristic scoring and optionally LLM judges; combines by weights.
        - judge_top_k>0: only the top-K (by heuristic) are sent to judges.
        """
        import asyncio, math

        if self.logger:
            try:
                self.logger.info(
                    f"[BESTOF/MULTI] starting gens={list(generator_models)} n_total={n_total} temps={tuple(temps)} "
                    f"selectors={list(selector_models) if selector_models else []} w_heur={weight_heuristic} w_llm={weight_llm} top_k={judge_top_k}"
                )
            except Exception:
                pass

        if not generator_models:
            # Fallback to single-model path (preserve behavior)
            return await self.generate_best_of(
                prompt=prompt,
                model_name=self.model_manager.get_active_model_name() or "gpt-4-turbo",
                system_prompt=system_prompt,
                question_text=question_text,
                context_hint=context_hint,
                n=n_total,
                temps=temps,
                max_tokens=max_tokens,
            )

        # Determine per-model counts
        n_per = max(1, math.ceil(max(1, n_total) / float(len(generator_models))))
        temps_seq = list(temps) if temps else [getattr(self.model_manager, 'default_temperature', 0.7)]
        if not temps_seq:
            temps_seq = [0.7]

        # Launch candidate generations
        gen_tasks: List[asyncio.Task] = []
        meta: List[Dict[str, Any]] = []
        for gm in generator_models:
            for i in range(n_per):
                t = temps_seq[i % len(temps_seq)]
                gen_tasks.append(
                    asyncio.create_task(
                        self.model_manager.generate_once(
                            prompt=prompt,
                            model_name=gm,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=t,
                        )
                    )
                )
                meta.append({"model": gm, "temp": t})

        results = await asyncio.gather(*gen_tasks, return_exceptions=True)
        candidates: List[Dict[str, Any]] = []
        for i, res in enumerate(results):
            src = meta[i]
            if isinstance(res, Exception):
                self.logger.error(f"[BESTOF/MULTI] {src['model']}@{src['temp']} error: {res}")
                text = ""
            else:
                text = (res or "").strip()
            candidates.append({"text": text, **src})
        if self.logger:
            try:
                self.logger.info(f"[BESTOF/MULTI] gathered {len(candidates)} candidates from {len(generator_models)} models")
            except Exception:
                pass

        # Heuristic scores (normalize to 0..10)
        heur_raw: List[float] = [
            self._score_answer(c["text"], question_text, context_hint) for c in candidates
        ]
        heur_norm: List[float] = self._minmax_normalize(heur_raw)
        for c, sc, raw in zip(candidates, heur_norm, heur_raw):
            c["score_heur"] = sc
            c["score_heur_raw"] = raw

        # If no selector models, pick best by heuristic
        selector_models = list(selector_models or [])
        if not selector_models or weight_llm <= 0.0:
            best = max(candidates, key=lambda x: x.get("score_heur", 0.0)) if candidates else {"text": ""}
            self.logger.info(
                f"[BESTOF/MULTI] selected by heuristic from {len(candidates)} candidates"
            )
            return best.get("text", "")

        # Optionally limit judged set
        judge_pool = candidates
        if judge_top_k and judge_top_k > 0 and len(candidates) > judge_top_k:
            judge_pool = sorted(candidates, key=lambda x: x.get("score_heur_raw", 0.0), reverse=True)[:judge_top_k]

        # Collect LLM judge scores
        judge_tasks: List[asyncio.Task] = []
        judge_meta: List[Dict[str, Any]] = []
        for c in judge_pool:
            for jm in selector_models:
                judge_tasks.append(
                    asyncio.create_task(
                        self._llm_judge_score(
                            judge_model=jm,
                            question_text=question_text,
                            answer_text=c["text"],
                            context_hint=context_hint,
                            max_tokens=selector_max_tokens,
                        )
                    )
                )
                judge_meta.append({"cid": id(c), "judge": jm})

        judge_scores: Dict[int, List[float]] = {id(c): [] for c in candidates}
        if judge_tasks:
            judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
            for m, res in zip(judge_meta, judge_results):
                sc = 0.0
                if isinstance(res, Exception):
                    self.logger.error(f"[JUDGE] error: {res}")
                else:
                    sc = float(res)
                judge_scores[m["cid"]].append(sc)

        for c in candidates:
            js = judge_scores.get(id(c), [])
            c["score_llm"] = float(sum(js) / len(js)) if js else 0.0

        # Final combined score
        w_h = max(0.0, min(1.0, weight_heuristic))
        w_l = max(0.0, min(1.0, weight_llm))
        if w_h + w_l <= 0:
            w_h = 1.0
            w_l = 0.0
        for c in candidates:
            c["score_total"] = w_h * c.get("score_heur", 0.0) + w_l * c.get("score_llm", 0.0)

        best = max(candidates, key=lambda x: x.get("score_total", 0.0)) if candidates else {"text": ""}
        self.logger.info(
            f"[BESTOF/MULTI] selected by ensemble from {len(candidates)} candidates"
        )
        return best.get("text", "")
