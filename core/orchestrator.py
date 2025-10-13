"""
# core/orchestrator.py

Module Contract
- Purpose: High‑level request orchestrator. Prepares prompts (topic detection, optional file processing, query rewrite), invokes the model, and persists the interaction to memory.
- Inputs:
  - user_input: str; optional files; mode flags (raw/enhanced)
  - Wired collaborators: model_manager, response_generator, file_processor, prompt_builder, memory_system, personality/topic/wiki/tokenizer managers
- Outputs:
  - process_user_query() → (assistant_text: str, debug_info: dict)
  - prepare_prompt() → (prompt_text: str, system_prompt: Optional[str])
- Key methods:
  - handle_commands(): simple topic switching commands
  - prepare_prompt(): topic update, file processing, optional query rewrite, resolve system prompt, build prompt via prompt_builder
  - process_user_query(): personality hook, deictic check, generate streamed response, store memory, schedule consolidation (now at shutdown)
- System prompt flow:
  - Resolved via config/app_config.load_system_prompt and/or path override; forwarded to response generation as system role message.
- Side effects:
  - Writes conversation+metadata to corpus DB/Chroma via memory_system; logs events.
- Async behavior:
  - Uses async streaming from response_generator; overall methods are async.
"""
import os
import processing.gate_system as gate_system
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from utils.logging_utils import get_logger
from integrations.wikipedia_api import WikipediaAPI

SYSTEM_PROMPT = "..."  # safe fallback (replace with your real default)
wiki_api = WikipediaAPI()
gate_system.wikipedia_api = wiki_api  # This sets it globally
# If you have a real helper, import that instead:
from utils.query_checker import is_deictic, analyze_query


class _SimplePromptBuilder:
    """Fallback prompt builder used when the unified builder is unavailable."""

    async def build_prompt(self, user_input: str, **_: Any) -> str:
        return user_input or ""


class _InMemoryCorpus:
    """Minimal corpus manager used by fallback memory coordinator."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add_entry(self, query: str, response: str, tags: Optional[List[str]] = None) -> None:
        self._entries.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now(),
            "tags": tags or [],
        })

    def get_recent_memories(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return self._entries[-limit:]

    def get_summaries(self, _limit: int) -> List[Dict[str, Any]]:
        return []


class _FallbackMemoryCoordinator:
    """Extremely small in-memory memory system for offline testing."""

    def __init__(self) -> None:
        self.corpus_manager = _InMemoryCorpus()
        self.gate_system = None

    async def store_interaction(self, query: str, response: str, tags: Optional[List[str]] = None) -> None:
        self.corpus_manager.add_entry(query, response, tags)

    async def get_memories(self, _query: str, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(reversed(self.corpus_manager.get_recent_memories(limit)))
        return [
            {
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "metadata": {"source": "recent", "final_score": 1.0},
            }
            for item in recent
        ]

    async def retrieve_relevant_memories(self, _query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        limit = (config or {}).get("recent_count", 5)
        recent = list(reversed(self.corpus_manager.get_recent_memories(limit)))
        memories = [
            {
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "source": "recent",
                "final_score": 1.0,
            }
            for item in recent
        ]
        counts = {
            "recent": len(memories),
            "semantic": 0,
            "hierarchical": 0,
        }
        return {"memories": memories, "counts": counts}

class DaemonOrchestrator:
    """
    Single orchestrator (prepare + generate split).
    - prepare_prompt: topic update, file processing, optional rewrite, prompt build
    - process_user_query: optional personality switch, commands, deictic check, generate, store
    """

    @staticmethod
    def _parse_thinking_block(response: str) -> Tuple[str, str]:
        """
        Parse response to extract thinking block and final answer.

        Args:
            response: Full LLM response potentially containing <thinking>...</thinking>

        Returns:
            Tuple of (thinking_part, final_answer_part)
            - If no thinking block found, thinking_part is empty and final_answer_part is the full response
        """
        if not response or not isinstance(response, str):
            return "", response or ""

        # Look for </thinking> delimiter
        delimiter = "</thinking>"
        if delimiter in response:
            parts = response.split(delimiter, 1)
            if len(parts) == 2:
                thinking_raw = parts[0]
                final_answer = parts[1].strip()

                # Extract thinking content (remove opening tag if present)
                thinking_content = thinking_raw
                if "<thinking>" in thinking_raw:
                    thinking_content = thinking_raw.split("<thinking>", 1)[1]

                return thinking_content.strip(), final_answer

        # No thinking block found - return empty thinking and full response as answer
        return "", response

    def __init__(
        self,
        *,
        model_manager,
        response_generator,
        file_processor,
        prompt_builder,
        memory_system=None,
        personality_manager=None,
        topic_manager=None,
        wiki_manager=None,
        tokenizer_manager=None,
        conversation_logger=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger("orchestrator")
        self.conversation_logger = conversation_logger  # kept for compatibility if referenced elsewhere
        self.model_manager = model_manager
        self.response_generator = response_generator
        self.file_processor = file_processor
        self.prompt_builder = prompt_builder
        self.memory_system = memory_system
        self.personality_manager = personality_manager
        self.topic_manager = topic_manager
        # inside __init__ after self.topic_manager = topic_manager
        try:
            if self.topic_manager and hasattr(gate_system, "set_topic_resolver"):
                # gate_system will now call TopicManager to turn “tell me about …” into a clean title
                gate_system.set_topic_resolver(self.topic_manager.get_primary_topic)
                if self.logger:
                    self.logger.debug("[orchestrator] Topic resolver registered with gate_system")
        except Exception as e:
            if self.logger:
                self.logger.debug(f"[orchestrator] Could not register topic resolver: {e}")

        self.wiki_manager = wiki_manager
        self.tokenizer_manager = tokenizer_manager

        # Use a single logger field throughout (can be stdlib logger or your own)


        self.config = config or {}
        self.current_topic = "general"
        self.topic_confidence_threshold = float(self.config.get("topic_confidence_threshold", 0.7))
        self.system_prompt_path = self.config.get("system_prompt_path")

    # ---------- 1) Commands & Topic ----------
    def handle_commands(self, user_input: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if user_input.startswith("/topic "):
            new_topic = user_input.replace("/topic ", "").strip()
            self.current_topic = new_topic
            if getattr(self.memory_system, "current_topic", None) is not None:
                self.memory_system.current_topic = new_topic
            return (f"Switched to topic: {new_topic}", {"command": "topic_switch"})

        if user_input == "/clear_topic":
            self.current_topic = "general"
            if getattr(self.memory_system, "current_topic", None) is not None:
                self.memory_system.current_topic = "general"
            return ("Cleared topic context, starting fresh conversation", {"command": "topic_clear"})

        return None  # no command handled

    # ---------- helpers ----------
    def _should_switch_topic(self, topics) -> bool:
        # Replace with your real threshold logic; conservative default:
        if not topics:
            return False
        if self.current_topic == "general":
            return True
        return (topics[0] or "").lower() != (self.current_topic or "").lower()

    # ---------- 2) Prepare (files, rewrite, prompt) ----------

    async def prepare_prompt(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """
        Performs pre-generation steps: topic update, file processing, optional query
        rewrite, and prompt building.

        Returns:
            (prompt, system_prompt)
        """
        # ---------------------------------------------------------------------
        # 0) Topic inference (non-fatal; use single canonical topic string)
        # ---------------------------------------------------------------------
        try:
            if getattr(self, "topic_manager", None):
                # Update TopicManager’s internal state from this turn’s input
                self.topic_manager.update_from_user_input(user_input)

                # Ask for a single canonical topic string (or None)
                primary = self.topic_manager.get_primary_topic()

                # Switch only if different from current
                if primary and (primary.lower() != (self.current_topic or "general").lower()):
                    self.current_topic = primary
                    if getattr(self, "memory_system", None) is not None:
                        self.memory_system.current_topic = self.current_topic
                    if getattr(self, "logger", None):
                        self.logger.info(f"Topic switched to: {self.current_topic}")
        except Exception:
            # Topic inference must never block the flow
            pass

        # ---------------------------------------------------------------------
        # 1) File processing (enhanced path only)
        # ---------------------------------------------------------------------
        combined_text = user_input
        if files and not use_raw_mode and getattr(self, "file_processor", None):
            try:
                combined_text = await self.file_processor.process_files(user_input, files)
            except Exception:
                # Fail-open: use raw user_input if file processing has issues
                combined_text = user_input

        # ---------------------------------------------------------------------
        # 1.5) Inline fact extraction for heavy topics/long messages
        # ---------------------------------------------------------------------
        fresh_facts: List[Dict] = []
        if not use_raw_mode and getattr(self, "memory_system", None):
            try:
                # Use async query analysis with LLM for best accuracy
                from utils.query_checker import analyze_query_async
                qinfo = await analyze_query_async(user_input, model_manager=self.model_manager)

                if qinfo.is_heavy_topic:
                    if getattr(self, "logger", None):
                        self.logger.info(
                            f"[Orchestrator] Heavy topic detected (len={len(user_input)}), "
                            "running inline fact extraction"
                        )

                    # Extract facts with timeout to avoid blocking
                    import asyncio
                    try:
                        fact_task = asyncio.create_task(
                            self.memory_system._extract_and_store_facts(
                                query=user_input,
                                response="",  # No response yet
                                truth_score=0.7  # Default for user-provided content
                            )
                        )

                        # Wait max 5 seconds
                        await asyncio.wait_for(fact_task, timeout=5.0)

                        # Retrieve just-extracted facts for prompt injection
                        fresh_facts = await self.memory_system.get_facts(
                            query=user_input,
                            limit=10  # Top 10 most relevant
                        )

                        if getattr(self, "logger", None) and fresh_facts:
                            self.logger.info(
                                f"[Orchestrator] Extracted {len(fresh_facts)} inline facts"
                            )

                    except asyncio.TimeoutError:
                        if getattr(self, "logger", None):
                            self.logger.warning("[Orchestrator] Inline fact extraction timed out")
                    except Exception as e:
                        if getattr(self, "logger", None):
                            self.logger.debug(f"[Orchestrator] Inline fact extraction failed: {e}")

            except Exception as e:
                # Never let fact extraction block the flow
                if getattr(self, "logger", None):
                    self.logger.debug(f"[Orchestrator] Heavy topic detection failed: {e}")

        # ---------------------------------------------------------------------
        # 2) Optional query rewrite (for retrieval/search phrasing)
        # ---------------------------------------------------------------------
        rewritten_query: Optional[str] = None
        qinfo = None
        try:
            qinfo = analyze_query(user_input)
        except Exception:
            qinfo = None
        # Only rewrite longer questions; skip tiny factoids to save latency
        # Allow config to disable query rewrite for lower latency
        _features = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
        _enable_rewrite = bool(_features.get("enable_query_rewrite", True))
        if _enable_rewrite and not use_raw_mode and qinfo and qinfo.is_question and qinfo.token_count >= 8:
            try:
                rewrite_prompt = (
                    'Rewrite the following user question into a concise, third-person declarative '
                    'statement suitable for a vector database search.\n\n'
                    f'User question: "{user_input}"\nRewritten statement:'
                )
                # Use a low-latency alias for quick rewrites; allow disabling timeout via config (<= 0)
                from config.app_config import REWRITE_TIMEOUT_S
                import asyncio as _a
                _rw_timeout = 0.0
                try:
                    _rw_timeout = float(REWRITE_TIMEOUT_S)
                except Exception:
                    _rw_timeout = 0.0
                _coro = self.model_manager.generate_once(
                    prompt=rewrite_prompt, model_name="gpt-4o-mini"
                )
                if _rw_timeout > 0:
                    rewritten_query = await _a.wait_for(_coro, timeout=_rw_timeout)
                else:
                    rewritten_query = await _coro
                if isinstance(rewritten_query, str):
                    rewritten_query = rewritten_query.strip().strip('"')
                else:
                    rewritten_query = user_input
            except Exception:
                rewritten_query = user_input

        # ---------------------------------------------------------------------
        # 3) Resolve system prompt (robust order + config-aware)
        # ---------------------------------------------------------------------
        SYSTEM_PROMPT_FALLBACK = (
            "You are Daemon, a helpful assistant with memory and RAG. "
            "Be direct, truthful, concise."
        )
        system_prompt: str = SYSTEM_PROMPT_FALLBACK

        # Merge persona config over base orchestrator config
        try:
            persona_cfg = self.personality_manager.get_current_config() if self.personality_manager else {}
        except Exception:
            persona_cfg = {}
        base_cfg = getattr(self, "config", {}) or {}
        merged_cfg = {**base_cfg, **(persona_cfg or {})}

        # Prefer centralized loader so it can read paths.* / prompts.* from cfg
        try:
            from config.app_config import load_system_prompt  # local import to avoid hard dep at import time
            loaded = load_system_prompt(merged_cfg)
            if isinstance(loaded, str) and loaded.strip():
                system_prompt = loaded
        except Exception:
            pass

        # Optional path override from persona or orchestrator
        override_path = None
        spf = (persona_cfg or {}).get("system_prompt_file")
        if isinstance(spf, str):
            override_path = spf
        elif isinstance(spf, dict):
            override_path = spf.get("system_prompt_file")

        if not override_path:
            override_path = getattr(self, "system_prompt_path", None)

        if override_path and isinstance(override_path, str):
            try:
                if os.path.exists(override_path):
                    with open(override_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        system_prompt = text
            except Exception:
                pass

            if getattr(self, "logger", None):
                self.logger.info(
                    f"[orchestrator] Using system prompt len={len(system_prompt)}; "
                    f"head={repr(system_prompt[:80])}"
                )

        # ---------------------------------------------------------------------
        # 4) Append resolved topic hint to the end of the system prompt
        #     (kept simple; if no topic was inferred, default to 'general').
        # ---------------------------------------------------------------------
        try:
            topic_str = (getattr(self, "current_topic", None) or "general").strip()
            if isinstance(system_prompt, str) and system_prompt.strip():
                system_prompt = system_prompt.rstrip() + f"\n\nQuery topic: {topic_str}"
        except Exception:
            # Never let topic hint break the flow
            pass

        # ---------------------------------------------------------------------
        # 4.25) Inject conversation thread context for tone/style adjustment
        # ---------------------------------------------------------------------
        if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
            try:
                if self.memory_system and hasattr(self.memory_system, 'get_thread_context'):
                    thread_ctx = self.memory_system.get_thread_context()

                    if thread_ctx and thread_ctx.get("thread_id"):
                        thread_depth = thread_ctx.get("thread_depth", 1)
                        is_heavy = thread_ctx.get("is_heavy_topic", False)
                        thread_topic = thread_ctx.get("thread_topic", "")

                        # Build thread context message
                        thread_msg = f"\n\n[THREAD CONTEXT]"
                        thread_msg += f"\nThis is message #{thread_depth} in an ongoing conversation thread"

                        if thread_topic:
                            thread_msg += f" about {thread_topic}"

                        if is_heavy:
                            thread_msg += "\nThis is a sensitive/heavy topic. "

                            if thread_depth >= 3:
                                thread_msg += (
                                    "You've been discussing this for multiple turns. "
                                    "Focus on specific details and avoid repeating general therapeutic questions. "
                                    "Reference concrete facts from earlier in the conversation."
                                )
                            else:
                                thread_msg += (
                                    "Be empathetic and specific. "
                                    "Engage with concrete details rather than generic therapeutic responses."
                                )
                        else:
                            # Non-heavy thread: subtle continuity hint
                            if thread_depth >= 3:
                                thread_msg += "\nMaintain conversational continuity and build on previous exchanges."

                        system_prompt = system_prompt.rstrip() + thread_msg

                        if getattr(self, "logger", None):
                            self.logger.debug(
                                f"[Orchestrator] Injected thread context: "
                                f"depth={thread_depth}, heavy={is_heavy}, topic={thread_topic}"
                            )
            except Exception as e:
                # Never let thread context injection break the flow
                if getattr(self, "logger", None):
                    self.logger.debug(f"[Orchestrator] Thread context injection failed: {e}")

        # ---------------------------------------------------------------------
        # 4.5) Add thinking block instruction to system prompt
        # ---------------------------------------------------------------------
        if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
            thinking_instruction = (
                "\n\n"
                "IMPORTANT: Before you provide your final answer, you must include a <thinking> block. "
                "Inside this block, detail your step-by-step reasoning and analysis of the user's request. "
                "After the </thinking> block, provide your final, concise, and helpful response to the user."
            )
            system_prompt = system_prompt.rstrip() + thinking_instruction

        # ---------------------------------------------------------------------
        # 5) Raw mode: return plain text, no system prompt
        # ---------------------------------------------------------------------
        if use_raw_mode:
            return combined_text, None

        # ---------------------------------------------------------------------
        # 5) Build prompt (unified)
        # ---------------------------------------------------------------------
        prompt = await self.prompt_builder.build_prompt(
            user_input=combined_text,
            search_query=rewritten_query,
            personality_config=persona_cfg,
            system_prompt=system_prompt,
            current_topic=getattr(self, "current_topic", "general"),
            fresh_facts=fresh_facts,  # Pass inline-extracted facts
        )

        # Some builders return a context dict; assemble it to a single string
        if isinstance(prompt, dict):
            prompt = self.prompt_builder._assemble_prompt(
                user_input=combined_text,
                context=prompt,
                system_prompt=system_prompt,
                directives_file="structured_directives.txt",
            )

        return prompt, system_prompt

    # ---------- 3) Generate & Store ----------
    async def process_user_query(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        personality: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates the full request:
          - optional personality switch (if provided)
          - commands (early exit)
          - deictic pre-check (optional early clarification)
          - prepare_prompt
          - generate + store
        Returns: (assistant_text, debug_info)
        """
        debug_info: Dict[str, Any] = {
            "start_time": datetime.now(),
            "user_input": user_input[:100],
            "files_count": len(files) if files else 0,
            "mode": "raw" if use_raw_mode else "enhanced",
        }

        try:
            # Personality hook: let GUI pass personality labels that flip the active config
            if personality and self.personality_manager:
                try:
                    self.personality_manager.switch_personality(personality)
                    if self.logger:
                        self.logger.info(f"Personality set to: {personality}")
                except Exception:
                    if self.logger:
                        self.logger.warning(f"Could not set personality: {personality}")


            # --- Commands: early exit ---
            cmd = self.handle_commands(user_input)
            if cmd:
                # shape matches handler expectations: (text, debug_info)
                text, meta = cmd
                debug_info.update(meta)
                return text, debug_info

            # --- Deictic pre-check (clarify before we build/stream) ---
            if not use_raw_mode and is_deictic(user_input) and self.memory_system:
                try:
                    retrieval_result = await self.memory_system.get_memories(user_input, limit=10)
                    if retrieval_result and retrieval_result[0].get("metadata", {}).get("needs_clarification"):
                        response = "I'm not sure what you're referring to. Could you be more specific?"
                        try:
                            await self.memory_system.store_interaction(
                                query=user_input, response=response, tags=["clarification"]
                            )
                        except Exception:
                            pass
                        debug_info.update({
                            "response_length": len(response),
                            "end_time": datetime.now(),
                            "prompt_length": 0,
                        })
                        debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()
                        return response, debug_info
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"[Orchestrator] Deictic pre-retrieval failed or skipped: {e}")

            # --- Build Prompt (unified path) ---
            prompt, system_prompt = await self.prepare_prompt(
                user_input=user_input, files=files, use_raw_mode=use_raw_mode
            )

            # --- Generate Response ---
            active_name_getter = getattr(self.model_manager, "get_active_model_name", None)
            model_name = active_name_getter() if callable(active_name_getter) else None
            model_name = model_name or "gpt-4-turbo"

            # Decide if we should use best-of (non-streaming) for quality; prefer runtime config
            try:
                from config.app_config import (
                    ENABLE_BEST_OF as DEFAULT_ENABLE_BEST_OF,
                    BEST_OF_N,
                    BEST_OF_TEMPS,
                    BEST_OF_MAX_TOKENS,
                    BEST_OF_MIN_QUESTION,
                    BEST_OF_MIN_TOKENS as _BEST_OF_MIN_TOKENS,
                )
            except Exception:
                DEFAULT_ENABLE_BEST_OF = True; BEST_OF_N = 2; BEST_OF_TEMPS = (0.2, 0.7); BEST_OF_MAX_TOKENS = 512; BEST_OF_MIN_QUESTION = True; _BEST_OF_MIN_TOKENS = 8

            _features = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
            _enable_bestof = bool(_features.get("enable_best_of", DEFAULT_ENABLE_BEST_OF))

            use_bestof = False
            try:
                qinfo = analyze_query(user_input)
                use_bestof = bool(
                    _enable_bestof and (
                        (qinfo.is_question and qinfo.token_count >= _BEST_OF_MIN_TOKENS)
                        or (not BEST_OF_MIN_QUESTION)
                    )
                )
            except Exception:
                use_bestof = bool(_enable_bestof)

            if use_bestof and not use_raw_mode:
                try:
                    from config.app_config import (
                        BEST_OF_MODEL,
                        BEST_OF_LATENCY_BUDGET_S,
                        BEST_OF_GENERATOR_MODELS as DEF_GEN_MODELS,
                        BEST_OF_SELECTOR_MODELS as DEF_SEL_MODELS,
                        BEST_OF_SELECTOR_MAX_TOKENS as DEF_SEL_MAXTOK,
                        BEST_OF_SELECTOR_WEIGHTS as DEF_SEL_WEIGHTS,
                        BEST_OF_SELECTOR_TOP_K as DEF_SEL_TOPK,
                        BEST_OF_DUEL_MODE as DEF_DUEL_MODE,
                        BEST_OF_MAX_TOKENS as DEF_BEST_MAXTOK,
                    )
                except Exception:
                    BEST_OF_MODEL = None; BEST_OF_LATENCY_BUDGET_S = 2.0
                    DEF_GEN_MODELS = []
                    DEF_SEL_MODELS = []
                    DEF_SEL_MAXTOK = 64
                    DEF_SEL_WEIGHTS = {"heuristic": 1.0, "llm": 0.0}
                    DEF_SEL_TOPK = 0
                    DEF_DUEL_MODE = False
                    DEF_BEST_MAXTOK = 512

                import asyncio as _a
                # Run best-of with an optional latency budget; if disabled (<=0), do not time out
                try:
                    _budget = 0.0
                    try:
                        _budget = float(BEST_OF_LATENCY_BUDGET_S)
                    except Exception:
                        _budget = 0.0

                    # Feature overrides (runtime-configurable via GUI)
                    _runtime = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
                    GEN_MODELS = list(_runtime.get('best_of_generator_models', DEF_GEN_MODELS))
                    SEL_MODELS = list(_runtime.get('best_of_selector_models', DEF_SEL_MODELS))
                    SEL_MAXTOK = int(_runtime.get('best_of_selector_max_tokens', DEF_SEL_MAXTOK))
                    SEL_WEIGHTS = dict(_runtime.get('best_of_selector_weights', DEF_SEL_WEIGHTS)) if isinstance(_runtime.get('best_of_selector_weights', DEF_SEL_WEIGHTS), dict) else DEF_SEL_WEIGHTS
                    SEL_TOPK = int(_runtime.get('best_of_selector_top_k', DEF_SEL_TOPK))
                    BEST_MAXTOK = int(_runtime.get('best_of_max_tokens', DEF_BEST_MAXTOK))

                    # If multi-model generators are configured, use ensemble path
                    use_ensemble = bool(GEN_MODELS)
                    if self.logger:
                        try:
                            self.logger.info(
                                f"[BESTOF] ensemble={use_ensemble} gens={list(GEN_MODELS)} "
                                f"selectors={list(SEL_MODELS)} weights={dict(SEL_WEIGHTS)} "
                                f"top_k={int(SEL_TOPK)} model={(BEST_OF_MODEL or model_name)} max_tokens={int(BEST_MAXTOK)}"
                            )
                        except Exception:
                            pass
                    temps_used = tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else (0.2, 0.7)
                    # Optional strict duel mode (two generators + one judge)
                    _DUEL_MODE = bool(_runtime.get('best_of_duel_mode', DEF_DUEL_MODE))
                    use_duel = bool(_DUEL_MODE and len(GEN_MODELS) == 2 and len(SEL_MODELS) >= 1)
                    if _budget > 0:
                        if use_duel:
                            m1, m2 = list(GEN_MODELS)[:2]
                            judge = list(SEL_MODELS)[0]
                            best_task = _a.create_task(
                                self.response_generator.generate_duel_and_judge(
                                    prompt=prompt,
                                    model_a=m1,
                                    model_b=m2,
                                    judge_model=judge,
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    max_tokens=BEST_MAXTOK,
                                    temperature_a=(temps_used[0] if len(temps_used) > 0 else None),
                                    temperature_b=(temps_used[1] if len(temps_used) > 1 else None),
                                    judge_max_tokens=int(SEL_MAXTOK),
                                )
                            )
                        elif use_ensemble:
                            best_task = _a.create_task(
                                self.response_generator.generate_best_of_ensemble(
                                    prompt=prompt,
                                    generator_models=list(GEN_MODELS),
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    n_total=BEST_OF_N,
                                    temps=temps_used,
                                    max_tokens=BEST_MAXTOK,
                                    selector_models=list(SEL_MODELS),
                                    selector_max_tokens=int(SEL_MAXTOK),
                                    weight_heuristic=float(SEL_WEIGHTS.get("heuristic", 0.5)),
                                    weight_llm=float(SEL_WEIGHTS.get("llm", 0.5)),
                                    judge_top_k=int(SEL_TOPK),
                                )
                            )
                        else:
                            best_task = _a.create_task(
                                self.response_generator.generate_best_of(
                                    prompt=prompt,
                                    model_name=BEST_OF_MODEL or model_name,
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    n=BEST_OF_N,
                                    temps=temps_used,
                                    max_tokens=BEST_MAXTOK,
                                )
                            )
                        full_response = await _a.wait_for(best_task, timeout=_budget)
                    else:
                        if use_duel:
                            m1, m2 = list(GEN_MODELS)[:2]
                            judge = list(SEL_MODELS)[0]
                            full_response = await self.response_generator.generate_duel_and_judge(
                                prompt=prompt,
                                model_a=m1,
                                model_b=m2,
                                judge_model=judge,
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                max_tokens=BEST_MAXTOK,
                                temperature_a=(temps_used[0] if len(temps_used) > 0 else None),
                                temperature_b=(temps_used[1] if len(temps_used) > 1 else None),
                                judge_max_tokens=int(SEL_MAXTOK),
                            )
                        elif use_ensemble:
                            full_response = await self.response_generator.generate_best_of_ensemble(
                                prompt=prompt,
                                generator_models=list(GEN_MODELS),
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                n_total=BEST_OF_N,
                                temps=temps_used,
                                max_tokens=BEST_MAXTOK,
                                selector_models=list(SEL_MODELS),
                                selector_max_tokens=int(SEL_MAXTOK),
                                weight_heuristic=float(SEL_WEIGHTS.get("heuristic", 0.5)),
                                weight_llm=float(SEL_WEIGHTS.get("llm", 0.5)),
                                judge_top_k=int(SEL_TOPK),
                            )
                        else:
                            full_response = await self.response_generator.generate_best_of(
                                prompt=prompt,
                                model_name=BEST_OF_MODEL or model_name,
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                n=BEST_OF_N,
                                temps=temps_used,
                                max_tokens=BEST_MAXTOK,
                            )
                except Exception:
                    # Timeout or error: cancel best-of (if applicable) and proceed with streaming
                    try:
                        if 'best_task' in locals():
                            best_task.cancel()
                    except Exception:
                        pass
                    # Accumulate full response (no yielding yet - need to parse thinking block)
                    full_response = ""
                    async for chunk in self.response_generator.generate_streaming_response(
                        prompt, model_name, system_prompt=system_prompt
                    ):
                        full_response += (chunk + " ")
                    full_response = full_response.strip()
            else:
                # Accumulate full response (no yielding yet - need to parse thinking block first)
                full_response = ""
                async for chunk in self.response_generator.generate_streaming_response(
                    prompt, model_name, system_prompt=system_prompt
                ):
                    full_response += (chunk + " ")
                full_response = full_response.strip()

            # --- Parse thinking block and extract final answer ---
            thinking_part, final_answer = self._parse_thinking_block(full_response)

            # Log thinking part for debugging if present
            if thinking_part:
                if self.logger:
                    self.logger.debug(f"[THINKING BLOCK]\n{thinking_part}")
                debug_info["thinking_length"] = len(thinking_part)

            # Store final answer (not the thinking part) in memory
            answer_for_storage = final_answer if final_answer else full_response

            # --- Store Interaction ---
            if self.memory_system and not use_raw_mode:
                try:
                    await self.memory_system.store_interaction(
                        query=user_input,
                        response=answer_for_storage,
                        tags=["conversation"]
                    )
                except Exception:
                    pass
            # Use instance logger
            if self.logger:
                self.logger.debug("[orchestrator] Persisted exchange; considering consolidation")

            # Summaries now run on shutdown; skip mid-session consolidation
            debug_info.update({
                "response_length": len(answer_for_storage),
                "full_response_length": len(full_response),
                "end_time": datetime.now(),
                "prompt_length": len(prompt),
            })
            debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()

            # Return only the final answer (thinking block removed)
            return answer_for_storage, debug_info

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing query: {e}", exc_info=True)
            if getattr(self, "conversation_logger", None):
                try:
                    self.conversation_logger.log_system_event("Error", str(e))
                except Exception:
                    pass
            debug_info["error"] = str(e)
            raise
