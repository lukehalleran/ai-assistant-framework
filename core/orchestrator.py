# core/orchestrator.py
import os
import processing.gate_system as gate_system
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from utils.logging_utils import get_logger
import processing.gate_system as gate_system
from integrations.wikipedia_api import WikipediaAPI

SYSTEM_PROMPT = "..."  # safe fallback (replace with your real default)
wiki_api = WikipediaAPI()
gate_system.wikipedia_api = wiki_api  # This sets it globally
# If you have a real helper, import that instead:
# from utils.query_checks import is_deictic
def is_deictic(text: str) -> bool:
    # Placeholder; replace with your real implementation
    t = text.strip().lower()
    return t in {"what about that?", "that?", "and that?"}


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

SYSTEM_PROMPT = "..."  # safe fallback (replace with your real default)
wiki_api = WikipediaAPI()
gate_system.wikipedia_api = wiki_api  # This sets it globally
class DaemonOrchestrator:
    """
    Single orchestrator (prepare + generate split).
    - prepare_prompt: topic update, file processing, optional rewrite, prompt build
    - process_user_query: optional personality switch, commands, deictic check, generate, store
    """

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
        # 2) Optional query rewrite (for retrieval/search phrasing)
        # ---------------------------------------------------------------------
        rewritten_query: Optional[str] = None
        if not use_raw_mode and (
            "?" in user_input
            or user_input.lower().startswith(("what", "who", "when", "how", "why"))
        ):
            try:
                rewrite_prompt = (
                    'Rewrite the following user question into a concise, third-person declarative '
                    'statement suitable for a vector database search.\n\n'
                    f'User question: "{user_input}"\nRewritten statement:'
                )
                rewritten_query = await self.model_manager.generate_once(
                    prompt=rewrite_prompt, model_name="gpt-4-turbo"
                )
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
        # 4) Raw mode: return plain text, no system prompt
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

            full_response = ""
            async for chunk in self.response_generator.generate_streaming_response(
                prompt, model_name, system_prompt=system_prompt
            ):
                full_response += (chunk + " ")
            full_response = full_response.strip()

            # --- Store Interaction ---
            if self.memory_system and not use_raw_mode:
                try:
                    await self.memory_system.store_interaction(
                        query=user_input,
                        response=full_response,
                        tags=["conversation"]
                    )
                except Exception:
                    pass
            # Use instance logger
            if self.logger:
                self.logger.debug("[orchestrator] Persisted exchange; considering consolidation")

            # --- Add this block ---
            try:
                # Consolidate every N exchanges (configurable)
                if hasattr(self, "prompt_builder") and hasattr(self.prompt_builder, "consolidator"):
                    await self.prompt_builder.consolidator.maybe_consolidate(
                        corpus_manager=getattr(self.memory_system, "corpus_manager", None)
                    )
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"[Consolidation] skipped: {e}")
            debug_info.update({
                "response_length": len(full_response),
                "end_time": datetime.now(),
                "prompt_length": len(prompt),
            })
            debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()
            return full_response, debug_info

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
