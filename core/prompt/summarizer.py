"""
# core/prompt/summarizer.py

Module Contract
- Purpose: LLM-based summarization and on-demand reflection generation for prompt building.
- Class: LLMSummarizer(model_manager, memory_coordinator)
- Key methods:
  - _llm_summarize_recent(conversations, topic, max_conversations) -> Optional[str]
    Async LLM call to summarize recent conversations (with SUM_TIMEOUT).
  - _reflect_on_demand(context, user_input, session_reflections) -> List[Dict]
    Generates N reflections to top up session reflections to REFLECTIONS_MAX_TARGET.
  - _persist_summary(summary_text, source_conversations, topic) -> None
    Stores generated summary via memory_coordinator.add_summary().
  - _fallback_micro_summary(conversations) -> str
    Simple keyword-based fallback when LLM summarization fails.
- Outputs:
  - Summary text strings or None on failure
  - Reflection dicts with content, timestamp, tags, source
- Dependencies:
  - models.model_manager (LLM API access for summarization/reflection)
  - memory.memory_coordinator (summary persistence via add_summary)
  - config.app_config (memory config for reflection target count)
- Side effects:
  - LLM API calls with configurable timeouts (SUM_TIMEOUT=30s, reflection=15s)
  - Summary persistence to memory coordinator
- Config env vars:
  - FORCE_LLM_SUMMARIES, SUM_TIMEOUT, REFLECTIONS_ON_DEMAND, REFLECTION_MAX_EXCERPTS
"""

import os
import asyncio
import inspect
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logging_utils import get_logger
from utils.ordered_slice import head as _ordered_head

logger = get_logger("prompt_summarizer")

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    """Parse boolean from string, with fallback."""
    if not s:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")

# Summary configuration
FORCE_LLM_SUMMARIES = _parse_bool(os.getenv("FORCE_LLM_SUMMARIES", "0"))
SUM_TIMEOUT = int(os.getenv("SUM_TIMEOUT", "30"))
REFLECTIONS_ON_DEMAND = _parse_bool(os.getenv("REFLECTIONS_ON_DEMAND", "1"))

# Target number of reflections to include in prompt (kept in sync with builder)
try:
    from config.app_config import config as _APP_CFG2
    _MEM_CFG2 = (_APP_CFG2.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG2 = {}

def _cfg2_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG2.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

REFLECTIONS_MAX_TARGET = _cfg2_int("prompt_max_reflections", 10)


class LLMSummarizer:
    """Handles LLM-based summarization and reflection generation."""

    def __init__(self, model_manager, memory_coordinator):
        self.model_manager = model_manager
        self.memory_coordinator = memory_coordinator
        self._summaries_model = None

    def _ensure_summaries_model(self):
        """Ensure we have a model for summarization tasks."""
        if not self._summaries_model:
            try:
                # Try to get a suitable model for summarization
                if hasattr(self.model_manager, 'get_active_model_name'):
                    model_name = self.model_manager.get_active_model_name()
                    self._summaries_model = model_name
                else:
                    # Fallback
                    self._summaries_model = "gpt-3.5-turbo"
            except (AttributeError, RuntimeError):
                self._summaries_model = "gpt-3.5-turbo"
        return self._summaries_model

    def _decide_gen_params(self, model_name: str) -> Dict[str, Any]:
        """Decide generation parameters for summarization."""
        return {
            "temperature": 0.3,
            "max_tokens": 500,
            "model": model_name
        }

    async def _generate_text(self, prompt: str, *, timeout: float,
                             max_tokens: int) -> str:
        """Run an internal generation and normalize streaming/non-streaming APIs.

        ``ModelManager.generate_async`` intentionally returns an async stream for
        API models but a plain string for local models.  Internal summarization
        needs one complete string, so prefer the non-streaming contract and keep
        a compatibility adapter for older/custom managers.  The timeout wraps
        both request creation and stream consumption.
        """
        model_name = self._ensure_summaries_model()

        async def _collect(result: Any) -> str:
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return ""
            if isinstance(result, str):
                return result.strip()
            if isinstance(result, dict):
                return str(result.get("content") or "").strip()
            # Only TEXT content counts: a Mock/stub manager auto-creates a
            # `.content` attribute, and stringifying it produced a fake
            # "<Mock …>" summary (test_llm_summarize_recent_no_model_manager).
            content = getattr(result, "content", None)
            if isinstance(content, str):
                return content.strip()
            if hasattr(result, "__aiter__"):
                parts: List[str] = []
                async for chunk in result:
                    if isinstance(chunk, str):
                        parts.append(chunk)
                        continue
                    if isinstance(chunk, dict):
                        piece = chunk.get("content", "")
                    else:
                        choices = getattr(chunk, "choices", None) or []
                        delta = getattr(choices[0], "delta", None) if choices else None
                        piece = getattr(delta, "content", "") if delta is not None else ""
                    if piece:
                        parts.append(str(piece))
                return "".join(parts).strip()
            return ""

        async def _run() -> str:
            generate_once = getattr(self.model_manager, "generate_once", None)
            if callable(generate_once):
                return await _collect(generate_once(
                    prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=0.3,
                ))

            generate_async = getattr(self.model_manager, "generate_async", None)
            if callable(generate_async):
                # generate_async(self, prompt, raw=False, images=None, **kwargs)
                # has no `model` param — it always uses the manager's active
                # model and forwards **kwargs into the local-model generate()
                # path, where an unexpected `model=` kwarg used to raise.
                return await _collect(generate_async(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,
                ))

            generate = getattr(self.model_manager, "generate", None)
            if callable(generate):
                result = await asyncio.to_thread(
                    generate,
                    prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return await _collect(result)
            return ""

        return await asyncio.wait_for(_run(), timeout=timeout)

    async def _llm_summarize_recent(self, conversations: List[Dict[str, Any]],
                                  topic: str = "", max_conversations: int = 10) -> Optional[str]:
        """
        Use LLM to summarize recent conversations.

        Args:
            conversations: List of conversation dicts with 'query' and 'response'
            topic: Optional topic focus for the summary
            max_conversations: Maximum number of conversations to include

        Returns:
            Summary text or None if generation fails
        """
        if not conversations:
            return None

        try:
            model_name = self._ensure_summaries_model()

            # Limit conversations to avoid token overflow — keep the NEWEST
            # max_conversations. conversations arrives newest-first (see
            # memory.corpus_manager._get_episodic_sorted, the source behind
            # gatherer_memory._get_recent_conversations), so a bare
            # conversations[-max_conversations:] took the OLDEST slice
            # whenever more than max_conversations were passed in — the
            # same newest-first-then-truncate class utils.ordered_slice.head
            # (single source of truth for "sort by timestamp before
            # slicing") exists to close (2026-09-04).
            limited_convs = _ordered_head(
                conversations, max_conversations,
                ts_key=lambda c: c.get("timestamp") if isinstance(c, dict) else None,
            ) if len(conversations) > max_conversations else conversations

            # Build conversation text
            conv_parts = []
            for conv in limited_convs:
                query = conv.get("query", "").strip()
                response = conv.get("response", "").strip()
                if query and response:
                    conv_parts.append(f"Human: {query}\nAssistant: {response}")
                elif response:
                    conv_parts.append(f"Assistant: {response}")
                elif query:
                    conv_parts.append(f"Human: {query}")

            if not conv_parts:
                return None

            conversation_text = "\n\n".join(conv_parts)

            # Build summary prompt
            topic_clause = f" focusing on {topic}" if topic else ""
            summary_prompt = f"""Please summarize the following conversation{topic_clause}. Extract key points, decisions, and important information in 2-3 sentences:

{conversation_text}

Summary:"""

            summary = await self._generate_text(
                summary_prompt,
                timeout=SUM_TIMEOUT,
                max_tokens=self._decide_gen_params(model_name)["max_tokens"],
            )

            if summary:
                logger.info(f"Generated LLM summary: {len(summary)} chars")
                return summary

        except asyncio.TimeoutError:
            logger.warning(f"LLM summarization timed out after {SUM_TIMEOUT}s")
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")

        return None

    def _persist_summary(self, summary_text: str, source_conversations: List[Dict[str, Any]],
                        topic: str = "") -> None:
        """
        Persist a generated summary to the memory coordinator.

        Args:
            summary_text: The summary content
            source_conversations: Original conversations that were summarized
            topic: Optional topic tag for the summary
        """
        try:
            # Create summary metadata
            timestamp = datetime.now().isoformat()
            conv_count = len(source_conversations)

            # Build tags
            tags = ["llm_generated", f"conversations_{conv_count}"]
            if topic:
                tags.append(f"topic_{topic.lower()}")

            # Create summary dict
            summary_dict = {
                "content": summary_text,
                "timestamp": timestamp,
                "tags": tags,
                "source": "llm_summarizer",
                "source_count": conv_count
            }

            # Store via memory coordinator
            if hasattr(self.memory_coordinator, 'add_summary'):
                self.memory_coordinator.add_summary(summary_dict)
            elif hasattr(self.memory_coordinator, 'corpus_manager'):
                # Fallback to corpus manager
                corpus_manager = self.memory_coordinator.corpus_manager
                if hasattr(corpus_manager, 'add_summary'):
                    corpus_manager.add_summary(summary_dict)

            logger.info(f"Persisted summary: {len(summary_text)} chars, {conv_count} conversations")

        except Exception as e:
            logger.warning(f"Failed to persist summary: {e}")

    def _fallback_micro_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Create a simple fallback summary when LLM summarization fails.

        Args:
            conversations: List of conversation dicts

        Returns:
            Simple summary string
        """
        if not conversations:
            return "No recent conversations to summarize."

        # Count conversations and extract key topics
        conv_count = len(conversations)

        # Extract some keywords from queries — the newest 5 (conversations
        # arrives newest-first; see the _llm_summarize_recent note above).
        keywords = set()
        for conv in _ordered_head(
            conversations, 5,
            ts_key=lambda c: c.get("timestamp") if isinstance(c, dict) else None,
        ):
            query = conv.get("query", "").strip().lower()
            if query:
                # Simple keyword extraction
                words = query.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        keywords.add(word)

        keyword_str = ", ".join(list(keywords)[:5]) if keywords else "various topics"

        return f"Recent {conv_count} conversations covering: {keyword_str}"

    async def _reflect_on_demand(self, context: Dict[str, Any], user_input: str,
                                session_reflections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate reflections on-demand based on current context and user input.

        Args:
            context: Current prompt context
            user_input: Current user query
            session_reflections: Existing session reflections

        Returns:
            List of new reflection dicts
        """
        if not REFLECTIONS_ON_DEMAND:
            return []

        try:
            # Check how many we need to top up to the target
            if len(session_reflections) >= REFLECTIONS_MAX_TARGET:
                return []  # Already at or above target
            needed = max(0, min(REFLECTIONS_MAX_TARGET - len(session_reflections), 6))  # cap to avoid latency spikes

            # Extract relevant content for reflection; fall back to recents/semantic if needed
            recent_memories = context.get("memories", [])[-3:]
            if not recent_memories:
                recent_memories = context.get("recent_conversations", [])[-3:]
            recent_facts = context.get("fresh_facts", [])[-5:]
            if not recent_facts:
                recent_facts = context.get("semantic_facts", [])[-5:]

            # Build reflection prompt
            reflection_context = []

            # Add recent memories
            if recent_memories:
                mem_text = []
                for mem in recent_memories:
                    query = mem.get("query", "")
                    response = mem.get("response", "")
                    if query and response:
                        mem_text.append(f"Q: {query}\nA: {response}")
                if mem_text:
                    reflection_context.append("Recent conversation:\n" + "\n\n".join(mem_text))

            # Add recent facts
            if recent_facts:
                fact_text = []
                for fact in recent_facts:
                    if isinstance(fact, dict):
                        content = fact.get("content", "")
                        if content:
                            fact_text.append(content)
                    else:
                        fact_text.append(str(fact))
                if fact_text:
                    reflection_context.append("Recent facts:\n" + "\n".join(fact_text))

            if not reflection_context:
                return []

            context_text = "\n\n".join(reflection_context)

            reflection_prompt = f"""Based on the following context, generate {needed} distinct reflections about patterns, insights, or meta-observations about the conversation. Focus on themes, learning patterns, or behavioral observations.

Context:
{context_text}

Current query: {user_input}

Return exactly {needed} reflections, each on its own line prefixed with "- ". Keep each to 1–2 sentences."""

            # Generate reflection using LLM
            model_name = self._ensure_summaries_model()
            gen_params = self._decide_gen_params(model_name)
            gen_params["max_tokens"] = 150 + (needed - 1) * 80  # allow a bit more for multiple items

            reflection_text = await self._generate_text(
                reflection_prompt,
                timeout=15,
                max_tokens=gen_params["max_tokens"],
            )

            reflection_text = (reflection_text or "").strip()
            if not reflection_text:
                return []

            # Parse multiple reflections from the generated text
            lines = [ln.strip() for ln in reflection_text.splitlines() if ln.strip()]
            items: List[Dict[str, Any]] = []
            for ln in lines:
                # Strip leading bullets or numbering
                cleaned = ln
                if cleaned.startswith("- "):
                    cleaned = cleaned[2:].strip()
                elif cleaned[:2].isdigit() and ". " in cleaned[:4]:
                    # handles "1. text" style
                    cleaned = cleaned.split(". ", 1)[-1].strip()
                if len(cleaned) < 5:
                    continue
                items.append({
                    "content": cleaned,
                    "timestamp": datetime.now().isoformat(),
                    "tags": ["on_demand", "session"],
                    "source": "reflection_generator"
                })

            if items:
                logger.info(f"Generated on-demand reflections: {len(items)} items")
                # Trim to 'needed' count just in case
                return items[:needed]

        except asyncio.TimeoutError:
            logger.warning("On-demand reflection timed out")
        except Exception as e:
            logger.warning(f"On-demand reflection failed: {e}")

        return []
