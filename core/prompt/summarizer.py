"""
# core/prompt/summarizer.py

Module Contract
- Purpose: LLM-based summarization and reflection generation for prompt building.
- Inputs:
  - summarize_conversations(conversations: List[Dict], model_name: str) -> str
  - generate_reflection(memories: List[Dict], query: str) -> str
  - cache_summary(key: str, summary: str) -> None
- Outputs:
  - Generated summaries of conversation history
  - Reflective insights from memory patterns
  - Cached summaries for performance optimization
- Behavior:
  - Uses LLM models to create concise summaries of recent conversations
  - Generates reflections by analyzing patterns in stored memories
  - Implements caching to avoid redundant summarization requests
  - Handles async operations with proper error handling and timeouts
- Dependencies:
  - models.model_manager (LLM access for summarization)
  - utils.time_manager (temporal context)
  - utils.logging_utils (operation logging)
- Side effects:
  - LLM API calls for summarization
  - Cache writes for performance optimization
  - Logging of summarization activities
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.time_manager import TimeManager
from utils.logging_utils import get_logger

logger = get_logger("prompt_summarizer")

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except Exception:
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except Exception:
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
except Exception:
    _MEM_CFG2 = {}

def _cfg2_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG2.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except Exception:
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
            except Exception:
                self._summaries_model = "gpt-3.5-turbo"
        return self._summaries_model

    def _decide_gen_params(self, model_name: str) -> Dict[str, Any]:
        """Decide generation parameters for summarization."""
        return {
            "temperature": 0.3,
            "max_tokens": 500,
            "model": model_name
        }

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

            # Limit conversations to avoid token overflow
            limited_convs = conversations[-max_conversations:] if len(conversations) > max_conversations else conversations

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

            # Generate summary
            gen_params = self._decide_gen_params(model_name)

            # Use model manager to generate
            if hasattr(self.model_manager, 'generate_async'):
                response = await asyncio.wait_for(
                    self.model_manager.generate_async(summary_prompt, **gen_params),
                    timeout=SUM_TIMEOUT
                )
            else:
                # Fallback to sync generation
                response = self.model_manager.generate(summary_prompt, **gen_params)

            if response and hasattr(response, 'content'):
                summary = response.content.strip()
            elif isinstance(response, str):
                summary = response.strip()
            else:
                summary = str(response).strip() if response else None

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

        # Extract some keywords from queries
        keywords = set()
        for conv in conversations[-5:]:  # Look at last 5 for keywords
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

            if hasattr(self.model_manager, 'generate_async'):
                async_stream = await asyncio.wait_for(
                    self.model_manager.generate_async(reflection_prompt, **gen_params),
                    timeout=15  # Shorter timeout for reflections
                )

                # Consume the async stream to get the actual text
                reflection_text = ""
                try:
                    async for chunk in async_stream:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                reflection_text += delta.content
                except Exception as e:
                    logger.warning(f"Error consuming reflection stream: {e}")
                    reflection_text = ""
            else:
                response = self.model_manager.generate(reflection_prompt, **gen_params)
                if response:
                    if hasattr(response, 'content'):
                        reflection_text = response.content.strip()
                    elif isinstance(response, str):
                        reflection_text = response.strip()
                    else:
                        reflection_text = str(response).strip()
                else:
                    reflection_text = ""

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
