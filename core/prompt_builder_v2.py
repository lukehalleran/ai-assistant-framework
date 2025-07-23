# core/prompt_builder_v2.py
import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logging_utils import get_logger, log_and_time

logger = get_logger("prompt_builder_v2")

class UnifiedPromptBuilder:
    """
    Unified prompt builder that combines all functionality:
    - Memory retrieval from multiple sources
    - Cosine similarity gating
    - Hierarchical memory expansion
    - Token budget management import PromptBuilder
    from core.prompt_builder import UnifiedHierarchicalPromptBuilder
    - Final prompt assembly
    """

    def __init__(
        self,
        model_manager,
        memory_coordinator,
        tokenizer_manager,
        wiki_manager,
        topic_manager,
        gate_system=None,
        max_tokens=4096,
        reserved_for_output=512
    ):
        self.model_manager = model_manager
        self.memory_coordinator = memory_coordinator
        self.tokenizer_manager = tokenizer_manager
        self.wiki_manager = wiki_manager
        self.topic_manager = topic_manager
        self.gate_system = gate_system
        self.max_tokens = max_tokens
        self.reserved_for_output = reserved_for_output
        self.token_budget = max_tokens - reserved_for_output

    @log_and_time("Build Prompt")
    async def build_prompt(
        self,
        user_input: str,
        include_dreams: bool = True,
        include_wiki: bool = True,
        include_semantic: bool = True,
        system_prompt: str = "",
        directives_file: str = "structured_directives.txt",
        personality_config: Dict = None,
        **kwargs
    ) -> str:
        """
        Main entry point for building prompts.
        Handles all retrieval, gating, and assembly in one place.
        """
        logger.debug(f"[PROMPT] Building prompt for: {user_input[:100]}...")

        # Step 1: Gather all context sources
        context = await self._gather_context(
            user_input,
            include_dreams=include_dreams,
            include_wiki=include_wiki,
            include_semantic=include_semantic,
            personality_config=personality_config
        )

        # Step 2: Apply gating if available
        if self.gate_system:
            context = await self._apply_gating(user_input, context)

        # Step 3: Manage token budget
        context = self._manage_token_budget(context)

        # Step 4: Assemble final prompt
        prompt = self._assemble_prompt(
            user_input=user_input,
            context=context,
            system_prompt=system_prompt,
            directives_file=directives_file
        )

        logger.debug(f"[PROMPT] Final length: {len(prompt)} chars")
        return prompt

    async def _gather_context(
        self,
        user_input: str,
        include_dreams: bool,
        include_wiki: bool,
        include_semantic: bool,
        personality_config: Dict = None
    ) -> Dict[str, Any]:
        """Gather all context sources in parallel"""
        context = {
            "memories": [],
            "summaries": [],
            "dreams": [],
            "wiki": "",
            "semantic_chunks": [],
            "recent_conversations": [],
            "time_context": self._get_time_context()
        }

        # Configure based on personality
        config = personality_config or {}
        memory_count = config.get("num_memories", 10)

        # Parallel retrieval using asyncio.gather
        tasks = []

        # Recent conversations (always included)
        tasks.append(self._get_recent_conversations(5))

        # Memories
        tasks.append(self.memory_coordinator.get_memories(user_input, limit=memory_count))

        # Summaries
        tasks.append(self._get_summaries(3))

        # Dreams (if enabled)
        if include_dreams:
            tasks.append(self._get_dreams(2))
        else:
            tasks.append(asyncio.create_task(asyncio.coroutine(lambda: [])()))

        # Wiki content (if enabled)
        if include_wiki:
            tasks.append(self._get_wiki_content(user_input))
        else:
            tasks.append(asyncio.create_task(asyncio.coroutine(lambda: "")()))

        # Semantic search (if enabled)
        if include_semantic:
            tasks.append(self._get_semantic_chunks(user_input))
        else:
            tasks.append(asyncio.create_task(asyncio.coroutine(lambda: [])()))

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        context["recent_conversations"] = results[0] if not isinstance(results[0], Exception) else []
        context["memories"] = results[1] if not isinstance(results[1], Exception) else []
        context["summaries"] = results[2] if not isinstance(results[2], Exception) else []
        context["dreams"] = results[3] if not isinstance(results[3], Exception) else []
        context["wiki"] = results[4] if not isinstance(results[4], Exception) else ""
        context["semantic_chunks"] = results[5] if not isinstance(results[5], Exception) else []

        return context

    async def _apply_gating(self, user_input: str, context: Dict) -> Dict:
        """Apply cosine similarity gating to filter context"""
        logger.debug("[PROMPT] Applying cosine similarity gating")

        # Gate memories
        if context["memories"]:
            context["memories"] = await self.gate_system.filter_memories(
                user_input, context["memories"]
            )

        # Gate wiki content
        if context["wiki"]:
            include_wiki, filtered_wiki = await self.gate_system.filter_wiki_content(
                user_input, context["wiki"]
            )
            context["wiki"] = filtered_wiki if include_wiki else ""

        # Gate semantic chunks
        if context["semantic_chunks"]:
            context["semantic_chunks"] = await self.gate_system.filter_semantic_chunks(
                user_input, context["semantic_chunks"]
            )

        logger.debug(f"[PROMPT] After gating: {len(context['memories'])} memories, "
                    f"{len(context['semantic_chunks'])} chunks")

        return context

    def _manage_token_budget(self, context: Dict) -> Dict:
        """Trim context to fit within token budget"""
        model_name = self.model_manager.get_active_model_name()
        current_tokens = 0

        # Priority order for context (most important first)
        priority_order = [
            ("recent_conversations", 5),
            ("memories", 10),
            ("summaries", 3),
            ("semantic_chunks", 5),
            ("wiki", 1),
            ("dreams", 2)
        ]

        trimmed_context = context.copy()

        for key, max_items in priority_order:
            if key not in context or not context[key]:
                continue

            if isinstance(context[key], list):
                # Handle list items
                items = []
                for item in context[key][:max_items]:
                    item_text = self._extract_text(item)
                    item_tokens = self.get_token_count(item_text, model_name)

                    if current_tokens + item_tokens < self.token_budget:
                        items.append(item)
                        current_tokens += item_tokens
                    else:
                        break

                trimmed_context[key] = items
            else:
                # Handle single text items
                text = str(context[key])
                text_tokens = self.get_token_count(text, model_name)

                if current_tokens + text_tokens < self.token_budget:
                    current_tokens += text_tokens
                else:
                    # Trim to fit
                    remaining_budget = self.token_budget - current_tokens
                    if remaining_budget > 50:  # Only include if meaningful
                        trimmed_context[key] = text[:remaining_budget * 4]  # Rough char estimate
                        current_tokens = self.token_budget
                    else:
                        trimmed_context[key] = ""

        logger.debug(f"[PROMPT] Token budget: {current_tokens}/{self.token_budget}")
        return trimmed_context

    def _assemble_prompt(
        self,
        user_input: str,
        context: Dict,
        system_prompt: str,
        directives_file: str
    ) -> str:
        """Assemble the final prompt from all components"""
        prompt_parts = []

        # System prompt
        if system_prompt:
            prompt_parts.append(f"[SYSTEM]\n{system_prompt}\n")

        # Time context
        if context.get("time_context"):
            prompt_parts.append(f"\n[TIME CONTEXT]\n{context['time_context']}\n")

        # Recent conversations (most important for context)
        if context.get("recent_conversations"):
            prompt_parts.append("\n[RECENT CONVERSATION]")
            for i, conv in enumerate(context["recent_conversations"][-5:]):
                q = conv.get("query", "").strip()
                r = conv.get("response", "").strip()
                if q or r:
                    prompt_parts.append(f"User: {q}\nAssistant: {r}\n")

        # Relevant memories
        if context.get("memories"):
            prompt_parts.append("\n[RELEVANT MEMORIES]")
            for mem in context["memories"]:
                prompt_parts.append(self._format_memory(mem))

        # Summaries
        if context.get("summaries"):
            prompt_parts.append("\n[CONVERSATION SUMMARIES]")
            for summary in context["summaries"]:
                prompt_parts.append(f"{summary}\n")

        # Wiki knowledge
        if context.get("wiki"):
            prompt_parts.append(f"\n[BACKGROUND KNOWLEDGE]\n{context['wiki']}\n")

        # Semantic chunks
        if context.get("semantic_chunks"):
            prompt_parts.append("\n[RELEVANT INFORMATION]")
            for chunk in context["semantic_chunks"]:
                text = chunk.get("text", chunk.get("content", ""))
                prompt_parts.append(f"- {text[:300]}...\n")

        # Dreams (if any)
        if context.get("dreams"):
            prompt_parts.append("\n[DREAMS/REFLECTIONS]")
            for dream in context["dreams"]:
                prompt_parts.append(f"{dream}\n")

        # Load and add directives
        directives = self._load_directives(directives_file)
        if directives:
            prompt_parts.append(f"\n[DIRECTIVES]\n{directives}\n")

        # User input
        prompt_parts.append(f"\n[USER INPUT]\n{user_input}\n")

        return "".join(prompt_parts)

    # === Helper Methods ===

    def get_token_count(self, text: str, model_name: str) -> int:
        """Get token count for text"""
        tokenizer = self.tokenizer_manager.get_tokenizer(model_name)
        if tokenizer is None:
            return len(text) // 4  # Rough estimate
        return len(tokenizer.encode(text, truncation=False))

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Try various keys
            for key in ["content", "text", "response", "filtered_content"]:
                if key in item:
                    return str(item[key])
            # Fallback to string representation
            return str(item)
        else:
            return str(item)

    def _format_memory(self, memory: Dict) -> str:
        """Format a memory for inclusion in prompt"""
        if "query" in memory and "response" in memory:
            return f"Q: {memory['query']}\nA: {memory['response']}\n"
        elif "content" in memory:
            return f"{memory['content']}\n"
        else:
            return f"{str(memory)}\n"

    def _get_time_context(self) -> str:
        """Get current time context"""
        now = datetime.now()
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    async def _get_recent_conversations(self, count: int) -> List[Dict]:
        """Get recent conversations"""
        return self.memory_coordinator.corpus_manager.get_recent_memories(count)

    async def _get_summaries(self, count: int) -> List[str]:
        """Get conversation summaries"""
        summaries = self.memory_coordinator.get_summaries(limit=count)
        return [s.get('content', '') for s in summaries if s.get('content')]

    async def _get_dreams(self, count: int) -> List[str]:
        """Get dreams/reflections"""
        dreams = self.memory_coordinator.get_dreams(limit=count)
        return [d.get('content', '') for d in dreams if d.get('content')]

    async def _get_wiki_content(self, query: str) -> str:
        """Get wiki content for query"""
        topics = self.topic_manager.update_from_user_input(query)
        wiki_content = []

        for topic in topics[:3]:  # Limit to top 3 topics
            snippet = self.wiki_manager.search_summary(topic)
            if snippet and not snippet.startswith("["):
                wiki_content.append(f"{topic}: {snippet}")

        return "\n".join(wiki_content)

    async def _get_semantic_chunks(self, query: str) -> List[Dict]:
        """Get semantic search results"""
        from knowledge.semantic_search import semantic_search
        return semantic_search(query, top_k=10)

    def _load_directives(self, directives_file: str) -> str:
        """Load directives from file"""
        if os.path.exists(directives_file):
            with open(directives_file, 'r') as f:
                return f.read()
        return ""
