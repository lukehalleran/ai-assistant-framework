from utils.logging_utils import get_logger, log_and_time
from typing import List, Dict, Any

logger = get_logger("prompt.py")

# Use the root logger or create a child logger that will inherit handlers
logger.debug("prompt builder.py is alive")

import re
import json
from datetime import datetime, timedelta
import os
from config.config import LOCAL_MODEL_CONTEXT_LIMIT, API_MODEL_CONTEXT_LIMIT
from memory.corpus_manager import CorpusManager
from memory.memory_interface import MemoryNode, MemoryType
from utils.logging_utils import log_and_time, get_logger
from knowledge.WikiManager import WikiManager
from knowledge.topic_manager import TopicManager
from knowledge.semantic_search import semantic_search
from models.model_manager import ModelManager
SEMANTIC_ONLY_MODE = False  # or False to use full system

"""
model_name: str
    Name of the model being used (e.g. "gpt-neo", "gpt-4", etc.)
    Used to determine tokenizer behavior and token limits.
"""
corpus_manager=CorpusManager()
wiki = WikiManager()
topic_manager=TopicManager()
model_mangaer=ModelManager()
class PromptBuilder:
    def __init__(self, tokenizer_manager, model_name,wiki,model_manager,  topic_manager, code_snapshot_path="DAEMONv3.py", changelog_path="daemon_changelog.md"):
        self.tokenizer_manager = tokenizer_manager
        self.model_manager=model_manager
        self.code_snapshot_path = code_snapshot_path
        self.changelog_path = changelog_path
        self.last_prompt = None  # Store the last prompt built
        self.corpus_manager=corpus_manager
        self.wiki=wiki
        self.topic_manager=topic_manager
    def load_directives_by_section(self, directives_file):
        sections = {}
        current_section = None
        if os.path.exists(directives_file):
            with open(directives_file, "r") as f:
                for line in f:
                    line = line.strip().lstrip('\ufeff')
                    if not line:
                        continue

                    if line.startswith("[") and line.endswith("]"):
                        current_section = line.strip("[]")
                        sections[current_section] = []
                    elif current_section:
                        sections[current_section].append(line)
        return sections


    def build_prompt(
        self,
        user_input: str,
        model_name: str,
        memories: list = None,
        summaries: list = None,
        dreams: list = None,
        wiki_snippet: str = None,
        recent_conversations: list = None,
        semantic_snippet: str = None,
        semantic_memory_results: list = None,
        time_context: str = None,
        is_api: bool = False,
        include_dreams: bool = True,
        include_code_snapshot: bool = False,
        include_changelog: bool = False,
        system_prompt: str = "",
        directives_file: str = "structured_directives.txt"
        ):


        logger.debug("🚧 Inside PromptBuilder.build_prompt")
        logger.debug(f"Received {len(memories) if memories else 0} memories")
        for i, m in enumerate(memories):
            logger.debug(f"Memory {i}: keys={list(m.keys())}, query={repr(m.get('query'))}, response={repr(m.get('response'))}")

        MAX_TOKENS = API_MODEL_CONTEXT_LIMIT if is_api else LOCAL_MODEL_CONTEXT_LIMIT
        model_name = self.model_manager.get_active_model_name()
        tokenizer = self.tokenizer_manager.get_tokenizer(model_name)

        recent_conversations = recent_conversations or []

        base_header = system_prompt + "\n\n"
        prompt_parts = ["=== BEGIN CONTEXT ===\n", "[SYSTEM HEADER]\n" + base_header]
        token_count = len(tokenizer.encode(base_header)) if tokenizer else 0

        # ---- TIME CONTEXT ----
        if time_context:
            # Accept both dict and plain-string formats
            if isinstance(time_context, dict):
                # Handle old and new key names gracefully
                current_time   = (
                    time_context.get("current_time")
                    or time_context.get("now")
                )
                since_last     = (
                    time_context.get("elapsed_since_last")
                    or time_context.get("since_last")
                )
                response_time  = (
                    time_context.get("last_response_time")
                    or time_context.get("response_time")
                )

                lines = []
                if current_time:
                    lines.append(f"- Current Time: {current_time}")
                if since_last:
                    lines.append(f"- Elapsed Since Last Query: {since_last}")
                if response_time:
                    lines.append(f"- Last Response Time: {response_time}")

                time_block = "\n[TIME CONTEXT]\n" + "\n".join(lines) + "\n"
            else:
                # Already formatted string
                time_block = f"\n[TIME CONTEXT]\n{time_context}\n"

            prompt_parts.append(time_block)
            if tokenizer:
                token_count += len(tokenizer.encode(time_block))

        directive_sections = self.load_directives_by_section(directives_file)

        core_directives = directive_sections.get("CORE DIRECTIVES", [])
        ethics_directives = directive_sections.get("ETHICS", [])
        personality_directives = directive_sections.get("PERSONALITY CARD", [])
        inject_safety = False
        inject_growth = False
        inject_behavior = True

        if personality_directives:
            personality_block = "\n[PERSONALITY CARD]\n"
            for directive in personality_directives:
                personality_block += f"{directive}\n"
            prompt_parts.append(personality_block)
            if tokenizer:
                token_count += len(tokenizer.encode(personality_block))

        if "danger" in user_input.lower() or "harm" in user_input.lower():
                inject_safety = True

        if user_input.lower().startswith("reflect") or user_input.lower().startswith("analyze"):
                inject_growth = True
        # === WIKI SNIPPET GENERATION ===
        wiki_snippet = ""
        if self.wiki and self.topic_manager:
            topics = self.topic_manager.update_from_user_input(user_input)
            for topic in topics:
                snippet = self.wiki.search_summary(topic)
                if snippet and not snippet.startswith("["):
                    wiki_snippet += f"\n[{topic}]\n{snippet}\n"

        if wiki_snippet and not wiki_snippet.startswith("[Disambiguation Error]") and not wiki_snippet.startswith("[Page Error]") and not wiki_snippet.startswith("[Error"):
            wiki_context = f"\n[RELEVANT BACKGROUND KNOWLEDGE]\n{wiki_snippet}\n"
            prompt_parts.append(wiki_context)
            if tokenizer:
                token_count += len(tokenizer.encode(wiki_context))

        if include_code_snapshot and self.code_snapshot_path and os.path.exists(self.code_snapshot_path):
            with open(self.code_snapshot_path, "r") as f:
                code_snapshot = f.read()
            code_context = f"\n[SOURCE CODE SNAPSHOT]\n{code_snapshot[:3000]}\n"
            prompt_parts.append(code_context)
            if tokenizer:
                token_count += len(tokenizer.encode(code_context))

        if include_changelog and self.changelog_path and os.path.exists(self.changelog_path):
            with open(self.changelog_path, "r") as f:
                changelog = f.read()
            changelog_context = f"\n[RECENT SELF-EDITS / CHANGELOG]\n{changelog[-3000:]}\n"
            prompt_parts.append(changelog_context)
            if tokenizer:
                token_count += len(tokenizer.encode(changelog_context))

       # === SEMANTICALLY RELEVANT CHUNKS (GATED) ===
        semantic_snippet_str = ""
        semantic_results = []
        try:
            semantic_results = semantic_search(user_input)
            semantic_snippet_str = "\n".join(
                f"[{r['title']}]\n{r['text']}" for r in semantic_results
            )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        if semantic_results:
            semantic_context = "\n[SEMANTICALLY RELEVANT WIKI KNOWLEDGE]\n"
            for chunk in semantic_results:
                title = chunk.get("title", "No title")
                text = chunk.get("text", "")
                semantic_context += f"- {title}: {text[:300]}...\n"
            prompt_parts.append(semantic_context)

            if tokenizer:
                token_count += len(tokenizer.encode(semantic_context))
                logger.debug(f"Added {len(semantic_results)} semantic chunks to the prompt")
                for i, chunk in enumerate(semantic_results[:5]):
                    logger.debug(f"[{i}] Title: {chunk.get('title')}, Snippet: {chunk.get('text', '')[:100]}...")


                for i, chunk in enumerate(semantic_snippet):
                    if not isinstance(chunk, dict):
                        logger.warning(f"Chunk {i} is not a dict: {chunk}")
                        continue
                    title = chunk.get("title", "No title")
                    text = chunk.get("text", "")
                    logger.debug(f"    [{i}] Title: {title}, Snippet: {text[:100]}...")

                    if i >= 4:  # Stop after 5 logs
                        break




        # === GATED SEMANTIC MEMORY ===
        if semantic_memory_results and isinstance(semantic_memory_results, dict):
            semantic_memory_context = "\n[SEMANTICALLY RELEVANT MEMORIES]\n"
            seen = set()
            for chunk in semantic_memory_results.get("documents", []):
                content = (
                    chunk if isinstance(chunk, str)
                    else chunk.get("filtered_content") or chunk.get("content") or chunk.get("text")
                )
                if content and content not in seen:
                    seen.add(content)
                    semantic_memory_context += f"- {content[:300]}...\n"
            prompt_parts.append(semantic_memory_context)

            if tokenizer:
                token_count += len(tokenizer.encode(semantic_memory_context))
            logger.debug(f"Added {len(seen)} gated semantic memory chunks to the prompt")

        else:
            logger.debug("No semantic memories passed the gate.")

        if recent_conversations:
            recent_block = "\n[RECENT CONVERSATION SNIPPETS]\nThese represent the latest exchange between the user and Daemon. Use them as the authoritative recent context for answering the current question."
            for i, conv in enumerate(recent_conversations[-5:]):
                query = conv.get("query", "").strip()
                response = conv.get("response", "").strip()
                if query or response:
                    recent_block += f"[{i+1}] User: {query}\nDaemon: {response}\n"
            logger.debug("[PROMPT BUILDER] Included RECENT CONVERSATION block with %d items.", len(recent_conversations))
            prompt_parts.append(recent_block)
            if tokenizer:
                token_count += len(tokenizer.encode(recent_block))

        if core_directives:
            core_block = "\n[CORE DIRECTIVES]\n"
            for directive in core_directives:
                core_block += f"- {directive}\n"
            prompt_parts.append(core_block)
            if tokenizer:
                token_count += len(tokenizer.encode(core_block))

        if ethics_directives:
            ethics_block = "\n[ETHICS]\n"
            for directive in ethics_directives:
                ethics_block += f"- {directive}\n"
            prompt_parts.append(ethics_block)
            if tokenizer:
                token_count += len(tokenizer.encode(ethics_block))

        if inject_behavior:
            interaction_directives = directive_sections.get("INTERACTION BEHAVIOR", [])
            if interaction_directives:
                interaction_block = "\n[INTERACTION BEHAVIOR]\n"
                for directive in interaction_directives:
                    interaction_block += f"- {directive}\n"
                prompt_parts.append(interaction_block)
                if tokenizer:
                    token_count += len(tokenizer.encode(interaction_block))

        if inject_safety:
            safety_directives = directive_sections.get("SAFETY & GUARDRAILS", [])
            if safety_directives:
                safety_block = "\n[SAFETY & GUARDRAILS]\n"
                for directive in safety_directives:
                    safety_block += f"- {directive}\n"
                prompt_parts.append(safety_block)
                if tokenizer:
                    token_count += len(tokenizer.encode(safety_block))

        if inject_growth:
            growth_directives = directive_sections.get("GROWTH & FEEDBACK", [])
            if growth_directives:
                growth_block = "\n[GROWTH & FEEDBACK]\n"
                for directive in growth_directives:
                    growth_block += f"- {directive}\n"
                prompt_parts.append(growth_block)
                if tokenizer:
                    token_count += len(tokenizer.encode(growth_block))

        prompt_parts.append("\n[USER INPUT]\n")
        user_msg = f"{user_input.strip()}\n"
        prompt_parts.append(user_msg)

        if tokenizer:
            if token_count + len(tokenizer.encode(user_msg)) > MAX_TOKENS:
                trimmed_user_msg = tokenizer.decode(tokenizer.encode(user_msg)[-MAX_TOKENS + token_count:])
                prompt_parts[-1] = trimmed_user_msg.strip() + "\n"

        prompt_parts.append("\n=== END CONTEXT ===\n")

        full_prompt = "".join(prompt_parts)
        self.last_prompt = full_prompt
        logger.debug(f"Prompt is {len(full_prompt)} chars and {token_count} tokens")
        logger.debug("=== FINAL PROMPT BUILD COMPLETE ===")
        logger.debug(f"Prompt length: {len(full_prompt)} characters")
        logger.debug(f"Token count: {token_count}")
        logger.debug(f"Semantic Snippet sample: {semantic_snippet[:2]}")
        logger.debug(f"Semantic Memory Results sample: {semantic_memory_results.get('documents', [])[:2] if isinstance(semantic_memory_results, dict) else 'Not a dict'}")


        return full_prompt


    def show_prompt(self):
        if self.last_prompt:
            print("\n===== CURRENT FULL PROMPT =====\n")
            print(self.last_prompt)
            print("\n===== END OF PROMPT =====\n")
        else:
            print("[PromptBuilder] No prompt built yet.")
