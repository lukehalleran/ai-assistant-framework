import os
from config import LOCAL_MODEL_CONTEXT_LIMIT, API_MODEL_CONTEXT_LIMIT

SEMANTIC_ONLY_MODE = True  # or False to use full system

"""
model_name: str
    Name of the model being used (e.g. "gpt-neo", "gpt-4", etc.)
    Used to determine tokenizer behavior and token limits.
"""

class PromptBuilder:
    def __init__(self, tokenizer_manager, code_snapshot_path="DAEMONv3.py", changelog_path="daemon_changelog.md"):
        self.tokenizer_manager = tokenizer_manager
        self.code_snapshot_path = code_snapshot_path
        self.changelog_path = changelog_path
        self.last_prompt = None  # Store the last prompt built

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

    def build_prompt(self, model_name, user_input, memories, summaries, dreams, wiki_snippet="", semantic_snippet=None,semantic_memory_results=None, time_context=None, is_api=False, include_dreams=True, include_code_snapshot=False, include_changelog=False, system_prompt="", directives_file="structured_directives.txt"):

        MAX_TOKENS = API_MODEL_CONTEXT_LIMIT if is_api else LOCAL_MODEL_CONTEXT_LIMIT

        tokenizer = self.tokenizer_manager.get_tokenizer(model_name)

        base_header = system_prompt + "\n\n"

        prompt_parts = ["=== BEGIN CONTEXT ===\n", "[SYSTEM HEADER]\n" + base_header]
        token_count = len(tokenizer.encode(base_header)) if tokenizer else 0

        if time_context:
            time_block = "\n[TIME CONTEXT]\n"
            time_block += f"- Current Time: {time_context['current_time']}\n"
            time_block += f"- Elapsed Since Last Query: {time_context['since_last']}\n"
            time_block += f"- Last Response Time: {time_context['response_time']}\n"
            prompt_parts.append(time_block)
            if tokenizer:
                token_count += len(tokenizer.encode(time_block))

        directive_sections = self.load_directives_by_section(directives_file)


        core_directives = directive_sections.get("CORE DIRECTIVES", [])
        ethics_directives = directive_sections.get("ETHICS", [])
        personality_directives = directive_sections.get("PERSONALITY CARD", [])
        inject_safety  = False
        inject_growth = False
        inject_behavior = True  # maybe always on

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

        if memories:
            prompt_parts.append("\n[SHORT-TERM MEMORIES]\n")
            for mem in reversed(memories):
                mem_text = f"[{mem['timestamp']}] User: {mem.get('query', '')}\nDaemon: {mem.get('response', '')}\n"
                prompt_parts.append(mem_text)
                if tokenizer:
                    token_count += len(tokenizer.encode(mem_text))

        if summaries:
            prompt_parts.append("\n[LONG-TERM SUMMARIES]\n")
            for s in summaries[-1:]:
                sum_text = f"Summary:\n{s['response']}\n"
                prompt_parts.append(sum_text)
                if tokenizer:
                    token_count += len(tokenizer.encode(sum_text))

        if include_dreams and dreams:
            prompt_parts.append("\n[DREAM FRAGMENTS]\n")
            for d in dreams[-1:]:
                dream_text = f"Dream Fragment:\n{d['dream']}\n"
                prompt_parts.append(dream_text)
                if tokenizer:
                    token_count += len(tokenizer.encode(dream_text))

        if semantic_snippet:
            semantic_context = "\n[SEMANTICALLY RELEVANT CONTEXT]\n"
            for chunk in semantic_snippet:
                semantic_context += f"- {chunk['title']}: {chunk['text'][:300]}...\n"
            prompt_parts.append(semantic_context)
            if tokenizer:
                token_count += len(tokenizer.encode(semantic_context))
                print(f"[DEBUG] Added {len(semantic_snippet)} semantic chunks to the prompt")
                for i, chunk in enumerate(semantic_snippet[:5]):  # show up to 5 for preview
                    print(f"    [{i}] Title: {chunk['title']}, Snippet: {chunk['text'][:100]}...")

                # Add semantically relevant memories
        if semantic_memory_results and semantic_memory_results['documents'][0]:
            semantic_memory_context = "\n[SEMANTICALLY RELEVANT MEMORIES]\n"
            for i, doc in enumerate(semantic_memory_results['documents'][0]):
                semantic_memory_context += f"- Memory {i+1}: {doc[:300]}...\n"
            prompt_parts.append(semantic_memory_context)
            print(f"[DEBUG] Added {len(semantic_memory_results['documents'][0])} semantic memory chunks to the prompt")


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
        self.last_prompt = full_prompt  # Save the latest built prompt
        return full_prompt

    def show_prompt(self):
        if self.last_prompt:
            print("\n===== CURRENT FULL PROMPT =====\n")
            print(self.last_prompt)
            print("\n===== END OF PROMPT =====\n")
        else:
            print("[PromptBuilder] No prompt built yet.")
