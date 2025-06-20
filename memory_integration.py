# memory_integration.py - Integration code for hierarchical memory system

import asyncio
from hierarchical_memory import HierarchicalMemorySystem, HierarchicalGatedPromptBuilder
from llm_gates import GatedPromptBuilder

def integrate_hierarchical_memory(gui_module):
    """
    Integrate hierarchical memory into gui.py with minimal changes
    """

    # Import necessary items from gui.py globals
    model_manager = gui_module.model_manager
    prompt_builder = gui_module.prompt_builder

    # Initialize hierarchical memory system
    hierarchical_memory = HierarchicalMemorySystem(model_manager)

    # Create the hierarchical gated prompt builder
    hierarchical_builder = HierarchicalGatedPromptBuilder(prompt_builder, model_manager)

    # Replace the build_daemon_prompt function
    original_build_daemon_prompt = gui_module.build_daemon_prompt

    def new_build_daemon_prompt(user_input):
        """Enhanced prompt builder with hierarchical memory"""

        # Get original debug info structure
        debug_info = {
            'topics_detected': [],
            'wiki_content': '',
            'semantic_results': {},
            'semantic_memory_results': {},
            'memory_chunks': 0,
            'summary_chunks': 0,
            'hierarchical_memories': {}  # NEW
        }

        # Get personality config
        personality_config = gui_module.personality_manager.get_current_config()

        # Run async memory retrieval in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get hierarchical memories
            hierarchical_memories = loop.run_until_complete(
                hierarchical_memory.retrieve_relevant_memories(user_input, max_memories=15)
            )

            debug_info['hierarchical_memories'] = {
                'retrieved': len(hierarchical_memories),
                'types': {},
                'top_relevance': []
            }

            # Count memory types
            for mem in hierarchical_memories:
                mem_type = mem['memory'].type.value
                debug_info['hierarchical_memories']['types'][mem_type] = \
                    debug_info['hierarchical_memories']['types'].get(mem_type, 0) + 1

            # Show top 3 most relevant
            for mem in hierarchical_memories[:3]:
                debug_info['hierarchical_memories']['top_relevance'].append({
                    'type': mem['memory'].type.value,
                    'score': f"{mem['final_score']:.3f}",
                    'preview': mem['memory'].content[:100] + '...'
                })

            # Get other context (wiki, semantic chunks, etc.)
            if personality_config.get("include_semantic_search", True):
                relevant_chunks, topic_nouns, topic_query = gui_module.get_relevant_context(user_input)
                debug_info['semantic_results'] = {
                    'query_nouns': topic_nouns,
                    'search_query': topic_query,
                    'results_found': len(relevant_chunks),
                    'top_results': [r['text'][:100] + "..." for r in relevant_chunks[:3]] if relevant_chunks else []
                }
            else:
                relevant_chunks, topic_nouns, topic_query = [], [], ""

            # Load system prompt
            with open(personality_config["system_prompt_file"], "r") as f:
                system_prompt = f.read()

            # Update topic manager
            gui_module.topic_manager.update_from_user_input(user_input)
            debug_info['topics_detected'] = list(gui_module.topic_manager.top_topics)[:5]

            # Wiki lookup
            wiki_snippet = ""
            if personality_config["include_wiki"]:
                for topic in list(gui_module.topic_manager.top_topics):
                    if topic.lower() in user_input.lower():
                        try:
                            wiki_snippet = gui_module.wiki_manager.search_summary(topic, sentences=5)
                            debug_info['wiki_content'] = f"Topic: {topic}\nContent: {wiki_snippet[:200]}..."

                            if gui_module.wiki_manager.should_fallback(wiki_snippet, user_input):
                                full_article = gui_module.wiki_manager.fetch_full_article(topic)
                                if "[Error" not in full_article and "[Disambiguation" not in full_article:
                                    wiki_snippet = full_article[:1500] + "..."
                                    debug_info['wiki_content'] += f"\n[Fallback to full article - {len(full_article)} chars]"
                        except Exception as e:
                            debug_info['wiki_content'] = f"Error: {str(e)}"
                        break

            # Time context
            time_context = {
                "current_time": gui_module.time_manager.get_current_datetime(),
                "since_last": gui_module.time_manager.get_elapsed_since_last(),
                "response_time": gui_module.time_manager.get_last_response_time()
            }

            # Build hierarchical prompt with all context
            context_sources = {
                "wiki_snippet": wiki_snippet,
                "semantic_chunks": relevant_chunks,
                "time_context": time_context,
                "summaries": []  # Can still use if needed
            }

            # Use hierarchical builder
            # Use hierarchical builder
            prompt = loop.run_until_complete(
                hierarchical_builder.build_hierarchical_prompt(
                    user_text=user_input,
                    tokenizer=gui_module.tokenizer_manager,
                    max_results=5
                )
            )


            return prompt, debug_info

        finally:
            loop.close()

    # Replace the function
    gui_module.build_daemon_prompt = new_build_daemon_prompt

    # Also update handle_submit to store interactions in hierarchical memory
    original_handle_submit = gui_module.handle_submit

    def new_handle_submit(user_text, files):
        """Enhanced submit handler with hierarchical memory storage"""
        if "top 3 topics" in user_text.lower() or "number one topic" in user_text.lower():
            print("[Memory Skipped] Summary-type prompt.")
            return user_display, response, debug_display

        # Call original handler
        user_display, response, debug_display = original_handle_submit(user_text, files)

        # Store in hierarchical memory
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Extract tags using existing tagger
            tags = gui_module.huggingface_auto_tag(
                f"User: {user_text}\nAssistant: {response}",
                gui_module.model_manager
            )

            # Store interaction
            memory_id = loop.run_until_complete(
                hierarchical_memory.store_interaction(user_text, response, tags)
            )

            print(f"[Hierarchical Memory] Stored interaction with ID: {memory_id}")

        finally:
            loop.close()

        return user_display, response, debug_display

    # Replace the function
    gui_module.handle_submit = new_handle_submit

    return hierarchical_memory, hierarchical_builder


# Standalone integration script
if __name__ == "__main__":
    """
    Run this to integrate hierarchical memory into your existing system
    """
    print("Integrating hierarchical memory system...")

    # First ensure llm_gates.py is in place
    import os
    if not os.path.exists("llm_gates.py"):
        print("ERROR: llm_gates.py not found. Please save the LLM gates code from the email first.")
        exit(1)

    # Check if hierarchical_memory.py exists
    if not os.path.exists("hierarchical_memory.py"):
        print("ERROR: hierarchical_memory.py not found. Please save the hierarchical memory code first.")
        exit(1)

    print("✓ All required files found")
    print("\nTo integrate into your GUI:")
    print("1. Add these imports to gui.py:")
    print("   from memory_integration import integrate_hierarchical_memory")
    print("   import asyncio")
    print("\n2. After all your manager initializations, add:")
    print("   hierarchical_memory, hierarchical_builder = integrate_hierarchical_memory(sys.modules[__name__])")
    print("\n3. The system will automatically use hierarchical memory for all interactions!")

    # Optionally patch gui.py automatically
    response = input("\nWould you like to automatically patch gui.py? (y/n): ")
    if response.lower() == 'y':
        with open("gui.py", "r") as f:
            gui_content = f.read()

        # Add imports after existing imports
        import_line = "from memory_integration import integrate_hierarchical_memory\nimport sys"
        if "from memory_integration import" not in gui_content:
            # Find a good place to insert (after other imports)
            import_pos = gui_content.find("# Initialize managers")
            if import_pos > 0:
                gui_content = gui_content[:import_pos] + import_line + "\n\n" + gui_content[import_pos:]

        # Add integration call
        integration_line = "\n# Integrate hierarchical memory\nhierarchical_memory, hierarchical_builder = integrate_hierarchical_memory(sys.modules[__name__])\n"
        if "integrate_hierarchical_memory" not in gui_content:
            # Add after manager initializations
            init_pos = gui_content.find("# Load memory")
            if init_pos > 0:
                gui_content = gui_content[:init_pos] + integration_line + "\n" + gui_content[init_pos:]

        # Save patched file
        with open("gui_hierarchical.py", "w") as f:
            f.write(gui_content)

        print("\n✓ Created gui_hierarchical.py with integrated memory system!")
        print("  Test it first, then rename to gui.py when ready.")
