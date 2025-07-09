# memory_integration.py - Integration code for hierarchical memory system
import logging
import asyncio
from datetime import datetime
import uuid

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("memory_integration.py is alive")

from hierarchical_memory import HierarchicalMemorySystem
from unified_hierarchical_prompt_builder import UnifiedHierarchicalPromptBuilder
from config import GATE_REL_THRESHOLD
from llm_gate_module import MultiStageGateSystem
from persistence import add_to_chroma_batch

def integrate_hierarchical_memory(
    model_manager,
    prompt_builder,
    personality_manager,
    topic_manager,
    time_manager,
    wiki_manager,
    huggingface_auto_tag,
    get_relevant_context,
    build_daemon_prompt_setter,
    handle_submit_setter,
    embed_model,
    collection,
    corpus,
    add_to_chroma_batch,
    add_to_corpus
):
    """
    Integrates hierarchical memory system with the existing components.
    Now includes cosine_threshold parameter for HierarchicalMemorySystem.
    """
    # Create hierarchical memory system with cosine threshold
    hierarchical_memory = HierarchicalMemorySystem(
        model_manager=model_manager,
        embed_model=embed_model,
        cosine_threshold=GATE_REL_THRESHOLD  # Add this parameter
    )

    # Create unified hierarchical prompt builder
    hierarchical_builder = UnifiedHierarchicalPromptBuilder(
        prompt_builder=prompt_builder,
        model_manager=model_manager
    )

    # Set the same memory system instance in the builder
    hierarchical_builder.memory_system = hierarchical_memory

    # Define the async build_daemon_prompt function
    async def new_build_daemon_prompt_async(user_input):
        """Enhanced prompt builder with hierarchical memory"""

        debug_info = {
            'topics_detected': [],
            'wiki_content': '',
            'semantic_results': {},
            'semantic_memory_results': {},
            'memory_chunks': 0,
            'summary_chunks': 0,
            'hierarchical_memories': {}
        }

        try:
            # Get personality config
            personality_config = personality_manager.get_current_config()

            # Get hierarchical memories
            hierarchical_memories = await hierarchical_memory.retrieve_relevant_memories(user_input, max_memories=15)

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

            # Get semantic context
            if personality_config.get("include_semantic_search", True):
                relevant_chunks, topic_nouns, topic_query = get_relevant_context(user_input)
                debug_info['semantic_results'] = {
                    'query_nouns': topic_nouns,
                    'search_query': topic_query,
                    'results_found': len(relevant_chunks),
                    'top_results': [r['text'][:100] + "..." for r in relevant_chunks[:3]] if relevant_chunks else []
                }
            else:
                relevant_chunks = []

            # Load system prompt
            with open(personality_config["system_prompt_file"], "r") as f:
                system_prompt = f.read()

            # Update topic manager
            topic_manager.update_from_user_input(user_input)
            debug_info['topics_detected'] = list(topic_manager.top_topics)[:5]

            # Wiki lookup
            wiki_snippet = ""
            if personality_config.get("include_wiki", True):
                for topic in list(topic_manager.top_topics):
                    if topic.lower() in user_input.lower():
                        try:
                            wiki_snippet = wiki_manager.search_summary(topic, sentences=5)
                            debug_info['wiki_content'] = f"Topic: {topic}\nContent: {wiki_snippet[:200]}..."

                            if wiki_manager.should_fallback(wiki_snippet, user_input):
                                full_article = wiki_manager.fetch_full_article(topic)
                                if "[Error" not in full_article and "[Disambiguation" not in full_article:
                                    wiki_snippet = full_article[:1500] + "..."
                                    debug_info['wiki_content'] += f"\n[Fallback to full article - {len(full_article)} chars]"
                        except Exception as e:
                            debug_info['wiki_content'] = f"Error: {str(e)}"
                        break

            # Get ChromaDB semantic memories
            results = collection.query(
                query_texts=[user_input],
                n_results=30,
                include=["documents", "metadatas"]
            )

            # Format all memories
            formatted_memories = []
            for mem_dict in hierarchical_memories:
                memory = mem_dict['memory']
                content_parts = memory.content.split('\nAssistant: ')
                if len(content_parts) == 2:
                    query = content_parts[0].replace("User: ", "").strip()
                    response = content_parts[1].strip()
                else:
                    query = memory.content[:100]
                    response = "[Could not parse response]"

                formatted_memories.append({
                    'query': query,
                    'response': response,
                    'timestamp': memory.timestamp
                })

            # Time context
            time_context = {
                "current_time": time_manager.get_current_datetime(),
                "since_last": time_manager.get_elapsed_since_last(),
                "response_time": time_manager.get_last_response_time()
            }

            # Gate memories - now using cosine similarity
            memory_gater = MultiStageGateSystem(model_manager, cosine_threshold=COSINE_SIMILARITY_THRESHOLD)
            raw_chunks = [
                {
                    "content": f"User: {m['query']}\nAssistant: {m['response']}",
                    "metadata": {"timestamp": m.get("timestamp", datetime.now())}
                }
                for m in formatted_memories
            ]
            gated_memory_chunks = await memory_gater.filter_memories(user_input, raw_chunks)

            # Build hierarchical prompt
            prompt = await hierarchical_builder.build_hierarchical_prompt(
                user_input=user_input,
                include_dreams=True,
                include_wiki=True,
                include_semantic=True,
                semantic_chunks=[],
                semantic_memory_results=gated_memory_chunks,
                wiki_content=wiki_snippet,
                system_prompt=system_prompt,
                directives_file=personality_config["directives_file"],
                time_context=time_context
            )

            return prompt, debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"[PROMPT ERROR] {e}")
            raise

    # Set the enhanced build function
    build_daemon_prompt_setter(new_build_daemon_prompt_async)

    # Create enhanced handle_submit
    def enhance_handle_submit(original_func):
        """Wrapper to enhance the original handle_submit with hierarchical memory storage"""
        async def enhanced_handle_submit(user_text, files, history):
            # Call original
            async for result in original_func(user_text, files, history):
                history, debug_display, status = result

                # After getting the response, store in hierarchical memory
                if history and len(history) > 0 and history[-1][1] != "Thinking...":
                    response = history[-1][1]

                    # Extract tags
                    tags = huggingface_auto_tag(f"User: {user_text}\nAssistant: {response}", model_manager)

                    # Store interaction in hierarchical memory
                    memory_id = await hierarchical_memory.store_interaction(user_text, response, tags)
                    logger.debug(f"[Hierarchical Memory] Stored interaction with ID: {memory_id}")

                yield result

        return enhanced_handle_submit

    # Store the enhancement function for later use
    hierarchical_memory._enhance_handle_submit = enhance_handle_submit

    # Return the instances
    return hierarchical_memory, hierarchical_builder
