import gradio as gr
import docx2txt
import pandas as pd
import time
from datetime import datetime
import  uuid

# Import from your modules
from tokenizer_manager import TokenizerManager
from prompt_builder import PromptBuilder
from memory import load_corpus, add_to_chroma, add_to_corpus, huggingface_auto_tag
from chromadb import PersistentClient
from topic_manager import TopicManager
from WikiManager import WikiManager
from time_manager import TimeManager
from models import run_model, model_manager
from search_faiss_with_metadata import semantic_search
from config import CONFIDENCE_THRESHOLD, DEBUG_MODE, SEMANTIC_ONLY_MODE
from personality_manager import PersonalityManager

# Constants



# Initialize managers (same as runtime.py)
tokenizer_manager = TokenizerManager()
prompt_builder = PromptBuilder(tokenizer_manager)
topic_manager = TopicManager()
wiki_manager = WikiManager()
time_manager = TimeManager()
personality_manager = PersonalityManager()

# Setup models (skip local model for faster startup)
# model_manager.load_model('gpt-neo', 'EleutherAI/gpt-neo-1.3B')  # Comment out for speed
model_manager.load_openai_model('gpt-4-turbo', 'gpt-4-turbo')
model_manager.switch_model('gpt-4-turbo')

# Load memory
corpus = load_corpus()
client = PersistentClient(path="chroma_db")


try:
    collection = client.get_or_create_collection("assistant-memory")
except KeyError as e:
    print("[ERROR] Collection metadata corrupted, resetting collection.")
    import shutil
    shutil.rmtree("/run/media/lukeh/T9/chroma_db")
    client = PersistentClient(path="/run/media/lukeh/T9/chroma_db")
    collection = client.get_or_create_collection("assistant-memory")

def get_relevant_context(user_input):
    """Run semantic search to retrieve relevant chunks for user input."""
    topic_nouns = topic_manager.extract_nouns(user_input)
    if not topic_nouns:
        return [], [], ""  # Return 3 values, not 1

    topic_query = " ".join(topic_nouns[:3])
    search_results = semantic_search(topic_query)

    # Filter out low-confidence matches
    filtered_results = [
        r for r in search_results if r['score'] < CONFIDENCE_THRESHOLD
    ]
    return filtered_results, topic_nouns, topic_query

def build_daemon_prompt(user_input):
    debug_info = {
    'topics_detected': [],
    'wiki_content': '',
    'semantic_results': {},
    'semantic_memory_results': {}, # empty dict — safe placeholder
    'memory_chunks': 0,
    'summary_chunks': 0
}

    """Build prompt using same logic as runtime.py"""
    personality_config = personality_manager.get_current_config()

    if personality_config.get("include_semantic_search", True):
        relevant_chunks, topic_nouns, topic_query = get_relevant_context(user_input)
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
    topic_manager.update_from_user_input(user_input)
    debug_info['topics_detected'] = list(topic_manager.top_topics)[:5]  # Show top 5

    # Wiki lookup if relevant
    wiki_snippet = ""
    wiki_topic_used = None
    if personality_config["include_wiki"]:
        for topic in list(topic_manager.top_topics):  # Convert set to list
            if topic.lower() in user_input.lower():
                try:
                    wiki_snippet = wiki_manager.search_summary(topic, sentences=5)  # Get MORE sentences
                    wiki_topic_used = topic
                    debug_info['wiki_content'] = f"Topic: {topic}\nContent: {wiki_snippet[:200]}..."

                    # Check if we should get more detailed info
                    if wiki_manager.should_fallback(wiki_snippet, user_input):
                        full_article = wiki_manager.fetch_full_article(topic)
                        if "[Error" not in full_article and "[Disambiguation" not in full_article:
                            wiki_snippet = full_article[:1500] + "..."  # Get MORE content
                            debug_info['wiki_content'] += f"\n[Fallback to full article - {len(full_article)} chars]"
                except Exception as e:
                    debug_info['wiki_content'] = f"Error: {str(e)}"
                break

    # Memories and summaries
    results = collection.query(query_texts=[user_input], n_results=30, include=["documents", "metadatas"])
    recents = [i for i in corpus if "@summary" not in i.get("tags", [])][-personality_config["num_memories"]:]
    summaries = [i for i in corpus if "@summary" in i.get("tags", [])][-1:]

    debug_info['memory_chunks'] = len(recents)
    debug_info['summary_chunks'] = len(summaries)

    debug_info['semantic_memory_results'] = {
        'results_found': len(results['documents'][0]),
        'top_results': [doc[:100] + "..." for doc in results['documents'][0][:3]] if results['documents'][0] else []
    }

    # Time context
    time_context = {
        "current_time": time_manager.get_current_datetime(),
        "since_last": time_manager.get_elapsed_since_last(),
        "response_time":time_manager.get_last_response_time()
    }

    # Build prompt
    if SEMANTIC_ONLY_MODE:
        prompt = prompt_builder.build_prompt(
            model_name=model_manager.get_active_model_name(),
            user_input=user_input,
            memories=[],
            summaries=[],
            dreams=[],
            wiki_snippet="",
            semantic_snippet=relevant_chunks,
            time_context=time_context,
            is_api=model_manager.is_api_model(model_manager.get_active_model_name()),
            include_dreams=False,
            include_code_snapshot=False,
            include_changelog=False,
             system_prompt=system_prompt,
    directives_file=personality_config["directives_file"]
        )
    else:
        prompt = prompt_builder.build_prompt(
            model_name=model_manager.get_active_model_name(),
            user_input=user_input,
            memories=recents,
            summaries=summaries,
            dreams=[],
            wiki_snippet=wiki_snippet,
            semantic_snippet=relevant_chunks,
            time_context=time_context,
            is_api=model_manager.is_api_model(model_manager.get_active_model_name()),
            include_dreams=False,
            include_code_snapshot=False,
            include_changelog=False,
             system_prompt=system_prompt,
    directives_file=personality_config["directives_file"]
        )

    return prompt, debug_info

# Function to handle UI interaction
def handle_submit(user_text, files):
    combined_text = user_text

    # Process each uploaded file
    if files is not None:
        for file in files:
            try:
                if file.name.endswith(".txt"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        combined_text += "\n\n" + f.read()
                elif file.name.endswith(".docx"):
                    combined_text += "\n\n" + docx2txt.process(file.name)
                elif file.name.endswith(".csv"):
                    df = pd.read_csv(file.name)
                    combined_text += "\n\n" + df.to_string()
                elif file.name.endswith(".py"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        combined_text += "\n\n" + f.read()
                else:
                    combined_text += f"\n\n[Unsupported file type: {file.name}]"
            except Exception as e:
                combined_text += f"\n\n[Error reading {file.name}: {str(e)}]"

    # Build prompt using same logic as REPL
    prompt, debug_info = build_daemon_prompt(combined_text)

    # Save full prompt to file for debugging
    with open(f"debug_prompt_{int(time.time())}.txt", "w") as f:
        f.write("=== FULL PROMPT DEBUG ===\n\n")
        f.write(prompt)
        f.write(f"\n\n=== PROMPT STATS ===\n")
        f.write(f"Total length: {len(prompt)} characters\n")
        f.write(f"Estimated tokens: ~{len(prompt)//4}\n")
        f.write(f"Debug info: {debug_info}\n")

    # Format debug info for display
    debug_display = ""
    if DEBUG_MODE:
        debug_display = f""" DEBUG INFO:

 Topics Detected: {', '.join(debug_info['topics_detected']) if debug_info['topics_detected'] else 'None'}

Wikipedia: {debug_info['wiki_content'] if debug_info['wiki_content'] else 'No wiki content retrieved'}

 Semantic Search:
  - Query: {debug_info['semantic_results'].get('search_query', '')}
  - Results: {debug_info['semantic_results'].get('results_found', 0)} chunks found
  - Top matches: {debug_info['semantic_results'].get('top_results', [])[:2] if debug_info['semantic_results'].get('top_results') else 'None'}

 Memory: {debug_info['memory_chunks']} recent conversations, {debug_info['summary_chunks']} summaries

  semanitc memories- Results: {debug_info['semantic_memory_results']['results_found']} memories found
  - Top matches: {debug_info['semantic_memory_results']['top_results'][:2] if debug_info['semantic_memory_results']['top_results'] else 'None'}

Prompt length: ~{len(prompt)} characters

Full Prompt:
{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}
"""


    # Run model
    response_start = datetime.now()
    response = run_model(prompt, model_name=model_manager.get_active_model_name())
    response_end = datetime.now()
    # Calculate the response time
    response_time = time_manager.measure_response_time(response_start, response_end)
    #  save it
    time_manager.save_response_time(response_time)

    # Update time manager
    time_manager.mark_query_time()

    # Update memory store (same as REPL)
    uid = str(uuid.uuid4())
    tags = huggingface_auto_tag(f"User: {user_text}\nAssistant: {response}", model_manager)
    add_to_chroma(f"User: {user_text}\nAssistant: {response}", uid, tags, collection, entry_type="memory")
    add_to_corpus(corpus, user_text, response, tags)

    return user_text, response, debug_display

# Build Gradio UI with Debug Panel
with gr.Blocks() as demo:
    # Dropdown to select personality
    def switch_personality_ui(selected):
        personality_manager.switch_personality(selected)
        return f"Switched to {selected} mode."

    gr.Markdown("### Personality Mode")
    personality_dropdown = gr.Dropdown(
        label="Choose Personality",
        choices=list(personality_manager.personalities.keys()),
        value="default"
    )
    personality_status = gr.Textbox(
        label="Personality Status",
        value="Using default mode.",
        interactive=False
    )

    personality_dropdown.change(
        switch_personality_ui,
        inputs=[personality_dropdown],
        outputs=[personality_status]
    )
    gr.Markdown("## Daemon Demo UI — Mobile Friendly")

    # Large Daemon Response box — at top, scrollable
    agent_display = gr.Textbox(
        label="Daemon Response",
        lines=15,
        interactive=False,
        max_lines=1000,
        show_copy_button=True
    )

    # Debug panel (collapsible)
    with gr.Accordion("🔍 Debug Info", open=False):
        debug_display = gr.Textbox(
            label="What's happening under the hood",
            lines=10,
            interactive=False,
            show_copy_button=True
        )

    # Small "You Typed" display — below response
    user_display = gr.Textbox(
        label="You Typed",
        lines=3,
        interactive=False
    )

    # User Input + Upload + Submit — compact row at bottom
    with gr.Row():
        user_input = gr.Textbox(
            lines=3,
            label="Type your message here"
        )
        file_input = gr.File(
            file_types=[".txt", ".docx", ".csv", ".py"],
            file_count="multiple",
            label="Upload files (max 50 files, 100MB each)"
        )
        submit_button = gr.Button("Submit")

    # Submit logic
    submit_button.click(
        handle_submit,
        inputs=[user_input, file_input],
        outputs=[user_display, agent_display, debug_display]
    )

# Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        max_file_size="100mb",
        max_threads=10
    )
