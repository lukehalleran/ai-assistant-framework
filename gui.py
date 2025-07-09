import logging
logging.basicConfig(level=logging.DEBUG)
from logging_utils import log_and_time, setup_logging, start_logging_monitor, GradioLogger, with_logging, log_async_operation
logger = setup_logging()
start_logging_monitor()
is_processing = False
# NOW import everything else including gradio
import gradio as gr
import docx2txt
import pandas as pd
import time
import uuid
import json
from queue import Queue, Empty
import httpx
import openai
import os, random
from datetime import datetime
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
import signal
import torch
import re
import gc
from typing import List, Dict
##choose random port
os.environ.setdefault("GRADIO_SERVER_PORT", str(random.randint(7800, 7999)))

# Your module imports
from tokenizer_manager import TokenizerManager
from prompt_builder import PromptBuilder
from persistence import add_to_chroma, add_to_chroma_batch
from memory import load_corpus, add_to_corpus, huggingface_auto_tag, save_corpus
from chromadb import PersistentClient
from topic_manager import TopicManager
from WikiManager import WikiManager
from time_manager import TimeManager
from models import run_model, model_manager
from search_faiss_with_metadata import semantic_search
from config import CONFIDENCE_THRESHOLD, DEBUG_MODE, SEMANTIC_ONLY_MODE, MAX_WORKING_MEMORY,GATE_REL_THRESHOLD
from personality_manager import PersonalityManager
from llm_gate_module import MultiStageGateSystem
from unified_hierarchical_prompt_builder import UnifiedHierarchicalPromptBuilder
from hierarchical_memory import get_cached_tokenizer
from shared_memory import hierarchical_memory, hierarchical_builder
from search_faiss_with_metadata import semantic_search
# Create a persistent logger instance
gradio_logger = GradioLogger()
gradio_logger.debug("GradioLogger initialized successfully")

if torch.cuda.is_available():
    # Check if GPU has enough free memory
    free_memory = torch.cuda.mem_get_info()[0] / 1e9  # in GB
    logger.debug(f"Free GPU memory at startup: {free_memory:.2f} GB")

    if free_memory < 2.0:  # Less than 2GB free
        logger.warning("Low GPU memory! Attempting cleanup...")
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_gpu():
    """Clean up GPU memory on exit"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU memory cleaned up")



# Replace the initialization section in gui.py with this corrected version:

# Initialize managers
tokenizer_manager = TokenizerManager()
prompt_builder = PromptBuilder(tokenizer_manager)
topic_manager = TopicManager()
wiki_manager = WikiManager()
time_manager = TimeManager()
personality_manager = PersonalityManager()

# === MODEL LOAD BLOCK ===
try:
    print("=== ENTERING MODEL LOAD BLOCK ===")
    gradio_logger.debug("=== ENTERING MODEL LOAD BLOCK ===")

    # First, always register and load GPT-4-Turbo as the primary model
    model_manager.load_openai_model("gpt-4-turbo", "gpt-4-turbo")
    model_manager.switch_model("gpt-4-turbo")
    gradio_logger.debug("[GUI] GPT-4-Turbo loaded and set as active model")

    # Then try to load local phi-2 for gating only (if available)
    try:
        gradio_logger.debug("[GUI] Attempting to load local model: phi-2 (for tagging/gating)...")
        model_manager.load_model("phi-2", "./models/phi-2")
        gradio_logger.debug(f"[GUI] Successfully loaded local model 'phi-2' for gating")
    except Exception as e:
        gradio_logger.error(f"[GUI] Failed to load local model 'phi-2': {e}")
        gradio_logger.debug("[GUI] Will use GPT-3.5-Turbo for tagging/gating when needed")
        # Register GPT-3.5 for gating tasks only
        model_manager.load_openai_model("gpt-3.5-turbo", "gpt-3.5-turbo")

    # Ensure GPT-4-Turbo remains the active model
    model_manager.switch_model("gpt-4-turbo")

except Exception as e:
    gradio_logger.error(f"[GUI] Critical error in model loading: {e}")
    raise RuntimeError("Failed to load required models")

print("=== MODELS AFTER INIT ===", model_manager.models.keys())
print("=== API MODELS ===", model_manager.api_models.keys())
print("=== ACTIVE MODEL ===", model_manager.active_model_name)
gradio_logger.debug(f"[GUI] Models after init: {model_manager.models.keys()}")
gradio_logger.debug(f"[GUI] API models: {model_manager.api_models.keys()}")
gradio_logger.debug(f"[GUI] Active model: {model_manager.active_model_name}")


import json
from datetime import datetime
import logging
from config import CORPUS_FILE  # <-- make sure you import this correctly

def load_corpus():
    with open(CORPUS_FILE, 'r') as f:
        corpus = json.load(f)
    latest = sorted(corpus, key=lambda x: x.get('timestamp', datetime.min.isoformat()), reverse=True)[:3]
    logging.debug(f"LOADED CORPUS LATEST ENTRIES: {latest}")
    return corpus

def force_test_memory():
    corpus = load_corpus()
    test_entry = {
        "content": "TEST MEMORY ENTRY",
        "timestamp": datetime.now().isoformat(),
        "tags": []
    }
    corpus.append(test_entry)
    with open(CORPUS_FILE, 'w') as f:
        json.dump(corpus, f, indent=2)
    logging.debug("FORCED TEST MEMORY WRITE SUCCESS")
def cleanup_corpus():
    corpus = load_corpus()
    cleaned = [entry for entry in corpus if "query" in entry and "response" in entry]
    print(f"Before cleanup: {len(corpus)} entries")
    print(f"After cleanup:  {len(cleaned)} entries")
    save_corpus(cleaned)
    print("✅ Corpus cleaned and saved.")

corpus = load_corpus()


def set_build_daemon_prompt(f):
    global build_daemon_prompt_async
    build_daemon_prompt_async = f

def set_handle_submit(f):
    global handle_submit
    handle_submit = f

def get_recent_memories_cached(count=3):
    # Force fresh retrieval every time
    corpus = load_corpus()  # Reload to ensure freshness
    return sorted(
        [i for i in corpus if "@summary" not in i.get("tags", [])],
        key=lambda x: x.get('timestamp', datetime.min),
        reverse=True
    )[:count]
##instante gate System
gate_system = MultiStageGateSystem(model_manager, cosine_threshold=GATE_REL_THRESHOLD)
# ============================================
# FULL RERANKING INTEGRATION
# ============================================

# Import reranking components
try:
    from rerank_standalone import LLMReranker, MemoryPruner

    # Create reranker using existing model_manager
    reranker = LLMReranker(model_manager)
    pruner = MemoryPruner()

    # Store original filter method
    original_filter_memories = gate_system.filter_memories

    async def enhanced_filter_memories(query: str, memories: List[Dict]) -> List[Dict]:
        """Full reranking pipeline"""

        # Step 1: Use higher threshold for initial filtering
        old_threshold = gate_system.gate_system.cosine_threshold
        gate_system.gate_system.cosine_threshold = 0.35

        try:
            # Call original filter
            filtered = await original_filter_memories(query, memories)
            gradio_logger.debug(f"[Rerank] Stage 1: {len(memories)} → {len(filtered)} (cosine filter)")

            # Step 2: If we still have too many, use advanced reranking
            if len(filtered) > 10:
                # Cross-encoder rerank
                reranked = await reranker.cross_encoder_rerank(query, filtered)
                gradio_logger.debug(f"[Rerank] Stage 2: {len(filtered)} → {len(reranked)} (cross-encoder)")

                # Diversity pruning
                reranked = pruner.prune_by_diversity(reranked[:20])

                # Take top 10 for LLM evaluation
                if len(reranked) > 10:
                    reranked = reranked[:10]


                gradio_logger.debug(f"[Rerank] Stage 3: {len(reranked)} → {len(final)} (LLM rerank)")
                ##Dedupe logic
                def prune_duplicates(memories):
                    seen = set()
                    unique = []
                    for mem in memories:
                        key = f"{mem.get('query', '')}_{mem.get('response', '')[:50]}"
                        if key not in seen:
                            seen.add(key)
                            unique.append(mem)
                    return unique
                # LLM rerank for final selection
                final = await reranker.llm_contextual_rerank(query, reranked, max_memories=5)

                # ✅ Apply deduplication
                final = prune_duplicates(final)
                for i, mem in enumerate(final):
                    q = mem.get("query", "")[:60]
                    r = mem.get("response", "")[:60]
                    gradio_logger.debug(f"[Final Memory {i+1}] Q: {q} | A: {r}")

                return final
            else:
                # Already small enough, just limit
                return filtered[:5]

        except Exception as e:
            gradio_logger.error(f"[Rerank] Error: {e}, falling back to simple filtering")
            # Fallback to simple filtering
            filtered = await original_filter_memories(query, memories)
            return filtered[:5]

        finally:
            # Always restore threshold
            gate_system.gate_system.cosine_threshold = old_threshold

    # Replace the method
    gate_system.filter_memories = enhanced_filter_memories
    gradio_logger.info("✅ Full reranking pipeline integrated successfully!")

except ImportError as e:
    gradio_logger.warning(f"Could not import reranking: {e}")
    gradio_logger.warning("Falling back to simple filtering")

    # Simple fallback
    original_filter_memories = gate_system.filter_memories

    async def simple_filter(query: str, memories: List[Dict]) -> List[Dict]:
        filtered = await original_filter_memories(query, memories)
        return filtered[:5]

    gate_system.filter_memories = simple_filter

# ============================================
# END OF RERANKING INTEGRATION
# ============================================


# DIAGNOSTIC: Check corpus structure
with open("corpus_diagnostic.txt", "w") as f:
    f.write("CORPUS DIAGNOSTIC\n")
    f.write("="*60 + "\n")
    f.write(f"Total corpus entries: {len(corpus)}\n\n")
    f.write("LAST 5 ENTRIES:\n")
    for i, item in enumerate(corpus[-5:]):
        f.write(f"\nEntry {i}:\n")
        f.write(f"  Type: {type(item)}\n")
        f.write(f"  Keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}\n")
        f.write(f"  Query: {item.get('query', 'MISSING')[:50] if isinstance(item, dict) else 'N/A'}...\n")
        f.write(f"  Response: {item.get('response', 'MISSING')[:50] if isinstance(item, dict) else 'N/A'}...\n")
        f.write(f"  Tags: {item.get('tags', 'MISSING') if isinstance(item, dict) else 'N/A'}\n")

gradio_logger.debug("Corpus diagnostic written to corpus_diagnostic.txt")

# Initialize ChromaDB
client = PersistentClient(path="chroma_db")
try:
    collection = client.get_or_create_collection("assistant-memory")
except KeyError as e:
    import traceback
    gradio_logger.debug(f"[ERROR] Failed to load Chroma collection — possibly corrupted.")
    traceback.print_exc()
    raise RuntimeError("Chroma DB appears corrupted. Manual intervention recommended.")

# Get embedder
embedder = model_manager.get_embedder()

# Define placeholder variables for callbacks
build_daemon_prompt_async = None
handle_submit = None

def set_build_daemon_prompt(f):
    global build_daemon_prompt_async
    build_daemon_prompt_async = f

# Define get_relevant_context function BEFORE using it
def get_relevant_context(user_input):
    """Run semantic search to retrieve relevant chunks for user input."""
    topic_nouns = topic_manager.extract_nouns(user_input)
    if not topic_nouns:
        return [], [], ""

    topic_query = " ".join(topic_nouns[:3])
    search_results = semantic_search(topic_query)

    filtered_results = [
        r for r in search_results if r['score'] < CONFIDENCE_THRESHOLD
    ]
    return filtered_results, topic_nouns, topic_query

# Helper functions
def extract_query_from_doc(doc):
    """Extract query from ChromaDB document"""
    if "User:" in doc and "Assistant:" in doc:
        return doc.split("Assistant:")[0].replace("User:", "").strip()
    return doc[:100]

def extract_response_from_doc(doc):
    """Extract response from ChromaDB document"""
    if "Assistant:" in doc:
        return doc.split("Assistant:")[1].strip()
    return doc[100:]
# Wikipedia cache
wiki_cache = {}
WIKI_CACHE_TTL = 3600  # 1 hour
_tokenizer_cache = {}
@functools.lru_cache(maxsize=128)
def cached_wiki_search(topic):
    """Cache Wikipedia searches"""
    return wiki_manager.search_summary(topic, sentences=5)

async def parallel_wiki_lookup(topics):
    """Parallel Wikipedia lookups"""
    loop = asyncio.get_event_loop()
    tasks = []

    for topic in topics[:3]:  # Limit to top 3
        if topic.lower() in wiki_cache:
            continue
        task = loop.run_in_executor(None, cached_wiki_search, topic)
        tasks.append(task)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    return []

# NOW integrate hierarchical memory with all dependencies available
from memory_integration import integrate_hierarchical_memory

hierarchical_memory, hierarchical_builder = integrate_hierarchical_memory(
    model_manager=model_manager,
    prompt_builder=prompt_builder,
    personality_manager=personality_manager,
    topic_manager=topic_manager,
    time_manager=time_manager,
    wiki_manager=wiki_manager,
    huggingface_auto_tag=huggingface_auto_tag,
    get_relevant_context=get_relevant_context,
    build_daemon_prompt_setter=set_build_daemon_prompt,
    handle_submit_setter=set_handle_submit,
    embed_model=embedder,
    collection=collection,
    corpus=corpus,
    add_to_chroma_batch=add_to_chroma_batch,
    add_to_corpus=add_to_corpus
)

# Update shared_memory module
import shared_memory
shared_memory.hierarchical_memory = hierarchical_memory
shared_memory.hierarchical_builder = hierarchical_builder

# Chat history storage
chat_history = []

# Memory diagnostic function
def check_memory_storage():
    """Diagnostic function to check what's actually in memory"""
    import json

    results = collection.query(
        query_texts=["What is my name", "Luke", "my name is"],
        n_results=10,
        include=["documents", "metadatas"]
    )

    with open("memory_diagnostic.txt", "w") as f:
        f.write("MEMORY STORAGE DIAGNOSTIC\n")
        f.write("="*60 + "\n\n")

        for query_idx, query in enumerate(["What is my name", "Luke", "my name is"]):
            f.write(f"\nQUERY: '{query}'\n")
            f.write("-"*40 + "\n")

            if results['documents'][query_idx]:
                for i, doc in enumerate(results['documents'][query_idx][:5]):
                    f.write(f"\nRESULT {i+1}:\n")
                    f.write(f"Content: {doc[:200]}...\n")
                    if results['metadatas'][query_idx][i]:
                        f.write(f"Metadata: {json.dumps(results['metadatas'][query_idx][i], indent=2)}\n")
            else:
                f.write("No results found\n")

        # Also check recent memories in corpus
        f.write("\n\nRECENT CORPUS ENTRIES:\n")
        f.write("-"*40 + "\n")
        recent = [item for item in corpus if "@summary" not in item.get("tags", [])][-5:]
        for i, item in enumerate(recent):
            f.write(f"\nENTRY {i+1}:\n")
            f.write(f"Query: {item.get('query', 'N/A')[:100]}\n")
            f.write(f"Response: {item.get('response', 'N/A')[:100]}\n")
            f.write(f"Tags: {item.get('tags', [])}\n")

# Call diagnostic
check_memory_storage()
gradio_logger.debug(f"Memory diagnostic written to memory_diagnostic.txt")

# NOW define the async functions that use the integrated components...


@log_and_time("Get Relevent Context")

def get_relevant_context(user_input):
    """Run semantic search to retrieve relevant chunks for user input."""
    topic_nouns = topic_manager.extract_nouns(user_input)
    if not topic_nouns:
        return [], [], ""

    topic_query = " ".join(topic_nouns[:3])
    search_results = semantic_search(topic_query)

    filtered_results = [
        r for r in search_results if r['score'] > CONFIDENCE_THRESHOLD
    ]
    return filtered_results, topic_nouns, topic_query

def extract_query_from_doc(doc):
    """Extract query from ChromaDB document"""
    if "User:" in doc and "Assistant:" in doc:
        return doc.split("Assistant:")[0].replace("User:", "").strip()
    return doc[:100]
def stitch_recent_conversation(corpus_entries):
    """Convert corpus entries to conversation format"""
    stitched = []

    for entry in corpus_entries:
        # Handle entries with query/response structure
        if "query" in entry and "response" in entry:
            stitched.append({
                "query": entry["query"],
                "response": entry["response"],
                "timestamp": entry.get("timestamp", datetime.now())
            })
        # Handle entries with just content (legacy format)
        elif "content" in entry:
            # Try to parse User/Assistant format
            content = entry["content"]
            if "User:" in content and "Assistant:" in content:
                parts = content.split("Assistant:", 1)
                query = parts[0].replace("User:", "").strip()
                response = parts[1].strip() if len(parts) > 1 else "[no response]"
                stitched.append({
                    "query": query,
                    "response": response,
                    "timestamp": entry.get("timestamp", datetime.now())
                })
            else:
                # Fallback for unparseable content
                stitched.append({
                    "query": content[:100],
                    "response": "[no response stored]",
                    "timestamp": entry.get("timestamp", datetime.now())
                })

    return stitched

def extract_response_from_doc(doc):
    """Extract response from ChromaDB document"""
    if "Assistant:" in doc:
        return doc.split("Assistant:")[1].strip()
    return doc[100:]
@log_and_time("GUI PROMPT BUILD")
async def build_daemon_prompt_async(user_input):
    """Async version of build_daemon_prompt"""
    gradio_logger.debug(f"[CANARY] Building prompt from combined text: {user_input[:100]}...")
    response_start = datetime.now()
    debug_info = {
        'topics_detected': [],
        'wiki_content': '',
        'semantic_results': {},
        'response_start': response_start,
        'semantic_memory_results': {},
        'memory_chunks': 0,
        'summary_chunks': 0,
        'memory_breakdown': {}
    }
    gradio_logger.debug(f"([PROMPT] Starting hierarchical prompt build")

    try:
        request_start = time_manager.mark_query_time()
        # Get personality config
        personality_config = personality_manager.get_current_config()

        # Get semantic chunks if enabled
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

        # Update topic manager and wiki lookup
        topic_manager.update_from_user_input(user_input)
        debug_info['topics_detected'] = list(topic_manager.top_topics)[:5]

        # Wiki lookup
        wiki_snippet = ""
        SKIP_WIKI_PATTERNS = [
            r"^(hey|hi|hello|yo|sup|what'?s up)",
            r"long day",
            r"tired",
            r"exhausted",
            r"how are you"
        ]

        skip_wiki = any(re.match(pattern, user_input.lower().strip()) for pattern in SKIP_WIKI_PATTERNS)

        if personality_config.get("include_wiki", True) and not skip_wiki:
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

        # Get hierarchical memories
        # === STEP 1: Retrieve Hierarchical Memories ===
        hierarchical_memories = await hierarchical_memory.retrieve_relevant_memories(user_input, max_memories=MAX_WORKING_MEMORY)

        # === STEP 2: Semantic Search via ChromaDB ===
        results = collection.query(
            query_texts=[user_input],
            n_results=30,
            include=["documents", "metadatas"]
        )

        # CURRENT BROKEN FLOW:
        # very_recent = [...] ← Loaded but never used!
        # all_memories = semantic_relevant + formatted_memories[:5] ← Missing very_recent!
        # unique_memories = deduplicated(all_memories)
        # recents = unique_memories ← Misleading name, no actual recents!

        # FIXED FLOW - Replace from STEP 3 onwards:

        # === STEP 3: Load Very Recent Corpus Entries ===
        very_recent = get_recent_memories_cached(3)

        gradio_logger.debug(f"[MEMORY DEBUG] Very recent: {len(very_recent)} entries")
        for i, mem in enumerate(very_recent):
            q = mem.get("query") or mem.get("content", "")
            gradio_logger.debug(f"  Recent {i + 1}: {q[:100]}...")


        # === STEP 4: Convert ChromaDB Results into Query/Response Chunks ===
        semantic_relevant = []
        if results['documents'][0]:
            for i, doc in enumerate(results['documents'][0][:personality_config["num_memories"]]):
                semantic_relevant.append({
                    'query': extract_query_from_doc(doc),
                    'response': extract_response_from_doc(doc),
                    'timestamp': results['metadatas'][0][i].get('timestamp', datetime.now())
                })

        # === STEP 5: Format Hierarchical Memories Safely ===
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

        # === STEP 6: PROPERLY Combine All Memory Types ===
        # Format very_recent to match structure
        stitched = stitch_recent_conversation(very_recent)
        very_recent_formatted = []
        for item in stitched:
            very_recent_formatted.append({
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "timestamp": item.get("timestamp", datetime.now()),
                "_source": "very_recent"
            })

        # NOW ACTUALLY INCLUDE very_recent!
        all_memories = very_recent_formatted + semantic_relevant + formatted_memories[:5]

        gradio_logger.debug(f"[MEMORY COMBINE] {len(very_recent_formatted)} very recent + {len(semantic_relevant)} semantic + {len(formatted_memories[:5])} hierarchical")

        # Remove duplicates but prioritize very_recent
        seen = set()
        unique_memories = []

        # First add all very_recent (these should always be included)
        for mem in all_memories:
            if mem.get('_source') == 'very_recent':
                unique_memories.append(mem)
                key = f"{mem.get('query', '')}_{mem.get('response', '')[:50]}"
                seen.add(key)

        # Then add others if not duplicate
        for mem in all_memories:
            if mem.get('_source') != 'very_recent':
                key = f"{mem.get('query', '')}_{mem.get('response', '')[:50]}"
                if key not in seen:
                    seen.add(key)
                    unique_memories.append(mem)
                    if len(unique_memories) >= personality_config["num_memories"] * 2:
                        break

        # Update debug info
        debug_info['memory_chunks'] = len(unique_memories)
        debug_info['memory_breakdown'] = {
            'very_recent': len([m for m in unique_memories if m.get('_source') == 'very_recent']),
            'semantic': len(semantic_relevant),
            'hierarchical': min(5, len(formatted_memories)),
            'total_unique': len(unique_memories)
        }

        # Get summaries
        summaries = [i for i in corpus if "@summary" in i.get("tags", [])][-1:]

        # === GATING SECTION - Only gate non-very-recent ===
        # Separate by source
        definitely_include = [m for m in unique_memories if m.get('_source') == 'very_recent']
        maybe_include = [m for m in unique_memories if m.get('_source') != 'very_recent']

        # Convert maybe_include to chunks for gating
        raw_chunks_to_gate = [
            {
                "content": f"User: {m['query']}\nAssistant: {m['response']}",
                "metadata": {"timestamp": m.get("timestamp", datetime.now())}
            }
            for m in maybe_include
        ]

        # Gate only the non-very-recent memories
        if raw_chunks_to_gate:
            gated_memory_chunks = await gate_system.filter_memories(user_input, raw_chunks_to_gate)
        else:
            gated_memory_chunks = []

        # Convert very_recent to chunk format (no gating needed)
        very_recent_chunks = [
            {
                "content": f"User: {m['query']}\nAssistant: {m['response']}",
                "metadata": {"timestamp": m.get("timestamp", datetime.now())}
            }
            for m in definitely_include
        ]


       # -------------------------------------------
        # 1. Build the always-include recent slice
        # -------------------------------------------
        override_memories = []
        for chunk in very_recent_chunks:
            q, r = chunk["content"].split("Assistant:", 1)
            override_memories.append({
                "query":     q.replace("User:", "").strip(),
                "response":  r.strip(),
                "timestamp": chunk["metadata"]["timestamp"],
            })

        # -------------------------------------------
        # 2. Older memories that passed LLM gating
        # (these are already the “semantic + hierarchical” lists returned
        #  by your retrieval-and-gate pipeline *before* you trimmed it)
        # -------------------------------------------
        semantic_memory_results = {
                "documents": [
                    {
                        "title": "",
                        "text": chunk["content"],
                        "filtered_content": chunk["content"][:300]
                    }
                    for chunk in gated_memory_chunks
                    if isinstance(chunk, dict) and "content" in chunk
                ]
            }




        gradio_logger.debug(
            f"[FINAL MEMORIES] {len(override_memories)} very recent (ungated) "
            f"+ {len(semantic_memory_results)} others (gated)"
        )
        recent_conversations = [
            {**m, "id": str(uuid.uuid4())} for m in override_memories
        ]

        # -------------------------------------------
        # 3. Build time context
        # -------------------------------------------
        time_context_sources = {
            "current_time": time_manager.current_iso(),
            "elapsed_since_last": time_manager.elapsed_since_last(),
            "last_response_time": time_manager.last_response(),
        }
        # -------------------------------------------
        # 4. Call the prompt builder – no duplication, time preserved
        # -------------------------------------------
        if semantic_memory_results.get('documents'):
            gradio_logger.debug(f"[DEBUG] semantic_memory_results sample: {semantic_memory_results['documents'][0]}")
        else:
            gradio_logger.debug("[DEBUG] semantic_memory_results is empty")

        prompt = await hierarchical_builder.build_hierarchical_prompt(
            user_input=user_input,
            include_dreams=False,
            include_wiki=True,
            include_semantic=True,        # we supply them manually
            override_memories=override_memories,
            semantic_chunks=[],            # leave empty
            semantic_memory_results=semantic_memory_results,
            wiki_content=wiki_snippet,
            recent_conversations=recent_conversations,
            system_prompt=system_prompt,
            directives_file=personality_config["directives_file"],
            time_context=time_context_sources,   # <<< make sure this is forwarded
        )

        return prompt, debug_info


    except Exception as e:
        debug_info["error"] = str(e)
        gradio_logger.debug(f"[PROMPT ERROR] {e}")
        raise
## Main Assistant Response Generation Funciton
@log_and_time("Generate Streaming Response")
async def generate_streaming_response_async(prompt, model_name=None):
    logger = logging.getLogger('daemon_app')
    gradio_logger.debug(f"[GENERATE] Starting async generation with model: {model_name}")
    start_time = time.time()
    first_token_time = None

    try:
        model_manager.switch_model("gpt-4-turbo")
        response = await model_manager.generate_async(prompt)

        if hasattr(response, "__aiter__"):
            buffer = ""
            async for chunk in response:
                try:
                    # Properly extract content from ChatCompletionChunk
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            delta_content = delta.content
                        else:
                            delta_content = ""
                    else:
                        delta_content = ""

                    if delta_content:
                        now = time.time()
                        if first_token_time is None:
                            first_token_time = now
                            gradio_logger.debug(f"[STREAMING] First token arrived after {now - start_time:.2f} seconds")

                        buffer += delta_content

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            yield line.strip()


                except Exception as e:
                    logger.error(f"[STREAMING] Error processing chunk: {e}")
                    continue

            # Yield remaining buffer content
            if buffer.strip():
                yield buffer.strip()

        else:
            # Handle non-streaming response
            logger.debug("[GENERATE] Received non-streaming response, handling in fallback mode.")

            # Extract content from response object
            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message"):
                    content = response.choices[0].message.content
                else:
                    content = str(response)
            else:
                content = str(response)

            # Simulate streaming by yielding in chunks
            words = content.split()
            buffer = ""
            for i, word in enumerate(words):
                buffer += word + " "
                if i % 3 == 0 and buffer.strip():
                    yield buffer.strip()
                    buffer = ""
            if buffer.strip():
                yield buffer.strip()

    except Exception as e:
        logger.error(f"[GENERATE] Error: {type(e).__name__}: {str(e)}")
        yield f"[Streaming Error] {e}"





# Replace your handle_submit function with this version that has explicit logging)
@log_and_time("Handle Submit")
async def handle_submit(user_text, files, history, use_raw_gpt=False):
    """Handle submission with a lock and comprehensive logging."""
    global is_processing
    import inspect
    if use_raw_gpt:
            print("[RAW MODE] Using raw GPT-4 with no memory or tone")

            import time
    # ------------------- LOCKING LOGIC START -------------------
    if is_processing:
        gradio_logger.warning("[SUBMIT] Request ignored, already processing.")
        return

    is_processing = True
    try:
        gradio_logger.debug(f"[HANDLE_SUBMIT] Starting with user_text: '{user_text[:50]}...'")
        gradio_logger.debug(f"[DEBUG] handle_submit is asyncgen: {inspect.isasyncgenfunction(handle_submit)}")

        def validate_history(history):
            if not isinstance(history, list):
                return False
            for i, item in enumerate(history):
                if not isinstance(item, list) or len(item) != 2:
                    logger.error(f"[FORMAT ERROR] history[{i}] = {item} — Not a 2-item list!")
                    return False
            return True

        if not user_text.strip():
            gradio_logger.debug("[HANDLE_SUBMIT] Empty user text, returning")
            if not validate_history(history):
                logger.error(f"[FATAL] Invalid history format before empty input yield: {history}")
                raise ValueError("Invalid history format passed to Gradio")
            yield history, "", ""
            return

        if not isinstance(history, list):
            history = []

        history.append([user_text, "Thinking..."])
        gradio_logger.debug(f"[HANDLE_SUBMIT] Added to history, length now: {len(history)}")

        if not validate_history(history):
            gradio_logger.error(f"[FATAL] Invalid history format before async call: {history}")
            raise ValueError("Invalid history format passed to Gradio")

        # Process uploaded files
        combined_text = user_text
        if files:
            gradio_logger.debug(f"[HANDLE_SUBMIT] Processing {len(files)} uploaded files")
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
                    gradio_logger.debug(f"[HANDLE_SUBMIT] Processed file: {file.name}")
                except Exception as e:
                    logger.error(f"[HANDLE_SUBMIT] Error reading {file.name}: {str(e)}")
                    combined_text += f"\n\n[Error reading {file.name}: {str(e)}]"

        # Build prompt
        gradio_logger.debug("[HANDLE_SUBMIT] Building daemon prompt...")
        prompt, debug_info = await build_daemon_prompt_async(combined_text)
        gradio_logger.debug(f"[HANDLE_SUBMIT] Prompt built, length: {len(prompt)}")

        debug_display = ""
        if DEBUG_MODE:
            debug_display = f"""🔍 DEBUG INFO:

📌 Topics: {', '.join(debug_info['topics_detected']) if debug_info['topics_detected'] else 'None'}

📚 Wikipedia: {debug_info['wiki_content'] if debug_info['wiki_content'] else 'No wiki content'}

🔎 Semantic Search:
  - Query: {debug_info['semantic_results'].get('search_query', '')}
  - Results: {debug_info['semantic_results'].get('results_found', 0)} chunks
  - Top: {debug_info['semantic_results'].get('top_results', [])[:2]}

💾 Memory: {debug_info['memory_chunks']} conversations
  - Recent: {debug_info['memory_breakdown'].get('very_recent', 0)}
  - Semantic: {debug_info['memory_breakdown'].get('semantic', 0)}
  - Hierarchical: {debug_info['memory_breakdown'].get('hierarchical', 0)}

📊 Prompt: ~{len(prompt)} chars"""

        # Stream response
        full_response = ""
        gradio_logger.debug("[HANDLE_SUBMIT] Starting streaming response generation...")

        async for partial_response in generate_streaming_response_async(prompt, "gpt-4-turbo"):
            full_response = (full_response + " " + partial_response).strip()
            temp_history = history[:-1] + [[user_text, full_response]]
            yield temp_history, debug_display, ""

        # Memory save
        response_end = datetime.now()
        response_time = (response_end - debug_info['response_start']).total_seconds()
        gradio_logger.debug(f"[HANDLE_SUBMIT] Response generation complete in {response_time:.2f} seconds")

        time_manager.mark_query_time()
        uid = str(uuid.uuid4())
        gradio_logger.debug(f"[HANDLE_SUBMIT] Generating tags and storing memory with ID: {uid}")

        tags = huggingface_auto_tag(f"User: {user_text}\nAssistant: {full_response}", model_manager, model_name="gpt-3.5-turbo")
        add_to_chroma(f"User: {user_text}\nAssistant: {full_response}", uid, tags, collection, entry_type="memory")
        add_to_corpus(corpus, user_text, full_response, tags)
        logging.debug(f"[add_to_corpus] Corpus updated successfully with entry: {user_text[:30]}...")

        asyncio.create_task(
            hierarchical_memory.store_interaction(user_text, full_response, tags)
        )

        gradio_logger.debug("[HANDLE_SUBMIT] Memory store dispatched to background task")
        history[-1][1] = full_response

        for i, msg in enumerate(history):
            if not isinstance(msg, list) or len(msg) != 2:
                logger.error(f"[FATAL FORMAT] history[{i}] = {msg} — Not a 2-item list!")
                raise ValueError(f"Gradio Chatbot requires history[{i}] to be [user, assistant]")

        gradio_logger.debug(f"[HANDLE_SUBMIT] Complete. Final history length: {len(history)}")
        yield history, debug_display, ""

    finally:
        # -------------------- LOCKING LOGIC END --------------------
        is_processing = False
        gradio_logger.debug("[SUBMIT] Processing finished, lock released.")




# Call this function right after the GUI loads


def switch_personality_ui(selected):
    """Switch personality mode"""
    personality_manager.switch_personality(selected)
    return f"Switched to {selected} mode."





# Build Gradio UI with improved layout
with gr.Blocks(css="""
    #chatbot { height: 500px; }
    #debug_panel { max-height: 300px; overflow-y: auto; }
    .user-message { background-color: #e3f2fd; }
    .assistant-message { background-color: #f5f5f5; }
""") as demo:
    gr.Markdown("# 🤖 Daemon Chat Interface")

    with gr.Row():
        with gr.Column(scale=1):
            # Personality selector
            personality_dropdown = gr.Dropdown(
                label="Personality Mode",
                choices=list(personality_manager.personalities.keys()),
                value="default",
                interactive=True
            )
            personality_status = gr.Textbox(
                label="Status",
                value="Using default mode.",
                interactive=False,
                max_lines=1
            )
        use_raw_gpt = gr.Checkbox(label="Bypass Memory System (Raw GPT)", value=False)

        # Debug panel
        with gr.Accordion("🔍 Debug Info", open=DEBUG_MODE):
                debug_display = gr.Textbox(
                    label="System Information",
                    lines=15,
                    interactive=False,
                    elem_id="debug_panel"
                )

        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chatbot"
            )

            with gr.Row():
                with gr.Column(scale=4):
                    user_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

            file_input = gr.File(
                file_types=[".txt", ".docx", ".csv", ".py"],
                file_count="multiple",
                label="Attach files (optional)"
            )

            # Status bar
            status_bar = gr.Textbox(
                label="System Status",
                value="Ready",
                interactive=False,
                max_lines=1
            )

    # Event handlers
    personality_dropdown.change(
        switch_personality_ui,
        inputs=[personality_dropdown],
        outputs=[personality_status]
    )



    # Use this instead
    submit_btn.click(
        handle_submit,
        inputs=[user_input, file_input, chatbot, use_raw_gpt],
        outputs=[chatbot, debug_display, status_bar]
    )


    gradio_logger.debug(f"✅ Submit button handler registered")
    user_input.submit(
        handle_submit,
        inputs=[user_input, file_input, chatbot, use_raw_gpt],
        outputs=[chatbot, debug_display, status_bar]
    )
    gradio_logger.debug(f"✅ Enter key handler registered")
    clear_btn.click(
        lambda: ([], "", ""),
        outputs=[chatbot, user_input, debug_display]
    )

# Launch app
if __name__ == "__main__":
    try:
        logger = setup_logging()
        gradio_logger = GradioLogger()

        gradio_logger.debug("Starting Gradio application...")

        def log_handler_check():
            root = logging.getLogger()
            handlers_info = [f"{type(h).__name__}({h.level})" for h in root.handlers]
            gradio_logger.debug(f"Current root handlers: {handlers_info}")

        log_handler_check()

        gr_logger = logging.getLogger('gradio')
        gr_logger.setLevel(logging.DEBUG)

        for name in ['gradio', 'gradio.routes', 'gradio.queueing', 'httpx', 'httpcore']:
            logger_instance = logging.getLogger(name)
            logger_instance.handlers = []
            logger_instance.propagate = True

        demo.queue()

        gradio_logger.debug("About to call demo.launch()...")

        demo.launch(
            server_name="0.0.0.0",
            max_file_size="100mb",
            max_threads=40,
            quiet=False,
            show_error=True,
            share=True
        )

    finally:
        gradio_logger.debug("Shutting down. Closing model manager HTTP client...")
        atexit.register(cleanup_gpu)
        signal.signal(signal.SIGINT, lambda s, f: (cleanup_gpu(), exit(0)))
        signal.signal(signal.SIGTERM, lambda s, f: (cleanup_gpu(), exit(0)))
        model_manager.close()
