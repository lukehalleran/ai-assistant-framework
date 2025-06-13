"""
memory.py - Persistent memory management for assistant interactions.

Responsibilities:
- Corpus load/save to JSON file
- Automatic tagging of interactions
- Periodic summarization of recent exchanges
- Trimming corpus size for efficiency
"""


import json, os
from datetime import datetime
from config import CORPUS_FILE, IN_HARM_TEST
from persistence import add_to_chroma
from transformers import pipeline
from config import DEFAULT_SUMMARY_PROMPT_HEADER, DEFAULT_TAGGING_PROMPT




def huggingface_auto_tag(text, model_manager, model_name=None):
    """Auto-tag text using the model's generate() function."""
    try:
        if model_name:
            model_manager.switch_model(model_name)

        prompt = DEFAULT_TAGGING_PROMPT.format(text=text)
        tags_text = model_manager.generate(prompt, max_tokens=40)
        return [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    except Exception as e:
        print(f"[Tagging Error] Could not generate tags: {e}")
        return ["misc"]


def format_timestamp(ts):

    """Format timestamp for saving to disk."""

    return ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else ts

def load_corpus():
    """Load memory corpus from disk."""
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE,  encoding="utf-8") as f:
            data = json.load(f)
            for e in data:
                if isinstance(e["timestamp"], str):
                    try:
                        e["timestamp"] = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
            return data
    return []

def save_corpus(corpus):
    """Save memory corpus to disk (atomic write with tmp file)."""
    tmp = CORPUS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([
            {**e, "timestamp": format_timestamp(e["timestamp"])} for e in corpus
        ], f, indent=2)
    os.replace(tmp, CORPUS_FILE)

def summarize_corpus_block(block):

    """Generate a summary node for the last 20 exchanges."""

    summary_lines = [f"Q: {e['query']} → A: {e['response'][:60]}..." for e in block[:20]]
    return {
        "query": "[SUMMARY NODE]",
        "response": DEFAULT_SUMMARY_PROMPT_HEADER + "\n".join(summary_lines) + "\n[truncated]",
        "timestamp": datetime.now(),
        "tags": ["@summary"]
    }


def add_to_corpus(corpus, query, response, tags):
    """Add a new interaction to the corpus, summarize every 20 interactions, trim if needed."""
    if IN_HARM_TEST:
        return  # 🚫 Block during harm test

    entry = {"query": query, "response": response, "timestamp": datetime.now(), "tags": tags}
    corpus.append(entry)
    real = [e for e in corpus if "@summary" not in e.get("tags", [])]
    if len(real) % 20 == 0:
        corpus.append(summarize_corpus_block(real[-20:]))
    if len(corpus) > 500:
        corpus[:] = corpus[-500:]
    save_corpus(corpus)
