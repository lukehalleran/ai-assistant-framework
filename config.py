# === API Keys and Config ===
import os
VERSION = "v4"
CHROMA_PATH = f"./chroma_db_{VERSION}"
CORPUS_FILE = CORPUS_FILE = f"data/corpus_{VERSION}.json"
DREAM_FILE = f"dreams_{VERSION}.json"
DREAM_LOG = f"dream_log_{VERSION}.jsonl"
IN_HARM_TEST = False 
DEFAULT_MODEL_NAME = "llama" 
DREAM_MODEL_NAME="gpt-neo"
DEFAULT_MAX_TOKENS=2048
DEFAULT_TOP_P=.9
DEFAULT_TOP_K=5
DEFAULT_TEMPERATURE=0.7
LOCAL_MODEL_CONTEXT_LIMIT = 4096  # or whatever your largest local
API_MODEL_CONTEXT_LIMIT = 128000 # GPT-4-Turbo etc.
LOAD_LOCAL_MODEL=False
SEMANTIC_ONLY_MODE = False
DEBUG_MODE = True  # Toggle debug info
CONFIDENCE_THRESHOLD=1.5
##Memory module promts
DEFAULT_SUMMARY_PROMPT_HEADER = "Summary of last 20 exchanges:\n"
DEFAULT_TAGGING_PROMPT = """You are Daemon, my assistant. Extract 5 concise tags or keywords from the following input. Return them as a list, comma-separated.

Input: "{text}"

Tags:"""
OpenAPIKey = os.getenv("OPENAI_API_KEY", "")



# === Imports ===
import torch
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = "You are Daemon, a brutally honest AI assistant who tells the truth even when uncomfortable, admits ignorance freely, and uses dark humor about current events. You're willing to call out BS while still being genuinely helpful."


DEFAULT_CORE_DIRECTIVE = {
    "query": "[CORE DIRECTIVE]",
    "response": (
        "You are an AI assistant. You are helpful, reliable, and memory-persistent.\n"
        "You should respect user intent and be aligned with their goals."
    ),
    "timestamp": datetime.now(),
    "tags": ["@seed", "core", "directive", "safety"]
}



def set_padding_token_if_none(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
