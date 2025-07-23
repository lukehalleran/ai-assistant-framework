# config.py
import os
from utils.logging_utils import get_logger

logger = get_logger("config")
logger.debug("config.py is alive")

VERSION = os.getenv("DAEMON_VERSION", "v4")

# Base directory for data (production)
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")

# Construct default production paths
_default_corpus_file = os.path.join(
    DEFAULT_DATA_DIR, f"corpus_{VERSION}.json"
)
_default_chroma_dir = os.path.join(
    DEFAULT_DATA_DIR, f"chroma_db_{VERSION}"
)

# Allow overrides via ENV
CORPUS_FILE = os.getenv("CORPUS_FILE", _default_corpus_file)
CHROMA_PATH = os.getenv("CHROMA_PATH", _default_chroma_dir)

logger.debug(f"Using CORPUS_FILE={CORPUS_FILE}")
logger.debug(f"Using CHROMA_PATH={CHROMA_PATH}")

#DREAM_FILE = f"dreams_{VERSION}.json"
#DREAM_LOG = f"dream_log_{VERSION}.jsonl"
IN_HARM_TEST = False 
DEFAULT_MODEL_NAME = "llama" 
DREAM_MODEL_NAME="gpt-neo"
DEFAULT_MAX_TOKENS=2048
DEFAULT_TOP_P=.9
DEFAULT_TOP_K=5
DEFAULT_TEMPERATURE=0.7
LOCAL_MODEL_CONTEXT_LIMIT = 4096  # or whatever your largest local
API_MODEL_CONTEXT_LIMIT = 128000 # GPT-4-Turbo etc.
LOAD_LOCAL_MODEL=True
SEMANTIC_ONLY_MODE = False
DEBUG_MODE = True  # Toggle debug info
CONFIDENCE_THRESHOLD=1.5
GATE_REL_THRESHOLD = 0.35 # temp for debug        # Minimum relevance score to pass LLM gate (0-1)
MAX_FINAL_MEMORIES = 5
RERANK_USE_LLM = True
CROSS_ENCODER_WEIGHT = 0.7# Maximum memories to return after all filtering
MEM_NO = 5                      # Number of memories to expand hierarchically
MEM_IMPORTANCE_SCORE = 0.6      # Minimum importance for parent/child expansion
MAX_WORKING_MEMORY=10
CHILD_MEM_LIMIT = 3  # Max children to include per memory
##Memory module promts
COSINE_SIMILARITY_THRESHOLD=0.25 #try values between .3 and .7, experiement
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
