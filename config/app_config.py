"""
# config/app_config.py

Module Contract
- Purpose: Central configuration loader/normalizer. Reads YAML (optional), merges env overrides, sets defaults, exposes strongly‑typed constants, and loads the system prompt text.
- Inputs:
  - Optional YAML (config.yaml) at several search paths
  - Environment variables (e.g., CORPUS_FILE, CHROMA_PATH, OPENAI_API_KEY, SUMMARY_* knobs, gating thresholds)
- Outputs:
  - Module‑level constants used across the stack: paths, memory/gating/model limits, SYSTEM_PROMPT text, etc.
- Key functions:
  - load_yaml_config(config_path) → dict: tolerant loader with variable resolution
  - ensure_config_defaults(config) → dict: fills missing critical defaults
  - load_system_prompt(cfg) → str: resolves from core/system_prompt[.txt] (stripping header comments) or falls back to inline
- Important constants:
  - CORPUS_FILE, CHROMA_PATH, SYSTEM_PROMPT, DEFAULT_* model knobs, gating thresholds, CORPUS_MAX_ENTRIES
- Side effects:
  - Creates data directories on import to ensure persistence paths exist.
- Error handling:
  - Logs and falls back to safe defaults if files/vars are missing.
"""
import os
import re
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from utils.logging_utils import get_logger

logger = get_logger("config")

# --------------------------------------------------------------------
# Variable resolution
# --------------------------------------------------------------------

def resolve_vars(config: dict) -> dict:
    """
    Recursively resolves placeholder variables in the config like ${section.key}.
    """
    if not isinstance(config, dict):
        return config

    def get_value_by_path(path: str, conf_dict: dict):
        keys = path.split(".")
        value = conf_dict
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def resolve_value(value, conf_dict):
        if isinstance(value, str):
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, value)
            for match in matches:
                replacement = get_value_by_path(match, conf_dict)
                if replacement is not None:
                    value = value.replace(f"${{{match}}}", str(replacement))
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v, conf_dict) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, conf_dict) for item in value]
        else:
            return value

    # Multiple passes to resolve nested references
    for _ in range(5):
        prev = str(config)
        config = resolve_value(config, config)
        if str(config) == prev:
            break

    return config

# --------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------

def load_yaml_config(config_path="config.yaml"):
    """Load configuration from YAML file with robust variable substitution."""
    # Try multiple paths
    paths_to_try = list(dict.fromkeys([
        Path(config_path),
        Path(__file__).parent / config_path,
        Path(__file__).parent.parent / config_path,
        Path.cwd() / config_path,
    ]))

    config = {}
    for path in paths_to_try:
        if path.exists():
            logger.info(f"Loading config from: {path}")
            try:
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        logger.error("Config file is not a valid dictionary.")
                        config = {}
                    break
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                config = {}

    if not config:
        logger.warning(f"Config file not found in any of: {paths_to_try}, using defaults.")

    logger.debug(f"Config before resolution - corpus_file: {config.get('memory', {}).get('corpus_file', 'NOT SET')}")
    config = resolve_vars(config)
    logger.debug(f"Config after resolution - corpus_file: {config.get('memory', {}).get('corpus_file', 'NOT SET')}")

    return config

# --------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------

def ensure_config_defaults(config):
    """Ensure critical config values have defaults after resolution."""
    daemon = config.setdefault("daemon", {})
    daemon.setdefault("version", "v4")
    daemon.setdefault("data_dir", "./data")
    daemon.setdefault("log_dir", "./conversation_logs")

    memory = config.setdefault("memory", {})

    # Only set defaults if the value is missing or unresolved
    existing_corpus = memory.get("corpus_file", None)
    if not existing_corpus or "${" in str(existing_corpus):
        memory["corpus_file"] = os.path.join(
            daemon["data_dir"], f"corpus_{daemon['version']}.json"
        )
        logger.info(f"Set default corpus_file to: {memory['corpus_file']}")

    existing_chroma = memory.get("chroma_path", None)
    if not existing_chroma or "${" in str(existing_chroma):
        memory["chroma_path"] = os.path.join(
            daemon["data_dir"], f"chroma_db_{daemon['version']}_v2"
        )
        logger.info(f"Set default chroma_path to: {memory['chroma_path']}")

    # Ensure other sections exist
    config.setdefault("models", {})
    config.setdefault("gating", {})
    config.setdefault("paths", {})
    config.setdefault("features", {})
    config.setdefault("prompts", {})

    return config

# --------------------------------------------------------------------
# Main Loading Sequence (only load once!)
# --------------------------------------------------------------------

logger.info("Loading configuration...")
config = load_yaml_config("config.yaml")
config = ensure_config_defaults(config)

# Extract commonly used values
VERSION = config.get("daemon", {}).get("version")
DEFAULT_DATA_DIR = config.get("daemon", {}).get("data_dir")
CORPUS_FILE = config.get("memory", {}).get("corpus_file")
CHROMA_PATH = config.get("memory", {}).get("chroma_path")

# Final validation and forcing if still unresolved
if CORPUS_FILE and "${" in CORPUS_FILE:
    logger.warning(f"CORPUS_FILE still contains variables: {CORPUS_FILE}")
    CORPUS_FILE = f"./data/corpus_{VERSION}.json"
    config['memory']['corpus_file'] = CORPUS_FILE

if CHROMA_PATH and "${" in CHROMA_PATH:
    logger.warning(f"CHROMA_PATH still contains variables: {CHROMA_PATH}")
    CHROMA_PATH = f"./data/chroma_db_{VERSION}"
    config['memory']['chroma_path'] = CHROMA_PATH

logger.info(f"Final CORPUS_FILE: {CORPUS_FILE}")
logger.info(f"Final CHROMA_PATH: {CHROMA_PATH}")

# Create data directories if needed
Path(DEFAULT_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Export all config values
# --------------------------------------------------------------------

IN_HARM_TEST = config.get("daemon", {}).get("in_harm_test", False)
DEBUG_MODE = config.get("daemon", {}).get("debug_mode", True)
DEFAULT_MODEL_NAME = config.get("models", {}).get("default", "llama")
DREAM_MODEL_NAME = config.get("models", {}).get("dream_model", "gpt-neo")
DEFAULT_MAX_TOKENS = config.get("models", {}).get("default_max_tokens", 2048)
HEAVY_TOPIC_MAX_TOKENS = config.get("models", {}).get("heavy_topic_max_tokens", 8192)
DEFAULT_TOP_P = config.get("models", {}).get("default_top_p", 0.9)
DEFAULT_TOP_K = config.get("models", {}).get("default_top_k", 5)
DEFAULT_TEMPERATURE = config.get("models", {}).get("default_temperature", 0.7)
LOCAL_MODEL_CONTEXT_LIMIT = config.get("models", {}).get("local_model_context_limit", 4096)
API_MODEL_CONTEXT_LIMIT = config.get("models", {}).get("api_model_context_limit", 128000)
LOAD_LOCAL_MODEL = config.get("models", {}).get("load_local_model", True)
SEMANTIC_ONLY_MODE = config.get("features", {}).get("semantic_only_mode", False)
CONFIDENCE_THRESHOLD = config.get("gating", {}).get("confidence_threshold", 1.5)
GATE_REL_THRESHOLD = config.get("gating", {}).get("gate_rel_threshold", 0.25)
MAX_FINAL_MEMORIES = config.get("memory", {}).get("max_final_memories", 5)
RERANK_USE_LLM = config.get("gating", {}).get("rerank_use_llm", True)
CROSS_ENCODER_WEIGHT = config.get("gating", {}).get("cross_encoder_weight", 0.7)
MEM_NO = config.get("memory", {}).get("mem_no", 5)
MEM_IMPORTANCE_SCORE = config.get("memory", {}).get("mem_importance_score", 0.6)
MAX_WORKING_MEMORY = config.get("memory", {}).get("max_working_memory", 10)
CHILD_MEM_LIMIT = config.get("memory", {}).get("child_mem_limit", 3)
CORPUS_MAX_ENTRIES = int(config.get("memory", {}).get("corpus_max_entries", 2000))
COSINE_SIMILARITY_THRESHOLD = config.get("gating", {}).get("cosine_similarity_threshold", 0.25)

# Hybrid Retrieval Configuration
# ------------------------------
# Enable hybrid retrieval (recent + semantic) for summaries and reflections
# When enabled, uses n/4 recent + 3n/4 semantic for summaries (1:3 ratio)
# and n/3 recent + 2n/3 semantic for reflections (1:2 ratio)
# Falls back to pure recency if disabled or unavailable
HYBRID_SUMMARIES_ENABLED = bool(config.get("memory", {}).get("hybrid_summaries_enabled", True))
HYBRID_REFLECTIONS_ENABLED = bool(config.get("memory", {}).get("hybrid_reflections_enabled", True))

# Summary/Reflection Budget Ratios
# --------------------------------
# Ratio of recent:semantic items (default: 0.25 = 1:3 for summaries, 0.33 = 1:2 for reflections)
# Lower values favor semantic retrieval, higher values favor recency
# Note: These are applied in memory_coordinator, not here (hardcoded for now)
# SUMMARY_RECENT_RATIO = float(config.get("memory", {}).get("summary_recent_ratio", 0.25))
# REFLECTION_RECENT_RATIO = float(config.get("memory", {}).get("reflection_recent_ratio", 0.33))

# Cosine Filtering Thresholds
# ---------------------------
# Threshold for cosine similarity filtering of summaries/reflections
# Range: 0.0-1.0, higher = more selective
# Summaries use 0.30 (should be clearly relevant, dense content)
# Reflections use 0.25 (more abstract, cast wider net)
SUMMARY_COSINE_THRESHOLD = float(config.get("gating", {}).get("summary_cosine_threshold", 0.30))
REFLECTION_COSINE_THRESHOLD = float(config.get("gating", {}).get("reflection_cosine_threshold", 0.25))

# Legacy config (keep for backward compatibility)
DEFAULT_SUMMARY_PROMPT_HEADER = config.get("memory", {}).get("default_summary_prompt_header", "Summary of last 20 exchanges:\n")
DEFAULT_TAGGING_PROMPT = config.get("memory", {}).get("default_tagging_prompt", "...")

OpenAPIKey = config.get("models", {}).get("openai_api_key", "")
topic_confidence_threshold = config.get("gating", {}).get("topic_confidence_threshold", 0.7)
RECENCY_DECAY_RATE = config.get("memory", {}).get("recency_decay_rate", 0.05)
TRUTH_SCORE_UPDATE_RATE = config.get("memory", {}).get("truth_score_update_rate", 0.02)
TRUTH_SCORE_MAX = config.get("memory", {}).get("truth_score_max", 0.95)
COLLECTION_BOOSTS = config.get("memory", {}).get("collection_boosts", {
    "facts": 0.15,
    "summaries": 0.10,
    "conversations": 0.0,
    "semantic": 0.05,
    "wiki": 0.05,
})

# -----------------------------
# Wikipedia defaults (config-driven)
# -----------------------------
# These drive how much wiki text is fetched and included. They act as defaults
# when environment variables are not explicitly set.
WIKI_CFG = config.get("wiki", {})
WIKI_FETCH_FULL_DEFAULT: bool = bool(WIKI_CFG.get("fetch_full", True))
WIKI_MAX_CHARS_DEFAULT: int = int(WIKI_CFG.get("max_chars", 15000))
# 0 (or <=0) disables sentence clipping; intro/full selection is handled elsewhere
WIKI_MAX_SENTENCES_DEFAULT: int = int(WIKI_CFG.get("max_sentences", 0))
WIKI_TIMEOUT_DEFAULT: float = float(WIKI_CFG.get("timeout_s", 1.2))
DEICTIC_THRESHOLD = config.get("gating", {}).get("deictic_threshold", 0.60)
NORMAL_THRESHOLD = config.get("gating", {}).get("normal_threshold", 0.35)
DEICTIC_ANCHOR_PENALTY = config.get("gating", {}).get("deictic_anchor_penalty", 0.1)
DEICTIC_CONTINUITY_MIN = config.get("gating", {}).get("deictic_continuity_min", 0.12)
SCORE_WEIGHTS = config.get("gating", {}).get("score_weights", {
    "relevance": 0.35,
    "recency": 0.25,
    "truth": 0.20,
    "importance": 0.05,
    "continuity": 0.10,
    "structure": 0.05,
})

# Best-of-N generation (answer-side reranking)
ENABLE_BEST_OF = config.get("features", {}).get("enable_best_of", True)
BEST_OF_N = int(config.get("features", {}).get("best_of_n", 2))
BEST_OF_TEMPS = tuple(config.get("features", {}).get("best_of_temps", [0.2, 0.7]))
BEST_OF_MIN_QUESTION = bool(config.get("features", {}).get("best_of_min_question", True))
BEST_OF_MAX_TOKENS = int(config.get("features", {}).get("best_of_max_tokens", 256))
BEST_OF_MODEL = config.get("features", {}).get("best_of_model", None)
BEST_OF_MIN_TOKENS = int(config.get("features", {}).get("best_of_min_tokens", 8))

# Optional multi-model generators/selectors (defaults keep current behavior)
BEST_OF_GENERATOR_MODELS = list(
    config.get("features", {}).get("best_of_generator_models", [])
)
BEST_OF_SELECTOR_MODELS = list(
    config.get("features", {}).get("best_of_selector_models", [])
)
BEST_OF_SELECTOR_MAX_TOKENS = int(
    config.get("features", {}).get("best_of_selector_max_tokens", 64)
)
BEST_OF_SELECTOR_WEIGHTS = dict(
    config.get("features", {}).get(
        "best_of_selector_weights", {"heuristic": 1.0, "llm": 0.0}
    )
)
BEST_OF_SELECTOR_TOP_K = int(
    config.get("features", {}).get("best_of_selector_top_k", 0)
)

# Optional strict 2-model duel mode (A vs B judged by a single judge)
BEST_OF_DUEL_MODE = bool(
    config.get("features", {}).get("best_of_duel_mode", False)
)

# Query rewrite toggle (can add latency on first token)
ENABLE_QUERY_REWRITE = bool(config.get("features", {}).get("enable_query_rewrite", True))
# Bound rewrite latency to keep first-token time low
REWRITE_TIMEOUT_S = float(config.get("features", {}).get("rewrite_timeout_s", 1.2))

# Soft latency budget for best-of reranking before falling back to streaming
BEST_OF_LATENCY_BUDGET_S = float(config.get("features", {}).get("best_of_latency_budget_s", 2.0))

system_prompt_file = config.get("paths", {}).get("system_prompt_file", {})

DEFAULT_CORE_DIRECTIVE = config.get("prompts", {}).get("default_core_directive", {
    "query": "[CORE DIRECTIVE]",
    "response": "You are an AI assistant...",
    "timestamp": datetime.now().isoformat(),
    "tags": ["@seed", "core", "directive", "safety"],
})

# --------------------------------------------------------------------
# System Prompt Loader
# --------------------------------------------------------------------

def load_system_prompt(cfg: Optional[Dict] = None) -> str:
    """Load system prompt with proper fallback chain"""
    cfg = cfg or config

    # Try different paths in order
    paths_to_try = []

    # From config paths section
    if cfg.get('paths', {}).get('system_prompt'):
        paths_to_try.append(Path(cfg['paths']['system_prompt']))

    # Standard locations (support both with and without .txt)
    paths_to_try.extend([
        Path(__file__).parent.parent / 'core' / 'system_prompt',
        Path.cwd() / 'core' / 'system_prompt',
        Path('core') / 'system_prompt',
        Path(__file__).parent.parent / 'core' / 'system_prompt.txt',
        Path.cwd() / 'core' / 'system_prompt.txt',
        Path('core') / 'system_prompt.txt',
    ])

    def _clean_header(text: str) -> str:
        """
        Drop leading file-header comment lines (e.g., '#core/system_prompt.txt', '# Daemon System Prompt …')
        while preserving markdown sections that follow. Stops at the first blank line.
        """
        if not text:
            return text
        lines = text.splitlines()
        i = 0
        # remove contiguous leading comment-ish lines (starting with '#') until a blank line
        while i < len(lines):
            line = lines[i].strip()
            if line == "":
                i += 1
                break
            if line.startswith('#'):
                i += 1
                continue
            # First non-comment, non-blank line reached; stop stripping
            break
        cleaned = "\n".join(lines[i:]).lstrip("\n")
        return cleaned or text

    for path in paths_to_try:
        if path and path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # normalize whitespace and remove any leading header comments
                    content = _clean_header(content).strip()
                if content:
                    logger.info(f"Loaded system prompt from: {path}")
                    return content
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")

    # Fallback to inline
    default = cfg.get('prompts', {}).get('default_system_prompt',
                                         "You are Daemon, a helpful AI assistant.")
    logger.info("Using default system prompt from config")
    return default

# --------------------------------------------------------------------
# Final setup
# --------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM_PROMPT = load_system_prompt(config)

# Environment overrides
VERSION = os.getenv("DAEMON_VERSION", VERSION)
CORPUS_FILE = os.getenv("CORPUS_FILE", CORPUS_FILE)
CHROMA_PATH = os.getenv("CHROMA_PATH", CHROMA_PATH)
OpenAPIKey = os.getenv("OPENAI_API_KEY", OpenAPIKey)
try:
    CORPUS_MAX_ENTRIES = int(os.getenv("CORPUS_MAX_ENTRIES", str(CORPUS_MAX_ENTRIES)))
except Exception:
    pass

logger.info(f"Config loaded successfully for VERSION={VERSION}")
logger.info(f"Using CORPUS_FILE={CORPUS_FILE}")
logger.info(f"Using CHROMA_PATH={CHROMA_PATH}")
logger.info(f"Using DEVICE={device}")
logger.info(f"Corpus max entries={CORPUS_MAX_ENTRIES}")
