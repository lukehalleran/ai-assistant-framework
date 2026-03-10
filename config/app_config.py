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
  - OBSIDIAN_ENABLED, OBSIDIAN_VAULT_PATH, OBSIDIAN_CHUNK_THRESHOLD, OBSIDIAN_MAX_NOTES_PROMPT [NEW]
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
GATE_REL_THRESHOLD = config.get("gating", {}).get("gate_rel_threshold", 0.18)
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

# --------------------------------------------------------------------
# Web Search Configuration (Tavily API)
# --------------------------------------------------------------------
# Enable real-time web search for queries requiring current information
# Uses Tavily API for search and content extraction
WEB_SEARCH_CFG = config.get("web_search", {})
WEB_SEARCH_ENABLED: bool = bool(WEB_SEARCH_CFG.get("enabled", True))
# Tavily API key (can also be set via TAVILY_API_KEY env var)
WEB_SEARCH_API_KEY: str = WEB_SEARCH_CFG.get("api_key", "") or os.getenv("TAVILY_API_KEY", "")
# Search timeout in seconds
WEB_SEARCH_TIMEOUT: float = float(WEB_SEARCH_CFG.get("timeout_s", 30.0))
# Maximum content characters per extracted page
WEB_SEARCH_MAX_CONTENT_CHARS: int = int(WEB_SEARCH_CFG.get("max_content_chars", 10000))
# Daily credit limit (Tavily free tier: 1000 credits/month ~ 33/day)
WEB_SEARCH_DAILY_CREDIT_LIMIT: int = int(WEB_SEARCH_CFG.get("daily_credit_limit", 100))
# Per-query credit limit
WEB_SEARCH_PER_QUERY_LIMIT: int = int(WEB_SEARCH_CFG.get("per_query_limit", 5))
# Cache TTL in hours
WEB_SEARCH_CACHE_TTL_HOURS: int = int(WEB_SEARCH_CFG.get("cache_ttl_hours", 72))
# Confidence threshold for triggering search (0.0-1.0)
WEB_SEARCH_CONFIDENCE_THRESHOLD: float = float(WEB_SEARCH_CFG.get("confidence_threshold", 0.5))
# Model for DEEP search link selection
WEB_SEARCH_LINK_SELECTOR_MODEL: str = WEB_SEARCH_CFG.get("link_selector_model", "gpt-4o-mini")

# Environment variable overrides for web search
WEB_SEARCH_ENABLED = bool(int(os.getenv("WEB_SEARCH_ENABLED", "1" if WEB_SEARCH_ENABLED else "0")))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", str(WEB_SEARCH_TIMEOUT)))
WEB_SEARCH_DAILY_CREDIT_LIMIT = int(os.getenv("WEB_SEARCH_DAILY_CREDIT_LIMIT", str(WEB_SEARCH_DAILY_CREDIT_LIMIT)))

# --------------------------------------------------------------------
# Wolfram Alpha Configuration
# --------------------------------------------------------------------
# Enable Wolfram Alpha for computational queries (math, science, data)
# Uses Wolfram Alpha LLM API for natural language computation
WOLFRAM_CFG = config.get("wolfram", {})
WOLFRAM_ENABLED: bool = bool(WOLFRAM_CFG.get("enabled", True))
# Wolfram Alpha App ID (can also be set via WOLFRAM_APP_ID env var)
WOLFRAM_APP_ID: str = WOLFRAM_CFG.get("app_id", "") or os.getenv("WOLFRAM_APP_ID", "")
# API endpoint
WOLFRAM_API_URL: str = WOLFRAM_CFG.get("api_url", "https://www.wolframalpha.com/api/v1/llm-api")
# Request timeout in seconds
WOLFRAM_TIMEOUT: float = float(WOLFRAM_CFG.get("timeout_s", 30.0))
# Maximum output characters from API
WOLFRAM_MAX_OUTPUT_CHARS: int = int(WOLFRAM_CFG.get("max_output_chars", 10000))
# Cache TTL in seconds (1 hour default - computational results don't change)
WOLFRAM_CACHE_TTL_SECONDS: int = int(WOLFRAM_CFG.get("cache_ttl_seconds", 3600))
# Rate limit per minute
WOLFRAM_RATE_LIMIT_PER_MINUTE: int = int(WOLFRAM_CFG.get("rate_limit_per_minute", 60))

# Environment variable overrides for Wolfram Alpha
WOLFRAM_ENABLED = bool(int(os.getenv("WOLFRAM_ENABLED", "1" if WOLFRAM_ENABLED else "0")))
WOLFRAM_TIMEOUT = float(os.getenv("WOLFRAM_TIMEOUT", str(WOLFRAM_TIMEOUT)))

# --------------------------------------------------------------------
# E2B Code Sandbox Configuration
# --------------------------------------------------------------------
# Secure Python code execution in Firecracker microVMs via E2B
# Use for multi-step calculations, data analysis, visualizations
SANDBOX_CFG = config.get("sandbox", {})
SANDBOX_ENABLED: bool = bool(SANDBOX_CFG.get("enabled", True))
# E2B API key (can also be set via E2B_API_KEY env var)
SANDBOX_API_KEY: str = SANDBOX_CFG.get("api_key", "") or os.getenv("E2B_API_KEY", "")
# Max execution time per code block in seconds
SANDBOX_TIMEOUT_SECONDS: int = int(SANDBOX_CFG.get("timeout_seconds", 60))
# Persistent session lifetime in minutes
SANDBOX_SESSION_TIMEOUT_MINUTES: int = int(SANDBOX_CFG.get("session_timeout_minutes", 30))
# Truncate large outputs (~1k tokens max)
SANDBOX_MAX_OUTPUT_CHARS: int = int(SANDBOX_CFG.get("max_output_chars", 4000))
# Cache TTL for identical code results (ephemeral mode only)
SANDBOX_CACHE_TTL_SECONDS: int = int(SANDBOX_CFG.get("cache_ttl_seconds", 3600))
# Rate limit per minute
SANDBOX_RATE_LIMIT_PER_MINUTE: int = int(SANDBOX_CFG.get("rate_limit_per_minute", 30))

# Environment variable overrides for E2B Sandbox
SANDBOX_ENABLED = bool(int(os.getenv("SANDBOX_ENABLED", "1" if SANDBOX_ENABLED else "0")))
SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", str(SANDBOX_TIMEOUT_SECONDS)))

# --------------------------------------------------------------------
# Git Memory Configuration
# --------------------------------------------------------------------
# Populate PROCEDURAL memory with git commit history
GIT_MEMORY_CFG = config.get("git_memory", {})
GIT_MEMORY_ENABLED: bool = bool(GIT_MEMORY_CFG.get("enabled", True))
GIT_MEMORY_INCLUDE_DIFFS: bool = bool(GIT_MEMORY_CFG.get("include_diffs", False))
GIT_MEMORY_DEFAULT_LIMIT: int = int(GIT_MEMORY_CFG.get("default_limit", 200))

# Environment variable overrides for Git Memory
GIT_MEMORY_ENABLED = bool(int(os.getenv("GIT_MEMORY_ENABLED", "1" if GIT_MEMORY_ENABLED else "0")))
GIT_MEMORY_INCLUDE_DIFFS = bool(int(os.getenv("GIT_MEMORY_INCLUDE_DIFFS", "1" if GIT_MEMORY_INCLUDE_DIFFS else "0")))
GIT_MEMORY_DEFAULT_LIMIT = int(os.getenv("GIT_MEMORY_DEFAULT_LIMIT", str(GIT_MEMORY_DEFAULT_LIMIT)))

# --------------------------------------------------------------------
# Procedural Skills Configuration
# --------------------------------------------------------------------
# Reusable problem-solving patterns ("How-To" memory)
SKILLS_CFG = config.get("procedural_skills", {})
PROCEDURAL_SKILLS_ENABLED: bool = bool(SKILLS_CFG.get("enabled", True))
PROMPT_MAX_SKILLS: int = int(SKILLS_CFG.get("prompt_max_skills", 5))
SKILL_DEDUP_THRESHOLD: float = float(SKILLS_CFG.get("dedup_threshold", 0.85))

# Environment variable overrides for Procedural Skills
PROCEDURAL_SKILLS_ENABLED = bool(int(os.getenv("PROCEDURAL_SKILLS_ENABLED", "1" if PROCEDURAL_SKILLS_ENABLED else "0")))
PROMPT_MAX_SKILLS = int(os.getenv("PROMPT_MAX_SKILLS", str(PROMPT_MAX_SKILLS)))
SKILL_DEDUP_THRESHOLD = float(os.getenv("SKILL_DEDUP_THRESHOLD", str(SKILL_DEDUP_THRESHOLD)))

# --------------------------------------------------------------------
# Code Proposals Configuration
# --------------------------------------------------------------------
# Goal-directed code change proposals generated from project analysis
PROPOSALS_CFG = config.get("code_proposals", {})
CODE_PROPOSALS_ENABLED: bool = bool(PROPOSALS_CFG.get("enabled", True))
CODE_PROPOSALS_COLLECTION: str = PROPOSALS_CFG.get("collection", "proposals")
CODE_PROPOSALS_DEDUP_THRESHOLD: float = float(PROPOSALS_CFG.get("dedup_threshold", 0.70))
CODE_PROPOSALS_MAX_PER_SESSION: int = int(PROPOSALS_CFG.get("max_per_session", 5))
CODE_PROPOSALS_REQUIRE_TESTS: bool = bool(PROPOSALS_CFG.get("require_tests", True))

# Prompt integration: surface proposals in [PROPOSED FEATURES] section
CODE_PROPOSALS_PROMPT_ENABLED: bool = bool(PROPOSALS_CFG.get("prompt_enabled", True))
CODE_PROPOSALS_PROMPT_MAX: int = int(PROPOSALS_CFG.get("prompt_max", 3))
CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD: float = float(PROPOSALS_CFG.get("keyword_dedup_tag_threshold", 0.60))
CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD: float = float(PROPOSALS_CFG.get("semantic_dedup_threshold", 0.75))
# LLM pairwise ranking: tournament-bracket comparison of top candidates
# Heavier (~1-2s) but far more accurate than pure semantic match
CODE_PROPOSALS_LLM_RANKING: bool = bool(PROPOSALS_CFG.get("llm_ranking", False))
CODE_PROPOSALS_LLM_RANKING_MODEL: str = PROPOSALS_CFG.get("llm_ranking_model", "gpt-4o-mini")
# Composite score weights (sum to 1.0)
CODE_PROPOSALS_WEIGHT_PRIORITY: float = float(PROPOSALS_CFG.get("weight_priority", 0.30))
CODE_PROPOSALS_WEIGHT_BREADTH: float = float(PROPOSALS_CFG.get("weight_breadth", 0.20))
CODE_PROPOSALS_WEIGHT_RECENCY: float = float(PROPOSALS_CFG.get("weight_recency", 0.10))
CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT: float = float(PROPOSALS_CFG.get("weight_goal_alignment", 0.40))

# Environment variable overrides for Code Proposals
CODE_PROPOSALS_ENABLED = bool(int(os.getenv("CODE_PROPOSALS_ENABLED", "1" if CODE_PROPOSALS_ENABLED else "0")))
CODE_PROPOSALS_DEDUP_THRESHOLD = float(os.getenv("CODE_PROPOSALS_DEDUP_THRESHOLD", str(CODE_PROPOSALS_DEDUP_THRESHOLD)))
CODE_PROPOSALS_MAX_PER_SESSION = int(os.getenv("CODE_PROPOSALS_MAX_PER_SESSION", str(CODE_PROPOSALS_MAX_PER_SESSION)))
CODE_PROPOSALS_PROMPT_ENABLED = bool(int(os.getenv("CODE_PROPOSALS_PROMPT_ENABLED", "1" if CODE_PROPOSALS_PROMPT_ENABLED else "0")))
CODE_PROPOSALS_PROMPT_MAX = int(os.getenv("CODE_PROPOSALS_PROMPT_MAX", str(CODE_PROPOSALS_PROMPT_MAX)))
CODE_PROPOSALS_LLM_RANKING = bool(int(os.getenv("CODE_PROPOSALS_LLM_RANKING", "1" if CODE_PROPOSALS_LLM_RANKING else "0")))

# --------------------------------------------------------------------
# Escalation Tracker Configuration
# --------------------------------------------------------------------
# Adaptive tone de-escalation with session momentum tracking.
# Tracks consecutive crisis/elevated messages and adapts response strategy
# to prevent therapeutic echo chamber (repeating identical validations).
ESCALATION_CFG = config.get("escalation_tracker", {})
ESCALATION_ENABLED: bool = bool(ESCALATION_CFG.get("enabled", True))
# Consecutive elevated/crisis messages before shifting from VALIDATE_AND_SUGGEST
ESCALATION_THRESHOLD: int = int(ESCALATION_CFG.get("threshold", 3))
# Consecutive calm messages before GENTLE_REENGAGEMENT ends
ESCALATION_DEESCALATION_WINDOW: int = int(ESCALATION_CFG.get("deescalation_window", 2))
# Sliding window size for tone history
ESCALATION_MAX_HISTORY: int = int(ESCALATION_CFG.get("max_history", 10))

# Environment variable overrides for Escalation Tracker
ESCALATION_ENABLED = bool(int(os.getenv("ESCALATION_ENABLED", "1" if ESCALATION_ENABLED else "0")))

# --------------------------------------------------------------------
# Cross-Collection Deduplication Configuration
# --------------------------------------------------------------------
# Unified dedup across facts, summaries, skills, and proposals.
# Detects near-duplicates across collection boundaries and resolves
# fact contradictions (same subject+predicate, different object).
CROSS_DEDUP_CFG = config.get("cross_dedup", {})
CROSS_DEDUP_ENABLED: bool = bool(CROSS_DEDUP_CFG.get("enabled", True))
CROSS_DEDUP_DUPLICATE_THRESHOLD: float = float(CROSS_DEDUP_CFG.get("duplicate_threshold", 0.92))
CROSS_DEDUP_CONTRADICTION_THRESHOLD: float = float(CROSS_DEDUP_CFG.get("contradiction_threshold", 0.85))
CROSS_DEDUP_MAX_DOCS_PER_COLLECTION: int = int(CROSS_DEDUP_CFG.get("max_docs_per_collection", 1000))
CROSS_DEDUP_ON_SHUTDOWN: bool = bool(CROSS_DEDUP_CFG.get("on_shutdown", True))
# Collections to scan for cross-duplicates
CROSS_DEDUP_COLLECTIONS: list = CROSS_DEDUP_CFG.get("collections", [
    "facts", "summaries", "procedural_skills", "proposals", "reflections",
])

# Environment variable overrides for Cross-Collection Dedup
CROSS_DEDUP_ENABLED = bool(int(os.getenv("CROSS_DEDUP_ENABLED", "1" if CROSS_DEDUP_ENABLED else "0")))
CROSS_DEDUP_DUPLICATE_THRESHOLD = float(os.getenv("CROSS_DEDUP_DUPLICATE_THRESHOLD", str(CROSS_DEDUP_DUPLICATE_THRESHOLD)))
CROSS_DEDUP_CONTRADICTION_THRESHOLD = float(os.getenv("CROSS_DEDUP_CONTRADICTION_THRESHOLD", str(CROSS_DEDUP_CONTRADICTION_THRESHOLD)))
CROSS_DEDUP_ON_SHUTDOWN = bool(int(os.getenv("CROSS_DEDUP_ON_SHUTDOWN", "1" if CROSS_DEDUP_ON_SHUTDOWN else "0")))

# --------------------------------------------------------------------
# Truth Scorer Configuration
# --------------------------------------------------------------------
# Evidence-based truth scoring with time decay.
# Replaces the old access-count echo chamber with decay-toward-uncertainty.
TRUTH_SCORER_CFG = config.get("truth_scorer", {})
TRUTH_SCORER_ENABLED: bool = bool(TRUTH_SCORER_CFG.get("enabled", True))
TRUTH_SCORER_INITIAL_SCORE: float = float(TRUTH_SCORER_CFG.get("initial_score", 0.7))
TRUTH_SCORER_CONFIRMED_BOOST: float = float(TRUTH_SCORER_CFG.get("confirmed_boost", 0.08))
TRUTH_SCORER_CORRECTION_PENALTY: float = float(TRUTH_SCORER_CFG.get("correction_penalty", 0.25))
TRUTH_SCORER_CONTRADICTION_PENALTY: float = float(TRUTH_SCORER_CFG.get("contradiction_penalty", 0.15))
TRUTH_SCORER_DECAY_RATE: float = float(TRUTH_SCORER_CFG.get("decay_rate_per_week", 0.02))
TRUTH_SCORER_DECAY_FLOOR: float = float(TRUTH_SCORER_CFG.get("decay_floor", 0.3))
TRUTH_SCORER_CORRECTION_DETECTION: bool = bool(TRUTH_SCORER_CFG.get("correction_detection", True))
TRUTH_SCORER_CONFIRMATION_DETECTION: bool = bool(TRUTH_SCORER_CFG.get("confirmation_detection", True))
# Source-based initial scores
TRUTH_SCORER_SOURCE_SCORES: dict = TRUTH_SCORER_CFG.get("source_scores", {
    "user_stated": 0.8, "corrected": 0.85, "llm_extracted": 0.7, "inferred": 0.5
})

# Environment variable overrides for Truth Scorer
TRUTH_SCORER_ENABLED = bool(int(os.getenv("TRUTH_SCORER_ENABLED", "1" if TRUTH_SCORER_ENABLED else "0")))

# --------------------------------------------------------------------
# User Profile Configuration
# --------------------------------------------------------------------
# Ephemeral relations that accumulate rapidly and should be pruned
PROFILE_CFG = config.get("user_profile", {})
PROFILE_EPHEMERAL_RELATIONS: list = PROFILE_CFG.get("ephemeral_relations", [
    # Mood / emotional state
    "current_activity", "current_feeling", "current_mood",
    "emotional_state", "current_state", "status", "recent_activity",
    "feeling", "feels",
    # Health / sleep
    "condition", "symptoms", "symptom", "current_condition",
    "current_health_status", "current_health_condition", "recent_condition",
    "recent_symptom", "medication_status", "medications_taken",
    "medications_taken_time", "medication_taken", "medication_time",
    "sleep_condition", "sleep_experience", "sleep_quality",
    "woke_up_time", "wake_up_time", "woke_up_at", "sleep_duration",
    # Work / schedule
    "work_status", "work_activity", "work_in", "work_hours_left",
    "work_duration", "time_until_work", "time_constraint",
    "work_start_time", "worked_hours_today", "work_hours_today",
    "workout_status", "workout_intent", "last_workout_time",
    "took_nap", "current_time", "upcoming_activity",
    "waiting_time", "note_taking_time",
    # Greetings / expressions (time-of-day context)
    "greeting", "expressed_feeling", "testing",
    # Generic contextual predicates (change every conversation)
    "is", "is_a", "has", "was",
    "thinks", "needs", "plans", "wants",
    "likes", "broke", "agree", "completed",
    "asks_about", "ask_about", "goal",
])
# Max historical (is_current=False) entries to keep per ephemeral relation
PROFILE_EPHEMERAL_MAX_HISTORY: int = int(PROFILE_CFG.get("ephemeral_max_history", 20))
# TTL in hours for ephemeral facts — stale ephemeral facts excluded from prompt context
PROFILE_EPHEMERAL_TTL_HOURS: int = int(PROFILE_CFG.get("ephemeral_ttl_hours", 24))
# Soft cap on total facts per category before pruning triggers
PROFILE_CATEGORY_SOFT_CAP: int = int(PROFILE_CFG.get("category_soft_cap", 200))

# Environment variable overrides for User Profile
PROFILE_EPHEMERAL_MAX_HISTORY = int(os.getenv("PROFILE_EPHEMERAL_MAX_HISTORY", str(PROFILE_EPHEMERAL_MAX_HISTORY)))
PROFILE_EPHEMERAL_TTL_HOURS = int(os.getenv("PROFILE_EPHEMERAL_TTL_HOURS", str(PROFILE_EPHEMERAL_TTL_HOURS)))
PROFILE_CATEGORY_SOFT_CAP = int(os.getenv("PROFILE_CATEGORY_SOFT_CAP", str(PROFILE_CATEGORY_SOFT_CAP)))
ESCALATION_THRESHOLD = int(os.getenv("ESCALATION_THRESHOLD", str(ESCALATION_THRESHOLD)))
ESCALATION_DEESCALATION_WINDOW = int(os.getenv("ESCALATION_DEESCALATION_WINDOW", str(ESCALATION_DEESCALATION_WINDOW)))
ESCALATION_MAX_HISTORY = int(os.getenv("ESCALATION_MAX_HISTORY", str(ESCALATION_MAX_HISTORY)))

# --------------------------------------------------------------------
# Obsidian Vault Configuration
# --------------------------------------------------------------------
# Enable personal notes integration from Obsidian vault
# Notes are embedded into ChromaDB and retrieved semantically
OBSIDIAN_CFG = config.get("obsidian", {})
OBSIDIAN_ENABLED: bool = bool(OBSIDIAN_CFG.get("enabled", True))
# Path to Obsidian vault directory
OBSIDIAN_VAULT_PATH: str = OBSIDIAN_CFG.get("vault_path", "") or os.path.expanduser("~/Documents/Luke Notes")
# Character threshold for chunking (notes < threshold = whole, >= threshold = chunk by headers)
OBSIDIAN_CHUNK_THRESHOLD: int = int(OBSIDIAN_CFG.get("chunk_threshold", 1500))
# Maximum notes to include in prompt
OBSIDIAN_MAX_NOTES_PROMPT: int = int(OBSIDIAN_CFG.get("max_notes_prompt", 5))

# Environment variable overrides for Obsidian
OBSIDIAN_ENABLED = bool(int(os.getenv("OBSIDIAN_ENABLED", "1" if OBSIDIAN_ENABLED else "0")))
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", OBSIDIAN_VAULT_PATH)

# Image loading for multimodal models
# When enabled, actual image data from notes will be loaded for multimodal-capable models
OBSIDIAN_INCLUDE_IMAGES: bool = bool(OBSIDIAN_CFG.get("include_images", True))
# Maximum images to load per note chunk
OBSIDIAN_MAX_IMAGES_PER_NOTE: int = int(OBSIDIAN_CFG.get("max_images_per_note", 3))
# Maximum total image data size in MB
OBSIDIAN_MAX_IMAGE_SIZE_MB: float = float(OBSIDIAN_CFG.get("max_image_size_mb", 10.0))

# Known multimodal models that can process images
# Format: partial model name/id patterns (case-insensitive matching)
MULTIMODAL_MODELS: list = OBSIDIAN_CFG.get("multimodal_models", [
    "opus-4", "claude-3", "sonnet-4", "gpt-4o", "gpt-4-vision", "gpt-4-turbo",
    "gemini-pro", "gemini-1.5", "gemini-2", "llava", "qwen-vl", "pixtral"
])

# --------------------------------------------------------------------
# Reference Documents Configuration
# --------------------------------------------------------------------
# User-uploaded reference documents (technical docs, project outlines, etc.)
REFERENCE_DOCS_CFG = config.get("reference_docs", {})
REFERENCE_DOCS_ENABLED: bool = bool(REFERENCE_DOCS_CFG.get("enabled", True))
# Character threshold for chunking (docs < threshold = whole, >= threshold = chunk by headers)
REFERENCE_DOCS_CHUNK_THRESHOLD: int = int(REFERENCE_DOCS_CFG.get("chunk_threshold", 2000))
# Maximum document chunks to include in prompt
REFERENCE_DOCS_MAX_PROMPT: int = int(REFERENCE_DOCS_CFG.get("max_prompt", 5))

# Environment variable overrides for Reference Docs
REFERENCE_DOCS_ENABLED = bool(int(os.getenv("REFERENCE_DOCS_ENABLED", "1" if REFERENCE_DOCS_ENABLED else "0")))

# --------------------------------------------------------------------
# Daily Notes Configuration (auto-generated conversation summaries)
# --------------------------------------------------------------------
DAILY_NOTES_CFG = config.get("daily_notes", {})
DAILY_NOTES_ENABLED: bool = bool(DAILY_NOTES_CFG.get("enabled", True))
# Subfolder within Obsidian vault for daily notes
DAILY_NOTES_FOLDER: str = DAILY_NOTES_CFG.get("folder", "Daily")
# Model for generating daily summaries
DAILY_NOTES_MODEL: str = DAILY_NOTES_CFG.get("model", "sonnet-4.5")
# Max tokens for LLM response
DAILY_NOTES_MAX_TOKENS: int = int(DAILY_NOTES_CFG.get("max_tokens", 800))

# Environment variable overrides for Daily Notes
DAILY_NOTES_ENABLED = bool(int(os.getenv("DAILY_NOTES_ENABLED", "1" if DAILY_NOTES_ENABLED else "0")))

# Weekly Notes Configuration (extends daily notes)
WEEKLY_NOTES_ENABLED: bool = bool(DAILY_NOTES_CFG.get("weekly_enabled", True))
WEEKLY_NOTES_MODEL: str = DAILY_NOTES_CFG.get("weekly_model", "sonnet-4.5")
WEEKLY_NOTES_MAX_TOKENS: int = int(DAILY_NOTES_CFG.get("weekly_max_tokens", 1200))

# Monthly Notes Configuration (extends daily/weekly notes)
MONTHLY_NOTES_ENABLED: bool = bool(DAILY_NOTES_CFG.get("monthly_enabled", True))
MONTHLY_NOTES_MODEL: str = DAILY_NOTES_CFG.get("monthly_model", "sonnet-4.5")
MONTHLY_NOTES_MAX_TOKENS: int = int(DAILY_NOTES_CFG.get("monthly_max_tokens", 2000))

# Environment variable override for Monthly Notes
MONTHLY_NOTES_ENABLED = bool(int(os.getenv("MONTHLY_NOTES_ENABLED", "1" if MONTHLY_NOTES_ENABLED else "0")))

# Tag Generation Configuration (for daily/weekly notes and future .md memories)
TAG_GENERATION_CFG = config.get("tag_generation", {})
TAG_GENERATION_ENABLED: bool = bool(TAG_GENERATION_CFG.get("enabled", True))
TAG_GENERATION_MODEL: str = TAG_GENERATION_CFG.get("model", "sonnet-4.5")
TAG_GENERATION_MAX_TAGS: int = int(TAG_GENERATION_CFG.get("max_tags", 10))
TAG_GENERATION_MIN_TAGS: int = int(TAG_GENERATION_CFG.get("min_tags", 3))

# Environment variable override for Tag Generation
TAG_GENERATION_ENABLED = bool(int(os.getenv("TAG_GENERATION_ENABLED", "1" if TAG_GENERATION_ENABLED else "0")))

# --------------------------------------------------------------------
# Narrative Context (Temporal Grounding) Configuration
# Synthesizes weekly/monthly summaries into a rolling "Life State" narrative
# that provides trajectory-aware context without per-query latency costs.
# --------------------------------------------------------------------
NARRATIVE_CONTEXT_CFG = config.get("narrative_context", {})
NARRATIVE_CONTEXT_ENABLED: bool = bool(NARRATIVE_CONTEXT_CFG.get("enabled", True))
NARRATIVE_CONTEXT_PATH: str = os.getenv(
    "NARRATIVE_CONTEXT_PATH",
    NARRATIVE_CONTEXT_CFG.get("path", "./data/narrative_context.txt")
)
NARRATIVE_MAX_TOKENS: int = int(NARRATIVE_CONTEXT_CFG.get("max_tokens", 500))
NARRATIVE_WEEKLIES_COUNT: int = int(NARRATIVE_CONTEXT_CFG.get("weeklies_count", 3))
NARRATIVE_MONTHLIES_COUNT: int = int(NARRATIVE_CONTEXT_CFG.get("monthlies_count", 1))
NARRATIVE_DAILIES_COUNT: int = int(NARRATIVE_CONTEXT_CFG.get("dailies_count", 6))
NARRATIVE_SYNTHESIS_MODEL: str = NARRATIVE_CONTEXT_CFG.get("synthesis_model", "sonnet-4.5")

# Environment variable overrides for Narrative Context
NARRATIVE_CONTEXT_ENABLED = bool(int(os.getenv("NARRATIVE_CONTEXT_ENABLED", "1" if NARRATIVE_CONTEXT_ENABLED else "0")))
NARRATIVE_MAX_TOKENS = int(os.getenv("NARRATIVE_MAX_TOKENS", str(NARRATIVE_MAX_TOKENS)))
NARRATIVE_WEEKLIES_COUNT = int(os.getenv("NARRATIVE_WEEKLIES_COUNT", str(NARRATIVE_WEEKLIES_COUNT)))
NARRATIVE_MONTHLIES_COUNT = int(os.getenv("NARRATIVE_MONTHLIES_COUNT", str(NARRATIVE_MONTHLIES_COUNT)))
NARRATIVE_DAILIES_COUNT = int(os.getenv("NARRATIVE_DAILIES_COUNT", str(NARRATIVE_DAILIES_COUNT)))
NARRATIVE_SYNTHESIS_MODEL = os.getenv("NARRATIVE_SYNTHESIS_MODEL", NARRATIVE_SYNTHESIS_MODEL)

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

# STM (Short-Term Memory) Pass Configuration
# ------------------------------------------
# Enable multi-pass STM analysis: lightweight LLM pass to summarize recent context
# before main response generation
USE_STM_PASS = bool(config.get("features", {}).get("use_stm_pass", True))
STM_MODEL_NAME = config.get("features", {}).get("stm_model_name", "gpt-4o-mini")
STM_MAX_RECENT_MESSAGES = int(config.get("features", {}).get("stm_max_recent_messages", 8))
# Minimum conversation depth before STM kicks in (avoid overhead for trivial exchanges)
STM_MIN_CONVERSATION_DEPTH = int(config.get("features", {}).get("stm_min_conversation_depth", 3))
# Topic similarity threshold for STM topic-change detection (0.0-1.0)
# Below this threshold = true topic change, STM skipped to avoid contamination
# Uses semantic similarity (embeddings) instead of string matching
STM_TOPIC_SIMILARITY_THRESHOLD = float(config.get("features", {}).get("stm_topic_similarity_threshold", 0.4))

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

# Memory Citation System
# Enable tracking and display of memory provenance in responses
# When enabled, Claude cites which memories it references, and citations
# are displayed in a separate tab (toggleable via GUI checkbox)
ENABLE_MEMORY_CITATIONS = bool(config.get("features", {}).get("enable_memory_citations", False))
MAX_CITATIONS_DISPLAY = int(config.get("features", {}).get("max_citations_display", 10))
CITATION_CONTENT_LENGTH = int(config.get("features", {}).get("citation_content_length", 200))

# Soft latency budget for best-of reranking before falling back to streaming
BEST_OF_LATENCY_BUDGET_S = float(config.get("features", {}).get("best_of_latency_budget_s", 2.0))

# --------------------------------------------------------------------
# File Upload Security Configuration (Added 2025-11-30)
# --------------------------------------------------------------------
# Maximum file size per individual file (10MB default)
FILE_UPLOAD_MAX_SIZE = int(config.get("security", {}).get("file_upload_max_size", 10 * 1024 * 1024))
# Maximum total size across all files in a single request (50MB default)
FILE_UPLOAD_MAX_TOTAL_SIZE = int(config.get("security", {}).get("file_upload_max_total_size", 50 * 1024 * 1024))
# Allowed file extensions for upload
FILE_UPLOAD_ALLOWED_EXTENSIONS = list(config.get("security", {}).get("file_upload_allowed_extensions", ['.txt', '.docx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']))
# CSV formula prefixes to escape (prevent formula injection)
FILE_UPLOAD_CSV_FORMULA_PREFIXES = tuple(config.get("security", {}).get("file_upload_csv_formula_prefixes", ['=', '+', '-', '@', '\t', '\r', '\n']))

# Directory for persisted upload images (created on demand)
FILE_UPLOAD_IMAGE_DIR = str(config.get("paths", {}).get("upload_image_dir", "data/uploads"))
# Maximum user uploads to surface in prompt
PROMPT_MAX_USER_UPLOADS = int(config.get("memory", {}).get("prompt_max_user_uploads", 5))

# Environment variable overrides for file upload security
FILE_UPLOAD_MAX_SIZE = int(os.getenv("FILE_UPLOAD_MAX_SIZE", FILE_UPLOAD_MAX_SIZE))
FILE_UPLOAD_MAX_TOTAL_SIZE = int(os.getenv("FILE_UPLOAD_MAX_TOTAL_SIZE", FILE_UPLOAD_MAX_TOTAL_SIZE))

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
# Intent Classifier Configuration
# --------------------------------------------------------------------
# Fast regex-first query intent classification (no LLM calls).
# Classifies queries into categorical intents that drive downstream
# retrieval counts, scoring weights, and gating thresholds.
INTENT_CFG = config.get("intent_classifier", {})
INTENT_ENABLED: bool = bool(INTENT_CFG.get("enabled", True))
# STM refinement threshold: below this confidence, STM free-text intent
# can upgrade the classification (no extra LLM call — STM already ran)
INTENT_STM_REFINEMENT_THRESHOLD: float = float(INTENT_CFG.get("stm_refinement_threshold", 0.50))

# Environment variable overrides for Intent Classifier
INTENT_ENABLED = bool(int(os.getenv("INTENT_ENABLED", "1" if INTENT_ENABLED else "0")))

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
