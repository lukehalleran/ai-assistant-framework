"""
# core/prompt/builder.py

Module Contract
- Purpose: Main UnifiedPromptBuilder orchestrating context retrieval, token budget
  management, and prompt assembly coordination. Delegates formatting to PromptFormatter
  and hygiene to ContentHygiene.
- Key methods:
  - build_prompt(user_input, config, context_result, ...) -> Dict
    Main entry point: parallel retrieval → hygiene → token budget → returns context dict.
    Sets/clears intent weight overrides and graph refs on scorer around retrieval.
  - build_prompt_from_context(user_input, config, context_result, ...) -> Dict
    Lightweight path skipping full retrieval (uses pre-gathered context).
  - _llm_compress_oversized(context) -> Dict
    Pre-compresses items ≥3x over token limit via LLM before middle-out fallback.
  - _assemble_prompt(context, user_input, directives, system_prompt) -> str
    Delegates to PromptFormatter._assemble_prompt().
  - _build_feature_inventory(context) -> str
    Delegates to PromptFormatter._build_feature_inventory().
  - _hygiene_and_caps(context, stm_summary) -> Dict
    Delegates to ContentHygiene._hygiene_and_caps().
  - _backfill_recent_conversations(...) -> List
    Delegates to ContentHygiene._backfill_recent_conversations().
  - Post-budget floors (Step 7.1): Guarantees minimum recent_conversations (PROMPT_MIN_RECENT_FLOOR=5),
    recent_summaries (PROMPT_MIN_SUMMARIES_FLOOR=2), and recent_reflections
    (PROMPT_MIN_REFLECTIONS_FLOOR=1) survive budget trimming. 2026-08-14: floors target the
    RENDERED split keys (the combined summaries/reflections keys are never rendered — the old
    floors topped up dead keys with retrieval calls whose output never reached the prompt) and
    are survival MINIMUMS, not restore-to-max (which would undo the trim).
  - Skill activation (Step 5): Creates SkillActivationPolicy in __init__, over-fetches procedural
    skills by SKILL_ACTIVATION_FETCH_MULTIPLIER (3x), applies policy after parallel gather completes
    (intent suppression, score threshold, STM bonus, cooldown filter, cap to max_skills).
  - Visual memory gating: passes `intent_type` into `get_visual_memories()` so image retrieval is
    gated by a visual/recall intent signal (a "show me"-type request), not a bare entity-name match
    (see gatherer_knowledge._query_wants_visual).
  - Proactive-insights distress gate (2026-08-05): _distress_active also suppresses the
    proactive_insights task — speculative cross-domain suggestions were landing in
    distress-adjacent conversations (same class as the reference-docs gate below).
  - Reference-docs ALLOW-gate (2026-08-05): _should_include_reference_docs() — self-docs
    surface only on meta_conversational/technical_help/project_work intents or a
    self-referential query cue (_SELF_REFERENTIAL_CUE_RE: "daemon", "your memory",
    "how do you score", ...); a conversational-tone personal pain turn had pulled 15
    [DAEMON DOCUMENTATION] chunks because the docs semantically match emotional language.
    The 2026-07-25 suppressions below always win.
  - Reference-docs distress gate (2026-07-25): _should_suppress_reference_docs() drops the
    [DAEMON DOCUMENTATION] self-docs task on distress/emotional_support turns — Daemon's own
    tone-detection docs semantically match distress language and were leaking crisis keyword
    lists + response-length rules into distress prompts.
- Outputs:
  - Context dictionary with all assembled data, metadata, and performance metrics
  - Formatted prompt string via _assemble_prompt delegation
- Dependencies:
  - .context_gatherer.ContextGatherer (parallel async data retrieval)
  - .formatter.PromptFormatter (section assembly, feature inventory, moved module-level helpers)
  - .hygiene.ContentHygiene (dedup, caps, backfill)
  - .summarizer.LLMSummarizer (on-demand reflections and summaries)
  - .token_manager.TokenManager (budget enforcement, middle-out compression)
  - .base._FallbackMemoryCoordinator (testing fallback)
  - memory.skill_activation.SkillActivationPolicy (post-retrieval skill filter)
  - memory.skill_activation.SkillCooldownStore (JSON-backed cooldown tracking)
  - processing.gate_system (relevance filtering)
  - memory.memory_scorer (intent weight overrides, graph refs set/cleared per call)
- Re-exports from .formatter (backward compatibility):
  - _staleness_prefix, _is_multimodal_model, _load_upload_image
- Side effects:
  - Memory system queries and parallel data retrieval
  - LLM API calls for summarization, reflection, and oversized item compression
  - Sets/clears _intent_weight_overrides and _graph_memory/_entity_resolver on scorer
  - Comprehensive logging and performance metrics
"""

import os
import re
import time
import asyncio
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context_pipeline import ContextResult
from datetime import datetime
from utils.time_manager import TimeManager
from utils.query_checker import (
    analyze_query, is_anaphoric_continuation, is_fragment_continuation,
    is_retry_continuation,
)
from memory.memory_consolidator import MemoryConsolidator
from utils.logging_utils import get_logger

# Import the modular components
from .context_gatherer import (
    ContextGatherer,
    PROMPT_MAX_RECENT_SUMMARIES,
    PROMPT_MAX_SEMANTIC_SUMMARIES,
    PROMPT_MAX_RECENT_REFLECTIONS,
    PROMPT_MAX_SEMANTIC_REFLECTIONS
)
from .formatter import (
    PromptFormatter, _parse_bool, _dedupe_keep_order, _sanitize_embedded_headers,
    _staleness_prefix, _is_multimodal_model, _load_upload_image,
)
from .summarizer import LLMSummarizer
from .token_manager import TokenManager, USER_PROFILE_MAX_TOKENS
from .base import _FallbackMemoryCoordinator
from .hygiene import ContentHygiene
from memory.skill_activation import SkillActivationPolicy, SkillCooldownStore
import hashlib as _hashlib
from utils.ordered_slice import newest_first as _ordered_newest_first

logger = get_logger("prompt_builder")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError) as e:
    logger.warning(f"[PromptBuilder] Could not load memory config: {e}, using defaults")
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError) as e:
        logger.debug(f"[PromptBuilder] Bad config value for '{key}': {e}, using default {default_val}")
        return int(default_val)

# Token and model configuration
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
RESERVE_FOR_COMPLETION = int(os.getenv("RESERVE_FOR_COMPLETION", "1024"))

# Model-aware token budget (replaces static PROMPT_TOKEN_BUDGET = 15000)
try:
    from config.app_config import (
        PROMPT_TOKEN_BUDGET_OVERRIDE,
        PROMPT_TOKEN_BUDGET_DEFAULT,
        PROMPT_TOKEN_BUDGET_LOCAL,
        PROMPT_TOKEN_BUDGET_FLOOR,
        PROMPT_TOKEN_BUDGET_CEILING,
        PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION,
    )
except ImportError:
    PROMPT_TOKEN_BUDGET_OVERRIDE = None
    PROMPT_TOKEN_BUDGET_DEFAULT = 40000
    PROMPT_TOKEN_BUDGET_LOCAL = 12000
    PROMPT_TOKEN_BUDGET_FLOOR = 8000
    PROMPT_TOKEN_BUDGET_CEILING = 60000
    PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION = 0.25

# LLM compression config (smart compression for heavily oversized items)
try:
    from config.app_config import (
        LLM_COMPRESSION_ENABLED,
        LLM_COMPRESSION_MODEL,
        LLM_COMPRESSION_TIMEOUT,
        LLM_COMPRESSION_RATIO_THRESHOLD,
        LLM_COMPRESSION_MAX_BATCH,
    )
except ImportError:
    LLM_COMPRESSION_ENABLED = False
    LLM_COMPRESSION_MODEL = "gpt-4o-mini"
    LLM_COMPRESSION_TIMEOUT = 3.0
    LLM_COMPRESSION_RATIO_THRESHOLD = 3.0
    LLM_COMPRESSION_MAX_BATCH = 8


# ---------------------------------------------------------------------------
# Eval snapshot hook (gated, read-only, disabled by default)
# ---------------------------------------------------------------------------

def _eval_capture_enabled() -> bool:
    """Check if eval snapshot capture is enabled via environment variable."""
    return os.environ.get("DAEMON_EVAL_CAPTURE", "0") == "1"


def _eval_capture_strict() -> bool:
    """Check if eval capture should raise on errors (vs log warnings)."""
    return os.environ.get("DAEMON_EVAL_CAPTURE_STRICT", "0") == "1"


def _maybe_capture_eval_snapshot(
    context: Dict[str, Any],
    user_input: str,
    sections: list,
    final_prompt: str,
) -> None:
    """Gated eval snapshot hook. Does nothing unless DAEMON_EVAL_CAPTURE=1.

    This function is read-only: it does not mutate context, sections, or prompt.
    It captures the post-hygiene assembled prompt for eval replay and saves it
    to disk. Failures log warnings but do not break normal chat (unless strict mode).
    """
    if not _eval_capture_enabled():
        return

    try:
        # Lazy import to avoid loading eval modules during normal operation
        from eval.snapshots import SnapshotCapture, save_snapshot
        from eval.schema import PromptProvenance
        from eval.section_registry import match_header_to_key
        from datetime import datetime, timezone
        import subprocess

        # Build formatted_sections map from the sections list
        formatted_sections: Dict[str, str] = {}
        for section_text in sections:
            if not section_text:
                continue
            first_line = section_text.split("\n", 1)[0]
            key = match_header_to_key(first_line)
            if key:
                formatted_sections[key] = section_text
            else:
                # Try to detect [CURRENT USER QUERY] which has a nested structure
                if "[CURRENT USER QUERY]" in first_line:
                    formatted_sections["current_query"] = section_text

        # Build provenance
        git_hash = ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

        provenance = PromptProvenance(
            model_name="",  # Not available in builder context
            git_commit_hash=git_hash,
            system_prompt_hash="",  # System prompt is in orchestrator
            capture_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Capture post_hygiene layer only (raw_retrieval would need pre-hygiene context)
        capture = SnapshotCapture()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context=context,
            formatted_sections=formatted_sections,
            prompt_text=final_prompt,
        )

        # Build minimal snapshot (single layer from builder hook)
        import uuid
        from eval.schema import PromptSnapshot

        snapshot = PromptSnapshot(
            snapshot_id=str(uuid.uuid4())[:8],
            query_text=user_input,
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            processed_query=user_input,
            detected_intent="",
            detected_tone="",
            provenance=provenance,
            layers={"post_hygiene": layer},
            retrieval_metadata={},
            assembly_metadata={"section_count": len(sections)},
        )

        save_snapshot(snapshot)
        logger.info(f"[EVAL] Snapshot captured: {snapshot.snapshot_id} ({len(formatted_sections)} sections)")

    except Exception as e:
        if _eval_capture_strict():
            raise
        logger.warning(f"[EVAL] Snapshot capture failed (non-fatal): {e}")


def _compute_token_budget(model_manager) -> int:
    """Compute prompt token budget based on model context window.

    Priority: env-var override > model-aware fraction > default.
    """
    # 1. Explicit env-var override (legacy compat: PROMPT_TOKEN_BUDGET=15000)
    if PROMPT_TOKEN_BUDGET_OVERRIDE is not None:
        logger.info(f"[PromptBuilder] Token budget: {PROMPT_TOKEN_BUDGET_OVERRIDE} (env override)")
        return PROMPT_TOKEN_BUDGET_OVERRIDE

    # 2. No model_manager available — use default
    if model_manager is None:
        logger.info(f"[PromptBuilder] Token budget: {PROMPT_TOKEN_BUDGET_DEFAULT} (default, no model_manager)")
        return PROMPT_TOKEN_BUDGET_DEFAULT

    # 3. Model-aware computation
    try:
        ctx_limit = model_manager.get_context_limit()
        is_local = not model_manager.is_api_model(model_manager.get_active_model_name())

        raw = int(ctx_limit * PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION)

        if is_local:
            budget = max(PROMPT_TOKEN_BUDGET_FLOOR, min(raw, PROMPT_TOKEN_BUDGET_LOCAL))
        else:
            # The experiment-adopted default (token_budget.default, 10000 as of
            # the 2026-07-15 preregistered budget experiment) is the OPERATIVE
            # cap for API models — the context fraction may only lower the
            # budget below it (small-ctx models), never raise it above. Before
            # 2026-07-25 the fraction path capped only at the 16K ceiling, so
            # any model with ctx >= ~84K silently ran ~15360 — the losing
            # 15K experiment arm — and the adopted 10000 was dead config.
            budget = max(
                PROMPT_TOKEN_BUDGET_FLOOR,
                min(raw, PROMPT_TOKEN_BUDGET_DEFAULT, PROMPT_TOKEN_BUDGET_CEILING),
            )

        logger.info(
            f"[PromptBuilder] Token budget: {budget} "
            f"(model-aware, ctx={ctx_limit}, local={is_local})"
        )
        return budget
    except Exception as e:
        logger.warning(f"[PromptBuilder] Could not determine context limit: {e}, using default")
        return PROMPT_TOKEN_BUDGET_DEFAULT

# Content limits (aligned with ContextGatherer defaults and user expectations)
# - Recent conversations: 15
# - Relevant memories: 15 (semantic search results only)
# - Facts: 15 semantic + 15 recent
# - Summaries: 10 (hybrid)
# - Reflections: 10 (hybrid)
PROMPT_MAX_RECENT = _cfg_int("prompt_max_recent", 15)
PROMPT_MAX_MEMS = _cfg_int("prompt_max_mems", 15)
PROMPT_MAX_FACTS = _cfg_int("prompt_max_facts", 30)
PROMPT_MAX_RECENT_FACTS = _cfg_int("prompt_max_recent_facts", 30)
PROMPT_MAX_SUMMARIES = _cfg_int("prompt_max_summaries", 10)
PROMPT_MAX_REFLECTIONS = _cfg_int("prompt_max_reflections", 10)
PROMPT_MAX_DREAMS = _cfg_int("prompt_max_dreams", 3)
PROMPT_MAX_SEMANTIC = _cfg_int("prompt_max_semantic", 8)
PROMPT_MAX_WIKI = _cfg_int("prompt_max_wiki", 3)
# Intents where encyclopedic wiki world-knowledge cannot help. The semantic-chunks
# task hits the external 41M-row wiki FAISS index (up to SEM_TIMEOUT_S), so for
# personal/social/emotional/meta/temporal/project turns we skip it entirely rather
# than pay the latency for zero benefit. (Web search is gated on intent the same way.)
WIKI_SEMANTIC_SUPPRESS_INTENTS = frozenset({
    "emotional_support", "casual_social", "meta_conversational",
    "temporal_recall", "project_work",
})
USER_PROFILE_FACTS_PER_CATEGORY = _cfg_int("user_profile_facts_per_category", 3)
PROMPT_MAX_PERSONAL_NOTES = _cfg_int("prompt_max_personal_notes", 5)
PROMPT_MIN_RECENT_FLOOR = _cfg_int("prompt_min_recent_floor", 5)
# Post-budget survival floors for the rendered summary/reflection sections
# (2026-08-14): the budget can now trim recent_summaries/recent_reflections
# (they were unmetered dead keys before), so the Step-7.1 floors guarantee a
# minimum survives — but only a minimum; restoring to MAX would undo the trim.
PROMPT_MIN_SUMMARIES_FLOOR = _cfg_int("prompt_min_summaries_floor", 2)
PROMPT_MIN_REFLECTIONS_FLOOR = _cfg_int("prompt_min_reflections_floor", 1)
# (2026-08-29): the Step-6.1 memory top-up is likewise a survival MINIMUM —
# its filler is ungated recent conversations relabeled as [RELEVANT MEMORIES];
# refilling to the intent cap (30) defeated the gate's honest-few doctrine.
MEMORY_TOPUP_FLOOR = _cfg_int("prompt_memory_topup_floor", 3)


def select_floor_topup(stored, have_contents, needed):
    """Pick the NEWEST `needed` items from a floor re-fetch, skipping ones
    already in the prompt.

    The floors used to iterate `stored[::-1]`, assuming an oldest-first list —
    but get_summaries/get_reflections return newest-first, so the floor
    restored the OLDEST items in the fetch buffer (live 2026-08-27:
    [RECENT SUMMARIES] rendered July 26/28 while Aug 22-26 summaries sat
    unread; same class as the agentic digest-order inversion). Sorts by
    timestamp explicitly (utils.ordered_slice.newest_first — single source
    of truth for "sort by timestamp before slicing", 2026-09-04) so either
    input order works.
    """
    if needed <= 0:
        return []

    def _ts_key(item):
        return item.get("timestamp") if isinstance(item, dict) else None

    have = set(have_contents or ())
    add = []
    ordered = _ordered_newest_first(
        [s for s in (stored or []) if isinstance(s, dict)], _ts_key,
    )
    for s in ordered:
        content = (s.get("content") or "").strip()
        if content and content not in have:
            add.append(s)
            have.add(content)
        if len(add) >= needed:
            break
    return add

# _staleness_prefix, _is_multimodal_model, _load_upload_image moved to formatter.py
# Re-exported above via: from .formatter import _staleness_prefix, _is_multimodal_model, _load_upload_image


def _should_suppress_reference_docs(files_suppress: bool, distress_active: bool, intent_type=None) -> bool:
    """Whether to skip the self-docs (reference_docs) retrieval task.

    Self-docs serve meta/technical queries about Daemon itself. On distress or
    emotional-support turns they are suppressed (2026-07-25): the tone-detection
    docs' crisis keyword lists semantically match distress language, so doc
    chunks — including the model's own response-length rules — were retrieved
    into every distress prompt (token waste + meta-leak).
    """
    return bool(
        files_suppress
        or distress_active
        or str(intent_type or "").lower() == "emotional_support"
    )


# Self-docs are ALLOW-listed, not just distress-suppressed (2026-08-05): a
# conversational-tone personal pain turn pulled 15 [DAEMON DOCUMENTATION]
# chunks (tone-detection docs, synthesis notes, even a lecture transcript)
# because distress suppression didn't fire and the docs semantically match
# emotional language. Self-docs exist for meta/technical queries about Daemon
# itself — they surface only for those intents or an explicit self-referential
# query cue.
# Analysis-query bound (2026-09-04, homework-attachment turn audit item 2):
# context.processed_query is normally either the short raw user text (no
# rewrite fired) or an LLM-rewritten short query — but when a huge merged
# attachment blob's word count clears the query-rewrite floor without the
# rewrite firing for some other reason, it can still be the RAW merged
# blob (a 130K-token homework turn measured ~130K of that as one giant
# "query"). Every retrieval-adjacent consumer downstream of build_prompt's
# `user_input` (obsidian keyword search, web-search trigger, agentic gate,
# memory semantic search) scores against that same string; a query that
# large either can't embed meaningfully (models truncate at their own max
# sequence length) or pathologically saturates keyword/overlap scoring.
# When processed_query is this large AND the untouched original (pre-merge)
# user text is short, fall back to the short text for analysis purposes —
# the full content still reaches the model via context.file_context in
# build_full_prompt's `[CURRENT QUERY]` assembly, a separate code path.
ANALYSIS_QUERY_MAX_CHARS = 2000

_SELF_DOC_INTENTS = {"meta_conversational", "technical_help", "project_work"}
_SELF_REFERENTIAL_CUE_RE = re.compile(
    r"\b(?:daemon|agentic|chroma(?:db)?|"
    r"your\s+(?:memory|memories|code|codebase|prompt|prompts|architecture|"
    r"retrieval|scoring|pipeline|system|tools?|docs|documentation|gate|collections?|"
    r"updates?|fixes|changes|changelog)|"
    r"(?:fixes|updates?|changes)\s+(?:on|to|in)\s+you|"
    r"how\s+(?:do|does|did)\s+you\s+(?:work|remember|retrieve|score|decide|search)|"
    r"truth[_ ]scores?|knowledge\s+graph)\b",
    re.IGNORECASE,
)


def _should_include_reference_docs(
    query: str, intent_type=None, files_suppress: bool = False, distress_active: bool = False
) -> bool:
    """Allow-list gate for the self-docs (reference_docs) retrieval task.

    Order matters: the 2026-07-25 suppressions (file uploads, distress,
    emotional_support intent) always win; otherwise self-docs surface only
    when the turn is plausibly ABOUT Daemon — meta/technical/project intent
    or a self-referential cue in the query.
    """
    if _should_suppress_reference_docs(files_suppress, distress_active, intent_type):
        return False
    if str(intent_type or "").lower() in _SELF_DOC_INTENTS:
        return True
    return bool(_SELF_REFERENTIAL_CUE_RE.search(query or ""))


def _should_include_note_images(model_name: str, query: str, intent_type=None) -> bool:
    """Gate for attaching Obsidian note images to the multimodal call.

    Mirrors the visual-memory gate: an attached image reads to the model as
    "the user just showed me this", so beyond config + model support the QUERY
    must signal visual intent (_query_wants_visual). Without the gate, a course
    note surfacing on an unrelated turn shipped its embedded screenshot and the
    model narrated it as a topic pivot (the diet-problem incident, 2026-07-14).
    The "[N image(s) attached]" text indicator still renders either way, so the
    model can offer to look when it's actually relevant.
    """
    if not (OBSIDIAN_INCLUDE_IMAGES and _is_multimodal_model(model_name)):
        return False
    from .gatherer_knowledge import _query_wants_visual
    return _query_wants_visual(query or "", intent_type)
PROMPT_MAX_REFERENCE_DOCS = _cfg_int("prompt_max_reference_docs", 15)
PROMPT_MAX_GIT_COMMITS = _cfg_int("prompt_max_git_commits", 10)
PROMPT_MAX_SKILLS = _cfg_int("prompt_max_skills", 5)
PROMPT_MAX_PROPOSALS = _cfg_int("prompt_max_proposals", 3)
PROMPT_MAX_USER_UPLOADS = _cfg_int("prompt_max_user_uploads", 5)
PROMPT_MAX_GRAPH_SENTENCES = _cfg_int("prompt_max_graph_sentences", 12)
PROMPT_MAX_SURFACED_THREADS = _cfg_int("prompt_max_surfaced_threads", 3)
PROMPT_MAX_PROACTIVE_INSIGHTS = _cfg_int("prompt_max_proactive_insights", 2)
PROMPT_MAX_VISUAL_MEMORIES = _cfg_int("prompt_max_visual_memories", 3)

# Feature toggles
REFLECTIONS_ON_DEMAND = _parse_bool(os.getenv("REFLECTIONS_ON_DEMAND", "0"))  # Off by default — blocks prompt build with LLM call
# Keep broad by default so we don't drop historical reflections
REFLECTIONS_SESSION_FILTER = _parse_bool(os.getenv("REFLECTIONS_SESSION_FILTER", "0"))
REFLECTIONS_TOPUP = _parse_bool(os.getenv("REFLECTIONS_TOPUP", "1"))

# Obsidian image loading for multimodal models
try:
    from config.app_config import (
        OBSIDIAN_INCLUDE_IMAGES,
        OBSIDIAN_MAX_IMAGES_PER_NOTE,
        PERSONAL_NOTES_GATE_THRESHOLD,
        REFERENCE_DOCS_GATE_THRESHOLD,
    )
except ImportError:
    OBSIDIAN_INCLUDE_IMAGES = True
    OBSIDIAN_MAX_IMAGES_PER_NOTE = 3
    PERSONAL_NOTES_GATE_THRESHOLD = 0.45
    REFERENCE_DOCS_GATE_THRESHOLD = 0.40



# Priority order for token budget management
PRIORITY_ORDER = [
    ("recent_conversations", 7),
    ("semantic_chunks", 6),
    ("personal_notes", 6),  # User's Obsidian notes - high priority
    ("user_uploads", 6),    # User uploaded files/images - high priority
    ("reference_docs", 5),  # Reference documents (system docs, project outlines)
    ("git_commits", 5),     # Git commit history (procedural memory)
    ("procedural_skills", 5),  # Reusable problem-solving patterns
    ("proposed_features", 3),  # Code proposals (trimmed before core context)
    ("unresolved_threads", 4),  # Open threads for proactive surfacing
    ("proactive_insights", 3),  # Cross-domain insights from knowledge graph
    ("memories", 5),
    ("semantic_facts", 4),
    ("fresh_facts", 4),
    ("summaries", 3),
    ("reflections", 2),
    ("wiki", 1),
    ("dreams", 2),
]


def _is_local_repo_audit_query(query: str) -> bool:
    """Detect explicit local repository inspection/verification turns."""
    q = (query or "").lower()
    action = any(term in q for term in (
        "read-only audit", "readonly audit", "repository audit", "repo audit",
        "audit of /", "inspect the repo", "inspect the repository",
        "verify the repo", "verify the repository",
    ))
    repo_cue = any(term in q for term in (
        "repo", "repository", "git status", "branch", "head", "pytest",
        "test suite", "lint", ".github/", "/home/",
    ))
    return action and repo_cue


# Self-report retrieval trim (2026-09-06): a one-line first-person status
# update with no request ("I took my stimulant at 10 AM today and I'm just
# resting this afternoon, feels good honestly even though I got nothing
# done") built an 11K-token prompt — [RELEVANT MEMORIES] n=14, [USER PROFILE]
# n=60 facts, 5 personal notes, a stale upload, full narrative — because
# nothing keyed off the message's SHAPE (as opposed to intent, which
# classified general@low-confidence and applied no override) trimmed
# retrieval for it. Distress/heavy-topic turns are deliberately exempt: a
# self-report shape during a crisis or emotionally heavy conversation still
# needs full context for safety-relevant continuity — see
# _apply_self_report_trim.
SELF_REPORT_RETRIEVAL_TRIM = {
    "max_mems": 4,
    "max_summaries": 2,
    "max_personal_notes": 2,
    "max_user_uploads": 0,
    "max_wiki": 0,
    "max_semantic": 0,
    "max_reference_docs": 0,
    "max_profile_tokens": 800,
    # 2026-09-06 retest: on a one-line status update the weight was NOT the
    # profile — [PROJECT COMMIT HISTORY] n=5 cost 2.4K tokens and ten
    # [RECENT CONVERSATION] turns 4.8K. Commits are noise on a personal
    # self-report; six recent turns keep continuity.
    "max_git_commits": 0,
    "max_recent": 6,
}


def _distress_from_crisis_level(crisis_level: Optional[str]) -> bool:
    """True when `crisis_level` encodes any distress-adjacent tone. Accepts
    any encoding the caller may pass ("CrisisLevel.CONCERN", "light_support",
    "crisis_support", ...) — extracted from build_prompt's distress-flagging
    logic (originally inline at the valence-aware-retrieval gate) so a second
    call site (the self-report retrieval trim) shares the identical rule
    rather than re-deriving it."""
    _cl = (crisis_level or "").upper()
    return any(
        k in _cl for k in (
            "HIGH", "MEDIUM", "CONCERN", "CRISIS", "ELEVATED",
            "LIGHT_SUPPORT", "ELEVATED_SUPPORT", "CRISIS_SUPPORT",
        )
    )


def _apply_self_report_trim(
    ro: Dict[str, Any], user_input: str, crisis_level: Optional[str]
) -> Dict[str, Any]:
    """Trim retrieval-override counts for a first-person status-update turn
    that requests nothing (see SELF_REPORT_RETRIEVAL_TRIM). Pure: returns a
    NEW dict — `ro` is never mutated. Min-merge semantics: an override an
    intent already set LOWER than the trim value is left alone (the trim
    never raises a count), a key absent from `ro` takes the trim value
    outright, and a higher intent value is capped down to the trim value.
    No-op (returns `ro` unchanged) unless the message is self-report shaped
    (`utils.query_checker.is_self_report`), and even then only when the turn
    is neither distress (`_distress_from_crisis_level`) nor a heavy topic
    (`utils.query_checker._is_heavy_topic_heuristic`) — both need full
    context regardless of the terse, request-free surface shape.
    """
    from utils.query_checker import is_self_report, _is_heavy_topic_heuristic

    if not is_self_report(user_input):
        return ro
    if _distress_from_crisis_level(crisis_level):
        return ro
    if _is_heavy_topic_heuristic(user_input):
        return ro

    merged = dict(ro)
    for key, trim_value in SELF_REPORT_RETRIEVAL_TRIM.items():
        if key in merged:
            try:
                merged[key] = min(int(merged[key]), int(trim_value))
            except (TypeError, ValueError):
                merged[key] = trim_value
        else:
            merged[key] = trim_value
    logger.debug(
        "[BUILD_PROMPT] Self-report shape — trimming retrieval overrides: "
        f"{user_input[:60]!r}"
    )
    return merged


class UnifiedPromptBuilder:
    """
    Unified prompt builder that coordinates all prompt building functionality.

    This class orchestrates the entire prompt building process by:
    1. Gathering context from various sources (memories, facts, wiki, etc.)
    2. Managing token budgets and content prioritization
    3. Formatting and assembling the final prompt
    4. Providing LLM summarization capabilities
    """

    def __init__(self, memory_coordinator=None, model_manager=None, tokenizer_manager=None,
                 consolidator=None, time_manager=None, token_budget: int = None,
                 wiki_manager=None, topic_manager=None, gate_system=None, **kwargs):
        """
        Initialize the UnifiedPromptBuilder.

        Args:
            memory_coordinator: Coordinator for memory operations
            model_manager: Manager for LLM interactions
            tokenizer_manager: Manager for token counting
            consolidator: Memory consolidation manager
            time_manager: Time management utilities
            token_budget: Maximum tokens for prompt context (None = auto-compute from model)
        """
        # Core dependencies
        self.memory_coordinator = memory_coordinator or self._build_default_memory_coordinator()
        self.model_manager = model_manager
        self.tokenizer_manager = tokenizer_manager
        self.consolidator = consolidator or MemoryConsolidator(model_manager)
        self.time_manager = time_manager or TimeManager()

        # Additional managers (for backward compatibility)
        self.wiki_manager = wiki_manager
        self.topic_manager = topic_manager
        self.gate_system = gate_system

        # Token management — model-aware if token_budget not explicitly passed
        if token_budget is None:
            token_budget = _compute_token_budget(model_manager)
        self.token_budget = token_budget

        # LLM-compression result cache: content-hash → compressed text, or None
        # for a recorded timeout (don't retry — middle-out handles it). The same
        # oversized item (git_commits especially) recurs turn after turn; without
        # this, each turn re-paid the full compression timeout for an identical
        # blob (observed: flat 3s on 7/7 turns of a session, 100% timeout rate).
        self._llm_compress_cache: Dict[str, Optional[str]] = {}

        # Initialize modular components
        self.token_manager = TokenManager(
            model_manager=self.model_manager,
            tokenizer_manager=self.tokenizer_manager,
            token_budget=token_budget
        )

        self.context_gatherer = ContextGatherer(
            memory_coordinator=self.memory_coordinator,
            model_manager=self.model_manager,
            token_manager=self.token_manager,
            gate_system=self.gate_system,
            time_manager=self.time_manager
        )

        self.formatter = PromptFormatter(
            token_manager=self.token_manager,
            time_manager=self.time_manager
        )

        self.summarizer = LLMSummarizer(
            model_manager=self.model_manager,
            memory_coordinator=self.memory_coordinator
        )

        self._hygiene = ContentHygiene(
            memory_coordinator=self.memory_coordinator,
            context_gatherer=self.context_gatherer
        )

        # Skill activation policy (post-retrieval filtering + cooldown)
        try:
            from config.app_config import (
                SKILL_ACTIVATION_ENABLED, SKILL_ACTIVATION_MAX_SKILLS,
                SKILL_ACTIVATION_MIN_SCORE, SKILL_ACTIVATION_COOLDOWN_HOURS,
                SKILL_ACTIVATION_FETCH_MULTIPLIER, SKILL_ACTIVATION_STM_BONUS,
                SKILL_ACTIVATION_USE_STM,
            )
            self._skill_activation_policy = SkillActivationPolicy(
                cooldown_store=SkillCooldownStore(),
                min_score=SKILL_ACTIVATION_MIN_SCORE,
                cooldown_hours=SKILL_ACTIVATION_COOLDOWN_HOURS,
                max_skills=SKILL_ACTIVATION_MAX_SKILLS,
                stm_bonus=SKILL_ACTIVATION_STM_BONUS,
                enabled=SKILL_ACTIVATION_ENABLED,
            )
            self._skill_fetch_multiplier = SKILL_ACTIVATION_FETCH_MULTIPLIER
            self._skill_activation_use_stm = SKILL_ACTIVATION_USE_STM
        except ImportError:
            self._skill_activation_policy = None
            self._skill_fetch_multiplier = 1
            self._skill_activation_use_stm = False

        # State tracking
        self._prompt_token_usage = 0

    def _build_default_memory_coordinator(self):
        """Build a fallback memory coordinator if none provided."""
        logger.warning("No memory coordinator provided, using fallback")
        return _FallbackMemoryCoordinator()

    _LLM_COMPRESS_CACHE_MAX = 200

    def _store_compress_cache(self, key: str, value: Optional[str]) -> None:
        """Record a compression outcome; bounded FIFO to cap memory."""
        cache = self._llm_compress_cache
        if len(cache) >= self._LLM_COMPRESS_CACHE_MAX:
            # Drop the oldest half — simple, no per-entry bookkeeping.
            for k in list(cache.keys())[: self._LLM_COMPRESS_CACHE_MAX // 2]:
                del cache[k]
        cache[key] = value

    async def _llm_compress_oversized(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-pass: LLM-compress heavily oversized items before budget trimming.

        Only targets items >= ratio_threshold * max_tokens (default 3x).
        Mildly oversized items still handled by middle-out in token_manager.
        """
        if not LLM_COMPRESSION_ENABLED:
            return context
        if not self.model_manager or not hasattr(self.model_manager, 'generate_once'):
            return context

        from .token_manager import MEMORY_ITEM_MAX_TOKENS, SEMANTIC_ITEM_MAX_TOKENS, PRIORITY_ORDER as TM_PRIORITY_ORDER

        try:
            model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else "default"
        except Exception:
            model_name = "default"

        # Scan all list sections for heavily oversized items
        candidates = []  # (section_name, index, item, item_tokens, max_tokens)
        for name, _prio in TM_PRIORITY_ORDER:
            val = context.get(name)
            if not val or not isinstance(val, list):
                continue
            if name in ("stm_summary", "user_profile", "narrative_state"):
                continue

            max_tokens = MEMORY_ITEM_MAX_TOKENS if name == "memories" else SEMANTIC_ITEM_MAX_TOKENS
            threshold = max_tokens * LLM_COMPRESSION_RATIO_THRESHOLD

            # Items above this ratio are too large for the LLM to compress
            # within the timeout — middle-out handles them instantly and well.
            skip_threshold = max_tokens * LLM_COMPRESSION_RATIO_THRESHOLD * 2

            for i, item in enumerate(val):
                # Keep user/assistant provenance intact. These fields have
                # independent deterministic caps in TokenManager; a summary
                # of the pair must never be written into the user's query.
                if self.token_manager._is_conversation_item(item):
                    continue
                item_text = self.token_manager._extract_text(item)
                try:
                    t = self.token_manager.get_token_count(item_text, model_name)
                except Exception:
                    t = len(item_text.split())
                if t >= threshold:
                    if t >= skip_threshold:
                        logger.debug(
                            f"[LLM-COMPRESS] Skipping {name}[{i}]: {t} tokens "
                            f"(>{skip_threshold:.0f}), middle-out will handle"
                        )
                        continue
                    candidates.append((name, i, item, t, max_tokens))

        if not candidates:
            return context

        # Sort by ratio (largest first), cap at max_batch
        candidates.sort(key=lambda c: c[3] / c[4], reverse=True)
        candidates = candidates[:LLM_COMPRESSION_MAX_BATCH]

        logger.info(f"[LLM-COMPRESS] {len(candidates)} items queued for LLM compression")
        from utils.turn_progress import emit as _progress_emit
        _progress_emit(f"🗜️ Compressing {len(candidates)} oversized items…")

        # Build compression tasks
        async def _compress_one(section: str, idx: int, item, item_tokens: int, max_tok: int):
            item_text = self.token_manager._extract_text(item)

            # Cache lookup: identical content recurs across turns (git_commits
            # especially). A cached string is reused; a cached None records a
            # prior timeout — skip straight to middle-out instead of re-paying
            # the timeout every turn.
            # Key includes the compression target: the same text compressed
            # for a different budget must not reuse the other target's result.
            _key = _hashlib.sha256(
                f"{max_tok}:{item_text}".encode("utf-8", "replace")
            ).hexdigest()
            if _key in self._llm_compress_cache:
                _cached = self._llm_compress_cache[_key]
                if _cached is None:
                    logger.debug(
                        f"[LLM-COMPRESS] {section}[{idx}]: cached timeout — middle-out"
                    )
                    return None
                logger.info(f"[LLM-COMPRESS] {section}[{idx}]: cache hit")
                return (section, idx, _cached)

            target = max_tok
            prompt = (
                f"Compress the following text to approximately {target} tokens. "
                f"Preserve ALL key facts, names, dates, numbers, and decisions. "
                f"Output ONLY the compressed text, nothing else.\n\n"
                f"Text:\n{item_text}"
            )
            try:
                compressed = await asyncio.wait_for(
                    self.model_manager.generate_once(
                        prompt,
                        model_name=LLM_COMPRESSION_MODEL,
                        system_prompt="You are a precise text compressor. Output only the compressed text.",
                        max_tokens=target + 64,  # small buffer for token estimation mismatch
                        temperature=0.0,
                    ),
                    timeout=LLM_COMPRESSION_TIMEOUT,
                )
                if compressed and isinstance(compressed, str) and len(compressed.strip()) > 20:
                    try:
                        new_tokens = self.token_manager.get_token_count(compressed.strip(), model_name)
                    except Exception:
                        new_tokens = len(compressed.strip().split())
                    if new_tokens >= item_tokens:
                        logger.warning("[LLM-COMPRESS] Rejected non-shrinking result for %s[%s]", section, idx)
                        return None
                    logger.info(
                        f"[LLM-COMPRESS] {section}[{idx}]: {item_tokens}→{new_tokens} tokens (LLM)"
                    )
                    self._store_compress_cache(_key, compressed.strip())
                    return (section, idx, compressed.strip())
            except asyncio.TimeoutError:
                logger.warning(f"[LLM-COMPRESS] Timeout compressing {section}[{idx}], falling back to middle-out")
                # Deterministic for identical content — record so the next turn
                # doesn't wait out the same timeout again.
                self._store_compress_cache(_key, None)
            except Exception as e:
                # Transient failures (API errors) are NOT cached — retry next turn.
                logger.warning(f"[LLM-COMPRESS] Failed {section}[{idx}]: {e}, falling back to middle-out")
            return None

        # Fire all compressions in parallel
        tasks = [
            _compress_one(section, idx, item, item_tokens, max_tok)
            for section, idx, item, item_tokens, max_tok in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Apply successful compressions back into context
        for result in results:
            if result is None or isinstance(result, Exception):
                continue
            section, idx, compressed_text = result
            items = context.get(section)
            if items and isinstance(items, list) and idx < len(items):
                original = items[idx]
                if isinstance(original, dict):
                    updated = dict(original)
                    text_key = self.token_manager._text_key(original)
                    if text_key is None:
                        continue
                    updated[text_key] = compressed_text
                    items[idx] = updated
                else:
                    items[idx] = compressed_text

        return context

    async def build_prompt(self, user_input: str, config: Optional[Dict[str, Any]] = None,
                          search_query: Optional[str] = None, personality_config: Optional[Dict[str, Any]] = None,
                          system_prompt: Optional[str] = None, current_topic: Optional[str] = None,
                          fresh_facts: Optional[List[Any]] = None, memories: Optional[List[Any]] = None,
                          stm_summary: Optional[Dict[str, Any]] = None,
                          crisis_level: Optional[str] = None,
                          retrieval_overrides: Optional[Dict[str, int]] = None,
                          weight_overrides: Optional[Dict[str, float]] = None,
                          intent_type: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Build a complete prompt context for the given user input.

        This is the main entry point for prompt building. It gathers context
        from all sources, applies token budget management, and returns a
        structured context dict ready for formatting.

        Args:
            user_input: The user's query/input
            config: Optional configuration overrides
            crisis_level: Current crisis level (HIGH/MEDIUM suppresses web search)
            retrieval_overrides: Optional dict of {max_*: count} to override
                global PROMPT_MAX_* constants. Used by intent classifier.
                `max_profile_tokens` overrides USER_PROFILE_MAX_TOKENS (the
                [USER PROFILE] section's token budget) — live since 2026-09-06,
                also set by the self-report retrieval trim (see
                SELF_REPORT_RETRIEVAL_TRIM / _apply_self_report_trim).
            weight_overrides: Optional dict of {weight_name: value} to override
                global SCORE_WEIGHTS. Used by intent classifier.

        Returns:
            Dict containing the built prompt context with sections like:
            - recent_conversations
            - memories
            - facts
            - fresh_facts
            - summaries
            - reflections
            - wiki
            - semantic_chunks
            - dreams
            - web_search_results (if triggered)
        """
        start_time = time.perf_counter()
        phase_timings = {}
        config = config or {}

        # Clear memory_id_map at start of each query to prevent memory leaks
        if hasattr(self.context_gatherer, 'clear_memory_id_map'):
            self.context_gatherer.clear_memory_id_map()

        logger.info(f"Building prompt for user input: {len(user_input)} chars")

        try:
            # Pre-fork: Detect first message + gather codebase changes BEFORE small-talk check
            # This ensures even "Yo" gets codebase change awareness.
            is_first_message = False
            codebase_changes = {}
            if self.time_manager:
                try:
                    gap = self.time_manager.time_since_previous_message()
                    is_first_message = isinstance(gap, str) and "N/A" in gap
                except (AttributeError, TypeError):
                    pass
            if is_first_message:
                _since_dt = getattr(self.time_manager, 'last_session_end_time', None)
                try:
                    codebase_changes = await self.context_gatherer.get_codebase_changes(_since_dt)
                except Exception as e:
                    logger.debug(f"[BUILD_PROMPT] Codebase changes failed: {e}")

            # Step 1: Analyze the query
            query_analysis = {}
            try:
                query_analysis = analyze_query(user_input)
                logger.debug(f"Query analysis: {query_analysis}")
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")

            # Check if this is small-talk that doesn't need heavy retrieval.
            # NOTE: QueryAnalysis.is_small_talk was never set before 2026-07-15
            # (dead getattr default), so this path never fired — a 7-word
            # "Hmm not working yet" pulled a 23K-token full-apparatus prompt.
            # Config-gated (light_prompt.enabled); crisis tone always gets the
            # full context regardless of message shape.
            if self._should_use_light_path(query_analysis, crisis_level):
                logger.info(
                    "[BUILD_PROMPT] Casual acknowledgment — using lightweight context "
                    f"(light_prompt path): {user_input[:60]!r}"
                )
                return await self._build_lightweight_context(user_input, stm_summary=stm_summary, codebase_changes=codebase_changes)

            # Continuation-answer: a short reply to the assistant's own prior
            # question ("What was the error?" -> "amplification"). Interpret it
            # from the immediate exchange, not corpus-wide semantic retrieval,
            # which otherwise pulls a topically-matching but unrelated memory
            # that hijacks the response. Heavy/crisis topics keep full context.
            if self._is_continuation_answer(user_input, query_analysis):
                logger.info(
                    "[BUILD_PROMPT] Continuation-answer to prior question — using "
                    f"lightweight context (no corpus retrieval): {user_input[:60]!r}"
                )
                return await self._build_lightweight_context(user_input, stm_summary=stm_summary, codebase_changes=codebase_changes)

            # Local repository audits should be grounded by file/git tools, not
            # unrelated private life context. Keep codebase-change awareness,
            # but disable every personal/semantic retrieval source.
            _local_repo_audit = _is_local_repo_audit_query(user_input)
            _ro = dict(retrieval_overrides or {})
            if _local_repo_audit:
                _ro.update({
                    "max_recent": 0, "max_mems": 0, "max_summaries": 0,
                    "max_reflections": 0, "max_dreams": 0,
                    "max_semantic": 0, "max_wiki": 0, "max_skills": 0,
                    "max_proposals": 0, "max_git_commits": 0,
                    "max_surfaced_threads": 0, "max_reference_docs": 0,
                    "max_user_uploads": 0, "max_proactive": 0,
                    "max_visual_memories": 0, "max_personal_notes": 0,
                    "max_upcoming_schedule": 0, "max_narrative": 0,
                })
                logger.info(
                    "[BUILD_PROMPT] Local repository audit — using minimized, "
                    "tool-grounded context"
                )

            # Self-report retrieval trim (2026-09-06): a terse first-person
            # status update with no request gets a tighter retrieval ceiling
            # (see SELF_REPORT_RETRIEVAL_TRIM) — no-op on distress/heavy-topic
            # turns or anything that isn't self-report shaped.
            _ro = _apply_self_report_trim(_ro, user_input, crisis_level)

            # Step 2: Gather narrative context (synchronous, cheap file read)
            # Gated by intent override: max_narrative=0 skips entirely
            narrative_state = ""
            _ro_pre = _ro
            if _ro_pre.get("max_narrative", 1) > 0:
                try:
                    narrative_state = self.context_gatherer.get_narrative_context()
                    if narrative_state:
                        logger.debug(f"[PromptBuilder] Got narrative context ({len(narrative_state)} chars)")
                except Exception as e:
                    logger.debug(f"[PromptBuilder] Failed to get narrative context: {e}")

            # Apply intent-driven retrieval count overrides
            eff_max_recent = _ro.get("max_recent", PROMPT_MAX_RECENT)
            eff_max_mems = _ro.get("max_mems", PROMPT_MAX_MEMS)
            eff_max_summaries_r = _ro.get("max_recent_summaries", PROMPT_MAX_RECENT_SUMMARIES)
            eff_max_summaries_s = _ro.get("max_semantic_summaries", PROMPT_MAX_SEMANTIC_SUMMARIES)
            # Convenience: "max_summaries" splits evenly into recent/semantic
            if "max_summaries" in _ro and "max_recent_summaries" not in _ro:
                total = max(0, int(_ro["max_summaries"]))
                eff_max_summaries_r = (total + 1) // 2
                eff_max_summaries_s = total // 2
            eff_max_reflections_r = _ro.get("max_recent_reflections", PROMPT_MAX_RECENT_REFLECTIONS)
            eff_max_reflections_s = _ro.get("max_semantic_reflections", PROMPT_MAX_SEMANTIC_REFLECTIONS)
            if "max_reflections" in _ro and "max_recent_reflections" not in _ro:
                total = max(0, int(_ro["max_reflections"]))
                eff_max_reflections_r = (total + 1) // 2
                eff_max_reflections_s = total // 2
            eff_max_dreams = _ro.get("max_dreams", PROMPT_MAX_DREAMS)
            eff_max_semantic = _ro.get("max_semantic", PROMPT_MAX_SEMANTIC)
            # Wiki-semantic intent gate: skip the external wiki FAISS lookup on turns
            # where encyclopedic world-knowledge is useless (emotional/casual/meta/
            # temporal/project). This removes the biggest prompt-build latency sink
            # (up to SEM_TIMEOUT_S) from the majority of a personal assistant's traffic.
            if intent_type and intent_type.lower() in WIKI_SEMANTIC_SUPPRESS_INTENTS:
                if eff_max_semantic > 0:
                    logger.debug(
                        f"[BUILD_PROMPT] Suppressing wiki-semantic chunks for intent={intent_type}"
                    )
                eff_max_semantic = 0
            eff_max_wiki = _ro.get("max_wiki", PROMPT_MAX_WIKI)
            _continuation_shaped = (
                is_anaphoric_continuation(user_input)
                or is_fragment_continuation(user_input)
                or is_retry_continuation(user_input)
            )
            if _continuation_shaped and (eff_max_wiki > 0 or eff_max_semantic > 0):
                logger.debug("[BUILD_PROMPT] Continuation-shaped query — suppressing wiki retrieval")
                eff_max_semantic = 0
                eff_max_wiki = 0
            eff_max_skills = _ro.get("max_skills", PROMPT_MAX_SKILLS)
            eff_max_proposals = _ro.get("max_proposals", PROMPT_MAX_PROPOSALS)
            eff_max_git = _ro.get("max_git_commits", PROMPT_MAX_GIT_COMMITS)
            eff_max_surfaced_threads = _ro.get("max_surfaced_threads", PROMPT_MAX_SURFACED_THREADS)
            eff_max_reference_docs = _ro.get("max_reference_docs", PROMPT_MAX_REFERENCE_DOCS)
            eff_max_user_uploads = _ro.get("max_user_uploads", PROMPT_MAX_USER_UPLOADS)
            eff_max_proactive = _ro.get("max_proactive", PROMPT_MAX_PROACTIVE_INSIGHTS)
            eff_max_visual_memories = _ro.get("max_visual_memories", PROMPT_MAX_VISUAL_MEMORIES)
            # Defensive fallback: if no intent overrides available (intent=None),
            # suppress visual memory for very short messages (likely casual greetings)
            if not _ro and len(user_input.split()) <= 5:
                eff_max_visual_memories = 0
                logger.debug("[BUILD_PROMPT] No intent overrides + short message — suppressing visual memory")
            eff_max_personal_notes = _ro.get("max_personal_notes", PROMPT_MAX_PERSONAL_NOTES)
            eff_max_upcoming_schedule = _ro.get("max_upcoming_schedule", 0)
            eff_user_profile_tokens = 0 if _local_repo_audit else int(
                _ro.get("max_profile_tokens", USER_PROFILE_MAX_TOKENS)
            )
            eff_max_graph_sentences = 0 if _local_repo_audit else _ro.get("max_graph_sentences", PROMPT_MAX_GRAPH_SENTENCES)

            # Schedule keyword gating: even without intent override, activate
            # schedule retrieval when query has schedule trigger + temporal signal
            if eff_max_upcoming_schedule == 0:
                _ql = user_input.lower()
                _sched_triggers = (
                    "schedule", "shift", "free", "busy", "exam", "when do i",
                    "am i free", "am i off", "do i have anything", "what do i have",
                )
                _temporal_signals = (
                    "today", "tomorrow", "tonight", "this week", "weekend",
                    "next", "monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday",
                    "coming up", "upcoming", "soon",
                )
                if (any(t in _ql for t in _sched_triggers)
                        and any(s in _ql for s in _temporal_signals)):
                    eff_max_upcoming_schedule = 10

            if _ro:
                logger.info(f"[BUILD_PROMPT] Intent retrieval overrides: {_ro}")

            # Set intent-driven weight overrides on scorer (cleared after gather)
            scorer = getattr(self.memory_coordinator, 'scorer', None)
            if scorer and weight_overrides:
                scorer._intent_weight_overrides = weight_overrides
                logger.info(f"[BUILD_PROMPT] Intent weight overrides set on scorer")

            # Set graph references on scorer for graph-boosted scoring
            if scorer:
                scorer._graph_memory = getattr(self.memory_coordinator, 'graph_memory', None)
                scorer._entity_resolver = getattr(self.memory_coordinator, 'entity_resolver', None)

            # Flag distress on the gatherer so valence-aware retrieval caps
            # mood-congruent recall this turn (cleared after gather). Accepts any
            # crisis_level encoding ("CrisisLevel.CONCERN", "light_support", ...).
            _distress_active = _distress_from_crisis_level(crisis_level)
            self.context_gatherer._distress_active = _distress_active
            if _distress_active:
                logger.info(f"[BUILD_PROMPT] Distress session — valence-aware retrieval active (crisis_level={crisis_level})")

            # Same-turn upload dedupe (2026-09-04, homework-attachment turn
            # audit item 3): basenames of files attached THIS turn, so
            # get_user_uploads can skip a just-persisted chunk of a file
            # whose full content is already verbatim in [CURRENT QUERY].
            self.context_gatherer._current_turn_upload_filenames = (
                kwargs.get('_uploaded_filenames') or []
            )

            # Apply intent-driven gate threshold override (cleared after gather)
            _gate_override = kwargs.get('_gate_threshold_override')
            _saved_gate_threshold = None
            _gate_obj = None
            if _gate_override is not None:
                # Access the gate system (triggers lazy init via property)
                _gs = self.context_gatherer.gate_system
                if _gs is not None:
                    # MultiStageGateSystem wraps CosineSimilarityGateSystem as .gate_system
                    _gate_obj = getattr(_gs, 'gate_system', _gs)
                    if hasattr(_gate_obj, 'cosine_threshold'):
                        _saved_gate_threshold = _gate_obj.cosine_threshold
                        _gate_obj.cosine_threshold = _gate_override
                        logger.info(f"[BUILD_PROMPT] Gate threshold override: {_saved_gate_threshold:.3f} -> {_gate_override:.3f}")

            # Step 3: Launch parallel data gathering tasks with per-task timing
            # Pre-embed the query once so all parallel ChromaDB lookups reuse it.
            chroma = getattr(self.memory_coordinator, 'chroma_store', None)
            if chroma and hasattr(chroma, 'clear_embedding_cache'):
                chroma.clear_embedding_cache()
                try:
                    chroma._cached_embed(user_input)
                except Exception:
                    pass  # Non-fatal; individual queries will embed as needed

            tasks = {}
            task_timings = {}

            # Live per-task progress for the streaming UI (no-op outside a turn)
            from utils.turn_progress import emit as _progress_emit

            _TASK_LABELS = {
                "recent": "recent conversations",
                "memories": "memory retrieval",
                "user_profile": "user profile",
                "summaries": "summaries",
                "reflections": "reflections",
                "semantic": "wiki semantic index",
                "wiki": "wiki articles",
                "personal_notes": "Obsidian notes",
                "reference_docs": "reference docs",
                "user_uploads": "past uploads",
                "web_search": "web search",
                "git_commits": "git history",
                "graph_context": "knowledge graph",
                "unresolved_threads": "open threads",
                "procedural_skills": "skills",
                "google_calendar": "calendar",
                "relevant_emails": "emails",
                "visual_memories": "visual memories",
                "daemon_self_notes": "self notes",
                "proposed_features": "proposals",
                "dreams": "synthesis insights",
            }

            async def _timed_task(name: str, coro):
                """Wrapper to time individual tasks"""
                _start = time.perf_counter()
                try:
                    result = await coro
                    task_timings[name] = time.perf_counter() - _start
                    # Surface slow/fruitful tasks live; skip sub-perceptual ones
                    _dur = task_timings[name]
                    if _dur >= 0.2:
                        _label = _TASK_LABELS.get(name, name)
                        _n = f" · {len(result)} hits" if isinstance(result, (list, tuple)) else ""
                        _progress_emit(f"📥 {_label} ✓ {_dur:.1f}s{_n}")
                    return result
                except Exception as e:
                    task_timings[name] = time.perf_counter() - _start
                    raise e

            # Recent conversations
            if eff_max_recent > 0:
                tasks["recent"] = asyncio.create_task(
                    _timed_task("recent", self.context_gatherer._get_recent_conversations(eff_max_recent))
                )

            # Query-relevant memories (semantic search results only)
            if eff_max_mems > 0:
                tasks["memories"] = asyncio.create_task(
                    _timed_task("memories", self.context_gatherer._get_semantic_memories(user_input, eff_max_mems))
                )

            # User Profile (replaces semantic_facts + fresh_facts with categorized hybrid retrieval)
            # Increased max_tokens to 3000 to accommodate 12 facts per category (up to 144 facts total)
            if eff_user_profile_tokens > 0:
                tasks["user_profile"] = asyncio.create_task(
                    _timed_task("user_profile", self.context_gatherer.get_user_profile_context(
                        user_input, max_tokens=eff_user_profile_tokens
                    ))
                )

            # Summaries (separated into recent + semantic)
            if eff_max_summaries_r > 0 or eff_max_summaries_s > 0:
                tasks["summaries"] = asyncio.create_task(
                    _timed_task("summaries", self.context_gatherer._get_summaries_separate(user_input, eff_max_summaries_r, eff_max_summaries_s))
                )

            # Dreams (if enabled)
            if eff_max_dreams > 0:
                tasks["dreams"] = asyncio.create_task(
                    _timed_task("dreams", self.context_gatherer._get_dreams(eff_max_dreams))
                )

            # Semantic chunks
            if eff_max_semantic > 0:
                tasks["semantic"] = asyncio.create_task(
                    _timed_task("semantic", self.context_gatherer._get_semantic_chunks(user_input, max_results=eff_max_semantic))
                )

            # Reflections (separated into recent + semantic)
            if eff_max_reflections_r > 0 or eff_max_reflections_s > 0:
                tasks["reflections"] = asyncio.create_task(
                    _timed_task("reflections", self.context_gatherer._get_reflections_separate(user_input, eff_max_reflections_r, eff_max_reflections_s))
                )

            # Wiki content
            if eff_max_wiki > 0:
                tasks["wiki"] = asyncio.create_task(
                    _timed_task("wiki", self.context_gatherer._get_wiki_content(user_input, eff_max_wiki))
                )

            # Personal notes from Obsidian vault
            # Check if model is multimodal to decide whether to load images
            current_model = getattr(self.model_manager, 'active_model_name', '') if self.model_manager else ''
            include_note_images = _should_include_note_images(current_model, user_input, intent_type)
            logger.debug(f"[PromptBuilder] image check: model={current_model}, OBSIDIAN_INCLUDE_IMAGES={OBSIDIAN_INCLUDE_IMAGES}, is_multimodal={_is_multimodal_model(current_model)}, include_note_images={include_note_images}")

            if eff_max_personal_notes > 0:
                # Rollup (weekly/monthly summary) notes only on retrospective
                # queries (2026-09-03): question-shaped or carrying a recall cue.
                try:
                    from core.agentic.gate import _recall_signal_hit  # lazy import: cycle
                    _notes_allow_rollups = ("?" in user_input) or _recall_signal_hit(user_input.lower())
                except Exception:
                    _notes_allow_rollups = True
                # Negative mood-section notes only when the turn has an emotional
                # cue: heavy topic, active distress, or negative affect in the
                # message itself (2026-09-03).
                try:
                    from utils.query_checker import _is_heavy_topic_heuristic
                    from memory.valence import negative_affect_score
                    from config.app_config import VALENCE_NEGATIVE_THRESHOLD  # lazy import: live-config read
                    _notes_allow_mood = bool(
                        getattr(self.context_gatherer, "_distress_active", False)
                        or _is_heavy_topic_heuristic(user_input)
                        or negative_affect_score(user_input) >= float(VALENCE_NEGATIVE_THRESHOLD)
                    )
                except Exception:
                    _notes_allow_mood = True
                tasks["personal_notes"] = asyncio.create_task(
                    _timed_task("personal_notes", self.context_gatherer.get_personal_notes(
                        user_input,
                        eff_max_personal_notes,
                        include_images=include_note_images,
                        max_images_per_note=OBSIDIAN_MAX_IMAGES_PER_NOTE,
                        allow_rollups=_notes_allow_rollups,
                        allow_mood_sections=_notes_allow_mood,
                    ))
                )

            # Reference documents (system docs, project outlines - excludes user uploads)
            # Allow-list gate (2026-08-05): self-docs surface only on
            # meta/technical/project turns or a self-referential query cue;
            # the 2026-07-25 suppressions (file uploads, distress,
            # emotional_support) still always win. See
            # _should_include_reference_docs.
            _include_self_docs = _should_include_reference_docs(
                user_input,
                intent_type,
                kwargs.get("_suppress_reference_docs", False),
                _distress_active,
            )
            if not _include_self_docs and not kwargs.get("_suppress_reference_docs", False):
                logger.info(
                    f"[BUILD_PROMPT] Skipping reference_docs "
                    f"(distress={_distress_active}, intent={intent_type} — not a self-docs turn)"
                )
            if _include_self_docs and eff_max_reference_docs > 0:
                tasks["reference_docs"] = asyncio.create_task(
                    _timed_task("reference_docs", self.context_gatherer.get_reference_docs(user_input, eff_max_reference_docs))
                )

            # User uploads (previously uploaded files/images)
            if eff_max_user_uploads > 0:
                tasks["user_uploads"] = asyncio.create_task(
                    _timed_task("user_uploads", self.context_gatherer.get_user_uploads(user_input, eff_max_user_uploads))
                )

            # Git commit history (procedural memory)
            if eff_max_git > 0:
                tasks["git_commits"] = asyncio.create_task(
                    _timed_task("git_commits", self.context_gatherer.get_git_commits(user_input, eff_max_git))
                )

            # Procedural skills (adaptive workflows)
            # Fetch wider window when activation policy is active so it can filter/rerank
            _skill_fetch_limit = eff_max_skills * self._skill_fetch_multiplier if self._skill_activation_policy else eff_max_skills
            if eff_max_skills > 0:
                tasks["procedural_skills"] = asyncio.create_task(
                    _timed_task("procedural_skills", self.context_gatherer.get_procedural_skills(user_input, _skill_fetch_limit))
                )

            # Proposed features (code proposals, only for project-related queries)
            if eff_max_proposals > 0:
                tasks["proposed_features"] = asyncio.create_task(
                    _timed_task("proposed_features", self.context_gatherer.get_proposed_features(user_input, eff_max_proposals))
                )

            # Knowledge graph context (entity relationships)
            if eff_max_graph_sentences > 0:
                tasks["graph_context"] = asyncio.create_task(
                    _timed_task("graph_context", self.context_gatherer.get_graph_context(
                        user_input, eff_max_graph_sentences
                    ))
                )

            # Unresolved threads (proactive surfacing)
            if eff_max_surfaced_threads > 0:
                tasks["unresolved_threads"] = asyncio.create_task(
                    _timed_task("unresolved_threads", self.context_gatherer.get_unresolved_threads(eff_max_surfaced_threads))
                )

            # Proactive cross-domain insights (non-blocking: warm cache or background warmup).
            # Distress gate (2026-08-05): speculative cross-domain suggestions
            # ("D&D could enhance your teamwork skills") were injected into
            # distress-adjacent medical conversations — same class as the
            # reference-docs distress suppression. Elevated tone gets no
            # proactive insights; they return when the session is conversational.
            if eff_max_proactive > 0 and _distress_active:
                logger.info("[BUILD_PROMPT] Proactive insights suppressed (distress session)")
            elif eff_max_proactive > 0:
                _surfacer = getattr(self.memory_coordinator, 'context_surfacer', None)
                if _surfacer and _surfacer._session_insights is not None:
                    # Cache warm — retrieve instantly via gatherer (adds attribution + citations)
                    tasks["proactive_insights"] = asyncio.create_task(
                        _timed_task("proactive_insights",
                                    self.context_gatherer.get_proactive_insights(user_input, eff_max_proactive))
                    )
                else:
                    # Cache cold (first message of session) — fire background warmup,
                    # skip insights for this message (available from message 2 onward)
                    async def _warmup_insights():
                        try:
                            await self.context_gatherer.get_proactive_insights(user_input, eff_max_proactive)
                            logger.debug("[BUILD_PROMPT] Proactive insights warmed up for next message")
                        except Exception as exc:
                            logger.debug(f"[BUILD_PROMPT] Insight warmup failed (non-fatal): {exc}")

                    asyncio.create_task(_warmup_insights())
                    logger.info("[BUILD_PROMPT] Proactive insights: cache cold, warming in background")

            # Upcoming schedule (gated by intent or keyword detection)
            if eff_max_upcoming_schedule > 0:
                tasks["upcoming_schedule"] = asyncio.create_task(
                    _timed_task("upcoming_schedule",
                                self.context_gatherer.get_upcoming_schedule(user_input, eff_max_upcoming_schedule))
                )

            # Visual memories (CLIP-based image search) — gated by visual intent
            # (a "show me"/recall signal), not just a bare entity name match.
            if eff_max_visual_memories > 0:
                tasks["visual_memories"] = asyncio.create_task(
                    _timed_task("visual_memories", self.context_gatherer.get_visual_memories(user_input, eff_max_visual_memories, intent_type=intent_type))
                )

            # Google Calendar events (real-time, cached 5 min)
            try:
                from config.app_config import GOOGLE_CALENDAR_ENABLED, GOOGLE_CALENDAR_MAX_EVENTS
                if GOOGLE_CALENDAR_ENABLED and not _local_repo_audit:
                    tasks["google_calendar"] = asyncio.create_task(
                        _timed_task("google_calendar",
                                    self.context_gatherer.get_google_calendar_events(GOOGLE_CALENDAR_MAX_EVENTS))
                    )
            except Exception:
                pass

            # Relevant emails (passive retrieval, cue-gated + distress-suppressed)
            try:
                from config.app_config import EMAIL_PASSIVE_CONTEXT_ENABLED, EMAIL_PASSIVE_MAX
                # 2026-09-03: intent profiles may zero passive email (casual_social /
                # emotional_support) via the max_relevant_emails override key.
                _eff_emails = int(_ro.get("max_relevant_emails", EMAIL_PASSIVE_MAX))
                if (EMAIL_PASSIVE_CONTEXT_ENABLED and _eff_emails > 0
                        and not _local_repo_audit):
                    tasks["relevant_emails"] = asyncio.create_task(
                        _timed_task("relevant_emails",
                                    self.context_gatherer.get_relevant_emails(user_input, _eff_emails))
                    )
            except Exception:
                pass

            # Daemon self-notes (working context from prior sessions)
            try:
                from config.app_config import DAEMON_NOTES_ENABLED, DAEMON_NOTES_MAX_PER_PROMPT
                if (DAEMON_NOTES_ENABLED and DAEMON_NOTES_MAX_PER_PROMPT > 0
                        and not _local_repo_audit):
                    tasks["daemon_self_notes"] = asyncio.create_task(
                        _timed_task("daemon_self_notes",
                                    self.context_gatherer.get_daemon_self_notes(user_input, DAEMON_NOTES_MAX_PER_PROMPT))
                    )
            except Exception:
                pass

            # Web search (triggered based on query analysis, suppressed during crisis).
            # Build a compact recent-turns digest so the trigger can resolve elliptical
            # follow-ups ("they're only giving us 7 days") against the topic just
            # discussed — the agentic gate already passes this; without it a pronoun-only
            # claim scores 0 on the standalone heuristic and the LLM is never consulted.
            _web_conv_ctx = None
            try:
                from core.agentic.gate import _build_recent_context
                _web_conv_ctx = _build_recent_context(
                    getattr(self.memory_coordinator, 'corpus_manager', None)
                )
            except Exception as _wc_err:
                logger.debug(f"[BUILD_PROMPT] recent-context for web trigger failed (non-fatal): {_wc_err}")
            if not _local_repo_audit:
                tasks["web_search"] = asyncio.create_task(
                    _timed_task("web_search", self.context_gatherer._get_web_search_results(
                        user_input, crisis_level, intent_type=intent_type,
                        conversation_context=_web_conv_ctx))
                )

            # Gather all results with timeout — use asyncio.wait so completed
            # tasks survive a timeout instead of wiping the entire context.
            _progress_emit(f"🔎 Retrieving context from {len(tasks)} sources in parallel…")
            _gather_start = time.perf_counter()
            phase_timings["before_retrieval"] = _gather_start - start_time
            try:
                done, pending = await asyncio.wait(
                    list(tasks.values()),
                    timeout=30.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
                _gather_elapsed = time.perf_counter() - _gather_start
                phase_timings["retrieval"] = _gather_elapsed

                gathered = {}
                timed_out_names = []
                for name, task in tasks.items():
                    if task in done:
                        try:
                            gathered[name] = task.result() or []
                            if name == "memories":
                                logger.debug(f"MEMORIES TASK: Got {len(gathered[name])} memories")
                            if name == "proposed_features":
                                logger.info(f"[PROPOSED_FEATURES] Task returned {len(gathered[name])} proposals")
                        except (Exception, asyncio.CancelledError) as exc:
                            logger.warning("Context task %s failed: %s", name, exc)
                            gathered[name] = []
                    else:
                        task.cancel()
                        gathered[name] = []
                        timed_out_names.append(name)

                if timed_out_names:
                    logger.warning(
                        "Prompt context retrieval timed out; partial context used. Pending: %s",
                        sorted(timed_out_names),
                    )

                if task_timings:
                    sorted_timings = sorted(task_timings.items(), key=lambda x: x[1], reverse=True)
                    timing_str = " | ".join([f"{k}={v:.2f}s" for k, v in sorted_timings])
                    logger.info(
                        f"[BUILD_PROMPT TIMING] total={_gather_elapsed:.2f}s | {timing_str}"
                    )
                _progress_emit(
                    f"🧱 Context retrieved ({_gather_elapsed:.1f}s) — gating, dedup, token budget…"
                )

            except Exception as _gather_exc:
                logger.warning("Unexpected error during context gathering: %s", _gather_exc)
                gathered = {name: [] for name in tasks.keys()}
            finally:
                # asyncio.wait does not cancel its children when this request
                # is cancelled. Drain them before clearing request-specific
                # scorer/gatherer state or admitting the next turn.
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks.values(), return_exceptions=True)
                # Clear intent weight overrides from scorer (set before gather)
                if scorer and weight_overrides:
                    scorer._intent_weight_overrides = None
                # Clear graph references from scorer
                if scorer:
                    scorer._graph_memory = None
                    scorer._entity_resolver = None
                # Clear distress flag from gatherer (set before gather)
                try:
                    self.context_gatherer._distress_active = False
                except Exception:
                    pass
                # Clear same-turn upload filenames (set before gather)
                try:
                    self.context_gatherer._current_turn_upload_filenames = []
                except Exception:
                    pass
                # Restore gate threshold (set before gather)
                if _saved_gate_threshold is not None and _gate_obj is not None:
                    _gate_obj.cosine_threshold = _saved_gate_threshold

            _post_start = time.perf_counter()
            # Step 3: Post-fetch processing

            # Apply skill activation policy (filter/rerank by intent, relevance, cooldown)
            if self._skill_activation_policy and "procedural_skills" in gathered:
                _stm_topics = None
                if self._skill_activation_use_stm and stm_summary:
                    _topic = stm_summary.get("topic", "")
                    _stm_topics = [_topic] if _topic and _topic.lower() != "general" else None
                gathered["procedural_skills"] = self._skill_activation_policy.filter(
                    gathered["procedural_skills"],
                    intent_type=intent_type,
                    stm_topics=_stm_topics,
                )

            # Handle separated summaries (recent + semantic)
            summaries_data = gathered.get("summaries", {})
            logger.debug(f"CONTEXT GATHERING: summaries_data type={type(summaries_data).__name__}, len={len(summaries_data) if isinstance(summaries_data, (list, dict)) else '?'}")
            if isinstance(summaries_data, dict):
                recent_summaries = summaries_data.get("recent", [])
                semantic_summaries = summaries_data.get("semantic", [])
                all_summaries = recent_summaries + semantic_summaries
                logger.debug(f"CONTEXT GATHERING: Extracted {len(recent_summaries)} recent, {len(semantic_summaries)} semantic summaries")
            else:
                # Backward compatibility for old format
                all_summaries = summaries_data or []
                recent_summaries = []
                semantic_summaries = []
                logger.debug(f"CONTEXT GATHERING: Using old format, got {len(all_summaries)} summaries")

            # Handle separated reflections (recent + semantic)
            reflections_data = gathered.get("reflections", {})
            if isinstance(reflections_data, dict):
                recent_reflections = reflections_data.get("recent", [])
                semantic_reflections = reflections_data.get("semantic", [])
                all_reflections = recent_reflections + semantic_reflections
            else:
                # Backward compatibility for old format
                all_reflections = reflections_data or []
                recent_reflections = []
                semantic_reflections = []

            # Filter reflections to session-level if enabled; if it empties the set,
            # fall back to original reflections to avoid dropping the section.
            if REFLECTIONS_SESSION_FILTER and all_reflections:
                session_reflections = [
                    r for r in all_reflections
                    if "session" in (r.get("tags", []) or []) or "session" in (r.get("source", "") or "")
                ]
                if not session_reflections:
                    session_reflections = all_reflections
            else:
                session_reflections = all_reflections

            # Sort reflections by timestamp (most recent first)
            try:
                session_reflections.sort(
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )
            except TypeError as e:
                logger.warning(f"[PromptBuilder] Could not sort reflections by timestamp: {e}")

            # Top-up with on-demand reflections if needed
            _effective_reflection_total = eff_max_reflections_r + eff_max_reflections_s
            if (REFLECTIONS_TOPUP and REFLECTIONS_ON_DEMAND
                    and _effective_reflection_total > 0
                    and len(session_reflections) < _effective_reflection_total):

                try:
                    context_for_reflection = {
                        "memories": gathered.get("memories", []),
                        "fresh_facts": gathered.get("recent_facts", [])
                    }

                    _progress_emit("💭 Generating on-demand reflection…")
                    on_demand_reflections = await self.summarizer._reflect_on_demand(
                        context_for_reflection,
                        user_input,
                        session_reflections
                    )

                    session_reflections.extend(on_demand_reflections)
                except Exception as e:
                    logger.warning(f"On-demand reflection failed: {e}")

            # Step 4: Build initial context
            gathered_memories = gathered.get("memories", [])
            logger.debug(f"CONTEXT BUILD: gathered memories count = {len(gathered_memories)}")

            # DEBUG: Check what's in recent conversations
            recent_convos = gathered.get("recent", [])
            logger.debug(f"[DEBUG RECENT] build_prompt: Got {len(recent_convos)} recent_conversations from gatherer")
            if recent_convos:
                # Log first 3 and last 3 with timestamps
                for i in range(min(3, len(recent_convos))):
                    mem = recent_convos[i]
                    ts = mem.get('timestamp', 'NO_TS')
                    query = mem.get('query', '')[:80]
                    logger.debug(f"[DEBUG RECENT] Item {i+1} (first): ts={ts}, query={query}...")
                if len(recent_convos) > 3:
                    for i in range(max(0, len(recent_convos) - 3), len(recent_convos)):
                        mem = recent_convos[i]
                        ts = mem.get('timestamp', 'NO_TS')
                        query = mem.get('query', '')[:80]
                        logger.debug(f"[DEBUG RECENT] Item {i+1} (last): ts={ts}, query={query}...")

            context = {
                "recent_conversations": recent_convos,
                "memories": gathered_memories,
                "user_profile": gathered.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "narrative_state": narrative_state,  # Temporal grounding (synthesized life context)
                "summaries": all_summaries,
                "recent_summaries": recent_summaries,
                "semantic_summaries": semantic_summaries,
                "reflections": session_reflections,
                "recent_reflections": recent_reflections,
                "semantic_reflections": semantic_reflections,
                "dreams": gathered.get("dreams", []),
                "semantic_chunks": gathered.get("semantic", []),
                "wiki": gathered.get("wiki", []),
                "personal_notes": gathered.get("personal_notes", []),  # User's Obsidian notes
                "reference_docs": gathered.get("reference_docs", []),  # System/project documentation
                "user_uploads": gathered.get("user_uploads", []),     # User uploaded files/images
                "git_commits": gathered.get("git_commits", []),      # Git commit history
                "procedural_skills": gathered.get("procedural_skills", []),  # Adaptive workflows
                "proposed_features": gathered.get("proposed_features", []),  # Code proposals
                "graph_context": gathered.get("graph_context", []),  # Knowledge graph relationships
                "unresolved_threads": gathered.get("unresolved_threads", []),  # Proactive thread surfacing
                "upcoming_schedule": gathered.get("upcoming_schedule", []),  # Schedule events (gated)
                "google_calendar": gathered.get("google_calendar", []),  # Real-time Google Calendar events
                "relevant_emails": gathered.get("relevant_emails", []),  # Relevant emails from Gmail/Outlook
                "proactive_insights": gathered.get("proactive_insights", []),  # Cross-domain insights
                "visual_memories": gathered.get("visual_memories", {"text_results": [], "images": []}),  # CLIP visual memories
                "web_search_results": gathered.get("web_search"),  # Real-time web search results
                "daemon_self_notes": gathered.get("daemon_self_notes", []),
                "codebase_changes": codebase_changes,  # Git changes since last session (first message only)
            }
            logger.debug(f"CONTEXT BUILT: recent_summaries={len(recent_summaries)}, semantic_summaries={len(semantic_summaries)}, recent_reflections={len(recent_reflections)}, semantic_reflections={len(semantic_reflections)}")
            logger.debug(f"CONTEXT BUILD: context memories count = {len(context['memories'])}")

            # Override with directly provided parameters (legacy interface)
            # Note: fresh_facts removed - now using user_profile instead
            if memories is not None:
                context["memories"] = memories

            # Step 5: Apply gating to filter by relevance
            try:
                # Avoid re-gating memories: ContextGatherer already applies
                # semantic filtering to the semantic half while preserving
                # the recency half. Re-gating here could drop the recents.

                # Do not gate wiki snippets here — wiki utility already applies
                # conservative cleaning and we prefer fail-open to ensure topical
                # knowledge flows into the prompt.

                # Allow semantic chunks to flow as-is; downstream token budgeting
                # and stitching will cap size. If we need gating later, prefer
                # the specialized filter_semantic_chunks in gate_system.

                # Gate personal notes through the multi-stage gate system
                personal_notes = context.get("personal_notes", [])
                if personal_notes and hasattr(self.context_gatherer, 'gate_system'):
                    try:
                        # min_results=0 (2026-09-03): notes never get the gate's
                        # forced below-threshold backfill (GATE_MIN_MEMORIES).
                        gated_notes = await self.context_gatherer.gate_system.filter_memories(
                            user_input, personal_notes, min_results=0
                        )
                        # Apply the personal-notes bar on the gate's BLENDED score
                        # (0.85·cosine + 0.15·truth; notes carry truth 0.9, so the
                        # 0.60 default ≈ cosine 0.55) — see obsidian.gate_threshold.
                        pre_filter_count = len(gated_notes)
                        gated_notes = [n for n in gated_notes
                                       if n.get("relevance_score", 0) >= PERSONAL_NOTES_GATE_THRESHOLD]
                        context["personal_notes"] = gated_notes[:PROMPT_MAX_PERSONAL_NOTES]
                        logger.debug(f"Gated personal notes: {len(personal_notes)} -> {pre_filter_count} (gate) -> {len(context['personal_notes'])} (threshold={PERSONAL_NOTES_GATE_THRESHOLD})")
                    except Exception as gate_err:
                        logger.warning(f"Personal notes gating failed, keeping original: {gate_err}")

                # Filter reference docs by relevance threshold to prevent
                # semantically-distant content from polluting the prompt
                reference_docs = context.get("reference_docs", [])
                if reference_docs:
                    pre_count = len(reference_docs)
                    reference_docs = [d for d in reference_docs
                                      if d.get("relevance_score", 0) >= REFERENCE_DOCS_GATE_THRESHOLD]
                    context["reference_docs"] = reference_docs[:PROMPT_MAX_REFERENCE_DOCS]
                    logger.debug(f"Reference docs gate: {pre_count} -> {len(context['reference_docs'])} (threshold={REFERENCE_DOCS_GATE_THRESHOLD})")
            except Exception as e:
                logger.warning(f"Gating failed: {e}")

            # Step 6: Apply hygiene and caps
            logger.debug(f"BEFORE HYGIENE_AND_CAPS: memories count = {len(context.get('memories', []))}")
            context = await self._hygiene_and_caps(context, stm_summary=stm_summary)

            # Step 6.1: Top-up relevant memories if cross-effects reduced them too much.
            try:
                mems = context.get("memories", []) or []
                recents = context.get("recent_conversations", []) or []
                # Respect intent-specific memory caps. Pre-2026-08-21 this
                # top-up used the global PROMPT_MAX_MEMS, so a casual_social
                # profile with max_mems=3 could be inflated back to the global
                # target after hygiene/dedup.
                # 2026-08-29: the top-up is a SURVIVAL MINIMUM, not a refill —
                # its filler is UNGATED recent conversations beyond the ones
                # [RECENT CONVERSATION] already shows, relabeled as "relevant".
                # Refilling to the intent cap defeated the gate's honest-few
                # doctrine (live: gate returned 1 memory, quality floor working
                # as designed; this top-up then added 30 recents — 10K tokens of
                # the prior days' heavy conversations on a logistics turn).
                target_mems = min(max(0, int(eff_max_mems or 0)), MEMORY_TOPUP_FLOOR)
                if len(mems) < target_mems:
                    # Pull extra recent conversations beyond the ones already shown
                    extra_recent = await self.context_gatherer._get_recent_conversations(PROMPT_MAX_RECENT + target_mems)
                    # Build keys for already used items
                    def _key(x):
                        return (str(x.get("query", "")) + str(x.get("response", ""))).strip().lower()

                    # CRITICAL: Check against BOTH recent_conversations AND existing memories to avoid duplicates
                    used = {_key(r) for r in recents}
                    used.update({_key(m) for m in mems})  # Also check against existing memories!

                    # Keep only items not already in either section
                    filler = []
                    skipped_count = 0
                    for item in extra_recent:
                        if _key(item) not in used:
                            filler.append(item)
                        else:
                            skipped_count += 1

                    needed = max(0, target_mems - len(mems))
                    if needed:
                        mems.extend(filler[:needed])
                        context["memories"] = mems
                        logger.debug(f"MEMORY TOP-UP: Added {min(needed, len(filler))} new memories (had {len(mems) - min(needed, len(filler))}, target {target_mems}), skipped {skipped_count} duplicates")
            except Exception as e:
                logger.warning(f"Memory top-up failed: {e}")

            logger.debug(f"AFTER MEMORY TOP-UP: memories count = {len(context.get('memories', []))}")

            # Step 6.2: Ensure minimum summaries and reflections by pulling directly from storage
            try:
                logger.debug(f"START OF SUMMARIES BLOCK: memories count = {len(context.get('memories', []))}")
                # Summaries — if we have too few, pull most recent without gating.
                # Targets recent_summaries: the formatter renders the SPLIT keys
                # only, so topping up the combined "summaries" key (pre-2026-08-14)
                # added content that never reached the prompt.
                if (eff_max_summaries_r > 0 and
                        len(context.get("recent_summaries", []) or []) < eff_max_summaries_r):
                    needed = eff_max_summaries_r - len(context.get("recent_summaries", []))
                    try:
                        # try memory_coordinator first (supports sync or async)
                        if hasattr(self.memory_coordinator, 'get_summaries'):
                            logger.debug(f"BEFORE get_summaries: memories count = {len(context.get('memories', []))}")
                            res = self.memory_coordinator.get_summaries(
                                max(1, eff_max_summaries_r + eff_max_summaries_s) * 2
                            )
                            import asyncio as _asyncio
                            stored = await res if _asyncio.iscoroutine(res) else res
                            logger.debug(f"AFTER get_summaries: memories count = {len(context.get('memories', []))}, stored type = {type(stored).__name__}")
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_summaries'):
                            stored = self.memory_coordinator.corpus_manager.get_summaries(
                                max(1, eff_max_summaries_r + eff_max_summaries_s) * 2
                            )
                        else:
                            stored = []
                    except Exception as e:
                        logger.warning(f"[PromptBuilder] Summary retrieval failed: {e}")
                        stored = []

                    # Keep the newest not already in context
                    # Normalize stored schema (legacy may use 'response'/'text')
                    norm = []
                    for s in (stored or []):
                        if isinstance(s, dict):
                            if not s.get('content'):
                                c = s.get('response') or s.get('text')
                                if c:
                                    s = {**s, 'content': c}
                        norm.append(s)
                    stored = norm

                    have = { (s.get('content') or '').strip()
                             for key in ('recent_summaries', 'semantic_summaries')
                             for s in (context.get(key) or []) if isinstance(s, dict) }
                    add = select_floor_topup(stored, have, needed)
                    if add:
                        context['recent_summaries'] = (context.get('recent_summaries') or []) + add

                # Reflections — if too few, pull most recent historical reflections
                # (recent_reflections is the rendered key; see summaries note above)
                if (eff_max_reflections_r > 0 and
                        len(context.get("recent_reflections", []) or []) < eff_max_reflections_r):
                    needed = eff_max_reflections_r - len(context.get("recent_reflections", []))
                    stored_refl = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_reflections'):
                            # get_reflections may be async; try both
                            res = self.memory_coordinator.get_reflections(
                                max(1, _effective_reflection_total) * 3
                            )
                            if asyncio.iscoroutine(res):
                                stored_refl = await res
                            else:
                                stored_refl = res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_reflections'):
                            res2 = self.memory_coordinator.corpus_manager.get_reflections(
                                max(1, _effective_reflection_total) * 3
                            )
                            stored_refl = res2 if isinstance(res2, list) else list(res2)
                    except Exception as e:
                        logger.warning(f"[PromptBuilder] Reflection retrieval failed: {e}")
                        stored_refl = []

                    have_refl = { (r.get('content') or '').strip()
                                  for key in ('recent_reflections', 'semantic_reflections')
                                  for r in (context.get(key) or []) if isinstance(r, dict) }
                    add_refl = select_floor_topup(stored_refl, have_refl, needed)
                    if add_refl:
                        context['recent_reflections'] = (context.get('recent_reflections') or []) + add_refl
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Reflection pre-budget top-up failed: {e}")

            # Step 6.9: LLM-compress heavily oversized items (async pre-pass)
            # Items >= 3x over their token limit get LLM summary instead of middle-out slicing.
            # Mildly oversized items (1x-3x) still use middle-out in token_manager.
            _compression_start = time.perf_counter()
            phase_timings["post_retrieval"] = _compression_start - _post_start
            context = await self._llm_compress_oversized(context)
            phase_timings["compression"] = time.perf_counter() - _compression_start
            _floor_start = time.perf_counter()

            # Complete recency candidates BEFORE budgeting. Restoring raw
            # records after trimming reintroduced unbounded attachment-heavy
            # conversations and repeated queries whose results were discarded.
            # Floors are best effort: the final hard budget takes precedence.
            try:
                # Recent conversations floor — guarantee session context survives
                recent_convos = context.get("recent_conversations", []) or []
                _recent_floor = min(PROMPT_MIN_RECENT_FLOOR, max(0, int(eff_max_recent)))
                if len(recent_convos) < _recent_floor:
                    needed_recent = _recent_floor - len(recent_convos)
                    try:
                        stored_recent = await self.context_gatherer._get_recent_conversations(PROMPT_MIN_RECENT_FLOOR * 2)
                    except Exception as e:
                        logger.debug(f"Failed to fetch recent conversations for floor: {e}")
                        stored_recent = []
                    if stored_recent:
                        def _recent_key(x):
                            return (str(x.get("query", "")) + str(x.get("response", ""))).strip().lower()
                        have_keys = {_recent_key(r) for r in recent_convos}
                        add_recent = []
                        for r in stored_recent:
                            if isinstance(r, dict) and _recent_key(r) not in have_keys:
                                add_recent.append(r)
                                have_keys.add(_recent_key(r))
                            if len(add_recent) >= needed_recent:
                                break
                        if add_recent:
                            context['recent_conversations'] = (context.get('recent_conversations') or []) + add_recent
                            logger.info(f"[RECENCY FLOOR] Added {len(add_recent)} conversation candidates (had {len(recent_convos)}, floor={_recent_floor})")

                # Summaries floor — survival minimum for the rendered key only
                # (restoring to MAX here would undo the budget trim)
                _summary_floor = min(
                    PROMPT_MIN_SUMMARIES_FLOOR,
                    max(0, int(eff_max_summaries_r)),
                )
                if len(context.get("recent_summaries", []) or []) < _summary_floor:
                    needed = _summary_floor - len(context.get("recent_summaries", []))
                    stored = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_summaries'):
                            res = self.memory_coordinator.get_summaries(PROMPT_MAX_SUMMARIES * 3)
                            import asyncio as _asyncio
                            stored = await res if _asyncio.iscoroutine(res) else res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_summaries'):
                            stored = self.memory_coordinator.corpus_manager.get_summaries(PROMPT_MAX_SUMMARIES * 3)
                        else:
                            stored = []
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Failed to fetch summaries for floor: {e}")
                        stored = []

                    # Normalize stored schema
                    norm = []
                    for s in (stored or []):
                        if isinstance(s, dict) and not s.get('content'):
                            c = s.get('response') or s.get('text')
                            if c:
                                s = {**s, 'content': c}
                        norm.append(s)
                    stored = norm

                    have = { (s.get('content') or '').strip()
                             for key in ('recent_summaries', 'semantic_summaries')
                             for s in (context.get(key) or []) if isinstance(s, dict) }
                    add = select_floor_topup(stored, have, needed)
                    if add:
                        context['recent_summaries'] = (context.get('recent_summaries') or []) + add

                logger.debug(f"AFTER SUMMARIES TOP-UP: memories count = {len(context.get('memories', []))}")

                # Reflections floor — survival minimum for the rendered key only
                _reflection_floor = min(
                    PROMPT_MIN_REFLECTIONS_FLOOR,
                    max(0, int(eff_max_reflections_r)),
                )
                if len(context.get("recent_reflections", []) or []) < _reflection_floor:
                    needed = _reflection_floor - len(context.get("recent_reflections", []))
                    stored_refl = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_reflections'):
                            res = self.memory_coordinator.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            import asyncio as _asyncio
                            if _asyncio.iscoroutine(res):
                                stored_refl = await res
                            else:
                                stored_refl = res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_reflections'):
                            res2 = self.memory_coordinator.corpus_manager.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            stored_refl = res2 if isinstance(res2, list) else list(res2)
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Failed to fetch reflections for floor: {e}")
                        stored_refl = []

                    # Normalize stored reflections schema
                    norm_r = []
                    for r in (stored_refl or []):
                        if isinstance(r, dict) and not r.get('content'):
                            c = r.get('response') or r.get('text')
                            if c:
                                r = {**r, 'content': c}
                        norm_r.append(r)
                    stored_refl = norm_r

                    have_refl = { (r.get('content') or '').strip()
                                  for key in ('recent_reflections', 'semantic_reflections')
                                  for r in (context.get(key) or []) if isinstance(r, dict) }
                    add_refl = select_floor_topup(stored_refl, have_refl, needed)
                    if add_refl:
                        context['recent_reflections'] = (context.get('recent_reflections') or []) + add_refl
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Recency floor top-up failed: {e}")

            # Last content mutation before assembly: every top-up must pass
            # the same field caps and priority budget as retrieved candidates.
            _budget_start = time.perf_counter()
            phase_timings["recency_topup"] = _budget_start - _floor_start
            context = self.token_manager._manage_token_budget(context)
            phase_timings["token_budget"] = time.perf_counter() - _budget_start

            logger.debug(f"BEFORE FINAL ASSEMBLY: memories count = {len(context.get('memories', []))}")

            # Step 8: Final context assembly
            prompt_ctx = {
                "recent_conversations": context.get("recent_conversations", []),
                "memories": context.get("memories", []),
                "user_profile": context.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "narrative_state": context.get("narrative_state", ""),  # Temporal grounding (synthesized life context)
                "summaries": context.get("summaries", []),
                "recent_summaries": context.get("recent_summaries", []),
                "semantic_summaries": context.get("semantic_summaries", []),
                "reflections": context.get("reflections", []),
                "recent_reflections": context.get("recent_reflections", []),
                "semantic_reflections": context.get("semantic_reflections", []),
                "dreams": context.get("dreams", []),
                "semantic_chunks": context.get("semantic_chunks", []),
                "wiki": context.get("wiki", []),
                "personal_notes": context.get("personal_notes", []),  # User's Obsidian notes
                "reference_docs": context.get("reference_docs", []),  # Daemon self-knowledge docs
                "user_uploads": context.get("user_uploads", []),     # User uploaded files/images
                "git_commits": context.get("git_commits", []),      # Git commit history
                "procedural_skills": context.get("procedural_skills", []),  # Adaptive workflows
                "proposed_features": context.get("proposed_features", []),  # Code proposals
                "graph_context": context.get("graph_context", []),  # Knowledge graph relationships
                "unresolved_threads": context.get("unresolved_threads", []),  # Proactive thread surfacing
                "upcoming_schedule": context.get("upcoming_schedule", []),  # Schedule events (gated)
                "google_calendar": context.get("google_calendar", []),  # Real-time Google Calendar events
                "relevant_emails": context.get("relevant_emails", []),  # Relevant emails from Gmail/Outlook
                "proactive_insights": context.get("proactive_insights", []),  # Cross-domain insights
                "visual_memories": context.get("visual_memories", {"text_results": [], "images": []}),  # CLIP visual memories
                "web_search_results": context.get("web_search_results"),  # Real-time web search results
                "daemon_self_notes": context.get("daemon_self_notes", []),
                "codebase_changes": context.get("codebase_changes", {}),
                "stm_summary": context.get("stm_summary"),  # STM context summary (dict or None)
                "memory_id_map": self.context_gatherer.memory_id_map if hasattr(self.context_gatherer, 'memory_id_map') else {}
            }

            build_time = time.perf_counter() - start_time
            logger.info(f"Prompt built in {build_time:.2f}s")
            logger.debug(f"RETURNING CONTEXT: memories count = {len(prompt_ctx.get('memories', []))}")

            # Attach timing metadata for interpretability (underscore-prefixed to avoid collision)
            prompt_ctx["_task_timings"] = dict(task_timings)
            prompt_ctx["_gather_elapsed"] = locals().get('_gather_elapsed', 0.0)
            prompt_ctx["_build_time"] = build_time
            prompt_ctx["_phase_timings"] = phase_timings
            logger.info("[BUILD_PROMPT PHASES] %s", " | ".join(
                f"{name}={elapsed:.3f}s" for name, elapsed in phase_timings.items()
            ))

            return prompt_ctx

        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return minimal context on error
            error_context = {
                "recent_conversations": [],
                "memories": [],
                "user_profile": "",
                "narrative_state": "",
                "summaries": [],
                "reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],
                "user_uploads": [],
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "upcoming_schedule": [],
                "google_calendar": [],
                "relevant_emails": [],
                "proactive_insights": [],
                "web_search_results": None,
                "memory_id_map": {}
            }
            # Include stm_summary if it was provided
            if stm_summary is not None:
                error_context["stm_summary"] = stm_summary
            return error_context

    async def build_prompt_from_context(
        self,
        context: "ContextResult",
        memories: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build prompt from a ContextResult object.

        This method provides a clean interface for building prompts from the
        ContextPipeline's output. It maps ContextResult fields to the existing
        build_prompt parameters.

        Args:
            context: ContextResult from ContextPipeline.build()
            memories: Optional pre-retrieved memories (if not provided, will be gathered)
            config: Optional configuration overrides

        Returns:
            Dict containing the built prompt context

        Example:
            context = await context_pipeline.build(user_input, files)
            prompt_ctx = await prompt_builder.build_prompt_from_context(context)
            final_prompt = prompt_builder._assemble_prompt(prompt_ctx, user_input)
        """
        # Import here to avoid circular dependency
        from core.context_pipeline import ContextResult

        if not isinstance(context, ContextResult):
            raise TypeError(f"Expected ContextResult, got {type(context)}")

        # Extract intent overrides (if intent classifier ran)
        retrieval_overrides = {}
        weight_overrides = {}
        gate_threshold_override = None
        if hasattr(context, 'intent') and context.intent is not None:
            retrieval_overrides = context.intent.retrieval_overrides or {}
            weight_overrides = context.intent.weight_overrides or {}
            gate_threshold_override = getattr(context.intent, 'gate_threshold_override', None)

        # Extract intent type for web search gating
        _intent_type = None
        if context.intent is not None:
            _it = getattr(context.intent, 'intent_type', None)
            _intent_type = getattr(_it, 'value', str(_it)) if _it else None

        # Bounded analysis query (item 2): fall back to the short original
        # (pre-merge) user text when processed_query is pathologically long
        # and the original text is not. See ANALYSIS_QUERY_MAX_CHARS above.
        _analysis_query = context.processed_query
        if (context.original_query
                and len(_analysis_query) > ANALYSIS_QUERY_MAX_CHARS
                and len(context.original_query) <= ANALYSIS_QUERY_MAX_CHARS):
            logger.info(
                f"[BUILD_PROMPT] processed_query is {len(_analysis_query)} chars — "
                f"using the {len(context.original_query)}-char original user text for "
                f"retrieval/gating analysis instead (full content still reaches the "
                f"rendered prompt via context.file_context)"
            )
            _analysis_query = context.original_query

        # Map ContextResult to build_prompt parameters
        # When files are uploaded, pass flag to suppress reference docs
        # so file content dominates the context window
        return await self.build_prompt(
            user_input=_analysis_query,
            config=config,
            search_query=_analysis_query if _analysis_query != context.original_query else None,
            current_topic=context.primary_topic,
            fresh_facts=context.extracted_facts if context.is_heavy_topic else None,
            memories=memories,
            stm_summary=context.stm_summary,
            crisis_level=context.crisis_level_str,
            retrieval_overrides=retrieval_overrides,
            weight_overrides=weight_overrides,
            intent_type=_intent_type,
            _suppress_reference_docs=context.has_files,
            _gate_threshold_override=gate_threshold_override,
            _uploaded_filenames=getattr(context, 'uploaded_filenames', None),
        )

    def _should_use_light_path(self, query_analysis: Any, crisis_level: Optional[str]) -> bool:
        """
        Route terse casual acknowledgments to the lightweight context.

        Config-gated (light_prompt.enabled); crisis/elevated tone always gets
        the full context regardless of message shape. crisis_level arrives as
        "CrisisLevel.HIGH" (str of enum, orchestrator path) or as the enum
        value ("crisis_support", "light_support") via crisis_level_str —
        the substring check covers both encodings.
        """
        from config.app_config import LIGHT_PROMPT_ENABLED
        if not LIGHT_PROMPT_ENABLED:
            return False
        if not getattr(query_analysis, "is_small_talk", False):
            return False
        _cl = (crisis_level or "").upper()
        return not any(k in _cl for k in ("HIGH", "MEDIUM", "CONCERN", "SUPPORT", "CRISIS"))

    def _is_continuation_answer(self, user_input: str, query_analysis: Any) -> bool:
        """
        True if `user_input` is a short reply directly answering the assistant's
        immediately-preceding question — route it to the lightweight path so a
        bare token isn't semantically matched against the whole corpus.

        Gated by light_prompt.enabled and the SAME heavy-topic exclusion as the
        casual-ack path (crisis/heavy short replies keep the full apparatus).
        Unlike _should_use_light_path it is NOT blocked by an elevated tone,
        because a short factual answer to the assistant's own question doesn't
        need the support apparatus — the answer is in the immediate exchange, and
        the system-prompt tone still applies to the lightweight prompt.
        """
        from config.app_config import LIGHT_PROMPT_ENABLED
        if not LIGHT_PROMPT_ENABLED:
            return False
        if getattr(query_analysis, "is_heavy_topic", False):
            return False  # crisis/heavy short replies keep full context
        # Content-based crisis guard: the heavy-topic heuristic MISSES crisis
        # phrasing that the tone keyword scorer catches (e.g. "i want to die"),
        # so check the message for any crisis/emotional keyword before stripping
        # context. Benign answers ("amplification") score None and pass.
        try:
            from utils.tone_detector import _check_keyword_crisis
            if _check_keyword_crisis(user_input) is not None:
                return False
        except Exception as e:
            logger.debug(f"[BUILD_PROMPT] continuation crisis-guard failed: {e}")
            return False
        # Cheap peek at the last stored assistant turn (in-memory, no LLM/DB).
        last_resp = ""
        try:
            cm = getattr(self.memory_coordinator, "corpus_manager", None)
            recent = cm.get_recent_memories(count=1) if cm else []
            if recent:
                last_resp = recent[0].get("response", "") or ""
        except Exception as e:
            logger.debug(f"[BUILD_PROMPT] continuation-answer peek failed: {e}")
            return False
        from utils.query_checker import is_continuation_answer
        return is_continuation_answer(user_input, last_resp)

    async def _build_lightweight_context(self, user_input: str, stm_summary: Optional[Dict[str, Any]] = None,
                                          codebase_changes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build lightweight context for small-talk queries."""
        try:
            # Just get recent conversations for small-talk
            recent = await self.context_gatherer._get_recent_conversations(3)

            context = {
                "recent_conversations": recent,
                "memories": [],
                "user_profile": "",
                "narrative_state": "",  # No narrative context for small-talk
                "summaries": [],
                "recent_summaries": [],
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],  # No personal notes for small-talk
                "user_uploads": [],   # No uploads for small-talk
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],  # No proposals for small-talk
                "graph_context": [],  # No graph for small-talk
                "unresolved_threads": [],  # No threads for small-talk
                "proactive_insights": [],  # No insights for small-talk
                "web_search_results": None,  # No web search for small-talk
                "codebase_changes": codebase_changes or {},  # Git changes since last session
            }

            # Add STM summary if provided
            if stm_summary is not None:
                context["stm_summary"] = stm_summary

            # Add memory ID map for citations
            context["memory_id_map"] = self.context_gatherer.memory_id_map if hasattr(self.context_gatherer, 'memory_id_map') else {}

            # Ambiguity detection: check if short user message references a phrase
            # that appears in multiple sessions (prevents content conflation)
            try:
                from core.ambiguity_detector import AmbiguityDetector
                ambiguity = AmbiguityDetector.detect(
                    user_input,
                    context.get("recent_conversations", []),
                )
                if ambiguity.is_ambiguous:
                    context["disambiguation_notes"] = [ambiguity.disambiguation_note]
                    logger.info(
                        f"[AmbiguityDetector] Detected: '{ambiguity.ambiguous_phrase}' "
                        f"in {len(ambiguity.matching_entries)} entries across sessions"
                    )
            except Exception as e:
                logger.debug(f"[AmbiguityDetector] Detection failed (non-fatal): {e}")

            # A short follow-up can follow a huge attachment turn. The light
            # path skips retrieval sources, not the conversation field caps.
            return self.token_manager._manage_token_budget(context)
        except Exception as e:
            logger.warning(f"Lightweight context building failed: {e}")
            return {
                "recent_conversations": [],
                "memories": [],
                "user_profile": "",
                "narrative_state": "",
                "summaries": [],
                "recent_summaries": [],
                "memory_id_map": {},
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],
                "user_uploads": [],
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "upcoming_schedule": [],
                "google_calendar": [],
                "relevant_emails": [],
                "proactive_insights": [],
                "web_search_results": None,
                "codebase_changes": codebase_changes or {},
            }

    async def _hygiene_and_caps(self, context: Dict[str, Any], stm_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply deduplication and caps. Delegates to ContentHygiene."""
        return await self._hygiene._hygiene_and_caps(context, stm_summary)

    async def _backfill_recent_conversations(
        self,
        existing_items: List[Dict[str, Any]],
        seen_embeddings: List[tuple],
        seen_content: set,
        target_count: int,
        offset: int,
        embedder,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Backfill recent conversations. Delegates to ContentHygiene."""
        return await self._hygiene._backfill_recent_conversations(
            existing_items, seen_embeddings, seen_content,
            target_count, offset, embedder, similarity_threshold
        )

    def get_token_count(self, text: str, model_name: str) -> int:
        """Get token count for text."""
        return self.token_manager.get_token_count(text, model_name)

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats."""
        return self.token_manager._extract_text(item)

    # Legacy support methods
    async def _gather_context(self, user_input: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy context gathering method - delegates to build_prompt."""
        return await self.build_prompt(user_input, config)

    def _build_feature_inventory(self, context: Dict[str, Any]) -> str:
        """Build a compact feature inventory. Delegates to PromptFormatter."""
        return self.formatter._build_feature_inventory(context)

    def _assemble_prompt(self, context: Dict[str, Any] = None, user_input: str = "",
                        directives: str = "", system_prompt: str = "", **kwargs) -> str:
        """Assemble final prompt string from context. Delegates to PromptFormatter."""
        return self.formatter._assemble_prompt(context, user_input, directives, system_prompt, **kwargs)


# Legacy compatibility class
class PromptBuilder:
    """
    Legacy compatibility wrapper for UnifiedPromptBuilder.

    Provides the old interface for backwards compatibility.
    """

    def __init__(self, model_manager_or_memory_coordinator=None, model_manager=None, **kwargs):
        # Handle both old and new calling conventions
        if model_manager is None and hasattr(model_manager_or_memory_coordinator, 'generate'):
            # Old style: PromptBuilder(model_manager)
            model_manager = model_manager_or_memory_coordinator
            memory_coordinator = None
        else:
            # New style: PromptBuilder(memory_coordinator, model_manager)
            memory_coordinator = model_manager_or_memory_coordinator

        self.unified_builder = UnifiedPromptBuilder(
            memory_coordinator=memory_coordinator,
            model_manager=model_manager,
            **kwargs
        )
        # Expose common attributes for backward compatibility
        self.model_manager = model_manager

    def _assemble_prompt(self, user_input: str = "", context: Dict[str, Any] = None,
                        system_prompt: str = "", directives: str = "", **kwargs) -> str:
        """Expose _assemble_prompt method for backward compatibility.

        Handles both signatures:
        - Legacy: _assemble_prompt(user_input=..., context=..., system_prompt=...)
        - New: _assemble_prompt(context, user_input, directives)
        """
        # Debug logging
        logger.debug(f"_assemble_prompt called with: user_input={type(user_input)}, context={type(context)}")

        # Handle different calling conventions
        if context is None:
            context = {}

        # Use system_prompt as directives if directives not provided
        if system_prompt and not directives:
            directives = system_prompt

        return self.unified_builder._assemble_prompt(context, user_input, directives)

    async def build_prompt(self, user_input: str = "", config: Optional[Dict[str, Any]] = None,
                          memories=None, summaries=None, dreams=None, wiki_snippet=None,
                          semantic_chunks=None, semantic_memory_results=None,
                          time_context=None, recent_conversations=None, **kwargs) -> str:
        """Build prompt and return formatted string.

        Supports both new interface (user_input, config) and legacy interface
        (user_input with specific argument overrides).
        """
        logger.debug(f"PROMPT BUILD LEGACY: Got {len(memories) if memories else 0} memories from parameters")
        if any([memories is not None, summaries is not None, dreams is not None,
                wiki_snippet is not None, semantic_chunks is not None]):
            # Legacy interface - build context manually
            context = {
                "recent_conversations": recent_conversations or [],
                "memories": memories or [],
                "user_profile": "",
                "summaries": summaries or [],
                "reflections": [],
                "dreams": dreams or [],
                "semantic_chunks": semantic_chunks or [],
                "wiki": [{"content": wiki_snippet}] if wiki_snippet else [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "upcoming_schedule": [],
                "google_calendar": [],
                "relevant_emails": [],
                "proactive_insights": [],
            }
            return self.unified_builder._assemble_prompt(context, user_input)
        else:
            # New interface - delegate to UnifiedPromptBuilder
            context = await self.unified_builder.build_prompt(user_input, config)
            return self.unified_builder._assemble_prompt(context, user_input)
