
# -----------------------------------------------------------------------------
# Unified Prompt Builder (refreshed)
#
# Goals of this refactor:
#  - Keep your original flow: gather → (optional) gate → budget → assemble
#  - Make priorities & budgets explicit and tunable via env vars
#  - Add small hygiene helpers (dedupe/cap) to reduce prompt noise
#  - Add thorough comments so future changes are easy/safe
#  - Do NOT delete existing behavior (wiki/semantic/dreams/facts/etc.)
#
# This version integrates:
#   • Unified wiki path via core/wiki_util.get_wiki_snippet
#   • Dreams/Reflections re-enabled in the main build_prompt() path
#   • Summaries diagnostics + graceful fallback when provider returns none
#   • LLM summaries ensure-model wiring + timeout + force switch (boolean or cadence)
#   • Best-effort persistence of generated summaries (so next run can load them)
#   • Unified boolean parsing for DREAMS flag
#   • Removed duplicate summaries section in final assembly
# -----------------------------------------------------------------------------

import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime
from utils.time_manager import TimeManager

from memory.memory_consolidator import MemoryConsolidator
from utils.logging_utils import get_logger, log_and_time
from core.wiki_util import get_wiki_snippet, clean_query

# --- robust feature flags / deps (do not let import-time errors kill this module)
try:
    from config.app_config import SEMANTIC_ONLY_MODE  # feature flag (do NOT rely on this alone)
except Exception:
    SEMANTIC_ONLY_MODE = False

# Try to import semantic search with a graceful no-op fallback
try:
    from knowledge.semantic_search import semantic_search_with_neighbors  # preferred
except Exception:
    # Last-resort no-op so this module always imports
    def semantic_search_with_neighbors(*_args, **_kwargs):
        return []

logger = get_logger("prompt_builder_v2")


class _FallbackCorpusManager:
    """Very small in-memory corpus to support compatibility builders."""

    def __init__(self) -> None:
        from collections import deque

        self._entries = deque(maxlen=500)

    def add_entry(self, query: str, response: str, tags=None, timestamp=None):
        self._entries.append({
            "query": query,
            "response": response,
            "tags": tags or [],
            "timestamp": timestamp or datetime.now(),
        })

    def get_recent_memories(self, count: int = 3):
        return list(self._entries)[-count:]

    def get_summaries(self, _count: int = 3):
        return []

    def add_summary(self, *_, **__):
        return None


class _FallbackMemoryCoordinator:
    """Fallback memory coordinator used when full storage stack is unavailable."""

    def __init__(self) -> None:
        self.corpus_manager = _FallbackCorpusManager()
        self.gate_system = None
        self.current_topic = "general"

    async def store_interaction(self, query: str, response: str, tags=None):
        self.corpus_manager.add_entry(query, response, tags)

    async def get_memories(self, _query: str, limit: int = 20, topic_filter: str | None = None):
        items = list(self.corpus_manager.get_recent_memories(limit))
        if topic_filter and topic_filter != "general":
            items = [i for i in items if f"topic:{topic_filter}" in (i.get("tags") or [])]
        return [
            {
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "timestamp": item.get("timestamp", datetime.now()),
                "source": "recent",
                "metadata": {"truth_score": 0.5},
            }
            for item in reversed(items)
        ][:limit]

    async def retrieve_relevant_memories(self, query: str, config=None):
        memories = await self.get_memories(query, limit=(config or {}).get("max_memories", 20))
        return {"memories": memories, "counts": {"recent": len(memories), "semantic": 0, "hierarchical": 0}}

    def get_summaries(self, limit: int = 3):
        return self.corpus_manager.get_summaries(limit)

    def get_dreams(self, _limit: int = 2):
        return []

    async def get_facts(self, *_, **__):
        return []



# === helpers ================================================================

def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    """Robust truthy/falsey parse that accepts 1/0, true/false, yes/no, on/off."""
    if s is None:
        return default
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

def _as_summary_dict(text: str, tags: list[str], source: str, timestamp: Optional[str] = None) -> dict:
    out = {"content": text, "type": "summary", "tags": tags, "source": source}
    if timestamp:
        out["timestamp"] = timestamp
    return out

def _dedupe_keep_order(items: Iterable[Any], key_fn=lambda x: str(x).strip().lower()) -> List[Any]:
    """De-duplicate while preserving order. key_fn extracts a stable identity."""
    seen = set()
    out: List[Any] = []
    for x in items or []:
        k = key_fn(x)
        if not k:
            continue
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out


def _truncate_list(items: List[Any], limit: int) -> List[Any]:
    """Fast, safe cap. Returns original if already within limit."""
    if not isinstance(items, list):
        return items
    return items[:limit] if len(items) > limit else items


# === Prompt-wide knobs (env overridable) =====================================

# Model capacity (tokens). If not provided, default to a sane ceiling.
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))

# Keep some headroom for the model’s answer; increase if answers get cut off.
RESERVE_FOR_COMPLETION = int(os.getenv("RESERVE_FOR_COMPLETION", "512"))

# Hard prompt token budget = model budget - completion reserve.
# NOTE: you can override explicitly with PROMPT_TOKEN_BUDGET if needed.
DEFAULT_PROMPT_BUDGET = MODEL_MAX_TOKENS - RESERVE_FOR_COMPLETION
PROMPT_TOKEN_BUDGET = int(os.getenv("PROMPT_TOKEN_BUDGET", str(DEFAULT_PROMPT_BUDGET)))

# Stable section priorities for trimming when we exceed budget.
# Higher number means *more* important (trim later).
PRIORITY_ORDER = [
    ("recent_conversations", 7),
    ("semantic_chunks",      6),
    ("memories",             5),
    ("facts",                4),
    ("summaries",            3),
    ("reflections", 2),  # below summaries; adjust if you want them stickier

    ("wiki",                 1),
    ("dreams",               2),   # still included; trimmed early if needed
]

# Soft caps for noisy sections (safe defaults; overridable)
PROMPT_MAX_FACTS       = int(os.getenv("PROMPT_MAX_FACTS", "12"))
PROMPT_MAX_RECENT      = int(os.getenv("PROMPT_MAX_RECENT", "5"))
PROMPT_MAX_SEMANTIC    = int(os.getenv("PROMPT_MAX_SEMANTIC", "8"))
PROMPT_MAX_MEMS        = int(os.getenv("PROMPT_MAX_MEMS", "10"))
PROMPT_MAX_SUMMARIES   = int(os.getenv("PROMPT_MAX_SUMMARIES", "5"))
PROMPT_MAX_DREAMS      = int(os.getenv("PROMPT_MAX_DREAMS", "3"))
PROMPT_MAX_REFLECTIONS=PROMPT_MAX_REFLECTIONS = int(os.getenv("PROMPT_MAX_REFLECTIONS", "3"))
SHOW_EMPTY_SECTIONS = _parse_bool(os.getenv("PROMPT_SHOW_EMPTY_SECTIONS", "1"))
ENABLE_MIDDLE_OUT   = _parse_bool(os.getenv("ENABLE_MIDDLE_OUT", "1"))
USER_INPUT_MAX_TOKENS = int(os.getenv("USER_INPUT_MAX_TOKENS", "4096"))
MEMORY_ITEM_MAX_TOKENS = int(os.getenv("MEMORY_ITEM_MAX_TOKENS", "512"))
SEMANTIC_ITEM_MAX_TOKENS = int(os.getenv("SEMANTIC_ITEM_MAX_TOKENS", "800"))

# Semantic search knobs
SEM_K = int(os.getenv("SEM_K", "8"))
SEM_TIMEOUT_S = float(os.getenv("SEM_TIMEOUT_S", "5.0"))  # allow first-load index/model costs

# Semantic stitching knobs
SEM_STITCH_MAX_CHARS = int(os.getenv("SEM_STITCH_MAX_CHARS", "0"))  # 0 disables stitching
SEM_STITCH_TOP_TITLES = int(os.getenv("SEM_STITCH_TOP_TITLES", "1"))
SEM_STITCH_MIN_CHUNKS = int(os.getenv("SEM_STITCH_MIN_CHUNKS", "2"))


# LLM summaries controls (timeouts + force switch)
SUM_TIMEOUT = float(os.getenv("LLM_SUMMARY_TIMEOUT", "2.0"))  # bound the *LLM call*
SUMMARIES_TASK_TIMEOUT = float(os.getenv("SUMMARIES_TASK_TIMEOUT", str(max(0.8, SUM_TIMEOUT + 0.25))))  # wrap the *task*

# --- FORCE_LLM_SUMMARIES supports: true/false/1/0/always/never OR an integer cadence (e.g., 3)
_raw_force = (os.getenv("FORCE_LLM_SUMMARIES", "") or "").strip().lower()
if _raw_force in ("1", "true", "yes", "always"):
    FORCE_LLM_SUMMARIES: bool | int = True
elif _raw_force in ("0", "false", "no", "never", ""):
    FORCE_LLM_SUMMARIES = False
else:
    try:
        FORCE_LLM_SUMMARIES = int(_raw_force)  # cadence every N prompts
    except ValueError:
        FORCE_LLM_SUMMARIES = False

# Feature flags
DREAMS_ENABLED = _parse_bool(os.getenv("DREAMS_ENABLED", "0"))
REFLECTIONS_ON_DEMAND = _parse_bool(os.getenv("REFLECTIONS_ON_DEMAND", "0"))
REFLECTIONS_ON_DEMAND_TOKENS = int(os.getenv("REFLECTIONS_ON_DEMAND_TOKENS", "180"))
REFLECTIONS_TOP_UP = _parse_bool(os.getenv("REFLECTIONS_TOP_UP", "1"))


# === Unified Prompt Builder ===================================================

class UnifiedPromptBuilder:
    """
    Unified prompt builder that combines all functionality:
      - Memory retrieval from multiple sources
      - Optional cosine/rerank gating via self.gate_system
      - Token budget management
      - Final prompt assembly

    Flow:
      build_prompt()
        ├─ _gather_context(...)
        ├─ _apply_gating(...)     [if gate_system provided]
        ├─ _hygiene_and_caps(...) [dedupe / cap noisy sections]
        ├─ _manage_token_budget(...)
        └─ _assemble_prompt(...)
    """

    def __init__(
        self,
        model_manager,
        memory_coordinator=None,
        tokenizer_manager=None,  # expects an instance; we preserve your fallback
        wiki_manager=None,
        topic_manager=None,
        gate_system=None,
        max_tokens: int = MODEL_MAX_TOKENS,
        reserved_for_output: int = RESERVE_FOR_COMPLETION
    ):
        self.model_manager = model_manager

        resolved_topic_manager = topic_manager
        if resolved_topic_manager is None:
            try:
                from utils.topic_manager import TopicManager

                resolved_topic_manager = TopicManager()
            except Exception:
                resolved_topic_manager = None

        resolved_gate_system = gate_system
        if resolved_gate_system is None:
            try:
                from processing.gate_system import MultiStageGateSystem

                resolved_gate_system = MultiStageGateSystem(model_manager)
            except Exception:
                resolved_gate_system = None

        resolved_memory = memory_coordinator
        if resolved_memory is None:
            resolved_memory = self._build_default_memory_coordinator(
                model_manager=model_manager,
                gate_system=resolved_gate_system,
                topic_manager=resolved_topic_manager,
            )

        self.memory_coordinator = resolved_memory
        self.consolidator = MemoryConsolidator(model_manager=self.model_manager)

        if tokenizer_manager is None or isinstance(tokenizer_manager, type):
            try:
                from models.tokenizer_manager import TokenizerManager

                self.tokenizer_manager = TokenizerManager(model_manager)
            except Exception:
                self.tokenizer_manager = None
        else:
            self.tokenizer_manager = tokenizer_manager

        if wiki_manager is None:
            try:
                from knowledge.WikiManager import WikiManager

                self.wiki_manager = WikiManager()
            except Exception:
                self.wiki_manager = None
        else:
            self.wiki_manager = wiki_manager

        self.topic_manager = resolved_topic_manager
        self.gate_system = resolved_gate_system

        # Token budgeting
        self.max_tokens = max_tokens
        self.reserved_for_output = reserved_for_output
        # Use env-driven budget if set; otherwise compute from ctor args.
        computed_budget = self.max_tokens - self.reserved_for_output
        self.token_budget = PROMPT_TOKEN_BUDGET if PROMPT_TOKEN_BUDGET else computed_budget

        # per-request counter for FORCE cadence
        self._force_counter = 0

        # Simple per-request cache (reset per builder instance)
        self._request_cache: Dict[str, Any] = {}
        logger.info(f"[PROMPT] Using builder class={self.__class__.__name__} module={self.__module__}")
        self._ensure_summaries_model()

        # Best-effort background warmup for semantic index (FAISS + parquet + embedder).
        # Keeps first-query latency low so we don't hit the bounded timeout.
        try:
            import threading
            def _warm_semantic_index_sync():
                try:
                    from knowledge.semantic_search import get_index
                    idx = get_index()
                    idx.load()
                    logger.debug("[PROMPT][Semantic] warmup complete")
                except Exception as e:
                    logger.debug(f"[PROMPT][Semantic] warmup failed: {e}")
            threading.Thread(target=_warm_semantic_index_sync, daemon=True).start()
        except Exception:
            pass

    def _build_default_memory_coordinator(self, *, model_manager, gate_system, topic_manager):
        try:
            from config.app_config import CORPUS_FILE, CHROMA_PATH
            from memory.corpus_manager import CorpusManager
            from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
            from memory.memory_coordinator import MemoryCoordinator
            from utils.time_manager import TimeManager

            corpus_manager = CorpusManager(corpus_file=CORPUS_FILE)
            chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

            if gate_system is None:
                from processing.gate_system import MultiStageGateSystem

                gate_system = MultiStageGateSystem(model_manager)

            return MemoryCoordinator(
                corpus_manager=corpus_manager,
                chroma_store=chroma_store,
                gate_system=gate_system,
                topic_manager=topic_manager,
                model_manager=model_manager,
                time_manager=TimeManager(),
            )
        except Exception as exc:
            logger.debug(f"[PromptBuilder] Falling back to in-memory coordinator: {exc}")
            fallback = _FallbackMemoryCoordinator()
            if topic_manager is not None:
                fallback.current_topic = getattr(topic_manager, "last_topic", "general") or "general"
            return fallback

    # --- Middle-out compression helpers --------------------------------------
    def _middle_out(self, text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
        """Compress text by keeping the head and tail, trimming the middle.

        Uses tokenizer_manager to decide if compression is needed, but slices by characters
        to avoid requiring a full encode/decode path. Good enough for budget safety.
        """
        if not ENABLE_MIDDLE_OUT:
            return text

        try:
            model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else "default"
            toks = self.get_token_count(text or "", model_name)
        except Exception:
            toks = len((text or "").split())
        if toks <= max_tokens:
            return text
        s = text or ""
        # Roughly map tokens to characters; assume ~4 chars per token as a conservative heuristic
        approx_chars = max_tokens * 4
        head_chars = int(approx_chars * head_ratio)
        tail_chars = max(0, approx_chars - head_chars)
        if len(s) <= approx_chars:
            return s
        head = s[:head_chars]
        tail = s[-tail_chars:] if tail_chars > 0 else ""
        snip = f"\n… [middle-out snipped {len(s) - (head_chars + tail_chars)} chars] …\n"
        return head + snip + tail

    def _ensure_summaries_model(self):
        """
        Make sure a usable API model is registered + active for LLM summaries.
        Controlled via env:
        LLM_SUMMARY_ALIAS (default: gpt-4o-mini)
        LLM_SUMMARY_PROVIDER_ID (default: openai/gpt-4o-mini)
        """
        alias = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
        provider_id = os.getenv("LLM_SUMMARY_PROVIDER_ID", "openai/gpt-4o-mini")

        # Register if missing
        if not getattr(self.model_manager, "api_models", None):
            self.model_manager.api_models = {}  # safety
        if alias not in self.model_manager.api_models:
            try:
                self.model_manager.load_openai_model(alias, provider_id)
                logger.info(f"[PROMPT][LLM] registered alias={alias} -> {provider_id}")
            except Exception as e:
                logger.error(f"[PROMPT][LLM] failed to register {alias} -> {provider_id}: {e}")

        # Do NOT switch the active model here; keep the user's choice for final responses.
        # We will temporarily switch during summaries generation and then restore.
        active = getattr(self.model_manager, "get_active_model_name", lambda: None)()
        key_present = "yes" if os.getenv("OPENAI_API_KEY") else "no"
        mapped = self.model_manager.api_models.get(alias)
        logger.info(f"[PROMPT][LLM] summaries model available: active={active} map[{alias}]={mapped} OPENAI_API_KEY? {key_present}")

    # Small helper to keep “speed run” replies concise.
    def _decide_gen_params(self, user_input: str):
        short = any(
            k in (user_input or "").lower()
            for k in ("time this", "quick", "faster", "speed test", "lol", "sup")
        )
        if short:
            # snappy
            return {"max_tokens": 220, "temperature": 0.5, "top_p": 0.9}
        return {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.9}

    # --- Facts retrieval (kept intact, with logging & dedupe) -----------------
    async def get_facts(self, query: str, limit: int = 8):
        """Fetch fact nodes from coordinator; fall back safely if API differs."""
        try:
            if hasattr(self.memory_coordinator, "get_facts"):
                facts = await self.memory_coordinator.get_facts(query=query, limit=limit)
            elif hasattr(self.memory_coordinator, "search_by_type"):
                facts = await self.memory_coordinator.search_by_type("facts", query=query, limit=limit)
            else:
                facts = []
        except Exception as e:
            logger.debug(f"[PROMPT][Facts] retrieval failed: {e}")
            facts = []

        # De-dupe by canonical string
        def _k(f):
            if isinstance(f, dict):
                return (f.get("content") or "").strip().lower()
            return str(getattr(f, "content", "")).strip().lower()

        deduped = _dedupe_keep_order(facts, key_fn=_k)
        logger.debug(f"[PROMPT][Facts] Included {len(deduped)} unique facts")
        return deduped

    # --- Unified wiki helpers --------------------------------------------------
    def _wiki_cache_key(self, raw_query: str) -> str:
        # Include both raw and cleaned variants to maximize hit rate but avoid collisions
        return f"{(raw_query or '').strip()}|{clean_query(raw_query)}"

    def _get_wiki_snippet_cached(self, raw_query: str) -> str:
        """
        Unified accessor with per-request cache.
        Returns a formatted snippet or '' on miss.
        """
        key = self._wiki_cache_key(raw_query)
        if key in self._request_cache:
            return self._request_cache[key]
        snippet = get_wiki_snippet(raw_query) or ""
        self._request_cache[key] = snippet
        return snippet

    # --- Gating wrapper (mem/wiki/semantic) -----------------------------------
    async def _apply_gating(self, user_input: str, ctx: dict) -> dict:
        """
        Gate memories/wiki/semantic using self.gate_system if available.
        Wiki uses the unified accessor and will NOT overwrite an existing good snippet.
        """
        if not self.gate_system:
            # Even without a gate system, populate wiki if missing (fast path).
            if not ctx.get("wiki"):
                ctx["wiki"] = self._get_wiki_snippet_cached(user_input)
            return ctx

        try:
            # Memories (fail-soft: if gate drops everything, keep a couple originals)
            orig_mems = ctx.get("memories", []) or []
            gated_mems = await self.gate_system.filter_memories(user_input, orig_mems)
            if not gated_mems and orig_mems:
                gated_mems = orig_mems[: min(2, len(orig_mems))]

            # Wiki: unified accessor (no legacy HTTP, small cached call)
            wiki_snip = ctx.get("wiki") or self._get_wiki_snippet_cached(user_input)

            # Semantic chunks (fail-soft)
            orig_sem = ctx.get("semantic_chunks", []) or []
            gated_sem = await self.gate_system.filter_semantic_chunks(user_input, orig_sem)
            if not gated_sem and orig_sem:
                gated_sem = orig_sem[: min(3, len(orig_sem))]

            return {
                **ctx,
                "memories": gated_mems,
                "wiki": wiki_snip,
                "semantic_chunks": gated_sem,
                "reflections": ctx.get("reflections", []),
                "wiki_snippet": wiki_snip,
            }


        except Exception as e:
            logger.debug(f"[PROMPT][Gate] Gating skipped due to error: {e}")
            # Ensure wiki present even if gating failed
            if not ctx.get("wiki"):
                ctx["wiki"] = self._get_wiki_snippet_cached(user_input)
            return ctx

    # --- Summaries (upgrade: prefer stored; else LLM; else micro) ------------
    async def _llm_summarize_recent(self, recents: List[dict], target_tokens: int = 160) -> str:
        if not recents:
            return ""

        def _clip(s: Optional[str], n: int) -> str:
            s = s or ""
            return s[:n]

        joined = "\n\n".join(
            f"User: {_clip(r.get('query'), 240)}\nAssistant: {_clip(r.get('response'), 300)}"
            for r in recents[:5]
            if (r.get('query') or r.get('response'))
        )

        system = (
            "You are a neutral note-taker. Summarize prior exchanges in third-person "
            "(no 'you'), concise and factual. 2–3 sentences. No praise, no coaching."
        )
        prompt = (
            f"{system}\n\nCONVERSATION EXCERPTS:\n{joined}\n\n"
            "Write one neutral recap (2–3 sentences). No second-person, no pep-talk."
        )

        mm = getattr(self, "model_manager", None)
        if not mm or not hasattr(mm, "generate_once"):
            logger.debug("[PROMPT][Summaries] No model_manager.generate_once; cannot LLM summarize")
            return ""
        alias = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
        if alias not in getattr(mm, "api_models", {}):
            logger.debug(f"[PROMPT][Summaries] alias '{alias}' not in api_models; skipping LLM summarize")
            return ""

        try:
            # 🔒 bound the call so it never stalls the build
            prev_model = None
            try:
                if hasattr(mm, "get_active_model_name"):
                    prev_model = mm.get_active_model_name()
                # temporarily switch to summaries alias
                if hasattr(mm, "switch_model"):
                    mm.switch_model(alias)
            except Exception:
                pass
            active_now = getattr(mm, "get_active_model_name", lambda: alias)()
            logger.debug(f"[PROMPT][Summaries] LLM call → model={active_now}, max_tokens={target_tokens}, timeout={SUM_TIMEOUT:.1f}")
            # ensure the generate_once itself respects SUM_TIMEOUT via asyncio.wait_for
            text = await asyncio.wait_for(mm.generate_once(prompt, max_tokens=target_tokens), timeout=SUM_TIMEOUT)
            # restore previous model
            try:
                if prev_model and hasattr(mm, "switch_model"):
                    mm.switch_model(prev_model)
                    try:
                        restored = mm.get_active_model_name() if hasattr(mm, "get_active_model_name") else prev_model
                        logger.info(f"[PROMPT][LLM] restored active model after summaries: {restored}")
                    except Exception:
                        pass
            except Exception:
                pass
            text = (text or "").strip()
            if not text:
                logger.debug("[PROMPT][Summaries] LLM returned empty text")
                return ""
            low = text.lower()
            if low.startswith(("q:", "that's great", "great to hear")):
                logger.debug("[PROMPT][Summaries] LLM output flagged as pep-talk/Q&A; discarding")
                return ""
            logger.debug(f"[PROMPT][Summaries] LLM result len={len(text)}")
            return text
        except asyncio.TimeoutError:
            logger.debug(f"[PROMPT][Summaries] LLM call timed out ({SUM_TIMEOUT:.1f}s)")
            return ""
        except Exception as e:
            logger.debug(f"[PROMPT][Summaries] generate_once error: {e}")
            return ""

    async def _persist_summary(self, text: str):
        """
        Best-effort persistence so next build can retrieve a stored summary.
        Will try memory_coordinator.add_summary, then corpus_manager.add_summary,
        then (optional) consolidator hook if available.
        """
        if not text:
            return
        # Coordinator path if available
        try:
            if hasattr(self.memory_coordinator, "add_summary"):
                await self.memory_coordinator.add_summary(text)
                return
        except Exception:
            pass
        # CorpusManager fallback
        try:
            cm = getattr(self.memory_coordinator, "corpus_manager", None)
            if cm and hasattr(cm, "add_summary"):
                tm = TimeManager()
                cm.add_summary({
                    "content": text,
                    # store as ISO for consistent downstream display
                    "timestamp": tm.current_iso(),
                    "type": "summary",
                    "tags": ["summary:stored", "source:llm"]
                })
                # if corpus manager persists to disk on add_summary internally, great;
                # otherwise an external save hook will pick it up later.
                return
        except Exception:
            pass
        # Optional consolidator hook
        try:
            if hasattr(self, "consolidator") and hasattr(self.consolidator, "record_summary"):
                await self.consolidator.record_summary(text)
        except Exception:
            pass

    async def _get_summaries(self, count: int) -> List[Dict]:
        logger.debug("[PROMPT][Summaries] _get_summaries START (count=%s)", count)

        # Determine whether to force an LLM summary for this build (boolean or cadence)
        _force = FORCE_LLM_SUMMARIES
        should_force = False
        if _force is True:
            should_force = True
        elif isinstance(_force, int) and _force > 0:
            should_force = (self._force_counter % _force == 0)

        summaries: List[Dict] = []

        # 0) Forced path (preempts stored reads for this build)
        if should_force:
            recent = await self._get_recent_conversations(max(5, count * 2))
            logger.debug("[PROMPT][Summaries] FORCE_LLM active; recent_count=%s", len(recent))
            if recent:
                llm = await self._llm_summarize_recent(recent, target_tokens=160)
                if llm:
                    logger.debug("[PROMPT][Summaries] FORCE_LLM result_len=%s", len(llm))
                    # Persist so the next build can load it as stored
                    try:
                        await self._persist_summary(llm)
                    except Exception as e:
                        logger.debug(f"[PROMPT][Summaries] persist failed: {e}")

                    # Also include the most recent stored ones (minus duplicates), up to `count`
                    stored: list[dict] = []
                    try:
                        cm = getattr(self.memory_coordinator, "corpus_manager", None)
                        if cm and hasattr(cm, "get_summaries"):
                            stored = cm.get_summaries(count * 2) or []
                    except Exception:
                        stored = []

                    def _is_real_text(t: str) -> bool:
                        t2 = (t or "").strip().lower()
                        return t2 and not (t2.startswith("summary of ") or t2.startswith("q:") or " q: " in t2 or " a: " in t2)

                    # Filter placeholders and dedupe by content text
                    cleaned = []
                    seen_txts = {llm.strip()}
                    for s in stored:
                        txt = (s.get("content") if isinstance(s, dict) else str(s)).strip()
                        if not _is_real_text(txt):
                            continue
                        if txt in seen_txts:
                            continue
                        seen_txts.add(txt)
                        cleaned.append({
                            "content": txt,
                            "timestamp": s.get("timestamp") if isinstance(s, dict) else None,
                            "type": "summary",
                            "tags": ["summary:stored"],
                            "source": "stored",
                        })

                    # Build final: forced first (with timestamp), then freshest stored, capped to count
                    from utils.time_manager import TimeManager
                    forced_ts = TimeManager().current_iso()
                    final = [_as_summary_dict(llm, ["llm_summary", "forced"], "llm_forced", forced_ts)]
                    final.extend(cleaned[: max(0, count - 1)])
                    return final


        # 1) Try stored summaries first
        try:
            if hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_summaries'):
                logger.debug("[PROMPT][Summaries] Using corpus_manager.get_summaries")
                summaries = self.memory_coordinator.corpus_manager.get_summaries(count)
            elif hasattr(self.memory_coordinator, 'get_summaries'):
                logger.debug("[PROMPT][Summaries] Using memory_coordinator.get_summaries")
                summaries = self.memory_coordinator.get_summaries(count)
        except Exception as e:
            logger.error(f"[PROMPT][Summaries] Error getting summaries: {e}")
            summaries = []

        logger.debug("[PROMPT][Summaries] loaded=%s", len(summaries) if summaries else 0)

        # Normalize and filter placeholders (the “Q:/A:” dump style)
        def _looks_placeholder(t: str) -> bool:
            t2 = (t or "").lower()
            return (t2.startswith("summary of ") or t2.startswith("q:") or " q: " in t2 or " a: " in t2)

        # Keep dicts and preserve timestamps, filter placeholders based on content
        normalized: List[Dict] = []
        for s in summaries or []:
            if isinstance(s, dict):
                txt = (s.get("content") or "").strip()
                ts = s.get("timestamp")
            else:
                txt = str(s).strip(); ts = None
            if txt and not _looks_placeholder(txt):
                normalized.append({
                    "content": txt,
                    "timestamp": ts,
                    "type": "summary",
                    "tags": ["summary:stored"],
                    "source": "stored",
                })
        logger.debug("[PROMPT][Summaries] have_real=%s", len(normalized))

        if normalized:
            return normalized[:count]

        # 2) LLM fallback (only if not forced above)
        try:
            recent = await self._get_recent_conversations(max(5, count * 2))
            if recent:
                logger.debug("[PROMPT][Summaries] Generating LLM fallback")
                llm = await self._llm_summarize_recent(recent, target_tokens=160)
                if llm:
                    logger.debug("[PROMPT][Summaries] Generated LLM summary")
                    # Persist so next call can load it stored
                    try:
                        await self._persist_summary(llm)
                    except Exception as e:
                        logger.debug(f"[PROMPT][Summaries] persist failed: {e}")
                    return [{
                        "content": llm,
                        "type": "summary",
                        "tags": ["llm_summary", "consolidated_summary"],
                        "source": "llm_fallback"
                    }]
            logger.debug("[PROMPT][Summaries] LLM returned nothing; falling back to micro")
        except Exception as e:
            logger.error(f"[PROMPT][Summaries] LLM fallback failed: {e}")

        # 3) Micro fallback (never return empty)
        logger.debug("[PROMPT][Summaries] Using micro fallback")
        micro = self._fallback_micro_summary(recent if 'recent' in locals() else [])
        return [{
            "content": micro,
            "type": "summary",
            "tags": ["summary:fallback_micro"],
            "source": "fallback_micro"
        }]
    async def _get_reflections(self, count: int):
        """Fetch small reflection notes, with safe fallbacks.

        Order of attempts:
          1) memory_coordinator.get_reflections(limit=count)
          2) if empty → memory_coordinator.search_by_type('reflections', ...)
          3) coordinator's chroma_store recent texts (if exposed)
        """
        out = []
        try:
            mc = self.memory_coordinator
            if hasattr(mc, "get_reflections"):
                res = await mc.get_reflections(limit=count)
                # If the coordinator returns nothing, fall through to other sources
                if res:
                    return res
            # fallback: typed search if available
            if hasattr(mc, "search_by_type"):
                items = await mc.search_by_type("reflections", query="", limit=count)
                for r in items or []:
                    txt = (r.get("content") if isinstance(r, dict) else str(r)).strip()
                    ts  = (r.get("timestamp") if isinstance(r, dict) else None)
                    if txt:
                        out.append({
                            "content": txt,
                            "type": "reflection",
                            "tags": ["source:semantic"],
                            "source": "semantic",
                            "timestamp": ts,
                        })
                return out[:count]
        except Exception:
            pass
        # If still empty, last-ditch: read directly from a chroma-like store, if present
        try:
            store = getattr(getattr(self, "memory_coordinator", None), "chroma_store", None)
            if not out and store:
                if hasattr(store, "get_recent"):
                    items = store.get_recent("reflections", limit=count) or []
                    for it in items:
                        txt = (it.get("content") or "").strip()
                        ts  = (it.get("metadata") or {}).get("timestamp")
                        if txt:
                            out.append({
                                "content": txt,
                                "type": "reflection",
                                "tags": ["source:chroma"],
                                "source": "chroma",
                                "timestamp": ts,
                            })
                    return out[:count]
                elif hasattr(store, "get_recent_texts"):
                    recent = store.get_recent_texts("reflections", limit=count) or []
                    for t in recent:
                        t = (t or "").strip()
                        if t:
                            out.append({
                                "content": t,
                                "type": "reflection",
                                "tags": ["source:chroma"],
                                "source": "chroma"
                            })
                    return out[:count]
        except Exception:
            pass

        return out[:count]
    async def _reflect_on_demand(self, recent_conversations: list) -> list:
        """If stored/semantic reflections are empty, synthesize a tiny reflection now."""
        try:
            if not REFLECTIONS_ON_DEMAND or not recent_conversations:
                return []
            # Build a very small prompt from the last few exchanges
            def _slice(e):
                q = (e.get("query") or "").strip()
                a = (e.get("response") or "").strip()
                bits = []
                if q: bits.append(f"User: {q[:200]}")
                if a: bits.append(f"Assistant: {a[:240]}")
                return "\n".join(bits)

            excerpts = [_slice(e) for e in recent_conversations[-8:] if (e.get("query") or e.get("response"))]
            if not excerpts:
                return []

            prompt = (
                "You are a neutral QA reviewer.\n"
                "Return three short bullets: (1) Went well, (2) Improve, (3) Insight.\n\n"
                + "\n\n".join(excerpts)
                + "\n\nBullets:"
            )

            # Use the same model manager the builder already has access to via coordinator
            mm = getattr(self.memory_coordinator, "model_manager", None)
            if not mm or not hasattr(mm, "generate_once"):
                return []

            # Temporarily switch to a reflection-friendly model alias if configured
            prev_model = None
            alias = os.getenv("LLM_REFLECTION_ALIAS", os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini"))
            try:
                if hasattr(mm, "get_active_model_name"):
                    prev_model = mm.get_active_model_name()
                if hasattr(mm, "api_models") and alias in getattr(mm, "api_models", {}):
                    if hasattr(mm, "switch_model"):
                        mm.switch_model(alias)
            except Exception:
                pass

            txt = await mm.generate_once(prompt, max_tokens=REFLECTIONS_ON_DEMAND_TOKENS)

            # Restore previous model
            try:
                if prev_model and hasattr(mm, "switch_model"):
                    mm.switch_model(prev_model)
                    try:
                        restored = mm.get_active_model_name() if hasattr(mm, "get_active_model_name") else prev_model
                        logger.info(f"[PROMPT][LLM] restored active model after reflections: {restored}")
                    except Exception:
                        pass
            except Exception:
                pass
            txt = (txt or "").strip()
            if not txt:
                return []

            # Best-effort persist (won’t crash if absent)
            try:
                if hasattr(self.memory_coordinator, "add_reflection"):
                    await self.memory_coordinator.add_reflection(txt, tags=["session:adhoc", "origin:builder"], source="builder")
            except Exception:
                pass

            # Stamp timestamp for immediate display
            try:
                from utils.time_manager import TimeManager
                ts = TimeManager().current_iso()
            except Exception:
                from datetime import datetime
                ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            return [{
                "content": txt,
                "type": "reflection",
                "tags": ["source:adhoc"],
                "source": "adhoc",
                "timestamp": ts,
            }]
        except Exception:
            return []

    # --- Public entry point ----------------------------------------------------

    async def _bounded(self, coro, timeout: float, default):
        """
        Await `coro` with a timeout. On timeout or error, return `default`.
        Accepts either a coroutine object or an awaitable.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"[PROMPT][_bounded] timeout after {timeout:.2f}s")
            return default
        except Exception as e:
            logger.debug(f"[PROMPT][_bounded] failed: {e}")
            return default

    @log_and_time("Build Prompt")
    async def build_prompt(
        self,
        *,
        user_input: str,
        search_query: Optional[str] = None,
        personality_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        current_topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parallelized prompt-context builder.
        Returns a context dict (prompt_ctx) compatible with orchestrator._assemble_prompt.
        """
        # increment per-call counter (used by FORCE cadence)
        self._force_counter += 1

        logger.debug(f"[PROMPT] Building prompt (user_input.head={repr((user_input or '')[:80])}, topic={current_topic})")
        start_time = time.time()

        logger.debug(f"[PROMPT][TRACE] timeouts: sums={SUMMARIES_TASK_TIMEOUT:.2f}s (LLM={SUM_TIMEOUT:.2f}s)")

        # Choose the retrieval query
        retrieval_query = (search_query or user_input or "").strip()

        # Dreams toggle (unified boolean parse)
        include_dreams = DREAMS_ENABLED

        # ----------------------------
        # Run all subfetches in parallel (bounded)
        # ----------------------------
        raw_user_input = user_input

        tasks = {
            "recent": self._bounded(self._get_recent_conversations(5), 0.3, []),
            "mems": self._bounded(
                self.memory_coordinator.get_memories(retrieval_query, limit=PROMPT_MAX_MEMS),
                1.5, []
            ),
            "facts": self._bounded(self.get_facts(retrieval_query, limit=PROMPT_MAX_FACTS), 0.8, []),
            "sums": self._bounded(self._get_summaries(PROMPT_MAX_SUMMARIES), SUMMARIES_TASK_TIMEOUT, []),
            "dreams":
            self._bounded(self._get_dreams(PROMPT_MAX_DREAMS), 0.3, []) if include_dreams else asyncio.sleep(0, result=[]),
            # Use unified semantic retrieval (includes FAISS + Chroma fallback and hot-cache)
            "sem": self._bounded(
                self._get_semantic_chunks(retrieval_query),
                SEM_TIMEOUT_S, []
            ),
            "refl": self._bounded(self._get_reflections(PROMPT_MAX_REFLECTIONS), 1.5, []),

            "wiki": asyncio.get_running_loop().run_in_executor(
                None, self._get_wiki_snippet_cached, (current_topic or retrieval_query)
            ),
        }

        recent, mems, facts, sums, dreams, sem, refl, wiki = await asyncio.gather(*tasks.values())
        logger.debug(
            f"[PROMPT] fetched | recent={len(recent)} sums={len(sums)} sem={len(sem)} "
            f"facts={len(facts)} dreams={len(dreams)} wiki_len={len(wiki) if wiki else 0}"
        )

        # Minimal fallbacks to avoid empty critical sections
        #  - If mems is empty, include up to 2 most recent Q/A pairs as soft memories
        if not mems and recent:
            soft = []
            for e in recent[:2]:
                q = (e.get("query") or "").strip(); a = (e.get("response") or "").strip()
                if q or a:
                    soft.append({"query": q, "response": a, "type": "memory", "tags": ["source:recent_fallback"]})
            mems = soft

        # Keep a copy of raw reflections before filtering
        raw_refl = list(refl or [])

        # Opportunistic synchronous reflection read from corpus if async path returned empty
        if not refl:
            try:
                cm = getattr(self.memory_coordinator, "corpus_manager", None)
                if cm and hasattr(cm, "get_items_by_type"):
                    items = cm.get_items_by_type("reflection", limit=max(PROMPT_MAX_REFLECTIONS * 3, 6))
                    refl = items or []
            except Exception:
                pass
        logger.debug(f"[PROMPT][Refl] raw_refl_count={len(raw_refl)} after_corpus_fallback={len(refl or [])}")

        # Filter reflections to "session-level" by default (shutdown-produced)
        try:
            def _is_session_refl(r: dict) -> bool:
                if not isinstance(r, dict):
                    return False
                tags = r.get("tags") or []
                src  = (r.get("source") or "").lower()
                # Accept shutdown-derived or any session:* reflections
                if any((str(t).lower().startswith("session:") for t in tags)):
                    return True
                if "session:end" in [str(t).lower() for t in tags]:
                    return True
                if src.startswith("shutdown"):
                    return True
                if any(str(t).lower().startswith("source:shutdown") for t in tags):
                    return True
                return False

            # Sort by timestamp desc when available
            def _parse_ts(r):
                try:
                    from datetime import datetime, timezone
                    ts = r.get("timestamp") if isinstance(r, dict) else None
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if isinstance(ts, datetime):
                        if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
                            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                        return ts
                except Exception:
                    pass
                from datetime import datetime as _dt
                return _dt.min

            refl = sorted((refl or []), key=_parse_ts, reverse=True)
            raw_refl = sorted((raw_refl or []), key=_parse_ts, reverse=True)

            session_refl = [r for r in (refl or []) if _is_session_refl(r)]
            if session_refl:
                # Top up with freshest non-session reflections to reach the cap
                cap = PROMPT_MAX_REFLECTIONS
                out_list = []
                seen = set()
                def _k(it):
                    if isinstance(it, dict):
                        return (it.get("content") or "").strip().lower()
                    return str(it).strip().lower()
                for r in session_refl:
                    k = _k(r)
                    if k and k not in seen:
                        out_list.append(r)
                        seen.add(k)
                        if len(out_list) >= cap:
                            break
                if len(out_list) < cap:
                    for r in (raw_refl or []):
                        k = _k(r)
                        if k and k not in seen:
                            out_list.append(r)
                            seen.add(k)
                            if len(out_list) >= cap:
                                break
                refl = out_list
            elif raw_refl:
                refl = raw_refl[:PROMPT_MAX_REFLECTIONS]
        except Exception:
            pass
        logger.debug(f"[PROMPT][Refl] final_before_adhoc={len(refl or [])}")

        # Last-resort/top-up: synthesize micro-reflections to reach the cap
        if REFLECTIONS_TOP_UP and recent:
            try:
                cap = PROMPT_MAX_REFLECTIONS
                tries = 0
                while len(refl or []) < cap and tries < cap:
                    od = await self._reflect_on_demand(recent)
                    tries += 1
                    if od:
                        # extend while deduping by text
                        existing = {(r.get('content') or '').strip() if isinstance(r, dict) else str(r).strip() for r in (refl or [])}
                        for item in od:
                            txt = (item.get('content') or '').strip() if isinstance(item, dict) else str(item).strip()
                            if txt and txt not in existing:
                                refl = (refl or []) + [item]
                                existing.add(txt)
                    else:
                        break
            except Exception:
                pass
        # Optional: always inject one fresh reflection even if cap already full
        try:
            force_new = _parse_bool(os.getenv("REFLECTIONS_FORCE_NEW", "0"))
            if force_new and recent:
                od = await self._reflect_on_demand(recent)
                if od:
                    # replace the oldest (tail) if different
                    existing_texts = [(r.get('content') or '').strip() if isinstance(r, dict) else str(r).strip() for r in (refl or [])]
                    new_txt = (od[0].get('content') or '').strip()
                    if new_txt and new_txt not in existing_texts:
                        if refl:
                            refl = list(refl)
                            refl[-1] = od[0]
                        else:
                            refl = od
        except Exception:
            pass
        logger.debug(f"[PROMPT][Refl] final_after_adhoc={len(refl or [])}")

        # Build initial ctx
        ctx: Dict[str, Any] = {
            "recent_conversations": recent or [],
            "memories": mems or [],
            "facts": facts or [],
            "summaries": sums or [],
            "dreams": dreams or [],
            "semantic_chunks": sem or [],
            "reflections": refl or [],   # session reflections preferred
            "wiki": wiki or "",
            "time_context": self._get_time_context() if hasattr(self, "_get_time_context") else "",
            "current_topic": current_topic or "",
            "system_prompt": system_prompt or "",
            "user_input": user_input or "",
            "raw_user_input": raw_user_input,
        }

        # ----------------------------
        # Gating (keep existing behavior)
        # ----------------------------
        try:
            gated_ctx = await self._apply_gating(retrieval_query, ctx)
        except Exception as e:
            logger.debug(f"[PROMPT] gating error (fail-open): {e}")
            gated_ctx = ctx

        raw_summaries = gated_ctx.get("summaries") or []
        logger.debug(f"[PROMPT][TRACE] summaries_pre_hygiene={len(raw_summaries)} FORCE_LLM={FORCE_LLM_SUMMARIES}")

        # ----------------------------
        # Hygiene & caps (dedupe + truncate)
        # ----------------------------
        def _safe_list(x):
            return x if isinstance(x, list) else []

        facts_c      = _truncate_list(_dedupe_keep_order(_safe_list(gated_ctx.get("facts"))), PROMPT_MAX_FACTS)
        summaries_c  = _truncate_list(_dedupe_keep_order(_safe_list(gated_ctx.get("summaries"))), PROMPT_MAX_SUMMARIES)
        refl_c = _truncate_list(_dedupe_keep_order(_safe_list(gated_ctx.get("reflections"))), PROMPT_MAX_REFLECTIONS)

        recent_c     = _truncate_list(_safe_list(gated_ctx.get("recent_conversations") or []), PROMPT_MAX_RECENT)
        # Dedupe semantic chunks by their cleaned or raw text
        def _sem_key(ch: Any) -> str:
            if isinstance(ch, dict):
                return (
                    (ch.get("filtered_content") or ch.get("text") or ch.get("content") or "")
                    .strip().lower()
                )
            return str(ch).strip().lower()
        sem_dedup = _dedupe_keep_order(_safe_list(gated_ctx.get("semantic_chunks") or []), key_fn=_sem_key)
        # Collapse near-duplicates where one text is a large substring of another
        collapsed: List[Dict[str, Any]] = []
        for ch in sem_dedup:
            try:
                t = (ch.get("filtered_content") or ch.get("text") or ch.get("content") or "").strip()
            except Exception:
                t = str(ch).strip()
            if not t:
                continue
            too_similar = False
            for kept in collapsed:
                kt = (kept.get("filtered_content") or kept.get("text") or kept.get("content") or "").strip()
                long, short = (kt, t) if len(kt) >= len(t) else (t, kt)
                if short and short in long and len(short) / max(1, len(long)) >= 0.85:
                    too_similar = True
                    break
            if not too_similar:
                collapsed.append(ch)
        # Optional stitching: merge multiple chunks per top title into large excerpts
        def _stitch_by_title(items: List[Dict]) -> List[Dict]:
            if SEM_STITCH_MAX_CHARS <= 0:
                return items
            by_title: Dict[str, List[Dict]] = {}
            for it in items:
                title = (it.get("title") or "").strip()
                key = title.lower()
                by_title.setdefault(key, []).append(it)
            scored = []
            for k, lst in by_title.items():
                mx = max((x.get("relevance_score", 0.0) for x in lst), default=0.0)
                scored.append((k, len(lst), mx))
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            out: List[Dict] = []
            picked = 0
            for key, _cnt, _mx in scored:
                parts: List[str] = []
                kept = 0
                for it in sorted(by_title[key], key=lambda x: x.get("relevance_score", 0.0), reverse=True):
                    txt = (it.get("filtered_content") or it.get("text") or it.get("content") or "").strip()
                    if not txt:
                        continue
                    parts.append(txt)
                    kept += 1
                    if kept >= max(1, SEM_STITCH_MIN_CHUNKS):
                        # keep adding; we'll clip by char budget
                        pass
                stitched = "\n\n".join(parts)
                stitched = stitched[: SEM_STITCH_MAX_CHARS]
                if stitched:
                    sample = by_title[key][0]
                    out.append({
                        "text": stitched,
                        "title": sample.get("title") or "",
                        "source": sample.get("source") or sample.get("namespace") or "wikipedia",
                        "timestamp": sample.get("timestamp") or sample.get("ts") or "",
                        "relevance_score": max((x.get("relevance_score", 0.0) for x in by_title[key]), default=0.0),
                    })
                    picked += 1
                if picked >= max(1, SEM_STITCH_TOP_TITLES):
                    break
            # Fill remaining slots with non-stitched items from other titles
            if len(out) < PROMPT_MAX_SEMANTIC:
                seen_keys = { (o.get("title") or "").strip().lower() for o in out }
                for it in items:
                    k = (it.get("title") or "").strip().lower()
                    if k in seen_keys:
                        continue
                    out.append(it)
                    if len(out) >= PROMPT_MAX_SEMANTIC:
                        break
            return out

        stitched = _stitch_by_title(collapsed)
        sem_chunks_c = _truncate_list(stitched, PROMPT_MAX_SEMANTIC)
        mems_c       = _truncate_list(_safe_list(gated_ctx.get("memories") or []), PROMPT_MAX_MEMS)
        if not mems_c:
            fallback_mems = _safe_list(ctx.get("memories") or [])
            if not fallback_mems:
                fallback_mems = _safe_list(ctx.get("recent_conversations") or [])
            if fallback_mems:
                limit = PROMPT_MAX_MEMS if PROMPT_MAX_MEMS else len(fallback_mems)
                if limit <= 0:
                    limit = len(fallback_mems)
                mems_c = fallback_mems[:limit]
        dreams_c     = _truncate_list(_safe_list(gated_ctx.get("dreams") or []), PROMPT_MAX_DREAMS) if include_dreams else []
        wiki_snip    = (gated_ctx.get("wiki") or gated_ctx.get("wiki_snippet") or "") or ""

        # If we still don't have reflections, synthesize a tiny one on-demand
        if not refl_c and REFLECTIONS_ON_DEMAND:
            try:
                od = await self._reflect_on_demand(recent_c)
                if od:
                    refl_c = _truncate_list(_dedupe_keep_order(od), PROMPT_MAX_REFLECTIONS)
            except Exception:
                pass

        prompt_ctx: Dict[str, Any] = {
            **gated_ctx,
            "recent_conversations": recent_c,
            "memories": mems_c,
            "facts": facts_c,
            "summaries": summaries_c,
            "reflections": refl_c,
            "semantic_chunks": sem_chunks_c,
            "wiki": wiki_snip,
            "dreams": dreams_c,
            "raw_user_input": raw_user_input,

        }

        # (Optional) quick token budget check if you budget pre-assembly
        try:
            hard_budget = int(os.getenv("PROMPT_TOKEN_BUDGET", str(getattr(self, "token_budget", 8192))))
            if hasattr(self, "_estimate_tokens"):
                usage = self._estimate_tokens(prompt_ctx)
                logger.debug(f"[PROMPT] pre-assembly token est: {usage}/{hard_budget}")
        except Exception:
            pass

        logger.debug(f"[PROMPT] END — Duration: {time.time() - start_time:.2f}s")
        # IMPORTANT: return a dict (orchestrator will _assemble_prompt)
        return prompt_ctx

    # --- Context gathering (legacy path kept if called elsewhere) -------------
    async def _gather_context(
        self,
        user_input: str,
        include_dreams: bool,
        include_wiki: bool,
        include_semantic: bool,
        personality_config: Dict = None
    ) -> Dict[str, Any]:

        context = {
            "memories": [],
            "facts": [],
            "summaries": [],
            "dreams": [],
            "wiki": "",
            "semantic_chunks": [],
            "recent_conversations": [],
            "time_context": self._get_time_context()
        }

        # Configure based on personality
        config = personality_config or {}
        memory_count = config.get("num_memories", 10)
        logger.debug("[PROMPT][TRACE] build_prompt ENTER (will_fetch_summaries=YES)")
        sums = await self._get_summaries(PROMPT_MAX_SUMMARIES)
        logger.debug("[PROMPT][TRACE] _get_summaries returned n=%s", len(sums) if sums else 0)

        # Launch parallel tasks
        tasks = []
        tasks.append(self._get_recent_conversations(5))                               # 0
        tasks.append(self.memory_coordinator.get_memories(user_input, limit=memory_count))  # 1
        tasks.append(self._get_summaries(PROMPT_MAX_SUMMARIES))                       # 2
        tasks.append(self._get_dreams(PROMPT_MAX_DREAMS) if (include_dreams and DREAMS_ENABLED) else asyncio.sleep(0, result=[]))  # 3
        # Wiki gate happens later; don't block here
        tasks.append(asyncio.sleep(0, result="" if include_wiki else ""))             # 4
        tasks.append(self._get_semantic_chunks(user_input) if include_semantic else asyncio.sleep(0, result=[]))  # 5
        tasks.append(self.get_facts(user_input, limit=PROMPT_MAX_FACTS))              # 6

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack with defensive checks
        context["recent_conversations"] = results[0] if not isinstance(results[0], Exception) else []
        context["memories"]             = results[1] if not isinstance(results[1], Exception) else []
        context["summaries"]            = results[2] if not isinstance(results[2], Exception) else []
        context["dreams"]               = results[3] if not isinstance(results[3], Exception) else []
        context["wiki"]                 = results[4] if not isinstance(results[4], Exception) else ""
        context["semantic_chunks"]      = results[5] if not isinstance(results[5], Exception) else []
        context["facts"]                = results[6] if not isinstance(results[6], Exception) else []

        # Debug logging
        logger.debug(f"[PROMPT] Retrieved {len(context['recent_conversations'])} recent conversations")
        logger.debug(f"[PROMPT] Retrieved {len(context['summaries'])} summaries")
        logger.debug(f"[PROMPT] Retrieved {len(context['semantic_chunks'])} semantic chunks")
        logger.debug(f"[PROMPT] Retrieved {len(context['facts'])} facts")

        return context
        # ctx already includes "recent_conversations" and your fetched "reflections"
        if not ctx.get("reflections"):  # nothing came back from corpus/semantic
            try:
                od = await self._reflect_on_demand(ctx.get("recent_conversations") or [])
                if od:
                    ctx["reflections"] = od
            except Exception:
                pass


    # --- Hygiene + caps before budgeting --------------------------------------
    def _hygiene_and_caps(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        De-duplicate and soft-cap the noisier lists before token budgeting.
        Keeps ordering and avoids heavy truncation later.
        """
        ctx = dict(context)  # shallow copy

        # De-dupe by content-ish keys
        def _k_textlike(d, keys=("content", "text", "response", "filtered_content")):
            if isinstance(d, str):
                return d.strip().lower()
            if isinstance(d, dict):
                for k in keys:
                    if d.get(k):
                        return str(d[k]).strip().lower()
            return str(d).strip().lower()

        ctx["facts"]                = _truncate_list(_dedupe_keep_order(ctx.get("facts", []), key_fn=_k_textlike), PROMPT_MAX_FACTS)
        ctx["summaries"]            = _truncate_list(_dedupe_keep_order(ctx.get("summaries", []), key_fn=_k_textlike), PROMPT_MAX_SUMMARIES)
        # Cap recent conversations strictly by the RECENT cap (not reflections)
        ctx["recent_conversations"] = _truncate_list(ctx.get("recent_conversations", []), PROMPT_MAX_RECENT)
        ctx["semantic_chunks"]      = _truncate_list(ctx.get("semantic_chunks", []), PROMPT_MAX_SEMANTIC)
        ctx["memories"]             = _truncate_list(ctx.get("memories", []), PROMPT_MAX_MEMS)
        ctx["dreams"]               = _truncate_list(ctx.get("dreams", []), PROMPT_MAX_DREAMS)

        return ctx

    # --- Token budgeting with stable priorities --------------------------------
    def _manage_token_budget(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trim sections in increasing priority order until we fit within token budget.
        For lists, we remove items from the tail in conservative chunks (25%) to
        avoid over-trimming; for strings, we blank the whole section if needed.
        """
        model_name = self.model_manager.get_active_model_name()
        trimmed = dict(context)
        current_tokens = 0

        # Helper to count tokens for any item
        def _item_tokens(item: Any) -> int:
            text = self._extract_text(item)
            return self.get_token_count(text, model_name)

        # First pass: optimistic inclusion respecting soft caps already applied
        for name, _prio in PRIORITY_ORDER:
            val = trimmed.get(name)
            if not val:
                continue
            if isinstance(val, list):
                kept = []
                for item in val:
                    t = _item_tokens(item)
                    if current_tokens + t <= self.token_budget:
                        kept.append(item)
                        current_tokens += t
                    else:
                        break
                trimmed[name] = kept
            else:
                t = self.get_token_count(str(val), model_name)
                if current_tokens + t <= self.token_budget:
                    current_tokens += t
                else:
                    # We'll consider dropping this later in the second pass.
                    pass

        # If we’re still over (due to some large strings), trim by priority
        def _total_tokens(ctx: Dict[str, Any]) -> int:
            total = 0
            for name, _ in PRIORITY_ORDER:
                v = ctx.get(name)
                if not v:
                    continue
                if isinstance(v, list):
                    for it in v:
                        total += _item_tokens(it)
                else:
                    total += self.get_token_count(str(v), model_name)
            return total

        usage = _total_tokens(trimmed)
        logger.debug(f"[PROMPT] Token budget (pre-trim check): {usage}/{self.token_budget}")

        if usage > self.token_budget:
            # Second pass: trim from lowest priority upward
            for name, prio in sorted(PRIORITY_ORDER, key=lambda x: x[1]):  # low → high
                v = trimmed.get(name)
                if not v:
                    continue

                if isinstance(v, list) and v:
                    # Drop a conservative slice from the tail
                    drop_n = max(1, int(len(v) * 0.25))
                    trimmed[name] = v[:-drop_n]
                elif isinstance(v, str) and v:
                    trimmed[name] = ""

                usage = _total_tokens(trimmed)
                if usage <= self.token_budget:
                    break

        logger.debug(f"[PROMPT] Token budget: {usage}/{self.token_budget}")
        return trimmed

    # --- Prompt assembly (kept, with light guard rails) ------------------------
    def _assemble_prompt(
        self,
        user_input: str,
        context: Dict[str, Any],
        system_prompt: str,
        directives_file: str
    ) -> str:
        """Assemble the final prompt from all components."""
        parts: List[str] = []

        def _fmt_ts(ts) -> str:
            """Format mixed ts (datetime or ISO string) as 'YYYY-MM-DD HH:MM'."""
            from datetime import datetime, timezone
            try:
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        return ""
                if isinstance(ts, datetime):
                    if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    return ts.strftime('%Y-%m-%d %H:%M')
            except Exception:
                return ""
            return ""

        # Do not inline the system prompt here; orchestrator already sends it
        # as a separate system message. Keeping it out avoids duplication.

        # Time context
        if context.get("time_context"):
            parts.append(f"\n[TIME CONTEXT]\n{context['time_context']}\n")

        # Recent conversations
        if context.get("recent_conversations"):
            parts.append(
                "\n[RECENT CONVERSATION — FOR CONTEXT ONLY]\n"
                "Do NOT quote, paraphrase, or restate anything from this section in your reply. "
                "Use it only to avoid repeating yourself. Answer strictly the new USER INPUT.\n"
            )

            for conv in context["recent_conversations"]:
                q = (conv.get("query") or "").strip()
                r = (conv.get("response") or "").strip()
                # Middle-out very long Q/A pairs
                if ENABLE_MIDDLE_OUT and q:
                    q = self._middle_out(q, MEMORY_ITEM_MAX_TOKENS)
                if ENABLE_MIDDLE_OUT and r:
                    r = self._middle_out(r, MEMORY_ITEM_MAX_TOKENS)
                ts = _fmt_ts(conv.get("timestamp"))
                if q or r:
                    ts_line = f"[{ts}]\n" if ts else ""
                    parts.append(f"{ts_line}User: {q}\nAssistant: {r}\n")

        # Relevant memories
        if context.get("memories") or SHOW_EMPTY_SECTIONS:
            mems = context.get("memories") or []
            parts.append(f"\n[RELEVANT MEMORIES] (n={len(mems)})\n")
            for mem in mems:
                # Format and compress long memory responses
                if isinstance(mem, dict) and (mem.get("response")):
                    mem = dict(mem)
                    mem["response"] = self._middle_out(mem.get("response", ""), MEMORY_ITEM_MAX_TOKENS)
                parts.append(self._format_memory(mem))

        # Facts
        if context.get("facts"):
            parts.append("\n[FACTS]")
            for f in context["facts"]:
                md = (f.get("metadata", {}) if isinstance(f, dict) else getattr(f, "metadata", {}) or {})
                subj = md.get("subject"); rel = md.get("relation"); obj = md.get("object"); conf = md.get("confidence")
                if subj and rel and obj:
                    line = f"- {subj} {rel} {obj}"
                    if isinstance(conf, (int, float)):
                        line += f"  (conf={conf:.2f})"
                    parts.append(line + "\n")
                else:
                    txt = (f.get("content") if isinstance(f, dict) else getattr(f, "content", "")) or ""
                    parts.append(f"- {txt}\n")
        logger.debug(f"[PROMPT][Facts] Rendering {len(context.get('facts', []))} facts")

        # --- Summaries (single, guarded section; duplicates removed) ----------
        if context.get("summaries"):
            parts.append(
                "\n[CONVERSATION SUMMARIES — FOR CONTEXT ONLY]\n"
                "Do NOT quote, paraphrase, praise, or restate anything from this section in your reply. "
                "Use it only to recall past topics. Answer strictly the new USER INPUT.\n"
            )
            first = True
            for summary in context["summaries"]:
                if isinstance(summary, dict):
                    text = summary.get("content")
                    ts = _fmt_ts(summary.get("timestamp"))
                else:
                    text = str(summary)
                    ts = ""
                prefix = f"[{ts}] " if ts else ""
                # Ensure a blank line between summaries
                if not first:
                    parts.append("\n")
                parts.append(f"{prefix}{text}\n")
                first = False
        logger.debug("[PROMPT][TRACE] assemble summaries present? %s", bool(context.get("summaries")))

        if context.get("reflections") or SHOW_EMPTY_SECTIONS:
            # Clarify these are session-level reflections (typically generated at shutdown)
            refl_list = context.get("reflections") or []
            parts.append(f"\n[SESSION REFLECTIONS] (n={len(refl_list)})\n")
            for r in refl_list:
                if isinstance(r, dict):
                    txt = (r.get("content") or "").strip()
                    ts = _fmt_ts(r.get("timestamp"))
                else:
                    txt = str(r).strip()
                    ts = ""
                if txt:
                    prefix = f"[{ts}] " if ts else ""
                    parts.append(f"- {prefix}{txt}\n")

        if context.get("semantic_chunks") or SHOW_EMPTY_SECTIONS:
            model_name = self.model_manager.get_active_model_name() if hasattr(self, "model_manager") and hasattr(self.model_manager, "get_active_model_name") else "default"
            total_sem_tokens = 0
            try:
                sem_list = context.get("semantic_chunks") or []
                total_sem_tokens = sum(
                    self.get_token_count((ch.get("text") or ch.get("content") or ""), model_name)
                    for ch in sem_list
                )
            except Exception:
                total_sem_tokens = 0
            parts.append(
                f"\n[RELEVANT INFORMATION] (n={len(context.get('semantic_chunks') or [])}, ~{total_sem_tokens} tokens)\n"
            )
            # How many characters to show for each semantic item in the prompt (preview only)
            try:
                _sem_disp_chars = int(os.getenv("SEMANTIC_ITEM_DISPLAY_CHARS", "1500"))
            except Exception:
                _sem_disp_chars = 300

            for chunk in (context.get("semantic_chunks") or []):
                # Prefer gated/cleaned text if available
                text = (
                    chunk.get("filtered_content")
                    or chunk.get("text")
                    or chunk.get("content")
                    or ""
                )
                if ENABLE_MIDDLE_OUT and text:
                    text = self._middle_out(text, SEMANTIC_ITEM_MAX_TOKENS)
                src  = chunk.get("source") or chunk.get("namespace") or "unknown"
                ts   = chunk.get("timestamp") or chunk.get("ts") or ""
                title = (chunk.get("title") or "").strip()
                title_part = f" title=\"{title}\"" if title else ""
                preview = text[: max(0, _sem_disp_chars)] if text else ""
                if preview and len(text) > len(preview):
                    preview += "..."
                parts.append(f"- src={src}{title_part} ts={ts} :: {preview}\n")

        # Background knowledge (wiki)
        wiki_text = context.get("wiki") or ""
        if wiki_text or SHOW_EMPTY_SECTIONS:
            parts.append("\n[BACKGROUND KNOWLEDGE]\n")
            if wiki_text:
                parts.append(wiki_text.strip() + "\n")

        # Dreams (if any)
        if context.get("dreams"):
            parts.append("\n[DREAMS]\n")
            for dream in context["dreams"]:
                parts.append(f"{dream}\n")

        # Load and add directives (kept)
        directives = self._load_directives(directives_file)
        if directives:
            parts.append(f"\n[DIRECTIVES]\n{directives}\n")

        # User input last (middle-out compress very long inputs)
        display_input = context.get("raw_user_input")
        if display_input is None or display_input == "":
            display_input = user_input or ""
        final_user_input = display_input
        if ENABLE_MIDDLE_OUT and final_user_input:
            final_user_input = self._middle_out(final_user_input, USER_INPUT_MAX_TOKENS)
        parts.append(f"\n[USER INPUT]\n{final_user_input}\n")

        return "".join(parts)

    # === Helper Methods =======================================================

    def get_token_count(self, text: str, model_name: str) -> int:
        """Delegate to tokenizer_manager; keeps compatibility with your models."""
        return self.tokenizer_manager.count_tokens(text or "", model_name)

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats for token counting."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("content", "text", "response", "filtered_content"):
                if key in item and item[key]:
                    return str(item[key])
            return str(item)
        return str(item)

    def _format_memory(self, memory: Dict) -> str:
        """Format a memory for inclusion in prompt output."""
        if "query" in memory and "response" in memory:
            q = (memory.get('query') or '').strip()
            a = (memory.get('response') or '').strip()
            return f"Q: {q}\nA: {a}\n"
        if "content" in memory:
            return f"{memory['content']}\n"
        return f"{str(memory)}\n"

    def _get_time_context(self) -> str:
        """Get current time context in a stable format."""
        now = datetime.now()
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    async def _get_recent_conversations(self, count: int, topic: str = None) -> List[Dict]:
        """
        Get recent conversations, optionally filtered by topic.

        Ordering & fallback:
          1) On-topic (tag == f"topic:{topic}")
          2) General (tag == "topic:general")
          3) Any remaining recent entries
        """
        all_recent = self.memory_coordinator.corpus_manager.get_recent_memories(max(count * 2, count))

        if topic is None and hasattr(self.memory_coordinator, "current_topic"):
            topic = self.memory_coordinator.current_topic
        topic = (topic or "general").strip().lower()

        if topic == "general":
            return all_recent[:count]

        def has_tag(entry, tag: str) -> bool:
            tags = entry.get("tags") or []
            return tag in tags

        on_topic_tag = f"topic:{topic}"
        on_topic  = [e for e in all_recent if has_tag(e, on_topic_tag)]
        general   = [e for e in all_recent if has_tag(e, "topic:general")]
        remainder = [e for e in all_recent if e not in on_topic and e not in general]

        merged: List[Dict] = []
        seen = set()
        for bucket in (on_topic, general, remainder):
            for e in bucket:
                key = id(e)  # stable enough for this in-process list
                if key not in seen:
                    seen.add(key)
                    merged.append(e)
                if len(merged) >= count:
                    break
            if len(merged) >= count:
                break

        return merged[:count]

    def _synthesize_summaries_from_recent(self, recents: List[Dict], max_items: int = 5) -> List[Dict]:
        """
        Lightweight fallback: compress recent Q/A into brief bullets.
        This is intentionally simple to avoid heavy token use or extra deps.
        """
        if not recents:
            return []
        out: List[Dict] = []
        for r in recents[:max_items]:
            q = (r.get("query") or "").strip()
            a = (r.get("response") or "").strip()
            if not (q or a):
                continue
            text = f"- User asked: {q[:120]}{'...' if len(q) > 120 else ''}\n  Assistant: {a[:160]}{'...' if len(a) > 160 else ''}"
            out.append({
                "content": text,
                "timestamp": r.get("timestamp", datetime.now()),
                "type": "summary",
                "tags": ["summary:fallback", "source:recent"],
                "source": "fallback_recent"
            })
        return out

    async def _get_dreams(self, count: int) -> List[str]:
        """Get dreams/reflections (light wrapper)."""
        try:
            dreams = self.memory_coordinator.get_dreams(limit=count)
            return [d.get('content', '') for d in dreams if d.get('content')]
        except Exception:
            return []

    async def _get_wiki_content(self, query: str) -> str:
        """
        Kept for compatibility; primary wiki path now happens via unified accessor.
        Tries the topic-first key and raw query using the cached get_wiki_snippet().
        """
        wiki_key = (
            (self.topic_manager.current_topic.strip()
             if self.topic_manager and getattr(self.topic_manager, "current_topic", None) else "")
            or query
        )
        snippet = self._get_wiki_snippet_cached(wiki_key)
        if snippet:
            return snippet

        if query and query != wiki_key:
            snippet = self._get_wiki_snippet_cached(query)
            if snippet:
                return snippet

        # Legacy fallback (if an external WikiManager provides a different, richer API)
        try:
            if self.wiki_manager and hasattr(self.wiki_manager, "get_article_text"):
                ok, title, text = await self.wiki_manager.get_article_text(query)
                if ok and text:
                    return f"## {title}\n{text}"
        except Exception as e:
            logger.debug(f"[Wiki] Legacy fallback failed: {e}")

        return ""

    async def _get_semantic_chunks(self, query: str) -> List[Dict]:
        """
        Retrieve semantic chunks for the prompt following the configured source order.

        Source options: "faiss" (global FAISS index), "chroma" (wiki_knowledge collection).
        Uses a small hot-cache: FAISS results are upserted into Chroma with a soft cap.
        """
        # Load config defaults lazily to avoid hard import-time deps
        try:
            from config.app_config import SEMANTIC_SOURCE_ORDER as _ORDER, WIKI_HOT_CACHE_LIMIT as _HOT_CAP
        except Exception:
            _ORDER = ["faiss", "chroma"]
            _HOT_CAP = 50000

        async def _from_chroma() -> List[Dict]:
            chunks: List[Dict] = []
            try:
                if getattr(self, "memory_coordinator", None):
                    results = await self.memory_coordinator.search_by_type(
                        "wiki_knowledge", query=query, limit=10
                    )
                    for r in results or []:
                        if not isinstance(r, dict):
                            continue
                        meta = r.get("metadata", {}) or {}
                        chunks.append({
                            "text": r.get("content") or meta.get("content") or "",
                            "title": meta.get("title") or "",
                            "source": "wiki_knowledge",
                            "timestamp": meta.get("timestamp"),
                            "relevance_score": r.get("relevance_score", 0.0),
                        })
            except Exception:
                return []
            return chunks

        async def _from_faiss_and_hotcache() -> List[Dict]:
            # Import FAISS neighbor search with guard
            try:
                from knowledge.semantic_search import semantic_search_with_neighbors as _sem
            except Exception:
                # If import path differs in this environment, use the one we imported at top
                try:
                    _sem = semantic_search_with_neighbors
                except Exception:
                    return []

            # Run search
            try:
                try:
                    faiss_res = _sem(query, k=SEM_K) or []
                except TypeError:
                    faiss_res = _sem(query, top_k=SEM_K) or []
            except Exception:
                faiss_res = []

            if not faiss_res:
                return []

            # Hot-cache into Chroma (best-effort, soft cap)
            try:
                store = getattr(getattr(self, "memory_coordinator", None), "chroma_store", None)
                if store is not None:
                    stats = store.get_collection_stats() or {}
                    wiki_count = int((stats.get("wiki_knowledge") or {}).get("count", 0))
                    if wiki_count < _HOT_CAP:
                        for r in faiss_res:
                            text = (r.get("text") or r.get("content") or "").strip()
                            if not text:
                                continue
                            title = (r.get("title") or "").strip()
                            # Clean wiki-ish markup before caching
                            try:
                                from processing.gate_system import clean_wikiish as _cw
                                text = _cw(text)
                            except Exception:
                                pass
                            chunk = {"title": title or "Wikipedia", "id": title or "unknown", "text": text, "chunk_index": 0}
                            try:
                                store.add_wiki_chunk(chunk)
                            except Exception:
                                pass
                        # Attempt pruning if we exceeded cap
                        try:
                            store.prune_collection_by_timestamp("wiki_knowledge", keep=_HOT_CAP)
                        except Exception:
                            pass
            except Exception:
                pass

            # Normalize FAISS records into chunk shape
            chunks: List[Dict] = []
            for r in faiss_res:
                text = (r.get("text") or r.get("content") or "").strip()
                if not text:
                    continue
                # Return cleaned text to the caller
                try:
                    from processing.gate_system import clean_wikiish as _cw
                    text = _cw(text)
                except Exception:
                    pass
                chunks.append({
                    "text": text,
                    "title": r.get("title") or "",
                    "source": r.get("source") or r.get("namespace") or "wikipedia",
                    "timestamp": r.get("timestamp"),
                    "relevance_score": float(r.get("similarity", 0.0)),
                })
            return chunks

        # Iterate sources per configured order
        order = [s.strip().lower() for s in (_ORDER or []) if isinstance(s, str)] or ["faiss", "chroma"]
        for src in order:
            if src == "chroma":
                chunks = await _from_chroma()
                if chunks:
                    return chunks
            elif src == "faiss":
                chunks = await _from_faiss_and_hotcache()
                if chunks:
                    return chunks

        # Nothing from any source
        return []

    def _load_directives(self, directives_file: str) -> str:
        """Load directives file if present (kept)."""
        if os.path.exists(directives_file):
            with open(directives_file, 'r') as f:
                return f.read()
        return ""


# -----------------------------------------------------------------------------
# Compatibility exports
# -----------------------------------------------------------------------------


class PromptBuilder:
    """Backwards-compatible facade around UnifiedPromptBuilder."""

    def __init__(self, model_manager, **kwargs):
        self._builder = UnifiedPromptBuilder(model_manager, **kwargs)
        self.consolidator = getattr(self._builder, "consolidator", None)

    async def build_prompt(self, *args, **kwargs):
        return await self._builder.build_prompt(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._builder, item)


__all__ = ["UnifiedPromptBuilder", "PromptBuilder"]
# Semantic search knobs
SEM_K = int(os.getenv("SEM_K", "8"))
SEM_TIMEOUT_S = float(os.getenv("SEM_TIMEOUT_S", "5.0"))  # allow first-load index/model costs
