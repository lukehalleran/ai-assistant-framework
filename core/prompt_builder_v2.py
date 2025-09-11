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

def _as_summary_dict(text: str, tags: list[str], source: str) -> dict:
    return {"content": text, "type": "summary", "tags": tags, "source": source}

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
        memory_coordinator,
        tokenizer_manager,  # expects an instance; we preserve your fallback
        wiki_manager,
        topic_manager,
        gate_system=None,
        max_tokens: int = MODEL_MAX_TOKENS,
        reserved_for_output: int = RESERVE_FOR_COMPLETION
    ):
        self.model_manager = model_manager
        self.memory_coordinator = memory_coordinator
        self.consolidator = MemoryConsolidator(model_manager=self.model_manager)
        # Preserve your original instance-or-fallback behavior for tokenizer_manager
        if tokenizer_manager is None or isinstance(tokenizer_manager, type):
            from models.tokenizer_manager import TokenizerManager
            self.tokenizer_manager = TokenizerManager(model_manager)
        else:
            self.tokenizer_manager = tokenizer_manager

        # Kept for compatibility with any external callers; not used in the unified path.
        self.wiki_manager = wiki_manager
        self.topic_manager = topic_manager
        self.gate_system = gate_system

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
                cm.add_summary({
                    "content": text,
                    "timestamp": datetime.now(),
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

                    # 👇 NEW: also include the most recent stored ones (minus duplicates), up to `count`
                    stored: list[dict] = []
                    try:
                        cm = getattr(self.memory_coordinator, "corpus_manager", None)
                        if cm and hasattr(cm, "get_summaries"):
                            stored = cm.get_summaries(count) or []
                    except Exception:
                        stored = []

                    # normalize to text list, filter placeholders
                    def _is_real(s):
                        t = (s or "").strip().lower()
                        return t and not (t.startswith("summary of ") or t.startswith("q:") or " q: " in t or " a: " in t)

                    stored_texts = []
                    for s in stored:
                        stored_texts.append((s.get("content") if isinstance(s, dict) else str(s)).strip())

                    stored_real = [t for t in stored_texts if _is_real(t)]

                    # Deduplicate against the just-forced text
                    dedup = []
                    seen = {llm.strip()}
                    for t in stored_real:
                        if t not in seen:
                            dedup.append(t)
                            seen.add(t)

                    # Build final list: forced first, then the freshest stored, capped
                    final_texts = [llm] + dedup[: max(0, count - 1)]
                    return [
                        _as_summary_dict(final_texts[0], ["llm_summary", "forced"], "llm_forced"),
                        *[_as_summary_dict(t, ["summary:stored"], "stored") for t in final_texts[1:]]
                    ]


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

        texts = []
        for s in summaries or []:
            texts.append((s.get("content") if isinstance(s, dict) else str(s)).strip())

        have_real = [t for t in texts if t and not _looks_placeholder(t)]
        logger.debug("[PROMPT][Summaries] have_real=%s", len(have_real))

        if have_real:
            return [{"content": t, "type": "summary", "tags": ["summary:stored"], "source": "stored"} for t in have_real[:count]]

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
                    if txt:
                        out.append({
                            "content": txt,
                            "type": "reflection",
                            "tags": ["source:semantic"],
                            "source": "semantic"
                        })
                return out[:count]
        except Exception:
            pass
        # If still empty, last-ditch: read directly from a chroma-like store, if present
        try:
            store = getattr(getattr(self, "memory_coordinator", None), "chroma_store", None)
            if not out and store and hasattr(store, "get_recent_texts"):
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

            return [{"content": txt, "type": "reflection", "tags": ["source:adhoc"], "source": "adhoc"}]
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
            "sem": self._bounded(
                asyncio.to_thread(
                    semantic_search_with_neighbors,
                    retrieval_query,
                    k=int(os.getenv("SEM_K", "8"))
                ),
                1.2, []
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
        sem_chunks_c = _truncate_list(_safe_list(gated_ctx.get("semantic_chunks") or []), PROMPT_MAX_SEMANTIC)
        mems_c       = _truncate_list(_safe_list(gated_ctx.get("memories") or []), PROMPT_MAX_MEMS)
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
                ts = _fmt_ts(conv.get("timestamp"))
                if q or r:
                    ts_line = f"[{ts}]\n" if ts else ""
                    parts.append(f"{ts_line}User: {q}\nAssistant: {r}\n")

        # Relevant memories
        if context.get("memories") or SHOW_EMPTY_SECTIONS:
            mems = context.get("memories") or []
            parts.append(f"\n[RELEVANT MEMORIES] (n={len(mems)})\n")
            for mem in mems:
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
            for chunk in (context.get("semantic_chunks") or []):
                text = chunk.get("text", chunk.get("content", ""))
                src  = chunk.get("source") or chunk.get("namespace") or "unknown"
                ts   = chunk.get("timestamp") or chunk.get("ts") or ""
                parts.append(f"- src={src} ts={ts} :: {text[:300]}...\n")

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

        # User input last
        parts.append(f"\n[USER INPUT]\n{user_input}\n")

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
        """Neighborhood semantic search with guarded import."""
        try:
            from knowledge.semantic_search import semantic_search_with_neighbors as _sem
        except Exception:
            # If import path differs in this environment, use the one we imported at top
            _sem = semantic_search_with_neighbors
        try:
            # Support both signatures (k or top_k)
            try:
                return _sem(query, k=10) or []
            except TypeError:
                return _sem(query, top_k=10) or []
        except Exception:
            return []

    def _load_directives(self, directives_file: str) -> str:
        """Load directives file if present (kept)."""
        if os.path.exists(directives_file):
            with open(directives_file, 'r') as f:
                return f.read()
        return ""
