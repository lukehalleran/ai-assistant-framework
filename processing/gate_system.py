#processing/gate_system.py
"""
Gating & lightweight retrieval helpers.

Goals:
- Be fast: batch embeddings, cache aggressively, avoid blocking the prompt build.
- Be stable: handle timeouts/network issues gracefully; fail-open when helpful.
- Be understandable: lots of comments explaining intent and trade-offs.

Public entry points most call sites use:
- MultiStageGateSystem.filter_memories(...)         -> batch gate memories
- MultiStageGateSystem.filter_wiki_content(...)     -> filter a wiki article
- MultiStageGateSystem.filter_semantic_chunks(...)  -> score semantic chunks
- GatedPromptBuilder.build_gated_prompt(...)        -> prompt build w/ gating

Internal helpers:
- CosineSimilarityGateSystem: cosine-based scorer (optional cross-encoder rerank)
- gated_wiki_fetch: fast, cancellable, cached wiki fetch + gating
"""

from __future__ import annotations

# --- stdlib ---
import os
import re
import time
import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable, List, Tuple, Dict, Any, Optional

# --- third-party ---
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx  # <— added for title search + summary fallback

# --- app-local ---
from config.app_config import (
    GATE_REL_THRESHOLD,
    WIKI_FETCH_FULL_DEFAULT,
    WIKI_MAX_CHARS_DEFAULT,
    WIKI_TIMEOUT_DEFAULT,
)
from utils.logging_utils import log_and_time, get_logger
from utils.query_checker import is_deictic_followup

logger = get_logger(__name__)
logger.debug("gate_system.py loaded — cosine similarity gating active")

"""
# processing/gate_system.py

Module Contract
- Purpose: Multi‑stage gating and reranking for candidate context (memories, semantic chunks, wiki). Provides a GatedPromptBuilder that prefilters context before prompt assembly.
- Key classes:
  - MultiStageGateSystem: cosine similarity + optional cross‑encoder reranking; filter_memories, filter_semantic_chunks, filter_wiki_content.
  - GatedPromptBuilder: adapter that invokes gate system then delegates to a prompt builder.
- Inputs:
  - Query text + lists of memories/semantic chunks/wiki content (dicts with text/metadata).
- Outputs:
  - Filtered lists with per‑item relevance scores; final prompt via underlying builder.
- Dependencies:
  - sentence-transformers, cross‑encoder model (optional), knowledge cleanup utilities, config gating thresholds.
- Side effects: None beyond logging.
"""

# =============================================================================
# Fast Wiki “assist” (cheap topical context that never blocks prompt build)
# =============================================================================

# Enables wiki assist; can be toggled if needed.
WIKI_ENABLED_DEFAULT = True
# Allow env override; otherwise honor config defaults
WIKI_FETCH_FULL = bool(int(os.getenv("WIKI_FETCH_FULL", "1" if WIKI_FETCH_FULL_DEFAULT else "0")))
try:
    WIKI_MAX_CHARS = int(os.getenv("WIKI_MAX_CHARS", str(WIKI_MAX_CHARS_DEFAULT)))
except Exception:
    WIKI_MAX_CHARS = WIKI_MAX_CHARS_DEFAULT

# Hard timeout to ensure we never stall prompt building.
try:
    WIKI_TIMEOUT_S = float(os.getenv("WIKI_TIMEOUT", str(WIKI_TIMEOUT_DEFAULT)))
except Exception:
    WIKI_TIMEOUT_S = WIKI_TIMEOUT_DEFAULT
if WIKI_FETCH_FULL and WIKI_TIMEOUT_S < 2.5:
    WIKI_TIMEOUT_S = 2.5

# Tiny in-process cache (query -> snippet); sized to avoid memory creep.
_WIKI_CACHE: dict[str, str] = {}

def _wiki_cache_key(q: str) -> str:
    base = (q or "").strip().lower()
    return f"{base}:full" if WIKI_FETCH_FULL else f"{base}:lead"

# Wikipedia HTTP settings
_WIKI_UA = {"User-Agent": "Daemon/1.0 (+https://example.com)"}
_WIKI_LANG = "en"

def get_cached_wiki(q: str) -> str:
    """Return cached snippet for normalized query if available."""
    return _WIKI_CACHE.get(_wiki_cache_key(q), "")


def _cache_wiki(q: str, text: str) -> None:
    """Store small snippets; keep cache bounded to ~64 entries."""
    key = _wiki_cache_key(q)
    if text:
        if len(_WIKI_CACHE) > 64:
            _WIKI_CACHE.pop(next(iter(_WIKI_CACHE)))
        _WIKI_CACHE[key] = text


def should_attempt_wiki(q: str) -> bool:
    """
    Cheap heuristic: only attempt wiki when the user likely asked a topical
    or definitional question (“tell me about X”, “what is Y”, or very short nouns).
    This avoids firing network fetches on follow-ups or multi-part instructions.
    """
    if not WIKI_ENABLED_DEFAULT:
        return False
    ql = (q or "").lower()
    # Avoid deictic follow-ups like “explain that again” or “another way”.
    if is_deictic_followup(ql):
        return False
    # Canonical topic intents.
    if any(p in ql for p in ("what is", "what are", "who is", "tell me about", "explain")):
        return True
    # Allow short, topic-like queries (<= 4 tokens of >=3 letters).
    tokens = [t for t in re.findall(r"[a-zA-Z]{3,}", ql)]
    return len(tokens) <= 4


async def gated_wiki_fetch(query: str, timeout: float = WIKI_TIMEOUT_S) -> tuple[bool, str]:
    """
    Non-blocking: returns (ok, snippet). If anything goes wrong or the content
    is judged irrelevant, returns (False, "") — callers can simply ignore.
    """
    # 0) cache hit (skip if full fetch requested to avoid stale short summaries)
    if not WIKI_FETCH_FULL:
        cached = get_cached_wiki(query)
        if cached:
            return True, cached

    # 1) pre-gate (no network)
    if not should_attempt_wiki(query):
        return False, ""

    # 2) attempt fetch with timeout
    try:
        ok, title, raw = await asyncio.wait_for(fetch_wiki_with_fallbacks(query), timeout=timeout)
        if not ok or not raw:
            return False, ""

        # 3) clean + gate content proper
        cleaned = clean_wikiish(raw)
        # Reuse cosine gate with default embedder and threshold
        res = await CosineSimilarityGateSystem().gate_content_async(query, cleaned, "wikipedia")
        if res.relevant and res.confidence >= GATE_REL_THRESHOLD:
            snippet = res.filtered_content or cleaned[:500]
            _cache_wiki(query, snippet)
            return True, snippet
        return False, ""
    except Exception:
        # timeout or transient error — just skip
        return False, ""


# --- local deictic helper (kept here to avoid circular imports) ---
_DEICTIC_HINTS = ("explain", "that", "it", "this", "again", "another way", "different way")




# --- semantic hygiene helpers (domain-agnostic) ---
_WIKI_NOISE_MARKERS = (
    "[[wp:", "[[ wP:", "[[category:", "[[user", "user:", "talk:", "redirect",
    "{{", "}}", "~~~~", "<page>", "</page>", "<revision", "&lt;", "&gt;", "rfa", "afd", "rfd"
)
_MARKUP_PAT = re.compile(r"(\{\{|\}\}|\[\[|\]\]|<[^>]+>|&[a-z]+;)", re.IGNORECASE)
_WORD_PAT = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b")


def _looks_wiki_noisy(text: str) -> bool:
    """
    Quick rejection for very markup-heavy text (talk pages, templates, etc.).
    """
    t = (text or "").lower()
    if any(m in t for m in _WIKI_NOISE_MARKERS):
        return True
    if not text:
        return False
    markup_hits = len(_MARKUP_PAT.findall(text))
    return (markup_hits / max(1, len(text))) > 0.01  # ~1% markup chars


# =============================================================================
# Topic-aware Wikipedia title resolution (non-breaking additions)
# =============================================================================

# Optional dependency injection point for your Topic Manager.
# Signature: resolver(user_query: str) -> str  (returns canonical topic or "")
_topic_resolver: Optional[Callable[[str], str]] = None


def set_topic_resolver(resolver: Callable[[str], str]) -> None:
    """
    Allow external code (e.g., TopicManager) to provide a canonical topic extractor.
    This avoids imports here and any circular dependencies.
    """
    global _topic_resolver
    _topic_resolver = resolver


_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

def _strip_articles(s: str) -> str:
    return _LEADING_ARTICLES.sub("", s or "").strip()


def wiki_title_candidates(q: str) -> list[str]:
    """
    Produce a few canonicalized title candidates from a conversational query.
    If a topic resolver is registered, prefer its canonical topic first.
    """
    base = (q or "").strip()

    # 1) Ask the topic manager (if provided)
    topic = ""
    try:
        if _topic_resolver:
            topic = (_topic_resolver(base) or "").strip()
    except Exception as e:
        logger.debug(f"[Wiki Fetch] topic_resolver failed: {e}")
        topic = ""

    seeds: list[str] = []
    if topic:
        seeds.append(topic)

    # Heuristic simplification: drop 'current' and temporal tails to improve hits
    def _simplify_title(s: str) -> str:
        import re as _re
        t = (s or "").strip()
        t = _re.sub(r"^(current|new|latest|recent|modern)\s+", "", t, flags=_re.IGNORECASE)
        t = _re.sub(r"\s+in\s+the\s+\d{1,2}(st|nd|rd|th)\s+century\b", "", t, flags=_re.IGNORECASE)
        t = _re.sub(r"\s+in\s+\d{3,4}s\b", "", t, flags=_re.IGNORECASE)
        t = _re.sub(r"\s+in\s+\d{4}\b", "", t, flags=_re.IGNORECASE)
        t = _re.sub(r"\s+", " ", t).strip()
        return t

    # 2) Conversational cleanups (kept for back-compat and robustness)
    seeds.append(base)
    simp = _simplify_title(base)
    if simp and simp not in seeds:
        seeds.append(simp)
    # Strip articles and add title-cased variant
    stripped = _strip_articles(base)
    if stripped and stripped not in seeds:
        seeds.append(stripped)
    titled = " ".join(w.capitalize() if len(w) > 2 else w for w in stripped.split())
    if titled and titled not in seeds:
        seeds.append(titled)

    # 3) Naive singularization for simple plurals
    if stripped.endswith("s") and len(stripped) > 3:
        sing = stripped[:-1]
        if sing not in seeds:
            seeds.append(sing)
        sing_t = " ".join(w.capitalize() if len(w) > 2 else w for w in sing.split())
        if sing_t not in seeds:
            seeds.append(sing_t)

    # 4) A few common nicknames / aliases
    alias = {"kitty": "Cat", "kitties": "Cat", "cats": "Cat", "dogs": "Dog"}
    al = alias.get(stripped.lower())
    if al and al not in seeds:
        seeds.append(al)

    # Deduplicate while preserving order
    seen = set()
    out: list[str] = []
    for s in seeds:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# Dependency-injected fetcher so we don’t hard-import a specific module.
_wiki_fetcher: Callable[[str], Awaitable[Tuple[bool, str]]] | None = None


def set_wiki_fetcher(fetcher: Callable[[str], Awaitable[Tuple[bool, str]]]) -> None:
    """Optionally allow other modules to provide a fetcher at runtime."""
    global _wiki_fetcher
    _wiki_fetcher = fetcher


# If some module set this in globals(), we’ll use it; otherwise we skip.
wikipedia_api = globals().get("wikipedia_api", None)


async def _wiki_search_title(query: str, timeout: float = 0.6) -> Optional[str]:
    """
    Use REST search to find a mainspace title for a freeform query.
    Returns best title string or None.
    """
    if not query:
        return None
    url = f"https://{_WIKI_LANG}.wikipedia.org/w/rest.php/v1/search/title"
    params = {"q": query, "limit": 7}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_WIKI_UA) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return None
            data = r.json() or {}
            pages = data.get("pages", [])
            for p in pages:
                title = (p or {}).get("title")
                if not title:
                    continue
                # crude main-namespace filter (no "X:Something")
                if ":" in title.split(" ", 1)[0]:
                    continue
                return title
    except Exception as e:
        logger.debug(f"[Wiki Search] failed for '{query}': {e}")
    return None


async def _wiki_summary_for_title(title: str, timeout: float = 0.6) -> Optional[str]:
    """
    Fetch the full introductory section (multi‑paragraph lead) for a known title.
    Tries the MediaWiki action API (exintro) first, then falls back to REST summary.
    """
    if not title:
        return None
    try:
        # 1) Action API: exintro (plain text)
        api = f"https://{_WIKI_LANG}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            # include only intro unless full fetch requested
            **({"exintro": 1} if not WIKI_FETCH_FULL else {}),
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }
        async with httpx.AsyncClient(timeout=timeout, headers=_WIKI_UA) as client:
            r = await client.get(api, params=params)
            if r.status_code == 200:
                js = r.json() or {}
                pages = (js.get("query", {}) or {}).get("pages", {}) or {}
                if pages:
                    _pid, pdata = next(iter(pages.items()))
                    extract = (pdata or {}).get("extract", "") or ""
                    if extract.strip():
                        extract = extract.strip()
                        if WIKI_MAX_CHARS and WIKI_MAX_CHARS > 0 and len(extract) > WIKI_MAX_CHARS:
                            # Clip at a sentence boundary if possible
                            clip = extract[:WIKI_MAX_CHARS]
                            m = re.search(r"[\.!?](?!.*[\.!?])", clip)
                            extract = clip if not m else clip[: m.end()]
                        logger.debug(f"[Wiki Summary] Action API {'FULL' if WIKI_FETCH_FULL else 'INTRO'} used for '{title}' (len={len(extract)})")
                        return extract
        # 2) Fallback: REST summary (shorter)
        url = f"https://{_WIKI_LANG}.wikipedia.org/api/rest_v1/page/summary/{httpx.utils.quote(title)}"
        async with httpx.AsyncClient(timeout=timeout, headers=_WIKI_UA) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            js = r.json() or {}
            extract = (js.get("extract") or "").strip()
            return extract or None
    except Exception as e:
        logger.debug(f"[Wiki Summary] failed for '{title}': {e}")
        return None


async def _wiki_summary_from_query(query: str, timeout: float = 0.8) -> Optional[Tuple[str, str]]:
    """
    Resolve a freeform query to a concrete title, then fetch its summary.
    Tries a few safe variants to avoid 404s like 'The_Space_Shuttle'.
    Returns (title, summary) or None.
    """
    if not query:
        return None

    for cand in wiki_title_candidates(query):
        title = await _wiki_search_title(cand, timeout=timeout)
        if not title:
            continue
        summary = await _wiki_summary_for_title(title, timeout=timeout)
        if summary:
            return title, summary
    return None


async def fetch_wiki_with_fallbacks(q: str) -> tuple[bool, str, str]:
    """
    Try topic-aware candidates. If a `wikipedia_api.fetch_article_text(title)` is
    available, use it first with a resolved title; otherwise fall back to the
    REST summary endpoint. Returns (ok, title, text).
    """
    # Extract the main topic from conversational queries (lightweight cleaner)
    q_lower = (q or "").lower().strip()
    cleaned = q_lower
    for lead in ("tell me about", "what is", "what are", "who is", "explain"):
        if lead in cleaned:
            cleaned = cleaned.split(lead, 1)[1].strip()
            break

    # Remove trailing conversational bits
    for phrase in ("how about", "please"):
        if phrase in cleaned:
            cleaned = cleaned.split(phrase, 1)[0].strip()
    cleaned = cleaned.strip(" ?.!")

    logger.debug(f"[Wiki Fetch] Cleaned query: '{cleaned}'")

    # Iterate candidates; resolve each to a concrete title via search
    for cand in wiki_title_candidates(cleaned or q):
        title = await _wiki_search_title(cand)
        if not title:
            continue

        # If we have a local wikipedia_api, try it first (may return full text)
        if (not WIKI_FETCH_FULL) and wikipedia_api is not None and hasattr(wikipedia_api, "fetch_article_text"):
            try:
                ok, text = await wikipedia_api.fetch_article_text(title)
                if ok and text and len(text) > 400:
                    logger.debug(f"[Wiki Fetch] Hit via wikipedia_api '{title}' (len={len(text)})")
                    return True, title, text
            except Exception as e:
                logger.debug(f"[Wiki Fetch] wikipedia_api failed for '{title}': {e}")

        # Fallback to REST summary (fast, avoids 404s)
        summary = await _wiki_summary_for_title(title)
        if summary:
            logger.debug(f"[Wiki Fetch] Hit via REST summary '{title}' (len={len(summary)})")
            return True, title, summary

    logger.debug("[Wiki Fetch] No suitable article found")
    return False, "", ""


def clean_wikiish(text: str) -> str:
    """
    Remove common wiki-style markup while preserving main prose.
    """
    # remove headings like == History ==
    text = re.sub(r"^=+[^=\n]+=+\s*$", "", text, flags=re.MULTILINE)
    # remove [citation needed] / [1] style refs
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
    # remove template braces crud {{...}}
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    return text.strip()


def _content_words(s: str) -> set[str]:
    return set(w.lower() for w in _WORD_PAT.findall(s or ""))


def _overlap_score(query: str, text: str) -> float:
    """
    Lexical overlap (normalized by the smaller set) as a soft rescue when cosine is borderline.
    """
    qw = _content_words(query)
    tw = _content_words(text)
    if not qw or not tw:
        return 0.0
    inter = len(qw & tw)
    return inter / max(1, min(len(qw), len(tw)))


def _source_weight(src: str) -> float:
    """
    Slightly prefer “docs/paper” over “unknown”. Numbers are gentle nudges, not gates.
    """
    s = (src or "").lower()
    if s in {"docs", "paper", "arxiv", "manual", "notebook"}:
        return 1.05
    if s in {"wikipedia", "wiki"}:
        return 1.00
    if s in {"unknown", ""}:
        return 0.85
    return 1.00


# Threshold & batching tunables (env-overridable)
DEFAULT_THRESHOLD = float(os.getenv("GATE_COSINE_THRESHOLD", "0.50"))
MAX_BATCH = int(os.getenv("GATE_EMBED_BATCH", "256"))


# =============================================================================
# Embedding cache (tiny, in-process)
# =============================================================================
class _EmbeddingCache:
    """
    Avoid re-encoding identical strings across queries (and within a batch).
    Keys are raw text; values are np.float32 normalized vectors.
    """
    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def set(self, key: str, val: np.ndarray) -> None:
        self._store[key] = val


_EMBED_CACHE = _EmbeddingCache()


# =============================================================================
# Core cosine gating
# =============================================================================
@dataclass
class GateResult:
    relevant: bool
    confidence: float
    reasoning: str
    filtered_content: str | None = None


class CosineSimilarityGateSystem:
    """
    Fast cosine similarity-based gating system.

    - Uses a single SentenceTransformer embedder everywhere (configurable).
    - Async-friendly: batch encodes; uses run_in_executor for sync encoders.
    - Optional cross-encoder re-ranker (CPU-friendly) for tie-breaking.
    """

    def __init__(
        self,
        model_manager=None,
        embedder: SentenceTransformer | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        # Stored for potential future hooks; not required by this class.
        self.model_manager = model_manager

        # Gate threshold (also used by content gate as cosine_threshold).
        self.threshold = float(threshold)
        self.cosine_threshold = self.threshold

        # Single embedder used everywhere. Default to MiniLM if none provided.
        self.embedder: SentenceTransformer = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        # Back-compat alias used by some call sites in this file.
        self.embed_model = self.embedder

        # Optional reranker; we keep it best-effort so we don’t fail hard if unavailable.
        try:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.use_reranking = True
            logger.debug("[Cosine Gate] Cross-encoder loaded for reranking")
        except Exception:
            self.cross_encoder = None
            self.use_reranking = False
            logger.debug("[Cosine Gate] No cross-encoder, using cosine only")

        # Lightweight in-class cache for gate_content_async
        self.cache: dict[str, GateResult] = {}

    # ----- small helpers -----

    def get_gating_model_name(self) -> str:
        return "cosine_similarity"

    def is_meta_query(self, query: str) -> bool:
        """
        Recognize “meta” questions about memory/recall to softly boost related content.
        """
        meta_keywords = [
            "memory",
            "memories",
            "what do you know",
            "how much do you remember",
            "what have you stored",
            "what do you recall",
        ]
        ql = (query or "").lower()
        return any(kw in ql for kw in meta_keywords)

    async def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts as L2-normalized vectors (np.float32), in batches.

        - If the embedder exposes `aencode`, use it directly.
        - Otherwise, run the sync `encode(...)` in a threadpool via run_in_executor.
        - Always request numpy + normalized embeddings for dot=cosine equivalence.
        """
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i : i + MAX_BATCH]
            maybe = getattr(self.embedder, "aencode", None)
            if callable(maybe):
                chunk_vecs = await self.embedder.aencode(
                    chunk, convert_to_numpy=True, normalize_embeddings=True
                )
            else:
                loop = asyncio.get_event_loop()
                # Use kwargs (positional booleans hit wrong params in ST).
                chunk_vecs = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.encode(
                        chunk,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                )
            vecs.append(chunk_vecs.astype(np.float32))
        return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)

    # ----- batch gating for memories -----

    async def batch_cosine_gate_memories(
        self, query: str, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score a batch of memory dicts against a query using cosine similarity.

        Behavior:
        - “Pinned” or episodic memories always pass (similarity recorded for logging).
        - Others pass if dot(q, m) >= threshold (with normalized vectors).
        - Embeddings are cached per text to avoid re-encoding.
        """
        t0 = time.time()
        logger.debug("[Batch Cosine Gate] START (n=%d, thr=%.3f)", len(memories), self.threshold)

        if not memories:
            logger.info("[Memory Filter] Gating complete: 0/0 (%.0fms)", (time.time() - t0) * 1000)
            return []

        # 1) Encode query once (offload to executor if needed)
        q_vec = await self._encode_texts([query or ""])  # (1, D)
        q = q_vec[0]

        # 2) Prepare memory vectors (cache + batch encode)
        m_texts: List[str] = []
        reusable_rows: List[np.ndarray | None] = []
        for m in memories:
            txt = (m.get("text") or m.get("content") or "").strip()
            m_texts.append(txt)
            reusable_rows.append(_EMBED_CACHE.get(txt))

        to_encode_idx = [i for i, v in enumerate(reusable_rows) if v is None]
        if to_encode_idx:
            to_encode = [m_texts[i] for i in to_encode_idx]
            new_vecs = await self._encode_texts(to_encode)
            for j, i in enumerate(to_encode_idx):
                row = new_vecs[j]
                reusable_rows[i] = row
                _EMBED_CACHE.set(m_texts[i], row)

        m_vecs = np.vstack(reusable_rows).astype(np.float32)  # (N, D)

        # 3) Cosine similarity via dot (vectors are normalized)
        sims = (m_vecs @ q.reshape(-1, 1)).reshape(-1)  # (N,)

        # 4) Bypass + thresholding (consistent episodic check: metadata.type == "episodic")
        passed: List[Dict[str, Any]] = []
        for m, s in zip(memories, sims):
            is_episodic = (m.get("metadata", {}).get("type") == "episodic")
            if m.get("pinned") or is_episodic:
                m["__score__"] = float(s)
                passed.append(m)
                continue
            if float(s) >= self.threshold:
                m["__score__"] = float(s)
                passed.append(m)

        dur_ms = (time.time() - t0) * 1000
        logger.info("[Memory Filter] Gating complete: %d/%d (%.0fms)", len(passed), len(memories), dur_ms)
        logger.debug("[Batch Cosine Gate] END")
        return passed

    # ----- single content gating -----

    @log_and_time("Cosine Gate Content")
    async def gate_content_async(self, query: str, content: str, content_type: str) -> GateResult:
        """
        Gate a single content blob (e.g., wiki text). Cache results on (query, content, type).
        """
        cache_key = f"{(query or '')[:50]}:{(content or '')[:100]}:{content_type}"
        if cache_key in self.cache:
            logger.debug(f"[Cosine Gate] Cache hit for {content_type}")
            return self.cache[cache_key]

        try:
            logger.debug(f"[Cosine Gate] Encoding query and {content_type} content")
            content = content or ""
            content_trim = content[:500]
            # Offload encoding to executor/aencode to avoid blocking event loop
            qv = await self._encode_texts([query or ""])  # (1,D)
            cv = await self._encode_texts([content_trim])  # (1,D)
            query_emb = qv[0]
            content_emb = cv[0]

            similarity = float(cosine_similarity([query_emb], [content_emb])[0][0])
            relevant = similarity >= self.cosine_threshold

            # Meta boost (soft nudge; capped at 1.0 to avoid weird logs)
            if self.is_meta_query(query) and any(w in content_trim.lower() for w in ("memory", "stored", "recall")):
                similarity = min(1.0, similarity + 0.10)
                relevant = True
                logger.debug(f"[Cosine Gate] Meta query boost applied to {content_type}")

            result = GateResult(
                relevant=relevant,
                confidence=similarity,
                reasoning=f"Cosine similarity: {similarity:.3f}",
                # Use full content when relevant; we only trim for embedding
                filtered_content=(content if relevant else None),
            )
            self.cache[cache_key] = result
            logger.info(
                f"[Cosine Gate] {content_type}: sim={similarity:.3f}, relevant={relevant}, "
                f"passed={'✓' if relevant else '✗'}"
            )
            return result

        except Exception as e:
            # Fail-open: we’d rather include something than drop context due to a transient error.
            logger.error(f"[Cosine Gate Error] {e}")
            return GateResult(relevant=True, confidence=0.5, reasoning="Gate failed, including by default")


# =============================================================================
# Multi-stage gate façade (keeps compatibility with call sites)
# =============================================================================
class MultiStageGateSystem:
    """
    Higher-level facade that:
    - Handles episodic bypass + blended scoring for memories.
    - Offers reranking when many items survive.
    - Wraps the cosine gate for wiki + semantic chunk flows.
    """

    def __init__(self, model_manager, cosine_threshold: float = GATE_REL_THRESHOLD):
        # Use the same embedder everywhere to maximize cache hits.
        self.gate_system = CosineSimilarityGateSystem(
            model_manager=model_manager, threshold=cosine_threshold
        )
        self.model_manager = model_manager
        self.embed_model = self.gate_system.embed_model

    def get_gating_model_name(self) -> str:
        return self.gate_system.get_gating_model_name()

    @log_and_time("Batch Cosine Gate Memories")
    async def batch_gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """
        Batch gate memories using cosine similarity with:
          - Episodic bypass (via metadata.type == "episodic")
          - Blended score: 70% cosine + 30% memory.truth_score
          - Optional re-ranker (if many survived)
        """
        if not memories:
            return []

        try:
            episodic: List[Dict] = []
            to_gate: List[Dict] = []

            # Separate episodic from others (episodic are “always include”)
            for mem in memories:
                if mem.get("metadata", {}).get("type") == "episodic":
                    logger.debug(f"[Batch Gate] BYPASS episodic: {mem.get('id', 'unknown')}")
                    episodic.append(mem)
                else:
                    to_gate.append(mem)

            logger.info(f"[Memory Filter] {len(episodic)} episodic memories bypassed gating")
            logger.info(f"[Memory Filter] Starting batch cosine gating for {len(to_gate)} memories")

            if not to_gate:
                return episodic  # nothing to gate

            # Encode once (offload to executor/aencode for responsiveness)
            qv = await self.gate_system._encode_texts([query or ""])  # (1,D)
            query_emb = qv[0]
            contents = [mem.get("content", "")[:500] for mem in to_gate]
            memory_embs = await self.gate_system._encode_texts(contents)

            similarities = cosine_similarity([query_emb], memory_embs)[0]

            # Deictic queries are often vague; don’t lower the bar for them.
            is_deictic = is_deictic_followup(query)

            gated: List[Dict] = []
            for mem, sim in zip(to_gate, similarities):
                truth = float(mem.get("metadata", {}).get("truth_score", 0.5))
                score = 0.7 * float(sim) + 0.3 * truth

                threshold = self.gate_system.cosine_threshold
                if is_deictic:
                    # Keep a floor for deictic follow-ups but allow env override; default to 0.25
                    try:
                        deictic_min = float(os.getenv("GATE_DEICTIC_MIN", "0.25"))
                    except Exception:
                        deictic_min = 0.25
                    threshold = max(threshold, deictic_min)

                if score >= threshold:
                    mem["relevance_score"] = float(score)
                    mem["filtered_content"] = mem.get("content", "")[:300]
                    gated.append(mem)
                else:
                    logger.debug(f"[Batch Gate] Memory filtered out: score={score:.3f}, threshold={threshold:.3f}")

            # Optional reranking when a lot survives (topical tie-breaker).
            if self.gate_system.use_reranking and len(gated) > 5:
                pairs = [[query, mem.get("content", "")[:300]] for mem in gated]
                rerank_scores = self.gate_system.cross_encoder.predict(pairs)
                for mem, score in zip(gated, rerank_scores):
                    mem["rerank_score"] = float(score)
                gated = sorted(gated, key=lambda x: x.get("rerank_score", 0), reverse=True)
            else:
                gated = sorted(gated, key=lambda x: x["relevance_score"], reverse=True)

            # Cap total items; always include episodic.
            final = episodic + gated[: max(0, 20 - len(episodic))]
            logger.info(f"[Memory Filter] Gating complete: {len(final)}/{len(memories)} memories passed")
            return final

        except Exception as e:
            logger.error(f"[Batch Gate Error] {e}")
            # Fallback: return a small slice with descending dummy scores (fail-open).
            for i, mem in enumerate(memories[:10]):
                mem["relevance_score"] = 0.5 - (i * 0.05)
                mem["filtered_content"] = mem.get("content", "")[:300]
            return memories[:10]

    @log_and_time("Filter Memories")
    async def filter_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """
        Main entry point for gating memories from the orchestrator.
        """
        if not memories:
            logger.debug("[Memory Filter] No memories to filter")
            return []
        logger.info(f"[Memory Filter] Starting batch cosine gating for {len(memories)} memories")
        filtered = await self.batch_gate_memories(query, memories)
        logger.info(f"[Memory Filter] Gating complete: {len(filtered)}/{len(memories)} memories passed")
        return filtered

    @log_and_time("Filter Wiki")
    async def filter_wiki_content(self, query: str, wiki_content: str) -> Tuple[bool, str]:
        """
        Filter wiki content with a conservative cleaner + cosine gate.
        Returns (pass, snippet).
        """
        if not wiki_content:
            logger.debug("[Wiki Filter] No wiki content passed; skipping")
            return False, ""
        try:
            # Clean instead of dropping; dropping was too aggressive.
            wiki_content = clean_wikiish(wiki_content)
            if not wiki_content or len(wiki_content) < 200:
                return False, ""
            result = await self.gate_system.gate_content_async(query, wiki_content, "wikipedia")

            if result.relevant and result.confidence > self.gate_system.cosine_threshold:
                # Prefer the full content passed through the gate
                return True, result.filtered_content or wiki_content

            # Secondary: minimal lexical overlap rescue (if cosine borderline)
            if result.confidence < 0.40:
                olap = _overlap_score(query, wiki_content[:1200])
                if olap >= 0.35:
                    logger.debug(f"[Wiki Filter Fallback] Overlap rescue (olap={olap:.2f})")
                    return True, wiki_content

            return False, ""

        except Exception as e:
            logger.debug(f"[Wiki Filter Error] {e}")
            return False, ""

    @log_and_time("Filter Semantic")
    async def filter_semantic_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Score semantic “chunks” with cosine + lexical overlap + source weighting.
        The intent is to keep topically right bits and avoid generic prose.
        """
        if not chunks:
            logger.debug("[Semantic Filter] No chunks to filter")
            return []

        try:
            logger.info(f"[Semantic Filter] Processing {len(chunks)} semantic chunks")
            query_emb = self.embed_model.encode(query or "", convert_to_numpy=True)

            scored_chunks: List[Dict] = []
            for i, chunk in enumerate(chunks[:30]):  # soft cap for latency
                text = chunk.get("text") or chunk.get("content") or ""
                title = chunk.get("title") or ""
                src = chunk.get("source") or chunk.get("namespace") or "unknown"

                # 1) Clean wiki-ish markup instead of dropping
                if _looks_wiki_noisy(title):
                    title = clean_wikiish(title)
                if _looks_wiki_noisy(text):
                    text = clean_wikiish(text)
                if not text:
                    continue

                # 2) Encode + cosine
                content_for_emb = f"{title} {text[:300]}".strip()
                chunk_emb = self.embed_model.encode(content_for_emb, convert_to_numpy=True)
                cosine_score = float(cosine_similarity([query_emb], [chunk_emb])[0][0])

                # 3) Lexical overlap
                olap = _overlap_score(query, text)

                # 4) Source weighting
                sw = _source_weight(src)

                # 5) Blended score (bounded)
                blended = max(0.0, min(1.0, 0.70 * cosine_score + 0.30 * olap))
                final_score = blended * sw

                # 6) Dynamic thresholding:
                #    - base tied to gate threshold
                #    - stricter when src unknown
                base_th = max(0.40, self.gate_system.cosine_threshold)
                if src.lower() in {"unknown", ""}:
                    base_th += 0.10  # be stricter on unknown
                # require at least a hint of lexical overlap if cosine is meh
                if olap < 0.15 and cosine_score < (base_th + 0.05):
                    logger.debug(
                        f"[Semantic Filter] Drop chunk {i} (src={src}): "
                        f"low overlap ({olap:.2f}) & low cosine ({cosine_score:.2f})"
                    )
                    continue

                chunk["relevance_score"] = float(final_score)
                chunk["filtered_content"] = text
                scored_chunks.append(chunk)

                if i < 5:
                    logger.debug(
                        f"[Semantic Filter] Chunk {i} ({title[:60] or 'untitled'}) "
                        f"src={src} cos={cosine_score:.3f} olap={olap:.2f} sw={sw:.2f} "
                        f"blend={blended:.3f} final={final_score:.3f}"
                    )

            # Prefilter with conservative floor, then (optionally) rerank
            prefiltered = [c for c in scored_chunks if c["relevance_score"] >= 0.40]
            logger.info(f"[Semantic Filter] Pre-filter: {len(prefiltered)}/{len(scored_chunks)} chunks >= 0.40")

            if self.gate_system.use_reranking and len(prefiltered) > 5:
                logger.info(f"[Semantic Filter] Running cross-encoder reranking on {len(prefiltered)} chunks")
                pairs = [
                    [query, f"{c.get('title','')} {(c.get('text') or c.get('content') or '')[:300]}"]
                    for c in prefiltered
                ]
                rerank_scores = self.gate_system.cross_encoder.predict(pairs)
                for chunk, score in zip(prefiltered, rerank_scores):
                    chunk["rerank_score"] = float(score)
                sorted_chunks = sorted(prefiltered, key=lambda x: x.get("rerank_score", 0), reverse=True)
            else:
                sorted_chunks = sorted(prefiltered, key=lambda x: x["relevance_score"], reverse=True)

            top_chunks = sorted_chunks[:5]
            if not top_chunks and scored_chunks:
                # fail-open: keep best-scoring item
                fallback = max(scored_chunks, key=lambda x: x.get("relevance_score", 0.0))
                logger.warning(f"[Semantic Filter] All filtered; fail-open with '{fallback.get('title','untitled')}'")
                top_chunks = [fallback]

            logger.info(f"[Semantic Filter] Final result: {len(top_chunks)} chunks selected")

            for i, chunk in enumerate(top_chunks):
                disp_score = chunk.get("rerank_score", chunk.get("relevance_score"))
                logger.debug(
                    f"[Semantic Filter] Selected chunk {i}: "
                    f"{chunk.get('title','untitled')} | src={chunk.get('source','unknown')} "
                    f"(score: {disp_score:.3f})"
                )

            return top_chunks

        except Exception as e:
            logger.error(f"[Semantic Filter Error] {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []


# =============================================================================
# Prompt builder wrapper that always uses the gate first
# =============================================================================
class GatedPromptBuilder:
    """
    Lightweight adapter around your existing prompt builder that first filters
    all context via MultiStageGateSystem, then calls into `prompt_builder`.
    """

    def __init__(self, prompt_builder, model_manager):
        self.prompt_builder = prompt_builder
        self.gate_system = MultiStageGateSystem(model_manager)

    @log_and_time("Cosine Gated Prompt Build")
    async def build_gated_prompt(
        self,
        user_input,
        memories,
        summaries,
        dreams,
        wiki_snippet: str = "",
        semantic_chunks: list[dict] | None = None,
        semantic_memory_results=None,
        time_context=None,
        recent_conversations=None,
        model_name=None,
        include_dreams=True,
        include_code_snapshot=False,
        include_changelog=False,
        system_prompt="",
        directives_file="structured_directives.txt",
    ):
        """
        Build the final prompt after passing all candidate context through gates.

        Design choices:
        - Errors in any gating stage are caught and fail-open in a bounded way.
        - Wiki is opportunistic (never blocks); semantic is conservative.
        - Memory gating is batched and cached for performance.
        """
        logger.debug("[Gated Prompt] Building prompt with cosine similarity gating")

        filtered_context: Dict[str, Any] = {}

        # --- Memories ---
        try:
            filtered_context["memories"] = await self.gate_system.filter_memories(user_input, memories)
        except Exception as e:
            logger.debug(f"[Gated Prompt - Memory Error] {e}")
            filtered_context["memories"] = memories[:5]  # fail-open small slice

        # --- Wikipedia (opportunistic) ---
        try:
            ok, wiki_text = await gated_wiki_fetch(user_input)
            filtered_context["wiki_snippet"] = wiki_text if ok else ""
        except Exception as e:
            logger.debug(f"[Gated Prompt - Wiki Error] {e}")
            filtered_context["wiki_snippet"] = ""

        # --- Semantic chunks ---
        try:
            filtered_context["semantic_chunks"] = await self.gate_system.filter_semantic_chunks(
                user_input, semantic_chunks or []
            )
        except Exception as e:
            logger.debug(f"[Gated Prompt - Semantic Error] {e}")
            filtered_context["semantic_chunks"] = []

        # --- Build final prompt (delegate to your existing builder) ---
        return self.prompt_builder.build_prompt(
            user_input=user_input,
            memories=filtered_context.get("memories", []),
            summaries=summaries,
            dreams=dreams if include_dreams else [],
            wiki_snippet=filtered_context.get("wiki_snippet", ""),
            semantic_chunks=filtered_context.get("semantic_chunks", []),
            semantic_memory_results=semantic_memory_results,
            time_context=time_context,
            model_name=(
                model_name
                or getattr(self.prompt_builder, "tokenizer_manager", None) and self.prompt_builder.tokenizer_manager.active_model_name
                or self.prompt_builder.model_manager.get_active_model_name()
            ),
            is_api=True,
            include_dreams=include_dreams,
            include_code_snapshot=include_code_snapshot,
            include_changelog=include_changelog,
            system_prompt=system_prompt,
            directives_file=directives_file,
        )
