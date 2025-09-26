# /knowledge/WikiManager.py
import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import quote
import string
import requests
from sentence_transformers import SentenceTransformer, util
from config.app_config import (
    WIKI_FETCH_FULL_DEFAULT,
    WIKI_MAX_CHARS_DEFAULT,
    WIKI_MAX_SENTENCES_DEFAULT,
    WIKI_TIMEOUT_DEFAULT,
)

log = logging.getLogger(__name__)

WIKI_UA = {"User-Agent": "Daemon/1.0 (+https://example.com)"}
# Preferred/stable summary base (works reliably in your logs)
WIKI_API_SUMMARY_BASE = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
# Search endpoints live under /w/rest.php/v1
WIKI_V1_BASE = "https://{lang}.wikipedia.org/w/rest.php/v1"

_PHRASE_STRIPPERS = (
    "information about", "info about", "tell me", "tell me about",
    "what can you tell me about", "can you tell me", "could you tell me",
    "who is", "what is", "define", "explain", "overview of", "summary of",
)

STOPWORDS = {
    "the", "a", "an", "of", "on", "for", "in", "about", "regarding", "re", "re:",
    "what", "who", "is", "are", "was", "were", "to", "and", "or", "me", "my", "your",
    "you", "please", "pls", "give", "show", "tell", "explain", "info", "information"
}
# Generic “topic words” we usually want to drop to find the head term:
GENERIC_TOPIC_WORDS = {
    "lottery", "game", "games", "company", "country", "president", "movie",
    "film", "band", "album", "book", "sport", "team", "city", "state"
}


def _keywords_from_query(q: str) -> List[str]:
    """
    Extract reasonable probe keywords from a verbose query.
    Keeps alphanumerics, drops stopwords and generic topic words, title-cases tokens.
    Also returns last-2-words bigram if useful.
    """
    tokens = re.findall(r"[A-Za-z0-9']+", (q or "").lower())
    core = [t for t in tokens if t not in STOPWORDS and t not in GENERIC_TOPIC_WORDS]
    core = list(dict.fromkeys(core))  # preserve order, dedupe
    probes: List[str] = []

    # Single-token probes (title-cased)
    for t in core:
        if len(t) > 1:
            probes.append(t[:1].upper() + t[1:])

    # Last-two-words bigram (often the head noun phrase)
    if len(core) >= 2:
        bigram = f"{core[-2][:1].upper() + core[-2][1:]} {core[-1][:1].upper() + core[-1][1:]}"
        if bigram not in probes:
            probes.append(bigram)

    return probes[:5]  # keep it tight


def _clean_query(q: str) -> str:
    q = (q or "").strip().lower()
    for p in _PHRASE_STRIPPERS:
        if q.startswith(p):
            q = q[len(p):].strip()
            break
    # remove trailing punctuation and extra spaces
    q = q.strip(string.punctuation + " \t\r\n")
    # collapse repeated spaces
    q = " ".join(q.split())
    # title-case improves wikipedia autosuggest in many cases
    return q.title()


def _token_overlap_score(query: str, candidate: str) -> float:
    qset = set(w for w in query.lower().split() if len(w) > 2)
    cset = set(w for w in candidate.lower().split() if len(w) > 2)
    if not qset or not cset:
        return 0.0
    return len(qset & cset) / len(qset)


def _pick_candidate(query: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    # 1) exactish match
    q_nospace = query.replace(" ", "").lower()
    for c in candidates:
        if c.replace(" ", "").lower() == q_nospace:
            return c
    # 2) best token overlap (handles “Powerball” vs “Information about the Powerball lottery”)
    best = max(candidates, key=lambda c: _token_overlap_score(query, c))
    return best


def _is_main_namespace(title: str) -> bool:
    # Reject obvious non-article namespaces
    return not re.match(
        r"^(Talk|User|User talk|Wikipedia|Wikipedia talk|File|File talk|Template|Template talk|Help|Help talk|Category|Category talk|Portal|Draft|TimedText|Module|Gadget|MediaWiki)(:|$)",
        title,
    )


def _clip_sentences(text: str, max_sentences: int = 6) -> str:
    if not text:
        return ""
    bits = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(bits[:max_sentences])


def _normalize_candidate(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # Replace underscores with spaces, collapse whitespace
    s = re.sub(r"_+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Wikipedia is case-insensitive on first character; normalize that
    return s[:1].upper() + s[1:]


def _title_case(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] if w else w for w in s.split())


def _title_variants(q: str) -> List[str]:
    """
    Gentle variants to improve direct resolution before/alongside search:
      - normalized
      - Title Case (if multiword)
      - singular/plural 's' toggle
      - remove leading 'the '
      - underscored version
    """
    out: List[str] = []
    base = _normalize_candidate(q)
    if not base:
        return out
    out.append(base)

    if " " in base:
        tcase = _title_case(base)
        if tcase not in out:
            out.append(tcase)

    # Drop leading 'the '
    if base.lower().startswith("the "):
        no_the = base[4:].strip()
        if no_the and no_the not in out:
            out.append(no_the)
        if " " in no_the:
            no_the_tc = _title_case(no_the)
            if no_the_tc not in out:
                out.append(no_the_tc)

    # very light pluralization tweak
    if base.endswith("s") and len(base) > 1:
        sing = base[:-1]
        if sing not in out:
            out.append(sing)
        sing_tc = _title_case(sing) if " " in sing else sing[:1].upper() + sing[1:]
        if sing_tc not in out:
            out.append(sing_tc)
    else:
        plur = base + "s"
        if plur not in out:
            out.append(plur)
        plur_tc = _title_case(plur) if " " in plur else plur[:1].upper() + plur[1:]
        if plur_tc not in out:
            out.append(plur_tc)

    underscored = base.replace(" ", "_")
    if underscored not in out:
        out.append(underscored)

    return out[:6]  # keep latency tight


@dataclass
class WikiPage:
    title: str
    url: str
    summary: str
    thumbnail: Optional[str] = None
    score: float = 0.0
    is_disambiguation: bool = False


# ---------- Lazy, per-process embedder (singleton) ----------
_EMBEDDER = None


def _get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


class WikiManager:
    """
    Dynamic Wikipedia resolver that:
      - tries light title variants against REST summary with redirects
      - uses REST search to find candidate pages
      - filters to main namespace
      - resolves disambiguation by scoring candidates vs the query
      - returns the highest-scoring page summary
    """

    def __init__(
        self,
        lang: str = "en",
        max_sentences: int = None,
        timeout: float = None,
        embed_model_name: str = "all-MiniLM-L6-v2",
        min_accept_score: float = 0.35,
        session: Optional[requests.Session] = None,
        max_attempts: Optional[int] = None,
    ):
        self.lang = lang
        # Timeout: env override -> explicit arg -> config default
        if timeout is not None:
            self.timeout = float(timeout)
        else:
            try:
                self.timeout = float(os.getenv("WIKI_TIMEOUT", str(WIKI_TIMEOUT_DEFAULT)))
            except Exception:
                self.timeout = float(WIKI_TIMEOUT_DEFAULT)
        # Allow env override; 0 or negative disables clipping
        try:
            env_ms = int(os.getenv("WIKI_MAX_SENTENCES", str(WIKI_MAX_SENTENCES_DEFAULT)))
        except Exception:
            env_ms = int(WIKI_MAX_SENTENCES_DEFAULT or 0)
        if env_ms > 0:
            self.max_sentences = env_ms
        else:
            self.max_sentences = max_sentences if (isinstance(max_sentences, int) and max_sentences > 0) else None
        self.embed_model_name = embed_model_name
        self.embed = None  # defer actual load
        self.min_accept_score = min_accept_score
        self.disable_embed_scoring = bool(int(os.getenv("WIKI_DISABLE_EMBED_SCORING", "0")))
        self.max_attempts = int(max_attempts if max_attempts is not None else os.getenv("WIKI_MAX_ATTEMPTS", "5"))
        # Overall time budget to avoid cascading probes when topic is weak
        # Budget for resolve-and-fetch across probes
        self.overall_budget_s = float(os.getenv("WIKI_BUDGET_S", "1.2"))
        # Cap probe counts per phase
        self.max_titles = int(os.getenv("WIKI_MAX_TITLES", "4"))

        # If full fetch is requested, expand default timeout/budget unless overridden
        try:
            _full = int(os.getenv("WIKI_FETCH_FULL", "1" if WIKI_FETCH_FULL_DEFAULT else "0")) == 1
        except Exception:
            _full = bool(WIKI_FETCH_FULL_DEFAULT)
        if _full:
            try:
                _tt = float(os.getenv("WIKI_TIMEOUT", "2.5"))
            except Exception:
                _tt = 2.5
            try:
                _bud = float(os.getenv("WIKI_BUDGET_S", "4.0"))
            except Exception:
                _bud = 4.0
            # Only increase if current values are smaller and not explicitly passed
            if timeout is None:
                self.timeout = max(self.timeout, _tt)
            self.overall_budget_s = max(self.overall_budget_s, _bud)

        # Reuse HTTP connections
        self.http = session or requests.Session()
        self.http.headers.update(WIKI_UA)

    def _fetch_extract_action_api(self, title: str) -> Optional[WikiPage]:
        """
        Fetch article extract via MediaWiki Action API. When WIKI_FETCH_FULL=1, fetches
        the full page text (no exintro). Otherwise, fetches only the intro.
        Applies WIKI_MAX_CHARS clipping if set (>0).
        """
        if not title:
            return None
        headers = {**WIKI_UA, "Accept": "application/json"}
        api = f"https://{self.lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }
        try:
            # Respect WIKI_FETCH_FULL; default is intro only
            if int(os.getenv("WIKI_FETCH_FULL", "0")) == 0:
                params["exintro"] = 1
        except Exception:
            params["exintro"] = 1

        try:
            r = self.http.get(api, params=params, headers=headers, timeout=self.timeout)
            if r.status_code != 200:
                return None
            js = r.json() or {}
            pages = js.get("query", {}).get("pages", {}) or {}
            if not pages:
                return None
            _pid, pdata = next(iter(pages.items()))
            extract = (pdata or {}).get("extract", "") or ""
            if not extract.strip():
                return None
            extract = extract.strip()
            try:
                max_chars = int(os.getenv("WIKI_MAX_CHARS", str(WIKI_MAX_CHARS_DEFAULT)))
            except Exception:
                max_chars = WIKI_MAX_CHARS_DEFAULT
            if max_chars and max_chars > 0 and len(extract) > max_chars:
                import re as _re
                clip = extract[:max_chars]
                m = _re.search(r"[\.!?](?!.*[\.!?])", clip)
                extract = clip if not m else clip[: m.end()]
            norm_title = (pdata or {}).get("title") or title
            return WikiPage(
                title=norm_title,
                url=f"https://{self.lang}.wikipedia.org/wiki/{quote(norm_title.replace(' ', '_'))}",
                summary=extract,
                thumbnail=None,
                is_disambiguation=False,
            )
        except Exception as e:
            log.debug("[Wiki] Action API extract failed for %s: %s", title, e)
            return None

    # -------- REST helpers --------
    def _search_titles(self, query: str, limit: int = 7) -> List[str]:
        # https://en.wikipedia.org/w/rest.php/v1/search/title?q=...&limit=...
        if not query:
            return []
        url = f"{WIKI_V1_BASE.format(lang=self.lang)}/search/title"
        params = {"q": query, "limit": limit}
        r = self.http.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        pages = data.get("pages", []) or []
        titles = [p.get("title", "") for p in pages if p.get("title")]
        # Keep only main namespace-looking titles
        return [t for t in titles if _is_main_namespace(t)]

    def _search_pages(self, query: str, limit: int = 7) -> List[str]:
        # https://en.wikipedia.org/w/rest.php/v1/search/page?q=...&limit=...
        if not query:
            return []
        url = f"{WIKI_V1_BASE.format(lang=self.lang)}/search/page"
        params = {"q": query, "limit": limit}
        r = self.http.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        pages = data.get("pages", []) or []
        titles = [p.get("title", "") for p in pages if p.get("title")]
        return [t for t in titles if _is_main_namespace(t)]

    def _fetch_summary_v1(self, title: str) -> Optional[WikiPage]:
        """
        Fetch summary with redirects. Prefer stable /api/rest_v1; fall back to /w/rest.php/v1;
        and finally to the MediaWiki action API if needed.
        """
        if not title:
            return None
        # If full fetch requested, prefer Action API right away
        try:
            if int(os.getenv("WIKI_FETCH_FULL", "1" if WIKI_FETCH_FULL_DEFAULT else "0")) == 1:
                full = self._fetch_extract_action_api(title)
                if full:
                    return full
        except Exception:
            pass

        headers = {**WIKI_UA, "Accept": "application/json"}

        # 1) Preferred REST path
        try:
            url_api = WIKI_API_SUMMARY_BASE.format(lang=self.lang, title=quote(title, safe="()"))
            r = self.http.get(url_api, params={"redirect": "true"}, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                js = r.json() or {}
                typ = (js.get("type") or "").lower()
                if typ in ("standard", "article") or js.get("extract"):
                    return WikiPage(
                        title=js.get("title") or title,
                        url=js.get("content_urls", {}).get("desktop", {}).get(
                            "page", f"https://{self.lang}.wikipedia.org/wiki/{quote(title)}"
                        ),
                        # Keep the full extract (lead section) without aggressive truncation
                        summary=(js.get("extract") or "").strip(),
                        thumbnail=js.get("thumbnail", {}).get("url"),
                        is_disambiguation=(js.get("type") == "disambiguation"),
                    )
        except Exception as e:
            log.debug("[Wiki] Preferred summary path failed for %s: %s", title, e)

        # 2) Fallback REST path
        try:
            url_rest = f"{WIKI_V1_BASE.format(lang=self.lang)}/page/summary/{quote(title, safe='()')}"
            r = self.http.get(url_rest, params={"redirect": "true"}, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                js = r.json() or {}
                typ = (js.get("type") or "").lower()
                if typ in ("standard", "article") or js.get("extract"):
                    return WikiPage(
                        title=js.get("title") or title,
                        url=js.get("content_urls", {}).get("desktop", {}).get(
                            "page", f"https://{self.lang}.wikipedia.org/wiki/{quote(title)}"
                        ),
                        summary=(js.get("extract") or "").strip(),
                        thumbnail=js.get("thumbnail", {}).get("url"),
                        is_disambiguation=(js.get("type") == "disambiguation"),
                    )
        except Exception as e:
            log.debug("[Wiki] Fallback REST summary path failed for %s: %s", title, e)

        # 3) MediaWiki Action API (plain extract)
        try:
            api = f"https://{self.lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                # Respect full-intro flag via env; default is intro-only
                **({"exintro": 1} if int(os.getenv("WIKI_FETCH_FULL", "0")) == 0 else {}),
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
                "titles": title,
            }
            r = self.http.get(api, params=params, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                js = r.json() or {}
                pages = js.get("query", {}).get("pages", {}) or {}
                if pages:
                    _pageid, pdata = next(iter(pages.items()))
                    extract = (pdata or {}).get("extract", "") or ""
                    if extract.strip():
                        norm_title = (pdata or {}).get("title") or title
                        extract = extract.strip()
                        try:
                            max_chars = int(os.getenv("WIKI_MAX_CHARS", "5000"))
                        except Exception:
                            max_chars = 5000
                        if max_chars and max_chars > 0 and len(extract) > max_chars:
                            # Attempt sentence-boundary clip
                            import re as _re
                            clip = extract[:max_chars]
                            m = _re.search(r"[\.!?](?!.*[\.!?])", clip)
                            extract = clip if not m else clip[: m.end()]
                        return WikiPage(
                            title=norm_title,
                            url=f"https://{self.lang}.wikipedia.org/wiki/{quote(norm_title.replace(' ', '_'))}",
                            summary=extract,
                            thumbnail=None,
                            is_disambiguation=False,
                        )
        except Exception as e:
            log.debug("[Wiki] Action API path failed for %s: %s", title, e)

        return None

    # -------- scoring & selection --------
    def _score(self, query: str, page: WikiPage) -> float:
        """Score a candidate page given the query."""
        if self.disable_embed_scoring:
            # Fast heuristic (no embeddings)
            return _token_overlap_score(query, f"{page.title} {page.summary}")

        # Combine title + summary; cosine similarity in MiniLM space
        if self.embed is None:
            self.embed = _get_embedder(self.embed_model_name)

        try:
            q = self.embed.encode(query, normalize_embeddings=True, show_progress_bar=False)
            doc = self.embed.encode(
                f"{page.title}\n\n{page.summary}", normalize_embeddings=True, show_progress_bar=False
            )
            return float(util.cos_sim(q, doc))
        except Exception as e:
            log.debug("[Wiki] embedding score failed: %s", e)
            # fall back to token overlap if embeddings fail
            return _token_overlap_score(query, f"{page.title} {page.summary}")

    def resolve_and_fetch(self, query: str) -> Optional[WikiPage]:
        try:
            import time as _t
            _t0 = _t.time()
            def _within_budget() -> bool:
                return (_t.time() - _t0) < self.overall_budget_s

            # Normalize verbose user phrasing like "information about X", "tell me about Y", etc.
            original_query = query or ""
            query = _clean_query(original_query)

            candidates: List[WikiPage] = []

            # 1) Quick direct attempts using gentle title variants (fast path)
            for t in _title_variants(query)[:min(3, self.max_titles)]:  # keep probe count small
                try:
                    if not _within_budget():
                        break
                    # Prefer action API when full fetch requested
                    if int(os.getenv("WIKI_FETCH_FULL", "1" if WIKI_FETCH_FULL_DEFAULT else "0")) == 1:
                        p = self._fetch_extract_action_api(t)
                        if not p:
                            p = self._fetch_summary_v1(t)
                    else:
                        p = self._fetch_summary_v1(t)
                    if p and _is_main_namespace(p.title):
                        candidates.append(p)
                except Exception as e:
                    log.debug("[Wiki] direct summary fetch failed for %s: %s", t, e)

            # 2) REST search for titles, then fetch summaries
            # Phase A: try title search on the cleaned query
            titles = self._search_titles(query)

            # Phase B: if no titles, try title search on original query
            if not titles:
                fallback_titles = self._search_titles(original_query)
                titles = fallback_titles or []

            # Phase C: if still nothing, derive probe keywords and try each via title search
            if not titles:
                for probe in _keywords_from_query(original_query):
                    tprobe = self._search_titles(probe)
                    if tprobe:
                        titles = tprobe
                        break

            # Phase D: if still nothing, use broader page search (content search)
            if not titles:
                ptitles = self._search_pages(query) or self._search_pages(original_query)
                if not ptitles:
                    kws = _keywords_from_query(original_query)
                    if kws:
                        # final hail mary: try page search on the strongest single keyword/bigram
                        ptitles = self._search_pages(kws[-1])
                titles = ptitles or []

            if not titles and not candidates:
                log.debug("[Wiki] No candidate titles for query=%r (cleaned) or original=%r", query, original_query)

            for t in titles[: self.max_titles]:
                try:
                    if not _within_budget():
                        break
                    if int(os.getenv("WIKI_FETCH_FULL", "1" if WIKI_FETCH_FULL_DEFAULT else "0")) == 1:
                        p = self._fetch_extract_action_api(t) or self._fetch_summary_v1(t)
                    else:
                        p = self._fetch_summary_v1(t)
                    if p and _is_main_namespace(p.title):
                        candidates.append(p)
                except Exception as e:
                    log.debug("[Wiki] summary fetch failed for %s: %s", t, e)

            # If we got no page objects but we do have titles, fetch a seed
            if titles and not candidates and _within_budget():
                seed = _pick_candidate(query, titles) or titles[0]
                try:
                    p = self._fetch_summary_v1(seed)
                    if p and _is_main_namespace(p.title):
                        candidates.append(p)
                except Exception:
                    pass

            # De-duplicate by title
            seen = set()
            deduped: List[WikiPage] = []
            for p in candidates:
                key = p.title.strip().lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(p)
            candidates = deduped

            if not candidates:
                return None

            # 3) If first is a disambiguation, expand by searching again with (query + title)
            if candidates and candidates[0].is_disambiguation and _within_budget():
                more_titles = self._search_titles(f"{query} {candidates[0].title}")
                for t in more_titles[: min(5, self.max_titles)]:
                    try:
                        if not _within_budget():
                            break
                        p = self._fetch_summary_v1(t)
                        if p and _is_main_namespace(p.title):
                            key = p.title.strip().lower()
                            if key not in seen:
                                seen.add(key)
                                candidates.append(p)
                    except Exception:
                        pass

            # 4) Score and pick best
            for p in candidates:
                try:
                    p.score = self._score(query, p)
                except Exception as e:
                    log.debug("[Wiki] scoring failed for %s: %s", p.title, e)
                    p.score = 0.0

            best = max(candidates, key=lambda x: x.score)
            if best.score < self.min_accept_score:
                log.debug("[Wiki] Best score below floor (%.3f < %.3f)", best.score, self.min_accept_score)
                return None

            # 5) Optional: Clip to N sentences (disabled when max_sentences is None/<=0)
            if isinstance(self.max_sentences, int) and self.max_sentences > 0 and best.summary:
                best.summary = _clip_sentences(best.summary, self.max_sentences)

            return best

        except Exception as e:
            log.debug("[Wiki] resolve_and_fetch error: %s", e)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    wm = WikiManager()
    tests = [
        "tell me about the powerball lottery",
        "Information about Powerball",
        "who is joe biden",
        "overview of u-boat",
        "explain powerball (australia)",
    ]
    for q in tests:
        page = wm.resolve_and_fetch(q)
        print("Q:", q)
        if page:
            print("→", page.title, " | ", page.url)
            s = page.summary or ""
            print("sum:", s[:180].replace("\n", " ") + ("..." if len(s) > 180 else ""))
        else:
            print("→ None")
        print("-" * 60)
"""
# knowledge/WikiManager.py

Module Contract
- Purpose: Retrieve short Wikipedia snippets given a query/topic. Provides light cleaning and scoring to prefer high‑quality pages.
- Inputs:
  - Public APIs (e.g., get_wiki_snippet) used by prompt builder and gating
- Outputs:
  - Short, cleaned text snippets suitable for inclusion in prompt context
- Side effects:
  - Optional hot‑cache of wiki chunks into Chroma (by callers); network requests to Wikipedia endpoints.
"""
