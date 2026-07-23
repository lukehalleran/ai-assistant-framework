"""
literature_oracle.py — Claim-level documented-ness oracle (OFFLINE measurement instrument).

Contract
--------
Inputs:  a synthesis candidate (concept_a, concept_b, claim_text) plus run metadata.
Outputs: an OracleRecord with an evidence-backed verdict:
         DOCUMENTED_EXACT | DOCUMENTED_ADJACENT | NOT_FOUND | UNCERTAIN | PARSE_FAILURE.
Side effects: network calls (arXiv API, Tavily via WebSearchManager) and JSONL appends
         via the module-level helpers. NO ChromaDB production writes (callers must inject
         an isolated WebSearchCache store), NO synthesis_filter imports, NO config.yaml /
         schema.py / app_config.py surface — constants below are module-local with
         env-var overrides only.

Key invariants (each maps to a documented failure mode of the old stage-3 oracle):
  * Evidence rule (code-enforced, not prompt-enforced): a DOCUMENTED_* verdict is valid
    only with >=1 citation whose url matches retrieved evidence and whose
    supporting_passage (<=40 words) appears verbatim in that evidence item's text.
    Otherwise the verdict is downgraded to UNCERTAIN with downgraded_from_prior=True.
    The adjudicator's own prior (llm_prior_verdict) is logged but never drives the verdict.
  * Parse failures are LOUD: exactly one retry, then verdict=PARSE_FAILURE — never a
    silent default (the dead-coherence-judge lesson).
  * Paraphrase-evasion countermeasure: query generation must include a query using
    CANONICAL names for the purported mapping, not the candidate's paraphrase.
  * arXiv politeness: >= ORACLE_ARXIV_DELAY_S between requests, serialized by a lock.

This module is a pure library: no CLI. Drivers live in scripts/oracle_calibrate.py and
scripts/oracle_measure_generator.py. Methodology + results: docs/LITERATURE_ORACLE.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from utils.logging_utils import get_logger

log = get_logger("literature_oracle")

# ---------------------------------------------------------------------------
# Module-local constants (env-var overrides only — deliberately NOT in the
# config.yaml/schema.py/app_config.py system: this instrument has zero
# production surface and must stay that way).
# ---------------------------------------------------------------------------
ORACLE_MODEL = os.getenv("ORACLE_MODEL", "")  # empty -> SYNTHESIS_COHERENCE_MODEL at runtime
ORACLE_QUERY_TEMP = float(os.getenv("ORACLE_QUERY_TEMP", "0.0"))  # 0 => stable queries => Tavily cache hits
ORACLE_ADJ_TEMP = float(os.getenv("ORACLE_ADJ_TEMP", "0.1"))
ORACLE_MIN_QUERIES = 2
ORACLE_MAX_QUERIES = 4
ORACLE_MAX_QUERY_CHARS = 400  # Tavily truncation limit
ORACLE_MAX_EVIDENCE = int(os.getenv("ORACLE_MAX_EVIDENCE", "8"))
ORACLE_MAX_ARXIV_SLOTS = int(os.getenv("ORACLE_MAX_ARXIV_SLOTS", "4"))  # reserve >=4 for web
ORACLE_PASSAGE_MAX_WORDS = 40
ORACLE_ARXIV_DELAY_S = float(os.getenv("ORACLE_ARXIV_DELAY_S", "3.1"))  # arXiv ToS: ~1 req / 3s
ORACLE_ARXIV_RETRIES = 2
ORACLE_ARXIV_MAX_RESULTS = 5
ORACLE_TAVILY_MAX_RESULTS = 4
ORACLE_TAVILY_QUERIES = 3  # how many of the generated queries also go to Tavily (credits)
ORACLE_VERDICT_FIRST = os.getenv("ORACLE_VERDICT_FIRST", "1") == "1"  # calibration A/B knob
ORACLE_ESCALATE = os.getenv("ORACLE_ESCALATE", "1") == "1"
ORACLE_QUERY_MAX_TOKENS = 400
ORACLE_ADJ_MAX_TOKENS = 700
# Rendering caps: snippets are short anyway; extracts are full pages fetched by the
# STANDARD-depth escalation — truncating them to snippet size would mean paying 2 credits
# for text the adjudicator never sees (and the evidence rule quotes against FULL text,
# so the model must be shown the region it is expected to cite from).
ORACLE_EVIDENCE_SNIPPET_CHARS = int(os.getenv("ORACLE_EVIDENCE_SNIPPET_CHARS", "800"))
ORACLE_EVIDENCE_EXTRACT_CHARS = int(os.getenv("ORACLE_EVIDENCE_EXTRACT_CHARS", "6000"))
# Worst case per candidate: 3 Tavily QUICK (3 credits) + 1 STANDARD escalation (2 credits).
ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE = 5.0

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_VERDICT_ALIASES = {
    "DOCUMENTED_EXACT": "DOCUMENTED_EXACT",
    "DOCUMENTED_ADJACENT": "DOCUMENTED_ADJACENT",
    "NOT_FOUND": "NOT_FOUND",
    "UNCERTAIN": "UNCERTAIN",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class OracleVerdict(str, Enum):
    DOCUMENTED_EXACT = "DOCUMENTED_EXACT"      # this specific mapping/identity is published
    DOCUMENTED_ADJACENT = "DOCUMENTED_ADJACENT"  # concepts connected in literature, not this claim
    NOT_FOUND = "NOT_FOUND"                    # search procedure surfaced no documentation
    UNCERTAIN = "UNCERTAIN"                    # evidence insufficient (incl. code-enforced downgrades)
    PARSE_FAILURE = "PARSE_FAILURE"            # adjudicator output unparseable after 1 retry — LOUD


class EvidenceItem(BaseModel):
    source: Literal["arxiv", "tavily_quick", "tavily_extract"]
    url: str
    title: str = ""
    text: str = ""
    query: str = ""
    score: float = 0.0


class Citation(BaseModel):
    url: str
    supporting_passage: str
    rationale: str = ""

    @field_validator("supporting_passage")
    @classmethod
    def _passage_word_cap(cls, v: str) -> str:
        if len(v.split()) > ORACLE_PASSAGE_MAX_WORDS:
            raise ValueError(f"supporting_passage exceeds {ORACLE_PASSAGE_MAX_WORDS} words")
        return v


class QueryPlan(BaseModel):
    queries: List[str] = Field(min_length=1)
    canonical_idx: int = 0      # index of the canonical-names query (paraphrase-evasion)
    relationship_idx: int = 0   # index of the relationship-type query ("X duality Y")
    fallback_used: bool = False

    @field_validator("queries")
    @classmethod
    def _clamp(cls, v: List[str]) -> List[str]:
        return [q[:ORACLE_MAX_QUERY_CHARS] for q in v[:ORACLE_MAX_QUERIES]]


class OracleRecord(BaseModel):
    run_id: str
    candidate_id: str
    timestamp: str = ""
    concept_a: str
    concept_b: str
    claim_text: str
    source: str = ""                 # corpus provenance tag
    pre_seen: bool = False           # calibration-contaminated stratum marker
    queries: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    verdict: OracleVerdict = OracleVerdict.UNCERTAIN
    verdict_name: str = ""           # established name/theorem, if the adjudicator gave one
    evidence_citations: List[Citation] = Field(default_factory=list)
    llm_prior_verdict: str = ""      # adjudicator's own-knowledge opinion — logged, never used
    llm_prior_citation_claim: str = ""
    downgraded_from_prior: bool = False
    escalated: bool = False
    parse_failures: int = 0
    aux_doc_cooccurrence: Optional[Dict[str, Any]] = None
    aux_concept_cosine: Optional[float] = None
    timing: Dict[str, float] = Field(default_factory=dict)
    credits_used: float = 0.0
    calibration_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Text / JSON utilities (pure)
# ---------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    """Whitespace-collapse + casefold + unicode dash/quote normalization,
    so 'verbatim substring' survives snippet reflow and typographic quotes."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s


def _extract_json(raw: str) -> Optional[dict]:
    """Strip markdown fences, then parse the first balanced-brace JSON object."""
    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _parse_verdict_string(v: Any) -> Optional[str]:
    """Map an adjudicator verdict string onto the enum; None if unrecognized."""
    if not isinstance(v, str):
        return None
    return _VERDICT_ALIASES.get(v.strip().upper().replace(" ", "_"))


# ---------------------------------------------------------------------------
# JSONL persistence (the resumability primitive)
# ---------------------------------------------------------------------------
def append_record(path: str, record: OracleRecord) -> None:
    parent = os.path.dirname(path)
    if parent:  # bare filename → cwd; makedirs('') raises FileNotFoundError
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_ids(path: str) -> Set[str]:
    """Candidate ids already scored in a results.jsonl; tolerates a truncated last line."""
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["candidate_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # truncated tail from an interrupted run
    return done


def check_budget(rate_limiter: Any) -> float:
    """Remaining Tavily credits; drivers stop BEFORE a candidate if below worst case."""
    try:
        return float(rate_limiter.get_remaining_credits())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_QUERY_SYSTEM = (
    "You write literature search queries. Output STRICT JSON only — no prose, no fences."
)

_QUERY_PROMPT = """A generator proposed this cross-domain structural claim:
  A: {a}
  B: {b}
  CLAIM: {claim}

Write {min_q}-{max_q} search queries to find whether THIS SPECIFIC mapping/identity is already
documented in the scientific literature. Requirements:
1. At least one query must use the CANONICAL/established names for the purported mapping
   (the names a textbook or review would use), NOT the claim's own paraphrase. If the mapping
   has a likely established name (a named theorem, duality, correspondence), name it.
2. At least one query must name the structural relationship type explicitly, e.g.
   "{a} {b} duality", "{a} {b} isomorphism", "{a} {b} formal equivalence theorem".
3. Each query <= {max_chars} characters. Plain keyword queries, no boolean operators.

Return JSON exactly:
{{"queries": ["...", "..."], "canonical_idx": <int>, "relationship_idx": <int>}}"""

_ADJ_SYSTEM = (
    "You adjudicate whether a SPECIFIC claimed cross-domain identity is already documented, "
    "judging STRICTLY from the retrieved evidence provided. You are not asked what you believe; "
    "you are asked what the evidence shows. Output STRICT JSON only — no prose, no fences."
)

_ADJ_SCHEMA_VERDICT_FIRST = """{{
  "verdict": "DOCUMENTED_EXACT | DOCUMENTED_ADJACENT | NOT_FOUND | UNCERTAIN",
  "name": "<established name/theorem or ''>",
  "citations": [{{"url": "<url of an evidence item>", "supporting_passage": "<verbatim passage copied from that item, <= {max_words} words>", "rationale": "<one sentence>"}}],
  "prior_verdict": "<your own-knowledge opinion, same vocabulary — logged only, never used>",
  "prior_citation_claim": "<reference you believe exists, or ''>",
  "reasoning": "<2-4 sentences>"
}}"""

_ADJ_SCHEMA_REASONING_FIRST = """{{
  "reasoning": "<2-4 sentences>",
  "citations": [{{"url": "<url of an evidence item>", "supporting_passage": "<verbatim passage copied from that item, <= {max_words} words>", "rationale": "<one sentence>"}}],
  "name": "<established name/theorem or ''>",
  "verdict": "DOCUMENTED_EXACT | DOCUMENTED_ADJACENT | NOT_FOUND | UNCERTAIN",
  "prior_verdict": "<your own-knowledge opinion, same vocabulary — logged only, never used>",
  "prior_citation_claim": "<reference you believe exists, or ''>"
}}"""

_ADJ_PROMPT = """Candidate cross-domain claim:
  A: {a}
  B: {b}
  CLAIM: {claim}

Retrieved evidence (the ONLY basis for your verdict):
{evidence}

Verdict definitions:
- DOCUMENTED_EXACT: the evidence shows THIS SPECIFIC identity/correspondence is an established,
  published result (named theorem, textbook identity, standard correspondence).
- DOCUMENTED_ADJACENT: the evidence connects these concepts (or a standard framing that contains
  them), but does NOT establish this specific structural claim.
- NOT_FOUND: the evidence does not document this claim or a containing framing.
- UNCERTAIN: the evidence is insufficient to judge either way.

A DOCUMENTED verdict REQUIRES citations: each citation's supporting_passage must be copied
VERBATIM from the text of the evidence item at that url, and be at most {max_words} words.
If you cannot cite such a passage, do not answer DOCUMENTED.

Return JSON exactly:
{schema}"""


def _claim_terms(claim: str) -> List[str]:
    """Distinctive lowercase terms from the claim (len > 3), for window selection."""
    stop = {"this", "that", "with", "from", "have", "both", "which", "their",
            "same", "when", "where", "into", "under", "over", "between"}
    return [w for w in re.findall(r"[a-z]{4,}", claim.lower()) if w not in stop]


def _select_window(body: str, terms: List[str], cap: int) -> str:
    """Pick the cap-sized window of body densest in claim terms (head if no hits).

    The evidence rule validates passages against the FULL text, but the model can only
    quote what it is shown — so long extracts must surface the claim-relevant region."""
    if len(body) <= cap:
        return body
    if not terms:
        return body[:cap]
    low = body.lower()
    best_start, best_hits = 0, -1
    stride = max(cap // 2, 1)
    for start in range(0, len(body) - cap + stride, stride):
        window = low[start:start + cap]
        hits = sum(window.count(t) for t in terms)
        if hits > best_hits:
            best_start, best_hits = start, hits
    return body[best_start:best_start + cap]


def _render_evidence(evidence: List[EvidenceItem], claim: str = "") -> str:
    terms = _claim_terms(claim)
    lines = []
    for i, e in enumerate(evidence, 1):
        cap = (ORACLE_EVIDENCE_EXTRACT_CHARS if e.source == "tavily_extract"
               else ORACLE_EVIDENCE_SNIPPET_CHARS)
        body = _select_window((e.text or "").strip().replace("\n", " "), terms, cap)
        lines.append(f"[E{i}] ({e.source}) {e.title}\n     url: {e.url}\n     {body}")
    return "\n".join(lines) if lines else "(no evidence retrieved)"


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------
class LiteratureOracle:
    """Per-candidate pipeline: [Q]uery-gen -> [R]etrieval -> [A]djudication -> [V]alidation.

    Pure library. Drivers inject a WebSearchManager whose cache store is ISOLATED from
    production ChromaDB (see scripts/oracle_calibrate.py)."""

    def __init__(
        self,
        model_manager: Any,
        web_search_manager: Optional[Any] = None,
        model_name: str = "",
        verdict_first: bool = ORACLE_VERDICT_FIRST,
        escalate: bool = ORACLE_ESCALATE,
        enable_aux: bool = False,
    ):
        self.mm = model_manager
        self.wsm = web_search_manager
        self._model_name = model_name or ORACLE_MODEL
        self.verdict_first = verdict_first
        self.escalate = escalate
        self.enable_aux = enable_aux
        self._arxiv_lock = asyncio.Lock()
        self._arxiv_last_call = 0.0

    @property
    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        from config.app_config import SYNTHESIS_COHERENCE_MODEL
        return SYNTHESIS_COHERENCE_MODEL

    # ------------------------------------------------------------------ [Q]
    def _fallback_queries(self, a: str, b: str) -> QueryPlan:
        return QueryPlan(
            queries=[
                f'"{a}" "{b}" correspondence equivalence known result',
                f"{a} {b} duality isomorphism theorem",
            ],
            canonical_idx=0,
            relationship_idx=1,
            fallback_used=True,
        )

    async def generate_queries(self, a: str, b: str, claim: str) -> QueryPlan:
        prompt = _QUERY_PROMPT.format(
            a=a, b=b, claim=claim,
            min_q=ORACLE_MIN_QUERIES, max_q=ORACLE_MAX_QUERIES,
            max_chars=ORACLE_MAX_QUERY_CHARS,
        )
        try:
            raw = await self.mm.generate_once(
                prompt,
                model_name=self.model_name,
                system_prompt=_QUERY_SYSTEM,
                max_tokens=ORACLE_QUERY_MAX_TOKENS,
                temperature=ORACLE_QUERY_TEMP,
                disable_reasoning=True,
            )
        except Exception as ex:
            log.warning(f"[oracle] query-gen LLM error, using fallback templates: {ex}")
            return self._fallback_queries(a, b)

        parsed = _extract_json(raw)
        queries = [q for q in (parsed or {}).get("queries", []) if isinstance(q, str) and q.strip()]
        if len(queries) < ORACLE_MIN_QUERIES:
            # Query-gen fallback is NOT a PARSE_FAILURE — only adjudication parses are terminal.
            log.warning("[oracle] query-gen parse/shape failure, using fallback templates")
            return self._fallback_queries(a, b)
        n = min(len(queries), ORACLE_MAX_QUERIES)

        def _idx(key: str, default: int) -> int:
            v = (parsed or {}).get(key, default)
            return v if isinstance(v, int) and 0 <= v < n else default

        return QueryPlan(
            queries=queries,
            canonical_idx=_idx("canonical_idx", 0),
            relationship_idx=_idx("relationship_idx", min(1, n - 1)),
        )

    # ------------------------------------------------------------------ [R]
    async def retrieve_arxiv(self, query: str) -> List[EvidenceItem]:
        """arXiv API search; polite (>=ORACLE_ARXIV_DELAY_S between requests, serialized)."""
        import httpx

        url = (
            f"{_ARXIV_API}?search_query=all:{urllib.parse.quote(query)}"
            f"&start=0&max_results={ORACLE_ARXIV_MAX_RESULTS}"
            f"&sortBy=relevance&sortOrder=descending"
        )
        for attempt in range(ORACLE_ARXIV_RETRIES + 1):
            async with self._arxiv_lock:
                wait = ORACLE_ARXIV_DELAY_S - (time.monotonic() - self._arxiv_last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                        resp = await client.get(url)
                    self._arxiv_last_call = time.monotonic()
                    if resp.status_code != 200:
                        raise RuntimeError(f"arXiv HTTP {resp.status_code}")
                    return self._parse_arxiv_atom(resp.text, query)
                except Exception as ex:
                    self._arxiv_last_call = time.monotonic()
                    if attempt < ORACLE_ARXIV_RETRIES:
                        log.warning(f"[oracle] arXiv attempt {attempt + 1} failed, retrying: {ex}")
                    else:
                        log.warning(f"[oracle] arXiv gave up after {attempt + 1} attempts: {ex}")
            if attempt < ORACLE_ARXIV_RETRIES:
                await asyncio.sleep(2.0 * (attempt + 1))  # backoff outside the lock
        return []  # retrieval failure != parse failure

    @staticmethod
    def _parse_arxiv_atom(xml_text: str, query: str) -> List[EvidenceItem]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as ex:
            log.warning(f"[oracle] arXiv XML parse error: {ex}")
            return []
        items: List[EvidenceItem] = []
        for e in root.findall("atom:entry", _ATOM_NS)[:ORACLE_ARXIV_MAX_RESULTS]:
            title = (e.findtext("atom:title", "", _ATOM_NS) or "").strip().replace("\n", " ")
            summary = (e.findtext("atom:summary", "", _ATOM_NS) or "").strip().replace("\n", " ")
            link = (e.findtext("atom:id", "", _ATOM_NS) or "").strip()
            if not (title or summary):
                continue
            items.append(EvidenceItem(
                source="arxiv", url=link or f"arxiv:{title[:60]}",
                title=title, text=summary, query=query,
            ))
        return items

    async def retrieve_tavily(self, query: str, depth: Any = None) -> Tuple[List[EvidenceItem], float]:
        """Tavily via WebSearchManager (QUICK default). Returns (items, credits_used)."""
        if self.wsm is None:
            return [], 0.0
        from knowledge.web_search_manager import WebSearchDepth
        depth = depth or WebSearchDepth.QUICK
        try:
            result = await self.wsm.search(
                query, depth=depth, max_results=ORACLE_TAVILY_MAX_RESULTS, use_cache=True,
                localize=False,
            )
        except Exception as ex:
            log.warning(f"[oracle] Tavily search error: {ex}")
            return [], 0.0
        if getattr(result, "error", None):
            log.warning(f"[oracle] Tavily returned error: {result.error}")
            return [], 0.0
        source = "tavily_extract" if depth != WebSearchDepth.QUICK else "tavily_quick"
        items = [
            EvidenceItem(
                source=source, url=p.url, title=p.title or "",
                text=(p.content or p.snippet or ""), query=query,
                score=float(getattr(p, "score", 0.0) or 0.0),
            )
            for p in (result.pages or [])
            if getattr(p, "url", "")
        ]
        credits = 0.0 if getattr(result, "from_cache", False) else float(result.total_credits_used or 0.0)
        return items, credits

    @staticmethod
    def merge_evidence(*lists: List[EvidenceItem]) -> List[EvidenceItem]:
        """Dedupe by canonical URL (extracts beat snippets), cap at ORACLE_MAX_EVIDENCE.

        Slot reservation: at most ORACLE_MAX_ARXIV_SLOTS go to arXiv so web evidence is
        never crowded out — textbook identities live in books/reviews/Wikipedia, which
        only the Tavily channel carries (observed in the first live smoke: 4 arXiv
        queries filled all 8 slots and the paid-for web results were dropped)."""
        from knowledge.web_search_manager import _canonical_url
        by_url: Dict[str, EvidenceItem] = {}
        for lst in lists:
            for item in lst:
                key = _canonical_url(item.url)
                prev = by_url.get(key)
                if prev is None or (prev.source == "tavily_quick" and item.source == "tavily_extract"):
                    by_url[key] = item
        arxiv = [e for e in by_url.values() if e.source == "arxiv"]
        web = sorted((e for e in by_url.values() if e.source != "arxiv"),
                     key=lambda e: -e.score)
        merged = arxiv[:ORACLE_MAX_ARXIV_SLOTS] + web
        if len(merged) < ORACLE_MAX_EVIDENCE:  # backfill spare slots with remaining arXiv
            merged += arxiv[ORACLE_MAX_ARXIV_SLOTS:ORACLE_MAX_ARXIV_SLOTS
                            + (ORACLE_MAX_EVIDENCE - len(merged))]
        return merged[:ORACLE_MAX_EVIDENCE]

    # ------------------------------------------------------------------ [A]
    def _adj_prompt(self, a: str, b: str, claim: str, evidence: List[EvidenceItem]) -> str:
        schema_tpl = _ADJ_SCHEMA_VERDICT_FIRST if self.verdict_first else _ADJ_SCHEMA_REASONING_FIRST
        return _ADJ_PROMPT.format(
            a=a, b=b, claim=claim,
            evidence=_render_evidence(evidence, claim=claim),
            max_words=ORACLE_PASSAGE_MAX_WORDS,
            schema=schema_tpl.format(max_words=ORACLE_PASSAGE_MAX_WORDS),
        )

    async def adjudicate(
        self, a: str, b: str, claim: str, evidence: List[EvidenceItem]
    ) -> Tuple[Optional[dict], str, int]:
        """One adjudication LLM call + exactly one retry. Returns (parsed|None, raw, parse_failures)."""
        prompt = self._adj_prompt(a, b, claim, evidence)
        raw = ""
        failures = 0
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\n\nReturn ONLY the JSON object. No other text."
            try:
                raw = await self.mm.generate_once(
                    prompt + suffix,
                    model_name=self.model_name,
                    system_prompt=_ADJ_SYSTEM,
                    max_tokens=ORACLE_ADJ_MAX_TOKENS,
                    temperature=ORACLE_ADJ_TEMP,
                    disable_reasoning=True,
                )
            except Exception as ex:
                log.warning(f"[oracle] adjudication LLM error (attempt {attempt + 1}): {ex}")
                raw = ""
            parsed = _extract_json(raw)
            if parsed is not None and _parse_verdict_string(parsed.get("verdict")) is not None:
                return parsed, raw, failures
            failures += 1
            log.warning(f"[oracle] adjudication parse failure {failures}/2")
        return None, raw, failures

    # ------------------------------------------------------------------ [V]
    @staticmethod
    def enforce_evidence_rule(
        parsed: dict, evidence: List[EvidenceItem]
    ) -> Tuple[OracleVerdict, List[Citation], bool]:
        """Code-enforced evidence rule (never prompt-only). Returns (verdict, citations, downgraded).

        DOCUMENTED_* survives only with >=1 citation where: url matches a retrieved item
        (canonical form), passage <= 40 words, and passage is a verbatim (normalized)
        substring of that item's title+text."""
        from knowledge.web_search_manager import _canonical_url

        verdict_str = _parse_verdict_string(parsed.get("verdict"))
        if verdict_str is None:
            return OracleVerdict.UNCERTAIN, [], True
        verdict = OracleVerdict(verdict_str)
        if verdict not in (OracleVerdict.DOCUMENTED_EXACT, OracleVerdict.DOCUMENTED_ADJACENT):
            return verdict, [], False

        by_url = {_canonical_url(e.url): e for e in evidence}
        valid: List[Citation] = []
        for c in parsed.get("citations", []) or []:
            if not isinstance(c, dict):
                continue
            url = str(c.get("url", ""))
            passage = str(c.get("supporting_passage", ""))
            item = by_url.get(_canonical_url(url))
            if item is None:
                continue
            if not passage or len(passage.split()) > ORACLE_PASSAGE_MAX_WORDS:
                continue
            haystack = _norm_text(f"{item.title} {item.text}")
            if _norm_text(passage) not in haystack:
                continue
            valid.append(Citation(url=url, supporting_passage=passage,
                                  rationale=str(c.get("rationale", ""))))
        if valid:
            return verdict, valid, False
        return OracleVerdict.UNCERTAIN, [], True

    # ------------------------------------------------------------ orchestrator
    async def score_candidate(
        self,
        candidate_id: str,
        a: str,
        b: str,
        claim: str,
        *,
        run_id: str,
        source: str = "",
        pre_seen: bool = False,
        calibration_label: Optional[str] = None,
    ) -> OracleRecord:
        record = OracleRecord(
            run_id=run_id, candidate_id=candidate_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            concept_a=a, concept_b=b, claim_text=claim,
            source=source, pre_seen=pre_seen, calibration_label=calibration_label,
        )
        timing: Dict[str, float] = {}
        credits = 0.0

        # [Q]
        t0 = time.monotonic()
        plan = await self.generate_queries(a, b, claim)
        timing["query_gen_s"] = round(time.monotonic() - t0, 2)
        record.queries = plan.queries

        # [R] arXiv for ALL queries (free); Tavily QUICK for canonical, relationship,
        # then first remaining — up to ORACLE_TAVILY_QUERIES.
        t0 = time.monotonic()
        arxiv_lists = [await self.retrieve_arxiv(q) for q in plan.queries]
        tavily_order: List[int] = []
        for idx in (plan.canonical_idx, plan.relationship_idx, *range(len(plan.queries))):
            if idx not in tavily_order and 0 <= idx < len(plan.queries):
                tavily_order.append(idx)
        tavily_lists: List[List[EvidenceItem]] = []
        for idx in tavily_order[:ORACLE_TAVILY_QUERIES]:
            items, spent = await self.retrieve_tavily(plan.queries[idx])
            tavily_lists.append(items)
            credits += spent
        evidence = self.merge_evidence(*arxiv_lists, *tavily_lists)
        timing["retrieval_s"] = round(time.monotonic() - t0, 2)
        record.evidence = evidence

        # [A] + [V]
        t0 = time.monotonic()
        parsed, _raw, failures = await self.adjudicate(a, b, claim, evidence)
        record.parse_failures = failures
        if parsed is None:
            record.verdict = OracleVerdict.PARSE_FAILURE
            timing["adjudication_s"] = round(time.monotonic() - t0, 2)
            record.timing = timing
            record.credits_used = round(credits, 2)
            return record

        record.llm_prior_verdict = str(parsed.get("prior_verdict", "") or "")
        record.llm_prior_citation_claim = str(parsed.get("prior_citation_claim", "") or "")
        record.verdict_name = str(parsed.get("name", "") or "")
        verdict, citations, downgraded = self.enforce_evidence_rule(parsed, evidence)
        timing["adjudication_s"] = round(time.monotonic() - t0, 2)

        # Escalation (flag 5): documented-per-model but unciteable from snippets ->
        # one STANDARD-depth re-search of the canonical query, one re-adjudication.
        # Triggers on the code-enforced downgrade AND on the honest variant (model answers
        # UNCERTAIN/NOT_FOUND because it cannot cite, while its prior says documented).
        prior_documented = "DOCUMENTED" in record.llm_prior_verdict.upper()
        claimed_documented = "DOCUMENTED" in str(parsed.get("verdict", "")).upper()
        final_documented = verdict in (OracleVerdict.DOCUMENTED_EXACT, OracleVerdict.DOCUMENTED_ADJACENT)
        if (
            self.escalate and not final_documented
            and (claimed_documented or prior_documented)
            and self.wsm is not None
        ):
            t0 = time.monotonic()
            from knowledge.web_search_manager import WebSearchDepth
            extra, spent = await self.retrieve_tavily(
                plan.queries[plan.canonical_idx], depth=WebSearchDepth.STANDARD
            )
            credits += spent
            evidence = self.merge_evidence(evidence, extra)
            record.evidence = evidence
            record.escalated = True
            parsed2, _raw2, failures2 = await self.adjudicate(a, b, claim, evidence)
            record.parse_failures += failures2
            if parsed2 is None:
                record.verdict = OracleVerdict.PARSE_FAILURE
                timing["escalation_s"] = round(time.monotonic() - t0, 2)
                record.timing = timing
                record.credits_used = round(credits, 2)
                return record
            record.llm_prior_verdict = str(parsed2.get("prior_verdict", "") or record.llm_prior_verdict)
            record.verdict_name = str(parsed2.get("name", "") or record.verdict_name)
            verdict, citations, downgraded = self.enforce_evidence_rule(parsed2, evidence)
            timing["escalation_s"] = round(time.monotonic() - t0, 2)

        record.verdict = verdict
        record.evidence_citations = citations
        record.downgraded_from_prior = downgraded

        # Aux signals: NEVER affect the verdict; default off (wiki-FAISS memory safety).
        if self.enable_aux:
            try:
                from knowledge.doc_cooccurrence import doc_cooccurrence
                r = doc_cooccurrence(a, b, depth=40)
                record.aux_doc_cooccurrence = {
                    "shared": r.shared, "mention": r.mention, "known": r.known,
                }
            except Exception as ex:
                log.warning(f"[oracle] aux doc_cooccurrence failed (non-fatal): {ex}")

        record.timing = timing
        record.credits_used = round(credits, 2)
        return record
