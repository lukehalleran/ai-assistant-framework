"""Structured PubMed E-utilities search shared by agentic and deliberation paths."""
from __future__ import annotations

from typing import Any, Iterable
import urllib.parse
import xml.etree.ElementTree as ET
import re


class PubMedRows(list):
    """List-compatible result carrying source-level retrieval status.

    Existing adapters consume plain lists, so keeping this as a list subclass
    lets the coordinator distinguish an empty, reachable PubMed search from a
    search that returned records which failed the relevance floor.
    """

    def __init__(self, rows: Iterable[dict[str, Any]] = (), *, status: str = "succeeded"):
        super().__init__(rows)
        self.status = status


_PUBMED_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has "
    "have how i if in is it its may me more most my of on or our people please "
    "research should study the their them there these this to use was were what "
    "when will with would you your about after before between during likely need "
    "question point treatment medication source specific query clinical data "
    "evidence trial person persons "
    # Tool/database names the user says when NAMING a source ("is there PubMed
    # evidence on…") — as query axes they select systematic reviews whose
    # abstracts describe their own search ("we searched PubMed…"): a 2026-08-31
    # live turn got 8/8 such reviews and zero topical papers. Plus bare
    # request verbs/prepositions from the same framing.
    "pubmed medline embase cochrane scopus wikipedia wiki arxiv wolfram "
    "google scholar check checked against last recorded"
    .split()
)
# Bare temporal units and reported-speech words — never useful as PubMed
# query axes. Transition/population terms (taper, cessation, stopping,
# spectrum, therapist…) are deliberately NOT here: they are the literature's
# own vocabulary, and stripping them breaks the very queries this ladder
# exists to serve (discontinuation studies, spectrum-disorder populations,
# therapy-outcome literature).
_PUBMED_NOISE = frozenset(
    "ago month months week weeks long term overall raised said helps help "
    # Analysis-framing vocabulary: as a query axis, "co-occurrence" selects
    # METHODOLOGY papers about co-occurrence of anything (live 2026-08-31:
    # drunkorexia/addiction-comorbidity junk for a study-vs-exercise
    # question). The substantive terms alongside it carry the real topic.
    "co-occurrence cooccurrence co-occur co-occurring co-occurred correlate "
    "correlation correlations association associations".split()
)

# Reported-speech frames ("my therapist says", "her doctor suggested") are
# evidential framing, not query content — strip the FRAME and keep what was
# said. Word-level stopwording can't make that distinction: "therapist" is
# genuine content in a therapy-outcomes question. Mirrors the pattern
# engine's reported-speech classifier.
_REPORTED_SPEECH_FRAME_RE = re.compile(
    r"\b(?:my|our|her|his|their|the|a)?\s*"
    r"(?:therapists?|doctors?|psychiatrists?|psychologists?|counsell?ors?|"
    r"physicians?|nurses?|friends?|mom|dad|parents?)\s+"
    r"(?:says?|said|think(?:s)?|thought|suggest(?:s|ed)?|believes?|"
    r"recommend(?:s|ed)?|raised|mentioned|told\s+me)\b",
    re.IGNORECASE,
)


def _query_terms(query: str) -> list[str]:
    """Extract content terms while discarding conversational/meta framing."""
    text = _REPORTED_SPEECH_FRAME_RE.sub(" ", str(query or ""))
    words = re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
    out: list[str] = []
    for word in words:
        if word in _PUBMED_STOPWORDS or word in _PUBMED_NOISE or word in out:
            continue
        out.append(word)
    return out


def _alias_group(term: str) -> tuple[str, ...]:
    return (term,)


def build_pubmed_query_ladder(
    query: str,
    *,
    supporting_facets: Iterable[str] = (),
    refuting_facets: Iterable[str] = (),
    rival_explanations: Iterable[str] = (),
    concept_synonyms: dict[str, Iterable[str]] | None = None,
    limit: int = 8,
) -> list[str]:
    """Build a deterministic, broadening PubMed query ladder.

    The first query preserves all substantive concepts. Later queries remove
    one axis at a time and add outcome/transition synonyms. This prevents a
    prose recovery query from being sent verbatim while still finding papers
    whose abstracts use adjacent endpoints such as agitation or aggression.
    """
    limit = max(1, min(int(limit), 8))
    raw = " ".join(str(query or "").split())
    def _values(value: Iterable[str]) -> list[str]:
        return [value] if isinstance(value, str) else list(value or [])

    supplemental = " ".join(
        str(value).strip() for value in (
            *_values(supporting_facets), *_values(refuting_facets),
            *_values(rival_explanations)
        ) if str(value).strip()
    )
    terms = _query_terms(f"{raw} {supplemental}")
    if not terms:
        return [raw[:500]] if raw else []

    # Keep first-occurrence order, but avoid turning generic facet words into
    # query axes. The source-specific query itself is preferred over facets.
    primary_terms = _query_terms(raw)
    if not primary_terms:
        primary_terms = terms
    aliases: dict[str, tuple[str, ...]] = {}
    for name, values in (concept_synonyms or {}).items():
        cleaned = tuple(dict.fromkeys([str(name).casefold(), *(str(v).casefold() for v in values)]))
        if cleaned:
            aliases[str(name).casefold()] = cleaned
    groups: list[tuple[str, ...]] = []
    for term in primary_terms:
        group = _alias_group(term)
        if group not in groups:
            groups.append(group)

    def _render(selected: list[tuple[str, ...]]) -> str:
        rendered = []
        for group in selected:
            choices = [f'"{item}"' if " " in item else item for item in group]
            rendered.append(
                choices[0] if len(choices) == 1 else f"({' OR '.join(choices)})"
            )
        return " AND ".join(rendered)

    candidates: list[str] = []
    # Direct, all-concept query.
    candidates.append(_render(groups))
    # Adjacent endpoint query: expand only recognized outcome groups.
    expanded = [
        aliases.get(term, (term,))
        for term in primary_terms
    ]
    candidates.append(_render(expanded))
    # Broaden one axis at a time, preserving the named entity/population when
    # present. The last one or two terms are commonly context words; dropping
    # them is safer than dropping the first named concepts.
    for drop_count in range(1, min(3, len(groups)) + 1):
        if len(groups) - drop_count >= 2:
            candidates.append(_render(groups[:-drop_count]))
    # A contextual population/outcome query is useful when the named exposure
    # has no indexed result, and a transition/outcome query catches withdrawal
    # or relapse literature.
    candidates.append(_render(expanded[:2]))

    return list(dict.fromkeys(candidate[:500] for candidate in candidates if candidate.strip()))[:limit]


def _row_relevance(row: dict[str, Any], query: str) -> tuple[float, int, bool]:
    """Return a lexical relevance score, distinct concept-hit count, and
    whether the query's FIRST (anchor) concept hits."""
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or row.get("text") or "").lower()
    terms = _query_terms(query)
    if not terms:
        return 0.0, 0, False
    # Boolean query expansions (e.g. irritability OR agitation) should count
    # as one concept, not as four independent hits.
    groups: list[tuple[str, ...]] = []
    for term in terms:
        group = _alias_group(term)
        if group not in groups:
            groups.append(group)
    hits = 0
    score = 0.0
    anchor_hit = False
    for position, aliases in enumerate(groups):
        title_hit = any(re.search(rf"\b{re.escape(alias)}\b", title) for alias in aliases)
        abstract_hit = any(re.search(rf"\b{re.escape(alias)}\b", abstract) for alias in aliases)
        if title_hit or abstract_hit:
            hits += 1
            score += 2.0 if title_hit else 1.0
            if position == 0:
                anchor_hit = True
    return score / max(1, len(groups)), hits, anchor_hit


def rank_pubmed_rows(
    rows: Iterable[dict[str, Any]],
    query: str,
    *,
    max_results: int = 20,
    min_distinct_hits: int = 1,
) -> PubMedRows:
    """Apply a conservative lexical relevance floor and rank accepted rows.

    PubMed's own relevance ranking is retained as the retrieval order, then
    the title/abstract floor prevents broad queries from contributing unrelated
    records. This is intentionally deterministic and does not pretend that a
    lexical score is clinical semantic judgment.
    """
    raw_rows = list(rows or [])
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    query_concepts = len({
        _alias_group(term) for term in _query_terms(query)
    })
    required_hits = max(
        1,
        int(min_distinct_hits),
        2 if query_concepts >= 2 else 1,
    )
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            continue
        score, hits, anchor_hit = _row_relevance(row, query)
        # The first concept is the query's subject anchor — a row matching
        # only later axes is topically adjacent noise (a sleep-quality
        # concept paper is not a caffeine paper).
        if hits < required_hits or not anchor_hit:
            continue
        item = dict(row)
        item["relevance_score"] = round(score, 4)
        item["relevance_hits"] = hits
        item["relevance_basis"] = "title/abstract lexical concept overlap"
        ranked.append((score, hits, -index, item))
    ranked.sort(reverse=True, key=lambda value: (value[0], value[1], value[2]))
    return PubMedRows(
        (item for _, _, _, item in ranked[:max(1, min(int(max_results), 20))]),
        status="succeeded" if ranked else ("no_relevant_results" if raw_rows else "no_results"),
    )


def parse_pubmed_articles(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed XML into citation-preserving evidence rows."""
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = (node.get("Label") or "").strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        authors = []
        for author in article.findall(".//Author"):
            collective = (author.findtext("CollectiveName") or "").strip()
            last = (author.findtext("LastName") or "").strip()
            initials = (author.findtext("Initials") or "").strip()
            name = collective or " ".join(part for part in (last, initials) if part)
            if name:
                authors.append(name)
        pub_date = article.find(".//JournalIssue/PubDate")
        published_date = ""
        if pub_date is not None:
            year = (pub_date.findtext("Year") or "").strip()
            month = (pub_date.findtext("Month") or "").strip()
            day = (pub_date.findtext("Day") or "").strip()
            medline = (pub_date.findtext("MedlineDate") or "").strip()
            published_date = "-".join(part for part in (year, month, day) if part) or medline
        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if (article_id.get("IdType") or "").lower() == "doi":
                doi = (article_id.text or "").strip()
                break
        rows.append({
            "pmid": pmid,
            "source_id": f"pmid:{pmid}" if pmid else "",
            "title": title or "Untitled PubMed record",
            "abstract": "\n".join(abstract_parts),
            "authors": authors,
            "published_date": published_date,
            "date": published_date,
            "journal": (article.findtext(".//Journal/Title") or "").strip(),
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "source": "PubMed",
        })
    return rows


async def search_pubmed(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    """Search and fetch PubMed rows while retaining PMIDs and citations."""
    import asyncio
    import httpx

    max_results = max(1, min(int(max_results), 20))
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}"
        "&sort=relevance&retmode=xml"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        search_response = None
        for attempt in range(2):
            search_response = await client.get(search_url)
            if search_response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt == 0:
                retry_after = search_response.headers.get("Retry-After", "0.5")
                try:
                    delay = min(2.0, max(0.1, float(retry_after)))
                except (TypeError, ValueError):
                    delay = 0.5
                await asyncio.sleep(delay)
        search_response.raise_for_status()
    root = ET.fromstring(search_response.text)
    ids = [node.text for node in root.findall(".//Id") if node.text]
    if not ids:
        return PubMedRows(status="no_results")

    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=xml"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        fetch_response = None
        for attempt in range(2):
            fetch_response = await client.get(fetch_url)
            if fetch_response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt == 0:
                await asyncio.sleep(0.5)
        fetch_response.raise_for_status()
    # The E-utilities result set is often broad for adjacent endpoint queries;
    # rank and floor it before it enters the shared evidence manifest.
    return rank_pubmed_rows(parse_pubmed_articles(fetch_response.text), query,
                            max_results=max_results)


def format_pubmed_results(query: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"[PubMed] No results for: {query}"
    lines = [f"[PUBMED SEARCH] {query}\n"]
    for index, row in enumerate(rows, 1):
        authors = ", ".join((row.get("authors") or [])[:3])
        if len(row.get("authors") or []) > 3:
            authors += " et al."
        marker = f"[{row.get('web_citation_id')}] " if row.get("web_citation_id") else ""
        lines.append(
            f"{index}. {marker}{row.get('title') or 'Untitled'}\n"
            f"   {authors}\n"
            f"   {(row.get('abstract') or 'No abstract')[:400]}\n"
            f"   {row.get('url') or ''}\n"
        )
    return "\n".join(lines)
