"""Structured PubMed E-utilities search shared by agentic and deliberation paths."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence
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
    "should the their them there these this to use was were what "
    "when will with would you your about after before between during likely need "
    "question point source specific query person persons "
    # Domain nouns (treatment, medication, clinical, research, study, evidence,
    # trial, data) are deliberately NOT stopped: a query can genuinely be ABOUT
    # one of them ("rest days off medication effects" lost its own subject
    # noun here until 2026-09-06 — the anchor concept then fell to whatever
    # survived first, an unrelated word). Evidentiary REQUEST framing built
    # from these words ("is there evidence on…") is stripped structurally by
    # _EVIDENTIARY_REQUEST_FRAME_RE below instead of by deleting the words.
    #
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

# "Is there evidence on X" / "any research about X" is a REQUEST frame, not
# content — the evidentiary noun plus its preposition names the ACT of
# asking, and X is the real subject. This is the sibling of the reported-
# speech strip above: the words themselves (evidence, research, study,
# trial, data) are real PubMed content in other positions ("clinical trial
# data") and must stay eligible as query axes, so only the request-shaped
# construction is removed, never the bare word.
_EVIDENTIARY_REQUEST_FRAME_RE = re.compile(
    r"\b(?:research|studies|study|evidence|trials?|data)\s+"
    r"(?:on|about|for|regarding|into|of|around|concerning)\b",
    re.IGNORECASE,
)


def _query_terms(query: str) -> list[str]:
    """Extract content terms while discarding conversational/meta framing."""
    text = _REPORTED_SPEECH_FRAME_RE.sub(" ", str(query or ""))
    text = _EVIDENTIARY_REQUEST_FRAME_RE.sub(" ", text)
    words = re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
    out: list[str] = []
    for word in words:
        if word in _PUBMED_STOPWORDS or word in _PUBMED_NOISE or word in out:
            continue
        out.append(word)
    return out


def _alias_group(
    term: str, concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> tuple[str, ...]:
    """Return (term, *declared synonyms) for one query concept.

    Lookup is case-insensitive and also matches a multiword synonym KEY
    against one of the term's own tokens (a planner-declared key like
    "rest days" should still resolve when the extracted term is "rest").
    """
    if not concept_synonyms:
        return (term,)
    term_cf = str(term).casefold()
    aliases: list[str] = []
    for name, values in concept_synonyms.items():
        name_cf = str(name).casefold()
        if term_cf != name_cf and term_cf not in name_cf.split():
            continue
        for value in values or ():
            value_str = str(value).strip()
            if value_str and value_str.casefold() != term_cf:
                aliases.append(value_str)
    return tuple(dict.fromkeys([term, *aliases]))


def _ordered_concept_groups(
    terms: Iterable[str],
    *,
    anchor_terms: Optional[Sequence[str]] = None,
    concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> list[tuple[str, ...]]:
    """Build alias groups in term order, then surface the caller's anchor.

    Ranking and query-broadening both treat the FIRST group as the query's
    subject anchor. Positional order (whichever word happened to survive
    stopwording first) is not reliably the caller's real subject — a frozen
    evidence spec's own outcome/series terms are a better signal when
    supplied, so that group is moved to the front. Falls back to positional
    order (prior behavior) when no anchor_terms are given or none matches.
    """
    groups: list[tuple[str, ...]] = []
    for term in terms:
        group = _alias_group(term, concept_synonyms)
        if group not in groups:
            groups.append(group)
    return _ordered_concept_groups_with_anchor_count(
        groups, anchor_terms=anchor_terms, extend_multiword=True,
        concept_synonyms=concept_synonyms,
    )[0]


def _ordered_concept_groups_with_anchor_count(
    groups: list[tuple[str, ...]],
    *,
    anchor_terms: Optional[Sequence[str]] = None,
    extend_multiword: bool = False,
    concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> tuple[list[tuple[str, ...]], int]:
    """Move EVERY group that matches a caller-declared anchor term to the
    front (anchor order preserved) and return how many groups are anchors.

    Fable referee (2026-09-06): surfacing only the FIRST matching anchor
    under-covered the live incident — the frozen spec listed its series
    terms as ["rest days", "medication"], so "rest" still became the sole
    anchor and a pitcher-workload abstract (rest + days) passed. A planner's
    series/outcome terms are the AXES of the comparison: an abstract missing
    any of them is off-topic, so all of them are mandatory hits. Positional
    fallback (anchor count 1 = first group) when no anchor_terms match.
    """
    if not groups or not anchor_terms:
        return groups, (1 if groups else 0)
    anchored: list[tuple[str, ...]] = []
    matched_groups: list[tuple[str, ...]] = []
    for raw in anchor_terms:
        candidate = str(raw or "").strip().casefold()
        if not candidate:
            continue
        # Planner keys arrive as identifiers ("rest_days", "medication_use")
        # as often as phrases — underscores are token separators too.
        candidate_tokens = [tok for tok in re.split(r"[\s_-]+", candidate) if tok]
        matched_here = False
        for group in groups:
            if group in matched_groups:
                continue
            group_terms = {item.casefold() for item in group}
            if candidate in group_terms or any(tok in group_terms for tok in candidate_tokens):
                if extend_multiword and len(candidate_tokens) > 1:
                    # A multiword axis ("rest days") reached this group through
                    # ONE of its tokens; for relevance scoring the axis hits
                    # when any of its own content tokens or aliases appears —
                    # "days off medication" satisfies the rest-days axis via
                    # "days". The CONJUNCTION of all axes is what discriminates
                    # (the pitcher abstract still fails the medication axis).
                    extra = tuple(
                        tok for tok in candidate_tokens
                        if len(tok) >= 3 and tok not in _PUBMED_STOPWORDS
                        and tok not in group_terms
                    )
                    anchored.append((*group, *extra))
                else:
                    anchored.append(group)
                matched_groups.append(group)
                matched_here = True
                break
        if not matched_here and extend_multiword:
            # A declared axis ABSENT from the query's own terms (the planner's
            # outcome terms "well-being"/"symptoms" never appear in
            # "rest days off medication effects") is still an axis of the
            # question: an abstract that mentions none of it is off-topic
            # (live: fundoplication / hip-surgery abstracts matched "rest" +
            # "off medication"). Add it as its own mandatory group with its
            # content tokens and declared synonyms.
            phrase = " ".join(t for t in re.split(r"[\s_]+", candidate) if t)
            content = [t for t in re.split(r"[\s_]+", candidate)
                       if len(t) >= 3 and t not in _PUBMED_STOPWORDS and t != phrase]
            synonyms = []
            for name, values in (concept_synonyms or {}).items():
                if str(name).casefold().replace("_", " ") == phrase:
                    synonyms.extend(str(v).strip().casefold() for v in (values or ()) if str(v).strip())
            new_group = tuple(dict.fromkeys([phrase, *content, *synonyms]))
            if new_group and new_group not in anchored:
                anchored.append(new_group)
    if not anchored:
        return groups, 1
    rest = [g for g in groups if g not in matched_groups]
    return [*anchored, *rest], len(anchored)


def build_pubmed_query_ladder(
    query: str,
    *,
    supporting_facets: Iterable[str] = (),
    refuting_facets: Iterable[str] = (),
    rival_explanations: Iterable[str] = (),
    concept_synonyms: dict[str, Iterable[str]] | None = None,
    anchor_terms: Optional[Sequence[str]] = None,
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
    groups = _ordered_concept_groups(
        primary_terms, anchor_terms=anchor_terms, concept_synonyms=concept_synonyms,
    )
    # Broadening (the drop-from-tail loop and the final expanded[:2] slice
    # below) must not silently lose the anchor concept just because it did
    # not survive stopwording first — reorder primary_terms to match groups
    # so every later slice keeps the anchor in its protected leading slots.
    primary_terms = [group[0] for group in groups]

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


def _row_relevance(
    row: dict[str, Any], query: str,
    *,
    anchor_terms: Optional[Sequence[str]] = None,
    concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> tuple[float, int, bool]:
    """Return a lexical relevance score, distinct concept-hit count, and
    whether the query's anchor concept hits.

    The anchor is the FIRST group in ``anchor_terms`` order that is present
    in the query's own extracted terms; with no anchor_terms (or no match)
    this falls back to positional order — the first surviving term, exactly
    the prior behavior.
    """
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or row.get("text") or "").lower()
    terms = _query_terms(query)
    if not terms:
        return 0.0, 0, False
    # Boolean query expansions (e.g. irritability OR agitation) should count
    # as one concept, not as four independent hits.
    base_groups: list[tuple[str, ...]] = []
    for term in terms:
        group = _alias_group(term, concept_synonyms)
        if group not in base_groups:
            base_groups.append(group)
    groups, anchor_count = _ordered_concept_groups_with_anchor_count(
        base_groups, anchor_terms=anchor_terms, extend_multiword=True,
        concept_synonyms=concept_synonyms,
    )
    hits = 0
    score = 0.0
    anchor_flags: list[bool] = []
    for position, aliases in enumerate(groups):
        title_hit = any(re.search(rf"\b{re.escape(alias)}\b", title) for alias in aliases)
        abstract_hit = any(re.search(rf"\b{re.escape(alias)}\b", abstract) for alias in aliases)
        matched = bool(title_hit or abstract_hit)
        if matched:
            hits += 1
            score += 2.0 if title_hit else 1.0
        if position < anchor_count:
            anchor_flags.append(matched)
    # Every declared anchor axis must hit (positional fallback = first group).
    anchor_hit = bool(anchor_flags) and all(anchor_flags)
    return score / max(1, len(groups)), hits, anchor_hit


def rank_pubmed_rows(
    rows: Iterable[dict[str, Any]],
    query: str,
    *,
    max_results: int = 20,
    min_distinct_hits: int = 1,
    anchor_terms: Optional[Sequence[str]] = None,
    concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> PubMedRows:
    """Apply a conservative lexical relevance floor and rank accepted rows.

    PubMed's own relevance ranking is retained as the retrieval order, then
    the title/abstract floor prevents broad queries from contributing unrelated
    records. This is intentionally deterministic and does not pretend that a
    lexical score is clinical semantic judgment.
    """
    raw_rows = list(rows or [])
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    query_concepts = len(_ordered_concept_groups(
        _query_terms(query), anchor_terms=anchor_terms, concept_synonyms=concept_synonyms,
    ))
    required_hits = max(
        1,
        int(min_distinct_hits),
        2 if query_concepts >= 2 else 1,
    )
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            continue
        score, hits, anchor_hit = _row_relevance(
            row, query, anchor_terms=anchor_terms, concept_synonyms=concept_synonyms,
        )
        # The anchor concept (positional first, or the caller's declared
        # subject when anchor_terms is supplied) — a row matching only other
        # axes is topically adjacent noise (a sleep-quality concept paper is
        # not a caffeine paper).
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


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_RETRY_BACKOFF_CAP_S = 8.0


async def _get_with_retry(client: Any, url: str, *, max_attempts: int = 3) -> Any:
    """GET with bounded retries, honoring a Retry-After header when present.

    Backoff is capped so a large/garbled header value can never stall a
    turn; falling back to a small exponential backoff otherwise. The final
    (non-retryable-or-exhausted) response is returned as-is — callers decide
    how to classify a persistent failure.
    """
    import asyncio

    response = None
    for attempt in range(max(1, max_attempts)):
        response = await client.get(url)
        if response.status_code not in _RETRYABLE_STATUS_CODES:
            break
        if attempt < max_attempts - 1:
            retry_after = response.headers.get("Retry-After")
            delay = None
            if retry_after:
                try:
                    delay = min(_RETRY_BACKOFF_CAP_S, max(0.1, float(retry_after)))
                except (TypeError, ValueError):
                    delay = None
            if delay is None:
                delay = min(_RETRY_BACKOFF_CAP_S, 0.5 * (2 ** attempt))
            await asyncio.sleep(delay)
    return response


async def search_pubmed(
    query: str,
    *,
    max_results: int = 5,
    anchor_terms: Optional[Sequence[str]] = None,
    concept_synonyms: Optional[dict[str, Iterable[str]]] = None,
) -> list[dict[str, Any]]:
    """Search and fetch PubMed rows while retaining PMIDs and citations.

    ``anchor_terms``/``concept_synonyms`` (2026-09-06) reach the per-rung
    ranker: without them a broadened rung ("rest AND days") anchored on its own
    first word and farm-labor / nurse-shift / bed-rest abstracts passed."""
    import httpx

    max_results = max(1, min(int(max_results), 20))
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}"
        "&sort=relevance&retmode=xml"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        search_response = await _get_with_retry(client, search_url)
    if search_response.status_code == 429:
        # Rate-limited on every attempt: the source is reachable but
        # throttled, not empty and not broken — partial, never
        # no_relevant_results (which would misreport "searched, found
        # nothing topical").
        return PubMedRows(status="partial")
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
        fetch_response = await _get_with_retry(client, fetch_url)
    if fetch_response.status_code == 429:
        return PubMedRows(status="partial")
    fetch_response.raise_for_status()
    # The E-utilities result set is often broad for adjacent endpoint queries;
    # rank and floor it before it enters the shared evidence manifest.
    return rank_pubmed_rows(parse_pubmed_articles(fetch_response.text), query,
                            max_results=max_results, anchor_terms=anchor_terms,
                            concept_synonyms=concept_synonyms)


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
