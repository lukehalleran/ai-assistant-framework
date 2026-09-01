"""Structured public research/API searches shared by agentic and insight paths."""
from __future__ import annotations

import html
from typing import Any
import urllib.parse
import xml.etree.ElementTree as ET

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def parse_arxiv_entries(xml_text: str) -> list[dict[str, Any]]:
    """Parse arXiv Atom without discarding stable IDs or citation metadata."""
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        url = (entry.findtext("atom:id", "", _ATOM_NS) or "").strip()
        title = " ".join(
            (entry.findtext("atom:title", "", _ATOM_NS) or "").split()
        )
        abstract = " ".join(
            (entry.findtext("atom:summary", "", _ATOM_NS) or "").split()
        )
        authors = [
            (author.findtext("atom:name", "", _ATOM_NS) or "").strip()
            for author in entry.findall("atom:author", _ATOM_NS)
        ]
        authors = [author for author in authors if author]
        if not (url or title or abstract):
            continue
        rows.append({
            "source_id": url or f"arxiv:{title[:120]}",
            "title": title or "Untitled arXiv record",
            "abstract": abstract,
            "authors": authors,
            "published_date": (
                entry.findtext("atom:published", "", _ATOM_NS) or ""
            ).strip(),
            "date": (
                entry.findtext("atom:published", "", _ATOM_NS) or ""
            ).strip(),
            "updated_date": (
                entry.findtext("atom:updated", "", _ATOM_NS) or ""
            ).strip(),
            "url": url,
            "source": "arXiv",
        })
    return rows


async def search_arxiv(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    import httpx

    max_results = max(1, min(int(max_results), 20))
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{urllib.parse.quote(query)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    return parse_arxiv_entries(response.text)


def parse_stackexchange_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Stack Exchange JSON and retain question IDs and links."""
    rows: list[dict[str, Any]] = []
    for item in data.get("items", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        question_id = item.get("question_id")
        body = html.unescape(str(item.get("body") or ""))
        # The API body is HTML; a compact text rendition is enough for evidence
        # selection while the canonical link remains available for citation.
        import re
        body = " ".join(re.sub(r"<[^>]+>", " ", body).split())
        rows.append({
            "question_id": question_id,
            "source_id": f"stackexchange:{question_id}" if question_id else str(item.get("link") or ""),
            "title": html.unescape(str(item.get("title") or "")),
            "text": body,
            "url": str(item.get("link") or ""),
            # E-utilities/Stack Exchange may return epoch seconds as an int;
            # normalize to text so downstream EvidenceItem date validation
            # cannot crash the entire research turn.
            "date": str(item.get("creation_date")) if item.get("creation_date") is not None else "",
            "score": item.get("score", 0),
            "is_answered": bool(item.get("is_answered")),
            "accepted_answer_id": item.get("accepted_answer_id"),
            "source": "Stack Exchange",
        })
    return rows


def _clean_html(value: Any) -> str:
    import re
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(str(value or ""))).split())


def attach_stackexchange_answers(rows: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach the accepted (or highest-voted) answer to each question."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in (data or {}).get("items", []):
        try:
            grouped.setdefault(int(item.get("question_id")), []).append(item)
        except (TypeError, ValueError):
            continue
    for row in rows:
        try:
            answers = grouped.get(int(row.get("question_id")), [])
        except (TypeError, ValueError):
            answers = []
        if not answers:
            continue
        accepted_id = row.get("accepted_answer_id")
        answer = next((a for a in answers if a.get("answer_id") == accepted_id), None)
        answer = answer or max(answers, key=lambda item: item.get("score", 0) or 0)
        row["answer_id"] = answer.get("answer_id")
        row["answer_text"] = _clean_html(answer.get("body"))
        row["answer_score"] = answer.get("score", 0)
        row["answer_url"] = f"{row.get('url', '')}/{answer.get('answer_id')}"
    return rows


async def search_stackexchange(
    query: str, *, site: str = "stackoverflow", max_results: int = 5,
) -> list[dict[str, Any]]:
    import httpx

    max_results = max(1, min(int(max_results), 20))
    url = (
        "https://api.stackexchange.com/2.3/search/advanced"
        f"?order=desc&sort=votes&q={urllib.parse.quote(query)}"
        f"&site={urllib.parse.quote(site)}&filter=withbody&pagesize={max_results}"
    )
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("error_message"):
            raise RuntimeError(str(data["error_message"]))
        rows = parse_stackexchange_items(data)
        ids = [str(row.get("question_id")) for row in rows if row.get("question_id")]
        if ids:
            answer_url = (
                f"https://api.stackexchange.com/2.3/questions/{';'.join(ids)}/answers"
                f"?order=desc&sort=votes&site={urllib.parse.quote(site)}&filter=withbody"
            )
            answer_response = await client.get(answer_url)
            answer_response.raise_for_status()
            answer_data = answer_response.json()
            if answer_data.get("error_message"):
                raise RuntimeError(str(answer_data["error_message"]))
            rows = attach_stackexchange_answers(rows, answer_data)
    return rows


def format_arxiv_results(query: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"[arXiv] No results for: {query}"
    lines = [f"[arXiv SEARCH] {query}\n"]
    for index, row in enumerate(rows, 1):
        authors = ", ".join((row.get("authors") or [])[:3])
        if len(row.get("authors") or []) > 3:
            authors += " et al."
        marker = f"[{row.get('web_citation_id')}] " if row.get("web_citation_id") else ""
        lines.append(
            f"{index}. {marker}{row.get('title') or 'Untitled'}\n"
            f"   {authors}\n"
            f"   {(row.get('abstract') or '')[:400]}\n"
            f"   {row.get('url') or ''}\n"
        )
    return "\n".join(lines)


def format_stackexchange_results(
    query: str, rows: list[dict[str, Any]], *, site: str,
) -> str:
    if not rows:
        return f"[Stack Exchange] No results for: {query}"
    lines = [f"[STACK EXCHANGE — {site}] {query}\n"]
    for index, row in enumerate(rows, 1):
        accepted = "ACCEPTED" if row.get("accepted_answer_id") else ""
        answered = "[ANSWERED]" if row.get("is_answered") else ""
        marker = f"[{row.get('web_citation_id')}] " if row.get("web_citation_id") else ""
        lines.append(
            f"{index}. {marker}[{row.get('score', 0)} votes] {answered} {accepted}\n"
            f"   {row.get('title') or ''}\n"
            f"   {(row.get('text') or '')[:500]}\n"
            f"   Answer: {(row.get('answer_text') or 'No answer retrieved')[:500]}\n"
            f"   {row.get('url') or ''}\n"
        )
    return "\n".join(lines)
