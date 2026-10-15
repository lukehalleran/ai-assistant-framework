"""Corpus-copy matcher for quarantine flips (2026-10-15, BC-26/BC-58).

Every conversation, summary and reflection lives TWICE: a Chroma document and a
dict in the rolling corpus JSON (``memory/corpus_manager.py``).  A quarantine
that flags only one copy hides half the record.  This leaf module is the ONE
place that maps a Chroma doc to its corpus entry and the ONE place that applies
the flag to the corpus copy, so every quarantine flip (owner tool, curation
engine) shares the same identity rule.

Identity is structural equality only (query / content / original_text /
thread_id / timestamp window) - no text heuristics.  Zero or several candidates
are reported, never guessed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from utils.date_coerce import to_naive_local

TIMESTAMP_WINDOW_S = 120.0

OUTCOME_MATCH = "match"
OUTCOME_NONE = "no_corpus_copy"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_UNSUPPORTED = "unsupported_collection"

SUPPORTED = ("conversations", "summaries", "reflections")


def _norm_ws(text: Any) -> str:
    return " ".join(str(text or "").split())


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return to_naive_local(value)
    if isinstance(value, str) and value.strip():
        try:
            return to_naive_local(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _entry_type(entry: dict) -> str:
    return str(entry.get("type") or "").lower()


def _is_conversation(entry: dict) -> bool:
    return _entry_type(entry) not in ("summary", "reflection") and "@summary" not in (entry.get("tags") or [])


def _doc_query(chroma_doc: dict) -> str:
    md = chroma_doc.get("metadata") or {}
    q = md.get("query")
    if q:
        return _norm_ws(q)
    content = str(chroma_doc.get("content") or chroma_doc.get("document") or "")
    if content.startswith("User:"):
        head = content[len("User:"):].split("\nAssistant:", 1)[0]
        return _norm_ws(head)
    return ""


def _resolve(indices: list[int]) -> tuple[Optional[int], str]:
    if len(indices) == 1:
        return indices[0], OUTCOME_MATCH
    if not indices:
        return None, OUTCOME_NONE
    return None, OUTCOME_AMBIGUOUS


def match_corpus_index(collection: str, chroma_doc: dict, corpus: list) -> tuple[Optional[int], str]:
    """Index of the corpus entry that is the same record as ``chroma_doc``.

    Returns ``(index, reason)``; reason in match / no_corpus_copy / ambiguous /
    unsupported_collection.  ``chroma_doc`` = {"id", "content"|"document", "metadata"}.
    """
    if collection not in SUPPORTED:
        return None, OUTCOME_UNSUPPORTED
    md = chroma_doc.get("metadata") or {}
    content = str(chroma_doc.get("content") or chroma_doc.get("document") or "")

    if collection == "conversations":
        query = _doc_query(chroma_doc)
        if not query:
            return None, OUTCOME_NONE
        cands = [i for i, e in enumerate(corpus)
                 if _is_conversation(e) and _norm_ws(e.get("query")) == query]
        thread = md.get("thread_id")
        if thread:
            same = [i for i in cands if corpus[i].get("thread_id") == thread]
            if same:
                cands = same
        doc_ts = _parse_ts(md.get("timestamp"))
        if doc_ts is not None:
            cands = [i for i in cands
                     if (t := _parse_ts(corpus[i].get("timestamp"))) is not None
                     and abs((t - doc_ts).total_seconds()) <= TIMESTAMP_WINDOW_S]
        return _resolve(cands)

    if collection == "summaries":
        body = content.strip()
        if not body:
            return None, OUTCOME_NONE
        cands = [i for i, e in enumerate(corpus)
                 if _entry_type(e) == "summary" and str(e.get("content") or "").strip() == body]
        return _resolve(cands)

    # reflections
    body = str(md.get("original_text") or "").strip() or content.strip()
    if not body:
        return None, OUTCOME_NONE
    cands = [i for i, e in enumerate(corpus)
             if _entry_type(e) == "reflection" and str(e.get("content") or "").strip() == body]
    if len(cands) > 1:
        ts = str(md.get("timestamp") or "")
        tied = [i for i in cands if str(corpus[i].get("timestamp") or "") == ts]
        if len(tied) == 1:
            cands = tied
    return _resolve(cands)


def set_corpus_flag(entry: dict, update: dict) -> None:
    """Apply a quarantine metadata update (the SAME keys
    ``scripts.quarantine_facts.metadata_update`` produces) to a corpus entry.
    The flag is a TOP-LEVEL key, which is what ``memory.utils.is_quarantined``
    reads on a corpus dict."""
    entry.update(update)
