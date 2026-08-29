"""
Shared memory utilities.

Common functions used across memory coordinator implementations.

Retrieval/storage junk guards:
- is_junk_conversation_doc(query, response) [2026-07-15, extended 07-25]: drops
  API-error-sentinel turns, bare "test" exchanges, and the owner's box-test
  battery (exact-match) at retrieval — hybrid + fallback paths both call it.
- is_junk_summary(text) [2026-07-25, ext 08-03]: rejects "-"/near-empty/
  truncated-fragment summaries AND API-error sentinel text (long error strings
  passed the length/letter checks and sat in the summaries collection);
  enforced at BOTH storage (chroma add_summary, corpus add_summary) and
  retrieval (MemoryRetriever.get_summaries).
"""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Any


# Bare connectivity-check exchanges ("test") — worthless as memories but they
# embed close to everything short and compete for retrieval slots.
# The four Python-project lines are the owner's standing box-test battery
# (exact-match only): re-run for months after changes, they had accreted 547
# stored conversation docs by 2026-07-25 and were surfacing as [RELEVANT
# MEMORIES] inside unrelated emotional conversations.
_TRIVIAL_TEST_QUERIES = frozenset({
    "test", "testing", "test test", "hello test",
    "hey, i'm working on a python project",
    "it's a rag system with vector search",
    "using chromadb for the vector store",
    "should i add prompt caching?",
})


def is_quarantined(metadata) -> bool:
    """True for docs the curation engine has quarantined (2026-08-28,
    docs/AUTONOMOUS_CURATION_DESIGN.md). Quarantine replaces deletion: the
    doc stays on disk (fully restorable via the Curation Center undo) but
    must never surface in a prompt. Checked beside the content predicates
    at every retrieval filter site."""
    return bool(metadata) and bool(metadata.get("curation_quarantined"))


def is_junk_conversation_doc(content: str = "", query: str = "", response: str = "") -> bool:
    """
    True for stored conversation docs that should never surface at retrieval:

    - API-error sentinel turns: transport failures persisted as replies
      before the 2026-07-03 storage-time guard (`_is_api_error_response`).
      Historical docs from Feb–March 2026 predate it and were seen ranking
      in top-10 retrieval (2026-07-15). Detected via the assistant side of
      the doc starting with any `models.model_manager.API_ERROR_PREFIXES`.
    - Bare connectivity-test exchanges (query is literally "test"/"testing").

    Works on either the flat doc text ("User: ...\\nAssistant: ...") or the
    separate query/response fields — callers pass whatever they have.
    This is the RETRIEVAL-TIME belt; `scripts/purge_error_memories.py` is the
    (dry-run-first) suspenders for the stored docs themselves.
    """
    from models.model_manager import API_ERROR_PREFIXES

    content = content or ""
    query = query or ""
    response = response or ""

    # Assistant text: explicit response field, else the doc's Assistant: part
    assistant_text = response
    user_text = query
    if content:
        head, sep, tail = content.partition("\nAssistant:")
        if sep:
            if not assistant_text:
                assistant_text = tail
            if not user_text and head.startswith("User:"):
                user_text = head[len("User:"):]
        elif not assistant_text:
            assistant_text = content

    if assistant_text.lstrip().startswith(API_ERROR_PREFIXES):
        return True

    if user_text.strip().lower() in _TRIVIAL_TEST_QUERIES:
        return True

    return False


# A real session summary is at least a sentence. Historical junk in the
# summaries collection ("-", truncated fragments like "* The user is
# experiencing sleep") embeds close to the null/empty embedding and was
# dominating "recent summaries" while the store held fresh full summaries.
SUMMARY_MIN_CHARS = 40

# Infra/harness sentinels stored as summaries: a doc that OPENS with a
# bracketed block naming an infrastructure failure is a tool message, not a
# summary. Seen live 2026-08-05: "[Wafer: response was truncated before the
# model finished its internal reasoning. Increase max_tokens ... then retry.]"
# surfacing under [SEMANTIC SUMMARIES]. API_ERROR_PREFIXES catches the known
# prefix tags; this catches the long tail of provider-specific wrappers by
# keyword instead of by exact prefix.
_HARNESS_SENTINEL_RE = re.compile(
    r"^\[[^\]]{0,200}?(?:error|truncat|timed?\s?out|unavailable|max_tokens|retry)",
    re.IGNORECASE,
)


def is_junk_summary(text: Any) -> bool:
    """True for summary docs that should never be stored or surfaced.

    Shared by the storage-time guard (reject before add_summary) and the
    retrieval-time filter in MemoryRetriever.get_summaries — same
    belt-and-suspenders split as is_junk_conversation_doc.
    """
    if not text or not isinstance(text, str):
        return True
    stripped = text.strip()
    if len(stripped) < SUMMARY_MIN_CHARS:
        return True
    # Placeholder/divider docs ("-", "...", "***") carry no letters to speak of.
    if sum(1 for c in stripped if c.isalpha()) < 20:
        return True
    # API-error sentinels stored as summaries (2026-08-03: ~a dozen
    # "[API Error] 402 ..." docs found in the summaries collection by the
    # cross-dedup queue review — long and letter-rich, so the length/letter
    # checks above pass them).
    from models.model_manager import API_ERROR_PREFIXES
    if stripped.startswith(API_ERROR_PREFIXES):
        return True
    # Provider/harness wrappers not covered by the fixed prefix list
    # ("[Wafer: response was truncated ...]" and friends).
    if _HARNESS_SENTINEL_RE.match(stripped):
        return True
    return False


def format_recent_conversations(
    entries: List[Dict],
    id_prefix: str = "recent",
    base_relevance: float = 0.9,
) -> List[Dict]:
    """
    Format raw corpus entries into standardized memory format.

    Args:
        entries: Raw entries from corpus_manager.get_recent_memories()
        id_prefix: Prefix for generated IDs
        base_relevance: Relevance score to assign (0.9 default, lower for
            temporal queries where semantic scores should dominate)

    Returns:
        List of formatted memory dicts with standardized structure
    """
    out: List[Dict] = []
    for e in entries:
        ts = e.get('timestamp', datetime.now())
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                ts = datetime.now()

        out.append({
            'id': f"{id_prefix}::{uuid.uuid4().hex[:8]}",
            'query': e.get('query', ''),
            'response': e.get('response', ''),
            'content': f"User: {e.get('query', '')}\nAssistant: {e.get('response', '')}",
            'timestamp': ts,
            'source': 'corpus',
            'collection': 'recent',
            'relevance_score': base_relevance,
            'metadata': {
                'timestamp': ts.isoformat() if isinstance(ts, datetime) else str(ts),
                'truth_score': e.get('truth_score', 0.6),
                'importance_score': e.get('importance_score', 0.5),
                'tags': e.get('tags', []),
                'access_count': 0,
            },
            'tags': e.get('tags', []),
            'truth_score': e.get('truth_score', 0.6),
            'importance_score': e.get('importance_score', 0.5),
        })
    return out
