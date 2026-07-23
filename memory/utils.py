"""
Shared memory utilities.

Common functions used across memory coordinator implementations.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any


# Bare connectivity-check exchanges ("test") — worthless as memories but they
# embed close to everything short and compete for retrieval slots.
_TRIVIAL_TEST_QUERIES = frozenset({"test", "testing", "test test", "hello test"})


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
