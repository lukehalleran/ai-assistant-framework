"""
Shared memory utilities.

Common functions used across memory coordinator implementations.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any


def format_recent_conversations(
    entries: List[Dict],
    id_prefix: str = "recent"
) -> List[Dict]:
    """
    Format raw corpus entries into standardized memory format.

    Args:
        entries: Raw entries from corpus_manager.get_recent_memories()
        id_prefix: Prefix for generated IDs

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
            'relevance_score': 0.9,  # fresh bias
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
