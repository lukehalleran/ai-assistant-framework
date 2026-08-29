"""Shared paginated chroma scan for curators (read-only)."""

from typing import Any, Dict, Iterator

from memory.curation.adapters import QUARANTINE_KEY

_BATCH = 200


def iter_collection(chroma_store, collection_name: str,
                    max_docs: int = 20000) -> Iterator[Dict[str, Any]]:
    """Yield {id, document, metadata} for every doc, paginated — never a
    full-collection load in one call (16GB-box doctrine)."""
    coll = chroma_store._get_collection(collection_name)
    if coll is None:
        return
    offset = 0
    seen = 0
    while seen < max_docs:
        res = coll.get(limit=_BATCH, offset=offset,
                       include=["documents", "metadatas"])
        ids = res.get("ids") or []
        if not ids:
            return
        docs = res.get("documents") or [None] * len(ids)
        metas = res.get("metadatas") or [{}] * len(ids)
        for i, doc_id in enumerate(ids):
            yield {
                "id": doc_id,
                "document": docs[i],
                "metadata": metas[i] or {},
            }
            seen += 1
            if seen >= max_docs:
                return
        offset += len(ids)


def already_quarantined(meta: Dict[str, Any]) -> bool:
    return bool(meta.get(QUARANTINE_KEY))
