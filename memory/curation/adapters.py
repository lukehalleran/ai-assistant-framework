"""Reversible store adapters with preflight capture and conflict-aware undo.

Chroma 1.0.7 merges metadata and rejects None. A previously absent quarantine
flag is restored semantically as False, and an absent quarantine reason as "".
Other newly introduced metadata keys require an explicit neutral policy;
unsupported changes are rejected before writing. Existing values are restored
exactly. No adapter deletes/reinserts documents. B3's mutation notifications
remain on both successful Chroma apply and revert.
"""
from typing import Any, Dict

from memory.curation.types import ItemChange
from memory.memory_expander import notify_chroma_mutation

QUARANTINE_KEY = "curation_quarantined"
QUARANTINE_REASON_KEY = "curation_quarantine_reason"
_METADATA_NEUTRALS = {QUARANTINE_KEY: False, QUARANTINE_REASON_KEY: ""}


class AdapterError(ValueError):
    pass


def _chroma_collection(chroma_store, store_ref: str):
    if chroma_store is None:
        raise AdapterError("chroma store not provided")
    name = store_ref.split(":", 1)[1]
    coll = chroma_store._get_collection(name)
    if coll is None:
        raise AdapterError(f"collection {name!r} unavailable")
    return coll


def _get_chroma_doc(coll, doc_id: str) -> Dict[str, Any]:
    res = coll.get(ids=[doc_id], include=["documents", "metadatas"])
    if not res or not res.get("ids"):
        raise AdapterError(f"doc {doc_id!r} not found")
    return {
        "document": (res.get("documents") or [None])[0],
        "metadata": (res.get("metadatas") or [{}])[0] or {},
    }


def _graph_node_attrs(graph_memory, entity_id: str):
    eid = (entity_id or "").lower().strip()
    graph = getattr(graph_memory, "graph", None)
    if graph is None or not graph.has_node(eid):
        raise AdapterError(f"graph node {entity_id!r} not found")
    return graph.nodes[eid]


def _find_profile_fact(user_profile, fact_id: str):
    cats = (getattr(user_profile, "profile", None) or {}).get("categories", {})
    for facts in cats.values():
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict) and fact.get("fact_id") == fact_id:
                    return fact
    raise AdapterError(f"profile fact {fact_id!r} not found")


def _updates(change):
    if change.store == "profile":
        if change.change_type != "supersede_profile_fact":
            raise AdapterError(f"unsupported profile change_type {change.change_type!r}")
        updates = {"is_current": False}
        if change.after.get("reason"):
            updates["curation_stale_reason"] = change.after["reason"]
        return updates
    updates = dict(change.after)
    if change.change_type in ("quarantine", "quarantine_node"):
        updates.setdefault(QUARANTINE_KEY, True)
    return updates


def _current_fields(change, *, chroma_store=None, user_profile=None, graph_memory=None):
    if change.store.startswith("chroma:"):
        if change.change_type not in ("set_metadata", "quarantine", "replace_content"):
            raise AdapterError(f"unsupported chroma change_type {change.change_type!r}")
        current = _get_chroma_doc(_chroma_collection(chroma_store, change.store), change.doc_id)
        return current if change.change_type == "replace_content" else current["metadata"]
    if change.store == "profile":
        return _find_profile_fact(user_profile, change.doc_id)
    if change.store == "graph":
        if change.change_type != "quarantine_node":
            raise AdapterError(f"unsupported graph change_type {change.change_type!r}")
        meta = _graph_node_attrs(graph_memory, change.doc_id).get("metadata")
        return meta if isinstance(meta, dict) else {}
    raise AdapterError(f"unknown store {change.store!r}")


def prepare_change(change: ItemChange, **stores) -> None:
    """Capture and validate all touched fields without changing the target."""
    updates = _updates(change)
    current = _current_fields(change, **stores)
    if not updates:
        raise AdapterError("empty change is not reversible")
    if change.change_type == "replace_content":
        if set(updates) != {"document"}:
            raise AdapterError("replace_content requires only a document field")
        if any(not isinstance(value, str) or not value.strip()
               for value in (updates["document"], current.get("document"))):
            raise AdapterError("replace_content requires nonempty content and pre-image")
    elif change.store.startswith("chroma:"):
        for key, value in updates.items():
            if not isinstance(value, (str, int, float, bool)):
                raise AdapterError(f"unsupported metadata value for {key!r}")
            if current.get(key) is None and key not in _METADATA_NEUTRALS:
                raise AdapterError(f"no reversible neutral value for metadata key {key!r}")
    change.before = {key: current.get(key) for key in updates}
    change.missing_before = [key for key in updates if key not in current]


def apply_change(change: ItemChange, *, chroma_store=None, user_profile=None,
                 graph_memory=None, prepared: bool = False) -> None:
    stores = dict(chroma_store=chroma_store, user_profile=user_profile, graph_memory=graph_memory)
    if not prepared:
        prepare_change(change, **stores)
    else:
        # Do not replace the durable pre-image with a new, unjournaled one.
        check = change.model_copy(deep=True)
        prepare_change(check, **stores)
        if check.before != change.before or check.missing_before != change.missing_before:
            raise AdapterError(f"target changed since preparation: {change.doc_id}")

    updates = _updates(change)
    if change.store.startswith("chroma:"):
        coll = _chroma_collection(chroma_store, change.store)
        if change.change_type == "replace_content":
            coll.update(ids=[change.doc_id], documents=[updates["document"]])
        else:
            coll.update(ids=[change.doc_id], metadatas=[updates])
        notify_chroma_mutation(change.doc_id)
    elif change.store == "profile":
        _find_profile_fact(user_profile, change.doc_id).update(updates)
        user_profile.save(raise_on_error=True)
    elif change.store == "graph":
        attrs = _graph_node_attrs(graph_memory, change.doc_id)
        if not isinstance(attrs.get("metadata"), dict):
            attrs["metadata"] = {}
        attrs["metadata"].update(updates)
        graph_memory._mark_dirty()
        graph_memory.save(raise_on_error=True)


def revert_change(change: ItemChange, *, chroma_store=None, user_profile=None,
                  graph_memory=None) -> None:
    """Restore touched fields, refusing to overwrite a later conflicting edit.

    Undo is idempotent: an interrupted rollback can safely be retried. Old
    journal entries use None to indicate absence; new ones record missing keys
    separately so JSON stores can also restore an explicitly stored null.
    """
    current = _current_fields(change, chroma_store=chroma_store,
                              user_profile=user_profile, graph_memory=graph_memory)
    updates = _updates(change)
    if not change.before or not set(updates).issubset(change.before):
        raise AdapterError("missing recovery pre-image")
    missing = (change.missing_before if change.missing_before is not None
               else [key for key, value in change.before.items() if value is None])
    chroma_metadata = change.store.startswith("chroma:") and change.change_type != "replace_content"
    restored = {}
    for key, before in change.before.items():
        value = before
        if chroma_metadata and key in missing:
            if key not in _METADATA_NEUTRALS:
                raise AdapterError(f"no reversible neutral value for metadata key {key!r}")
            value = _METADATA_NEUTRALS[key]
        actual = current.get(key)
        if actual != before and actual != value and actual != updates.get(key):
            raise AdapterError(f"undo conflicts with a later edit: {change.doc_id}/{key}")
        restored[key] = value

    if change.store.startswith("chroma:"):
        coll = _chroma_collection(chroma_store, change.store)
        if change.change_type == "replace_content":
            old_doc = restored.get("document")
            if not isinstance(old_doc, str) or not old_doc.strip():
                raise AdapterError("nonempty content pre-image required")
            coll.update(ids=[change.doc_id], documents=[old_doc])
        else:
            coll.update(ids=[change.doc_id], metadatas=[restored])
        notify_chroma_mutation(change.doc_id)
        return

    target = current
    if change.store == "graph":
        attrs = _graph_node_attrs(graph_memory, change.doc_id)
        if not isinstance(attrs.get("metadata"), dict):
            attrs["metadata"] = target
    for key, value in restored.items():
        if key in missing:
            target.pop(key, None)
        else:
            target[key] = value
    if change.store == "profile":
        user_profile.save(raise_on_error=True)
    else:
        graph_memory._mark_dirty()
        graph_memory.save(raise_on_error=True)
