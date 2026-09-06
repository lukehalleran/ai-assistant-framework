"""Store adapters: apply / revert one ItemChange.

Every apply captures the pre-image of the exact fields it overwrites into
``change.before`` BEFORE writing, so revert() is always possible. Adapters
support only reversible operations — there is deliberately NO delete adapter;
deletion never happens through the curation engine (quarantine flips a
metadata flag the retrieval filters respect; emptying quarantine is a
separate, explicit human surface).

Stores addressed:
- "chroma:<collection>"  — metadata set / content replace / quarantine flag
- "profile"              — supersede a profile fact (is_current=False)
"""

from typing import Any, Dict

from memory.curation.types import ItemChange
from utils.logging_utils import get_logger

logger = get_logger("curation_adapters")

# Metadata key the retrieval filters check. A quarantined doc stays on disk,
# fully restorable, but never surfaces in a prompt.
QUARANTINE_KEY = "curation_quarantined"
QUARANTINE_REASON_KEY = "curation_quarantine_reason"


class AdapterError(RuntimeError):
    pass


def _chroma_collection(chroma_store, store_ref: str):
    name = store_ref.split(":", 1)[1]
    coll = chroma_store._get_collection(name)
    if coll is None:
        raise AdapterError(f"collection {name!r} unavailable")
    return coll


def _get_chroma_doc(coll, doc_id: str) -> Dict[str, Any]:
    res = coll.get(ids=[doc_id], include=["documents", "metadatas"])
    if not res or not res.get("ids") or not res["ids"]:
        raise AdapterError(f"doc {doc_id!r} not found")
    return {
        "document": (res.get("documents") or [None])[0],
        "metadata": (res.get("metadatas") or [{}])[0] or {},
    }


def _graph_node_attrs(graph_memory, entity_id: str):
    """The live networkx attribute dict for a node (metadata lives at
    attrs["metadata"]); None when the node does not exist."""
    eid = (entity_id or "").lower().strip()
    g = getattr(graph_memory, "graph", None)
    if g is None or not g.has_node(eid):
        return None
    return g.nodes[eid]


def apply_change(change: ItemChange, *, chroma_store=None, user_profile=None,
                 graph_memory=None) -> None:
    """Apply one change, filling change.before with the pre-image first."""
    if change.store.startswith("chroma:"):
        if chroma_store is None:
            raise AdapterError("chroma store not provided")
        coll = _chroma_collection(chroma_store, change.store)
        current = _get_chroma_doc(coll, change.doc_id)

        if change.change_type in ("set_metadata", "quarantine"):
            updates = dict(change.after)
            if change.change_type == "quarantine":
                updates.setdefault(QUARANTINE_KEY, True)
            # Pre-image: only the keys we touch (None marks "was absent").
            change.before = {
                k: current["metadata"].get(k) for k in updates
            }
            merged = {**current["metadata"], **updates}
            coll.update(ids=[change.doc_id], metadatas=[merged])
        elif change.change_type == "replace_content":
            new_doc = change.after.get("document")
            if not isinstance(new_doc, str) or not new_doc.strip():
                # Prime directive: a repair may never blank a document.
                raise AdapterError("replace_content with empty document refused")
            change.before = {"document": current["document"]}
            coll.update(ids=[change.doc_id], documents=[new_doc])
        else:
            raise AdapterError(f"unsupported chroma change_type {change.change_type!r}")
        return

    if change.store == "profile":
        if user_profile is None:
            raise AdapterError("user profile not provided")
        if change.change_type != "supersede_profile_fact":
            raise AdapterError(f"unsupported profile change_type {change.change_type!r}")
        fact = _find_profile_fact(user_profile, change.doc_id)
        if fact is None:
            raise AdapterError(f"profile fact {change.doc_id!r} not found")
        change.before = {
            "is_current": fact.get("is_current", True),
            "curation_stale_reason": fact.get("curation_stale_reason"),
        }
        fact["is_current"] = False
        if change.after.get("reason"):
            fact["curation_stale_reason"] = change.after["reason"]
        user_profile.save()
        return

    if change.store == "graph":
        # Node quarantine (2026-09-05): reversible metadata flag on a graph
        # node; GraphMemory.edge_is_suppressed drops every edge touching a
        # quarantined node at read time. The node and its edges stay on disk
        # — deletion remains the owner's terminal-only hygiene step.
        if graph_memory is None:
            raise AdapterError("graph memory not provided")
        if change.change_type != "quarantine_node":
            raise AdapterError(f"unsupported graph change_type {change.change_type!r}")
        attrs = _graph_node_attrs(graph_memory, change.doc_id)
        if attrs is None:
            raise AdapterError(f"graph node {change.doc_id!r} not found")
        meta = attrs.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            attrs["metadata"] = meta
        updates = {QUARANTINE_KEY: True, **{k: v for k, v in change.after.items()}}
        change.before = {k: meta.get(k) for k in updates}
        meta.update(updates)
        graph_memory._mark_dirty()
        graph_memory.save()
        return

    raise AdapterError(f"unknown store {change.store!r}")


def revert_change(change: ItemChange, *, chroma_store=None, user_profile=None,
                  graph_memory=None) -> None:
    """Restore the pre-image captured at apply time."""
    if change.store.startswith("chroma:"):
        if chroma_store is None:
            raise AdapterError("chroma store not provided")
        coll = _chroma_collection(chroma_store, change.store)
        current = _get_chroma_doc(coll, change.doc_id)
        if change.change_type in ("set_metadata", "quarantine"):
            restored = dict(current["metadata"])
            for k, v in change.before.items():
                if v is None:
                    restored.pop(k, None)
                else:
                    restored[k] = v
            coll.update(ids=[change.doc_id], metadatas=[restored])
        elif change.change_type == "replace_content":
            old_doc = change.before.get("document")
            if isinstance(old_doc, str) and old_doc:
                coll.update(ids=[change.doc_id], documents=[old_doc])
        return

    if change.store == "profile":
        if user_profile is None:
            raise AdapterError("user profile not provided")
        fact = _find_profile_fact(user_profile, change.doc_id)
        if fact is None:
            raise AdapterError(f"profile fact {change.doc_id!r} not found")
        fact["is_current"] = change.before.get("is_current", True)
        prev_reason = change.before.get("curation_stale_reason")
        if prev_reason is None:
            fact.pop("curation_stale_reason", None)
        else:
            fact["curation_stale_reason"] = prev_reason
        user_profile.save()
        return

    if change.store == "graph":
        if graph_memory is None:
            raise AdapterError("graph memory not provided")
        attrs = _graph_node_attrs(graph_memory, change.doc_id)
        if attrs is None:
            raise AdapterError(f"graph node {change.doc_id!r} not found")
        meta = attrs.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            attrs["metadata"] = meta
        for k, v in change.before.items():
            if v is None:
                meta.pop(k, None)
            else:
                meta[k] = v
        graph_memory._mark_dirty()
        graph_memory.save()
        return

    raise AdapterError(f"unknown store {change.store!r}")


def _find_profile_fact(user_profile, fact_id: str):
    """Locate a fact dict by fact_id across all profile categories."""
    cats = (getattr(user_profile, "profile", None) or {}).get("categories", {})
    for facts_list in cats.values():
        if not isinstance(facts_list, list):
            continue
        for fact in facts_list:
            if isinstance(fact, dict) and fact.get("fact_id") == fact_id:
                return fact
    return None
