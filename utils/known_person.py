"""Known-person test shared by prompt gathering and the personal-claim check.

Moved out of ``core/prompt/gatherer_knowledge`` (2026-10-21, class: BC-84) so
``core/personal_claim_check`` can use it without importing the prompt package.
The live knowledge graph is registered once by the memory coordinator
(``register_graph``); callers that hold a graph pass it explicitly.
"""
from __future__ import annotations

from typing import Any, Optional

_REGISTERED_GRAPH: Any = None


def register_graph(graph_memory: Any) -> None:
    """Remember the live graph for callers that have no handle on it."""
    global _REGISTERED_GRAPH
    _REGISTERED_GRAPH = graph_memory


def is_known_person(graph_memory: Optional[Any], name: str) -> bool:
    """True when ``name`` is a KNOWN PERSON: a knowledge-graph entity of type
    person (alias-resolved, read-only) or a name already in the Google
    Contacts cache (cache-only, never an API call). ``graph_memory=None`` uses
    the registered graph. Tries the whole phrase, then each word of it.
    Unknown -> False."""
    if not name or not name.strip():
        return False
    if graph_memory is None:
        graph_memory = _REGISTERED_GRAPH
    candidates = [name.strip()] + [w for w in name.split() if len(w) >= 3]
    if graph_memory is not None:
        for cand in candidates:
            try:
                eid = graph_memory.resolve_entity(cand)
                if not isinstance(eid, str) or not eid.strip():
                    continue
                node = graph_memory.get_entity(eid)
                et = getattr(node, "entity_type", None)
                if isinstance(et, str) and et.strip().lower() == "person":
                    return True
            except Exception:  # degrades: graph lookup failure -> try the contacts cache, may under-fire
                continue
    try:
        from core.actions.google_contacts import is_cached_contact_name  # lazy import: layering (utils must not import core at module level)
        return bool(is_cached_contact_name(name))
    except Exception:  # degrades: contacts cache unreadable -> name not known
        return False
