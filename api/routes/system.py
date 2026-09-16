"""System routes: GET /api/status (memory stats) and GET /api/graph (knowledge graph JSON)."""

import json
import os
import threading
import uuid

from fastapi import APIRouter, HTTPException, Request

import knowledge.obsidian_manager as obsidian_manager
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status")
async def status(request: Request):
    orchestrator = request.app.state.daemon.orchestrator
    out = {}
    try:
        cm = orchestrator.memory_system.corpus_manager
        corpus = getattr(cm, "corpus", []) or []
        out["total_entries"] = len(corpus)
    except Exception as e:
        logger.debug(f"[API] status corpus read failed: {e}")
    try:
        mm = orchestrator.model_manager
        out["active_model"] = mm.get_active_model_name()
    except Exception:
        pass
    try:
        graph = getattr(orchestrator.memory_system, "graph_memory", None)
        if graph is not None and hasattr(graph, "edge_count"):
            # Relation-level counts (2026-09-03): the nx DiGraph holds one
            # edge per node PAIR, so number_of_edges() under-reported
            # multi-relation pairs (836 vs 982 live).
            out["graph_nodes"] = int(graph.node_count())
            out["graph_edges"] = int(graph.edge_count())
        elif graph is not None and hasattr(graph, "graph"):
            out["graph_nodes"] = int(graph.graph.number_of_nodes())
            out["graph_edges"] = int(graph.graph.number_of_edges())
    except Exception:
        pass
    return out


def _sync_message(result) -> str:
    if result.errors:
        return f"⚠️ Sync completed with errors: {', '.join(result.errors)}"
    if result.embedded_files == 0 and result.updated_files == 0 and result.skipped_files > 0:
        return f"✓ All {result.skipped_files} notes unchanged"
    parts = []
    if result.embedded_files:
        parts.append(f"{result.embedded_files} new")
    if result.updated_files:
        parts.append(f"{result.updated_files} updated")
    return (f"✅ Synced {', '.join(parts)} notes ({result.total_chunks} chunks) "
            f"in {result.duration_seconds:.1f}s. Skipped {result.skipped_files} unchanged.")


def _sync_result_payload(result) -> dict:
    return {
        "embedded_files": result.embedded_files,
        "updated_files": result.updated_files,
        "skipped_files": result.skipped_files,
        "processed_files": result.processed_files,
        "total_files": result.total_files,
        "total_chunks": result.total_chunks,
        "errors": list(result.errors),
        "duration_seconds": result.duration_seconds,
    }


def _run_notes_sync(daemon, task_id: str) -> None:
    """Worker body: embed the vault with the LIVE store, then record the outcome.

    Runs on its own thread so the outcome survives the requesting client
    (BC-80). Uses the orchestrator's Chroma store — a bare ObsidianManager()
    lazily built a second store + embedder inside the running process (BC-81).
    """
    state = daemon.notes_sync
    try:
        store = daemon.orchestrator.memory_system.chroma_store
        if store is None:
            raise RuntimeError("live Chroma store is unavailable")
        manager = obsidian_manager.ObsidianManager(chroma_store=store)
        result = manager.embed_vault(force_reindex=False)
        status = "failed" if result.errors else "succeeded"
        state.finish(
            task_id,
            status,
            _sync_message(result),
            error="; ".join(result.errors) if result.errors else None,
            result=_sync_result_payload(result),
        )
    except Exception as exc:
        logger.error(f"[API] notes sync failed: {exc}")
        state.finish(task_id, "failed", f"❌ Sync failed: {exc}", error=str(exc))


@router.post("/sync-notes")
async def sync_notes(request: Request):
    """Start the vault sync as a retained background job and return at once.

    Single-flight: a second tap while one runs reports "already running" with
    the running job's task_id. The outcome is read from GET /sync-notes/status.
    The response keeps the ``message`` key the SPA and Gradio paths rely on.
    """
    daemon = request.app.state.daemon
    state = daemon.notes_sync
    task_id = uuid.uuid4().hex
    if not state.try_start(task_id):
        current = state.snapshot()
        return {
            "message": "Notes sync already running",
            "status": "running",
            "task_id": current["task_id"],
        }
    worker = threading.Thread(
        target=_run_notes_sync, args=(daemon, task_id), name="NotesSync", daemon=True,
    )
    state.worker = worker
    try:
        worker.start()
    except Exception as exc:
        # A job that never started must not stay "running" forever.
        logger.error(f"[API] notes sync worker failed to start: {exc}")
        state.finish(task_id, "failed", f"❌ Sync failed to start: {exc}", error=str(exc))
        current = state.snapshot()
        return {"message": current["message"], "status": "failed", "task_id": task_id}
    return {"message": "Notes sync started", "status": "running", "task_id": task_id}


@router.get("/sync-notes/status")
async def sync_notes_status(request: Request):
    """Current or retained notes-sync outcome (survives a dropped POST response)."""
    return request.app.state.daemon.notes_sync.snapshot()


@router.get("/graph")
async def graph(request: Request, limit: int = 300):
    """Read-only knowledge-graph payload for the (stretch) graph view.

    Trims to the top-`limit` nodes by degree so the client render stays fast.

    F09 (2026-09-09, docs/HANDOFF_20260909_independent_bug_audit.md): the
    on-disk schema written by GraphMemory.save() (memory/graph_memory.py) is
    `nodes` as an id -> attributes DICT and edges carrying `source_id`/
    `target_id` (the relation-level edge index — a node pair can have more
    than one edge, one per relation). This route promises its consumers a
    node LIST of `{"id": ..., ...}` dicts and edges with `source`/`target`
    keys; convert at this boundary instead of assuming the writer's shape,
    which used to crash (`'str' object has no attribute 'get'`) or silently
    return the wrong shape once the node count exceeded `limit`.
    """
    from config.app_config import KNOWLEDGE_GRAPH_PERSIST_PATH  # lazy import: live-config

    path = KNOWLEDGE_GRAPH_PERSIST_PATH
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Knowledge graph file not found.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_nodes = data.get("nodes", {})
    raw_edges = data.get("edges", [])

    # GraphMemory.save() always writes an id -> attrs dict; tolerate an
    # already-list shape defensively (e.g. a hand-built fixture) rather than
    # assuming one or the other.
    if isinstance(raw_nodes, dict):
        nodes = [{"id": node_id, **(attrs or {})} for node_id, attrs in raw_nodes.items()]
    else:
        nodes = [n for n in raw_nodes if isinstance(n, dict) and n.get("id") is not None]

    edges = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source_id", e.get("source"))
        tgt = e.get("target_id", e.get("target"))
        if src is None or tgt is None:
            continue
        edges.append({
            "source": src,
            "target": tgt,
            "relation": e.get("relation"),
            "weight": e.get("weight", 1.0),
            "truth_score": e.get("truth_score"),
            "metadata": e.get("metadata", {}),
        })

    if limit and len(nodes) > limit:
        degree: dict[str, int] = {}
        for e in edges:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        nodes = sorted(nodes, key=lambda n: degree.get(n.get("id"), 0), reverse=True)[:limit]
        keep = {n.get("id") for n in nodes}
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]

    return {"nodes": nodes, "edges": edges}
