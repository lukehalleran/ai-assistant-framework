"""System routes: GET /api/status (memory stats) and GET /api/graph (knowledge graph JSON)."""

import json
import os

from fastapi import APIRouter, HTTPException, Request

from utils.logging_utils import get_logger
import asyncio

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
        if graph is not None and hasattr(graph, "graph"):
            out["graph_nodes"] = int(graph.graph.number_of_nodes())
            out["graph_edges"] = int(graph.graph.number_of_edges())
    except Exception:
        pass
    return out


@router.post("/sync-notes")
async def sync_notes(request: Request):
    """Embed the Obsidian vault into ChromaDB (same helper as the Gradio button).

    embed_vault is synchronous and can take a while on big vaults — run it in a
    worker thread so the event loop (and any in-flight chat stream) stays live.
    """

    def _sync() -> str:
        try:
            from knowledge.obsidian_manager import ObsidianManager
            manager = ObsidianManager()
            result = manager.embed_vault(force_reindex=False)

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
        except Exception as e:
            return f"❌ Sync failed: {e}"

    message = await asyncio.to_thread(_sync)
    return {"message": message}


@router.get("/graph")
async def graph(request: Request, limit: int = 300):
    """Read-only knowledge-graph payload for the (stretch) graph view.

    Trims to the top-`limit` nodes by degree so the client render stays fast.
    """
    from config.app_config import KNOWLEDGE_GRAPH_PERSIST_PATH

    path = KNOWLEDGE_GRAPH_PERSIST_PATH
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Knowledge graph file not found.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if limit and len(nodes) > limit:
        degree = {}
        for e in edges:
            degree[e.get("source")] = degree.get(e.get("source"), 0) + 1
            degree[e.get("target")] = degree.get(e.get("target"), 0) + 1
        nodes = sorted(nodes, key=lambda n: degree.get(n.get("id"), 0), reverse=True)[:limit]
        keep = {n.get("id") for n in nodes}
        edges = [e for e in edges if e.get("source") in keep and e.get("target") in keep]

    return {"nodes": nodes, "edges": edges}
