"""ObsidianManager.get_notes semantic query runs off the event loop.

Incident 2026-10-18, class BC-41 (sync work on the event loop): live
[LoopStall] 762 ms at query_collection via obsidian_manager.get_notes on turn 1.
The recording seam notes which thread ran the sync store call; clean and
indented/wrapped query forms are both asserted (fixtures are synthetic).
"""
import threading
from unittest.mock import MagicMock

import pytest

from knowledge.obsidian_manager import ObsidianManager

QUERIES = [
    "what did Rowan write about the quasar planning",
    "    what did Rowan write about\n    the quasar planning\n",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", QUERIES)
async def test_semantic_query_runs_off_loop(query):
    seen = {}

    def fake_query(name, text, n_results=3, **kw):
        seen["t"] = threading.get_ident()
        return [{"content": "Quasar notes", "metadata": {"title": "Q"}, "relevance_score": 0.8}]

    mgr = ObsidianManager(chroma_store=MagicMock(), vault_path="/nonexistent")
    mgr.chroma_store.query_collection = fake_query
    mgr._keyword_search = lambda q, limit=10: []
    loop_thread = threading.get_ident()
    out = await mgr.get_notes(query, limit=6)
    assert [n["metadata"]["title"] for n in out] == ["Q"]
    assert seen["t"] != loop_thread
