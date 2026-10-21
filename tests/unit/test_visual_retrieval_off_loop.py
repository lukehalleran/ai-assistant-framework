"""VisualRetriever.retrieve_visual_memories must not run its sync CLIP /
store / image-load seams on the event loop thread.

Incident 2026-10-17 (lane UPL finding): the async method called
_search_clip, search_by_text, get_by_entity and image loading directly.
class: BC-41 (blocking call on the loop).
"""

import asyncio
import threading

from knowledge.visual_retrieval import VisualRetriever


class _Clip:
    def __init__(self, rec):
        self.rec = rec

    def encode_text(self, q):
        self.rec["encode_text"] = threading.get_ident()
        return [0.0]


class _Store:
    def __init__(self, rec):
        self.rec = rec

    def search_by_clip(self, emb, k=5):
        self.rec["search_by_clip"] = threading.get_ident()
        return [{"image_path": "/x/a.jpg", "score": 0.5, "entity_ids": ["rowan"]}]

    def search_by_text(self, q, k=5):
        self.rec["search_by_text"] = threading.get_ident()
        return []

    def get_by_entity(self, ents):
        self.rec["get_by_entity"] = threading.get_ident()
        return [{"image_path": "/x/b.jpg", "score": 0.4, "entity_ids": ["tamsin"]}]


def test_sync_seams_run_off_the_loop(monkeypatch):
    rec = {}
    r = VisualRetriever(_Clip(rec), _Store(rec))

    def fake_load(self, result):
        rec["load"] = threading.get_ident()
        return None

    monkeypatch.setattr(VisualRetriever, "_load_image_for_multimodal", fake_load)

    async def run():
        loop_tid = threading.get_ident()
        await r.retrieve_visual_memories("tamsin", target_entities={"tamsin"})
        return loop_tid

    loop_tid = asyncio.run(run())
    assert set(rec) == {
        "encode_text", "search_by_clip", "search_by_text", "get_by_entity", "load",
    }
    for name, tid in rec.items():
        assert tid != loop_tid, f"{name} ran on the loop thread"
