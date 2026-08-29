"""Curation Center API routes (api/routes/curation.py)."""

import httpx
import pytest

from api.app import create_app
from memory.curation.engine import CurationEngine, StoreBundle, new_proposal_id
from memory.curation.journal import CurationJournal
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
)
from tests.unit.helpers_orchestrator import _make_orchestrator
from tests.unit.test_curation_engine import (
    FakeChromaStore,
    FakeCollection,
    StubCurator,
)


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t")


def _install_engine(tmp_path, monkeypatch, proposals=None):
    import memory.curation.service as service

    store = FakeChromaStore({"conversations": FakeCollection(
        {"d1": {"document": "x", "metadata": {}}})})
    engine = CurationEngine(
        StoreBundle(chroma_store=store),
        queue_path=str(tmp_path / "q.json"),
        journal=CurationJournal(str(tmp_path / "a.jsonl")),
    )
    if proposals:
        engine.register(StubCurator(proposals))
        engine.run_scan()
    monkeypatch.setattr(service, "_engine", engine)
    return engine, store


def _proposal(doc_id="d1"):
    return CurationProposal(
        proposal_id=new_proposal_id(), curator="test",
        instrument=Instrument.METADATA, confidence=Confidence.DETERMINISTIC,
        title="Quarantine test doc", evidence="deployed predicate",
        items=[ItemChange(store="chroma:conversations", doc_id=doc_id,
                          change_type="quarantine")],
    )


class TestCurationRoutes:

    @pytest.mark.asyncio
    async def test_queue_lists_pending(self, tmp_path, monkeypatch):
        _install_engine(tmp_path, monkeypatch, [_proposal()])
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/curation/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["max_mode"] == "queue"
        assert len(body["proposals"]) == 1
        assert body["proposals"][0]["title"] == "Quarantine test doc"

    @pytest.mark.asyncio
    async def test_apply_then_undo_roundtrip(self, tmp_path, monkeypatch):
        engine, store = _install_engine(tmp_path, monkeypatch, [_proposal()])
        pid = engine.pending()[0].proposal_id
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post(f"/api/curation/{pid}/apply")
            assert resp.status_code == 200
            assert store.collections["conversations"].docs["d1"]["metadata"][
                "curation_quarantined"]
            resp = await client.post(f"/api/curation/{pid}/undo")
            assert resp.status_code == 200
            assert "curation_quarantined" not in \
                store.collections["conversations"].docs["d1"]["metadata"]

    @pytest.mark.asyncio
    async def test_dismiss_and_conflict_states(self, tmp_path, monkeypatch):
        engine, _ = _install_engine(tmp_path, monkeypatch, [_proposal()])
        pid = engine.pending()[0].proposal_id
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post(f"/api/curation/{pid}/dismiss",
                                     json={"reason": "not junk"})
            assert resp.status_code == 200
            # Dismissed proposal can't be applied.
            resp = await client.post(f"/api/curation/{pid}/apply")
            assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_unknown_proposal_404(self, tmp_path, monkeypatch):
        _install_engine(tmp_path, monkeypatch)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post("/api/curation/cur_nope/apply")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_activity_returns_journal(self, tmp_path, monkeypatch):
        engine, _ = _install_engine(tmp_path, monkeypatch, [_proposal()])
        engine.apply(engine.pending()[0].proposal_id)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/curation/activity")
        assert resp.status_code == 200
        events = [e["event"] for e in resp.json()["events"]]
        assert "applied" in events and "scan_finished" in events
