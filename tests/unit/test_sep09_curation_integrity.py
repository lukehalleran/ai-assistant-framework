"""B2: real-store undo, write-ahead recovery, and bounded curation overlap."""
import json
import asyncio
import threading
from pathlib import Path

import pytest

import memory.curation.engine as engine_module
from memory.curation.engine import CurationEngine, StoreBundle
from memory.curation.journal import CurationJournal
from memory.curation.types import CurationProposal, ItemChange
from memory.user_profile import UserProfile


def proposal(*items):
    return CurationProposal(proposal_id="synthetic-proposal", curator="synthetic",
                            instrument="metadata", confidence="deterministic",
                            title="Synthetic curation", evidence="Synthetic fixture",
                            items=list(items))


def engine_at(tmp_path, stores):
    return CurationEngine(stores, queue_path=str(tmp_path / "queue.json"),
                          journal=CurationJournal(str(tmp_path / "journal.jsonl")))


def profile_engine(tmp_path):
    profile = UserProfile(str(tmp_path / "profile.json"))
    profile.profile["categories"]["career"] = [
        {"fact_id": "synthetic-fact", "is_current": True}]
    profile.save()
    engine = engine_at(tmp_path, StoreBundle(user_profile=profile))
    p = proposal(ItemChange(store="profile", doc_id="synthetic-fact",
                            change_type="supersede_profile_fact",
                            after={"reason": "synthetic"}))
    engine._proposals[p.proposal_id] = p
    return engine, profile, p


def persisted_current(profile):
    return json.loads(Path(profile.profile_path).read_text())["categories"]["career"][0]["is_current"]


@pytest.fixture
def real_chroma():
    import chromadb
    from chromadb.config import Settings
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    import uuid
    collection = client.create_collection("curation_" + uuid.uuid4().hex,
                                          embedding_function=None)
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"conversations": collection}
    yield store, collection
    client.delete_collection(collection.name)


@pytest.mark.parametrize("before", [None, False, True])
def test_real_driver_undo_restores_quarantine_semantics(tmp_path, real_chroma, before, monkeypatch):
    import memory.curation.adapters as adapters
    store, collection = real_chroma
    metadata = {"source": "synthetic", "unrelated": 17}
    if before is not None:
        metadata.update(curation_quarantined=before, curation_quarantine_reason="prior")
    collection.add(ids=["synthetic"], documents=["Synthetic durable memory"],
                   embeddings=[[1.0, 0.0]], metadatas=[metadata])
    observed = []
    notify = adapters.notify_chroma_mutation

    def recording_notify(doc_id):
        observed.append(doc_id)
        notify(doc_id)

    monkeypatch.setattr(adapters, "notify_chroma_mutation", recording_notify)
    engine = engine_at(tmp_path, StoreBundle(chroma_store=store))
    p = proposal(ItemChange(store="chroma:conversations", doc_id="synthetic",
                            change_type="quarantine", after={"curation_quarantine_reason": "new"}))
    engine._proposals[p.proposal_id] = p
    engine.apply(p.proposal_id)
    engine.undo(p.proposal_id)
    restored = collection.get(ids=["synthetic"], include=["metadatas"])["metadatas"][0]
    assert restored["curation_quarantined"] is (False if before is None else before)
    assert restored["curation_quarantine_reason"] == ("" if before is None else "prior")
    assert restored["unrelated"] == 17 and restored["source"] == "synthetic"
    assert observed == ["synthetic", "synthetic"]  # retain B3's invalidation hooks


def test_unrestorable_metadata_is_refused_before_any_write(tmp_path, real_chroma):
    store, collection = real_chroma
    collection.add(ids=["synthetic"], documents=["Synthetic memory"],
                   embeddings=[[1.0, 0.0]], metadatas=[{"source": "synthetic"}])
    engine = engine_at(tmp_path, StoreBundle(chroma_store=store))
    p = proposal(ItemChange(store="chroma:conversations", doc_id="synthetic",
                            change_type="set_metadata", after={"new_arbitrary_key": "value"}))
    engine._proposals[p.proposal_id] = p
    with pytest.raises(ValueError, match="revers|neutral|restore"):
        engine.apply(p.proposal_id)
    assert collection.get(ids=["synthetic"], include=["metadatas"])["metadatas"] == [{"source": "synthetic"}]


@pytest.mark.parametrize("blocked", ["queue", "journal", "both"])
def test_unwritable_recovery_evidence_prevents_target_mutation(tmp_path, blocked):
    engine, profile, p = profile_engine(tmp_path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("synthetic")
    if blocked in ("queue", "both"):
        engine.queue_path = str(blocker / "queue.json")
    if blocked in ("journal", "both"):
        engine.journal.path = str(blocker / "journal.jsonl")
    with pytest.raises(OSError):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is True
    assert profile.profile["categories"]["career"][0]["is_current"] is True


def test_preimage_is_durable_before_the_target_save(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    save = profile.save
    observations = []

    def observe_save(**kwargs):
        events = engine.journal.tail()
        started = [e for e in events if e["event"] == "apply_started"]
        assert started and started[0]["items"][0]["before"]["is_current"] is True
        queued = json.loads(Path(engine.queue_path).read_text())["proposals"][0]
        assert queued["items"][0]["before"]["is_current"] is True
        observations.append(True)
        return save(**kwargs)

    monkeypatch.setattr(profile, "save", observe_save)
    engine.apply(p.proposal_id)
    assert observations and persisted_current(profile) is False


def test_batch_validates_later_targets_before_writing_the_first(tmp_path):
    engine, profile, p = profile_engine(tmp_path)
    p.items.append(ItemChange(store="profile", doc_id="missing-fact",
                              change_type="supersede_profile_fact"))
    with pytest.raises(ValueError, match="not found"):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is True
    assert not any(e["event"] == "apply_started" for e in engine.journal.tail())


class SimulatedCrash(BaseException):
    pass


def test_crash_after_target_write_is_recoverable_from_journal(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    apply = engine_module.apply_change

    def write_then_crash(*args, **kwargs):
        apply(*args, **kwargs)
        raise SimulatedCrash()

    monkeypatch.setattr(engine_module, "apply_change", write_then_crash)
    with pytest.raises(SimulatedCrash):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is False
    Path(engine.queue_path).write_text("{synthetic broken queue")
    reopened_profile = UserProfile(profile.profile_path)
    restarted = engine_at(tmp_path, StoreBundle(user_profile=reopened_profile))
    recovered = restarted.get(p.proposal_id)
    assert recovered is not None and recovered.status.value == "interrupted"
    assert p.proposal_id in [x.proposal_id for x in restarted.pending()]
    restarted.undo(p.proposal_id)
    assert persisted_current(profile) is True
    assert engine_at(tmp_path, restarted.stores).get(p.proposal_id).status.value == "undone"


def test_item_that_raises_after_writing_is_also_rolled_back(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    apply = engine_module.apply_change

    def write_then_fail(*args, **kwargs):
        apply(*args, **kwargs)
        raise OSError("synthetic post-write failure")

    monkeypatch.setattr(engine_module, "apply_change", write_then_fail)
    with pytest.raises(OSError, match="post-write"):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is True
    assert p.status.value == "failed"
    event = next(e for e in engine.journal.tail() if e["event"] == "apply_failed")
    assert event["rollback"] and all(row["restored"] for row in event["rollback"])


def test_rollback_failure_remains_undoable_after_restart(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    apply = engine_module.apply_change

    def write_then_fail(*args, **kwargs):
        apply(*args, **kwargs)
        raise OSError("synthetic target failure")

    def failed_rollback(*args, **kwargs):
        raise OSError("synthetic rollback failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(engine_module, "apply_change", write_then_fail)
        patcher.setattr(engine_module, "revert_change", failed_rollback)
        with pytest.raises(OSError):
            engine.apply(p.proposal_id)
    restarted = engine_at(tmp_path, engine.stores)
    assert restarted.get(p.proposal_id).status.value == "interrupted"
    restarted.undo(p.proposal_id)
    assert persisted_current(profile) is True


def test_completed_apply_survives_failed_queue_refresh(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    save = engine._save_queue
    saves = []

    def fail_refresh():
        saves.append(True)
        if len(saves) == 2:
            raise OSError("synthetic queue refresh failure")
        save()

    monkeypatch.setattr(engine, "_save_queue", fail_refresh)
    with pytest.raises(OSError, match="queue refresh"):
        engine.apply(p.proposal_id)
    restarted = engine_at(tmp_path, engine.stores)
    assert restarted.get(p.proposal_id).status.value == "applied"
    restarted.undo(p.proposal_id)
    assert persisted_current(profile) is True


def test_failed_commit_record_rolls_back_and_reopens_as_failed(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    record = engine.journal.record

    def fail_commit(event, **detail):
        if event == "applied":
            raise OSError("synthetic journal commit failure")
        record(event, **detail)

    monkeypatch.setattr(engine.journal, "record", fail_commit)
    with pytest.raises(OSError, match="journal commit"):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is True
    assert engine_at(tmp_path, engine.stores).get(p.proposal_id).status.value == "failed"


def test_strict_profile_save_exposes_target_write_failure(tmp_path, monkeypatch):
    import memory.user_profile as profile_module
    engine, profile, p = profile_engine(tmp_path)

    def fail_write(*args, **kwargs):
        raise OSError("synthetic profile write failure")

    monkeypatch.setattr(profile_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="profile write"):
        engine.apply(p.proposal_id)
    assert persisted_current(profile) is True
    # Both the failed target save and its failed rollback save are surfaced.
    assert p.status.value == "interrupted"


def test_graph_restores_null_and_missing_keys_after_reopening(tmp_path):
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphNode

    path = str(tmp_path / "graph.json")
    graph = GraphMemory(persist_path=path)
    graph.add_entity(GraphNode(entity_id="synthetic", display_name="Synthetic",
                               metadata={"curation_quarantine_reason": None, "unrelated": 17}))
    graph.save()
    engine = engine_at(tmp_path, StoreBundle(graph_memory=graph))
    p = proposal(ItemChange(store="graph", doc_id="synthetic", change_type="quarantine_node",
                            after={"curation_quarantine_reason": "synthetic"}))
    engine._proposals[p.proposal_id] = p
    engine.apply(p.proposal_id)
    reopened = GraphMemory(persist_path=path)
    assert reopened.get_entity("synthetic").metadata["curation_quarantined"] is True
    engine_at(tmp_path, StoreBundle(graph_memory=reopened)).undo(p.proposal_id)
    final = GraphMemory(persist_path=path)
    assert final.get_entity("synthetic").metadata == {
        "curation_quarantine_reason": None, "unrelated": 17}


def test_graph_save_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    import memory.graph_memory as graph_module
    from memory.graph_models import GraphNode

    graph = graph_module.GraphMemory(persist_path=str(tmp_path / "graph.json"))
    graph.add_entity(GraphNode(entity_id="synthetic", display_name="Synthetic"))
    graph.save()
    original = Path(graph.persist_path).read_bytes()
    engine = engine_at(tmp_path, StoreBundle(graph_memory=graph))
    p = proposal(ItemChange(store="graph", doc_id="synthetic", change_type="quarantine_node"))
    engine._proposals[p.proposal_id] = p

    def fail_write(*args, **kwargs):
        raise OSError("synthetic graph failure")

    monkeypatch.setattr(graph_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="graph failure"):
        engine.apply(p.proposal_id)
    assert Path(graph.persist_path).read_bytes() == original
    assert p.status.value == "interrupted" and graph._dirty


def test_undo_refuses_to_overwrite_a_later_edit(tmp_path, real_chroma):
    store, collection = real_chroma
    collection.add(ids=["synthetic"], documents=["Synthetic original"],
                   embeddings=[[1.0, 0.0]], metadatas=[{"source": "synthetic"}])
    engine = engine_at(tmp_path, StoreBundle(chroma_store=store))
    p = proposal(ItemChange(store="chroma:conversations", doc_id="synthetic",
                            change_type="quarantine", after={"curation_quarantine_reason": "first"}))
    engine._proposals[p.proposal_id] = p
    engine.apply(p.proposal_id)
    collection.update(ids=["synthetic"], metadatas=[{"curation_quarantine_reason": "later edit"}])
    with pytest.raises(ValueError, match="conflict"):
        engine.undo(p.proposal_id)
    metadata = collection.get(ids=["synthetic"], include=["metadatas"])["metadatas"][0]
    assert metadata["curation_quarantine_reason"] == "later edit"
    assert metadata["curation_quarantined"] is True
    assert p.status.value == "interrupted"


def test_fake_collection_matches_installed_merge_and_none_rules(real_chroma):
    from tests.unit.test_curation_engine import FakeCollection
    _, real = real_chroma
    real.add(ids=["synthetic"], documents=["Synthetic"], embeddings=[[1.0, 0.0]],
             metadatas=[{"source": "synthetic", "flag": True}])
    fake = FakeCollection({"synthetic": {"document": "Synthetic",
                                        "metadata": {"source": "synthetic", "flag": True}}})
    for collection in (real, fake):
        collection.update(ids=["synthetic"], metadatas=[{"flag": False}])
        assert collection.get(ids=["synthetic"], include=["metadatas"])["metadatas"][0] == {
            "source": "synthetic", "flag": False}
        with pytest.raises(ValueError):
            collection.update(ids=["synthetic"], metadatas=[{"flag": None}])


def test_torn_journal_append_does_not_hide_later_recovery(tmp_path):
    engine, profile, p = profile_engine(tmp_path)
    Path(engine.journal.path).write_text('{"event":"synthetic torn')
    engine.apply(p.proposal_id)
    Path(engine.queue_path).write_text("{synthetic corrupt queue")
    restarted = engine_at(tmp_path, engine.stores)
    assert restarted.get(p.proposal_id).status.value == "applied"
    restarted.undo(p.proposal_id)
    assert persisted_current(profile) is True


def test_crash_during_undo_can_be_retried(tmp_path, monkeypatch):
    engine, profile, p = profile_engine(tmp_path)
    engine.apply(p.proposal_id)
    revert = engine_module.revert_change

    def revert_then_crash(*args, **kwargs):
        revert(*args, **kwargs)
        raise SimulatedCrash()

    with monkeypatch.context() as patcher:
        patcher.setattr(engine_module, "revert_change", revert_then_crash)
        with pytest.raises(SimulatedCrash):
            engine.undo(p.proposal_id)
    assert persisted_current(profile) is True
    restarted = engine_at(tmp_path, engine.stores)
    assert restarted.get(p.proposal_id).status.value == "interrupted"
    restarted.undo(p.proposal_id)
    assert persisted_current(profile) is True


class BlockingCurator:
    name = "synthetic"

    def __init__(self, entered, release, items=()):
        self.entered, self.release, self.items = entered, release, items

    def sentinels(self, stores):
        return []

    def scan(self, stores):
        self.entered.set()
        assert self.release.wait(5), "test must release the scanner"
        return list(self.items)


def test_busy_engine_refuses_all_conflicting_mutations(tmp_path):
    engine, profile, p = profile_engine(tmp_path)
    entered, release = threading.Event(), threading.Event()
    engine.register(BlockingCurator(entered, release))
    errors = []

    def scan():
        try:
            engine.run_scan()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=scan, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        for method, args in [(engine.run_scan, ()), (engine.apply, (p.proposal_id,)),
                             (engine.dismiss, (p.proposal_id,)), (engine.undo, (p.proposal_id,))]:
            with pytest.raises(ValueError, match="busy"):
                method(*args)
        assert persisted_current(profile) is True
    finally:
        release.set()
        worker.join(5)
    assert not errors and not worker.is_alive()
    engine.apply(p.proposal_id)
    assert persisted_current(profile) is False


def test_auto_apply_is_reentrant_under_scan_lock(tmp_path):
    from memory.curation.types import CuratorMode
    engine, profile, p = profile_engine(tmp_path)
    engine.max_mode = CuratorMode.AUTO
    engine.curator_modes = {"synthetic": CuratorMode.AUTO}
    engine.anomaly_fraction = 1.0
    engine._proposals.clear()
    release = threading.Event()
    release.set()
    engine.register(BlockingCurator(threading.Event(), release, [p]))
    errors = []

    def scan():
        try:
            engine.run_scan()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=scan, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "nested auto-apply deadlocked"
    assert not errors and persisted_current(profile) is False
    assert p.status.value == "applied"


def test_auto_apply_does_not_swallow_journal_failure(tmp_path, monkeypatch):
    from memory.curation.types import CuratorMode

    engine, profile, p = profile_engine(tmp_path)
    engine.max_mode = CuratorMode.AUTO
    engine.curator_modes = {"synthetic": CuratorMode.AUTO}
    engine.anomaly_fraction = 1.0
    engine._proposals.clear()
    release = threading.Event()
    release.set()
    engine.register(BlockingCurator(threading.Event(), release, [p]))
    record = engine.journal.record

    def fail_prepare(event, **detail):
        if event == "apply_started":
            raise OSError("synthetic auto-apply journal failure")
        record(event, **detail)

    monkeypatch.setattr(engine.journal, "record", fail_prepare)
    with pytest.raises(OSError, match="auto-apply journal"):
        engine.run_scan()
    assert persisted_current(profile) is True


@pytest.mark.asyncio
async def test_api_exposes_and_undoes_recovered_interrupted_proposal(tmp_path, monkeypatch):
    import memory.curation.service as service
    from tests.unit.api_launch_auth_client import authed_client, make_test_app as create_app
    from tests.unit.helpers_orchestrator import _make_orchestrator

    engine, profile, p = profile_engine(tmp_path)
    apply = engine_module.apply_change

    def write_then_crash(*args, **kwargs):
        apply(*args, **kwargs)
        raise SimulatedCrash()

    with monkeypatch.context() as patcher:
        patcher.setattr(engine_module, "apply_change", write_then_crash)
        with pytest.raises(SimulatedCrash):
            engine.apply(p.proposal_id)
    Path(engine.queue_path).write_text("{synthetic corrupt queue")
    restarted = engine_at(tmp_path, StoreBundle(user_profile=UserProfile(profile.profile_path)))
    monkeypatch.setattr(service, "_engine", restarted)
    app = create_app(_make_orchestrator(), start_background=False)
    async with authed_client(app) as client:
        queued = (await client.get("/api/curation/queue")).json()["proposals"]
        assert queued[0]["status"] == "interrupted"
        for route in ("apply", "dismiss"):
            assert (await client.post(f"/api/curation/{p.proposal_id}/{route}")).status_code == 409
        response = await client.post(f"/api/curation/{p.proposal_id}/undo")
        assert response.status_code == 200 and response.json()["status"] == "undone"
        assert (await client.get("/api/curation/queue")).json()["proposals"] == []
    assert persisted_current(profile) is True


@pytest.mark.asyncio
async def test_api_timeout_keeps_worker_busy_until_actual_completion(tmp_path, monkeypatch):
    import config.app_config as config
    import memory.curation.service as service
    from tests.unit.api_launch_auth_client import authed_client, make_test_app as create_app
    from tests.unit.helpers_orchestrator import _make_orchestrator

    engine, profile, p = profile_engine(tmp_path)
    entered, release = threading.Event(), threading.Event()
    engine.register(BlockingCurator(entered, release))
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    scan = engine.run_scan

    def scan_and_signal():
        try:
            result = scan()
        except engine_module.CurationBusyError:
            raise
        except BaseException:
            loop.call_soon_threadsafe(finished.set)
            raise
        else:
            loop.call_soon_threadsafe(finished.set)
            return result

    monkeypatch.setattr(engine, "run_scan", scan_and_signal)
    monkeypatch.setattr(service, "_engine", engine)
    monkeypatch.setattr(config, "CURATION_SCAN_TIMEOUT_S", 0.1)
    app = create_app(_make_orchestrator(), start_background=False)
    async with authed_client(app) as client:
        first = asyncio.create_task(client.post("/api/curation/scan"))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
            result = await asyncio.wait_for(first, 2)
            assert result.status_code == 504 and "still running" in result.json()["detail"]
            for route in ("scan", f"{p.proposal_id}/apply", f"{p.proposal_id}/dismiss", f"{p.proposal_id}/undo"):
                response = await asyncio.wait_for(client.post("/api/curation/" + route), 1)
                assert response.status_code == 409 and "busy" in response.json()["detail"]
            assert persisted_current(profile) is True
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 3)
            await first
        response = await client.post(f"/api/curation/{p.proposal_id}/apply")
        assert response.status_code == 200 and persisted_current(profile) is False
