"""Tests for the remaining API surface: uploads, models, status/graph, config schema."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from tests.unit.api_launch_auth_client import authed_client as _client
from tests.unit.api_launch_auth_client import make_test_app as create_app
from tests.unit.helpers_orchestrator import _make_orchestrator
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


def _write_graph_fixture(path, entities, relations):
    """Build a real on-disk graph file through the deployed GraphMemory API
    (F09, docs/HANDOFF_20260909_independent_bug_audit.md) — never hand-write
    the JSON, since the writer schema (nodes as an id->attrs dict, edges with
    source_id/target_id) is exactly what the route under test must handle.

    entities: [(entity_id, display_name), ...]
    relations: [(source_id, relation, target_id), ...]
    """
    gm = GraphMemory(persist_path=str(path))
    for eid, display in entities:
        gm.add_entity(GraphNode(entity_id=eid, display_name=display))
    for src, rel, tgt in relations:
        gm.add_relation(GraphEdge(source_id=src, relation=rel, target_id=tgt))
    # save() no-ops when nothing has changed (dirty-flag optimization) — an
    # empty-graph fixture needs a real file on disk to exercise the route's
    # empty-payload path, so force the write through the real save() method.
    gm._dirty = True
    gm.save(raise_on_error=True)
    return gm


class TestUploads:
    @pytest.mark.asyncio
    async def test_upload_registers_files(self):
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post(
                "/api/uploads",
                files=[("files", ("notes.txt", b"hello world", "text/plain"))],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["files"]) == 1
        info = body["files"][0]
        assert info["name"] == "notes.txt"
        assert info["size"] == 11

        # file_id resolves to a shim object with .name pointing at a real temp file
        shims = app.state.daemon.resolve_uploads([info["file_id"]])
        assert len(shims) == 1
        assert shims[0].name.endswith(".txt")
        with open(shims[0].name, "rb") as f:
            assert f.read() == b"hello world"

    @pytest.mark.asyncio
    async def test_unknown_file_id_is_skipped(self):
        app = create_app(_make_orchestrator(), start_background=False)
        assert app.state.daemon.resolve_uploads(["nope"]) == []

    @pytest.mark.asyncio
    async def test_upload_limit_rolls_back_prior_files(self, monkeypatch):
        import api.routes.files as files_route
        monkeypatch.setattr(files_route, "MAX_TOTAL_BYTES", 5)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post(
                "/api/uploads",
                files=[
                    ("files", ("one.txt", b"1234", "text/plain")),
                    ("files", ("two.txt", b"56", "text/plain")),
                ],
            )
        assert resp.status_code == 413
        assert app.state.daemon._uploads == {}

    @pytest.mark.asyncio
    async def test_cancelled_upload_removes_partial_file(self, tmp_path, monkeypatch):
        import api.routes.files as files_route

        class CancelledUpload:
            filename = "partial.txt"

            def __init__(self):
                self.calls = 0

            async def read(self, _size):
                self.calls += 1
                if self.calls == 1:
                    return b"partial"
                raise asyncio.CancelledError

        monkeypatch.setattr(files_route, "_UPLOAD_DIR", str(tmp_path))
        app = create_app(_make_orchestrator(), start_background=False)
        request = SimpleNamespace(app=app)

        with pytest.raises(asyncio.CancelledError):
            await files_route.upload_files(request, [CancelledUpload()])

        assert list(tmp_path.iterdir()) == []
        assert app.state.daemon._uploads == {}


class TestModels:
    def _app(self):
        orch = _make_orchestrator()
        orch.model_manager.api_models = {"deepseek-v4": {}, "gpt-4-turbo": {}}
        orch.model_manager.models = {}
        orch.model_manager.get_active_model_name = MagicMock(return_value="deepseek-v4")
        return create_app(orch, start_background=False), orch

    @pytest.mark.asyncio
    async def test_list_models(self):
        app, _ = self._app()
        async with _client(app) as client:
            resp = await client.get("/api/models")
        body = resp.json()
        assert body["active"] == "deepseek-v4"
        assert set(body["models"]) == {"deepseek-v4", "gpt-4-turbo"}

    @pytest.mark.asyncio
    async def test_set_active_model(self, tmp_path, monkeypatch):
        # chdir so the yaml persist writes a scratch config/, not the real one
        monkeypatch.chdir(tmp_path)
        app, orch = self._app()
        async with _client(app) as client:
            resp = await client.put("/api/models/active", json={"name": "gpt-4-turbo"})
        assert resp.status_code == 200
        orch.model_manager.switch_model.assert_called_once_with("gpt-4-turbo")

    @pytest.mark.asyncio
    async def test_set_active_model_empty_name(self):
        app, _ = self._app()
        async with _client(app) as client:
            resp = await client.put("/api/models/active", json={"name": "  "})
        assert resp.status_code == 422


class TestSystem:
    @pytest.mark.asyncio
    async def test_status(self):
        orch = _make_orchestrator()
        orch.memory_system.corpus_manager.corpus = [{"x": 1}, {"x": 2}]
        app = create_app(orch, start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/status")
        body = resp.json()
        assert body["total_entries"] == 2
        assert body["active_model"] == "test-model"


class TestGraphEndpoint:
    """F09 (docs/HANDOFF_20260909_independent_bug_audit.md): the saved graph
    schema is `nodes` as an id->attrs DICT and edges with `source_id`/
    `target_id` (memory/graph_memory.py GraphMemory.save); the route used to
    assume `nodes` was already a list of `{"id": ...}` dicts with edge
    `source`/`target` fields — a schema the writer never produces — and
    crashed with `'str' object has no attribute 'get'` above the node limit.
    Every fixture here is produced by the deployed GraphMemory API, never
    hand-written in the route's old (wrong) assumed shape.
    """

    @pytest.mark.asyncio
    async def test_below_limit_returns_everything(self, tmp_path, monkeypatch):
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("alice", "Alice"), ("bob", "Bob")],
            [("alice", "knows", "bob")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 300})

        assert resp.status_code == 200
        body = resp.json()
        assert {n["id"] for n in body["nodes"]} == {"alice", "bob"}
        assert len(body["edges"]) == 1
        edge = body["edges"][0]
        assert edge["source"] == "alice"
        assert edge["target"] == "bob"
        assert edge["relation"] == "knows"

    @pytest.mark.asyncio
    async def test_equal_to_limit_returns_everything(self, tmp_path, monkeypatch):
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("alice", "Alice"), ("bob", "Bob")],
            [("alice", "knows", "bob")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 2})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        assert {n["id"] for n in body["nodes"]} == {"alice", "bob"}
        assert len(body["edges"]) == 1
        assert body["edges"][0]["source"] == "alice"
        assert body["edges"][0]["target"] == "bob"

    @pytest.mark.asyncio
    async def test_above_limit_trims_by_degree_without_crashing(self, tmp_path, monkeypatch):
        # This is the exact reproduction shape from the handoff: nodes above
        # the limit used to raise AttributeError iterating node ID strings.
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("hub", "Hub"), ("a", "A"), ("b", "B"), ("c", "C")],
            [("hub", "knows", "a"), ("hub", "knows", "b"), ("hub", "knows", "c")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 1})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 1
        assert body["nodes"][0]["id"] == "hub"  # highest degree (3) survives
        # No leaf survived the trim, so no edge can have both endpoints kept.
        assert body["edges"] == []

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_payload(self, tmp_path, monkeypatch):
        path = tmp_path / "kg.json"
        _write_graph_fixture(path, [], [])
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 300})

        assert resp.status_code == 200
        assert resp.json() == {"nodes": [], "edges": []}

    @pytest.mark.asyncio
    async def test_isolated_node_survives_with_zero_degree(self, tmp_path, monkeypatch):
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("alice", "Alice"), ("lonely", "Lonely")],
            [],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 300})

        assert resp.status_code == 200
        body = resp.json()
        assert {n["id"] for n in body["nodes"]} == {"alice", "lonely"}
        assert body["edges"] == []

    @pytest.mark.asyncio
    async def test_degree_tie_keeps_exactly_limit_nodes_and_valid_edges(self, tmp_path, monkeypatch):
        # Two disjoint pairs: every node has degree 1 (a tie).
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")],
            [("a", "knows", "b"), ("c", "knows", "d")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 2})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        ids = {n["id"] for n in body["nodes"]}
        for e in body["edges"]:
            assert e["source"] in ids and e["target"] in ids

    @pytest.mark.asyncio
    async def test_multiple_relations_between_one_pair_are_both_kept(self, tmp_path, monkeypatch):
        # The relation-level edge index (2026-09-03 fix) stores each relation
        # on a pair as its own edge — the route must not collapse them.
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("alice", "Alice"), ("bob", "Bob")],
            [("alice", "knows", "bob"), ("alice", "works_with", "bob")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 300})

        assert resp.status_code == 200
        body = resp.json()
        relations = {
            e["relation"] for e in body["edges"] if e["source"] == "alice" and e["target"] == "bob"
        }
        assert relations == {"knows", "works_with"}

    @pytest.mark.asyncio
    async def test_selected_edges_reference_only_selected_nodes(self, tmp_path, monkeypatch):
        path = tmp_path / "kg.json"
        _write_graph_fixture(
            path,
            [("hub", "Hub"), ("a", "A"), ("b", "B")],
            [("hub", "knows", "a"), ("hub", "knows", "b"), ("a", "knows", "b")],
        )
        monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 2})

        assert resp.status_code == 200
        body = resp.json()
        ids = {n["id"] for n in body["nodes"]}
        assert len(ids) == 2
        for e in body["edges"]:
            assert e["source"] in ids
            assert e["target"] in ids


class TestSyncNotes:
    """BC-80 / BC-81 (docs/BUG_CLASSES.md): the sync is a retained background
    job. The POST returns at once (still carrying ``message``), a second tap
    joins the running job, and the outcome is read from
    ``GET /api/sync-notes/status`` — so a client whose response dropped
    (2026-09-14 and 2026-09-16: server succeeded, SPA showed "failed") can
    read the truth after a reload. The worker uses the orchestrator's LIVE
    Chroma store, never a bare ObsidianManager().
    """

    @staticmethod
    def _result(**over):
        base = dict(errors=[], embedded_files=2, updated_files=1, skipped_files=5,
                    processed_files=8, total_files=8, total_chunks=12, duration_seconds=1.5)
        base.update(over)
        return MagicMock(**base)

    @staticmethod
    def _install_manager(monkeypatch, manager):
        import knowledge.obsidian_manager as om
        factory = MagicMock(return_value=manager)
        monkeypatch.setattr(om, "ObsidianManager", factory)
        return factory

    @staticmethod
    def _join(app, timeout=5.0):
        worker = app.state.daemon.notes_sync.worker
        assert worker is not None
        worker.join(timeout=timeout)
        assert not worker.is_alive(), "notes-sync worker did not finish"

    @pytest.mark.asyncio
    async def test_start_returns_immediately_and_status_reads_the_outcome(self, monkeypatch):
        manager = MagicMock()
        manager.embed_vault.return_value = self._result()
        factory = self._install_manager(monkeypatch, manager)
        orch = _make_orchestrator()
        app = create_app(orch, start_background=False)
        async with _client(app) as client:
            resp = await client.post("/api/sync-notes")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "running" and body["task_id"]
            assert "message" in body  # contract kept for the SPA toast + Gradio parity
            self._join(app)
            status = (await client.get("/api/sync-notes/status")).json()
        assert status["status"] == "succeeded"
        assert status["task_id"] == body["task_id"]
        assert "2 new" in status["message"] and "1 updated" in status["message"]
        assert status["error"] is None
        assert status["finished_at"] >= status["started_at"]
        assert isinstance(status["server_time"], float)
        last = status["last_result"]
        assert last["status"] == "succeeded" and last["task_id"] == body["task_id"]
        assert last["result"]["embedded_files"] == 2 and last["result"]["processed_files"] == 8
        manager.embed_vault.assert_called_once_with(force_reindex=False)
        # BC-81: the LIVE store is injected, not a lazily-built second one.
        factory.assert_called_once_with(chroma_store=orch.memory_system.chroma_store)

    @pytest.mark.asyncio
    async def test_status_is_idle_before_any_sync(self):
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            status = (await client.get("/api/sync-notes/status")).json()
        assert status["status"] == "idle" and status["last_result"] is None
        assert status["task_id"] is None and status["started_at"] is None

    @pytest.mark.asyncio
    async def test_single_flight_while_running_then_outcome_survives_the_request(self, monkeypatch):
        import threading
        release = threading.Event()
        entered = threading.Event()

        def _slow_embed(force_reindex=False):
            entered.set()
            assert release.wait(timeout=5), "test never released the worker"
            return self._result()

        manager = MagicMock()
        manager.embed_vault.side_effect = _slow_embed
        self._install_manager(monkeypatch, manager)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            first = (await client.post("/api/sync-notes")).json()
            assert first["status"] == "running"
            assert entered.wait(timeout=5)
            # The POST returned while the work is still in flight (the BC-80 shape).
            mid = (await client.get("/api/sync-notes/status")).json()
            assert mid["status"] == "running" and mid["task_id"] == first["task_id"]
            second = (await client.post("/api/sync-notes")).json()
            assert second["status"] == "running"
            assert second["task_id"] == first["task_id"]
            assert "already running" in second["message"]
            release.set()
            self._join(app)
            final = (await client.get("/api/sync-notes/status")).json()
        assert final["status"] == "succeeded" and final["task_id"] == first["task_id"]
        assert manager.embed_vault.call_count == 1  # the second tap did not start a second job

    @pytest.mark.asyncio
    async def test_embed_exception_is_a_failed_outcome_with_the_error(self, monkeypatch):
        manager = MagicMock()
        manager.embed_vault.side_effect = RuntimeError("vault missing")
        self._install_manager(monkeypatch, manager)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            await client.post("/api/sync-notes")
            self._join(app)
            status = (await client.get("/api/sync-notes/status")).json()
        assert status["status"] == "failed"
        assert status["error"] == "vault missing"
        assert "Sync failed" in status["message"]
        assert status["last_result"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_per_file_errors_are_reported_as_failed_with_counts(self, monkeypatch):
        manager = MagicMock()
        manager.embed_vault.return_value = self._result(errors=["Error processing a.md: bad"])
        self._install_manager(monkeypatch, manager)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            await client.post("/api/sync-notes")
            self._join(app)
            status = (await client.get("/api/sync-notes/status")).json()
        assert status["status"] == "failed"
        assert "completed with errors" in status["message"]
        assert status["last_result"]["result"]["errors"] == ["Error processing a.md: bad"]

    @pytest.mark.asyncio
    async def test_missing_live_store_fails_instead_of_building_a_private_one(self, monkeypatch):
        factory = self._install_manager(monkeypatch, MagicMock())
        orch = _make_orchestrator()
        orch.memory_system.chroma_store = None
        app = create_app(orch, start_background=False)
        async with _client(app) as client:
            await client.post("/api/sync-notes")
            self._join(app)
            status = (await client.get("/api/sync-notes/status")).json()
        assert status["status"] == "failed"
        assert "live Chroma store is unavailable" in status["error"]
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_start_failure_is_recorded_not_left_running(self, monkeypatch):
        import threading
        self._install_manager(monkeypatch, MagicMock())

        def _no_start(self_thread):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", _no_start)
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            body = (await client.post("/api/sync-notes")).json()
            status = (await client.get("/api/sync-notes/status")).json()
        assert body["status"] == "failed" and "failed to start" in body["message"]
        assert status["status"] == "failed"
        assert "can't start new thread" in status["error"]
        # A later tap is not blocked by a phantom "running" job.
        assert status["task_id"] == body["task_id"]

    def test_notes_sync_state_ignores_a_stale_task_outcome(self):
        from api.state import NotesSyncState
        state = NotesSyncState()
        assert state.try_start("a")
        assert not state.try_start("b")  # single flight
        state.finish("b", "failed", "stale")  # a job that never owned the slot
        assert state.snapshot()["status"] == "running"
        state.finish("a", "succeeded", "ok", result={"embedded_files": 1})
        snap = state.snapshot()
        assert snap["status"] == "succeeded" and snap["last_result"]["result"] == {"embedded_files": 1}
        assert state.try_start("c")  # a new job may start after a terminal state
        assert state.snapshot()["last_result"]["task_id"] == "a"  # retained across the new start


class TestApiConfigSchema:
    def test_api_section_defaults(self):
        from config.schema import ApiSection

        s = ApiSection()
        assert s.host == "127.0.0.1"
        assert s.port == 8000
        assert s.cors_origins == ["http://localhost:5173"]
        assert s.serve_frontend is True
        assert s.frontend_dist_dir == "web/dist"

    def test_api_section_on_daemon_config(self):
        from config.schema import DaemonConfig

        cfg = DaemonConfig()
        assert cfg.api.port == 8000
