"""Tests for the remaining API surface: uploads, models, status/graph, config schema."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import httpx
from unittest.mock import MagicMock

from api.app import create_app
from tests.unit.helpers_orchestrator import _make_orchestrator


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t")


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

    @pytest.mark.asyncio
    async def test_graph_trims_to_top_degree(self, tmp_path, monkeypatch):
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
            ],
        }
        path = tmp_path / "kg.json"
        path.write_text(json.dumps(graph))
        monkeypatch.setattr(
            "config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path)
        )

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/graph", params={"limit": 2})
        body = resp.json()
        assert len(body["nodes"]) == 2
        ids = {n["id"] for n in body["nodes"]}
        assert "a" in ids  # highest degree survives the trim
        for e in body["edges"]:
            assert e["source"] in ids and e["target"] in ids


class TestSyncNotes:
    @pytest.mark.asyncio
    async def test_sync_notes_returns_helper_message(self, monkeypatch):
        result = MagicMock(
            errors=[], embedded_files=2, updated_files=1,
            skipped_files=5, total_chunks=12, duration_seconds=1.5,
        )
        manager = MagicMock()
        manager.embed_vault.return_value = result

        import knowledge.obsidian_manager as om
        monkeypatch.setattr(om, "ObsidianManager", MagicMock(return_value=manager))

        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.post("/api/sync-notes")

        assert resp.status_code == 200
        msg = resp.json()["message"]
        assert "2 new" in msg and "1 updated" in msg
        manager.embed_vault.assert_called_once_with(force_reindex=False)


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
