"""Tests for the debug/provenance/settings API routes (Gradio tabs → SPA, 2026-07-14)."""

import pytest
import httpx

from api.app import create_app
from tests.unit.helpers_orchestrator import _make_orchestrator


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t")


def _record(**overrides):
    rec = {
        "mode": "enhanced",
        "query": "what is my cat's name?",
        "prompt": "[MEMORY]\ncat is Biscuit\n[QUERY]\nwhat is my cat's name?",
        "system_prompt": "You are Daemon.",
        "response": "Biscuit.",
        "model": "test-model",
        "prompt_tokens": 20,
        "system_tokens": 5,
        "total_tokens": 25,
        "citations_enabled": True,
        "citations": [{"id": "mem_1"}],
        "provenance": {
            "response_mode": "enhanced",
            "session_id": "sess-1",
            "model_name": "test-model",
            "cited_ids": ["mem_1"],
            "thinking_block": "t" * 900,
        },
        "phase_timings": {"total_wall": 2.0, "prompt_build": 1.0},
        "task_timings": {"retrieve_memories": 0.4},
        "gather_elapsed": 0.5,
    }
    rec.update(overrides)
    return rec


class TestDebugRoutes:
    @pytest.mark.asyncio
    async def test_empty_records(self):
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/debug")
        assert resp.status_code == 200
        assert resp.json() == {"records": [], "count": 0}

    @pytest.mark.asyncio
    async def test_records_returned_in_order(self):
        app = create_app(_make_orchestrator(), start_background=False)
        app.state.daemon.session.debug_records.extend(
            [_record(query="q1"), _record(query="q2")]
        )
        async with _client(app) as client:
            resp = await client.get("/api/debug")
        body = resp.json()
        assert body["count"] == 2
        assert [r["query"] for r in body["records"]] == ["q1", "q2"]
        assert body["records"][0]["prompt"].startswith("[MEMORY]")

    @pytest.mark.asyncio
    async def test_prompt_export_latest(self, monkeypatch):
        monkeypatch.setattr("config.app_config.DAEMON_MODE", "dev")
        app = create_app(_make_orchestrator(), start_background=False)
        app.state.daemon.session.debug_records.append(_record())
        async with _client(app) as client:
            resp = await client.get("/api/debug/prompt")
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert "[USER QUERY]" in resp.text
        assert "what is my cat's name?" in resp.text
        assert "[SYSTEM PROMPT]" in resp.text  # dev mode includes it

    @pytest.mark.asyncio
    async def test_prompt_export_hides_system_prompt_in_user_mode(self, monkeypatch):
        monkeypatch.setattr("config.app_config.DAEMON_MODE", "user")
        app = create_app(_make_orchestrator(), start_background=False)
        app.state.daemon.session.debug_records.append(_record())
        async with _client(app) as client:
            resp = await client.get("/api/debug/prompt")
        assert "[SYSTEM PROMPT]" not in resp.text
        assert "[FULL CONTEXT PROMPT]" in resp.text

    @pytest.mark.asyncio
    async def test_prompt_export_404_when_empty(self):
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/debug/prompt")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_prompt_export_by_index(self, monkeypatch):
        monkeypatch.setattr("config.app_config.DAEMON_MODE", "dev")
        app = create_app(_make_orchestrator(), start_background=False)
        app.state.daemon.session.debug_records.extend(
            [_record(query="first"), _record(query="second")]
        )
        async with _client(app) as client:
            resp = await client.get("/api/debug/prompt", params={"index": 0})
        assert "first" in resp.text

    @pytest.mark.asyncio
    async def test_provenance_merges_and_truncates(self):
        app = create_app(_make_orchestrator(), start_background=False)
        app.state.daemon.session.debug_records.append(_record())
        async with _client(app) as client:
            resp = await client.get("/api/provenance")
        prov = resp.json()
        assert prov["session_id"] == "sess-1"
        assert prov["mode"] == "enhanced"
        assert prov["model"] == "test-model"
        assert prov["citations_enabled"] is True
        assert prov["prompt_tokens"] == 20
        assert prov["thinking_block"].endswith(" [truncated]")
        assert len(prov["thinking_block"]) == 500 + len(" [truncated]")

    @pytest.mark.asyncio
    async def test_provenance_404_when_empty(self):
        app = create_app(_make_orchestrator(), start_background=False)
        async with _client(app) as client:
            resp = await client.get("/api/provenance")
        assert resp.status_code == 404


class TestSettingsRoutes:
    def _app(self, tmp_path, monkeypatch):
        # chdir so config.yaml reads/writes hit a scratch dir, never the real one
        monkeypatch.chdir(tmp_path)
        orch = _make_orchestrator()
        # Non-empty like a real orchestrator config (the core's `cfg or {}`
        # guard replaces an empty dict, so mutations would miss it)
        orch.config = {"features": {}, "web_search": {}, "models": {}}
        orch.model_manager.api_models = {"m-a": {}, "m-b": {}}
        orch.model_manager.models = {}
        orch.model_manager.default_max_tokens = 2048
        orch.model_manager.default_temperature = 0.7
        return create_app(orch, start_background=False), orch

    @pytest.mark.asyncio
    async def test_get_settings_snapshot(self, tmp_path, monkeypatch):
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.get("/api/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["streaming"]["disable_best_of"] is False
        assert body["web_search"]["enabled"] is True
        assert set(body["model_choices"]) == {"m-a", "m-b"}
        assert body["tokens"]["streaming_max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_put_streaming_persists_and_mutates(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/streaming", json={
                "disable_best_of": True,
                "disable_query_rewrite": False,
                "disable_llm_summaries": True,
                "best_of_latency_budget_s": 5.0,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True and body["persisted"] is True
        # Live orchestrator mutated (same as the Gradio button)
        assert orch.config["features"]["enable_best_of"] is False
        assert orch.config["features"]["disable_llm_summaries"] is True
        # Persisted to the scratch config.yaml
        import yaml
        data = yaml.safe_load(open(tmp_path / "config" / "config.yaml"))
        assert data["features"]["best_of_latency_budget_s"] == 5.0

    @pytest.mark.asyncio
    async def test_put_web_search(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/web-search", json={
                "enabled": False, "daily_credit_limit": 50,
            })
        assert resp.json()["ok"] is True
        assert orch.config["web_search"]["enabled"] is False
        import config.app_config as app_cfg
        assert app_cfg.WEB_SEARCH_ENABLED is False
        # restore module global for other tests
        app_cfg.WEB_SEARCH_ENABLED = True

    @pytest.mark.asyncio
    async def test_put_duel_same_model_is_400(self, tmp_path, monkeypatch):
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/duel", json={
                "enabled": True, "model_1": "m-a", "model_2": "m-a",
            })
        assert resp.status_code == 400
        assert "different models" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_put_duel_valid(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/duel", json={
                "enabled": True, "model_1": "m-a", "model_2": "m-b",
            })
        assert resp.json()["ok"] is True
        assert orch.config["features"]["best_of_generator_models"] == ["m-a", "m-b"]

    @pytest.mark.asyncio
    async def test_put_tokens_updates_model_manager(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/tokens", json={
                "best_of_max_tokens": 256,
                "judge_max_tokens": 64,
                "streaming_max_tokens": 4096,
            })
        assert resp.json()["ok"] is True
        assert orch.model_manager.default_max_tokens == 4096

    @pytest.mark.asyncio
    async def test_put_temperature(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/temperature",
                                    json={"temperature": 0.3})
        assert resp.json()["ok"] is True
        assert orch.model_manager.default_temperature == 0.3

    @pytest.mark.asyncio
    async def test_put_temperature_out_of_range_is_422(self, tmp_path, monkeypatch):
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/temperature",
                                    json={"temperature": 3.0})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_put_summary_cadence(self, tmp_path, monkeypatch):
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/summary-cadence",
                                    json={"every_n": 25})
        assert resp.json()["ok"] is True
        assert orch.memory_system.consolidator.consolidation_threshold == 25

    @pytest.mark.asyncio
    async def test_get_snapshot_includes_shutdown_llm_sections(self, tmp_path, monkeypatch):
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.get("/api/settings")
        body = resp.json()
        assert set(body["synthesis"]) == {"enabled", "candidates_per_session"}
        assert set(body["proposals"]) == {"enabled", "max_per_session"}

    @pytest.mark.asyncio
    async def test_put_synthesis_off_mutates_runtime_and_persists(self, tmp_path, monkeypatch):
        import config.app_config as app_cfg
        monkeypatch.setattr(app_cfg, "SYNTHESIS_POOLED_ENABLED", True)
        monkeypatch.setattr(app_cfg, "SYNTHESIS_GENERATOR_ENABLED", True)
        monkeypatch.setattr(app_cfg, "SYNTHESIS_POOLED_CANDIDATES_PER_SESSION", 8)
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/synthesis", json={
                "enabled": False, "candidates_per_session": 4,
            })
        assert resp.json()["ok"] is True
        assert app_cfg.SYNTHESIS_POOLED_ENABLED is False
        # OFF also forces the legacy generator flag off — the dreaming gate is
        # (GENERATOR or POOLED), so a stale yaml flag can't resurrect dreaming
        assert app_cfg.SYNTHESIS_GENERATOR_ENABLED is False
        assert app_cfg.SYNTHESIS_POOLED_CANDIDATES_PER_SESSION == 4
        assert orch.config["synthesis_pooled"]["enabled"] is False
        import yaml
        data = yaml.safe_load(open(tmp_path / "config" / "config.yaml"))
        assert data["synthesis_pooled"] == {"enabled": False, "candidates_per_session": 4}
        assert data["synthesis_generator"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_put_synthesis_on_leaves_retired_generator_off(self, tmp_path, monkeypatch):
        import config.app_config as app_cfg
        monkeypatch.setattr(app_cfg, "SYNTHESIS_POOLED_ENABLED", False)
        monkeypatch.setattr(app_cfg, "SYNTHESIS_GENERATOR_ENABLED", False)
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/synthesis", json={
                "enabled": True, "candidates_per_session": 8,
            })
        assert resp.json()["ok"] is True
        assert app_cfg.SYNTHESIS_POOLED_ENABLED is True
        assert app_cfg.SYNTHESIS_GENERATOR_ENABLED is False  # retired tier stays retired

    @pytest.mark.asyncio
    async def test_put_synthesis_count_out_of_range_is_422(self, tmp_path, monkeypatch):
        app, _ = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/synthesis", json={
                "enabled": True, "candidates_per_session": 50,
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_put_proposals_mutates_runtime_and_persists(self, tmp_path, monkeypatch):
        import config.app_config as app_cfg
        monkeypatch.setattr(app_cfg, "CODE_PROPOSALS_ENABLED", True)
        monkeypatch.setattr(app_cfg, "CODE_PROPOSALS_MAX_PER_SESSION", 5)
        app, orch = self._app(tmp_path, monkeypatch)
        async with _client(app) as client:
            resp = await client.put("/api/settings/proposals", json={
                "enabled": False, "max_per_session": 2,
            })
        assert resp.json()["ok"] is True
        assert app_cfg.CODE_PROPOSALS_ENABLED is False
        assert app_cfg.CODE_PROPOSALS_MAX_PER_SESSION == 2
        assert orch.config["code_proposals"]["enabled"] is False
        import yaml
        data = yaml.safe_load(open(tmp_path / "config" / "config.yaml"))
        assert data["code_proposals"] == {"enabled": False, "max_per_session": 2}


class TestGradioParityWiring:
    """The Gradio tab and the API must call THE SAME core functions."""

    def test_settings_tab_imports_core(self):
        import inspect
        import gui.tabs.settings as tab
        src = inspect.getsource(tab)
        for fn in ("apply_streaming", "apply_web_search", "apply_duel",
                   "apply_tokens", "apply_temperature", "apply_summary_cadence",
                   "apply_synthesis", "apply_proposals"):
            assert f"settings_core.{fn}" in src, f"Gradio tab no longer calls settings_core.{fn}"

    def test_api_routes_import_core(self):
        import inspect
        import api.routes.settings as routes
        src = inspect.getsource(routes)
        for fn in ("apply_streaming", "apply_web_search", "apply_duel",
                   "apply_tokens", "apply_temperature", "apply_summary_cadence",
                   "apply_synthesis", "apply_proposals"):
            assert f"settings_core.{fn}" in src, f"API route no longer calls settings_core.{fn}"
