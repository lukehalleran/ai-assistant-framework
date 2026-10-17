"""B2 (2026-10-12): /api/setup routes + the first-run launch decision.

class: BC-83, BC-16, BC-58. TestClient-style (ASGI) against the deployed app; fake HTTP.

Original batch label: oct12_b2 (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""

import os
import stat
from types import SimpleNamespace

import pytest

from gui import setup_core
from tests.unit.api_launch_auth_client import authed_client as _client
from tests.unit.api_launch_auth_client import make_test_app as create_app
from tests.unit.helpers_orchestrator import _make_orchestrator

KEY = "sk-or-v1-" + "z" * 40


class _Resp:
    def __init__(self, code):
        self.status_code = code


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path))
    for n in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "WOLFRAM_APP_ID", "E2B_API_KEY"):
        monkeypatch.delenv(n, raising=False)
    return tmp_path


def _app(reloads=None):
    orch = _make_orchestrator()
    ident = SimpleNamespace(name="", pronouns="")
    prefs = SimpleNamespace(style="balanced")

    class _P:
        identity, preferences = ident, prefs

        def update_identity(self, name, pronouns):
            ident.name, ident.pronouns = name, pronouns

        def update_preferences(self, style):
            prefs.style = style
    orch.user_profile = _P()
    calls = reloads if reloads is not None else []
    orch.model_manager = SimpleNamespace(reload_api_keys=lambda: calls.append(1) or True)
    orch.memory_system = SimpleNamespace(corpus_manager=SimpleNamespace(corpus=[]))
    return create_app(orch, start_background=False)


class TestSetupRoutes:
    @pytest.mark.asyncio
    async def test_status_initially_incomplete(self, env):
        async with _client(_app()) as c:
            r = await c.get("/api/setup/status")
        assert r.status_code == 200
        body = r.json()
        assert body["complete"] is False
        assert body["steps"]["openrouter"] == {"state": "missing", "hint": None}

    @pytest.mark.asyncio
    async def test_openrouter_ok_writes_env_0600_rekeys_never_echoes(self, env, monkeypatch):
        monkeypatch.setattr(setup_core, "default_http_get", lambda *a: _Resp(200))
        reloads = []
        (env / ".env").write_text("KEEP=me\n")
        async with _client(_app(reloads)) as c:
            r = await c.post("/api/setup/openrouter", json={"key": KEY})
            st = await c.get("/api/setup/status")
        assert r.json() == {"ok": True, "error": None, "hint": "…" + KEY[-4:], "unreachable": False}
        assert KEY not in r.text and KEY not in st.text
        p = env / ".env"
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
        assert p.read_text().splitlines() == ["KEEP=me", f"OPENROUTER_API_KEY={KEY}"]
        assert reloads == [1]
        assert st.json()["steps"]["openrouter"]["state"] == "found"

    @pytest.mark.asyncio
    async def test_401_not_saved(self, env, monkeypatch):
        monkeypatch.setattr(setup_core, "default_http_get", lambda *a: _Resp(401))
        reloads = []
        async with _client(_app(reloads)) as c:
            r = await c.post("/api/setup/openrouter", json={"key": KEY, "save_unverified": True})
        body = r.json()
        assert body["ok"] is False and body["error"] and KEY not in r.text
        assert not (env / ".env").exists() and reloads == []

    @pytest.mark.asyncio
    async def test_network_error_not_saved_unless_unverified(self, env, monkeypatch):
        def boom(*a):
            raise OSError("down")
        monkeypatch.setattr(setup_core, "default_http_get", boom)
        async with _client(_app()) as c:
            r1 = await c.post("/api/setup/openrouter", json={"key": KEY})
            assert r1.status_code == 200 and r1.json()["ok"] is False and r1.json()["unreachable"] is True
            assert r1.json()["error"].startswith("Could not reach OpenRouter") and not (env / ".env").exists()
            r2 = await c.post("/api/setup/openrouter", json={"key": KEY, "save_unverified": True})
        assert r2.json()["ok"] is True and (env / ".env").exists()

    @pytest.mark.asyncio
    async def test_optional_key_and_unknown(self, env):
        async with _client(_app()) as c:
            bad = await c.post("/api/setup/key/tavily", json={"key": "x"})
            ok = await c.post("/api/setup/key/tavily", json={"key": "tvly-" + "q" * 25})
            unk = await c.post("/api/setup/key/openai", json={"key": "x"})
        assert bad.json()["ok"] is False
        assert ok.json()["ok"] is True and "tvly-qqq" not in ok.text
        assert unk.status_code == 404

    @pytest.mark.asyncio
    async def test_full_flow_to_complete(self, env, monkeypatch, tmp_path):
        monkeypatch.setattr(setup_core, "default_http_get", lambda *a: _Resp(200))
        import gui.wizard as wizard
        monkeypatch.setattr(wizard, "_apply_personality_style", lambda s: None)
        async with _client(_app()) as c:
            early = await c.post("/api/setup/complete")
            assert early.json()["complete"] is False
            await c.post("/api/setup/openrouter", json={"key": KEY})
            pr = await c.post("/api/setup/profile", json={"name": "Sam", "pronouns": "he/him", "style": "direct"})
            assert pr.json()["ok"] is True
            done = await c.post("/api/setup/complete")
            st = await c.get("/api/setup/status")
        assert done.json()["complete"] is True and st.json()["complete"] is True
        assert st.json()["steps"]["profile"]["name"] == "Sam"

    @pytest.mark.asyncio
    async def test_obsidian_missing_dir_rejected(self, env):
        async with _client(_app()) as c:
            r = await c.post("/api/setup/obsidian", json={"path": str(env / "nope")})
        assert r.json()["ok"] is False

    @pytest.mark.asyncio
    async def test_routes_behind_launch_auth(self, env):
        import httpx
        app = _app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1:8000") as c:
            r = await c.get("/api/setup/status")
        assert r.status_code in (401, 403)


class TestFirstRunLaunchDecision:
    def test_first_run_does_not_take_gradio_path(self):
        import main
        assert main.use_legacy_gradio_path(legacy_gui=False, force_wizard=False, needs_wizard=True) is False

    def test_dev_flags_keep_legacy_path(self):
        import main
        assert main.use_legacy_gradio_path(True, False, False) is True
        assert main.use_legacy_gradio_path(False, True, True) is True

    def test_main_branch_uses_the_decision(self):
        import inspect
        import main
        src = inspect.getsource(main)
        assert "if use_legacy_gradio_path(legacy_gui, force_wizard, needs_wizard):" in src
        assert "if legacy_gui or needs_wizard:" not in src


class TestModelManagerReload:
    def test_reload_api_keys_rebuilds_from_env(self, monkeypatch):
        from models.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        seen = []
        monkeypatch.setattr(ModelManager, "reinitialize_clients", lambda self, k=None: seen.append(k) or True)
        assert mm.reload_api_keys() is True and seen == [None]

    def test_reinitialize_picks_up_env_key(self, monkeypatch):
        from models.model_manager import ModelManager
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        mm = ModelManager(api_key="")
        mm.api_key = None
        assert mm.reload_api_keys() is True and mm.api_key == KEY
