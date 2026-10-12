"""B2 (2026-10-12): gui/setup_core.py -- the first-run setup core.

class: BC-83 (one .env authority), BC-16, BC-58. Fakes only: no network.
"""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui import setup_core


KEY = "sk-or-v1-" + "a" * 40


class _Resp:
    def __init__(self, code):
        self.status_code = code


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path))
    for n in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "WOLFRAM_APP_ID", "E2B_API_KEY"):
        monkeypatch.delenv(n, raising=False)
    return tmp_path


def _orch(name="", corpus=0, reloads=None):
    ident = SimpleNamespace(name=name, pronouns="")
    prefs = SimpleNamespace(style="balanced")
    saved = {}

    class _Profile:
        identity = ident
        preferences = prefs

        def update_identity(self, name, pronouns):
            ident.name, ident.pronouns = name, pronouns

        def update_preferences(self, style):
            prefs.style = style
            saved["style"] = style

    mm = SimpleNamespace(reload_api_keys=lambda: (reloads.append(1) if reloads is not None else None) or True)
    return SimpleNamespace(
        user_profile=_Profile(), model_manager=mm,
        memory_system=SimpleNamespace(corpus_manager=SimpleNamespace(corpus=[0] * corpus)),
    )


class TestEnvAuthority:
    def test_env_path_is_bootstrap_path(self, data_dir):
        from utils.bootstrap import get_user_data_dir
        assert str(setup_core.env_path()) == os.path.join(get_user_data_dir(), ".env")
        assert setup_core.env_path().parent == data_dir

    def test_write_is_0600_atomic_and_preserves_lines(self, data_dir):
        p = setup_core.env_path()
        p.write_text("# comment\nOTHER=1\nTAVILY_API_KEY=old\n")
        assert setup_core.write_env_vars({"TAVILY_API_KEY": "tvly-new", "NEW_ONE": "x"})
        lines = p.read_text().splitlines()
        assert lines == ["# comment", "OTHER=1", "TAVILY_API_KEY=tvly-new", "NEW_ONE=x"]
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
        assert os.environ["NEW_ONE"] == "x"
        assert not [f for f in os.listdir(data_dir) if f.endswith(".tmp")]
        os.environ.pop("NEW_ONE", None)

    def test_injection_rejected_nothing_written(self, data_dir):
        assert not setup_core.write_env_vars({"A": "x\nEVIL=1"})
        assert not setup_core.write_env_vars({"bad name": "x"})
        assert not setup_core.env_path().exists()


class TestKeyPresence:
    def test_placeholder_and_empty_are_missing(self, data_dir, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "your_key_here")
        monkeypatch.setenv("E2B_API_KEY", "  ")
        assert not setup_core.key_present("TAVILY_API_KEY")
        assert not setup_core.key_present("E2B_API_KEY")
        assert setup_core.key_hint("TAVILY_API_KEY") is None

    def test_hint_is_last_four_only(self, data_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        hint = setup_core.key_hint("OPENROUTER_API_KEY")
        assert hint == "…" + KEY[-4:]
        assert KEY[:-4] not in hint


class TestValidateOpenRouter:
    def test_200_ok_uses_typed_key_and_endpoint(self):
        seen = {}

        def fake(url, headers, timeout):
            seen.update(url=url, headers=headers)
            return _Resp(200)

        assert setup_core.validate_openrouter_key(KEY, http_get=fake)["status"] == "ok"
        assert seen["url"] == "https://openrouter.ai/api/v1/key"
        assert seen["headers"] == {"Authorization": f"Bearer {KEY}"}

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_failures_invalid(self, code):
        assert setup_core.validate_openrouter_key(KEY, http_get=lambda *a: _Resp(code))["status"] == "invalid"

    def test_other_status_and_exception_unreachable(self):
        assert setup_core.validate_openrouter_key(KEY, http_get=lambda *a: _Resp(500))["status"] == "unreachable"

        def boom(*a):
            raise OSError("down")
        assert setup_core.validate_openrouter_key(KEY, http_get=boom)["status"] == "unreachable"

    def test_empty_key_invalid_without_http(self):
        def never(*a):
            raise AssertionError("no call for an empty key")
        assert setup_core.validate_openrouter_key("  ", http_get=never)["status"] == "invalid"


class TestSaveOpenRouter:
    def test_ok_writes_openrouter_env_and_rekeys(self, data_dir):
        reloads = []
        res = setup_core.save_openrouter_key(_orch(reloads=reloads), KEY, http_get=lambda *a: _Resp(200))
        assert res == {"ok": True, "error": None, "hint": "…" + KEY[-4:], "unreachable": False}
        text = setup_core.env_path().read_text()
        assert f"OPENROUTER_API_KEY={KEY}" in text and "OPENAI_API_KEY" not in text
        assert os.environ["OPENROUTER_API_KEY"] == KEY
        assert reloads == [1]

    def test_invalid_not_saved(self, data_dir):
        reloads = []
        res = setup_core.save_openrouter_key(_orch(reloads=reloads), KEY, http_get=lambda *a: _Resp(401),
                                             save_unverified=True)
        assert res["ok"] is False and not setup_core.env_path().exists() and reloads == []

    def test_unreachable_not_saved_unless_unverified(self, data_dir):
        def boom(*a):
            raise OSError
        assert setup_core.save_openrouter_key(_orch(), KEY, http_get=boom)["ok"] is False
        assert not setup_core.env_path().exists()
        assert setup_core.save_openrouter_key(_orch(), KEY, http_get=boom, save_unverified=True)["ok"] is True
        assert setup_core.env_path().exists()


class TestOptionalKeys:
    def test_format_rejected_not_saved(self, data_dir):
        assert setup_core.save_optional_key(_orch(), "tavily", "nope")["ok"] is False
        assert not setup_core.env_path().exists()

    def test_tavily_flips_live_manager(self, data_dir):
        mgr = SimpleNamespace(api_key="", _api_key_invalid=True, _tavily_client=object(), _initialized=True)
        orch = _orch()
        orch.prompt_builder = SimpleNamespace(context_gatherer=SimpleNamespace(_web_search_manager=mgr))
        res = setup_core.save_optional_key(orch, "tavily", "tvly-" + "b" * 25)
        assert res["ok"] and mgr.api_key.startswith("tvly-")
        assert mgr._api_key_invalid is False and mgr._initialized is False

    def test_real_web_search_manager_becomes_available(self, data_dir, monkeypatch):
        from knowledge.web_search_manager import WebSearchManager
        mgr = WebSearchManager(api_key="")
        monkeypatch.setattr(WebSearchManager, "is_enabled", lambda self: True)
        monkeypatch.setattr(WebSearchManager, "_ensure_tavily", lambda self: True)
        assert mgr.is_available() is False
        orch = _orch()
        orch.prompt_builder = SimpleNamespace(context_gatherer=SimpleNamespace(_web_search_manager=mgr))
        setup_core.save_optional_key(orch, "tavily", "tvly-" + "c" * 25)
        assert mgr.is_available() is True


class TestProfileAndMarker:
    def test_profile_writes(self, data_dir, monkeypatch):
        applied = []
        import gui.wizard as wizard
        monkeypatch.setattr(wizard, "_apply_personality_style", lambda s: applied.append(s))
        o = _orch()
        assert setup_core.save_profile(o, " Sam ", None, "warm")["ok"]
        assert (o.user_profile.identity.name, o.user_profile.identity.pronouns) == ("Sam", "they/them")
        assert applied == ["warm"]
        assert setup_core.save_profile(o, "", None, None)["ok"] is False
        assert setup_core.save_profile(o, "Sam", None, "weird")["ok"] is False

    def test_status_flow_and_marker(self, data_dir, monkeypatch):
        o = _orch()
        st = setup_core.get_status(o)
        assert st["complete"] is False and st["steps"]["openrouter"]["state"] == "missing"
        setup_core.save_openrouter_key(o, KEY, http_get=lambda *a: _Resp(200))
        setup_core.save_profile(o, "Sam", None, None)
        # keys + name alone are NOT complete: the marker is the Finish step
        st = setup_core.get_status(o)
        assert st["steps"]["openrouter"]["state"] == "found" and st["complete"] is False
        st = setup_core.complete_setup(o)
        assert st["complete"] is True and os.path.exists(setup_core.marker_path())
        assert KEY not in repr(st)

    def test_complete_refused_without_required(self, data_dir):
        assert setup_core.complete_setup(_orch())["complete"] is False
        assert not os.path.exists(setup_core.marker_path())

    def test_established_profile_is_grandfathered(self, data_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        assert setup_core.get_status(_orch(name="Owner", corpus=50))["complete"] is True
        assert setup_core.get_status(_orch(name="Owner", corpus=1))["complete"] is False

    def test_corrupt_marker_is_not_complete(self, data_dir):
        open(setup_core.marker_path(), "w").write("{nope")
        assert setup_core.marker_exists() is False


class TestObsidian:
    def test_missing_dir_rejected(self, tmp_path):
        assert setup_core.save_obsidian_path(str(tmp_path / "nope"), save=lambda u: (True, None))["ok"] is False

    def test_dir_persisted_via_save(self, tmp_path):
        d = {}
        res = setup_core.save_obsidian_path(str(tmp_path), save=lambda u: (u(d), (True, None))[1])
        assert res["ok"] and d["obsidian"] == {"vault_path": str(tmp_path), "enabled": True}

    def test_none_means_no_vault(self):
        d = {}
        res = setup_core.save_obsidian_path(None, save=lambda u: (u(d), (True, None))[1])
        assert res["ok"] and res["path"] is None and d["obsidian"]["enabled"] is False


class TestCoordinatorContract:
    def test_unreachable_flag_and_text(self, data_dir):
        def boom(*a):
            raise OSError
        r = setup_core.save_openrouter_key(_orch(), KEY, http_get=boom)
        assert r["ok"] is False and r["unreachable"] is True and r["error"].startswith("Could not reach OpenRouter")
        r5 = setup_core.save_openrouter_key(_orch(), KEY, http_get=lambda *a: _Resp(500))
        assert r5["unreachable"] is True and r5["error"].startswith("Could not reach OpenRouter")
        r4 = setup_core.save_openrouter_key(_orch(), KEY, http_get=lambda *a: _Resp(401))
        assert r4["unreachable"] is False

    def test_profile_idempotent_and_style_null_keeps_style(self, data_dir, monkeypatch):
        import gui.wizard as wizard
        monkeypatch.setattr(wizard, "_apply_personality_style", lambda s: None)
        o = _orch()
        setup_core.save_profile(o, "Sam", "he/him", None)
        setup_core.save_profile(o, "Sam", None, "direct")
        setup_core.save_profile(o, "Sam", None, None)
        assert o.user_profile.identity.pronouns == "he/him"
        assert o.user_profile.preferences.style == "direct"


class TestReferee1012:
    """Referee additions: the dev-mode .env authority and the legacy key name."""

    def test_source_checkout_without_override_uses_repo_root_env(self, monkeypatch):
        import utils.bootstrap as bootstrap
        monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
        monkeypatch.setattr(bootstrap, "IS_FROZEN", False)
        # main.py's bare load_dotenv() reads the repo-root .env, never data/.env
        assert setup_core.resolve_env_path() == Path(bootstrap.get_app_dir()) / ".env"

    def test_frozen_uses_user_data_dir(self, monkeypatch, tmp_path):
        import utils.bootstrap as bootstrap
        monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
        monkeypatch.setattr(bootstrap, "IS_FROZEN", True)
        monkeypatch.setattr(bootstrap, "get_user_data_dir", lambda: str(tmp_path))
        assert setup_core.resolve_env_path() == tmp_path / ".env"

    def test_legacy_openai_key_counts_as_the_openrouter_key(self, data_dir, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-" + "k" * 30 + "wxyz")
        st = setup_core.get_status(_orch(name="Sam"))
        assert st["steps"]["openrouter"]["state"] == "found"
        assert st["steps"]["openrouter"]["hint"] == "…wxyz"


def test_conftest_sandboxes_the_env_file(tmp_path):
    """Any test that drives a setup/wizard key save writes a temp file, never a
    checkout's real .env (a wizard test once wrote the repo-root .env)."""
    assert setup_core.env_path() == tmp_path / ".env"
    assert setup_core.resolve_env_path() != tmp_path / ".env"


def test_status_reports_mode_live(data_dir, monkeypatch):
    """The SPA hides the dev-only E2B key outside dev mode (legacy wizard parity)."""
    import config.app_config as app_cfg
    monkeypatch.setattr(app_cfg, "DAEMON_MODE", "user")
    assert setup_core.get_status(_orch(name="Sam"))["mode"] == "user"
    monkeypatch.setattr(app_cfg, "DAEMON_MODE", "dev")
    assert setup_core.get_status(_orch(name="Sam"))["mode"] == "dev"
