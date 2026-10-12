"""First-run setup core (2026-10-12, B2) -- pure functions, settings_core pattern.

Both the FastAPI routes (api/routes/setup.py) and the legacy Gradio wizard
(gui/wizard.py) call THESE functions, so there is one .env authority, one set
of key validators and one profile-write path (class: BC-83, BC-16).

Rules:
- The .env file is the file utils/bootstrap.setup_environment loads
  (``<get_user_data_dir()>/.env``) -- never APPDATA on Linux.
- Env writes are atomic (temp + os.replace), mode 0600, preserve other lines.
- Keys are never logged and never returned (only a masked last-4 hint).
- The OpenRouter key is validated with the TYPED key against the free key-info
  endpoint (GET /api/v1/key) -- never a generation call. Only HTTP 200 is valid.
"""

import os
import re
from pathlib import Path
from typing import Callable, Optional

from utils.bootstrap import _PLACEHOLDER_PATTERNS, get_user_data_dir
from utils.logging_utils import get_logger
from utils.safe_json import CorruptStoreError, atomic_write_json, atomic_write_text, load_critical_json

logger = get_logger("setup_core")

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_ENV = "OPENROUTER_API_KEY"

# step name -> env var for the optional keys
OPTIONAL_KEYS = {
    "tavily": "TAVILY_API_KEY",
    "wolfram": "WOLFRAM_APP_ID",
    "e2b": "E2B_API_KEY",
}
VALID_STYLES = ("warm", "balanced", "direct")

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BAD_VALUE_RE = re.compile(r"[\x00-\x1f\x7f\s]")  # newline injection / whitespace in a key


# ---------------------------------------------------------------------------
# .env authority + atomic writer
# ---------------------------------------------------------------------------

def env_path() -> Path:
    """The .env file every setup write targets (tests/conftest.py sandboxes this)."""
    return resolve_env_path()


def resolve_env_path() -> Path:
    """The ONE .env authority: the file the running launch path loads.

    Packaged (frozen) or with ``DAEMON_DATA_DIR`` set: ``<get_user_data_dir()>/.env``,
    the file utils/bootstrap.setup_environment + main.py's frozen branch load.
    Source checkout without an override: main.py's bare ``load_dotenv()`` reads
    the repo-root ``.env`` -- writing ``data/.env`` there would save a key that
    is never read (class: BC-83).
    """
    import utils.bootstrap as _bootstrap  # lazy import: patch-point (tests flip IS_FROZEN)
    if _bootstrap.IS_FROZEN or _bootstrap._data_dir_override():
        return Path(os.path.join(get_user_data_dir(), ".env"))
    return Path(os.path.join(_bootstrap.get_app_dir(), ".env"))


def _read_env_lines(path: Path) -> list:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def write_env_vars(mapping: dict) -> bool:
    """Atomically write/update KEY=VALUE pairs in the profile .env (0600).

    Other lines and comments are preserved; an existing key is updated in
    place; new keys append. Also sets os.environ. Returns False (nothing
    written) on an invalid name/value or an I/O failure. Never logs values.
    """
    try:
        for name, value in mapping.items():
            if not _ENV_NAME_RE.match(str(name)):
                raise ValueError(f"invalid env var name {name!r}")
            if _BAD_VALUE_RE.search(str(value)):
                raise ValueError(f"invalid characters in value for {name}")
        path = env_path()
        lines = _read_env_lines(path)
        pending = {str(k): str(v) for k, v in mapping.items()}
        for i, line in enumerate(lines):
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and m.group(1) in pending:
                lines[i] = f"{m.group(1)}={pending.pop(m.group(1))}"
        for name, value in pending.items():
            lines.append(f"{name}={value}")
        atomic_write_text(str(path), "\n".join(lines) + "\n", mode=0o600)
        for name, value in mapping.items():
            os.environ[str(name)] = str(value)
        logger.info("[Setup] wrote %d env var(s) to the profile .env", len(mapping))
        return True
    except Exception as e:
        logger.error("[Setup] env write failed: %s", type(e).__name__)
        return False


def _is_placeholder(value: str) -> bool:
    lower = value.lower()
    return any(p.lower() in lower for p in _PLACEHOLDER_PATTERNS)


def key_present(name: str) -> bool:
    """True when the env var holds a real (non-empty, non-placeholder) value."""
    value = (os.environ.get(name) or "").strip()
    return bool(value) and not _is_placeholder(value)


# The legacy wizard wrote the OpenRouter key as OPENAI_API_KEY, and
# ModelManager still reads it as the fallback -- so a profile set up that way
# HAS a working key and must not be asked again.
_KEY_FALLBACKS = {OPENROUTER_ENV: ("OPENAI_API_KEY",)}


def _effective_key_var(name: str) -> Optional[str]:
    for var in (name,) + _KEY_FALLBACKS.get(name, ()):
        if key_present(var):
            return var
    return None


def key_hint(name: str) -> Optional[str]:
    """Masked hint '...abcd' (last 4) of a present key (or its legacy fallback), else None."""
    var = _effective_key_var(name)
    if var is None:
        return None
    return "…" + os.environ[var].strip()[-4:]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_api_key_format(key: str) -> bool:
    """OpenRouter key shape (legacy wizard pre-check)."""
    key = key.strip()
    return key.startswith("sk-or-") and len(key) > 20


def validate_tavily_key_format(key: str) -> bool:
    key = key.strip()
    return key.startswith("tvly-") and len(key) > 20


def validate_wolfram_key_format(key: str) -> bool:
    return len(key.strip()) >= 6


def validate_e2b_key_format(key: str) -> bool:
    key = key.strip()
    return key.startswith("e2b_") and len(key) > 10


_FORMAT_VALIDATORS = {
    "tavily": validate_tavily_key_format,
    "wolfram": validate_wolfram_key_format,
    "e2b": validate_e2b_key_format,
}


def validate_optional_key_format(name: str, key: str) -> Optional[str]:
    """None when the format is plausible, else a user-facing error string."""
    if name not in _FORMAT_VALIDATORS:
        return "Unknown key."
    if _BAD_VALUE_RE.search(key.strip()) or not key.strip():
        return "That key contains spaces or line breaks."
    if not _FORMAT_VALIDATORS[name](key):
        return f"That doesn't look like a valid {name} key."
    return None


def default_http_get(url: str, headers: dict, timeout: float):
    """Real HTTP GET (the only network call here). Tests inject a fake instead."""
    import httpx  # lazy import: startup-cost
    return httpx.get(url, headers=headers, timeout=timeout)


def validate_openrouter_key(key: str, http_get: Callable = None, timeout: float = 10.0) -> dict:
    """Validate the TYPED key via OpenRouter's free key-info endpoint.

    Returns {"status": "ok"|"invalid"|"unreachable", "error": str|None}.
    Only HTTP 200 is ok; 401/403 invalid; anything else/exception unreachable.
    """
    key = (key or "").strip()
    if not key or _BAD_VALUE_RE.search(key):
        return {"status": "invalid", "error": "Paste your OpenRouter key (no spaces or line breaks)."}
    http_get = http_get or default_http_get
    try:
        resp = http_get(OPENROUTER_KEY_URL, {"Authorization": f"Bearer {key}"}, timeout)
        code = int(getattr(resp, "status_code", 0))
    except Exception as e:  # degrades: reported as unreachable, key not saved
        logger.warning("[Setup] OpenRouter key check unreachable: %s", type(e).__name__)
        return {"status": "unreachable", "error": "Could not reach OpenRouter. Check your connection."}
    if code == 200:
        return {"status": "ok", "error": None}
    if code in (401, 403):
        return {"status": "invalid", "error": "OpenRouter rejected that key."}
    return {"status": "unreachable", "error": f"Could not reach OpenRouter (unexpected response {code})."}


# ---------------------------------------------------------------------------
# Live refresh after a key save
# ---------------------------------------------------------------------------

def _reload_model_manager(orchestrator) -> bool:
    mm = getattr(orchestrator, "model_manager", None)
    reload_fn = getattr(mm, "reload_api_keys", None)
    if not callable(reload_fn):
        return False
    try:
        return bool(reload_fn())
    except Exception as e:  # degrades: key saved, chat needs a restart
        logger.warning("[Setup] model manager re-key failed: %s", type(e).__name__)
        return False


def _refresh_live_managers(orchestrator, env_var: str, value: str) -> None:
    """Make a saved optional key live: app_config constant + live managers."""
    try:
        import config.app_config as app_cfg  # lazy import: live-config
        attr = {"TAVILY_API_KEY": "WEB_SEARCH_API_KEY", "WOLFRAM_APP_ID": "WOLFRAM_APP_ID",
                "E2B_API_KEY": "SANDBOX_API_KEY"}.get(env_var)
        if attr:
            setattr(app_cfg, attr, value)
    except Exception:  # degrades: next start picks the key up from .env
        pass
    if env_var == "TAVILY_API_KEY":
        from gui import settings_core  # lazy import: layering
        for mgr in settings_core._live_web_search_managers(orchestrator):
            mgr.api_key = value
            mgr._api_key_invalid = False
            mgr._tavily_client = None
            mgr._initialized = False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def save_openrouter_key(orchestrator, key: str, *, http_get: Callable = None,
                        save_unverified: bool = False) -> dict:
    """Validate (typed key) then save + re-key in-process. Never returns the key."""
    key = (key or "").strip()
    check = validate_openrouter_key(key, http_get=http_get)
    unreachable = check["status"] == "unreachable"
    if check["status"] == "invalid" or (unreachable and not save_unverified):
        return {"ok": False, "error": check["error"], "hint": None, "unreachable": unreachable}
    if not write_env_vars({OPENROUTER_ENV: key}):
        return {"ok": False, "error": "Could not save the key to your profile folder.", "hint": None,
                "unreachable": False}
    _reload_model_manager(orchestrator)
    return {"ok": True, "error": None, "hint": key_hint(OPENROUTER_ENV), "unreachable": False}


def save_optional_key(orchestrator, name: str, key: str) -> dict:
    """Format-validate (no paid call), save, refresh live managers."""
    err = validate_optional_key_format(name, key or "")
    if err:
        return {"ok": False, "error": err, "hint": None}
    env_var = OPTIONAL_KEYS[name]
    key = key.strip()
    if not write_env_vars({env_var: key}):
        return {"ok": False, "error": "Could not save the key to your profile folder.", "hint": None}
    _refresh_live_managers(orchestrator, env_var, key)
    return {"ok": True, "error": None, "hint": key_hint(env_var)}


def save_profile(orchestrator, name: str, pronouns: Optional[str], style: Optional[str]) -> dict:
    """Identity + style, the same writes the wizard's finalize does."""
    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "Please enter a name or nickname."}
    # Idempotent update: pronouns None/blank keeps the stored value (they/them if none yet).
    try:
        existing = (orchestrator.user_profile.identity.pronouns or "").strip()
    except Exception:  # degrades: no stored pronouns known
        existing = ""
    pronouns = (pronouns or "").strip() or existing or "they/them"
    style = (style or "").strip().lower() or None
    if style is not None and style not in VALID_STYLES:
        return {"ok": False, "error": "Unknown response style."}
    try:
        profile = getattr(orchestrator, "user_profile", None)
        if profile is None:
            from memory.user_profile import UserProfile  # lazy import: import-side-effect
            profile = UserProfile()
            orchestrator.user_profile = profile
        profile.update_identity(name=name, pronouns=pronouns)
        if style is not None:
            profile.update_preferences(style=style)
            from gui.wizard import _apply_personality_style  # lazy import: cycle (wizard imports setup_core)
            _apply_personality_style(style)
    except Exception as e:
        logger.error("[Setup] profile save failed: %s", type(e).__name__)
        return {"ok": False, "error": "Could not save your profile."}
    return {"ok": True, "error": None}


def save_obsidian_path(path: Optional[str], save: Optional[Callable] = None) -> dict:
    """Persist the vault path via Settings' config.local.yaml writer. None/empty = no vault."""
    path = (path or "").strip()
    resolved = ""
    if path:
        p = Path(os.path.expanduser(path))
        if not p.is_dir():
            return {"ok": False, "error": "That folder doesn't exist.", "path": None}
        resolved = str(p)
    if save is None:
        from gui import settings_core  # lazy import: layering
        save = settings_core.save_settings
    ok, err = save(lambda d: d.setdefault("obsidian", {}).update(
        {"vault_path": resolved, "enabled": bool(resolved)}))
    if not ok:
        logger.error("[Setup] vault path persist failed")
        return {"ok": False, "error": "Could not save the vault setting.", "path": None}
    try:
        import config.app_config as app_cfg  # lazy import: live-config
        app_cfg.OBSIDIAN_VAULT_PATH = resolved
        app_cfg.OBSIDIAN_ENABLED = bool(resolved)
    except Exception:  # degrades: takes effect on next start
        pass
    return {"ok": True, "error": None, "path": resolved or None}


# ---------------------------------------------------------------------------
# Setup-complete marker + status
# ---------------------------------------------------------------------------

def marker_path() -> str:
    return os.path.join(get_user_data_dir(), "setup_complete.json")


def marker_exists() -> bool:
    try:
        return bool((load_critical_json(marker_path(), "setup_complete") or {}).get("complete"))
    except CorruptStoreError:
        # quarantined by load_critical_json; the marker carries no user data,
        # so re-running the (resumable) setup is the safe answer.
        return False


def write_marker() -> None:
    atomic_write_json(marker_path(), {"complete": True, "schema_version": 1})


def _current_vault_path() -> Optional[str]:
    try:
        import config.app_config as app_cfg  # lazy import: live-config
        if app_cfg.OBSIDIAN_ENABLED and str(app_cfg.OBSIDIAN_VAULT_PATH or "").strip() \
                and Path(os.path.expanduser(app_cfg.OBSIDIAN_VAULT_PATH)).is_dir():
            return str(app_cfg.OBSIDIAN_VAULT_PATH)
    except Exception:  # degrades: reported as no vault
        pass
    return None


def _established_profile(orchestrator) -> bool:
    """An existing user (corpus history AND a named profile) -- never sees Setup.

    Deliberately stricter than UserProfile.is_first_run: that turns False the
    moment a name is saved, which would end Setup mid-flow (optional keys and
    the vault come after the profile step).
    """
    try:
        profile = orchestrator.user_profile
        corpus = orchestrator.memory_system.corpus_manager
        count = len(corpus.corpus) if hasattr(corpus, "corpus") else 0
        return count >= 5 and bool((profile.identity.name or "").strip())
    except Exception:  # degrades: no profile/corpus info -> not established
        return False


def get_status(orchestrator) -> dict:
    profile = getattr(orchestrator, "user_profile", None)
    name = pronouns = style = None
    try:
        if profile is not None:
            name = (profile.identity.name or "").strip() or None
            pronouns = (profile.identity.pronouns or "").strip() or None
            style = profile.preferences.style
    except Exception:  # degrades: profile fields shown as missing
        pass
    marker = marker_exists()
    steps = {
        "openrouter": {"state": "found" if _effective_key_var(OPENROUTER_ENV) else "missing",
                       "hint": key_hint(OPENROUTER_ENV)},
        "profile": {"state": "found" if name else "missing", "name": name, "pronouns": pronouns},
        "style": {"state": "found" if (marker or (style and style != "balanced")) else "missing",
                  "value": style},
    }
    for step, env_var in OPTIONAL_KEYS.items():
        steps[step] = {"state": "found" if key_present(env_var) else "missing",
                       "hint": key_hint(env_var)}
    vault = _current_vault_path()
    steps["obsidian"] = {"state": "found" if vault else "missing", "path": vault}
    required = steps["openrouter"]["state"] == "found" and steps["profile"]["state"] == "found"
    complete = (required and marker) or (required and _established_profile(orchestrator))
    # The SPA shows dev-only optional keys (E2B code sandbox) only in dev mode,
    # as the legacy wizard did. Read live (Settings can change the mode).
    import config.app_config as app_cfg  # lazy import: live-config
    mode = "dev" if str(getattr(app_cfg, "DAEMON_MODE", "user")).lower() == "dev" else "user"
    return {"complete": bool(complete), "mode": mode, "steps": steps}


def complete_setup(orchestrator) -> dict:
    """Write the marker (only if the required steps are done); return status."""
    status = get_status(orchestrator)
    required = (status["steps"]["openrouter"]["state"] == "found"
                and status["steps"]["profile"]["state"] == "found")
    if required:
        write_marker()
        status = get_status(orchestrator)
    return status
