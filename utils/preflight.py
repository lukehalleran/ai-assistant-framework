"""
Module Contract — utils/preflight.py

Purpose:
    Startup preflight checks for full-orchestrator modes (gui/cli). Turns
    "cryptic failure on first use" into an actionable message at launch:
      - data directory not writable        → FATAL (memory could not persist)
      - OPENROUTER_API_KEY / OPENAI_API_KEY (either) missing/placeholder
                                           → WARNING (chat will return
        [AUTH ERROR]; local-model setups are legitimate, so not fatal)
      - TAVILY_API_KEY missing             → NOTE (web search disabled)
      - Google token expired, no refresh   → WARNING (2026-09-27, BC-47,
        BC-58, BC-70, BC-72: disk-only check, no network — every downstream
        Gmail/Calendar/Contacts read would otherwise silently read empty)
      - installed packages drift from requirements.lock.txt → WARNING
        (2026-10-24: one floating transitive release broke spaCy loading;
        advisory only — never installs, skipped when frozen / no lock)
      - spaCy en_core_web_sm missing       → NOTE (fact extraction degrades
        to regex-only; core paths already guard this lazily)

Inputs:  environment variables + config.app_config paths.
Outputs: PreflightResult(fatal, warnings); print_preflight() renders it.
Side effects: creates the data directory if missing; writes+removes a
    small probe file to verify writability. No network calls.

Key decision: only unrecoverable-data conditions are fatal. Everything
    else warns loudly and lets the app start (graceful degradation is a
    project-wide pattern).
"""

import importlib.metadata
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List

from utils.logging_utils import get_logger

logger = get_logger("preflight")

_PLACEHOLDER_FRAGMENTS = ("your_", "your-", "placeholder", "xxx", "todo", "test-key", "changeme")


@dataclass
class PreflightResult:
    fatal: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal


def _looks_like_placeholder(value: str) -> bool:
    # Fragment matches only count for short values: a real 35-char sk-or- key
    # can legitimately contain e.g. "test-key" in its random section
    # (observed 2026-07-14 — working key, false-positive warning).
    if len(value) >= 30:
        return False
    lowered = value.lower()
    return any(frag in lowered for frag in _PLACEHOLDER_FRAGMENTS)


def _check_data_dir_writable(result: PreflightResult) -> None:
    from config.app_config import CHROMA_PATH  # lazy import: live-config

    data_dir = os.path.dirname(CHROMA_PATH) or "."
    probe = os.path.join(data_dir, ".preflight_write_probe")
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        result.fatal.append(
            f"Data directory '{data_dir}' is not writable ({e}). "
            f"Daemon cannot persist memory there. Fix the directory permissions, "
            f"or point CHROMA_PATH at a writable location."
        )


def _check_llm_key(result: PreflightResult) -> None:
    # The app resolves OPENROUTER_API_KEY first, then OPENAI_API_KEY
    # (models/model_manager.py); the preflight must accept either (P3 F6).
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        result.warnings.append(
            "OPENROUTER_API_KEY (or OPENAI_API_KEY) is not set — chat will fail "
            "with [AUTH ERROR] unless a local model is configured. Add it to the "
            ".env file (OPENROUTER_API_KEY=sk-or-...) or re-run the setup wizard: "
            "python main.py wizard"
        )
    elif _looks_like_placeholder(key):
        result.warnings.append(
            "The LLM API key (OPENROUTER_API_KEY / OPENAI_API_KEY) looks like a "
            "placeholder value — chat will fail with [AUTH ERROR]. Replace it "
            "with a real key in the .env file."
        )


def _check_web_search_key(result: PreflightResult) -> None:
    if not os.environ.get("TAVILY_API_KEY", ""):
        result.warnings.append(
            "TAVILY_API_KEY is not set — web search is disabled. "
            "Add it to the .env file to enable it."
        )


def _check_google_token(result: PreflightResult) -> None:
    """Disk-only: an existing Google token file that is expired with no
    refresh token cannot recover itself — every downstream Gmail/Calendar/
    Contacts read would otherwise silently return no results until the
    owner re-authorizes. `token_expired_no_refresh` is documented disk-only
    (no network call), matching this module's startup-safety contract.
    2026-09-27 (BC-47, BC-58, BC-70, BC-72)."""
    try:
        from core.actions.google_auth import get_google_auth  # lazy import: patch-point
    except ImportError:
        return

    auth = get_google_auth()
    if auth is None:
        return  # not configured — nothing to check

    if auth.token_expired_no_refresh:
        result.warnings.append(
            "Google token file exists but is expired with no refresh token — "
            "Gmail/Calendar/Contacts reads will silently return no results "
            "until you re-authorize: python scripts/reauth_google.py"
        )


def _check_spacy_model(result: PreflightResult) -> None:
    # The model installs as an importable package; find_spec avoids the
    # cost of importing spaCy itself here.
    if importlib.util.find_spec("spacy") is None:
        return  # spaCy not installed at all — optional dependency path
    if importlib.util.find_spec("en_core_web_sm") is None:
        result.warnings.append(
            "spaCy model 'en_core_web_sm' is not installed — fact extraction "
            "falls back to regex-only (lower quality). Install it with: "
            "python -m spacy download en_core_web_sm"
        )


_LOCK_LINE_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
_LOCK_FIX_COMMAND = "pip install -r requirements.txt -c requirements.lock.txt"


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _repo_lock_path() -> str:
    # Resolved from this module's location, never the cwd.
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "requirements.lock.txt")


def _installed_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_dependency_drift(result: PreflightResult) -> None:
    """One advisory warning when installed packages differ from the lock."""
    if getattr(sys, "frozen", False):
        return  # the frozen build carries its own bundled environment
    lock_path = _repo_lock_path()
    if not os.path.isfile(lock_path):
        return
    drifted = []
    with open(lock_path, encoding="utf-8") as fh:
        for line in fh:
            m = _LOCK_LINE_RE.match(line)
            if not m:
                continue
            name, want = m.group(1), m.group(2).split("+", 1)[0]
            have = _installed_version(_pep503(name))
            if have is None:
                drifted.append(f"{name}: not installed, lock {want}")
            elif have.split("+", 1)[0] != want:
                drifted.append(f"{name}: installed {have.split('+', 1)[0]}, lock {want}")
    if drifted:
        examples = "; ".join(drifted[:5])
        result.warnings.append(
            f"{len(drifted)} installed package(s) differ from requirements.lock.txt "
            f"({examples}). Behaviour may differ from the tested environment. "
            f"Fix: {_LOCK_FIX_COMMAND}"
        )


def run_preflight() -> PreflightResult:
    """Run all startup checks. Never raises; returns findings."""
    result = PreflightResult()
    for check in (_check_data_dir_writable, _check_llm_key,
                  _check_web_search_key, _check_google_token, _check_spacy_model,
                  _check_dependency_drift):
        try:
            check(result)
        except Exception as e:  # a broken check must not block startup itself
            name = getattr(check, "__name__", repr(check))
            logger.error(f"[Preflight] Check {name} failed to run: {e}")
    return result


def print_preflight(result: PreflightResult) -> None:
    """Render findings to console + log. Caller decides whether to exit."""
    for msg in result.warnings:
        print(f"[Preflight] WARNING: {msg}")
        logger.warning(f"[Preflight] {msg}")
    for msg in result.fatal:
        print(f"[Preflight] FATAL: {msg}")
        logger.critical(f"[Preflight] {msg}")
