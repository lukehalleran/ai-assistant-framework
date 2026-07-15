"""
Module Contract — utils/preflight.py

Purpose:
    Startup preflight checks for full-orchestrator modes (gui/cli). Turns
    "cryptic failure on first use" into an actionable message at launch:
      - data directory not writable        → FATAL (memory could not persist)
      - OPENAI_API_KEY missing/placeholder → WARNING (chat will return
        [AUTH ERROR]; local-model setups are legitimate, so not fatal)
      - TAVILY_API_KEY missing             → NOTE (web search disabled)
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

import importlib.util
import os
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
    from config.app_config import CHROMA_PATH

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
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        result.warnings.append(
            "OPENAI_API_KEY is not set — chat will fail with [AUTH ERROR] unless "
            "a local model is configured. Add it to the .env file "
            "(OPENAI_API_KEY=sk-or-...) or re-run the setup wizard: "
            "python main.py wizard"
        )
    elif _looks_like_placeholder(key):
        result.warnings.append(
            "OPENAI_API_KEY looks like a placeholder value — chat will fail with "
            "[AUTH ERROR]. Replace it with a real key in the .env file."
        )


def _check_web_search_key(result: PreflightResult) -> None:
    if not os.environ.get("TAVILY_API_KEY", ""):
        result.warnings.append(
            "TAVILY_API_KEY is not set — web search is disabled. "
            "Add it to the .env file to enable it."
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


def run_preflight() -> PreflightResult:
    """Run all startup checks. Never raises; returns findings."""
    result = PreflightResult()
    for check in (_check_data_dir_writable, _check_llm_key,
                  _check_web_search_key, _check_spacy_model):
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
