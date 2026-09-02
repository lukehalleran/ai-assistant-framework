"""Tests for utils/preflight.py — startup checks with actionable messages."""

import os
from unittest.mock import patch

import pytest

from utils.preflight import (
    PreflightResult,
    _check_data_dir_writable,
    _check_llm_key,
    _check_spacy_model,
    _check_web_search_key,
    run_preflight,
)


class TestLLMKeyCheck:
    def test_missing_key_warns_with_fix(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = PreflightResult()
        _check_llm_key(result)
        assert len(result.warnings) == 1
        assert "wizard" in result.warnings[0]
        assert not result.fatal  # local-model setups are legitimate

    def test_placeholder_key_warns(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "your_key_here")
        result = PreflightResult()
        _check_llm_key(result)
        assert len(result.warnings) == 1
        assert "placeholder" in result.warnings[0]

    def test_real_key_passes(self, monkeypatch):
        realistic_key = "sk-" + "or-" + "v1-" + "abc123" + "def456"
        monkeypatch.setenv("OPENAI_API_KEY", realistic_key)
        result = PreflightResult()
        _check_llm_key(result)
        assert not result.warnings
        assert not result.fatal

    def test_long_key_with_fragment_substring_passes(self, monkeypatch):
        """A real key whose random section contains e.g. 'test-key' must not
        warn (observed false positive 2026-07-14 on a working 35-char key)."""
        realistic_key = (
            "sk-" + "or-" + "v1-" + "a1" + "test-key" + "b2c3d4" + "e5f6a7" + "b8c9d0"
        )
        monkeypatch.setenv("OPENAI_API_KEY", realistic_key)
        result = PreflightResult()
        _check_llm_key(result)
        assert not result.warnings

    def test_short_placeholder_still_warns(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        result = PreflightResult()
        _check_llm_key(result)
        assert len(result.warnings) == 1


class TestWebSearchKeyCheck:
    def test_missing_tavily_warns(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = PreflightResult()
        _check_web_search_key(result)
        assert len(result.warnings) == 1
        assert "web search" in result.warnings[0]

    def test_present_tavily_passes(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-abc")
        result = PreflightResult()
        _check_web_search_key(result)
        assert not result.warnings


class TestDataDirCheck:
    def test_writable_dir_passes(self, tmp_path):
        with patch("config.app_config.CHROMA_PATH", str(tmp_path / "chroma")):
            result = PreflightResult()
            _check_data_dir_writable(result)
        assert not result.fatal

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
    def test_unwritable_dir_is_fatal(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        os.chmod(locked, 0o500)
        try:
            with patch("config.app_config.CHROMA_PATH", str(locked / "chroma")):
                result = PreflightResult()
                _check_data_dir_writable(result)
            assert len(result.fatal) == 1
            assert "not writable" in result.fatal[0]
        finally:
            os.chmod(locked, 0o700)


class TestSpacyModelCheck:
    def test_missing_model_warns(self):
        def fake_find_spec(name):
            return object() if name == "spacy" else None

        with patch("utils.preflight.importlib.util.find_spec", side_effect=fake_find_spec):
            result = PreflightResult()
            _check_spacy_model(result)
        assert len(result.warnings) == 1
        assert "en_core_web_sm" in result.warnings[0]

    def test_no_spacy_at_all_is_silent(self):
        with patch("utils.preflight.importlib.util.find_spec", return_value=None):
            result = PreflightResult()
            _check_spacy_model(result)
        assert not result.warnings


class TestRunPreflight:
    def test_never_raises_when_check_breaks(self):
        with patch("utils.preflight._check_llm_key", side_effect=RuntimeError("boom")):
            result = run_preflight()  # must not raise
        assert isinstance(result, PreflightResult)

    def test_ok_property(self):
        result = PreflightResult()
        assert result.ok
        result.warnings.append("w")
        assert result.ok  # warnings don't block startup
        result.fatal.append("f")
        assert not result.ok
