"""2026-09-03 — Codex audit follow-ups implemented by Fable.

1. A parseable-but-structurally-invalid user profile is quarantined and aborts
   (same class as corrupt JSON — never continue with state a save would wipe).
4. Routine prompt/token/STM/image diagnostics are DEBUG, not WARNING, and the
   STM value dump is redacted.
6. An absent external FAISS index is an explicit disabled state (INFO), not an
   ERROR.
7. E2B sandbox cleanup treats "already gone" (404 / NotFoundException) as
   idempotent success.
8. Unused imports flagged by vulture removed (modules still import).
"""
import json
import logging
import re
from pathlib import Path

import pytest

from memory.user_profile import profile_shape_error
from utils.safe_json import CorruptStoreError

REPO = Path(__file__).resolve().parents[2]


# ── 1. profile shape validation ───────────────────────────────────────────
class TestProfileShapeValidation:
    def test_shape_error_messages(self):
        assert profile_shape_error([]) .startswith("top-level JSON is list")
        assert "categories" in profile_shape_error({"categories": []})
        assert "raw_log" in profile_shape_error({"categories": {}, "raw_log": {}})
        assert "quick_profile" in profile_shape_error({"quick_profile": "x"})
        assert "categories['health']" in profile_shape_error({"categories": {"health": {}}})
        assert profile_shape_error({"categories": {"health": []}, "raw_log": [], "quick_profile": {}}) == ""
        assert profile_shape_error({}) == ""  # legacy minimal file: containers defaulted

    def test_wrong_shape_file_is_quarantined_and_aborts(self, tmp_path):
        from memory.user_profile import UserProfile
        p = tmp_path / "user_profile.json"
        p.write_text(json.dumps(["not", "a", "profile"]))
        with pytest.raises(CorruptStoreError):
            UserProfile(profile_path=str(p))
        assert list(tmp_path.glob("user_profile.json.corrupt-*")), "quarantine copy missing"
        # the original is never replaced by an empty profile
        assert not p.exists() or p.read_text().startswith("[")

    def test_valid_file_loads(self, tmp_path):
        from memory.user_profile import UserProfile
        p = tmp_path / "user_profile.json"
        p.write_text(json.dumps({"user_id": "t", "quick_profile": {}, "categories": {}, "raw_log": []}))
        prof = UserProfile(profile_path=str(p))
        assert prof.profile["user_id"] == "t"


# ── 4. log-noise demotion ─────────────────────────────────────────────────
_ROUTINE = re.compile(
    r'logger\.warning\(f?"(?:PROMPT ASSEMBLY|PROMPT BUILD|STM RENDERING|\[DEBUG RECENT\]|'
    r'\[TOKEN BUDGET\] (?:Processing|Kept|Stopped|Preserving)|\[PromptBuilder\] IMAGE DEBUG|'
    r'\[ContextGatherer\] IMAGE DEBUG)'
)


class TestRoutineDiagnosticsAreDebug:
    @pytest.mark.parametrize("rel", [
        "core/prompt/token_manager.py", "core/prompt/formatter.py", "core/prompt/builder.py",
        "core/prompt/gatherer_memory.py", "core/prompt/gatherer_knowledge.py",
    ])
    def test_no_routine_warning_sites(self, rel):
        src = (REPO / rel).read_text()
        assert not _ROUTINE.search(src), rel

    def test_stm_value_is_not_logged(self):
        src = (REPO / "core/prompt/formatter.py").read_text()
        assert "stm_summary = {stm_summary}" not in src
        assert "value = {context.get('stm_summary')}" not in src

    def test_real_degradation_stays_warning(self):
        src = (REPO / "core/prompt/token_manager.py").read_text()
        assert "Still over budget after 3 trim passes" in src
        assert re.search(r'logger\.warning\(\s*f"\[TOKEN BUDGET\] Still over budget', src)


# ── 6. external FAISS absence ─────────────────────────────────────────────
class TestSemanticIndexAbsenceIsExplicit:
    def test_missing_index_is_info_not_error(self, monkeypatch, tmp_path, caplog):
        import knowledge.semantic_search as ss
        monkeypatch.setattr(ss, "INDEX_PATH", str(tmp_path / "none.faiss"))
        monkeypatch.setattr(ss, "META_PATH", str(tmp_path / "none.parquet"))
        monkeypatch.setattr(ss, "_warned_missing", False)
        idx = ss.SemanticSearchIndex()
        with caplog.at_level(logging.INFO, logger=ss.logger.name):
            idx.load()
        assert idx.loaded is False
        assert "external index not present" in idx.disabled_reason
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors
        assert any("DISABLED" in r.getMessage() for r in caplog.records)


# ── 7. E2B idempotent cleanup ─────────────────────────────────────────────
class TestSandboxCleanupIdempotent:
    def test_not_found_is_success(self, caplog):
        from knowledge.sandbox_manager import _kill_sandbox_quietly

        class NotFoundException(Exception):
            pass

        class _Gone:
            def kill(self):
                raise NotFoundException("sandbox not found")

        class _Http404:
            def kill(self):
                raise RuntimeError("DELETE /sandboxes/x returned 404")

        class _Broken:
            def kill(self):
                raise RuntimeError("connection reset")

        with caplog.at_level(logging.DEBUG):
            assert _kill_sandbox_quietly(_Gone(), "s") is True
            assert _kill_sandbox_quietly(_Http404(), "s") is True
            assert _kill_sandbox_quietly(_Broken(), "s") is False
        assert any(r.levelno == logging.WARNING and "connection reset" in r.getMessage()
                   for r in caplog.records)
        assert not any(r.levelno >= logging.WARNING and "not found" in r.getMessage().lower()
                       for r in caplog.records)


# ── 8. dead imports removed ───────────────────────────────────────────────
class TestDeadImportsRemoved:
    def test_modules_import(self):
        import eval.snapshots  # noqa: F401
        import knowledge.synthesis_filter  # noqa: F401
        import models.model_manager as mm
        assert not hasattr(mm, "_openai_module")
