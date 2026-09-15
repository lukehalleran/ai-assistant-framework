"""memory/memory_consolidator.py read sites raise RetrievalError instead of
silently returning empty (CGR-009 #117-#120). Deployed-function tests: the
real `_current_status_facts` / Obsidian reader / `generate_narrative_context`
methods on a `MemoryConsolidator` built with an explicit fake profile (never
a real UserProfile() or OBSIDIAN_VAULT_PATH); readers use tmp_path or a
monkeypatched `_get_obsidian_notes_path`; the model manager is a stub
`generate_once`.
"""

import asyncio
from pathlib import Path

import pytest

from memory.memory_consolidator import MemoryConsolidator
from utils.retrieval_outcome import RetrievalError

_MARKER = "SYNTH_MARKER_f12a_9e21bd"


class _FakeModelManager:
    def __init__(self):
        self.calls = 0

    async def generate_once(self, *a, **k):
        self.calls += 1
        return "The user has been doing fine this week."


def _cons(profile=None, model_manager=None):
    """A MemoryConsolidator with an explicit fake profile -- never triggers
    the lazy real UserProfile()."""
    return MemoryConsolidator(model_manager or _FakeModelManager(), user_profile=profile or object())


class TestCurrentStatusFacts:
    def test_profile_unavailable_raises_retrieval_error(self, monkeypatch):
        cons = _cons()
        monkeypatch.setattr(type(cons), "user_profile", property(lambda self: None))
        with pytest.raises(RetrievalError) as exc:
            cons._current_status_facts()
        assert exc.value.source == "status_facts"
        assert exc.value.reason == "profile_unavailable"

    def test_get_current_view_raising_raises_retrieval_error(self):
        class RaisingProfile:
            def get_current_view(self):
                raise ValueError(f"boom {_MARKER}")

        cons = _cons(profile=RaisingProfile())
        with pytest.raises(RetrievalError) as exc:
            cons._current_status_facts()
        assert exc.value.source == "status_facts"
        assert exc.value.reason == "ValueError"
        assert _MARKER not in str(exc.value)

    def test_healthy_profile_no_status_relations_returns_empty(self):
        class NoStatusProfile:
            def get_current_view(self):
                return {"identity": [{"relation": "likes", "value": "coffee", "is_current": True}]}

        cons = _cons(profile=NoStatusProfile())
        assert cons._current_status_facts() == []

    def test_healthy_profile_with_status_fact_returns_it(self):
        fact = {"relation": "enrolled_in", "value": "Course XYZ", "is_current": True}

        class HealthyProfile:
            def get_current_view(self):
                return {"identity": [fact]}

        cons = _cons(profile=HealthyProfile())
        assert cons._current_status_facts() == [fact]


def _make_layout(tmp_path, kind):
    """Build a minimal Obsidian notes layout for one reader kind and write
    its note file. Returns nothing -- the reader discovers it via glob."""
    if kind == "weekly":
        d = tmp_path / "Week 1 Jan 2026"
        d.mkdir()
        (d / "Week 1 Jan 2026 Summary.md").write_text(
            "---\ngenerated: 2026-01-05\n---\nBody text.", encoding="utf-8"
        )
    elif kind == "monthly":
        d = tmp_path / "January 2026"
        d.mkdir()
        (d / "January 2026 Summary.md").write_text(
            "---\ngenerated: 2026-01-05\n---\nBody text.", encoding="utf-8"
        )
    else:
        d = tmp_path / "Week 1 Jan 2026"
        d.mkdir()
        (d / "Mon Daily Note.md").write_text(
            "---\ndate: 2026-01-05\n---\nBody text.", encoding="utf-8"
        )


READER_CASES = [
    ("_read_obsidian_weekly_summaries", "obsidian_weekly", "weekly"),
    ("_read_obsidian_monthly_summaries", "obsidian_monthly", "monthly"),
    ("_read_obsidian_daily_notes", "obsidian_daily", "daily"),
]


class TestObsidianReaders:
    @pytest.mark.parametrize("method_name, source, kind", READER_CASES)
    def test_no_notes_path_returns_empty(self, method_name, source, kind, monkeypatch):
        cons = _cons()
        monkeypatch.setattr(cons, "_get_obsidian_notes_path", lambda: None)
        assert getattr(cons, method_name)() == []

    @pytest.mark.parametrize("method_name, source, kind", READER_CASES)
    def test_read_failure_raises_retrieval_error(self, method_name, source, kind, tmp_path, monkeypatch):
        _make_layout(tmp_path, kind)
        cons = _cons()
        monkeypatch.setattr(cons, "_get_obsidian_notes_path", lambda: str(tmp_path))

        def _raise(self, *a, **k):
            raise OSError(f"disk fail {_MARKER}")

        monkeypatch.setattr(Path, "read_text", _raise)
        with pytest.raises(RetrievalError) as exc:
            getattr(cons, method_name)()
        assert exc.value.source == source
        assert exc.value.reason == "OSError"
        assert _MARKER not in str(exc.value)

    @pytest.mark.parametrize("method_name, source, kind", READER_CASES)
    def test_healthy_tmp_note_is_parsed(self, method_name, source, kind, tmp_path, monkeypatch):
        _make_layout(tmp_path, kind)
        cons = _cons()
        monkeypatch.setattr(cons, "_get_obsidian_notes_path", lambda: str(tmp_path))
        result = getattr(cons, method_name)()
        assert len(result) == 1
        assert result[0]["source"] == source
        assert "Body text." in result[0]["content"]
        assert result[0]["timestamp"] == "2026-01-05"


class TestGenerateNarrativeContext:
    def test_raising_reader_returns_empty_and_skips_llm(self, monkeypatch, caplog):
        mm = _FakeModelManager()
        cons = _cons(model_manager=mm)
        monkeypatch.setattr(cons, "_current_status_facts", lambda: [])
        monkeypatch.setattr(cons, "_read_obsidian_monthly_summaries", lambda limit: [])

        def _raise(limit):
            raise RetrievalError(source="obsidian_weekly", reason="OSError")

        monkeypatch.setattr(cons, "_read_obsidian_weekly_summaries", _raise)
        monkeypatch.setattr(cons, "_read_obsidian_daily_notes", lambda limit: [])

        with caplog.at_level("WARNING"):
            result = asyncio.run(cons.generate_narrative_context())

        assert result == ""
        assert mm.calls == 0
        assert "Narrative not regenerated" in caplog.text
        assert "obsidian_weekly" in caplog.text and "OSError" in caplog.text

    def test_raising_status_facts_returns_empty_and_skips_llm(self, monkeypatch, caplog):
        mm = _FakeModelManager()
        cons = _cons(model_manager=mm)
        monkeypatch.setattr(cons, "_read_obsidian_monthly_summaries", lambda limit: [])
        monkeypatch.setattr(
            cons, "_read_obsidian_weekly_summaries",
            lambda limit: [{"content": "week", "timestamp": "2026-01-01"}],
        )
        monkeypatch.setattr(cons, "_read_obsidian_daily_notes", lambda limit: [])

        def _raise():
            raise RetrievalError(source="status_facts", reason="profile_unavailable")

        monkeypatch.setattr(cons, "_current_status_facts", _raise)

        with caplog.at_level("WARNING"):
            result = asyncio.run(cons.generate_narrative_context())

        assert result == ""
        assert mm.calls == 0
        assert "Narrative not regenerated" in caplog.text
        assert "status_facts" in caplog.text and "profile_unavailable" in caplog.text

    def test_all_sources_empty_returns_empty(self, monkeypatch):
        mm = _FakeModelManager()
        cons = _cons(model_manager=mm)
        monkeypatch.setattr(cons, "_read_obsidian_monthly_summaries", lambda limit: [])
        monkeypatch.setattr(cons, "_read_obsidian_weekly_summaries", lambda limit: [])
        monkeypatch.setattr(cons, "_read_obsidian_daily_notes", lambda limit: [])

        result = asyncio.run(cons.generate_narrative_context())
        assert result == ""
        assert mm.calls == 0

    def test_healthy_inputs_keep_status_guard_and_remove_conflict(self, monkeypatch):
        class ConflictingModelManager:
            def __init__(self):
                self.calls = 0

            async def generate_once(self, prompt, **kwargs):
                self.calls += 1
                assert "AUTHORITATIVE CURRENT FACTS" in prompt
                assert "enrolled_in = Course XYZ" in prompt
                return "The user withdrew from the fall semester after a health scare."

        class HealthyProfile:
            def get_current_view(self):
                return {
                    "identity": [
                        {"relation": "enrolled_in", "value": "Course XYZ", "is_current": True},
                    ]
                }

        mm = ConflictingModelManager()
        cons = _cons(profile=HealthyProfile(), model_manager=mm)
        monkeypatch.setattr(cons, "_read_obsidian_monthly_summaries", lambda limit: [])
        monkeypatch.setattr(
            cons, "_read_obsidian_weekly_summaries",
            lambda limit: [{"content": "week", "timestamp": "2026-01-01"}],
        )
        monkeypatch.setattr(cons, "_read_obsidian_daily_notes", lambda limit: [])

        result = asyncio.run(cons.generate_narrative_context())
        assert mm.calls == 1
        body = result.split("[CAUTION")[0]
        assert "withdrew" not in body
        assert "enrolled_in=Course XYZ" in result
