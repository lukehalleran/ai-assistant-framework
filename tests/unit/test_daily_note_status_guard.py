"""utils/daily_notes_generator.py never writes a daily note without its
status-claim guard (BC-20, BC-47, CM-05): `_current_status_facts` raises a
typed `RetrievalError` instead of silently returning `[]`, and
`generate_for_date` treats that as a hard stop -- before any LLM call, tag
generation, note write or narrative refresh -- returning the same
`GenerationResult` object with `success=False` and the constant label
`error="status_guard_unavailable"`. This mirrors the shape F12a gave
`memory/memory_consolidator.py._current_status_facts` for the sibling read
site named in failure_outcome_design.md
("duplicate at utils/daily_notes_generator.py:440-452").

Deployed-function tests: the real `DailyNotesGenerator`, built with an
explicit fake profile/corpus/model manager and a tmp_path vault (never a
real UserProfile() or OBSIDIAN_VAULT_PATH). `_trigger_narrative_refresh` is
stubbed on every instance -- it is not under test here, only the
status-facts read and the guard it feeds are.
"""

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.daily_notes_generator import DailyNotesGenerator
from utils.retrieval_outcome import RetrievalError

_MARKER = "SYNTH_MARKER_f12b_7c3a1d"

# Long enough to pass generate_for_date's 100-char minimum-response check.
_MOCK_LLM_RESPONSE = (
    "## Summary\nA quiet day of synthetic conversations about test coverage.\n\n"
    "## Main Quest: Testing\n- Verified the status guard\n\n"
    "## Side Quests\nNone today.\n\n## Life Events\n- **Work**: Wrote tests.\n\n"
    "## Emotional State\nFocused.\n\n## Key Decisions\nNone.\n\n"
    "## Knowledge Gained\nNone.\n\n## Open Threads\nAll resolved.\n\n## Intensity: 2/10\n"
)


class _FakeModelManager:
    """Stub model_manager -- the first model in the fallback list always
    succeeds, so generate_for_date never needs to walk the fallback list."""

    def __init__(self, response=_MOCK_LLM_RESPONSE):
        self.calls = 0
        self.last_prompt = None
        self._response = response

    async def generate_once(self, prompt, **kwargs):
        self.calls += 1
        self.last_prompt = prompt
        return self._response


class _RaisingProfile:
    def get_current_view(self):
        raise ValueError(f"boom {_MARKER}")


class _HealthyProfile:
    def __init__(self, facts=None):
        self._facts = facts or {}

    def get_current_view(self):
        return self._facts


def _gen(tmp_path, profile=None, model_manager=None, convos=None):
    """A DailyNotesGenerator built with explicit fakes -- never triggers the
    lazy real UserProfile(), a real MemoryConsolidator, or OBSIDIAN_VAULT_PATH.
    tag generation is disabled so the tag_generator property (which would
    lazy-load a real TagGenerator) is never touched, and
    `_trigger_narrative_refresh` is replaced outright, since it is not the
    method under test in this file."""
    corpus = MagicMock()
    corpus.corpus = convos if convos is not None else []
    gen = DailyNotesGenerator(
        corpus_manager=corpus,
        model_manager=model_manager or _FakeModelManager(),
        vault_path=str(tmp_path / "vault"),
        user_profile=profile if profile is not None else object(),
    )
    gen.tag_generation_enabled = False
    gen._trigger_narrative_refresh = AsyncMock(return_value=None)
    return gen


def _convos_for(target_date: date):
    return [{
        "timestamp": datetime(target_date.year, target_date.month, target_date.day, 9, 0, 0),
        "query": "Hi",
        "response": "Hey",
    }]


class TestCurrentStatusFacts:
    """Unit-level: the read site itself."""

    def test_profile_none_raises_profile_unavailable(self, tmp_path, monkeypatch):
        gen = _gen(tmp_path)
        monkeypatch.setattr(type(gen), "user_profile", property(lambda self: None))
        with pytest.raises(RetrievalError) as exc:
            gen._current_status_facts()
        assert exc.value.source == "status_facts"
        assert exc.value.reason == "profile_unavailable"

    def test_get_current_view_raising_raises_retrieval_error(self, tmp_path):
        gen = _gen(tmp_path, profile=_RaisingProfile())
        with pytest.raises(RetrievalError) as exc:
            gen._current_status_facts()
        assert exc.value.source == "status_facts"
        assert exc.value.reason == "ValueError"
        assert _MARKER not in str(exc.value)

    def test_healthy_profile_no_status_relations_returns_empty(self, tmp_path):
        profile = _HealthyProfile({"identity": [{"relation": "likes", "value": "coffee", "is_current": True}]})
        gen = _gen(tmp_path, profile=profile)
        assert gen._current_status_facts() == []

    def test_healthy_profile_with_status_fact_returns_it(self, tmp_path):
        fact = {"relation": "enrolled_in", "value": "Course XYZ", "is_current": True}
        gen = _gen(tmp_path, profile=_HealthyProfile({"identity": [fact]}))
        assert gen._current_status_facts() == [fact]


class TestGenerateForDateStatusGuard:
    """Integration-level: `generate_for_date` reacts to the typed failure."""

    @pytest.mark.asyncio
    async def test_raising_profile_blocks_generation_before_any_write(self, tmp_path, caplog):
        target = date(2026, 5, 20)
        mm = _FakeModelManager()
        gen = _gen(tmp_path, profile=_RaisingProfile(), model_manager=mm, convos=_convos_for(target))

        with caplog.at_level("WARNING"):
            result = await gen.generate_for_date(target)

        assert result.success is False
        assert result.error == "status_guard_unavailable"
        assert result.skipped_reason is None
        assert mm.calls == 0
        gen._trigger_narrative_refresh.assert_not_awaited()
        assert not gen._get_note_path(target).exists()
        assert _MARKER not in (result.error or "")
        assert _MARKER not in caplog.text
        assert "status_facts" in caplog.text and "ValueError" in caplog.text

    @pytest.mark.asyncio
    async def test_profile_unavailable_blocks_generation(self, tmp_path, monkeypatch):
        target = date(2026, 5, 20)
        mm = _FakeModelManager()
        gen = _gen(tmp_path, model_manager=mm, convos=_convos_for(target))
        monkeypatch.setattr(type(gen), "user_profile", property(lambda self: None))

        result = await gen.generate_for_date(target)

        assert result.success is False
        assert result.error == "status_guard_unavailable"
        assert mm.calls == 0
        gen._trigger_narrative_refresh.assert_not_awaited()
        assert not gen._get_note_path(target).exists()

    @pytest.mark.asyncio
    async def test_healthy_status_facts_writes_note_with_guard_block(self, tmp_path):
        """Control: a healthy status-facts read still writes the note, and
        the guard block reaches the prompt sent to the model."""
        target = date(2026, 5, 20)
        mm = _FakeModelManager()
        fact = {"relation": "enrolled_in", "value": "Course XYZ", "is_current": True}
        gen = _gen(tmp_path, profile=_HealthyProfile({"identity": [fact]}), model_manager=mm,
                   convos=_convos_for(target))

        result = await gen.generate_for_date(target)

        assert result.success is True
        assert result.error is None
        assert mm.calls == 1
        assert "AUTHORITATIVE CURRENT FACTS" in mm.last_prompt
        assert "enrolled_in = Course XYZ" in mm.last_prompt
        gen._trigger_narrative_refresh.assert_awaited_once()
        assert result.output_path is not None and result.output_path.exists()
