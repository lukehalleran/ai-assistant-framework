"""
Unit tests for daily note auto-update logic in utils/daily_notes_generator.py

Tests the new behavior where daily notes are regenerated when later sessions
add significantly more conversations than the existing note captured.
"""

import pytest
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, MagicMock, AsyncMock

from utils.daily_notes_generator import DailyNotesGenerator, GenerationResult


# A long enough LLM response to pass the 100-char minimum check
_MOCK_LLM_RESPONSE = (
    "## Summary\nLuke had a full day of conversations covering health recovery and technical work.\n\n"
    "## Main Quest: Recovery and Return\n- Recovered from illness\n- Resumed normal routine\n\n"
    "## Side Quests\nNone today.\n\n## Life Events\n- **Health**: Feeling better.\n\n"
    "## Emotional State\nUpbeat and ready to go.\n\n## Key Decisions\nNone.\n\n"
    "## Knowledge Gained\nNone.\n\n## Open Threads\nAll resolved.\n\n## Intensity: 3/10\n"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vault_path(tmp_path):
    """Create a temporary vault structure."""
    vault = tmp_path / "vault"
    vault.mkdir()
    daily = vault / "Daily"
    daily.mkdir()
    return vault


@pytest.fixture
def generator(vault_path):
    """Create a DailyNotesGenerator with temp vault, no lazy-loading."""
    mock_corpus = MagicMock()
    mock_corpus.corpus = []
    mock_model = MagicMock()

    with patch("utils.daily_notes_generator.DailyNotesGenerator.__init__", lambda self, **kw: None):
        gen = DailyNotesGenerator.__new__(DailyNotesGenerator)

    # Set required attributes directly
    gen._corpus_manager = mock_corpus
    gen._model_manager = mock_model
    gen._tag_generator = None
    gen.vault_path = vault_path
    gen.output_dir = vault_path / "Daily Notes and To Do's"
    gen.output_dir.mkdir(parents=True)
    gen.enabled = True
    gen.daily_folder = "Daily Notes and To Do's"
    gen.model_name = "sonnet-4.5"
    gen.max_tokens = 800
    gen.tag_generation_enabled = False

    return gen


def _write_note(gen, target_date: date, conversations: int = 1) -> Path:
    """Helper: write a minimal daily note with given conversation count."""
    filename = gen._format_filename(target_date)
    month_folder = gen._get_month_folder_name(target_date)
    week_folder = gen._get_week_folder_name(target_date)

    note_dir = gen.output_dir / month_folder / week_folder
    note_dir.mkdir(parents=True, exist_ok=True)
    note_path = note_dir / filename

    content = f"""---
date: {target_date.isoformat()}
usage_intensity: 3
conversations: {conversations}
span_hours: 1.0
active_hours: 0.5
main_quest: "Test Quest"
tags: ["daily", "daemon-generated"]
generated: {datetime.now().isoformat()}
---

# Daily Note - {target_date.strftime('%B %d, %Y')}

## Summary
Test note with {conversations} conversations.
"""
    note_path.write_text(content, encoding='utf-8')
    return note_path


# =============================================================================
# _find_existing_note_path tests
# =============================================================================

class TestFindExistingNotePath:
    """Tests for resolving the actual path of an existing note."""

    def test_finds_note_in_monthly_weekly_layout(self, generator):
        """Should find note in current monthly/weekly structure."""
        target = date(2026, 5, 18)
        written = _write_note(generator, target, conversations=3)
        found = generator._find_existing_note_path(target)
        assert found == written

    def test_returns_none_when_no_note(self, generator):
        """Should return None when no note exists."""
        target = date(2026, 5, 18)
        assert generator._find_existing_note_path(target) is None

    def test_finds_note_in_flat_layout(self, generator):
        """Should find note at flat directory (legacy layout)."""
        target = date(2026, 5, 18)
        filename = generator._format_filename(target)
        flat_path = generator.output_dir / filename
        flat_path.write_text("---\nconversations: 2\n---\nTest\n", encoding='utf-8')
        found = generator._find_existing_note_path(target)
        assert found == flat_path

    def test_finds_note_in_weekly_at_root_layout(self, generator):
        """Should find note at weekly-at-root (legacy layout)."""
        target = date(2026, 5, 18)
        filename = generator._format_filename(target)
        week_folder = generator._get_week_folder_name(target)
        weekly_path = generator.output_dir / week_folder / filename
        weekly_path.parent.mkdir(parents=True)
        weekly_path.write_text("---\nconversations: 2\n---\nTest\n", encoding='utf-8')
        found = generator._find_existing_note_path(target)
        assert found == weekly_path

    def test_prefers_monthly_weekly_over_legacy(self, generator):
        """Monthly/weekly layout should be checked first."""
        target = date(2026, 5, 18)
        canonical = _write_note(generator, target, conversations=3)

        # Also create flat copy
        filename = generator._format_filename(target)
        flat_path = generator.output_dir / filename
        flat_path.write_text("---\nconversations: 1\n---\nOld\n", encoding='utf-8')

        found = generator._find_existing_note_path(target)
        assert found == canonical


# =============================================================================
# _get_existing_conversation_count tests
# =============================================================================

class TestGetExistingConversationCount:
    """Tests for reading conversation count from existing note frontmatter."""

    def test_reads_count_from_frontmatter(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=7)
        assert generator._get_existing_conversation_count(target) == 7

    def test_returns_zero_when_no_note(self, generator):
        target = date(2026, 5, 18)
        assert generator._get_existing_conversation_count(target) == 0

    def test_returns_zero_for_malformed_frontmatter(self, generator):
        target = date(2026, 5, 18)
        note_path = _write_note(generator, target)
        note_path.write_text("No frontmatter here\nJust text\n", encoding='utf-8')
        assert generator._get_existing_conversation_count(target) == 0

    def test_returns_zero_for_missing_conversations_field(self, generator):
        target = date(2026, 5, 18)
        note_path = _write_note(generator, target)
        note_path.write_text("---\ndate: 2026-05-18\n---\nNo count\n", encoding='utf-8')
        assert generator._get_existing_conversation_count(target) == 0

    def test_handles_conversations_with_spaces(self, generator):
        """Frontmatter might have spaces around the value."""
        target = date(2026, 5, 18)
        note_path = _write_note(generator, target)
        note_path.write_text("---\nconversations:  12 \n---\n", encoding='utf-8')
        assert generator._get_existing_conversation_count(target) == 12


# =============================================================================
# _should_auto_update tests
# =============================================================================

class TestShouldAutoUpdate:
    """Tests for auto-update threshold logic."""

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_triggers_when_delta_meets_threshold(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=1)
        assert generator._should_auto_update(target, current_count=4) is True

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_triggers_when_delta_exceeds_threshold(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=1)
        assert generator._should_auto_update(target, current_count=10) is True

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_skips_when_delta_below_threshold(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=5)
        assert generator._should_auto_update(target, current_count=7) is False

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_skips_when_same_count(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=5)
        assert generator._should_auto_update(target, current_count=5) is False

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_skips_when_no_existing_note(self, generator):
        """No existing note means count=0, so any conversations trigger update.
        But this path is only reached when note_exists() is True."""
        target = date(2026, 5, 18)
        # No note written — _get_existing_conversation_count returns 0
        # 3 new conversations = delta of 3 >= threshold 3
        assert generator._should_auto_update(target, current_count=3) is True

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 1)
    def test_custom_threshold(self, generator):
        """Even 1 new conversation triggers if threshold is 1."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=2)
        assert generator._should_auto_update(target, current_count=3) is True

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_exact_boundary(self, generator):
        """Exactly at threshold should trigger."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=4)
        # delta = 7 - 4 = 3, threshold = 3
        assert generator._should_auto_update(target, current_count=7) is True

    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    def test_one_below_boundary(self, generator):
        """One below threshold should not trigger."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=4)
        # delta = 6 - 4 = 2, threshold = 3
        assert generator._should_auto_update(target, current_count=6) is False


# =============================================================================
# note_exists integration with _find_existing_note_path
# =============================================================================

class TestNoteExists:
    """Verify note_exists() delegates to _find_existing_note_path correctly."""

    def test_returns_true_when_note_exists(self, generator):
        target = date(2026, 5, 18)
        _write_note(generator, target)
        assert generator.note_exists(target) is True

    def test_returns_false_when_no_note(self, generator):
        target = date(2026, 5, 18)
        assert generator.note_exists(target) is False


# =============================================================================
# generate_for_date auto-update integration
# =============================================================================

class TestGenerateForDateAutoUpdate:
    """Test that generate_for_date uses auto-update when note exists with fewer conversations."""

    @pytest.mark.asyncio
    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    async def test_auto_update_proceeds_when_threshold_met(self, generator):
        """When existing note has 1 conversation and corpus now has 5, should proceed."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=1)

        # Populate corpus with 5 conversations for the date
        convos = []
        for i in range(5):
            convos.append({
                "timestamp": datetime(2026, 5, 18, 9 + i, 0, 0),
                "query": f"Message {i}",
                "response": f"Response {i}",
            })
        generator._corpus_manager.corpus = convos

        # Mock the LLM call so we don't actually call an API
        generator._model_manager.generate_once = AsyncMock(return_value=_MOCK_LLM_RESPONSE)

        result = await generator.generate_for_date(target)

        # Should have proceeded (not skipped)
        assert result.skipped_reason is None
        assert result.success is True
        assert result.conversation_count == 5

    @pytest.mark.asyncio
    @patch("config.app_config.DAILY_NOTES_UPDATE_MIN_NEW", 3)
    async def test_no_auto_update_when_below_threshold(self, generator):
        """When existing note has 3 conversations and corpus now has 4, should skip."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=3)

        convos = []
        for i in range(4):
            convos.append({
                "timestamp": datetime(2026, 5, 18, 9 + i, 0, 0),
                "query": f"Message {i}",
                "response": f"Response {i}",
            })
        generator._corpus_manager.corpus = convos

        result = await generator.generate_for_date(target)

        assert result.skipped_reason == "already_exists"

    @pytest.mark.asyncio
    async def test_force_overrides_auto_update_check(self, generator):
        """force=True should always regenerate regardless of conversation count."""
        target = date(2026, 5, 18)
        _write_note(generator, target, conversations=10)

        convos = [{"timestamp": datetime(2026, 5, 18, 9, 0, 0), "query": "Hi", "response": "Hey"}]
        generator._corpus_manager.corpus = convos

        generator._model_manager.generate_once = AsyncMock(return_value=_MOCK_LLM_RESPONSE)

        result = await generator.generate_for_date(target, force=True)
        assert result.skipped_reason is None
