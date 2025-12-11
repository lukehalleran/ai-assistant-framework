"""
tests/test_wizard.py

Tests for onboarding wizard state machine, validation, and profile integration.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from gui.wizard import (
    WizardState,
    WizardStep,
    process_wizard_message,
    get_welcome_message,
    write_api_key_to_env,
    validate_api_key_format,
    parse_style_preference,
    is_skip,
)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.model_manager = Mock()
    orchestrator.model_manager.generate_once = AsyncMock(return_value="OK")
    orchestrator.user_profile = None
    return orchestrator


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


class TestWizardStep:
    """Test WizardStep enum."""

    def test_wizard_steps_exist(self):
        """All required wizard steps should exist."""
        assert WizardStep.WELCOME
        assert WizardStep.API_KEY
        assert WizardStep.STYLE
        assert WizardStep.NAME
        assert WizardStep.PRONOUNS
        assert WizardStep.BACKGROUND
        assert WizardStep.COMPLETE


class TestWizardState:
    """Test WizardState dataclass."""

    def test_wizard_state_defaults(self):
        """WizardState should have sensible defaults."""
        state = WizardState()
        assert state.step == WizardStep.WELCOME
        assert state.collected_data == {}
        assert state.error_count == 0
        assert state.max_retries == 3

    def test_wizard_state_mutable_collected_data(self):
        """WizardState.collected_data should be mutable."""
        state = WizardState()
        state.collected_data['test'] = 'value'
        assert state.collected_data['test'] == 'value'


class TestValidateApiKeyFormat:
    """Test API key format validation."""

    def test_validate_api_key_format_valid(self):
        """Valid OpenRouter key should pass validation."""
        assert validate_api_key_format("sk-or-v1-abc123def456ghi789") is True

    def test_validate_api_key_format_invalid_prefix(self):
        """Key without 'sk-or-' prefix should fail."""
        assert validate_api_key_format("sk-abc123def456") is False
        assert validate_api_key_format("or-abc123def456") is False

    def test_validate_api_key_format_too_short(self):
        """Key too short should fail even with correct prefix."""
        assert validate_api_key_format("sk-or-short") is False

    def test_validate_api_key_format_with_whitespace(self):
        """Key with leading/trailing whitespace should be stripped and validated."""
        assert validate_api_key_format("  sk-or-v1-abc123def456ghi789  ") is True


class TestWriteApiKeyToEnv:
    """Test writing API key to .env file."""

    def test_write_api_key_creates_new_file(self, temp_env_file):
        """Should create new .env file if it doesn't exist."""
        os.remove(temp_env_file)  # Start with no file

        with patch('gui.wizard.Path') as mock_path:
            mock_path.return_value = Path(temp_env_file)
            result = write_api_key_to_env("sk-or-test-key-12345")

        assert result is True
        assert os.path.exists(temp_env_file)

        with open(temp_env_file, 'r') as f:
            content = f.read()
        assert 'OPENAI_API_KEY=sk-or-test-key-12345' in content

    def test_write_api_key_updates_existing_file(self, temp_env_file):
        """Should update existing OPENAI_API_KEY line in .env."""
        # Create existing .env with old key
        with open(temp_env_file, 'w') as f:
            f.write('OPENAI_API_KEY=old-key\n')
            f.write('OTHER_VAR=value\n')

        with patch('gui.wizard.Path') as mock_path:
            mock_path.return_value = Path(temp_env_file)
            result = write_api_key_to_env("sk-or-new-key-67890")

        assert result is True

        with open(temp_env_file, 'r') as f:
            lines = f.readlines()

        assert 'OPENAI_API_KEY=sk-or-new-key-67890\n' in lines
        assert 'OTHER_VAR=value\n' in lines

    def test_write_api_key_sets_os_environ(self):
        """Should set os.environ['OPENAI_API_KEY'] for immediate effect."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            temp_path = f.name

        try:
            with patch('gui.wizard.Path') as mock_path:
                mock_path.return_value = Path(temp_path)
                write_api_key_to_env("sk-or-test-key-99999")

            assert os.environ.get('OPENAI_API_KEY') == 'sk-or-test-key-99999'
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestParseStylePreference:
    """Test style preference parsing."""

    def test_parse_style_numeric_warm(self):
        """Numeric input '1' should map to 'warm'."""
        assert parse_style_preference("1") == "warm"

    def test_parse_style_numeric_balanced(self):
        """Numeric input '2' should map to 'balanced'."""
        assert parse_style_preference("2") == "balanced"

    def test_parse_style_numeric_direct(self):
        """Numeric input '3' should map to 'direct'."""
        assert parse_style_preference("3") == "direct"

    def test_parse_style_text_warm(self):
        """Text 'warm' should map to 'warm'."""
        assert parse_style_preference("warm") == "warm"
        assert parse_style_preference("warm & supportive") == "warm"
        assert parse_style_preference("supportive") == "warm"

    def test_parse_style_text_balanced(self):
        """Text 'balanced' should map to 'balanced'."""
        assert parse_style_preference("balanced") == "balanced"
        assert parse_style_preference("default") == "balanced"
        assert parse_style_preference("adapt") == "balanced"

    def test_parse_style_text_direct(self):
        """Text 'direct' should map to 'direct'."""
        assert parse_style_preference("direct") == "direct"
        assert parse_style_preference("concise") == "direct"
        assert parse_style_preference("direct & concise") == "direct"

    def test_parse_style_description_warm(self):
        """Description with 'empathy' keywords should infer 'warm'."""
        assert parse_style_preference("I want more empathy and support") == "warm"
        assert parse_style_preference("be warm and caring") == "warm"

    def test_parse_style_description_direct(self):
        """Description with 'short' keywords should infer 'direct'."""
        assert parse_style_preference("I want short responses") == "direct"
        assert parse_style_preference("be brief and concise") == "direct"
        assert parse_style_preference("keep it quick") == "direct"

    def test_parse_style_default_balanced(self):
        """Unknown input should default to 'balanced'."""
        assert parse_style_preference("something random") == "balanced"
        assert parse_style_preference("xyz") == "balanced"


class TestIsSkip:
    """Test skip detection."""

    def test_is_skip_explicit_skip(self):
        """Explicit 'skip' should return True."""
        assert is_skip("skip") is True
        assert is_skip("SKIP") is True

    def test_is_skip_alternatives(self):
        """Common skip alternatives should return True."""
        assert is_skip("none") is True
        assert is_skip("n/a") is True
        assert is_skip("pass") is True
        assert is_skip("no") is True
        assert is_skip("-") is True
        assert is_skip("") is True

    def test_is_skip_non_skip(self):
        """Non-skip input should return False."""
        assert is_skip("Alice") is False
        assert is_skip("they/them") is False
        assert is_skip("I'm a developer") is False


class TestWizardFlow:
    """Test wizard flow through state transitions."""

    @pytest.mark.asyncio
    async def test_wizard_welcome_advances_to_api_key(self, mock_orchestrator):
        """WELCOME step should advance to API_KEY."""
        state = WizardState(step=WizardStep.WELCOME)
        response, new_state, complete = await process_wizard_message("", state, mock_orchestrator)

        assert new_state.step == WizardStep.API_KEY
        assert "api key" in response.lower() or "openrouter" in response.lower()
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_api_key_valid_advances_to_style(self, mock_orchestrator):
        """Valid API key should advance to STYLE."""
        state = WizardState(step=WizardStep.API_KEY)
        response, new_state, complete = await process_wizard_message(
            "sk-or-test-key-12345678901234567890",
            state,
            mock_orchestrator
        )

        assert new_state.step == WizardStep.STYLE
        assert "style" in response.lower() or "talk" in response.lower()
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_api_key_invalid_stays_on_api_key(self, mock_orchestrator):
        """Invalid API key should stay on API_KEY step."""
        state = WizardState(step=WizardStep.API_KEY)
        response, new_state, complete = await process_wizard_message("invalid-key", state, mock_orchestrator)

        assert new_state.step == WizardStep.API_KEY
        assert "doesn't look like" in response.lower() or "valid" in response.lower()
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_style_advances_to_name(self, mock_orchestrator):
        """Style selection should advance to NAME."""
        state = WizardState(step=WizardStep.STYLE)
        response, new_state, complete = await process_wizard_message("2", state, mock_orchestrator)

        assert new_state.step == WizardStep.NAME
        assert "name" in response.lower() or "call you" in response.lower()
        assert new_state.collected_data['style'] == 'balanced'
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_name_advances_to_pronouns(self, mock_orchestrator):
        """Name entry should advance to PRONOUNS."""
        state = WizardState(step=WizardStep.NAME)
        response, new_state, complete = await process_wizard_message("TestUser", state, mock_orchestrator)

        assert new_state.step == WizardStep.PRONOUNS
        assert "pronouns" in response.lower()
        assert new_state.collected_data['name'] == 'TestUser'
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_name_skip_advances_to_pronouns(self, mock_orchestrator):
        """Skipping name should advance to PRONOUNS with empty name."""
        state = WizardState(step=WizardStep.NAME)
        response, new_state, complete = await process_wizard_message("skip", state, mock_orchestrator)

        assert new_state.step == WizardStep.PRONOUNS
        assert new_state.collected_data['name'] == ''
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_pronouns_advances_to_background(self, mock_orchestrator):
        """Pronouns entry should advance to BACKGROUND."""
        state = WizardState(step=WizardStep.PRONOUNS)
        response, new_state, complete = await process_wizard_message("they/them", state, mock_orchestrator)

        assert new_state.step == WizardStep.BACKGROUND
        assert "anything" in response.lower() or "background" in response.lower()
        assert new_state.collected_data['pronouns'] == 'they/them'
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_background_skip_completes(self, mock_orchestrator):
        """Skipping background should complete wizard."""
        state = WizardState(step=WizardStep.BACKGROUND, collected_data={'style': 'balanced'})

        # Mock user profile (patch at import location)
        with patch('memory.user_profile.UserProfile') as mock_profile_class:
            mock_profile = Mock()
            mock_profile_class.return_value = mock_profile

            response, new_state, complete = await process_wizard_message("skip", state, mock_orchestrator)

        assert new_state.step == WizardStep.COMPLETE
        assert "all set" in response.lower() or "ready" in response.lower()
        assert complete is True

    @pytest.mark.asyncio
    async def test_wizard_background_with_text_extracts_facts(self, mock_orchestrator):
        """Background text should trigger fact extraction."""
        state = WizardState(step=WizardStep.BACKGROUND, collected_data={'style': 'balanced'})

        # Mock LLMFactExtractor (patch at import location)
        with patch('memory.llm_fact_extractor.LLMFactExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract_triples = AsyncMock(return_value=[
                {'relation': 'occupation', 'value': 'developer', 'confidence': 0.8}
            ])
            mock_extractor_class.return_value = mock_extractor

            # Mock UserProfile (patch at import location)
            with patch('memory.user_profile.UserProfile') as mock_profile_class:
                mock_profile = Mock()
                mock_profile_class.return_value = mock_profile

                response, new_state, complete = await process_wizard_message(
                    "I'm a software developer",
                    state,
                    mock_orchestrator
                )

        assert new_state.step == WizardStep.COMPLETE
        assert complete is True
        assert 'initial_facts' in new_state.collected_data
        assert len(new_state.collected_data['initial_facts']) == 1


class TestWizardErrorRecovery:
    """Test wizard error handling and recovery."""

    @pytest.mark.asyncio
    async def test_wizard_api_key_test_failure_stays_on_step(self, mock_orchestrator):
        """Failed API key test should stay on API_KEY step."""
        mock_orchestrator.model_manager.generate_once = AsyncMock(side_effect=Exception("API error"))

        state = WizardState(step=WizardStep.API_KEY)
        response, new_state, complete = await process_wizard_message(
            "sk-or-test-key-12345678901234567890",
            state,
            mock_orchestrator
        )

        assert new_state.step == WizardStep.API_KEY
        assert "didn't work" in response.lower()
        assert complete is False

    @pytest.mark.asyncio
    async def test_wizard_max_retries_exceeded(self, mock_orchestrator):
        """Exceeding max retries should provide helpful error message."""
        state = WizardState(step=WizardStep.API_KEY, error_count=3)

        # Force an exception
        with patch('gui.wizard._handle_api_key', side_effect=Exception("Test error")):
            response, new_state, complete = await process_wizard_message(
                "sk-or-test-key-12345678901234567890",
                state,
                mock_orchestrator
            )

        assert "having trouble" in response.lower()
        assert "manually" in response.lower()


class TestWizardCompletion:
    """Test wizard completion and profile saving."""

    @pytest.mark.asyncio
    async def test_wizard_completion_saves_profile(self, mock_orchestrator):
        """Wizard completion should save identity and preferences to profile."""
        state = WizardState(
            step=WizardStep.BACKGROUND,
            collected_data={
                'name': 'TestUser',
                'pronouns': 'they/them',
                'style': 'balanced'
            }
        )

        # Create mock profile with expected methods
        mock_profile = Mock()
        mock_profile.update_identity = Mock()
        mock_profile.update_preferences = Mock()
        mock_profile.add_fact = Mock()
        mock_profile.save = Mock()

        # Set orchestrator.user_profile to our mock (so hasattr check passes and returns our mock)
        mock_orchestrator.user_profile = mock_profile

        response, new_state, complete = await process_wizard_message("skip", state, mock_orchestrator)

        # Verify profile methods were called
        mock_profile.update_identity.assert_called_once_with(name='TestUser', pronouns='they/them')
        mock_profile.update_preferences.assert_called_once_with(style='balanced')

        assert complete is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
