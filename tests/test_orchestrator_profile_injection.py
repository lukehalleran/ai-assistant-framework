"""
tests/test_orchestrator_profile_injection.py

Tests for orchestrator runtime identity and style injection.
Validates that user profile name/pronouns are substituted in system prompts
and style modifiers are applied to tone instructions.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from utils.tone_detector import CrisisLevel
from memory.user_profile_schema import ProfileIdentity, ProfilePreferences


@pytest.fixture
def mock_user_profile():
    """Create a mock user profile with identity and preferences."""
    profile = Mock()
    profile.identity = ProfileIdentity(name="TestUser", pronouns="they/them")
    profile.preferences = ProfilePreferences(style="balanced")
    profile.get_style_modifier = Mock(return_value="")
    return profile


@pytest.fixture
def mock_orchestrator():
    """Create a minimal mock orchestrator for testing."""
    from core.orchestrator import DaemonOrchestrator

    # Create all required dependencies as mocks
    model_manager = Mock()
    response_generator = Mock()
    response_generator.time_manager = None
    file_processor = Mock()
    prompt_builder = Mock()
    # Mock build_prompt as async - returns a simple string (the actual prompt doesn't matter for these tests)
    prompt_builder.build_prompt = AsyncMock(return_value="mocked_full_prompt")
    memory_system = Mock()
    memory_system.corpus_manager = Mock()

    orchestrator = DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder,
        memory_system=memory_system
    )

    return orchestrator


@pytest.fixture
def mock_emotional_context():
    """Create a mock emotional context to bypass tone detection in tests."""
    from utils.emotional_context import EmotionalContext
    from utils.tone_detector import CrisisLevel
    from utils.need_detector import NeedType

    return EmotionalContext(
        crisis_level=CrisisLevel.CONVERSATIONAL,
        need_type=NeedType.NEUTRAL,
        tone_confidence=0.8,
        need_confidence=0.7,
        tone_trigger=None,
        need_trigger=None,
        explanation="Test emotional context"
    )


class TestUserProfileLoading:
    """Test that user profile is loaded in orchestrator __init__."""

    def test_orchestrator_loads_user_profile(self, mock_orchestrator):
        """Orchestrator should load user profile on init."""
        # user_profile should be loaded (or None if loading failed)
        assert hasattr(mock_orchestrator, 'user_profile')


class TestSystemPromptPlaceholderSubstitution:
    """Test runtime placeholder substitution in system prompts."""

    @pytest.mark.asyncio
    async def test_system_prompt_name_substitution(self, mock_orchestrator, mock_user_profile, mock_emotional_context):
        """System prompt should replace {USER_NAME} with actual name."""
        mock_orchestrator.user_profile = mock_user_profile

        # Create a system prompt with placeholder
        test_prompt = "You are talking to {USER_NAME}. Be helpful."

        # Mock the system prompt loading to return our test prompt
        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        # Verify placeholder was replaced
        assert "{USER_NAME}" not in sys_prompt
        assert "TestUser" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_pronouns_substitution(self, mock_orchestrator, mock_user_profile, mock_emotional_context):
        """System prompt should replace {USER_PRONOUNS} with actual pronouns."""
        mock_orchestrator.user_profile = mock_user_profile

        test_prompt = "Use {USER_PRONOUNS} when referring to the user."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        assert "{USER_PRONOUNS}" not in sys_prompt
        assert "they/them" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_pronoun_variants_they_them(self, mock_orchestrator, mock_user_profile, mock_emotional_context):
        """System prompt should replace pronoun variants for they/them."""
        mock_orchestrator.user_profile = mock_user_profile

        test_prompt = "{PRONOUN_SUBJ} likes coding. Talk to {PRONOUN_OBJ} about {PRONOUN_POSS} projects."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        assert "{PRONOUN_SUBJ}" not in sys_prompt
        assert "they likes coding" in sys_prompt.lower()
        assert "talk to them about their projects" in sys_prompt.lower()

    @pytest.mark.asyncio
    async def test_system_prompt_pronoun_variants_she_her(self, mock_orchestrator, mock_emotional_context):
        """System prompt should replace pronoun variants for she/her."""
        profile = Mock()
        profile.identity = ProfileIdentity(name="Alice", pronouns="she/her")
        profile.get_style_modifier = Mock(return_value="")
        mock_orchestrator.user_profile = profile

        test_prompt = "{PRONOUN_SUBJ} is working. Support {PRONOUN_OBJ} with {PRONOUN_POSS} tasks."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        assert "she is working" in sys_prompt.lower()
        assert "support her with her tasks" in sys_prompt.lower()

    @pytest.mark.asyncio
    async def test_system_prompt_pronoun_variants_he_him(self, mock_orchestrator, mock_emotional_context):
        """System prompt should replace pronoun variants for he/him."""
        profile = Mock()
        profile.identity = ProfileIdentity(name="Bob", pronouns="he/him")
        profile.get_style_modifier = Mock(return_value="")
        mock_orchestrator.user_profile = profile

        test_prompt = "{PRONOUN_SUBJ} needs help. Assist {PRONOUN_OBJ} with {PRONOUN_POSS} work."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        assert "he needs help" in sys_prompt.lower()
        assert "assist him with his work" in sys_prompt.lower()

    @pytest.mark.asyncio
    async def test_system_prompt_defaults_when_no_profile(self, mock_orchestrator, mock_emotional_context):
        """System prompt should use defaults when user profile is missing."""
        mock_orchestrator.user_profile = None

        test_prompt = "You are talking to {USER_NAME} using {USER_PRONOUNS}."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            prompt, sys_prompt = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        # Placeholders should remain unchanged if no profile
        # (or be replaced with defaults - depends on implementation)
        # Based on the code, if profile is None, placeholders won't be replaced
        assert "{USER_NAME}" in sys_prompt or "the user" in sys_prompt.lower()


class TestStyleModifierInjection:
    """Test style modifier injection in tone instructions."""

    def test_get_style_modifier_warm(self, mock_orchestrator):
        """Warm style should inject warm modifier in CONVERSATIONAL mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: WARM & SUPPORTIVE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.CONVERSATIONAL)

        assert "WARM & SUPPORTIVE" in instructions
        assert "CONVERSATIONAL" in instructions

    def test_get_style_modifier_direct(self, mock_orchestrator):
        """Direct style should inject direct modifier in CONVERSATIONAL mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: DIRECT & CONCISE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.CONVERSATIONAL)

        assert "DIRECT & CONCISE" in instructions
        assert "CONVERSATIONAL" in instructions

    def test_get_style_modifier_balanced_empty(self, mock_orchestrator):
        """Balanced style should return empty modifier (no injection)."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.CONVERSATIONAL)

        # Should only have CONVERSATIONAL mode, no style modifier
        assert "CONVERSATIONAL" in instructions
        assert "WARM" not in instructions
        assert "DIRECT" not in instructions

    def test_style_modifier_not_applied_in_crisis(self, mock_orchestrator):
        """Style modifier should NOT be applied in HIGH crisis mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: DIRECT & CONCISE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.HIGH)

        # Crisis mode should override style preference
        assert "CRISIS SUPPORT" in instructions
        assert "DIRECT & CONCISE" not in instructions
        # get_style_modifier should not have been called for HIGH crisis
        assert profile.get_style_modifier.call_count == 0

    def test_style_modifier_applied_in_medium(self, mock_orchestrator):
        """Style modifier should be applied in MEDIUM support mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: WARM & SUPPORTIVE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.MEDIUM)

        assert "WARM & SUPPORTIVE" in instructions
        assert "ELEVATED SUPPORT" in instructions

    def test_style_modifier_applied_in_concern(self, mock_orchestrator):
        """Style modifier should be applied in CONCERN mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: DIRECT & CONCISE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.CONCERN)

        assert "DIRECT & CONCISE" in instructions
        assert "LIGHT SUPPORT" in instructions


class TestProfileChangesReflectedImmediately:
    """Test that profile changes are reflected immediately (no restart needed)."""

    @pytest.mark.asyncio
    async def test_profile_changes_reflected_immediately(self, mock_orchestrator, mock_emotional_context):
        """Profile changes should be reflected in next prepare_prompt call."""
        # Start with one profile
        profile1 = Mock()
        profile1.identity = ProfileIdentity(name="User1", pronouns="he/him")
        profile1.get_style_modifier = Mock(return_value="")
        mock_orchestrator.user_profile = profile1

        test_prompt = "Talking to {USER_NAME} ({USER_PRONOUNS})."

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            _, sys_prompt1 = await mock_orchestrator.prepare_prompt("Hello", use_raw_mode=False)

        assert "User1" in sys_prompt1
        assert "he/him" in sys_prompt1

        # Change profile
        profile2 = Mock()
        profile2.identity = ProfileIdentity(name="User2", pronouns="she/her")
        profile2.get_style_modifier = Mock(return_value="")
        mock_orchestrator.user_profile = profile2

        with patch('config.app_config.load_system_prompt', return_value=test_prompt), \
             patch('core.orchestrator.analyze_emotional_context', new=AsyncMock(return_value=mock_emotional_context)):
            _, sys_prompt2 = await mock_orchestrator.prepare_prompt("Hi again", use_raw_mode=False)

        # Should reflect new profile
        assert "User2" in sys_prompt2
        assert "she/her" in sys_prompt2
        assert "User1" not in sys_prompt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
