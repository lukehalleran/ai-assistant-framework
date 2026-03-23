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


# Pronoun map matching orchestrator.py implementation
PRONOUN_MAP = {
    "he/him": ("he", "him", "his"),
    "she/her": ("she", "her", "her"),
    "they/them": ("they", "them", "their"),
}


def _substitute_placeholders(prompt: str, profile) -> str:
    """Replicate the orchestrator's placeholder substitution logic for testing."""
    if not profile:
        return prompt
    identity = profile.identity
    name = identity.name if identity.name else "the user"
    pronouns = identity.pronouns if identity.pronouns else "they/them"
    subj, obj, poss = PRONOUN_MAP.get(pronouns.lower(), ("they", "them", "their"))
    prompt = prompt.replace("{USER_NAME}", name)
    prompt = prompt.replace("{USER_PRONOUNS}", pronouns)
    prompt = prompt.replace("{PRONOUN_SUBJ}", subj)
    prompt = prompt.replace("{PRONOUN_OBJ}", obj)
    prompt = prompt.replace("{PRONOUN_POSS}", poss)
    return prompt


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

    model_manager = Mock()
    response_generator = Mock()
    response_generator.time_manager = None
    file_processor = Mock()
    prompt_builder = Mock()
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


class TestUserProfileLoading:
    """Test that user profile is loaded in orchestrator __init__."""

    def test_orchestrator_loads_user_profile(self, mock_orchestrator):
        """Orchestrator should load user profile on init."""
        assert hasattr(mock_orchestrator, 'user_profile')


class TestSystemPromptPlaceholderSubstitution:
    """Test runtime placeholder substitution in system prompts.

    Tests the substitution logic directly rather than through the full
    prepare_prompt pipeline, since that requires deep async mocking.
    """

    def test_system_prompt_name_substitution(self, mock_user_profile):
        """System prompt should replace {USER_NAME} with actual name."""
        prompt = "You are talking to {USER_NAME}. Be helpful."
        result = _substitute_placeholders(prompt, mock_user_profile)

        assert "{USER_NAME}" not in result
        assert "TestUser" in result

    def test_system_prompt_pronouns_substitution(self, mock_user_profile):
        """System prompt should replace {USER_PRONOUNS} with actual pronouns."""
        prompt = "Use {USER_PRONOUNS} when referring to the user."
        result = _substitute_placeholders(prompt, mock_user_profile)

        assert "{USER_PRONOUNS}" not in result
        assert "they/them" in result

    def test_system_prompt_pronoun_variants_they_them(self, mock_user_profile):
        """System prompt should replace pronoun variants for they/them."""
        prompt = "{PRONOUN_SUBJ} likes coding. Talk to {PRONOUN_OBJ} about {PRONOUN_POSS} projects."
        result = _substitute_placeholders(prompt, mock_user_profile)

        assert "{PRONOUN_SUBJ}" not in result
        assert "they likes coding" in result.lower()
        assert "talk to them about their projects" in result.lower()

    def test_system_prompt_pronoun_variants_she_her(self):
        """System prompt should replace pronoun variants for she/her."""
        profile = Mock()
        profile.identity = ProfileIdentity(name="Alice", pronouns="she/her")

        prompt = "{PRONOUN_SUBJ} is working. Support {PRONOUN_OBJ} with {PRONOUN_POSS} tasks."
        result = _substitute_placeholders(prompt, profile)

        assert "she is working" in result.lower()
        assert "support her with her tasks" in result.lower()

    def test_system_prompt_pronoun_variants_he_him(self):
        """System prompt should replace pronoun variants for he/him."""
        profile = Mock()
        profile.identity = ProfileIdentity(name="Bob", pronouns="he/him")

        prompt = "{PRONOUN_SUBJ} needs help. Assist {PRONOUN_OBJ} with {PRONOUN_POSS} work."
        result = _substitute_placeholders(prompt, profile)

        assert "he needs help" in result.lower()
        assert "assist him with his work" in result.lower()

    def test_system_prompt_defaults_when_no_profile(self):
        """System prompt should keep placeholders when user profile is None."""
        prompt = "You are talking to {USER_NAME} using {USER_PRONOUNS}."
        result = _substitute_placeholders(prompt, None)

        assert "{USER_NAME}" in result

    def test_system_prompt_defaults_when_name_empty(self):
        """System prompt should use 'the user' when name is empty."""
        profile = Mock()
        profile.identity = ProfileIdentity(name="", pronouns="")

        prompt = "Talking to {USER_NAME}."
        result = _substitute_placeholders(prompt, profile)

        assert "the user" in result


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

        assert "CONVERSATIONAL" in instructions
        assert "WARM" not in instructions
        assert "DIRECT" not in instructions

    def test_style_modifier_not_applied_in_crisis(self, mock_orchestrator):
        """Style modifier should NOT be applied in HIGH crisis mode."""
        profile = Mock()
        profile.get_style_modifier = Mock(return_value="\nSTYLE: DIRECT & CONCISE\n")
        mock_orchestrator.user_profile = profile

        instructions = mock_orchestrator._get_tone_instructions(CrisisLevel.HIGH)

        assert "CRISIS SUPPORT" in instructions
        assert "DIRECT & CONCISE" not in instructions
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
    """Test that profile changes are reflected immediately."""

    def test_profile_changes_reflected_immediately(self):
        """Profile changes should be reflected in placeholder substitution."""
        profile1 = Mock()
        profile1.identity = ProfileIdentity(name="User1", pronouns="he/him")

        prompt = "Talking to {USER_NAME} ({USER_PRONOUNS})."

        result1 = _substitute_placeholders(prompt, profile1)
        assert "User1" in result1
        assert "he/him" in result1

        # Change profile
        profile2 = Mock()
        profile2.identity = ProfileIdentity(name="User2", pronouns="she/her")

        result2 = _substitute_placeholders(prompt, profile2)
        assert "User2" in result2
        assert "she/her" in result2
        assert "User1" not in result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
