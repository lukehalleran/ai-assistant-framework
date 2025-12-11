"""
Tests for force_wizard flag override after wizard completion.

Verifies that once wizard is completed (identity.name exists), the force_wizard
flag is ignored on subsequent launches.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from gui.launch import launch_gui


def test_force_wizard_ignored_when_identity_exists(capsys):
    """Force wizard flag should be ignored if identity.name exists."""
    # Create mock orchestrator with completed wizard (identity exists)
    mock_orchestrator = Mock()
    mock_orchestrator.user_profile = Mock()
    mock_orchestrator.user_profile.identity = Mock()
    mock_orchestrator.user_profile.identity.name = "Luke"  # Wizard completed
    mock_orchestrator.user_profile.is_first_run = Mock(return_value=False)  # Return actual boolean
    mock_orchestrator.personality_manager = Mock()
    mock_orchestrator.personality_manager.personalities = {"default": Mock()}  # Mock personalities dict
    mock_orchestrator.memory_system = Mock()
    mock_orchestrator.memory_system.corpus_manager = Mock()
    mock_orchestrator.memory_system.corpus_manager.corpus = []  # Empty corpus

    # Mock _launch_wizard_ui and the normal chat UI launch
    with patch('gui.launch._launch_wizard_ui') as mock_wizard_ui, \
         patch('gui.launch.gr.Blocks') as mock_blocks:

        # Configure mock for normal chat UI
        mock_demo = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_demo

        # Launch with force_wizard=True but identity exists
        launch_gui(mock_orchestrator, force_wizard=True)

        # Verify wizard UI was NOT launched
        mock_wizard_ui.assert_not_called()

        # Verify normal chat UI was launched instead
        mock_blocks.assert_called_once()

        # Check debug output
        captured = capsys.readouterr()
        assert "Force wizard mode enabled BUT wizard already completed" in captured.out
        assert "Identity name: 'Luke'" in captured.out
        assert "Ignoring force_wizard flag" in captured.out
        assert "Launching normal chat UI" in captured.out


def test_force_wizard_honored_when_no_identity(capsys):
    """Force wizard flag should be honored if identity.name is empty."""
    # Create mock orchestrator with NO identity
    mock_orchestrator = Mock()
    mock_orchestrator.user_profile = Mock()
    mock_orchestrator.user_profile.identity = Mock()
    mock_orchestrator.user_profile.identity.name = ""  # No identity
    mock_orchestrator.personality_manager = Mock()
    mock_orchestrator.memory_system = Mock()
    mock_orchestrator.memory_system.corpus_manager = Mock()
    mock_orchestrator.memory_system.corpus_manager.corpus = []

    # Mock _launch_wizard_ui
    with patch('gui.launch._launch_wizard_ui') as mock_wizard_ui:
        mock_wizard_ui.return_value = None

        # Launch with force_wizard=True and no identity
        launch_gui(mock_orchestrator, force_wizard=True)

        # Verify wizard UI WAS launched
        mock_wizard_ui.assert_called_once()

        # Check debug output
        captured = capsys.readouterr()
        assert "Force wizard mode enabled" in captured.out
        assert "Launching wizard UI" in captured.out


def test_force_wizard_honored_when_no_profile(capsys):
    """Force wizard flag should be honored if no profile exists."""
    # Create mock orchestrator with no profile
    mock_orchestrator = Mock()
    mock_orchestrator.user_profile = None  # No profile at all
    mock_orchestrator.personality_manager = Mock()
    mock_orchestrator.memory_system = Mock()
    mock_orchestrator.memory_system.corpus_manager = Mock()
    mock_orchestrator.memory_system.corpus_manager.corpus = []

    # Mock _launch_wizard_ui
    with patch('gui.launch._launch_wizard_ui') as mock_wizard_ui:
        mock_wizard_ui.return_value = None

        # Launch with force_wizard=True and no profile
        launch_gui(mock_orchestrator, force_wizard=True)

        # Verify wizard UI WAS launched
        mock_wizard_ui.assert_called_once()

        # Check debug output
        captured = capsys.readouterr()
        assert "Force wizard mode enabled" in captured.out
        assert "Launching wizard UI" in captured.out


def test_normal_launch_after_wizard_completion(capsys):
    """Normal launch (no force flag) should skip wizard if identity exists."""
    # Create mock orchestrator with completed wizard
    mock_orchestrator = Mock()
    mock_orchestrator.user_profile = Mock()
    mock_orchestrator.user_profile.identity = Mock()
    mock_orchestrator.user_profile.identity.name = "Luke"  # Wizard completed
    mock_orchestrator.user_profile.is_first_run = Mock(return_value=False)
    mock_orchestrator.personality_manager = Mock()
    mock_orchestrator.memory_system = Mock()
    mock_orchestrator.memory_system.corpus_manager = Mock()
    mock_orchestrator.memory_system.corpus_manager.corpus = ["entry1", "entry2"]

    # Mock _launch_wizard_ui and normal chat UI
    with patch('gui.launch._launch_wizard_ui') as mock_wizard_ui, \
         patch('gui.launch.gr.Blocks') as mock_blocks:

        mock_demo = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_demo

        # Launch without force_wizard flag
        launch_gui(mock_orchestrator, force_wizard=False)

        # Verify wizard UI was NOT launched
        mock_wizard_ui.assert_not_called()

        # Verify normal chat UI was launched
        mock_blocks.assert_called_once()

        # Check debug output
        captured = capsys.readouterr()
        assert "Launching normal chat UI" in captured.out
        assert "Is first run: False" in captured.out
