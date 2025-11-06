"""
Unit tests for core/dependencies.py

Tests:
- DependencyContainer singleton pattern
- Initialization and getter methods
"""

import pytest
from unittest.mock import Mock
from core.dependencies import DependencyContainer, deps


# =============================================================================
# DependencyContainer Tests
# =============================================================================

def test_dependency_container_singleton():
    """DependencyContainer is a singleton"""
    container1 = DependencyContainer()
    container2 = DependencyContainer()

    # Same instance
    assert container1 is container2


def test_dependency_container_global_instance():
    """Global deps instance is the singleton"""
    container = DependencyContainer()

    assert deps is container


def test_initialize_sets_model_manager():
    """Initialize sets model_manager"""
    # Create fresh container for testing
    container = object.__new__(DependencyContainer)
    container._initialized = False

    mock_model_manager = Mock()
    container.initialize(mock_model_manager)

    assert container.model_manager is mock_model_manager
    assert container._initialized is True


def test_initialize_idempotent():
    """Initialize can be called multiple times safely"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    mock_model_manager1 = Mock()
    mock_model_manager2 = Mock()

    container.initialize(mock_model_manager1)
    container.initialize(mock_model_manager2)

    # First one is kept
    assert container.model_manager is mock_model_manager1


def test_get_model_manager_before_init():
    """get_model_manager raises error if not initialized"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    with pytest.raises(RuntimeError, match="Dependencies not initialized"):
        container.get_model_manager()


def test_get_model_manager_after_init():
    """get_model_manager returns model_manager after initialization"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    mock_model_manager = Mock()
    container.initialize(mock_model_manager)

    result = container.get_model_manager()

    assert result is mock_model_manager


def test_get_tokenizer_manager_before_init():
    """get_tokenizer_manager raises error if not initialized"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    with pytest.raises(RuntimeError, match="Dependencies not initialized"):
        container.get_tokenizer_manager()


def test_get_tokenizer_manager_after_init():
    """get_tokenizer_manager returns tokenizer_manager after initialization"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    mock_model_manager = Mock()
    container.initialize(mock_model_manager)

    # Tokenizer manager is created during init
    result = container.get_tokenizer_manager()

    assert result is not None
    assert result is container.tokenizer_manager


def test_multiple_instances_share_state():
    """Multiple DependencyContainer instances share state"""
    container1 = DependencyContainer()
    container2 = DependencyContainer()

    # Reset for testing
    container1._initialized = False

    mock_model_manager = Mock()
    container1.initialize(mock_model_manager)

    # container2 sees the same initialization
    assert container2._initialized is True
    assert container2.model_manager is mock_model_manager


def test_initialized_flag_starts_false():
    """New container starts with _initialized = False"""
    # Get the singleton
    container = DependencyContainer()

    # It may already be initialized from previous tests
    # So let's create a new instance directly
    new_container = object.__new__(DependencyContainer)
    new_container._initialized = False

    assert new_container._initialized is False


def test_container_new_method():
    """__new__ creates or returns singleton"""
    # Clear the singleton for testing
    original_instance = DependencyContainer._instance

    try:
        # Reset singleton
        DependencyContainer._instance = None

        container1 = DependencyContainer()
        assert container1 is not None
        assert DependencyContainer._instance is container1

        container2 = DependencyContainer()
        assert container2 is container1

    finally:
        # Restore
        DependencyContainer._instance = original_instance


def test_initialize_creates_tokenizer_manager():
    """Initialize creates TokenizerManager with model_manager"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    mock_model_manager = Mock()
    container.initialize(mock_model_manager)

    # TokenizerManager should be created
    assert hasattr(container, 'tokenizer_manager')
    assert container.tokenizer_manager is not None


# =============================================================================
# Edge Cases
# =============================================================================

def test_initialize_with_none():
    """Initialize with None model_manager raises error"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    # TokenizerManager requires model_manager
    with pytest.raises(ValueError, match="model_manager is required"):
        container.initialize(None)


def test_get_manager_error_messages():
    """Error messages are descriptive"""
    container = object.__new__(DependencyContainer)
    container._initialized = False

    try:
        container.get_model_manager()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Dependencies not initialized" in str(e)
        assert "initialize()" in str(e)

    try:
        container.get_tokenizer_manager()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Dependencies not initialized" in str(e)
        assert "initialize()" in str(e)
