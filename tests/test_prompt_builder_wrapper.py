"""Tests for core/prompt_builder.py (compatibility wrapper)"""
import pytest
from core.prompt_builder import UnifiedHierarchicalPromptBuilder, UnifiedPromptBuilder, PromptBuilder


class MockPromptBuilder:
    """Mock prompt builder for testing."""

    def __init__(self):
        self.consolidator = "mock_consolidator"
        self.build_prompt_called = False
        self.custom_attribute = "test_value"

    async def build_prompt(self, *args, **kwargs):
        """Mock async build_prompt."""
        self.build_prompt_called = True
        return f"prompt with args={args}, kwargs={kwargs}"


class MockGatedBuilder:
    """Mock gated prompt builder."""

    def __init__(self):
        self.consolidator = None

    async def build_gated_prompt(self, *args, **kwargs):
        """Mock async build_gated_prompt."""
        return f"gated prompt with args={args}"


class SyncMockBuilder:
    """Mock synchronous prompt builder."""

    def build_prompt(self, *args, **kwargs):
        """Mock sync build_prompt."""
        return "sync prompt"


class NoMethodBuilder:
    """Mock builder without build methods."""
    pass


def test_unified_hierarchical_builder_initialization():
    """Test that UnifiedHierarchicalPromptBuilder initializes correctly."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    assert wrapper._delegate == mock_builder
    assert wrapper.model_manager == mock_model_manager
    assert wrapper.consolidator == "mock_consolidator"


def test_unified_hierarchical_builder_with_chroma():
    """Test initialization with chroma_store."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()
    mock_chroma = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager,
        chroma_store=mock_chroma
    )

    assert wrapper.chroma_store == mock_chroma


def test_unified_hierarchical_builder_no_consolidator():
    """Test initialization when delegate has no consolidator."""
    mock_builder = NoMethodBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    assert wrapper.consolidator is None


@pytest.mark.asyncio
async def test_build_prompt_delegates_to_async():
    """Test that build_prompt delegates to async build_prompt."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    result = await wrapper.build_prompt("user_input", test_arg="value")

    assert mock_builder.build_prompt_called
    assert "user_input" in result
    assert "test_arg" in result


@pytest.mark.asyncio
async def test_build_prompt_delegates_to_gated():
    """Test that build_prompt delegates to build_gated_prompt if no build_prompt."""
    mock_builder = MockGatedBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    result = await wrapper.build_prompt("user_input", arg2="value")

    assert "gated prompt" in result
    assert "user_input" in result


@pytest.mark.asyncio
async def test_build_prompt_handles_sync_builder():
    """Test that build_prompt handles synchronous builders."""
    mock_builder = SyncMockBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    result = await wrapper.build_prompt("test")

    assert result == "sync prompt"


@pytest.mark.asyncio
async def test_build_prompt_raises_on_no_method():
    """Test that build_prompt raises AttributeError if no method available."""
    mock_builder = NoMethodBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    with pytest.raises(AttributeError, match="does not support build_prompt"):
        await wrapper.build_prompt("test")


def test_getattr_delegation():
    """Test that __getattr__ delegates to underlying builder."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    # Should delegate to mock_builder.custom_attribute
    assert wrapper.custom_attribute == "test_value"


def test_getattr_delegation_missing_attribute():
    """Test that __getattr__ raises AttributeError for missing attributes."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()

    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager
    )

    with pytest.raises(AttributeError):
        _ = wrapper.nonexistent_attribute


def test_unified_prompt_builder_export():
    """Test that UnifiedPromptBuilder is exported correctly."""
    # Should be re-exported from core.prompt
    assert UnifiedPromptBuilder is not None
    # Should be a class we can reference
    assert hasattr(UnifiedPromptBuilder, "__name__")


def test_prompt_builder_export():
    """Test that PromptBuilder is exported correctly."""
    # Should be imported from core.prompt
    assert PromptBuilder is not None
    assert hasattr(PromptBuilder, "__name__")


def test_all_exports():
    """Test that __all__ contains expected exports."""
    from core import prompt_builder as pb_module

    assert hasattr(pb_module, "__all__")
    assert "UnifiedPromptBuilder" in pb_module.__all__
    assert "UnifiedHierarchicalPromptBuilder" in pb_module.__all__
    assert "PromptBuilder" in pb_module.__all__


def test_wrapper_with_extra_kwargs():
    """Test that wrapper ignores extra kwargs."""
    mock_builder = MockPromptBuilder()
    mock_model_manager = object()

    # Should not raise with extra kwargs
    wrapper = UnifiedHierarchicalPromptBuilder(
        prompt_builder=mock_builder,
        model_manager=mock_model_manager,
        extra_arg1="value1",
        extra_arg2="value2"
    )

    assert wrapper._delegate == mock_builder
