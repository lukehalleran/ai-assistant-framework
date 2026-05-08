"""Tests for side-effect-free eval generation."""

import pytest

from eval.no_store_generation import EvalGenerationConfig, EvalGenerator
from eval.schema import EvalGenerationResult


class TestEvalGenerationConfig:
    """Test safety constraints on EvalGenerationConfig."""

    def test_all_flags_default_false(self):
        config = EvalGenerationConfig()
        for field_name, value in vars(config).items():
            assert value is False, f"{field_name} should default to False"

    def test_store_interaction_true_raises(self):
        with pytest.raises(ValueError, match="store_interaction"):
            EvalGenerationConfig(store_interaction=True)

    def test_update_graph_true_raises(self):
        with pytest.raises(ValueError, match="update_graph"):
            EvalGenerationConfig(update_graph=True)

    def test_extract_facts_true_raises(self):
        with pytest.raises(ValueError, match="extract_facts"):
            EvalGenerationConfig(extract_facts=True)

    def test_update_threads_true_raises(self):
        with pytest.raises(ValueError, match="update_threads"):
            EvalGenerationConfig(update_threads=True)

    def test_run_shutdown_processing_true_raises(self):
        with pytest.raises(ValueError, match="run_shutdown_processing"):
            EvalGenerationConfig(run_shutdown_processing=True)

    def test_allow_web_search_true_raises(self):
        with pytest.raises(ValueError, match="allow_web_search"):
            EvalGenerationConfig(allow_web_search=True)

    def test_update_summaries_true_raises(self):
        with pytest.raises(ValueError, match="update_summaries"):
            EvalGenerationConfig(update_summaries=True)

    def test_update_reflections_true_raises(self):
        with pytest.raises(ValueError, match="update_reflections"):
            EvalGenerationConfig(update_reflections=True)

    def test_update_stm_true_raises(self):
        with pytest.raises(ValueError, match="update_stm"):
            EvalGenerationConfig(update_stm=True)

    def test_trigger_skills_true_raises(self):
        with pytest.raises(ValueError, match="trigger_skills"):
            EvalGenerationConfig(trigger_skills=True)

    def test_multiple_flags_true_raises(self):
        """Even if multiple flags are True, the first one encountered raises."""
        with pytest.raises(ValueError):
            EvalGenerationConfig(store_interaction=True, update_graph=True)


class TestEvalGenerator:
    """Test EvalGenerator with a fake model manager."""

    def test_generate_no_model_manager_raises(self):
        gen = EvalGenerator()
        with pytest.raises(RuntimeError, match="model_manager"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                gen.generate("prompt", "model")
            )

    @pytest.mark.asyncio
    async def test_generate_with_fake_model_manager(self):
        class FakeModelManager:
            async def generate_once(self, prompt, model_name=None,
                                    system_prompt="", max_tokens=256,
                                    temperature=None, top_p=None):
                return "This is a test response."

        gen = EvalGenerator(model_manager=FakeModelManager())
        result = await gen.generate(
            assembled_prompt="[RECENT CONVERSATION] n=1\ntest\n\n[CURRENT USER QUERY]\nhi",
            model="test-model",
            temperature=0.3,
        )

        assert isinstance(result, EvalGenerationResult)
        assert result.response_text == "This is a test response."
        assert result.model == "test-model"
        assert result.temperature == 0.3
        assert result.prompt_hash != ""
        assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_does_not_import_orchestrator(self):
        """EvalGenerator should not import or call the orchestrator."""
        import sys

        class FakeModelManager:
            async def generate_once(self, **kwargs):
                return "response"

        gen = EvalGenerator(model_manager=FakeModelManager())

        # Track imports during generation
        imported_before = set(sys.modules.keys())
        await gen.generate("prompt", "model")
        imported_after = set(sys.modules.keys())

        new_imports = imported_after - imported_before
        orchestrator_imports = [m for m in new_imports if "orchestrator" in m]
        assert orchestrator_imports == [], (
            f"EvalGenerator imported orchestrator modules: {orchestrator_imports}"
        )

    @pytest.mark.asyncio
    async def test_generate_passes_system_message(self):
        class SpyModelManager:
            def __init__(self):
                self.last_system_prompt = None

            async def generate_once(self, prompt, model_name=None,
                                    system_prompt="", max_tokens=256,
                                    temperature=None, top_p=None):
                self.last_system_prompt = system_prompt
                return "ok"

        spy = SpyModelManager()
        gen = EvalGenerator(model_manager=spy)
        await gen.generate("prompt", "model", system_message="Custom system msg")
        assert spy.last_system_prompt == "Custom system msg"

    @pytest.mark.asyncio
    async def test_generate_default_system_message(self):
        class SpyModelManager:
            def __init__(self):
                self.last_system_prompt = None

            async def generate_once(self, prompt, model_name=None,
                                    system_prompt="", max_tokens=256,
                                    temperature=None, top_p=None):
                self.last_system_prompt = system_prompt
                return "ok"

        spy = SpyModelManager()
        gen = EvalGenerator(model_manager=spy)
        await gen.generate("prompt", "model")
        assert "helpful" in spy.last_system_prompt.lower()
