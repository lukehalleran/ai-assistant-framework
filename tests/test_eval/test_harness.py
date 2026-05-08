"""Tests for the Phase 4 generation harness."""

import json
from pathlib import Path

import pytest

from eval.harness import (
    BASELINE_VARIANT_ID,
    GenerationHarness,
    GenerationPair,
    HarnessConfig,
    RunManifest,
    load_run_manifest,
    load_run_results,
)
from eval.no_store_generation import EvalGenerator
from eval.schema import (
    EvalGenerationResult,
    PromptProvenance,
    PromptSnapshot,
    SectionSnapshot,
    SnapshotLayer,
    compute_prompt_hash,
)
from eval.section_registry import SECTION_REGISTRY


# ---------------------------------------------------------------------------
# Fake model manager for testing
# ---------------------------------------------------------------------------

class FakeModelManager:
    """Returns canned responses. Tracks calls for assertion."""

    def __init__(self, response: str = "Fake LLM response."):
        self.response = response
        self.calls: list[dict] = []

    async def generate_once(
        self,
        prompt: str,
        model_name: str = None,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        self.calls.append({
            "prompt_len": len(prompt),
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self.response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section(key: str, text: str = "") -> SectionSnapshot:
    reg = SECTION_REGISTRY.get(key)
    if reg is None:
        raise ValueError(f"Unknown section key: {key}")
    if not text:
        text = f"[{reg.header}] n=1\nContent for {key}"
    return SectionSnapshot(
        key=key,
        header=reg.header,
        structured_content=text,
        formatted_text=text,
        token_count=max(1, len(text) // 4),
        source_field=reg.source_field,
        category=reg.category.value,
        eligible_for_ablation=reg.eligible_for_ablation,
        structurally_required=reg.structurally_required,
        assembly_order=reg.assembly_order,
    )


def _make_snapshot(
    section_keys: list[str] | None = None,
    snapshot_id: str = "test1234",
    query_text: str = "test query",
) -> PromptSnapshot:
    if section_keys is None:
        section_keys = [
            "current_query",
            "time_context",
            "memories",
            "user_profile",
        ]

    sections = {k: _make_section(k) for k in section_keys}
    prompt_text = "\n\n".join(
        s.formatted_text
        for s in sorted(sections.values(), key=lambda s: s.assembly_order)
    )

    layer = SnapshotLayer(
        layer_name="post_hygiene",
        sections=sections,
        layer_content_hash="fakehash",
        prompt_text=prompt_text,
        prompt_hash_exact=compute_prompt_hash(prompt_text),
        prompt_hash_normalized=compute_prompt_hash(prompt_text, normalize=True),
        capture_timestamp="2026-05-05T00:00:00+00:00",
    )

    return PromptSnapshot(
        snapshot_id=snapshot_id,
        query_text=query_text,
        query_timestamp="2026-05-05T00:00:00+00:00",
        processed_query=query_text,
        detected_intent="",
        detected_tone="",
        provenance=PromptProvenance(
            model_name="test-model",
            git_commit_hash="abc1234",
            system_prompt_hash="sys_hash",
        ),
        layers={"post_hygiene": layer},
        retrieval_metadata={},
        assembly_metadata={},
    )


# ---------------------------------------------------------------------------
# HarnessConfig tests
# ---------------------------------------------------------------------------

class TestHarnessConfig:
    def test_defaults(self):
        c = HarnessConfig()
        assert c.model == "gpt-4o-mini"
        assert c.temperature == 0.3
        assert c.max_tokens == 2048

    def test_roundtrip(self):
        c = HarnessConfig(model="sonnet-4.5", temperature=0.7)
        d = c.to_dict()
        c2 = HarnessConfig.from_dict(d)
        assert c2.model == "sonnet-4.5"
        assert c2.temperature == 0.7


# ---------------------------------------------------------------------------
# GenerationPair tests
# ---------------------------------------------------------------------------

class TestGenerationPair:
    def test_result_filename(self):
        p = GenerationPair(
            snapshot_id="snap1234",
            variant_id="snap1234_LOO_memories",
            query_text="test",
            strategy="leave_one_out",
            description="LOO: removed memories",
        )
        assert p.result_filename() == "snap1234_snap1234_LOO_memories.json"

    def test_baseline_filename(self):
        p = GenerationPair(
            snapshot_id="snap1234",
            variant_id=BASELINE_VARIANT_ID,
            query_text="test",
            strategy="baseline",
            description="Full prompt",
        )
        assert BASELINE_VARIANT_ID in p.result_filename()


# ---------------------------------------------------------------------------
# RunManifest tests
# ---------------------------------------------------------------------------

class TestRunManifest:
    def test_save_load_roundtrip(self, tmp_path):
        m = RunManifest(
            run_id="test_run",
            config={"model": "test"},
            started_at="2026-05-05T00:00:00",
            total_pairs=10,
            completed_pairs=5,
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        m2 = RunManifest.load(path)
        assert m2.run_id == "test_run"
        assert m2.total_pairs == 10
        assert m2.completed_pairs == 5


# ---------------------------------------------------------------------------
# Plan pairs tests
# ---------------------------------------------------------------------------

class TestPlanPairs:
    def test_includes_baseline(self):
        snap = _make_snapshot()
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"test1234": snap})
        baselines = [p for p in pairs if p.variant_id == BASELINE_VARIANT_ID]
        assert len(baselines) == 1
        assert baselines[0].strategy == "baseline"

    def test_includes_variants(self):
        snap = _make_snapshot()
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"test1234": snap})
        variants = [p for p in pairs if p.variant_id != BASELINE_VARIANT_ID]
        assert len(variants) > 0

    def test_multiple_snapshots(self):
        snap1 = _make_snapshot(snapshot_id="snap_aaa")
        snap2 = _make_snapshot(snapshot_id="snap_bbb")
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"snap_aaa": snap1, "snap_bbb": snap2})
        baselines = [p for p in pairs if p.variant_id == BASELINE_VARIANT_ID]
        assert len(baselines) == 2

    def test_baseline_has_full_tokens(self):
        snap = _make_snapshot()
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"test1234": snap})
        baseline = [p for p in pairs if p.variant_id == BASELINE_VARIANT_ID][0]
        assert baseline.token_count_estimate > 0
        assert baseline.token_count_estimate == baseline.parent_token_count

    def test_pair_query_text_from_snapshot(self):
        snap = _make_snapshot(query_text="What is my cat's name?")
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"test1234": snap})
        for p in pairs:
            assert p.query_text == "What is my cat's name?"

    def test_skips_snapshot_without_post_hygiene(self):
        snap = _make_snapshot()
        snap.layers.clear()
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        pairs = harness.plan_pairs({"test1234": snap})
        assert pairs == []


# ---------------------------------------------------------------------------
# Generation tests (async, with fake model)
# ---------------------------------------------------------------------------

class TestGeneration:
    @pytest.mark.asyncio
    async def test_baseline_generation(self, tmp_path):
        fake_mm = FakeModelManager(response="Test response for baseline")
        generator = EvalGenerator(model_manager=fake_mm)
        harness = GenerationHarness(generator=generator)

        snap = _make_snapshot()
        manifest = await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="test_run",
        )

        assert manifest.completed_pairs > 0
        assert manifest.failed_pairs == 0
        assert manifest.total_generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_results_saved_to_disk(self, tmp_path):
        fake_mm = FakeModelManager(response="Saved response")
        generator = EvalGenerator(model_manager=fake_mm)
        harness = GenerationHarness(generator=generator)

        snap = _make_snapshot()
        await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="test_run",
        )

        results_dir = tmp_path / "test_run" / "results"
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) > 0

        # Check result structure
        data = json.loads(result_files[0].read_text())
        assert "pair" in data
        assert "result" in data
        assert "prompt_text" in data
        assert data["result"]["response_text"] == "Saved response"

    @pytest.mark.asyncio
    async def test_manifest_saved(self, tmp_path):
        fake_mm = FakeModelManager()
        generator = EvalGenerator(model_manager=fake_mm)
        harness = GenerationHarness(generator=generator)

        snap = _make_snapshot()
        await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="test_run",
        )

        manifest = load_run_manifest(str(tmp_path / "test_run"))
        assert manifest is not None
        assert manifest.run_id == "test_run"
        assert manifest.completed_at is not None

    @pytest.mark.asyncio
    async def test_resume_skips_completed(self, tmp_path):
        fake_mm = FakeModelManager()
        generator = EvalGenerator(model_manager=fake_mm)
        harness = GenerationHarness(generator=generator)

        snap = _make_snapshot()

        # First run
        m1 = await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="resume_test",
        )
        first_completed = m1.completed_pairs

        # Second run (resume) — same run_id
        fake_mm.calls.clear()
        m2 = await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="resume_test",
        )

        assert m2.skipped_pairs == first_completed
        assert m2.completed_pairs == 0  # nothing new to generate
        assert len(fake_mm.calls) == 0  # no LLM calls made

    @pytest.mark.asyncio
    async def test_model_receives_correct_params(self, tmp_path):
        fake_mm = FakeModelManager()
        config = HarnessConfig(
            model="test-model-x",
            temperature=0.5,
            max_tokens=1024,
        )
        generator = EvalGenerator(model_manager=fake_mm)
        harness = GenerationHarness(generator=generator, config=config)

        snap = _make_snapshot(
            section_keys=["current_query", "time_context"],  # minimal
        )
        await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="param_test",
        )

        # Check that the fake model saw the right params
        assert len(fake_mm.calls) > 0
        call = fake_mm.calls[0]
        assert call["model_name"] == "test-model-x"
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_generation_error_recorded(self, tmp_path):
        """If generation fails, error is recorded but run continues."""

        class FailingModelManager:
            call_count = 0

            async def generate_once(self, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("API error")
                return "OK response"

        generator = EvalGenerator(model_manager=FailingModelManager())
        harness = GenerationHarness(generator=generator)

        snap = _make_snapshot()
        manifest = await harness.run(
            {"test1234": snap},
            output_dir=str(tmp_path),
            run_id="error_test",
        )

        assert manifest.failed_pairs >= 1
        assert len(manifest.errors) >= 1
        assert manifest.errors[0]["stage"] == "generation"
        # Other pairs should still complete
        assert manifest.completed_pairs > 0


# ---------------------------------------------------------------------------
# Load results tests
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_load_run_results(self, tmp_path):
        results_dir = tmp_path / "test_run" / "results"
        results_dir.mkdir(parents=True)

        # Write a fake result
        data = {
            "pair": {"snapshot_id": "s1", "variant_id": "__baseline__"},
            "result": {"response_text": "hello"},
            "prompt_text": "test prompt",
        }
        (results_dir / "s1___baseline__.json").write_text(json.dumps(data))

        results = load_run_results(str(tmp_path / "test_run"))
        assert len(results) == 1
        assert "s1___baseline__.json" in results

    def test_load_empty_dir(self, tmp_path):
        results = load_run_results(str(tmp_path / "nonexistent"))
        assert results == {}

    def test_load_manifest(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        m = RunManifest(
            run_id="test", config={}, started_at="now",
            total_pairs=5, completed_pairs=3,
        )
        m.save(run_dir / "manifest.json")

        loaded = load_run_manifest(str(run_dir))
        assert loaded is not None
        assert loaded.completed_pairs == 3

    def test_load_missing_manifest(self, tmp_path):
        loaded = load_run_manifest(str(tmp_path / "no_run"))
        assert loaded is None


# ---------------------------------------------------------------------------
# Snapshot loading tests
# ---------------------------------------------------------------------------

class TestSnapshotLoading:
    def test_load_snapshots_from_dir(self, tmp_path):
        from eval.snapshots import save_snapshot
        snap = _make_snapshot()
        save_snapshot(snap, str(tmp_path))

        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        snapshots = harness.load_snapshots(str(tmp_path))
        assert len(snapshots) == 1
        assert "test1234" in snapshots

    def test_load_empty_dir(self, tmp_path):
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        snapshots = harness.load_snapshots(str(tmp_path))
        assert snapshots == {}

    def test_load_nonexistent_dir(self):
        harness = GenerationHarness(
            generator=EvalGenerator(model_manager=None),
        )
        snapshots = harness.load_snapshots("/nonexistent/path")
        assert snapshots == {}
