"""Tests for the Phase 5 pairwise judge harness."""

import json
from pathlib import Path

import pytest

from eval.judge import (
    JUDGE_CRITERIA,
    BatchJudge,
    JudgeRunManifest,
    JudgeVerdict,
    PairwiseJudge,
    _parse_scores,
    aggregate_by_section,
    format_section_report,
    load_verdicts,
    parse_judge_response,
)
from eval.no_store_generation import EvalGenerator


# ---------------------------------------------------------------------------
# Fake model manager
# ---------------------------------------------------------------------------

class FakeJudgeModelManager:
    """Returns canned judge output."""

    def __init__(self, winner: str = "A", confidence: float = 0.8):
        self.winner = winner
        self.confidence = confidence
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
        self.calls.append({"prompt_len": len(prompt), "model_name": model_name})
        return (
            f"SCORES_A: accuracy=4 helpfulness=4 conciseness=5 tone=4 grounding=4\n"
            f"SCORES_B: accuracy=3 helpfulness=3 conciseness=3 tone=3 grounding=3\n"
            f"WINNER: {self.winner}\n"
            f"CONFIDENCE: {self.confidence}\n"
            f"EXPLANATION: Response {self.winner} was more natural and concise."
        )


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_basic(self):
        scores = _parse_scores("accuracy=4 helpfulness=3 conciseness=5 tone=4 grounding=3")
        assert scores == {
            "accuracy": 4,
            "helpfulness": 3,
            "conciseness": 5,
            "tone": 4,
            "grounding": 3,
        }

    def test_empty(self):
        assert _parse_scores("") == {}

    def test_partial(self):
        scores = _parse_scores("accuracy=4 helpfulness=bad")
        assert scores == {"accuracy": 4}


class TestParseJudgeResponse:
    def test_full_response(self):
        raw = (
            "SCORES_A: accuracy=4 helpfulness=5 conciseness=4 tone=4 grounding=5\n"
            "SCORES_B: accuracy=3 helpfulness=3 conciseness=2 tone=3 grounding=3\n"
            "WINNER: A\n"
            "CONFIDENCE: 0.85\n"
            "EXPLANATION: Response A was more grounded and concise."
        )
        result = parse_judge_response(raw)
        assert result["winner"] == "A"
        assert result["confidence"] == 0.85
        assert result["scores_a"]["accuracy"] == 4
        assert result["scores_b"]["conciseness"] == 2
        assert "grounded" in result["explanation"]

    def test_tie(self):
        raw = "WINNER: tie\nCONFIDENCE: 0.5\nEXPLANATION: Both similar."
        result = parse_judge_response(raw)
        assert result["winner"] == "tie"

    def test_winner_b(self):
        raw = "WINNER: B\nCONFIDENCE: 0.9"
        result = parse_judge_response(raw)
        assert result["winner"] == "B"

    def test_malformed_defaults(self):
        raw = "Some random LLM output that doesn't follow format"
        result = parse_judge_response(raw)
        assert result["winner"] == "tie"
        assert result["confidence"] == 0.5
        assert result["scores_a"] == {}

    def test_case_insensitive_winner(self):
        raw = "WINNER: a"
        result = parse_judge_response(raw)
        assert result["winner"] == "A"


# ---------------------------------------------------------------------------
# Position randomization tests
# ---------------------------------------------------------------------------

class TestPositionRandomization:
    def test_deterministic(self):
        """Same pair always gets same position."""
        gen = EvalGenerator(model_manager=FakeJudgeModelManager())
        judge = PairwiseJudge(generator=gen)
        _, _, pos1 = judge._randomize_position("bl", "var", "snap1", "var1")
        _, _, pos2 = judge._randomize_position("bl", "var", "snap1", "var1")
        assert pos1 == pos2

    def test_different_pairs_can_differ(self):
        """Different pairs may get different positions."""
        gen = EvalGenerator(model_manager=FakeJudgeModelManager())
        judge = PairwiseJudge(generator=gen)
        positions = set()
        for i in range(20):
            _, _, pos = judge._randomize_position("bl", "var", f"snap{i}", f"var{i}")
            positions.add(pos)
        # With 20 different pairs, extremely likely to get both A and B
        assert len(positions) == 2

    def test_baseline_in_correct_position(self):
        gen = EvalGenerator(model_manager=FakeJudgeModelManager())
        judge = PairwiseJudge(generator=gen)
        resp_a, resp_b, pos = judge._randomize_position(
            "baseline_text", "variant_text", "snap1", "var1"
        )
        if pos == "A":
            assert resp_a == "baseline_text"
            assert resp_b == "variant_text"
        else:
            assert resp_a == "variant_text"
            assert resp_b == "baseline_text"


# ---------------------------------------------------------------------------
# JudgeVerdict tests
# ---------------------------------------------------------------------------

class TestJudgeVerdict:
    def test_roundtrip(self):
        v = JudgeVerdict(
            snapshot_id="snap1",
            variant_id="snap1_LOO_memories",
            strategy="leave_one_out",
            sections_removed=["memories"],
            sections_added=[],
            query_text="test query",
            winner="A",
            winner_is_baseline=True,
            confidence=0.8,
            baseline_position="A",
            judge_model="gpt-4o-mini",
            scores_a={"accuracy": 4},
            scores_b={"accuracy": 3},
        )
        d = v.to_dict()
        v2 = JudgeVerdict.from_dict(d)
        assert v2.winner == "A"
        assert v2.winner_is_baseline is True
        assert v2.sections_removed == ["memories"]


# ---------------------------------------------------------------------------
# JudgeRunManifest tests
# ---------------------------------------------------------------------------

class TestJudgeRunManifest:
    def test_save_load(self, tmp_path):
        m = JudgeRunManifest(
            run_id="test",
            source_generation_run="20260506_100331",
            judge_model="gpt-4o-mini",
            started_at="2026-05-06T12:00:00",
            total_pairs=100,
            completed_pairs=50,
            baseline_wins=30,
            variant_wins=10,
            ties=10,
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        m2 = JudgeRunManifest.load(path)
        assert m2.completed_pairs == 50
        assert m2.baseline_wins == 30


# ---------------------------------------------------------------------------
# Judge pair tests (async)
# ---------------------------------------------------------------------------

class TestJudgePair:
    @pytest.mark.asyncio
    async def test_basic_judgment(self):
        fake_mm = FakeJudgeModelManager(winner="A", confidence=0.8)
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)

        verdict = await judge.judge_pair(
            query="Hey man",
            baseline_response="Hey! How's it going?",
            variant_response="Hello. How are you doing today? Is there anything I can help you with?",
            snapshot_id="snap1",
            variant_id="snap1_LOO_memories",
            strategy="leave_one_out",
            sections_removed=["memories"],
            sections_added=[],
        )

        assert verdict.winner in ("A", "B", "tie")
        assert verdict.confidence > 0
        assert verdict.judge_model == "gpt-4o-mini"
        assert verdict.judge_time_ms >= 0
        assert len(verdict.scores_a) > 0

    @pytest.mark.asyncio
    async def test_unblinding_baseline_a(self):
        """When baseline is A and judge picks A, winner_is_baseline=True."""
        fake_mm = FakeJudgeModelManager(winner="A")
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)

        # Force a deterministic position by trying many pairs
        # and checking one where baseline is A
        for i in range(50):
            verdict = await judge.judge_pair(
                query="test", baseline_response="bl", variant_response="var",
                snapshot_id=f"snap{i}", variant_id=f"var{i}",
                strategy="leave_one_out",
                sections_removed=["memories"], sections_added=[],
            )
            if verdict.baseline_position == "A":
                assert verdict.winner_is_baseline is True
                return

        pytest.fail("Could not find a pair where baseline was in position A")

    @pytest.mark.asyncio
    async def test_unblinding_baseline_b(self):
        """When baseline is B and judge picks A, winner_is_baseline=False."""
        fake_mm = FakeJudgeModelManager(winner="A")
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)

        for i in range(50):
            verdict = await judge.judge_pair(
                query="test", baseline_response="bl", variant_response="var",
                snapshot_id=f"snap{i}", variant_id=f"var{i}",
                strategy="leave_one_out",
                sections_removed=["memories"], sections_added=[],
            )
            if verdict.baseline_position == "B":
                assert verdict.winner_is_baseline is False
                return

        pytest.fail("Could not find a pair where baseline was in position B")

    @pytest.mark.asyncio
    async def test_tie_unblinding(self):
        fake_mm = FakeJudgeModelManager(winner="tie")
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)

        verdict = await judge.judge_pair(
            query="test", baseline_response="bl", variant_response="var",
            snapshot_id="snap1", variant_id="var1",
            strategy="leave_one_out",
            sections_removed=["memories"], sections_added=[],
        )
        assert verdict.winner_is_baseline is None


# ---------------------------------------------------------------------------
# Batch judge tests
# ---------------------------------------------------------------------------

class TestBatchJudge:
    @pytest.mark.asyncio
    async def test_batch_run(self, tmp_path):
        """Run batch judging against synthetic Phase 4 results."""
        # Create fake Phase 4 results
        results_dir = tmp_path / "gen_run" / "results"
        results_dir.mkdir(parents=True)

        # Baseline
        baseline = {
            "pair": {
                "snapshot_id": "snap1",
                "variant_id": "__baseline__",
                "query_text": "Hey man",
                "strategy": "baseline",
                "sections_removed": [],
                "sections_added": [],
            },
            "result": {"response_text": "Hey! What's up?"},
        }
        (results_dir / "snap1___baseline__.json").write_text(json.dumps(baseline))

        # Variant
        variant = {
            "pair": {
                "snapshot_id": "snap1",
                "variant_id": "snap1_LOO_memories",
                "query_text": "Hey man",
                "strategy": "leave_one_out",
                "sections_removed": ["memories"],
                "sections_added": [],
            },
            "result": {"response_text": "Hello! How can I assist you today?"},
        }
        (results_dir / "snap1_snap1_LOO_memories.json").write_text(json.dumps(variant))

        fake_mm = FakeJudgeModelManager(winner="A")
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)
        batch = BatchJudge(judge=judge, requests_per_minute=0)

        manifest = await batch.run(
            str(tmp_path / "gen_run"),
            output_dir=str(tmp_path / "judgments"),
        )

        assert manifest.completed_pairs == 1
        assert manifest.failed_pairs == 0
        assert manifest.completed_at is not None

    @pytest.mark.asyncio
    async def test_plan_judgments(self, tmp_path):
        results_dir = tmp_path / "gen_run" / "results"
        results_dir.mkdir(parents=True)

        baseline = {
            "pair": {"snapshot_id": "s1", "variant_id": "__baseline__",
                     "query_text": "q", "strategy": "baseline",
                     "sections_removed": [], "sections_added": []},
            "result": {"response_text": "bl"},
        }
        v1 = {
            "pair": {"snapshot_id": "s1", "variant_id": "s1_LOO_a",
                     "query_text": "q", "strategy": "leave_one_out",
                     "sections_removed": ["a"], "sections_added": []},
            "result": {"response_text": "v1"},
        }
        v2 = {
            "pair": {"snapshot_id": "s1", "variant_id": "s1_LOO_b",
                     "query_text": "q", "strategy": "leave_one_out",
                     "sections_removed": ["b"], "sections_added": []},
            "result": {"response_text": "v2"},
        }
        (results_dir / "s1___baseline__.json").write_text(json.dumps(baseline))
        (results_dir / "s1_s1_LOO_a.json").write_text(json.dumps(v1))
        (results_dir / "s1_s1_LOO_b.json").write_text(json.dumps(v2))

        fake_mm = FakeJudgeModelManager()
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)
        batch = BatchJudge(judge=judge)

        pairs = batch.plan_judgments(str(tmp_path / "gen_run"))
        assert len(pairs) == 2  # 2 variants, 1 baseline

    @pytest.mark.asyncio
    async def test_resume_skips_existing(self, tmp_path):
        results_dir = tmp_path / "gen_run" / "results"
        results_dir.mkdir(parents=True)

        baseline = {
            "pair": {"snapshot_id": "s1", "variant_id": "__baseline__",
                     "query_text": "q", "strategy": "baseline",
                     "sections_removed": [], "sections_added": []},
            "result": {"response_text": "bl"},
        }
        variant = {
            "pair": {"snapshot_id": "s1", "variant_id": "s1_LOO_mem",
                     "query_text": "q", "strategy": "leave_one_out",
                     "sections_removed": ["memories"], "sections_added": []},
            "result": {"response_text": "var"},
        }
        (results_dir / "s1___baseline__.json").write_text(json.dumps(baseline))
        (results_dir / "s1_s1_LOO_mem.json").write_text(json.dumps(variant))

        fake_mm = FakeJudgeModelManager()
        gen = EvalGenerator(model_manager=fake_mm)
        judge = PairwiseJudge(generator=gen)
        batch = BatchJudge(judge=judge, requests_per_minute=0)

        # First run
        m1 = await batch.run(
            str(tmp_path / "gen_run"),
            output_dir=str(tmp_path / "judgments"),
            run_id="resume_test",
        )
        assert m1.completed_pairs == 1

        # Second run (resume)
        fake_mm.calls.clear()
        m2 = await batch.run(
            str(tmp_path / "gen_run"),
            output_dir=str(tmp_path / "judgments"),
            run_id="resume_test",
        )
        assert m2.skipped_pairs == 1
        assert m2.completed_pairs == 0
        assert len(fake_mm.calls) == 0


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def _make_verdict(
        self,
        section: str,
        winner_is_baseline: bool,
        confidence: float = 0.8,
        baseline_pos: str = "A",
    ) -> JudgeVerdict:
        winner = baseline_pos if winner_is_baseline else ("B" if baseline_pos == "A" else "A")
        scores_a = {c: 4 for c in JUDGE_CRITERIA}
        scores_b = {c: 3 for c in JUDGE_CRITERIA}
        if not winner_is_baseline and baseline_pos == "A":
            scores_a, scores_b = scores_b, scores_a
        return JudgeVerdict(
            snapshot_id="snap1",
            variant_id=f"snap1_LOO_{section}",
            strategy="leave_one_out",
            sections_removed=[section],
            sections_added=[],
            query_text="test",
            winner=winner,
            winner_is_baseline=winner_is_baseline,
            confidence=confidence,
            baseline_position=baseline_pos,
            scores_a=scores_a,
            scores_b=scores_b,
        )

    def test_aggregate_basic(self):
        verdicts = [
            self._make_verdict("memories", True),
            self._make_verdict("memories", True),
            self._make_verdict("memories", False),
        ]
        stats = aggregate_by_section(verdicts)
        assert "memories" in stats
        assert stats["memories"]["baseline_wins"] == 2
        assert stats["memories"]["variant_wins"] == 1
        assert stats["memories"]["total"] == 3
        assert stats["memories"]["baseline_win_rate"] == pytest.approx(2/3)

    def test_aggregate_ignores_non_loo(self):
        v = JudgeVerdict(
            snapshot_id="s1", variant_id="s1_BUNDLE_x",
            strategy="bundle", sections_removed=["a", "b"],
            sections_added=[], query_text="test",
            winner="A", winner_is_baseline=True,
        )
        stats = aggregate_by_section([v])
        assert len(stats) == 0

    def test_format_report(self):
        verdicts = [
            self._make_verdict("memories", True),
            self._make_verdict("memories", False),
            self._make_verdict("user_profile", True),
        ]
        stats = aggregate_by_section(verdicts)
        report = format_section_report(stats)
        assert "memories" in report
        assert "user_profile" in report
        assert "Section Ablation" in report


# ---------------------------------------------------------------------------
# Load verdicts tests
# ---------------------------------------------------------------------------

class TestLoadVerdicts:
    def test_load_from_dir(self, tmp_path):
        verdicts_dir = tmp_path / "verdicts"
        verdicts_dir.mkdir()

        v = JudgeVerdict(
            snapshot_id="s1", variant_id="s1_LOO_mem",
            strategy="leave_one_out", sections_removed=["memories"],
            sections_added=[], query_text="test", winner="A",
        )
        (verdicts_dir / "s1_s1_LOO_mem.json").write_text(json.dumps(v.to_dict()))

        loaded = load_verdicts(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].variant_id == "s1_LOO_mem"

    def test_load_empty(self, tmp_path):
        loaded = load_verdicts(str(tmp_path / "nonexistent"))
        assert loaded == []
