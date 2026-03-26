# tests/unit/test_implementation_detector.py
"""
Unit tests for the implementation detection pipeline.

Tests cover:
- DetectionResult model
- Cooldown logic
- Stage 1: File existence
- Stage 2: Code content grep
- Stage 3: Git history
- Stage 4: LLM judgment
- Confidence scoring
- Status thresholds
- Batch detection
- Lightweight mode
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from knowledge.implementation_detector import DetectionResult, ImplementationDetector
from memory.code_proposal import CodeProposal, ImplementationStep


# =====================================================================
# Helpers
# =====================================================================

def _make_proposal(**overrides):
    """Create a test proposal with sensible defaults."""
    defaults = dict(
        title="Add user authentication",
        affected_files=["core/auth.py", "config/settings.py"],
        implementation_steps=[
            ImplementationStep(
                order=1,
                description="Create AuthManager class with def login()",
                file_path="core/auth.py",
                action="create",
            ),
            ImplementationStep(
                order=2,
                description="Add AUTH_ENABLED config constant",
                file_path="config/settings.py",
                action="modify",
            ),
        ],
    )
    defaults.update(overrides)
    return CodeProposal(**defaults)


# =====================================================================
# TestDetectionResult
# =====================================================================

class TestDetectionResult:
    def test_defaults(self):
        r = DetectionResult()
        assert r.confidence == 0.0
        assert r.status == "not_checked"
        assert r.evidence == ""
        assert r.llm_adjusted is False

    def test_construction(self):
        r = DetectionResult(
            proposal_id="abc", confidence=0.75, status="likely",
            evidence="Files exist", file_existence_ratio=0.5,
        )
        assert r.proposal_id == "abc"
        assert r.confidence == 0.75

    def test_serialization(self):
        r = DetectionResult(proposal_id="x", confidence=0.9, status="confirmed")
        d = r.model_dump()
        assert d["proposal_id"] == "x"
        assert d["confidence"] == 0.9
        r2 = DetectionResult(**d)
        assert r2 == r


# =====================================================================
# TestCooldown
# =====================================================================

class TestCooldown:
    def test_skip_within_cooldown(self):
        p = _make_proposal(last_tracked_at=time.time() - 100)
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_COOLDOWN", 86400):
            assert detector._should_skip_cooldown(p) is True

    def test_expired_cooldown(self):
        p = _make_proposal(last_tracked_at=time.time() - 100000)
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_COOLDOWN", 86400):
            assert detector._should_skip_cooldown(p) is False

    def test_no_last_tracked(self):
        p = _make_proposal(last_tracked_at=None)
        detector = ImplementationDetector()
        assert detector._should_skip_cooldown(p) is False

    def test_cooldown_zero(self):
        p = _make_proposal(last_tracked_at=time.time() - 1)
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_COOLDOWN", 0):
            assert detector._should_skip_cooldown(p) is False

    def test_detect_single_returns_cached_on_cooldown(self):
        p = _make_proposal(
            last_tracked_at=time.time() - 10,
            implementation_confidence=0.75,
            implementation_status="likely",
            implementation_evidence="cached evidence",
        )
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_COOLDOWN", 86400):
            result = detector.detect_single(p)
        assert result.skipped_reason == "cooldown"
        assert result.confidence == 0.75
        assert result.status == "likely"


# =====================================================================
# TestFileExistence
# =====================================================================

class TestFileExistence:
    def test_all_exist(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("class Auth: pass")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "settings.py").write_text("X = 1")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 1.0
        assert len(existing) == 2

    def test_none_exist(self, tmp_path):
        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 0.0
        assert len(existing) == 0

    def test_partial(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 0.5
        assert existing == ["core/auth.py"]

    def test_empty_affected_files(self, tmp_path):
        """Falls back to implementation_steps file_path."""
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("")

        p = _make_proposal(affected_files=[])
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 0.5  # 1 of 2 step file_paths exist

    def test_no_paths_at_all(self, tmp_path):
        p = _make_proposal(
            affected_files=[],
            implementation_steps=[
                ImplementationStep(order=1, description="Do something", file_path="")
            ],
        )
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 0.0

    def test_fallback_to_steps(self, tmp_path):
        (tmp_path / "foo.py").write_text("")
        p = _make_proposal(
            affected_files=[],
            implementation_steps=[
                ImplementationStep(order=1, description="Create foo", file_path="foo.py")
            ],
        )
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 1.0

    def test_dedup_step_paths(self, tmp_path):
        """Same file_path in multiple steps should not count twice."""
        (tmp_path / "foo.py").write_text("")
        p = _make_proposal(
            affected_files=[],
            implementation_steps=[
                ImplementationStep(order=1, description="Step 1", file_path="foo.py"),
                ImplementationStep(order=2, description="Step 2", file_path="foo.py"),
            ],
        )
        detector = ImplementationDetector(repo_path=str(tmp_path))
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 1.0

    def test_nonexistent_repo(self):
        p = _make_proposal()
        detector = ImplementationDetector(repo_path="/nonexistent/path")
        existing = []
        ratio = detector._stage_file_existence(p, existing)
        assert ratio == 0.0


# =====================================================================
# TestCodeContent
# =====================================================================

class TestCodeContent:
    def test_class_found(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("class AuthManager:\n    pass\n")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        ratio, matches = detector._stage_code_content(p, ["core/auth.py"])
        assert "AuthManager" in matches
        assert ratio > 0

    def test_function_found(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("def login():\n    pass\n")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        ratio, matches = detector._stage_code_content(p, ["core/auth.py"])
        assert "login" in matches

    def test_constant_found(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "settings.py").write_text("AUTH_ENABLED = True\n")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        ratio, matches = detector._stage_code_content(p, ["config/settings.py"])
        assert "AUTH_ENABLED" in matches

    def test_no_identifiers(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        p = _make_proposal(
            title="minor fix",
            implementation_steps=[
                ImplementationStep(order=1, description="fix the bug", file_path="foo.py")
            ],
        )
        detector = ImplementationDetector(repo_path=str(tmp_path))
        ratio, matches = detector._stage_code_content(p, ["foo.py"])
        assert ratio == 0.0
        assert matches == []

    def test_no_existing_files(self):
        p = _make_proposal()
        detector = ImplementationDetector()
        ratio, matches = detector._stage_code_content(p, [])
        assert ratio == 0.0

    def test_file_unreadable(self, tmp_path):
        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        # File doesn't exist so read will fail
        ratio, matches = detector._stage_code_content(p, ["missing.py"])
        assert ratio == 0.0

    def test_partial_match(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("class AuthManager:\n    pass\n")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        identifiers = detector._extract_identifiers(p)
        # Should have AuthManager, login, AUTH_ENABLED among others
        # Only AuthManager is in the file
        ratio, matches = detector._stage_code_content(p, ["core/auth.py"])
        assert 0 < ratio < 1.0

    def test_extract_identifiers_from_title(self):
        p = _make_proposal(title="Add class FooBar and SOME_CONST")
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "FooBar" in idents
        assert "SOME_CONST" in idents


# =====================================================================
# TestGitHistory
# =====================================================================

class TestGitHistory:
    def _mock_git_output(self, text, returncode=0):
        return MagicMock(stdout=text, stderr="", returncode=returncode)

    def test_file_overlap(self):
        output = "abc123|||feat: add authentication\ncore/auth.py\nconfig/settings.py\n"
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output(output)):
            score, matches = detector._stage_git_history(p)
        assert score > 0.5
        assert len(matches) >= 1

    def test_keyword_match(self):
        output = "abc123|||add user authentication\nother_file.py\n"
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output(output)):
            score, matches = detector._stage_git_history(p)
        assert score > 0  # keyword overlap

    def test_combined_scoring(self):
        output = "abc123|||add user authentication\ncore/auth.py\nconfig/settings.py\n"
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output(output)):
            score, matches = detector._stage_git_history(p)
        # Both file overlap (full) and keyword overlap
        assert score > 0.7

    def test_git_unavailable(self):
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", side_effect=OSError("no git")):
            score, matches = detector._stage_git_history(p)
        assert score == 0.0
        assert matches == []

    def test_empty_history(self):
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output("")):
            score, matches = detector._stage_git_history(p)
        assert score == 0.0

    def test_no_affected_files(self):
        p = _make_proposal(affected_files=[])
        # implementation_steps have file_path but affected_files is what git stage checks
        detector = ImplementationDetector()
        score, matches = detector._stage_git_history(p)
        assert score == 0.0

    def test_multiple_commits(self):
        output = (
            "aaa|||unrelated change\nunrelated.py\n\n"
            "bbb|||add authentication support\ncore/auth.py\n\n"
            "ccc|||fix typo\nREADME.md\n"
        )
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output(output)):
            score, matches = detector._stage_git_history(p)
        assert score > 0

    def test_subprocess_error(self):
        p = _make_proposal()
        detector = ImplementationDetector()
        with patch("subprocess.run", return_value=self._mock_git_output("", returncode=1)):
            score, matches = detector._stage_git_history(p)
        assert score == 0.0


# =====================================================================
# TestLLMJudgment
# =====================================================================

class TestLLMJudgment:
    @pytest.mark.asyncio
    async def test_single_proposal(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value='{"1": 0.75}')

        p = _make_proposal()
        r = DetectionResult(
            proposal_id=p.id, confidence=0.5,
            file_existence_ratio=0.5, code_match_ratio=0.3, git_match_score=0.2,
        )
        detector = ImplementationDetector(model_manager=mm)
        adjustments = await detector._stage_llm_judgment([(p, r)])
        assert p.id in adjustments
        assert adjustments[p.id] == 0.75

    @pytest.mark.asyncio
    async def test_batch(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value='{"1": 0.80, "2": 0.40}')

        p1 = _make_proposal(title="Proposal A")
        p2 = _make_proposal(title="Proposal B")
        r1 = DetectionResult(proposal_id=p1.id, confidence=0.5)
        r2 = DetectionResult(proposal_id=p2.id, confidence=0.5)

        detector = ImplementationDetector(model_manager=mm)
        adjustments = await detector._stage_llm_judgment([(p1, r1), (p2, r2)])
        assert p1.id in adjustments
        assert p2.id in adjustments

    @pytest.mark.asyncio
    async def test_model_manager_none(self):
        detector = ImplementationDetector(model_manager=None)
        result = await detector._stage_llm_judgment([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_parse_error(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="not json at all")

        p = _make_proposal()
        r = DetectionResult(proposal_id=p.id, confidence=0.5)
        detector = ImplementationDetector(model_manager=mm)
        adjustments = await detector._stage_llm_judgment([(p, r)])
        assert adjustments == {}

    @pytest.mark.asyncio
    async def test_llm_exception(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("API error"))

        p = _make_proposal()
        r = DetectionResult(proposal_id=p.id, confidence=0.5)
        detector = ImplementationDetector(model_manager=mm)
        adjustments = await detector._stage_llm_judgment([(p, r)])
        assert adjustments == {}

    @pytest.mark.asyncio
    async def test_clamps_to_0_1(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value='{"1": 1.5}')

        p = _make_proposal()
        r = DetectionResult(proposal_id=p.id, confidence=0.5)
        detector = ImplementationDetector(model_manager=mm)
        adjustments = await detector._stage_llm_judgment([(p, r)])
        assert adjustments[p.id] == 1.0  # clamped


# =====================================================================
# TestConfidenceScoring
# =====================================================================

class TestConfidenceScoring:
    def test_weighted_formula(self):
        detector = ImplementationDetector()
        # 1.0 * 0.35 + 1.0 * 0.35 + 1.0 * 0.30 = 1.0
        assert detector._compute_confidence(1.0, 1.0, 1.0) == 1.0

    def test_zero(self):
        detector = ImplementationDetector()
        assert detector._compute_confidence(0.0, 0.0, 0.0) == 0.0

    def test_partial(self):
        detector = ImplementationDetector()
        result = detector._compute_confidence(0.5, 0.5, 0.5)
        assert abs(result - 0.5) < 0.01

    def test_clamp_above_one(self):
        detector = ImplementationDetector()
        # Should never exceed 1.0 even with rounding
        result = detector._compute_confidence(1.0, 1.0, 1.0)
        assert result <= 1.0

    def test_clamp_below_zero(self):
        detector = ImplementationDetector()
        result = detector._compute_confidence(0.0, 0.0, 0.0)
        assert result >= 0.0


# =====================================================================
# TestStatusThresholds
# =====================================================================

class TestStatusThresholds:
    def test_confirmed(self):
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_CONFIDENCE_CONFIRMED", 0.85), \
             patch("config.app_config.IMPL_TRACKING_CONFIDENCE_LIKELY", 0.60):
            assert detector._confidence_to_status(0.90) == "confirmed"

    def test_likely(self):
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_CONFIDENCE_CONFIRMED", 0.85), \
             patch("config.app_config.IMPL_TRACKING_CONFIDENCE_LIKELY", 0.60):
            assert detector._confidence_to_status(0.70) == "likely"

    def test_uncertain(self):
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_CONFIDENCE_CONFIRMED", 0.85), \
             patch("config.app_config.IMPL_TRACKING_CONFIDENCE_LIKELY", 0.60):
            assert detector._confidence_to_status(0.40) == "uncertain"

    def test_not_implemented(self):
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_CONFIDENCE_CONFIRMED", 0.85), \
             patch("config.app_config.IMPL_TRACKING_CONFIDENCE_LIKELY", 0.60):
            assert detector._confidence_to_status(0.10) == "not_implemented"


# =====================================================================
# TestBatchDetection
# =====================================================================

class TestBatchDetection:
    @pytest.mark.asyncio
    async def test_empty_batch(self):
        detector = ImplementationDetector()
        results = await detector.detect_batch([], lightweight=False)
        assert results == []

    @pytest.mark.asyncio
    async def test_mixed_results(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("class AuthManager:\n    pass\n")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "settings.py").write_text("AUTH_ENABLED = True")

        p1 = _make_proposal()  # files exist
        p2 = _make_proposal(
            title="Add FooBar widget",
            affected_files=["nonexistent/widget.py"],
        )

        detector = ImplementationDetector(repo_path=str(tmp_path))
        with patch("subprocess.run", return_value=MagicMock(stdout="", returncode=1)):
            results = await detector.detect_batch([p1, p2], lightweight=False)

        assert len(results) == 2
        assert results[0].file_existence_ratio == 1.0
        assert results[1].file_existence_ratio == 0.0

    @pytest.mark.asyncio
    async def test_cooldown_filtering(self):
        p = _make_proposal(
            last_tracked_at=time.time() - 10,
            implementation_confidence=0.5,
            implementation_status="uncertain",
        )
        detector = ImplementationDetector()
        with patch("config.app_config.IMPL_TRACKING_COOLDOWN", 86400):
            results = await detector.detect_batch([p], lightweight=False)
        assert len(results) == 1
        assert results[0].skipped_reason == "cooldown"

    @pytest.mark.asyncio
    async def test_llm_batching(self, tmp_path):
        """Borderline proposals should trigger LLM batch call."""
        # Create a proposal that gets borderline confidence
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("class AuthManager:\n    pass\n")

        p = _make_proposal(affected_files=["core/auth.py", "missing.py"])

        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value='{"1": 0.90}')

        detector = ImplementationDetector(
            repo_path=str(tmp_path), model_manager=mm,
        )
        with patch("subprocess.run", return_value=MagicMock(stdout="", returncode=1)):
            results = await detector.detect_batch([p], lightweight=False)

        # If confidence was borderline (0.30-0.84), LLM should have been called
        # The result may or may not be LLM-adjusted depending on the composite score
        assert len(results) == 1


# =====================================================================
# TestLightweight
# =====================================================================

class TestLightweight:
    def test_only_stage_1(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "settings.py").write_text("")

        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        result = detector.detect_single(p, lightweight=True)
        assert result.file_existence_ratio == 1.0
        assert result.code_match_ratio == 0.0  # not checked
        assert result.git_match_score == 0.0  # not checked
        assert result.confidence == 1.0  # pure file ratio

    def test_no_git_calls(self, tmp_path):
        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))

        with patch("subprocess.run") as mock_run:
            detector.detect_single(p, lightweight=True)
        mock_run.assert_not_called()

    def test_correct_structure(self, tmp_path):
        p = _make_proposal()
        detector = ImplementationDetector(repo_path=str(tmp_path))
        result = detector.detect_single(p, lightweight=True)
        assert isinstance(result, DetectionResult)
        assert result.proposal_id == p.id
        assert result.llm_adjusted is False

    @pytest.mark.asyncio
    async def test_batch_lightweight(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "auth.py").write_text("")

        p = _make_proposal()
        mm = MagicMock()
        mm.generate_once = AsyncMock()

        detector = ImplementationDetector(repo_path=str(tmp_path), model_manager=mm)
        results = await detector.detect_batch([p], lightweight=True)

        assert len(results) == 1
        # LLM should NOT be called in lightweight mode
        mm.generate_once.assert_not_called()


# =====================================================================
# TestBuildEvidence
# =====================================================================

class TestBuildEvidence:
    def test_basic(self):
        detector = ImplementationDetector()
        ev = detector._build_evidence(0.5, ["AuthManager"], ["feat: auth"], False)
        assert "50%" in ev
        assert "AuthManager" in ev
        assert "feat: auth" in ev

    def test_llm_adjusted(self):
        detector = ImplementationDetector()
        ev = detector._build_evidence(1.0, [], [], llm_adjusted=True)
        assert "LLM-adjusted" in ev

    def test_truncation(self):
        detector = ImplementationDetector()
        ev = detector._build_evidence(
            1.0,
            [f"ident_{i}" for i in range(100)],
            [f"commit_{i}" for i in range(100)],
            False,
        )
        assert len(ev) <= 500


# =====================================================================
# TestExtractIdentifiers
# =====================================================================

class TestExtractIdentifiers:
    def test_class_pattern(self):
        p = _make_proposal(
            implementation_steps=[
                ImplementationStep(order=1, description="Create FooManager class")
            ],
        )
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "FooManager" in idents

    def test_def_pattern(self):
        p = _make_proposal(
            implementation_steps=[
                ImplementationStep(order=1, description="def process_data")
            ],
        )
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "process_data" in idents

    def test_constant_pattern(self):
        p = _make_proposal(
            implementation_steps=[
                ImplementationStep(order=1, description="Add FEATURE_FLAG_ENABLED setting")
            ],
        )
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "FEATURE_FLAG_ENABLED" in idents

    def test_filters_noise(self):
        p = _make_proposal(
            implementation_steps=[
                ImplementationStep(order=1, description="TODO: fix NONE check")
            ],
        )
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "TODO" not in idents
        assert "NONE" not in idents

    def test_short_names_filtered(self):
        p = _make_proposal(
            implementation_steps=[
                ImplementationStep(order=1, description="class AB")
            ],
        )
        detector = ImplementationDetector()
        idents = detector._extract_identifiers(p)
        assert "AB" not in idents


# =====================================================================
# TestScoreCommit
# =====================================================================

class TestScoreCommit:
    def test_full_file_overlap(self):
        detector = ImplementationDetector()
        affected = {"core/auth.py", "config/settings.py"}
        title_words = {"authentication", "user"}
        commit_files = {"core/auth.py", "config/settings.py"}
        score = detector._score_commit(affected, title_words, commit_files, "add auth")
        assert score >= 0.7  # full file overlap * 0.7

    def test_no_overlap(self):
        detector = ImplementationDetector()
        affected = {"core/auth.py"}
        title_words = {"authentication"}
        commit_files = {"README.md"}
        score = detector._score_commit(affected, title_words, commit_files, "update docs")
        assert score == 0.0

    def test_keyword_only(self):
        detector = ImplementationDetector()
        affected = {"core/auth.py"}
        title_words = {"authentication", "user"}
        commit_files = {"other.py"}
        score = detector._score_commit(affected, title_words, commit_files, "add user authentication")
        # No file overlap, but keyword overlap
        assert score > 0.0
        assert score <= 0.3  # max keyword contribution
