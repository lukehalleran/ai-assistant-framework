"""
Strict JSON-field contract tests for knowledge.proposal_generator (S03, BC-21
sibling of F03/A04's core.grounding_check._parse_verdict, S01's
utils.web_search_trigger.LLMSearchTriggerResponse.parse, and S02's
core.response_planner.ResponsePlanner._parse_review).

Covers:
- GoalDirectedGenerator._parse_proposal(): rejections for every field this
  batch owns (requires_tests, priority, tags, affected_files,
  estimated_complexity) when PRESENT with the wrong type; controls for
  valid/minimal/absent-field/clamped/truncated/whitelist-normalized
  payloads. Unlike S01/S02, `_parse_proposal` already takes a `dict` (JSON
  decoding happens one level up, in `_parse_response`), so there is no
  top-level-object case here.
- The supervision seam: `classify_proposal` (memory/proposal_risk.py) must
  never be called with a rejected payload's touched_paths — today a bad
  `affected_files` string is silently exploded into single-character
  "paths" (`list("abcdef") == ['a','b',...]`) and fed to the risk
  classifier before the proposal is eventually dropped by an unrelated
  Pydantic error. Supervision fields (risk_level, touches_core_system) must
  still come ONLY from the real, unmocked `classify_proposal` for a valid
  payload (never defaulted, never changed by this batch).
- A round trip through the generator's public parse entry point
  (`generate_proposals`, `_parse_response` -> `_parse_proposal`) with a
  fake model_manager (no network): a batch with one wrong-typed proposal
  and one valid proposal keeps only the valid one.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import knowledge.proposal_generator as pg
from knowledge.proposal_generator import GoalDirectedGenerator
from memory.code_proposal import RiskLevel
from memory.proposal_risk import classify_proposal as real_classify_proposal


def _valid_proposal(**overrides) -> dict:
    """A fully-typed, valid raw proposal dict, with overrides."""
    data = {
        "title": "Add response caching",
        "proposal_type": "feature",
        "priority": 8,
        "reasoning": "Speed improvement",
        "description": "Add Redis caching for hot paths",
        "implementation_steps": [
            {"order": 1, "description": "Add redis client",
             "file_path": "core/cache.py", "action": "create"},
        ],
        "affected_files": ["core/cache.py"],
        "tags": ["performance"],
        "estimated_complexity": "medium",
        "requires_tests": True,
    }
    data.update(overrides)
    return data


@pytest.fixture
def generator():
    return GoalDirectedGenerator(model_manager=None, repo_path=".")


# ---------------------------------------------------------------------------
# Deployed function: GoalDirectedGenerator._parse_proposal — rejections
# ---------------------------------------------------------------------------


class TestParseProposalStrictContractRejections:

    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, None, "yes"])
    def test_requires_tests_wrong_type_rejected(self, generator, bad):
        """Failing-before proof: today `bool(data.get("requires_tests", True))`
        coerces EVERY value here to a truthy/falsy bool with no error at
        all — `bool("false")` is `True`, `bool(None)` is `False` — so a
        malformed proposal is indistinguishable from a genuine, deliberate
        choice. Must now reject."""
        assert generator._parse_proposal(_valid_proposal(requires_tests=bad)) is None

    @pytest.mark.parametrize("bad", [True, False, 7.9, "8", None, [10]])
    def test_priority_wrong_type_rejected(self, generator, bad):
        """Failing-before proof: `int(data.get("priority", 5))` silently
        turns a bool into 1/0 and truncates a float; `int("8")` silently
        accepts a JSON string. Must now reject."""
        assert generator._parse_proposal(_valid_proposal(priority=bad)) is None

    @pytest.mark.parametrize("bad", ["abc", 123, None, {"a": 1}, ["ok", 5], [None]])
    def test_tags_wrong_type_or_element_rejected(self, generator, bad):
        assert generator._parse_proposal(_valid_proposal(tags=bad)) is None

    @pytest.mark.parametrize("bad", ["abcdef", 123, None, {"a": 1}, ["ok", 5], [None]])
    def test_affected_files_wrong_type_or_element_rejected(self, generator, bad):
        """Failing-before proof (see TestClassifyProposalNeverSeesRejectedPayload):
        a string `affected_files` is not just eventually rejected — it is
        first exploded into single characters (`list("abcdef")`) and fed to
        the risk classifier as if they were real paths."""
        assert generator._parse_proposal(_valid_proposal(affected_files=bad)) is None

    @pytest.mark.parametrize("bad", [5, True, None, ["low"]])
    def test_estimated_complexity_wrong_type_rejected(self, generator, bad):
        assert generator._parse_proposal(_valid_proposal(estimated_complexity=bad)) is None

    def test_rejection_warning_names_only_field_and_type(self, generator, caplog):
        """Privacy: the one warning names the field/type, never the model's
        own title/description/reasoning text (which may quote the user's
        session content)."""
        secret = "the user's private medical detail"
        data = _valid_proposal(
            priority=True, title=secret, description=secret, reasoning=secret,
        )
        with caplog.at_level(logging.WARNING, logger="proposal_generator"):
            result = generator._parse_proposal(data)
        assert result is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert secret not in warnings[0]
        assert "priority" in warnings[0]


# ---------------------------------------------------------------------------
# Deployed function: GoalDirectedGenerator._parse_proposal — controls
# ---------------------------------------------------------------------------


class TestParseProposalStrictContractControls:

    def test_valid_full_payload_accepted(self, generator):
        proposal = generator._parse_proposal(_valid_proposal())
        assert proposal is not None
        assert proposal.title == "Add response caching"
        assert proposal.priority == 8
        assert proposal.tags == ["performance"]
        assert proposal.affected_files == ["core/cache.py"]
        assert proposal.estimated_complexity == "medium"
        assert proposal.requires_tests is True

    def test_minimal_valid_payload_title_only(self, generator):
        proposal = generator._parse_proposal({"title": "Minimal proposal"})
        assert proposal is not None
        assert proposal.title == "Minimal proposal"

    def test_absent_optional_fields_use_documented_defaults(self, generator):
        proposal = generator._parse_proposal({"title": "A valid title"})
        assert proposal is not None
        assert proposal.requires_tests is True
        assert proposal.priority == 5
        assert proposal.tags == []
        assert proposal.affected_files == []
        assert proposal.estimated_complexity == "medium"

    def test_priority_clamped_above_and_below_range(self, generator):
        above = generator._parse_proposal(_valid_proposal(priority=99))
        below = generator._parse_proposal(_valid_proposal(priority=-5))
        assert above.priority == 10
        assert below.priority == 1

    def test_tags_truncated_to_ten(self, generator):
        proposal = generator._parse_proposal(
            _valid_proposal(tags=[f"tag{i}" for i in range(20)])
        )
        assert proposal is not None
        assert len(proposal.tags) == 10

    def test_unknown_estimated_complexity_normalizes_to_default(self, generator):
        """An unknown (but correctly-typed) whitelist string keeps today's
        normalization to the default rather than being rejected outright —
        same precedent as S01's search_depth/document_source."""
        proposal = generator._parse_proposal(_valid_proposal(estimated_complexity="extreme"))
        assert proposal is not None
        assert proposal.estimated_complexity == "medium"

    def test_requires_tests_explicit_false_accepted(self, generator):
        proposal = generator._parse_proposal(_valid_proposal(requires_tests=False))
        assert proposal is not None
        assert proposal.requires_tests is False

    def test_affected_files_valid_list_of_str_accepted(self, generator):
        proposal = generator._parse_proposal(
            _valid_proposal(affected_files=["a/b.py", "c/d.py"])
        )
        assert proposal is not None
        assert proposal.affected_files == ["a/b.py", "c/d.py"]


# ---------------------------------------------------------------------------
# Supervision seam: classify_proposal must never see a rejected payload
# ---------------------------------------------------------------------------


class TestClassifyProposalNeverSeesRejectedPayload:
    """`classify_proposal` (memory/proposal_risk.py) computes the
    supervision fields (risk_level, touches_core_system) and must never be
    fed a rejected payload's data — not even indirectly through
    `touched_paths`. This batch does not touch classify_proposal itself
    (out of scope); it closes the seam feeding it."""

    @pytest.mark.parametrize("field,bad", [
        ("requires_tests", "false"),
        ("priority", True),
        ("tags", "abc"),
        ("affected_files", "abcdef"),
        ("estimated_complexity", 5),
    ])
    def test_rejected_payload_never_reaches_classify_proposal(self, generator, field, bad):
        with patch.object(pg, "classify_proposal") as mock_classify:
            result = generator._parse_proposal(_valid_proposal(**{field: bad}))
        assert result is None
        mock_classify.assert_not_called()

    def test_valid_payload_reaches_classify_proposal_with_clean_touched_paths(self, generator):
        """Paired control: a valid payload still reaches the real
        classify_proposal, with touched_paths built exactly from
        affected_files + step file_paths — no character fragments."""
        captured = {}

        def spy(*args, **kwargs):
            captured["touched_paths"] = args[0] if args else kwargs.get("affected_files")
            return real_classify_proposal(*args, **kwargs)

        with patch.object(pg, "classify_proposal", side_effect=spy) as mock_classify:
            proposal = generator._parse_proposal(_valid_proposal())

        assert proposal is not None
        mock_classify.assert_called_once()
        assert captured["touched_paths"] == ["core/cache.py", "core/cache.py"]


# ---------------------------------------------------------------------------
# Round trip through the public parse entry point (fake LLM text, no network)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.generate_once = AsyncMock()
    return mm


class TestRoundTripPublicParseEntryPoint:

    @pytest.mark.asyncio
    async def test_generate_proposals_drops_wrong_typed_proposal_keeps_valid_ones(
        self, mock_model_manager
    ):
        """Round trip: _parse_response -> _parse_proposal, driven by the
        deployed generate_proposals() with a fake model_manager. One
        proposal has a bool `priority` (today silently accepted as 1); the
        other is fully valid. Only the valid one must survive."""
        response = json.dumps([
            _valid_proposal(title="Bad priority proposal", priority=True),
            _valid_proposal(title="Good proposal", priority=6),
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        proposals = await gen.generate_proposals(extra_context="Test context")

        assert len(proposals) == 1
        assert proposals[0].title == "Good proposal"
        assert proposals[0].priority == 6

    @pytest.mark.asyncio
    async def test_paired_control_all_valid_proposals_pass_through(self, mock_model_manager):
        response = json.dumps([
            _valid_proposal(title="First proposal", priority=3),
            _valid_proposal(title="Second proposal", priority=9),
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        proposals = await gen.generate_proposals(extra_context="Test context")

        assert len(proposals) == 2
        assert {p.title for p in proposals} == {"First proposal", "Second proposal"}

    @pytest.mark.asyncio
    async def test_supervision_fields_only_from_classify_proposal(self, mock_model_manager):
        """Do-not-change-supervision-classification proof: a valid proposal
        touching a CRITICAL supervision-layer file gets risk_level/
        touches_core_system exactly as the real, unmocked classify_proposal
        computes for the same inputs — never defaulted, never overridden by
        this batch's contract."""
        response = json.dumps([
            _valid_proposal(
                title="Touch the risk classifier itself",
                affected_files=["memory/proposal_risk.py"],
            )
        ])
        mock_model_manager.generate_once.return_value = response

        gen = GoalDirectedGenerator(model_manager=mock_model_manager, repo_path=".")
        proposals = await gen.generate_proposals(extra_context="Test context")

        assert len(proposals) == 1
        # A CRITICAL-path affected_files match alone forces touches_core=True,
        # risk=CRITICAL regardless of code_texts/title/description, so this
        # single-argument call is a faithful, non-brittle "what would the
        # real classifier say" oracle.
        expected_core, expected_risk = real_classify_proposal(["memory/proposal_risk.py"])
        assert expected_core is True and expected_risk == RiskLevel.CRITICAL
        assert proposals[0].touches_core_system == expected_core
        assert proposals[0].risk_level == expected_risk
