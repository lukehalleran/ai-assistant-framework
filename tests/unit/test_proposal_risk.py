# tests/unit/test_proposal_risk.py
"""Tests for proposal supervision classification + live-registry conflict wiring.

Covers the three gaps the audit surfaced:
1. A proposal touching the supervision/safety layer is forced to CRITICAL.
2. An INDIRECT edit (a new/clean-path file that imports a core module) still
   trips touches_core_system (the refactor-and-move / re-export gap).
3. Conflict annotation reflects whatever is in the LIVE feature registry — an
   entry added after an agent branched is seen, because the read is live.
"""

import textwrap

import pytest

from memory.code_proposal import ProposalType, RiskLevel
from memory.proposal_risk import classify_proposal


# -- 1. supervision / safety layer is unconditionally CRITICAL ---------------

@pytest.mark.parametrize("path", [
    "memory/code_proposal.py",
    "memory/proposal_store.py",
    "knowledge/proposal_generator.py",
    "config/feature_registry.py",
    "config/feature_registry.yaml",
    "agent_branch/eval_static.py",          # under agent_branch/ dir prefix
    "utils/python_fs_guard.py",
    "utils/shell_cmd_guard.py",
    "core/actions/email.py",                # under core/actions/ dir prefix
    "scripts/safe_git.sh",
])
def test_supervision_and_safety_touch_forced_critical(path):
    touches_core, risk = classify_proposal([path])
    assert touches_core is True
    assert risk == RiskLevel.CRITICAL, path


def test_core_spine_is_high_not_critical():
    touches_core, risk = classify_proposal(["core/orchestrator.py"])
    assert touches_core is True
    assert risk == RiskLevel.HIGH


def test_prompt_dir_prefix_matches_any_file_under_it():
    # the old exact-string list missed everything but the one enumerated file
    touches_core, risk = classify_proposal(["core/prompt/gatherer_web.py"])
    assert touches_core is True and risk == RiskLevel.HIGH


def test_plain_feature_is_not_core():
    touches_core, risk = classify_proposal(["knowledge/new_widget.py"])
    assert touches_core is False
    assert risk == RiskLevel.MEDIUM


def test_docs_only_is_low():
    touches_core, risk = classify_proposal(
        ["docs/NEW.md"], proposal_type=ProposalType.DOCS)
    assert touches_core is False and risk == RiskLevel.LOW


# -- 2. indirect edits trip touches_core_system (import-based detection) ------

def test_new_file_importing_core_module_trips_touches_core():
    # Clean path, but the code imports a core orchestration module.
    code = "from core.orchestrator import Orchestrator\n\ndef helper(): ...\n"
    touches_core, risk = classify_proposal(
        ["plugins/brand_new_helper.py"], code_texts=[code])
    assert touches_core is True
    assert risk == RiskLevel.HIGH


def test_new_file_importing_safety_module_is_critical():
    code = "import utils.python_fs_guard as g\n"
    touches_core, risk = classify_proposal(
        ["plugins/sneaky.py"], code_texts=[code])
    assert touches_core is True
    assert risk == RiskLevel.CRITICAL  # safety layer, even via import


def test_unrelated_imports_do_not_trip():
    code = "import os\nfrom typing import List\nimport knowledge.web_search_manager\n"
    touches_core, risk = classify_proposal(
        ["plugins/clean.py"], code_texts=[code])
    assert touches_core is False
    assert risk == RiskLevel.MEDIUM


# -- 3. conflict annotation reflects the LIVE registry -----------------------

def test_check_conflicts_reflects_registry_added_after_branch(tmp_path, monkeypatch):
    """A registry entry added to the file the live process reads is reflected by
    check_conflicts immediately — the read is live, not a frozen branch-time copy."""
    import config.feature_registry as fr

    reg = tmp_path / "feature_registry.yaml"
    reg.write_text(textwrap.dedent("""
        features:
          - proposal_id: shipped_after_branch
            title: A feature merged to main after the agent branched
            implemented_files:
              - memory/target_module.py
    """), encoding="utf-8")
    monkeypatch.setattr(fr, "_registry_path", lambda: reg)
    fr.load_registry(force=True)
    try:
        hits = fr.check_conflicts(["memory/target_module.py"])
        assert [h.proposal_id for h in hits] == ["shipped_after_branch"]
        assert fr.check_conflicts(["memory/unrelated.py"]) == []
    finally:
        fr.load_registry(force=True)  # restore the real registry cache


def test_generator_annotates_depends_on_from_conflicts(monkeypatch):
    """The generator populates depends_on from live-registry conflicts so the
    reviewer sees what a proposal collides with."""
    from knowledge.proposal_generator import GoalDirectedGenerator
    import config.feature_registry as fr

    class _Feat:
        proposal_id = "existing_feature"

    monkeypatch.setattr(fr, "check_conflicts", lambda files: [_Feat()] if files else [])
    monkeypatch.setattr(fr, "get_dependencies", lambda pid: ["upstream_dep"])

    proposal = GoalDirectedGenerator()._parse_proposal({
        "title": "Touch an existing feature's files",
        "proposal_type": "refactor",
        "affected_files": ["memory/truth_scorer.py"],
    })
    assert proposal is not None
    GoalDirectedGenerator._annotate_conflicts([proposal])
    assert "existing_feature" in proposal.depends_on
    assert "upstream_dep" in proposal.depends_on
