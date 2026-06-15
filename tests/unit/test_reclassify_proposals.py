# tests/unit/test_reclassify_proposals.py
"""Tests for the proposal re-classification script's metadata extraction.

The script's value is that it reconstructs the SAME inputs the live generator
feeds the classifier — affected_files + step targets + step snippets +
description/reasoning — from a stored proposal's ChromaDB metadata, so a
proposal that predates the classifier wiring is re-scored correctly. These cover
that adapter (``_classify_from_metadata``); the classifier itself is covered in
test_proposal_risk.py."""

import importlib.util
import json
import os

from memory.code_proposal import RiskLevel

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "reclassify_proposals.py")
_spec = importlib.util.spec_from_file_location("reclassify_proposals", _PATH)
reclass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reclass)


def _md(**over):
    base = {
        "proposal_id": "x", "title": "T", "proposal_type": "feature",
        "affected_files_json": json.dumps([]), "steps_json": json.dumps([]),
        "description": "", "reasoning": "",
    }
    base.update(over)
    return base


def test_core_path_in_affected_files_trips_core():
    md = _md(affected_files_json=json.dumps(["core/prompt/context_gatherer.py"]))
    touched, core, risk = reclass._classify_from_metadata(md)
    assert core is True and risk == RiskLevel.HIGH
    assert "core/prompt/context_gatherer.py" in touched


def test_supervision_path_in_step_targets_is_critical():
    steps = [{"order": 1, "description": "edit", "file_path": "knowledge/proposal_generator.py"}]
    md = _md(steps_json=json.dumps(steps))
    _touched, core, risk = reclass._classify_from_metadata(md)
    assert core is True and risk == RiskLevel.CRITICAL


def test_import_in_step_snippet_trips_core():
    steps = [{"order": 1, "description": "new helper", "file_path": "plugins/p.py",
              "code_snippet": "from core.orchestrator import Orchestrator\n"}]
    md = _md(steps_json=json.dumps(steps))
    _touched, core, risk = reclass._classify_from_metadata(md)
    assert core is True and risk == RiskLevel.HIGH


def test_plain_feature_stays_medium_no_core():
    md = _md(affected_files_json=json.dumps(["knowledge/new_widget.py"]))
    _touched, core, risk = reclass._classify_from_metadata(md)
    assert core is False and risk == RiskLevel.MEDIUM


def test_malformed_json_metadata_degrades_gracefully():
    # A corrupt steps_json must not crash the adapter — it falls back to empty.
    md = _md(affected_files_json="{not json", steps_json="also bad")
    touched, core, risk = reclass._classify_from_metadata(md)
    assert touched == [] and core is False and risk == RiskLevel.MEDIUM
