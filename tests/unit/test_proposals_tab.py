# tests/unit/test_proposals_tab.py
"""Tests for the Proposals-tab supervision surfacing.

The Proposals tab is where the human acts on a proposal's computed supervision
metadata. These cover the pure, stateless renderer (_render_proposal_card): a
HIGH/CRITICAL or core-system proposal must be visually flagged, and its
registry overlaps (depends_on) must be shown — so a reviewer can't approve a
dangerous change on a glance. The Approve/Mark-Built acknowledge GATE itself is
the requires_human_ack policy (tested in test_proposal_risk.py)."""

import json

from gui.tabs.proposals import _render_proposal_card


def _meta(**over):
    base = {
        "proposal_id": "p1",
        "title": "Some proposal",
        "proposal_type": "feature",
        "status": "pending",
        "priority": 7,
        "risk_level": "medium",
        "touches_core_system": False,
    }
    base.update(over)
    return base


def test_critical_proposal_renders_critical_badge():
    html = _render_proposal_card(_meta(risk_level="critical"), 0, json)
    assert "CRITICAL" in html
    assert "#dc2626" in html  # critical red


def test_high_proposal_renders_high_badge():
    html = _render_proposal_card(_meta(risk_level="high"), 1, json)
    assert "HIGH" in html
    assert "#f97316" in html  # high orange


def test_core_system_touch_renders_core_badge():
    html = _render_proposal_card(_meta(touches_core_system=True), 0, json)
    assert "CORE-SYSTEM" in html


def test_plain_medium_proposal_has_no_core_badge():
    html = _render_proposal_card(_meta(), 0, json)
    assert "CORE-SYSTEM" not in html
    assert "MEDIUM" in html


def test_depends_on_overlaps_are_surfaced():
    html = _render_proposal_card(
        _meta(depends_on_json=json.dumps(["shipped_feature_a", "upstream_dep"])),
        0, json,
    )
    assert "Overlaps / depends on" in html
    assert "shipped_feature_a" in html and "upstream_dep" in html


def test_missing_supervision_fields_default_to_medium_no_crash():
    # Pre-supervision proposals lack risk_level/touches_core_system entirely.
    bare = {"proposal_id": "old", "title": "Legacy", "proposal_type": "feature",
            "status": "pending", "priority": 5}
    html = _render_proposal_card(bare, 0, json)
    assert "MEDIUM" in html        # safe default
    assert "CORE-SYSTEM" not in html
