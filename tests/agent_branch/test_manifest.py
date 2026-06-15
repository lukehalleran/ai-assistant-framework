# tests/agent_branch/test_manifest.py
"""Unit tests: manifest issuance, immutability, and tamper detection."""

import json

import pytest
from pydantic import ValidationError

from agent_branch.manifest import (
    DEFAULT_FORBIDDEN_PATHS,
    BranchManifest,
    ManifestTamperError,
)


def _issue(**over):
    base = dict(branch_id="b1", objective="do a thing", allowed_paths=["sandbox/**"])
    base.update(over)
    return BranchManifest.issue(**base)


def test_issue_stamps_a_verifying_hash():
    m = _issue()
    assert m.manifest_hash
    assert m.verify() is True
    m.assert_intact()  # does not raise


def test_hash_is_deterministic_for_same_fields():
    a = _issue(token_budget=1000)
    b = _issue(token_budget=1000)
    assert a.manifest_hash == b.manifest_hash


def test_hash_changes_when_a_field_changes():
    a = _issue(max_diff_lines=10)
    b = _issue(max_diff_lines=11)
    assert a.manifest_hash != b.manifest_hash


def test_manifest_is_frozen_immutable():
    m = _issue()
    with pytest.raises(ValidationError):
        m.objective = "something else"


def test_tampered_supervisor_copy_detected_on_reload(tmp_path):
    """Red-team #6: a doctored supervisor-side copy fails the hash check."""
    m = _issue()
    doctored = m.model_dump()
    doctored["objective"] = "exfiltrate everything"  # change a field, keep old hash
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(ManifestTamperError):
        BranchManifest.from_json_file(p)


def test_intact_roundtrip_passes(tmp_path):
    m = _issue()
    p = m.to_json_file(tmp_path / "m.json")
    reloaded = BranchManifest.from_json_file(p)
    assert reloaded.verify()
    assert reloaded.objective == m.objective


def test_assert_intact_raises_on_blank_hash():
    raw = BranchManifest(branch_id="x", objective="y")  # never issued → no hash
    assert raw.verify() is False
    with pytest.raises(ManifestTamperError):
        raw.assert_intact()


def test_default_forbidden_paths_cover_safety_surface():
    m = _issue()
    for required in ("tests/", "data/", ".env", "scripts/safe_git.sh", "scripts/bin/"):
        assert required in m.forbidden_paths
    assert "tests/" in DEFAULT_FORBIDDEN_PATHS
