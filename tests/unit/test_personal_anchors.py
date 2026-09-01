"""Tests for personal anchors (school, employer, orgs) in institution_resolver."""

import json
import os
import pytest
from pathlib import Path
from utils.institution_resolver import (
    InstitutionResolver,
    get_user_anchors,
    _INSTITUTION_VALUE_RE,
)
from utils.web_search_trigger import terms_are_private_sphere_generic


@pytest.fixture
def temp_profile(tmp_path):
    """Create a temporary profile file for testing."""
    profile_path = tmp_path / "user_profile.json"

    def make_profile(school=None, employer=None, org=None):
        """Build a profile with optional school/employer/org facts."""
        profile = {
            "user_id": "test",
            "categories": {},
        }
        categories = []

        if school:
            categories.append({
                "relation": "school",
                "value": school,
                "category": "education",
                "confidence": 1.0,
                "source_excerpt": "",
                "timestamp": "2026-01-01T00:00:00",
                "fact_id": "school-fact",
                "is_current": True,
            })

        if employer:
            categories.append({
                "relation": "works_at",
                "value": employer,
                "category": "career",
                "confidence": 1.0,
                "source_excerpt": "",
                "timestamp": "2026-01-01T00:00:00",
                "fact_id": "employer-fact",
                "is_current": True,
            })

        if org:
            categories.append({
                "relation": "member_of",
                "value": org,
                "category": "relationships",
                "confidence": 1.0,
                "source_excerpt": "",
                "timestamp": "2026-01-01T00:00:00",
                "fact_id": "org-fact",
                "is_current": True,
            })

        if categories:
            # Group by category
            profile["categories"]["education"] = [c for c in categories if c.get("category") == "education"]
            profile["categories"]["career"] = [c for c in categories if c.get("category") == "career"]
            profile["categories"]["relationships"] = [c for c in categories if c.get("category") == "relationships"]

        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        return profile_path

    return make_profile, profile_path


def test_personal_anchors_school_employer_org(temp_profile):
    """Test that get_user_anchors returns school, employer, org in order."""
    make_profile, profile_path = temp_profile
    make_profile(
        school="Georgia Tech",
        employer="Acme Corporation",
        org="Local Tech Meetup"
    )

    resolver = InstitutionResolver(str(profile_path))
    anchors = resolver.get_anchors()

    assert anchors == [
        "Georgia Tech",
        "Acme Corporation",
        "Local Tech Meetup",
    ]


def test_personal_anchors_junk_employer_excluded(temp_profile):
    """Test that junk-shaped employer values are excluded."""
    make_profile, profile_path = temp_profile
    profile_path.write_text(json.dumps({
        "user_id": "test",
        "categories": {
            "career": [
                {
                    "relation": "employer",
                    "value": "in third best company in the nation",
                    "category": "career",
                    "confidence": 0.8,
                    "source_excerpt": "",
                    "timestamp": "2026-01-01T00:00:00",
                    "fact_id": "bad-employer",
                    "is_current": True,
                }
            ]
        }
    }), encoding="utf-8")

    resolver = InstitutionResolver(str(profile_path))
    anchors = resolver.get_anchors()

    # Junk value should not match the regex and should be excluded
    assert anchors == []


def test_personal_anchors_no_employment(temp_profile):
    """Test that school-only profile returns only school."""
    make_profile, profile_path = temp_profile
    make_profile(school="Georgia Tech")

    resolver = InstitutionResolver(str(profile_path))
    anchors = resolver.get_anchors()

    assert anchors == ["Georgia Tech"]


def test_personal_anchors_empty_profile(temp_profile):
    """Test that empty profile returns empty list."""
    make_profile, profile_path = temp_profile
    make_profile()

    resolver = InstitutionResolver(str(profile_path))
    anchors = resolver.get_anchors()

    assert anchors == []


def test_private_sphere_guard_with_anchors():
    """Test that private-sphere guard works with multiple anchors."""
    # Single anchor
    assert terms_are_private_sphere_generic(
        ["Acme standup meeting time"],
        ["Georgia Tech"]
    ) is False

    # Multiple anchors — match with second
    assert terms_are_private_sphere_generic(
        ["Acme standup meeting time"],
        ["Georgia Tech", "Acme"]
    ) is True

    # No anchors
    assert terms_are_private_sphere_generic(
        ["Acme standup meeting time"],
        []
    ) is False


def test_institution_value_regex():
    """Test that _INSTITUTION_VALUE_RE correctly validates institution shapes."""
    # Valid forms
    assert _INSTITUTION_VALUE_RE.match("Georgia Tech")
    assert _INSTITUTION_VALUE_RE.match("University of Chicago")
    assert _INSTITUTION_VALUE_RE.match("MIT")
    assert _INSTITUTION_VALUE_RE.match("Johns Hopkins")
    assert _INSTITUTION_VALUE_RE.match("Acme Corporation")
    assert _INSTITUTION_VALUE_RE.match("Local Tech Meetup")

    # Invalid forms (junk)
    assert not _INSTITUTION_VALUE_RE.match("in third best company in the nation")
    assert not _INSTITUTION_VALUE_RE.match("get into school stuff")
    assert not _INSTITUTION_VALUE_RE.match("for a bit")
    assert not _INSTITUTION_VALUE_RE.match("yesterday")


def test_get_user_anchors_module_level():
    """Test that module-level get_user_anchors() function works."""
    # This tests the global singleton, so we can't easily mock the profile path.
    # Instead, just verify the function exists and returns a list.
    result = get_user_anchors()
    assert isinstance(result, list)
    # It may be empty if the real profile doesn't have those relations
    for anchor in result:
        assert isinstance(anchor, str)
