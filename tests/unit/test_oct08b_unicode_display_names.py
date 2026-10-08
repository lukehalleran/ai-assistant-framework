"""Display-name validation is structural, not ASCII-only (2026-10-08).

class: BC-61.  Driven through the deployed resolver
(UserIdentityResolver.get_display_name via the env-override path and the
profile path) — never the helper alone.
"""
import json
import unicodedata
from unittest.mock import patch

import pytest

from utils.user_identity import UserIdentityResolver

ACCEPT = [
    "José",
    "Zoë",  # NFC
    unicodedata.normalize("NFD", "Zoë"),
    "Łukasz",
    "Élodie",
    "王芳",
    "김민준",
    "O'Brien",
    "Mary-Jane",
    "Anna Maria Lopez",
]
REJECT = [
    "josé",
    "123 junk",
    "One Two Three Four",
    "Sam‮evil",
    "Sam\x07",
    "Sa‍m",
    "",
]


@pytest.fixture
def empty_profile(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"quick_profile": {"name": "Fallback"}, "categories": {}}))
    return str(p)


@pytest.mark.parametrize("name", ACCEPT)
def test_env_override_accepts_unicode_names(name, empty_profile):
    with patch("utils.user_identity.USER_NAME_OVERRIDE", name):
        assert UserIdentityResolver(empty_profile).get_display_name() == name


@pytest.mark.parametrize("name", REJECT)
def test_env_override_rejects_implausible_names(name, empty_profile):
    with patch("utils.user_identity.USER_NAME_OVERRIDE", name):
        assert UserIdentityResolver(empty_profile).get_display_name() == "Fallback"


@pytest.mark.parametrize("name", ACCEPT)
def test_profile_fact_accepts_unicode_names(name, tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"categories": {"identity": [
        {"relation": "name", "value": name, "is_current": True, "confidence": 0.9}
    ]}}))
    with patch("utils.user_identity.USER_NAME_OVERRIDE", ""):
        assert UserIdentityResolver(str(p)).get_display_name() == name


@pytest.mark.parametrize("name", REJECT)
def test_profile_fact_rejects_implausible_names(name, tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"categories": {"identity": [
        {"relation": "name", "value": name, "is_current": True, "confidence": 0.9}
    ]}}))
    with patch("utils.user_identity.USER_NAME_OVERRIDE", ""):
        assert UserIdentityResolver(str(p)).get_display_name() == "the user"


def test_quick_profile_name_accepts_unicode(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"quick_profile": {"name": "Łukasz"}, "categories": {}}))
    with patch("utils.user_identity.USER_NAME_OVERRIDE", ""):
        assert UserIdentityResolver(str(p)).get_display_name() == "Łukasz"
