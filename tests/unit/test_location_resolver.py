# tests/unit/test_location_resolver.py
"""
Unit tests for utils/location_resolver.py.

Covers the resolution chain (override → IP geolocation → profile fallback),
the place-shape filter that rejects junk lives_in values ("joke state"),
non-blocking behavior of the IP path, and formatting helpers.
"""

import json
import os
import tempfile

import pytest

import utils.location_resolver as lr
from utils.location_resolver import LocationResolver


@pytest.fixture
def profile_file():
    """Temp profile JSON with realistic lives_in history, including junk."""
    entries = [
        {
            "relation": "lives_in",
            "value": "Saint Charles, IL",
            "timestamp": "2026-01-16T00:41:28",
            "is_current": False,
        },
        {
            "relation": "lives_in",
            "value": "Springfield, IL",
            "timestamp": "2026-03-22T23:57:07",
            "is_current": False,
        },
        {
            # Sarcasm captured as a fact — the current entry is NOT a place
            "relation": "lives_in",
            "value": "joke state",
            "timestamp": "2026-05-28T21:16:19",
            "is_current": True,
        },
        {
            "relation": "lives_in",
            "value": "a bad mood",
            "timestamp": "2026-06-04T18:51:03",
            "is_current": True,
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"categories": {"identity": entries}}, f)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def resolver(profile_file, monkeypatch):
    """Resolver with IP lookup disabled so tests exercise the profile path."""
    monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
    monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
    monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")
    return LocationResolver(profile_path=profile_file)


class TestResolutionChain:
    def test_disabled_returns_none(self, resolver, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_ENABLED", False)
        assert resolver.get_location() is None

    def test_override_wins(self, resolver, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "Geneva, IL")
        assert resolver.get_location() == "Geneva, IL"

    def test_profile_fallback_skips_junk(self, resolver):
        # "joke state" and "a bad mood" are current but not place-shaped —
        # the newest place-shaped fact must win instead.
        assert resolver.get_location() == "Springfield, IL"

    def test_profile_prefers_current_place_fact(self, profile_file, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
        monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")
        with open(profile_file) as f:
            profile = json.load(f)
        profile["categories"]["identity"].append({
            "relation": "lives_in",
            "value": "Milton, IL",
            "timestamp": "2026-02-01T00:00:00",
            "is_current": True,
        })
        with open(profile_file, "w") as f:
            json.dump(profile, f)
        resolver = LocationResolver(profile_path=profile_file)
        # Older than Springfield entry but is_current — current wins.
        assert resolver.get_location() == "Milton, IL"

    def test_missing_profile_returns_none(self, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
        monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")
        resolver = LocationResolver(profile_path="/nonexistent/profile.json")
        assert resolver.get_location() is None

    def test_junk_only_profile_returns_none(self, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
        monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"categories": {"identity": [
                    {"relation": "lives_in", "value": "such a bad mood",
                     "timestamp": "2026-04-25T15:03:05", "is_current": True},
                ]}}, f)
            resolver = LocationResolver(profile_path=path)
            assert resolver.get_location() is None
        finally:
            os.remove(path)


class TestIpPath:
    def test_cached_ip_location_returned(self, resolver, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", True)
        monkeypatch.setattr(lr, "requests", object())  # non-None sentinel
        resolver._ip_location = "Saint Charles, Illinois"
        resolver._ip_fetched_at = __import__("time").time()
        assert resolver.get_location() == "Saint Charles, Illinois"

    def test_no_cache_falls_back_to_profile_without_blocking(self, resolver, monkeypatch):
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", True)
        monkeypatch.setattr(lr, "requests", object())
        started = []
        monkeypatch.setattr(resolver, "_start_background_refresh",
                            lambda: started.append(True))
        # First call: refresh kicked off in background, profile answers now.
        assert resolver.get_location() == "Springfield, IL"
        assert started == [True]

    def test_refresh_success_populates_cache(self, resolver, monkeypatch):
        monkeypatch.setattr(resolver, "_fetch_ip_location",
                            lambda: "Saint Charles, Illinois")
        resolver._refresh_ip_location()
        assert resolver._ip_location == "Saint Charles, Illinois"
        assert resolver._ip_fetched_at > 0

    def test_refresh_failure_sets_backoff(self, resolver, monkeypatch):
        monkeypatch.setattr(resolver, "_fetch_ip_location", lambda: None)
        resolver._refresh_ip_location()
        assert resolver._ip_location is None
        assert resolver._ip_failed_at > 0
        # Backoff window suppresses immediate re-refresh.
        resolver._start_background_refresh()
        assert resolver._refresh_in_flight is False


class TestFormatPlace:
    def test_city_and_region(self):
        assert LocationResolver._format_place("Saint Charles", "Illinois") == \
            "Saint Charles, Illinois"

    def test_city_only(self):
        assert LocationResolver._format_place("Saint Charles", "") == "Saint Charles"

    def test_empty(self):
        assert LocationResolver._format_place(None, None) is None


class TestModuleAccessor:
    def test_get_user_location_never_raises(self, monkeypatch):
        # Even with a broken resolver, the accessor must be safe to call
        # from the search hot path.
        monkeypatch.setattr(lr, "_resolver", None)
        monkeypatch.setattr(lr, "LOCATION_ENABLED", False)
        assert lr.get_user_location() is None
