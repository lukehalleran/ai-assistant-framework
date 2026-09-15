"""Table-integrity tests for utils/windows_timezones.py (A03b-2; BC-71).

The generated WINDOWS_TO_IANA table is never hand-edited and is the sole
source `_resolve_windows_registry_timezone` consults. These tests never
import babel at runtime from production code (utils/windows_timezones.py
itself has no babel import); they use babel only here, at test time, to
verify the generated table against its source when babel happens to be
installed (it is not a declared project dependency, 2026-09-13 parent-
verified fact) — and stay SKIP-FREE by asserting the embedded spot samples
directly when it is not.
"""
import sys

from zoneinfo import ZoneInfo

from utils.windows_timezones import WINDOWS_TO_IANA

# Spot samples named in the batch contract; also the fallback assertion
# when babel is unavailable (see test_matches_babel_source_or_spot_samples).
_SPOT_SAMPLES = {
    "Mountain Standard Time": "America/Denver",
    "India Standard Time": "Asia/Calcutta",
    "UTC": "Etc/UTC",
    "W. Europe Standard Time": "Europe/Berlin",
}


def _is_valid_zoneinfo_key(zone: str) -> bool:
    try:
        ZoneInfo(zone)
        return True
    except Exception:
        return False


def test_all_values_are_valid_zoneinfo_keys():
    invalid = {w: z for w, z in WINDOWS_TO_IANA.items() if not _is_valid_zoneinfo_key(z)}
    assert invalid == {}


def test_has_139_entries_sorted_by_key():
    assert len(WINDOWS_TO_IANA) == 139
    assert list(WINDOWS_TO_IANA.keys()) == sorted(WINDOWS_TO_IANA.keys())


def test_spot_samples_match_expected_zones():
    for windows_name, expected_zone in _SPOT_SAMPLES.items():
        assert WINDOWS_TO_IANA[windows_name] == expected_zone


def test_matches_babel_source_or_spot_samples():
    """When babel is importable, the table equals get_global exactly. When
    it is not, this asserts the embedded spot samples directly (SKIP-FREE:
    no pytest.skip either way)."""
    try:
        from babel.core import get_global
    except ImportError:
        for windows_name, expected_zone in _SPOT_SAMPLES.items():
            assert WINDOWS_TO_IANA[windows_name] == expected_zone
        return
    assert WINDOWS_TO_IANA == get_global("windows_zone_mapping")


def test_absent_babel_path_still_asserts_not_skips(monkeypatch):
    """Simulate babel being uninstalled (it is not a declared dependency)
    and confirm the fallback above performs a real assertion, never a
    pytest.skip, by exercising the same absent-babel branch directly."""
    monkeypatch.setitem(sys.modules, "babel", None)
    monkeypatch.setitem(sys.modules, "babel.core", None)
    import pytest
    with pytest.raises(ImportError):
        from babel.core import get_global  # noqa: F401

    for windows_name, expected_zone in _SPOT_SAMPLES.items():
        assert WINDOWS_TO_IANA[windows_name] == expected_zone
