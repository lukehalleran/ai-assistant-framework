"""Tests for the single profile-path authority (F02 / BC-10 / BC-58).

Drives the deployed `utils.bootstrap.get_user_profile_path()` and the five
consumers that must resolve through it: UserIdentityResolver, LocationResolver,
InstitutionResolver, TimezoneResolver, and UserProfile. Every scenario uses
tmp_path; the clone's real data/ directory is never created, written, or read
for content (a plain os.path.exists check in dev mode mirrors production
behavior and is unavoidable, but this suite never opens that file). All names
and places below are synthetic; this repo is public.
"""
import json
import os

from memory.user_profile import UserProfile
from utils.bootstrap import get_user_data_dir, get_user_profile_path
from utils.institution_resolver import InstitutionResolver
from utils.location_resolver import LocationResolver
from utils.timezone_resolver import TimezoneResolver
from utils.user_identity import UserIdentityResolver


def _fact(relation, value):
    return {
        "relation": relation,
        "value": value,
        "is_current": True,
        "confidence": 0.9,
        "timestamp": "2026-01-01T00:00:00",
    }


def _synthetic_profile(name, place, school, tz):
    return {
        "user_id": "synthetic",
        "quick_profile": {"name": name, "school": school, "timezone": tz},
        "categories": {
            "identity": [_fact("name", name), _fact("lives_in", place), _fact("timezone", tz)],
            "education": [_fact("school", school)],
        },
        "identity": {"name": name, "pronouns": ""},
        "raw_log": [],
    }


def _write_profile(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _resolve_all(profile_path=None):
    """Instantiate all five consumers through their real read paths and
    return the resolved OUTCOMES (not just the stored profile_path)."""
    kwargs = {} if profile_path is None else {"profile_path": profile_path}
    return {
        "name": UserIdentityResolver(**kwargs).get_display_name(),
        "location": LocationResolver(**kwargs).get_location(),
        "institution": InstitutionResolver(**kwargs).get_institution(),
        "timezone": TimezoneResolver(**kwargs).get_timezone(),
        "profile_name": UserProfile(**kwargs).identity.name,
    }


NOVA = _synthetic_profile("Nova Ashworth", "Rivermont, Cascadia", "Rivermont Institute", "America/Denver")
LEGACY_PERSON = _synthetic_profile("Wren Castellan", "Halden, Meridia", "Halden Polytechnic", "America/New_York")
AUTHORITY_PERSON = _synthetic_profile("Iris Delacroix", "Portmere, Solvane", "Portmere College", "America/Los_Angeles")


class TestEnvPathReadByAllFiveConsumers:
    """(a) USER_PROFILE_PATH set to a tmp file -> every consumer reads it."""

    def test_all_five_consumers_resolve_outcomes_from_env_path(self, tmp_path, monkeypatch):
        profile_path = tmp_path / "synthetic_profile.json"
        _write_profile(str(profile_path), NOVA)
        monkeypatch.setenv("USER_PROFILE_PATH", str(profile_path))

        assert get_user_profile_path() == str(profile_path)

        outcomes = _resolve_all()
        assert outcomes == {
            "name": "Nova Ashworth",
            "location": "Rivermont, Cascadia",
            "institution": "Rivermont Institute",
            "timezone": "America/Denver",
            "profile_name": "Nova Ashworth",
        }


class TestCwdIndependence:
    """(b) Env unset + chdir(tmp_path): resolution does not depend on cwd."""

    def test_resolved_path_ignores_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("USER_PROFILE_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        expected = os.path.join(get_user_data_dir(), "user_profile.json")
        assert get_user_profile_path() == expected


class TestFrozenSimulation:
    """(c) Frozen Windows simulation -> <tmp>/Daemon/user_profile.json.

    Windows semantics are simulated on Linux here; platform acceptance on a
    real Windows machine remains pending (see the batch evidence packet).
    """

    def test_frozen_windows_path(self, tmp_path, monkeypatch):
        import utils.bootstrap as bootstrap

        monkeypatch.setattr(bootstrap, "IS_FROZEN", True)
        monkeypatch.setattr(bootstrap, "IS_WINDOWS", True)
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.delenv("USER_PROFILE_PATH", raising=False)

        expected = os.path.join(str(tmp_path), "Daemon", "user_profile.json")
        assert get_user_profile_path() == expected


class TestCompatibilityRead:
    """(d) Authority absent + legacy present; both/neither present controls.

    Uses the frozen simulation so the authoritative and legacy roots both
    live inside tmp_path -- no real filesystem location outside the sandbox
    is ever touched.
    """

    def _frozen_paths(self, tmp_path, monkeypatch):
        import utils.bootstrap as bootstrap

        monkeypatch.setattr(bootstrap, "IS_FROZEN", True)
        monkeypatch.setattr(bootstrap, "IS_WINDOWS", True)
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.delenv("USER_PROFILE_PATH", raising=False)
        authority = os.path.join(str(tmp_path), "Daemon", "user_profile.json")
        legacy = os.path.join("data", "user_profile.json")  # cwd-relative, matches old default
        return authority, legacy

    def test_legacy_read_when_authority_absent(self, tmp_path, monkeypatch, caplog):
        authority, legacy = self._frozen_paths(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        _write_profile(legacy, LEGACY_PERSON)
        before_bytes = open(legacy, "rb").read()
        before_mtime = os.path.getmtime(legacy)

        assert not os.path.exists(authority)
        with caplog.at_level("WARNING"):
            resolved = get_user_profile_path()
        assert resolved == legacy
        assert any(authority in r.getMessage() for r in caplog.records)

        # The four read-only resolvers read the legacy file through their
        # real read paths -- assert OUTCOMES, not just profile_path.
        assert UserIdentityResolver().get_display_name() == "Wren Castellan"
        assert LocationResolver().get_location() == "Halden, Meridia"
        assert InstitutionResolver().get_institution() == "Halden Polytechnic"
        assert TimezoneResolver().get_timezone() == "America/New_York"

        # No copy/move/write: get_user_profile_path() itself, and the four
        # read-only resolvers above, never touched the legacy file.
        assert open(legacy, "rb").read() == before_bytes
        assert os.path.getmtime(legacy) == before_mtime
        assert not os.path.exists(authority)

        # UserProfile is the fifth consumer; its constructor legitimately
        # runs a pre-existing schema migration/save unrelated to this batch,
        # so only its resolved OUTCOME is asserted here, not file bytes.
        assert UserProfile().identity.name == "Wren Castellan"

    def test_authority_wins_when_both_present(self, tmp_path, monkeypatch):
        authority, legacy = self._frozen_paths(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        _write_profile(legacy, LEGACY_PERSON)
        _write_profile(authority, AUTHORITY_PERSON)

        assert get_user_profile_path() == authority
        outcomes = _resolve_all()
        assert outcomes["name"] == "Iris Delacroix"

    def test_authority_path_when_both_absent(self, tmp_path, monkeypatch):
        authority, legacy = self._frozen_paths(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        assert not os.path.exists(authority)
        assert not os.path.exists(legacy)
        assert get_user_profile_path() == authority


class TestExplicitOverridesWinOverEnv:
    """(e) Controls: explicit profile_path= and monkeypatched DEFAULT_PATH."""

    def test_explicit_profile_path_beats_env(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env_profile.json"
        explicit_path = tmp_path / "explicit_profile.json"
        _write_profile(str(env_path), LEGACY_PERSON)
        _write_profile(str(explicit_path), AUTHORITY_PERSON)
        monkeypatch.setenv("USER_PROFILE_PATH", str(env_path))

        resolver = UserIdentityResolver(profile_path=str(explicit_path))
        assert resolver.get_display_name() == "Iris Delacroix"

        profile = UserProfile(profile_path=str(explicit_path))
        assert profile.identity.name == "Iris Delacroix"

    def test_monkeypatched_default_path_beats_env(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env_profile.json"
        patched_path = tmp_path / "patched_profile.json"
        _write_profile(str(env_path), LEGACY_PERSON)
        _write_profile(str(patched_path), AUTHORITY_PERSON)
        monkeypatch.setenv("USER_PROFILE_PATH", str(env_path))
        monkeypatch.setattr(UserProfile, "DEFAULT_PATH", str(patched_path))

        profile = UserProfile()
        assert profile.profile_path == str(patched_path)
        assert profile.identity.name == "Iris Delacroix"


class TestBackupTargetsFollowTheAuthority:
    """Parent review R1 (BC-58): utils.backup_manager.backup_targets() read
    the old string UserProfile.DEFAULT_PATH. With DEFAULT_PATH = None the
    profile must stay a backup target (existing_only=True) and a restore
    destination (existing_only=False) through the authority."""

    def test_backup_targets_include_resolved_profile_path(self, tmp_path, monkeypatch):
        from utils.backup_manager import backup_targets

        profile_path = tmp_path / "synthetic_profile.json"
        _write_profile(str(profile_path), NOVA)
        monkeypatch.setenv("USER_PROFILE_PATH", str(profile_path))

        assert str(profile_path) in backup_targets(existing_only=True)
        assert str(profile_path) in backup_targets(existing_only=False)
