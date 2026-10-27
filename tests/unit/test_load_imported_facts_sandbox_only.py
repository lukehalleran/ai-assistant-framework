"""Imported ChatGPT facts never reach the live profile.

Incident 2026-10-15 (class: BC-46): scripts/load_imported_facts.py called
``add_facts_batch`` on the DEFAULT (live) profile -- facts authored by another
system, bypassing the fact_source user-span join.  The script is sandbox-only
now (like generate_test_facts.py): a real load needs --profile and refuses the
live profile; a dry run touches nothing.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "load_imported_facts_under_test", _ROOT / "scripts" / "load_imported_facts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def facts_file(tmp_path):
    p = tmp_path / "imported.json"
    p.write_text(json.dumps({"metadata": {"source": "chatgpt"},
                             "facts": [{"relation": "likes", "object": "tea", "category": "prefs"}]}))
    return str(p)


def test_real_load_without_profile_is_refused(facts_file, monkeypatch, tmp_path):
    live = tmp_path / "live_profile.json"
    monkeypatch.setenv("USER_PROFILE_PATH", str(live))
    mod = _load()
    with pytest.raises(mod.LiveProfileRefused):
        mod.load_facts(facts_file, None, dry_run=False)
    assert not live.exists()


def test_real_load_onto_live_profile_is_refused(facts_file, monkeypatch, tmp_path):
    live = tmp_path / "live_profile.json"
    monkeypatch.setenv("USER_PROFILE_PATH", str(live))
    mod = _load()
    for form in (str(live), str(tmp_path / "sub" / ".." / "live_profile.json")):
        with pytest.raises(mod.LiveProfileRefused):
            mod.load_facts(facts_file, form, dry_run=False)
    assert not live.exists()


def test_dry_run_needs_no_profile_and_writes_nothing(facts_file, monkeypatch, tmp_path):
    live = tmp_path / "live_profile.json"
    monkeypatch.setenv("USER_PROFILE_PATH", str(live))
    stats = _load().load_facts(facts_file, None, dry_run=True)
    assert stats["dry_run"] is True and not live.exists()


def test_sandbox_profile_loads(facts_file, monkeypatch, tmp_path):
    monkeypatch.setenv("USER_PROFILE_PATH", str(tmp_path / "live_profile.json"))
    sandbox = tmp_path / "sandbox" / "user_profile.json"
    stats = _load().load_facts(facts_file, str(sandbox), dry_run=False)
    assert stats["facts_added"] >= 1 and sandbox.exists()
