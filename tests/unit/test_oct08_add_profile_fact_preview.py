"""2026-10-08 (class: BC-70, BC-58): the add_profile_fact.py dry-run preview
uses add_fact's own supersession rule (relation_classifier.supersedes_on_new_value),
so it never says "SUPERSEDE" for a multi-valued relation that --apply would
append alongside. Synthetic tmp profile only; fictional drug names."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from memory import relation_classifier as rc
from memory.user_profile import UserProfile

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "add_profile_fact.py"


@pytest.fixture
def script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("add_profile_fact_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(UserProfile, "DEFAULT_PATH", str(tmp_path / "profile.json"))
    monkeypatch.setattr(mod, "_daemon_running", lambda: False)
    return mod


def _seed(tmp_path, facts):
    p = UserProfile(profile_path=str(tmp_path / "profile.json"))
    for rel, val in facts:
        assert p.add_fact(rel, val, 0.9, "seed")
    p.save()


def _run(script, monkeypatch, capsys, relation, value, apply=False):
    argv = ["add_profile_fact.py", "--relation", relation, "--value", value]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr("sys.argv", argv)
    assert script.main() == 0
    return capsys.readouterr().out


def _previewed(out):
    return set(re.findall(r"^\s+- '([^']*)'$", out, flags=re.M))


def _historical(tmp_path, relation):
    p = UserProfile(profile_path=str(tmp_path / "profile.json"))
    return {
        f["value"]
        for facts in p.profile["categories"].values()
        for f in facts
        if isinstance(f, dict) and f.get("relation") == relation and not f.get("is_current", True)
    }


def test_helper_rule():
    assert rc.supersedes_on_new_value("age", "34", "33") is True
    assert rc.supersedes_on_new_value("medication_name", "Lorvatin", "kavarin") is False
    assert rc.supersedes_on_new_value("medication_dose", "kavarin 30 mg", "20 mg kavarin daily") is True
    assert rc.supersedes_on_new_value("medication_dose", "kavarin 30 mg", "Zelphex 10 mg") is False


def test_multi_valued_relation_does_not_claim_supersede(script, tmp_path, monkeypatch, capsys):
    _seed(tmp_path, [("medication_name", "kavarin")])
    out = _run(script, monkeypatch, capsys, "medication_name", "Lorvatin")
    assert "SUPERSEDE" not in out
    assert "ADDS this value alongside 1 current value(s)" in out
    assert _previewed(out) == set()
    _run(script, monkeypatch, capsys, "medication_name", "Lorvatin", apply=True)
    assert _historical(tmp_path, "medication_name") == set()


def test_single_valued_relation_supersedes(script, tmp_path, monkeypatch, capsys):
    _seed(tmp_path, [("age", "33")])
    out = _run(script, monkeypatch, capsys, "age", "34")
    assert "SUPERSEDE" in out
    assert "ADDS this value" not in out
    previewed = _previewed(out)
    assert previewed == {"33"}
    _run(script, monkeypatch, capsys, "age", "34", apply=True)
    assert _historical(tmp_path, "age") == previewed


def test_dose_supersedes_only_same_referent(script, tmp_path, monkeypatch, capsys):
    _seed(tmp_path, [
        ("medication_name", "kavarin"),
        ("medication_dose", "20 mg kavarin daily"),
        ("medication_dose", "Zelphex 10 mg"),
    ])
    out = _run(script, monkeypatch, capsys, "medication_dose", "kavarin 30 mg")
    previewed = _previewed(out)
    assert previewed == {"20 mg kavarin daily"}
    assert "Zelphex" not in "".join(previewed)
    assert "ADDS this value alongside 1 current value(s)" in out
    _run(script, monkeypatch, capsys, "medication_dose", "kavarin 30 mg", apply=True)
    assert _historical(tmp_path, "medication_dose") == previewed
