"""utils/preflight._check_dependency_drift — advisory lock-drift warning.

Incident 2026-10-24: one floating transitive release (preshed 3.0.14) broke
spacy.load on a fresh install and nothing at startup said so.
Class: proposed BC (floating transitive dependencies); silent degradation family.
"""
import sys

import utils.preflight as pf
from utils.preflight import PreflightResult


def _lock(tmp_path, lines):
    p = tmp_path / "requirements.lock.txt"
    p.write_text("# header\n" + "\n".join(lines) + "\n")
    return str(p)


def _run(monkeypatch, tmp_path, lock_lines, installed):
    monkeypatch.setattr(pf, "_repo_lock_path", lambda: _lock(tmp_path, lock_lines))
    monkeypatch.setattr(pf, "_installed_version", lambda n: installed.get(n))
    monkeypatch.delattr(sys, "frozen", raising=False)
    r = PreflightResult()
    pf._check_dependency_drift(r)
    return r


def test_no_drift_no_warning(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, ["numpy==1.26.4", "Foo_Bar==2.0"],
             {"numpy": "1.26.4", "foo-bar": "2.0"})
    assert r.warnings == []


def test_drift_one_warning_with_count_examples_and_fix(monkeypatch, tmp_path):
    lines = [f"pkg{i}==1.0" for i in range(7)]
    installed = {f"pkg{i}": "2.0" for i in range(7)}
    r = _run(monkeypatch, tmp_path, lines, installed)
    assert len(r.warnings) == 1
    w = r.warnings[0]
    assert w.startswith("7 installed")
    assert "pkg0: installed 2.0, lock 1.0" in w
    assert "pkg4: installed 2.0, lock 1.0" in w
    assert "pkg5" not in w  # at most 5 examples
    assert "pip install -r requirements.txt -c requirements.lock.txt" in w


def test_missing_package_counts_as_drift(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, ["ghost==1.0"], {})
    assert len(r.warnings) == 1
    assert "ghost: not installed, lock 1.0" in r.warnings[0]


def test_local_label_is_not_drift(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, ["torch==2.11.0"], {"torch": "2.11.0+cu126"})
    assert r.warnings == []


def test_frozen_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_repo_lock_path", lambda: _lock(tmp_path, ["ghost==1.0"]))
    monkeypatch.setattr(pf, "_installed_version", lambda n: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    r = PreflightResult()
    pf._check_dependency_drift(r)
    assert r.warnings == []


def test_absent_lock_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_repo_lock_path", lambda: str(tmp_path / "nope.txt"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    r = PreflightResult()
    pf._check_dependency_drift(r)
    assert r.warnings == []


def test_registered_in_run_preflight(monkeypatch):
    called = []
    monkeypatch.setattr(pf, "_check_dependency_drift", lambda r: called.append(1))
    # run_preflight iterates a tuple built at call time from module globals
    pf.run_preflight()
    assert called
