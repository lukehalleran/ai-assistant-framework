"""DAEMON_TEST_MODE defence in depth for owner-data writers (2026-09-02).

The 2026-09-02 audit suspected a pytest run had triggered a shutdown backup and
appended to the live curation audit; the archived daemon log showed it was the
daemon's own restart. The entry points still had no guard, so a subprocess or
ad-hoc run that bypasses the conftest sandbox could write the owner's files.
"""

import os

from memory.curation.engine import resolve_queue_path
from memory.curation.journal import CurationJournal, resolve_journal_path
from utils.backup_manager import run_shutdown_backup


def test_shutdown_backup_is_skipped_in_test_mode(monkeypatch):
    monkeypatch.setenv("DAEMON_TEST_MODE", "1")
    result = run_shutdown_backup()
    assert result.ok is True
    assert result.skipped_reason == "test_mode"
    assert not result.files_copied


def test_journal_default_redirects_only_when_it_would_hit_prod(monkeypatch):
    import memory.curation.journal as cj
    monkeypatch.setenv("DAEMON_TEST_MODE", "1")
    prod = os.path.join("logs", "curation_audit.jsonl")
    # Un-sandboxed module default (what a subprocess would see).
    monkeypatch.setattr(cj, "_DEFAULT_JOURNAL_PATH", prod)
    assert resolve_journal_path() == os.path.join("logs", "test_curation_audit.jsonl")
    assert CurationJournal().path.endswith("test_curation_audit.jsonl")
    # An explicit path always wins (the engine's own tests pass tmp paths).
    assert resolve_journal_path("/tmp/x.jsonl") == "/tmp/x.jsonl"
    # A sandboxed default (conftest repoints it) is left alone.
    monkeypatch.setattr(cj, "_DEFAULT_JOURNAL_PATH", "/tmp/sandbox/audit.jsonl")
    assert resolve_journal_path() == "/tmp/sandbox/audit.jsonl"


def test_journal_default_untouched_outside_test_mode(monkeypatch):
    import memory.curation.journal as cj
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    prod = os.path.join("logs", "curation_audit.jsonl")
    monkeypatch.setattr(cj, "_DEFAULT_JOURNAL_PATH", prod)
    assert resolve_journal_path() == prod


def test_queue_default_redirects_only_when_it_would_hit_prod(monkeypatch):
    import memory.curation.engine as ce
    monkeypatch.setenv("DAEMON_TEST_MODE", "1")
    prod = os.path.join("data", "curation_queue.json")
    monkeypatch.setattr(ce, "_DEFAULT_QUEUE_PATH", prod)
    assert resolve_queue_path() == os.path.join("data", "test_curation_queue.json")
    assert resolve_queue_path("/tmp/q.json") == "/tmp/q.json"
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    assert resolve_queue_path() == prod
