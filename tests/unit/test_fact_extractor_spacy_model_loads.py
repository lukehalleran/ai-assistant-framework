"""memory/fact_extractor._get_nlp — the spaCy model must load, and a failure is loud.

Incident 2026-10-24: preshed 3.0.14 made spacy.load raise OverflowError; the
failure was cached at DEBUG so 167 spaCy-dependent tests silently took their
no-parser fallback in CI. Class: silent swallow / degraded-without-notice
(proposed BC for floating transitive dependencies).
conftest check: no spaCy stubbing under DAEMON_TEST_MODE (see handoff).
"""
import importlib.util
import logging

import pytest

import memory.fact_extractor as fe


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(fe, "_NLP", None)
    monkeypatch.setattr(fe, "_NLP_LOADED", False)


def test_model_loads_with_sane_lemmas(fresh_cache):
    if importlib.util.find_spec("en_core_web_sm") is None:
        pytest.skip("en_core_web_sm not installed")
    nlp = fe._get_nlp()
    assert nlp is not None, "spaCy model failed to load (dependency drift?)"
    assert [t.lemma_ for t in nlp("clusters ate")] == ["cluster", "eat"]


def test_load_failure_logs_warning_with_exception_type(fresh_cache, monkeypatch, caplog):
    spacy = pytest.importorskip("spacy")

    def boom(*a, **k):
        raise OverflowError("Python int too large to convert to C ssize_t")

    monkeypatch.setattr(spacy, "load", boom)
    with caplog.at_level(logging.DEBUG, logger=fe.logger.name):
        assert fe._get_nlp() is None
        assert fe._get_nlp() is None  # cached: not retried per call
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "OverflowError" in warns[0].getMessage()
    assert "parsing degraded" in warns[0].getMessage()


def test_missing_spacy_stays_debug(fresh_cache, monkeypatch, caplog):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "spacy":
            raise ImportError("no spacy")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with caplog.at_level(logging.DEBUG, logger=fe.logger.name):
        assert fe._get_nlp() is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_model_not_installed_stays_debug_with_install_hint(fresh_cache, monkeypatch, caplog):
    spacy = pytest.importorskip("spacy")
    real = importlib.util.find_spec

    def boom(*a, **k):
        raise OSError("[E050] Can't find model 'en_core_web_sm'")

    monkeypatch.setattr(spacy, "load", boom)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: None if n == "en_core_web_sm" else real(n, *a, **k))
    with caplog.at_level(logging.DEBUG, logger=fe.logger.name):
        assert fe._get_nlp() is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not installed" in r.getMessage() and "spacy download" in r.getMessage()
               for r in caplog.records)
