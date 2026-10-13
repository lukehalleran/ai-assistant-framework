"""Tone detector degraded mode (2026-10-13, beta path): no embedder / no exemplar
embeddings / encode raising makes ``_semantic_crisis_detection`` return raw_scores={}.
``detect_crisis_level`` used to crash on ``max({}.values())``; the pipeline turned the crash
into CONVERSATIONAL, skipping the arbiter and the domain-strain floor. All tests drive THE
deployed ``detect_crisis_level``; the arbiter is stubbed at ``_llm_crisis_fallback`` (no network).
class: empty-iterable reduction (BC-93), BC-69
"""
import asyncio

import pytest

import utils.tone_detector as td
from utils.tone_detector import CrisisLevel, detect_crisis_level

NEUTRAL = "can you help me plan my week"
PLAIN = "the neighbours repainted their fence recently"
STRAIN = "work has been rough lately honestly"
# a message the deployed _domain_strain_hit floor recognises (STRAIN above does not)
FLOOR_MSG = "taking care of my dad after his stroke is wearing me out"


class _BadEmbedder:
    def encode(self, *a, **k):
        raise RuntimeError("encode boom")


class _OkEmbedder:
    def encode(self, *a, **k):  # pragma: no cover - only used by path 3
        raise RuntimeError("encode boom")


def _degrade(monkeypatch, path):
    if path == "no_embedder":
        monkeypatch.setattr(td, "_get_embedder", lambda mm=None: None)
    elif path == "no_exemplars":
        monkeypatch.setattr(td, "_get_embedder", lambda mm=None: _OkEmbedder())
        monkeypatch.setattr(td, "_get_exemplar_embeddings", lambda mm=None: {})
    else:
        monkeypatch.setattr(td, "_get_embedder", lambda mm=None: _BadEmbedder())
        monkeypatch.setattr(td, "_get_exemplar_embeddings", lambda mm=None: {"high": 1})


PATHS = ["no_embedder", "no_exemplars", "encode_raises"]


def _run(msg, mm=None, **kw):
    return asyncio.run(detect_crisis_level(msg, model_manager=mm, **kw))


@pytest.fixture
def arbiter(monkeypatch):
    calls = []
    verdict = {"v": (CrisisLevel.CONCERN, 0.7)}

    async def fake(message, model_manager=None):
        calls.append(message)
        return verdict["v"]

    monkeypatch.setattr(td, "_llm_crisis_fallback", fake)
    monkeypatch.setattr(td, "_learn_tone_exemplar", lambda *a, **k: calls.append("LEARN"))
    return calls, verdict


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("msg", [NEUTRAL, PLAIN, STRAIN])
def test_never_raises(monkeypatch, path, msg):
    _degrade(monkeypatch, path)
    result = _run(msg)
    assert isinstance(result, td.ToneAnalysis)


@pytest.mark.parametrize("path", PATHS)
def test_arbiter_concern_wins(monkeypatch, arbiter, path):
    _degrade(monkeypatch, path)
    calls, _ = arbiter
    result = _run(PLAIN, mm=object())
    assert result.level == CrisisLevel.CONCERN
    assert result.trigger == "llm_fallback"
    assert calls.count(PLAIN) == 1


@pytest.mark.parametrize("path", PATHS)
def test_no_model_manager_strain_floor_still_fires(monkeypatch, path):
    _degrade(monkeypatch, path)
    assert td._domain_strain_hit(FLOOR_MSG)
    result = _run(FLOOR_MSG, mm=None)
    assert result.level == CrisisLevel.CONCERN
    assert result.trigger == td.DOMAIN_STRAIN_TRIGGER


def test_explicit_casual_skips_arbiter(monkeypatch, arbiter):
    _degrade(monkeypatch, "no_embedder")
    calls, _ = arbiter
    result = _run("lol ok", mm=object())
    assert calls == []
    assert result.level == CrisisLevel.CONVERSATIONAL


def test_task_directive_skips_arbiter(monkeypatch, arbiter):
    _degrade(monkeypatch, "no_embedder")
    calls, _ = arbiter
    msg = "please write a python function that sorts a list of numbers"
    assert td._is_task_directive_signal(msg)
    _run(msg, mm=object())
    assert calls == []


def test_degraded_does_not_teach_exemplars(monkeypatch, arbiter):
    _degrade(monkeypatch, "no_embedder")
    calls, _ = arbiter
    _run(PLAIN, mm=object())
    assert "LEARN" not in calls


def test_conversational_explanation_names_degraded_mode(monkeypatch, arbiter):
    _degrade(monkeypatch, "no_embedder")
    calls, verdict = arbiter
    verdict["v"] = None  # arbiter unavailable
    result = _run(PLAIN, mm=object())
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert "semantic stage unavailable" in result.explanation


def test_missing_embedder_warning_logs_once(monkeypatch, caplog):
    _degrade(monkeypatch, "no_embedder")
    monkeypatch.setattr(td, "_SEMANTIC_UNAVAILABLE_WARNED", False, raising=False)
    import logging
    with caplog.at_level(logging.WARNING, logger=td.logger.name):
        _run(PLAIN)
        _run(PLAIN)
    hits = [r for r in caplog.records if "No embedder available" in r.getMessage()]
    assert len(hits) == 1
