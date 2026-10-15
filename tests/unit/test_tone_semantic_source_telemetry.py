"""2026-10-15 (class: BC-72): which text the tone semantic stage scored
("head" = the authored opening of a paste, "whole") lived only in the
explanation string, so turn records could not show how often the head path
decided a turn. Now ToneAnalysis.semantic_source -> EmotionalContext.
tone_semantic_source -> the turn record's `tone_semantic_source`. Enum only.
Synthetic fake embedder (same as test_oct14e_paste_tone_head).
"""
import inspect

import numpy as np
import pytest

import utils.tone_detector as td
import utils.turn_telemetry as tt
from types import SimpleNamespace
from utils import emotional_context as ec
from core import orchestrator

VENT = "I feel terrible. I worked for about 5 hours but I feel like I wasted my time. FEELBAD FEELBAD"
PASTE = " ".join(["MATH"] * 200)
INCIDENT = VENT + ":\n\n" + PASTE
AXES = {"high": 0, "medium": 1, "concern": 2, "conversational": 3}


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True):
        v = np.zeros(4)
        v[2] = text.count("FEELBAD")
        v[3] = text.count("MATH") + 0.01
        return v


@pytest.fixture(autouse=True)
def fake_space(monkeypatch):
    monkeypatch.setattr(td, "_get_embedder", lambda mm=None: FakeEmbedder())
    monkeypatch.setattr(td, "_get_exemplar_embeddings",
                        lambda mm=None: {k: np.eye(4)[i] for k, i in AXES.items()})
    monkeypatch.setattr(td, "_recent_distress_from_history", lambda h: False)
    monkeypatch.setattr(td, "_learn_tone_exemplar", lambda *a, **k: None)
    # the need detector runs inside analyze_emotional_context; it is not under test
    import utils.emotional_context as _ec
    monkeypatch.setattr(
        _ec, "detect_need_type",
        lambda m, mm=None: SimpleNamespace(need_type=_ec.NeedType.NEUTRAL if hasattr(_ec.NeedType, "NEUTRAL") else list(_ec.NeedType)[0],
                                           confidence=0.5, trigger="stub", explanation="stub"),
    )


@pytest.mark.asyncio
async def test_long_paste_with_authored_head_reports_head():
    ctx = await ec.analyze_emotional_context(INCIDENT)
    assert ctx.tone_semantic_source == "head"
    assert ec.tone_semantic_source_field(ctx) == "head"


@pytest.mark.asyncio
async def test_short_message_reports_whole():
    tone = await td.detect_crisis_level("FEELBAD FEELBAD I feel bad today", None, None, None)
    assert tone.semantic_source == "whole"
    ctx = await ec.analyze_emotional_context("FEELBAD FEELBAD I feel bad today")
    assert ctx.tone_semantic_source == "whole"


@pytest.mark.asyncio
async def test_no_semantic_stage_reports_none(monkeypatch):
    monkeypatch.setattr(td, "_get_embedder", lambda mm=None: None)
    ctx = await ec.analyze_emotional_context("hello there")
    assert ctx.tone_semantic_source is None
    assert ec.tone_semantic_source_field(ctx) is None


def test_field_helper_is_enum_only():
    class C:
        tone_semantic_source = "some free text"
    assert ec.tone_semantic_source_field(C()) is None
    assert ec.tone_semantic_source_field(None) is None


def test_turn_record_builder_writes_the_field_and_docs_list_it():
    src = inspect.getsource(orchestrator)
    assert '"tone_semantic_source": tone_semantic_source_field(' in src
    assert "tone_semantic_source" in tt.__doc__
