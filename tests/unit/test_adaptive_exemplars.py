"""
Tests for the adaptive exemplar store + tone-detector learning loop
(2026-08-02: replaces the hand-maintained-keyword-list treadmill — three
borderline tone misses in one day were each fixed by hand-editing exemplar
lists; confirmed classifications now grow the semantic channel per-user).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import utils.adaptive_exemplars as ae
from utils.adaptive_exemplars import AdaptiveExemplarStore, get_store
from utils.tone_detector import (
    CRISIS_EXEMPLARS,
    CrisisLevel,
    TONE_CONFIG,
    detect_crisis_level,
)


class TestStore:
    def test_record_and_reload_roundtrip(self, tmp_path):
        p = tmp_path / "ex.json"
        s = AdaptiveExemplarStore(str(p))
        assert s.record("tone", "concern", "I feel completely hollowed out today", "test")
        assert s.version == 1
        s2 = AdaptiveExemplarStore(str(p))
        assert s2.get_learned("tone", "concern") == [
            "I feel completely hollowed out today"
        ]

    def test_too_short_rejected(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        assert not s.record("tone", "concern", "sad", "test")
        assert s.version == 0

    def test_exact_duplicate_rejected_vs_learned_and_seeds(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        assert s.record("tone", "concern", "I feel completely hollowed out", "test")
        assert not s.record("tone", "concern", "i feel COMPLETELY hollowed out", "test")
        assert not s.record(
            "tone", "concern", "I am so unhappy about everything", "test",
            seed_texts=["i am so unhappy about everything"],
        )

    def test_near_duplicate_rejected_with_embedder(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        s.record("tone", "concern", "I feel completely hollowed out", "test")
        emb = MagicMock()
        # query vector identical to the stored exemplar's vector → cos = 1.0
        emb.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert not s.record(
            "tone", "concern", "I feel absolutely hollowed out", "test", embedder=emb
        )

    def test_per_label_cap_evicts_oldest(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        for i in range(45):
            s.record("tone", "concern", f"unique distress phrasing number {i} here", "t")
        learned = s.get_learned("tone", "concern")
        assert len(learned) == 40
        assert "number 0 " not in " | ".join(learned)

    def test_corrupt_file_starts_empty(self, tmp_path):
        p = tmp_path / "ex.json"
        p.write_text("{broken")
        s = AdaptiveExemplarStore(str(p))
        assert s.get_learned("tone", "concern") == []

    def test_long_text_clipped(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        s.record("tone", "medium", "x" * 1000, "test")
        assert len(s.get_learned("tone", "medium")[0]) == 300


class TestToneLearningLoop:
    @pytest.mark.asyncio
    async def test_keyword_confirmation_teaches(self):
        # A keyword-routed distress message must land in the adaptive store.
        msg = "I'm having a panic attack and I can't breathe, everything is falling apart"
        analysis = await detect_crisis_level(msg, model_manager=None)
        assert analysis.level in (CrisisLevel.MEDIUM, CrisisLevel.HIGH)
        learned = get_store().get_learned("tone", analysis.level.name.lower())
        assert any("panic attack" in t for t in learned)

    @pytest.mark.asyncio
    async def test_arbiter_confirmation_teaches(self):
        msg = (
            "I am back now. I keep thinking I am a stupid piece of shit. "
            "I know it's the meds but I wanna cry"
        )
        scores = {"high": 0.365, "medium": 0.390, "concern": 0.349,
                  "conversational": 0.302}
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, scores),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.CONCERN, 0.7)),
        ):
            analysis = await detect_crisis_level(msg, model_manager=MagicMock())
        assert analysis.level == CrisisLevel.CONCERN
        assert any("wanna cry" in t for t in get_store().get_learned("tone", "concern"))

    @pytest.mark.asyncio
    async def test_backstop_never_teaches(self):
        # Heuristic channel must not self-reinforce.
        msg = (
            "I am back now. I keep thinking I am a stupid piece of shit. "
            "I know it's the meds but I wanna cry"
        )
        scores = {"high": 0.365, "medium": 0.390, "concern": 0.349,
                  "conversational": 0.302}
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, scores),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=None),
        ):
            analysis = await detect_crisis_level(msg, model_manager=MagicMock())
        assert analysis.trigger == "borderline_backstop"
        assert get_store().get_learned("tone", "concern") == []

    @pytest.mark.asyncio
    async def test_conversational_never_teaches(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.7,
                          {"high": 0.1, "medium": 0.1, "concern": 0.1,
                           "conversational": 0.7}),
        ):
            await detect_crisis_level(
                "Just got back from the gym, feeling alright honestly",
                model_manager=None,
            )
        assert get_store()._data == {}

    @pytest.mark.asyncio
    async def test_learning_disable_flag(self):
        msg = "I'm having a panic attack and I can't breathe right now"
        with patch.dict(TONE_CONFIG, {"exemplar_learning": False}):
            await detect_crisis_level(msg, model_manager=None)
        assert get_store()._data == {}


class TestPrototypeMerge:
    def test_learned_exemplars_join_prototypes(self):
        import utils.tone_detector as td
        get_store().record("tone", "concern", "everything feels heavy and gray today", "t")
        seen = []

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True):
                seen.append(list(texts))
                return np.ones((len(texts), 4))

        with patch.object(td, "_get_embedder", return_value=FakeEmbedder()):
            td._exemplar_embeddings_cache = None
            protos = td._get_exemplar_embeddings(None)
        assert set(protos) == set(CRISIS_EXEMPLARS)
        concern_batch = next(
            b for b in seen if "everything feels heavy and gray today" in b
        )
        assert set(CRISIS_EXEMPLARS["concern"]).issubset(set(concern_batch))

    def test_cache_invalidates_on_new_learning(self):
        import utils.tone_detector as td

        class FakeEmbedder:
            calls = 0
            def encode(self, texts, convert_to_numpy=True):
                FakeEmbedder.calls += 1
                return np.ones((len(texts), 4))

        with patch.object(td, "_get_embedder", return_value=FakeEmbedder()):
            td._exemplar_embeddings_cache = None
            td._get_exemplar_embeddings(None)
            first = FakeEmbedder.calls
            td._get_exemplar_embeddings(None)
            assert FakeEmbedder.calls == first  # cached
            get_store().record("tone", "medium", "a brand new confirmed phrasing here", "t")
            td._get_exemplar_embeddings(None)
            assert FakeEmbedder.calls > first  # recomputed after learning
