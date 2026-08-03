"""
Tests for the second wave of adaptive-exemplar adopters (2026-08-02/03):

- need_detector: high-confidence KEYWORD classifications teach the store
  (never neutral); prototypes merge seeds + learned, cache keyed on store
  version.
- web_search_trigger: anchors merge learned "search_worthy" (positive) and
  "no_search" (negative) phrases; cache keyed on store version.
- gate veto: a tone-corroborated veto teaches "no_search" (deterministic
  confirmation that a vent must not search).
- handlers telemetry hook: a response that actually cited [WEB_ results
  teaches "search_worthy" — but never from elevated-tone turns.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.agentic.gate import AgenticDecision, apply_intent_veto
from utils.adaptive_exemplars import get_store
from utils.need_detector import (
    NEED_CONFIG,
    NEED_EXEMPLARS,
    NeedType,
    detect_need_type,
)

VENT = "I am embarrassed for how I reacted earlier. My dad came by for a bit. I am so unhappy"


def _decision():
    return AgenticDecision(
        should_trigger=True, modes=["web_search"], search_terms=["x"],
        matched_entities=[], doc_gen_intent=None, self_note_intent=None,
        skip_initial_search=True, reason="triggered", veto_exempt=False,
    )


class TestNeedDetectorLearning:
    def test_high_confidence_keyword_teaches(self):
        # Multiple presence patterns + short length → keyword fast path.
        msg = "I just need someone to listen, I don't want advice"
        analysis = detect_need_type(msg, model_manager=None)
        if analysis.trigger == "keyword" and analysis.confidence >= NEED_CONFIG[
            "high_confidence_threshold"
        ]:
            learned = get_store().get_learned("need", analysis.need_type.value)
            assert any("listen" in t for t in learned)
        else:
            pytest.skip("message did not hit the keyword fast path — see next test")

    def test_fast_path_teaches_via_mocked_keyword_result(self):
        from utils import need_detector as nd
        strong = nd.NeedAnalysis(
            need_type=NeedType.PRESENCE, confidence=0.9, trigger="keyword",
            raw_scores={}, explanation="",
        )
        with patch.object(nd, "_keyword_need_detection", return_value=strong):
            nd.detect_need_type("I really just need you here with me tonight")
        assert any(
            "here with me tonight" in t
            for t in get_store().get_learned("need", "presence")
        )

    def test_neutral_never_teaches(self):
        from utils import need_detector as nd
        neutral = nd.NeedAnalysis(
            need_type=NeedType.NEUTRAL, confidence=0.95, trigger="keyword",
            raw_scores={}, explanation="",
        )
        with patch.object(nd, "_keyword_need_detection", return_value=neutral), \
             patch.object(nd, "_semantic_need_detection",
                          return_value=nd.NeedAnalysis(NeedType.NEUTRAL, 0.0,
                                                       "semantic", {}, "")):
            nd.detect_need_type("some ambiguous message about things")
        assert get_store().get_learned("need", "neutral") == []

    def test_prototypes_merge_learned(self):
        from utils import need_detector as nd
        get_store().record("need", "presence", "please just stay with me a while", "t")
        seen = []

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True):
                seen.append(list(texts))
                return np.ones((len(texts), 4))

        with patch.object(nd, "_get_embedder", return_value=FakeEmbedder()):
            nd._need_exemplar_embeddings_cache = None
            protos = nd._get_need_exemplar_embeddings(None)
        assert set(protos) == set(NEED_EXEMPLARS)
        batch = next(b for b in seen if "please just stay with me a while" in b)
        assert set(NEED_EXEMPLARS["presence"]).issubset(set(batch))


class TestWebTriggerAnchors:
    def test_anchors_merge_learned_and_invalidate_on_version(self):
        from utils import web_search_trigger as wt
        get_store().record("web_search", "search_worthy",
                           "what changed in the transit schedule this week", "t")
        seen = []

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
                seen.append(list(texts))
                return np.ones((len(texts), 4))

        with patch("models.model_manager.ModelManager._get_cached_embedder",
                   return_value=FakeEmbedder()):
            wt._search_anchor_embs = None
            wt._no_search_anchor_embs = None
            wt._search_anchor_version = None
            pos, neg = wt._get_search_anchors()
            assert pos is not None
            assert any("transit schedule" in " ".join(b) for b in seen)
            calls = len(seen)
            wt._get_search_anchors()
            assert len(seen) == calls  # cached
            get_store().record("web_search", "no_search",
                               "i just feel awful about the whole thing today", "t")
            wt._get_search_anchors()
            assert len(seen) > calls  # recomputed after learning
        wt._search_anchor_embs = None
        wt._no_search_anchor_embs = None
        wt._search_anchor_version = None


class TestVetoTeachesNoSearch:
    def test_tone_veto_records_no_search(self):
        d = apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="light_support", query=VENT,
        )
        assert d.should_trigger is False
        assert any(
            "so unhappy" in t
            for t in get_store().get_learned("web_search", "no_search")
        )

    def test_no_veto_no_learning(self):
        apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="conversational", query=VENT,
        )
        assert get_store().get_learned("web_search", "no_search") == []


class TestCitationTeachesSearchWorthy:
    def _ctx(self, tone="conversational"):
        ctx = MagicMock()
        ctx.user_text = "what's the latest on the kimi k3 release"
        ctx.telemetry = {}
        ctx.t_prepare_elapsed = 1.0
        ctx.orchestrator._last_turn_signals = {"tone_level": tone}
        return ctx

    def test_web_cited_response_teaches(self):
        from gui.handlers import _write_turn_telemetry
        _write_turn_telemetry(
            self._ctx(), "enhanced", "s", "m", 100,
            response_text="Per recent coverage [WEB_1], the release...",
        )
        assert any(
            "kimi k3" in t
            for t in get_store().get_learned("web_search", "search_worthy")
        )

    def test_uncited_response_does_not_teach(self):
        from gui.handlers import _write_turn_telemetry
        _write_turn_telemetry(
            self._ctx(), "enhanced", "s", "m", 100,
            response_text="Here's what I know from memory.",
        )
        assert get_store().get_learned("web_search", "search_worthy") == []

    def test_elevated_tone_never_teaches(self):
        from gui.handlers import _write_turn_telemetry
        _write_turn_telemetry(
            self._ctx(tone="light_support"), "enhanced", "s", "m", 100,
            response_text="I looked this up [WEB_1] because you asked.",
        )
        assert get_store().get_learned("web_search", "search_worthy") == []
