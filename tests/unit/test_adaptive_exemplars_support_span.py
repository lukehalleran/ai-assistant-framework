"""
A learned exemplar keeps the text that supports its label.

Incident date: 2026-10-23 (oct23c LEARN lane; audit findings A06 + A09).
Class: BC-29 / BC-52 (A06: label learned from clipped text that no longer
carries the evidence), BC-20 / BC-36 (A09: malformed persisted entry crashes
the reader of a self-healing derived store).

Drives the deployed store, the deployed tone teacher (`_learn_tone_exemplar`),
the deployed intent teacher (`IntentClassifier.classify`) and the real
`get_learned`. All fixture text is invented.
"""

import json

import pytest

import core.intent_classifier as ic
import utils.tone_detector as td
from utils.adaptive_exemplars import AdaptiveExemplarStore, get_store

_NEUTRAL = (
    "We went through the grocery list for the week and talked about the pasta "
    "recipe, the oven timing, and who is bringing the salad on Saturday. "
)
_TAIL = "Then it hit me and I had a panic attack in the hallway."


def _long_distress(head_repeats=3):
    return _NEUTRAL * head_repeats + _TAIL


def _wrapped(text, width=60):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n    ".join(lines)


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch):
    monkeypatch.setattr(td, "_get_embedder", lambda mm=None: None)


@pytest.mark.parametrize("shape", ["clean", "wrapped"])
class TestToneTeacherKeepsSupport:
    def test_long_message_stores_window_with_keyword(self, shape):
        msg = _long_distress()
        assert len(msg) > 300
        if shape == "wrapped":
            msg = _wrapped(msg)
        td._learn_tone_exemplar(msg, "medium", "keyword")
        learned = get_store().get_learned("tone", "medium")
        assert learned, "a supported window should be learned"
        assert all(len(t) <= 300 for t in learned)
        assert all("panic attack" in t for t in learned)

    def test_short_message_unchanged(self, shape):
        msg = "I had a panic attack in the hallway today."
        if shape == "wrapped":
            msg = _wrapped(msg, 20)
        td._learn_tone_exemplar(msg, "medium", "keyword")
        assert get_store().get_learned("tone", "medium") == [" ".join(msg.split())]

    def test_long_message_without_span_source_declines(self, shape):
        msg = _long_distress()
        if shape == "wrapped":
            msg = _wrapped(msg)
        td._learn_tone_exemplar(msg, "medium", "arbiter")
        assert get_store().get_learned("tone", "medium") == []


@pytest.mark.parametrize("shape", ["clean", "wrapped"])
class TestIntentTeacherKeepsSupport:
    def test_long_regex_hit_in_tail_stores_the_hit(self, shape):
        q = _NEUTRAL * 3 + "What's my sister's name?"
        assert len(q) > 300
        if shape == "wrapped":
            q = _wrapped(q)
        result = ic.IntentClassifier().classify(q)
        assert result.source == "regex" and result.confidence >= 0.85
        learned = get_store().get_learned("intent", result.intent.value)
        assert learned and all(len(t) <= 300 for t in learned)
        assert all("sister" in t for t in learned)

    def test_short_query_unchanged(self, shape):
        q = "What's my sister's name?"
        result = ic.IntentClassifier().classify(q)
        assert get_store().get_learned("intent", result.intent.value) == [q]

    def test_stm_refined_long_query_declines(self, shape):
        clf = ic.IntentClassifier()
        weak = clf._build_result(ic.IntentType.GENERAL, 0.0)
        q = _NEUTRAL * 3 + "honestly today just broke me a little"
        if shape == "wrapped":
            q = _wrapped(q)
        refined = clf.refine_with_stm(
            weak, "seeking emotional support and comfort", query=q
        )
        assert refined.source == "stm_refined"
        assert get_store().get_learned("intent", refined.intent.value) == []


class TestStoreBoundary:
    def test_invalid_spans_decline_on_long_text(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        long = _NEUTRAL * 3
        for span in (None, (5, 5), (-1, 4), (10, 99999), ("a", "b"), (7,)):
            assert not s.record("tone", "medium", long, "t", support_span=span)
        assert s.version == 0

    def test_window_contains_span_in_the_middle(self, tmp_path):
        s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
        text = ("alpha " * 100) + "MARKERWORD " + ("omega " * 100)
        i = text.index("MARKERWORD")
        assert s.record("tone", "medium", text, "t", support_span=(i, i + 10))
        out = s.get_learned("tone", "medium")[0]
        assert "MARKERWORD" in out and len(out) <= 300
        assert not out.startswith(" ") and not out.endswith(" ")
        assert all(w in ("alpha", "MARKERWORD", "omega") for w in out.split())


class TestMalformedEntriesOnLoad:
    def test_empty_entry_dropped_no_keyerror(self, tmp_path):
        p = tmp_path / "ex.json"
        p.write_text(json.dumps({"tone": {"medium": [{}]}}))
        s = AdaptiveExemplarStore(str(p))
        assert s.get_learned("tone", "medium") == []

    def test_mixed_file_keeps_valid_entries(self, tmp_path):
        p = tmp_path / "ex.json"
        good = {"text": "I feel hollowed out today", "ts": "2026-10-01", "source": "t"}
        p.write_text(json.dumps({
            "tone": {
                "medium": [{}, good, "bare string", {"text": 5}, {"text": "  "}, None],
                "high": "not a list",
            },
            "intent": ["wrong", "shape"],
            "need": {"presence": [good]},
            7: {},
        }))
        s = AdaptiveExemplarStore(str(p))
        assert s.get_learned("tone", "medium") == [good["text"]]
        assert s.get_learned("tone", "high") == []
        assert s.get_learned("intent", "x") == []
        assert s.get_learned("need", "presence") == [good["text"]]
        # the loaded store still accepts new exemplars
        assert s.record("tone", "medium", "a brand new confirmed phrasing here", "t")


def test_spanless_long_decline_is_logged_at_info_with_reason(tmp_path, caplog):
    # 2026-10-26 (lane H1, #101 follow-up 2c; class: BC-58): the decline used
    # to be DEBUG-only, invisible in prod. Same instance: span given -> learned
    # (no decline line); span absent -> declined AND a visible reason.
    import logging
    s = AdaptiveExemplarStore(str(tmp_path / "ex.json"))
    long = "x " * 400
    with caplog.at_level(logging.INFO):
        assert s.record("tone", "medium", long, "gate_veto", support_span=(10, 15))
        assert "not learned" not in caplog.text
        assert not s.record("tone", "medium", long, "gate_veto")
    assert "not learned" in caplog.text
    assert "no usable supporting span" in caplog.text and "source=gate_veto" in caplog.text
    # indented / wrapped form of the same long text declines identically
    wrapped = "\n    ".join(long.split(" ")[:400])
    assert not s.record("tone", "medium", wrapped, "gate_veto")
