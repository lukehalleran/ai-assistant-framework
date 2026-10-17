"""2026-10-14 (class: BC-09, BC-06): a vent followed by an inline paste.

The user's own two opening sentences were diluted by 1,500 chars of neutral
homework in ONE embedding (tone CONVERSATIONAL, conf 0.19). The deployed
`detect_crisis_level` now also scores the structural authored head
(`utils.paste_shape.authored_head`) and keeps the more severe result.
Synthetic fake embedder: marker tokens map to axes, no real model.

Original batch label: oct14e (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""
import numpy as np
import pytest

import utils.tone_detector as td
from utils.paste_shape import authored_head

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
    taught = []
    monkeypatch.setattr(td, "_learn_tone_exemplar", lambda *a, **k: taught.append(a))
    return taught


async def _detect(msg):
    return await td.detect_crisis_level(msg, None, None, None)


class TestAuthoredHead:
    def test_incident_shape_returns_vent_only(self):
        h = authored_head(INCIDENT)
        assert h.startswith("I feel terrible") and h.endswith("FEELBAD FEELBAD:")
        assert "MATH" not in h

    def test_short_message_empty(self):
        assert authored_head("I feel terrible. I worked all day.") == ""

    def test_no_boundary_caps_at_60_words(self):
        t = " ".join(f"w{i}" for i in range(200))
        assert authored_head(t) == " ".join(f"w{i}" for i in range(60))

    def test_list_first_paste_empty(self):
        t = "- item one\n- item two\n" + " ".join(["x"] * 200)
        assert authored_head(t) == ""

    def test_blank_line_boundary(self):
        t = "I feel really bad about today.\n\n" + " ".join(["x"] * 200)
        assert authored_head(t) == "I feel really bad about today."


@pytest.mark.asyncio
async def test_head_distress_lifts_diluted_paste():
    r = await _detect(INCIDENT)
    assert r.level.value != "conversational"
    assert getattr(r.raw_scores, "source", "whole") == "head"
    assert "semantic_source=head" in r.explanation


@pytest.mark.asyncio
async def test_whole_message_alone_is_conversational_in_fake_space():
    # the same text without a recognised boundary/size: head path off -> unchanged
    r = await _detect(("FEELBAD " * 2) + PASTE.replace("MATH", "MATH")[:0] + "MATH " * 20)
    assert r.level.value == "conversational"


@pytest.mark.asyncio
async def test_neutral_head_neutral_paste_unchanged():
    r = await _detect("Here is my progress summary for today and notes:\n\n" + PASTE)
    assert r.level.value == "conversational"
    assert getattr(r.raw_scores, "source", "whole") == "whole"


@pytest.mark.asyncio
async def test_distressed_whole_never_lowered():
    msg = "Here is my update for the week and notes:\n\n" + " ".join(["FEELBAD"] * 200)
    r = await _detect(msg)
    assert r.level.value != "conversational"


@pytest.mark.asyncio
async def test_head_path_teaches_nothing(fake_space):
    await _detect(INCIDENT)
    assert fake_space == []


# --- referee addition (2026-10-14) -------------------------------------------
@pytest.mark.asyncio
async def test_arbiter_teaches_the_head_not_the_paste(monkeypatch):
    """When the authored head drove the semantic level and the LLM arbiter
    confirms, the exemplar taught is the head -- never the pasted body (BC-29)."""
    scores = td._SemanticScores({"high": 0.1, "medium": 0.2, "concern": 0.45, "conversational": 0.2})
    scores.source = "head"
    monkeypatch.setattr(td, "_semantic_crisis_detection",
                        lambda *a, **k: (td.CrisisLevel.CONCERN, 0.45, scores))

    async def arbiter(message, model_manager):
        return (td.CrisisLevel.CONCERN, 0.8), "llm", {"backend": "llm"}

    monkeypatch.setattr(td, "_run_arbiter", arbiter)
    taught = []
    monkeypatch.setattr(td, "_learn_tone_exemplar", lambda text, *a, **k: taught.append(text))
    r = await td.detect_crisis_level(INCIDENT, None, object(), None)
    assert r.level == td.CrisisLevel.CONCERN
    assert taught and "MATH" not in taught[0] and taught[0].startswith("I feel terrible")
