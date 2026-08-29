"""
Forced-minimum backfill quality floor (2026-08-23).

The 2026-08-23 pattern-doc turn: "Only 0 memories passed, forcing minimum of
8" back-filled exam scores and git commits (blended 0.523–0.610) into a
therapy-pattern prompt. Fail-soft means "don't starve an ordinary turn", not
"force in anything": backfill now stops at threshold − GATE_FORCED_FLOOR_MARGIN
(default 0.05) — near-misses may be rescued, far-below junk is refused and the
turn renders fewer (or zero) memories.

Drives THE DEPLOYED batch_gate_memories with a controllable-cosine embedder
(blended = 0.85·cos + 0.15·truth; truth 0.5 → +0.075). With threshold 0.60:
natural pass needs cos ≥ ~0.618, default rescue band is cos ~0.559–0.618.
"""

import math

import numpy as np
import pytest

from processing.gate_system import MultiStageGateSystem

DIM = 8


class AngleEmbedder:
    """Query → e0; each memory text → [c, sqrt(1−c²), 0…] so cosine(query, mem) = c."""

    def __init__(self, query, cos_map):
        self.query = query
        self.cos_map = cos_map

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **kw):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            if t == self.query:
                out[i, 0] = 1.0
            else:
                c = float(self.cos_map.get(t, 0.0))
                out[i, 0] = c
                out[i, 1] = math.sqrt(max(0.0, 1.0 - c * c))
        return out


class NoRerankMM:
    def get_cross_encoder(self, model_name=""):
        raise RuntimeError("no cross encoder in tests")


QUERY = "what patterns have we seen across my medication conversations"


def _make_gate(cos_map, threshold=0.60):
    embedder = AngleEmbedder(QUERY, cos_map)
    ms = MultiStageGateSystem(NoRerankMM(), retrieval_embedder=embedder,
                              retrieval_threshold=threshold)
    ms.gate_system.embedder = embedder
    ms.gate_system.embed_model = embedder
    ms.embed_model = embedder
    return ms


def _mem(text):
    return {"content": text, "metadata": {"truth_score": 0.5}}


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.setenv("GATE_MIN_MEMORIES", "8")
    monkeypatch.delenv("GATE_FORCED_FLOOR_MARGIN", raising=False)


@pytest.mark.asyncio
async def test_near_misses_rescued_junk_refused():
    # blended: near-miss 0.568 (≥ floor 0.55, < threshold 0.60) — rescued;
    # junk 0.415 (< floor) — refused even though the minimum wants 8.
    cos_map = {"near miss a": 0.58, "near miss b": 0.58}
    cos_map.update({f"junk {i}": 0.40 for i in range(6)})
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem(t) for t in cos_map])
    assert sorted(m["content"] for m in out) == ["near miss a", "near miss b"]
    assert all(m.get("forced_minimum") for m in out)


@pytest.mark.asyncio
async def test_all_far_below_yields_zero():
    # The 08-23 failure shape: nothing passes, nothing is near — render NOTHING
    # instead of the 8 best-of-the-junk.
    cos_map = {f"unrelated {i}": 0.30 for i in range(10)}
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem(t) for t in cos_map])
    assert out == []


@pytest.mark.asyncio
async def test_natural_passes_unaffected_by_floor():
    cos_map = {"on topic": 0.70, "near miss": 0.58, "junk": 0.30}
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem(t) for t in cos_map])
    by_content = {m["content"]: m for m in out}
    assert set(by_content) == {"on topic", "near miss"}
    assert not by_content["on topic"].get("forced_minimum")
    assert by_content["near miss"].get("forced_minimum")


@pytest.mark.asyncio
async def test_margin_env_override_widens_band():
    cos_map = {"borderline junk": 0.40}  # blended 0.415
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem("borderline junk")])
    assert out == []  # default floor 0.55 refuses it

    import os
    os.environ["GATE_FORCED_FLOOR_MARGIN"] = "0.30"
    try:
        ms2 = _make_gate(cos_map)
        out2 = await ms2.batch_gate_memories(QUERY, [_mem("borderline junk")])
        assert [m["content"] for m in out2] == ["borderline junk"]
    finally:
        del os.environ["GATE_FORCED_FLOOR_MARGIN"]


@pytest.mark.asyncio
async def test_margin_one_restores_legacy_backfill(monkeypatch):
    monkeypatch.setenv("GATE_FORCED_FLOOR_MARGIN", "1.0")
    cos_map = {f"unrelated {i}": 0.30 for i in range(10)}
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem(t) for t in cos_map])
    assert len(out) == 8  # pre-floor behavior: fill to GATE_MIN_MEMORIES


@pytest.mark.asyncio
async def test_min_results_cap_still_applies_with_floor():
    cos_map = {f"near miss {i}": 0.58 for i in range(6)}
    ms = _make_gate(cos_map)
    out = await ms.batch_gate_memories(QUERY, [_mem(t) for t in cos_map],
                                       min_results=3)
    assert len(out) == 3  # intent budget caps rescues too
