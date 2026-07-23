"""
tests/integration/test_golden_distress_replay.py

ACCEPTANCE TEST for the 2026-07 tone-flatline incident.

Replays the frozen golden distress transcript through the real tone→escalation
sub-pipeline (tone detector → CrisisLevel→ToneLevel mapping → EscalationTracker →
strategy selection) and asserts the INTENDED trajectory:

  (a) tone never flatlines at CONVERSATIONAL after turn 2, despite short messages
  (b) consecutive_distress_count accumulates across the session
  (c) GROUNDING_PRESENCE / QUIET_COMPANIONSHIP engages by mid-session
  (d) while grounding is active, the strategy forbids excavating questions

This goes RED if EITHER root cause is reintroduced:
  - the <8-word tone fast-path bypass (tones would flatline → a,b,c fail), or
  - the CrisisLevel.value vs ToneLevel.from_string dead-wire (tracker fed
    CONVERSATIONAL every turn → b,c fail).

Determinism: the tone path uses the sentence-transformer embedder, whose output
is deterministic for a fixed model. The test is embedder-gated (skips if the
model can't load) exactly like the revived semantic tone tests.
"""

import json
import os

import pytest

from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.context_pipeline import ToneLevel


_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_distress_session.json")


def _embedder_available() -> bool:
    try:
        from utils.tone_detector import _get_embedder
        return _get_embedder(None) is not None
    except Exception:
        return False


def _load_fixture():
    with open(os.path.abspath(_FIXTURE), "r", encoding="utf-8") as f:
        return json.load(f)


async def _replay(turns, distress_threshold):
    """
    Faithful replay of the pipeline's tone→escalation wiring:
      context_pipeline._detect_tone: analyze_emotional_context(..., previous_tone=_last_tone_level)
                                     then ToneLevel.from_string(crisis_level.value)
      orchestrator:                  escalation_tracker.update(tone_level, msg, need_type)
    """
    from utils.emotional_context import analyze_emotional_context
    from utils.query_checker import _is_heavy_topic_heuristic

    tracker = EscalationTracker(distress_threshold=distress_threshold)
    prev = None
    history = []
    rows = []
    for msg in turns:
        ctx = await analyze_emotional_context(
            msg, conversation_history=history[-3:], model_manager=None, previous_tone=prev
        )
        crisis = ctx.crisis_level
        tone_level = ToneLevel.from_string(crisis.value)  # exactly what the pipeline passes
        prev = crisis
        strategy = tracker.update(tone_level, msg, need_type=None)
        rows.append({
            "msg": msg,
            "crisis": crisis,
            "tone_level": tone_level,
            "strategy": strategy,
            "distress": tracker.consecutive_distress_count,
            "elevated": tracker.consecutive_elevated_count,
            "instructions": tracker.get_strategy_instructions(),
        })
        history.append({"query": msg, "is_heavy_topic": _is_heavy_topic_heuristic(msg)})
    return rows


_GROUNDING = {ResponseStrategy.GROUNDING_PRESENCE, ResponseStrategy.QUIET_COMPANIONSHIP}


@pytest.mark.skipif(not _embedder_available(),
                    reason="ALLOW_SKIP: env-gate — sentence-transformer embedder not loadable")
async def test_golden_distress_trajectory():
    fx = _load_fixture()
    turns = fx["turns"]
    threshold = fx["distress_threshold"]
    exp = fx["expected_trajectory"]
    rows = await _replay(turns, threshold)

    # Human-readable trace on failure.
    trace = "\n".join(
        f"T{i+1} crisis={r['crisis'].value:16s} tone={r['tone_level'].name:14s} "
        f"distress={r['distress']} elev={r['elevated']} strat={r['strategy'].value}"
        for i, r in enumerate(rows)
    )

    # (a) No flatline: from the configured turn onward, tone is never CONVERSATIONAL.
    start = exp["no_flatline_from_turn"] - 1  # 1-indexed → 0-indexed
    for i in range(start, len(rows)):
        assert rows[i]["tone_level"] != ToneLevel.CONVERSATIONAL, (
            f"Turn {i+1} flatlined to CONVERSATIONAL — fast-path regression?\n{trace}"
        )
        assert rows[i]["crisis"].name != "CONVERSATIONAL", f"\n{trace}"

    # (b) Distress accumulates: monotonically non-decreasing, and reaches the
    #     threshold by the configured turn.
    distress_series = [r["distress"] for r in rows]
    assert distress_series == sorted(distress_series), (
        f"consecutive_distress_count not monotonic — a turn reset it?\n{trace}"
    )
    by_turn = exp["distress_threshold_reached_by_turn"]
    assert rows[by_turn - 1]["distress"] >= threshold, (
        f"distress {rows[by_turn-1]['distress']} < threshold {threshold} by turn {by_turn}\n{trace}"
    )
    assert distress_series[-1] >= threshold, f"\n{trace}"

    # (c) Grounding engages by mid/late session.
    first_grounding = next(
        (i + 1 for i, r in enumerate(rows) if r["strategy"] in _GROUNDING), None
    )
    assert first_grounding is not None, (
        f"GROUNDING/QUIET never engaged — escalation tracker starved (dead-wire regression?)\n{trace}"
    )
    assert first_grounding <= exp["grounding_active_by_turn"], (
        f"Grounding first engaged at turn {first_grounding}, "
        f"expected by turn {exp['grounding_active_by_turn']}\n{trace}"
    )

    # (d) Whenever grounding is active, the injected instructions forbid excavation.
    for i, r in enumerate(rows):
        if r["strategy"] in _GROUNDING:
            instr = r["instructions"].lower()
            assert "probing question" in instr, (
                f"Turn {i+1} grounding active but excavation not forbidden\n{trace}"
            )


@pytest.mark.skipif(not _embedder_available(),
                    reason="ALLOW_SKIP: env-gate — sentence-transformer embedder not loadable")
async def test_golden_first_two_turns_seed_distress():
    """Guard the bootstrap: the opening turns must register as distress so the
    session can accumulate — this is where the old fast-path silently reset."""
    fx = _load_fixture()
    rows = await _replay(fx["turns"], fx["distress_threshold"])
    assert rows[0]["tone_level"] in {ToneLevel.CONCERN, ToneLevel.ELEVATED, ToneLevel.CRISIS}
    assert rows[0]["distress"] >= 1
