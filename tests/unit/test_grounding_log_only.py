"""
Grounding verifier LOG-ONLY mode (2026-09-04).

Telemetry over the window: 42 verifier fires -> 27 flags -> 25 shipped
corrections, >=9 documented false, 0 documented true — the same evidence
class that made the post-answer review gate LOG-ONLY on 2026-08-28. Mirrors
that fix: config `grounding_check.mode` (log_only | correct, default
log_only) gates whether gui.handlers._apply_grounding_check is allowed to
touch the shown/stored response. LOG-ONLY still runs the full
prefilter->verifier->demotion pipeline (so precision can be measured) but
never integrates or appends a correction.

Reuses the fakes from tests/unit/test_grounding_wiring.py (same house style
as test_deferred_request_clarify.py: fakes, not getsource pins, drive THE
deployed helper).
"""

import pytest

import gui.handlers as handlers
from tests.unit.test_grounding_wiring import (
    FIRING_RESPONSE,
    VERDICT_JSON,
    _ctx,
    _StubModelManager,
)


@pytest.fixture
def log_only_mode(monkeypatch):
    """Default as of 2026-09-04 — pin explicitly so the test is robust to a
    future config default change."""
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_MODE", "log_only")


@pytest.fixture
def correct_mode(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_MODE", "correct")


def test_default_config_mode_is_log_only():
    """The default (no override) config value must resolve to log_only —
    this is what makes it safe to omit `mode` from config.yaml/env."""
    import config.app_config as ac
    assert ac.GROUNDING_MODE == "log_only"


@pytest.mark.asyncio
async def test_log_only_returns_input_unchanged_with_empty_suffix(log_only_mode):
    ctx = _ctx()
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)

    assert revised == FIRING_RESPONSE
    assert suffix == ""


@pytest.mark.asyncio
async def test_log_only_records_flagged_verdict_and_mode_telemetry(log_only_mode):
    ctx = _ctx()
    await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)

    assert ctx.telemetry["grounding_prefilter_fired"] is True
    assert ctx.telemetry["grounding_verifier_fired"] is True
    assert ctx.telemetry["grounding_flagged"] is True
    assert ctx.telemetry["grounding_confidence"] == 0.95
    assert ctx.telemetry["grounding_mode"] == "log_only"
    assert "discredited" in ctx.telemetry["grounding_verdict"]
    assert len(ctx.telemetry["grounding_verdict"]) <= 300
    # Never shipped in log-only mode.
    assert "grounding_corrected" not in ctx.telemetry
    assert "grounding_integrated" not in ctx.telemetry


@pytest.mark.asyncio
async def test_log_only_never_calls_integrator(log_only_mode):
    mm = _StubModelManager(responses=[VERDICT_JSON, "should never be requested"])
    ctx = _ctx(mm=mm)
    await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    # Only the verifier call — no integrator round-trip in log-only mode.
    assert mm.calls == 1


@pytest.mark.asyncio
async def test_log_only_logs_warning(log_only_mode, caplog):
    import logging
    ctx = _ctx()
    with caplog.at_level(logging.WARNING, logger="gradio_gui"):
        await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert any("[Grounding] LOG-ONLY flagged conf=" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_log_only_skips_below_confidence_threshold(log_only_mode):
    """Below-threshold verdicts never reach the mode branch at all — same
    no-action contract as mode=='correct'."""
    import json
    low = json.loads(VERDICT_JSON)
    low["confidence"] = 0.5
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(low)))
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert "grounding_mode" not in ctx.telemetry


@pytest.mark.asyncio
async def test_correct_mode_still_ships_suffix_correction(correct_mode, monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_INTEGRATE_ENABLED", False)
    ctx = _ctx()
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)

    assert revised is None
    assert suffix.startswith("\n\n> ⚠️ Correction:")
    assert ctx.telemetry["grounding_corrected"] is True
    assert ctx.telemetry["grounding_mode"] == "correct"


@pytest.mark.asyncio
async def test_correct_mode_still_ships_integration():
    """Default GROUNDING_INTEGRATE_ENABLED is True; only mode is pinned here
    to prove 'correct' is the pre-09-04 behavior end-to-end."""
    import json
    revised_text = (
        "The old \"refrigerator mother\" frame put the cause on the mother — "
        "correction: that theory was discredited long ago; autism is "
        "neurodevelopmental, not caused by parenting. It deserves a longer look."
    )
    mm = _StubModelManager(responses=[VERDICT_JSON, revised_text])
    ctx = _ctx(mm=mm)
    import config.app_config as ac
    from unittest.mock import patch
    with patch.object(ac, "GROUNDING_MODE", "correct"):
        revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)

    assert revised == revised_text
    assert suffix == ""
    assert ctx.telemetry["grounding_integrated"] is True
    assert ctx.telemetry["grounding_corrected"] is True
