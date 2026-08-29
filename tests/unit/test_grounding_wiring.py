"""Factual-grounding floor — handlers wiring (gui.handlers._apply_grounding_check).

Drives THE deployed helper with fakes (test_deferred_request_clarify.py style):
telemetry accumulation, elevated-tone wording, confidence gating, fail-open on
verifier failure, enabled/pre-filter short-circuits, and source-level pins that
both the enhanced and agentic runners append the suffix to final_output (the
storage copy must carry the correction — a false endorsement must not become
retrievable corpus ground truth).
"""
import inspect
import json
from types import SimpleNamespace

import pytest

import gui.handlers as handlers
from core.grounding_check import GroundingVerdict


FIRING_RESPONSE = (
    "The old \"refrigerator mother\" frame put the cause on the mother — "
    "which, given what you lived, lands closer to truth than the other version. "
    "It deserves a longer look."
)
NON_FIRING_RESPONSE = (
    "I hear you, and I'm right here with you. That took real courage to send, "
    "and you don't have to carry the rest of today alone at all."
)

VERDICT_JSON = json.dumps({
    "false_claim_present": True,
    "claim": "refrigerator mother theory lands closer to truth",
    "why_false": "The theory was scientifically discredited decades ago.",
    "confidence": 0.95,
    "correction": "The refrigerator mother theory was discredited long ago; autism is neurodevelopmental, not caused by parenting.",
})


class _StubModelManager:
    def __init__(self, raw=VERDICT_JSON, exc=None):
        self.raw = raw
        self.exc = exc
        self.calls = 0

    async def generate_once(self, prompt, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.raw


def _ctx(response_tone="CrisisLevel.CONVERSATIONAL", mm=None, user_text="tell me about it"):
    return SimpleNamespace(
        user_text=user_text,
        telemetry={},
        orchestrator=SimpleNamespace(model_manager=mm if mm is not None else _StubModelManager()),
        raw_context={"tone_level": response_tone},
    )


@pytest.mark.asyncio
async def test_firing_text_high_confidence_appends_and_records():
    ctx = _ctx()
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix.startswith("\n\n> ⚠️ Correction:")
    assert "discredited" in suffix
    assert ctx.telemetry["grounding_prefilter_fired"] is True
    assert ctx.telemetry["grounding_verifier_fired"] is True
    assert ctx.telemetry["grounding_flagged"] is True
    assert ctx.telemetry["grounding_confidence"] == 0.95
    assert ctx.telemetry["grounding_corrected"] is True


@pytest.mark.asyncio
async def test_elevated_tone_gets_gentle_wording():
    ctx = _ctx(response_tone="CrisisLevel.MEDIUM")
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert "gently set straight" in suffix
    assert "Correction:" not in suffix


@pytest.mark.asyncio
async def test_concern_tone_counts_as_elevated():
    # The live failure happened at CONCERN (LIGHT SUPPORT tier).
    ctx = _ctx(response_tone="CrisisLevel.CONCERN")
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert "gently set straight" in suffix


@pytest.mark.asyncio
async def test_low_confidence_no_action():
    low = json.loads(VERDICT_JSON)
    low["confidence"] = 0.5
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(low)))
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert ctx.telemetry.get("grounding_flagged") is True
    assert "grounding_corrected" not in ctx.telemetry


@pytest.mark.asyncio
async def test_verifier_failure_fails_open():
    ctx = _ctx(mm=_StubModelManager(exc=RuntimeError("boom")))
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert ctx.telemetry.get("grounding_verifier_fired") is True
    assert "grounding_corrected" not in ctx.telemetry


@pytest.mark.asyncio
async def test_no_flag_verdict_no_action():
    clean = {"false_claim_present": False, "claim": "", "why_false": "",
             "confidence": 0.9, "correction": ""}
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(clean)))
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert ctx.telemetry["grounding_flagged"] is False


@pytest.mark.asyncio
async def test_disabled_short_circuits_before_verifier(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_CHECK_ENABLED", False)
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert mm.calls == 0
    assert ctx.telemetry == {}


@pytest.mark.asyncio
async def test_non_firing_text_never_calls_verifier():
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    suffix = await handlers._apply_grounding_check(ctx, NON_FIRING_RESPONSE)
    assert suffix == ""
    assert mm.calls == 0
    assert "grounding_prefilter_fired" not in ctx.telemetry


@pytest.mark.asyncio
async def test_short_response_skipped():
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    assert await handlers._apply_grounding_check(ctx, "myth.") == ""
    assert mm.calls == 0


@pytest.mark.asyncio
async def test_missing_model_manager_fails_open():
    ctx = _ctx()
    ctx.orchestrator = SimpleNamespace()  # no model_manager attr
    suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert ctx.telemetry.get("grounding_prefilter_fired") is True


# ---------------------------------------------------------------------------
# Source-level wiring pins (house precedent: deferred-request tests) — both
# runner paths must call the helper and append the suffix to final_output.
# ---------------------------------------------------------------------------

def test_enhanced_path_wired():
    src = inspect.getsource(handlers._run_enhanced)
    assert "_apply_grounding_check" in src
    assert "_gc_suffix" in src
    assert 'final_output = (final_output or "").rstrip() + _gc_suffix' in src


def test_agentic_path_wired():
    src = inspect.getsource(handlers._run_agentic_search)
    assert "_apply_grounding_check" in src
    assert "_ag_gc_suffix" in src
    assert 'final_output = (final_output or "").rstrip() + _ag_gc_suffix' in src


def test_correction_survives_storage_sanitize():
    # The stored copy must keep the correction: _sanitize_response_text is the
    # storage boundary transform applied to final_output.
    from core.grounding_check import build_grounding_correction
    body = "Here is the answer to your question about the theory."
    suffix = build_grounding_correction("The theory was discredited long ago.")
    sanitized = handlers._sanitize_response_text(body + suffix)
    assert "> ⚠️" in sanitized
    assert "discredited" in sanitized
