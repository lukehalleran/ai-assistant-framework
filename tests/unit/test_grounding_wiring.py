"""Factual-grounding floor — handlers wiring (gui.handlers._apply_grounding_check).

Drives THE deployed helper with fakes (test_deferred_request_clarify.py style):
telemetry accumulation, elevated-tone wording, confidence gating, fail-open on
verifier failure, enabled/pre-filter short-circuits, and source-level pins that
both the enhanced and agentic runners handle BOTH outcomes of the helper's
(revised, suffix) contract (2026-08-29): integration replaces display AND
final_output wholesale; suffix append remains the fallback. The storage copy
must carry the correction either way — a false endorsement must not become
retrievable corpus ground truth.
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

# In-bounds revision of FIRING_RESPONSE (length ratio must sit inside the
# integrator's [0.75, 1.30] guard) with the correction woven into the prose.
REVISED_RESPONSE = (
    "The old \"refrigerator mother\" frame put the cause on the mother — "
    "correction: that theory was discredited long ago; autism is "
    "neurodevelopmental, not caused by parenting. It deserves a longer look."
)


class _StubModelManager:
    """Scripted responses: consumed one per generate_once call, last repeats."""

    def __init__(self, raw=VERDICT_JSON, exc=None, responses=None):
        self.responses = list(responses) if responses is not None else [raw]
        self.exc = exc
        self.calls = 0
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        idx = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[idx]


def _ctx(response_tone="CrisisLevel.CONVERSATIONAL", mm=None, user_text="tell me about it"):
    return SimpleNamespace(
        user_text=user_text,
        telemetry={},
        orchestrator=SimpleNamespace(model_manager=mm if mm is not None else _StubModelManager()),
        raw_context={"tone_level": response_tone},
    )


@pytest.fixture
def no_integrate(monkeypatch):
    """Suffix-contract tests: pin the integrator off so behavior is the
    pre-2026-08-29 append path. Also pins mode=="correct" (2026-09-04
    default is log_only) — these tests assert a SHIPPED correction."""
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_INTEGRATE_ENABLED", False)
    monkeypatch.setattr(ac, "GROUNDING_MODE", "correct")


@pytest.fixture
def correct_mode(monkeypatch):
    """Pin mode=='correct' (2026-09-04 default is log_only) for tests that
    assert a SHIPPED correction/integration."""
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_MODE", "correct")


@pytest.mark.asyncio
async def test_firing_text_high_confidence_delivers_spliced_fallback(no_integrate):
    # A05b-1: integrator disabled -> the ONE integrated fallback ships
    # through the revised path, never draft-plus-suffix (BC-45/BC-71).
    ctx = _ctx()
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert revised is not None
    assert "> ⚠️ Correction:" not in revised  # retired suffix marker
    assert "discredited" in revised
    assert ctx.telemetry["grounding_prefilter_fired"] is True
    assert ctx.telemetry["grounding_verifier_fired"] is True
    assert ctx.telemetry["grounding_flagged"] is True
    assert ctx.telemetry["grounding_confidence"] == 0.95
    assert ctx.telemetry["grounding_corrected"] is True
    assert ctx.telemetry["grounding_status"] == "fallback"
    assert ctx.telemetry["grounding_fallback"] == "spliced:claim_located_overlap"


@pytest.mark.asyncio
async def test_elevated_tone_gets_gentle_wording(no_integrate):
    ctx = _ctx(response_tone="CrisisLevel.MEDIUM")
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert "gently set straight" in revised
    assert "Correction:" not in revised


@pytest.mark.asyncio
async def test_concern_tone_counts_as_elevated(no_integrate):
    # The live failure happened at CONCERN (LIGHT SUPPORT tier).
    ctx = _ctx(response_tone="CrisisLevel.CONCERN")
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert "gently set straight" in revised


@pytest.mark.asyncio
async def test_low_confidence_no_action():
    low = json.loads(VERDICT_JSON)
    low["confidence"] = 0.5
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(low)))
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert ctx.telemetry.get("grounding_flagged") is True
    assert "grounding_corrected" not in ctx.telemetry


@pytest.mark.asyncio
async def test_verifier_failure_fails_open():
    ctx = _ctx(mm=_StubModelManager(exc=RuntimeError("boom")))
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert ctx.telemetry.get("grounding_verifier_fired") is True
    assert "grounding_corrected" not in ctx.telemetry


@pytest.mark.asyncio
async def test_no_flag_verdict_no_action():
    clean = {"false_claim_present": False, "claim": "", "why_false": "",
             "confidence": 0.9, "correction": ""}
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(clean)))
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert ctx.telemetry["grounding_flagged"] is False


@pytest.mark.asyncio
async def test_disabled_short_circuits_before_verifier(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "GROUNDING_CHECK_ENABLED", False)
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert mm.calls == 0
    # 2026-09-15 (audit F6 / BC-70): a check that never ran records an explicit
    # SKIPPED receipt with its reason -- never nothing, never "complete".
    assert ctx.telemetry["grounding_status"] == "skipped"
    assert ctx.telemetry["grounding_skip_reason"] == "disabled"
    assert "grounding_verifier_fired" not in ctx.telemetry


@pytest.mark.asyncio
async def test_non_firing_text_never_calls_verifier():
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    revised, suffix = await handlers._apply_grounding_check(ctx, NON_FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert mm.calls == 0
    assert "grounding_prefilter_fired" not in ctx.telemetry


@pytest.mark.asyncio
async def test_short_response_skipped():
    mm = _StubModelManager()
    ctx = _ctx(mm=mm)
    assert await handlers._apply_grounding_check(ctx, "myth.") == (None, "")
    assert mm.calls == 0


@pytest.mark.asyncio
async def test_missing_model_manager_fails_open():
    ctx = _ctx()
    ctx.orchestrator = SimpleNamespace()  # no model_manager attr
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert ctx.telemetry.get("grounding_prefilter_fired") is True


# ---------------------------------------------------------------------------
# Integration path (2026-08-29): correction woven INTO the response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_integration_returns_revised_text_no_suffix(correct_mode):
    mm = _StubModelManager(responses=[VERDICT_JSON, REVISED_RESPONSE])
    ctx = _ctx(mm=mm)
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert revised == REVISED_RESPONSE
    assert suffix == ""
    assert mm.calls == 2  # verifier + integrator
    assert ctx.telemetry["grounding_corrected"] is True
    assert ctx.telemetry["grounding_integrated"] is True


@pytest.mark.asyncio
async def test_integration_failure_falls_back_to_integrated_fallback(correct_mode):
    # Integrator returns something wildly out of length bounds → guard trips
    # → the integrated fallback (spliced/standalone) ships, never the old
    # draft-plus-suffix shape.
    mm = _StubModelManager(responses=[VERDICT_JSON, "Nope."])
    ctx = _ctx(mm=mm)
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert revised is not None
    assert "> ⚠️ Correction:" not in revised
    assert "discredited" in revised
    assert "grounding_integrated" not in ctx.telemetry
    assert ctx.telemetry["grounding_status"] == "fallback"
    assert ctx.telemetry["grounding_fallback"] == "spliced:claim_located_overlap"


@pytest.mark.asyncio
async def test_builder_none_ships_draft_unchanged(no_integrate, monkeypatch):
    """A verdict clears _apply_grounding_check's own gate (flagged, non-empty
    correction, confidence above threshold) but build_integrated_fallback
    finds nothing substantive to deliver. Same contract as a verifier-failure
    verdict: ship the draft unmodified, no marker, never the retired suffix.
    (build_integrated_fallback is faked here because a verdict that clears
    verify_grounding's own advice/PSA/date demotions always yields a
    non-None result from the real builder — see A05b-1 packet.)"""
    import core.grounding_check as gc
    monkeypatch.setattr(gc, "build_integrated_fallback", lambda *a, **k: None)
    ctx = _ctx()
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert (revised, suffix) == (None, "")
    assert ctx.telemetry["grounding_status"] == "complete"  # set by verify_grounding
    assert ctx.telemetry["grounding_fallback"] == "none"
    assert "grounding_corrected" not in ctx.telemetry


@pytest.mark.asyncio
async def test_claim_not_located_delivers_standalone_without_draft_prose(no_integrate):
    """Claim absent from the draft -> a standalone corrective reply, never
    the flawed prose. A distinctive draft-only phrase must be absent from
    the delivered text, and the receipt's grounding_fallback stays a
    constant label (no claim/correction text)."""
    unrelated = json.loads(VERDICT_JSON)
    unrelated["claim"] = "the meeting moved to next Tuesday at noon"
    unrelated["correction"] = "The meeting is still on Thursday at 3pm."
    ctx = _ctx(mm=_StubModelManager(raw=json.dumps(unrelated)))
    revised, suffix = await handlers._apply_grounding_check(ctx, FIRING_RESPONSE)
    assert suffix == ""
    assert revised is not None
    assert "refrigerator mother" not in revised  # distinctive draft token
    assert "Thursday at 3pm" in revised
    assert ctx.telemetry["grounding_status"] == "fallback"
    assert ctx.telemetry["grounding_fallback"] == "standalone:claim_not_located"


@pytest.mark.asyncio
async def test_fallback_reattaches_trailing_proposal_card(no_integrate):
    """A trailing action-proposal card survives verbatim in the delivered
    fallback (A05a contract) — the card is authoritative backend state, not
    prose, and A05b-1 must not special-case it at the call sites."""
    card = "\n\n---\n**calendar_create**\nSummary: Team sync\n"
    ctx = _ctx()
    revised, suffix = await handlers._apply_grounding_check(
        ctx, FIRING_RESPONSE + card)
    assert suffix == ""
    assert revised is not None
    assert revised.endswith(card)


@pytest.mark.asyncio
async def test_source_material_reaches_verifier_prompt():
    mm = _StubModelManager(responses=[json.dumps({
        "false_claim_present": False, "claim": "", "why_false": "",
        "confidence": 0.9, "correction": ""})])
    ctx = _ctx(mm=mm)
    await handlers._apply_grounding_check(
        ctx, FIRING_RESPONSE, source_material="ABC 1234 Fall 2026 syllabus: HW1 due 9/13")
    assert mm.calls == 1
    assert "ABC 1234 Fall 2026 syllabus" in mm.prompts[0]
    assert "AUTHORITATIVE" in mm.prompts[0]


# ---------------------------------------------------------------------------
# Source-level wiring pins (house precedent: deferred-request tests) — both
# runner paths must call the helper, handle the revised-text outcome by
# REPLACING final_output, and keep the suffix append as fallback. The agentic
# path must feed the loop's tool-round results to the verifier as source
# material (the "Fall 2026" conf-0.9 misfire was source-blindness).
# ---------------------------------------------------------------------------

def test_enhanced_path_wired():
    src = inspect.getsource(handlers._run_enhanced)
    assert "_apply_grounding_check" in src
    assert "_gc_revised, _gc_suffix" in src
    assert "final_output = _gc_revised" in src
    # A05b-1: the suffix-append branch is retired — no dead code left behind.
    assert '.rstrip() + _gc_suffix' not in src


def test_agentic_path_wired():
    src = inspect.getsource(handlers._run_agentic_search)
    assert "_apply_grounding_check" in src
    assert "_ag_gc_revised, _ag_gc_suffix" in src
    assert "final_output = _ag_gc_revised" in src
    assert "source_material=_ag_source" in src
    # A05b-1: the suffix-append branch is retired — no dead code left behind.
    assert '.rstrip() + _ag_gc_suffix' not in src


def test_integrated_fallback_survives_storage_sanitize():
    # The stored copy must keep the correction: _sanitize_response_text is the
    # storage boundary transform applied to final_output. Drive the ACTUAL
    # live fallback (build_integrated_fallback), not the retired suffix
    # builder — one spliced result and one standalone result.
    from core.grounding_check import build_integrated_fallback

    draft = "Here is the answer to your question about the theory."
    verdict = GroundingVerdict(
        claim="the theory holds up",
        why_false="It is well established to be false.",
        correction="The theory was discredited long ago.",
        confidence=0.9,
        false_claim_present=True,
    )
    # Negative control: the original draft does NOT contain the correction
    # text, so its presence after sanitizing must come from the fallback.
    assert "discredited" not in draft

    spliced = build_integrated_fallback(
        "I hear you. The theory holds up under scrutiny. Take care.", verdict)
    standalone = build_integrated_fallback(draft, verdict)
    assert spliced is not None and standalone is not None
    assert spliced.kind == "spliced" and standalone.kind == "standalone"

    sanitized_spliced = handlers._sanitize_response_text(spliced.text)
    sanitized_standalone = handlers._sanitize_response_text(standalone.text)
    assert "discredited" in sanitized_spliced
    assert "Correction:" in sanitized_spliced  # spliced lead survives
    assert "discredited" in sanitized_standalone
    # standalone lead survives
    assert "found something that needs correcting" in sanitized_standalone
