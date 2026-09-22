"""Grounding and personal-claim corrections compose on the clean delivery body."""
import inspect
from types import SimpleNamespace

import pytest
import gui.handlers as handlers
from utils.personal_claim_provenance import KEY, annotate_personal_claim_memory, clean_personal_claim_receipt


def test_agentic_and_enhanced_sites_use_one_sequential_revision_pipeline():
    for route in (handlers._run_agentic_search, handlers._run_enhanced):
        source = inspect.getsource(route)
        assert "await _apply_delivery_revisions(" in source
        assert "_delivery_body.rstrip() + (_guard_suffix or \"\")" in source or (
            "_delivery_body.rstrip() + (_ag_guard_suffix or \"\")" in source
        )
        assert "_apply_personal_claim_check_for_delivery(\n                ctx, _pre_suffix" not in source


@pytest.mark.asyncio
async def test_second_correction_receives_first_correction(monkeypatch):
    seen = []

    async def grounding(_ctx, body, source_material=""):
        assert body == "Draft"
        return "Grounded draft", ""

    async def personal(_ctx, body):
        seen.append(body)
        return "Grounded and personal-safe draft"

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)

    result = await handlers._apply_delivery_revisions(SimpleNamespace(), "Draft")
    assert seen == ["Grounded draft"]
    assert result == "Grounded and personal-safe draft"


@pytest.mark.asyncio
async def test_log_only_pending_inputs_equal_final_clean_body_after_grounding_revision(monkeypatch):
    async def grounding(_ctx, body, source_material=""):
        assert body == "Draft"
        return "Corrected clean body", ""

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    ctx = SimpleNamespace(personal_claim_mode="log_only", telemetry={})

    final_body = await handlers._apply_delivery_revisions(ctx, "Draft")
    assert final_body == "Corrected clean body"
    assert ctx.personal_claim_pending == final_body


@pytest.mark.asyncio
async def test_log_only_checks_keep_suffix_out_and_receipt_hash_matches_stored_body(monkeypatch):
    async def grounding(ctx, body, source_material=""):
        ctx.grounding_pending = (body, source_material)
        return None, ""

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    body = "The answer body."
    guard_notice = (
        "\n\n> ⚠️ I don't see that on your calendar — nothing "
        "was created. Say \"add it\" and I'll queue a card."
    )
    ctx = SimpleNamespace(personal_claim_mode="log_only", telemetry={})

    final_body = await handlers._apply_delivery_revisions(ctx, body)
    delivered = final_body + guard_notice
    assert ctx.grounding_pending[0] == final_body
    assert ctx.personal_claim_pending == final_body

    receipt = clean_personal_claim_receipt(
        {"status": "checked", "delivery": "unchanged", "insufficient_count": 1},
        response=delivered,
    )
    marked = annotate_personal_claim_memory({"response": delivered, KEY: receipt})
    assert marked["response"].startswith(final_body)
    assert guard_notice.strip() in marked["response"]
    assert "[Personal-claim check:" in marked["response"]
