"""Figure check wiring in the delivery-revision pipeline (lane FIGB, 2026-10-25, BC-103).

Drives THE deployed ``handlers._apply_delivery_revisions``: the reply the user would
have received on the incident turn must carry the corrected elapsed time, grounding
and personal-claim must audit the corrected body, and a checker failure is fail-open.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import gui.handlers as handlers
from config import app_config

BAD = "Given how wrecked you were feeling an hour ago, take care."


def _ctx():
    stamp = (datetime.now() - timedelta(minutes=1)).isoformat()  # clock-safe: the check compares elapsed seconds, never calendar days
    return SimpleNamespace(
        user_text="still rough tonight", telemetry={}, personal_claim_mode="log_only",
        raw_context={"recent_conversations": [
            {"query": "rough night", "response": "I'm sorry you feel wrecked, Rowan.", "timestamp": stamp}]},
        history=[], orchestrator=None,
    )


@pytest.fixture
def seen(monkeypatch):
    got = {}

    async def grounding(ctx, body, source_material=""):
        got["grounding"] = body
        return None, ""

    async def personal(ctx, body):
        got["personal"] = body
        return None

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)
    monkeypatch.setattr(handlers.checker_findings, "record_finding",
                        lambda checker, **kw: got.setdefault("findings", []).append((checker, kw)) or True)
    return got


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [BAD, "    Given how wrecked you were\n    feeling an hour ago, take care."])
async def test_incident_reply_is_corrected_and_audited_downstream(seen, body):
    ctx = _ctx()
    final = await handlers._apply_delivery_revisions(ctx, body)
    assert "a minute ago" in final and "an hour ago" not in final
    assert seen["grounding"] == final and seen["personal"] == final
    assert ctx.telemetry["figure_check_status"] == "ok"
    assert ctx.telemetry["figure_check_corrected"] == 1
    assert ctx.telemetry["figure_check_findings"] == {"elapsed_precise": 1}
    assert any(c == "figure_check" and kw["verdict"] == "elapsed_precise" for c, kw in seen["findings"])


@pytest.mark.asyncio
async def test_receipt_carries_counts_and_labels_only(seen):
    ctx = _ctx()
    await handlers._apply_delivery_revisions(ctx, BAD)
    text = repr({k: v for k, v in ctx.telemetry.items() if k.startswith("figure_check")})
    assert "wrecked" not in text and "hour" not in text


@pytest.mark.asyncio
async def test_checker_exception_leaves_reply_unchanged_with_error_status(seen, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("core.figure_check.check_reply", boom)
    ctx = _ctx()
    final = await handlers._apply_delivery_revisions(ctx, BAD)
    assert final == BAD and seen["grounding"] == BAD
    assert ctx.telemetry["figure_check_status"] == "error"


@pytest.mark.asyncio
async def test_toggle_same_instance_before_after(seen, monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(app_config, "FIGURE_CHECK_ENABLED", False)
    assert await handlers._apply_delivery_revisions(ctx, BAD) == BAD
    assert ctx.telemetry["figure_check_status"] == "skipped:disabled"
    monkeypatch.setattr(app_config, "FIGURE_CHECK_ENABLED", True)
    assert "a minute ago" in await handlers._apply_delivery_revisions(ctx, BAD)
    assert ctx.telemetry["figure_check_status"] == "ok"


@pytest.mark.asyncio
async def test_log_only_class_is_recorded_not_edited(seen, monkeypatch):
    monkeypatch.setattr(app_config, "FIGURE_CHECK_CORRECT_CLASSES", ())
    monkeypatch.setattr(app_config, "FIGURE_CHECK_LOG_CLASSES", ("elapsed_precise",))
    ctx = _ctx()
    assert await handlers._apply_delivery_revisions(ctx, BAD) == BAD
    assert seen["findings"][0][1]["mode"] == "log_only"


@pytest.mark.asyncio
async def test_world_fact_and_derived_figures_log_nothing(seen):
    ctx = _ctx()
    body = "Withdrawal typically resolves within 7 days. Your note says you felt wrecked a minute ago."
    await handlers._apply_delivery_revisions(ctx, body)
    assert "findings" not in seen


def test_config_defaults_match_the_yaml_section():
    assert app_config.FIGURE_CHECK_ENABLED is True
    assert set(app_config.FIGURE_CHECK_CORRECT_CLASSES) == {"elapsed_precise", "weekday_date"}
    assert set(app_config.FIGURE_CHECK_LOG_CLASSES) == {
        "elapsed_vague", "elapsed_unconfirmed", "copied_mismatch", "unsupported",
        "cross_owner_date", "unsupported_detail", "stale_day_count"}   # last three: log-only siblings, 2026-10-27
    assert app_config.FIGURE_CHECK_MIN_REFERENT_OVERLAP == 1
