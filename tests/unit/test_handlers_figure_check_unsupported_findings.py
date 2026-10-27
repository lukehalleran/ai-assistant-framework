"""Unsupported figures are persisted with their figure text and clause (lane C1, wave 2).

Incident 2026-10-26 (oct26b M2 sheets): the figure check's `unsupported` class was
counted in the receipt but never written to checker_findings.jsonl, so B1-B6 could
not be labelled. Class: BC-103 (figure provenance), BC-58 (log class not wired).

Drives THE deployed ``handlers._apply_delivery_revisions``.
"""
from types import SimpleNamespace

import pytest

import gui.handlers as handlers
from config import app_config

BODY = "You have been studying for 3 hours straight tonight, Rowan."


def _ctx():
    return SimpleNamespace(
        user_text="still rough tonight", telemetry={}, personal_claim_mode="log_only",
        raw_context={"recent_conversations": [{"query": "rough night", "response": "I hear you."}]},
        history=[], orchestrator=None,
    )


@pytest.fixture
def seen(monkeypatch):
    got = {}

    async def grounding(ctx, body, source_material=""):
        return None, ""

    async def personal(ctx, body):
        return None

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)
    monkeypatch.setattr(handlers.checker_findings, "record_finding",
                        lambda checker, **kw: got.setdefault("findings", []).append((checker, kw)) or True)
    return got


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [BODY, "    You have been studying\n    for 3 hours straight\n    tonight, Rowan."])
async def test_unsupported_figure_is_recorded_with_span_and_clause(seen, body):
    ctx = _ctx()
    final = await handlers._apply_delivery_revisions(ctx, body)
    assert final == body  # log-only: never edited
    rows = [kw for c, kw in seen.get("findings", []) if c == "figure_check" and kw["verdict"] == "unsupported"]
    assert rows, ctx.telemetry
    row = rows[0]
    assert row["claim_span"].startswith("3")
    assert row["mode"] == "log_only"
    assert "studying" in " ".join(row["clause"].split())


@pytest.mark.asyncio
async def test_toggle_same_instance_unsupported_class(seen, monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(app_config, "FIGURE_CHECK_LOG_CLASSES", ("elapsed_vague",))
    await handlers._apply_delivery_revisions(ctx, BODY)
    assert "findings" not in seen
    monkeypatch.setattr(app_config, "FIGURE_CHECK_LOG_CLASSES", ("elapsed_vague", "unsupported"))
    await handlers._apply_delivery_revisions(ctx, BODY)
    assert any(kw["verdict"] == "unsupported" for _, kw in seen["findings"])


def test_deployed_default_logs_unsupported():
    assert "unsupported" in app_config.FIGURE_CHECK_LOG_CLASSES
