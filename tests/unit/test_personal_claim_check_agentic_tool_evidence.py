"""Agentic tool results reach the personal-claim evidence, under the same caps.

Incident date: 2026-10-27 (wave-2 follow-up: wave 2 added this turn's PRE-GATHERED web pages as
``source`` evidence, but the agentic loop's own tool results -- ``ctx.citation_sources_text`` -- were
never passed, so a claim grounded in a fetched page came back `insufficient`).
Class: BC-84 (personal-claim boundary precision).

Drives THE deployed ``gui.handlers._apply_personal_claim_check`` with a scripted auditor model that
records its prompt; blocks asserted clean and wrapped/indented.
"""
import json
from types import SimpleNamespace

import pytest

import config.app_config as config
from core.personal_claim_check import SOURCE_MAX_PAGES, build_personal_evidence
from gui import handlers

BLOCK = "[WEB_1] **Northwind Bakery hours** (https://example.org/nw)\nNorthwind Bakery opens at nine on weekdays."
WRAPPED = "\n".join("    " + ln for ln in BLOCK.split("\n"))


class _Model:
    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps({"claims": []})


def _ctx(model, tool_text):
    return SimpleNamespace(
        user_text="When does the bakery open?", telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=model),
        raw_context={}, history=(), personal_claim_mode="log_only",
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
        turn_attachments=[], citation_sources_text=tool_text,
    )


def _evidence_of(model):
    return json.loads(model.prompts[0].split("EVIDENCE:\n", 1)[1].split("\n\nDRAFT:\n", 1)[0])


@pytest.mark.parametrize("tool_text", [BLOCK, WRAPPED])
async def test_handler_passes_agentic_tool_results_as_source_evidence(monkeypatch, tool_text):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    model = _Model()
    await handlers._apply_personal_claim_check(
        _ctx(model, tool_text), "The bakery opens at nine.", mode="log_only")
    ev = _evidence_of(model)
    assert [r["role"] for r in ev] == ["user", "source"]
    assert "opens at nine" in ev[1]["text"]


async def test_no_tool_text_adds_no_source_row(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    model = _Model()
    await handlers._apply_personal_claim_check(_ctx(model, ""), "Fine.", mode="log_only")
    assert [r["role"] for r in _evidence_of(model)] == ["user"]


def test_extra_sources_share_the_page_cap_with_gathered_pages():
    pages = SimpleNamespace(pages=[SimpleNamespace(title=f"p{i}", content=f"gathered page {i}") for i in range(SOURCE_MAX_PAGES)])
    ev = build_personal_evidence("q", {"web_search_results": pages}, extra_sources=[("t", "extra tool text")])
    assert [r["role"] for r in ev].count("source") == SOURCE_MAX_PAGES
    assert not any("extra tool text" in r["text"] for r in ev)


def test_extra_source_is_page_capped():
    from core.personal_claim_check import SOURCE_PAGE_MAX_CHARS
    ev = build_personal_evidence("q", {}, extra_sources=[("t", "x " * (SOURCE_PAGE_MAX_CHARS * 2))], max_chars=100000)
    src = [r for r in ev if r["role"] == "source"]
    assert len(src) == 1 and len(src[0]["text"]) <= SOURCE_PAGE_MAX_CHARS
