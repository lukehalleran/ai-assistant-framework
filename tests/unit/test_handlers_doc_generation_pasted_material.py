"""Derivative documents from PASTED material (oct26b D1 item 1; incident 2026-09-20).
class: BC-15, BC-58

A resume pasted into the message + "make a clean version of this" ran the
research/report pipeline: the derivative path (classify_deliverable ->
compose_from_material) was keyed on attachments only. Driven through THE
deployed `_run_doc_generation` with a fake generator (no LLM).
"""

from types import SimpleNamespace

import pytest

import gui.handlers as handlers
import knowledge.document_generator as dg_mod
from memory.memory_storage import is_paste_sized

RESUME = "Pricing Analyst, Acme Mutual. Maintained rate models for Rowan Group.\n" * 40
REQUEST = "make a clean one-page version of this resume:\n\n" + RESUME


def _ctx(user_text, *, topic="resume"):
    return SimpleNamespace(
        orchestrator=SimpleNamespace(
            prompt_builder=None, memory_system=None, active_documents=None,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        doc_gen_intent={"topic": topic, "doc_type": "report", "focus": None, "source": None},
        user_text=user_text, merged_input=user_text, history=[], handled=False,
        turn_attachments=[],
    )


@pytest.fixture
def got(monkeypatch):
    rec = {"classified": None}

    class FakeGenerator:
        verdict = "derivative"

        def __init__(self, **kw):
            pass

        async def classify_deliverable(self, request):
            rec["classified"] = request
            return FakeGenerator.verdict

        async def compose_from_material(self, **kw):
            rec["composed"] = kw
            return SimpleNamespace(title="T", path="/tmp/drafts/x.md", doc_type="draft",
                                   sources=[], sections_count=1, word_count=5)

        async def generate(self, **kw):
            rec["generated"] = kw
            return SimpleNamespace(title="T", path="/tmp/reports/x.md", doc_type="report",
                                   sources=[], sections_count=1, word_count=5)

        def repoint_index(self, a, b):
            pass

    rec["cls"] = FakeGenerator
    monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeGenerator)
    monkeypatch.setattr(handlers, "_write_turn_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "_get_session_id", lambda *a: "s1")
    return rec


async def _run(ctx):
    return [c async for c in handlers._run_doc_generation(ctx)]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True], ids=["clean", "indented"])
async def test_pasted_material_with_derivative_request_composes(got, wrap):
    text = REQUEST if not wrap else "\n".join("    " + ln for ln in REQUEST.splitlines())
    assert is_paste_sized(text)
    ctx = _ctx(text)
    await _run(ctx)
    assert ctx.handled is True
    assert "generated" not in got
    assert got["composed"]["material"] == text  # full paste rides as MATERIAL
    assert "clean one-page version" in got["composed"]["request"]


@pytest.mark.asyncio
async def test_instruction_after_the_paste_is_seen_by_the_classifier(got):
    text = RESUME * 3 + "\nplease rewrite the above as a cleaner version"
    ctx = _ctx(text)
    await _run(ctx)
    assert "rewrite the above" in got["classified"]
    assert len(got["classified"]) < len(text)
    assert "composed" in got


@pytest.mark.asyncio
async def test_pasted_material_with_analysis_request_keeps_report_pipeline(got):
    got["cls"].verdict = "analysis"
    ctx = _ctx("write a report evaluating this:\n\n" + RESUME)
    await _run(ctx)
    assert "composed" not in got and got["generated"]["source_material"].endswith(RESUME)


@pytest.mark.asyncio
async def test_short_request_without_material_is_unchanged(got):
    text = "write a report about climate change"
    ctx = _ctx(text, topic="climate change")
    await _run(ctx)
    assert got["classified"] is None and "composed" not in got
    assert got["generated"]["source_material"] == text
