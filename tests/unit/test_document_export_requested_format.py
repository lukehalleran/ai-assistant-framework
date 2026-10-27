"""Requested-format export on the agentic generate_document tool and on insight
document saves (oct26b D1 item 2; incident 2026-09-20). class: BC-89, BC-15

`export_document` was wired on the main doc-gen path only; the agentic tool and
the insight save wrote markdown regardless of "as a pdf". All three now go
through `document_export.export_requested_format`. Converters are faked.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gui.handlers as handlers
import knowledge.document_export as dx
import knowledge.document_generator as dg_mod
from core.agentic.formatters import AgenticFormatter
from core.agentic.tools import ToolExecutor
from core.insight.types import Assessment, ClaimAssessment, EvidenceItem, FacetPlan, FacetQuery


@pytest.fixture
def md(tmp_path):
    p = tmp_path / "rates-2026-10-26.md"
    p.write_text("# Rates\n\nbody\n", encoding="utf-8")
    return p


@pytest.fixture
def fake_converter(monkeypatch):
    calls = []

    def fake_export(path, fmt, *, reference_doc=None):
        calls.append((str(path), fmt))
        out = path.with_suffix(f".{fmt}") if hasattr(path, "with_suffix") else __import__("pathlib").Path(path).with_suffix(f".{fmt}")
        out.write_text("converted", encoding="utf-8")
        return out

    monkeypatch.setattr(dx, "export_document", fake_export)
    return calls


def _fake_generator(md_path, rec):
    class FakeDG:
        def __init__(self, **kw):
            pass

        async def generate(self, **kw):
            return SimpleNamespace(title="Rates", path=str(md_path), doc_type="report",
                                   sources=[], sections_count=1, word_count=3)

        def save_prewritten(self, markdown, **kw):
            return SimpleNamespace(title="Rates", path=str(md_path))

        def repoint_index(self, old, new):
            rec["repointed"] = (str(old), str(new))

    return FakeDG


def _executor(request):
    ex = ToolExecutor(model_manager=MagicMock(), web_search_manager=MagicMock(),
                      formatter=AgenticFormatter())
    ex.set_fetch_provenance_context(current_user_text=request)
    return ex


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", [
    "write a report on rates as a pdf",
    "    write a report on rates\n    and save it as a pdf",
])
async def test_agentic_generate_document_exports_requested_format(md, fake_converter, monkeypatch, request_text):
    rec = {}
    monkeypatch.setattr(dg_mod, "DocumentGenerator", _fake_generator(md, rec))
    out = await _executor(request_text)._execute_generate_document("rates", "report")
    assert fake_converter == [(str(md), "pdf")]
    assert str(md.with_suffix(".pdf")) in out and "rates-2026-10-26.md" not in out
    assert not md.exists() and rec["repointed"][1] == str(md.with_suffix(".pdf"))


@pytest.mark.asyncio
async def test_agentic_generate_document_without_format_is_markdown_only(md, fake_converter, monkeypatch):
    monkeypatch.setattr(dg_mod, "DocumentGenerator", _fake_generator(md, {}))
    out = await _executor("write a report on rates")._execute_generate_document("rates", "report")
    assert fake_converter == [] and md.exists() and str(md) in out


@pytest.mark.asyncio
async def test_agentic_export_failure_is_reported_and_keeps_markdown(md, monkeypatch):
    def boom(*a, **k):
        raise dx.DocumentExportError("no converter")
    monkeypatch.setattr(dx, "export_document", boom)
    monkeypatch.setattr(dg_mod, "DocumentGenerator", _fake_generator(md, {}))
    out = await _executor("write it up as a docx")._execute_generate_document("rates", "report")
    assert "not written" in out and md.exists() and str(md) in out


# --- insight-mode save, through THE deployed _run_insight_mode -----------------

def _insight_ctx(raw_query):
    orchestrator = MagicMock()
    orchestrator.model_manager.get_active_model_name = MagicMock(return_value="kimi-3")
    orchestrator.memory_system.chroma_store = MagicMock()
    ctx = handlers.SubmitContext(
        user_text=raw_query, files=None, history=[], use_raw_gpt=False,
        orchestrator=orchestrator, personality=None, fast_mode=False,
        conversation_logger=None, file_names=[], merged_input="q", files_result=None,
    )
    ctx.raw_context = {}
    ctx.gate_decision = MagicMock()
    ctx.gate_decision.insight_intent = {"kind": "theme_sweep", "theme": "rates",
                                        "wants_document": True, "raw_query": raw_query}
    return ctx


@pytest.fixture
def insight_stages(monkeypatch):
    import core.insight.assessor as assessor_mod
    import core.insight.facets as facets_mod
    import core.insight.sweep as sweep_mod
    import core.insight.synthesizer as synth_mod

    async def synthesize_stream(intent, evidence, assessment, **kw):
        yield "The record shows a pattern (2026-08-18)."

    monkeypatch.setattr(facets_mod, "decompose", AsyncMock(return_value=FacetPlan(
        facets=[FacetQuery(name="f", query_text="q")], claims=["c"])))
    monkeypatch.setattr(sweep_mod, "run_sweep", AsyncMock(return_value=[
        EvidenceItem(doc_id="d1", text="note", date="2026-08-18", collection="conversations", speaker="user")]))
    monkeypatch.setattr(assessor_mod, "assess", AsyncMock(return_value=Assessment(
        overall="agree", claims=[ClaimAssessment(claim="c", verdict="agree", support=["E1"])])))
    monkeypatch.setattr(synth_mod, "synthesize_stream", synthesize_stream)
    monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)


def _collect(ctx):
    async def _run():
        return [c async for c in handlers._run_insight_mode(ctx)]
    return asyncio.run(_run())


def test_insight_document_save_exports_requested_format(md, fake_converter, insight_stages, monkeypatch):
    rec = {}
    monkeypatch.setattr(dg_mod, "DocumentGenerator", _fake_generator(md, rec))
    chunks = _collect(_insight_ctx("summarize my rates pattern and save it as a pdf"))
    assert fake_converter == [(str(md), "pdf")]
    assert str(md.with_suffix(".pdf")) in chunks[-1]["content"]


def test_insight_document_save_without_format_is_markdown_only(md, fake_converter, insight_stages, monkeypatch):
    monkeypatch.setattr(dg_mod, "DocumentGenerator", _fake_generator(md, {}))
    chunks = _collect(_insight_ctx("summarize my rates pattern and save it"))
    assert fake_converter == [] and md.exists() and str(md) in chunks[-1]["content"]
