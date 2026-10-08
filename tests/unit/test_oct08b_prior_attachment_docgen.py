"""Doc-gen from an attachment sent in a PRIOR turn (2026-10-08).
class: BC-58, BC-15

A document request that names (or, per the trigger's "attachment" source
declaration, clearly means) a file registered in an EARLIER turn ran research
mode on the bare request: the attachment text lives in ctx.merged_input only
on the turn it was sent. Driven through THE deployed `_run_doc_generation`
with the fixture pattern of test_sep20_doc_attachment_source.py.
"""

from types import SimpleNamespace

import pytest

import gui.handlers as handlers
import knowledge.document_generator as dg_mod
from core.active_document import AMBIGUOUS, ActiveDocumentRegistry, select_source_document
from utils.web_search_trigger import LLMSearchTriggerResponse

BODY = "Pricing Analyst, Acme Mutual. Maintained rate models. " * 20
OTHER = "Quarterly lab notes on titration. " * 30
REQUEST = "write a clean new version of fair_resume.docx for me"


def _ctx(user_text, docs, *, source=None, history=()):
    registry = SimpleNamespace(documents=lambda: list(docs))
    return SimpleNamespace(
        orchestrator=SimpleNamespace(
            prompt_builder=None, memory_system=None, active_documents=registry,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        doc_gen_intent={"topic": user_text, "doc_type": "report", "focus": None, "source": source},
        user_text=user_text, merged_input=user_text,
        history=list(history), handled=False, turn_attachments=[],
    )


def _doc(name, text=BODY, turn=3):
    return SimpleNamespace(display_name=name, text=text, registered_turn=turn)


@pytest.fixture
def captured(monkeypatch):
    got = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        async def assign_attachment_roles(self, request, attachments):
            return {"template": None}

        async def classify_deliverable(self, request):
            got["classified"] = request
            return "analysis"

        async def compose_from_material(self, **kwargs):
            got["composed"] = kwargs
            return SimpleNamespace(title="T", path="/tmp/draft.md", doc_type="draft",
                                   sources=[], sections_count=1, word_count=10)

        async def generate(self, **kwargs):
            got.update(kwargs)
            return SimpleNamespace(title="T", path="/tmp/doc.md", doc_type="report",
                                   sources=[], sections_count=1, word_count=10)

        def repoint_index(self, old, new):
            pass

    monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeGenerator)
    monkeypatch.setattr(handlers, "_write_turn_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "_get_session_id", lambda *a: "s1")
    return got


async def _run(ctx):
    return [c async for c in handlers._run_doc_generation(ctx)]


@pytest.mark.asyncio
async def test_filename_mention_uses_prior_document_text(captured):
    history = [{"role": "assistant", "content": "Fix: bold the section headers."}]
    ctx = _ctx(REQUEST, [_doc("hw3.pdf", OTHER), _doc("fair_resume.docx")], history=history)
    chunks = await _run(ctx)
    assert ctx.handled is True
    material = captured["source_material"]
    assert BODY in material and OTHER not in material
    assert "bold the section headers" in material  # same transcript tail as this-turn path
    assert captured["topic"] == "fair resume"
    assert "attachment" in [c for c in chunks if c.get("is_progress")][0]["content"]


@pytest.mark.asyncio
async def test_declared_attachment_with_one_document_uses_it(captured):
    ctx = _ctx("make a clean version of the file I sent", [_doc("fair_resume.docx")],
               source="attachment")
    await _run(ctx)
    assert BODY in captured["source_material"]
    assert captured["topic"] == "fair resume"


@pytest.mark.asyncio
async def test_declared_attachment_with_two_documents_falls_through(captured):
    ctx = _ctx("make a clean version of the file I sent",
               [_doc("fair_resume.docx"), _doc("hw3.pdf", OTHER)], source="attachment")
    chunks = await _run(ctx)
    assert ctx.handled is False
    assert chunks == [] and "source_material" not in captured and "composed" not in captured


@pytest.mark.asyncio
async def test_two_mentioned_documents_fall_through(captured):
    ctx = _ctx("merge fair_resume.docx and hw3.pdf into a summary",
               [_doc("fair_resume.docx"), _doc("hw3.pdf", OTHER)])
    await _run(ctx)
    assert ctx.handled is False and "source_material" not in captured


@pytest.mark.asyncio
async def test_undeclared_unmentioned_document_stays_research(captured):
    text = "write a report about climate change"
    ctx = _ctx(text, [_doc("fair_resume.docx")])
    ctx.doc_gen_intent["topic"] = "climate change"
    await _run(ctx)
    assert captured["source_material"] == text and captured["topic"] == "climate change"


@pytest.mark.asyncio
async def test_short_document_text_stays_research(captured):
    ctx = _ctx(REQUEST, [_doc("fair_resume.docx", text="tiny")])
    assert len("tiny") < dg_mod.DOCUMENT_PROVIDED_MIN_CHARS
    await _run(ctx)
    assert captured["source_material"] == REQUEST


class TestSelectSourceDocument:
    def test_none_when_nothing_signals(self):
        assert select_source_document([_doc("a.docx")], "write a report", declared_attachment=False) is None

    def test_short_stem_needs_the_full_filename(self):
        docs = [_doc("a.docx")]
        assert select_source_document(docs, "write a report", declared_attachment=False) is None
        assert select_source_document(docs, "use a.docx", declared_attachment=False) is docs[0]

    def test_mentioned_beats_declaration_and_registry_fakes_work(self):
        docs = [_doc("a.docx"), _doc("b.pdf")]
        assert select_source_document(docs, "from B.pdf please", declared_attachment=True).display_name == "b.pdf"

    def test_never_picks_newest_on_ambiguity(self):
        docs = [_doc("a.docx", turn=1), _doc("b.pdf", turn=9)]
        assert select_source_document(docs, "from the file", declared_attachment=True) is AMBIGUOUS

    def test_works_over_the_real_registry(self):
        reg = ActiveDocumentRegistry()
        reg.register("fair_resume.docx", BODY, "docx", 1)
        got = select_source_document(reg.documents(), "x", declared_attachment=True)
        assert got.display_name == "fair_resume.docx"


class TestTriggerParsesAttachmentSource:
    def _parse(self, value):
        resp = LLMSearchTriggerResponse.parse(
            '{"should_search": false, "search_terms": [], '
            '"needs_document_generation": true, "document_topic": "t", '
            f'"document_type": "report", "document_source": "{value}"}}'
        )
        assert resp is not None
        return resp

    def test_attachment_accepted(self):
        assert self._parse("attachment").document_source == "attachment"
        assert self._parse(" Attachment ").document_source == "attachment"

    def test_unknown_value_rejected(self):
        assert self._parse("banana").document_source == ""
