"""
Attachment-sourced document generation (2026-09-20). class: BC-04

Live incident: "instead, please write a new document using info in attachment
and applied formatting fixes" with fair_resume.docx attached. _run_doc_generation
passed ctx.user_text (180 chars, no attachment text — that lives only in
ctx.merged_input) as source_material, which failed DOCUMENT_PROVIDED_MIN_CHARS,
so the generator web-searched the literal request and saved a report about the
PHRASE "please find attached". Driven through THE deployed _run_doc_generation.
"""

from types import SimpleNamespace

import pytest

import gui.handlers as handlers
import knowledge.document_generator as dg_mod

LIVE_MESSAGE = (
    "instead, please write a new document using info in attachment and applied "
    "formatting fixes. Do not give it to me here in chat, I am asking you to "
    "use tools to create a new document"
)
RESUME_TEXT = "Actuarial Analyst, Mercer. Maintained rate models. " * 20


def _ctx(user_text, merged_input, *, docs=(), history=()):
    registry = SimpleNamespace(documents=lambda: list(docs))
    return SimpleNamespace(
        orchestrator=SimpleNamespace(
            prompt_builder=None, memory_system=None, active_documents=registry,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        doc_gen_intent={"topic": user_text, "doc_type": "report", "focus": None},
        user_text=user_text, merged_input=merged_input,
        history=list(history), handled=False,
    )


@pytest.fixture
def captured(monkeypatch):
    got = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        async def generate(self, **kwargs):
            got.update(kwargs)
            return SimpleNamespace(
                title="T", path="/tmp/doc.md", doc_type="report",
                sources=[], sections_count=1, word_count=10,
            )

    monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeGenerator)
    monkeypatch.setattr(handlers, "_write_turn_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(handlers, "_get_session_id", lambda *a: "s1")
    return got


@pytest.mark.asyncio
async def test_live_attachment_turn_uses_attachment_as_material(captured):
    merged = f"{LIVE_MESSAGE}\n\n[FILE: fair_resume.docx]\n{RESUME_TEXT}"
    doc = SimpleNamespace(display_name="fair_resume.docx", registered_turn=15)
    older = SimpleNamespace(display_name="hw3.pdf", registered_turn=4)
    history = [{"role": "assistant", "content": "Fix: bold the section headers."}]
    ctx = _ctx(LIVE_MESSAGE, merged, docs=[older, doc], history=history)

    chunks = [c async for c in handlers._run_doc_generation(ctx)]

    assert ctx.handled is True
    material = captured["source_material"]
    assert RESUME_TEXT in material
    assert len(material) >= dg_mod.DOCUMENT_PROVIDED_MIN_CHARS  # web/wiki suppressed
    assert "bold the section headers" in material  # "applied fixes" = prior turns
    assert material.index(RESUME_TEXT) < material.index("bold the section headers")
    assert captured["topic"] == "fair resume"  # never the raw imperative
    assert "attachment" in [c for c in chunks if c.get("is_progress")][0]["content"]


@pytest.mark.asyncio
async def test_no_attachment_stays_research(captured):
    text = "write a report about climate change"
    ctx = _ctx(text, text, docs=[SimpleNamespace(display_name="hw3.pdf", registered_turn=4)])
    ctx.doc_gen_intent["topic"] = "climate change"
    [c async for c in handlers._run_doc_generation(ctx)]
    assert captured["source_material"] == text
    assert captured["topic"] == "climate change"  # an idle active doc never renames it


@pytest.mark.asyncio
async def test_tiny_merge_delta_is_not_an_attachment(captured):
    text = "write a report about climate change"
    ctx = _ctx(text, text + "\n[ATTACHMENT NOTE] x")
    [c async for c in handlers._run_doc_generation(ctx)]
    assert captured["source_material"] == text


def test_attachment_topic_without_registry_falls_back():
    ctx = SimpleNamespace(orchestrator=SimpleNamespace())
    assert handlers._attachment_topic(ctx) == ""


def test_documents_dir_is_sandboxed(tmp_path):
    """conftest autouse: a test-constructed DocumentGenerator never writes the
    owner's documents/ (61 'my-problem-is-x' summaries leaked from
    test_insight_mode_handler before 2026-09-20)."""
    gen = dg_mod.DocumentGenerator(model_manager=None)
    assert str(gen.output_dir.resolve()).startswith(str(tmp_path.resolve()))
