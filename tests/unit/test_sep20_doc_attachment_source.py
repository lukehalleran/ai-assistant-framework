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
RESUME_TEXT = "Pricing Analyst, Acme Mutual. Maintained rate models. " * 20


def _ctx(user_text, merged_input, *, docs=(), history=(), attachments=()):
    registry = SimpleNamespace(documents=lambda: list(docs))
    return SimpleNamespace(
        orchestrator=SimpleNamespace(
            prompt_builder=None, memory_system=None, active_documents=registry,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        doc_gen_intent={"topic": user_text, "doc_type": "report", "focus": None},
        user_text=user_text, merged_input=merged_input,
        history=list(history), handled=False, turn_attachments=list(attachments),
    )


@pytest.fixture
def captured(monkeypatch):
    got = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        async def assign_attachment_roles(self, request, attachments):
            got["roles_called"] = True
            got["roles_request"] = request
            got["roles_attachments"] = attachments
            return {"template": got.get("_template_name")}

        async def classify_deliverable(self, request):
            got["classified"] = request
            return got.get("_kind", "analysis")

        async def compose_from_material(self, **kwargs):
            got["composed"] = kwargs
            return SimpleNamespace(
                title="T", path="/tmp/draft.md", doc_type="draft",
                sources=[], sections_count=1, word_count=10,
            )

        async def generate(self, **kwargs):
            got.update(kwargs)
            return SimpleNamespace(
                title="T", path="/tmp/doc.md", doc_type="report",
                sources=[], sections_count=1, word_count=10,
            )

        def repoint_index(self, old, new):
            pass

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


# --- derivative deliverable (round 2): the first fix routed the resume in as
# [INPUT_1] and the report pipeline wrote a cited ANALYSIS of it. ---

@pytest.mark.asyncio
async def test_derivative_request_composes_instead_of_reporting(captured):
    captured["_kind"] = "derivative"
    merged = f"{LIVE_MESSAGE}\n\n[FILE: fair_resume.docx]\n{RESUME_TEXT}"
    doc = SimpleNamespace(display_name="fair_resume.docx", registered_turn=15)
    ctx = _ctx(LIVE_MESSAGE, merged, docs=[doc])
    chunks = [c async for c in handlers._run_doc_generation(ctx)]
    assert captured["classified"] == LIVE_MESSAGE
    assert RESUME_TEXT in captured["composed"]["material"]
    assert captured["composed"]["topic"] == "fair resume"
    assert "source_material" not in captured  # generate() never ran
    assert "/tmp/draft.md" in chunks[-1]["content"]


@pytest.mark.asyncio
async def test_classifier_never_runs_without_an_attachment(captured):
    captured["_kind"] = "derivative"
    text = "write a report about climate change"
    [c async for c in handlers._run_doc_generation(_ctx(text, text))]
    assert "classified" not in captured and "composed" not in captured


def _gen(reply):
    async def generate_once(prompt, **kw):
        if isinstance(reply, Exception):
            raise reply
        return reply
    return dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,expected", [
    ("DERIVATIVE", "derivative"), (" derivative.\n", "derivative"),
    ("DERIVATIVEe", "derivative"), ("ANALYSISe", "analysis"),  # kimi-3 trailing-'e' artifact (live)
    ("ANALYSIS", "analysis"), ("not derivative, analysis", "analysis"),
    ("", "analysis"), (None, "analysis"), (RuntimeError("boom"), "analysis"),
    (dg_mod._LLM_ERROR_SENTINELS[0] + " derivative", "analysis"),
])
async def test_classify_deliverable_fails_safe_to_analysis(reply, expected):
    assert await _gen(reply).classify_deliverable("rewrite my resume") == expected


@pytest.mark.asyncio
async def test_compose_writes_a_draft_with_no_report_scaffolding(tmp_path):
    gen = _gen("# Jordan Example\n\nPricing Analyst, Acme Mutual.")
    doc = await gen.compose_from_material(
        request="new version", material=RESUME_TEXT, topic="fair resume")
    text = open(doc.path, encoding="utf-8").read()
    assert "/drafts/" in doc.path and doc.doc_type == "draft"
    assert "Pricing Analyst, Acme Mutual." in text and "## Sources" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["", "   ", dg_mod._LLM_ERROR_SENTINELS[0] + " 402"])
async def test_compose_refuses_error_or_empty_output(reply):
    with pytest.raises(RuntimeError):
        await _gen(reply).compose_from_material(request="x", material="y", topic="t")


@pytest.mark.asyncio
async def test_classifier_call_survives_a_reasoning_model():
    """Live 20:21: kimi-k3 burned max_tokens=8 in its reasoning channel → empty
    answer → 'analysis' → a report. The call must disable reasoning and leave room."""
    seen = {}

    async def generate_once(prompt, **kw):
        seen.update(kw)
        return "The deliverable is DERIVATIVE."

    gen = dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))
    assert await gen.classify_deliverable(LIVE_MESSAGE) == "derivative"
    assert seen["disable_reasoning"] is True and seen["max_tokens"] >= 32


@pytest.mark.asyncio
async def test_compose_retries_reasoning_only_completion():
    calls = []

    async def generate_once(prompt, **kw):
        calls.append(kw.get("disable_reasoning", False))
        return "" if len(calls) == 1 else "# Resume\n\nBody."

    gen = dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))
    doc = await gen.compose_from_material(request="r", material="m", topic="fair resume")
    assert calls == [False, True] and doc.doc_type == "draft"


@pytest.mark.asyncio
async def test_compose_strips_leaked_special_token_so_title_is_the_h1():
    """Live 20:29: body began '<|sep|># JORDAN EXAMPLE' → title 'SUMMARY', token in the .docx."""
    gen = _gen("<|sep|># Jordan Example\n\n## SUMMARY\n\nBody.")
    doc = await gen.compose_from_material(request="r", material="m", topic="fair resume")
    assert doc.title == "Jordan Example"
    assert "<|sep|>" not in open(doc.path, encoding="utf-8").read()


# --- attachment role assignment (round 3): "write a new resume, cast into ---
# --- this ATS template" with two attachments — one CONTENT, one LAYOUT. ----

TEMPLATE_TEXT = "[Full Name]\n[Job Title]\n\nSUMMARY\n[summary]\n\nEXPERIENCE\n[job]\n"


@pytest.mark.asyncio
async def test_assign_attachment_roles_valid_json_names_the_template():
    gen = _gen('{"template": "ats_template.docx"}')
    attachments = [
        {"name": "resume.docx", "text": RESUME_TEXT},
        {"name": "ats_template.docx", "text": TEMPLATE_TEXT},
    ]
    roles = await gen.assign_attachment_roles("cast into the template", attachments)
    assert roles == {"template": "ats_template.docx"}


@pytest.mark.asyncio
async def test_assign_attachment_roles_name_not_in_list_is_none():
    gen = _gen('{"template": "not_a_real_attachment.docx"}')
    attachments = [{"name": "a.docx", "text": "x"}, {"name": "b.docx", "text": "y"}]
    assert await gen.assign_attachment_roles("req", attachments) == {"template": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "not json at all", "", "   ", None, RuntimeError("boom"),
    dg_mod._LLM_ERROR_SENTINELS[0] + ' {"template": "a.docx"}',
])
async def test_assign_attachment_roles_fails_safe_to_none(reply):
    gen = _gen(reply)
    attachments = [{"name": "a.docx", "text": "x"}, {"name": "b.docx", "text": "y"}]
    assert await gen.assign_attachment_roles("req", attachments) == {"template": None}


@pytest.mark.asyncio
async def test_assign_attachment_roles_disables_reasoning():
    seen = {}

    async def generate_once(prompt, **kw):
        seen.update(kw)
        return '{"template": null}'

    gen = dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))
    attachments = [{"name": "a.docx", "text": "x"}, {"name": "b.docx", "text": "y"}]
    await gen.assign_attachment_roles("req", attachments)
    assert seen["disable_reasoning"] is True and seen["max_tokens"] == 120


@pytest.mark.asyncio
async def test_compose_with_template_text_adds_layout_block_and_rule():
    seen = {}

    async def generate_once(prompt, **kw):
        seen["prompt"] = prompt
        return "# Jordan Example\n\n## Experience\n\nPricing Analyst, Acme Mutual."

    gen = dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))
    await gen.compose_from_material(
        request="cast into the template", material=RESUME_TEXT,
        topic="fair resume", template_text=TEMPLATE_TEXT,
    )
    assert "[LAYOUT TEMPLATE" in seen["prompt"] and TEMPLATE_TEXT in seen["prompt"]
    assert "never copy its placeholder facts" in seen["prompt"]


@pytest.mark.asyncio
async def test_compose_without_template_text_has_no_layout_block():
    seen = {}

    async def generate_once(prompt, **kw):
        seen["prompt"] = prompt
        return "# Jordan Example\n\nBody."

    gen = dg_mod.DocumentGenerator(model_manager=SimpleNamespace(generate_once=generate_once))
    await gen.compose_from_material(request="r", material=RESUME_TEXT, topic="fair resume")
    assert "[LAYOUT TEMPLATE" not in seen["prompt"]


# --- handler wiring: template excluded from material, used as layout+export ---

@pytest.mark.asyncio
async def test_template_attachment_excluded_from_material_and_used_as_layout(captured):
    captured["_template_name"] = "ats_template.docx"
    request = "write a new resume from my old one, cast into the template"
    attachments = [
        {"name": "resume.docx", "path": "/tmp/resume.docx", "extension": ".docx", "text": RESUME_TEXT},
        {"name": "ats_template.docx", "path": "/tmp/ats_template.docx", "extension": ".docx", "text": TEMPLATE_TEXT},
    ]
    merged = f"{request}\n\n{RESUME_TEXT}\n\n{TEMPLATE_TEXT}"
    ctx = _ctx(request, merged, attachments=attachments)

    chunks = [c async for c in handlers._run_doc_generation(ctx)]

    assert ctx.handled is True
    assert captured["roles_called"] is True
    assert "classified" not in captured  # a template turn skips classify_deliverable
    composed = captured["composed"]
    # .strip(): the material's overall trailing whitespace is trimmed, which
    # can eat RESUME_TEXT's own trailing space when it lands at the very end.
    assert RESUME_TEXT.strip() in composed["material"]
    assert TEMPLATE_TEXT not in composed["material"]
    assert composed["template_text"] == TEMPLATE_TEXT
    assert "**Template**: ats_template.docx" in chunks[-1]["content"]


@pytest.mark.asyncio
async def test_export_receives_the_template_file_as_reference_doc(captured, monkeypatch):
    captured["_template_name"] = "ats_template.docx"
    request = "write a new resume from my old one as a docx, cast into the template"
    attachments = [
        {"name": "resume.docx", "path": "/tmp/resume.docx", "extension": ".docx", "text": RESUME_TEXT},
        {"name": "ats_template.docx", "path": "/tmp/ats_template.docx", "extension": ".docx", "text": TEMPLATE_TEXT},
    ]
    merged = f"{request}\n\n{RESUME_TEXT}\n\n{TEMPLATE_TEXT}"
    ctx = _ctx(request, merged, attachments=attachments)

    export_calls = {}

    def fake_export(md_path, fmt, *, reference_doc=None):
        export_calls["reference_doc"] = reference_doc
        return "/tmp/draft.docx"

    monkeypatch.setattr(handlers.document_export, "export_document", fake_export)
    [c async for c in handlers._run_doc_generation(ctx)]
    assert export_calls["reference_doc"] == "/tmp/ats_template.docx"


@pytest.mark.asyncio
async def test_single_attachment_never_calls_role_assignment(captured):
    """One attachment: assign_attachment_roles is only meaningful with >=2 —
    the pre-existing single-attachment behaviour (every attachment is
    content) must be unchanged."""
    merged = f"{LIVE_MESSAGE}\n\n[FILE: fair_resume.docx]\n{RESUME_TEXT}"
    doc = SimpleNamespace(display_name="fair_resume.docx", registered_turn=15)
    ctx = _ctx(LIVE_MESSAGE, merged, docs=[doc], attachments=[
        {"name": "fair_resume.docx", "path": "/tmp/fair_resume.docx", "extension": ".docx", "text": RESUME_TEXT},
    ])
    [c async for c in handlers._run_doc_generation(ctx)]
    assert "roles_called" not in captured


def test_attachment_topic_never_names_the_output_after_the_template():
    docs = [SimpleNamespace(display_name="fair_resume.docx", registered_turn=20),
            SimpleNamespace(display_name="single column resume template.docx", registered_turn=21)]
    ctx = SimpleNamespace(orchestrator=SimpleNamespace(
        active_documents=SimpleNamespace(documents=lambda: docs)))
    assert handlers._attachment_topic(ctx) == "single column resume template"  # newest, no exclusion
    assert handlers._attachment_topic(
        ctx, exclude_name="single column resume template.docx") == "fair resume"


# --- trailing-artifact guard (round 4): kimi-3 sometimes glues a stray 'e' --
# onto the FINAL word of a composed document ("...MS Excel, and MS Accesse" —
# live), which the existing stream-artifact stripper only catches after
# terminal punctuation. Purely structural: never a keyword/section check. ----

@pytest.mark.parametrize("body,grounding,expected", [
    (
        "Skilled in SQL, Excel, and MS Accesse",
        "Software: SQL, Excel, MS Access, PowerPoint",
        "Skilled in SQL, Excel, and MS Access",
    ),
    (
        "Experienced with Microsoft Office",
        "Daily use of the Office suite",
        "Experienced with Microsoft Office",
    ),
    (
        "Ends with an invented worde",
        "None of these words appear anywhere",
        "Ends with an invented worde",
    ),
    ("", "grounding text", ""),
    (None, None, None),
])
def test_strip_ungrounded_trailing_e(body, grounding, expected):
    assert dg_mod._strip_ungrounded_trailing_e(body, grounding) == expected


@pytest.mark.asyncio
async def test_compose_strips_ungrounded_trailing_e_using_material_grounding():
    """The trailing-e guard runs on compose_from_material's own output,
    grounded against material + template_text + request."""
    gen = _gen("# Resume\n\nSkills: SQL, Excel, MS Accesse")
    doc = await gen.compose_from_material(
        request="rewrite", material="Tools: MS Access, SQL, Excel", topic="fair resume",
    )
    text = open(doc.path, encoding="utf-8").read()
    assert "MS Accesse" not in text and "MS Access" in text


@pytest.mark.asyncio
async def test_compose_leaves_genuine_trailing_word_alone():
    """A word that ends in 'e' and IS itself grounded must never be touched."""
    gen = _gen("# Resume\n\nWorked extensively with Office")
    doc = await gen.compose_from_material(
        request="rewrite", material="Uses the Office suite daily", topic="fair resume",
    )
    text = open(doc.path, encoding="utf-8").read()
    assert text.rstrip().endswith("Office")
