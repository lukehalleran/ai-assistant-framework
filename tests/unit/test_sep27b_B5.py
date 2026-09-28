"""
Tests for lane B batch B5 (2026-09-27): doc-gen polish.

Module Contract
- Purpose: Validate the two B5 closures against the DEPLOYED functions —
  never a re-derivation:
  1. knowledge/document_export._versioned derives the export filename from
     the md stem and versions ONCE (an already-versioned md, "…-2.md",
     used to stack a second "-2" onto its own export on collision,
     "…-2-2.docx", instead of "…-3.docx"). class: BC-45.
  2. knowledge/document_generator.DocumentGenerator now records the real
     LLM prompt(s)/system_prompt(s) + token counts it sends while producing
     a document, exposed on GeneratedDocument (debug_prompt,
     debug_system_prompt, prompt_tokens, system_tokens, total_tokens) —
     previously nothing captured this, so a caller building the doc-gen
     debug record had only the bare query and zeros. class: BC-72.
  utils/turn_telemetry.record_turn (the field names document_generator's
  new data is meant to reach, once a caller is wired) still round-trips the
  new small-integer fields faithfully through its generic sanitizer.
- Inputs: tmp_path fixtures, an in-file fake ModelManager (no network).
- Outputs: pass/fail assertions.
- Dependencies: pytest, pytest-asyncio, standard library.
"""

import json

import pytest

import knowledge.document_export as dx
from knowledge.document_generator import DocumentGenerator, GeneratedDocument
from utils import turn_telemetry


# ============================================================================
# B5(b): export versioning derives from the md stem, versions once
# ============================================================================

class TestExportVersioningDerivesFromStem:

    def test_already_versioned_md_stacks_no_further_on_first_export(self, tmp_path):
        """First export of an already-versioned md (…-2.md, from the
        generator's OWN collision handling) matches its stem exactly — no
        extra suffix appended just because the stem already carries one."""
        md = tmp_path / "resume-2026-09-20-2.md"
        md.write_text("# Title\n\nBody text.\n", encoding="utf-8")
        out = dx.export_document(md, "txt")
        assert out.name == "resume-2026-09-20-2.txt"

    def test_second_export_of_versioned_md_increments_once(self, tmp_path):
        """The bug (FOLLOWUPS 09-20 r7): exporting the SAME already-versioned
        md a second time (an older .txt output already occupies the natural
        name) used to stack a second '-2' onto the whole stem
        ('resume-2026-09-20-2-2.txt'). It must instead increment the ONE
        version marker the stem already carries."""
        md = tmp_path / "resume-2026-09-20-2.md"
        md.write_text("# Title\n\nBody text.\n", encoding="utf-8")
        first = dx.export_document(md, "txt")
        second = dx.export_document(md, "txt")
        assert first.name == "resume-2026-09-20-2.txt"
        assert second.name == "resume-2026-09-20-3.txt"
        assert "-2-2" not in second.name
        assert first.exists() and second.exists()

    def test_third_export_continues_the_single_suffix(self, tmp_path):
        md = tmp_path / "resume-2026-09-20-2.md"
        md.write_text("# Title\n\nBody text.\n", encoding="utf-8")
        dx.export_document(md, "txt")
        dx.export_document(md, "txt")
        third = dx.export_document(md, "txt")
        assert third.name == "resume-2026-09-20-4.txt"

    def test_plain_stem_without_a_date_is_unaffected(self, tmp_path):
        """Regression guard: a plain (non date-suffixed) md stem keeps the
        pre-existing behavior exactly (tests/unit/test_document_export.py::
        test_never_overwrites covers the .md-generator-naming-free case;
        this repeats it here so the whole B5 contract lives in one file)."""
        md = tmp_path / "fair-resume.md"
        md.write_text("# Title\n\nBody.\n", encoding="utf-8")
        first = dx.export_document(md, "txt")
        second = dx.export_document(md, "txt")
        assert first.name == "fair-resume.txt"
        assert second.name == "fair-resume-2.txt"

    def test_first_version_dated_stem_not_misparsed_as_a_version_marker(self, tmp_path):
        """Guard against a defect the fix could have introduced: a FIRST-
        version dated stem ('resume-2026-09-20', no explicit '-N' yet) must
        not have its date's day-of-month digits ('-20') mistaken for a
        version marker and incremented into a bogus date ('...-21.txt')."""
        md = tmp_path / "resume-2026-09-20.md"
        md.write_text("# Title\n\nBody.\n", encoding="utf-8")
        first = dx.export_document(md, "txt")
        second = dx.export_document(md, "txt")
        assert first.name == "resume-2026-09-20.txt"
        assert second.name == "resume-2026-09-20-2.txt"


# ============================================================================
# B5: doc-gen debug record carries the real prompt(s) + token counts
# ============================================================================

class _FakeModelManager:
    """No network: returns queued replies in call order, records every call
    it received so a test can additionally inspect what was actually sent."""

    def __init__(self, replies):
        self.default_model = "test-model"
        self._replies = list(replies)
        self.calls = []

    async def generate_once(self, prompt, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, **kwargs})
        if not self._replies:
            raise AssertionError("_FakeModelManager: no more queued replies")
        return self._replies.pop(0)


_SUMMARY_BODY = (
    "## Overview\nSummary content about the topic. [WEB_1]\n\n"
    "## Sources\n- [WEB_1] Test Source\n"
)
_OUTLINE = "## Section 1\nIntro.\n## Section 2\nDetails.\n"
_REPORT_BODY = (
    "## Section 1\nIntro content. [WEB_1]\n\n## Section 2\nMore content. [WEB_1]\n\n"
    "## Sources\n- [WEB_1] Test Source\n"
)


@pytest.mark.asyncio
async def test_generate_summary_populates_real_prompt_and_tokens(tmp_path):
    mm = _FakeModelManager([_SUMMARY_BODY])
    dg = DocumentGenerator(model_manager=mm, output_dir=tmp_path, repo_root=tmp_path)
    result = await dg.generate("a specific narrow topic", doc_type="summary")

    assert isinstance(result, GeneratedDocument)
    # PROMPT is no longer just the bare query — the real draft prompt (which
    # embeds the topic and the source/citation instructions) reached it.
    assert "a specific narrow topic" in result.debug_prompt
    assert "[draft]" in result.debug_prompt
    assert result.debug_system_prompt  # real system prompt, not None/""
    assert result.prompt_tokens > 0
    assert result.system_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.system_tokens


@pytest.mark.asyncio
async def test_generate_report_records_both_outline_and_draft_prompts(tmp_path):
    mm = _FakeModelManager([_OUTLINE, _REPORT_BODY])
    dg = DocumentGenerator(model_manager=mm, output_dir=tmp_path, repo_root=tmp_path)
    result = await dg.generate("a specific narrow topic", doc_type="report")

    assert "[outline]" in result.debug_prompt
    assert "[draft]" in result.debug_prompt
    assert result.total_tokens > 0


@pytest.mark.asyncio
async def test_generate_resets_recorded_calls_between_runs(tmp_path):
    """A second generate() on the SAME instance must not accumulate the
    first run's prompts into the second run's debug summary."""
    mm = _FakeModelManager([_SUMMARY_BODY, _SUMMARY_BODY])
    dg = DocumentGenerator(model_manager=mm, output_dir=tmp_path, repo_root=tmp_path)
    first = await dg.generate("topic one", doc_type="summary")
    second = await dg.generate("topic two", doc_type="summary")

    assert first.debug_prompt.count("[draft]") == 1
    assert second.debug_prompt.count("[draft]") == 1
    assert "topic one" not in second.debug_prompt
    assert "topic two" in second.debug_prompt


@pytest.mark.asyncio
async def test_compose_from_material_records_the_derivative_prompt(tmp_path):
    mm = _FakeModelManager(["# Rewritten Document\n\nRewritten body.\n"])
    dg = DocumentGenerator(model_manager=mm, output_dir=tmp_path, repo_root=tmp_path)
    result = await dg.compose_from_material(
        request="rewrite this as a one-pager",
        material="Original material content here.",
        topic="rewrite request",
    )

    assert "[derivative]" in result.debug_prompt
    assert "rewrite this as a one-pager" in result.debug_prompt
    assert "Original material content here." in result.debug_prompt
    assert result.prompt_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.system_tokens


def test_save_prewritten_direct_call_reports_honest_zeros(tmp_path):
    """save_prewritten's OWN contract (2026-08-23): no research pipeline, no
    LLM call. A caller that saves already-authored text (insight mode) on a
    FRESH instance must get real zeros, not fabricated non-zero data."""
    mm = _FakeModelManager([])
    dg = DocumentGenerator(model_manager=mm, output_dir=tmp_path, repo_root=tmp_path)
    result = dg.save_prewritten("# Title\n\nAlready written body.\n", topic="insight save")

    assert result.debug_prompt == ""
    assert result.debug_system_prompt == ""
    assert result.prompt_tokens == 0
    assert result.system_tokens == 0
    assert result.total_tokens == 0


# ============================================================================
# turn_telemetry: the doc-gen token fields round-trip through record_turn
# ============================================================================

def test_record_turn_round_trips_doc_gen_token_fields(tmp_path, monkeypatch):
    """Once a caller wires GeneratedDocument's real counts into the turn
    record (doc_gen_prompt_tokens/doc_gen_system_tokens/doc_gen_total_tokens
    — documented in turn_telemetry's module docstring), record_turn's
    generic sanitizer must carry them through as real integers, not stringify
    or drop them."""
    log_path = tmp_path / "turn_records.jsonl"
    import config.app_config as app_config
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_PATH", str(log_path), raising=False)

    ok = turn_telemetry.record_turn({
        "mode": "doc-generation",
        "doc_gen_prompt_tokens": 842,
        "doc_gen_system_tokens": 17,
        "doc_gen_total_tokens": 859,
    })
    assert ok is True

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["doc_gen_prompt_tokens"] == 842
    assert payload["doc_gen_system_tokens"] == 17
    assert payload["doc_gen_total_tokens"] == 859


def test_turn_telemetry_docstring_names_the_doc_gen_fields():
    """The module contract (what a future caller should name the fields)
    documents the exact field names GeneratedDocument's data is meant to
    populate — a plain grep-able contract, not a behavior change."""
    doc = turn_telemetry.__doc__ or ""
    assert "doc_gen_prompt_tokens" in doc
    assert "doc_gen_system_tokens" in doc
    assert "doc_gen_total_tokens" in doc


# ─── CGR-20260927-002: a failed source search is not "zero sources" ──────

class _FailingWeb:
    def is_available(self):
        return True

    async def search(self, **_kw):
        raise ConnectionError("provider down")


class _FailingChroma:
    def query_collection(self, *_a, **_kw):
        raise TimeoutError("chroma stalled")


@pytest.mark.asyncio
async def test_failed_source_searches_are_recorded_not_read_as_empty(tmp_path):
    gen = DocumentGenerator(web_search_manager=_FailingWeb(), chroma_store=_FailingChroma())
    gen._source_failures = []
    sources = await gen._gather_sources("a topic")
    assert sources == []
    assert sorted(gen._source_failures) == ["notes", "web", "wiki"]


def test_generated_document_carries_source_failures_default_empty():
    doc = GeneratedDocument(path="p", title="t", doc_type="report", topic="x",
                            focus=None, sources=[], created_at="now")
    assert doc.source_failures == []
