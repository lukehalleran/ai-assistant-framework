"""Bounded analysis query without a length cliff.

Incident 2026-10-17 17:09, class BC-09 (analysis run on the merged attachment
blob): a 2,079-char authored paste plus a ~10K-char PDF skipped the
"original <= cap" fallback, so retrieval/gating analyzed the whole blob.
"""
import pytest

from core.prompt.builder import ANALYSIS_QUERY_MAX_CHARS, _bounded_analysis_query

CAP = ANALYSIS_QUERY_MAX_CHARS
QUESTION = "do i have a quick bridge to 4, or should I lay out all 3 screen divisions?"


def _long_original(n=2079, indent=False):
    body = "Opening words here. " + ("When we open Arena we see 3 sections. " * 80)
    body = body[: n - len(QUESTION) - 1].rsplit(" ", 1)[0] + " " + QUESTION
    if indent:
        body = "\n".join("    " + ln for ln in body.replace(". ", ".\n").splitlines())
    return body


def test_short_processed_unchanged():
    assert _bounded_analysis_query("hello there", "other") == "hello there"


def test_long_processed_short_original_uses_original():
    assert _bounded_analysis_query("x " * 3000, "short question?") == "short question?"


@pytest.mark.parametrize("indent", [False, True])
def test_long_original_keeps_head_and_final_question(indent):
    original = _long_original(indent=indent)
    assert len(original) > CAP
    out = _bounded_analysis_query(original + "\nPDF " * 2000, original)
    assert len(out) <= CAP
    assert out.lstrip().startswith(original.split()[0])
    assert "quick bridge to 4" in out and out.rstrip().endswith("divisions?")
    assert "\n…\n" in out


def test_no_original_head_tail_of_processed():
    processed = "Start of text. " + "filler word " * 1000 + "Final question here?"
    for orig in (None, ""):
        out = _bounded_analysis_query(processed, orig)
        assert len(out) <= CAP
        assert out.startswith("Start") and out.endswith("question here?")


def test_cuts_land_on_whitespace():
    words = [f"word{i:04d}" for i in range(1000)]
    text = " ".join(words)
    out = _bounded_analysis_query(text, text)
    head, tail = out.split("\n…\n")
    assert all(w in words for w in head.split())
    assert all(w in words for w in tail.split())


@pytest.mark.asyncio
async def test_deployed_build_path_uses_helper(monkeypatch):
    import core.prompt.builder as b
    from core.context_pipeline import ContextResult, ToneLevel

    original = _long_original()
    huge = original + "\n" + "pdf text " * 2000
    context = ContextResult(
        processed_query=huge, original_query=original,
        tone_level=ToneLevel.CONVERSATIONAL, tone_instructions="", file_context=huge,
    )
    builder = object.__new__(b.UnifiedPromptBuilder)
    captured = {}

    async def fake_build_prompt(user_input, **kwargs):
        captured["user_input"] = user_input
        return {}

    builder.build_prompt = fake_build_prompt
    await b.UnifiedPromptBuilder.build_prompt_from_context(builder, context)
    assert len(captured["user_input"]) <= CAP
    assert "quick bridge to 4" in captured["user_input"]
    assert "pdf text" not in captured["user_input"]
