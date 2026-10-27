"""Behavioural sibling for ``TestDocGenAndSelfNoteArtifactStrip.
test_document_generator_strips_trailing_artifact`` in
tests/unit/test_stream_artifacts.py (BC-63: it grepped
``inspect.getsource(DocumentGenerator.generate)`` for the call text).

Incident date: 2026-09-01 (kimi-3 lone ``e`` / ``<|sep|>`` tokens reaching saved
documents). Bug classes: BC-63, BC-62. THE deployed ``generate`` is driven with
a fake model manager whose reply carries the artifact, and the SAVED file is
read back.
"""
from pathlib import Path

import pytest

from knowledge.document_generator import DocumentGenerator


class _FakeMM:
    default_model = "test-model"

    def __init__(self, reply):
        self.reply = reply

    async def generate_once(self, prompt, system_prompt=None, **kwargs):
        return self.reply


BODY = "## Overview\nSummary content about the topic. [WEB_1]\n\n## Sources\n- [WEB_1] Test Source"


async def _saved(tmp_path, reply):
    dg = DocumentGenerator(model_manager=_FakeMM(reply), output_dir=tmp_path,
                           repo_root=tmp_path)
    result = await dg.generate("a narrow topic", doc_type="summary")
    return Path(result.path).read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_lone_e_glued_to_terminal_punctuation_is_not_saved(tmp_path, wrap):
    reply = BODY + ".e\n"
    if wrap:
        reply = "\n".join("  " + ln if ln else ln for ln in reply.split("\n"))
    text = await _saved(tmp_path, reply)
    assert ".e" not in text.replace("i.e", "")
    assert "Summary content about the topic." in text


@pytest.mark.asyncio
async def test_special_token_leak_is_not_saved(tmp_path):
    text = await _saved(tmp_path, BODY + "\n<|sep|>")
    assert "<|sep|>" not in text


@pytest.mark.asyncio
async def test_ordinary_trailing_e_word_is_kept(tmp_path):
    # control: only the glued-artifact shape is stripped, never a real word
    text = await _saved(tmp_path, BODY + " for the date\n")
    assert "for the date" in text
