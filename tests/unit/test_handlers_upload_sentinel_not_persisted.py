"""An unsupported attachment is shown as a notice but never persisted as document content.

Incident 2026-10-17: "[Unsupported file type: X.zip]" was stored via upload_text as a
user_upload. Bug class: BC-91 (machinery text stored as content).
"""
import asyncio
from types import SimpleNamespace

from gui import handlers
from utils import file_processor as fp


class FakeRefDocs:
    def __init__(self):
        self.calls = []

    def upload_text(self, **kw):
        self.calls.append(kw)


def _orch(ref):
    gatherer = SimpleNamespace(reference_docs_manager=ref)
    return SimpleNamespace(prompt_builder=SimpleNamespace(context_gatherer=gatherer))


def test_unsupported_file_not_persisted_but_notice_visible():
    f = SimpleNamespace(name="tool.exe", orig_name="tool.exe", read=lambda: b"MZ")
    r = asyncio.run(fp.FileProcessor().process_files_structured("hi", [f]))
    assert "Unsupported file type: tool.exe" in r.text_content
    assert not r.documents
    ref = FakeRefDocs()
    asyncio.run(handlers._persist_uploads(_orch(ref), r))
    assert ref.calls == []


def test_persist_loop_skips_sentinel_even_if_marked_document():
    doc = fp.ProcessedFile(filename="x.zip", content_text=fp.unsupported_file_notice("x.zip"))
    ok = fp.ProcessedFile(filename="a.txt", content_text="real")
    r = fp.ProcessedFilesResult(documents=[doc, ok])
    ref = FakeRefDocs()
    asyncio.run(handlers._persist_uploads(_orch(ref), r))
    assert [c["title"] for c in ref.calls] == ["upload:a.txt"]
