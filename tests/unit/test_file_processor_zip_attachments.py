"""process_files_structured: .srt/.vtt at top level and .zip expansion into per-member documents.

Incident 2026-10-17 13:25: a zip of 14 .srt lecture transcripts was stored as an
"[Unsupported file type]" sentinel. Bug classes: BC-15 (attachment content never reached the
model), BC-91 (machinery text stored as content).
"""
import asyncio
import io
import zipfile
from types import SimpleNamespace

import pytest

from utils import file_processor as fp

SRT = "1\n00:00:01,000 --> 00:00:03,000\nLecture {n} opening line.\n"


def _upload(name, data):
    return SimpleNamespace(name=name, orig_name=name, read=lambda: data)


def _zip(members, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _run(files):
    return asyncio.run(fp.FileProcessor().process_files_structured("hi", files))


def test_top_level_srt_is_read():
    r = _run([_upload("L1.srt", SRT.format(n=1).encode())])
    assert [d.filename for d in r.documents] == ["L1.srt"]
    assert "[00:01] Lecture 1 opening line." in r.text_content


def test_zip_expands_into_per_member_documents_in_natural_order():
    data = _zip([("L10.srt", SRT.format(n=10)), ("L2.srt", SRT.format(n=2)), ("L1.srt", SRT.format(n=1)),
                 ("__MACOSX/._L1.srt", "x"), (".hidden.srt", "x"), ("dir/", "")])
    r = _run([_upload("Mod 5.zip", data)])
    assert [d.filename for d in r.documents] == ["Mod 5.zip/L1.srt", "Mod 5.zip/L2.srt", "Mod 5.zip/L10.srt"]
    assert "Lecture 10 opening line." in r.documents[2].content_text
    assert "skipped" not in r.text_content
    assert not any(p.error for p in r.processed_files)


def test_image_and_nested_zip_skipped_and_named_once():
    inner = _zip([("a.srt", SRT.format(n=1))])
    data = _zip([("L1.srt", SRT.format(n=1)), ("pic.png", b"\x89PNG"), ("inner.zip", inner)])
    r = _run([_upload("m.zip", data)])
    assert [d.filename for d in r.documents] == ["m.zip/L1.srt"]
    assert r.text_content.count("[zip m.zip: skipped 2 member(s): pic.png, inner.zip]") == 1


@pytest.mark.parametrize("bad", ["../evil.srt", "/abs/evil.srt", "ok/../../evil.srt"])
def test_zip_slip_rejected(bad):
    r = _run([_upload("m.zip", _zip([(bad, SRT.format(n=1))]))])
    assert not r.documents
    assert r.processed_files[0].error and "unsafe member path" in r.processed_files[0].error


def test_bomb_ratio_refused():
    data = _zip([("big.txt", "A" * 5_000_000)])
    r = _run([_upload("b.zip", data)])
    assert not r.documents
    assert "zip bomb" in r.processed_files[0].error


def test_member_count_cap():
    data = _zip([(f"f{i}.srt", SRT.format(n=i) + f"{i}\n" * 3) for i in range(fp.ZIP_MAX_MEMBERS + 1)])
    r = _run([_upload("many.zip", data)])
    assert not r.documents
    assert "members" in r.processed_files[0].error


def test_total_size_counts_across_batch(monkeypatch):
    monkeypatch.setattr(fp, "MAX_TOTAL_SIZE", 400)
    data = _zip([("L1.srt", SRT.format(n=1))])
    first = _upload("pad.txt", b"x" * 390)
    r = _run([first, _upload("m.zip", data)])
    assert not [d for d in r.documents if d.filename.startswith("m.zip/")]


def test_bad_zip_is_error_not_crash():
    r = _run([_upload("broken.zip", b"this is not a zip")])
    assert not r.documents
    assert "not a valid zip" in r.processed_files[0].error
    assert "broken.zip" in r.text_content


def test_unsupported_member_marked_error_not_document():
    data = _zip([("L1.srt", SRT.format(n=1)), ("tool.exe", b"MZ")])
    r = _run([_upload("m.zip", data)])
    assert [d.filename for d in r.documents] == ["m.zip/L1.srt"]
    assert "skipped 1 member(s): tool.exe" in r.text_content
