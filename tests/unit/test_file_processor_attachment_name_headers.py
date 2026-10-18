"""Attached documents carry a file-name header line in the merged text.

Incident 2026-10-17 15:45: 26 zip-member transcripts were appended anonymously and the
model said "you've pasted the transcripts directly into the message". Class: BC-15
(attachment identity never reached the model).
"""
import asyncio
import io
import zipfile
from types import SimpleNamespace

from utils import file_processor as fp
from utils import read_time_markers

SRT = "1\n00:00:01,000 --> 00:00:03,000\nLecture {n} opening line.\n"


def _upload(name, data):
    return SimpleNamespace(name=name, orig_name=name, read=lambda: data)


def _run(files):
    return asyncio.run(fp.FileProcessor().process_files_structured("hi", files))


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, d in members:
            zf.writestr(n, d)
    return buf.getvalue()


def test_header_helper_format():
    assert fp.attachment_header("a.txt") == "[Attached file: a.txt]\n"


def test_plain_text_header_precedes_text():
    r = _run([_upload("notes.txt", b"alpha beta gamma")])
    assert "[Attached file: notes.txt]\nalpha beta gamma" in r.text_content


def test_zip_members_headers_in_order():
    data = _zip([("L2.srt", SRT.format(n=2)), ("L1.srt", SRT.format(n=1))])
    r = _run([_upload("Mod.zip", data)])
    t = r.text_content
    h1, h2 = "[Attached file: Mod.zip/L1.srt]\n", "[Attached file: Mod.zip/L2.srt]\n"
    assert h1 in t and h2 in t and t.index(h1) < t.index(h2)
    assert t.index(h1) < t.index("Lecture 1 opening") < t.index(h2) < t.index("Lecture 2 opening")


def test_documents_content_text_has_no_header():
    r = _run([_upload("notes.txt", b"alpha beta gamma")])
    assert r.documents[0].content_text == "alpha beta gamma"


def test_unsupported_file_has_no_header():
    r = _run([_upload("blob.xyz", b"\x00\x01")])
    assert "[Attached file:" not in r.text_content


def test_errored_file_has_no_header():
    r = _run([_upload("big.zip", _zip([("../evil.srt", SRT.format(n=1))]))])
    assert r.processed_files[0].error
    assert "[Attached file:" not in r.text_content


def test_duplicate_same_content_collapses_to_one():
    r = _run([_upload("a.txt", b"same body text"), _upload("a.txt", b"same body text")])
    assert r.text_content.count("[Attached file: a.txt]") == 1
    assert r.text_content.count("same body text") == 1


def test_header_survives_read_time_stripper():
    r = _run([_upload("notes.txt", b"alpha beta gamma")])
    assert "[Attached file: notes.txt]" in read_time_markers.strip_machinery(r.text_content)
    wrapped = "\n    ".join(r.text_content.split("\n"))
    assert "[Attached file: notes.txt]" in read_time_markers.strip_machinery(wrapped)
