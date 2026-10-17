"""Stdlib-only text extraction for OpenDocument text (.odt) files.

An .odt is a zip whose ``content.xml`` holds the document body. This module turns it into
plain text (headings as ``#`` lines, list items as ``- `` lines, tables as pipe-separated
rows) with no third-party dependency. Raises ValueError (clear message) for anything that is
not a readable, safely-sized .odt; the caller turns that into an error entry.
"""
import io
import xml.etree.ElementTree as ET
import zipfile

_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def _q(prefix: str, local: str) -> str:
    return "{%s}%s" % (_NS[prefix], local)


_SKIP = {_q("office", "annotation"), _q("text", "tracked-changes"), _q("office", "annotation-end")}
_P, _H = _q("text", "p"), _q("text", "h")
_SPAN_LIKE = {_q("text", "span"), _q("text", "a"), _q("text", "note"), _q("text", "ruby")}


def _inline(elem) -> str:
    """Text of a paragraph/heading element, honouring spaces, tabs and line breaks."""
    out = [elem.text or ""]
    for child in elem:
        tag = child.tag
        if tag in _SKIP:
            pass
        elif tag == _q("text", "s"):
            try:
                out.append(" " * max(int(child.get(_q("text", "c"), "1")), 1))
            except ValueError:
                out.append(" ")
        elif tag == _q("text", "tab"):
            out.append("\t")
        elif tag == _q("text", "line-break"):
            out.append("\n")
        else:
            out.append(_inline(child))
        out.append(child.tail or "")
    return "".join(out)


def _cell_text(cell) -> str:
    parts = [_inline(p).strip() for p in cell.iter() if p.tag in (_P, _H)]
    return " ".join(x for x in parts if x).replace("|", "/")


def _render_table(table, lines):
    for row in table.iter(_q("table", "table-row")):
        cells = [_cell_text(c) for c in row if c.tag in (_q("table", "table-cell"), _q("table", "covered-table-cell"))]
        if any(cells):
            lines.append(" | ".join(cells))


def _walk(elem, lines, depth=0):
    for child in elem:
        tag = child.tag
        if tag in _SKIP:
            continue
        if tag == _H:
            try:
                level = min(max(int(child.get(_q("text", "outline-level"), "1")), 1), 6)
            except ValueError:
                level = 1
            lines.append("#" * level + " " + _inline(child).strip())
        elif tag == _P:
            lines.append(_inline(child))
        elif tag == _q("text", "list"):
            _render_list(child, lines, depth)
        elif tag == _q("table", "table"):
            _render_table(child, lines)
        elif tag in (_q("text", "section"), _q("text", "list-item")):
            _walk(child, lines, depth)


def _render_list(lst, lines, depth):
    for item in lst:
        if item.tag != _q("text", "list-item"):
            continue
        for sub in item:
            if sub.tag in (_P, _H):
                lines.append("  " * depth + "- " + _inline(sub).strip())
            elif sub.tag == _q("text", "list"):
                _render_list(sub, lines, depth + 1)


def odt_to_text(data: bytes) -> str:
    # lazy import: cycle (utils.file_processor imports this module); reuses the zip-safety constants
    from utils.file_processor import MAX_FILE_SIZE, ZIP_MAX_RATIO

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("not a valid .odt (zip) file")
    with zf:
        try:
            info = zf.getinfo("content.xml")
        except KeyError:
            raise ValueError("no content.xml in .odt (damaged or not an OpenDocument file)")
        if info.flag_bits & 0x1:
            raise ValueError("encrypted .odt cannot be read")
        if info.file_size > MAX_FILE_SIZE:
            raise ValueError(f"content.xml too large ({info.file_size} bytes)")
        if info.file_size and (info.compress_size == 0 or info.file_size / info.compress_size > ZIP_MAX_RATIO):
            raise ValueError("compression ratio too high (possible zip bomb)")
        try:
            raw = zf.read("content.xml")
        except Exception as e:  # degrades: unreadable member becomes an error entry via ValueError
            raise ValueError(f"content.xml unreadable: {e}")
    # ElementTree never fetches external entities; DOCTYPE/ENTITY declarations are refused
    # outright so internal entity expansion (billion laughs) cannot happen either.
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("content.xml contains a DTD/entity declaration; refused")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise ValueError(f"content.xml is not valid XML: {e}")
    body = root.find(_q("office", "body"))
    text_root = body.find(_q("office", "text")) if body is not None else None
    if text_root is None:
        raise ValueError("no document text found in content.xml")
    lines: list = []
    _walk(text_root, lines)
    return "\n".join(lines).strip()
