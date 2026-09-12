"""Strict, read-only parser for ``docs/BUG_CLASSES.md``.

This module deliberately parses the Markdown as a document contract rather
than importing any application code.  It discovers body IDs and index IDs
independently, so a catalog can report drift instead of silently accepting an
ID-only coverage claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable, Mapping


ID_RE = re.compile(r"^BC-(?P<number>0[1-9]|[1-9][0-9]+)$")
HEADING_RE = re.compile(r"^###\s+(?P<id>BC-[0-9]+)\s+(?P<title>.+?)\s*$")
INDEX_ROW_RE = re.compile(
    r"^\|\s*(?P<id>BC-[0-9]+)\s*\|(?P<class>.*?)\|(?P<family>.*?)\|"
    r"(?P<status>.*?)\|\s*$"
)
METHOD_ROW_RE = re.compile(
    r"^\|\s*(?P<id>(?:DM|CM)-[0-9]+)\s*\|(?P<name>.*?)\|"
    r"(?P<body>.*?)\|\s*$"
)
FIELD_RE = re.compile(
    r"^-\s+(Mechanism|Incidents|Find|Closure|Status):\s*(.*)$", re.MULTILINE
)
STATUS_RE = re.compile(r"^(closed|partial|recurs|open)\b", re.IGNORECASE)
METHOD_HEADING_RE = re.compile(
    r"^##\s+(?P<kind>Detection methods|Closure methods)\s+\((?P<rest>.*?)\)"
)
SINGLETON_HEADING = "## Unclassified singletons"

REQUIRED_FIELDS = ("Mechanism", "Incidents", "Find", "Closure", "Status")
VALID_STATUSES = frozenset({"closed", "partial", "recurs", "open"})


class CatalogParseError(ValueError):
    """Raised when the catalog cannot be decoded or structurally parsed."""


@dataclass(frozen=True)
class IndexRecord:
    id: str
    title: str
    family: str
    status: str
    line: int


@dataclass(frozen=True)
class MethodRecord:
    id: str
    name: str
    body: str
    line: int


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    title: str
    fields: Mapping[str, str]
    line: int
    raw: str

    @property
    def status(self) -> str:
        match = STATUS_RE.match(self.fields.get("Status", "").strip())
        return match.group(1).lower() if match else ""


@dataclass
class Catalog:
    text: str
    entries: dict[str, CatalogEntry]
    index: dict[str, IndexRecord]
    detection_methods: dict[str, MethodRecord]
    closure_methods: dict[str, MethodRecord]
    singleton_bullets: list[str] = field(default_factory=list)

    @property
    def methods(self) -> dict[str, MethodRecord]:
        return {**self.detection_methods, **self.closure_methods}


def _read_source(source: str | bytes | Path) -> str:
    if isinstance(source, bytes):
        try:
            return source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CatalogParseError(f"malformed UTF-8 at byte {exc.start}") from exc
    if isinstance(source, Path):
        try:
            return source.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CatalogParseError(f"malformed UTF-8 at byte {exc.start}") from exc
        except OSError as exc:
            raise CatalogParseError(f"cannot read catalog: {exc}") from exc
    return source


def _validate_id(value: str, kind: str, line: int) -> str:
    if not ID_RE.fullmatch(value):
        raise CatalogParseError(f"line {line}: malformed {kind} ID {value!r}")
    return value


def parse_index(text: str | bytes | Path) -> dict[str, IndexRecord]:
    """Parse index rows independently from the body entries."""
    source = _read_source(text)
    result: dict[str, IndexRecord] = {}
    for line_no, line in enumerate(source.splitlines(), 1):
        match = INDEX_ROW_RE.match(line)
        if not match:
            continue
        ident = _validate_id(match.group("id"), "index", line_no)
        status = match.group("status").strip().lower()
        if status not in VALID_STATUSES:
            # Keep the row so diagnostics can name the invalid value.
            status = f"!invalid:{status}"
        if ident in result:
            raise CatalogParseError(f"line {line_no}: duplicate index ID {ident}")
        result[ident] = IndexRecord(
            ident, match.group("class").strip(), match.group("family").strip(),
            status, line_no,
        )
    if not result:
        raise CatalogParseError("index discovery is empty")
    return result


def parse_methods(text: str | bytes | Path) -> dict[str, dict[str, MethodRecord]]:
    """Parse DM and CM tables independently of class entries."""
    source = _read_source(text)
    groups: dict[str, dict[str, MethodRecord]] = {"DM": {}, "CM": {}}
    current: str | None = None
    for line_no, line in enumerate(source.splitlines(), 1):
        heading = METHOD_HEADING_RE.match(line)
        if heading:
            current = "DM" if heading.group("kind") == "Detection methods" else "CM"
            continue
        match = METHOD_ROW_RE.match(line)
        if not match or current is None:
            continue
        ident = match.group("id")
        if not re.fullmatch(rf"{current}-[0-9]+", ident):
            raise CatalogParseError(f"line {line_no}: {ident} in {current} table")
        if ident in groups[current]:
            raise CatalogParseError(f"line {line_no}: duplicate method ID {ident}")
        groups[current][ident] = MethodRecord(
            ident, match.group("name").strip(), match.group("body").strip(), line_no
        )
    for kind, methods in groups.items():
        if not methods:
            raise CatalogParseError(f"{kind} discovery is empty")
    return groups


def _body_blocks(source: str) -> Iterable[tuple[str, str, int]]:
    lines = source.splitlines()
    starts: list[tuple[str, str, int, int]] = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match:
            if re.search(r"\bBC-[0-9]+\b", match.group("title")):
                raise CatalogParseError(f"line {index + 1}: malformed or merged class heading")
            starts.append((match.group("id"), match.group("title"), index, index + 1))
        elif line.startswith("### ") and "BC-" in line:
            raise CatalogParseError(f"line {index + 1}: malformed or merged class heading")
    for pos, (ident, title, start, body_start) in enumerate(starts):
        end = starts[pos + 1][2] if pos + 1 < len(starts) else len(lines)
        yield ident, title, body_start + 1, "\n".join(lines[start:end])


def parse_catalog(source: str | bytes | Path) -> Catalog:
    """Parse the full catalog and retain all text needed for diagnostics."""
    text = _read_source(source)
    if not text.strip():
        raise CatalogParseError("catalog is empty")
    entries: dict[str, CatalogEntry] = {}
    for ident, title, first_body_line, raw in _body_blocks(text):
        line = text[: text.find(raw)].count("\n") + 1
        if ident in entries:
            raise CatalogParseError(f"line {line}: duplicate body ID {ident}")
        fields: dict[str, str] = {}
        for field_match in FIELD_RE.finditer(raw):
            name, value = field_match.groups()
            if name in fields:
                raise CatalogParseError(f"line {line}: duplicate {name} field in {ident}")
            fields[name] = value.strip()
        missing = [name for name in REQUIRED_FIELDS if not fields.get(name)]
        if missing:
            raise CatalogParseError(
                f"line {first_body_line}: {ident} missing field(s): {', '.join(missing)}"
            )
        if not STATUS_RE.match(fields["Status"]):
            raise CatalogParseError(f"line {first_body_line}: {ident} has malformed status")
        entries[ident] = CatalogEntry(ident, title.strip(), fields, first_body_line, raw)

    index = parse_index(text)
    methods = parse_methods(text)
    lines = text.splitlines()
    singleton_bullets: list[str] = []
    in_singletons = False
    for line in lines:
        if line.strip().startswith(SINGLETON_HEADING):
            in_singletons = True
            continue
        if in_singletons and line.startswith("## "):
            break
        if in_singletons and line.startswith("-"):
            singleton_bullets.append(line[1:].strip())
    return Catalog(text, entries, index, methods["DM"], methods["CM"], singleton_bullets)


def incident_digest(entry: CatalogEntry | str) -> str:
    """Return a stable digest of the incident field's exact UTF-8 text."""
    incident = entry.fields["Incidents"] if isinstance(entry, CatalogEntry) else entry
    return sha256(incident.strip().encode("utf-8")).hexdigest()


def catalog_diagnostics(catalog: Catalog) -> list[str]:
    """Return admission findings without rewriting or normalizing the catalog."""
    findings: list[str] = []
    body_ids = set(catalog.entries)
    index_ids = set(catalog.index)
    for ident in sorted(index_ids - body_ids):
        findings.append(f"index/body disagreement: {ident} appears only in index")
    for ident in sorted(body_ids - index_ids):
        findings.append(f"index/body disagreement: {ident} appears only in body")
    for ident in sorted(body_ids & index_ids):
        entry_status = catalog.entries[ident].status
        index_status = catalog.index[ident].status
        if index_status.startswith("!invalid:"):
            findings.append(f"invalid index status for {ident}: {index_status[9:]}")
        elif entry_status != index_status:
            findings.append(
                f"index/body status disagreement: {ident} index={index_status} "
                f"body={entry_status}"
            )
        for field_name in REQUIRED_FIELDS:
            if not catalog.entries[ident].fields.get(field_name, "").strip():
                findings.append(f"missing {field_name} field: {ident}")
        for method_id in re.findall(r"\b(?:DM|CM)-[0-9]+\b", catalog.entries[ident].fields["Find"] + " " + catalog.entries[ident].fields["Closure"]):
            group = catalog.detection_methods if method_id.startswith("DM-") else catalog.closure_methods
            if method_id not in group:
                findings.append(f"unknown method reference: {ident} -> {method_id}")
    if not catalog.entries:
        findings.append("body discovery is empty")
    if not catalog.index:
        findings.append("index discovery is empty")
    if not catalog.detection_methods:
        findings.append("DM discovery is empty")
    if not catalog.closure_methods:
        findings.append("CM discovery is empty")
    return findings
