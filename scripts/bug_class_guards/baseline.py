"""Content-anchored baseline for the gate scanners (schema 2).

The baseline is a MULTISET of full candidate anchors — ``(scanner, path,
qualname, kind, digest)`` — not a set of line numbers: the anchoring lesson
from ``tests/unit/test_ordered_slice_guard.py`` (a690a91 went red when
unrelated hunks shifted allowlisted slices).  Two identical candidates inside
one function therefore need two rows; a third identical one still surfaces
as NEW.  Each row also carries a bounded ``excerpt`` for humans; it is not
part of the identity.

Ratchet semantics, both directions:
  * an anchor present in the tree and absent from the baseline is NEW;
  * an anchor present in the baseline and absent from the tree is STALE
    (the debt was fixed, or the candidate changed) — the row must be removed,
    so fixed debt can never be laundered by a leftover row.

The loader fails closed: a missing file is an error even when the tree has no
finding, and any structural defect — wrong schema type or value, unknown or
missing fields, empty or non-string anchor parts, unknown or report-only
scanners, undeclared kinds, malformed digests, non-canonical row order —
raises ``BaselineError``.  Rows are written only by an explicit
``--write-baseline`` bootstrap, which still needs reviewed dispositions.

The schema-1 reader and renderer remain for one purpose: proving the reviewed
2026-09-13 migration mapped every legacy occurrence.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .policy import is_repo_relpath
from .scanners.common import ANCHOR_VERSION, Finding

SCHEMA = 2
LEGACY_SCHEMA = 1
DEFAULT_BASELINE = "config/bug_class_baseline.json"

Anchor = tuple[str, str, str, str, str]
LegacyFingerprint = tuple[str, str, str, str]
ANCHOR_FIELDS = ("scanner", "path", "symbol", "kind", "digest")
ENTRY_FIELDS = ANCHOR_FIELDS + ("excerpt",)
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class BaselineError(RuntimeError):
    """The baseline file is missing or cannot be used."""


@dataclass(frozen=True)
class Baseline:
    counter: Counter
    excerpts: Mapping[Anchor, str] = field(default_factory=dict)

    @property
    def occurrences(self) -> int:
        return sum(self.counter.values())

    def by_scanner(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for anchor, count in self.counter.items():
            counts[anchor[0]] = counts.get(anchor[0], 0) + count
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class Comparison:
    matched: int
    new: list[Anchor]
    stale: list[Anchor]


def anchor_problem(
    entry: object,
    gate_kinds: Mapping[str, Sequence[str]],
    known_scanners: Iterable[str],
    fields: Sequence[str] = ANCHOR_FIELDS,
) -> str | None:
    """Why ``entry`` is not a valid anchor object, or None."""
    if not isinstance(entry, dict):
        return "is not an object"
    missing = [name for name in fields if name not in entry]
    extra = sorted(set(entry) - set(fields))
    if missing:
        return f"is missing {', '.join(missing)}"
    if extra:
        return f"has unknown field(s) {', '.join(extra)}"
    for name in ANCHOR_FIELDS:
        value = entry[name]
        if not isinstance(value, str) or not value:
            return f"field {name} must be a non-empty string"
    scanner = entry["scanner"]
    if scanner not in gate_kinds:
        if scanner in set(known_scanners):
            return f"names report-only scanner {scanner}, which never has baseline rows"
        return f"names unknown scanner {scanner}"
    if entry["kind"] not in gate_kinds[scanner]:
        return f"kind {entry['kind']!r} is not declared by {scanner}"
    if not is_repo_relpath(entry["path"]):
        return f"path {entry['path']!r} is not repository-relative"
    if not DIGEST_RE.match(entry["digest"]):
        return "digest must be 64 lowercase hex characters"
    return None


def anchor_tuple(entry: Mapping[str, str]) -> Anchor:
    return tuple(entry[name] for name in ANCHOR_FIELDS)  # type: ignore[return-value]


def load(
    path: Path,
    gate_kinds: Mapping[str, Sequence[str]],
    known_scanners: Iterable[str],
) -> Baseline:
    if not path.is_file():
        raise BaselineError(f"baseline missing: {path} (a missing baseline is never 'no debt')")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "anchor", "entries"}:
        raise BaselineError(f"baseline {path} must be an object with exactly schema, anchor, entries")
    if type(payload["schema"]) is not int or payload["schema"] != SCHEMA:
        raise BaselineError(f"unsupported baseline schema {payload['schema']!r} in {path} (expected {SCHEMA})")
    if payload["anchor"] != ANCHOR_VERSION:
        raise BaselineError(f"baseline {path} anchor version {payload['anchor']!r} is not {ANCHOR_VERSION}")
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise BaselineError(f"baseline {path} entries must be a list")
    known = list(known_scanners)
    counter: Counter = Counter()
    excerpts: dict[Anchor, str] = {}
    previous: Anchor | None = None
    for index, entry in enumerate(entries):
        problem = anchor_problem(entry, gate_kinds, known, ENTRY_FIELDS)
        if problem is None and not isinstance(entry["excerpt"], str):
            problem = "field excerpt must be a string"
        if problem:
            raise BaselineError(f"baseline {path} entry {index} {problem}")
        anchor = anchor_tuple(entry)
        if previous is not None and anchor < previous:
            raise BaselineError(f"baseline {path} entry {index} is out of canonical sorted order")
        previous = anchor
        counter[anchor] += 1
        excerpts.setdefault(anchor, entry["excerpt"])
    return Baseline(counter, excerpts)


def from_findings(findings: Iterable[Finding]) -> Baseline:
    counter: Counter = Counter()
    excerpts: dict[Anchor, str] = {}
    for finding in findings:
        anchor = finding.fingerprint()
        counter[anchor] += 1
        excerpts.setdefault(anchor, finding.excerpt)
    return Baseline(counter, excerpts)


def render(baseline: Baseline) -> str:
    entries = []
    for anchor in sorted(baseline.counter):
        row = dict(zip(ANCHOR_FIELDS, anchor))
        row["excerpt"] = baseline.excerpts.get(anchor, "")
        entries.extend(dict(row) for _ in range(baseline.counter[anchor]))
    payload = {"schema": SCHEMA, "anchor": ANCHOR_VERSION, "entries": entries}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def save(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(baseline), encoding="utf-8")


def compare(found: Counter, stored: Counter) -> Comparison:
    new_counter = found - stored
    stale_counter = stored - found
    matched = sum((found & stored).values())
    return Comparison(matched, _expand(new_counter), _expand(stale_counter))


def _expand(counter: Counter) -> list:
    out: list = []
    for key in sorted(counter):
        out.extend([key] * counter[key])
    return out


def subset(counter: Counter, scanner_ids: Iterable[str]) -> Counter:
    allowed = set(scanner_ids)
    return Counter({key: count for key, count in counter.items() if key[0] in allowed})


# ------------------------------------------------- schema 1 (migration only)


def legacy_load(path: Path) -> Counter:
    """The 2026-09-11 schema-1 multiset of (scanner, path, symbol, text)."""
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BaselineError(f"cannot read legacy baseline {path}: {exc}") from exc
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload["schema"] != LEGACY_SCHEMA:
        raise BaselineError(f"{path} is not a schema-1 baseline")
    counter: Counter = Counter()
    for index, entry in enumerate(payload.get("entries") or []):
        if not isinstance(entry, dict) or set(entry) != {"scanner", "path", "symbol", "text"}:
            raise BaselineError(f"legacy entry {index} is malformed")
        if not all(isinstance(entry[name], str) for name in entry):
            raise BaselineError(f"legacy entry {index} has a non-string field")
        counter[(entry["scanner"], entry["path"], entry["symbol"], entry["text"])] += 1
    return counter


def legacy_render(counter: Counter) -> str:
    """Byte-identical to the schema-1 writer, so a digest proves reconciliation."""
    entries = []
    for key in sorted(counter):
        scanner, path, symbol, text = key
        for _ in range(counter[key]):
            entries.append({"scanner": scanner, "path": path, "symbol": symbol, "text": text})
    payload = {"schema": LEGACY_SCHEMA, "entries": entries}
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
