"""The per-occurrence review ledger: ``config/bug_class_dispositions.json`` (schema 1).

Every active baseline occurrence has exactly one ``accepted_debt`` record,
bound to the SHA-256 of its source file as reviewed.  A fixed or disproved
candidate keeps its record as history with ``confirmed_fixed`` (which must
name the source-owner request it came through) or ``false_positive``, and no
longer has a baseline row.  So accepted debt is visibly distinct from a
confirmed fix, and a fixed row cannot stay behind as green-laundered debt.

Validation is two-level:

* ``load`` rejects a structurally invalid ledger (``DispositionsError``, the
  scan exits 2): wrong schema, unknown or missing fields, bad statuses,
  ordinals, dates, hashes or anchors, duplicate occurrences, or legacy
  records without a legacy snapshot.
* ``reconcile`` compares a valid ledger to the baseline and the tree (the
  scan exits 1): an undisposed occurrence, an accepted record with no row
  (orphan), a changed or missing source file (re-review needed), and — on a
  full scan — a legacy history that no longer re-renders to the exact
  schema-1 baseline bytes recorded when it was migrated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import baseline as baseline_mod
from .scanners.common import ANCHOR_VERSION

SCHEMA = 1
DEFAULT_DISPOSITIONS = "config/bug_class_dispositions.json"
STATUSES = ("accepted_debt", "confirmed_fixed", "false_positive")
ASSESSMENTS = ("product_risk", "reviewed_benign", "uncertain")
RECORD_FIELDS = (
    "anchor", "ordinal", "legacy", "source_sha256", "status", "assessment",
    "rationale", "reviewer", "review_date", "evidence", "request",
)
LEGACY_FIELDS = ("scanner", "path", "symbol", "text", "ordinal")
LEGACY_BASELINE_FIELDS = ("schema", "sha256", "occurrences")
REQUEST_RE = re.compile(r"^CGR-[0-9]{8}-[0-9]{3}(?:-[0-9]+)?$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class DispositionsError(RuntimeError):
    """The ledger is missing or structurally invalid."""


@dataclass(frozen=True)
class Record:
    anchor: tuple[str, str, str, str, str]
    ordinal: int
    legacy: tuple[str, str, str, str] | None
    legacy_ordinal: int | None
    source_sha256: str
    status: str
    assessment: str
    rationale: str
    reviewer: str
    review_date: str
    evidence: str
    request: str | None


@dataclass(frozen=True)
class Ledger:
    records: tuple[Record, ...]
    legacy_sha256: str | None
    legacy_occurrences: int | None

    def count(self, status: str) -> int:
        return sum(1 for record in self.records if record.status == status)

    @property
    def legacy_records(self) -> int:
        return sum(1 for record in self.records if record.legacy is not None)


@dataclass(frozen=True)
class Problem:
    code: str
    detail: str


def _positive(value: object) -> bool:
    return type(value) is int and value >= 1


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _record(
    index: int,
    raw: object,
    gate_kinds: Mapping[str, Sequence[str]],
    known: Sequence[str],
    has_legacy: bool,
) -> Record:
    where = f"record {index}"
    if not isinstance(raw, dict) or set(raw) != set(RECORD_FIELDS):
        missing = sorted(set(RECORD_FIELDS) - set(raw)) if isinstance(raw, dict) else []
        extra = sorted(set(raw) - set(RECORD_FIELDS)) if isinstance(raw, dict) else []
        raise DispositionsError(f"{where} must have exactly {', '.join(RECORD_FIELDS)} (missing {missing}, extra {extra})")
    problem = baseline_mod.anchor_problem(raw["anchor"], gate_kinds, known)
    if problem:
        raise DispositionsError(f"{where} anchor {problem}")
    if not _positive(raw["ordinal"]):
        raise DispositionsError(f"{where} ordinal must be a positive integer")
    legacy = legacy_ordinal = None
    if raw["legacy"] is not None:
        item = raw["legacy"]
        if not has_legacy:
            raise DispositionsError(f"{where} has legacy history but the ledger has no legacy_baseline")
        if not isinstance(item, dict) or set(item) != set(LEGACY_FIELDS):
            raise DispositionsError(f"{where} legacy must have exactly {', '.join(LEGACY_FIELDS)}")
        if not all(isinstance(item[name], str) for name in ("scanner", "path", "symbol", "text")):
            raise DispositionsError(f"{where} legacy fields must be strings")
        if not (item["scanner"] and item["path"] and item["text"]) or not _positive(item["ordinal"]):
            raise DispositionsError(f"{where} legacy key is incomplete")
        legacy = (item["scanner"], item["path"], item["symbol"], item["text"])
        legacy_ordinal = item["ordinal"]
    if not isinstance(raw["source_sha256"], str) or not SHA_RE.match(raw["source_sha256"]):
        raise DispositionsError(f"{where} source_sha256 must be 64 lowercase hex characters")
    if raw["status"] not in STATUSES:
        raise DispositionsError(f"{where} status {raw['status']!r} is not one of {', '.join(STATUSES)}")
    if raw["assessment"] not in ASSESSMENTS:
        raise DispositionsError(f"{where} assessment {raw['assessment']!r} is not one of {', '.join(ASSESSMENTS)}")
    for name in ("rationale", "reviewer", "evidence"):
        if not _text(raw[name]):
            raise DispositionsError(f"{where} {name} must be a non-empty string")
    review_date = raw["review_date"]
    try:
        valid_date = isinstance(review_date, str) and bool(DATE_RE.match(review_date)) and bool(date.fromisoformat(review_date))
    except ValueError:
        valid_date = False
    if not valid_date:
        raise DispositionsError(f"{where} review_date must be YYYY-MM-DD")
    request = raw["request"]
    if request is not None and (not isinstance(request, str) or not REQUEST_RE.match(request)):
        raise DispositionsError(f"{where} request {request!r} is not a CGR-YYYYMMDD-NNN id")
    if raw["status"] == "confirmed_fixed" and request is None:
        raise DispositionsError(f"{where} confirmed_fixed must name the source-owner request")
    return Record(
        baseline_mod.anchor_tuple(raw["anchor"]), raw["ordinal"], legacy, legacy_ordinal,
        raw["source_sha256"], raw["status"], raw["assessment"], raw["rationale"],
        raw["reviewer"], review_date, raw["evidence"], request,
    )


def _contiguous(groups: Mapping[object, list[int]], label: str) -> None:
    for key, ordinals in groups.items():
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise DispositionsError(f"{label} ordinals for {key} must be 1..{len(ordinals)} without repeats, got {sorted(ordinals)}")


def load(
    path: Path,
    gate_kinds: Mapping[str, Sequence[str]],
    known_scanners: Iterable[str],
) -> Ledger:
    if not path.is_file():
        raise DispositionsError(f"dispositions missing: {path}")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DispositionsError(f"cannot read dispositions {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "anchor", "legacy_baseline", "records"}:
        raise DispositionsError(f"{path} must have exactly schema, anchor, legacy_baseline, records")
    if type(payload["schema"]) is not int or payload["schema"] != SCHEMA:
        raise DispositionsError(f"unsupported dispositions schema {payload['schema']!r}")
    if payload["anchor"] != ANCHOR_VERSION:
        raise DispositionsError(f"dispositions anchor version {payload['anchor']!r} is not {ANCHOR_VERSION}")
    legacy = payload["legacy_baseline"]
    legacy_sha = legacy_count = None
    if legacy is not None:
        if not isinstance(legacy, dict) or set(legacy) != set(LEGACY_BASELINE_FIELDS):
            raise DispositionsError("legacy_baseline must be null or have exactly schema, sha256, occurrences")
        if type(legacy["schema"]) is not int or legacy["schema"] != baseline_mod.LEGACY_SCHEMA:
            raise DispositionsError("legacy_baseline.schema must be 1")
        if not isinstance(legacy["sha256"], str) or not SHA_RE.match(legacy["sha256"]):
            raise DispositionsError("legacy_baseline.sha256 must be 64 lowercase hex characters")
        if not _positive(legacy["occurrences"]):
            raise DispositionsError("legacy_baseline.occurrences must be a positive integer")
        legacy_sha, legacy_count = legacy["sha256"], legacy["occurrences"]
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise DispositionsError("records must be a list")
    known = list(known_scanners)
    records = tuple(
        _record(index, raw, gate_kinds, known, legacy is not None)
        for index, raw in enumerate(raw_records)
    )
    by_anchor: dict[object, list[int]] = {}
    by_legacy: dict[object, list[int]] = {}
    for record in records:
        by_anchor.setdefault(record.anchor, []).append(record.ordinal)
        if record.legacy is not None:
            by_legacy.setdefault(record.legacy, []).append(record.legacy_ordinal)
    _contiguous(by_anchor, "anchor")
    _contiguous(by_legacy, "legacy")
    return Ledger(records, legacy_sha, legacy_count)


def reconcile(
    ledger: Ledger,
    stored: Counter,
    root: Path,
    compared_scanners: Iterable[str],
    *,
    full_scope: bool,
) -> list[Problem]:
    compared = set(compared_scanners)
    problems: list[Problem] = []
    accepted: Counter = Counter(
        record.anchor
        for record in ledger.records
        if record.status == "accepted_debt" and record.anchor[0] in compared
    )
    rows = baseline_mod.subset(stored, compared)
    for anchor in sorted(rows):
        missing = rows[anchor] - accepted[anchor]
        if missing > 0:
            problems.append(Problem(
                "disposition_missing",
                f"{missing} baseline occurrence(s) of {_describe(anchor)} have no accepted_debt review",
            ))
    for anchor in sorted(accepted):
        extra = accepted[anchor] - rows[anchor]
        if extra > 0:
            problems.append(Problem(
                "disposition_orphan",
                f"{extra} accepted_debt record(s) for {_describe(anchor)} have no baseline occurrence",
            ))
    hashes: dict[str, str | None] = {}
    for record in ledger.records:
        if record.status != "accepted_debt" or record.anchor[0] not in compared:
            continue
        rel = record.anchor[1]
        if rel not in hashes:
            target = root / rel
            hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        current = hashes[rel]
        if current != record.source_sha256:
            state = "is missing" if current is None else f"now hashes {current}"
            problems.append(Problem(
                "disposition_source_changed",
                f"{rel} {state}; accepted_debt for {_describe(record.anchor)} #{record.ordinal} "
                f"was reviewed at {record.source_sha256} — re-review before retaining it",
            ))
    if full_scope and ledger.legacy_sha256 is not None:
        history = Counter(record.legacy for record in ledger.records if record.legacy is not None)
        rendered = baseline_mod.legacy_render(history).encode("utf-8")
        digest = hashlib.sha256(rendered).hexdigest()
        occurrences = sum(history.values())
        if digest != ledger.legacy_sha256 or occurrences != ledger.legacy_occurrences:
            problems.append(Problem(
                "legacy_reconciliation",
                f"legacy history re-renders to {occurrences} occurrence(s), sha256 {digest}; "
                f"the migrated schema-1 baseline was {ledger.legacy_occurrences} at {ledger.legacy_sha256}",
            ))
    return problems


def _describe(anchor: Sequence[str]) -> str:
    scanner, path, symbol, kind, digest = anchor
    return f"{scanner}:{path}[{symbol}] {kind} {digest[:12]}"
