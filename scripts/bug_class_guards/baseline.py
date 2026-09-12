"""Content-anchored baseline for the bug-class scanners.

The baseline is a MULTISET of fingerprints, not a set of line numbers: the
anchoring lesson from ``tests/unit/test_ordered_slice_guard.py`` (a690a91 went
red when unrelated hunks shifted allowlisted slices).  Two identical flagged
lines inside one function therefore need two entries — a third identical line
still surfaces as unexplained.

Ratchet semantics, both directions:
  * a fingerprint present in the tree and absent from the baseline is NEW
    (the change introduced an instance of the class);
  * a fingerprint present in the baseline and absent from the tree is STALE
    (the debt was fixed, or the anchor moved — either way the entry must be
    removed, so fixed debt can never be laundered by a leftover row).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .scanners.common import Finding

SCHEMA = 1
DEFAULT_BASELINE = "config/bug_class_baseline.json"

Fingerprint = tuple[str, str, str, str]


class BaselineError(RuntimeError):
    """The baseline file exists but cannot be used."""


@dataclass(frozen=True)
class Comparison:
    baselined: int
    new: list[Fingerprint]
    stale: list[Fingerprint]


def fingerprints(findings: Iterable[Finding]) -> Counter[Fingerprint]:
    return Counter(finding.fingerprint() for finding in findings)


def load(path: Path) -> Counter[Fingerprint]:
    if not path.exists():
        return Counter()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise BaselineError(f"unsupported baseline schema in {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise BaselineError(f"baseline {path} has no entries list")
    counter: Counter[Fingerprint] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise BaselineError(f"baseline {path} has a non-object entry")
        try:
            key = (
                str(entry["scanner"]),
                str(entry["path"]),
                str(entry["symbol"]),
                str(entry["text"]),
            )
        except KeyError as exc:
            raise BaselineError(f"baseline {path} entry missing {exc}") from exc
        counter[key] += 1
    return counter


def render(counter: Counter[Fingerprint]) -> str:
    entries = []
    for key in sorted(counter):
        scanner, path, symbol, text = key
        for _ in range(counter[key]):
            entries.append(
                {"scanner": scanner, "path": path, "symbol": symbol, "text": text}
            )
    payload = {"schema": SCHEMA, "entries": entries}
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def save(path: Path, counter: Counter[Fingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(counter), encoding="utf-8")


def compare(
    found: Counter[Fingerprint], baseline: Counter[Fingerprint]
) -> Comparison:
    new_counter = found - baseline
    stale_counter = baseline - found
    matched = sum((found & baseline).values())
    return Comparison(matched, _expand(new_counter), _expand(stale_counter))


def _expand(counter: Counter[Fingerprint]) -> list[Fingerprint]:
    out: list[Fingerprint] = []
    for key in sorted(counter):
        out.extend([key] * counter[key])
    return out


def subset(counter: Counter[Fingerprint], scanner_ids: Sequence[str]) -> Counter[Fingerprint]:
    allowed = set(scanner_ids)
    return Counter({k: v for k, v in counter.items() if k[0] in allowed})
