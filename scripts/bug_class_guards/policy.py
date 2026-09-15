"""The enforcement contract: ``config/bug_class_policy.json`` (schema 1).

The policy is the human-reviewed statement of what the scan enforces: the
exact scanner IDs, modes, classes and contract versions; the input legs and
the roots each owns; the classification of every top-level Python root; and
the harness/guard receipts CI must produce.  The registry in
``bug_class_guards.scanners`` is the implementation.  ``registry_mismatches``
compares the two in full, so a deleted, renamed, re-moded, re-classed or
unreviewed scanner — or a narrowed leg — stops the scan before any scanner
runs.  Neither file makes the other tamper-proof: changing both together is
an ordinary reviewed code change.

``evaluate_inventory`` inspects only top-level names and the direct ``*.py``
presence of unknown directories.  Excluded and pattern-matched private roots
are never listed, and unknown hidden directories are reported without being
opened.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .scanners.common import EXCLUDED_DIR_NAMES, LEG_KINDS, Exemption, Leg, Scanner

POLICY_PATH = "config/bug_class_policy.json"
SCHEMA = 1
MODES = ("gate", "report")

_CLASS_ID_RE = re.compile(r"^BC-(0[1-9]|[1-9][0-9]+)$")
_SCANNER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_TOP_KEYS = {
    "schema", "policy_version", "claim", "paths", "walk_exclusions", "scanners",
    "legs", "inventory", "harness", "repo_wide_guards",
}


class PolicyError(ValueError):
    """The policy file is missing, unreadable, or violates schema 1."""


@dataclass(frozen=True)
class ScannerContract:
    id: str
    mode: str
    class_ids: tuple[str, ...]
    contract_version: int
    legs: tuple[str, ...]
    kinds: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "class_ids": list(self.class_ids),
            "contract_version": self.contract_version,
            "legs": list(self.legs),
            "kinds": list(self.kinds),
        }


@dataclass(frozen=True)
class Classified:
    name: str
    reason: str = ""


@dataclass(frozen=True)
class Policy:
    version: str
    claim: str
    sha256: str
    baseline_path: str
    dispositions_path: str
    catalog_path: str
    walk_exclusions: tuple[str, ...]
    scanners: tuple[ScannerContract, ...]
    legs: Mapping[str, Leg]
    unscanned_python_roots: tuple[Classified, ...]
    non_python_roots: tuple[str, ...]
    excluded_roots: tuple[Classified, ...]
    excluded_patterns: tuple[Classified, ...]
    harness_path: str
    expected_test_cases: int
    repo_wide_guards: tuple[str, ...]

    def scanner(self, scanner_id: str) -> ScannerContract:
        for contract in self.scanners:
            if contract.id == scanner_id:
                return contract
        raise KeyError(scanner_id)

    @property
    def scanner_ids(self) -> tuple[str, ...]:
        return tuple(contract.id for contract in self.scanners)

    @property
    def scanned_roots(self) -> frozenset[str]:
        """Top-level names read by at least one Python leg."""
        return frozenset(
            root.split("/", 1)[0]
            for leg in self.legs.values()
            if leg.is_python
            for root in leg.roots
        )


# -------------------------------------------------------------- validation


def is_repo_relpath(value: object) -> bool:
    """A non-empty POSIX path inside the repository, without ``.``/``..``."""
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if value.startswith("/") or "\\" in value or value.endswith("/"):
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _fail(where: str, message: str) -> None:
    raise PolicyError(f"{where}: {message}")


def _object(value: Any, keys: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing:
        _fail(where, f"missing field(s) {', '.join(missing)}")
    if extra:
        _fail(where, f"unknown field(s) {', '.join(extra)}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be a non-empty string")
    return value


def _flag(value: Any, where: str) -> bool:
    if type(value) is not bool:
        _fail(where, "must be true or false")
    return value


def _count(value: Any, where: str) -> int:
    if type(value) is not int or value < 1:
        _fail(where, "must be a positive integer")
    return value


def _unique_list(value: Any, where: str, *, allow_empty: bool = False) -> list:
    if not isinstance(value, list) or (not value and not allow_empty):
        _fail(where, "must be a non-empty list" if not allow_empty else "must be a list")
    seen = set()
    for item in value:
        marker = json.dumps(item, sort_keys=True)
        if marker in seen:
            _fail(where, f"duplicate entry {item!r}")
        seen.add(marker)
    return value


def _segment(value: Any, where: str) -> str:
    name = _text(value, where)
    if "/" in name or name in {".", ".."}:
        _fail(where, f"{name!r} must be a single top-level name")
    return name


def _parse_leg(raw: Any, index: int) -> Leg:
    where = f"legs[{index}]"
    raw = _object(raw, {"id", "kind", "roots", "required", "exempt"}, where)
    leg_id = _text(raw["id"], f"{where}.id")
    if not _SCANNER_ID_RE.match(leg_id):
        _fail(f"{where}.id", f"malformed leg id {leg_id!r}")
    kind = raw["kind"]
    if kind not in LEG_KINDS:
        _fail(f"{where}.kind", f"unknown leg kind {kind!r}")
    roots = _unique_list(raw["roots"], f"{where}.roots")
    for root in roots:
        if not is_repo_relpath(root):
            _fail(f"{where}.roots", f"{root!r} is not a repository-relative path")
    if kind == "file" and len(roots) != 1:
        _fail(f"{where}.roots", "a file leg has exactly one root")
    required = _flag(raw["required"], f"{where}.required")
    exempt = []
    for number, item in enumerate(_unique_list(raw["exempt"], f"{where}.exempt", allow_empty=True)):
        item = _object(item, {"path", "reason"}, f"{where}.exempt[{number}]")
        path = item["path"]
        if not is_repo_relpath(path) or not any(
            path == root or path.startswith(root + "/") for root in roots
        ):
            _fail(f"{where}.exempt[{number}].path", f"{path!r} is not under the leg's roots")
        exempt.append(Exemption(path, _text(item["reason"], f"{where}.exempt[{number}].reason")))
    return Leg(leg_id, kind, tuple(roots), required, tuple(exempt))


def _parse_scanner(raw: Any, index: int, legs: Mapping[str, Leg]) -> ScannerContract:
    where = f"scanners[{index}]"
    raw = _object(raw, {"id", "mode", "class_ids", "contract_version", "legs", "kinds"}, where)
    scanner_id = _text(raw["id"], f"{where}.id")
    if not _SCANNER_ID_RE.match(scanner_id):
        _fail(f"{where}.id", f"malformed scanner id {scanner_id!r}")
    if raw["mode"] not in MODES:
        _fail(f"{where}.mode", f"unknown mode {raw['mode']!r}")
    class_ids = _unique_list(raw["class_ids"], f"{where}.class_ids")
    for class_id in class_ids:
        if not isinstance(class_id, str) or not _CLASS_ID_RE.match(class_id):
            _fail(f"{where}.class_ids", f"malformed class id {class_id!r}")
    version = _count(raw["contract_version"], f"{where}.contract_version")
    leg_ids = _unique_list(raw["legs"], f"{where}.legs")
    for leg_id in leg_ids:
        if leg_id not in legs:
            _fail(f"{where}.legs", f"unknown leg {leg_id!r}")
    kinds = _unique_list(raw["kinds"], f"{where}.kinds")
    for kind in kinds:
        if not isinstance(kind, str) or not _KIND_RE.match(kind):
            _fail(f"{where}.kinds", f"malformed kind {kind!r}")
    return ScannerContract(
        scanner_id, raw["mode"], tuple(class_ids), version, tuple(leg_ids), tuple(kinds)
    )


def parse_policy(payload: Any, sha256: str = "") -> Policy:
    top = _object(payload, _TOP_KEYS, "policy")
    if type(top["schema"]) is not int or top["schema"] != SCHEMA:
        _fail("policy.schema", f"unsupported schema {top['schema']!r} (expected {SCHEMA})")
    version = _text(top["policy_version"], "policy.policy_version")
    claim = _text(top["claim"], "policy.claim")
    paths = _object(top["paths"], {"baseline", "dispositions", "catalog"}, "policy.paths")
    for key, value in paths.items():
        if not is_repo_relpath(value):
            _fail(f"policy.paths.{key}", f"{value!r} is not a repository-relative path")
    exclusions = [
        _segment(name, "policy.walk_exclusions")
        for name in _unique_list(top["walk_exclusions"], "policy.walk_exclusions")
    ]

    legs: dict[str, Leg] = {}
    for index, raw in enumerate(_unique_list(top["legs"], "policy.legs")):
        leg = _parse_leg(raw, index)
        if leg.id in legs:
            _fail(f"legs[{index}].id", f"duplicate leg id {leg.id!r}")
        legs[leg.id] = leg

    scanners: list[ScannerContract] = []
    for index, raw in enumerate(_unique_list(top["scanners"], "policy.scanners")):
        contract = _parse_scanner(raw, index, legs)
        if any(existing.id == contract.id for existing in scanners):
            _fail(f"scanners[{index}].id", f"duplicate scanner id {contract.id!r}")
        scanners.append(contract)
    used = {leg_id for contract in scanners for leg_id in contract.legs}
    unused = sorted(set(legs) - used)
    if unused:
        _fail("policy.legs", f"leg(s) no scanner reads: {', '.join(unused)}")

    inventory = _object(
        top["inventory"],
        {"unscanned_python_roots", "non_python_roots", "excluded_roots", "excluded_patterns"},
        "policy.inventory",
    )
    unscanned = tuple(
        Classified(
            _segment(_object(item, {"name", "reason"}, "inventory.unscanned_python_roots")["name"], "inventory.unscanned_python_roots"),
            _text(item["reason"], "inventory.unscanned_python_roots.reason"),
        )
        for item in _unique_list(inventory["unscanned_python_roots"], "inventory.unscanned_python_roots", allow_empty=True)
    )
    non_python = tuple(
        _segment(_object(item, {"name"}, "inventory.non_python_roots")["name"], "inventory.non_python_roots")
        for item in _unique_list(inventory["non_python_roots"], "inventory.non_python_roots", allow_empty=True)
    )
    excluded = tuple(
        Classified(
            _segment(_object(item, {"name", "reason"}, "inventory.excluded_roots")["name"], "inventory.excluded_roots"),
            _text(item["reason"], "inventory.excluded_roots.reason"),
        )
        for item in _unique_list(inventory["excluded_roots"], "inventory.excluded_roots", allow_empty=True)
    )
    patterns = tuple(
        Classified(
            _text(_object(item, {"pattern", "reason"}, "inventory.excluded_patterns")["pattern"], "inventory.excluded_patterns"),
            _text(item["reason"], "inventory.excluded_patterns.reason"),
        )
        for item in _unique_list(inventory["excluded_patterns"], "inventory.excluded_patterns", allow_empty=True)
    )

    harness = _object(top["harness"], {"path", "expected_test_cases"}, "policy.harness")
    if not is_repo_relpath(harness["path"]):
        _fail("policy.harness.path", "must be a repository-relative path")
    expected = _count(harness["expected_test_cases"], "policy.harness.expected_test_cases")
    guards = _unique_list(top["repo_wide_guards"], "policy.repo_wide_guards")
    for guard in guards:
        if not is_repo_relpath(guard) or not guard.endswith(".py"):
            _fail("policy.repo_wide_guards", f"{guard!r} is not a Python test path")

    policy = Policy(
        version, claim, sha256, paths["baseline"], paths["dispositions"], paths["catalog"],
        tuple(exclusions), tuple(scanners), legs, unscanned, non_python, excluded, patterns,
        harness["path"], expected, tuple(guards),
    )
    classes = [policy.scanned_roots, {c.name for c in unscanned}, set(non_python), {c.name for c in excluded}]
    labels = ["scanned", "unscanned_python_roots", "non_python_roots", "excluded_roots"]
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            overlap = sorted(set(classes[i]) & set(classes[j]))
            if overlap:
                _fail("policy.inventory", f"{', '.join(overlap)} classified as both {labels[i]} and {labels[j]}")
    return policy


def load_policy(path: Path) -> Policy:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PolicyError(f"policy missing: {path}") from exc
    except OSError as exc:
        raise PolicyError(f"cannot read policy {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolicyError(f"policy {path} is not valid JSON: {exc}") from exc
    return parse_policy(payload, hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------- registry parity


def observed_contract(scanner: Scanner) -> ScannerContract:
    return ScannerContract(
        scanner.id, scanner.mode, tuple(scanner.class_ids), scanner.contract_version,
        tuple(leg.id for leg in scanner.legs), tuple(scanner.kinds),
    )


def registry_mismatches(policy: Policy, registry: Mapping[str, Scanner]) -> list[str]:
    """Every difference between the reviewed contract and the implementation."""
    problems: list[str] = []
    expected_ids = list(policy.scanner_ids)
    for missing in [sid for sid in expected_ids if sid not in registry]:
        problems.append(f"scanner {missing} is required by the policy but absent from the registry")
    for extra in [sid for sid in registry if sid not in expected_ids]:
        problems.append(f"scanner {extra} is in the registry but not reviewed in the policy")
    for key, scanner in registry.items():
        if key != scanner.id:
            problems.append(f"registry key {key} holds scanner {scanner.id}")
    for contract in policy.scanners:
        scanner = registry.get(contract.id)
        if scanner is None:
            continue
        observed = observed_contract(scanner)
        for field_name in ("mode", "class_ids", "contract_version", "legs", "kinds"):
            want, got = getattr(contract, field_name), getattr(observed, field_name)
            if want != got:
                problems.append(f"{contract.id}.{field_name}: policy {want!r}, registry {got!r}")
        for leg in scanner.legs:
            declared = policy.legs.get(leg.id)
            if declared is None:
                continue
            if (leg.kind, leg.roots, leg.required) != (declared.kind, declared.roots, declared.required):
                problems.append(
                    f"{contract.id} leg {leg.id}: policy {declared.kind} {list(declared.roots)} "
                    f"required={declared.required}, registry {leg.kind} {list(leg.roots)} required={leg.required}"
                )
            if leg.exempt_paths != declared.exempt_paths:
                problems.append(
                    f"{contract.id} leg {leg.id}: exemptions differ "
                    f"(policy {sorted(declared.exempt_paths)}, registry {sorted(leg.exempt_paths)})"
                )
    if set(policy.walk_exclusions) != set(EXCLUDED_DIR_NAMES):
        problems.append(
            f"walk exclusions: policy {sorted(policy.walk_exclusions)}, "
            f"implementation {sorted(EXCLUDED_DIR_NAMES)}"
        )
    return problems


# --------------------------------------------------------------- inventory


@dataclass
class InventoryResult:
    evaluated: bool = False
    unclassified_python: list[str] = field(default_factory=list)
    unclassified_non_python: list[str] = field(default_factory=list)
    unscanned_python_present: list[str] = field(default_factory=list)
    excluded_present: list[str] = field(default_factory=list)
    not_inspected: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "evaluated": self.evaluated,
            "unclassified_python": self.unclassified_python,
            "unclassified_non_python": self.unclassified_non_python,
            "unscanned_python_present": self.unscanned_python_present,
            "excluded_present": self.excluded_present,
            "not_inspected": self.not_inspected,
        }


def _has_direct_python(path: Path) -> bool | None:
    try:
        with os.scandir(path) as entries:
            return any(
                entry.name.endswith(".py") and entry.is_file(follow_symlinks=False)
                for entry in entries
            )
    except OSError:
        return None


def evaluate_inventory(root: Path, policy: Policy) -> InventoryResult:
    """Classify every top-level entry without descending into any of them."""
    result = InventoryResult(evaluated=True)
    scanned = policy.scanned_roots
    unscanned = {item.name for item in policy.unscanned_python_roots}
    non_python = set(policy.non_python_roots)
    excluded = {item.name for item in policy.excluded_roots}
    patterns = [item.name for item in policy.excluded_patterns]
    with os.scandir(root) as entries:
        names = sorted((entry.name, entry) for entry in entries)
    for name, entry in names:
        if name in excluded or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            result.excluded_present.append(name)
            continue
        known = name in scanned or name in unscanned or name in non_python
        if entry.is_symlink():
            if not known:
                result.not_inspected.append(name)
            continue
        if entry.is_file(follow_symlinks=False):
            if not name.endswith(".py") or name in scanned:
                continue
            if name in unscanned:
                result.unscanned_python_present.append(name)
            else:
                result.unclassified_python.append(name)
            continue
        if not entry.is_dir(follow_symlinks=False) or name in scanned:
            continue
        if name.startswith(".") and not known:
            result.not_inspected.append(name)
            continue
        has_python = _has_direct_python(root / name)
        if has_python is None:
            result.not_inspected.append(name)
        elif name in unscanned:
            if has_python:
                result.unscanned_python_present.append(name)
        elif has_python:
            result.unclassified_python.append(name)
        elif name not in non_python:
            result.unclassified_non_python.append(name)
    return result
