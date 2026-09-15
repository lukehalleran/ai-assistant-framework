"""Reviewed schema-1 → schema-2 baseline migration (2026-09-13, one-time).

Contract v1 anchored an occurrence as ``(scanner, path, qualname, clipped
source line)``.  Contract v2 anchors it as ``(scanner, path, qualname, kind,
SHA-256 of the canonical candidate AST)``.  Rewriting the committed baseline
silently would lose the review trail, so this command only PROPOSES:

* it re-runs every gate scanner and maps each legacy occurrence to exactly one
  current finding with the same v1 key, taking identical keys in source order
  (the stable legacy ordinal);
* any legacy key whose current multiplicity differs is PENDING — the command
  fails closed and nothing is proposed for it;
* current findings no legacy occurrence claims are listed as new candidates;
* it writes a proposed schema-2 baseline and a per-occurrence mapping (legacy
  key and ordinal, v2 anchor and ordinal, line span, full source expression,
  source-file SHA-256) to ``--out``, which may not be inside ``config/``.

The legacy file must be canonical (its schema-1 re-render is byte-identical),
so the mapping's legacy history can later be proven by digest alone.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import baseline as baseline_mod
from .policy import POLICY_PATH, PolicyError, load_policy, registry_mismatches
from .scanners import SCANNERS
from .scanners.common import ScannerError

MAX_EXPRESSION_LINES = 40
LEGACY_KEY = ("scanner", "path", "symbol", "text")


def _row(root: Path, finding, legacy_key, legacy_ordinal, cache: dict) -> dict:
    if finding.path not in cache:
        raw = (root / finding.path).read_bytes()
        cache[finding.path] = (hashlib.sha256(raw).hexdigest(), raw.decode("utf-8").splitlines())
    digest, lines = cache[finding.path]
    start, end = finding.span if finding.span != (0, 0) else (finding.line, finding.line)
    segment = lines[start - 1 : end]
    if len(segment) > MAX_EXPRESSION_LINES:
        segment = segment[:MAX_EXPRESSION_LINES] + [f"… ({end - start + 1} lines)"]
    return {
        "legacy": None
        if legacy_key is None
        else {**dict(zip(LEGACY_KEY, legacy_key)), "ordinal": legacy_ordinal},
        "anchor": dict(zip(baseline_mod.ANCHOR_FIELDS, finding.fingerprint())),
        "ordinal": None,
        "line": finding.line,
        "span": [start, end],
        "excerpt": finding.excerpt,
        "expression": "\n".join(segment),
        "source_sha256": digest,
        "unresolved": finding.unresolved,
    }


def propose(root: Path, legacy_path: Path) -> tuple[dict, baseline_mod.Baseline]:
    policy = load_policy(root / POLICY_PATH)
    mismatches = registry_mismatches(policy, SCANNERS)
    if mismatches:
        raise PolicyError("registry does not match the policy: " + "; ".join(mismatches))
    raw = legacy_path.read_bytes()
    legacy = baseline_mod.legacy_load(legacy_path)
    if baseline_mod.legacy_render(legacy).encode("utf-8") != raw:
        raise baseline_mod.BaselineError(f"{legacy_path} is not a canonical schema-1 rendering")

    gate_ids = [contract.id for contract in policy.scanners if contract.mode == "gate"]
    findings = [finding for sid in gate_ids for finding in SCANNERS[sid].scan(root).findings]
    findings.sort(key=lambda f: (f.scanner_id, f.path, f.line, f.digest))
    by_legacy = defaultdict(list)
    for finding in findings:
        by_legacy[finding.legacy_fingerprint()].append(finding)

    cache: dict = {}
    rows: list[dict] = []
    claimed: set[int] = set()
    pending = []
    for key in sorted(legacy):
        count = legacy[key]
        current = by_legacy.get(key, [])
        if len(current) != count:
            pending.append({
                "legacy": dict(zip(LEGACY_KEY, key)),
                "legacy_multiplicity": count,
                "current_multiplicity": len(current),
            })
            continue
        for ordinal, finding in enumerate(current, 1):
            claimed.add(id(finding))
            rows.append(_row(root, finding, key, ordinal, cache))
    for finding in findings:
        if id(finding) not in claimed:
            rows.append(_row(root, finding, None, None, cache))

    rows.sort(key=lambda r: (tuple(r["anchor"].values()), r["line"]))
    seen: Counter = Counter()
    for row in rows:
        anchor = tuple(row["anchor"].values())
        seen[anchor] += 1
        row["ordinal"] = seen[anchor]
    rows.sort(key=lambda r: (r["anchor"]["scanner"], r["anchor"]["path"], r["line"], r["ordinal"]))

    mapped = Counter(
        tuple(row["legacy"][name] for name in LEGACY_KEY) for row in rows if row["legacy"]
    )
    proposed = baseline_mod.from_findings(findings)
    per_scanner = {}
    for sid in gate_ids:
        per_scanner[sid] = {
            "legacy": sum(v for k, v in legacy.items() if k[0] == sid),
            "mapped": sum(1 for r in rows if r["legacy"] and r["anchor"]["scanner"] == sid),
            "new_candidates": sum(1 for r in rows if not r["legacy"] and r["anchor"]["scanner"] == sid),
            "proposed": sum(v for k, v in proposed.counter.items() if k[0] == sid),
        }
    proof = {
        "legacy_path": str(legacy_path),
        "legacy_sha256": hashlib.sha256(raw).hexdigest(),
        "legacy_occurrences": sum(legacy.values()),
        "legacy_unique_keys": len(legacy),
        "legacy_duplicate_extras": sum(legacy.values()) - len(legacy),
        "mapped_occurrences": sum(mapped.values()),
        "legacy_multiset_equal": mapped == legacy,
        "pending": pending,
        "new_candidates": sum(1 for row in rows if not row["legacy"]),
        "proposed_occurrences": proposed.occurrences,
        "proposed_unique_anchors": len(proposed.counter),
        "per_scanner": per_scanner,
    }
    return {"proof": proof, "rows": rows}, proposed


def run(root: Path, legacy_path: Path, out_dir: Path) -> int:
    out = out_dir.resolve()
    protected = (root / "config").resolve()
    if out == protected or protected in out.parents:
        print(f"migrate-baseline: refusing to write into {protected}; proposals go elsewhere")
        return 2
    try:
        mapping, proposed = propose(root, legacy_path)
    except (PolicyError, baseline_mod.BaselineError, ScannerError, OSError) as exc:
        print(f"migrate-baseline: {exc}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposed_baseline.json").write_text(baseline_mod.render(proposed), encoding="utf-8")
    (out / "migration_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    proof = mapping["proof"]
    print(json.dumps({k: v for k, v in proof.items() if k != "pending"}, indent=2))
    for item in proof["pending"]:
        print(f"PENDING {item}")
    ok = not proof["pending"] and proof["legacy_multiset_equal"]
    print("migration proposal: " + ("one-to-one" if ok else "NOT one-to-one — fail closed"))
    return 0 if ok else 1
