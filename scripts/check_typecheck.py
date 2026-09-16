#!/usr/bin/env python3
"""Pyright type-checking lane: baseline-backed gate + ratchet.

Contract (docs/PLAN_20260916_typecheck_lane.md §2-3): each ``(file, rule)``
pair has a checked-in ceiling; a diagnostic count above it, or above a
suppression-comment ceiling, is a finding. ``--update-baseline`` may only
LOWER a row; a rise needs ``--allow-raise``. ``--changed-since`` adds a
changed-line rule so a diagnostic that moved within its ceiling still fails
on a touched line. Exit 0 clean, 1 findings, 2 could-not-evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 1
TOOL = "pyright"
DEFAULT_VERSION = "1.1.403"

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

# Conforming suppression: rule-scoped + a reason. Anything else matching the
# broad ignore prefix is a "legacy" (blanket) suppression.
CONFORMING_IGNORE_RE = re.compile(r"#\s*pyright:\s*ignore\[[A-Za-z, ]+\]\s*#\s*why:\s*\S")
ANY_IGNORE_RE = re.compile(r"#\s*(pyright|type):\s*ignore")
VERSION_RE = re.compile(r"pyright (\d+\.\d+\.\d+)")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
ALL_ZEROS_RE = re.compile(r"^0+$")


def pyright_command() -> list[str]:
    env_bin = os.environ.get("PYRIGHT_BIN")
    if env_bin:
        return [env_bin]
    which = shutil.which("pyright")
    if which:
        return [which]
    return [sys.executable, "-m", "pyright"]


def _forced_env(version: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYRIGHT_PYTHON_FORCE_VERSION"] = version
    env["PYRIGHT_PYTHON_IGNORE_WARNINGS"] = "1"
    return env


def measured_pyright_version(cmd: list[str], forced_version: str) -> str | None:
    result = subprocess.run([*cmd, "--version"], capture_output=True, text=True, env=_forced_env(forced_version))
    match = VERSION_RE.search(result.stdout)
    return match.group(1) if match else None


def run_pyright(cmd: list[str], root: Path, forced_version: str) -> tuple[int, str, str, float]:
    started = time.monotonic()
    result = subprocess.run(
        [*cmd, "--outputjson", "--pythonpath", sys.executable, "-p", str(root)],
        capture_output=True, text=True, cwd=str(root), env=_forced_env(forced_version),
    )
    wall = time.monotonic() - started
    return result.returncode, result.stdout, result.stderr, wall


def display_path(root: Path, path: Path) -> str:
    """Path relative to root for messages — never an absolute path in output."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_baseline(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read baseline ({exc.strerror or exc})"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"baseline is not valid JSON: {exc}"


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-typecheck-baseline-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, sort_keys=True, indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def pyright_config(root: Path) -> dict:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data.get("tool", {}).get("pyright", {})


def excluded_dir_names(patterns: list[str]) -> frozenset[str]:
    names = set()
    for pattern in patterns:
        name = pattern.rsplit("/", 1)[-1]
        if name and name != "**":
            names.add(name)
    return frozenset(names)


def included_python_files(root: Path, config: dict) -> list[str]:
    include = config.get("include", [])
    exclude_names = excluded_dir_names(config.get("exclude", []))
    files: set[str] = set()
    for entry in include:
        target = root / entry
        if target.is_file() and target.suffix == ".py":
            files.add(target.relative_to(root).as_posix())
            continue
        if not target.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in exclude_names]
            for filename in filenames:
                if filename.endswith(".py"):
                    files.add((Path(dirpath) / filename).relative_to(root).as_posix())
    return sorted(files)


def normalize_diagnostics(raw: dict, root: Path) -> list[dict]:
    out = []
    for diag in raw.get("generalDiagnostics", []):
        path = Path(diag["file"])
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        start = (diag.get("range") or {}).get("start") or {}
        out.append({
            "file": rel,
            "rule": diag.get("rule") or "<none>",
            "line": start.get("line", 0) + 1,
            "severity": diag["severity"],
        })
    return out


def count_by_file_rule(diags: list[dict]) -> Counter:
    return Counter((d["file"], d["rule"]) for d in diags)


def lines_by_file_rule(diags: list[dict]) -> dict:
    out: dict = defaultdict(list)
    for d in diags:
        out[(d["file"], d["rule"])].append(d["line"])
    return out


def scan_suppressions(root: Path, files: list[str]) -> tuple[Counter, Counter]:
    valid: Counter = Counter()
    legacy: Counter = Counter()
    for rel in files:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            if CONFORMING_IGNORE_RE.search(line):
                valid[rel] += 1
            elif ANY_IGNORE_RE.search(line):
                legacy[rel] += 1
    return valid, legacy


def rule_a_findings(counts: Counter, line_map: dict, baseline_files: dict) -> list[str]:
    findings = []
    for (file, rule), measured in sorted(counts.items()):
        ceiling = baseline_files.get(file, {}).get(rule, 0)
        if measured > ceiling:
            lines = ", ".join(str(n) for n in sorted(line_map[(file, rule)]))
            findings.append(f"count exceeded: {file} {rule} measured {measured} ceiling {ceiling} (lines {lines})")
    return findings


def suppression_findings(valid: Counter, legacy: Counter, baseline_ignores: dict, baseline_legacy: dict) -> list[str]:
    findings = []
    for file in sorted(set(valid) | set(baseline_ignores)):
        measured, ceiling = valid.get(file, 0), baseline_ignores.get(file, 0)
        if measured > ceiling:
            findings.append(f"suppression added: {file} ignores measured {measured} ceiling {ceiling}")
    for file in sorted(set(legacy) | set(baseline_legacy)):
        measured, ceiling = legacy.get(file, 0), baseline_legacy.get(file, 0)
        if measured > ceiling:
            findings.append(f"suppression added: {file} legacy_ignores measured {measured} ceiling {ceiling}")
    return findings


def resolve_changed_ref(root: Path, ref: str) -> bool:
    if not ref or ALL_ZEROS_RE.match(ref):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=str(root), capture_output=True, text=True,
    )
    return result.returncode == 0


def changed_ranges(root: Path, ref: str, files: list[str]) -> dict:
    if not files:
        return {}
    result = subprocess.run(["git", "diff", "-U0", ref, "--", *files], cwd=str(root), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed (exit {result.returncode}): {result.stderr[-500:]}")
    ranges: dict = defaultdict(list)
    current_file = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current_file = None if target == "/dev/null" else target.removeprefix("b/")
            continue
        match = HUNK_RE.match(line)
        if match and current_file is not None:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            if count == 0:
                continue
            ranges[current_file].append((start, start + count - 1))
    return ranges


def rule_b_findings(ranges: dict, gate_diags: list[dict]) -> list[str]:
    findings = set()
    for d in gate_diags:
        for start, end in ranges.get(d["file"], ()):
            if start <= d["line"] <= end:
                findings.add(f"touched line still reports: {d['file']}:{d['line']} {d['rule']}")
    return sorted(findings)


def build_measured_baseline(counts: Counter, valid: Counter, legacy: Counter, version: str, mode: str, python_version: str) -> dict:
    files: dict = defaultdict(dict)
    for (file, rule), n in counts.items():
        files[file][rule] = n
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "version": version,
        "mode": mode,
        "python": python_version,  # major.minor — never an interpreter path (machine-specific)
        "files": {f: dict(sorted(r.items())) for f, r in sorted(files.items())},
        "ignores": {f: n for f, n in sorted(valid.items()) if n},
        "legacy_ignores": {f: n for f, n in sorted(legacy.items()) if n},
        "notes": {},
    }


def existing_keys(baseline: dict) -> set[str]:
    keys = set()
    for file, rules in baseline.get("files", {}).items():
        keys.update(f"{file}::{rule}" for rule in rules)
    keys.update(f"{file}::ignores" for file in baseline.get("ignores", {}))
    keys.update(f"{file}::legacy_ignores" for file in baseline.get("legacy_ignores", {}))
    return keys


def detect_raises(new_baseline: dict, old_baseline: dict | None) -> list[tuple[str, int, int]]:
    """Rows that would rise vs. old_baseline (absent row treated as ceiling 0)."""
    raises = []
    old_files = (old_baseline or {}).get("files", {})
    for file, rules in new_baseline["files"].items():
        for rule, n in rules.items():
            old = old_files.get(file, {}).get(rule, 0)
            if n > old:
                raises.append((f"{file}::{rule}", old, n))
    for kind in ("ignores", "legacy_ignores"):
        old_kind = (old_baseline or {}).get(kind, {})
        for file, n in new_baseline[kind].items():
            old = old_kind.get(file, 0)
            if n > old:
                raises.append((f"{file}::{kind}", old, n))
    return sorted(raises)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="check_typecheck", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--changed-since", default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--allow-raise", default=None, metavar="REASON")
    parser.add_argument("--pyright-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _read_pyright_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except OSError as exc:
        return None, f"cannot read --pyright-json ({exc.strerror or exc})"
    except json.JSONDecodeError as exc:
        return None, f"--pyright-json is not valid JSON: {exc}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    baseline_path = (args.baseline or (root / "config" / "typecheck_baseline.json")).resolve()

    baseline, load_error = load_baseline(baseline_path)
    if load_error:
        if not args.update_baseline:
            print(f"could not evaluate: {load_error}", file=sys.stderr)
            return EXIT_ERROR
        print(f"warning: {load_error}; starting a fresh baseline", file=sys.stderr)
        baseline = None
    if baseline is None and not args.update_baseline:
        print(
            f"could not evaluate: no baseline at {display_path(root, baseline_path)}; "
            "run --update-baseline first",
            file=sys.stderr,
        )
        return EXIT_ERROR

    cmd = pyright_command()
    forced_version = (baseline or {}).get("version") or DEFAULT_VERSION

    if args.pyright_json:
        raw, read_error = _read_pyright_json(args.pyright_json)
        if read_error:
            print(f"could not evaluate: {read_error}", file=sys.stderr)
            return EXIT_ERROR
        measured_version = raw.get("version")
    else:
        measured_version = measured_pyright_version(cmd, forced_version)
        if measured_version is None:
            print("could not evaluate: could not parse pyright --version output", file=sys.stderr)
            return EXIT_ERROR
        if baseline is not None and measured_version != baseline.get("version"):
            print(
                f"could not evaluate: pyright version mismatch (measured {measured_version}, "
                f"baseline expects {baseline.get('version')})",
                file=sys.stderr,
            )
            return EXIT_ERROR
        returncode, stdout, stderr, wall = run_pyright(cmd, root, forced_version)
        print(f"pyright wall: {wall:.1f} s")
        if returncode not in (0, 1):
            print(f"could not evaluate: pyright exited {returncode}: {stderr[-1000:]}", file=sys.stderr)
            return EXIT_ERROR
        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as exc:
            print(
                f"could not evaluate: pyright output was not valid JSON: {exc}; stderr: {stderr[-1000:]}",
                file=sys.stderr,
            )
            return EXIT_ERROR

    diags = normalize_diagnostics(raw, root)
    gate_diags = [d for d in diags if d["severity"] == "error"]
    warning_count = sum(1 for d in diags if d["severity"] != "error")
    counts = count_by_file_rule(gate_diags)
    line_map = lines_by_file_rule(gate_diags)

    try:
        config = pyright_config(root)
    except OSError as exc:
        print(f"could not evaluate: cannot read pyproject.toml ({exc.strerror or exc})", file=sys.stderr)
        return EXIT_ERROR
    except tomllib.TOMLDecodeError as exc:
        print(f"could not evaluate: pyproject.toml is not valid TOML: {exc}", file=sys.stderr)
        return EXIT_ERROR
    mode = config.get("typeCheckingMode", "standard")
    files = included_python_files(root, config)
    valid, legacy = scan_suppressions(root, files)

    if args.update_baseline:
        return _update_baseline(root, baseline_path, baseline, counts, valid, legacy, measured_version, mode, args)

    findings = []
    findings.extend(rule_a_findings(counts, line_map, baseline.get("files", {})))
    findings.extend(suppression_findings(valid, legacy, baseline.get("ignores", {}), baseline.get("legacy_ignores", {})))

    if args.changed_since is not None:
        if resolve_changed_ref(root, args.changed_since):
            try:
                ranges = changed_ranges(root, args.changed_since, files)
            except RuntimeError as exc:
                print(f"could not evaluate: {exc}", file=sys.stderr)
                return EXIT_ERROR
            findings.extend(rule_b_findings(ranges, gate_diags))
        else:
            print(f"changed-line rule skipped: {args.changed_since} unresolvable")

    return _report(counts, warning_count, valid, legacy, baseline, findings, args.as_json)


def _update_baseline(root, baseline_path, baseline, counts, valid, legacy, measured_version, mode, args) -> int:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    new_baseline = build_measured_baseline(counts, valid, legacy, measured_version, mode, python_version)
    raises = detect_raises(new_baseline, baseline)
    if raises and not args.allow_raise:
        for key, old, new in raises:
            print(f"would raise: {key} {old} -> {new}", file=sys.stderr)
        print('refused: raising rows needs --allow-raise "<reason>"; baseline NOT written', file=sys.stderr)
        return EXIT_FINDINGS
    if baseline is None:
        # A fresh baseline has no prior rows to "raise" from: ONE note records
        # the reason, not a copy of it on every row.
        notes = {"__initial__": args.allow_raise}
        print(f"initial baseline ({len(raises)} rows): {args.allow_raise}")
    else:
        notes = {
            k: v for k, v in baseline.get("notes", {}).items()
            if k.startswith("__") or k in existing_keys(new_baseline)
        }
        for key, old, new in raises:
            notes[key] = args.allow_raise
            print(f"raise allowed: {key} {old} -> {new} ({args.allow_raise})")
    new_baseline["notes"] = notes
    atomic_write_json(baseline_path, new_baseline)
    total = sum(sum(r.values()) for r in new_baseline["files"].values())
    print(
        f"baseline written: {display_path(root, baseline_path)} — {total} error diagnostics "
        f"in {len(new_baseline['files'])} files, ignores {sum(new_baseline['ignores'].values())}, "
        f"legacy_ignores {sum(new_baseline['legacy_ignores'].values())}"
    )
    return EXIT_OK


def _report(counts, warning_count, valid, legacy, baseline, findings, as_json) -> int:
    total_measured = sum(counts.values())
    total_baseline = sum(sum(r.values()) for r in baseline.get("files", {}).values())
    files_with_diags = len({file for file, _rule in counts})
    valid_total, legacy_total = sum(valid.values()), sum(legacy.values())

    report = {
        "exit": EXIT_OK if not findings else EXIT_FINDINGS,
        "error_diagnostics": total_measured,
        "files_with_diagnostics": files_with_diags,
        "baseline_total": total_baseline,
        "warnings_not_gated": warning_count,
        "valid_suppressions": valid_total,
        "legacy_suppressions": legacy_total,
        "findings": findings,
    }
    if as_json:
        print(json.dumps(report, sort_keys=True, indent=2))

    if findings:
        print(f"typecheck: FINDINGS ({len(findings)})")
        for finding in findings:
            print(f"  {finding}")
        print(
            "fix the diagnostics in the touched file, or (owner) "
            'scripts/check_typecheck.py --update-baseline --allow-raise "<reason>"'
        )
        return EXIT_FINDINGS

    print(
        f"typecheck: OK — {total_measured} error diagnostics in {files_with_diags} files "
        f"(baseline {total_baseline}), {warning_count} warnings not gated, "
        f"suppressions {valid_total}/{legacy_total}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
