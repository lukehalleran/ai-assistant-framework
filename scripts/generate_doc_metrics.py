#!/usr/bin/env python3
"""Generate docs/METRICS_SNAPSHOT.md — the single source of truth for volatile
repo metrics (LOC, file/test counts, tool/collection counts, latest benchmark).

Prime directive (see docs/DOC_CONSISTENCY_CHECKLIST.md): the CODEBASE is the
source of truth. This script derives every number from primary sources and
records the exact invocation next to each, so a reviewer can reproduce it and
so docs never drift silently again.

Usage:
    python scripts/generate_doc_metrics.py            # write the snapshot
    python scripts/generate_doc_metrics.py --check     # exit 1 if snapshot is stale
    python scripts/generate_doc_metrics.py --update-readme  # also rewrite README markers

Docs embed volatile numbers between HTML markers that this script rewrites:
    <!-- METRICS:BEGIN --> ... <!-- METRICS:END -->
Everything else in a doc is left untouched.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Run as `python scripts/generate_doc_metrics.py`, sys.path[0] is scripts/ —
# the tool-count import (core.agentic.tools) needs the repo root on the path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SNAPSHOT = REPO / "docs" / "METRICS_SNAPSHOT.md"
BEGIN = "<!-- METRICS:BEGIN -->"
END = "<!-- METRICS:END -->"


def _run(cmd: list[str], cwd: Path = REPO) -> str:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    ).stdout.strip()


# ── Primary-source derivations ──────────────────────────────────────────────

def python_loc() -> dict:
    """Count git-tracked Python files + lines, excluding vendored venv/ and data/.

    Method: git ls-files '*.py' | grep -vE '^(venv|data)/'
    """
    files = [
        f for f in _run(["git", "ls-files", "*.py"]).splitlines()
        if not re.match(r"^(venv|data)/", f)
    ]
    total = 0
    code = 0
    for rel in files:
        p = REPO / rel
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        total += len(lines)
        for ln in lines:
            s = ln.strip()
            if s and not s.startswith("#"):
                code += 1
    return {"files": len(files), "total_lines": total, "code_lines": code}


def test_counts() -> dict:
    """Collected test count (pytest --collect-only) + test-file count.

    The collected count honours pytest.ini markers/exclusions, so it is the
    number a reviewer reproduces with the recorded invocation.
    """
    out = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    m = re.search(r"(\d+)\s+tests?\s+collected", out)
    collected = int(m.group(1)) if m else None
    # Count files on disk (matches pytest's collection universe, which includes
    # not-yet-committed test files), i.e. `find tests -name "test_*.py"`.
    files = len([p for p in (REPO / "tests").rglob("test_*.py")])
    return {
        "collected": collected,
        "files": files,
        "invocation": "python -m pytest --collect-only -q",
    }


def tool_count() -> dict:
    """Agentic tool count from the dispatch table (single source of truth)."""
    try:
        from core.agentic.tools import DISPATCH_TABLE  # noqa: PLC0415
        rows = len(DISPATCH_TABLE)
    except Exception:  # pragma: no cover - import guard
        rows = None
    # recall_image is registered but excluded from the loop to save API credits.
    exposed = rows - 1 if rows else None
    return {"dispatch_rows": rows, "exposed_in_loop": exposed}


def collection_count() -> int | None:
    """ChromaDB collection count from the store's registration block."""
    src = (REPO / "memory/storage/multi_collection_chroma_store.py").read_text()
    m = re.search(r"self\.collections\s*=\s*\{(.*?)\}", src, re.DOTALL)
    if not m:
        return None
    return len(re.findall(r"^\s*['\"][\w]+['\"]\s*:", m.group(1), re.MULTILINE))


def benchmark_row() -> dict:
    """Latest retrieval benchmark from the designated ledger (BENCHMARK_METRICS.md).

    Also computes the same metrics from data/benchmark_per_case.csv as an
    independent cross-check; a divergence is surfaced (never silently resolved).
    """
    ledger = REPO / "docs" / "BENCHMARK_METRICS.md"
    result: dict = {"ledger": None, "csv": None}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            # | **Combined** | **272** | **0.8911** | **0.8309** | ... |
            if re.search(r"\bCombined\b", line) and "|" in line:
                nums = re.findall(r"[-+]?\d*\.?\d+", line.replace("**", ""))
                if len(nums) >= 5:
                    result["ledger"] = {
                        "n": int(float(nums[0])),
                        "mrr": float(nums[1]),
                        "r_at_1": float(nums[2]),
                        "source": "docs/BENCHMARK_METRICS.md (Combined row)",
                    }
                    break
    csv_path = REPO / "data" / "benchmark_per_case.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        ret = [r for r in rows if str(r.get("has_retrieval", "")).strip().lower()
               in ("true", "1", "yes")]
        def _avg(col: str) -> float:
            vals = [float(r[col]) for r in ret if r.get(col) not in ("", "None", None)]
            return round(sum(vals) / len(vals), 4) if vals else float("nan")
        if ret:
            result["csv"] = {
                "n": len(ret),
                "mrr": _avg("mrr"),
                "r_at_1": _avg("recall_at_1"),
                "source": "data/benchmark_per_case.csv (computed)",
            }
    return result


def git_sha() -> str:
    return _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"


# ── Snapshot rendering ──────────────────────────────────────────────────────

def build_snapshot_block() -> str:
    loc = python_loc()
    tests = test_counts()
    tools = tool_count()
    colls = collection_count()
    bench = benchmark_row()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha = git_sha()

    lines = []
    lines.append(f"- **Python:** {loc['files']} files · "
                 f"{loc['total_lines']:,} total lines · "
                 f"{loc['code_lines']:,} non-blank/non-comment  "
                 f"<sub>(git ls-files '*.py', excl venv/ data/)</sub>")
    tf = tests["collected"]
    lines.append(f"- **Tests:** {tf if tf is not None else '?'} collected across "
                 f"{tests['files']} test files  "
                 f"<sub>({tests['invocation']}; pytest.ini exclusions applied)</sub>")
    lines.append(f"- **Agentic tools:** {tools['dispatch_rows']} dispatch-table types "
                 f"({tools['exposed_in_loop']} exposed in the loop; recall_image excluded)")
    lines.append(f"- **ChromaDB collections:** {colls}")
    led = bench.get("ledger")
    if led:
        lines.append(f"- **Retrieval benchmark (ledger):** MRR={led['mrr']} · "
                     f"R@1={led['r_at_1']} · n={led['n']}  <sub>({led['source']})</sub>")
    csvb = bench.get("csv")
    if csvb:
        lines.append(f"- **Retrieval benchmark (CSV cross-check):** MRR={csvb['mrr']} · "
                     f"R@1={csvb['r_at_1']} · n={csvb['n']}  <sub>({csvb['source']})</sub>")
    if led and csvb and (led["n"] != csvb["n"] or abs(led["mrr"] - csvb["mrr"]) > 0.005):
        lines.append(f"  - ⚠️ **Ledger and CSV disagree** (n {led['n']}≠{csvb['n']}, "
                     f"MRR {led['mrr']}≠{csvb['mrr']}). Needs an owner-blessed re-run "
                     f"before either is quoted as canonical.")
    body = "\n".join(lines)
    return (f"{BEGIN}\n"
            f"<!-- Generated by scripts/generate_doc_metrics.py — do not edit by hand. -->\n"
            f"_Snapshot: {date} · git `{sha}`_\n\n"
            f"{body}\n"
            f"{END}")


def render_snapshot_file() -> str:
    return (
        "# METRICS_SNAPSHOT.md — Generated Repo Metrics\n\n"
        "Volatile numbers derived from primary sources. Regenerate with "
        "`python scripts/generate_doc_metrics.py`. Other docs reference this "
        "file (or embed the marker block below) instead of hand-copying numbers.\n\n"
        + build_snapshot_block() + "\n"
    )


def rewrite_markers(path: Path, block: str) -> bool:
    """Replace the marker block in `path`. Returns True if the file changed."""
    if not path.exists():
        return False
    text = path.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pat.search(text):
        return False
    new = pat.sub(block.replace("\\", "\\\\"), text)
    if new != text:
        path.write_text(new)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the snapshot file is stale")
    ap.add_argument("--update-readme", action="store_true",
                    help="also rewrite README.md marker block")
    args = ap.parse_args()

    content = render_snapshot_file()
    if args.check:
        current = SNAPSHOT.read_text() if SNAPSHOT.exists() else ""
        # Compare everything except the snapshot date/sha line (always changes).
        strip = lambda s: re.sub(r"_Snapshot:.*?_", "", s)
        if strip(current) != strip(content):
            print("METRICS_SNAPSHOT.md is stale — run: python scripts/generate_doc_metrics.py")
            return 1
        print("METRICS_SNAPSHOT.md is up to date.")
        return 0

    SNAPSHOT.write_text(content)
    print(f"Wrote {SNAPSHOT.relative_to(REPO)}")
    block = build_snapshot_block()
    if args.update_readme:
        if rewrite_markers(REPO / "README.md", block):
            print("Updated README.md metrics markers")
        if _update_readme_test_badge(REPO / "README.md"):
            print("Updated README.md test-count badge")
    return 0


def _update_readme_test_badge(path: Path) -> bool:
    """Rewrite the shields test-count badge with the live collected count."""
    if not path.exists():
        return False
    n = test_counts()["collected"]
    if n is None:
        return False
    text = path.read_text()
    # Match e.g. tests-5%2C731-brightgreen or tests-~5%2C000-brightgreen
    badge = re.compile(r"(badge/tests-)~?[\d]+(?:%2C[\d]+)*(-brightgreen)")
    formatted = f"{n:,}".replace(",", "%2C")
    new = badge.sub(rf"\g<1>{formatted}\g<2>", text)
    if new != text:
        path.write_text(new)
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
