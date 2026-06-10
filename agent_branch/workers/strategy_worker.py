#!/usr/bin/env python3
# agent_branch/workers/strategy_worker.py
"""
Scripted strategy worker — the DETERMINISTIC driver of the M2 reaper proof.

The bounded objective is "implement subtract() in sandbox/calc.py, keep add()
working". Three strategies produce measurably DIFFERENT diffs that all pass the
same supervisor proof test (so divergence is real, not theatre):

  WORKER_STRATEGY=surgical   → append subtract (smallest diff, 1 file)
  WORKER_STRATEGY=refactor   → restructure calc into an ops table, then both ops
  WORKER_STRATEGY=testfirst  → add a stdlib branch test, then implement (2 files)

Deterministic so the proof is reproducible (an LLM trio is the manual demo).
Stdlib only.
"""

import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("WORKER_REPO", "/work/repo")
ARTIFACTS = os.environ.get("WORKER_ARTIFACTS", "/work/artifacts")
STRATEGY = os.environ.get("WORKER_STRATEGY", "surgical")

_SURGICAL = "\n\ndef subtract(a, b):\n    return a - b\n"

_REFACTOR = '''"""Arithmetic operations (refactored into a dispatch table)."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


OPS = {"add": add, "subtract": subtract}


def apply(op, a, b):
    return OPS[op](a, b)
'''

# stdlib branch test — runs its asserts on `python sandbox/test_subtract.py`
# (the script's own dir is on sys.path, so `import calc` resolves).
_TESTFIRST_TEST = (
    "import calc\n"
    "assert calc.subtract(5, 2) == 3\n"
    "assert calc.add(2, 3) == 5\n"
    "print('branch test ok')\n"
)


def main() -> int:
    Path(ARTIFACTS).mkdir(parents=True, exist_ok=True)
    calc = Path(REPO, "sandbox", "calc.py")
    report = {"strategy": STRATEGY, "objective": "implement subtract"}

    if STRATEGY == "surgical":
        calc.write_text(calc.read_text() + _SURGICAL, encoding="utf-8")
        report["edits"] = ["sandbox/calc.py (append)"]
    elif STRATEGY == "refactor":
        calc.write_text(_REFACTOR, encoding="utf-8")
        report["edits"] = ["sandbox/calc.py (restructure)"]
    elif STRATEGY == "testfirst":
        Path(REPO, "sandbox", "test_subtract.py").write_text(_TESTFIRST_TEST, encoding="utf-8")
        calc.write_text(calc.read_text() + _SURGICAL, encoding="utf-8")
        report["edits"] = ["sandbox/test_subtract.py (new)", "sandbox/calc.py (append)"]
    else:
        print(f"unknown strategy {STRATEGY!r}", file=sys.stderr)
        return 2

    Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
    print(f"strategy worker '{STRATEGY}' implemented subtract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
