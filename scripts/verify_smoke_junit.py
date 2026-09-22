#!/usr/bin/env python3
"""Reject empty, skipped, or failing fresh-runtime smoke JUnit receipts."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED_CLASSNAME = "tests.smoke.test_fresh_clone_contract"
EXPECTED_TESTNAME = "test_isolated_runtime_two_boot_cycle"


def verify(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"JUnit receipt is missing: {path}"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return False, f"JUnit receipt is unreadable: {exc}"

    cases = root.findall(".//testcase")
    expected = [
        case for case in cases
        if case.get("classname") == EXPECTED_CLASSNAME
        and case.get("name") == EXPECTED_TESTNAME
    ]
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    passed = len(cases) - skipped - failed - errors
    if not cases:
        return False, "JUnit receipt contains zero test cases"
    if len(cases) != 1 or len(expected) != 1:
        labels = [f"{case.get('classname')}::{case.get('name')}" for case in cases]
        return False, (
            "JUnit receipt must contain exactly the isolated-runtime two-boot test; "
            f"found {labels!r}"
        )
    if skipped:
        return False, f"JUnit receipt contains {skipped} skipped test case(s)"
    if failed or errors:
        return False, f"JUnit receipt contains {failed} failure(s) and {errors} error(s)"
    if passed < 1:
        return False, "JUnit receipt contains no passed test case"
    return True, f"JUnit receipt verified: {passed} passed, 0 skipped"


def workflow_problems(text: str) -> list[str]:
    """Small textual contract preventing removal of any required smoke leg."""
    required = {
        "model provisioning": "scripts/prepare_smoke_model_cache.py --receipt receipts/smoke-models.json",
        "required mode": 'DAEMON_SMOKE_REQUIRED: "1"',
        "offline smoke": "HF_HUB_OFFLINE: \"1\"",
        "two-boot test": "tests/smoke/test_fresh_clone_contract.py",
        "JUnit output": "--junitxml=receipts/isolated-runtime-smoke.xml",
        "JUnit verifier": "scripts/verify_smoke_junit.py receipts/isolated-runtime-smoke.xml",
    }
    problems = []
    for label, fragment in required.items():
        if text.count(fragment) != 1:
            problems.append(f"workflow must include {label} exactly once: {fragment}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    ok, message = verify(args.receipt)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
