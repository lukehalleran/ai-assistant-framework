"""DM-44 ratchet (BC-06/BC-58): a Tier-1 keyword arm scanning the whole message.

Inside `evaluate_agentic_gate`, a call to `_hit_non_negated` / `_find_trigger_hits`
with a Tier-1 `_*_HIT` keyword matcher must go through `_tier1_keyword_hit`
(code-stripped + head/tail window on long messages). A direct call scans a pasted
body and routes an agentic loop on an incidental word (2026-10-14: 'integral'
mid-paste, 36 s turn).
"""
import ast
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "core" / "agentic" / "gate.py"
CALLEES = {"_hit_non_negated", "_find_trigger_hits"}
# Deliberate exceptions, by matcher name -> reason.
ALLOWLIST = {
    "_EMAIL_INTENT_CUE_HIT": "email arm: own clause-level request rules (sentence scoped)",
    "_EMAIL_INTENT_VERB_HIT": "email arm: own clause-level request rules (sentence scoped)",
    "_EXPLICIT_SEARCH_HIT": "veto-exemption only (never routes a mode); reviewed as sibling",
}


def scan(source: str):
    tree = ast.parse(source)
    found = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef))
                and fn.name == "evaluate_agentic_gate"):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in CALLEES):
                for a in node.args[1:2]:
                    if (isinstance(a, ast.Name) and a.id.endswith("_HIT")
                            and a.id not in ALLOWLIST):
                        found.append((node.lineno, a.id))
    return found


def test_no_unscoped_tier1_keyword_arm():
    assert scan(GATE.read_text(encoding="utf-8")) == []


def test_sensitivity_old_computation_line_is_flagged():
    fixture = (
        "async def evaluate_agentic_gate(user_text):\n"
        "    needs_computation = _hit_non_negated(_lower, _COMPUTATION_HIT)\n"
    )
    assert scan(fixture) == [(2, "_COMPUTATION_HIT")]


def test_allowlist_entries_still_used():
    src = GATE.read_text(encoding="utf-8")
    for name in ALLOWLIST:
        assert name in src
