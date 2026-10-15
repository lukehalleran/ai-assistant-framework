"""BC-97 ratchet (DM-43): a personal-claim provenance note must name its claim.

The generic PERSONAL_CLAIM_MARKER fired on nearly every stored assistant turn
and so carried no information about WHICH claim was unverified (2026-10-13).
Only the two owner modules may reference it; every other consumer asks
`marker_suffix` / `annotate_personal_claim_memory`.
"""
import re
from pathlib import Path

from utils.personal_claim_provenance import (
    MARKER,
    annotate_personal_claim_memory,
    response_digest,
)

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"utils/read_time_markers.py", "utils/personal_claim_provenance.py"}
SCAN_DIRS = ("core", "gui", "api", "models", "utils", "knowledge", "memory", "processing", "config")
_REF = re.compile(r"\bPERSONAL_CLAIM_MARKER\b|\bLEGACY_PERSONAL_CLAIM_MARKERS\b|personal_claim_provenance import[^\n]*\bMARKER\b")


def _hits(files):
    out = []
    for rel, text in files:
        if rel in ALLOWED:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if not line.lstrip().startswith("#") and _REF.search(line):
                out.append(f"{rel}:{n}: {line.strip()}")
    return out


def _production():
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "tests" not in p.parts:
                yield p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8")


def test_generic_marker_referenced_only_by_its_owners():
    assert not _hits(_production()), "generic personal-claim marker referenced outside its owners (BC-97)"


def test_guard_sensitivity():
    bad = [("memory/x.py", "from utils.personal_claim_provenance import MARKER as M\n")]
    assert _hits(bad)
    assert _hits([("gui/y.py", "a += PERSONAL_CLAIM_MARKER\n")])


def _item(spans, **extra):
    resp = "Intro. The deadline is the 25th. Outro."
    receipt = {"status": "checked", "reason": "ok", "delivery": "unchanged",
               "insufficient_count": 1, "response_sha256": response_digest(resp), **extra}
    if spans is not None:
        receipt["flagged_spans"] = spans
    return {"response": resp, "personal_claim_support": receipt}


def test_deployed_annotate_specificity():
    named = annotate_personal_claim_memory(_item(["The deadline is the 25th."]))["response"]
    assert MARKER not in named and "The deadline is the 25th." in named
    assert MARKER not in annotate_personal_claim_memory(_item([]))["response"]
    assert annotate_personal_claim_memory(_item(None))["response"].endswith(MARKER)
