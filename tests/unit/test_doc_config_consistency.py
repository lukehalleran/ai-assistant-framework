"""Tripwire: documented config values must equal the live constants.

This is intentionally SMALL and high-value (see docs/DOC_CONSISTENCY_CHECKLIST.md
§2.2) — a drift alarm, not full coverage. Each entry pins a doc line to a live
`app_config` constant (or config.yaml value). When a constant changes without
the doc being updated, the anchored regex (built from the *live* value) stops
matching and this test fails, naming the doc to fix.

Seeded from the 2026-06-30 synthesis recalibration (§1.3) and the memory
score-weight resolution (§1.4) — the two families that caused the audit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import config.app_config as app_config

REPO = Path(__file__).resolve().parent.parent.parent


def _num_variants(v: float) -> str:
    """Regex alternation of plausible textual forms of a number (0.6 → 0.6|0.60)."""
    forms = {repr(float(v)), f"{float(v):.2f}", f"{float(v):g}"}
    return "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"


def _score_weights() -> dict:
    cfg = yaml.safe_load((REPO / "config/config.yaml").read_text())
    return cfg["gating"]["score_weights"]


_SW = _score_weights()

# (label, live_value, doc_relpath, anchor_regex_with_{v})
# {v} is replaced by a regex matching the live value's textual forms. The anchor
# text ties the value to its meaning so a coincidental match elsewhere won't pass.
CHECKS = [
    # ── §1.3 synthesis composite (app_config constants) ──
    ("WEIGHT_COHERENCE", app_config.SYNTHESIS_WEIGHT_COHERENCE,
     "docs/SYNTHESIS_FILTER.md", r"{v} \* coherence"),
    ("WEIGHT_NOVELTY", app_config.SYNTHESIS_WEIGHT_NOVELTY,
     "docs/SYNTHESIS_FILTER.md", r"{v} \* novelty"),
    ("WEIGHT_DISTANCE", app_config.SYNTHESIS_WEIGHT_DISTANCE,
     "docs/SYNTHESIS_FILTER.md", r"{v} \* distance"),
    ("WEIGHT_STRUCTURAL", app_config.SYNTHESIS_WEIGHT_STRUCTURAL,
     "docs/SYNTHESIS_FILTER.md", r"{v} \* structural"),
    ("COMPOSITE_MIN_SCORE", app_config.SYNTHESIS_COMPOSITE_MIN_SCORE,
     "docs/SYNTHESIS_FILTER.md", r"composite >= {v}"),
    ("COMPOSITE_MIN_SCORE(skeleton)", app_config.SYNTHESIS_COMPOSITE_MIN_SCORE,
     "docs/PROJECT_SKELETON.md", r"SYNTHESIS_COMPOSITE_MIN_SCORE = {v}"),
    ("CONCEPT_COSINE_KNOWN", app_config.SYNTHESIS_CONCEPT_COSINE_KNOWN_THRESHOLD,
     "docs/SYNTHESIS_FILTER.md", r"cos\(A,B\)` > {v}"),
    ("WEIGHT_NOVELTY(qref)", app_config.SYNTHESIS_WEIGHT_NOVELTY,
     "docs/QUICK_REFERENCE.md", r"SYNTHESIS_WEIGHT_NOVELTY = {v}"),
    ("COMPOSITE_MIN_SCORE(readme)", app_config.SYNTHESIS_COMPOSITE_MIN_SCORE,
     "README.md", r"composite ≥ {v}"),

    # ── §1.4 memory score weights (config.yaml gating.score_weights) ──
    ("score_weights.relevance", _SW["relevance"],
     "docs/QUICK_REFERENCE.md", r'"relevance": {v}'),
    ("score_weights.recency", _SW["recency"],
     "docs/QUICK_REFERENCE.md", r'"recency": {v}'),
    ("score_weights.truth", _SW["truth"],
     "docs/QUICK_REFERENCE.md", r'"truth": {v}'),
    ("score_weights.topic_match", _SW["topic_match"],
     "docs/QUICK_REFERENCE.md", r'"topic_match": {v}'),
    ("score_weights.relevance(formal)", _SW["relevance"],
     "docs/FORMAL_MODEL.md", r"relevance\(d, x\) \+ collection_boost \| {v}"),
    ("score_weights.relevance(memsys)", _SW["relevance"],
     "docs/MEMORY_SYSTEM.md", r"relevance:  {v}"),
]


@pytest.mark.parametrize("label,value,doc,anchor", CHECKS,
                         ids=[c[0] for c in CHECKS])
def test_doc_matches_live_constant(label, value, doc, anchor):
    path = REPO / doc
    assert path.exists(), f"{doc} missing"
    text = path.read_text()
    pattern = anchor.replace("{v}", _num_variants(value))
    assert re.search(pattern, text), (
        f"{doc} is out of sync with the live value for {label} ({value}). "
        f"Expected to find /{pattern}/. Update the doc (or the constant) so they "
        f"agree — see docs/DOC_CONSISTENCY_CHECKLIST.md."
    )


def test_score_weights_sum_to_one():
    """Guard the invariant the docs assert ('sum to 1.0')."""
    total = sum(float(v) for v in _SW.values())
    assert abs(total - 1.0) < 1e-6, f"gating.score_weights sum to {total}, not 1.0"
