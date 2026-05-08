"""
Phase 6: Objective checks for prompt ablation eval.

Automated checks that don't require subjective judgment. Each check
produces a pass/fail/score result per response, and aggregates across
variants to detect section-level quality deltas.

Checks:
    1. Response length analysis — too short (truncated) or too long (padding)
    2. Thinking leak detection — reasoning content leaked into final answer
    3. Profile grounding — mentioned names/facts match what's in the prompt
    4. Citation validity — [MEM_*] and [WEB_*] markers reference real sources

Inputs:
    - Phase 4 generation results (eval/runs/<run_id>/results/)
Outputs:
    - Per-response check results
    - Aggregate check pass rates per section (LOO comparison)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Check result model
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of one objective check on one response."""

    check_name: str
    passed: bool
    score: float = 1.0  # 0.0 = worst, 1.0 = best
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CheckResult:
        return cls(**d)


@dataclass
class ResponseCheckResults:
    """All check results for one (snapshot, variant) response."""

    snapshot_id: str
    variant_id: str
    strategy: str
    query_text: str
    sections_removed: List[str]
    checks: List[CheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResponseCheckResults:
        d = dict(d)
        d["checks"] = [CheckResult.from_dict(c) for c in d.get("checks", [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# Check 1: Response length analysis
# ---------------------------------------------------------------------------

# Length thresholds by query type heuristic
_CASUAL_PATTERNS = re.compile(
    r"^(hey|hi|hello|sup|yo|lol|haha|blugh|nice|thanks|ok)\b",
    re.IGNORECASE,
)

_SHORT_THRESHOLD = 20    # chars — likely truncated
_CASUAL_LONG = 500       # casual queries shouldn't produce essays
_COMPLEX_LONG = 5000     # even complex queries shouldn't hit this


def check_response_length(
    response_text: str,
    query_text: str,
) -> CheckResult:
    """Check if response length is appropriate for the query type."""
    length = len(response_text.strip())
    is_casual = bool(_CASUAL_PATTERNS.match(query_text.strip()))

    details: Dict[str, Any] = {
        "response_length": length,
        "is_casual_query": is_casual,
    }

    # Too short — likely truncated or empty
    if length < _SHORT_THRESHOLD:
        return CheckResult(
            check_name="response_length",
            passed=False,
            score=0.0,
            details={**details, "issue": "truncated", "threshold": _SHORT_THRESHOLD},
        )

    # Too long for casual
    if is_casual and length > _CASUAL_LONG:
        over_ratio = length / _CASUAL_LONG
        score = max(0.0, 1.0 - (over_ratio - 1.0) * 0.5)  # penalize proportionally
        return CheckResult(
            check_name="response_length",
            passed=False,
            score=score,
            details={**details, "issue": "verbose_casual", "threshold": _CASUAL_LONG},
        )

    # Too long for anything
    if length > _COMPLEX_LONG:
        return CheckResult(
            check_name="response_length",
            passed=False,
            score=0.3,
            details={**details, "issue": "excessive", "threshold": _COMPLEX_LONG},
        )

    return CheckResult(
        check_name="response_length",
        passed=True,
        score=1.0,
        details=details,
    )


# ---------------------------------------------------------------------------
# Check 2: Thinking leak detection
# ---------------------------------------------------------------------------

# Patterns that indicate leaked chain-of-thought
_THINKING_PATTERNS = [
    re.compile(r"<thinking>", re.IGNORECASE),
    re.compile(r"</thinking>", re.IGNORECASE),
    re.compile(r"^(Let me think|I need to consider|First, I'll analyze)", re.MULTILINE),
    re.compile(r"(Step \d+:|First,.*Second,.*Third,)", re.DOTALL),
    re.compile(r"\bthe user (is|wants|seems|might|asked)\b", re.IGNORECASE),
    re.compile(r"\bI should (respond|check|look|consider|mention)\b", re.IGNORECASE),
    re.compile(r"\bmy (instructions|system prompt|context) (say|tell|indicate)\b", re.IGNORECASE),
    re.compile(r"\b(according to|based on) (my|the) (memory|profile|context)\b", re.IGNORECASE),
]


def check_thinking_leak(response_text: str) -> CheckResult:
    """Detect leaked reasoning/thinking content in the response."""
    hits: List[str] = []
    for pattern in _THINKING_PATTERNS:
        matches = pattern.findall(response_text)
        if matches:
            hits.extend(str(m) for m in matches[:3])

    if len(hits) >= 2:
        return CheckResult(
            check_name="thinking_leak",
            passed=False,
            score=0.0,
            details={"leak_patterns": hits[:5], "hit_count": len(hits)},
        )

    return CheckResult(
        check_name="thinking_leak",
        passed=True,
        score=1.0,
        details={"hit_count": len(hits)},
    )


# ---------------------------------------------------------------------------
# Check 3: Profile grounding
# ---------------------------------------------------------------------------

def _extract_profile_facts(prompt_text: str) -> Dict[str, str]:
    """Extract name=value facts from the [USER PROFILE] section of a prompt."""
    facts: Dict[str, str] = {}

    # Find the USER PROFILE section
    match = re.search(
        r"\[USER PROFILE\].*?\n(.*?)(?=\n\[(?:ACTIVE|CODEBASE|TIME|TEMPORAL|SHORT|CURRENT))",
        prompt_text,
        re.DOTALL,
    )
    if not match:
        return facts

    profile_text = match.group(1)

    # Extract key=value pairs like "brother_name=Dillion"
    for m in re.finditer(r"(\w+)=([^;\[\]\n,]+)", profile_text):
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if len(value) > 1 and value.lower() not in ("true", "false", "none"):
            facts[key] = value

    return facts


def _extract_names_from_response(response_text: str) -> List[str]:
    """Extract proper names (capitalized words) from a response."""
    # Match capitalized words that look like names (not sentence starts)
    names: List[str] = []
    # Find words that are capitalized mid-sentence
    for m in re.finditer(r"(?<=[a-z] )([A-Z][a-z]{2,})", response_text):
        names.append(m.group(1))
    # Also find names after possessives, articles, etc.
    for m in re.finditer(r"(?:named?|called|is|brother|sister|pet|cat|dog|boss|friend)\s+\**([A-Z][a-z]+)", response_text):
        names.append(m.group(1))
    return list(set(names))


def check_profile_grounding(
    response_text: str,
    prompt_text: str,
) -> CheckResult:
    """Check if names/facts mentioned in the response are grounded in the prompt.

    Looks for proper names in the response and checks if they appear
    somewhere in the prompt (profile, memories, or any section).
    """
    names_in_response = _extract_names_from_response(response_text)
    if not names_in_response:
        return CheckResult(
            check_name="profile_grounding",
            passed=True,
            score=1.0,
            details={"names_found": 0, "all_grounded": True},
        )

    prompt_lower = prompt_text.lower()
    grounded: List[str] = []
    ungrounded: List[str] = []

    for name in names_in_response:
        if name.lower() in prompt_lower:
            grounded.append(name)
        else:
            ungrounded.append(name)

    total = len(names_in_response)
    grounded_ratio = len(grounded) / total if total > 0 else 1.0

    return CheckResult(
        check_name="profile_grounding",
        passed=len(ungrounded) == 0,
        score=grounded_ratio,
        details={
            "names_found": total,
            "grounded": grounded,
            "ungrounded": ungrounded,
            "grounded_ratio": grounded_ratio,
        },
    )


# ---------------------------------------------------------------------------
# Check 4: Citation validity
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(r"\[(MEM|WEB|GRAPH)_\d+\]")


def check_citation_validity(
    response_text: str,
    prompt_text: str,
) -> CheckResult:
    """Check if citation markers in the response reference sources in the prompt.

    Validates [MEM_N], [WEB_N], [GRAPH_N] markers against what's actually
    in the prompt context.
    """
    citations = _CITATION_PATTERN.findall(response_text)
    full_citations = _CITATION_PATTERN.findall(response_text)
    full_citation_strs = re.findall(r"\[(?:MEM|WEB|GRAPH)_\d+\]", response_text)

    if not full_citation_strs:
        return CheckResult(
            check_name="citation_validity",
            passed=True,
            score=1.0,
            details={"citations_found": 0, "all_valid": True},
        )

    valid: List[str] = []
    invalid: List[str] = []

    for cite in full_citation_strs:
        if cite in prompt_text:
            valid.append(cite)
        else:
            invalid.append(cite)

    total = len(full_citation_strs)
    valid_ratio = len(valid) / total if total > 0 else 1.0

    return CheckResult(
        check_name="citation_validity",
        passed=len(invalid) == 0,
        score=valid_ratio,
        details={
            "citations_found": total,
            "valid": valid,
            "invalid": invalid,
            "valid_ratio": valid_ratio,
        },
    )


# ---------------------------------------------------------------------------
# Check 5: Filler detection
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    re.compile(r"feel free to (ask|reach out|let me know)", re.IGNORECASE),
    re.compile(r"if you (have any|need any|want any) (other |more )?(questions|help|info)", re.IGNORECASE),
    re.compile(r"(don't hesitate|I'm here) to help", re.IGNORECASE),
    re.compile(r"Is there anything else", re.IGNORECASE),
    re.compile(r"Let me know if (you'd like|you want|there's)", re.IGNORECASE),
    re.compile(r"I hope (this|that) helps", re.IGNORECASE),
    re.compile(r"Happy to (help|assist|elaborate)", re.IGNORECASE),
]


def check_filler(response_text: str) -> CheckResult:
    """Detect generic filler phrases that pad responses without adding value."""
    hits: List[str] = []
    for pattern in _FILLER_PATTERNS:
        matches = pattern.findall(response_text)
        if matches:
            hits.extend(str(m) for m in matches[:2])

    # Filler at the end is worse than filler mid-response
    last_100 = response_text[-100:] if len(response_text) > 100 else response_text
    tail_filler = sum(1 for p in _FILLER_PATTERNS if p.search(last_100))

    score = 1.0
    if hits:
        score = max(0.0, 1.0 - len(hits) * 0.2 - tail_filler * 0.3)

    return CheckResult(
        check_name="filler",
        passed=len(hits) == 0,
        score=score,
        details={
            "filler_count": len(hits),
            "tail_filler": tail_filler,
            "examples": hits[:3],
        },
    )


# ---------------------------------------------------------------------------
# Run all checks on one response
# ---------------------------------------------------------------------------

def run_all_checks(
    response_text: str,
    prompt_text: str,
    query_text: str,
) -> List[CheckResult]:
    """Run all objective checks on a single response."""
    return [
        check_response_length(response_text, query_text),
        check_thinking_leak(response_text),
        check_profile_grounding(response_text, prompt_text),
        check_citation_validity(response_text, prompt_text),
        check_filler(response_text),
    ]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_checks_on_generation_run(
    gen_run_dir: str,
) -> List[ResponseCheckResults]:
    """Run all checks on every result from a Phase 4 generation run."""
    results_path = Path(gen_run_dir) / "results"
    all_results: List[ResponseCheckResults] = []

    for path in sorted(results_path.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            pair = data["pair"]
            result = data["result"]
            prompt = data.get("prompt_text", "")

            checks = run_all_checks(
                response_text=result["response_text"],
                prompt_text=prompt,
                query_text=pair["query_text"],
            )

            all_results.append(ResponseCheckResults(
                snapshot_id=pair["snapshot_id"],
                variant_id=pair["variant_id"],
                strategy=pair["strategy"],
                query_text=pair["query_text"],
                sections_removed=pair.get("sections_removed", []),
                checks=checks,
            ))
        except Exception:
            continue

    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_checks_by_section(
    results: List[ResponseCheckResults],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate check results by removed section (LOO only).

    For each section, compares baseline pass rates with variant pass rates
    to detect which sections help or hurt on each check.

    Returns: {section: {check_name: {baseline_pass_rate, variant_pass_rate, delta}}}
    """
    # Collect baseline check results per snapshot
    baseline_checks: Dict[str, Dict[str, CheckResult]] = {}
    for r in results:
        if r.variant_id == "__baseline__":
            baseline_checks[r.snapshot_id] = {c.check_name: c for c in r.checks}

    # Collect LOO variant results by section
    section_data: Dict[str, Dict[str, List[tuple]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        if r.strategy != "leave_one_out" or not r.sections_removed:
            continue
        section = r.sections_removed[0]
        bl = baseline_checks.get(r.snapshot_id, {})

        for check in r.checks:
            bl_check = bl.get(check.check_name)
            bl_score = bl_check.score if bl_check else 1.0
            section_data[section][check.check_name].append(
                (bl_score, check.score)
            )

    # Compute aggregates
    result: Dict[str, Dict[str, Any]] = {}
    for section, checks in sorted(section_data.items()):
        section_stats: Dict[str, Any] = {}
        for check_name, pairs in checks.items():
            bl_scores = [p[0] for p in pairs]
            var_scores = [p[1] for p in pairs]
            bl_avg = sum(bl_scores) / len(bl_scores) if bl_scores else 0
            var_avg = sum(var_scores) / len(var_scores) if var_scores else 0
            section_stats[check_name] = {
                "baseline_avg_score": round(bl_avg, 3),
                "variant_avg_score": round(var_avg, 3),
                "delta": round(var_avg - bl_avg, 3),
                "n": len(pairs),
            }
        result[section] = section_stats

    return result


def format_checks_report(
    section_stats: Dict[str, Dict[str, Any]],
) -> str:
    """Human-readable report of objective check results per section."""
    check_names = ["response_length", "thinking_leak", "profile_grounding",
                   "citation_validity", "filler"]

    lines = [
        "=== Objective Checks Report (LOO) ===",
        "",
        "Delta = variant_score - baseline_score",
        "Positive delta = removing section IMPROVED this check (section was hurting)",
        "Negative delta = removing section WORSENED this check (section was helping)",
        "",
    ]

    # Per-check tables
    for check_name in check_names:
        lines.append(f"--- {check_name} ---")
        lines.append(
            f"{'Section':<28s} {'BL avg':>7s} {'Var avg':>8s} {'Delta':>7s} {'N':>4s}"
        )
        lines.append("-" * 58)

        entries = []
        for section, checks in sorted(section_stats.items()):
            stats = checks.get(check_name)
            if not stats:
                continue
            entries.append((section, stats))

        # Sort by delta (most positive first = sections that hurt most)
        entries.sort(key=lambda x: x[1]["delta"], reverse=True)

        for section, stats in entries:
            lines.append(
                f"{section:<28s} {stats['baseline_avg_score']:>7.3f} "
                f"{stats['variant_avg_score']:>8.3f} "
                f"{stats['delta']:>+7.3f} "
                f"{stats['n']:>4d}"
            )
        lines.append("")

    return "\n".join(lines)
