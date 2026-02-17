"""
Markdown report generator for retrieval benchmark results.

Produces a grouped summary table by intent type with per-case details
for failures.
"""

from typing import Dict, List, Any


def generate_report(results: List[Dict[str, Any]]) -> str:
    """
    Generate a markdown report from benchmark results.

    Args:
        results: List of BenchmarkResult.to_dict() outputs.

    Returns:
        Markdown-formatted report string.
    """
    lines: List[str] = []
    lines.append("# Retrieval Quality Benchmark Report\n")

    # Group by intent type
    by_intent: Dict[str, List[Dict]] = {}
    for r in results:
        intent = r.get("intent_expected", "unknown")
        by_intent.setdefault(intent, []).append(r)

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Intent | Cases | Passed | Failed | Avg Recall | Avg MRR |")
    lines.append("|--------|------:|-------:|-------:|-----------:|--------:|")

    total_cases = total_pass = total_fail = 0
    for intent in sorted(by_intent.keys()):
        cases = by_intent[intent]
        passed = sum(1 for c in cases if c.get("passed"))
        failed = len(cases) - passed
        total_cases += len(cases)
        total_pass += passed
        total_fail += failed
        avg_recall = sum(c.get("recall_at_k", 0) for c in cases) / max(len(cases), 1)
        avg_mrr = sum(c.get("mrr", 0) for c in cases) / max(len(cases), 1)
        lines.append(
            f"| {intent} | {len(cases)} | {passed} | {failed} "
            f"| {avg_recall:.3f} | {avg_mrr:.3f} |"
        )

    lines.append(
        f"| **TOTAL** | **{total_cases}** | **{total_pass}** | **{total_fail}** | | |"
    )
    lines.append("")

    # Failed cases detail
    failures = [r for r in results if not r.get("passed")]
    if failures:
        lines.append("## Failed Cases\n")
        for f in failures:
            lines.append(f"### {f['case_id']}")
            lines.append(f"- **Intent**: expected `{f['intent_expected']}`, "
                         f"got `{f['intent_actual']}` (conf={f['intent_confidence']:.2f})")
            lines.append(f"- **Recall@K**: {f['recall_at_k']:.3f}")
            lines.append(f"- **Retrieved**: {f.get('retrieved_ids', [])}")
            for reason in f.get("failure_reasons", []):
                lines.append(f"- {reason}")
            if f.get("false_retrievals"):
                lines.append(f"- **Should NOT have retrieved**: {f['false_retrievals']}")
            if f.get("order_violations"):
                lines.append(f"- **Order violations**: {f['order_violations']}")
            lines.append("")
    else:
        lines.append("## All cases passed.\n")

    return "\n".join(lines)
