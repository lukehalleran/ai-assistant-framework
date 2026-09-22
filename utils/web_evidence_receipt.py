"""Per-turn web-evidence receipt and the spent-budget notice (2026-09-12).

Adversarial review F4. When the daily web-search budget is spent, the budget
veto turns a turn that WANTED fresh evidence into one that looks exactly like
"no search needed", and the reply answers from priors with nothing telling the
user. On 2026-09-11 fourteen turns after 19:13 got zero web evidence this way;
one reply said so.

The fix is a receipt built at final delivery from what THIS turn recorded,
plus one short notice when a need went unmet because of the budget. The notice
is generated after retries, the action guard and the grounding check, so the
streamed bubble and the stored reply carry the same single line.

Sources (all per-turn, never the previous turn's state):
  * the agentic gate decision's ``web_evidence_blocked`` — Tier 1/Tier 4 stood
    a web arm down because a paid search could not run;
  * the prompt builder's ``raw_context["web_search_decision"]`` receipt — the
    enhanced-path trigger's requested / blocked / results / error;
  * this turn's agentic session rounds — web results, the manager's budget
    refusal, and URL fetches that returned a page;
  * a round's or a ``WebSearchResult``/``MultiSearchResult``'s own typed
    ``blocked`` field (2026-09-12, follow-up findings 2/3) — set even when
    the same round/result also carries pages, so a PARTIAL budget refusal
    (one sub-query funded, one refused; a free direct fetch that could not
    afford its billed Tavily fallback) is never silently discarded just
    because something else in the round succeeded.

Only a BUDGET block produces a notice. A disabled toggle is the owner's
choice, and other failures (timeouts, provider errors) are out of scope here.

Leaf module: no application imports. Both delivery sites in gui/handlers.py
and the tests share this one implementation.
"""

from typing import Any, Dict, Optional

import utils.read_time_markers as read_time_markers

# The error prefix WebSearchManager.search returns when the limiter refuses a
# search (multi_search repeats it per sub-query). Pinned against the deployed
# manager by tests/unit/test_sep12_web_evidence_budget.py.
BUDGET_ERROR_MARKER = "Daily credit limit reached"

BUDGET_NOTICE = read_time_markers.delivery_notice(
    read_time_markers.NOTICE_WEB_BUDGET,
    " is used up, so this answer isn't checked against current sources.",
)
BUDGET_NOTICE_PARTIAL = read_time_markers.delivery_notice(
    read_time_markers.NOTICE_WEB_BUDGET_PARTIAL,
    " is used up, so this answer relies only on the sources I could already get.",
)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_budget_block(value: Any) -> bool:
    """True only for the literal string "budget" — a MagicMock attribute (or
    any other non-str truthy stand-in) never counts as a real block reason."""
    return isinstance(value, str) and value == "budget"


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def build_web_evidence_receipt(
    gate_decision: Any = None,
    web_decision: Optional[Dict[str, Any]] = None,
    session: Any = None,
) -> Dict[str, Any]:
    """What this turn wanted from the web and what it actually got.

    Returns ``{"requested", "blocked", "acquired", "fetched"}``: whether fresh
    web evidence was wanted; the block reason when a paid search could not
    run (``"budget"``/``"disabled"``) or None; search/cache pages acquired;
    pages returned by URL fetches."""
    receipt: Dict[str, Any] = {
        "requested": False, "blocked": None, "acquired": 0, "fetched": 0,
    }

    gate_blocked = getattr(gate_decision, "web_evidence_blocked", None)
    if _is_text(gate_blocked):
        receipt["requested"] = True
        receipt["blocked"] = gate_blocked

    if isinstance(web_decision, dict):
        if web_decision.get("requested") is True or web_decision.get("triggered") is True:
            receipt["requested"] = True
        decision_blocked = web_decision.get("blocked")
        if _is_text(decision_blocked):
            receipt["requested"] = True
            receipt["blocked"] = receipt["blocked"] or decision_blocked
        receipt["acquired"] += _count(web_decision.get("results"))
        error = web_decision.get("error")
        if isinstance(error, str) and BUDGET_ERROR_MARKER in error:
            receipt["requested"] = True
            receipt["blocked"] = receipt["blocked"] or "budget"

    rounds = getattr(session, "rounds", None) if session is not None else None
    for rnd in rounds if isinstance(rounds, list) else []:
        if _is_budget_block(getattr(rnd, "blocked", None)):
            receipt["requested"] = True
            receipt["blocked"] = receipt["blocked"] or "budget"
        query = getattr(getattr(rnd, "request", None), "query", "")
        if isinstance(query, str) and query.startswith("[Fetch URL]"):
            # Fetch failures render as bracketed notes ("[Could not fetch
            # content from …]", "[URL fetch error: …]"); each fetched page
            # renders its own "Title: …\nURL: …" header.
            summary = getattr(rnd, "summary", None)
            if isinstance(summary, str):
                receipt["fetched"] += summary.count("\nURL: ")
            continue
        results = getattr(rnd, "results", None)
        if results is None:
            continue
        pages = getattr(results, "pages", None)
        if isinstance(pages, list):
            receipt["acquired"] += len(pages)
        if _is_budget_block(getattr(results, "blocked", None)):
            receipt["requested"] = True
            receipt["blocked"] = receipt["blocked"] or "budget"
        error = getattr(results, "error", None)
        if isinstance(error, str) and BUDGET_ERROR_MARKER in error:
            receipt["requested"] = True
            receipt["blocked"] = receipt["blocked"] or "budget"

    return receipt


def web_evidence_notice(receipt: Optional[Dict[str, Any]]) -> str:
    """The one notice for an unmet web-evidence need, or "" when none applies.
    Partial wording when some search/cache/fetch evidence was still acquired:
    a successfully fetched page must never be described as a failed check."""
    if not isinstance(receipt, dict):
        return ""
    if receipt.get("requested") is not True or receipt.get("blocked") != "budget":
        return ""
    if _count(receipt.get("acquired")) or _count(receipt.get("fetched")):
        return BUDGET_NOTICE_PARTIAL
    return BUDGET_NOTICE


def apply_web_evidence_notice(text: Optional[str], receipt: Optional[Dict[str, Any]]) -> str:
    """Append the notice once (idempotent) and return the text."""
    body = text or ""
    notice = web_evidence_notice(receipt)
    if not notice:
        return body
    if BUDGET_NOTICE.strip() in body or BUDGET_NOTICE_PARTIAL.strip() in body:
        return body
    return body.rstrip() + notice
