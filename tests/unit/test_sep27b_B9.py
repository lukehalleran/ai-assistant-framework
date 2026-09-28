"""Lane B batch B9: personal-claim omission scope + timeout budget.

Evidence: ``~/daemon_exec/sep27_runs/S10/personal_claim_rollup.md`` (378
production checks, 2026-09-15..09-27): 37.3% of checks timed out at the old
``timeout_s=5.0`` and the checked-only p90 (4.711s) already sat right at that
ceiling; separately, 79% of successfully checked replies had >=1 unsupported
claim, but a 10-sample manual review found 7/10 were discussion/advice/
explanatory sentences (a news-story opinion, interview-prep advice, a stats
explanation) the auditor's necessarily broad system prompt flags as claims
about "another person"/"another episode" even though they assert nothing
about what the user did.

Fixes (class: BC-47, BC-46, BC-84), staying in ``log_only``:
1. ``omit_unsupported_claims`` only removes a sentence for a completion /
   personal_completion / event-kind claim (``_is_completion_kind`` already
   gates the demotion path above it) -- a discussion/plan/suggestion/other
   claim is recorded on the receipt but never physically omitted.
2. ``config.yaml``'s ``personal_claim_check.timeout_s`` moves from 5.0 to
   6.5 (measured p90 + margin against the code's actual model default,
   gpt-4o-mini) so the check stops discarding a third of its coverage as
   pure timeout waste; ``audit_personal_claims``'s own default keyword
   argument documents the same number for a direct call.

No model call is made here -- ``PersonalClaimResult``/``omit_unsupported_claims``
are exercised directly (as the existing suite does), and the timeout number is
read from the actually-loaded ``config.app_config`` constant, not re-derived.
"""

from __future__ import annotations

import inspect

import config.app_config as app_config
from core.personal_claim_check import (
    PersonalClaimResult,
    _omission_eligible_kind,
    audit_personal_claims,
    omit_unsupported_claims,
)


def _result(claims):
    return PersonalClaimResult("checked", "ok", claims=claims)


def test_discussion_kind_unsupported_claim_is_recorded_never_omitted():
    """A non-completion-kind claim stays in the response even when flagged.

    Live sample #6 (09-20 18:32, technical_help): "That's the right instinct
    -- those two are exactly the kind of questions where 'I used it' and 'I
    und..." was flagged insufficient/demoted as a claim about "another
    person" although it is career-prep advice, not a claim the user did
    anything. Before this fix, ``omit_unsupported_claims`` stripped it purely
    on status; after, kind gates it too.
    """
    response = "That's the right instinct for interviews. You uploaded the resume yesterday."
    result = _result([
        {"text": "That's the right instinct for interviews", "status": "insufficient",
         "kind": "discussion", "evidence": []},
        {"text": "You uploaded the resume yesterday", "status": "insufficient",
         "kind": "personal_completion", "evidence": []},
    ])
    revised = omit_unsupported_claims(response, result)
    assert "That's the right instinct for interviews." in revised
    assert "You uploaded the resume yesterday" not in revised
    # The receipt still carries both -- recorded, never silently dropped.
    counts = result.receipt()
    assert counts["insufficient_count"] == 2


def test_contradicted_discussion_kind_is_also_not_omission_eligible():
    """Scope is kind-based, not status-based: even a *contradicted* verdict
    on a discussion-kind claim is left in place -- only claims about the
    user's own state are physically removable."""
    response = "The purges clearly worked out fine. You finished the repair yesterday."
    result = _result([
        {"text": "The purges clearly worked out fine", "status": "contradicted",
         "kind": "discussion", "evidence": []},
        {"text": "You finished the repair yesterday", "status": "contradicted",
         "kind": "completion", "evidence": []},
    ])
    revised = omit_unsupported_claims(response, result)
    assert "The purges clearly worked out fine." in revised
    assert "You finished the repair yesterday" not in revised


def test_all_completion_kind_variants_remain_eligible():
    """personal_completion/event/personal_event/achievement all still omit --
    only the demotion path's own kind test changed scope, not its logic."""
    for kind in ("completion", "personal_completion", "event", "personal_event", "achievement"):
        assert _omission_eligible_kind(kind)
        response = "Setup line. You completed that task on Tuesday."
        result = _result([
            {"text": "You completed that task on Tuesday", "status": "insufficient",
             "kind": kind, "evidence": []},
        ])
        assert omit_unsupported_claims(response, result) == "Setup line."


def test_non_personal_kinds_are_never_eligible_personal_state_kinds_are():
    for kind in ("discussion", "suggestion", "quote", "causal", "Discussion"):
        assert not _omission_eligible_kind(kind)
    # Statements about the user's own state stay eligible — the 09-15 live
    # incident was a PLAN claim; a mixed label keeps its personal half.
    for kind in ("plan", "future plan", "conditional", "negation", "partial",
                 "cancellation", "obligation", "other", "plan/suggestion", "completed action"):
        assert _omission_eligible_kind(kind)


def test_only_unsupported_claims_of_any_kind_are_unaffected():
    """A supported claim was never omission-eligible and still is not,
    regardless of kind -- this fix only narrows the unsupported branch."""
    response = "You aced the exam. That's a great strategy in general."
    result = _result([
        {"text": "You aced the exam", "status": "supported", "kind": "personal_completion",
         "evidence": [{"source_id": "s1", "quote": "aced"}]},
        {"text": "That's a great strategy in general", "status": "supported", "kind": "discussion",
         "evidence": [{"source_id": "s1", "quote": "strategy"}]},
    ])
    assert omit_unsupported_claims(response, result) == response


def test_timeout_default_matches_measured_p90_plus_margin():
    """S10 rollup: checked-only p90 was 4.711s against the old 5.0s ceiling
    (37.3% of all checks timed out). 6.5s = measured p90 + ~1.8s margin."""
    sig = inspect.signature(audit_personal_claims)
    assert sig.parameters["timeout_s"].default == 6.5


def test_deployed_config_timeout_is_the_raised_budget_not_the_old_ceiling():
    """Reads config.app_config's ACTUAL loaded constant (YAML -> schema ->
    app_config), not a re-derivation or a hardcoded historical baseline."""
    assert app_config.PERSONAL_CLAIM_TIMEOUT_S == 6.5
    assert app_config.PERSONAL_CLAIM_TIMEOUT_S > 5.0
