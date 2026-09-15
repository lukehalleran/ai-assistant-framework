=== F8b: web search section outcome agrees with the web receipt (typed failure for exceptions, provider errors and budget refusals) ===
(Parent draft, 2026-09-14. It moves to docs/execution/generalization/briefs/F8b.md after F8a is integrated.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Decisions per request" → CGR-007;
- "F7 split and gatherer outcome shape" (typed return);
- the parent's F8 split (F8b = web gatherer #92, response `-5`).
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md. This batch answers anchor #92, whose packet note reads "confirm that the web evidence receipt distinguishes this outcome".
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-007-5.md.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL.

COMPLIANCE (read twice): source write only after the failing-first run is recorded; no non-pytest Python of any kind; check fixture inputs against the trigger/decision gates; disclose any breach.

MANIFEST CHECK: against `$S/manifest_post_F8a.txt` / `$S/manifest_paths_post_F8a.txt` (same two commands as other briefs; S = the parent scratchpad).
GUARD, MEMORY, DATA NOTE: as in R_common_rules.
- The data/ baseline has 7 entries, including an untouched user_profile.json.
- tests/test_web_search_manager.py is NEVER run: it writes data/web_search_credits.json and data/chroma_multi.
ORDER REMINDER: print `sha256sum core/prompt/gatherer_web.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): two existing tests pin today's `None`; update them only as described below.

PARENT-VERIFIED FACTS (core/prompt/gatherer_web.py `270b6208…` = the packet's source sha, 364 lines; re-verify)
- `_get_web_search_results` (def 68; ONLY caller: builder task "web_search" at builder.py 1513).
  - `self.last_web_decision` is initialised at 96–108 and updated on every branch (the receipt).
- Success path:
  - `result = await manager.multi_search(...)` (242 / 253);
  - `pages = list(getattr(result, "pages", None) or [])`, and `last_web_decision["results"] = len(pages)` (262–263);
  - a budget-blocked result sets `last_web_decision["requested"]`/`["blocked"] = "budget"` (268–273);
  - `if result.has_results:` → `memory_id_map["WEB_SEARCH"]` → `return result` (275–294). `WebSearchResult` is a plain dataclass (knowledge/web_search_manager.py:250, `has_results` = pages and no error at 274–276). It is always truthy.
  - else, if `result.error`: `last_web_decision["error"] = str(result.error)`, then `return None` (295–299). This is the PROVIDER-ERROR path (e.g. an F2 Tavily failure). It currently reads no_results in `_section_outcomes`.
  - with no error and no results (a genuine empty search, or a budget refusal without an error) → `return None`.
- Outer `except Exception as e` (301–304): `last_web_decision["error"] = type(e).__name__`, warning, `return None`. This is ANCHOR #92.
- Earlier returns of None (not triggered, vetoed, crisis-suppressed, no manager, …) are deliberate non-searches; leave them as `None` (no_results / not attempted). Find each one and confirm.
- Consumers:
  - the gather loop stores `raw or []` (builder 1552);
  - builder 1757 `"web_search_results": gathered.get("web_search")`; 1758 `"web_search_decision": last_web_decision`;
  - formatter 1090–1094 renders only `if web_search is not None and hasattr(web_search, 'has_results') and web_search.has_results`, so an empty OutcomeList is handled exactly like `[]`;
  - the formatter's `web_search=` label (833–851) reads ONLY `web_search_decision`, so a typed return does NOT change the prompt label;
  - F6b's `sections_not_checked` reads `_section_outcomes`, so after F8b a web failure appears there.
- Existing tests that pin `None` (FIXTURE RULE candidates):
  - tests/unit/test_sep10_web_search_gap.py::test_gatherer_exposes_search_exception (≈218–233): the except path, asserting `is None` plus receipt fields.
  - tests/unit/test_tavily_failure_outcomes.py::TestDeployedConsumers::test_gatherer_receipt_records_provider_failure (≈312–339): asserts `result is None`, the error set and the marker absent. Determine by reading which path it takes: the except, or the provider-error `result.error` path.
- Other references: test_sep10_web_search_gap.py (not-triggered decision), test_sep12_web_evidence_budget.py, test_sep12_followup_budget_outcomes.py, test_sep09_live_controls.py, test_web_fallback_general_intent.py, test_sep12_repository_status_context.py, test_independent_prompt_audit.py.

OWNERSHIP
- core/prompt/gatherer_web.py: the body of `_get_web_search_results` ONLY, plus one import line.
- The two FIXTURE RULE test edits above.
- New tests/unit/test_gatherer_outcomes_web.py.
- New docs/execution/generalization/batches/F8b.md.
- The response file named above.
- Read-only: knowledge/web_search_manager.py, utils/web_search_trigger.py, builder, formatter, orchestrator, handlers, everything else, and every class-guard path.

PARENT-VERIFIED (2026-09-14, read-only, knowledge/web_search_manager.py): a budget refusal sets BOTH fields.
- `search()` returns `WebSearchResult(error="Daily credit limit reached. Remaining: …", blocked="budget")` at 1317–1322 (reservation refused) and 1401–1406 (spend refused).
- A merged `MultiSearchResult` sets `error="; ".join(errors)` ONLY when no pages remain, and `blocked="budget"` when any sub-query was budget-refused (2521–2532).
- Other error results carry `error` with no `blocked`:
  - "Tavily client not available" (1327–1331);
  - "Tavily API key is invalid" (1418–1422, 1433–1437);
  - "Web search provider failed (<reason>)" (1423–1427).
- Therefore the gatherer must check `blocked == "budget"` BEFORE checking `error`.

CONTRACT
1. Anchor #92 (outer except): keep the receipt write and the log; return `OutcomeList.failed(type(e).__name__)` instead of None.
2. Budget refusal, checked FIRST inside the `not result.has_results` branch: when `getattr(result, "blocked", None) == "budget"`, keep the receipt writes (including `last_web_decision["error"]` exactly as today) and return `OutcomeList.unavailable("budget")`.
3. Provider-error result (`not result.has_results`, not budget-blocked, `result.error` set): keep the receipt write and return `OutcomeList.failed("provider_error")`.
   - Use a constant label, never `str(result.error)`.
   - Do NOT string-match the error text to split "client not available" or "invalid key" into other states (BC-76). Record that single label as a limitation.
4. A genuine empty search (no error, not blocked) and every deliberate non-search stay `None`: no_results / not attempted.
5. The success path is unchanged (it returns the `WebSearchResult` object). The receipts (`last_web_decision`) are byte-for-byte unchanged on every path.
6. Privacy: reasons are constant labels or exception class names only.

TESTS (tests/unit/test_gatherer_outcomes_web.py; fakes only; NO real WebSearchManager state files, limiter, cache, Tavily or network; reuse the fakes in test_sep10_web_search_gap.py and test_tavily_failure_outcomes.py, which build managers on tmp_path)
- FAILING FIRST in one command: `sha256sum core/prompt/gatherer_web.py`, then the new tests on the UNEDITED source.
- Cases:
  - raising `multi_search` → failed / class, `== []`, receipt error = class;
  - a provider-error result → failed / provider_error, receipt error set, marker text absent from the reason;
  - a budget-refused empty result → unavailable / budget, receipt blocked = "budget";
  - a genuine empty result → None;
  - a not-triggered decision → None;
  - success → the same `WebSearchResult`.
- Through the builder (`full_builder`): a raising search → `_section_outcomes["web_search"] == failed/<class>`; the prompt label still reads `web_search=ON(error)`; the web section is absent.
- FIXTURE RULE edits:
  - update `test_gatherer_exposes_search_exception` to assert `outcome_status(result) == ("failed", "TimeoutError")` and `result == []`, keeping every receipt assertion;
  - update `test_gatherer_receipt_records_provider_failure` to assert the typed failure for whichever path it takes, keeping the receipt and marker assertions;
  - the paired controls are the existing not-triggered test and the new genuine-empty test.
- Focused: the new file plus the 9 web test files above (2 chunks). Sweep: every remaining tests/unit importer of core.prompt.gatherer_web / WebSearchMixin, with the usual exclusions.

SCAN: expect the dm18 row for #92 to go STALE, with new = 0.
RESPONSE FILE (`-5`):
- #92 fixed, with receipt-agreement evidence;
- the provider-error and budget paths as BC-58 siblings fixed in the same method;
- CGR-007 fully answered (-1…-5), except #85, #86 and #89, which F8a answered with evidence (no code).
SIZE: target ≤300 changed lines; HARD stop at ≤450.
ORDER: manifest → create F8b.md → pre-edit scan → tests → failing-first (digest in the same command) → edit → FIXTURE RULE edits → focused → sweep → ruff → scan → data/logs listing → packet → response file.
