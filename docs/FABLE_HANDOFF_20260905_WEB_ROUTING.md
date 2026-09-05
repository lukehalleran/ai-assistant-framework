# Fable handoff — September 5 web-routing follow-up

User requested economical delegation: frontier handles architecture/review; cheaper
delegates own bounded patches and ask frontier for help when uncertain. User gave
25% then 10% credit warnings. Scope frozen; stop broad audits and finish validation.
Do not trust an agent's claims without checking files/evidence. No commits/restarts.

## Read first / preserve

- Earlier completed pass: `docs/INDEPENDENT_AUDIT_20260905.md` and
  `docs/INDEPENDENT_AUDIT_HANDOFF_20260905.md`. Earlier 280-test pass and isolated
  startup/persistence smoke DO NOT validate the additional changes below.
- Branch master, original HEAD e6e515de82aff941484c1d2b388746f8a33749af. Dirty tree
  contains original interrupted Claude work plus the previous audit plus this batch.
  No changes staged or committed. Preserve all unrelated work.
- Original starting snapshot: `/tmp/daemon-audit-baseline-cAK40N1q/`.
- Before root's latest trigger/gate edits: `/tmp/daemon-web-pass-baseline-Znk9t247/`
  holds utils/web_search_trigger.py and core/agentic/gate.py; gatherer_web.py there
  already includes the delegate's removal of the general-intent veto.
- Production Python main.py PID1062969 was previously identified from this repo,
  started11:15 September5; not stopped/restarted. Recheck identity/state, do not
  assume source edits are loaded or kill a stale recorded PID.

## Exact confirmed incident

User query: `Farage caught doing crimes this week? Meeting with Russian proxies and receiving illegal money. This has to be releated`

Root daemon_debug.log at11:23:23: one strong recency cue scored0.40, threshold0.50.
gpt-4o-mini classifier timed out after5s; fallback reused no-search heuristic.
Intent classifier separately returned general/confidence0.00. The enhanced web
gatherer then unconditionally skipped intent=general. Answer had no fresh evidence.
Reproduced classifier-unavailable outcome read-only before edits: false/0.40/0.50.

Turn1 at11:16 DID fetch BBC URL directly: 9195chars. Base prompt web_search=ON(0)
preceded agentic fetching and is not final-generation evidence status. Do not infer
that Turn1 fabricated a fetch or that the underlying political allegations are true.
This is a routing investigation, not verification of those allegations.

## Implemented latest batch

1. Cheap delegate web_fallback (gpt-5.6-luna/high) owns
   core/prompt/gatherer_web.py + tests/unit/test_web_fallback_general_intent.py.
   Removed ONLY general from hard non-search intent set. Unknown/general now consult
   the shared trigger; existing explicit non-search/crisis/disabled protections stay.
   Delegate9tests passed. Root read full patch/tests and corrected explanatory comment.
2. Root utils/web_search_trigger.py adds requires_fresh_public_evidence(query), a
   conservative POSITIVE rule: recency cue + question anywhere + public-event terms,
   excluding personal/mixed/document requests and explicit no-search instructions.
   No politician/party/outlet is special-cased. False does NOT mean no search needed:
   it leaves existing routing in control. Examples beyond this narrow scope still
   need future handling (pronoun-only followups, public questions with personal wrappers).
3. should_search_heuristic uses that rule to return one quick search, without another
   model classifier. core/agentic/gate.py uses SAME helper in Tier1 and seeds search
   terms, so the exact incident routes to web even without any model manager.
4. Trigger cache key now includes enabled flag, crisis level and remaining credits.
   Previously a cached positive could bypass later disabled/crisis policy because
   lookup happened before guards and its key excluded policy.
5. Unavailable classifier fallback now has source=fallback and an explanatory reason,
   rather than looking like successful classification. It does not distinguish
   timeout vs parse/other classifier failure; lower-level logs provide detail.
6. Root tests/unit/test_public_news_verification.py adds19mocked cases: exact incident,
   additional public-event questions, personal/static/mixed/negated negative controls,
   gate without model, cache policy isolation, visible classifier failure.

## Validation state at handoff

- Ruff fatal subset passed on all five latest Python files:
  `python -m ruff check utils/web_search_trigger.py core/agentic/gate.py core/prompt/gatherer_web.py tests/unit/test_public_news_verification.py tests/unit/test_web_fallback_general_intent.py --select E9,F63,F7,F82`
- git diff --check passed. No new runtime/provider validation, no restart.
- The broader command completed before the interrupt took effect: **198 passed,
  3 FAISS deprecation warnings, 200.43s**. Exact command:
  `python -m pytest tests/unit/test_public_news_verification.py tests/unit/test_web_fallback_general_intent.py tests/test_web_search_trigger.py tests/unit/test_agentic_gate.py tests/unit/test_stm_gate_recent_history.py -q --color=no`
  It appeared stalled after37passes but finished normally. Legacy heuristic tests
  can call real semantic-anchor embedding code; isolate that for a reliably fast lane.
- Final two-file isolated run result will be appended below when available.

## Prepared proposals — NOT IMPLEMENTED

Cheap delegates public_context and tool_receipts were read-only and stopped at25%.
Root has not accepted all their recommendations; use them as starting hypotheses.

### A. Better public-news context (next priority after validating routing)

Candidate seam: ContextPipeline.build() classifies short ORIGINAL user_input before
attachment merging and records a metadata flag. build_prompt_from_context() passes
bounded retrieval overrides to builder. Do not run the helper on giant merged text.
For clearly public questions, test reducing recent history to2 and avoiding unrelated
memories/summaries/reflections/wiki/notes/calendar/proactive sources. Preserve original
attachments, last exchange, identity/time, location when relevant, and mixed personal
questions. The pure helper is deliberately narrow; do not treat it as a full policy.

Fact extraction on this incident took2.38s and yielded0new facts. STM input4669tokens.
Consider skipping or bounding these stages for public news, but DON'T blindly remove
STM: reference resolution and continuity must be tested first. The delegate suggested
zeroing most sources; root deferred that material quality decision, not approved it.
Regression coverage needed: current event, personal/doc negative controls, attached
public article retained, mixed personal+public untouched, local-news location retained,
older ContextResult without metadata compatible. Existing query_checker.is_question
misses an internal question mark when the message ends with a statement; not fixed.

### B. Actual web tool receipts (small coherent follow-up)

Existing seams:
- core/agentic/controller.py initial URL fastpath ~516–566 fetches URLs directly;
  normal tool dispatch ~1171–1213 and wrappers ~1311–1355 also matter. Instrumenting
  only tools.py misses controller wrappers. Line numbers shift with edits.
- core/agentic/types.py AgenticSearchSession.get_provenance_summary() currently emits
  bounded agentic_rounds. gui/handlers.py _attach_agentic_provenance copies selected
  keys; _build_debug_record preserves provenance. API /provenance forwards/redacts it.
- Add bounded metadata receipts (tool, round, target, attempted, result count/content
  length, duration, status), NOT raw result bodies or another full prompt dump.
- Preserve distinction between base prompt inventory and later tool evidence.
- IMPORTANT root caveat: do not label every unavailable tool attempted=true; and
  do not invent timeout/empty distinctions by guessing from prose. Direct fetch
  manager collapses some timeout/network failures to empty results. Exact states
  need structured manager return status, or report unknown/error truthfully.
- Possible files: core/agentic/types.py, controller.py, gui/handlers.py, new tests.
  No API schema/frontend rewrite required to expose provenance metadata.
- Tests: successful URL fastpath metadata, empty/error/unavailable results, bounded
  multiround receipts, search result counts, handler/API redaction; no body duplication.

## Recommended continuation pattern

1. Inspect git status/diff; verify final isolated tests below; resolve broader tests
   with real providers/stores blocked and semantic boosts mocked where appropriate.
2. Frontier review edge cases and policy, delegate only clearly owned mechanical
   implementation/test work. Delegates stop and ask frontier when semantics change.
3. Take ONE of A or B, not both plus another audit. Require focused regression and
   an evidence artifact before expanding scope. Do not spend model calls to decide
   a clear search requirement or feed a full GUI export to every delegate.
4. Before deploying, identify the actual live process/version and choose a safe
   restart window; previous smoke used disposable stores and deterministic generation.
5. Keep original user changes separate in any eventual commits; no blanket commit.

## Fable follow-up (2026-09-05 afternoon) — verification, hardening, live probes

- Verified the batch against the files (not the report): diff read, deployed-function probes, 198+28 → cross-batch 101 → combined 648-test regression. Ruff floor + `git diff --check` clean.
- Deployed-function probe of `requires_fresh_public_evidence` found two gaps, both fixed by a sonnet delegate under a Fable spec and re-probed: (a) UNDER-fire on second-person request wrappers ("Can you check if Farage was arrested this week?", "Did you see that the president was charged this week?") — a leading wrapper is stripped and satisfies the question shape; (b) OVER-fire on pronoun-less private questions ("Is the homework due this week? The syllabus reports Sep 13.") — weak public terms (reports/announced/company) count only when no private-sphere noun is present. The incident string still fires; "What did you say yesterday about the courts?" and "Can you check my email from the court this week?" still don't.
- Owner relaunched at 12:58 (PID 1075790) with the three batches. Live probe 1 (`Were any UK or US politicians charged with crimes this week? What did the police or courts announce?`): gate `web_search`, 5 Tavily searches, cited answer that also corrected the 11:24 miss (Channel 4 sting / Electoral Commission → Met referral; no charge). Live probe 2 (77-word Outlook read request): gate `tools`, two `email_search` rounds against Gmail, honest "Outlook is disabled on my end".
- Three secondary defects found in the probe records and fixed on disk (NOT in the running process): heavy-topic news turn latched the sticky floor onto the next request (tone_detector); [THREAD CONTEXT] asserted continuity with the news thread under the STM recall override (orchestrator); the same query string searched twice at two depths (web_search_manager cache). See CLAUDE_CHANGELOG.md 2026-09-05.
- Proposal A/B status: neither started. B (tool receipts) is the recommended next milestone; it overlaps the audit's duplicate-answer item (response-boundary IDs/hashes).
