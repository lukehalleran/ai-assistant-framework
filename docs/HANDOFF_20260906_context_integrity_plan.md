# Fable handoff: evidence integrity, context continuity, and efficient inference

Prepared September 6, 2026 for Fable 5.1 / high. **Planning only; no application fixes or data repairs implemented by this handoff.** Reviewed working tree was clean at HEAD `778a37a`. Recheck HEAD, diffs, and loaded runtime before implementing. Line numbers below are navigation aids, not test assertions.

## 1. Owner requirements and corrections

- Build general behavior for users writing American English, including contractions, fragments, typos, implicit subjects, quotations, and follow-ups. Do not hardcode the owner's identity, medications, contacts, routines, timestamps, or wording. Replacing one incident regex with a larger incident regex is not generalization.
- This remains a **single-tenant personalized desktop application**. No multi-tenant accounts, tenant isolation architecture, or genericization that removes useful personal memory is requested. Source ownership and test/live distinctions still matter within one user's corpus.
- Earliest intended target is 2028, with a switch to all-local models. Preserve richer context and model-independent interfaces. Stronger future inference is a design opportunity, not an established capability or a reason to tolerate corrupted evidence. Do not optimize solely for today's API token price.
- Credits are below 50%. Fable owns architecture and adversarial review; use two cheap delegates for genuinely independent, precisely scoped implementation. Freeze contracts before parallel edits. No broad audit, new framework, or repeated full-suite runs.
- The owner confirms that the earlier same-day morning-dose claim was **test input, not an actual event**. Preserve that correction; do not reinterpret it as medication nonadherence or a conflicting genuine report. A separate false dose/source-excerpt record also exists and needs its own provenance investigation.
- The owner reports no alcohol on the current day. Historical consumption records exist; bounded inspection found no evidence establishing a newly stored affirmative current-day alcohol event. Do not claim otherwise. The earlier review mentioned alcohol as an interaction example, not as an observation about the owner.
- The assistant did recommend valerian: the first scheduling response said to take melatonin, magnesium, and valerian before bed, and the next response recommended the nightly three-item stack. Possession of an option was promoted into a recommendation without establishing the full regimen.

Read `docs/DEVELOPMENT_WORKFLOW.md` and the critical/session-safety sections of `CLAUDE.md`. Follow owner-gated commits, pushes, restarts, and store repairs. Use synthetic evidence in tracked tests/docs; raw personal dumps stay out of this public repository.

## 2. Evidence and boundaries of this review

The relevant local trace is root `daemon_debug.log`, September 6 around 20:16–20:26; it rotates. Capture only bounded diagnostic metadata or sanitized fixtures before relying on those offsets. Do not reload entire large stores or duplicate the full conversation dump.

| Finding | Evidence / deployed seam | Status |
|---|---|---|
| Search results disappear after budgeting | `core/prompt/token_manager.py:352–362` treats a non-dict structured result as string-like and writes compressed `str(val)` back. `core/prompt/formatter.py:1014–1050` only renders the object with `has_results`. Its separate helper at 658 supports objects/dicts but is not this assembly branch. | **Reproduced** through the actual budget manager and formatter using a synthetic `WebSearchResult`, one long page, and deterministic tokenizer. Before: object, web section present. After: string, web section and source marker absent. No provider/store was used. |
| Agentic generation has a separate evidence path | `core/agentic/controller.py:697–735, 2008–2110, 2210–2308` gives the decision counts plus four recent turns clipped to 220 characters per message. Reuse at 1228 checks answer usability, not visibility of relevant evidence. `_build_final_prompt` at 2319 ignores initial `web_search_results`; accumulated tool context is separately added at 2436. | **Code-confirmed**. Trace: search returned nine pages, used six search credits, then a memory-triggered loop made no tool call and reused a 447-character decision answer. Do not equate the exported base prompt with this actual request. |
| Current profile contains unsupported dose and test-derived facts | Cheap delegate inspected bounded profile/corpus evidence. One affirmative dose's source excerpt is a current-day denial; the original turn also contains an affirmative prior-day dose. `memory/fact_source.py:264–325, 631–728` now has clause-level support checks. Per-turn path: `memory/memory_storage.py:1073–1102`; shutdown paths: `memory/shutdown_processor.py:547–605, 839–848`. | **Stored contradiction confirmed; exact writer/version not proven.** Existing fixes may postdate the record. Do not announce a fresh extractor regression without replaying current code. |
| Historical consumption can masquerade as current state | `memory/user_profile.py:487–522`: `is_current` means latest unsuperseded relation, with expiry decided by relation-name classification. `memory/relation_classifier.py:96–101` treats one consumption relation as ephemeral while another event-shaped relation can remain durable. | **Code/data confirmed**. This is event-versus-habit and observation-time-versus-event-time ambiguity, not grounds to purge drinking history. |
| Narrative suppresses an appointment using unrelated attachment text | `utils/completed_plan_claims.py:162–217`: completion cue anywhere plus two shared tokens and weak date/digit compatibility. Caution at 258–271 quotes the matched statement's beginning. | **Mechanism confirmed**; observed unrelated homework statement became completion evidence. Raising the token threshold is insufficient. |
| Routing and STM overinterpret follow-ups | Appreciation receives temporal-recall instructions; medication scheduling receives creative-exploration instructions; an explicit goal clarification is called a topic change. STM upgrades one wake-up observation to a recent pattern. | **Visible in dump**. Distinguish routing metadata error from a proven effect on generated behavior. |
| Context includes junk and tenuous associations | Incorrect graph identity/name edges, duplicate relationships, probability notes during appreciation, pet-to-workplace insights, and long git history during an alarm question. | **Visible**. Larger context alone is not a failure; unsupported claims and absent relevance explanations are. |
| Fluent answers overstate support | Unsupported outreach response-rate and spam guarantees; a pending administrative change described as resolved; optional supplements promoted to a routine; an alarm characterized as the entire circadian mechanism. | **Visible**. These are evidence/calibration errors, not a requirement to prohibit suggestions or strip every number. |
| Debug totals are not full usage | `gui/handlers.py:379–397` returns locally counted prompt + system tokens, with character fallback. `web/src/components/debug/DebugPage.tsx:106–121` labels their sum total. No output/auxiliary usage is included in that sum. `core/prompt/builder.py:1212` times overlapping asynchronous tasks. | **Code-confirmed**. Input estimates, provider billing, cumulative task time, and critical-path latency must remain distinct. |

Additional observations worth covering with the same changes: exported git diff counts are sometimes redacted as phone numbers; capability flags do not report connector health; the matching trace has expired Google authorization and semantic-search timeouts. These warrant accurate diagnostics, not an unrelated OAuth or search-engine rebuild.

## 3. Minimal shared contracts

Extend existing types/adapters at their current seams; do not build a second memory pipeline or a universal ontology before fixing the observed failures.

**Evidence identity.** An item should carry its existing ID plus source turn/document ID, source kind and speaker, a source span or stable claim reference, capture time, event time or interval when supported, and origin (`live`, `test`, `quoted`, `generated`, or `unknown`). Claims additionally need polarity/modality and kind (`event`, `state`, `habit`, `plan`, or hypothesis). Reuse metadata already present; missing fields mean unknown, not permission to invent them. Keep semantic confidence separate from provenance validity.

**Authority and corrections.** A current direct correction outranks an earlier generated claim about that same subject, proposition, and time interval. It does not erase unrelated history. Track reversible supersession/exclusion with reason and source IDs. A source excerpt existing verbatim is necessary for traceability but does not prove it entails the claim. Two shared words are not entailment. A generated summary remains generated even when copied into a user-owned notes vault.

**Evidence visibility.** Carry structured evidence through gather → budget → render → decision/final request → debug. Preserve source identity and kind; report what was retrieved, admitted, rendered, and actually sent. An inventory count is not evidence content. Only sources visible to the answering request may support its citations.

**Semantic versus deterministic work.** Deterministically enforce IDs, schema, span bounds, origin permissions, time arithmetic, citation maps, and typed rendering. Use the existing contextual semantic stage for ambiguous meaning, clause ownership, relation kind, and event matching. Cache/batch judgements with source/version-aware keys. Do not add a paid call per fact, a giant English keyword catalog, or a new critic on every trivial turn. Ambiguous matches stay unresolved; useful supported answers may still proceed.

## 4. Implementation sequence

### A. Restore evidence delivery and truthful receipts first

Owner: one cheap delegate; Fable approves the shared representation before edits. Likely files: prompt token manager/formatter, web result types, agentic controller/types, handler debug plumbing. Keep overlapping edits sequential.

1. Add a failing reproduction of the real `WebSearchResult → _manage_token_budget → _assemble_prompt` path. Include one oversized page, multiple pages, dict adapters, empty/error results, Unicode and tokenizer-dense text. Existing list-only mock fixtures do not establish this contract.
2. Budget renderable page content without converting a structured object into its dataclass repr. Preserve page/source IDs, URLs, titles, dates, order and immutability of cached originals. Assign IDs once within the turn and retain them through trimming; dropped pages cannot leave falsely visible citations. Unknown structured types must use a registered adapter or be explicitly excluded with a reason.
3. Reuse this renderer/evidence bundle in agentic decision and final generation. Inspect all supported response paths, including answer reuse, ordinary synthesis, timeout fallback and supported tool protocols. Initial retrieval must not disappear merely because routing chose agentic mode.
4. Permit decision-answer reuse when that request had the answer-relevant evidence and instructions. Otherwise perform final synthesis with the admitted evidence. Do not unconditionally disable reuse or pretend a mechanical visibility check proves semantic sufficiency. An inventory saying “10 memories” must never prevent a necessary lookup.
5. Make request receipts identify route, answer-producing call, model, prompt/system hashes, visible source IDs/kinds, omissions and reason, and reuse status. Preserve an actual decision-prompt hash instead of `decision-answer-reuse` as a pretend hash. Label base prompt exports explicitly; provide an opt-in bounded export of the actual answering request through the existing debug surface.
6. Accumulated memory/file/tool results must retain their source kinds, not all become a web-results section. Propagate citeable IDs consistently through reuse and final paths. Empty, unavailable, timed-out and successful calls require distinct statuses only where the underlying tool actually reports them.

Acceptance: synthetic source markers survive the complete route; reused answers cannot silently bypass missing admitted evidence; no duplicate search is required to recover already-fetched content; citation maps match visible pages. Prove this with captured provider-call arguments and deterministic generated responses, not just assertions about prompt strings in helper tests.

### B. Repair provenance, temporal state and completion identity

Owner: second cheap delegate after Fable freezes metadata changes; separate from A's files. Reuse `fact_source`, profile metadata, relation classifier, completion matcher, existing correction/curation infrastructure.

1. Replay both per-turn and shutdown extraction through their deployed storage adapters with disposable stores. First determine whether current code already prevents the observed false dose. Keep the existing clause-negation fix; address surviving callers or temporal projection rather than rewriting a working helper.
2. Preserve affirmative yesterday and negative today as distinct claims. A previous-day extra amount is neither today's dose nor a standing prescription. Preserve subject ownership and units. A source-span validator must reject unsupported actor/action/time/quantity even when entity tokens overlap.
3. Add explicit non-persistent test/replay mode at the existing submission/storage boundary, propagated to background storage and shutdown synthesis. Tests may exercise normal inference without teaching the live profile/graph. Ordinary conversation stays persistent. Mixed messages can contain both a test block and genuine user commentary; source spans allow retaining the latter. Do not infer “test” merely from unusual wording, repeated questions, or a medication name.
4. Accept retrospective owner corrections tied to the relevant turn/claim and propagate invalidation to dependent facts, graph edges, summaries, notes and narrative. Keep unrelated real events. Where old generated artifacts lack lineage, flag them for scoped review/recomputation with uncertainty; do not pretend exact ancestry is known.
5. Separate event time from storage time and latest relation from applicability now. Infer event/state/habit from the claim and context, not just a relation-name allowlist. “I drank tea last night” differs from “I drink tea nightly.” Relative dates use the source turn's timezone/date; unknown AM/PM or date remains uncertain. Historical queries can still retrieve old events.
6. Replace generic token-overlap completion with compatible actor + event/action/object identity + temporal scope + actual completed status. Explicit reference chains (“that appointment”, “got it done”) can use nearby context. Future, cancelled, attempted, conditional and completed remain distinct. Same-day ordering uses timestamps, not only calendar dates. Digit overlap alone never identifies an event.
7. A completion caution should reference the exact supporting claim and its date, or abstain. “Attached the exercise about completed tasks” cannot complete an unrelated real-world task. A completed occurrence of a recurring plan does not close all future occurrences.
8. Prepare a dry-run contamination report with IDs, reasons, proposed read suppression/supersession and dependent artifacts. Reuse safe JSON, backups, live-process guards and curation queues. Owner reviews the concrete report before applying with Daemon down; no silent data deletion or automatic global reset. Validate repair idempotence and undo on a copy first.

Acceptance: no test-derived personal fact appears as live evidence; true historical consumption remains retrievable without becoming today's state; current negation coexists with past positive events; unrelated prose cannot close a plan; unsupported legacy claims are explicitly uncertain/suppressed through an auditable route.

### C. Preserve intent and calibrate conclusions

Fable owns the semantic contract and reviews outputs; delegate bounded implementation after A/B land.

Current seams: `core/intent_classifier.py:726–831` and its `refine_with_stm` method; `core/context_pipeline.py:528–534` refines intent using free-text STM; `core/orchestrator.py:118–150` decides topic shift using finite continuation predicates and STM labels. These are candidates to reconcile, not proof of which classifier produced every observed label. `core/response_planner.py:178–260` builds another fixed-size evidence digest; include it in visibility/parity tests rather than adding yet another analysis pass.

1. Treat topic, dialogue act, user goal, factual state and evidence needs as separate fields in the existing contextual analysis. A clarification can refine the same goal without being a new topic/event. Short acknowledgements should preserve referents without manufacturing a historical pattern. Handle openers/typos through surrounding meaning, not hardcoded exact phrases.
2. Reconcile system, style and response-plan instructions. Scope public-source requirements to externally checkable claims: user-supplied numbers can cite user evidence, arithmetic can be computed, and speculative suggestions can be identified as such. Financial-source tiers are not a universal medical or personal-memory authority hierarchy. Runtime tool availability is authoritative; remove contradictory static capability claims and duplicated competing style instructions where present.
3. Use one general decision-support principle: separate what is observed, inferred, optional, recommended, pending and confirmed. Having an option is not selecting it; a missed input is not proof of no interactions; institutional advocacy is not approval; one observation is not a trend. Apply this across health, finance, school, scheduling and software decisions.
4. Unknown material details should prompt a targeted lookup or concise clarification while still answering supported parts. Avoid blanket referrals, exhaustive precaution lists, or interrogating the user when the record already answers. Do not hardcode medication schedules, alarm times, combinations, or named substances into policy.
5. Evaluate actual answers for unsupported numerical rates, absolutes (“zero risk”, “whole mechanism”), causal leaps, and failure to revise a prior conclusion after correction. Do not create a regex that deletes all strong language or all numbers. Prefer fixing evidence/instructions first; add a targeted semantic check only if residual measured failures justify its cost.

Medical review boundary: the observed recommendation lacked enough regimen information to establish combination safety. This plan does not supply a replacement dosing schedule. Primary background references used by the review: https://www.nccih.nih.gov/health/valerian and https://www.nhlbi.nih.gov/health/circadian-rhythm-disorders/treatment.

### D. Improve contextual value without blanket restriction

Confirmed selection seam: `memory/user_profile.py:681–709` calls raw keyword overlap semantic relevance, admits top items even when all score zero, then adds recency slots; `:749–806` visits every category. This is a weak selector, but rejecting every zero-keyword match would also reject semantic matches and helpful persistent identity. Preserve a deliberate small identity/core-context allowance and evaluate the rest by meaning and task utility. `core/prompt/builder.py:1375–1479` is the source-gathering seam; prefer a shared policy over separate contradictory per-source vetoes.

1. Keep broad retrieval available and potentially useful, particularly for longitudinal questions and stronger local models. Separate retrieval breadth from what is admitted, emphasized, and treated as evidence. An old or emotionally negative memory is not automatically irrelevant.
2. Admission/ranking should consider the current goal, resolved referents, authorship, time applicability, contradictions and marginal information. Low-confidence cross-domain suggestions need an explicit evidential bridge; a pet relationship alone does not support workplace-management advice. Existing source/relevance stages should carry those reasons, avoiding an extra classifier per section.
3. Normalize graph duplicates and assess type/claim plausibility at ingestion and read time. Do not blacklist particular names or literal strings that happened to be junk here. Keep legitimate unusual names and nonstandard self-descriptions. Durable personalized identities/preferences remain available even for short turns.
4. Remove only demonstrably redundant instructions and unhelpful duplication after measuring behavior. Keep source text retrievable through expansion. Middle-out compression must preserve correction/negation/qualification and authorship boundaries; for evidence that cannot survive a small allowance, prefer an explicit reference/omission over misleading excerpts.
5. Compare current, relevance-ranked-rich and compact/adaptive context on the same frozen cases. Judge factuality, personalization, recall coverage, correction uptake, user-goal continuity and uncertainty alongside tokens/latency. Choose policies by measured capability/task/context limits; do not embed provider names, deployment year or today's token prices in semantic logic.

### E. Measure real work, then optimize

1. Keep existing counters compatible but label local prompt/system counts **input estimates**. Collect per-call provider-reported input/output/cached/reasoning usage where actually available; unknown values remain unknown. Local backends report token counts and timing without fabricated dollar billing. Track search credits separately.
2. Correlate chat, intent, STM, planner, verifier, retrieval, embeddings and background synthesis using turn/call/parent IDs. Attribute background work once; do not force it into synchronous response wall time. Add runtime/source version so a loaded process can be distinguished from edited files.
3. Expose monotonic start/end/duration, overlap and critical-path preparation/first-token/completion times. Do not sum concurrent retrieval durations as latency or count nested phases twice. Add a modest daily aggregate by call site before redesigning scheduling.
4. Report enabled versus attempted versus healthy versus evidence-visible separately. Surface persistent authorization failure once with a useful reconnect action through existing UI; agents do not change credentials. Debounce repeated failed refresh work through existing credential/error handling only if measured and properly invalidated on reconnect.
5. After A–D, optimize confirmed duplicate embeddings, repeated retrieval, static prompt assembly and cache misses. Keys must include effective query, corpus/source version, policy, model capability and relevant temporal scope. Cache reuse must not revive retracted facts or bypass changed availability/budgets. Verify timeout behavior for worker tasks before claiming cancelled work stopped.
6. Make export redaction field-aware: preserve numerical diff counts without weakening real phone/contact redaction. Label display truncation separately from actual model truncation. Prompt hashes describe actual inputs; receipts remain bounded and use existing privacy settings.

## 5. Validation and adversarial review

Use existing suites before adding overlapping ones: `test_independent_prompt_audit.py`, `test_web_search_citations.py`, `test_agentic_decision_answer_reuse.py`, `test_agentic_digest_order.py`, `test_agentic_timeout_fallback.py`, `test_fact_source_negation_scope.py`, `test_sep06_evidence_ownership.py`, `test_completed_plan_claims.py`, `test_user_profile_schema.py`, `test_thread_context_and_resolution.py`, and `test_intent_style_instructions.py` under `tests/unit/`. These are starting points, not proof that every deployed path is covered.

Fable should freeze a small semantic matrix before implementation, including held-out cases unseen by patch authors:

| Class | Positive case and adversarial control |
|---|---|
| Time/polarity | Yesterday's use plus today's omission; reverse clause order, contractions, implicit subjects; a durable habit remains durable. |
| Source ownership | First-person live report versus quote, generated daily note, test prompt and third-party statement; genuine commentary after a test block still persists. |
| Corrections | Correct quantity/time/identity for one event; preserve unrelated events and valid historical retrieval; invalidate derived copies without multiplying occurrences. |
| Completion | Actual completed appointment/repair/payment versus unrelated document sharing words; “not yet”, planned, cancelled, rescheduled, recurring and same-day replan. |
| Dialogue | Acknowledgement, clarification, goal refinement, true topic switch, pronoun follow-up and mixed personal/public question. |
| Grounding | Available versus recommended option; request versus approval; unsupported rate versus sourced rate; user-provided number versus invented number. |
| Evidence transport | Long structured web object, dict/empty/error, compressed page, multiple tools, reuse/final/fallback paths, cache hit, source dropped under budget. |
| Rich context | Relevant remote memory embedded among distractors; cross-domain insight with/without an actual bridge; compact context must not win by omitting the user's relevant history. |

Use substitutions across ordinary domains and change syntax/order, not just names. Deterministic tests should assert semantic metadata and captured inputs/outputs of deployed functions; do not mirror the implementation's regex or require exact answer wording. Do not make all low-confidence cases disappear to obtain a pass.

Example focused commands, split by change ownership:

```bash
python -m pytest tests/unit/test_independent_prompt_audit.py tests/unit/test_web_search_citations.py tests/unit/test_agentic_decision_answer_reuse.py tests/unit/test_agentic_digest_order.py tests/unit/test_agentic_timeout_fallback.py -q --color=no
python -m pytest tests/unit/test_fact_source_negation_scope.py tests/unit/test_sep06_evidence_ownership.py tests/unit/test_completed_plan_claims.py tests/unit/test_user_profile_schema.py -q --color=no
python -m pytest tests/unit/test_thread_context_and_resolution.py tests/unit/test_intent_style_instructions.py -q --color=no
git diff --check
```

Add new tests to the relevant command. Block live providers, embeddings downloads and production stores in the fast lane. Run Ruff on changed Python files using repository settings. Run the four repository contract guards named in `docs/DEVELOPMENT_WORKFLOW.md` before owner push; batch broader checks for the machine's memory limit. No full suite is claimed here.

Generation acceptance is separate: use sanitized frozen evidence and bounded provider/local-model replay; inspect complete answers for the matrix above. Fable is the adversarial reviewer, not merely another same-prompt judge. Predeclare a call/token budget; no unattended retries or all-model benchmark sweep. Live owner probes happen after owner restart. Record runtime version, actual answering request, response and receipts. Passing deterministic transport tests does not prove medical reasoning or future-model quality.

## 6. Execution and stopping rules

Deliver A and B as separate reviewable changes before broader optimization. Fable settles shared metadata/rendering decisions first; cheap agents edit disjoint files with explicit acceptance cases. Each reports changed files, failed-before/passed-after evidence, limits and unresolved questions. No agent opportunistically expands its scope.

| If… | Then… |
|---|---|
| Current extraction replay already rejects the bad fact | Repair the legacy contamination through owner-reviewed workflow; do not add a redundant parser patch. Verify shutdown and alternative writers before closing. |
| Event/source identity is ambiguous | Keep unresolved/uncertain; seek missing context only when material. No guessed completion or auto-deletion. |
| A structured source cannot fit | Preserve minimal valid source metadata and an explicit omission/expansion path; never silently convert its type or drop a negation. |
| Decision reuse lacks required visible evidence | Synthesize with that evidence; preserve reuse for genuinely sufficient decision requests. |
| Rich context preserves personalization but raises cost | Keep the quality result and expose/tune the resource policy; do not silently reduce evidence to chase token counts. |
| A delegate encounters a shared-contract change, unexpected store state or unrelated failure | Stop that workstream and return concrete evidence to Fable. Continue independent work only. |
| Credits approach exhaustion | Stop new investigation; finish the current bounded verification and write an exact resume handoff. Do not claim the remaining phases implemented. |

Final handoff should enumerate code on disk, tests, remaining risks, data repair proposed versus applied, and deployed versus merely edited version. Owner commits/pushes/restarts and approves any concrete store repair. Completion means the observed issue classes pass general, held-out counterexamples while useful personalization remains; it does not mean promising perfect American-English understanding or guaranteed 2028 model performance.
