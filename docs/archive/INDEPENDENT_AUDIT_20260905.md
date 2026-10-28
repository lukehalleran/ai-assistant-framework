# Independent engineering audit — September 4–5, 2026

## Outcome

Important context-budget, response-admission, cancellation, and index-initialization
bugs are fixed on disk, with targeted reproductions. Existing uncommitted work was
preserved. **The reported repeated final answer is not explained or claimed fixed.**
This is a completed engineering pass, not a claim that the entire system is verified.

The best next architecture is a bounded, source-aware evidence pipeline with explicit
request ownership and measurable routing—not more layers of prompts asking models
to compensate for oversized, redundant context.

## Baseline and architecture

Initial branch `master`, HEAD `e6e515de82aff941484c1d2b388746f8a33749af`, no staged
changes. The nine dirty tracked files and one untracked test were inspected with
recent commits before editing. Exact initial copies and patches remain in
`/tmp/daemon-audit-baseline-cAK40N1q/`. There was no checkpoint commit or reset.

The interrupted batch addressed conversation-field metering, attachment names and
deduplication, email-read intent detection, missing-file recognition, and deadline
notes. Its focused tests passed at baseline; that did not establish overall correctness.
The token fix was incomplete because other builder paths bypassed it.

Main path:

`UI/API → shared submit pipeline → ContextPipeline → parallel retrieval → hygiene/top-ups/budget → formatter/planner → provider or agentic loop → response assembly → UI snapshots + background persistence`

The source retrieval fan-out is concurrent, but several expensive phases surround
it serially. The API uses mutable session state; builder/scorer/gatherer also carry
request-specific mutable fields. The API reservation fixed here is not a universal
cross-interface lock for every possible caller of that shared state.

## Findings and fixes

All bugs below existed before this audit unless explicitly noted otherwise.

### F1 — High, confirmed and fixed: context budget could be bypassed

The lightweight follow-up path returned uncapped history. The full path budgeted
before recency floors, which then restored original records. Four regressions run
against the original full/light builder methods failed: a 21,015-token fixture
survived a 1,200-token budget; a light-path query retained 400,000 characters;
cancelled retrieval remained alive; retrieved self notes disappeared.

The old four-characters-per-token clipping estimate was not a hard token cap.
Dictionary compression could meter hypothetical compressed text without changing
the retained dictionary. Structured `codebase_changes` could also be converted
into a string incompatible with its formatter, or escape second-pass trimming.

Changes: tokenizer-measured clipping including its omission marker; budget after
all top-ups; budget light prompts; meter retained values; preserve or drop structured
sections without destroying their schema. Tests preserve original stored records.
Files: `utils/text_budget.py`, `core/prompt/{token_manager,builder}.py`.

**Boundary:** these are context-section caps using the application's tokenizer
estimate, not a provider-exact whole-request cap. System text, the current query,
attachments, tool schemas and rendering overhead still need a unified final budget.
Tiny budgets can now drop recency-floor candidates; the floor is not a guarantee.

### F2 — Medium, confirmed and fixed: priority and rendering lost useful evidence

Admission iterated the literal priority list, not its numeric priorities. Lower-value
memories could consume space before explicitly higher-priority live web/calendar
context. `daemon_self_notes` was fetched then discarded, and `codebase_changes`
was absent from final assembly. The formatter ignored a preformatted `text` field
that the meter recognized. Regression tests cover these mismatches.

Admission now follows descending priority, and gathered/rendered fields agree.
The profile keeps the gatherer's existing 3,000-token allocation rather than being
treated as an 800-token snippet. This last change prevents stricter clipping from
unintentionally worsening personalization; it is not a measured quality improvement.
Files: `core/prompt/{builder,token_manager,formatter}.py`.

### F3 — High for provenance, medium for latency; confirmed and fixed

Optional LLM compression could write compressed conversation text into `query`
while leaving the response intact, choose an empty field rather than the extracted
field, and accept an expanded result. Conversation pairs now use independent
deterministic field caps; other items write back to the exact extracted field, and
non-shrinking LLM results are rejected.

In the available September 4 logs: 14 queued compression batches, 39 item timeouts,
zero logged successful compressions, and two timeout-cache hits. The matching turn
spent approximately three seconds waiting for a compression timeout. The feature
was already using `gpt-4o-mini`: a cheap model did not make that work useful.

Default LLM compression is now **off**, consistently in YAML, configuration fallback,
schema and builder fallback. It remains opt-in and tested. Deterministic excerpts can
lose middle facts; long documents ultimately need query-targeted passages with source
links, not indiscriminate clipping or an extra summarizer on every critical path.
Files: `core/prompt/builder.py`, `config/{config.yaml,app_config.py,schema.py}`.

### F4 — High, confirmed and fixed: overlapping turns and incomplete cancellation

The route checked a lock before returning its response, but acquired it only when
the response generator started. Two requests could pass admission. The frontend
also relied on asynchronously updated state to prevent repeated sends. Session
clear was allowed during streaming, and the client cleared its UI even on a failed
DELETE. These paths could corrupt ordering/state, although none is established as
the cause of the user's duplicated paragraphs.

The route now reserves admission before returning, closes the pipeline before
releasing the reservation, and handles early cancellation/send failure. Clear during
a stream returns 409. The browser has a synchronous in-flight guard and preserves
history on failed clear. The builder cancels and drains async retrieval children
before resetting request-scoped shared state.

Files: `api/routes/chat.py`, `web/src/api/useChatStream.ts`, `core/prompt/builder.py`.
Tests cover admission before body consumption, early cancellation, send failure
with the lock held through cleanup, and rejected clear. **Running worker threads
are not forcibly stopped by cancelling their awaiting coroutine.**

### F5 — Medium, confirmed and fixed: one large tool result escaped the limit

The agentic accumulator removed older delimiter-separated blocks until one remained.
One oversized result with no delimiter therefore survived unchanged. The remaining
block now passes a measured cap. Both initially empty and already-populated contexts
have regression coverage. File: `core/agentic/controller.py`.

### F6 — High resource risk, confirmed and fixed: concurrent cold index loading

The semantic-index singleton lock protected object creation, not `load()`. Two
worker threads could both load the multi-GB FAISS index and concurrently mutate
its metadata offsets. A controlled overlapping-load test observed two loader calls
before the fix and one afterward. Initialization now publishes one completed index
under an instance lock. File: `knowledge/semantic_search.py`.

Separately, the first clean-path smoke observed a 1.5-second wiki-search timeout,
then index initialization completing in 54.27 seconds after the requests ended
(40,982,675 metadata rows indexed; the full text corpus was not loaded into RAM).
Existing dedicated worker/semaphore limits bound the concurrency, but do not stop
disk work or guarantee prompt process exit. Single initialization fixes the race,
**not** that resource-lifetime problem. The disposable smoke now explicitly disables
wiki retrieval through its numeric source limits.

## What the supplied runtime evidence actually shows

The matched trace is now archived in `daemon_debug_20260904_195742.log`:
retrieval at line 554; zero of 14 memories retained at line 623; full prompt at
line 669; decision-answer reuse at line 1283. Log filenames may rotate again.

| Phase | Observed duration | Interpretation |
| --- | ---: | --- |
| Context pipeline | ~5.24 s | Tone/heavy-topic/STM preparation before the main retrieval fan-out |
| Retrieval wall | 6.94 s | Memories 6.92, reflections 6.90, summaries 6.89 overlap; do not add them |
| LLM compression | ~3 s | Timeout after retrieval; not hidden by retrieval concurrency |
| Builder total | ~10.39 s | Retrieval, compression, and remaining context work |
| Full prompt build | 12.13 s | Includes ~1.7 s response planning after builder work |
| Prepare prompt | ~17.45 s | Pipeline plus full build and surrounding overhead |
| Agentic loop | ~20.10 s | Includes the decision-round generation |

The three ~6.9-second retrieval timings include awaits and shared compute contention,
not isolated CPU time. The main gating path re-embeds the query and candidates across
sources (`processing/gate_system.py:_encode_texts`, batch filter paths). Some paths
have two reranking stages (`memory/memory_retriever.py:_maybe_cross_encoder_rerank`
and gate filtering). Duplicate work is plausible from these call chains; its exact
wall-time contribution and quality value are not isolated. No speculative removal
of reranking or semantic memories was made.

The turn paid to retrieve 14 memories and retained zero. This is strong evidence of
poor retrieval-to-prompt yield, not proof that every discarded memory was useful.
Measure admitted evidence and answer utility, not just retrieved counts.

The ~469,591-character “snip” was the **omitted count**, not retained snippet size:
471,991 stored query characters minus the old 2,400-character excerpt. The large
stored source was real. The light-path budget bypass made it dangerous on follow-ups.
The reported 11,667 prompt + 7,747 system = 19,414 estimated tokens also illustrates
why a nominal 10,000 context budget is not a 10,000 total request budget.

### Repeated final answer: unresolved

The matching agentic path reused an 855-character decision-round answer and skipped
final synthesis. The persisted answer did not contain repeated long paragraphs.
A September 4 scan found no exact repeated paragraphs of at least 80 characters in
the sampled saved answers. This does not exclude near-duplicates, another turn,
provider behavior, or a display/debug-export duplication.

The traced ordinary, agentic, and insight provider consumers expect incremental
`delta.content`; API/UI updates use cumulative snapshots and replacement. No general
mid-stream retry-and-append loop was found. A provider emitting cumulative text in a
delta field would violate that assumption, but the inspected samples did not prove
that happened. Do not add a heuristic paragraph deduplicator that could corrupt
legitimate repeated text. Capture one real incident end-to-end instead.

Current observability is insufficient to settle the issue. Add request/attempt IDs,
chunk sequence and lengths, and response hashes at provider assembly, persistence,
API emission and browser completion. Keep raw content opt-in and bounded. This pass
adds monotonic builder phase timings, not the complete response-boundary tracing.

## Frontier workflow and economics

The described external frontier protocol was not found as a complete implemented
policy in this repository. `agent_branch` has concrete manifest budgets, isolated
execution, diffs/tests and proxy accounting; its Layer 2 intent judge explicitly
returns a deferred stub verdict. Treat those tests/budgets as real mechanisms, but
do not mistake the stub for independent semantic supervision.

`ModelManager.generate_once` permits explicit model selection; ordinary streaming
uses the active model, and supported reasoning requests commonly specify `medium`.
A prompt declaring identity/effort is useful provenance only when matched to actual
request/response metadata. It neither proves capability nor creates economical routing.

Keep bounded delegation, but change the contract:

- Route mechanical tasks using explicit requirements and measured acceptance rates,
  latency and total cost. Include escalation/rework in cost, not just the cheap call.
- Give delegates a narrow question, relevant evidence, acceptance tests, permitted
  actions, and one aggregate budget/depth limit. Escalate on missing evidence or
  failed criteria, not vague model self-confidence.
- Verify artifacts and run tests; sample independent reasoning for risky claims.
  Re-solving every delegated task duplicates the expensive work delegation should save.
- Return a bounded structured answer: conclusion, evidence IDs, uncertainty,
  measurements, changes/tests and requested escalation. Store raw dumps separately
  for lazy retrieval. Do not send full conversation/debug exports by default.

`api/routes/debug.py` redacts but returns all retained debug records; those records
can contain full prompts/history. Redaction does not bound size or eliminate repeated
evidence. Export-and-paste feedback can recursively import prior prompts, making it
a major context-pollution risk. Verbose parser logs and full-prompt logging also make
large diagnostic artifacts. Their exact latency contribution was not benchmarked.

## Validation and measurements

Final focused regression command (280 passed, one pre-existing midnight-dependent
skip, 19.85 seconds; FAISS/websockets deprecation warnings):

```bash
python -m pytest tests/unit/test_independent_prompt_audit.py tests/unit/test_semantic_load_concurrency.py tests/unit/test_semantic_search_metric.py tests/unit/test_api_chat.py tests/unit/test_handle_submit.py tests/unit/test_stream_artifacts.py tests/unit/test_request_path_parity.py tests/unit/test_llm_compression.py tests/unit/test_middle_out_no_grow.py tests/unit/test_light_prompt_path.py tests/unit/test_budget_meters_rendered_sections.py tests/unit/test_token_budget_cap.py tests/unit/test_prompt_builder_methods.py tests/unit/test_sep04_evening_fixes.py -q --color=no
```

The preceding run was 279 passed / one failed / one skipped: the old no-growth test
required returning a 900-token original for an 800-token cap. Its assertion was
updated to require both no growth and compliance, rather than exempting dense text.

Other validation (overlapping batches, **do not sum**):

- Initial baseline: `test_sep04_evening_fixes.py` +
  `test_budget_meters_rendered_sections.py`: 30 passed.
- `python -m pytest tests/unit/test_independent_prompt_audit.py tests/unit/test_sep04_evening_fixes.py tests/unit/test_sep04_attachment_turn.py tests/unit/test_file_processor.py tests/unit/test_context_pipeline.py -q`:
  135 passed, 189.83 s, before the final profile/wiki adjustments.
- Earlier focused token/builder/compressor, API/handler, and light-context batches:
  99 passed / one skip, 147 passed, and 71 passed respectively; superseded where
  overlapping by the final command above.
- New negative reproductions failed before their fixes: initial clipping/priority/
  compression tests; four original-builder tests; cold-index race and profile allocation.
- `npm run build` in `web`: passed TypeScript and Vite; existing circular
  KaTeX/vendor chunk warning. No browser E2E or real-provider replay claimed.
- `python -m ruff check` on audit-edited Python sources/tests/scripts with
  `--select E9,F63,F7,F82`: passed. `git diff --check`: passed.
- An earlier agent's broad 327-test attempt stalled and was interrupted. This is
  not a full-suite result; some nominal unit tests instantiate real embedding models.

Read-only size replay: `python scripts/audit_context_budget.py` selects three real
stored turns through September 4 17:26:27, prints only aggregate counts, and makes
no provider calls. Before: 476,719 chars / 141,395 estimated tokens. After: 6,815 chars /
1,728 estimated tokens. Median cap time across five runs: 195.75 ms. Largest query:
471,991 chars before, 600 estimated tokens after. This compares old unbudgeted light
history with capped history, not complete provider requests or answer quality.

Clean-path validation uses `python scripts/audit_runtime_smoke.py`: real fresh-process
orchestrator, cached embedders, Chroma, HTTP/SSE, prompt builder, upload parser and
disk persistence; temporary data, deterministic generation and optional integrations
disabled. The September 4 successful result used PID 1054124, startup 3.145 s,
two requests, four ordered history messages, three persisted/reloaded entries,
8,891-character first prompt after a ~504k-character synthetic query, and exactly
one attachment marker despite a repeated file ID. Its later wiki load motivated F6.
The final September 5 rerun also passed and the command completed: PID 1068000,
temporary data `/tmp/daemon-runtime-audit-jksxo9ef`, two requests, four messages,
three reloaded entries, 8,893-character first prompt, one attachment marker.
Log: `/tmp/daemon-runtime-audit-20260905.log`. Startup was 58.973 s, including
50.58 s in coordinator/builder initialization. Full-path retrieval was 4.148 s,
compression 0.000 s, final budgeting 0.060 s. Startup variability and differing
stores/configuration make these **not** a controlled latency comparison with the
user's 37.55-second production request. No live-provider improvement is claimed.

## Git state and deployment

Branch `master`; 21 modified tracked files and eight untracked files; no commits
created, nothing staged. Existing user work remains
uncommitted alongside these fixes. Do not package everything into one opaque commit:
the original attachment/deadline batch has separate behavior and remaining risks.

Original modified files untouched by this audit: README.md, core/agentic/gate.py,
core/context_pipeline.py, docs/HANDOFF_20260904_hw_attachment_turn.md,
docs/METRICS_SNAPSHOT.md, gui/handlers.py, utils/attachment_audit.py,
utils/file_processor.py. Original core/prompt/token_manager.py changes were preserved
and extended. Original untracked test_sep04_evening_fixes.py remains untracked.

Additional modified files: api/routes/chat.py; config/app_config.py, config/config.yaml,
config/schema.py; core/agentic/controller.py; core/prompt/builder.py,
core/prompt/formatter.py; knowledge/semantic_search.py; tests/unit/test_api_chat.py,
tests/unit/test_llm_compression.py, tests/unit/test_middle_out_no_grow.py;
web/src/api/useChatStream.ts.

New audit files: this report; INDEPENDENT_AUDIT_HANDOFF_20260905.md;
scripts/audit_context_budget.py; scripts/audit_runtime_smoke.py;
tests/unit/test_independent_prompt_audit.py; tests/unit/test_semantic_load_concurrency.py;
utils/text_budget.py. Temporary smoke data/logs and the baseline are retained in /tmp.

At September 5 inspection, PID 1062969 (`python main.py`, started 11:15:13) had this
repository as cwd. It was not stopped. Its loaded modules predate the final wiki/profile
changes; cwd alone does not establish an exact loaded source version. Apply those
changes to the live daemon through a planned restart, not assumptions about hot reload.

## Remaining risks

- Real Kimi/provider streaming, the reported duplicate display, full production
  integrations, and a production-store restart were not validated.
- No personalization/relevance benchmark proves the changed selection is better;
  per-field excerpts can omit important middle facts. Whole-request budgeting remains.
- Shared state outside API admission, thread-backed work after cancellation,
  background persistence shutdown behavior, and cross-process store safety need
  more targeted lifecycle tests; the smoke proves normal persistence, not crash safety.
- The original deadline note uses today's date for timezone conversion and omits
  day rollover. Static inspection confirms that mechanism; dated/DST/cross-midnight
  correctness is not validated or repaired. It also pairs first matching deadlines,
  which may refer to different assignments. Do not treat it as authoritative scheduling.
- Full repository regression and external frontier economics were not established.
- Startup variability (3.145 versus 58.973 seconds in isolated runs) is unisolated;
  the slower run's largest measured phase was coordinator/builder initialization.

## Highest-leverage next steps

1. **Capture and identify one real duplicate incident** (high value, small/medium
   effort): versioned startup fingerprint plus response-boundary IDs/hashes. Compare
   provider, stored text and main chat DOM; distinguish debug exports from the answer.
2. **Replace debug-dump feedback with bounded evidence envelopes, then enforce a
   fully rendered request budget** (high value, medium effort). Preserve documents
   as addressable artifacts with query-selected passages; make raw diagnostics opt-in.
3. **Fuse retrieval and measure evidence yield** (high value, medium effort): share
   one query embedding per model, deduplicate source candidates before repeated
   gating/reranking, record candidate→admitted→cited counts and compute queue time.
   Validate recall on a small personalized question set before deleting stages.
4. **Make request/resource ownership explicit** (high value, medium effort): one
   cancellation/deadline scope across interfaces, bounded worker lifecycle and storage
   drain, and a fast isolated test lane plus separate model/integration tests.

## Independent ideas and credit-efficient continuation

- Give evidence a stable source ID/version and distinguish current facts from historic
  beliefs. Correlated memory/reflection/summary copies should not gain extra weight
  merely because the same assertion appears three times.
- Track a source's marginal value per millisecond/token. Repeatedly fetching then
  discarding every candidate is a signal to change routing, not increase retrieval limits.
- Make frontier supervision artifact-based: small testable claims, trace references
  and budget ledgers. Remove semantic-monitoring layers that only repeat the work or
  return a stub; preserve isolation, accounting and independent tests.
- There is no reliable numeric credit/session estimate available to this agent.
  A useful unit is one bounded milestone: evidence/reproduction, patch, focused
  validation, then durable handoff. Avoid broad parallel agent audits and repeated
  full log/context dumps. The handoff records completed work and unresolved questions
  so the next frontier spends its context on the next decision, not rediscovery.
