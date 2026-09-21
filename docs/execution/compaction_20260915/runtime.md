# Runtime compaction audit — 2026-09-15

Scope: `core/`, `memory/`, `knowledge/`, `processing/`, `models/`, and `utils/`.
Plan only: no application files, stores, generated data, or tests were changed.

## Coverage

Mechanical inventory covered every tracked path under the six directories: 244
files total (242 `.py`, one `.txt`, one extensionless path), 122,805 Python
lines plus 401 text lines. Directory counts/lines: core 89/53,274; memory
59/28,321; knowledge 35/16,889; processing 2/1,656; models 3/1,706; utils
55/21,360. The scan also counted 1,522 `try:` blocks, 974 `except Exception`
occurrences, 457 async definitions, and 585 awaits. These counts identify
review hotspots; they are not savings estimates.

Deep semantic review (representative/high fan-in or largest files):
`core/agentic/controller.py`, `core/orchestrator.py`, `core/prompt/builder.py`,
`core/prompt/base.py`, `core/prompt/formatter.py`, `core/prompt/gatherer_knowledge.py`,
`memory/memory_retriever.py`, `memory/memory_storage.py`,
`memory/shutdown_processor.py`, `knowledge/web_search_manager.py`,
`processing/gate_system.py`, `models/model_manager.py`, `utils/web_search_trigger.py`,
`utils/query_checker.py`, `utils/tone_detector.py`, and the three note generators
(`utils/daily_notes_generator.py`, `weekly_notes_generator.py`,
`monthly_notes_generator.py`). All remaining tracked files received inventory,
line-count, literal, and repeated-block scans only; they remain deferred for
semantic behavior review. In particular, this is not a claim of full review of
all 244 files.

## Concrete opportunities

Priority 1 — repeated session selection in `memory/shutdown_processor.py`.
The same `session_conversations`/corpus fallback, timestamp filter, and
malformed-timestamp fallback appears at lines 648–665, 1053–1071, 1195–1213,
and 1525–1543. A private helper can centralize this policy. Estimated local
reduction: about 30–40 lines after retaining the distinct empty-result policy
(one caller uses `corpus=[]`, three return). Risk: changing return-on-error,
session ordering, or timestamp fallback changes persistence and shutdown
behavior. Verify with focused shutdown, fact, skill, proposal, and thread
extraction tests plus AST/source hash review of accepted-debt files.

Priority 1 — duplicated fallback model roster in
`utils/daily_notes_generator.py:838–850`, `utils/weekly_notes_generator.py:549–561`,
and `utils/monthly_notes_generator.py:539–551`. Extract one immutable roster
and retain each caller's primary-model precedence and logging. Estimated local
reduction: about 24–30 lines. Risk: import cycles, list mutation, or changing
fallback order (which changes model output and cost). Verify exact roster/order,
primary de-duplication, and each generator's fallback/error tests.

Priority 2 — duplicate order-preserving dedupe helper in
`core/prompt/base.py:58–67` and `core/prompt/formatter.py:105–114`. One shared
implementation or compatibility alias should save about 8–10 lines. Risk:
different accepted iterables/key defaults or external imports. Verify callers,
ordering, duplicate keys, and public import compatibility.

Priority 2 — duplicate minimal context construction in
`core/prompt/builder.py:2137–2161` and `2376–2409`. A factory can own common
empty keys while callers add `stm_summary`, split render keys, and light-path
metadata. Estimated local reduction: about 18–25 lines. Risk: missing a key
silently changes prompt sections or debug consumers. Verify dictionary key sets,
`None` versus empty-list values, and both error paths with snapshot-style prompt
assertions.

Priority 2 — duplicate gate module contracts in
`processing/gate_system.py:2–26` and `64–87`. Consolidating the public contract
and embedding-space warning saves about 20–25 lines of documentation. Risk is
documentation drift rather than runtime behavior; preserve the embedding-space
threshold warning and public entry points. Verify import/compile and manually
review the resulting contract.

Priority 3 — multiline import cleanup, for example
`core/prompt/builder.py:82–85` and `90–100`. Mechanical flattening may save only
1–3 lines per block and has poor payoff. Preserve import grouping, lazy imports,
and TYPE_CHECKING boundaries; require compile/lint and import smoke checks.

## AST duplicate pass

An AST scan covered all 242 tracked Python files. It compared function bodies
after removing location attributes while retaining identifiers, constants, and
control structure. It found 19 duplicate groups. Most are trivial helpers,
same-file API methods, or candidates already listed above. The five additional
nontrivial groups inspected were:

* `knowledge/web_search_manager.py:278–302` and `373–396`, identical
  `get_formatted_content` methods on two result types. A private formatter taking
  pages and `max_chars` could remove about 15–18 local lines while preserving
  each public method and output byte-for-byte. Verify truncation boundaries,
  empty pages, and both result classes' callers.
* `memory/memory_retriever.py:921–927` and `1057–1063`, identical nested
  `get_item_id` functions. A module helper can remove about 7 lines. Risk is
  changing the deduplication key used by retrieval ordering; verify timestamp
  conversion, non-dict values, and both hybrid retrieval paths.
* `utils/monthly_notes_generator.py:258–273` and
  `utils/weekly_notes_generator.py:271–286`, identical frontmatter parsers.
  Shared helper extraction saves about 9 lines. Preserve YAML failure behavior,
  delimiter splitting, and body stripping; verify both note readers.
* `knowledge/wikidata_enrichment.py:49–53` and
  `knowledge/wikidata_resolver.py:52–56`, identical `_slugify` helpers. A
  shared utility saves about 4 lines, but import ownership and public/private
  compatibility should be checked first. Verify punctuation, whitespace, and
  Unicode word behavior.
* `utils/completed_plan_claims.py:255–265` and
  `utils/streak_claims.py:128–138`, identical `_coerce_date` helpers. Shared
  date parsing saves about 9 lines. Preserve `datetime` precedence, `Z`
  handling, and invalid-input `None`; verify both claim extractors.

The scan also found `_parse_bool` copies in four prompt modules, `_now`/`_now_iso`
copies in three memory modules, and several constructor/API wrappers. Those are
deferred because config semantics, import layering, or public compatibility may
outweigh their small local savings. The estimates above are local reductions,
not a repository-wide total.

A safe formatting example for a later mechanical pass is
`core/actions/github_write.py:37–40`: the call can become
`result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5)`.
The call AST, string literals, keyword order, and surrounding `try` block stay
identical; only the short call layout changes. This is the intended boundary for
signature/call folding, unlike restructuring the async or exception-control
blocks themselves.

## Guardrails and deferred work

Do not flatten multiline async calls/definitions, `await`/gather blocks, or
`try`/`except` paths based on line count alone. Short signatures or calls may be
folded when their AST/tokens/literals are identical and comments, argument
evaluation order, cancellation, and control-block layout remain unchanged. The
assigned code has cancellation,
lazy imports, persistence writes, timeout boundaries, and fail-open behavior
whose visible shape is part of the contract. Do not normalize or deduplicate
literal prompt/template bodies merely to reduce lines: `core/system_prompt.txt`
(401 lines), long module contracts, and large prompt literals may affect exact
LLM output, token budgets, whitespace-sensitive parsers, or human auditability.

The scan found repeated JSON/schema fragments (for example
`core/actions/registry.py:574–590` and `core/agentic/types.py:1270–1286`) and
repeated generator/model setup beyond the examples above. These need semantic
ownership decisions before extraction. No savings number is assigned to them.

Resume checkpoint: start with the four Priority 1/2 candidates above, inspect
current tests and accepted-debt/source-hash rules, then make one narrow change
at a time. Re-run the mechanical inventory and targeted behavior tests after
each change; manually rereview any accepted-debt file rather than blanket
rebaselining its disposition. Root's metrics/token report remains authoritative
for whole-repository totals.

## Runtime-coupled documentation safety check

`docs/PROJECT_SKELETON.md:2610` is a real product contract. The deployed reader
is `knowledge/proposal_generator.py:115–207`: `gather_context()` reads the first
existing path from `docs/PROJECT_SKELETON.md`/root fallback, `docs/GOALS.md`/root
fallback, root `CLAUDE.md` then `docs/CLAUDE.md`, and
`docs/QUICK_REFERENCE.md` then root fallback. Skeleton text is filtered by
`_filter_skeleton_sections()` (`210–241`), goals are truncated to 4,000
characters (`157–159`), while CLAUDE and QUICK_REFERENCE are read in full.
`_build_prompt()` emits the exact section order and headings at `243–265`:
CLAUDE, QUICK_REFERENCE, filtered skeleton, goals, recent commits, then extra
context. `generate_proposals()` and `generate_proposals_with_context()` call
this reader (`487` and `561`); `agent_branch/goal_runner.py:119–129` reuses it,
so agent-branch objective generation inherits these document inputs.

`core/prompt/proposal_filter.py:99–152` independently reads `docs/GOALS.md`
(repo-root fallback), hashes the full text for its cache, then extracts `###`
lines only between `Active Goals` and the next `---` to build the utility query.
Changing headings, ordering, backticks, or prose can therefore change proposal
ranking even when the information appears equivalent. The core system prompt
and `section_instructions.py` mention these names for the model but do not load
the files themselves. The `eval/` sources found no direct reads of these four
files; their `reference_docs` flags describe expected retrieved sections.

There is a second deployed coupling outside the assigned directories:
`config/config.yaml:378–381` enables `docs` as the reference-doc seed path;
`gui/launch.py:271–302` calls `ReferenceDocsManager.sync_directory()`, whose
default `*.md` glob (`knowledge/reference_docs_manager.py:642–676`) hashes and
indexes every markdown file in `docs/` into the `reference_docs` collection.
Those chunks feed normal retrieval, so every compacted docs file can alter
retrieval candidates, scores, prompt text, and stored content hashes. Root
`CLAUDE.md` is not included by that default docs-directory seed, but remains a
full proposal-generator input and a protected shell-guard filename.

Safety contract for any documentation compaction: treat the four named files
and all auto-seeded `docs/*.md` as runtime data. Do not summarize, reorder,
rename headings, normalize whitespace, or delete “redundant” prose under a
formatting-only plan. An exact reconstruction is acceptable only when it
preserves the reader's resulting strings (including skeleton filter boundaries,
GOALS first-4,000-character slice, proposal section order/headings, and the
bytes hashed/indexed by reference-doc seeding). Otherwise classify the change
as a behavior/product change, preserve the old file for comparison, and verify
proposal prompt snapshots, GOALS utility-query output, reference-doc content
hash/chunk metadata, and representative retrieval results before approval.
