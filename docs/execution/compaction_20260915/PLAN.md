# Codebase compaction plan — 2026-09-15

Goal: minimize maintained LOC and LLM context tokens while preserving behavior,
documentation information, and readable code. This is a plan, not an implementation.
Three GPT-5.6 Luna agents audited separate areas; Astra measured and reviewed results.

## Findings and recommended direction

Start with a small formatting pilot, then remove proven duplicated implementations
and test setup. Treat documentation as its own workstream with an information map.
Do not set a repository percentage target before measuring those pilots.

Line folding alone has modest token value: the conservative 100-column scan found
4,140 removable Python lines but only 6,081 tokens saved. Larger opportunities need
actual removal of repeated text and logic. Some apparently verbose code expresses
failure boundaries, ordering, or regression evidence and should stay explicit.

Documentation has substantial context cost. However, proposal generation reads
several docs directly, and startup indexes top-level `docs/*.md`. Those documents
are runtime inputs: equivalent prose does not imply equivalent model behavior.
This audit lives in a subdirectory outside that default, nonrecursive seed glob.

## Baseline and coverage

Start: `216b513352270b73cd2aa42985f461d97f78dc58`; the user's pull moved HEAD to
`eccd39f66d21efceead1344f29b86ab7dedf20b8`, changing only
`tests/unit/test_background_storage_receipts.py`. Measurements were refreshed.
Working-tree source was clean at the start; two pre-existing untracked user docs
were left untouched. Recheck [revision_check.json](revision_check.json) before work.
After measurement, external edits changed `docs/BUG_CLASSES.md` and local context
docs. Those current texts are outside the frozen totals/review; the checkpoint
names each file needing remeasurement.

| Tracked material | Files | Physical lines | Tokens, cl100k_base |
|---|---:|---:|---:|
| Python, including tests/scripts/probes | 1,022 | 335,785 | 3,212,545 |
| Markdown | 245 | 89,040 | 1,509,964 |
| TS/TSX/CSS/HTML | 31 | 4,074 | 33,716 |
| Other text | 90 | 23,578 | 271,053 |
| Lockfiles/generated or data records | 4 | 5,863 | 95,425 |
| Binary assets | 9 | — | — |
| Total tracked | 1,401 | 458,340 | 5,122,703 |

Tokens use a cached comparative tokenizer, not a claim about Astra's tokenizer or
billing. Lines count `splitlines()`, including comments/blanks. Categories describe
file types, not edit eligibility; prompts, policies, certificates and ledgers remain
contracts even if classified as ordinary text. Generated artifacts are not savings.

Python contains 52,998 blank lines, 24,658 comment-only lines and 37,140 docstring
lines. These counts overlap and are not a deletion budget. Tests alone contain
152,355 Python lines; improving repeated fixture/setup code merits a dedicated pass.

Local files outside tracked totals: `CLAUDE.md` is 456 lines / 45,120 tokens;
`CLAUDE_CHANGELOG.md` is 895 lines / 78,944 tokens. Giant prose lines already hide
verbosity. Moving history out of default context can save reading tokens but saves
zero repository LOC by itself, and CLAUDE is also a product prompt input.

All tracked text received mechanical metrics; all 1,022 Python files parsed and
received tokenizer/AST folding analysis, with no scan errors. Runtime agents also
scanned 242 Python files for identical function ASTs, finding 19 duplicate groups.
Semantic review covered selected regions, not every function in the repository.
Each area report lists inspected files and the remaining review boundary:
[runtime](runtime.md), [interfaces/config/eval](surfaces.md), [tests/scripts/docs](tests_docs.md).

Exact per-file hashes and scan status: [inventory.tsv](inventory.tsv).
Candidate locations: [fold_candidates.tsv](fold_candidates.tsv).
Totals: [summary.json](summary.json). Reproduce from the repository root:

```bash
python docs/execution/compaction_20260915/measure.py
```

The script reads tracked files without importing the application, simulates folds
in memory and compares whole-file ASTs without location attributes. It writes only
audit evidence. It excludes this audit directory and separately measures four named
local docs; ignored stores, environments, secrets, logs and backups are not audited.
An AST match does not cover source inspection, comments, hash contracts or tracebacks.

## Compaction rules

1. Prefer one short statement/call/signature per line; use 100 columns as the pilot
   limit, with human-reviewed 110-column exceptions. Keep longer calls and meaningful
   data rows multiline. Preserve logical whitespace and one assertion's purpose.
2. Never pack statements with semicolons, compress control blocks onto one line,
   shorten descriptive identifiers, or introduce nested ternaries/dense comprehensions
   for LOC. Do not remove type information, contracts, rationale or edge cases.
3. Preserve argument evaluation, await/cancellation, exceptions, logging, ordering,
   defaults, missing-vs-null distinctions, lazy imports, signatures and mutation scope.
   Keep public compatibility wrappers when needed; count their cost in net savings.
4. Treat prompts, regexes, SQL, rendered text, schemas, receipts and test fixtures as
   data. Preserve literal values and output bytes. A dict-to-defaultdict rewrite or
   eager default lookup is a semantic refactor, not a formatting change.
5. Extract a helper only for equivalent responsibilities. Count helper/import/wrapper
   overhead and extra files a reader must open; moving code to modules is not savings.
   Check `__doc__`, reflection and source-reading consumers before docstring edits.
6. For prose, shorten sentences and remove repeated explanations only after mapping
   each fact, exception, command, number, date, rationale and provenance to its retained
   home. Preserve valid links/anchors and distinguish historical from current facts.
   Reflowing paragraphs or replacing prose with huge tables is not token compaction.
7. Keep entry instructions self-contained for critical rules. Link optional detail
   by task; avoid chains of links that force an LLM or person to open many documents.
   Do not add generated mirror files that erase the net maintenance/token benefit.

## Ordered execution batches

| Batch | Scope and first candidates | Acceptance and stop boundary |
|---|---|---|
| C0: refreshed baseline | Recheck revision/hashes, source consumers and tests; select a 5–10-file pilot | Rebase locations after merges; record pre-existing failures separately |
| C1: mechanical pilot | Short calls, signatures, imports and literal containers from the candidate TSV; examples `api/routes/debug.py:67`, `core/actions/github_write.py:37` | Same location-free AST and literal values; comments intact; net tokens decrease; readable diff; no source-contract drift |
| C2: expand mechanical pass | Apply accepted C1 rules by subsystem, with disjoint cheap-agent file ownership | No blanket formatter migration; reject awkward folds; measure each batch before extending |
| C3: local shared logic | Shutdown session selection; repeated fallback rosters; prompt dedupe; web result rendering; frontmatter parsing | Inspect all callers first; preserve exact failure/order/output behavior; focused outcome tests |
| C4: tests and scripts | Repeated mock-store builders/setup; equivalent CLI/environment scaffolding; compact explicit case records | Preserve test cases, outcomes, fixtures, marks, isolation and nodeids; scripts preserve exit/dry-run/backup behavior |
| C5: uncoupled documentation | Pilot `eval/README.md`/`eval/PLAN.md`, then prove consumer status for each further doc | Information mapping, working links/anchors, historical status and instructions preserved; no runtime text/hash change |
| C6: runtime docs and immutable records | Proposal input docs, top-level seeded docs, bug catalog, historical receipts | Defer prose changes under strict behavior preservation; proceed only with exact consumer output or a separately scoped behavior change |

For C1/C2, measured fold candidates at different widths are alternatives, not additive:

| Maximum width | Logical statements | Removable lines | Token decrease |
|---|---:|---:|---:|
| 88 | 533 | 1,409 | 2,076 |
| 100 | 1,728 | 4,140 | 6,081 |
| 110 | 3,066 | 7,323 | 10,643 |
| 120 | 4,400 | 10,842 | 15,470 |

These are measured simulations, not reviewed/approved edits or an upper bound on
all opportunities. Even 120 columns saves only about 0.48% of Python tokens; 100
columns saves about 0.19%. The heuristic excludes comment-bearing statements,
multiline string tokens and backslashes; it does not exhaust legal formatting.

High-value C3 candidates, with rough local estimates after a small helper but before
full import/test/compatibility accounting; do not sum these as a promised total:

| Candidate | Current sites | Estimated local LOC reduction |
|---|---|---:|
| Shutdown session selection | `memory/shutdown_processor.py:648,1053,1195,1525` | 30–40 |
| Note-generator fallback roster | daily/weekly/monthly generators, `utils/`, lines 838/549/539 | 24–30 |
| Minimal error contexts | `core/prompt/builder.py:2137,2376` | 18–25 |
| Web result formatting | Two result classes in `knowledge/web_search_manager.py` | 15–18 |
| Config search-path setup | `config/app_config.py:89,135` | 10–14 |
| Prompt dedupe helper | `core/prompt/base.py:58`, `formatter.py:105` | 8–10 |

Keep the smallest 4–9-line cross-module deduplications low priority if their imports
and indirection outweigh readability. `gui/handlers.py` (6,273 lines) merits later
semantic review; splitting it alone is not a compaction result. API models and
streaming state machines already carry dense contracts and should not be first pilots.

## Documentation dependencies and preservation

`knowledge/proposal_generator.py:115` reads CLAUDE and QUICK_REFERENCE in full,
filters PROJECT_SKELETON by headings and slices GOALS to 4,000 characters.
`core/prompt/proposal_filter.py:99` parses GOALS headings into ranking queries.
`gui/launch.py:271` seeds configured docs through
`knowledge/reference_docs_manager.py:642`, using a nonrecursive `*.md` glob and
content hashes. Whitespace/prose edits can change truncation, chunks and retrieval.

Potential canonical map after those dependencies are satisfied: README for onboarding,
ARCHITECTURE_GUIDE for conceptual design, subsystem docs for contracts, QUICK_REFERENCE
for navigation, one metrics source for volatile counts. PROJECT_SKELETON currently
costs 92,855 tokens; QUICK_REFERENCE 46,429; ARCHITECTURE_GUIDE 42,349. These are
review priorities, not claims that their text is redundant or safe to remove.

Historical batch/brief/response files contain substantial repeated framing, but also
unique evidence and receipts. Start with one field-preserving comparison, retain
original evidence, and check machine consumers. An added index may improve navigation;
it does not reduce total LOC. Do not replace retained history with lossy summaries.
`docs/BUG_CLASSES.md` has a strict parser and incident fingerprints, so its layout
and text are also contracts (`scripts/bug_class_guards/catalog.py`).

## Verification and execution discipline

For formatting: compare ASTs/literals, review comments and source-sensitive guards,
then run existing focused tests and `ruff check` on the touched scope. AST identity
alone is insufficient. Do not add tests that merely restate a formatting operation.

For shared logic: capture outputs through deployed functions; test success/failure,
empty/missing/corrupt data, ordering and relevant cancellation branches. Compare
serialized payloads, prompt strings, receipts and relevant side effects. For tests,
compare collection/nodeids and marks under the same selection; preserve fixture scope,
monkeypatch cleanup and assertion independence. Do not turn separate cases into a
loop that stops after the first failure. Parametrization must preserve case identity.

Use the established lanes in [TEST_LANES.md](../../TEST_LANES.md): affected suites,
the five repo-wide guards and bug-class contract; frontend changes also run
`npm run typecheck` and `npm test`. Follow memory caps and Daemon-down requirements
for non-unit batches; do not run the full suite beside the live application.

Accepted-debt dispositions bind source SHA-256, so even formatting can require
re-review (`scripts/bug_class_guards/dispositions.py:249`). Some guards bind source
text or document fingerprints. Preserve enforcement, review remaining debt and update
only justified records; never blanket-rebaseline or weaken guards to land compaction.
The post-pull `test_background_storage_receipts.py:363` also uses a fixed source-text
window: formatting can change that test's result without changing runtime behavior.

Use cheap agents on disjoint small batches; parent reviews semantics and evidence.
Do not combine compaction with bug fixes, dependency changes, renamed public symbols
or prompt tuning. Stop a batch on unexplained behavior/guard drift. Keep successful
work reviewable for the owner's normal commit process; this audit creates no commits.

## Checkpoint and remaining work

Completed: repository-wide inventory/metrics, Python syntax and folding scan, runtime
duplicate-function scan, representative area reviews, consumer tracing, and this plan.
No application edits, application model calls, store writes, test-suite execution or commits occurred.
The measurement script passed Ruff; inventory/candidate totals and local document
links were checked. External catalog drift is recorded, not treated as a clean snapshot.

Remaining: exhaustive semantic review of uninspected functions/doc sections, pilot
implementation, measured net savings, full information mappings and behavior checks.
No repository-wide semantic-review or behavior-equivalence claim is made.

Resume at C0, then C1 with the candidate TSV and area reports. At each stop record
revision, exact files/regions reviewed, next candidate, line/token deltas, tests actually
run and unresolved risks. All inventory-only files remain unreviewed semantically;
use their hashes and each area's inspected list instead of restarting the whole audit.
