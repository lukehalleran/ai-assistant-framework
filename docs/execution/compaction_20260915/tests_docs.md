# Compaction audit: tests, scripts, docs, and eval (2026-09-15)

## Scope and method

This is a plan-only audit. No source, tests, data, generated artifacts, or user
work were changed; no test suite was run. `git ls-files -z` mechanically
inventoried the assigned paths, with `wc` byte/line counts and targeted `rg`
pattern scans. I read the local contract in `CLAUDE.md`, root `PLAN.md`,
`eval/PLAN.md`, `eval/README.md`, and representative high-value files. This is
not an exhaustive semantic review of every test, script, or document.

Tracked inventory (root measurement, refreshed at `eccd39f`; `splitlines()` counts):

| Area | Files | Lines | Bytes |
|---|---:|---:|---:|
| `tests/**` | 560 | 155,416 | 6,497,517 |
| `scripts/**` | 160 | 32,020 | 1,329,730 |
| `docs/**` | 259 | 97,604 | 6,074,736 |
| `eval/**` | 18 | 6,446 | 230,906 |

These rows are disjoint by path. `docs/PLAN_20260912_session_defects.md` (318
lines, 18,993 bytes) and `docs/SOURCE_DOCUMENT_TIER_DESIGN.md` (204 lines,
12,316 bytes) are untracked user work: inspected only and must not be edited.
`CLAUDE_CHANGELOG.md` is local context, not included in the assigned tracked
inventory.

## Findings and candidate ordering

1. **Docs history and generated execution records (structural repetition, but
last in implementation order).**
`docs/execution/generalization/batches/` is 73 files / 35,271 lines / 2.25 MB;
the parallel `briefs/` is 38 / 4,512 / 517 KB. Their repeated per-batch
structure is visibly shared, but the inspected F13c-1 batch contains unique
timestamps, hashes, command output, deviations, failures, and limitations. Its
F13c-1 brief is a separate contract/checklist, while the inspected CGR response
is an immutable receipt with before/after SHAs and test evidence. Do not rewrite
or delete immutable records in bulk. First prove a machine-readable index or
generated view can preserve every ID, verdict, receipt, failure outcome, source
excerpt, and link byte-exactly. The 30
`class_guard_responses/` files (7,147 lines / 424 KB) and 10 request records
(820 / 64 KB) similarly repeat request/response boilerplate. No savings estimate
is justified before that pilot; measure only after a field-preserving render/diff
and consumer search.

2. **Canonical docs map (medium risk; first classify runtime consumers).**
`docs/PROJECT_SKELETON.md:2600-2635` records that proposal generation reads
`CLAUDE.md` and `QUICK_REFERENCE.md` in full and a filtered skeleton, while
reference-doc seeding can place markdown into the `[DAEMON DOCUMENTATION]`
prompt. Those runtime-coupled inputs require exact emitted-prompt/index
comparison and cannot be compacted as ordinary narrative. `CLAUDE.md` also has
high context cost (the current cached metric is about 45,120 tokens); moving
repeated text out of the default context may reduce prompt cost, but produces
zero repository LOC savings unless the runtime reader changes. `docs/ARCHITECTURE_GUIDE.md`
is the broad architecture reference (for example request lifecycle at lines
246–341 and memory architecture at 638–714); `docs/QUICK_REFERENCE.md` repeats
operational module descriptions and current fixes (for example lines 81–236
and 279–405). `README.md` repeats architecture, metrics, and test commands
(lines 150–195, 270–286). Keep README as entry-point/reviewer path, ARCHITECTURE
as the canonical conceptual reference, QUICK_REFERENCE as a deliberately short
operator index, and replace repeated narrative with links plus a generated fact
block only for files proven unconsumed by runtime. Preserve headings/anchors
used by links. Savings are unestimated until a section-level before/after map
exists. Validate all relative links and heading IDs.

3. **Eval plan/readme overlap (low risk).** `eval/PLAN.md` repeats architecture,
section order, phase deliverables, and test counts that `eval/README.md` also
explains (`PLAN.md` lines 95–236; README lines 1–23, 111–179). Keep PLAN as
historical design/status and README as runnable usage; move volatile counts to
one generated table and link the architecture reference. Potential savings are
unestimated pending a section-level pilot.

4. **Tests: repeated scaffolding, not blanket deduplication (medium risk).**
`tests/unit/test_sep10_probe_dump_actions.py:1-65` and
`tests/unit/test_sep10_probe_dump_interpretation.py` contain intentionally
verbatim live turns and round labels; preserve those IDs/nodeids and source
texts. `tests/unit/test_cross_deduplicator.py:28-91` has reusable mock-store and
embedding builders; similar local builders recur across unit tests. A later
pass may centralize only demonstrably identical helpers, with compatibility
wrappers preserving nodeids. `tests/test_eval/test_variants.py:20-82` has a
small domain-specific snapshot builder that should remain local unless all eval
tests share the same contract. No savings estimate is made until AST/helper
equivalence and pytest nodeid diffs are measured.

5. **Fixtures and assertion blocks (medium risk).** Long case tables in
`tests/test_tone_detection.py:57-302`, `tests/unit/test_tone_observational_mild_distress.py:52-169`,
`tests/unit/test_tone_keyword_boundaries.py:175-190`, and
`tests/unit/test_sep12_keyword_boundary_and_budget.py:158+` look mechanically
compressible, but each row can encode a regression boundary. Convert only to a
schema-backed table with stable IDs and one assertion loop; retain comments,
expected values, provenance, and parametrized IDs. Do not remove edge cases or
replace deployed-function calls with proxies. No savings estimate is made
pending a per-row semantic diff.

6. **Scripts (moderate savings, behavior-sensitive).** `scripts/audit_runtime_smoke.py:1-43`
and `:80-150` deliberately spell out isolation, network fail-closed behavior,
and persistence receipts; keep those guards explicit. `scripts/generate_doc_metrics.py:1-120`
has useful single-source derivation and should remain canonical, while repeated
CLI/environment setup across migration/probe scripts is a candidate for a
shared helper only if exit codes, dry-run defaults, and receipt fields remain
byte-for-byte contract-compatible. The bug-class scanner contract is explicit
in `scripts/bug_class_guards/scanners/dm01_raw_substring.py:1-31`; preserve
contract/version/leg semantics. No savings estimate is made until helper
equivalence, exit-code parity, and dry-run/receipt tests are measured.

7. **Empty package initializers.** Exact-hash scan found identical empty
`__init__.py` files in `tests/agent_branch`, `tests/benchmarks`,
`tests/test_eval`, and `tests/unit`. They are negligible and may be left alone;
removing one risks import/package behavior for no meaningful token gain.

## No-information-loss pilot

Use a small non-private pair before any broad rewrite: keep the numeric and
exception-bearing contract in `eval/README.md:111-124` (the seven named test
commands) and the corresponding phase/test inventory in `eval/PLAN.md:257-306`.
The compact form may move shared prose to one canonical section, but must retain
each command, test filename, phase status, count, and the warning that eval
generation bypasses persistence side effects. Compare old/new rendered text,
links, headings, and extracted command/filename sets. A passing diff is a
prerequisite for touching larger docs families.

## Preservation and verification requirements

Every compaction candidate must retain: test names and parametrized IDs (or a
checked nodeid mapping), all edge-case rows and expected values, deployed
function invocation, source excerpts and provenance, dry-run/delete guards,
receipts and audit fields, document links and heading anchors, instructions,
historical status/date, and generated-file ownership markers. Do not compact by
deleting incidents, changing assertions to broad snapshots, or collapsing a
failure/not-run result into an empty success.

Recommended order: (1) docs canonicalization and link/anchor check using the
small eval pilot; (2) a read-only index/view pilot for batch/brief/response
records; (3)
eval README/PLAN deduplication; (4) script helper extraction; (5) test helper
and fixture normalization. Before/after checks should include `git diff
--check`, markdown link/anchor validation, exact manifest and byte counts plus
the repository's cached tokenizer metric (never a bytes/4 proxy),
AST parse/compile, `pytest --collect-only` nodeid comparison, targeted tests for
changed families, and the repository's prescribed test lanes. Data and live
stores remain out of scope.

## Exact coverage manifest

Mechanically inventoried (all tracked files): `tests/**` (560), `scripts/**`
(160), `docs/**` (259), and `eval/**` (18), using `git ls-files -z` plus byte
and line counts. Root markdown was included only where it matched those path
globs; no separate overlapping root rollup is reported. Binary/non-text paths
were counted but not semantically read; no user data tree was opened.

Inspected in detail: `CLAUDE.md`, `PLAN.md`, `README.md`, `eval/PLAN.md`,
`eval/README.md`, `tests/unit/test_sep10_probe_dump_actions.py`,
`tests/unit/test_sep10_probe_dump_interpretation.py`,
`tests/unit/test_cross_deduplicator.py`, `tests/test_eval/test_variants.py`,
`tests/test_tone_detection.py`, `scripts/audit_runtime_smoke.py`,
`scripts/generate_doc_metrics.py`, `scripts/bug_class_guards/scanners/dm01_raw_substring.py`,
`docs/ARCHITECTURE_GUIDE.md`, `docs/QUICK_REFERENCE.md`, `docs/BUILD_GUIDE.md`,
`docs/DEVELOPMENT_WORKFLOW.md`, `docs/TEST_LANES.md`, `docs/METRICS_SNAPSHOT.md`,
`docs/BUG_CLASSES.md`, `docs/BUG_RETROSPECTIVE_20260715_20260904.md`, plus the
two named untracked design/plan docs (read-only).

Deferred semantic review: every tracked file under those globs outside the
explicit inspected list above. The mechanical inventory is complete, but the
semantic read was partial and representative; this report makes no claim of
exhaustive semantic review or of a precise remaining-file count.

## Resume checkpoint

Next agent should start with the generated generalization families, build a
field-preserving compact representation and manifest, then run the verification
requirements above. Re-read both untracked design docs before any consolidation
that touches source-document tiers or session-defect contracts. No edits or
commits were made by this audit.
