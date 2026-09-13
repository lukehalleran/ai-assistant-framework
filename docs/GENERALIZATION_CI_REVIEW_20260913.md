# Independent generalization sweep and three CI layers — 2026-09-13

The implemented class ratchet works for its declared patterns. It does not
block every known bug class, and GitHub currently does not require its success
before changes land. The third layer, assembled-turn probes, remains a design.
Generalization has useful shared mechanisms, but fresh-user, privacy, Windows,
and response-integrity requirements still have concrete gaps.

## Scope and evidence

Reviewed local `432c217` with three cheap GPT-5.6 Luna auditors: runtime and
language, platform and lifecycle, and scanner sensitivity. The parent reviewed
the evidence, checked the installed hook and GitHub run, and ran the five
additional repo-wide guards. Coverage includes all G01–G13 documents, both
older generalization inventories, the development workflow, test-lane ledger,
beta workplan, and probe design. This is a source/test sweep, not external
population validation, a Windows run, or a live-session audit.

No application code, baseline, stores, or remote settings were changed. No paid
model probes or full local suite were run. The three pre-existing untracked
design/plan documents remain untracked. Documentation preimages and synthetic
scanner evidence were saved additively under
`.agent_snapshots/generalization_ci_20260913T161049Z/`; no snapshot rotation
or deletion was needed for this documentation batch.

## What actually ran

[GitHub run 34728362771](https://github.com/lukehalleran/ai-assistant-framework/actions/runs/34728362771)
tested pushed commit `85435ed` and completed successfully on September 13 UTC
(September 12 local time). The guard step executed; it was not skipped.

| Layer | Verified result | Limit |
|---|---|---|
| Backend tests | 10,185 passed; 18 skipped; 383 deselected; 1 xfailed | Result belongs to `85435ed`, not local `432c217` |
| Frontend | Typecheck succeeded; 9 tests passed in 2 files | No Windows, build, or accessibility job |
| Class harness | 63 passed remotely and locally | Tests selected scanner patterns and catalog rules |
| Five repo-wide guards | 85 passed locally at `432c217` | Git-state, slices, prompt budgets, tool wiring, model capabilities |
| Catalog inspection | 78 classes, 31 detection methods, 14 closure methods; no diagnostics | Consistency does not prove behavioral coverage |
| Probe harness | No runner or CI job found | Design and a starter backlog only |

The installed `.git/hooks/pre-push` is the symlink to `hooks/pre-push`; its
contents match. `.github/workflows/tests.yml` invokes the scan and isolated
harness before the main test suite. Events cover pushes to `master` and
`refactor/prompt-modular`, and PRs to `master`; there is no scheduled run.

GitHub's branch endpoint reported `master.protected=false`, protection
`enabled=false`, and no required checks. Thus the hook can refuse a local
push, and CI can fail a completed push, but CI success is not a server-enforced
condition for landing. Local HEAD is one commit ahead of the pushed tree.
Read-only verification command:

```bash
gh api repos/lukehalleran/ai-assistant-framework/branches/master \
  --jq '{protected: .protected, sha: .commit.sha, protection: .protection}'
```

## Why the first scanner push could be green

The September 11 bounded scan brief explicitly superseded the earlier all-class
contract/receipt plan as over-scoped. The delivered policy accepts matching
baseline findings; it rejects additions and stale entries. It never meant
“zero known defects.” Current observations:

| Scanner | Mode | Local files | Findings / baseline | Pushed CI files |
|---|---|---:|---:|---:|
| DM-01 raw substring | gate | 393 | 12 / 12 | 392 |
| DM-17 apply/store isolation candidates | gate | 593 | 39 / 39 | 592 |
| DM-18 broad exception → empty | gate | 122 | 79 / 79 | 122 |
| DM-31 literal live-state defaults | gate | 393 | 3 / 3 | 392 |
| Catalog consistency | gate | 1 | 0 / 0 | 1 |
| DM-16 config reachability | report | 392 | 87 / n/a | 391 |
| DM-29 repeated phrase additions | report | 1 | 0 / n/a | 0 |

All 133 gated findings match the baseline. The 87 DM-16 candidates do not
affect success. DM-29's input is the untracked local changelog, so CI's zero
files means **unavailable evidence**, not a clean phrase-generalization scan.
These are candidate counts, not counts of confirmed product defects.

### Sensitivity proof and remaining holes

The auditor ran the deployed CLI entry point on disposable synthetic trees,
with mutations confined to those trees or the scratch process's registry:

| Control | Observed result |
|---|---|
| New finding in each of all five gates | Exit 1 for each |
| Removed finding with stale baseline, each gate | Exit 1 for each |
| Empty input, four Python gates | Exit 1 for each |
| Missing catalog | Exit 2 |
| Malformed Python/catalog, each gate; malformed baseline JSON | Exit 2 |
| Identical fingerprint multiplicity increases/decreases | Exit 1, NEW / STALE respectively |
| Exact baselines; harmless line movement | Green in the isolated harness |
| Remove a scanner from the in-memory registry | Exit 0: obligation disappears |
| Change a scanner from gate to report | Exit 0: obligation disappears |
| Raw substring added beside a trigger-match import | No finding: module exemption |

The last three controls expose limitations. Registry mutation evidence is
about the CLI, not an actual modified workflow run on GitHub. Current registry
tests check valid metadata, not the required set and modes
(`tests/bug_class_guards/test_scanners.py:53`). The CLI filters baseline rows
by selected active gates (`scripts/check_bug_classes.py:112`;
`scripts/bug_class_guards/baseline.py:110`).

DM-01 deliberately exempts an entire module importing the shared matcher
(`scripts/bug_class_guards/scanners/dm01_raw_substring.py:105`); its green
fixture explicitly contains this blind spot. That matches the bounded brief
but does not establish that every relevant expression uses the matcher.
DM-17 likewise recognizes a guard reference, not proof that it runs before
every write; its two source inventories share one nonzero-file count.
The Python inventory has fixed roots; nonzero totals do not prove every
subsystem or required root was examined.

Missing baseline on a finding-free tree can pass the standalone CLI; the
wired harness separately requires the committed baseline to exist. Explicit
`--write-baseline` intentionally accepts the current findings. That flag must
never appear in enforcement commands or an automatic baseline refresh.

The older five guards also have scoped heuristics: prompt-budget extraction
matches a particular `context.get` spelling; tool parity includes source-token
checks. Their 85 passing tests are useful evidence, not a proof of arbitrary
rewrites or behavioral correctness.

## Generalization findings, reconciled with the roadmap

Priority here means ordering for the next engineering batches. “Release gap”
means a stated requirement is unimplemented or lacks acceptance evidence;
it does not assert a regression from a previously qualified release.

| ID / priority | Verified finding and source | Contract / next acceptance |
|---|---|---|
| F01 / P1 current boundary gap | API factory installs CORS but no launch-token/Origin/Host authorization (`api/app.py:69`); approval and note-sync POSTs have no such checks (`api/routes/actions.py:38`, `api/routes/system.py:47`) | G06-T02/A03: reject unapproved origins at the server, including simple POSTs; test the deployed app with fake executors. Source evidence; no live attack performed. |
| F02 / P1 packaged-user gap | Four resolvers default to relative `data/user_profile.json`: identity, location, institution, timezone. Frozen `UserProfile` uses APPDATA (`memory/user_profile.py:108`). Timezone system lookup only understands `/etc/localtime`, then falls back to Central (`utils/timezone_resolver.py:64,112`). Calendar create consumes it (`core/actions/google_calendar_create.py:227`) | G02 path/context contract, G08 clean install: use one configured profile path and validate non-Central Windows event times. Earlier timezone closure was partial. |
| F03 / P1 parser defect | `_parse_verdict` uses `bool(data.get(...))`, making JSON string `"false"` true (`core/grounding_check.py:527`) | G12 strict boolean requirement: reject strings/numbers/missing required fields; drive parser and delivery tests. Default log-only limits reply mutation, not false verdict telemetry. |
| F04 / P1 opt-in correction gap | Failed/disabled integration returns an appended suffix (`gui/handlers.py:3252-3266`; `core/grounding_check.py:899`); tests preserve it | G12 acceptance 4–6: integrated fallback and identical displayed/stored/indexed final text. Default log-only mitigates exposure; it does not validate `correct` mode. |
| F05 / P1 beta release gap | Beta workplan claimed content never leaves while requiring hosted model keys. No runtime privacy-mode contract was found | G01/G03: label hosted inference and tool egress explicitly. The misleading plan paragraph is corrected in this batch; runtime disclosure remains to build. |
| F06 / P1 beta release gap | Normal telemetry includes query text and response plans (`core/orchestrator.py:343`, `utils/turn_telemetry.py:137`). Ordinary debug/info logs also contain content (`utils/query_rewriter.py:134`, `utils/logging_utils.py:111`) | G03-T09, G11: separate owner-local incident records from exportable beta metrics; fixed allowlist and seeded-canary tests. Local debug content is intentional today, not evidence of an existing automatic export. |
| F07 / P1 privacy-mode gap | IP lookup is on by default; location resolver contacts external services and contributes location to web-trigger prompts (`utils/location_resolver.py:97,158`, `utils/web_search_trigger.py:1607`) | G03-A07/A12/A13: explicit location/egress policy and no-network tests. Test mode already suppresses the background lookup; that does not qualify production OFFLINE mode. |
| F08 / P2 fresh-user assumptions | Default personality assumes an established close relationship and gendered user (`config/prompts/default_personality.txt:1,42,61`); loaded by orchestrator (`core/orchestrator.py:1236`) | G02-T04, G09-T05: neutral empty-profile prompt; relationship/style settings backed by explicit preferences. |
| F09 / P2 name support | Display-name regex requires ASCII uppercase initials (`utils/user_identity.py:33`); accented/non-Latin initials fail | G02-A09: Unicode-name fixtures through deployed resolution, not only regex matching. |
| F10 / P2 prominence gap | Quick Profile promotes a fixed relation set (`memory/user_profile.py:411`); no signal-based path for other durable identity facts | G02-T05/G04-T06: confirmation/durability/pinning-based prominence, with demotion and diverse-profile tests. Previously parked work remains open. |
| F11 / P2 adaptation/calibration gap | Learned exemplars have caps and provenance but no unified reset/explanation contract (`utils/adaptive_exemplars.py:125`); fixed retrieval defaults remain (`config/app_config.py:263`) | G04: inventory learning stores; reset and poisoning controls; sparse/mature corpus evaluation before threshold changes. |
| F12 / P2 domain-coverage gap | Fast tool cues remain developer-heavy (`core/agentic/gate.py:174`); school-specific prompt expansion remains although employer/org anchors reach the private-sphere guard (`utils/web_search_trigger.py:1372,1614`) | G04-T07/G09-T07: categorized tasks and organization anchors across consumers; measure fallback behavior before claiming entire domains fail. |
| F13 / P2 accessibility gap | Chat lacks explicit live-region semantics and icon controls lack explicit accessible names; smooth scrolling follows each content change (`web/src/components/chat/MessageList.tsx:21`, `ChatInput.tsx:63,94`) | G09-T01/T02: keyboard, accessible-name, buffered announcement, and reduced-motion checks, followed by assistive-technology users. Static inspection is not a browser accessibility audit. |
| F14 / release gap | Session-only/off/ask-to-store policy and deletion across derivatives are not implemented; profile history is append-only (`memory/user_profile.py:20`), settings expose runtime knobs | G07: inventory writers, then enforce storage policy across foreground/background/shutdown paths and test deletion/recovery. Curation undo is a different contract. |
| F15 / release gap | `daemon.spec` exists but lacks current SPA bundle; no planned fresh-clone smoke or Windows CI found (`daemon.spec:45,96`; `.github/workflows/tests.yml:12`) | G08: config-free smoke including `web/dist`, then clean Windows launch/recovery matrix. Corrected the stale “spec absent” doc claim. |
| F16 / release gap | No unified inference-role gateway/qualification registry; hosted and local generation coexist (`models/model_manager.py:404`) | G10: role/call-site inventory, qualified artifacts, explicit local failure, and network-denied validation. Model-slug parity is narrower. |
| F17 / evidence gap | Existing eval is a prompt-ablation corpus without cohort/consent/subgroup fields (`eval/corpus.py:67`); no inspected E3–E5 population evidence | G01/G05: machine-readable product/eval contracts, separate owner/synthetic/held-out evidence. No population claim follows from this sweep. |

G11 also lacks the planned user feedback/incident-bundle workflow and private
canary export gate. G13's top status note correctly marks its older body as
historical: graph filtering has production call sites, and an integrator exists;
do not revive the older “no caller/suffix only” claims. G12's fallback issue
above remains current. Expanded insight vocabulary and shared keyword matching
are verified improvements from the older sweeps, not findings to reopen wholesale.

## Third layer: starter query backlog

The existing [probe design](PROBE_HARNESS_DESIGN.md) is the third-prong document.
The following 15 cases are **proposed synthetic fixtures**, not completed
probes or copied private transcripts. Each needs a machine-readable case,
seeded state, expected receipts, forbidden side effects, and counterexample.
Cases span G02/G04/G06/G09/G12 and the existing incident classes.

| Case / class | Synthetic query and seed | Required outcome / counterexample |
|---|---|---|
| Q01 / BC-01,28 | Coursework seed; `Why does the number column contain NA?` | No distress floor from “number”; paired genuine distress turn must retain appropriate routing. |
| Q02 / BC-01,58 | `Check the deadline warning.`; repeat wrapped/indented | No “dead”/“war” substring hit; positive true-word/inflection controls. |
| Q03 / BC-32,78 | News seed; `Please verify that public announcement.`; zero web credits | Consistent budget receipt, no dispatched search, honest unmet-evidence response; positive budget case searches within its cap. |
| Q04 / BC-73,58 | `I have a new doc to review.`; assistant history previously mentioned a doctor | User-authored evidence controls referent; no invented doctor in STM, key points, strategy, or response. |
| Q05 / BC-74 | Stale offer seed; `Add a meeting tomorrow at 2 PM for 30 minutes.` | New explicit request determines card parameters; positive short affirmation still accepts a valid current offer. |
| Q06 / BC-46,64 | Q05 card followed by approval, fixed clock and non-Central timezone | Fake calendar executor receives valid aware times, executes once, returns matching receipt; malformed clock/card fails. |
| Q07 / BC-77 | Duplicate-content seed; `Save a note titled Reading Plan with these steps.` | Explicit save produces a saved/updated outcome or honest refusal; autonomous dedup cannot silently masquerade as success. |
| Q08 / BC-46,75 | `Is the event already on my calendar?`; prior assistant claim but no execution receipt | Prior assistant prose is not completion evidence; paired real receipt permits supported completion. |
| Q09 / BC-05,30 | `What does the document I uploaded say?`; document uploaded five days before fixed clock | Correct document identity; no “uploaded today” claim; fresh-upload control remains valid. |
| Q10 / BC-31,04 | `What changed in this repository since yesterday?`; temporary repository with newer commit than seeded index | Fresh local history reaches final context within budget; historical-document query keeps its intended retrieval. |
| Q11 / BC-11,12 | Same test instance, disable web search, then `Look up the latest forecast.` | Both gate and dispatch observe changed state; honest disabled outcome; re-enable control exercises restoration. |
| Q12 / BC-20,47 | `What notes do I have about the project?`; injected retrieval failure | Distinguish failure from genuinely empty store; no false “no notes exist” inference. |
| Q13 / BC-63,64; G12 | `Summarize the supported finding.`; cassette verifier emits string `"false"` | Invalid verdict does not become a positive correction; actual JSON booleans and a substantive true error are controls. |
| Q14 / BC-58; G12 | Seed draft with one material error; force integrator timeout | Final display/storage/index receipt agree, with no contradictory suffix; successful revision and log-only controls. |
| Q15 / BC-14,46 | `Explain the next step.`; stream fixtures contain split reasoning markers, empty/reasoning-only output, and provider error | No raw protocol artifact; explicit error/empty-result handling; clean stream preserved. |

Run cases under clean/wrapped spelling, terse/polite wording, fresh/mature
profiles, school/work/caregiving/hobby contexts, varied names, and timezones
where relevant. Keep semantic invariants constant; a fixture-specific phrase
patch is not closure. Add targeted non-query contracts for F01/F06/F14/F15:
API authorization, telemetry canaries, storage policy, and installer behavior.

Before running assembled turns, create a disposable instance with one explicit
root for corpus/profile/vector/graph/uploads/actions/logs/learning/backups;
scrub provider credentials and deny unexpected networking. Test shutdown and
background writes too. `DAEMON_TEST_MODE` alone does not establish isolation.

Drive the real turn path and stub only provider/tool boundaries. Cassette
misses fail explicitly; never fall through to a paid provider. Record all
inference roles and streamed chunks. Prompt hashes help identify requests but
must be accompanied by model/settings/tool-schema and fixture versions; a
changed prompt is a reviewable cassette update, not permission to auto-record.
Assert final outcomes and write/execution receipts as well as routing. Missing
required receipt fields fail. Shape-only prose assertions begin report-only;
provider-drift probes are a separate explicitly budgeted live lane.

## Bounded execution order

1. **Guard contract batch:** pin required scanner IDs/modes, reject orphaned
   baseline obligations on full scans, test missing source legs, and protect
   hook/workflow commands from silent omission or baseline-writing flags.
   Preserve explicit partial-scan behavior. Precision-audit DM-01 before
   removing its module exemption; do not bulk rebaseline away new candidates.
2. **Current defects:** separate small batches for strict verifier parsing and
   unified resolver paths/timezone. Add failing deployed-function tests first.
   Triage the API boundary before sharing a build with testers.
3. **Probe foundation:** schema + Q01–Q15 seeds, sandbox construction and
   isolation canaries, cassette boundary, then the real turn runner. Add a
   few proven cases per batch; no claim of 15 passing probes until executed.
4. **CI integration:** deterministic receipt/outcome job with no path filters,
   missing-case/empty-collection failures, and measured resource bounds. Add
   schedule only after measuring it. Local heavy runs remain sequential with
   Daemon down; the existing pre-push 6 GiB cap is below the documented 8 GiB
   non-unit requirement and needs a separate measured repair.
5. **Landing enforcement:** after check names and negative runs are verified,
   configure required checks on protected master and use reviewed PRs. This
   audit only read remote policy; it did not change it or create a failing PR.
6. **Generalization milestones:** connect F01–F17 to the existing G requirements;
   carry privacy, memory control, packaging, accessibility, inference roles,
   and external population evidence as explicit release gates.

Retain the workflow's cheap-executor/frontier-review split, disjoint file
ownership, outcome fixtures, BC-58 sibling enumeration, and small reviewable
batches. Its “three development prongs” are collaborators; the three quality
layers here are tests, class scans, and assembled-turn probes. They are
different groupings. Current hook selections and optional `PREPUSH_FULL` run
as separate pytest processes; older “ONE process” wording in historical
workflow/ledger passages is stale. There is still no nightly job implemented.

Reproduction of cheap checks:

```bash
python scripts/check_bug_classes.py scan --root .
python scripts/check_bug_classes.py inspect --catalog docs/BUG_CLASSES.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  --confcutdir=tests/bug_class_guards tests/bug_class_guards
env DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 120s python -m pytest -q \
  -p no:cacheprovider tests/unit/test_no_git_state_in_tests.py \
  tests/unit/test_ordered_slice_guard.py \
  tests/unit/test_budget_meters_rendered_sections.py \
  tests/unit/test_tool_wiring_parity.py tests/unit/test_model_capability_wiring.py
```
