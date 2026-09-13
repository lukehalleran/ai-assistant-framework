# Generalization execution plan — 2026-09-13

Status: execution plan only. No application, CI, data, or runtime changes are
authorized by this document. This plan is paired with the independent class-guard
plan and the evidence in
[GENERALIZATION_CI_REVIEW_20260913.md](GENERALIZATION_CI_REVIEW_20260913.md).

## Purpose and evidence boundary

Turn review findings F01–F17, the G01–G13 roadmap, and probe cases Q01–Q15 into
small reviewable implementation batches that inexpensive agents can execute
without inventing product policy. The parent agent owns integration, judgment,
and review. A cheap worker receives one behavior batch at a time, changes only
its listed files, writes a concise evidence packet, and returns the patch for
review. A worker does not commit, push, restart Daemon, use live stores, recruit
testers, or make population claims.

The source review is not external validation, a Windows run, an accessibility
audit, an attack exercise, or a live-user session. A green unit test closes only
the named deterministic contract. It does not make a release milestone complete.
The prior GitHub result belongs to commit 85435ed; it is not evidence for later
trees.

Release words used below are deliberately distinct:

| State | Evidence allowed | What it does not establish |
|---|---|---|
| Code drafted | Diff exists | Correctness |
| Focused contract green | Named tests pass on the candidate tree | Whole-system behavior |
| Synthetic assembled probe green | Real turn path, isolated fixture, deterministic receipts | External-user behavior or provider drift |
| Owner-canary observed | Owner runs a reviewed artifact and reports the versioned result | Population evidence |
| E3 development evidence | Consented external labeled set with a reproducible manifest | Held-out generalization |
| E4 held-out evidence | Separately held-out external users and declared sample scope | Universal correctness |
| E5 independent evidence | Independent review or replication | A guarantee of universal correctness |

No milestone below may be promoted by counting code or tests as external
evidence. Stable population claims require the G05 evidence level and scope
described in the source roadmap; no population-wide 1.0 claim follows from this
plan's synthetic matrix.

## Fresh-session start

A new Claude session with no memory of the planning conversation executes
this plan; everything it needs is in this file and the documents it names.

1. **Base.** Before launch the owner commits the planning documents on
   `master`: this plan, the class-guard plan
   (`docs/execution/class_guards/PLAN_20260913_class_guard_completion.md`),
   `docs/GENERALIZATION_CI_REVIEW_20260913.md`, `docs/PROBE_HARNESS_DESIGN.md`,
   and the review's edits to `docs/DEVELOPMENT_WORKFLOW.md`,
   `docs/TEST_LANES.md`, `docs/WORKPLAN_2_hardening_executable_beta.md`,
   `docs/generalization/README.md`, and
   `docs/generalization/08-windows-executable.md`. Both sessions start from
   that commit. If either plan or the review is missing from HEAD, or the
   tree is dirty at start, stop and report.
2. **Checkout.** Work only in `/home/lukeh/daemon_exec/generalization`, a
   plain clone of `/home/lukeh/Daemon_v1` at that commit with its push URL
   disabled; the owner creates it before launch. Never edit, reset, or run
   tests in `/home/lukeh/Daemon_v1`. The class-guard plan runs at the same
   time in `/home/lukeh/daemon_exec/class_guards`.
3. **Roles.** This session is the parent: it owns integration, review, and
   escalation decisions. It delegates one batch at a time to a subagent on a
   cheaper model (Agent tool, for example `model: "sonnet"`), passing that
   batch's prompt built from “Starter cheap-subagent prompt” below. Workers
   run one at a time in this checkout. Do not use `isolation: "worktree"`:
   integrated batches stay uncommitted, so a fresh worktree would not contain
   them. A worker cannot message the parent mid-run; its escalation packet is
   its final report, and the parent continues that worker with SendMessage.
   The parent reviews each complete diff before the next batch starts.
4. **Dirty tree after the first batch.** Once a batch is integrated the tree
   is intentionally dirty. The parent gives each later worker the expected
   changed-file manifest with SHA-256 digests; the worker stops only when
   `git status` differs from that manifest or an assigned file is already
   changed.
5. **Interpreter and frontend.** `python --version` must print 3.11.8. The
   version comes from `/home/lukeh/.python-version`, so it resolves only under
   `/home/lukeh`; elsewhere `python` is system 3.13. There is no project venv:
   use the shared pyenv interpreter read-only and never `pip install`. A
   missing dependency is a stop-and-report. Frontend batches first run
   `npm ci` inside this checkout's `web/`; never touch
   `/home/lukeh/Daemon_v1/web/node_modules`.
6. **What the clone lacks, by design.** Gitignored `data/`, `.env`, and
   `config/config.local.yaml` are absent: no API keys, owner vocabulary, or
   live stores. Relative `./data/...` paths resolve inside the clone; never
   point `CHROMA_PATH` or another path override at `/home/lukeh/Daemon_v1`.
   Git does not clone hooks, so `hooks/pre-push` is not installed.
7. **Test resources.** Wrap every pytest in
   `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M`.
   Run one pytest process at a time across both sessions: if
   `pgrep -af pytest` shows the other session's run, wait. Run only the
   assigned focused tests; do not run the full fast lane or the non-unit
   remainder locally, since CI owns them. Never search or traverse `data/` or
   `.agent_snapshots/`.
8. **Class-guard exchange.** The only class-guard path this session reads is
   `/home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/`,
   read-only. Responses, integration signals, and CI proposals are written in
   this checkout at the paths under “File ownership and parallel-work
   contract”. All relative paths in this plan are relative to this checkout.
9. **Hard limits and deliverable.** No commit, amend, push, pull request,
   remote-setting change, deploy, daemon restart, or call to Daemon's LLM
   providers or paid tool APIs. The owner has adopted pull-request-required
   protection for `master` (class-guard plan, “Owner decisions”), so
   integrated work lands later through a pull request the owner opens. The
   deliverable is the uncommitted tree in this checkout plus evidence under
   `docs/execution/generalization/`; the owner commits there and fetches it
   into the main checkout.

## File ownership and parallel-work contract

The generalization lane owns all application source, tests, configs, scripts,
package specifications, and generalization artifacts except the class-guard
plan's exclusive files. This explicit denylist belongs to the parallel class-
guard plan; do not edit any of it:

- scripts/check_bug_classes.py; scripts/bug_class_guards/**;
  config/bug_class_baseline.json; planned config/bug_class_policy.json and
  config/bug_class_dispositions.json; tests/bug_class_guards/**.
- tests/unit/test_no_git_state_in_tests.py;
  tests/unit/test_ordered_slice_guard.py;
  tests/unit/test_budget_meters_rendered_sections.py;
  tests/unit/test_tool_wiring_parity.py;
  tests/unit/test_model_capability_wiring.py.
- hooks/pre-push; .github/**; docs/BUG_CLASSES.md;
  docs/DEVELOPMENT_WORKFLOW.md; docs/TEST_LANES.md;
  docs/execution/class_guards/**.

The class-guard plan owns scanner policy, detector implementation, class
catalog/baseline edits, and CI/hook changes. This plan owns application fixes
that the class-guard plan requests to close an observed baseline item. The
class-guard owner may adjust the baseline/catalog only after the application
patch and tests are integrated and reviewed. Do not hide these repairs inside an
unrelated batch or edit the other plan's files to make a scan green.

Cross-plan handoff follows the class-guard plan's request contract; there is no
shared or joint status tracker. Requests arrive as immutable packets under
docs/execution/class_guards/requests/, written only by the class-guard owner
and read-only here. Answer each with one immutable response file at
docs/execution/generalization/class_guard_responses/<request-id>.md: request
ID, owning batch ID, source SHA-256 before and after, changed files, the
focused deployed-function test with its exact command and result, and any
limitation. Never edit a request packet or a submitted response; a follow-up is
a new response file with a `-2`, `-3` suffix. A request with no response stays
open, and its gate stays red.

All source and test files belong to at most one active worker. The following
shared seams require sequential ownership, even if the workers have different
task names: core/orchestrator.py; gui/handlers.py; config/app_config.py,
config/schema.py, and config/config.yaml; tests/conftest.py; utils/user_identity.py;
utils/location_resolver.py; web chat/client files; memory/user_profile.py; and
models/model_manager.py. Assign one integration owner for any batch that spans
these seams. The integration owner resolves imports and public contracts after
the isolated behavior batch passes; no parallel agent edits the same files.
Anything involving a shared schema, cache key, dispatcher, receipt format, or
store writer follows the workflow's stop-and-escalate rule.

Work only in the checkout named under “Fresh-session start”. Before each
batch, the parent confirms the base and each worker records:

~~~bash
git status --short
git rev-parse HEAD
git diff --stat
git remote -v
sha256sum <each assigned source and test file>
~~~

Repeat the status, HEAD, diff-stat, and digest check before handoff. If a
checkout's origin points at the active local repository, disable its push URL
before any remote operation; workers do not push at all. Never reset, clean, or
overwrite pre-existing work to obtain a clean start. If scratch checkouts are
unavailable, use the same tree only with a written file lock, no overlapping
agents, and one integration/testing writer. Do not edit the working runtime.

Do not install into the shared interpreter, and do not mutate another
checkout's node_modules, a model cache, a Chroma/profile store, or other user
data. No live Daemon state is used. Disable downloads in test
processes and put any required disposable cache under a unique task root.
Configuration import itself can write: config/app_config.py creates CHROMA_PATH
before its late environment override assignments around line 2027. Set every
path override before importing application modules, then fail closed on any
write outside the sandbox root. DAEMON_TEST_MODE alone is not isolation.
Persist only small sanitized diffs and evidence packets under
docs/execution/generalization/; large test logs must use the workflow's durable
checkpoint location, not /tmp.

## Execution states and batch sizing

Every batch follows this state sequence:

1. **Ready:** parent records base SHA, file ownership, requirement IDs, fixture
   version, expected contract, and exact test command.
2. **Failing proof:** worker adds or runs a test that drives the deployed
   function and observes the defect. A source-shape proxy is not sufficient.
3. **Patch:** worker changes one behavior with its counterexample and avoids
   adding the incident's literal as a phrase exception.
4. **Focused verification:** exact tests pass, ruff/type checks pass where
   relevant, and sibling sites are enumerated using BC-58's Find method.
5. **Review:** parent inspects the complete diff, ownership, privacy boundaries,
   receipts, and test evidence. A failed or unreviewable batch remains open.
6. **Integrated:** parent accepts the patch into the candidate tree and reruns
   the exact tests there. Only then can another batch own a shared file.

Keep each worker's behavior batch near 200–450 changed lines including tests.
Aim below 350 lines when the patch touches a high-risk path. If a coherent
behavior needs more than 450 lines, split at an API seam and require the first
batch to leave no behaviorally active partial migration. Do not bundle unrelated
docs or formatting changes to meet the target. Workers return the patch, command
outputs, changed-file manifest, and unresolved decisions; the parent owns
integration and any commit/restart decision.

## Dependency map and release milestones

| Milestone | Entry condition | Scope | Exit evidence |
|---|---|---|---|
| M0 — isolated execution | Reviewed base and exclusive file map are confirmed | Session setup, F01 security contract, fixture plan | Clean resume manifest and synthetic-only test root |
| M1 — safe current behavior | M0 complete | F01–F07, F03–F04 first, plus F05 disclosure | API boundary, path/timezone, parser, correction, telemetry, and egress tests green |
| M2 — fresh-user mechanisms | M1 complete | F08–F12 and neutral profile fixtures | Empty/Unicode/diverse synthetic profiles, signal-based prominence, resettable learning, category/anchor coverage |
| M3 — memory and interface controls | M1 contracts stable | F13–F14 | Accessible chat baseline and tested storage/deletion writers; any unimplemented derivative remains a release blocker |
| M4 — install and inference route | M1 AppPaths/profile contract stable | F15–F16 | Built React asset smoke, clean-machine Windows evidence, role inventory, honest local failure |
| M5 — deterministic assembled probes | M1 contracts stable; fixture sandbox and cassette boundary approved | Probe schema, runner, Q01–Q15 in slices | Every enabled hard probe has receipts, counterexample, and no-skip execution |
| M6 — product/evaluation claims | M1–M5 and privacy consent controls complete | F17 / G01 / G05 / G11 | Separate owner, synthetic, external-development, and held-out manifests; E3/E4/E5 reported honestly |
| M7 — release qualification | All prior gates and external evidence available | Full G01–G13 exit gates | Clean Windows matrix, offline/security/accessibility evidence, migration/deletion proof, qualified roles, reviewed limitations |

Some work can proceed independently after its prerequisite contract is frozen:
G09 accessibility checks can run beside G04 learning inventory if each owns
different files; the F01 API security core and F08 neutral prompt can also run
in parallel. In the shared-tree fallback, do not parallelize. F02 must precede
packaging's profile path assumptions; F03 precedes F04; G03 egress policy and
G07 storage rules precede a claim that probe sandboxing is safe; F16 role
inventory precedes role migration; G05 evaluation schema precedes E3/E4 claims.
G11 incident practice applies to every wave.

The phases in the older generalization README are strategic time horizons, not
promises that the listed work fits in one coding sprint. Keep the current
release lane scoped to contracts that can be implemented and proven. Model
qualification, external recruitment, longitudinal evidence, signing, and
independent review remain separate external or release milestones.

## First bounded batch: F01 loopback API authorization

Start with the loopback boundary before sharing a build with testers. A CORS
allowlist by itself is not authorization. The batch must define and test the
server-side trust contract across the React SPA, SSE POST, legacy /admin UI,
download endpoints, health endpoint, and internal background work.

Assigned files for the first server batch:

- api/app.py and a new focused api/launch_auth.py (or a similarly small,
  API-owned module after inspecting current seams).
- New tests/unit/test_api_origin_security.py.
- Existing tests/unit/test_api_actions.py and tests/unit/test_api_chat.py may
  be changed only by this batch's single worker if their fixtures need the
  explicit launch secret.

Contract:

1. Generate or receive one high-entropy per-launch secret at process startup;
   tests inject a deterministic secret. Do not persist it in settings or
   ordinary logs.
2. Enforce the expected loopback Host and strict same-origin policy on the
   server for every stateful API route, including simple form POSTs. Require
   the launch secret for reads and writes that can expose personal or diagnostic
   data; state-changing routes also reject missing, stale, or mismatched Origin.
3. Enumerate all mounted routers and route methods. A new route defaults to
   protected. `/health` may expose only a minimal liveness result and remains
   read-only. Static assets are public, contain no personal data, and cannot
   invoke actions.
4. The bundled React stream is a POST using fetch-event-source; custom auth
   headers are supported. The client batch must attach auth to ordinary fetch,
   SSE, upload, action, and download calls. Do not weaken the server for a UI
   path that cannot attach the contract.
5. `/admin` is development-only unless its transport is shown to carry the same
   authorization. Packaged mode must not expose an unauthenticated Gradio
   mutation surface. Internal background tasks call service functions directly
   and do not create an unauthenticated HTTP bypass.
6. Test the deployed FastAPI app with fake action executors: rejected origins
   make zero executor calls; a correct same-origin authorized request executes
   exactly once; altered Host, missing token, stale token, form-encoded POST,
   websocket/stream route, and disallowed scheme fail closed. Test health
   disclosure separately.

If the current startup surface has no safe way to pass a per-launch token to
the bundled page, record the exact launch/UI seam and escalate that contract
blocker to the parent. Do not replace it with wildcard CORS, a tokenless path,
or a blanket exception. The server and web transport sub-batches must be
sequential at their shared API contract boundary; they may use separate test
files and must never edit the same file at once.

Acceptance: deployed-route tests cover approved and rejected origins, Host,
token, simple POST, SSE, uploads/downloads, API debug, action approval, note
sync, /admin packaging policy, health, and background service calls. At least
one test drives a rejected approval and proves the fake executor was not called.
This is a source/test closure, not a live hostile-site exercise.

## Bounded application batches

The file lists below are starting assignments. A worker may add one new
submodule only when its parent approves the seam. Do not grab a file from a
neighboring batch to make the patch convenient. Test additions live in new
or listed test files; no two workers write one test file simultaneously.

### P1 current gaps

| Batch | Finding / contract | File ownership and meaningful tests | Dependencies and acceptance |
|---|---|---|---|
| A01 API origin/Host/launch auth | F01 / G06-T02, A03–A04 | api/app.py, api/launch_auth.py, tests/unit/test_api_origin_security.py; targeted tests/unit/test_api_actions.py and tests/unit/test_api_chat.py | First batch above. Server rejects unauthorized SPA/admin/simple POST routes before route code and fake executors. |
| A02 UI transport authorization | F01 / G06-T02 | web/src/api/client.ts, web/src/api/useChatStream.ts, new web/src/api/client.security.test.ts and existing web/src/api/useChatStream.test.ts | After A01 token contract is frozen. Cover every direct fetch, upload, native download, model/settings/action/curation call, SSE reconnect and abort. `cd web && npm run typecheck && npm test -- --run`. |
| A03 profile path and timezone | F02 / G02-T03, G08-T08 | One worker owns utils/user_identity.py, utils/location_resolver.py, utils/institution_resolver.py, utils/timezone_resolver.py, memory/user_profile.py and only the necessary path bootstrap/config files; tests/unit/test_user_identity.py, test_location_resolver.py, test_institution_resolver.py, test_timezone_resolver.py, test_calendar_create.py | Use the existing bootstrap/config authority including USER_PROFILE_PATH and the frozen UserProfile path behavior. Do not invent a second AppPaths abstraction or circular app import. Preserve existing data in this compatibility batch; migration later is dry-run, backed up, idempotent. Simulate Windows timezone mapping and pass a non-Central IANA zone through calendar creation to an aware executor payload. If profile and OS zone are both unknown, ask before scheduling; never silently choose Central or UTC. |
| A04 strict verifier JSON | F03 / G12 strict boolean contract | core/grounding_check.py, tests/unit/test_grounding_check.py, tests/unit/test_grounding_wiring.py only as needed | Reject missing required fields, string/number/null booleans, malformed objects, and out-of-range fields as abstention. Controls: actual false, actual true substantive error, and valid no-error case. Never use truthiness coercion. |
| A05 atomic correction delivery | F04 / G12 acceptance 3–6 | Sole integration worker owns gui/handlers.py, core/grounding_check.py, tests/unit/test_grounding_wiring.py, tests/unit/test_grounding_log_only.py and an outcome test in tests/unit/test_grounding_delivery_atomicity.py | Runs only after A04; no overlapping worker edits these shared files. In correct mode, a flagged draft cannot be streamed as settled final text before review. Either buffer the final response until review or define/test explicit provisional replacement semantics with an accessible client. On revision failure/timeout/disabled integration, emit one integrated fallback, never append a suffix to a flawed draft. Assert display_text == stored_text == indexed_text == final_text and no draft/verdict enters retrieval. Preserve log-only as the default. |
| A06 hosted disclosure | F05 / G01-T01/02, G03-T05 | docs/WORKPLAN_2_hardening_executable_beta.md, product privacy/capability UI source chosen after locating its owner, and its focused test; new sanitized artifact under docs/execution/generalization/ | Verify the corrected workplan wording in the review is still present. Implement visible HOSTED_TRANSITION disclosure by inference role and connected-tool destination before an operation. Do not claim content stays local while hosted keys are required. Connected egress text must identify data leaving and destination. |
| A07 privacy-safe telemetry/logging | F06 / G03-T09, G11 | utils/turn_telemetry.py, tests/unit/test_turn_telemetry.py, tests/unit/test_privacy_redaction.py; core/orchestrator.py only in a later sole-owner integration batch | Exportable metrics use a fixed allowlist, no query, response plan, retrieved content, names, or free text. Preserve intentional owner-local debug in memory-allowed development mode, with access and retention explicitly local. Memory-off/session-only must not duplicate denied turn content into logs/backups. Seed private canaries and assert absent from normal telemetry and any export bundle; distinguish local debug from exported metrics. |
| A08 optional location and egress | F07 / G03-T05/06, A09–A13 | utils/location_resolver.py and tests/unit/test_location_resolver.py in a separate sequential batch from A03; location settings owner and utils/web_search_trigger.py only after the policy contract is stable | IP lookup defaults off unless disclosed and explicitly enabled. OFFLINE blocks resolver network calls and suppresses location in web prompt; LOCAL_CONNECTED uses explicit user location and minimal query data. Test no network with socket/HTTP stubs, background refresh, cache, query justification, and positive opted-in location. Test mode alone is not offline qualification. |
| A09 neutral empty-profile prompt | F08 / G02-T04, G09-T05 | config/prompts/default_personality.txt, a prompt-render test in tests/test_orchestrator_profile_injection.py or a new isolated test, plus prompt owner only if required | Replace assumed intimacy and gender with neutral defaults. Relationship, pronoun, warmth, directness, and proactive behavior come from explicit preferences. Test no name/pronouns/profile, opt-in close style, and preference combinations; do not edit core/orchestrator.py concurrently with another batch. |

### P2 fresh-user and learning gaps

| Batch | Finding / contract | File ownership and meaningful tests | Dependencies and acceptance |
|---|---|---|---|
| B01 Unicode display names | F09 / G02-A09 | utils/user_identity.py and tests/unit/test_user_identity.py, tests/unit/test_sep12_followup_identity_names.py | After A03 releases the resolver files. Cover accented Latin, non-Latin, combining marks, apostrophe/hyphen, multiword names, empty name, malicious/control characters, and end-to-end identity resolution. Normalize safely without rejecting valid scripts or treating a test fixture as the only proof. |
| B02 signal-based profile prominence | F10 / G02-T05, A10–A11; G04-T06 | memory/user_profile.py; tests/unit/test_user_profile.py, tests/unit/test_profile_confirmation_recurrent.py, tests/unit/test_user_profile_schema.py | After A03 and with sole ownership of memory/user_profile.py. Score confirmation, durability, recurrence, currentness, relevance, and explicit pin. Add demotion and explanation, user-defined relation, and varied-profile tests. Sensitive inferred facts require confirmation and cannot enter always-visible context. No relation-list or phrase append. |
| B03 learning registry and reset | F11 / G04-T01/03/04/05/08/10 | Start with read-only inventory artifact docs/execution/generalization/adaptive_inventory.md; then one worker owns utils/adaptive_exemplars.py and tests/unit/test_adaptive_exemplars.py plus a distinct reset contract test. | Inventory all adaptive stores/thresholds before mutating. Add provenance, evidence class/count, source-event IDs, bounds, decay/currentness, explanation, version, quarantine, and reset-to-neutral. Do not let assistant output, one session, or retrieved external content confirm its own learning. Reset proves immediate prompt/decision effect. Keep threshold changes report-only until sparse/mature corpus evaluation and rollback evidence exist. |
| B04 categorized task coverage and organization anchors | F12 / G04-T07, G09-T07 | Separate sequential owners for core/agentic/gate.py tests/unit/test_agentic_gate.py and utils/web_search_trigger.py tests/test_web_search_trigger.py, tests/unit/test_location_resolver.py only for anchor propagation | Enumerate consumers and preserve general categories/context anchors through gate, query rewrite, private-sphere guard, and prompt generation. Add school, employer/org, care, caregiving, household, hobby, and public-work examples with non-trigger controls. Measure fallback/tool behavior. Never close a miss by appending another specific phrase. |

### Release, interface, and evidence gaps

| Batch | Finding / contract | File ownership and meaningful tests | Dependencies and acceptance |
|---|---|---|---|
| C01 accessible chat semantics | F13 / G09-T01/02/04 | web/src/components/chat/MessageList.tsx, ChatInput.tsx, relevant component tests, and one new accessible chat test file | Add accessible names to icon-only actions; semantic status/live regions; buffer announcements by meaningful message rather than token; preserve keyboard focus; honor reduced motion. Use React Testing Library and accessible-role queries. `cd web && npm run typecheck && npm test -- --run`. Browser/NVDA evidence remains manual and external. |
| C02 storage-writer and derivative inventory | F14 / G03-T01, G07-T01, G11 | New inventory under docs/execution/generalization/ with a single read-only inventory owner; no runtime writers changed in this batch | Enumerate conversation, profile, vector, graph, summaries, reflections, adaptive records, telemetry, logs, uploads, caches, backups, indexes, generated notes, shutdown jobs, and background writers with lineage and deletion owner. Explicitly distinguish correction/undo, hide, forget, and delete. |
| C03 memory policy before storage | F14 / G07-T02/03 | One worker owns the policy schema/entry point and tests. Next batches own disjoint writer adapters; do not let parallel workers change config/schema or tests/conftest.py. | Implement off, session_only, ask_to_store, automatic_bounded, custom modes before write. Denied turns leave no personal corpus/vector/graph/summary/adaptive/log/backup record. Local owner debug is retained only in memory-allowed development mode, subject to A07. Test foreground, background, shutdown, crash/restart, and policy changes. |
| C04 deletion cascade and export/backup scope | F14 / G07-T04–09; G03-T10 | First worker owns deletion ledger/registry and a disposable canary contract. Then each storage adapter has one named worker/file lock. Tests live in new tests/unit/test_memory_policy_deletion.py plus existing store-specific tests. | Seed a unique canary through every derivative, delete source, verify all stores/caches/indexes/logs/backups, preserve independently supported facts, test idempotence/recovery, and disclose retained immutable backups. Deletion is blocked from release claims until every registered writer has a tested adapter. |
| C05 React package and Windows clean install | F15 / G08-T02/08 | daemon.spec, web build metadata only as needed, tests/unit/test_packaged_frontend_assets.py (new), docs/execution/generalization/windows_matrix.md | `cd web && npm ci && npm run build`; then verify web/dist is collected and served in a clean scratch package smoke with no checkout dependency. Do not edit .github/** or installer CI. Windows 11 clean VM evidence is a later gate: no Python/Node/Git/admin assumption, fresh profile, restart, offline, upgrade, rollback, export/delete, and uninstall. |
| C06 inference role inventory and honest failure | F16 / G03-T07, G10-T01/02/03/07/08/13 | First artifact docs/execution/generalization/inference_roles.yaml and tests for completeness; then migrate one role per batch through models/model_manager.py owned by one integration worker. Add tests/test_model_manager_methods.py only with that role batch. | Inventory every generate/call site, role, input data class, model/provider, fallback, structured output, offline behavior, and latency budget. Gateway selects roles, not provider brands. If local role is missing/unqualified/fails, return explicit capability-unavailable with no hosted retry. Never rewrite forty call sites at once. Each role promotion requires G05 report, role-specific fixtures, network-denied run, rollback, and no partial persistence/action on malformed output. |
| C07 product/evaluation evidence registry | F17 / G01-T05, G05-T01–04/10/11, G11-T12 | eval/corpus.py, eval/harness.py, focused tests/test_eval/test_corpus.py, test_harness.py, plus new manifest validation tests | Add case ID/version, requirement IDs, task/risk, consent/privacy status, evidence level, cohort/subgroup scope, identity split, prompt/model/runtime/tool-schema/config versions, expected receipt, count/uncertainty, and retention. Physically/logically separate owner, synthetic, external-development, and held-out data. External labels require consent/revocation. Do not add real participant data in this worktree. |

## Probe layer: schema, sandbox, stubs, receipts, and Q01–Q15

The third quality prong is an assembled-turn behavioral harness, distinct from
the workflow's three collaborator prongs and from unit/static gates. The review's
Q cases are proposed synthetic fixtures, not completed probes. Use the actual
turn path and stub only provider/tool boundaries. Cassette misses fail and
cannot fall through to a paid service. Live/provider drift belongs to a
separately budgeted, report-only lane.

### Required case record

Each YAML or JSON case is versioned and validated against one schema. Required
fields:

~~~yaml
id: Q01-coursework-number-na
version: 1
requirement_ids: [G02, G04, G06, G09]
bug_classes: [BC-01, BC-28]
evidence_level: synthetic
fixture:
  seed_id: coursework_session_v1
  fixed_clock: 2026-09-13T15:00:00Z
  timezone: America/Denver
  user_profile_version: 1
input:
  text: "Why does the number column contain NA?"
  surface_forms: [clean, wrapped_indented]
provider:
  cassette_id: synthetic-provider-fixture-v1
  roles: [intent, response]
  prompt_hash: required-with-model-settings-and-tool-schema-hash
  miss: fail
tools:
  stubs: []
expect:
  receipts: {}
  response_shape: {}
  required_writes: []
  forbidden_effects: []
  counterexamples: []
~~~

The real schema must require expected receipts and counterexamples where the
case uses those features; the abbreviated example above is not permission to
omit assertions. Every case has: synthetic seed state; frozen clock/timezone
where applicable; clean and client-wrapped/indented form; provider role and
cassette/version metadata; tool stubs; required outcome/execution/write
receipts; forbidden side effects; paired positive or non-triggering
counterexample; and report/gate classification.

Prompt hashes identify but do not uniquely define cassettes. Bind cassettes to
model ID/version, role, generation settings, system/prompt-section hashes,
tool-schema version, fixture version, and parser version. Prompt or schema
changes make the cassette stale and require a reviewed fixture update. Never
auto-record a new cassette from an unreviewed run. Exclude private text, tokens,
real paths, and owner data.

### Sandbox construction

Create one disposable instance with explicit roots for corpus, profile,
vector/Chroma, graph, uploads, action proposals, logs, telemetry, learning,
backups, model cache, and temporary files. Seed it from synthetic fixtures.
Scrub provider and integration credentials. Deny outbound networking by default;
provider and tool stubs are explicit in-process boundaries. Freeze clock,
timezone, random seed, and feature settings.

Set all root/path overrides before importing application modules. In particular,
guard the import-time CHROMA_PATH mkdir in config/app_config.py; do not rely on
the late overrides around line 2027 or DAEMON_TEST_MODE. In a subprocess, trap
filesystem writes outside the sandbox root and fail on them. Also assert the
resolved path map has no workspace/user-home/private-store paths. Exercise
shutdown and background writers after the turn, then scan all roots for canary
strings and unexpected files. If isolation cannot be proven, do not run the
assembled turn; repair the fixture boundary first.

The turn runner drives the real in-process turn endpoint/handler that exercises
production orchestration; HTTP/SSE is added only for cases whose contract is
the API/stream itself. Provider generation is cassette-backed; missing cassette
or undeclared role is a hard test failure with zero network requests. Tool
executors return deterministic receipts and record call count/arguments. Tests
assert receipts/outcomes and write lineage, not exact model wording. Shape
predicates (forbidden markers, length bands, no contradictory suffix) are
report-only until false-positive rates are reviewed. Hard invariants such as
unauthorized action, forbidden egress, incorrect/missing receipt, cross-profile
write, missing required case, empty probe collection, or unregistered role gate.

Initial tests belong in tests/probes/test_schema.py,
tests/probes/test_sandbox.py, and tests/probes/test_runner.py. Keep probe
fixtures under tests/probes/cases/ with named seed fixtures under
tests/probes/seeds/. Do not edit tests/conftest.py for probe bootstrap until a
sole integration owner takes it; isolated bootstrap is preferred.

### Case backlog, grouping, and acceptance

| Case | Seed and critical invariant | Required receipts and counterexample |
|---|---|---|
| Q01 | Coursework; “Why does the number column contain NA?” | No distress floor from number. Paired real distress turn still routes appropriately; assert tone and escalation receipt. |
| Q02 | Deadline warning, repeated wrapped/indented | No dead/war substring false positive; true-word and inflection positive control. |
| Q03 | News, zero web credits; “Please verify that public announcement.” | Budget receipt consistent, no dispatch, honest unmet-evidence result. Positive-budget control searches within cap. |
| Q04 | Assistant history mentioned doctor; user says “I have a new doc to review.” | User-authored referent wins; no invented doctor in STM, key points, strategy, or response. Run clean and wrapped form. |
| Q05 | Stale offer then “Add a meeting tomorrow at 2 PM for 30 minutes.” | New explicit request sets proposal/card; valid short affirmation accepts a current offer control. |
| Q06 | Q05 approval, frozen clock, non-Central timezone | Fake calendar executor receives valid aware times exactly once; matching receipt; malformed clock/card fails. |
| Q07 | Duplicate-content seed; explicit “Save a note titled Reading Plan…” | Saved/updated receipt or honest refusal. Autonomous dedup cannot claim success; assert no false completion receipt. |
| Q08 | “Is the event already on my calendar?” prior assistant claim only | No execution receipt means no completion claim; paired actual matching receipt permits supported completion. |
| Q09 | Document uploaded five days before fixed clock | Correct document identity, no “today” assertion; fresh-upload positive control. |
| Q10 | Temporary repository with newer commit than seeded index | Fresh local history reaches final context within budget; historical-document query preserves intended retrieval. |
| Q11 | Same process disables web then asks latest forecast | Gate and dispatch both see disabled state; honest unavailable result; re-enable control proves restoration. |
| Q12 | “What notes do I have about the project?” with injected retrieval failure | Failure is distinct from empty store; no unsupported “no notes exist.” Pair with truly empty store. |
| Q13 | Supported finding; verifier cassette emits string false | Invalid verdict never becomes positive correction; true JSON booleans and substantive true error controls. |
| Q14 | Materially wrong draft, forced integrator timeout | Final display/storage/index receipt match; no contradictory suffix or visible failed draft. Include successful revision and log-only controls. |
| Q15 | Split reasoning markers, empty/reasoning-only output, provider error | No raw protocol artifacts; explicit error/empty result; clean stream preserved. Assert emitted chunks and final receipt. |

Add non-query contracts in the first probe releases for F01 API
authorization, F06 seeded telemetry canaries, F14 denied storage/deletion, and
F15 packaged React asset smoke. Queries alone do not close these requirements.

Group implementation to preserve review size: foundation/schema/sandbox first;
Q01–Q05 next; Q06–Q10 next; Q11–Q15 last. Each group receives new test files
and a distinct fixture file set; only one runner integration owner edits the
runner. No case is called passing until it runs against the actual candidate
turn path with required fields present. Missing case IDs and empty collection
are failures, not skips. Keep provider-drift probes separately budgeted and
report-only until a human approves a rate limit and run window.

## CI, local checks, and patch proposals

The generalization lane must deliver exact CI proposals as reviewable artifacts
under docs/execution/generalization/. It must not edit .github/workflows,
hooks/pre-push, TEST_LANES.md, or other class-guard-owned files. The class-guard
owner alone applies workflow changes. Proposed deterministic probe command:

~~~bash
DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 180s python -m pytest -q \
-p no:cacheprovider tests/probes/
~~~

Only enable this command after schema, sandbox and receipt tests exist and
resource use is measured. The CI proposal must have no path filter, require a
non-empty expected probe set, verify every expected case is collected, fail on
unregistered roles or cassette misses, record outcome receipts as artifacts
without private text, and use an explicit resource timeout. Schedule a nightly
run only after full-suite duration and memory are measured. No code-complete
claim until the proposed CI check has actually run on the integrated candidate.

Focused Python checks use the repo's bounded environment:

~~~bash
DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
<exact assigned test files>
~~~

Run Python lint on changed files, then ruff check . on the reviewed integration
candidate. Frontend batches run from web with npm run typecheck, npm test --
--run, and npm run build where packaging is touched. The ordinary full fast lane
is python -m pytest -q -m "not slow and not benchmark and not semantic" --tb=short.
The non-unit remainder must not run beside the live Daemon; the documented
8 GiB / 512 MiB-swap durable-output wrapper applies. Do not amend CI or hook
caps in this plan.

Each patch-proposal artifact records: intended workflow job, exact command,
required environment, collection check, timeout/resource measurement, outputs
to upload, redaction proof, and a negative control showing that a missing case,
empty collection, forbidden egress, or missing receipt goes red. The artifact
is a proposal until the class-guard owner merges it and branch protection
requires it; current master protection is disabled.

## Routine failure contingencies

Workers use these decisions without parent consultation. Escalate only when a
listed contingency fails or the decision changes a hard privacy/security/data
contract.

| If this happens | Worker action |
|---|---|
| Focused test fails before edits for a known unrelated reason | Capture exact failure and base SHA; run only the assigned tests; do not repair unrelated code in this batch. Report the blocking dependency with one command and minimal evidence. |
| Existing tests encode the defect (suffix output, Central default, ASCII regex) | Replace the obsolete expectation in the same owned test only after adding a deployed-function outcome assertion and paired non-triggering control. Do not preserve behavior to keep a green test. |
| Test needs a new shared schema or config contract | Stop before schema/config edits; bundle the proposed fields, current readers/writers, migration compatibility, and tests. Parent assigns one integration owner. |
| A platform API is unavailable on Linux | Unit-test with a Windows registry/timezone/path stub and add a Windows CI/manual evidence row; mark platform acceptance pending. Do not infer Windows success. |
| Profile path migration would move or overwrite existing data | Keep compatibility read path, write a dry-run report/backup/rollback design, and defer migration. Never copy or rewrite real user data in tests. |
| Verifier JSON is malformed or wrong-typed | Treat it as abstain/invalid; leave response unchanged only in log-only mode. In correct mode use the integrated fallback, never stringify or coerce. |
| Integration timeout/error or revised text fails contract checks | Use the safe integrated fallback and record a non-success receipt. Do not return the bad draft plus suffix. Preserve the old index until atomic final commit. |
| A location/tool request is disabled or offline | Return honest unavailable/needs-consent state, emit no request, and keep local profile location optional. Never enable IP lookup as a fallback. |
| Learning corpus is too sparse or evaluation disagrees | Retain the population baseline and current thresholds, mark calibration report-only, gather evidence. No threshold promotion by intuition. |
| A phrase/control fails in one domain or subgroup | Inspect all consumers, add category/context anchoring or a learned per-user channel, and keep paired non-triggering cases. Do not append a phrase regex. |
| Delete cascade discovers an unregistered writer/backup | Fail deletion acceptance and inventory the writer. No claim that source-row deletion equals delete-all. |
| React semantics pass unit tests but depend on browser timing/AT behavior | Keep static/browser tests scoped and mark NVDA/keyboard/zoom external evidence pending. Do not claim accessibility audit complete. |
| Packaged build cannot find web/dist or starts only from checkout | Fix collection/serving and add clean temp-directory smoke. If launcher/installer signing is missing, mark Windows release gate pending. |
| Local inference role has no qualified model or returns an error | Return explicit unavailable result, no hosted fallback, and no partial store/action. Keep role disabled pending G05 evidence. |
| Probe cassette key changes after a prompt/tool schema change | Mark cassette stale and request a reviewed fixture recapture against synthetic state. Do not auto-record on test failure. |
| Memory/path sandbox sees a write outside its root | Stop the run, preserve the path and stack, fix isolation before rerunning. Do not delete the outside file or assume DAEMON_TEST_MODE protected it. |
| Test resource use approaches the measured bound | Split the batch or reduce fixture size without weakening assertions; record peak and elapsed time. Do not run the non-unit suite with Daemon active. |
| Generalization scan exposes a baseline finding in a file the batch edits | Record a pending integration signal at docs/execution/generalization/class_guard_signals/<batch-id>.md: scanner, path, symbol, source SHA-256, and the reproduction command. Do not fix it outside a request, rebaseline, or edit the catalog; the class-guard owner decides whether to issue a request. |

## Parent escalation packet

Only escalate a hard contract blocker, unexpected private/state effect, schema
or writer ambiguity, or failed contingency. Bundle one concise packet:

1. Requirement/batch ID and candidate SHA.
2. Exact files read/changed and current file digests.
3. The failed test or contract, with one bounded command and the relevant output.
4. Two feasible options, their compatibility/privacy/data effects, and the
   worker's recommendation.
5. Whether state was written, where, and how isolation was checked.
6. The smallest next decision the parent must make.

Do not escalate ordinary import errors, test fixture edits within the assigned
files, or a known platform check that has a prescribed stub path. Never send
external messages or private data. A worker cannot message the parent
mid-run: it stops and returns this packet as its final report.

## Evidence packet and stop conditions

At the end of every batch, save a sanitized packet under
docs/execution/generalization/batches/<batch-id>.md with:

- Base and candidate HEAD; status and changed-file manifest; file digests.
- Requirement and defect IDs, invariant, and exact reason the source seam owns
  the fix.
- Failing-before evidence when reproducible on the reviewed base.
- Exact commands, exit codes, pass/skip/fail counts, elapsed time, peak memory,
  and relevant receipt summaries.
- Counterexample results, BC-58 sibling-site search, and any exemptions.
- Privacy scan result and confirmation that no live store, network credential,
  or provider was used.
- Remaining milestone state: code drafted, deterministic contract, synthetic
  assembled, owner canary, E3/E4/E5, or pending external evidence.

Immediately stop the batch and return the packet if ownership is already dirty,
the parent changed the base during work, a shared file has an active writer, a
test would touch live stores, a stub falls through to a provider, a test writes
outside its sandbox, a contract would weaken privacy/security/deletion, or a
change requires irreversible data migration. No worker commits, pushes,
restarts the app, updates model caches, changes real settings, or writes live
data.

## G01–G13 completion register

This execution plan maps findings to roadmap requirements; it does not replace
the source requirements. Track each acceptance ID with one of: open, code,
focused, synthetic, owner, E3, E4, E5, blocked-by-evidence. A requirement with
external/manual acceptance cannot be marked complete by unit tests.

| Roadmap | Plan coverage and remaining release proof |
|---|---|
| G01 Product contract | A06/C07 establish hosted disclosure, machine-readable claim/evidence lineage, and wording checks. Hardware tiers, lifecycle/rollback, UX comprehension, and claim expiry remain release work. |
| G02 Owner-neutral runtime | A03/A09/B01/B02 plus matrix fixtures cover configured profile source, neutral prompts, Unicode names, identity isolation, currentness and pinning. Full UserContext/AppPaths threading and all direct identity reads remain a staged migration; do not invent a second path authority. |
| G03 Private data/egress | A06–A08, A07, C02–C06, probe sandbox cover disclosure, allowed location, logs, writers, and gateway roles. Central broker, secret protection, unknown-destination audit, agent isolation, delete/uninstall are not complete until their own acceptance tests pass. |
| G04 Adaptive personalization | B02/B03/B04 cover signals, evidence, reset, bounds, poisoning, anchors, and conservative calibration. Store inventory, user-facing explanation/usability, rollback/migration, and mature-corpus calibration remain. |
| G05 Population evaluation | C07 plus synthetic probes create reproducible manifests and keep splits separate. E3–E5 recruitment, uncertainty, subgroup/worst-group results, 60–90 day evidence, and qualified model reports are external gates. |
| G06 Safety/security | A01/A02, Q06/Q08, and sandbox contracts cover F01, authorization receipts, non-replay, and no-egress test path. Threat model, prompt injection, memory poison, SSRF/path traversal, supply chain, and independent review remain. |
| G07 User memory control | C02–C04 and Q07/Q12/Q14 cover inventory, pre-write policy, provenance, correction, deletion and backups. Full user ledger, accessible controls, exports/restores, and observed user comprehension remain. |
| G08 Windows executable | A03/C05 plus F15 smoke define authoritative paths and bundled React assets. Signed launcher, clean standard-user Windows install, crash recovery, migration, update/rollback, offline pack, uninstall and resource matrix remain. |
| G09 Accessibility/communication | A09/B04/C01 and Q01/Q02/Q04 establish neutral style, language counterexamples and stream semantics. Full keyboard, 200% zoom, contrast, NVDA, reduced motion, onboarding/recovery, and external AT review remain. |
| G10 Local model migration | C06 inventory and role-by-role honest failure establish the gateway contract. Managed runtime, artifact signatures/licenses, hardware qualification, full role migration, prompt budgeting, scheduling, and zero hosted runtime are later gates. |
| G11 Dogfooding/incidents | Evidence packet, class-guard request responses and integration signals, synthetic fixtures, privacy scans, and C07 separation implement the bounded repair loop. User feedback UX, private incident ledger, sanitized export review, owner canary and long-lived external evidence remain. |
| G12 Response integrity | A04/A05 and Q13–Q15 enforce strict verdict parsing and atomic final answer across stream/storage/index. Keep log-only default until correction precision evidence supports an explicit mode change. |
| G13 Fable handoff / historical work | Treat docs/generalization/13-fable-midflight-handoff.md as historical status with its later corrections. Current production has graph filtering call sites and an integrator; do not revive the old “no caller” or “suffix only” claims. Current G12 fallback failure remains F04. Require resume manifest and checkpoint review before any old midflight task resumes. |

## Starter cheap-subagent prompt

Use this exact prompt for the first implementation worker after the parent
confirms the base and scratch checkout. It is intentionally bounded to A01:

~~~text
You own batch A01 only: F01 loopback API origin/Host/launch-token authorization.
Read docs/PLAN_20260913_generalization_execution.md, the F01 row, the first
bounded batch contract, and docs/generalization/06-safety-and-security.md
G06-T02. Do not edit any file outside A01's assigned list:
api/app.py, a small new api/launch_auth.py, tests/unit/test_api_origin_security.py,
and only the fixtures in tests/unit/test_api_actions.py or
tests/unit/test_api_chat.py if required. Do not edit web files, .github,
hooks, TEST_LANES, BUG_CLASSES, any class-guard files, app_config/schema/YAML,
or data/.

Before editing, report read-only git status, HEAD, diff stat, assigned-file
digests, active owners, and remote URLs. If the tree is dirty, HEAD changes, or
one assigned file has another writer, stop and report. Never reset/clean. Do not
commit or push. If origin points at the active local repository, disable its
push URL; do not contact a remote.

Add deployed-FastAPI tests with fake executors. Prove strict loopback Host,
same-origin, launch-token authorization, simple form POST rejection, and that
unauthorized approval calls the executor zero times. Enumerate every route and
method, including React SSE POST, downloads/debug, uploads, action approval,
note sync, /admin, health and background service calls. Protect every personal
or mutating route by default. Keep /health minimal and read-only. Packaged
/admin must not be an unauthenticated Gradio mutation surface. Internal
background tasks call services directly; do not add an HTTP bypass.

If the existing launcher cannot securely provide a per-launch secret to the
bundled page, stop and send the parent the smallest contract blocker with
two options. Do not weaken to wildcard CORS or tokenless exceptions.

Run only the assigned API tests with the documented offline/sandbox env, then
ruff on changed Python files. Before handoff, repeat status/HEAD/diff/digests;
run BC-58 sibling-site search; save a concise evidence packet under
docs/execution/generalization/batches/A01.md. Include exact command, counts,
failing-before result if available, fake-executor call counts, limitations,
and proof no network/live store was used. Return the patch for parent review.
~~~

## Follow-on scheduling

After A01 review, schedule A02 against its frozen token/header contract, then
A03 path/timezone, A04 strict parser, A05 atomic delivery, A06 disclosure,
A07 telemetry, and A08 location policy. A03's path contract unblocks the
synthetic sandbox and packaged smoke. A04 must pass before A05; A05 must pass
before Q13–Q15 can gate. After the P1 lane, run A09 and B01/B02/B03/B04 as
separate ownership windows. C01 can proceed in parallel only while its React
files are disjoint from A02 and the F04 streaming decision is frozen. C02's
inventory precedes C03/C04. C06's role inventory precedes any local role
migration. C07 precedes any population claim.

The probe work starts with schema, write-boundary sandbox and cassette fail-closed
tests; then Q01–Q05, Q06–Q10, and Q11–Q15 in the groups above. The exact CI
proposal is written only after collection, timing and memory are measured. Any
probe red from an expected outcome blocks the next batch until understood; a
newly observed issue becomes a sanitized incident with a counterexample and a
new bounded batch. Do not retroactively loosen expected receipts to obtain a
green run.

Before external testers, require A01/A02, A03 compatibility, A06/A07/A08
privacy disclosures, C03/C04 storage and deletion previews, C05 clean package
smoke, G06 threat triage, and a parent-approved privacy review. Before calling
the product locally complete, require every runtime inference role to pass
C06's role contract with credentials empty and networking denied. Before a
stable population claim, require G05 E4 or E5 evidence and the G01 claims
review. Code completion, a green synthetic probe suite, and owner-canary
success remain separate facts.

## Ready-to-paste parent brief

~~~text
Execute docs/PLAN_20260913_generalization_execution.md as the parent session.
Read it fully first, then docs/GENERALIZATION_CI_REVIEW_20260913.md,
docs/DEVELOPMENT_WORKFLOW.md §3/§3a, docs/BUG_CLASSES.md, and the
docs/generalization/ documents each batch names. Follow “Fresh-session start”
exactly: stay in /home/lukeh/daemon_exec/generalization, confirm python 3.11.8
and a clean tree at the committed base, and never touch the main checkout or
any class-guard-owned path. Run batches in the order under “Follow-on
scheduling”, starting with A01: one cheaper subagent per batch, sequential,
each given its assigned-file list and a prompt built from the starter
template. Review each diff, rerun its exact tests under the memory cap, and
save its evidence packet before starting the next batch. Answer class-guard
requests only through response files; record scan findings only as
integration signals. Stop on the plan's stop conditions. No commits, pushes,
pull requests, remote settings, restarts, or calls to Daemon's LLM providers.
Report per batch: changed files, commands and counts, open escalations, open
class-guard requests, and limitations.
~~~
