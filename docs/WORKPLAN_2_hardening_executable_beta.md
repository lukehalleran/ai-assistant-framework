# Workplan 2 — hardening → executable → five-friend beta

_Written 2026-09-08 (Fable) from the owner's direction: the current priority is
completeness and absence of bugs, not features; a fresh executable; a way to
hand Daemon to ~5 friends and use their data to improve it, under a hard
constraint that the owner **cannot** see the testers' chats. Constraints on the
owner's side: internship applications start ~2026-09-22, a 3-month leave just
ended, one easy course plus an incomplete. So: small verified batches, nothing
that needs a week of uninterrupted attention._

Workplan 1 = the 2026-09-08 handoff batches (B1–B6, correctness). Workplan 2
starts when those ship and runs in the same loop (`docs/DEVELOPMENT_WORKFLOW.md`).

## Phase F — retired 2026-10-08

_The career-fair cut (inserted 2026-09-21, fair ~2026-09-29) is past; its
detailed body was removed rather than left as a stale plan - it was only ever an
uncommitted draft, so it is not in git history; the 09-21 handoff with the
Docker/exe findings is
`~/daemon_exec/followups_0921_runs/HANDOFF_20260921_employer_ready.md`)._

Outcomes: **A-D done** (public-surface baseline and repo credibility, fresh-clone
smoke in CI - `tests/smoke/test_fresh_clone_contract.py`, required job in
`.github/workflows/tests.yml:87-105` - and the generic-config audit; the
baseline/credibility/audit items are per the owner's 09-30 lane records, not
re-verified here). **E partial**: `daemon.spec` bundles `web/dist` (`19c313b`,
2026-10-02) and paths are frozen-aware; the Linux build and binary smoke are
still pending (P3, Phase 2 item 1). **F**: Docker rebuilt statically, never
built. **G**: README screenshots and a React-UI demo re-record not started.

Carried forward: Docker build + README "Docker Ready" badge only after the
acceptance in Phase 2 item 4; README screenshots/link pass is a pre-invite item
in Phase 3 (item 5); the talking-points prep is dropped.

## Phase 0 — finish what is in flight (this week)

- Ship B1–B6, one commit per batch, restart, live probes. Owner applies the
  pending `data/profile_junk_candidates_20260908.txt`.
- Turn on the nightly full suite (workflow §7.1) as a `systemd --user` timer
  with the memory-capped three-batch procedure; a red night blocks the next
  batch. Without it, "no bugs" is unmeasurable.

## Phase 1 — completeness and bug-class closure (1–2 weeks, interleaved)

"Complete" means: every feature a fresh clone advertises works with the
generic committed config, and every feature that is OFF is hidden from the UI
rather than half-present.

1. **Fresh-clone smoke in CI** (`tests/smoke/test_fresh_clone.py`, one job):
   no `config.local.yaml`, no `data/`, placeholder key → startup preflight
   warns and continues, wizard completes, three enhanced turns against a
   stub model, clean shutdown with backup phase, second start reads the
   stores back. Fails on any traceback in the log. This is the executable's
   acceptance test too.
2. **Bug-class ledger**: `docs/BUG_RETROSPECTIVE_20260715_20260904.md` lists
   the recurring classes (substring matches, newest-first slicing, dead
   wiring, write-back bugs, proxy validation). For each class, name the guard
   test that closes it structurally (several exist: ordered-slice guard,
   budget-meter parity, tool-wiring parity, capability parity). Classes with
   no guard get one; a class with a guard gets no more per-incident patches.
3. **Feature audit for the beta build**: list every Settings toggle and
   agentic tool; mark each SHIP / HIDE / REMOVE. Known HIDE candidates:
   synthesis generators (off since 07-15), graph walk (retired), Outlook
   (dormant), curation auto mode (locked), pattern analysis surfaces beyond
   insight mode. Hidden = not rendered in the SPA and not in the tool list.
4. **Sensor that isn't the owner** (workflow §7.4): nightly agent over
   `logs/turn_records.jsonl` + the day's debug records ranking anomalies.
   This becomes the beta roll-up in Phase 3, so build it once, content-free
   from the start (see the telemetry allowlist below).
5. **Priced cold-start probe** (also G04-T02): ~15 real-model turns, each wrapped
   in `[test]…[/test]`, on an EMPTY store with synthetic personas, measuring
   answer quality and gate behaviour. Gate thresholds (e.g.
   `gate_rel_threshold_retrieval` 0.60, `COSINE_SIMILARITY_THRESHOLD` 0.15) were
   calibrated on the owner's ~7K-document corpus; nothing yet shows how they
   behave for a new user with 0–50 memories. Price the run first (call count ×
   default-model rate, per the paid-probe rule) and stop at the first 402.

## Phase 2 — the executable (one focused day, then fixes)

Facts: `daemon.spec` was last rebuilt 2026-05-18; since then `core/insight/`,
`memory/curation/`, `core/email/`, `core/actions/*`, and ~30 `utils/` modules
were added. `collect_submodules` on project packages should pick them up, but
third-party hidden imports (httpx SSE, google-auth, openpyxl, faiss, open_clip)
are the usual breakage. Model weights (bge-small, MiniLM, cross-encoder, spaCy
`en_core_web_sm`, CLIP ViT-B/32) are hundreds of MB; decide bundle vs
first-run download per model (first-run download with a progress line in the
wizard is the smaller build; bundling is the more reliable one for non-technical
testers — pick bundling for the beta, size be damned).

1. Rebuild on Linux (owner's box): `pyinstaller daemon.spec --clean --noconfirm`,
   then run the Phase-1 fresh-clone smoke AGAINST THE BINARY in a clean
   container (`docs/DOCKER_README.md` has the base image) — not in the dev venv.
2. Windows build: needs a Windows runner (GitHub Actions `windows-latest` with
   CPU-only torch, mirroring `tests.yml`'s pin) — one workflow file, artifact
   upload; do not try to cross-build. Two of five friends will likely be on
   Windows; ask before building.
3. First-run wizard must collect exactly: OpenRouter key, data directory,
   personality choice, and the beta consent text (Phase 3). Everything else
   defaults generic.
4. Alternative for the technical testers: Docker image from the same commit.
   Two distribution paths, one codebase, same smoke test.
5. **Store compatibility across weekly builds.** The beta ships a build per week
   to testers who keep their data directory. Today only `knowledge_graph.json`
   (`memory/graph_memory.py:753`), `claim_index.json`
   (`memory/claim_tracker.py:349`) and the curation queue
   (`memory/curation/engine.py:160`) carry a `schema_version`
   (checked with `grep -rn schema_version memory/ utils/`). Required before the
   first weekly update: `schema_version` on every persistent JSON store
   (e.g. profile, entity aliases, learned relations, tone state, pending
   actions) via `utils/safe_json.check_schema_version`; the Chroma version
   pinned (`chromadb==1.0.7` in `requirements.txt`/`requirements.lock.txt`) and
   a pre-upgrade Chroma compatibility check; and an AUTOMATIC backup (reuse
   `utils/backup_manager.py`) before an upgraded build first opens the data
   directory, with a refusal-to-start message that names the backup if a store
   is newer than the build.
6. **Packaged first run: SPA or Gradio wizard?** The React SPA has no onboarding
   screen (no wizard/onboard/first-run code under `web/src` or `api/`); the
   only first-run flow is the standalone Gradio wizard (`main.py:893-895`,
   `gui/wizard.py`). Decide whether the packaged build gets a SPA first-run
   with a consent screen (key, data directory, personality, beta consent text)
   or keeps the Gradio wizard for the beta. Whichever is chosen, the consent
   screen must carry the honest `HOSTED_TRANSITION` wording ("local storage,
   not private inference"). `gui/wizard.py:89` currently says "Daemon is
   architected for full local operation — with a 4090 or better GPU, you can
   swap in local models and keep 100% of your data on your own machine. The
   memory system itself already runs entirely locally." Reconcile that text
   with A06 (docs only here; the `.py` is not edited by this plan).

## Phase 3 — five-friend beta with a structural privacy boundary

**Principle:** this beta uses `HOSTED_TRANSITION`: prompts and selected
conversation/memory context leave the device for hosted inference. Optional
network tools also transmit request data. Personal stores remain local.
Sharing a bug report with the developer is a separate, explicit choice;
the default report must contain metadata only, with any opted-in text
previewed and redacted. The UI and consent text must explain provider egress
and developer report sharing separately. Local storage is not private
inference. See `generalization/03-private-data-and-egress.md` and the
[2026-09-13 review](GENERALIZATION_CI_REVIEW_20260913.md).

**Tester gate (2026-10-08).** Before the first invitation, all of: (a) A06
consent/disclosure of hosted inference and tool egress, (b) A07 content-free
telemetry (item 2 below), (c) F08 neutral default personality, (d) data-wipe
instructions (item 5), and (e) the **cost budget** below. C03/C04 (memory
storage policy, deletion cascade) and the independent privacy review move to
"pre-public release" (see the 2026-10-08 note in
`PLAN_20260913_generalization_execution.md`, "Follow-on scheduling").

**Cost budget (gate, before inviting testers).** Measure $/tester/day on the
chosen default model including the shutdown LLM jobs (reflections, summaries,
fact extraction, daily note) and agentic loops; choose the beta default model
(the committed `config/config.yaml` `models.active` is currently `kimi-3`,
line 76); confirm per-tester spend caps (OpenRouter sub-keys, item 1) work by
exhausting one on a test key and checking the failure is honest, not a silent
empty reply. Output: a one-line $/tester/day figure and the chosen model,
recorded here.

1. **Per-tester model keys.** Each tester uses their own OpenRouter key (or a
   provisioned sub-key with a spend cap, created by the owner). The owner's
   OpenRouter dashboard then shows spend only. OpenRouter's account-level
   "prompt logging" must be OFF on any key the owner controls — verify in the
   dashboard before handing out keys, and state it in the consent text.
2. **Content-free telemetry by construction.** New `TELEMETRY_PROFILE=beta`
   (config + env): `utils/turn_telemetry.record_turn` passes records through
   an ALLOWLIST (`utils/telemetry_schema.py`): intent, confidence, source, tone
   level, gate triggered/modes/reason, mode, model, timings, token estimates,
   response/query LENGTHS, error/sentinel flags, grounding flags, agentic
   round count, sandbox created/executed counts, empty-response flag, hashed
   session id. Dropped: `query`, `plan_points`, `response_plan`,
   `gate_reason` free text beyond a fixed enum, anything else. A unit test
   feeds a record containing every known content field and asserts the
   written line has none of them; a second test fails when a NEW field is
   added to `record_turn` callers without an allowlist decision (the
   budget-meter-parity pattern).
3. **Local-only stores.** Conversations, corpus, profile, graph, debug
   records, `daemon_debug.log` stay on the tester's disk. Backups stay local.
   The `[relay:` convention and the privacy redaction module are unchanged.
4. **Bug bundle = metadata by default.** The SPA's debug export becomes
   "Send a bug report": it packages the beta telemetry lines for the session,
   the anomaly roll-up, versions, and timings — no prompt or response text.
   A per-turn checkbox "include this turn's text (redacted)" lets the tester
   opt a specific turn in; the redacted text is shown to them in full before
   the bundle is written. The bundle is a zip the tester sends however they
   like (email is fine for five people); no upload endpoint in v1.
5. **Consent + expectations text** (`docs/BETA_README.md`): what Daemon is,
   that it is not a crisis service (the tone system exists but is not care),
   the verbatim telemetry allowlist, how to send a report, how to wipe
   everything (`data/` directory + key), and that the owner cannot read
   their chats.
   Same pass: README screenshots (React chat UI, Debug/Provenance view) from a
   generic config, and a check that every Reviewer Quick Path / Key
   Documentation link resolves at HEAD.
6. **Roll-up script** `scripts/beta_rollup.py`: reads N testers' telemetry
   files (and the owner's own — dogfood the same profile), prints per-tester
   and pooled tables: turns, agentic zero-round rate by gate reason, latency
   by call site, verifier flags, empty responses, error sentinels, sandbox
   waste. Same output feeds the nightly sensor. This is the "use data to
   improve" loop: weekly roll-up → top three anomaly classes → fix batch →
   new build.
7. **Five-tester runbook**: one build per week at most; testers pin a
   version; a fix is not "done" until it shows in the next roll-up.

## Sequencing and honest risk

| Item | Effort | Risk |
|---|---|---|
| Phase 0 | in flight | low |
| Docker fix lane per 09-21 handoff (after the binary smoke) | 1–2 h + build | medium; badge returns only on acceptance |
| README screenshot + link pass (before invitations) | ½ day | low; screenshots must come from a generic config |
| Cold-start probe (Phase 1 item 5) | ½ day + priced model calls | low; price it first |
| Cost measurement (Phase 3 gate) | ½ day + model calls | low; decides the beta default model |
| Store-compatibility + first-run decision (Phase 2 items 5–6) | 1–2 days | medium; touches every JSON store's load path |
| Fresh-clone smoke | 1 day | low; will expose config assumptions |
| Feature audit + hiding | 1 day | low |
| Executable rebuild + clean-container smoke | 1 day, likely +2 of fixes | **highest** — 4 months of drift, large ML deps |
| Windows build | ½ day if CI does it | medium (torch/faiss wheels) |
| Beta telemetry profile + allowlist tests | 1 day | low |
| Bug bundle UI + consent doc | 1–2 days | low |
| Roll-up script | ½ day | low |

Order: Phase 0 → smoke → feature audit → executable → telemetry profile →
bug bundle + README → invite testers. Before invitations, also close the
hosted-inference disclosure gap (A06) identified in the 2026-09-13 review; the
loopback API authorization (F01) and packaged profile/timezone (F02) gaps are
closed as of 2026-10-08 (`GENERALIZATION_CI_REVIEW_20260913.md` "Status
updates"). The executable alone does not establish those properties. The
telemetry allowlist remains the gate for accepting any data back.
Nothing here needs a new subsystem; every piece reuses existing modules
(preflight, wizard, privacy_redaction, turn_telemetry, backup_manager, the
SPA debug view).

## Owner decisions needed

1. Bundle model weights or first-run download (recommendation: bundle).
2. Testers' OS mix (decides whether the Windows CI build is needed now).
3. Per-tester own keys vs owner-provisioned capped keys (recommendation:
   provisioned sub-keys with a cap; prompt logging verified OFF).
4. Which HIDE candidates from the feature audit are actually REMOVE.
5. Beta default model, after the Phase 3 cost measurement (2026-10-08).
6. Packaged first run: SPA with a consent screen, or the Gradio wizard for the
   beta (Phase 2 item 6).
