# Compaction audit: API, UI, agents, eval, config

Date: 2026-09-15
Scope: `api/`, `gui/`, `web/`, `agent_branch/`, `eval/`, `integrations/`,
`config/`, `main.py`, `conftest.py`. Plan only; no application files, tests,
generated files, locks, or data were changed.

## Inventory and coverage

The tracked-file inventory was produced with `git ls-files` for the scope,
excluding `web/package-lock.json` and dependency/build/coverage directories.
It contains 128 files and 38,996 lines:

| area | files | lines |
| --- | ---: | ---: |
| api | 16 | 1,673 |
| gui | 10 | 11,062 |
| web (source/config; lock excluded) | 33 | 4,135 |
| agent_branch | 33 | 4,426 |
| eval | 18 | 6,446 |
| integrations | 1 | 44 |
| config (YAML/JSON/prompts included) | 15 | 9,418 |
| root (`main.py`, `conftest.py`) | 2 | 1,792 |

Representative code inspected directly: `api/app.py`, `api/chat_service.py`,
`api/routes/chat.py`, `api/schemas.py`, `api/sse.py`; `gui/handlers.py`,
`gui/launch.py`, `gui/wizard.py`, `gui/settings_core.py`; `web/src/api/
useChatStream.ts`, `client.ts`, `types.ts`, `MarkdownMessage.tsx`,
`MessageBubble.tsx`, `ChatInput.tsx`, `SettingsPage.tsx`;
`agent_branch/provisioning.py`, `queue.py`, `supervisor.py`,
`proposal_bridge.py`; `eval/schema.py`, `harness.py`, `variants.py`,
`section_registry.py`, `utilization.py`; `config/app_config.py`, `schema.py`,
`feature_registry.py`; `main.py`, `conftest.py`.

Additional representative reads (structural review, not line-by-line review)
covered all API route modules (`api/routes/actions.py`, `curation.py`,
`debug.py`, `files.py`, `models.py`, `settings.py`, `system.py`), the remaining
worker/queue entry points (`agent_branch/workers/*.py`, `run_queue.py`), eval
phase runners (`eval/run_phase2.py`, `run_phase4.py`, `run_phase5.py`,
`run_phase6.py`) and persistence checks, plus CI/build/install hooks listed
below. Findings from these reads are limited to the cited regions; they do not
claim deep review of every function in those files.

Launch/build/hooks/installer checks also covered `.github/workflows/tests.yml`,
`Dockerfile`, `pyproject.toml`, `setup.cfg`, `installer/daemon_installer.iss`,
`installer/build_installer.bat`, `scripts/start_daemon.sh`,
`scripts/build-docker.sh`, and `scripts/systemd/install_daily_notes.sh`.
`.gradio/certificate.pem` was inventory-only; certificate contents were not
loaded or reproduced.

Deferred from line-by-line review: remaining files in the inventory, especially
the many route/tab/worker/run-phase modules, all YAML/JSON values, prompt text,
and the remainder of the 6,273-line `gui/handlers.py`. The inventory is
reproducible with:

    git ls-files -- api gui web agent_branch eval integrations config main.py conftest.py

The deferred files still require symbol-level search and targeted tests before
mechanical edits. Existing literals, prompts, user-facing copy, and schema
values are treated as information/behavior and are not compaction targets.

## Findings and prioritized plan

### P0: preserve contracts while compacting

1. `api/schemas.py:23-142` is compact in code already, but its defaults,
   bounds, `Literal` event names, and `Field(default_factory=...)` are wire
   contracts. Keep each model field and default explicit. In particular,
   `api/schemas.py:31-33` and the module event table at `:3-13` must stay in
   sync with `api/chat_service.py:95-141` and the browser reducer.
2. `api/routes/chat.py:20-61,64-86` and `api/chat_service.py:77-172` encode
   admission-lock ownership, cancellation, cumulative message replacement,
   keepalives, and final history append ordering. Do not collapse wrappers or
   move `ensure_future`, `shield`, `aclose`, or reservation release without
   an async cancellation/order test matrix. A shorter implementation can
   accidentally permit concurrent turns or append a duplicate assistant turn.
3. `web/src/api/useChatStream.ts:214-323` is deliberately stateful: the
   in-flight ref, `sendId`, abort controller, complete-debug follow-up, and
   timer cleanup protect against stale streams. Preserve event names and
   reducer ordering. `web/src/api/types.ts` and `api/schemas.py` should be
   treated as one schema pair during any edit.

### P1: plausible readability/LOC wins

1. `gui/handlers.py:294-318` has two XML cleaners with shared replacement and
   blank-line normalization. A private helper parameterized by “remove whole
   tagged blocks” could reduce roughly 25 lines to 14 while retaining the
   deliberate distinction between marker-only streaming cleanup and final
   block removal. Verify all call sites around `:1107-1119`, plus literal code
   fences and tool names, before changing it.
2. `config/app_config.py:89-121` and `:135-158` repeat the same four-path
   search. A `_candidate_config_paths(filename)` helper could remove about
   10-14 lines and make precedence readable. Preserve ordering, de-duplication,
   `Path.cwd()` behavior, local-override precedence, and error fallback; these
   determine deployment defaults.
3. `eval/schema.py:89-95,141-154,179-204,231-237,254-260,269-282` repeats
   dataclass serialization boilerplate. A small explicit mixin/helper may save
   25-35 lines, but JSON keys and non-mutating `from_dict` behavior (especially
   `SnapshotLayer.from_dict`) are compatibility contracts. Prefer a narrow
   helper over reflection that silently accepts/drops fields.
4. `config/feature_registry.py:71-108` manually maps YAML fields into a
   Pydantic model. It is readable but repetitive; compaction is low priority
   because the explicit defaults document malformed/missing registry behavior.
   If changed, retain empty-cache-on-error and `force` invalidation semantics.
5. `gui/handlers.py` is the dominant readability issue (6,273 lines and many
   mode-specific branches, including `:5572-6140`). Module extraction alone is
   not a compaction win: it adds context hops and may increase total LOC. Defer
   extraction unless a specific duplicated block is removed and the resulting
   call graph remains easier to follow. Do not inline or globally deduplicate
   prompt/action strings.
6. `.github/workflows/tests.yml:1-228` is intentionally verbose policy and
   receipt wiring. Keep the `always()` gate, dependency-result checks, and
   artifact paths. `pyproject.toml:30-47` is a lint policy contract; avoid
   reformatting it as style cleanup. `installer/daemon_installer.iss:13-93`
   contains install identity, packaged paths, and an explicit destructive-data
   prompt at uninstall; shorten comments only if exact paths and prompt
   behavior remain obvious.

### P2: measured local folds only

1. `api/routes/debug.py:67-70` contains a formatting-only candidate saving
   three lines with the same location-free AST:

       before:
           content = build_redacted_prompt_export(
               record,
               include_system=DAEMON_MODE == "dev",
           )
       after:
           content = build_redacted_prompt_export(record, include_system=DAEMON_MODE == "dev")

   Retain keyword names, evaluation order and source-sensitive contracts.
   `api/routes/actions.py:28` already uses a single-line `ChatMessage` call
   and offers no corresponding saving.
2. `api/routes/curation.py:82-100` repeats `await` plus JSON serialization for
   three mutation endpoints. Inlining `p` into the return would save 3 lines,
   but it is an AST-changing local-expression refactor, not a formatting fold:

       before:
           p = await _run_operation(engine.undo, proposal_id)
           return p.model_dump(mode="json")
       after:
           return (await _run_operation(engine.undo, proposal_id)).model_dump(mode="json")

   Defer until focused tests prove exception translation, await-before-serialize
   ordering, and response serialization are unchanged. Do not count its lines
   as a safe compaction saving.
3. `eval/run_phase6.py:51-57` could use `defaultdict(Counter)`, but that changes
   AST and missing-key/type behavior. Defer it unless every escaping value and
   key read is proven compatible; it is not a formatting opportunity.
4. `web/src/components/settings/SettingsPage.tsx:34-59` has many state fields
   mirroring API settings. A metadata-driven form may reduce repetition, but
   field-specific ranges, defaults, apply payloads, and notification text are
   behavior. Defer unless a measured fold preserves the current event trace and
   rendered literals exactly.
5. `main.py:1-28,42-83,186-240` contains long frozen-mode/bootstrap and fallback
   contracts. The import ordering before `config.app_config`, environment
   setup, and module-level side effects are startup behavior. Trim stale prose
   only after comparing frozen and normal launch paths; never reorder imports.
6. `agent_branch/provisioning.py:181-222` has intentionally verbose security
   rationale around trusted checkout and hardened git. Preserve those comments
   and the exact trusted baseline/diff sequence. `agent_branch/queue.py` and
   `supervisor.py` similarly encode worker lifecycle and failure handling;
   readability extraction is safer than expression-level shortening.
7. `eval/section_registry.py:1-12,30-47` is a source-of-truth registry, and
   `eval/variants.py` consumes its flags/order. Keep comments, headers, and
   assembly order literal. Prompt/eval text in `config/prompts/`, YAML, and
   JSON is data, not dead prose, even when repetitive.

## Verification required after any implementation

Run focused API schema/SSE serialization tests; stream cancellation, duplicate
submit, keepalive, and history-order tests; frontend typecheck plus reducer,
accessibility, keyboard, and literal-render tests; config default/override and
frozen bootstrap smoke tests; agent trusted-diff/security tests; and eval
snapshot round-trip/hash/ablation tests. Compare serialized API payloads,
prompt hashes/section order, UI event traces, and default config values before
and after. Full-suite execution is outside this audit request.

## Resume checkpoint

Start with `gui/handlers.py:294-318` and `config/app_config.py:89-158` as the
lowest-risk local refactors. Before touching either, inspect all call sites and
capture focused characterization outputs. Then handle `eval/schema.py` only if
round-trip fixtures exist. Defer module extraction of `gui/handlers.py`, any
SSE/frontend state-machine edits, and all prompt/config data changes until the
P0 contract checks are green. Existing untracked docs were left untouched.
