# Independent audit — resumable state (September 5, 2026)

Status: this engineering pass is complete. Read `INDEPENDENT_AUDIT_20260905.md` for
the final evidence, changes, validation, git state, risks and ranked next steps.
User selected Astra 6 / xhigh, authorized a broad independent audit and repairs, and
asked for efficient credit use and a durable handoff. Runtime credits are not visible
to the agent. Do not restart the investigation or trust previous agents' conclusions
without checking the evidence below. Do not commit merely to clean the tree.

## Preservation and scope

- Repository: `/home/lukeh/Daemon_v1`, branch `master`, initial HEAD
  `e6e515de82aff941484c1d2b388746f8a33749af`. No audit commits or staged changes.
- Exact starting dirty files and patches: `/tmp/daemon-audit-baseline-cAK40N1q/`.
- Original dirty files: README.md, core/agentic/gate.py, core/context_pipeline.py,
  core/prompt/token_manager.py, docs/HANDOFF_20260904_hw_attachment_turn.md,
  docs/METRICS_SNAPSHOT.md, gui/handlers.py, utils/attachment_audit.py,
  utils/file_processor.py; original untracked tests/unit/test_sep04_evening_fixes.py.
- Original diffs and recent commits were inspected. Existing work was preserved.
  No production process was deliberately killed or restarted by this audit.
- Three bounded agents were used; all exhausted their available usage. Do not
  restart them. Their suggestions were independently checked before adopting edits.

## Confirmed evidence

- Matching September 4 19:44 trace is in ROOT `daemon_debug.log`, not logs/.
  Retrieval wall 6.94 s; memories/reflections/summaries overlap at ~6.9 s each.
  They are not sequential. Context pipeline ~5.24 s precedes retrieval.
  Builder ~10.39 s includes a 3 s LLM compression timeout; response planning
  adds ~1.7 s, giving prompt_build ~12.13 s and preparation ~17.45 s.
- September 4 log sample: 14 queued compression batches, 39 timeouts, zero
  successful compressions, two timeout-cache hits. Default LLM compression is
  now off (optional implementation remains). Do not claim a measured 3 s
  end-to-end improvement without a comparable replay.
- `469591` denotes REMOVED characters: a stored query has 471991 characters;
  old 600-token approximation kept 2400 chars. Not a 469591-character retained snip.
- Matched agentic answer reuses its decision-round answer (855 chars); final
  synthesis was skipped. Scanning September 4 persisted answers found no exact
  repeated paragraphs >=80 characters. Display duplication is unresolved, not fixed.
- Old budget admission ignored numeric priority; late floors reinserted uncapped
  entries; light prompts bypassed the budget entirely. Old full-builder methods
  reproduced four new regression failures (floor 21015 > 1200; light query
  400000 chars; cancellation did not drain retrieval; self notes were discarded).
- Main context budget is NOT a whole-request cap: system/current query/rendering
  overhead/tool schemas remain outside it. Fully rendered accounting is next work.
- Real GUI debug exports contain full prompt/system/history/debug records: feeding
  these back can multiply context. No implemented version of the user's external
  frontier protocol was located. `agent_branch/eval_layer2.py` intent judge is a stub.

## Implemented changes

- utils/text_budget.py: tokenizer-measured head/tail cap including truncation marker.
- core/prompt/token_manager.py: measured caps; priority-sorted admission; accurate
  write-back accounting; preserve structured sections or drop them whole.
- core/prompt/builder.py: final budget AFTER all floors; budget light path; cancel
  and drain async retrieval before resetting shared request state; preserve retrieved
  daemon_self_notes and codebase_changes; monotonic phase timings; skip conversational
  LLM compression, reject expansion, write back into the exact extracted text field.
- core/prompt/formatter.py: render preformatted `text` as well as `content` consistently.
- core/agentic/controller.py: cap even one oversized accumulated tool-result block.
- config/{config.yaml,app_config.py,schema.py}: default LLM compression off.
- api/routes/chat.py: reserve turn before returning streaming response, release on
  failures/cancellation, close iterator before releasing lock, reject clear during turn.
- web/src/api/useChatStream.ts: synchronous in-flight guard; preserve history on failed clear.
- Focused tests in test_independent_prompt_audit.py, test_api_chat.py; optional
  compressor tests explicitly enable the now-opt-in feature.
- scripts/audit_runtime_smoke.py: real startup/HTTP/prompt/file/persistence pipeline,
  temporary stores, deterministic generation, no provider spending or background jobs.

## Validation already obtained (overlapping batches; do not sum)

- Initial unchanged baseline: 30 passed (evening fixes + rendered-section metering).
- New targeted repro before first fixes: 8 failed / 1 passed, then passed after fixes.
- Token/builder/compressor batch: 99 passed / 1 pre-existing time-dependent skip.
- Handler/API/stream/path parity batch: 147 passed.
- Prompt/compressor/light-path/rendered-metering batch: 71 passed.
- Prompt/original evening and attachment edits/file processing/context pipeline:
  135 passed in 189.83 s.
- Web `npm run build`: passed; pre-existing circular KaTeX/vendor chunk warning.
- Ruff fatal-error subset and git diff --check: passed before last handoff update.
- An agent's much broader 327-test attempt stalled and was interrupted. This is
  NOT a full-suite pass. Some nominal unit tests load real embedding models.

## Clean-path result recovered September 5

Captured log: `/tmp/daemon-runtime-audit-final.log`.
Fresh Python PID 1054124, startup 3.145 s, current code root confirmed.
Disposable data `/tmp/daemon-runtime-audit-u4v4d4fa`.
Two real HTTP/SSE requests, four ordered history messages, three entries reloaded
from disk, first prompt 8891 chars despite a ~504k-character prior query, repeated
attachment ID rendered its unique marker once. AUDIT_RESULT emitted successfully.
Generation was deterministic: this does NOT validate real Kimi streaming or the
production daemon's optional integrations. No production daemon restart claimed.
After the result an in-flight wiki search loaded 40,982,675 metadata rows in 54.27 s:
investigate timeout/to_thread lifetime and disabled-source routing before declaring
resource cleanup proven. The main smoke assertions passed; process-exit timing was
not captured reliably after the session ended.

## Final September 5 checkpoint — completed, do not repeat

- Wiki investigation found a real concurrent initialization race: singleton creation
  was locked but load() was not. Controlled test observed two loader calls before
  the fix and one afterward. `knowledge/semantic_search.py` now serializes cold load.
  It does NOT cancel in-flight disk work. Numeric wiki source limits, rather than
  top-level enabled flags, control this path; the disposable smoke now sets both to 0.
- Shared `USER_PROFILE_MAX_TOKENS=3000` aligns builder and meter. Regression proves
  identity context within that allocation is not clipped to the generic snippet cap.
- `scripts/audit_context_budget.py` read-only actual-corpus replay: three turns,
  476719 chars / 141395 estimated tokens before, 6815 chars / 1728 tokens after,
  median 195.75 ms over five cap runs. Largest query 471991 chars → 600 estimated
  tokens. Counts are local estimates, not provider billing. No text is printed.
- Final exact 14-file regression command is in the report: 280 passed, one existing
  midnight-dependent skip, 19.85 s. A preceding run failed one legacy assertion
  requiring over-budget input remain unchanged; it now asserts no growth AND the cap.
- Fatal-error Ruff checks and git diff --check passed. Web build previously passed;
  no frontend changes after that build. No full-suite pass claimed.
- Final current-code clean smoke: PID 1068000, `/tmp/daemon-runtime-audit-jksxo9ef`,
  log `/tmp/daemon-runtime-audit-20260905.log`. Two requests, four messages, three
  entries reloaded from disk, first prompt 8893 chars, unique attachment marker once.
  AUDIT_RESULT emitted and process command completed. No provider calls, optional
  integrations disabled, production stores untouched.
- Startup 58.973 s (phase 4 coordinator/builder 50.58 s) versus prior 3.145 s.
  This variability is not isolated; do not claim a measured startup improvement.
- Host process PID 1062969, Python main.py, started September 5 11:15:13; cwd and
  executable confirmed read-only as this repo and Python 3.11.8. Left running.
  It predates the final profile/wiki changes. No production restart was performed.
- Final tree: master at original HEAD, 21 modified tracked and eight untracked files;
  no stages/commits. The report enumerates original versus audit-owned changes.

## Instructions for the next frontier

Do not begin another broad sweep or commit all dirty files blindly. Read the report,
inspect any new user changes, then choose one of its four ranked follow-ups. The
highest priority unresolved incident is duplicate display/provider output: acquire
one matching request across provider assembly, persistence, API and main chat DOM,
with IDs/lengths/hashes and a version fingerprint. Existing stored evidence does not
show repeated long paragraphs. Do not mask the symptom with heuristic text deletion.

Then bound debug/evidence handoffs and the fully rendered prompt, or measure/fuse
retrieval and its repeated embeddings/reranking with a personalization evaluation.
Original deadline timezone notes still use today's date and omit day rollover;
do not call them date/DST-correct. Existing exact midnight tests do not establish that.
Keep the production process/data safe; a cwd match does not prove loaded source identity.

Credit workflow recommendation: one bounded investigation/implementation milestone
at a time, write evidence and handoff at each milestone, reuse focused tests, avoid
parallel broad agent audits and repeated large log dumps. No numeric credit/session
capacity estimate is justified by the available telemetry.
