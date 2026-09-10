# Handoff — B6 speed batch, phase 1 (Codex overnight, Fable referee in the morning)

_Written 2026-09-09 ~19:40 by Fable from the day's live logs. Executor: Codex.
Referee: Fable. Owner commits in the morning after referee PASS. Nothing here
touches `data/`; the daemon may stay up while you work (it only reads the tree
at launch)._

## STATE
- Tree: `52a7fbb` pushed, CI green (first run with the frontend job + 17
  un-ignored files). Uncommitted at write time: `tests/unit/test_no_vacuous_assertions.py`
  (new guard, untracked) and a Claude subagent repairing 31 vacuous tests in a
  worktree — **do not touch `tests/test_*` files named in that guard's ALLOWLIST**;
  those land separately.
- Daemon: owner relaunches tonight; runs the pushed tree.
- Numbers (from `logs/turn_records.jsonl` + `daemon_debug*.log`, 2026-09-09):
  prepare_prompt median 8.6 s enhanced / 12.6 s agentic (flat vs 09-08 and
  09-05 — no batch today changed latency). A typical text turn is 15–25 s wall.

## Verified root causes (each pinned to a log line + code)
| # | Symptom | Evidence | Cause | Fix (deterministic) |
|---|---|---|---|---|
| S1 | First image turn after a restart took 36 s BEFORE the prompt build started (19:11:05 ingress → 19:11:41 gate) | `daemon_debug.log` 19:11:09–19:11:35: `[CLIPManager] Loading ViT-B-32…` → `Model loaded (dim=512)` = 26 s; nothing else logged meanwhile | `_persist_uploads` is a background `asyncio.create_task` (`gui/handlers.py:4808`) but `VisualMemoryPipeline.ingest_image` (`knowledge/visual_memory_pipeline.py:60`) calls `self._clip.encode_image_from_path(...)` (L90) SYNCHRONOUSLY inside the coroutine — the CLIP cold load + encode block the event loop, so the gate/prepare of the SAME turn cannot start | (a) run the CLIP encode (and any other blocking step in `ingest_image`) via `await asyncio.to_thread(...)`; (b) add warmup step 8 in `gui/launch._run_model_warmup` (after step 7, same guarded shape): `get_clip_manager().load()` + open the visual store (`knowledge/visual_memory_store`) — so the first image never pays the load |
| S2 | Every agentic turn spends ~5 s after the answer is ready (18:25:42 "ready to answer" → 18:25:47 grounding line → 18:25:47 storage) | `[GroundingCheck] … demoting` at +5 s; `_apply_grounding_check` is `await`ed inline at `gui/handlers.py:3660` (agentic) and `:4296` (enhanced); verifier timeouts 5 s / 6 s (`core/grounding_check.py:729,882`) | The grounding verifier is LOG-ONLY by default (2026-09-04) yet sits on the critical path | when the check cannot change the shipped text (log-only: `GROUNDING_INTEGRATE_ENABLED` false AND the correction-append path disabled — read `_apply_grounding_check` L2875–3015 to confirm exactly which flags make it side-effect-free), schedule it as a background task AFTER the response is yielded; the task mutates `ctx.telemetry` grounding fields and the server-held debug record dict (`session.debug_records`, same dict object — `api/routes/debug.py:4`) when it finishes. When integration/append is enabled it MUST stay inline (display == storage contract). Telemetry write (`_write_turn_telemetry`) must still include the grounding fields — either await the task before the telemetry write with a short cap, or write telemetry from the task's completion callback. Say which you chose and why. |
| S3 | A long in-flight turn looks like a hang and counts as idle | `main.update_activity_timestamp()` is poked only at the END of `handle_submit` (`gui/handlers.py:4653`); no progress line during image work | (a) poke the activity timestamp at ingress too (top of `_handle_submit_inner`); (b) `_idle_monitor_thread` (`main.py:757`) skips its cycle while a turn is in flight — use the existing in-flight key set in handlers (`_inflight_key`, L4554) or a simple module counter incremented at ingress/decremented in `finally`; (c) emit a progress line ("📷 describing 1 image…") before the image-description call (the existing `_progress_emit` hook used in `context_pipeline.build`) |
| S4 | No per-turn latency roll-up; today's medians took a custom script | `turn_records.jsonl` carries only `prepare_elapsed_s`; the debug record already has `phase_timings` + `task_timings` (`gui/handlers.py:628–683`) | (a) add `phase_timings` and `task_timings` (rounded, ≤ 20 keys) to the turn-record line in `utils/turn_telemetry.py`; (b) new read-only `scripts/latency_rollup.py --days 7`: per day × mode → n, median/p90 wall & prepare; top-6 task poles by median; image turns separately (flag = `task_timings` has an image/vision key or `has_images`); exits 0, never writes (workflow §7.8) |
| S5 (measure only) | `wiki` task is the slowest gather leg (7.3 s in probe 1) | `core/prompt/gatherer_knowledge._get_wiki_content` (L1654): chroma `wiki_knowledge` query → FAISS leg (`SEM_TIMEOUT_S`=1.5, dedicated 2-worker executor) → snippet/compress | Do NOT change behaviour. Add ONE debug line at the end of the wiki task with sub-timings (chroma_ms, faiss_ms, compress_ms, timed_out flag) so tomorrow's phase 2 knows which leg to cut |

## PLANNED (in this order; tests first, record failing counts)
1. `git status` must show only the two items noted in STATE plus your work. Read the five code sites above before editing.
2. **S1** — tests: `tests/unit/test_sep09_speed_batch.py::TestClipOffLoop`: drive the deployed `VisualMemoryPipeline.ingest_image` with a fake CLIP whose `encode_image_from_path` records `threading.current_thread().name` and sleeps 0.2 s; assert the loop stayed responsive (a concurrently scheduled `asyncio.sleep(0.05)` task completes before the encode returns) and the encode ran off the main thread. Warmup: `_run_model_warmup` calls the CLIP loader once (fake manager records `load()`), and a raising loader is swallowed with the `[Warmup] clip skip:` line (existing per-step guard shape).
3. **S2** — tests: with log-only flags, `handle_submit`'s agentic and enhanced paths yield the final response BEFORE `_apply_grounding_check`'s verifier coroutine resolves (fake verifier blocks on an Event; assert the response arrived, then release; assert the debug record + telemetry fields carry the verdict after release). With integration enabled, assert the check is still awaited inline and can revise the text (existing `test_grounding_wiring.py` cases must keep passing unchanged).
4. **S3** — tests: activity timestamp advances at ingress (fake `main.update_activity_timestamp` recorder called before `prepare_prompt`); idle monitor cycle skipped while the in-flight counter > 0 (call the deployed `_idle_monitor_thread` body once with a monkeypatched sleep and a fake orchestrator; assert `_run_shutdown_tasks` not called); progress line emitted before the image-description call.
5. **S4** — tests: `utils/turn_telemetry` line contains `phase_timings`/`task_timings` when provided and omits them when absent; `scripts/latency_rollup.py` run on a synthetic JSONL (10 rows, two modes, one image turn) prints the expected medians (assert on captured stdout) and never opens anything for writing (monkeypatch `open` mode check).
6. **S5** — one debug line; a test that the wiki task's timing dict has the four keys after a run with fake chroma/FAISS.
7. Verification: new file + `tests/unit/test_grounding_wiring.py tests/unit/test_grounding_check.py tests/unit/test_handle_submit.py tests/unit/test_warmup_guards.py tests/unit/test_sep04_attachment_turn.py tests/unit/test_request_path_parity.py tests/unit/test_sep08_agentic_answer_integrity.py` + the five repo-wide guards + `ruff check .`. `tests/unit/test_no_git_state_in_tests.py` bans git reads in tests.
8. Docs: append `### B6 phase 1 — speed (2026-09-10, Codex)` to `docs/PLAN_20260909_audit_repairs.md` (per-item failed-before/after, the S2 design choice, S5's first live sub-timings once the owner restarts). `commit_message_21.txt` (subject ≤72; one paragraph; `Tests:`; `Docs:`; trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01EMdQbyDbZf7uE1ZawZ6XBN`).

## CONTINGENCY
| Condition | Action |
|---|---|
| `ingest_image` has other blocking calls (caption LLM sync? PIL decode?) | move each to `to_thread`; list them in the doc |
| The grounding flags do not cleanly separate "cannot change text" from "can" | STOP; report the flag matrix — Fable decides the boundary |
| `_inflight_key` set is per-request-dedupe, not a counter | add a plain module-level counter with `try/finally`; do not repurpose the dedupe set |
| A test needs a model download / live store / network | STOP; replace the boundary with a fake or report |
| `test_handle_submit.py` shape makes the S2 ordering test brittle | assert via the deployed `_apply_grounding_check` scheduling helper instead of the full SSE stream, and say so |
| > ~500 source lines | split S4/S5 into a second commit message |

## WHY
Today's five batches were correctness; latency was untouched and is flat.
The two biggest wins are not retrieval at all: a cold CLIP load blocking the
event loop on the first image after every restart, and a log-only verifier
costing 5 s per agentic turn on the critical path. Retrieval contention
(memories/wiki poles) is real but needs the S4/S5 numbers before anyone cuts
it — that is phase 2, planned after a day of roll-ups.

## OWNER (morning)
Read Fable's referee note, run the commit runner Fable leaves, `git push`
(your own line), restart, send one image message and one calendar delete:
the image turn should start its prompt build within ~2 s of ingress and the
agentic turn should finish ~5 s sooner than tonight's 12–20 s.
