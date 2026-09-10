# Handoff — open items after the 2026-09-09 audit repairs

_Written 2026-09-09 ~18:10 by Fable. Everything the Codex audit confirmed
(F01–F11) plus F12 is fixed and pushed (`e651582` B1, `d18dd15` B3, `6f0d310`
B2, `0a3528c` hotfix, `e2428df` B5, `52a7fbb` B4). This file is the list of
what is NOT yet closed, in priority order, so the next session (owner, Fable
or Codex) can pick it up cold. Batch record and per-finding evidence:
`docs/PLAN_20260909_audit_repairs.md`._

## STATE
- Tree clean at `52a7fbb`, pushed. CI run for that push was in progress at
  write time — first run with the new `frontend` job and the 17 previously
  ignored test files. If it is red, read the log before anything else.
- Daemon: restarted 16:54 on `6f0d310` (B1+B3+B2). It does NOT have the
  hotfix, B5 or B4 yet.
- Uncommitted: nothing.

## OWNER — do when convenient (one at a time)
1. **Restart the daemon** (picks up B4's forced-action fix and the SPA card
   chaining; the SPA build is served from `web/dist`, so run `cd web && npm
   run build` before the restart or the browser keeps the old bundle).
2. **B4 live probe** (the exact shape that failed at 15:02):
   - `Add two calendar events called "B4 probe A" and "B4 probe B" tomorrow at 3pm and 4pm, 30 minutes each.`
     → ONE batched card; approve → both created.
   - `Delete the "B4 probe A" event tomorrow.` → a DELETE card appears (not a
     create; not a narrated "queued" with no card); approve.
   - `Delete the "B4 probe B" event tomorrow.` → same.
   Fable reads the debug records / `logs/actions_audit.jsonl` afterwards; no
   paste needed.
3. **B3 live check (F05)**: nothing to do — after the restart in step 1, the
   NEXT clean shutdown writes summaries with `source_doc_ids`. Fable checks
   the summaries collection metadata after that.
4. **Curation Center**: apply the 5-doc "Repair stream artifacts" card
   (`cur_b3b3…`) — it covers the 4 docs you undid plus one new one.
5. **Data queue** (daemon DOWN, dry-run first, pre-image backup is automatic):
   - `python scripts/purge_adaptive_exemplars.py --from-file data/exemplar_purge_candidates_20260908.txt` then `--apply`
   - Profile junk `data/profile_junk_candidates_20260908.txt` → prefer the
     pending `profile_junk_facts` curation card if it lists the same facts
     (reversible); otherwise `scripts/purge_profile_facts.py`.
   - Older 09-03 candidate files: check the Curation Center first — most are
     now covered by pending cards.

## FOLLOW-UPS FOUND TODAY (not yet fixed; each is small)
| # | Item | Where | Fix direction |
|---|---|---|---|
| 1 | **Undo is hard to find** (owner-flagged) — an applied card vanishes from the queue and Undo lives only in the Activity list at the bottom | `web/src/components/curation/CurationPage.tsx` ~L166–249 | keep the applied card in place for the session with an "Applied · Undo" button, or a "Recently applied" strip above the queue |
| 2 | **Scan dedupe** — a shutdown scan re-proposes docs already targeted by a pending card (the 5-doc card overlapped 4/4 with the applied one) | `memory/curation/engine.run_scan` + curator batch assembly | skip doc ids already in a PENDING/INTERRUPTED proposal of the same curator; at apply, an item whose target already equals `after` reports "already repaired" instead of rewriting |
| 3 | **Queue GET returns 409 during a scan** — B2 put `pending()` under the non-blocking op lock; the page shows an error + Refresh while a scan runs | `memory/curation/engine.py` `@_serialized pending`, `api/routes/curation.py` | serve a snapshot from a separate short lock (or a copy taken at the end of each operation) |
| 4 | **Settings save strips config.yaml comments** (53 comment lines lost by two toggles today; restored via `git checkout`) | `gui/settings_core.save_settings` | comment-preserving writer (ruamel.yaml round-trip) or a separate `config.settings.yaml` overlay merged at load — same pattern as `config.local.yaml` |
| 6 | **Multi-target delete/update yields ONE card** — "Delete both X and Y" produced a single `calendar_delete_event` proposal (B1 probe) and an honest note that Y still needs its own; creates batch via `events[]` but delete/update have no batch form and the forced round yields one proposal, so F07's browser chaining was not exercised | `core/agentic/controller.py` forced round, `core/actions/registry.py` calendar delete/update specs | either allow N same-type proposals in one forced round (each its own card → exercises F07 chaining) or add an `events[]`-style batch to delete/update; write a chaining probe once one of these lands |
| 7 | **Slow image turn read as a stall (19:11)** — an image-attached message spent 36 s in the vision description call before the prompt build even started, the owner killed the daemon at ~19:12, and the clean-shutdown tasks then ran. NOT the idle monitor (60-min timeout, never fired). Two hardening items fall out: `update_activity_timestamp()` is only poked at the END of `handle_submit` (~L4653), so a long in-flight turn counts as idle; and there is no progress signal to the browser during the pre-prompt image-description stage | `gui/handlers.py` ingress, `main._idle_monitor_thread`, SPA progress indicator | poke the activity timestamp at ingress; have the idle monitor skip its cycle while a turn is in flight; emit a progress line ("describing 1 image…") during the vision call so the UI does not look hung |
| 8 | **`MemoryCoordinator.debug_memory_state()` raises on lazily-unopened collections** — it iterates `chroma_store.collections` and calls `.count()` on every value, but the store keeps `None` placeholders for the 14 collections until first use (the common case) | `memory/memory_coordinator.py` `debug_memory_state`; pinned by `xfail(strict=True, reason="FINDING…")` in `tests/test_memory_coordinator_advanced.py` | iterate through `store._get_collection(name)` (or skip `None`), report unopened collections as `unopened`; then drop the xfail — strict xfail turns green-and-loud the moment it is fixed |
| 5 | **F12 residual** — `detect_action_intent` picks one type; if the gate's detection is ever wrong, a forced round now REJECTS the model's (possibly correct) sibling type. Acceptable (fails safe to "no card" + notice) but worth telemetry | `core/agentic/controller.py` forced round | log `forced_type` vs `proposed_type` mismatches to turn telemetry; review after a week |

## TEST-GAP LEADS STILL OPEN (from the audit's T-list; no confirmed defect behind them)
- **T10** request lifecycle parity (normal / agentic / recovery / cancel / action-chain through storage+display+debug receipts).
- **T11** registry entries through actual dispatch + provider payloads.
- **T12** meaningful retrieval/provenance assertions + whole-request budget ceiling.
- **T15** fresh-process import-order isolation checks.
- **T16** semantic-lane execution receipts (embedder-gated tests must report that they RAN).
- Audit "remaining leads": calendar read-cache lacks `max_events`/`lookahead_days` in its key + 250-event un-paginated fetches; `/api/actions` history append outside the chat stream lock; upload temp-file lifetime (owner policy); naive/aware datetime mixing in `get_recent()`.

## WHY (so nobody re-derives it)
- Push discipline: runners END at the commit; `git push` is the owner's own line (`feedback_push_is_owner_typed`).
- The 17 CI ignores were stale since 2026-05-11; un-ignoring them adds ~200 s and ~440 tests to CI. If the first run is red on one of those files, fix or re-ignore THAT file with a reason in `docs/TEST_LANES.md` — do not restore the blanket list.
- Codex is out of credits until ~20:59 on 2026-09-09; B5 was finished by a Claude subagent in place. Codex has NOT seen B4 or the B5 finish — a relay summarising `52a7fbb` is owed when it returns.
