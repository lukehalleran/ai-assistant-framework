# Web Frontend (FastAPI + React)

**Last updated:** 2026-10-27

**Status:** M1–M2 of the migration plan shipped (2026-07-14). FastAPI owns the
process; the React SPA is the primary UI; the legacy Gradio UI survives at
`/admin` for dev/admin tabs (Proposals, Synthesis grading, Settings, Logs,
Debug Trace, Status/Dedup, Personality). The first-run wizard is now the SPA
Setup page (`/api/setup/*`); the Gradio wizard survives only for
`main.py wizard` / `--wizard`.

## Architecture

```
python main.py
  └─ build_orchestrator()                    (unchanged)
  └─ check_first_run()                       (gui/launch.py — shared)
       ├─ --legacy-gui / `main.py wizard` / --wizard → launch_gui()
       │     (legacy standalone Gradio on :7860; `use_legacy_gradio_path`)
       └─ else (including a first run) → uvicorn.run(create_app(orch))
             a first run shows the SPA Setup page
            api/app.py:create_app
              ├─ lifespan startup: start_background_tasks() + idle monitor
              ├─ lifespan shutdown: main.run_shutdown_tasks_async()
              │    (pending storage → reflection/facts → daily note →
              │     synthesis dreaming; same double-run guard as legacy)
              ├─ /api/* routers, /health (reused utils/health_check.py)
              └─ mount_admin_and_frontend:
                   gr.mount_gradio_app(build_demo(orch), path="/admin")
                   StaticFiles(web/dist) at /
```

- `gui/launch.py` was split: `build_demo()` (Blocks construction + queue, no
  launch), `start_background_tasks()`, `check_first_run()`, and `launch_gui()`
  as the legacy wrapper. `share=True` tunnels are NOT available when mounted.
- The chat pipeline is NOT forked: `api/chat_service.py:submit_stream()` wraps
  `gui.handlers.handle_submit` (the deployed async generator) and translates
  its yield dicts into typed SSE events.
- Action approve/reject: `gui/handlers.py` now has transport-agnostic
  `execute_pending_action_core` / `reject_pending_action_core` returning
  `ActionOutcome` (core/actions/types.py); the Gradio button wrappers and the
  API routes both call them — same audit + executor path.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | SSE stream (events: progress, thinking, duel_thinking, message, complete, error; `message` carries CUMULATIVE content — replace-render) |
| `GET/DELETE /api/session` | restore on refresh / clear in-memory UI state ONLY (never stored memory) |
| `POST /api/actions/{id}/approve|reject` | audited human-in-the-loop action decisions |
| `POST /api/uploads` | multipart ≤100MB; returns `file_ids` referenced by ChatRequest |
| `GET /api/models` / `PUT /api/models/active` | model list + switch (persists to config.yaml) |
| `GET /api/status`, `GET /api/graph?limit=N` | showcase-panel data (corpus stats; degree-trimmed knowledge graph) |
| `GET /api/debug` | server-held per-turn debug records (query, full prompt, response, tokens, timings, provenance), PII/credential-redacted at the server boundary since 2026-09-02 (`utils/privacy_redaction.py`; the held record stays raw) — cleared with DELETE /api/session. Default view (2026-09-04): the SPA shows EVERY record the server has, newest first; "only since this page load" is an explicit opt-in toggle (default off) — a page refresh used to re-capture the session baseline AFTER the turn the owner just ran and hide it even though this endpoint still returned it [toggle added 2026-09-04, baseline mechanism from 2026-07-15]. The doc-generation and self-note bypass paths (`gui/handlers.py` `_run_doc_generation`/`_save_daemon_note`) now attach a debug record to their final SSE chunk too (2026-09-04) — before that fix, `api/chat_service.py`'s `is_final = "debug" in chunk` check never saw one, so a session made only of those turns produced an empty Debug view and a 404 on Provenance |
| `GET /api/debug/prompt?index=-1` | one turn's full prompt as redacted TXT (Content-Disposition attachment; system prompt included only in dev mode — mirrors the Gradio download button) [2026-07-14] |
| `GET /api/provenance?index=-1` | one turn's provenance view (provenance dict + mode/model/citations/tokens, thinking display-capped at 500 chars) [2026-07-14] |
| `GET /api/settings` / `PUT /api/settings/{streaming,web-search,duel,tokens,temperature,summary-cadence,synthesis,proposals}` | Settings tab parity: GET snapshot + per-section apply. Thin layer over `gui/settings_core.py` — THE same functions the Gradio tab calls (no forked logic); 400 = validation error, `persisted:false` = runtime applied but YAML write failed [2026-07-14; synthesis/proposals shutdown-LLM toggles + count sliders added 2026-07-15] |
| `GET /api/setup/status`, `POST /api/setup/{openrouter,profile,obsidian,complete}`, `POST /api/setup/key/{name}` | first-run Setup page backend (`api/routes/setup.py`, logic in `gui/setup_core.py`): status, OpenRouter key save/verify, optional keys, profile, vault path, completion |
| `POST /api/sync-notes`, `GET /api/sync-notes/status` | start the Obsidian vault sync and poll its retained outcome (`api/state.NotesSyncState`; a finished sync is not reported failed when the initiating request disconnects) |
| `GET /api/report/turns`, `POST /api/report/build` | report-a-problem: list turns, build a local consented diagnostic bundle (returned, never written or sent by the server) |
| `GET /api/curation/queue`, `POST /api/curation/scan`, `POST /api/curation/{proposal_id}/{apply,dismiss,undo}`, `GET /api/curation/activity` | Curation Center backend (`api/routes/curation.py`, engine in `memory/curation/`) |
| `POST /api/heartbeat`, `POST /api/shutdown` | SPA liveness ping and clean Quit (see "App lifecycle" below) |
| `GET /health` | pre-existing health check, reused |

**Security.** Every `/api/*` request passes `api/launch_auth.py` (`LaunchAuthMiddleware`):
the Host header must be loopback, `api.host` (when a specific address) or an exact entry
in `api.allowed_hosts` (shipped `[]`; no suffix or wildcard matching); writes also need
the per-launch secret in the `X-Daemon-Launch-Token` header and a same-origin `Origin`.
The token is generated per launch, never persisted or logged. Note: the API is still
unauthenticated beyond that, so keep the bind address on loopback or a tailnet.

Config: `api:` section in config.yaml (`host` 127.0.0.1, `port` 8000,
`cors_origins`, `serve_frontend`, `frontend_dist_dir`, `allowed_hosts`) → `API_*` constants in
app_config; env overrides `DAEMON_API_HOST` / `DAEMON_API_PORT`.

## Frontend (web/)

Vite + React 18 + TypeScript (strict) + Mantine 8, dark theme matching
gui/theme.py (#111827 / #1f2937 / #3b82f6, JetBrains Mono). Markdown via
react-markdown + remark-gfm/math + rehype-katex/highlight. SSE client:
@microsoft/fetch-event-source (POST body + AbortController stop button).
State: one `useChatStream` reducer hook — no Redux/Query.

Shipped: streaming chat, progress/thinking indicator + elapsed timer, inline
ActionApprovalCard, model selector, fast-mode + raw-GPT + memory-citations
toggles, file attach (📎 FileButton → /api/uploads → file_ids), sync-notes
button (POST /api/sync-notes), duel-mode thinking accordion, mobile burger
layout (navbar collapses; dvh heights), session restore/clear, and the
**Memory Transparency panel** (🧠 header toggle → right aside rendering each
turn's debug record: mode/model/token badges, pipeline-phase + retrieval-task
waterfalls, cited memories, provenance; follows the latest turn, per-turn
selector), live **ActivityLog** (per-tool agent steps streamed during long
agentic turns; keepalives filtered; log archived onto the turn's debug record
and shown in the MemoryPanel), and `[WIKI_N]` Wikipedia citations (session-wide
ids + source map + Sources-footer links, mirroring `[WEB_N]` — see
`core/agentic/tools.py _current_wiki_source_map`, `gui/handlers.py
_apply_web_citations(wiki_map=)`).

**Views (2026-07-14):** the navbar SegmentedControl switches the main pane
between Chat / Debug / Provenance / Settings, plus Curation (the 🧹 entry,
shown only in dev mode, `availability.dev_mode` in `web/src/App.tsx`) (no react-router; the chat column
stays mounted-but-hidden so an in-flight stream keeps rendering). Debug =
per-turn accordion (query → prompt → response, token counts, phase + retrieval
waterfalls, full-prompt TXT download, 📋 copy button on every text block);
Provenance = per-turn JSON (copy button) with turn selector; Settings = the
eight Gradio sections (streaming, web search, duel, tokens, temperature,
summary cadence, and — 2026-07-15 — the two shutdown-LLM steps: synthesis
dreaming and code proposals, each an on/off toggle plus a per-shutdown count
slider) applying through the shared `gui/settings_core.py`. Debug and
Provenance default to showing the WHOLE session the server holds, newest
first (2026-09-04 — see the `GET /api/debug` row above for why the prior
"ongoing UI session only" default was wrong on a page refresh); a "since
this page load only" `Switch` on each view opts back into the 2026-07-15
Gradio-parity scoping (`web/src/api/debugSession.ts` still supplies the
baseline for that toggle and for the absolute indices prompt-export links
and the turn selector need; self-heals if the count shrinks after a server
restart). Components: `web/src/components/debug/DebugPage.tsx`,
`debug/ProvenancePage.tsx`, `settings/SettingsPage.tsx`.

Not yet: graph view in the SPA (the `/api/graph` endpoint exists). The frozen build
is covered by `docs/BUILD_GUIDE.md`.

## Single-instance guard

`utils/single_instance.py` — `main.py` gui/cli modes take an exclusive flock
on `<chroma parent>/daemon.lock` before building the orchestrator; a second
launch exits immediately with the holder's PID (two instances sharing
ChromaDB/corpus caused the duplicate-threads incident). The kernel releases
the lock on any process death (incl. SIGKILL), so zombies can't strand it.

## Gradio share default

`GRADIO_SHARE` defaults **off** (2026-07-14): the legacy Gradio path used to
default to requesting a public gradio.live tunnel — a public URL to an
unauthenticated app with the full memory behind it. Set `GRADIO_SHARE=1`
to opt in explicitly; remote/mobile access is Tailscale + FastAPI (below).

## Startup preflight

`utils/preflight.py` (2026-07-14) — runs in gui/cli modes right after the
single-instance lock. An unwritable data directory aborts startup (memory
could not persist); a missing/placeholder `OPENAI_API_KEY`, missing
`TAVILY_API_KEY`, or missing spaCy `en_core_web_sm` model print actionable
warnings and continue (graceful degradation). Separately, a corrupt critical
JSON store (graph/aliases/profile/corpus/claims) raises `CorruptStoreError`
during orchestrator build — `main.py` prints the recovery message (quarantine
path included) and exits instead of running with, then overwriting, empty state.

## Remote / mobile access

Server binds `api.host` (config.local.yaml overrides to the Tailscale IP;
`DAEMON_API_HOST` env also works). Tailnet-only binding keeps the
unauthenticated API off the home LAN. From a phone: Tailscale app on →
`http://<tailscale-ip>:8000`. When launching over SSH, run inside tmux so a
dropped connection doesn't SIGHUP the server past the clean shutdown sequence.

## Dev workflow

```bash
python main.py                      # backend :8000 (serves web/dist if built, /admin, /health)
cd web && npm run dev               # Vite :5173, proxies /api + /health → 8000
npm run build                       # tsc --noEmit + vite build → web/dist (function-form manualChunks: react / mantine / highlight / katex / vendor; Debug/Provenance/Settings/Curation views are React.lazy — main chunk 1.07 MB → ~21 kB app + cacheable vendor chunks, 2026-09-03)
python main.py --legacy-gui         # old standalone Gradio on :7860
```

Node is a dev-only dependency; production serves the prebuilt `web/dist`.
`web/node_modules` + `web/dist` are gitignored.

## Tests

`tests/unit/test_api_chat.py` (SSE contract via httpx ASGITransport driving the
REAL handle_submit), `test_api_actions.py` (core outcomes + audit + Gradio
wrapper tuple), `test_api_misc.py` (uploads/models/status/graph/schema).
Mock-orchestrator factory shared via `tests/unit/helpers_orchestrator.py`
(re-exports from test_handle_submit.py, which is unchanged).

## App lifecycle: Quit and the packaged-mode watchdog (B3, 2026-10-12)

Closing a terminal window does not reliably hang up the process (Fedora's
Ptyxis never sends SIGHUP), so the app owns its shutdown (`api/lifecycle.py`,
`api/routes/lifecycle.py`; class BC-58).

- **`POST /api/shutdown`** (launch token + same-origin required like every
  `/api/*` write; additionally refused for non-loopback peers) schedules
  `SIGINT` to our own process ~0.5 s after the 202 is sent, so uvicorn's
  graceful exit runs the lifespan shutdown sequence exactly as Ctrl+C does.
  Idempotent: a repeat call returns 202 `already_shutting_down` with no second
  signal. A turn in flight is reported (`turn_in_flight`) but does not block it.
- **`POST /api/heartbeat`**: the SPA pings on mount, every 30 s and on
  `visibilitychange -> visible` (`web/src/api/useHeartbeat.ts`).
- **Watchdog** (YAML `app_lifecycle`): armed by the first heartbeat ever; when
  the gap exceeds `client_grace_s` (120, env `DAEMON_CLIENT_GRACE_S`) and no
  turn is in flight it requests the same clean shutdown. `exit_when_clients_gone`
  is `null` = auto (on when frozen, off in dev). Config is read live per tick.
- **SPA**: navbar "Quit Daemon" (confirm modal) -> POST -> a "Daemon has shut
  down -- you can close this tab" screen; heartbeats stop.
- The frozen smoke (`scripts/smoke_frozen_build.py`, Smoke B) ends by calling
  `/api/shutdown` and asserts 401 without the token, 202 with it, exit 0 and
  all shutdown phases.
