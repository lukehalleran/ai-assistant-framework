# Web Frontend (FastAPI + React)

**Status:** M1–M2 of the migration plan shipped (2026-07-14). FastAPI owns the
process; the React SPA is the primary UI; the legacy Gradio UI survives at
`/admin` for dev/admin tabs (Proposals, Synthesis grading, Settings, Logs,
Debug Trace, Status/Dedup, Personality) and the first-run wizard.

## Architecture

```
python main.py
  └─ build_orchestrator()                    (unchanged)
  └─ check_first_run()                       (gui/launch.py — shared)
       ├─ first run / --legacy-gui → launch_gui()   (legacy standalone Gradio)
       └─ else → uvicorn.run(create_app(orch))      (new default)
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
| `GET /api/debug` | server-held per-turn debug records (query, full prompt, response, tokens, timings, provenance), PII/credential-redacted at the server boundary since 2026-09-02 (`utils/privacy_redaction.py`; the held record stays raw) — cleared with DELETE /api/session. The SPA shows only the ongoing UI session's turns (Gradio parity): `web/src/api/debugSession.ts` snapshots the record count at page load and Debug/Provenance hide everything before it [2026-07-15] |
| `GET /api/debug/prompt?index=-1` | one turn's full prompt as redacted TXT (Content-Disposition attachment; system prompt included only in dev mode — mirrors the Gradio download button) [2026-07-14] |
| `GET /api/provenance?index=-1` | one turn's provenance view (provenance dict + mode/model/citations/tokens, thinking display-capped at 500 chars) [2026-07-14] |
| `GET /api/settings` / `PUT /api/settings/{streaming,web-search,duel,tokens,temperature,summary-cadence,synthesis,proposals}` | Settings tab parity: GET snapshot + per-section apply. Thin layer over `gui/settings_core.py` — THE same functions the Gradio tab calls (no forked logic); 400 = validation error, `persisted:false` = runtime applied but YAML write failed [2026-07-14; synthesis/proposals shutdown-LLM toggles + count sliders added 2026-07-15] |
| `GET /health` | pre-existing health check, reused |

Config: `api:` section in config.yaml (`host` 127.0.0.1, `port` 8000,
`cors_origins`, `serve_frontend`, `frontend_dist_dir`) → `API_*` constants in
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
between Chat / Debug / Provenance / Settings (no react-router; the chat column
stays mounted-but-hidden so an in-flight stream keeps rendering). Debug =
per-turn accordion (query → prompt → response, token counts, phase + retrieval
waterfalls, full-prompt TXT download, 📋 copy button on every text block);
Provenance = per-turn JSON (copy button) with turn selector; Settings = the
eight Gradio sections (streaming, web search, duel, tokens, temperature,
summary cadence, and — 2026-07-15 — the two shutdown-LLM steps: synthesis
dreaming and code proposals, each an on/off toggle plus a per-shutdown count
slider) applying through the shared `gui/settings_core.py`. Both Debug and
Provenance are scoped to the ongoing UI session (2026-07-15): Gradio's
`debug_state` is a per-page-load `gr.State`, so its debug tab never shows a
backlog — the SPA matches by snapshotting the server record count at app mount
(`web/src/api/debugSession.ts`; reset by Clear chat, self-heals if the count
shrinks after a server restart; prompt-export links keep absolute indices).
Components: `web/src/components/debug/DebugPage.tsx`,
`debug/ProvenancePage.tsx`, `settings/SettingsPage.tsx`.

Not yet: graph view (stretch), PyInstaller packaging.

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
npm run build                       # tsc --noEmit + vite build → web/dist
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
