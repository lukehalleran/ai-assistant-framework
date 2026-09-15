"""FastAPI application factory for Daemon.

create_app(orchestrator) builds the parent app:
- lifespan startup: the launch-time background daemon threads (notes catch-up,
  reference-docs seed, model warmup) + the idle-monitor thread from main.py.
- lifespan shutdown: main.run_shutdown_tasks_async() — the same pending-storage
  → reflection/facts → daily-note → synthesis-dreaming sequence as the legacy
  path, with the same double-run guard.
- /api/* routers, /health (reused utils.health_check), Gradio dev UI mounted at
  /admin, and the built React SPA (web/dist) served at /.

Mount order matters: routers → /admin → static /.
"""

import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.launch_auth import (
    LOOPBACK_HOSTNAMES,
    LaunchAuthMiddleware,
    generate_launch_secret,
    normalize_trusted_hostnames,
)
from api.routes import actions, chat, curation, debug, files, models, settings, system
from api.state import AppState
from utils.logging_utils import get_logger

logger = get_logger("api_app")


def create_app(orchestrator, start_background: bool = True, launch_secret: str | None = None,
                trusted_hosts=None) -> FastAPI:
    """Build the FastAPI app around an already-constructed orchestrator.

    start_background=False skips the startup threads + shutdown sequence
    (used by tests, which stub the orchestrator).

    launch_secret injects a deterministic per-launch token (tests only); a
    real launch generates one high-entropy secret per process, never
    persisted or logged. See api/launch_auth.py (F01, G06-T02) for the
    Host/Origin/token contract the LaunchAuthMiddleware added below enforces.

    trusted_hosts (A01b): an explicit iterable of hostnames/IPs to trust as a
    Host header beyond loopback (tests only); None (the default) reads the
    owner's `api.host` + `api.allowed_hosts` from config.app_config instead.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_background:
            try:
                from gui.launch import start_background_tasks
                start_background_tasks(orchestrator)
            except Exception as e:
                logger.warning(f"[API] Background startup tasks failed (non-fatal): {e}")

            # Idle monitor + activity timestamp live in main.py (handle_submit
            # pokes main.update_activity_timestamp via its `import main` hook).
            try:
                import main as main_mod
                main_mod._orchestrator_ref = orchestrator
                idle_thread = threading.Thread(
                    target=main_mod._idle_monitor_thread, daemon=True, name="IdleMonitor"
                )
                idle_thread.start()
                logger.info("[API] Idle monitor started")
            except Exception as e:
                logger.warning(f"[API] Idle monitor not started: {e}")

        yield

        if start_background:
            try:
                import main as main_mod
                await main_mod.run_shutdown_tasks_async(orchestrator)
            except Exception as e:
                logger.error(f"[API] Lifespan shutdown tasks failed: {e}")

    app = FastAPI(title="Daemon", lifespan=lifespan)
    app.state.daemon = AppState(orchestrator)
    secret = launch_secret or generate_launch_secret()
    app.state.launch_secret = secret

    # CORS is only needed when Vite is NOT proxying /api. Read once so
    # LaunchAuthMiddleware's Origin allowlist (below) shares this exact
    # trusted-origin source with CORSMiddleware.
    cors_origins: list = []
    try:
        from config.app_config import API_CORS_ORIGINS
        cors_origins = list(API_CORS_ORIGINS or [])
        if cors_origins:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    except Exception as e:
        logger.warning(f"[API] CORS setup skipped: {e}")

    app.include_router(chat.router)
    app.include_router(actions.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(system.router)
    app.include_router(debug.router)
    app.include_router(settings.router)
    app.include_router(curation.router)

    # Minimal, read-only liveness (G06-T02) — not the detailed
    # utils.health_check payload; the legacy --legacy-gui path still wires
    # that one onto Gradio's own app directly (BC-58 sibling, unowned here).
    @app.get("/health", include_in_schema=False)
    async def _health():
        return {"status": "ok"}

    # A01b: owner-configured trusted-Host allowance (api.host + api.allowed_hosts,
    # exact-match only — e.g. the owner's Tailscale address), on top of A01's
    # loopback-only Host check. trusted_hosts=None (the normal path) reads the
    # config; an explicit iterable (tests) is normalized the same way. Never
    # log a hostname/IP value — counts only.
    if trusted_hosts is None:
        try:
            from config.app_config import API_ALLOWED_HOSTS, API_HOST
            trusted_hosts, rejected_hosts = normalize_trusted_hostnames(API_HOST, API_ALLOWED_HOSTS)
        except Exception as e:
            trusted_hosts, rejected_hosts = frozenset(), 0
            logger.warning(f"[API] Trusted-host config unavailable ({e}); loopback only")
    else:
        trusted_hosts, rejected_hosts = normalize_trusted_hostnames(None, trusted_hosts)
    logger.info(f"[API] Trusted non-loopback hostnames: {len(trusted_hosts - LOOPBACK_HOSTNAMES)}")
    if rejected_hosts:
        logger.warning(f"[API] Rejected trusted-host entries: {rejected_hosts}")

    # Added LAST: add_middleware() prepends, so this ends up outermost and
    # runs first — before CORSMiddleware and any router. See launch_auth.py.
    # route_source=app is read fresh per-request (app.routes), not snapshotted
    # here, because mount_admin_and_frontend adds routes after this returns.
    app.add_middleware(
        LaunchAuthMiddleware,
        secret=secret,
        cors_origins=cors_origins,
        admin_packaged=getattr(sys, "frozen", False),
        route_source=app,
        trusted_hostnames=trusted_hosts,
    )

    return app


def mount_admin_and_frontend(app: FastAPI, orchestrator) -> FastAPI:
    """Mount the Gradio dev UI at /admin and the built SPA at /.

    Separate from create_app so tests can build the API without importing
    Gradio or requiring a frontend build. Note: gr.mount_gradio_app returns
    the (possibly wrapped) parent app — use the return value.
    """
    try:
        import gradio as gr
        from gui.launch import build_demo
        demo = build_demo(orchestrator)
        app = gr.mount_gradio_app(app, demo, path="/admin", max_file_size="100mb")

        # Bare /admin gets swallowed by the StaticFiles root mount before the
        # sub-app's slash redirect can fire — redirect it explicitly.
        from fastapi.responses import RedirectResponse

        @app.get("/admin", include_in_schema=False)
        async def _admin_redirect():
            return RedirectResponse(url="/admin/")

        logger.info("[API] Gradio admin UI mounted at /admin")
    except Exception as e:
        logger.error(f"[API] Gradio /admin mount failed: {e}")

    try:
        from config.app_config import API_SERVE_FRONTEND, FRONTEND_DIST_DIR
        if API_SERVE_FRONTEND and os.path.isdir(FRONTEND_DIST_DIR):
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles

            index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
            if os.path.isfile(index_path):
                # Only delivery channel for the token: a plain GET can't carry a
                # custom header, and LaunchAuthMiddleware already Host-checked
                # this request, so only a loopback OR owner-trusted Host caller
                # (A01b: api.host / api.allowed_hosts) sees the body — for a
                # trusted host, the owner's own network ACL (e.g. the tailnet)
                # is the trust boundary, not this server. Never cached, never
                # framed (clickjacking the approval buttons via a framed page),
                # and the secret is HTML-escaped before going into an attribute.
                import html as html_lib

                @app.get("/", include_in_schema=False)
                async def _serve_shell():
                    with open(index_path, encoding="utf-8") as f:
                        html = f.read()
                    token = html_lib.escape(app.state.launch_secret, quote=True)
                    tag = f'<meta name="daemon-launch-token" content="{token}">'
                    html = html.replace("<head>", "<head>" + tag, 1) if "<head>" in html else tag + html
                    return HTMLResponse(html, headers={
                        "Cache-Control": "no-store",
                        "X-Frame-Options": "DENY",
                        "Content-Security-Policy": "frame-ancestors 'none'",
                    })

            # Registered after "/" above, which claims the exact path; this
            # Mount only ever answers other asset paths — public/inert.
            app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
            logger.info(f"[API] Frontend served from {FRONTEND_DIST_DIR}")
        else:
            logger.info("[API] No frontend build found (web/dist) — API + /admin only")
    except Exception as e:
        logger.warning(f"[API] Frontend static mount failed: {e}")

    return app
