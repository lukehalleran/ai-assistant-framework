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
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import actions, chat, curation, debug, files, models, settings, system
from api.state import AppState
from utils.logging_utils import get_logger

logger = get_logger("api_app")


def create_app(orchestrator, start_background: bool = True) -> FastAPI:
    """Build the FastAPI app around an already-constructed orchestrator.

    start_background=False skips the startup threads + shutdown sequence
    (used by tests, which stub the orchestrator).
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

    # CORS is only needed when the Vite dev server is NOT proxying /api;
    # the config default covers the dev-server origin as a fallback.
    try:
        from config.app_config import API_CORS_ORIGINS
        if API_CORS_ORIGINS:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(API_CORS_ORIGINS),
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

    try:
        from utils.health_check import add_health_endpoint
        add_health_endpoint(app, orchestrator)
    except Exception as e:
        logger.warning(f"[API] Health endpoint not added: {e}")

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
            from fastapi.staticfiles import StaticFiles
            app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
            logger.info(f"[API] Frontend served from {FRONTEND_DIST_DIR}")
        else:
            logger.info("[API] No frontend build found (web/dist) — API + /admin only")
    except Exception as e:
        logger.warning(f"[API] Frontend static mount failed: {e}")

    return app
