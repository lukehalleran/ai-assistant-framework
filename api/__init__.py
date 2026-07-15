"""FastAPI backend for the Daemon web frontend.

FastAPI owns the process (uvicorn); the legacy Gradio UI is mounted at /admin
for dev/admin tabs, and the built React SPA (web/dist) is served at /.

Entry point: api.app.create_app(orchestrator).
"""
