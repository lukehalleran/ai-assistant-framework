"""F01 / G06-T02: loopback Host, same-origin, launch-token authorization for
the deployed FastAPI app (api/app.py, api/launch_auth.py). Drives the real
ASGI app via httpx.ASGITransport (never a live socket) with fake executors.
Routes are enumerated from `app.routes`, never hardcoded, so a new route is
proven protected by construction (parent review round 1: F1-F6)."""

import re
from unittest.mock import MagicMock, patch

import httpx
import pytest

from api.app import mount_admin_and_frontend
from api.launch_auth import (
    LAUNCH_TOKEN_HEADER,
    LaunchAuthMiddleware,
    build_allowed_origins,
    classify_request,
    is_allowed_origin,
    is_loopback_host,
)
from tests.unit.api_launch_auth_client import TEST_LAUNCH_SECRET, TEST_ORIGIN, authed_client
from tests.unit.api_launch_auth_client import make_test_app as create_app_with_secret
from tests.unit.helpers_orchestrator import _make_orchestrator
from tests.unit.test_api_actions import _make_proposal, _patches

FOREIGN_HOST_URL = "http://evil.example"
_PUBLIC_ALLOWLIST = {("/health", "GET"), ("/", "GET"), ("/admin", "GET")}
_MUTATING = ("POST", "PUT", "PATCH", "DELETE")


def _raw_client(app, base_url=FOREIGN_HOST_URL, headers=None):
    # No token, no Origin, and (by default) a non-loopback Host.
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=base_url, headers=headers or {})


def _route_cases(app, exclude=("HEAD", "OPTIONS")):
    # (concrete_path, method) for every declared route+method.
    cases = []
    for route in app.routes:
        path, methods = getattr(route, "path", None), getattr(route, "methods", None)
        if path is None or methods is None:
            continue  # e.g. the StaticFiles Mount, which has no method list
        cases += [(re.sub(r"\{[^}]+\}", "x", path), m) for m in methods if m not in exclude]
    return cases


def _approval_app_and_registry():
    app = create_app_with_secret(_make_orchestrator(), start_background=False)
    app.state.daemon.session.pending_action_id = "act-1"
    store = MagicMock()
    store.approve.return_value = _make_proposal()
    patches, _, registry = _patches(store, execute_result=MagicMock(success=True, message="x"))
    for p in patches:
        p.start()
    return app, registry


def _full_app(tmp_path, packaged=False, **kwargs):
    """The deployed app with admin + a tmp-dir frontend build mounted, so the
    live route table includes "/", "/admin", and every static/app route —
    stubs Gradio's demo so no real orchestrator/UI is built."""
    (tmp_path / "index.html").write_text("<html><head></head><body>app</body></html>")
    (tmp_path / "app.js").write_text("console.log('hi');")
    with patch("sys.frozen", packaged, create=True):
        app = create_app_with_secret(_make_orchestrator(), start_background=False, **kwargs)
    with patch("config.app_config.API_SERVE_FRONTEND", True), \
         patch("config.app_config.FRONTEND_DIST_DIR", str(tmp_path)), \
         patch("gui.launch.build_demo", return_value=__import__("gradio").Blocks()):
        app = mount_admin_and_frontend(app, _make_orchestrator())
    return app


class TestLaunchAuthHelpers:  # pure helper functions in api/launch_auth.py

    def test_loopback_host(self):
        assert is_loopback_host("127.0.0.1:8000", 8000)
        assert is_loopback_host("localhost:8000", 8000)
        assert is_loopback_host("[::1]:8000", 8000)
        assert is_loopback_host("127.0.0.1", None)  # test transport: no server port known
        assert not is_loopback_host("evil.example", 8000)
        assert not is_loopback_host("127.0.0.1:9999", 8000)  # wrong port for THIS connection
        assert not is_loopback_host("", 8000)
        assert not is_loopback_host(None, 8000)
        # F3: str.isdigit() accepts non-ASCII digits ("²") that int() rejects — must not raise.
        assert is_loopback_host("127.0.0.1:²", 8000) is False

    def test_origin_allowlist_and_request_classification(self):
        allowed = build_allowed_origins(["http://localhost:5173"], 8000)
        assert is_allowed_origin("http://127.0.0.1:8000", allowed)
        assert is_allowed_origin("http://localhost:5173", allowed)
        for bad in ("http://evil.example", "null", None, "https://127.0.0.1:8000"):
            assert not is_allowed_origin(bad, allowed)
        # F3: SplitResult.port raises ValueError for an out-of-range port — must not raise.
        assert is_allowed_origin("http://127.0.0.1:99999", allowed) is False
        # classify_request's path-based steps (admin/public/api) need no route
        # table and fall back to "default" without one; step 3's live
        # route-matching is proven end-to-end below, against the real app.
        assert classify_request({"path": "/admin/", "method": "GET"}, None) == "admin"
        assert classify_request({"path": "/health", "method": "GET"}, None) == "public"
        assert classify_request({"path": "/health", "method": "POST"}, None) == "default"


@pytest.mark.asyncio
async def test_websocket_rejected_but_lifespan_passes_through():
    calls, sent = [], []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    async def send(message):
        sent.append(message)

    mw = LaunchAuthMiddleware(inner, secret="s3cr3t")
    ws_scope = {
        "type": "websocket", "path": "/ws", "method": "GET",
        "headers": [(b"host", b"127.0.0.1:8000")], "server": ("127.0.0.1", 8000),
    }
    await mw(ws_scope, None, send)
    assert calls == []  # inner app never called for a websocket scope
    assert sent == [{"type": "websocket.close", "code": 4403}]

    await mw({"type": "lifespan"}, None, None)
    assert calls == ["lifespan"]


@pytest.mark.asyncio
async def test_non_ascii_token_header_rejects_instead_of_raising():
    """Parent review F7: header bytes decode as latin-1, and
    hmac.compare_digest raises TypeError on non-ASCII str arguments, so a
    hostile token byte must still 401 (never a 500) while the valid token
    passes."""
    calls, sent = [], []

    async def inner(scope, receive, send):
        calls.append(scope["path"])

    async def send(message):
        sent.append(message)

    mw = LaunchAuthMiddleware(inner, secret="s3cr3t")

    def scope(token: bytes):
        return {
            "type": "http", "path": "/api/session", "method": "GET",
            "headers": [(b"host", b"127.0.0.1:8000"), (b"x-daemon-launch-token", token)],
            "server": ("127.0.0.1", 8000),
        }

    await mw(scope(b"s3cr3\xe9"), None, send)
    assert calls == [] and sent[0]["status"] == 401
    await mw(scope(b"s3cr3t"), None, send)
    assert calls == ["/api/session"]


class TestDeployedEnforcement:
    def _app(self, **kwargs):
        return create_app_with_secret(_make_orchestrator(), start_background=False, **kwargs)

    @pytest.mark.asyncio
    async def test_every_route_rejects_a_non_loopback_host_before_route_code(self):
        """Dynamic enumeration proves the Host check runs before any classification."""
        app = self._app()
        cases = _route_cases(app)
        assert cases, "route enumeration must not be empty"
        async with _raw_client(app) as client:
            for path, method in cases:
                resp = await client.request(method, path, content=b"not json at all")
                assert resp.status_code == 400, f"{method} {path} -> {resp.status_code}"
                assert resp.json()["detail"] == "invalid host"

    @pytest.mark.asyncio
    async def test_token_enforcement_on_get(self):
        """Missing, stale (a DIFFERENT launch's secret), and valid token."""
        own_secret = "secret-A" + "0" * 32
        app = self._app(launch_secret=own_secret)
        other_secret = "secret-B" + "0" * 32
        for headers, expected in (
            ({}, 401),
            ({LAUNCH_TOKEN_HEADER: other_secret}, 401),
            ({LAUNCH_TOKEN_HEADER: own_secret}, 200),
        ):
            async with _raw_client(app, base_url=TEST_ORIGIN, headers=headers) as client:
                resp = await client.get("/api/session")
            assert resp.status_code == expected, headers

    @pytest.mark.asyncio
    async def test_mutating_route_origin_enforcement(self, monkeypatch):
        """DELETE /api/session: bad Origin rejects; the configured Vite origin
        passes. F6: CORS origins are pinned, not read from config.yaml."""
        monkeypatch.setattr("config.app_config.API_CORS_ORIGINS", ["http://localhost:5173"])
        app = self._app()
        cases = [(o, 403) for o in ("null", "http://evil.example", "https://127.0.0.1:8000")]
        cases += [(None, 403), ("http://localhost:5173", 204)]
        for origin, expected in cases:
            headers = {LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET}
            if origin is not None:
                headers["Origin"] = origin
            async with _raw_client(app, base_url=TEST_ORIGIN, headers=headers) as client:
                resp = await client.delete("/api/session")
            assert resp.status_code == expected, f"origin={origin!r} -> {resp.status_code}"

    @pytest.mark.asyncio
    async def test_cors_preflight_on_protected_route_rejected_before_cors_middleware(self):
        """F4: the preflight itself is what stops a cross-origin page from ever
        attaching the token — no CORS header means the browser never sends
        the real request."""
        app = self._app()
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-daemon-launch-token",
        }
        async with _raw_client(app, base_url=TEST_ORIGIN, headers=headers) as client:
            resp = await client.options("/api/session")
        assert resp.status_code == 401
        assert "access-control-allow-origin" not in resp.headers

    @pytest.mark.asyncio
    async def test_action_approval_executor_call_count_tracks_authorization(self):
        """Zero calls for no credentials and for a simple form-encoded POST;
        exactly one call once authorized — same app throughout."""
        app, registry = _approval_app_and_registry()
        try:
            async with _raw_client(app, base_url=TEST_ORIGIN) as client:
                bare = await client.post("/api/actions/act-1/approve")
                form = await client.post(
                    "/api/actions/act-1/approve", content=b"a=1&b=2",
                    headers={"content-type": "application/x-www-form-urlencoded"},
                )
            assert bare.status_code == 401 and form.status_code == 401
            registry.execute.assert_not_called()
            async with authed_client(app) as client:
                ok = await client.post("/api/actions/act-1/approve")
            assert ok.status_code == 200
            registry.execute.assert_awaited_once()
        finally:
            patch.stopall()

    @pytest.mark.asyncio
    async def test_sse_chat_route_authorization(self):
        """Unauthorized POST never reaches the pipeline (exactly 401 — token
        is missing, so Origin is never even reached); authorized streams to
        `complete`."""
        from tests.unit.helpers_orchestrator import _make_file_processor_mock
        app = create_app_with_secret(_make_orchestrator(streaming_chunks=["ok"]), start_background=False)
        mock_submit = MagicMock()
        with patch("gui.handlers.handle_submit", mock_submit):
            async with _raw_client(app, base_url=TEST_ORIGIN) as client:  # missing token+origin
                unauth = await client.post("/api/chat", json={"text": "hi"})
        assert unauth.status_code == 401
        mock_submit.assert_not_called()
        with patch("gui.handlers.file_processor", _make_file_processor_mock("hi")), \
             patch("gui.handlers.get_conversation_logger", return_value=MagicMock()):
            async with authed_client(app) as client:
                resp = await client.post("/api/chat", json={"text": "hi"})
                body = (await resp.aread()).decode()
        assert resp.status_code == 200
        assert "event: complete" in body and "event: error" not in body

    @pytest.mark.asyncio
    async def test_health_is_minimal_public_but_still_host_checked(self):
        app = self._app()
        async with _raw_client(app, base_url=TEST_ORIGIN) as client:  # no token, no origin
            ok = await client.get("/health")
        assert ok.status_code == 200 and ok.json() == {"status": "ok"}
        async with _raw_client(app) as client:  # foreign Host
            bad = await client.get("/health")
        assert bad.status_code == 400

    @pytest.mark.asyncio
    async def test_new_route_added_after_create_app_defaults_to_protected(self):
        """F1/F5(c): the middleware reads app.routes fresh per request, so a
        route registered after create_app (as mount_admin_and_frontend does)
        is still classified "protected", not silently public — including a
        method the route never declared (Starlette Match.PARTIAL, not just
        Match.FULL: a route existing at this path is what matters)."""
        app = self._app()

        @app.get("/new-export")
        async def _new_export():
            return {"ok": True}

        async with _raw_client(app, base_url=TEST_ORIGIN) as client:  # no token
            unauth_get = await client.get("/new-export")
            unauth_post = await client.post("/new-export")  # method the route never declared
        assert unauth_get.status_code == 401 and unauth_post.status_code == 401
        async with authed_client(app) as client:
            auth = await client.get("/new-export")
        assert auth.status_code == 200


class TestAdminAndFrontendPolicy:
    @pytest.mark.asyncio
    async def test_admin_policy(self, tmp_path):
        """Packaged blocks outright; dev needs Origin but no token on a mutation."""
        packaged = _full_app(tmp_path, packaged=True)
        async with authed_client(packaged) as client:  # full valid creds
            blocked = await client.get("/admin")
        assert blocked.status_code == 404

        dev = _full_app(tmp_path, packaged=False)
        bad_origin = {"Origin": "http://evil.example"}
        async with _raw_client(dev, base_url=TEST_ORIGIN, headers=bad_origin) as client:
            rejected = await client.post("/admin/api/predict")
        async with _raw_client(dev, base_url=TEST_ORIGIN) as client:  # no token needed for /admin
            reachable = await client.get("/admin", follow_redirects=False)
        assert rejected.status_code == 403
        assert reachable.status_code in (302, 307)

    @pytest.mark.asyncio
    async def test_public_get_paths_are_unauthenticated_but_still_host_checked(self, tmp_path):
        """"/" and "/app.js" need no token/Origin but still 400 a bad Host;
        the shell response also carries F2's no-cache/no-frame headers."""
        app = _full_app(tmp_path)
        for path, expect_in_body in (("/", app.state.launch_secret), ("/app.js", "console.log")):
            async with _raw_client(app, base_url=TEST_ORIGIN) as client:  # no token/origin
                ok = await client.get(path)
            assert ok.status_code == 200 and expect_in_body in ok.text
            if path == "/":
                assert ok.headers["cache-control"] == "no-store"
                assert ok.headers["x-frame-options"] == "DENY"
                assert ok.headers["content-security-policy"] == "frame-ancestors 'none'"
            async with _raw_client(app) as client:  # foreign Host
                bad = await client.get(path)
            assert bad.status_code == 400

    @pytest.mark.asyncio
    async def test_every_route_defaults_to_protected(self, tmp_path):
        """F5(a): every non-Mount route+method, with a valid Host+Origin but
        NO token, 401s — except the explicit public allowlist, asserted to be
        exactly {("/health","GET"), ("/","GET"), ("/admin","GET")} so the skip
        list can't silently grow. Also covers FastAPI's own /docs, /redoc,
        /openapi.json (protected, per F1 item 3 — intended). F5(b): every
        MUTATING route+method, with a valid token but a foreign Origin, still
        403s (uploads, note-sync, settings, curation, models, session,
        actions) — rejected before route code, so nothing is written."""
        app = _full_app(tmp_path)
        cases = _route_cases(app)
        assert cases
        skipped = set()
        no_token = {"Origin": TEST_ORIGIN}  # allowed Origin, deliberately no token
        async with _raw_client(app, base_url=TEST_ORIGIN, headers=no_token) as client:
            for path, method in cases:
                resp = await client.request(method, path)
                if resp.status_code != 401:
                    skipped.add((path, method))
        assert skipped == _PUBLIC_ALLOWLIST

        mutating = [(p, m) for p, m in cases if m in _MUTATING]
        assert mutating
        bad_origin = {LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET, "Origin": "http://evil.example"}
        async with _raw_client(app, base_url=TEST_ORIGIN, headers=bad_origin) as client:
            for path, method in mutating:
                resp = await client.request(method, path)
                assert resp.status_code == 403, f"{method} {path} -> {resp.status_code}"
