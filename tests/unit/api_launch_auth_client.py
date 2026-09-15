"""Shared deployed-app test client for the A01 loopback/Host/launch-token
contract (api/launch_auth.py).

create_app() now rejects a request that fails its loopback Host, same-
origin, or launch-token check before any route runs. Every existing API test
file built its ASGI client against a non-loopback base_url ("http://t" /
"http://test") with no auth headers, which the new contract now (correctly)
rejects. This module centralizes the deterministic test app/client so each
test file only needs to swap its own client construction, not its test
bodies or assertions.
"""

import httpx

from api.app import create_app
from api.launch_auth import LAUNCH_TOKEN_HEADER

# Deterministic, injected (never generated) launch secret — well over the
# >=32-byte entropy floor api.launch_auth.generate_launch_secret() uses.
TEST_LAUNCH_SECRET = "test-launch-secret-" + "0" * 40
TEST_ORIGIN = "http://127.0.0.1:8000"


def make_test_app(orchestrator, **kwargs):
    """create_app() with a fixed injected launch secret, for deterministic tests."""
    kwargs.setdefault("launch_secret", TEST_LAUNCH_SECRET)
    return create_app(orchestrator, **kwargs)


def authed_client(app, *, secret: str = TEST_LAUNCH_SECRET, origin: str = TEST_ORIGIN, **kwargs):
    """AsyncClient wired for the loopback Host + launch-token contract:
    base_url is a loopback authority on the app's configured port (Host
    check) and default headers carry the launch token + a same-origin
    Origin (state-changing routes). Tests that exercise a REJECTED request
    build their own client/headers instead of using this helper."""
    headers = {LAUNCH_TOKEN_HEADER: secret, "Origin": origin}
    headers.update(kwargs.pop("headers", None) or {})
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=TEST_ORIGIN, headers=headers, **kwargs)
