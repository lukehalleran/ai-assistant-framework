"""Canary: the `websockets.legacy` modules uvicorn's default WS stack imports.

Incident date: 2026-10-12 (conditional noted in the oct12 web audit; test added
2026-10-26, lane H1). Class: BC-58 (a dependency upgrade silently removes a
module the runtime imports lazily, found at the first websocket connection).

uvicorn's `websockets` implementation (the default `ws="auto"` choice) imports
`websockets.legacy.handshake` and `websockets.legacy.server` at import time.
websockets deprecates and will eventually drop `legacy`. If this test fails,
the upgrade surfaced it in CI instead of at runtime: start uvicorn with
ws="websockets-sansio" (and check Gradio's mounted /admin websocket use), then
update this canary. No runtime code is touched by this test.
"""
import importlib

import pytest

_NEEDED = (
    "websockets.legacy.handshake",
    "websockets.legacy.server",
    "uvicorn.protocols.websockets.websockets_impl",
)

_HELP = (
    "`{mod}` no longer imports ({err}). uvicorn's default websocket stack needs "
    "websockets.legacy: start uvicorn with ws='websockets-sansio' (in api/ / "
    "main.py's uvicorn.run or Config) before upgrading, then update this canary."
)


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # canary: convert to a clear, actionable failure
        pytest.fail(_HELP.format(mod=name, err=exc))


@pytest.mark.parametrize("name", _NEEDED)
def test_websockets_legacy_modules_import(name):
    assert _import(name) is not None


def test_uvicorn_sansio_alternative_exists():
    # The documented escape hatch must itself still exist in the installed uvicorn.
    mod = _import("uvicorn.protocols.websockets.websockets_sansio_impl")
    assert hasattr(mod, "WebSocketsSansIOProtocol")


def test_pointer_names_the_remedy():
    assert "websockets-sansio" in _HELP
