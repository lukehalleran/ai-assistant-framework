"""App lifecycle (B3, 2026-10-12): the app owns its own shutdown.

Closing a terminal window does not reliably hang up the process (Fedora's
Ptyxis never sends SIGHUP), so a packaged Daemon outlived its window. Two
triggers now run the SAME clean shutdown as Ctrl+C -- SIGINT to ourselves, so
uvicorn's graceful exit runs the lifespan shutdown sequence in main.py:

* ``POST /api/shutdown`` (the SPA's Quit control), and
* a heartbeat watchdog (packaged mode): the SPA POSTs ``/api/heartbeat``; once
  the first heartbeat has ever arrived (armed), a gap longer than the grace
  period with no turn in flight requests the shutdown.

``request_clean_shutdown`` is the single function both use. class: BC-58.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import signal
import time
from typing import Callable, Optional

from config import app_config
from utils.logging_utils import get_logger

logger = get_logger("api_lifecycle")

WATCHDOG_INTERVAL_S = 15.0
SIGNAL_DELAY_S = 0.5  # lets the 202 response flush before uvicorn starts exiting


def _default_inflight() -> bool:
    from gui.handlers import has_inflight_turns  # lazy import: cycle
    return has_inflight_turns()


class LifecycleState:
    """Per-app heartbeat + shutdown bookkeeping. ``clock`` and ``inflight_fn`` are injectable."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 inflight_fn: Optional[Callable[[], bool]] = None,
                 signal_delay_s: float = SIGNAL_DELAY_S):
        self.clock = clock
        self.inflight_fn = inflight_fn or _default_inflight
        self.signal_delay_s = signal_delay_s
        self.last_heartbeat: Optional[float] = None  # None = not armed yet
        self.shutting_down = False

    def heartbeat(self) -> None:
        self.last_heartbeat = self.clock()

    def inflight(self) -> bool:
        try:
            return bool(self.inflight_fn())
        except Exception as e:  # degrades: unknown in-flight state counts as busy -> never shut down mid-turn
            logger.warning(f"[Lifecycle] in-flight check failed ({e}); treating as busy")
            return True


def request_clean_shutdown(state: LifecycleState, reason: str) -> bool:
    """Schedule SIGINT to this process. Returns False when a shutdown is already under way."""
    if state.shutting_down:
        logger.info(f"[Lifecycle] shutdown already requested; ignoring ({reason})")
        return False
    state.shutting_down = True
    logger.info(f"[Lifecycle] clean shutdown requested: {reason}")
    loop = asyncio.get_running_loop()
    loop.call_later(state.signal_delay_s, lambda: os.kill(os.getpid(), signal.SIGINT))
    return True


def watchdog_tick(state: LifecycleState) -> bool:
    """One watchdog evaluation (live config reads). True when it requested a shutdown."""
    if not app_config.APP_LIFECYCLE_EXIT_WHEN_CLIENTS_GONE:
        return False
    if state.shutting_down or state.last_heartbeat is None:
        return False
    gap = state.clock() - state.last_heartbeat
    if gap <= app_config.APP_LIFECYCLE_CLIENT_GRACE_S:
        return False
    if state.inflight():
        return False
    return request_clean_shutdown(state, f"no client heartbeat for {gap:.0f}s")


async def watchdog_loop(state: LifecycleState, interval_s: float = WATCHDOG_INTERVAL_S) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            watchdog_tick(state)
        except Exception as e:  # degrades: a bad tick must not kill the watchdog
            logger.warning(f"[Lifecycle] watchdog tick failed: {e}")


def is_loopback_client(host: Optional[str]) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback
