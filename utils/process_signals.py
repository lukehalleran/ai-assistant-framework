"""Process-signal hygiene for the long-running daemon (2026-09-07).

SIGHUP means "the controlling terminal went away". The daemon is routinely
launched from a phone SSH session over Tailscale and served for hours after
that session was forgotten; on 2026-09-07 15:02:27 the phone's connection
reset ("Connection reset by peer" in the sshd journal), the shell hung up,
and the process — which handled only SIGTERM/SIGINT — was killed one second
into its own idle-triggered shutdown tasks: no session summary, no LLM fact
pass, no backup, no curation scan.

``install_hangup_handler`` turns a hangup into the SAME clean shutdown the
user gets from Ctrl+C: it first detaches stdout/stderr from the dead pty
(every later ``print`` would otherwise raise ``OSError: [Errno 5]``), then
forwards SIGINT to the process so whichever layer owns SIGINT (uvicorn's
graceful exit → lifespan shutdown, or the legacy Gradio path's handler)
runs the normal sequence. Leaf module: stdlib only.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Optional

_HANGUP_SEEN = False


def detach_stdio() -> None:
    """Point stdout/stderr at /dev/null; the terminal they were attached to is gone."""
    try:
        devnull = open(os.devnull, "w")  # noqa: SIM115 — kept open for the process lifetime
    except OSError:
        return
    for name in ("stdout", "stderr"):
        try:
            getattr(sys, name).flush()
        except Exception:
            pass
        setattr(sys, name, devnull)


def install_hangup_handler(
    *,
    forward_signal: int = signal.SIGINT,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Register SIGHUP → detach stdio → re-raise ``forward_signal`` on ourselves.

    Returns False on platforms without SIGHUP (Windows). Idempotent: a second
    hangup while the first shutdown is in flight is ignored.
    """
    if not hasattr(signal, "SIGHUP"):
        return False

    def _on_hangup(signum, frame):  # noqa: ARG001 — signal-handler signature
        global _HANGUP_SEEN
        detach_stdio()
        if _HANGUP_SEEN:
            return
        _HANGUP_SEEN = True
        if logger is not None:
            logger.warning(
                "[Signal] SIGHUP (controlling terminal closed) — forwarding signal %s for a clean shutdown",
                forward_signal,
            )
        os.kill(os.getpid(), forward_signal)

    signal.signal(signal.SIGHUP, _on_hangup)
    return True
