"""
# utils/logging_utils.py
Module Contract
- Purpose: Central logging utilities used throughout the project. Provides named loggers and simple timing decorators.
- Inputs:
  - get_logger(name), log_and_time(label)
- Outputs:
  - Logger instances; wrapped functions with timing logs.
- Side effects:
  - None (root configuration happens in entrypoints via configure_logging()).
  - configure_logging() pins the provider/transport loggers (openai, httpx, httpcore,
    urllib3) to WARNING and filters them on every handler — the DEBUG file sink was
    persisting full request bodies (every prompt). Raw bodies require BOTH
    DAEMON_MODE=dev and DAEMON_ALLOW_SENSITIVE_HTTP_LOGS=1 (2026-09-02).
"""
from typing import Callable, Optional
import logging
import os
import time
import inspect
import functools


# These libraries log complete HTTP bodies (including model prompts) at DEBUG.
# Daemon's file sink is intentionally DEBUG even when the UI is in normal mode,
# so transport loggers need an independent privacy floor.
_SENSITIVE_TRANSPORT_LOGGERS = (
    "openai",
    "httpx",
    "httpcore",
    "urllib3",
)


class _SensitiveTransportFilter(logging.Filter):
    """Drop verbose provider/transport records that may contain prompt bodies."""

    def __init__(self, allow_sensitive: bool = False) -> None:
        super().__init__()
        self.allow_sensitive = allow_sensitive

    def filter(self, record: logging.LogRecord) -> bool:
        if self.allow_sensitive or record.levelno >= logging.WARNING:
            return True
        return not any(
            record.name == name or record.name.startswith(name + ".")
            for name in _SENSITIVE_TRANSPORT_LOGGERS
        )


def _sensitive_http_logging_enabled() -> bool:
    """True only for a deliberately opted-in developer process.

    Both variables are required so merely switching the UI to dev mode cannot
    cause raw provider request bodies to be persisted.
    """

    return (
        os.getenv("DAEMON_MODE", "").strip().lower() == "dev"
        and os.getenv("DAEMON_ALLOW_SENSITIVE_HTTP_LOGS", "").strip() == "1"
    )


def configure_logging(
    level: int = logging.INFO,
    file_path: Optional[str] = "daemon_debug.log",
    file_level: int = logging.DEBUG,
    console_level: Optional[int] = None,
) -> None:
    """Configure root logger once and avoid duplicate handlers.

    On startup, call this before creating any loggers to ensure a clean
    configuration and avoid duplicate/garbled console lines.
    """
    # Test isolation (2026-08-28): a pytest run that imports main/gui.launch
    # used to land HERE with the prod path — rotating the LIVE daemon's log
    # out from under it and writing test output into daemon_debug.log (the
    # 08-28 12:20 "doc boom"/quantum test records sat in a rotated prod log).
    # Under DAEMON_TEST_MODE the file sink is redirected to a test-only path.
    if os.getenv("DAEMON_TEST_MODE") and file_path:
        file_path = os.path.join("logs", "test_debug.log")

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    # Always set root level to the lowest of console/file levels to avoid filtering
    root.setLevel(min(level, file_level))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    allow_sensitive_http = _sensitive_http_logging_enabled()
    transport_filter = _SensitiveTransportFilter(allow_sensitive_http)

    # Set a logger floor as the first line of defence. The handler filter below
    # is retained as defence in depth if an SDK changes its logger level later.
    for logger_name in _SENSITIVE_TRANSPORT_LOGGERS:
        logging.getLogger(logger_name).setLevel(
            logging.DEBUG if allow_sensitive_http else logging.WARNING
        )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level if console_level is not None else level)
    ch.setFormatter(fmt)
    ch.addFilter(transport_filter)
    root.addHandler(ch)

    # File handler (optional)
    if file_path:
        try:
            # Rotate existing log instead of truncating — preserves debug
            # data from prior runs (critical when app restarts mid-session).
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                mtime = os.path.getmtime(file_path)
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(mtime))
                base, ext = os.path.splitext(file_path)
                rotated = f"{base}_{ts}{ext}"
                # Avoid overwriting if rotated name already exists
                if not os.path.exists(rotated):
                    os.rename(file_path, rotated)
        except Exception:
            pass
        try:
            fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(fmt)
            fh.addFilter(transport_filter)
            root.addHandler(fh)
        except Exception:
            # If file can't be opened, continue with console-only
            pass


def get_logger(name: str = "daemon_app") -> logging.Logger:
    """Return a module-specific logger.

    Root configuration should be done once via `configure_logging()` in the
    application entrypoint (e.g., main.py) to avoid duplicate handlers.
    """
    return logging.getLogger(name)


# --- Lightweight decorators ---

def log_and_time(label: str = "Function") -> Callable:
    """Decorator to log start/end and duration at DEBUG level."""
    def decorator(func):
        log = get_logger(func.__module__)

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                start = time.time()
                log.debug(f"[{label}] START")
                async for result in func(*args, **kwargs):
                    yield result
                log.debug(f"[{label}] END — Duration: {time.time() - start:.2f}s")
            return async_gen_wrapper

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_func_wrapper(*args, **kwargs):
                start = time.time()
                log.debug(f"[{label}] START")
                result = await func(*args, **kwargs)
                log.debug(f"[{label}] END — Duration: {time.time() - start:.2f}s")
                return result
            return async_func_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            log.debug(f"[{label}] START")
            result = func(*args, **kwargs)
            log.debug(f"[{label}] END — Duration: {time.time() - start:.2f}s")
            return result
        return sync_wrapper

    return decorator


def log_duration(tag: str) -> Callable:
    """Decorator to log only duration (DEBUG level)."""
    def decorator(func):
        log = get_logger(func.__module__)

        if inspect.isasyncgenfunction(func):
            return func  # Not supported nicely

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                log.debug(f"[TIMING] {tag} took {time.time() - start:.2f}s")
                return result
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            log.debug(f"[TIMING] {tag} took {time.time() - start:.2f}s")
            return result
        return sync_wrapper

    return decorator


def log_async_operation(func):
    """Decorator to log async operation start/complete/errors."""
    log = get_logger(func.__module__)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        log.debug(f"[ASYNC START] {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            log.debug(f"[ASYNC COMPLETE] {func.__name__}")
            return result
        except Exception as e:
            log.error(f"[ASYNC ERROR] {func.__name__}: {type(e).__name__}: {e}")
            raise

    return wrapper


