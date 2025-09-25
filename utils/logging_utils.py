"""
Logging utilities for the project.

Key goals:
- Provide a simple `get_logger(name)` for modules.
- Configure root logging once on startup without duplicate handlers.
"""

from typing import Callable, Optional
import logging
import time
import inspect
import functools


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
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    # Always set root level to the lowest of console/file levels to avoid filtering
    root.setLevel(min(level, file_level))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level if console_level is not None else level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (optional)
    if file_path:
        try:
            # Truncate once at startup to start a fresh session log,
            # then attach a file handler in append mode for stability.
            open(file_path, "w", encoding="utf-8").close()
        except Exception:
            # If truncate fails, we will still attempt to attach handler
            pass
        try:
            fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(fmt)
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


"""
Module Contract
- Purpose: Central logging utilities used throughout the project. Provides named loggers and simple timing decorators.
- Inputs:
  - get_logger(name), log_and_time(label)
- Outputs:
  - Logger instances; wrapped functions with timing logs.
- Side effects:
  - None (root configuration happens in entrypoints via configure_logging()).
"""
