import os
import sys
import time
import logging
import inspect
import threading
import functools
import multiprocessing
from datetime import datetime

# === Direct File Logger (Failsafe) ===
class DirectFileLogger:
    def __init__(self, filename="daemon_debug.log"):
        self.filename = filename
        self.lock = threading.Lock()

    def _write(self, level, logger_name, message):
        with self.lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = f"{timestamp} [{level}] [{logger_name}] {message}\n"
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def debug(self, msg, logger_name="daemon_app"):
        self._write("DEBUG", logger_name, msg)

    def info(self, msg, logger_name="daemon_app"):
        self._write("INFO", logger_name, msg)

    def warning(self, msg, logger_name="daemon_app"):
        self._write("WARNING", logger_name, msg)

    def error(self, msg, logger_name="daemon_app"):
        self._write("ERROR", logger_name, msg)

direct_logger = DirectFileLogger()

# === Logging Decorators ===

def log_and_time(label="Function"):
    def decorator(func):
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                start = time.time()
                logger.debug(f"[{label}] START")
                async for result in func(*args, **kwargs):
                    yield result
                duration = time.time() - start
                logger.debug(f"[{label}] END — Duration: {duration:.2f}s")
            return async_gen_wrapper

        elif inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_func_wrapper(*args, **kwargs):
                start = time.time()
                logger.debug(f"[{label}] START")
                result = await func(*args, **kwargs)
                duration = time.time() - start
                logger.debug(f"[{label}] END — Duration: {duration:.2f}s")
                return result
            return async_func_wrapper

        elif inspect.isfunction(func):
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                logger.debug(f"[{label}] START")
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.debug(f"[{label}] END — Duration: {duration:.2f}s")
                return result
            return sync_wrapper

        else:
            raise TypeError(f"@log_and_time cannot be applied to: {func.__name__}")
    return decorator


def log_duration(tag):
    def decorator(func):
        if inspect.isasyncgenfunction(func):
            return func
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                logger.debug(f"[TIMING] {tag} took {time.time() - start:.2f}s")
                return result
            return async_wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logger.debug(f"[TIMING] {tag} took {time.time() - start:.2f}s")
            return result
        return sync_wrapper
    return decorator

def with_logging(func):
    if inspect.isasyncgenfunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ensure_logging_persistence()
            async for item in func(*args, **kwargs):
                yield item
    else:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ensure_logging_persistence()
            return await func(*args, **kwargs)
    return wrapper

def log_async_operation(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.debug(f"[ASYNC START] {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"[ASYNC COMPLETE] {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"[ASYNC ERROR] {func.__name__}: {type(e).__name__}: {e}")
            raise
    return wrapper

# === Gradio Logger Wrapper ===

class GradioLogger:
    def __init__(self):
        self.direct_logger = direct_logger
        self.logger = logging.getLogger("daemon_app")

    def _log(self, level, msg):
        try:
            ensure_logging_persistence()
            getattr(self.logger, level)(msg)
        except Exception:
            pass
        getattr(self.direct_logger, level)(msg)

    def debug(self, msg): self._log("debug", msg)
    def info(self, msg): self._log("info", msg)
    def warning(self, msg): self._log("warning", msg)
    def error(self, msg): self._log("error", msg)

# === File Handler that Flushes Immediately ===

class ImmediateFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        direct_logger._write(record.levelname, record.name, record.getMessage())

# === Logging Setup ===

def setup_logging():
    import asyncio

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for name in list(logging.Logger.manager.loggerDict):
        logger_instance = logging.getLogger(name)
        logger_instance.handlers.clear()
        logger_instance.propagate = True
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = ImmediateFileHandler("daemon_debug.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    multiprocessing.log_to_stderr().setLevel(logging.DEBUG)
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    critical_loggers = [
        '', '__main__', 'daemon_app', 'httpx', 'httpcore', 'openai', 'asyncio',
        'gradio', 'gradio.queueing', 'gradio.routes', 'transformers',
        'chromadb', 'models', 'hierarchical_memory', 'llm_gates', 'unified_hierarchical_prompt_builder'
    ]
    for name in critical_loggers:
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.propagate = True

    root_logger.debug(f"🌀 New session started at {datetime.now().isoformat()}")
    direct_logger.debug(f"🌀 Direct logger also active at {datetime.now().isoformat()}")

    return root_logger

def ensure_logging_persistence():
    root_logger = logging.getLogger()
    if not any(isinstance(h, ImmediateFileHandler) for h in root_logger.handlers):
        setup_logging()
        direct_logger.warning("Logging was reset — reconfigured handlers")

def start_logging_monitor():
    def monitor():
        while True:
            time.sleep(5)
            ensure_logging_persistence()
            direct_logger.debug(f"Logging heartbeat — {datetime.now()}")
    threading.Thread(target=monitor, daemon=True).start()

# === Initialize on Import ===

logger = setup_logging()
start_logging_monitor()
