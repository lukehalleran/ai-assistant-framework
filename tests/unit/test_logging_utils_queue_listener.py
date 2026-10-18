"""configure_logging puts file/console sinks behind a QueueListener.

Incident 2026-10-17, class: BC-41 (sync I/O on the event loop): a one-line
logger.debug blocked the loop 1088 ms inside FileHandler.flush ([LoopStall]).
"""
import asyncio
import logging
import logging.handlers
import threading

import pytest

import utils.logging_utils as lu


@pytest.fixture()
def configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    root = logging.getLogger()
    saved, level = root.handlers[:], root.level
    saved_levels = {n: logging.getLogger(n).level for n in ("openai", "httpx", "httpcore", "urllib3")}
    lu.configure_logging(file_path="ignored-under-test.log")
    try:
        yield tmp_path / "logs" / "test_debug.log"
    finally:
        lu.stop_logging_listener()
        root.handlers[:] = saved
        root.setLevel(level)
        for n, lv in saved_levels.items():
            logging.getLogger(n).setLevel(lv)


def test_root_has_one_queue_handler_and_no_direct_sinks(configured):
    hs = logging.getLogger().handlers
    qhs = [h for h in hs if isinstance(h, logging.handlers.QueueHandler)]
    assert len(qhs) == 1
    assert not [h for h in hs if isinstance(h, logging.FileHandler)]
    assert not [h for h in hs if type(h) is logging.StreamHandler]


def test_log_from_coroutine_is_written_off_the_loop_thread(configured, monkeypatch):
    writers = []
    orig_emit = logging.FileHandler.emit

    def spy(self, record):
        if "thread-probe-marker" in record.getMessage():
            writers.append(threading.get_ident())
        return orig_emit(self, record)

    monkeypatch.setattr(logging.FileHandler, "emit", spy)
    loop_thread = []

    async def main():
        loop_thread.append(threading.get_ident())
        logging.getLogger("daemon.test").debug("thread-probe-marker")

    asyncio.run(main())
    lu.stop_logging_listener()  # drains the queue
    assert writers and loop_thread
    assert writers[0] != loop_thread[0]
    assert "thread-probe-marker" in configured.read_text()


def test_reconfigure_leaves_one_live_listener(configured):
    first = lu._listener
    lu.configure_logging(file_path="ignored-under-test.log")
    second = lu._listener
    assert first is not second
    assert first._thread is None  # stopped
    assert second._thread is not None and second._thread.is_alive()
    qhs = [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.QueueHandler)]
    assert len(qhs) == 1


def test_stop_flushes_everything_logged_before(configured):
    log = logging.getLogger("daemon.flush")
    for i in range(200):
        log.debug("flush-line-%d", i)
    lu.stop_logging_listener()
    text = configured.read_text()
    assert "flush-line-0" in text and "flush-line-199" in text


def test_get_log_file_path_reports_the_sink(configured):
    path = lu.get_log_file_path()
    assert path is not None and path.endswith("test_debug.log")
    lu.stop_logging_listener()
    assert lu.get_log_file_path() is None


def test_sensitive_transport_debug_record_not_written(configured, monkeypatch):
    monkeypatch.delenv("DAEMON_ALLOW_SENSITIVE_HTTP_LOGS", raising=False)
    # Defence in depth: raise the logger floor so the HANDLER filter is what
    # is exercised once the record has crossed the queue.
    logging.getLogger("openai._base_client").setLevel(logging.DEBUG)
    logging.getLogger("openai._base_client").debug("private-prompt-marker")
    logging.getLogger("daemon.test").debug("ordinary-marker")
    lu.stop_logging_listener()
    text = configured.read_text()
    assert "ordinary-marker" in text
    assert "private-prompt-marker" not in text
