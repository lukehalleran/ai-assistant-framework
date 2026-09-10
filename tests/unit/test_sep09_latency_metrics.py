"""B6 latency records, read-only rollup, and deployed retrieval timings."""

import builtins
import io
import json
import math
import sys
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from utils.turn_telemetry import record_turn


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch):
    import config.app_config as config

    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
    return path


@pytest.mark.parametrize("field", ["phase_timings", "task_timings"])
def test_telemetry_bounds_and_rounds_valid_timings(telemetry_path, field):
    timings = {
        "nan": math.nan, "infinity": math.inf, "negative": -1,
        "boolean": True, "text": "3.5", "nested": {"value": 1}, "huge": 10**1000,
        **{f"task_{i}": i + 0.123456 for i in range(25)},
    }
    assert record_turn({field: timings})
    row = json.loads(telemetry_path.read_text())
    assert row[field] == {f"task_{i}": i + 0.123 for i in range(20)}
    assert timings["task_0"] == 0.123456


@pytest.mark.parametrize("value", [None, [], "bad"])
def test_telemetry_omits_absent_or_malformed_timings(telemetry_path, value):
    assert record_turn({"phase_timings": value})
    row = json.loads(telemetry_path.read_text())
    assert "phase_timings" not in row
    assert "task_timings" not in row


def test_rollup_ten_turns_read_only(tmp_path, monkeypatch, capsys):
    from scripts import latency_rollup

    now = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
    rows = [
        {
            "ts": now.isoformat(), "mode": "enhanced" if i <= 5 else "agentic-search",
            "wall_elapsed_s": 2 * i, "prepare_elapsed_s": i,
            "task_timings": {"wiki": i, "memories": 2 * i},
            "has_images": i == 10,
        }
        for i in range(1, 11)
    ]
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    before = path.read_bytes()
    builtin_open, io_open = builtins.open, io.open
    opens = []

    def checked_open(real_open):
        def check(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+")
            opens.append(mode)
            return real_open(file, mode, *args, **kwargs)
        return check

    monkeypatch.setattr(builtins, "open", checked_open(builtin_open))
    monkeypatch.setattr(io, "open", checked_open(io_open))
    assert latency_rollup.main(["--days", "7", "--path", str(path)], now=now) == 0
    output = capsys.readouterr().out
    assert "2026-09-09 enhanced text n=5 wall_s median=6.000 p90=9.200 n=5 prepare_s median=3.000 p90=4.600 n=5" in output
    assert "2026-09-09 agentic-search text n=4 wall_s median=15.000 p90=17.400 n=4" in output
    assert "2026-09-09 agentic-search image n=1 wall_s median=20.000 p90=20.000 n=1" in output
    assert "tasks_s: memories=6.000, wiki=3.000" in output
    assert opens
    assert path.read_bytes() == before


def test_rollup_filters_bad_rows_and_reports_missing_wall(tmp_path, capsys):
    from scripts import latency_rollup

    now = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
    base = {"ts": now.isoformat(), "mode": "enhanced", "prepare_elapsed_s": 4}
    rows = [
        base,
        {**base, "test_env": True, "wall_elapsed_s": 1000},
        {**base, "ts": (now - timedelta(days=8)).isoformat()},
        {**base, "ts": (now + timedelta(days=1)).isoformat()},
        {**base, "ts": "not a date"},
        {**base, "phase_timings": {"total_wall": 999, "prepare_prompt": 100}},
        {**base, "mode": ["bad"]},
        {**base, "task_timings": {"vision_description": 2}, "wall_elapsed_s": math.inf},
        {**base, "prepare_elapsed_s": -1, "wall_elapsed_s": True},
        [],
    ]
    path = tmp_path / "turns.jsonl"
    path.write_text("garbled\n" + "\n".join(json.dumps(row) for row in rows))
    assert latency_rollup.main(["--path", str(path)], now=now) == 0
    output = capsys.readouterr().out
    assert "enhanced text n=3 wall_s median=n/a p90=n/a n=0 prepare_s median=4.000 p90=4.000 n=2" in output
    assert "enhanced image n=1 wall_s median=n/a p90=n/a n=0" in output
    assert "999" not in output


def test_rollup_limits_top_tasks_and_uses_calendar_days(tmp_path, capsys):
    from scripts import latency_rollup

    now = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
    row = {
        "ts": "2026-09-03T00:00:00+00:00", "mode": "enhanced",
        "task_timings": {f"leg{i}": i for i in range(8)},
    }
    path = tmp_path / "turns.jsonl"
    path.write_text(json.dumps(row))
    assert latency_rollup.main(["--days", "7", "--path", str(path)], now=now) == 0
    output = capsys.readouterr().out
    assert "tasks_s: leg7=7.000, leg6=6.000, leg5=5.000, leg4=4.000, leg3=3.000, leg2=2.000" in output
    assert "leg1=" not in output


def test_rollup_missing_file_is_success(tmp_path, capsys):
    from scripts import latency_rollup

    assert latency_rollup.main(["--path", str(tmp_path / "missing.jsonl")]) == 0
    assert "No telemetry available" in capsys.readouterr().out


def test_rollup_reports_image_ingress_and_background_verifier_time(tmp_path, capsys):
    from scripts import latency_rollup

    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    rows = [
        {"ts": now.isoformat(), "mode": "enhanced", "has_images": True,
         "pre_prepare_elapsed_s": 0.4, "wall_elapsed_s": 8},
        {"ts": now.isoformat(), "mode": "agentic-search", "has_images": False,
         "grounding_verifier_elapsed_s": 4.5, "wall_elapsed_s": 12},
    ]
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert latency_rollup.main(["--path", str(path)], now=now) == 0
    output = capsys.readouterr().out
    image_line = next(line for line in output.splitlines() if "enhanced image" in line)
    agentic_line = next(line for line in output.splitlines() if "agentic-search text" in line)
    assert "pre_prepare_s median=0.400 p90=0.400 n=1" in image_line
    assert "grounding_s median=n/a p90=n/a n=0" in image_line
    assert "grounding_s median=4.500 p90=4.500 n=1" in agentic_line
    assert "wall_s median=12.000" in agentic_line


@pytest.fixture
def wiki(monkeypatch):
    import core.prompt.gatherer_knowledge as gk

    gatherer = gk.KnowledgeRetrievalMixin()
    gatherer.memory_coordinator = SimpleNamespace(chroma_store=None)
    gatherer._get_wiki_snippet_cached = AsyncMock(return_value={"content": "A stellar object."})
    logger = Mock()
    monkeypatch.setattr(gk, "logger", logger)
    monkeypatch.setattr(gk, "_WIKI_CHROMA_INFLIGHT", threading.Semaphore(2))
    monkeypatch.setattr(gk, "_WIKI_SEM_INFLIGHT", threading.Semaphore(2))
    monkeypatch.setattr(gk, "semantic_search_with_neighbors", Mock(return_value=[]))
    monkeypatch.setitem(sys.modules, "knowledge.WikiManager", SimpleNamespace(
        _keywords_from_query=Mock(return_value=["quasar"]),
    ))
    tracker = Mock()
    monkeypatch.setitem(sys.modules, "knowledge.wiki_tracker", SimpleNamespace(
        WikiArticleTracker=SimpleNamespace(get_instance=lambda: tracker),
    ))
    return gk, gatherer, logger


def _timing_line(logger, task):
    calls = [call for call in logger.debug.call_args_list
             if call.args and call.args[0] == f"[WikiTiming] task={task} %s"]
    assert len(calls) == 1
    return calls[0].args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "hello"])
async def test_wiki_skip_still_emits_one_timing_line(wiki, query):
    _, gatherer, logger = wiki
    assert await gatherer._get_wiki_content(query) == []
    assert _timing_line(logger, "wiki") == {
        "chroma_ms": 0.0, "fallback_ms": 0.0, "timed_out": False,
    }
    gatherer._get_wiki_snippet_cached.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_hit", [False, True])
async def test_wiki_times_chroma_and_actual_fallback(wiki, monkeypatch, local_hit):
    gk, gatherer, logger = wiki
    result = {"content": "A stellar object.", "metadata": {"title": "Quasar"}}
    chroma = SimpleNamespace(
        collections={"wiki_knowledge": Mock(count=Mock(return_value=1))},
        query_collection=Mock(return_value=[result] if local_hit else []),
    )
    gatherer.memory_coordinator.chroma_store = chroma
    ticks = iter([1.0, 1.125, 2.0, 2.25])
    monkeypatch.setattr(gk, "_t", SimpleNamespace(perf_counter=lambda: next(ticks)))
    output = await gatherer._get_wiki_content("describe stellar quasars")
    assert output[0]["content"] == result["content"]
    assert _timing_line(logger, "wiki") == {
        "chroma_ms": 125.0, "fallback_ms": 0.0 if local_hit else 250.0,
        "timed_out": False,
    }
    chroma.query_collection.assert_called_once()
    assert gatherer._get_wiki_snippet_cached.await_count == (0 if local_hit else 1)
    gk.semantic_search_with_neighbors.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["wiki", "semantic"])
async def test_wiki_timeout_records_elapsed_without_fallback(wiki, monkeypatch, task):
    gk, gatherer, logger = wiki
    release = threading.Event()
    done = threading.Event()

    def blocked(*args, **kwargs):
        try:
            release.wait(2)
            return []
        finally:
            done.set()

    if task == "wiki":
        gatherer.memory_coordinator.chroma_store = SimpleNamespace(
            collections={"wiki_knowledge": Mock(count=Mock(return_value=1))},
            query_collection=blocked,
        )
        monkeypatch.setattr(gk, "WIKI_CHROMA_TIMEOUT_S", 0.01)
        function = gatherer._get_wiki_content
        field = "chroma_ms"
    else:
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", blocked)
        monkeypatch.setattr(gk, "SEM_TIMEOUT_S", 0.01)
        function = gatherer._get_semantic_chunks
        field = "faiss_ms"
    try:
        assert await function("describe stellar quasars") == []
        timings = _timing_line(logger, task)
        assert timings["timed_out"] is True
        assert 0 < timings[field] < 1000
        gatherer._get_wiki_snippet_cached.assert_not_awaited()
    finally:
        release.set()
        assert done.wait(2)


@pytest.mark.asyncio
async def test_semantic_times_actual_faiss_task(wiki, monkeypatch):
    gk, gatherer, logger = wiki
    ticks = iter([1.0, 1.25])
    monkeypatch.setattr(gk, "_t", SimpleNamespace(perf_counter=lambda: next(ticks)))
    assert await gatherer._get_semantic_chunks("describe stellar quasars") == []
    gk.semantic_search_with_neighbors.assert_called_once()
    assert _timing_line(logger, "semantic") == {"faiss_ms": 250.0, "timed_out": False}


@pytest.mark.asyncio
async def test_wiki_records_swallowed_live_snippet_timeout(wiki, monkeypatch):
    gk, gatherer, logger = wiki
    del gatherer._get_wiki_snippet_cached
    gatherer._wiki_cache_key = lambda query: query
    monkeypatch.setattr(gk, "_wiki_cache", {})
    snippet = Mock(return_value=None)
    monkeypatch.setattr(gk, "get_wiki_snippet", snippet)

    async def timeout(future, timeout):
        await future
        raise TimeoutError

    monkeypatch.setattr(gk, "asyncio", SimpleNamespace(
        wait_for=timeout, get_event_loop=gk.asyncio.get_event_loop, TimeoutError=TimeoutError,
    ))
    assert await gatherer._get_wiki_content("describe stellar quasars") == []
    snippet.assert_called_once_with("quasar")
    timings = _timing_line(logger, "wiki")
    assert timings["timed_out"] is True
    assert timings["fallback_ms"] > 0
    logger.reset_mock()
    assert await gatherer._get_wiki_content("hello") == []
    assert _timing_line(logger, "wiki")["timed_out"] is False
