"""2026-10-08 turn-path fixes (lane T): trigger token cap, noisy-library log
floor, first-person-only "used to", in-process completed-turn resend record,
single attachment parse per turn.

Every test drives THE deployed function; only the model call / file stub is
substituted.
"""

import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.intent_classifier import IntentClassifier, IntentType


# ---------------------------------------------------------------------------
# T1 — web-search trigger completion cap
# ---------------------------------------------------------------------------

_LONG_REASON = (
    "The user is asking about a specific, time-sensitive development that "
    "depends on current reporting rather than general knowledge, and the "
    "phrasing refers to a recent announcement whose details would have "
    "changed after any training cutoff, so a live lookup is warranted here. "
) * 2


def _long_valid_json():
    return json.dumps({
        "should_search": True,
        "confidence": 0.9,
        "reason": _LONG_REASON,
        "search_terms": [
            "claude biometric id requirement announcement 2026",
            "claude identity verification policy news",
            "anthropic user verification biometrics reaction",
            "claude id check rollout date",
        ],
        "search_depth": "standard",
        "num_searches": 3,
    })


class TestTriggerCompletionCap:
    @pytest.mark.asyncio
    async def test_max_tokens_passed_is_at_least_400_and_long_json_parses(self):
        from utils.web_search_trigger import _classify_with_llm_unified

        payload = _long_valid_json()
        assert len(payload) >= 650
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=payload)
        result = await _classify_with_llm_unified(
            query="check the news on the claude id requirement thing",
            model_manager=mm,
        )
        assert mm.generate_once.call_args.kwargs["max_tokens"] >= 400
        assert result is not None
        assert result.should_search is True
        assert len(result.search_terms) == 4

    @pytest.mark.asyncio
    async def test_truncated_completion_returns_none_and_warns_with_length(self, caplog):
        from utils.web_search_trigger import _classify_with_llm_unified

        truncated = _long_valid_json()[:420]  # cut mid-object, the live shape
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=truncated)
        with caplog.at_level(logging.WARNING):
            result = await _classify_with_llm_unified(
                query="check the news on the claude id requirement thing",
                model_manager=mm,
            )
        from utils import web_search_trigger as wst
        assert result is wst._TRIGGER_UNPARSEABLE
        msgs = [r for r in caplog.records
                if "JSON parse error" in r.getMessage()]
        assert msgs and msgs[0].levelno == logging.WARNING
        assert f"length={len(truncated)}" in msgs[0].getMessage()


# ---------------------------------------------------------------------------
# T2 — pdfminer & friends do not flood the DEBUG file sink
# ---------------------------------------------------------------------------

class TestNoisyLibraryLoggers:
    def test_configure_logging_floors_parser_libraries_at_warning(self):
        from utils.logging_utils import configure_logging

        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        names = ("pdfminer", "pdfplumber", "PIL")
        saved_levels = {n: logging.getLogger(n).level for n in names}
        try:
            configure_logging(file_path=None)
            for child in ("pdfminer.psparser", "pdfminer.pdfinterp",
                          "pdfplumber.page", "PIL.PngImagePlugin"):
                assert logging.getLogger(child).getEffectiveLevel() >= logging.WARNING, child
            # privacy floor unchanged
            assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)
            for n, lvl in saved_levels.items():
                logging.getLogger(n).setLevel(lvl)


# ---------------------------------------------------------------------------
# T3 — "used to" only as a first-person habitual-past shape
# ---------------------------------------------------------------------------

LIVE_HOMEWORK_QUERY = (
    "I am making an error, but cant find it 7. Ok now recall, the variance "
    "you are used to is Population Variance, this is Sample Varience, formula "
    "is \xa0S^2=1/n−1∑_i_n​(xi​−xˉ)^2\nWhy Bessel correction though? "
    "1/n just a bit too small, since this variance is a function of Population "
    "Variance, and we’re est"
)


class TestUsedToAnchor:
    @pytest.fixture(scope="class")
    def clf(self):
        return IntentClassifier()

    def test_live_homework_paste_is_not_temporal_recall(self, clf):
        assert clf.classify(LIVE_HOMEWORK_QUERY).intent != IntentType.TEMPORAL_RECALL

    @pytest.mark.parametrize("text", [
        "I'm used to waking up early",
        "I am used to the cold by now",
        "I got used to the noise quickly",
        "the variance you are used to is Population Variance",
        "getting used to the new schedule has been rough",
    ])
    def test_be_get_used_to_is_not_temporal_recall(self, clf, text):
        assert clf.classify(text).intent != IntentType.TEMPORAL_RECALL

    @pytest.mark.parametrize("text", [
        "I used to run every morning — when did I stop?",
        "we always used to talk about this",
    ])
    def test_first_person_habitual_past_is_temporal_recall(self, clf, text):
        assert clf.classify(text).intent == IntentType.TEMPORAL_RECALL


# ---------------------------------------------------------------------------
# T4 — delivered-but-not-yet-stored turn is recognised as a completed duplicate
# ---------------------------------------------------------------------------

class TestCompletedTurnRecord:
    @pytest.fixture(autouse=True)
    def _clean(self):
        import gui.handlers as handlers
        handlers._COMPLETED_TURNS.clear()
        yield
        handlers._COMPLETED_TURNS.clear()

    @pytest.fixture(autouse=True)
    def _owner(self):
        self._o = self._orch()
        yield

    def _orch(self, entries=()):
        # One orchestrator per process in prod; tests share self._o.
        if entries == () and getattr(self, "_o", None) is not None:
            return self._o
        return SimpleNamespace(
            memory_system=SimpleNamespace(corpus_manager=SimpleNamespace(corpus=list(entries))))

    QUERY = "Please summarize where we landed on the homework problem set"

    def test_registered_turn_is_a_duplicate_with_empty_corpus(self):
        import gui.handlers as handlers
        norm = " ".join(self.QUERY.lower().split())
        assert handlers._recent_completed_duplicate(self._orch(), norm) is None
        handlers._register_completed_turn(self._o, self.QUERY, "the delivered reply")
        assert handlers._recent_completed_duplicate(self._orch(), norm) == "the delivered reply"

    def test_whitespace_and_case_variants_match(self):
        import gui.handlers as handlers
        handlers._register_completed_turn(self._o, self.QUERY, "reply")
        resend = "  " + self.QUERY.upper().replace(" ", "  ") + "\n"
        norm = " ".join(resend.lower().split())
        assert handlers._recent_completed_duplicate(self._orch(), norm) == "reply"

    def test_expires_after_the_resend_window(self, monkeypatch):
        import gui.handlers as handlers
        norm = " ".join(self.QUERY.lower().split())
        handlers._register_completed_turn(self._o, self.QUERY, "reply")
        real = handlers._time.monotonic
        monkeypatch.setattr(
            handlers._time, "monotonic",
            lambda: real() + handlers._COMPLETED_RESEND_WINDOW_S + 1,
        )
        assert handlers._recent_completed_duplicate(self._orch(), norm) is None

    def test_different_query_or_different_files_do_not_match(self):
        import gui.handlers as handlers
        handlers._register_completed_turn(self._o, self.QUERY, "reply", ["a.pdf"])
        other = "a completely different question about something else"
        assert handlers._recent_completed_duplicate(self._orch(), other) is None
        norm = " ".join(self.QUERY.lower().split())
        assert handlers._recent_completed_duplicate(self._orch(), norm, ["b.pdf"]) is None
        assert handlers._recent_completed_duplicate(self._orch(), norm, ["a.pdf"]) == "reply"

    def test_other_orchestrator_does_not_match(self):
        import gui.handlers as handlers
        handlers._register_completed_turn(self._o, self.QUERY, "reply")
        norm = " ".join(self.QUERY.lower().split())
        other = SimpleNamespace(memory_system=SimpleNamespace(
            corpus_manager=SimpleNamespace(corpus=[])))
        assert handlers._recent_completed_duplicate(other, norm) is None

    def test_registry_is_bounded(self):
        import gui.handlers as handlers
        for i in range(handlers._COMPLETED_TURN_CAP + 10):
            handlers._register_completed_turn(self._o, f"distinct question number {i} for the cap", "r")
        assert len(handlers._COMPLETED_TURNS) == handlers._COMPLETED_TURN_CAP

    @pytest.mark.asyncio
    async def test_dispatch_storage_registers_the_delivered_turn(self, monkeypatch):
        import gui.handlers as handlers

        async def fake_store(**kwargs):
            return None

        monkeypatch.setattr(handlers, "_background_store_interaction", fake_store)
        task = handlers._dispatch_storage(
            self._o, self.QUERY, "final stored reply",
            self.QUERY, "final stored reply", None, [], None, "sid", {}, "enhanced",
        )
        await task
        norm = " ".join(self.QUERY.lower().split())
        assert handlers._recent_completed_duplicate(self._orch(), norm) == "final stored reply"

    @pytest.mark.asyncio
    async def test_handle_submit_serves_resend_before_corpus_write_lands(self, monkeypatch):
        import gui.handlers as handlers
        handlers._register_completed_turn(self._o, self.QUERY, "# the first reply")
        inner_called = []

        async def fake_inner(*args, **kwargs):
            inner_called.append(True)
            yield {"role": "assistant", "content": "should not run"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        chunks = [c async for c in handlers.handle_submit(
            self.QUERY, None, [], False, self._orch())]
        assert not inner_called
        assert "# the first reply" in chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_deliberate_retry_still_runs_fresh(self, monkeypatch):
        """_resend_serve_appropriate semantics unchanged: the client history
        already holds the reply -> the user saw it -> run fresh."""
        import gui.handlers as handlers
        handlers._register_completed_turn(self._o, self.QUERY, "# the first reply")
        inner_called = []

        async def fake_inner(*args, **kwargs):
            inner_called.append(True)
            yield {"role": "assistant", "content": "fresh"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        history = [{"role": "assistant", "content": "# the first reply"}]
        chunks = [c async for c in handlers.handle_submit(
            self.QUERY, None, history, False, self._orch())]
        assert inner_called
        assert chunks[-1]["content"] == "fresh"


# ---------------------------------------------------------------------------
# T5 — one attachment parse per turn
# ---------------------------------------------------------------------------

class TestSingleAttachmentParse:
    @pytest.mark.asyncio
    async def test_stage3_reuses_handler_parse_byte_identical(self, tmp_path, monkeypatch):
        from core.context_pipeline import (
            ContextPipeline, set_precomputed_files, clear_precomputed_files,
        )
        from utils.file_processor import FileProcessor

        assert str(tmp_path).startswith("/tmp/")
        doc = tmp_path / "notes.txt"
        doc.write_text("Line one of the attachment.\nLine two.")
        f = SimpleNamespace(name=str(doc))

        calls = []
        real = FileProcessor._process_text_file

        def counting(self, file):
            calls.append(1)
            return real(self, file)

        monkeypatch.setattr(FileProcessor, "_process_text_file", counting)

        fp = FileProcessor()
        user_text = "what does the attachment say?"
        analysis_text = user_text + "\n\n[ATTACHMENT NOTE] something"  # differs from user_text

        # Old path reference (fresh parse for the pipeline's own input).
        reference = await fp.process_files(analysis_text, [f])
        calls.clear()

        # Handler parse (once) then Stage 3.
        handler_result = await fp.process_files_structured(user_text, [f])
        assert set_precomputed_files([f], user_text, handler_result.text_content)
        pipe = ContextPipeline.__new__(ContextPipeline)
        pipe.file_processor = fp
        try:
            merged = await pipe._process_files(analysis_text, [f])
        finally:
            clear_precomputed_files()
        assert len(calls) == 1
        assert merged == reference

    @pytest.mark.asyncio
    async def test_mismatched_file_list_falls_back_to_parsing(self, tmp_path, monkeypatch):
        from core.context_pipeline import (
            ContextPipeline, set_precomputed_files, clear_precomputed_files,
        )
        from utils.file_processor import FileProcessor

        doc = tmp_path / "other.txt"
        doc.write_text("different file")
        f1 = SimpleNamespace(name=str(doc))
        f2 = SimpleNamespace(name=str(doc))  # equal content, distinct object
        calls = []
        real = FileProcessor._process_text_file
        monkeypatch.setattr(
            FileProcessor, "_process_text_file",
            lambda self, file: (calls.append(1), real(self, file))[1],
        )
        fp = FileProcessor()
        res = await fp.process_files_structured("q", [f1])
        calls.clear()
        set_precomputed_files([f1], "q", res.text_content)
        pipe = ContextPipeline.__new__(ContextPipeline)
        pipe.file_processor = fp
        try:
            merged = await pipe._process_files("q", [f2])
        finally:
            clear_precomputed_files()
        assert len(calls) == 1  # parsed again
        assert merged == await fp.process_files("q", [f2])

    @pytest.mark.asyncio
    async def test_no_handoff_means_current_behaviour(self, tmp_path):
        from core.context_pipeline import ContextPipeline, clear_precomputed_files
        from utils.file_processor import FileProcessor

        clear_precomputed_files()
        doc = tmp_path / "a.txt"
        doc.write_text("abc")
        f = SimpleNamespace(name=str(doc))
        fp = FileProcessor()
        pipe = ContextPipeline.__new__(ContextPipeline)
        pipe.file_processor = fp
        assert await pipe._process_files("q", [f]) == await fp.process_files("q", [f])
