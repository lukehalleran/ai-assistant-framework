"""ContextPipeline: ONE early query rewrite, concurrent, additive-hint only.

2026-10-24/25 owner decision (query rewrite ON; settings audit F1). Classes:
BC-82 (live config), BC-72 (latency), BC-31 (two routes to one purpose),
BC-06 (raw vs processed text).

Contracts, all driven through THE deployed ``ContextPipeline.build`` /
``_rewrite_query`` (only the unrelated stages are stubbed on the instance):
  * the heavy-topic LLM check, the rewrite and STM run CONCURRENTLY (ordering
    stamps, never wall-clock bounds — BC-101);
  * the rewrite never replaces ``processed_query`` (gate scoring keeps the raw
    query — reason a; an upload's merged text is never overwritten — reason b)
    and is published as ``ContextResult.retrieval_query`` + the per-turn hint;
  * a misbehaving model (hangs / raises) yields NO rewrite, never a stall.
"""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from core.context_pipeline import ContextPipeline, ToneLevel
from utils.query_rewriter import get_turn_hint

QUERY = ("please explain in detail how the memory gate scores its candidate "
         "documents before the reranker runs today?")
WRAPPED = "   " + QUERY.replace(" how ", "\n   how ") + "\n"


def _pipeline(enable=True, timeout=1.2, model_manager=None):
    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        return ContextPipeline(
            model_manager=model_manager, topic_manager=None,
            config={"enable_query_rewrite": enable, "REWRITE_TIMEOUT_S": timeout},
        )


def _stub_unrelated(cp, monkeypatch):
    async def topics(user_input, last_exchange=None, **_kw):
        return "general", ["general"]

    async def tone(user_input, history):
        return ToneLevel.CONVERSATIONAL, None

    async def thread(current_query=None):
        return None

    monkeypatch.setattr(cp, "_extract_topics", topics)
    monkeypatch.setattr(cp, "_detect_tone", tone)
    monkeypatch.setattr(cp, "_get_identity_context", lambda: ("", None))
    monkeypatch.setattr(cp, "_get_thread_context", thread)
    monkeypatch.setattr(cp, "_get_tone_instructions", lambda level: "")
    monkeypatch.setattr(cp, "_intent_classifier", None)


def _stub_calls(cp, monkeypatch, stamps, *, rewrite_result="REWRITTEN query", stm=False):
    async def _yield_a_few():
        for _ in range(3):
            await asyncio.sleep(0)

    async def heavy(user_input, topics_):
        stamps.append("heavy:start")
        await _yield_a_few()
        stamps.append("heavy:end")
        return True, [{"fact": "f"}], NS(is_question=True, is_command=False, token_count=20)

    async def rewrite(user_input, query_analysis):
        stamps.append("rewrite:start")
        await _yield_a_few()
        stamps.append("rewrite:end")
        return rewrite_result

    async def analyze_stm(user_input, history):
        stamps.append("stm:start")
        await _yield_a_few()
        stamps.append("stm:end")
        return {"intent": "x", "topic": "t"}

    monkeypatch.setattr(cp, "_check_heavy_topics", heavy)
    monkeypatch.setattr(cp, "_rewrite_query", rewrite)
    monkeypatch.setattr(cp, "_analyze_stm", analyze_stm)
    monkeypatch.setattr(cp, "_should_run_stm", lambda history: stm)


def _build(cp, query, **kw):
    return asyncio.run(cp.build(query, **kw))


def _build_with_hints(cp, query, probes):
    """Run build() and read the per-turn hint for each probe query from the
    SAME task context (what the prompt-build chain that awaits build() sees)."""
    async def go():
        result = await cp.build(query)
        return result, [get_turn_hint(p) for p in probes]
    return asyncio.run(go())


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
class TestConcurrency:
    def test_heavy_rewrite_stm_start_before_any_finishes(self, monkeypatch, text):
        cp = _pipeline()
        _stub_unrelated(cp, monkeypatch)
        stamps: list = []
        _stub_calls(cp, monkeypatch, stamps, stm=True)
        _build(cp, text, conversation_history=[{"query": "a", "response": "b"}])
        starts = [i for i, s in enumerate(stamps) if s.endswith(":start")]
        ends = [i for i, s in enumerate(stamps) if s.endswith(":end")]
        assert len(starts) == 3 and len(ends) == 3, stamps
        assert max(starts) < min(ends), f"calls ran serially: {stamps}"

    def test_results_of_every_concurrent_call_reach_the_result(self, monkeypatch, text):
        cp = _pipeline()
        _stub_unrelated(cp, monkeypatch)
        _stub_calls(cp, monkeypatch, [], stm=True)
        r = _build(cp, text, conversation_history=[{"query": "a", "response": "b"}])
        assert r.is_heavy_topic is True
        assert r.extracted_facts == [{"fact": "f"}]
        assert r.query_analysis.token_count == 20
        assert r.stm_summary == {"intent": "x", "topic": "t"}
        assert r.retrieval_query == "REWRITTEN query"


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
class TestRewriteIsRetrievalHintOnly:
    def test_gate_query_stays_raw_and_hint_is_published(self, monkeypatch, text):
        cp = _pipeline()
        _stub_unrelated(cp, monkeypatch)
        _stub_calls(cp, monkeypatch, [])
        r, hints = _build_with_hints(
            cp, text,
            [text, text.strip() + " Rowan Tamsin", "an unrelated tool query"],
        )
        assert r.processed_query == text, "gate scoring must keep the RAW query"
        assert r.retrieval_query == "REWRITTEN query"
        assert hints[0].rewritten == "REWRITTEN query"
        assert hints[0].topics == ["general"], "Stage-1 topic is published for reuse"
        # graph expansion appends terms to the raw text: still this turn's query
        assert hints[1] == hints[0]
        assert hints[2] is None

    def test_upload_processed_query_is_never_overwritten(self, monkeypatch, text):
        cp = _pipeline()
        _stub_unrelated(cp, monkeypatch)
        _stub_calls(cp, monkeypatch, [])
        merged = text + "\n\n[ATTACHED: notes.txt]\nline one of the file"

        async def process_files(user_input, files):
            return merged

        monkeypatch.setattr(cp, "_process_files", process_files)
        r = _build(cp, text, files=[NS(name="notes.txt")])
        assert r.processed_query == merged
        assert r.retrieval_query == "REWRITTEN query"

    def test_stale_hint_never_leaks_into_next_turn(self, monkeypatch, text):
        cp = _pipeline()
        _stub_unrelated(cp, monkeypatch)
        _stub_calls(cp, monkeypatch, [])
        async def two_turns():
            await cp.build(text)
            first = get_turn_hint(text)
            _stub_calls(cp, monkeypatch, [], rewrite_result=None)
            r2 = await cp.build(text)  # same task context: turn 2 follows turn 1
            return first, r2, get_turn_hint(text)

        first, r, second = asyncio.run(two_turns())
        assert first.rewritten == "REWRITTEN query"
        assert r.retrieval_query is None
        assert second.rewritten is None, "turn 1's rewrite leaked into turn 2"
        assert second.topics == ["general"]

    def test_disabled_pipeline_makes_no_rewrite_call_but_still_shares_topics(self, monkeypatch, text):
        cp = _pipeline(enable=False)
        _stub_unrelated(cp, monkeypatch)
        stamps: list = []
        _stub_calls(cp, monkeypatch, stamps)

        async def go():
            r = await cp.build(text)
            return r, get_turn_hint(text)

        r, hint = asyncio.run(go())
        assert "rewrite:start" not in stamps
        assert r.retrieval_query is None and r.processed_query == text
        # topic reuse is not a rewrite: the switch does not remove it
        assert hint.rewritten is None and hint.topics == ["general"]


class TestDeployedRewriteQueryWithMisbehavingModels:
    """Real ``_rewrite_query``; the model double is the misbehaving part."""

    def _mm(self, behaviour, calls):
        async def generate_once(**kw):
            calls.append(kw)
            return await behaviour()
        return NS(generate_once=generate_once)

    def test_shipped_timeout_is_on(self):
        import config.app_config as cfg
        assert cfg.REWRITE_TIMEOUT_S == 1.2 and cfg.ENABLE_QUERY_REWRITE is True

    def test_hung_model_times_out_to_no_rewrite(self):
        calls: list = []

        async def hang():
            await asyncio.Event().wait()

        cp = _pipeline(timeout=0.01, model_manager=self._mm(hang, calls))
        assert asyncio.run(cp._rewrite_query(QUERY, None)) is None
        assert len(calls) == 1

    def test_raising_model_yields_no_rewrite(self):
        async def boom():
            raise RuntimeError("provider 500")

        cp = _pipeline(model_manager=self._mm(boom, []))
        assert asyncio.run(cp._rewrite_query(QUERY, None)) is None

    def test_zero_timeout_makes_no_call(self):
        calls: list = []

        async def ok():
            return "x"

        cp = _pipeline(timeout=0, model_manager=self._mm(ok, calls))
        assert asyncio.run(cp._rewrite_query(QUERY, None)) is None
        assert calls == []

    def test_standard_turn_makes_exactly_one_rewrite_call(self, monkeypatch):
        calls: list = []

        async def ok():
            return "expanded retrieval phrasing"

        cp = _pipeline(model_manager=self._mm(ok, calls))
        _stub_unrelated(cp, monkeypatch)

        async def heavy(user_input, topics_):
            return False, [], None

        monkeypatch.setattr(cp, "_check_heavy_topics", heavy)
        monkeypatch.setattr(cp, "_should_run_stm", lambda history: False)
        r = _build(cp, QUERY)
        assert len(calls) == 1
        assert r.retrieval_query == "expanded retrieval phrasing"
        assert r.processed_query == QUERY


class TestRewriteUnwrapsTestEnvelope:
    """R1 (2026-10-26, class: BC-05, BC-06): the LLM rewrite and its
    should-rewrite decision judge the probe's inner text, not the tags."""

    INNER = "how does the quillfern gate score candidate documents\nbefore the reranker runs today?"

    def _run(self, text):
        prompts: list = []

        async def generate_once(**kw):
            prompts.append(kw["prompt"])
            return "REWRITTEN"

        cp = _pipeline(model_manager=NS(generate_once=generate_once))
        return asyncio.run(cp._rewrite_query(text, None)), prompts

    def test_wrapped_and_indented_reach_prompt_without_tags(self):
        wrapped = f"[test]\n  {self.INNER}\n[/test]"
        indented = "   [test]" + self.INNER.replace("\n", "\n      ") + "[/test]\n"
        for text in (self.INNER, wrapped, indented):
            result, prompts = self._run(text)
            assert result == "REWRITTEN" and len(prompts) == 1
            assert "[test]" not in prompts[0] and "[/test]" not in prompts[0]
            assert "quillfern gate" in prompts[0]

    def test_decision_matches_unwrapped_text(self):
        # fallback heuristic: starts with "how" and >=5 words -> rewrite
        _, clean = self._run(self.INNER)
        _, wrapped = self._run(f"[test]{self.INNER}[/test]")
        assert len(clean) == len(wrapped) == 1
