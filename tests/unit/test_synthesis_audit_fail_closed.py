"""memory/shutdown_processor.py _run_synthesis_dreaming's audit-check block
(1708-1725) becomes fail-closed (owner decision 2026-09-14, item 2; see
docs/execution/generalization/failure_outcome_design.md "Synthesis audit
auto-halt" and briefs/PARENT_STATE.md "Owner decisions received 2026-09-14",
item 2): the typed RetrievalError from a failed audit-stats read (raised by
memory/synthesis_memory.py's get_audit_stats -> get_all_results, CGR-009
#143, landed in F12d) now skips dreaming with one labels-only warning
instead of being swallowed by a broad except that let dreaming proceed as
if the audit were healthy. Any OTHER exception in the audit check is no
longer caught here either -- it propagates to the method's existing outer
except (1831), which ends dreaming non-fatally.

Split off SIZE from F12d (batches/F12d.md "SIZE checkpoint"); F12d shipped
only the synthesis reads. No CGR-009 scan anchor (a bundled owner decision),
so no response file -- evidence is in batches/F12d-2.md.

Fakes/spies only, patched at their import site inside _run_synthesis_dreaming
(no real generator, filter, graph, embedder, ModelManager or Chroma store is
ever built):
- knowledge.synthesis_filter.SynthesisFilter (imported at shutdown_processor
  ~1727) -- a spy class recording construction; process_batch returns a
  stub dict.
- knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator
  (imported at ~1741) -- with SYNTHESIS_POOLED_ENABLED=True and the other
  generator flags False, this is the ONLY generator branch the method can
  reach (the retired retrieval/graph_walk/tier2 branch is the `else`), so
  patching just this one class covers "every generator the method would
  construct" for the flag set these tests use. A spy recording construction
  and returning a configurable candidate list.
- memory.synthesis_memory.SynthesisMemory (imported at ~1710 for the audit
  read, again at ~1814 before the filter) -- a spy with a configurable
  get_audit_stats() return and separate construct/audit-read counters, so a
  test can show the audit read specifically was or was not made. EXCEPTION:
  the typed-degrade test uses the REAL class over a local FakeChromaStore
  whose query_collection raises, so get_audit_stats's real
  get_all_results(limit=500) raises the real, deployed RetrievalError.

ShutdownProcessor is built with __new__ (precedent:
tests/unit/test_thread_outcomes.py's _make_processor), setting only
chroma_store / memory_coordinator / model_manager -- the attributes the
method reads before the generators.
"""
from unittest.mock import MagicMock

import pytest

from memory.shutdown_processor import ShutdownProcessor

MARKER = "F12D2_H7RQXM"


# ---------------------------------------------------------------------------
# Fakes / spies
# ---------------------------------------------------------------------------

class FakeChromaStore:
    """Minimal stand-in exposing only query_collection -- the one method
    SynthesisMemory.get_all_results calls -- raising on demand."""

    def __init__(self, query_raise):
        self._query_raise = query_raise

    def query_collection(self, collection_name, query_text, n_results=10):
        raise self._query_raise


def _make_filter_spy():
    """Fresh spy class for knowledge.synthesis_filter.SynthesisFilter -- a
    factory so each test gets its own isolated counters."""
    class _FilterSpy:
        construct_calls = 0

        def __init__(self, **kwargs):
            _FilterSpy.construct_calls += 1

        async def process_batch(self, candidates):
            return {"accepted": 0, "rejected": 0}

    return _FilterSpy


def _make_generator_spy(candidates):
    """Fresh spy class for PooledConceptSynthesisGenerator."""
    class _GeneratorSpy:
        construct_calls = 0

        def __init__(self, **kwargs):
            _GeneratorSpy.construct_calls += 1

        async def generate_candidates(self, count):
            return list(candidates)

    return _GeneratorSpy


def _make_memory_spy(audit_stats):
    """Fresh spy class for memory.synthesis_memory.SynthesisMemory with a
    configurable get_audit_stats() return and separate construct/audit-read
    counters (so a test can prove the audit read specifically was, or was
    not, made -- distinct from the later pre-filter construction)."""
    class _MemorySpy:
        construct_calls = 0
        audit_stats_calls = 0

        def __init__(self, *args, **kwargs):
            _MemorySpy.construct_calls += 1

        def get_audit_stats(self):
            _MemorySpy.audit_stats_calls += 1
            return dict(audit_stats)

    return _MemorySpy


def _make_processor(chroma_store=None, memory_coordinator=None):
    proc = ShutdownProcessor.__new__(ShutdownProcessor)
    proc.chroma_store = chroma_store
    proc.memory_coordinator = memory_coordinator
    proc.model_manager = None
    return proc


def _patch_pooled_only(monkeypatch, *, audit_enabled=True):
    """Gate combination shared by every test below except the early-return
    control: only the pooled generator branch is reachable."""
    monkeypatch.setattr("config.app_config.SYNTHESIS_GENERATOR_ENABLED", False)
    monkeypatch.setattr("config.app_config.SYNTHESIS_POOLED_ENABLED", True)
    monkeypatch.setattr("config.app_config.SYNTHESIS_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr("config.app_config.GRAPH_WALK_ENABLED", False)
    monkeypatch.setattr("config.app_config.SYNTHESIS_AUDIT_ENABLED", audit_enabled)


# ---------------------------------------------------------------------------
# Typed degrade (deployed chain): RetrievalError skips dreaming, one warning
# ---------------------------------------------------------------------------

class TestAuditFailClosedTypedDegrade:
    @pytest.mark.asyncio
    async def test_retrieval_error_skips_dreaming_one_warning_labels_only(
        self, monkeypatch, caplog
    ):
        _patch_pooled_only(monkeypatch)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )
        store = FakeChromaStore(RuntimeError(f"boom {MARKER}"))
        proc = _make_processor(chroma_store=store)

        with caplog.at_level("WARNING"):
            await proc._run_synthesis_dreaming()  # must not raise

        assert filter_spy.construct_calls == 0
        assert gen_spy.construct_calls == 0
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "synthesis_memory" in msg
        assert "all_results:RuntimeError" in msg
        assert MARKER not in msg


# ---------------------------------------------------------------------------
# Broad failure: NOT swallowed here -- propagates to the outer except
# ---------------------------------------------------------------------------

class TestAuditBroadFailurePropagatesToOuterExcept:
    @pytest.mark.asyncio
    async def test_other_exception_ends_dreaming_non_fatal(self, monkeypatch, caplog):
        _patch_pooled_only(monkeypatch)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )

        class _RaisingSynthesisMemory:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(f"boom {MARKER}")

        monkeypatch.setattr(
            "memory.synthesis_memory.SynthesisMemory", _RaisingSynthesisMemory
        )
        proc = _make_processor(chroma_store=MagicMock())

        with caplog.at_level("WARNING"):
            await proc._run_synthesis_dreaming()  # must not raise

        assert filter_spy.construct_calls == 0
        assert gen_spy.construct_calls == 0
        assert "Synthesis dreaming failed (non-fatal)" in caplog.text


# ---------------------------------------------------------------------------
# Controls (contract item 4: unchanged behaviour)
# ---------------------------------------------------------------------------

class TestControlsUnchangedBehaviour:
    @pytest.mark.asyncio
    async def test_auto_halt_true_logs_halted_no_filter_or_generator(
        self, monkeypatch, caplog
    ):
        _patch_pooled_only(monkeypatch)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )
        mem_spy = _make_memory_spy({
            "auto_halt": True, "fp_rate": 0.75, "fp_halt_threshold": 0.5,
            "total_graded": 20, "min_graded_for_halt": 10,
        })
        monkeypatch.setattr("memory.synthesis_memory.SynthesisMemory", mem_spy)
        proc = _make_processor(chroma_store=MagicMock())

        with caplog.at_level("WARNING"):
            await proc._run_synthesis_dreaming()

        assert filter_spy.construct_calls == 0
        assert gen_spy.construct_calls == 0
        assert "Synthesis HALTED by audit queue" in caplog.text

    @pytest.mark.asyncio
    async def test_auto_halt_false_healthy_stats_dreaming_proceeds(self, monkeypatch):
        _patch_pooled_only(monkeypatch)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )
        mem_spy = _make_memory_spy({
            "auto_halt": False, "fp_rate": 0.1, "fp_halt_threshold": 0.5,
            "total_graded": 20, "min_graded_for_halt": 10,
        })
        monkeypatch.setattr("memory.synthesis_memory.SynthesisMemory", mem_spy)
        proc = _make_processor(chroma_store=MagicMock())

        await proc._run_synthesis_dreaming()

        assert mem_spy.audit_stats_calls == 1
        assert filter_spy.construct_calls == 1
        assert gen_spy.construct_calls == 1

    @pytest.mark.asyncio
    async def test_audit_disabled_skips_read_dreaming_proceeds(self, monkeypatch):
        _patch_pooled_only(monkeypatch, audit_enabled=False)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )
        mem_spy = _make_memory_spy({})
        monkeypatch.setattr("memory.synthesis_memory.SynthesisMemory", mem_spy)
        proc = _make_processor(chroma_store=MagicMock())

        await proc._run_synthesis_dreaming()

        assert mem_spy.audit_stats_calls == 0  # no audit read at all
        assert filter_spy.construct_calls == 1
        assert gen_spy.construct_calls == 1

    @pytest.mark.asyncio
    async def test_both_generator_flags_false_early_return_nothing_constructed(
        self, monkeypatch
    ):
        monkeypatch.setattr("config.app_config.SYNTHESIS_GENERATOR_ENABLED", False)
        monkeypatch.setattr("config.app_config.SYNTHESIS_POOLED_ENABLED", False)
        filter_spy = _make_filter_spy()
        gen_spy = _make_generator_spy(["candidate"])
        mem_spy = _make_memory_spy({})
        monkeypatch.setattr("knowledge.synthesis_filter.SynthesisFilter", filter_spy)
        monkeypatch.setattr(
            "knowledge.synthesis_pooled_generator.PooledConceptSynthesisGenerator",
            gen_spy,
        )
        monkeypatch.setattr("memory.synthesis_memory.SynthesisMemory", mem_spy)
        proc = _make_processor(chroma_store=MagicMock())

        await proc._run_synthesis_dreaming()

        assert mem_spy.construct_calls == 0
        assert filter_spy.construct_calls == 0
        assert gen_spy.construct_calls == 0
