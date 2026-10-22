"""Shutdown writers never build DERIVED artifacts from owner [test] probes
(incident 2026-10-16; bug classes BC-58 one predicate / every derived-artifact
reader, BC-81 second corpus instance in a live process, BC-91-adjacent
message truth).

Evidence: at the 12:54 shutdown the only conversation of the day was an owner
probe wrapped [test]...[/test]. Fact extraction honoured the envelope; the
daily note and the narrative rebuilt from it did not. These tests drive the
DEPLOYED functions (real CorpusManager on a tmp file, the real
ShutdownProcessor methods, the real DailyNotesGenerator, the real
_do_shutdown_async) and assert no probe text reaches an LLM prompt.
Fixture values are invented.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import memory.shutdown_processor as sp
from memory.corpus_manager import CorpusManager
from memory.fact_source import is_test_turn

PROBE = "What was I stressed about in early October, and is any of it still open?"
PROBE_WRAPPED = f"[test] {PROBE} [/test]"
PROBE_BLOCK = f"[test]\n{PROBE}\n[/test]"
PROBE_INDENTED = f"   [test]\n   {PROBE}\n   [/test]"
REAL = "Rowan and I finished the Tamsin chapter outline today."

NOW = datetime.now().replace(microsecond=0)
# Fixture turns sit up to an hour before NOW; in the first hours after midnight they would fall on the
# previous day while the tests ask for NOW.date() (CI at 00:26 UTC, 2026-10-22). Keep the whole fixture
# window inside one past day.
if NOW.hour < 2:
    NOW = (NOW - timedelta(days=1)).replace(hour=23, minute=0, second=0)


def _entry(q, minutes=0, **extra):
    e = {"query": q, "response": f"Reply to: {q[:20]}", "type": "episodic",
         "timestamp": NOW - timedelta(minutes=60 - minutes), "tags": []}
    e.update(extra)
    return e


def _corpus(tmp_path, entries):
    cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    cm.corpus = list(entries)
    cm.invalidate_read_cache()
    return cm


# ---------------------------------------------------------------- predicate

@pytest.mark.parametrize("text", [PROBE_WRAPPED, PROBE_BLOCK, PROBE_INDENTED,
                                  f"[test]{PROBE}[/test]", f"{REAL}\n{PROBE_WRAPPED}"])
def test_is_test_turn_detects_inline_block_and_wrapped(text):
    assert is_test_turn({"query": text})
    assert is_test_turn({"user": text})


def test_is_test_turn_ignores_plain_and_dangling_envelope():
    assert not is_test_turn({"query": REAL})
    assert not is_test_turn({"query": "[test] never closed"})
    assert not is_test_turn({})


def test_is_test_turn_authored_text_has_precedence_over_merged_query():
    # user_text is authoritative even when the merged attachment blob differs
    assert not is_test_turn({"user_text": REAL, "query": f"{REAL}\n{PROBE_WRAPPED}"})
    assert is_test_turn({"user_text": PROBE_WRAPPED, "query": "merged blob without markers"})
    assert not is_test_turn({"user_text": "", "query": PROBE_WRAPPED})  # attachment-only turn


# ---------------------------------------------------------------- accessor

def test_derivable_excludes_test_and_quarantined_visible_unchanged(tmp_path):
    entries = [_entry(REAL, 1), _entry(PROBE_WRAPPED, 2), _entry("hidden one", 3, curation_quarantined=True),
               _entry(PROBE_BLOCK, 4)]
    cm = _corpus(tmp_path, entries)
    assert [e["query"] for e in cm.derivable_entries()] == [REAL]
    # retrieval visibility is unchanged: probes stay (an in-session probe can test recall)
    assert [e["query"] for e in cm.visible_entries()] == [REAL, PROBE_WRAPPED, PROBE_BLOCK]
    # the recent window for derived artifacts drops probes BEFORE counting
    assert [e["query"] for e in cm.get_recent_derivable(1)] == [REAL]
    assert PROBE_WRAPPED in [e["query"] for e in cm.get_recent_memories(5)]


# ---------------------------------------------------------------- shutdown processor sites

class _EchoModel:
    """Misbehaving-by-design fake: records every prompt it is handed."""

    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kw):
        self.prompts.append(str(prompt))
        return "[]"


class _Consolidator:
    consolidation_threshold = 3

    def __init__(self):
        self.seen = []

    async def consolidate_memories(self, content_list):
        self.seen.append(content_list)
        return SimpleNamespace(content="A summary of the real work done across these sessions.")


def _proc(cm, model=None, cons=None, session_start=None):
    return sp.ShutdownProcessor(cm, None, cons or _Consolidator(), None, model or _EchoModel(), None, None,
                                session_start or (NOW - timedelta(hours=3)))


def _mixed(tmp_path, n_real=4):
    entries = [_entry(f"{REAL} (part {i})", i) for i in range(n_real)]
    entries.insert(2, _entry(PROBE_WRAPPED, 10))
    entries.append(_entry(PROBE_BLOCK, 20))
    return _corpus(tmp_path, entries)


def _no_probe(text):
    return PROBE not in text and "[test]" not in text.lower()


def test_site_summary_selection_excludes_probe(tmp_path):
    cm = _mixed(tmp_path, n_real=6)
    proc = _proc(cm)
    sel = proc._select_uncovered.__func__  # noqa: F841 - deployed method used below
    # coverage is judged by a chroma store; stub the reader only, not the selection
    import memory.summary_coverage as cov
    orig = cov.covered_ranges
    cov.covered_ranges = lambda store: []
    try:
        non_summ, N, wm, blocks = proc._select_uncovered(5)
    finally:
        cov.covered_ranges = orig
    assert all(_no_probe(e["query"]) for e in non_summ)
    assert all(_no_probe(e["query"]) for b in blocks for e in b)
    assert len(non_summ) == 6


def test_summary_coverage_rule_probe_never_counts_toward_a_block_or_stalls(tmp_path):
    """RULE: probes are removed from the pool BEFORE blocks are cut. They neither
    count toward N nor appear in a block; a trailing probe newer than the real
    turns is simply not a candidate, so the content-time watermark (anchored on
    real turns) can never be held back or skipped by it. A pool of ONLY probes
    produces no block and no summary call."""
    import memory.summary_coverage as cov
    orig = cov.covered_ranges
    cov.covered_ranges = lambda store: []
    try:
        # 2 real + 5 probes with N=3: probes must not complete a block
        entries = [_entry(REAL, 1), _entry(REAL + " two", 2)] + [_entry(PROBE_WRAPPED, 3 + i) for i in range(5)]
        proc = _proc(_corpus(tmp_path, entries))
        _, N, _, blocks = proc._select_uncovered(5)
        assert N == 3 and blocks == []
        # only probes -> nothing at all
        proc2 = _proc(_corpus(tmp_path, [_entry(PROBE_WRAPPED, i) for i in range(6)]))
        assert proc2._select_uncovered(5)[3] == []
        # real turns after a probe: the block is cut from real turns alone, in order
        entries = [_entry(REAL + " a", 1), _entry(PROBE_WRAPPED, 2), _entry(REAL + " b", 3),
                   _entry(REAL + " c", 4), _entry(PROBE_BLOCK, 5)]
        cons = _Consolidator()
        proc3 = _proc(_corpus(tmp_path, entries), cons=cons)
        _, _, _, blocks = proc3._select_uncovered(5)
        assert [[e["query"][-1] for e in b] for b in blocks] == [["a", "b", "c"]]
    finally:
        cov.covered_ranges = orig


def test_site_summary_prompt_excludes_probe(tmp_path):
    cm = _mixed(tmp_path, n_real=6)
    cons = _Consolidator()
    proc = _proc(cm, cons=cons)
    import memory.summary_coverage as cov
    orig = cov.covered_ranges
    cov.covered_ranges = lambda store: []
    proc._store_summary = lambda text, block: True
    try:
        created = asyncio.run(proc.summarize_uncovered_blocks(5))
    finally:
        cov.covered_ranges = orig
    assert created == 2
    flat = " ".join(c["content"] for call in cons.seen for c in call)
    assert _no_probe(flat)


def test_site_reflection_excludes_probe(tmp_path):
    cm = _mixed(tmp_path)
    model = _EchoModel()
    proc = _proc(cm, model=model)
    stored = []

    async def _add_reflection(text, **kw):
        stored.append(text)
        return True

    proc._storage = SimpleNamespace(add_reflection=_add_reflection)
    asyncio.run(proc.run_shutdown_reflection(None, None))
    assert any(REAL in p for p in model.prompts), "reflection should have run on the real turns"
    assert all(_no_probe(p) for p in model.prompts)


def test_site_llm_facts_excludes_probe(tmp_path, monkeypatch):
    cm = _mixed(tmp_path)
    model = _EchoModel()
    proc = _proc(cm, model=model)
    monkeypatch.setenv("LLM_FACTS_ENABLED", "1")
    asyncio.run(proc._extract_llm_facts(None))
    assert any(REAL in p for p in model.prompts)
    assert all(_no_probe(p) for p in model.prompts)


def test_site_regex_facts_excludes_probe(tmp_path):
    cm = _mixed(tmp_path)
    seen = []

    class _Extractor:
        async def extract_facts(self, text, *a, **k):  # the deployed call site awaits this
            seen.append(text)
            return []

    proc = _proc(cm)
    proc.fact_extractor = _Extractor()
    asyncio.run(proc._extract_session_facts(None))
    assert seen, "extractor should have been handed the real turns"
    assert any(REAL in t for t in seen if isinstance(t, str))
    assert all(_no_probe(t) for t in seen if isinstance(t, str))


def test_site_procedural_skills_excludes_probe(tmp_path):
    cm = _mixed(tmp_path)
    model = _EchoModel()
    proc = _proc(cm, model=model)
    asyncio.run(proc._extract_procedural_skills(None))
    assert any(REAL in p for p in model.prompts)
    assert all(_no_probe(p) for p in model.prompts)


def test_site_threads_excludes_probe(tmp_path):
    cm = _mixed(tmp_path)
    model = _EchoModel()
    proc = _proc(cm, model=model)
    proc.thread_store = SimpleNamespace(list_open_threads=lambda *a, **k: [])
    asyncio.run(proc._process_open_threads(None))
    assert any(REAL in p for p in model.prompts)
    assert all(_no_probe(p) for p in model.prompts)


def test_site_proposals_selection_excludes_probe(tmp_path, monkeypatch):
    cm = _mixed(tmp_path)
    proc = _proc(cm, model=_EchoModel())
    captured = {}

    async def _cold(sess_items, generator, dedup_context):
        captured["items"] = sess_items
        return []

    proc._generate_proposals_cold = _cold
    import knowledge.proposal_generator as pg
    import memory.proposal_store as ps
    monkeypatch.setattr(pg, "GoalDirectedGenerator", lambda **k: SimpleNamespace())
    monkeypatch.setattr(ps, "ProposalStore", lambda **k: SimpleNamespace(get_for_dedup=lambda: ""))
    monkeypatch.setattr("config.app_config.CODE_PROPOSALS_ENABLED", True, raising=False)
    asyncio.run(proc._generate_proposals(None))
    assert len(captured.get("items", [])) == 4
    for e in captured["items"]:
        assert _no_probe(e["query"])


# ---------------------------------------------------------------- recent-window derived readers

def test_consolidator_recent_window_excludes_probe(tmp_path):
    from memory.memory_consolidator import MemoryConsolidator

    cm = _mixed(tmp_path, n_real=4)
    seen = []

    class _C(MemoryConsolidator):
        async def consolidate_memories(self, recent):
            seen.append([m["query"] for m in recent])
            return SimpleNamespace(content="summary text")

    c = _C.__new__(_C)
    c.consolidation_threshold = 3
    c.lookback = 10
    c.model_manager = None
    stored = []
    cm.add_summary = stored.append  # type: ignore[assignment]
    assert asyncio.run(c.maybe_consolidate(cm)) is True
    assert seen and all(_no_probe(q) for q in seen[0]) and len(seen[0]) == 4


# ---------------------------------------------------------------- daily note

_NOTE = (
    "## Summary\nA quiet day of synthetic conversations about test coverage.\n\n"
    "## Main Quest: Testing\n- Verified the guard\n\n## Side Quests\nNone today.\n\n"
    "## Life Events\n- **Work**: Wrote tests.\n\n## Emotional State\nFocused.\n\n"
    "## Key Decisions\nNone.\n\n## Knowledge Gained\nNone.\n\n## Open Threads\nAll resolved.\n\n"
    "## Intensity: 2/10\n"
)


class _NoteModel:
    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kw):
        self.prompts.append(str(prompt))
        return _NOTE


class _Profile:
    def get_current_view(self):
        return {}


def _gen(tmp_path, cm, model):
    from unittest.mock import AsyncMock
    from utils.daily_notes_generator import DailyNotesGenerator

    g = DailyNotesGenerator(corpus_manager=cm, model_manager=model, vault_path=str(tmp_path / "vault"),
                            user_profile=_Profile())
    g.tag_generation_enabled = False
    g._trigger_narrative_refresh = AsyncMock(return_value=None)
    return g


@pytest.mark.parametrize("probe", [PROBE_WRAPPED, PROBE_BLOCK, PROBE_INDENTED])
def test_daily_note_only_probe_takes_the_no_conversations_path(tmp_path, probe):
    cm = _corpus(tmp_path, [_entry(probe, 5)])
    model = _NoteModel()
    gen = _gen(tmp_path, cm, model)
    result = asyncio.run(gen.generate_for_date(NOW.date()))
    assert result.skipped_reason == "no_conversations" and not result.success
    assert model.prompts == []  # no LLM call
    assert not list((tmp_path / "vault").rglob("*.md"))  # nothing written


def test_daily_note_mixed_day_prompt_has_real_turn_not_probe(tmp_path):
    cm = _corpus(tmp_path, [_entry(REAL, 5), _entry(PROBE_WRAPPED, 6)])
    model = _NoteModel()
    gen = _gen(tmp_path, cm, model)
    result = asyncio.run(gen.generate_for_date(NOW.date()))
    assert result.conversation_count == 1
    assert model.prompts and any(REAL in p for p in model.prompts)
    assert all(_no_probe(p) for p in model.prompts)


# ---------------------------------------------------------------- main.py shutdown sequence

@pytest.fixture
def shutdown_env(monkeypatch):
    import main

    built = []

    class _Gen:
        def __init__(self, **kw):
            built.append(kw)

        async def generate_for_date(self, d, force=False):
            return SimpleNamespace(success=False, skipped_reason="no_conversations", error=None, conversation_count=0)

    async def _noop(*a, **k):
        return None

    async def _pending(timeout=10.0):
        return None

    live_cm = object()
    ms = SimpleNamespace(
        corpus_manager=live_cm,
        run_shutdown_reflection=_noop,
        process_shutdown_memory=_noop,
        run_synthesis_dreaming=_noop,
    )
    orch = SimpleNamespace(memory_system=ms, model_manager=object())
    monkeypatch.setattr(main.daily_notes_generator, "DailyNotesGenerator", _Gen)
    monkeypatch.setattr(main.app_config, "DAILY_NOTES_ENABLED", True)
    monkeypatch.setattr(main.app_config, "CURATION_ENABLED", False)
    monkeypatch.setattr(main.app_config, "BACKUP_ENABLED", True)
    monkeypatch.setattr("gui.handlers.wait_for_pending_storage", _pending)
    monkeypatch.setattr(main.backup_manager, "run_shutdown_backup",
                        lambda: SimpleNamespace(ok=True, path="/tmp/bk", chroma_included=False,
                                                skipped_reason=None, error=None))
    return main, orch, built, live_cm


def test_backup_line_says_exiting_only_on_a_real_exit(shutdown_env, monkeypatch, capsys):
    main, orch, _, _ = shutdown_env
    monkeypatch.setattr(main, "_process_exiting", False)  # idle-monitor flush: the server stays up
    asyncio.run(main._do_shutdown_async(orch, [], []))
    idle = capsys.readouterr().out
    assert "Backup written: /tmp/bk — server still running." in idle and "exiting." not in idle
    monkeypatch.setattr(main, "_process_exiting", True)  # signal handler / lifespan
    asyncio.run(main._do_shutdown_async(orch, [], []))
    assert "Backup written: /tmp/bk — exiting." in capsys.readouterr().out


def test_daily_note_step_gets_the_live_corpus_manager_and_builds_no_second_one(shutdown_env, monkeypatch):
    main, orch, built, live_cm = shutdown_env
    constructed = []
    real_init = CorpusManager.__init__
    monkeypatch.setattr(CorpusManager, "__init__",
                        lambda self, *a, **k: (constructed.append(1), real_init(self, *a, **k))[1])
    asyncio.run(main._do_shutdown_async(orch, [], []))
    assert built and built[0]["corpus_manager"] is live_cm
    assert constructed == []


# ---------------------------------------------------------------- explicit list entry (referee round 1)

def test_explicit_session_list_is_filtered_at_the_process_entry(tmp_path, monkeypatch):
    """An explicit [real, test] list must not carry the probe into any LLM call."""
    cm = _corpus(tmp_path, [])
    model = _EchoModel()
    proc = _proc(cm, model=model)
    monkeypatch.setenv("LLM_FACTS_ENABLED", "1")
    items = [_entry(REAL, 1), _entry(PROBE_WRAPPED, 2)]
    import memory.summary_coverage as cov
    monkeypatch.setattr(cov, "covered_ranges", lambda store: [])
    asyncio.run(proc.process_shutdown_memory(items))
    asyncio.run(proc.run_shutdown_reflection(items, None))
    assert any(REAL in p for p in model.prompts)
    assert all(_no_probe(p) for p in model.prompts)
