"""Lane BF: offline summary backlog catch-up through THE deployed summary path (BC-95, BC-27).

Real CorpusManager (tmp file), real ephemeral MultiCollectionChromaStore, real
MemoryConsolidator + ShutdownProcessor; only the LLM is faked.  Synthetic text only.

Original batch label: oct15 (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""
import asyncio
import importlib.util
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
from chromadb.config import Settings

import memory.shutdown_processor as sp
from memory import summary_coverage as cov
from memory.corpus_manager import CorpusManager
from memory.memory_consolidator import MemoryConsolidator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

_spec = importlib.util.spec_from_file_location(
    "backfill_summaries", Path(__file__).resolve().parents[2] / "scripts/backfill_summaries.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

T0 = datetime(2026, 9, 20, 8, 0)
N = 20


class _Emb:
    def __call__(self, input):
        return [[1.0, 0.0] for _ in input]

    def name(self):
        return "fixed-test-embedder"


def _store():
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    s = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    s.collections = {n: client.create_collection(name=f"{n}_{uuid.uuid4().hex[:8]}", embedding_function=_Emb())
                     for n in ("conversations", "summaries")}
    return s


def _conv(i):
    return {"query": f"STAT 5100 question number {i}", "response": f"STAT 5100 answer number {i}",
            "timestamp": (T0 + timedelta(hours=i)).isoformat(), "type": "episodic", "tags": []}


class _Model:
    """Fake LLM: records prompts; optionally raises (the dry run must never reach it)."""
    def __init__(self, raises=False):
        self.prompts, self.raises = [], raises

    async def generate_once(self, prompt, **kw):
        if self.raises:
            raise AssertionError("model called")
        self.prompts.append(prompt)
        return f"- The user worked on STAT 5100 coursework, summary call {len(self.prompts)}, in detail."


def _world(tmp_path, n_new=100, n_old=8, extra=None):
    """Corpus of n_old entries before the watermark + n_new after; a Chroma conversation per entry."""
    store = _store()
    entries = [_conv(i) for i in range(n_old + n_new)]
    if extra:
        extra(entries)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    cm = CorpusManager(str(path))
    for e in entries:
        store.add_to_collection("conversations", f"User: {e['query']}\nAssistant: {e['response']}",
                                {"timestamp": e["timestamp"], "query": e["query"]})
    wm = T0 + timedelta(hours=n_old - 1)
    store.add_to_collection("summaries", "- An earlier consolidated summary of STAT 5100 work in detail.", {
        "timestamp": "2026-10-01T00:00:00", "tags": "summary:consolidated,source:consolidator",
        "temporal_anchor_start": T0.isoformat(), "temporal_anchor_end": wm.isoformat()})
    return store, cm, path


def _install(monkeypatch, store, cm, model, tmp_path, running=False):
    monkeypatch.setattr(bf, "_open_store", lambda: store)
    monkeypatch.setattr(bf, "_open_corpus", lambda: cm)
    monkeypatch.setattr(bf, "_open_model_manager", lambda: model)
    monkeypatch.setattr(bf, "_open_claim_index", lambda: None)
    monkeypatch.setattr(bf, "_open_entity_resolver", lambda: None)
    monkeypatch.setattr(bf, "_summary_block_size", lambda: N)
    monkeypatch.setattr(bf, "daemon_running", lambda: running)


def _n_summaries(store):
    return len(store.collections["summaries"].get()["ids"])


def _proc(store, cm, model):
    return bf.make_processor(store, cm, consolidator=MemoryConsolidator(model, consolidation_threshold=N))


def test_shutdown_calls_the_shared_method_and_creates_min_cap_backlog(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SUMMARY_MAX_BLOCKS_PER_SHUTDOWN", 2)
    store, cm, _ = _world(tmp_path)  # 5 full blocks of backlog
    model = _Model()
    p = _proc(store, cm, model)
    calls = []
    real = p.summarize_uncovered_blocks

    async def _spy(cap):
        calls.append(cap)
        return await real(cap)
    p.summarize_uncovered_blocks = _spy
    for name in ("_extract_session_facts", "_extract_llm_facts",
                 "_extract_procedural_skills", "_generate_proposals", "_check_implementation_tracking",
                 "_process_open_threads", "_run_wiki_enrichment", "_run_wikidata_enrichment",
                 "_run_cross_collection_dedup"):
        async def _noop(*a, **k):
            return None
        setattr(p, name, _noop)
    p._save_knowledge_graph = lambda: None
    p._maybe_spawn_agent_branch = lambda: None
    before = _n_summaries(store)
    asyncio.run(p.process_shutdown_memory([]))
    assert calls == [2]
    assert _n_summaries(store) - before == 2 and len(model.prompts) == 2


def test_plan_matches_what_is_summarized_and_makes_no_model_call(tmp_path):
    store, cm, _ = _world(tmp_path)
    p = _proc(store, cm, _Model(raises=True))
    corpus_before = Path(cm.corpus_file).read_bytes()
    plan = p.plan_uncovered_blocks(3)
    assert [x["n_entries"] for x in plan] == [N, N, N]
    assert plan[0]["start"] == T0 + timedelta(hours=8)
    assert plan[0]["end"] == T0 + timedelta(hours=27)
    assert plan[1]["start"] > plan[0]["end"]  # oldest first, disjoint
    assert _n_summaries(store) == 1 and Path(cm.corpus_file).read_bytes() == corpus_before
    # the real generator then summarizes exactly those spans
    model = _Model()
    p2 = _proc(store, cm, model)
    assert asyncio.run(p2.summarize_uncovered_blocks(3)) == 3
    anchors = sorted((m["temporal_anchor_start"], m["temporal_anchor_end"])
                     for m in store.collections["summaries"].get()["metadatas"])[1:]
    assert anchors == [(x["start"].isoformat(), x["end"].isoformat()) for x in plan]


def test_unreadable_coverage_plans_and_creates_nothing(tmp_path):
    store, cm, _ = _world(tmp_path)
    p = _proc(store, cm, _Model(raises=True))
    p.chroma_store = SimpleNamespace(list_all=lambda n: (_ for _ in ()).throw(RuntimeError("down")))
    assert p.plan_uncovered_blocks(5) == []
    assert asyncio.run(p.summarize_uncovered_blocks(5)) == 0


def test_limit_runs_drain_backlog_without_overlap(tmp_path, monkeypatch, capsys):
    store, cm, _ = _world(tmp_path)
    model = _Model()
    _install(monkeypatch, store, cm, model, tmp_path)
    counts = []
    for _ in range(4):
        assert bf.main(["--limit", "2", "--apply"]) == 0
        counts.append(_n_summaries(store) - 1)
    assert counts == [2, 4, 5, 5]  # 2, then the NEXT 2, then 1, then 0
    mds = [m for m in store.collections["summaries"].get()["metadatas"] if m.get("temporal_anchor_end")]
    spans = sorted((m["temporal_anchor_start"], m["temporal_anchor_end"]) for m in mds)
    assert len(spans) == 6
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 > e1  # no overlap, no duplicate coverage (BC-95)
    assert len(model.prompts) == 5
    corpus_summaries = [e for e in json.loads(Path(cm.corpus_file).read_text()) if "summary" in str(e.get("type"))]
    assert len(corpus_summaries) == 5 and all(e.get("covers_end") for e in corpus_summaries)
    assert "New watermark" in capsys.readouterr().out


def test_dry_run_writes_nothing_and_never_builds_the_model(tmp_path, monkeypatch, capsys):
    store, cm, path = _world(tmp_path)
    _install(monkeypatch, store, cm, None, tmp_path)
    monkeypatch.setattr(bf, "_open_model_manager", lambda: pytest.fail("model manager built in dry run"))
    before = path.read_bytes()
    assert bf.main(["--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "planned blocks: 2" in out and "estimated LLM calls: 2" in out and "DRY RUN" in out
    assert path.read_bytes() == before and _n_summaries(store) == 1


def test_daemon_running_refuses_apply(tmp_path, monkeypatch):
    store, cm, path = _world(tmp_path)
    _install(monkeypatch, store, cm, _Model(raises=True), tmp_path, running=True)
    before = path.read_bytes()
    assert bf.main(["--apply"]) == 1
    assert path.read_bytes() == before and _n_summaries(store) == 1


def test_preflight_refuses_unflagged_corpus_copy_then_proceeds_when_flagged(tmp_path, monkeypatch, capsys):
    secret = "ZEBRA-QUARANTINED-CLAIM"
    store, cm, path = _world(tmp_path)
    cm.corpus[30]["query"] = secret
    cm.corpus[30]["response"] = secret
    cm.save_corpus()
    ids = store.collections["conversations"].get(include=["documents"])
    target = next(i for i, d in zip(ids["ids"], ids["documents"]) if "number 30" in d)
    # the Chroma copy of that turn carries the (edited) text so the matcher can join the copies
    store.collections["conversations"].update(ids=[target], documents=[f"User: {secret}\nAssistant: {secret}"],
                                              metadatas=[{"timestamp": cm.corpus[30]["timestamp"].isoformat(), "query": secret}])
    assert store.update_metadata("conversations", target, {"curation_quarantined": True})
    model = _Model()
    _install(monkeypatch, store, cm, model, tmp_path)
    before = path.read_bytes()
    assert bf.main(["--apply"]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert path.read_bytes() == before and _n_summaries(store) == 1 and model.prompts == []

    cm.corpus[30]["curation_quarantined"] = True  # what scripts/quarantine_memory.py does to the corpus copy
    cm.save_corpus()
    assert bf.main(["--apply"]) == 0
    assert model.prompts and all(secret not in p for p in model.prompts)
    assert _n_summaries(store) > 1


def test_no_corpus_copy_does_not_block(tmp_path, monkeypatch):
    store, cm, _ = _world(tmp_path)
    did = store.add_to_collection("conversations", "User: orphan\nAssistant: only in chroma",
                                  {"timestamp": (T0 + timedelta(hours=50)).isoformat(), "query": "orphan"})
    store.update_metadata("conversations", did, {"curation_quarantined": True})
    assert bf.preflight(store, cm, cov.coverage_watermark(cov.covered_ranges(store))) == []


class _FlakyModel(_Model):
    """Fake LLM whose Nth call returns nothing (an empty/failed summary)."""
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on, self.calls = fail_on, 0

    async def generate_once(self, prompt, **kw):
        self.calls += 1
        if self.calls == self.fail_on:
            return ""
        return await super().generate_once(prompt, **kw)


def test_failed_block_stops_the_run_and_is_retried_not_leapfrogged(tmp_path):
    """Referee 2026-10-15: coverage is a watermark, so a block stored AFTER a failed one
    would move it past the gap and the failed block would never be summarized."""
    store, cm, _ = _world(tmp_path)  # 5 full blocks
    plan = _proc(store, cm, _Model(raises=True)).plan_uncovered_blocks(5)
    model = _FlakyModel(fail_on=2)
    p = _proc(store, cm, model)
    assert asyncio.run(p.summarize_uncovered_blocks(5)) == 1  # block 1 stored, block 2 failed → stop
    assert _n_summaries(store) == 2
    assert asyncio.run(p.summarize_uncovered_blocks(5)) == 4  # resumes AT block 2
    anchors = sorted((m["temporal_anchor_start"], m["temporal_anchor_end"])
                     for m in store.collections["summaries"].get()["metadatas"])[1:]
    assert anchors == [(x["start"].isoformat(), x["end"].isoformat()) for x in plan]


def test_store_summary_reports_whether_the_coverage_copy_landed(tmp_path):
    store, cm, _ = _world(tmp_path, n_new=20)
    p = _proc(store, cm, _Model())
    block = [e for e in cm.corpus][8:28]
    text = "- The user reviewed STAT 5100 lecture notes on variance in some detail."
    assert p._store_summary(text, block) is True
    assert p._store_summary(text, block) is False  # duplicate → not stored
    assert p._store_summary("-", block) is False   # junk → not stored
