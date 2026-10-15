"""Lane SUM: summary coverage by content time (BC-95/BC-96/BC-52)."""
from datetime import datetime, timedelta

from memory import summary_coverage as sc
from memory.memory_retriever import MemoryRetriever, _summary_item
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore as S

T0 = datetime(2026, 8, 1, 12, 0)


def _entries(n, start=0):
    return [{"id": i, "timestamp": T0 + timedelta(hours=i)} for i in range(start, start + n)]


def _ts(e):
    return e["timestamp"]


def _md(a, b, created="2026-10-13T18:55:00", tags="summary:consolidated,source:consolidator"):
    return {"timestamp": created, "tags": tags,
            "temporal_anchor_start": a.isoformat(), "temporal_anchor_end": b.isoformat()}


class FakeStore:
    def __init__(self, mds, raises=False):
        self.mds, self.raises = mds, raises

    def list_all(self, name):
        if self.raises:
            raise RuntimeError("chroma down")
        return [{"id": str(i), "content": "x", "metadata": m} for i, m in enumerate(self.mds)]


def test_trim_shift_summarizes_only_uncovered_content_once():
    entries = _entries(60)
    ranges = sc.covered_ranges(FakeStore([_md(entries[0]["timestamp"], entries[39]["timestamp"])]))
    # corpus rolls: oldest 25 trimmed -> positions shift, old code re-summarized 25..39
    window = entries[25:]
    blocks = sc.uncovered_blocks(window, sc.coverage_watermark(ranges), 20, 5, _ts)
    assert [[e["id"] for e in b][0] for b in blocks] == [40]
    assert len(blocks) == 1 and [e["id"] for e in blocks[0]] == list(range(40, 60))
    # after those are covered, nothing is due: no re-summary
    ranges2 = ranges + [(entries[40]["timestamp"], entries[59]["timestamp"])]
    assert sc.uncovered_blocks(window, sc.coverage_watermark(ranges2), 20, 5, _ts) == []


def test_unreadable_store_is_none_not_empty():
    assert sc.covered_ranges(FakeStore([], raises=True)) is None
    assert sc.covered_ranges(None) is None
    assert sc.covered_ranges(FakeStore([])) == []


def test_fresh_install_two_blocks_oldest_first_and_cap():
    e = _entries(45)
    b = sc.uncovered_blocks(e, None, 20, 5, _ts)
    assert [x[0]["id"] for x in b] == [0, 20] and all(len(x) == 20 for x in b)
    assert len(sc.uncovered_blocks(e, None, 20, 1, _ts)) == 1
    assert sc.remaining_full_blocks(e, None, 20, _ts) == 2


def test_quarantined_and_unanchored_rows():
    md_q = dict(_md(T0, T0 + timedelta(hours=3)), curation_quarantined=True)
    md_legacy = {"timestamp": "2026-02-01T00:00:00", "tags": "summary:consolidated,source:consolidator"}
    md_other = _md(T0, T0, tags="reflection")
    assert len(sc.ranges_from_metadatas([md_q, md_legacy, md_other])) == 1


def test_retriever_item_content_time_covers_and_legacy():
    row = {"content": "c", "metadata": _md(T0, T0 + timedelta(days=2))}
    it = _summary_item(row)
    assert it["timestamp"] == (T0 + timedelta(days=2)).isoformat()
    assert it["created_at"] == "2026-10-13T18:55:00"
    assert it["covers"] == "2026-08-01 – 2026-08-03"
    legacy = _summary_item({"content": "c", "metadata": {"timestamp": "2026-02-01T00:00:00"}})
    assert legacy["timestamp"] == "2026-02-01T00:00:00" and "covers" not in legacy


def test_get_recent_time_keys_default_unchanged_and_override():
    rows = [{"id": "a", "metadata": {"timestamp": "2026-10-01T00:00:00", "k": "2026-01-01T00:00:00"}},
            {"id": "b", "metadata": {"timestamp": "2026-09-01T00:00:00", "k": "2026-02-01T00:00:00"}}]
    st = FakeStore([])
    st.list_all = lambda n: [dict(r) for r in rows]
    assert [r["id"] for r in S.get_recent(st, "x", 2)] == ["a", "b"]
    assert [r["id"] for r in S.get_recent(st, "x", 2, ("k", "timestamp"))] == ["b", "a"]


def test_formatter_renders_covers_never_creation_stamp():
    from core.prompt.formatter import _format_summary_section
    it = {"content": "Homework due on the 27th.", "timestamp": "2026-09-19T17:03:47",
          "created_at": "2026-10-13T18:55:00", "covers": "2026-09-17 – 2026-09-19"}
    out = _format_summary_section([it], "RECENT SUMMARIES")
    assert "1) [covers 2026-09-17 – 2026-09-19] " in out
    assert "2026-10-13" not in out and "17:03" not in out
    refl = _format_summary_section([{"content": "r", "timestamp": "2026-10-01"}], "R")
    assert "1) 2026-10-01: r" in refl


def test_overlap_resummaries_exact_time_not_same_date():
    h = lambda x: T0 + timedelta(hours=x)
    rows = [
        {"id": "first", "created": h(100), "start": h(0), "end": h(10), "text": ""},
        {"id": "dup", "created": h(200), "start": h(2), "end": h(8), "text": ""},
        {"id": "same_day_later", "created": h(300), "start": h(11), "end": h(12), "text": ""},
    ]
    assert [r["id"] for r in sc.overlap_resummaries(rows)] == ["dup"]


def test_report_script_on_temp_sqlite(tmp_path):
    import sqlite3, importlib.util
    from pathlib import Path
    db = tmp_path / "chroma.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE collections(id TEXT, name TEXT);
    CREATE TABLE segments(id TEXT, scope TEXT, collection TEXT);
    CREATE TABLE embeddings(id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
    CREATE TABLE embedding_metadata(id INTEGER, key TEXT, string_value TEXT, int_value INTEGER, float_value REAL);
    INSERT INTO collections VALUES('c1','summaries'),('c2','conversations');
    INSERT INTO segments VALUES('s1','METADATA','c1'),('s2','METADATA','c2');
    """)
    def add(rid, seg, eid, md):
        con.execute("INSERT INTO embeddings VALUES(?,?,?)", (rid, seg, eid))
        for k, v in md.items():
            con.execute("INSERT INTO embedding_metadata VALUES(?,?,?,NULL,NULL)", (rid, k, v))
    add(1, "s1", "first", _md(T0, T0 + timedelta(hours=10), created="2026-09-01T00:00:00") | {"chroma:document": "alpha"})
    add(2, "s1", "dup", _md(T0 + timedelta(hours=2), T0 + timedelta(hours=8), created="2026-10-01T00:00:00") | {"chroma:document": "beta"})
    add(3, "s2", "conv", {"timestamp": (T0 + timedelta(days=5)).isoformat()})
    con.commit(); con.close()
    spec = importlib.util.spec_from_file_location("rsc", Path(__file__).resolve().parents[2] / "scripts/report_summary_coverage.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    lines, dups = m.build_report(db)
    assert [d["id"] for d in dups] == ["dup"]
    assert "conversations newer than watermark: 1" in lines


# ---------------------------------------------------------------------------
# DEPLOYED process_shutdown_memory selection (fake corpus / chroma / consolidator)
# ---------------------------------------------------------------------------
import asyncio
from types import SimpleNamespace

import memory.shutdown_processor as sp


class _Corpus:
    def __init__(self, entries):
        self.corpus = list(entries)

    def visible_entries(self):  # mirrors CorpusManager.visible_entries
        return list(self.corpus)

    def add_summary(self, node):
        self.corpus.append(dict(node, tags=list(node.get("tags") or []) + ["@summary"]))

    def get_summaries(self, n):
        return [e for e in self.corpus if "summary" in str(e.get("type", ""))][:n]


class _Chroma(FakeStore):
    def __init__(self, mds, raises=False):
        super().__init__(mds, raises)
        self.added = []

    def add_to_collection(self, name, text, md):
        self.added.append((text, md))
        self.mds.append(md)
        return "id"


class _Consolidator:
    consolidation_threshold = 20

    def __init__(self):
        self.calls = []

    async def consolidate_memories(self, content_list):
        self.calls.append(len(content_list))
        return SimpleNamespace(content=f"Summary number {len(self.calls)}: the user discussed their project work in detail.")


def _proc(corpus, chroma, cons):
    p = sp.ShutdownProcessor(corpus, chroma, cons, None, None, None, None, datetime.now())
    for name in ("_extract_session_facts", "_extract_llm_facts", "_extract_behavioral_patterns",
                 "_extract_procedural_skills", "_generate_proposals", "_check_implementation_tracking",
                 "_process_open_threads", "_run_wiki_enrichment", "_run_wikidata_enrichment",
                 "_run_cross_collection_dedup"):
        async def _noop(*a, **k):
            return None
        setattr(p, name, _noop)
    p._save_knowledge_graph = lambda: None
    p._maybe_spawn_agent_branch = lambda: None
    return p


def _conv(i):
    return {"query": f"q{i}", "response": f"a{i}", "timestamp": T0 + timedelta(hours=i), "type": "episodic"}


def test_deployed_selection_after_corpus_trim_summarizes_only_new_content_once(monkeypatch):
    monkeypatch.setattr(sp, "SUMMARY_MAX_BLOCKS_PER_SHUTDOWN", 5)
    allc = [_conv(i) for i in range(60)]
    # block_n tags of the old design, covering entries 0-39, are gone with the trimmed window
    chroma = _Chroma([_md(allc[0]["timestamp"], allc[39]["timestamp"])])
    corpus = _Corpus(allc[25:])  # rolling window dropped the oldest 25
    cons = _Consolidator()
    asyncio.run(_proc(corpus, chroma, cons).process_shutdown_memory([]))
    assert cons.calls == [20]  # entries 40-59 only; 25-39 NOT re-summarized
    (text, md), = chroma.added
    assert md["temporal_anchor_start"] == allc[40]["timestamp"].isoformat()
    assert md["temporal_anchor_end"] == allc[59]["timestamp"].isoformat()
    assert "block_n" not in md["tags"]
    new = [e for e in corpus.corpus if e.get("covers_end")]
    assert len(new) == 1 and new[0]["covers_end"] == allc[59]["timestamp"].isoformat()
    assert not any("block_n" in t for t in new[0]["tags"])
    # a second shutdown finds nothing due
    cons2 = _Consolidator()
    asyncio.run(_proc(corpus, chroma, cons2).process_shutdown_memory([]))
    assert cons2.calls == []


def test_deployed_selection_unreadable_coverage_creates_nothing():
    corpus = _Corpus([_conv(i) for i in range(45)])
    cons = _Consolidator()
    chroma = _Chroma([], raises=True)
    asyncio.run(_proc(corpus, chroma, cons).process_shutdown_memory([]))
    assert cons.calls == [] and chroma.added == []


def test_corpus_side_recency_uses_covers_end():
    from memory.corpus_manager import CorpusManager
    cm = CorpusManager.__new__(CorpusManager)
    cm.corpus = [
        {"content": "old content, written late", "type": "summary", "tags": ["@summary"],
         "timestamp": datetime(2026, 10, 13), "covers_start": "2026-09-17T10:00:00", "covers_end": "2026-09-19T10:00:00"},
        {"content": "newer content, written early", "type": "summary", "tags": ["@summary"],
         "timestamp": datetime(2026, 10, 1), "covers_start": "2026-09-30T10:00:00", "covers_end": "2026-09-30T12:00:00"},
        {"content": "legacy", "type": "summary", "tags": ["@summary"], "timestamp": datetime(2026, 9, 1)},
    ]
    assert [e["content"][:6] for e in cm.get_summaries(3)] == ["newer ", "old co", "legacy"]
    from memory.memory_retriever import _corpus_summary_item
    it = _corpus_summary_item(cm.corpus[0])
    assert it["timestamp"] == "2026-09-19T10:00:00" and it["covers"] == "2026-09-17 – 2026-09-19"
    assert it["created_at"] == datetime(2026, 10, 13)
    assert _corpus_summary_item(cm.corpus[2]) is cm.corpus[2]
