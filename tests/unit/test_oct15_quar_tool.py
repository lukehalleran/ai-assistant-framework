"""Lane QUAR: quarantine_sync matcher + scripts/quarantine_memory.py (class: BC-26, BC-58).

Drives the DEPLOYED matcher, tool functions, MultiCollectionChromaStore.update_metadata (real
ephemeral Chroma) and CorpusManager.save_corpus on tmp stores - never the live data/.
Synthetic text only.
"""
import importlib.util
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from memory.corpus_manager import CorpusManager
from memory.quarantine_sync import match_corpus_index
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("quarantine_memory", _ROOT / "scripts" / "quarantine_memory.py")
qm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qm)

T = datetime(2026, 10, 14, 18, 30, 0)


def _doc(content, **md):
    return {"id": "d1", "content": content, "metadata": md}


def _conv(q, ts=T, **extra):
    return {"query": q, "response": "r", "timestamp": ts.isoformat(), "tags": [], **extra}


# --- matcher ------------------------------------------------------------------------------

def test_conversation_matches_by_thread_query_and_close_timestamp():
    corpus = [_conv("STAT 5100 due when", thread_id="t1"), _conv("other", thread_id="t2")]
    doc = _doc("User: STAT 5100 due when\nAssistant: x", query="STAT 5100 due when",
               thread_id="t1", timestamp=(T + timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S"))
    assert match_corpus_index("conversations", doc, corpus) == (0, "match")


def test_conversation_query_falls_back_to_document_user_line():
    corpus = [_conv("STAT 5100 due when")]
    doc = _doc("User: STAT 5100 due when\nAssistant: x", timestamp=T.strftime("%Y-%m-%d %H:%M:%S"))
    assert match_corpus_index("conversations", doc, corpus) == (0, "match")


def test_same_query_twice_within_window_without_thread_is_ambiguous():
    corpus = [_conv("same words"), _conv("same words", T + timedelta(seconds=30))]
    doc = _doc("x", query="same words", timestamp=T.strftime("%Y-%m-%d %H:%M:%S"))
    assert match_corpus_index("conversations", doc, corpus) == (None, "ambiguous")


def test_same_query_far_apart_is_disambiguated_by_timestamp_and_thread_narrows():
    corpus = [_conv("same words"), _conv("same words", T + timedelta(days=1))]
    doc = _doc("x", query="same words", timestamp=T.strftime("%Y-%m-%d %H:%M:%S"))
    assert match_corpus_index("conversations", doc, corpus) == (0, "match")
    both = [_conv("q", thread_id="a"), _conv("q", thread_id="b")]
    assert match_corpus_index("conversations", _doc("x", query="q", thread_id="b",
                                                    timestamp=T.isoformat()), both) == (1, "match")


def test_no_corpus_copy():
    assert match_corpus_index("conversations", _doc("x", query="absent", timestamp=T.isoformat()),
                              [_conv("present")]) == (None, "no_corpus_copy")


def test_summary_exact_content_and_reflection_via_original_text():
    corpus = [{"content": "  a summary body  ", "type": "summary", "timestamp": T.isoformat()},
              {"content": "reflection body", "type": "reflection", "timestamp": T.isoformat()},
              {"content": "reflection body", "type": "reflection", "timestamp": "2026-01-01T00:00:00"}]
    assert match_corpus_index("summaries", _doc("a summary body"), corpus) == (0, "match")
    refl = _doc("retrieval text soup", original_text="reflection body", timestamp=T.isoformat())
    assert match_corpus_index("reflections", refl, corpus) == (1, "match")  # timestamp tie-break
    assert match_corpus_index("reflections", _doc("reflection body", timestamp="2000-01-01"), corpus) == (None, "ambiguous")


def test_facts_unsupported_and_aware_vs_naive_timestamps_do_not_raise():
    assert match_corpus_index("facts", _doc("x"), []) == (None, "unsupported_collection")
    aware = datetime.now(timezone.utc)
    corpus = [_conv("q", aware.replace(tzinfo=None))]
    doc = _doc("x", query="q", timestamp=aware.isoformat())
    idx, reason = match_corpus_index("conversations", doc, corpus)
    assert reason in ("match", "no_corpus_copy")  # no TypeError either way
    corpus2 = [dict(_conv("q"), timestamp=aware.isoformat())]
    match_corpus_index("conversations", _doc("x", query="q", timestamp=T.isoformat()), corpus2)


# --- tool on tmp chroma + tmp corpus ------------------------------------------------------------

class _Embed:
    def __call__(self, input):
        return [[1.0, 0.0] for _ in input]

    def name(self):
        return "fixed-test-embedder"


@pytest.fixture
def env(tmp_path):
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {n: client.create_collection(name=f"{n}_{uuid.uuid4().hex[:8]}", embedding_function=_Embed())
                         for n in ("conversations", "summaries", "reflections", "facts")}
    store.collections["conversations"].add(
        ids=["c1"], documents=["User: STAT 5100 due when\nAssistant: x"],
        metadatas=[{"query": "STAT 5100 due when", "thread_id": "t1",
                    "timestamp": (T + timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")}])
    store.collections["summaries"].add(ids=["s1"], documents=["a summary body"], metadatas=[{"k": "v"}])
    store.collections["facts"].add(ids=["f1"], documents=["user | likes | tea"], metadatas=[{"k": "v"}])
    entries = [_conv("STAT 5100 due when", thread_id="t1"), _conv("untouched turn", T + timedelta(hours=1)),
               {"content": "a summary body", "type": "summary", "tags": ["@summary"], "timestamp": T.isoformat()}]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    cm = CorpusManager(str(path))
    return store, cm, path, tmp_path / "backups"


TARGETS = [("conversations", "c1"), ("summaries", "s1"), ("facts", "f1"), ("conversations", "nope")]


def _flag(store, coll, i):
    return bool((store.collections[coll].get(ids=[i], include=["metadatas"])["metadatas"][0] or {}).get("curation_quarantined"))


def test_dry_run_writes_nothing(env, capsys):
    store, cm, path, backups = env
    before = path.read_bytes()
    rows = qm.plan_targets(store, cm, TARGETS)
    qm._print_plan(rows, undo=False)
    out = capsys.readouterr().out
    assert "match" in out and "unsupported_collection" in out and "MISSING" in out
    assert path.read_bytes() == before and not backups.exists()
    assert not _flag(store, "conversations", "c1")


def test_apply_flips_both_copies_with_one_preimage_and_undo_restores(env):
    store, cm, path, backups = env
    rows = qm.plan_targets(store, cm, TARGETS)
    res = qm.run(rows, store, cm, undo=False, reason="owner review", backup_dir=backups)
    assert res["chroma_changed"] == 3 and res["corpus_changed"] == 2
    assert _flag(store, "conversations", "c1") and _flag(store, "summaries", "s1")
    reloaded = CorpusManager(str(path))
    assert [bool(e.get("curation_quarantined")) for e in reloaded.corpus] == [True, False, True]
    assert "STAT 5100" not in " ".join(str(e.get("query")) for e in reloaded.get_recent_memories(5))
    lines = [json.loads(x) for x in res["preimage"].read_text().splitlines()]
    assert len(lines) == len(TARGETS) and len(list(backups.glob("quarantine_memory_preimage_*.jsonl"))) == 1
    conv = next(x for x in lines if x["id"] == "c1")
    assert "curation_quarantined" not in conv["chroma_metadata"] and "curation_quarantined" not in conv["corpus_entry"]

    rows = qm.plan_targets(store, cm, TARGETS)
    res2 = qm.run(rows, store, cm, undo=True, reason="", backup_dir=backups)
    assert res2["chroma_changed"] == 3 and res2["corpus_changed"] == 2
    assert not _flag(store, "conversations", "c1")
    again = CorpusManager(str(path))
    assert not any(e.get("curation_quarantined") for e in again.corpus)
    assert "STAT 5100" in " ".join(str(e.get("query")) for e in again.get_recent_memories(5))


def test_already_flagged_chroma_gets_corpus_only_flip(env):
    store, cm, path, backups = env
    store.update_metadata("conversations", "c1", {"curation_quarantined": True})
    rows = qm.plan_targets(store, cm, [("conversations", "c1")])
    res = qm.run(rows, store, cm, undo=False, reason="r", backup_dir=backups)
    assert res["chroma_changed"] == 0 and res["corpus_changed"] == 1
    assert CorpusManager(str(path)).corpus[0].get("curation_quarantined") is True


def test_ambiguous_corpus_copy_does_not_block_chroma_flip(env):
    store, cm, path, backups = env
    cm.corpus.append(_conv("STAT 5100 due when", T + timedelta(seconds=1), thread_id="t1"))
    rows = qm.plan_targets(store, cm, [("conversations", "c1")])
    assert rows[0]["reason"] == "ambiguous"
    res = qm.run(rows, store, cm, undo=False, reason="r", backup_dir=backups)
    assert res["chroma_changed"] == 1 and res["corpus_changed"] == 0
    assert _flag(store, "conversations", "c1")


def test_main_refuses_apply_while_daemon_running(env, monkeypatch, tmp_path, capsys):
    store, cm, path, backups = env
    before = path.read_bytes()
    monkeypatch.setattr(qm, "daemon_running", lambda: True)
    monkeypatch.setattr(qm, "BACKUP_DIR", backups)
    monkeypatch.setattr(qm, "_open_store", lambda: store)
    monkeypatch.setattr(qm, "_open_corpus", lambda: cm)
    tf = tmp_path / "t.tsv"
    tf.write_text("# c\nconversations\tc1\n")
    assert qm.main(["--from-file", str(tf), "--apply"]) == 1
    assert path.read_bytes() == before and not backups.exists() and not _flag(store, "conversations", "c1")
    # dry run still allowed and silent
    monkeypatch.setattr(qm, "daemon_running", lambda: True)
    assert qm.main(["--from-file", str(tf)]) == 0
    assert path.read_bytes() == before


def test_main_apply_end_to_end_when_daemon_down(env, monkeypatch, tmp_path):
    store, cm, path, backups = env
    monkeypatch.setattr(qm, "daemon_running", lambda: False)
    monkeypatch.setattr(qm, "BACKUP_DIR", backups)
    monkeypatch.setattr(qm, "_open_store", lambda: store)
    monkeypatch.setattr(qm, "_open_corpus", lambda: cm)
    tf = tmp_path / "t.tsv"
    tf.write_text("conversations\tc1\n")
    assert qm.main(["--from-file", str(tf), "--apply", "--reason", "test"]) == 0
    assert _flag(store, "conversations", "c1") and CorpusManager(str(path)).corpus[0]["curation_quarantined"] is True


def test_parse_targets_skips_malformed_and_dedupes():
    assert qm.parse_targets("# x\nsummaries\ts1\nsummaries\ts1\nbadline\nfacts\tf1  # c\n\n") == [
        ("summaries", "s1"), ("facts", "f1")]
