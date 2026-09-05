"""The optional multi-GB index must load once even with overlapping searches."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading


def test_overlapping_cold_loads_share_one_initialization(monkeypatch):
    import pyarrow.parquet as pq
    from knowledge import semantic_search as semantic

    entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def read_index(path):
        calls.append(path)
        (entered if len(calls) == 1 else second_entered).set()
        assert release.wait(3), "test did not release fake index load"
        return SimpleNamespace(metric_type=1)

    monkeypatch.setattr(semantic, "faiss", SimpleNamespace(
        read_index=read_index, METRIC_L2=1, METRIC_INNER_PRODUCT=0,
    ))
    monkeypatch.setattr(semantic.os.path, "exists", lambda path: True)
    monkeypatch.setattr(semantic, "_load_embedder", lambda name: object())
    monkeypatch.setattr(pq, "ParquetFile", lambda path: SimpleNamespace(
        metadata=SimpleNamespace(num_row_groups=0),
    ))
    index = semantic.SemanticSearchIndex()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(index.load)
        try:
            assert entered.wait(2)
            second = pool.submit(index.load)
            # Before the fix both workers enter the expensive loader. With
            # single-flight initialization the second waits for publication.
            second_entered.wait(0.2)
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert index.loaded
    index.load()  # warm access must also reuse the published resources
    assert len(calls) == 1, "concurrent cold searches loaded the large index twice"
