"""B2: writer overlap uses independent temporary files, with real file I/O."""
import json
import threading

import pytest

from utils import safe_json


def test_overlapping_writers_both_publish_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "store.json"
    first_written = threading.Event()
    release_first = threading.Event()
    real_dump = json.dump
    real_replace = safe_json.os.replace
    published, errors = [], []

    def controlled_dump(data, stream, **kwargs):
        real_dump(data, stream, **kwargs)
        stream.flush()
        if data["writer"] == "first":
            first_written.set()
            assert release_first.wait(5), "second writer did not finish"

    def observe_replace(src, dst):
        real_replace(src, dst)
        published.append(json.loads(path.read_text()))

    monkeypatch.setattr(safe_json.json, "dump", controlled_dump)
    monkeypatch.setattr(safe_json.os, "replace", observe_replace)

    def write_first():
        try:
            safe_json.atomic_write_json(str(path), {"writer": "first", "text": "a" * 8000})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=write_first)
    thread.start()
    try:
        assert first_written.wait(5)
        safe_json.atomic_write_json(str(path), {"writer": "second", "text": "b"})
    finally:
        release_first.set()
        thread.join(5)
    assert not thread.is_alive() and not errors, errors
    assert [row["writer"] for row in published] == ["second", "first"]
    assert published[1]["text"] == "a" * 8000
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_text_preserves_unicode_and_private_mode(tmp_path):
    path = tmp_path / "token.json"
    safe_json.atomic_write_text(path, "synthetic café\n", mode=0o600)
    assert path.read_text() == "synthetic café\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_failed_publish_preserves_destination_and_cleans_own_temp(tmp_path, monkeypatch):
    path = tmp_path / "store.json"
    path.write_text('{"original": true}')

    def failed_replace(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(safe_json.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        safe_json.atomic_write_json(str(path), {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [path]
