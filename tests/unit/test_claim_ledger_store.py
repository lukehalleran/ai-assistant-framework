"""User-claim ledger store (memory/claim_ledger_store.py).

Incident 2026-10-20 (restatement drift: a conditional came back as fact, a
superseded estimate was quoted). Classes: BC-75, BC-52. The store is a
persistent JSON file, so CLAUDE.md's Critical Rule applies: load strict
(missing = empty; corrupt = quarantine + CorruptStoreError, never an empty
stand-in the next save would write over the user's data), save atomic, refuse a
newer schema.

Drives the deployed store; values are invented. conftest's autouse
``_sandbox_claim_ledger`` points the store at a tmp path.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

import memory.claim_ledger_store as store
from memory import claim_ledger as cl
from utils import safe_json


def claim(value="6", at="2026-10-20T09:00:00"):
    return cl.LedgerClaim(
        id=cl.make_claim_id("days_behind", "duration", at, value, f"{value} days"), subject_key="days_behind",
        description="days behind", kind="duration", value=value, unit="days", modality="asserted",
        stated_at=at, quote=f"{value} days")


def test_missing_file_is_an_empty_ledger_and_nothing_is_created():
    ledger = store.get_ledger()
    assert ledger.claims == [] and not os.path.exists(store.store_path())


def test_save_is_atomic_json_with_schema_version_and_round_trips():
    ledger = store.get_ledger()
    ledger.add(claim())
    store.save_ledger()
    payload = json.loads(open(store.store_path()).read())
    assert payload["schema_version"] == 1 and len(payload["claims"]) == 1
    store.reset_for_tests()
    assert [c.value for c in store.get_ledger().claims] == ["6"]
    leftovers = [p for p in os.listdir(os.path.dirname(store.store_path())) if ".tmp" in p]
    assert leftovers == []


def test_get_ledger_is_one_process_wide_instance():
    assert store.get_ledger() is store.get_ledger()


def test_corrupt_file_is_quarantined_and_raises_never_an_empty_ledger():
    path = store.store_path()
    open(path, "w").write("{ not json")
    with pytest.raises(safe_json.CorruptStoreError):
        store.get_ledger()
    siblings = os.listdir(os.path.dirname(path))
    assert any(".corrupt-" in name for name in siblings)          # quarantine copy exists
    assert open(path).read() == "{ not json"                       # the original was not touched
    with pytest.raises(safe_json.CorruptStoreError):               # still refuses; no silent empty state
        store.get_ledger()
    with pytest.raises(RuntimeError):                              # and nothing can be saved over it
        store.save_ledger()
    assert open(path).read() == "{ not json"


def test_wrong_shape_is_a_corrupt_store():
    open(store.store_path(), "w").write(json.dumps({"schema_version": 1, "claims": "nope"}))
    with pytest.raises(safe_json.CorruptStoreError):
        store.get_ledger()


def test_newer_schema_is_refused_and_the_file_is_left_alone():
    path = store.store_path()
    body = json.dumps({"schema_version": 99, "claims": []})
    open(path, "w").write(body)
    with pytest.raises(safe_json.StoreVersionError):
        store.get_ledger()
    assert open(path).read() == body


def test_a_changed_store_path_reloads(tmp_path, monkeypatch):
    first = store.get_ledger()
    first.add(claim())
    store.save_ledger()
    other = tmp_path / "other.json"
    monkeypatch.setattr(store, "_PATH_OVERRIDE", str(other))
    assert store.get_ledger() is not first and store.get_ledger().claims == []


def test_configured_path_is_read_live_not_frozen(monkeypatch, tmp_path):
    from config import app_config
    monkeypatch.setattr(store, "_PATH_OVERRIDE", None)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_STORE_PATH", str(tmp_path / "a.json"))
    assert store.store_path() == str(tmp_path / "a.json")
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_STORE_PATH", str(tmp_path / "b.json"))
    assert store.store_path() == str(tmp_path / "b.json")


def test_write_lock_serialises_concurrent_writers():
    async def scenario():
        order = []

        async def writer(name):
            async with store.write_lock():
                order.append(f"{name}-in")
                await asyncio.sleep(0.02)
                order.append(f"{name}-out")

        await asyncio.gather(writer("a"), writer("b"))
        return order

    order = asyncio.run(scenario())
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


def test_write_lock_is_rebuilt_for_a_new_event_loop():
    async def grab():
        return store.write_lock()

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second


def test_backup_manager_lists_the_store_when_it_exists():
    from utils import backup_manager
    store.get_ledger()
    store.save_ledger()
    assert store.store_path() in backup_manager.backup_targets()
    os.remove(store.store_path())
    assert store.store_path() not in backup_manager.backup_targets()
    assert store.store_path() in backup_manager.backup_targets(existing_only=False)
