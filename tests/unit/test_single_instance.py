"""Tests for utils/single_instance.py — the double-launch guard."""

import os
import subprocess
import sys
import textwrap

import pytest

from utils.single_instance import (
    LOCK_FILENAME,
    SingleInstanceError,
    acquire_single_instance_lock,
)


class TestSingleInstanceLock:
    def test_acquire_writes_pid(self, tmp_path):
        fh = acquire_single_instance_lock(str(tmp_path))
        try:
            with open(tmp_path / LOCK_FILENAME) as f:
                assert f.read().strip() == str(os.getpid())
        finally:
            fh.close()

    def test_second_acquire_fails_with_holder_pid(self, tmp_path):
        fh = acquire_single_instance_lock(str(tmp_path))
        try:
            with pytest.raises(SingleInstanceError) as exc:
                acquire_single_instance_lock(str(tmp_path))
            assert str(os.getpid()) in str(exc.value)
        finally:
            fh.close()

    def test_lock_released_on_close(self, tmp_path):
        fh = acquire_single_instance_lock(str(tmp_path))
        fh.close()
        fh2 = acquire_single_instance_lock(str(tmp_path))
        fh2.close()

    def test_lock_released_on_process_death(self, tmp_path):
        """A killed holder must not strand the lock (the zombie scenario)."""
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {repr(os.getcwd())})
            from utils.single_instance import acquire_single_instance_lock
            fh = acquire_single_instance_lock({repr(str(tmp_path))})
            print("LOCKED", flush=True)
            import time; time.sleep(30)
        """)
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert proc.stdout.readline().strip() == "LOCKED"
            # Child holds it → we can't take it
            with pytest.raises(SingleInstanceError):
                acquire_single_instance_lock(str(tmp_path))
        finally:
            proc.kill()
            proc.wait(timeout=10)

        # Kernel released the dead child's lock; stale file must not block us
        fh = acquire_single_instance_lock(str(tmp_path))
        fh.close()

    def test_creates_lock_dir(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        fh = acquire_single_instance_lock(str(target))
        try:
            assert (target / LOCK_FILENAME).exists()
        finally:
            fh.close()
