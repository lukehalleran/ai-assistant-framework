"""A02/A03 (2026-10-23): the profile .env writer is one serialized transaction
and an update changes the EFFECTIVE value.

class: BC-38 (read/modify/write race), BC-33, BC-26, BC-45. Synthetic .env only.
"""
import threading

from dotenv import dotenv_values

from gui import setup_core


def _race(monkeypatch, tmp_path, first, second):
    """Two real write_env_vars calls; a barrier after each READS the file."""
    barrier = threading.Barrier(2, timeout=1.0)
    real_read = setup_core._read_env_lines

    def gated_read(path):
        lines = real_read(path)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass  # serialized: the second reader never meets the first (expected after the fix)
        return lines

    monkeypatch.setattr(setup_core, "_read_env_lines", gated_read)
    results = {}

    def run(tag, mapping):
        results[tag] = setup_core.write_env_vars(mapping)

    threads = [threading.Thread(target=run, args=("a", first)), threading.Thread(target=run, args=("b", second))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return results


def test_concurrent_saves_of_different_keys_both_survive(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OTHER=keep\n", encoding="utf-8")
    results = _race(monkeypatch, tmp_path, {"TAVILY_API_KEY": "tvly-synthetic"},
                    {"WOLFRAM_APP_ID": "WOLF-SYNTH"})
    assert results == {"a": True, "b": True}
    values = dotenv_values(tmp_path / ".env")
    assert values["TAVILY_API_KEY"] == "tvly-synthetic"
    assert values["WOLFRAM_APP_ID"] == "WOLF-SYNTH"
    assert values["OTHER"] == "keep"


def test_duplicate_assignment_update_changes_effective_value(tmp_path):
    (tmp_path / ".env").write_text(
        "TAVILY_API_KEY=old_synthetic\n# note\nTAVILY_API_KEY=stale_synthetic\n", encoding="utf-8")
    assert setup_core.write_env_vars({"TAVILY_API_KEY": "new_synthetic"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert dotenv_values(tmp_path / ".env")["TAVILY_API_KEY"] == "new_synthetic"
    assert text.count("TAVILY_API_KEY") == 1
    assert "# note" in text


def test_export_form_duplicates_are_canonicalized(tmp_path):
    (tmp_path / ".env").write_text(
        "export WOLFRAM_APP_ID=old\n  export WOLFRAM_APP_ID = stale\n", encoding="utf-8")
    assert setup_core.write_env_vars({"WOLFRAM_APP_ID": "fresh"})
    assert dotenv_values(tmp_path / ".env")["WOLFRAM_APP_ID"] == "fresh"
    assert (tmp_path / ".env").read_text(encoding="utf-8").count("WOLFRAM_APP_ID") == 1


def test_untouched_duplicate_of_another_key_is_left_alone(tmp_path):
    (tmp_path / ".env").write_text("KEEP=a\nKEEP=b\nTAVILY_API_KEY=x\n", encoding="utf-8")
    assert setup_core.write_env_vars({"TAVILY_API_KEY": "y"})
    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert lines == ["KEEP=a", "KEEP=b", "TAVILY_API_KEY=y"]
