"""truncate_prompt reports an oversize prefix through the logger, not stdout.

Incident date: 2026-10-27 (lane PRINT). Class: BC-70 (print in app code).
"""
import logging

from models.model_manager import ModelManager


class _Tok:
    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def _run(prompt):
    return ModelManager.truncate_prompt(prompt, _Tok(), 3, preserve_prefix="You are Rowan")


def test_oversize_prefix_logs_warning_not_stdout(caplog, capsys):
    prompt = "a b c d e You are Rowan x y"  # prefix 8 tokens > max 3
    with caplog.at_level(logging.WARNING):
        out = _run(prompt)
    assert out == "Rowan x y"
    assert any("Prefix alone exceeds max input size" in r.getMessage() for r in caplog.records)
    assert "Prefix alone" not in capsys.readouterr().out


def test_oversize_prefix_logs_warning_indented(caplog, capsys):
    prompt = "    a b c d e\n    You are Rowan\n    x y"
    with caplog.at_level(logging.WARNING):
        _run(prompt)
    assert any("Prefix alone exceeds max input size" in r.getMessage() for r in caplog.records)
    assert "Prefix alone" not in capsys.readouterr().out
