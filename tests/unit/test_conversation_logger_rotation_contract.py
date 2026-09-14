"""Regression contract for text and JSONL conversation-log rotation."""

import subprocess
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "utils" / "conversation_logger.py"


class ConversationLoggerRotationContractTests(unittest.TestCase):
    def test_rotation_formats_and_concurrent_writes_complete_in_bounded_child(self):
        child = r'''import importlib.util
import json
import sys
import tempfile
import threading
from pathlib import Path

source = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("conversation_logger_under_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Logger = module.ConversationLogger

with tempfile.TemporaryDirectory() as root:
    root = Path(root)

    text_dir = root / "text"
    text = Logger(log_dir=str(text_dir), log_format="text", max_file_size_mb=0)
    text.log_interaction("text-one", "reply-one")
    text.log_interaction("text-two", "reply-two")
    text_files = sorted(text_dir.glob("*.txt"))
    assert len(text_files) == 3, text_files
    text_bodies = [path.read_text(encoding="utf-8") for path in text_files]
    assert "text-one" not in text_bodies[0] and "text-two" not in text_bodies[0]
    assert sum("text-one" in body for body in text_bodies) == 1
    assert sum("text-two" in body for body in text_bodies) == 1
    assert all("DAEMON CONVERSATION LOG" in body for body in text_bodies)

    json_dir = root / "json"
    structured = Logger(log_dir=str(json_dir), log_format="json", max_file_size_mb=0)
    structured.log_interaction("json-one", "reply-one")
    structured.log_interaction("json-two", "reply-two")
    structured.log_interaction("json-three", "reply-three")
    json_files = sorted(json_dir.glob("*.jsonl"))
    assert len(json_files) == 3, json_files
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in json_files]
    assert [row["user_input"] for row in rows] == ["json-one", "json-two", "json-three"]
    assert [row["conversation_id"] for row in rows] == [1, 2, 3]

    concurrent_dir = root / "concurrent"
    concurrent = Logger(log_dir=str(concurrent_dir), log_format="text")
    start = threading.Barrier(9)
    failures = []
    def write_batch(batch):
        try:
            start.wait(timeout=2)
            for item in range(20):
                value = f"{batch}-{item}"
                concurrent.log_interaction(value, f"reply-{value}")
        except BaseException as exc:
            failures.append(exc)
    threads = [threading.Thread(target=write_batch, args=(batch,)) for batch in range(8)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=3)
    assert not any(thread.is_alive() for thread in threads)
    assert not failures, failures
    assert concurrent.conversation_count == 160
    body = next(concurrent_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert body.count("--- Conversation #") == 160
    assert all(body.count("USER:" + chr(10) + f"{batch}-{item}" + chr(10)) == 1 for batch in range(8) for item in range(20))

print("rotation contract passed")
'''
        result = subprocess.run(
            [sys.executable, "-c", child, str(SOURCE)],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
        self.assertIn("rotation contract passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
