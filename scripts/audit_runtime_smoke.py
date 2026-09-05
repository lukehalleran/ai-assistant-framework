"""Fresh-process smoke: real startup, HTTP, prompt builder and local persistence.

Run from the repository: python scripts/audit_runtime_smoke.py
Uses disposable stores and cached embedding models; generation is deterministic.
No production stores, provider calls, startup maintenance or shutdown jobs.
The temporary directory is retained so failures can be inspected.
"""

import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


async def run(orch, run_dir, captured):
    import httpx
    import uvicorn
    from api.app import create_app
    from api.routes import files
    from gui import handlers
    from memory.corpus_manager import CorpusManager

    files._UPLOAD_DIR = str(run_dir / "uploads")
    cm = orch.memory_system.corpus_manager
    cm.add_entry("Audit attachment: " + "large reference text " * 24000,
                 "The earlier attachment was received.", tags=["audit"])
    app = create_app(orch, start_background=False)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(45):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("server exited before startup")
                await asyncio.sleep(0.05)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
            assert (await client.get("/api/session")).json()["history"] == []
            first = await client.post("/api/chat", json={"text": "thanks"})
            assert first.status_code == 200 and "event: complete" in first.text
            assert "event: error" not in first.text
            first_prompt_chars = len(captured[-1])
            assert first_prompt_chars < 50000, first_prompt_chars
            upload = await client.post("/api/uploads", files={"files": ("audit_note.txt", b"AUDIT_ATTACHMENT_UNIQUE\nOne small reference note.")})
            upload.raise_for_status()
            fid = upload.json()["files"][0]["file_id"]
            second = await client.post("/api/chat", json={"text": "Please explain the attached note.", "file_ids": [fid, fid]})
            assert second.status_code == 200 and "event: complete" in second.text
            assert "event: error" not in second.text
            assert captured[-1].count("AUDIT_ATTACHMENT_UNIQUE") == 1
            history = (await client.get("/api/session")).json()["history"]
            assert [x["role"] for x in history] == ["user", "assistant", "user", "assistant"]
            assert all(x["content"] for x in history)
            if handlers._pending_storage_tasks:
                await asyncio.wait_for(asyncio.gather(*tuple(handlers._pending_storage_tasks)), 30)
            reloaded = CorpusManager(corpus_file=cm.corpus_file)
            saved = reloaded.get_recent_memories(count=3)
            assert any(item.get("query") == "thanks" for item in saved)
            assert len(saved) >= 3
            print("AUDIT_RESULT=" + json.dumps({
                "code_root": str(REPO), "pid": os.getpid(), "data_root": str(run_dir),
                "requests": 2, "history_messages": len(history), "persisted_entries": len(saved),
                "first_prompt_chars": first_prompt_chars,
                "attachment_occurrences": captured[-1].count("AUDIT_ATTACHMENT_UNIQUE"),
                "generation": "deterministic; real provider behavior not tested",
            }), flush=True)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        listener.close()


def main():
    import yaml
    run_dir = Path(tempfile.mkdtemp(prefix="daemon-runtime-audit-"))
    cfg = yaml.safe_load((REPO / "config/config.yaml").read_text())
    # Disable optional integrations and maintenance. The main request pipeline,
    # embeddings, Chroma, corpus, file parser, HTTP adapter and renderer are real.
    for section in cfg.values():
        if isinstance(section, dict) and "enabled" in section:
            section["enabled"] = False
    cfg["light_prompt"]["enabled"] = True
    cfg["features"]["use_stm_pass"] = False
    cfg["obsidian"]["vault_path"] = str(run_dir / "vault")
    cfg["memory"]["corpus_file"] = str(run_dir / "data/corpus.json")
    cfg["memory"]["chroma_path"] = str(run_dir / "data/chroma")
    # Wiki retrieval is controlled by numeric limits, not an enabled flag.
    # A timed-out FAISS load otherwise keeps its worker alive after the test.
    cfg["memory"]["prompt_max_semantic"] = 0
    cfg["memory"]["prompt_max_wiki"] = 0
    (run_dir / "data").mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "config.local.yaml").write_text("{}\n")
    os.chdir(run_dir)
    sys.path.insert(0, str(REPO))
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DAEMON_TEST_MODE": "1",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = value
    captured = []

    async def generate_once(self, prompt, **kwargs):
        return '{"topic":"Audit","tone":"neutral","is_heavy_topic":false}'

    async def generate_async(self, prompt, **kwargs):
        captured.append(prompt)
        async def stream():
            for text in ("The audit fixture ", "contains one note."):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")])
        return stream()

    # Fail closed on outbound sockets, even if an optional integration has an
    # overlooked activation path. Local HTTP is the only permitted connection.
    original_connect = socket.socket.connect
    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("audit blocks external network connections")
        return original_connect(sock, address)

    with patch("dotenv.load_dotenv", return_value=False), patch("socket.socket.connect", local_connect):
        import torch
        torch.set_num_threads(2)
        from models.model_manager import ModelManager
        with patch.object(ModelManager, "generate_once", generate_once), patch.object(ModelManager, "generate_async", generate_async):
            import main as runtime
            started = time.perf_counter()
            orch = runtime.build_orchestrator()
            print(f"AUDIT_STARTUP_SECONDS={time.perf_counter() - started:.3f}", flush=True)
            asyncio.run(run(orch, run_dir, captured))


if __name__ == "__main__":
    main()
