"""
Lightweight GUI smoke-run to validate that the Gradio app
builds successfully (especially the Logs tab code viewer).

It uses a stub orchestrator to avoid loading models or data.
The server launches and blocks; run under a short timeout
in CI or tooling to simply exercise construction and launch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any


class DummyPersonalityManager:
    def __init__(self) -> None:
        self.personalities: Dict[str, str] = {"Default": "default"}
        self.current_personality: str = "Default"

    def switch_personality(self, name: str) -> None:
        if name in self.personalities:
            self.current_personality = name


class DummyCorpusManager:
    def __init__(self) -> None:
        self.corpus: List[Dict[str, Any]] = []

    def get_summaries(self, limit: int | None = None) -> List[Dict[str, Any]]:
        return []

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        return []


class DummyChromaColl:
    def count(self) -> int:
        return 0


class DummyChromaStore:
    def __init__(self) -> None:
        self.collections = {"reflections": DummyChromaColl()}


class DummyConsolidator:
    def __init__(self) -> None:
        self.consolidation_threshold: int = 10


class DummyMemorySystem:
    def __init__(self) -> None:
        self.corpus_manager = DummyCorpusManager()
        self.chroma_store = DummyChromaStore()
        self.consolidator = DummyConsolidator()

    async def run_shutdown_reflection(self, **kwargs) -> None:
        return None

    async def process_shutdown_memory(self, **kwargs) -> None:
        return None

    async def store_interaction(self, **kwargs) -> None:
        return None


class DummyPromptBuilder:
    def __init__(self) -> None:
        self.consolidator = DummyConsolidator()


class DummyOrchestrator:
    def __init__(self) -> None:
        self.personality_manager = DummyPersonalityManager()
        self.memory_system = DummyMemorySystem()
        self.prompt_builder = DummyPromptBuilder()


def main() -> None:
    # Ensure no external share/browser during smoke run
    os.environ.setdefault("GRADIO_SHARE", "0")
    os.environ.setdefault("GRADIO_OPEN_BROWSER", "0")
    # Ensure project root is importable when running as a script
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from gui.launch import launch_gui

    launch_gui(DummyOrchestrator())


if __name__ == "__main__":
    main()
