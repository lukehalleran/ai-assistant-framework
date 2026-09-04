#!/usr/bin/env python3
"""Thin entrypoint for the daily-note catch-up job (2026-09-03).

`python main.py daily-note-catchup` imports the whole runtime at module level
(orchestrator, Chroma store, gate system, GUI, wiki manager) before the mode
dispatch runs — the 02:00 systemd job paid ~8 s CPU and up to 1.6 GB for a task
that only needs the notes generator and the LLM client. Worse, the live unit
ran it under the system interpreter (Python 3.13 / numpy 2 / chromadb 0.6.3),
which printed numpy ABI tracebacks on every run and could not safely open the
1.0.7 Chroma store. This script imports only what the job needs and MUST run
under the project interpreter (pyproject: python >=3.11,<3.12 — the pyenv
3.11.8 env). See scripts/systemd/README.md for the timer install.

Exit codes: 0 = note generated / already exists / skipped; 1 = generation
error (so a systemd OnFailure= drop-in can alert).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

# .env takes precedence over the shell environment (mirrors main.py).
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)


def run_catchup() -> int:
    """Generate yesterday's daily note if it is missing. Mirrors main.py's
    ``daily-note-catchup`` mode output exactly."""
    import asyncio

    from utils.logging_utils import configure_logging
    configure_logging()
    try:
        from utils.python_fs_guard import activate as _activate_fs_guard
        _activate_fs_guard()
    except Exception:
        pass

    from models.model_manager import ModelManager
    from utils.daily_notes_generator import DailyNotesGenerator

    print(f"\n{'=' * 60}")
    print("DAILY NOTE CATCH-UP")
    print(f"{'=' * 60}")

    # Create ModelManager with API key to ensure LLM is available
    model_manager = ModelManager()
    generator = DailyNotesGenerator(model_manager=model_manager)
    result = asyncio.run(generator.generate_yesterday_if_missing())

    if result is None:
        print("Yesterday's note already exists, nothing to do.")
        return 0
    if result.success:
        print("Generated yesterday's note:")
        print(f"  Output: {result.output_path}")
        print(f"  Conversations: {result.conversation_count}")
        return 0
    if result.skipped_reason:
        print(f"Skipped: {result.skipped_reason}")
        return 0
    print(f"Failed: {result.error}")
    return 1


if __name__ == "__main__":
    sys.exit(run_catchup())
