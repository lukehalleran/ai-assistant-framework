# Repository Guidelines
Please note this is users very first python project, they are not a programmer and have knowledge gaps. Be mindful.

## Project Structure & Module Organization
- `core/` Orchestrator, prompt builder, response generator.
- `memory/` Chroma store, corpus manager, coordinator, storage.
- `processing/` Multi‑stage gate system and pipeline utilities.
- `models/` Model and tokenizer managers.
- `utils/` Logging, file processing, time/topic helpers.
- `knowledge/` `WikiManager` and retrieval helpers.
- `gui/` Gradio UI; `main.py` launches it.
- `config/` `app_config.py` and optional `config.yaml` (defaults apply if absent).
- `tests/` Runnable scripts and integration stubs.
- `data/`, `chroma_db/`, `embedded_parquet/` Large artifacts (do not commit).

## Build, Test, and Development Commands
- Setup: `pip install -r requirements.txt`
- Run GUI (fast/balanced/max): `make -f Makefile.fast run`, or `python main.py`
- CLI sanity run: `python main.py cli`
- Wikipedia pipeline (sample): `make -f Makefile.fast pipeline`
- Tests (preferred if pytest installed): `python -m pytest -q`
- Test scripts (no pytest): `python tests/test_basic_pipeline.py`, `python tests/test_file_processor.py`

## Coding Style & Naming Conventions
- Python 3.8+; 4‑space indentation; follow PEP 8.
- Names: files/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use type hints and short, single‑purpose functions.
- Logging: use `utils.logging_utils.get_logger(name)`; avoid print in library code.

## Testing Guidelines
- Place tests under `tests/` as `test_*.py`.
- Prefer pytest; async tests can use `asyncio.run(...)` in scripts.
- Mock external/model calls (see `tests/test_basic_pipeline.py`).
- Aim for coverage of core flows: orchestration, memory, gating, and file processing.

## Commit & Pull Request Guidelines
- Messages are short and imperative, often referencing touched modules/files (e.g., "Update file_processor.py: handle temp files").
- PRs include: clear summary, scope of changes, run instructions, and screenshots for UI changes (`gui/`). Link issues when applicable.

## Security & Configuration Tips
- Never commit secrets. Set `OPENAI_API_KEY` via env.
- Heavy data/indexes are local only; verify `.gitignore` before adding files.
- Configuration comes from `config/app_config.py` with env/YAML overrides. Example env overrides: `CHROMA_PATH=./data/chroma_db_v4 CORPUS_FILE=./data/corpus_v4.json python main.py`.
- Performance profiles come from `Makefile.*` (e.g., `CHROMA_DEVICE=cpu|cuda`).

## How to run commands
- Always use a clean shell and our project venv.
- Never run `pyenv rehash` automatically. If dependencies change, I will run it manually.

### Wrapper
Use this exact wrapper for every command:
```bash
bash --noprofile --norc -lc 'source ".venv/bin/activate" && "$CMD"'

