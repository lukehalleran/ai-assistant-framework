# Daemon Desktop Executable Build Guide

## Overview

This guide covers building Daemon as a standalone desktop executable using PyInstaller and packaging it with Inno Setup.

**Build Status**: Linux-first packaging fixes landed 2026-10-02 (SPA + models bundled, CLIP off); not yet built end to end. Windows v1.0.0 was last verified 2026-04-02, before the FastAPI migration.

## Prerequisites

1. **Python 3.11+** (tested with 3.11.9)
2. **PyInstaller 6.0+**: `pip install pyinstaller`
3. **All Daemon dependencies**: `pip install -r requirements.txt`
4. **spaCy model**: `python -m spacy download en_core_web_sm`
5. **Inno Setup 6** (for installer): https://jrsoftware.org/issetup.html

## Quick Build (Linux first)

Run from the repo root, in order. Linux is the primary, current path.

```bash
# 1. venv + dependencies (+ spaCy model, see Prerequisites)
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pyinstaller

# 2. Build the React SPA (daemon.spec refuses to run without web/dist/index.html)
cd web && npm ci && npm run build && cd ..

# 3. Stage the offline models (bge-small, all-MiniLM-L6-v2, ms-marco cross-encoder
#    + tiktoken cl100k_base, ~305 MB) from your local HF cache into build/
python scripts/stage_frozen_models.py            # dry-run: shows what it would copy
python scripts/stage_frozen_models.py --apply

# 4. Freeze (10-20 minutes)
pyinstaller daemon.spec --clean --noconfirm

# 5. Run
dist/Daemon/Daemon
```

Notes:
- Step 3 needs the three models already in `~/.cache/huggingface/hub` (run Daemon
  once from source with network, or `huggingface_hub.snapshot_download`). The
  script names any missing model. Without `build/models` the spec only warns, and
  the frozen app has no offline embedder (HF is forced offline in the build).
- **CLIP / visual memory is OFF in the build**: the ~350 MB weights are not
  bundled, so `bootstrap` sets `VISUAL_MEMORY_ENABLED=0` unless CLIP weights are
  present in `_internal/models`. An explicit env var still wins.
- **First run:** the wizard is still the legacy Gradio page (http://localhost:7860);
  after setup the app serves the SPA at http://127.0.0.1:8000 (Gradio dev tabs at `/admin`).
- User data lives in `~/.daemon/` (see User Data Directory), never the install dir.

## Windows (not yet re-verified since the FastAPI migration)

The last verified Windows build (v1.0.0, 2026-04-02) predates the FastAPI/SPA
frontend and the model/SPA bundling above. The steps are the same in PowerShell
(`venv\Scripts\activate`, `python scripts\stage_frozen_models.py --apply`,
`pyinstaller daemon.spec --clean --noconfirm`), followed by
`installer\build_installer.bat` for the Inno Setup installer. Treat the
installer sections below as unverified until a Windows build is re-tested.

## Build Output

```
dist/Daemon/
├── Daemon.exe          # Main executable
├── _internal/          # Python runtime and packages
└── assets/             # Icons and splash screen

installer/output/
└── DaemonSetup-1.0.0.exe  # Distributable installer (~293MB)
```

## Files Created for Build

| File | Purpose |
|------|---------|
| `daemon.spec` | PyInstaller configuration (one-dir mode); bundles `web/dist`, `build/models`, `build/tiktoken` |
| `scripts/stage_frozen_models.py` | Copies the offline HF models + tiktoken file into `build/` |
| `utils/bootstrap.py` | Frozen executable bootstrap (paths, migration) |
| `utils/startup.py` | Progress tracking during startup |
| `hooks/hook-*.py` | Custom PyInstaller hooks for complex packages |
| `hooks/runtime_hook.py` | Pre-import environment setup |
| `assets/daemon_icon.ico` | Windows icon |
| `assets/daemon_icon.png` | Linux/general icon |
| `assets/splash.png` | Splash screen image |
| `installer/daemon_installer.iss` | Inno Setup installer script |
| `installer/build_installer.bat` | Automated installer build script |

## Architecture Decisions

### One-Dir Mode (Not One-File)
- **Reason**: One-file mode extracts to temp on every launch (30s+ delay)
- **Benefit**: Instant startup, easier debugging, smaller AV false positive rate

### Console Window (console=True)
- Console window is visible during operation — closing the browser tab triggers graceful shutdown via Gradio's `unload` event
- Closing the console window also kills the process (Windows `CTRL_CLOSE_EVENT`)
- Stdout is redirected to `%APPDATA%/Daemon/daemon_startup.log` as a safety net when `sys.stdout` is None

### External Data Not Bundled
- Wikipedia FAISS index (~2.2GB) and metadata (~12GB) are NOT bundled
- Features gracefully disabled when external data is missing
- Users can optionally provide external data via `DAEMON_EXTERNAL_DATA` env var or wizard setup
- See [Wikipedia FAISS Setup](#wikipedia-faiss-setup-optional) below for download instructions

### User Data Directory
When running as frozen executable, data is stored in:
- **Windows**: `%APPDATA%/Daemon/`
- **macOS**: `~/Library/Application Support/Daemon/`
- **Linux**: `~/.daemon/`

This includes: corpus, ChromaDB, user profile, .env, logs, narrative context

### User vs Developer Mode (DAEMON_MODE)
The first-run wizard asks users to choose between Personal and Developer mode:

| Feature | User Mode | Dev Mode |
|---------|-----------|----------|
| Chat, memory, web search | Yes | Yes |
| Synthesis pipeline | Disabled | Enabled |
| Code proposals | Disabled | Enabled |
| Reference docs (architecture) | Disabled | Enabled |
| Auto-dedup on shutdown | Executed | Dry-run preview |
| Debug Trace tab | Hidden | Visible |
| Logs tab | Hidden | Visible |
| Proposals tab | Hidden | Visible |
| Synthesis tab | Hidden | Visible |
| Memory Maintenance UI | Hidden | Visible |

### Migration Support
The bootstrap module automatically migrates data from `./data/` to the user data directory on first frozen run.

## First-Run Wizard Flow

```
WELCOME → INTRO → MODE → API_KEY → TAVILY_KEY → WOLFRAM_KEY →
(E2B_KEY if dev) → STYLE → NAME → PRONOUNS → OBSIDIAN →
WIKI_INDEX → BACKGROUND → COMPLETE
```

| Step | Required | Description |
|------|----------|-------------|
| MODE | Yes | Personal or Developer mode |
| API_KEY | Yes | OpenRouter key (validated with live API call) |
| TAVILY_KEY | Optional | Web search (Tavily) |
| WOLFRAM_KEY | Optional | Computational engine (Wolfram Alpha) |
| E2B_KEY | Optional, dev only | Code sandbox (E2B) |
| STYLE | Yes | Warm / Balanced / Direct |
| NAME | Optional | User's name |
| PRONOUNS | Optional | User's pronouns |
| OBSIDIAN | Optional | Obsidian vault path (strongly recommended) |
| WIKI_INDEX | Optional | Wikipedia FAISS index path |
| BACKGROUND | Optional | Initial facts about user |

## Testing the Executable

After build completes:

```bash
# Windows
dist\Daemon\Daemon.exe
```

### Test Checklist

- [ ] App launches (splash appears within 1-2s)
- [ ] GUI appears (within 15-20s)
- [ ] First-run wizard works (if no existing data)
- [ ] Mode selection (personal vs dev) works
- [ ] API key validation works
- [ ] Optional keys can be skipped
- [ ] Obsidian vault path validation works
- [ ] Chat functionality works
- [ ] File upload works (uploaded doc visible in response)
- [ ] Model selector on chat page works
- [ ] Closing browser tab kills process
- [ ] Graceful shutdown (daily note generated, dedup runs)
- [ ] Relaunch skips wizard (profile persisted)

### Clearing User Data for Fresh Test

```bash
del "%APPDATA%\Daemon\.env"
del "%APPDATA%\Daemon\user_profile.json"
del "%APPDATA%\Daemon\corpus_v4.json"
```

## Troubleshooting

### Build Fails with Missing Module
Add to `hiddenimports` in `daemon.spec`:
```python
hiddenimports += ['missing_module']
```

### Build Fails with Missing Data File
Add to `datas` in `daemon.spec`:
```python
datas.append(('path/to/file', 'destination'))
```

### Executable Crashes on Start
1. Check `%APPDATA%\Daemon\daemon_startup.log` for errors
2. If no log exists, temporarily set `console=True` in `daemon.spec`, rebuild, and run from terminal

### Port Already in Use
Previous Daemon instance may still be running (zombie process):
```bash
taskkill /F /IM Daemon.exe
```

### OneDrive Lock Errors During Build
Pause OneDrive sync before building:
- Right-click OneDrive tray icon → Pause syncing → 2 hours

### Large Bundle Size
Current: ~1.2GB uncompressed, ~293MB installer

To reduce:
- Use CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Creating the Installer

### Windows (Inno Setup)
```bash
cd installer
build_installer.bat
# Output: installer/output/DaemonSetup-1.0.0.exe
```

### Distribution
Upload `DaemonSetup-1.0.0.exe` to GitHub Releases along with:
- `daemon-wiki-index-v1.zip` (optional Wikipedia FAISS index, separate download)

## Performance Targets

| Metric | Actual | Target |
|--------|--------|--------|
| Installer size | 293MB | <400MB |
| Cold start to splash | ~1s | <2s |
| Cold start to GUI | ~15s | <20s |
| Memory usage | ~2GB | <2GB |

## Key Files Reference

- `main.py`: Entry point with freeze_support(), bootstrap, splash updates, shutdown tasks
- `utils/bootstrap.py`: Path resolution, data migration, environment setup, splash helpers
- `daemon.spec`: PyInstaller configuration
- `gui/wizard.py`: First-run wizard (mode, keys, obsidian, wiki index)
- `gui/launch.py`: Gradio UI with mode-gated tabs
- `config/app_config.py`: Central config with DAEMON_MODE gating

## Wikipedia FAISS Setup (Optional)

Daemon can use a pre-built FAISS index over 6.5M+ Wikipedia articles (~41M vectors) for knowledge retrieval. This is optional — the assistant works without it.

### Download

```bash
pip install huggingface_hub

# Download the index and metadata (~14.5 GB total)
huggingface-cli download MochiLives/daemon-wiki-faiss \
    --repo-type dataset \
    --local-dir ~/daemon-wiki-data/wiki_data
```

### Configure

Set `WIKI_DATA_ROOT` to the **parent** directory of `wiki_data/`:

```bash
export WIKI_DATA_ROOT=~/daemon-wiki-data
```

Or set individual file paths:

```bash
export FAISS_INDEX_PATH=/path/to/wiki_data/vector_index_ivf.faiss
export FAISS_META_PATH=/path/to/wiki_data/metadata.parquet
```

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| Disk | ~14.5 GB |
| RAM | ~2.6 GB (2.2 GB index + 0.4 GB embedding model) |
| GPU | Not required (CPU works fine) |

Metadata is read on-demand via zero-copy parquet row-group access — no DataFrame is loaded into memory.

### Building From Scratch

If you want to build the index yourself instead of downloading the pre-built one:

```bash
python scripts/build_faiss_index.py
```

This downloads the latest Wikipedia dump, parses/chunks/embeds all articles, and builds an IVF4096,PQ48 index. Requires ~102 GB disk for the raw dump plus ~60 GB for intermediate embeddings.

---

## Development vs Frozen Mode

| Aspect | Development | Frozen |
|--------|-------------|--------|
| Data location | `./data/` | `%APPDATA%/Daemon/` |
| Config | `config/config.yaml` | Bundled in `_internal/` |
| .env | Project root | `%APPDATA%/Daemon/.env` |
| Wiki data | Optional local | Via `DAEMON_EXTERNAL_DATA` |
| IS_FROZEN | False | True |
| Browser close | Server keeps running | Triggers shutdown |
| Splash screen | Print to console | PyInstaller splash |
